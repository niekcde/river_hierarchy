"""Apply vector manual edits to a binary raster mask.

Manual edits are expected as polygon features with an ``action`` field:

- ``add`` burns pixels to 1
- ``remove`` burns pixels to 0

The output keeps the base raster's CRS, transform, dimensions, and nodata
convention, and writes a strict uint8 binary GeoTIFF.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.features import rasterize


ACTION_ALIASES = {
    "add": "add",
    "water": "add",
    "1": "add",
    "true": "add",
    "remove": "remove",
    "erase": "remove",
    "delete": "remove",
    "0": "remove",
    "false": "remove",
}


def main() -> None:
    args = parse_args()
    summary = apply_manual_edits(
        base_mask=args.base_mask,
        edits=args.edits,
        output=args.output,
        layer=args.layer,
        action_field=args.action_field,
        all_touched=not args.center_only,
        summary_path=args.summary,
    )
    print(json.dumps(summary, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-mask", type=Path, required=True, help="Prepared binary GeoTIFF to edit.")
    parser.add_argument("--edits", type=Path, required=True, help="GeoPackage/vector file with edit polygons.")
    parser.add_argument("--output", type=Path, required=True, help="Cleaned binary GeoTIFF output path.")
    parser.add_argument("--layer", default=None, help="Vector layer name. Defaults to the first layer.")
    parser.add_argument("--action-field", default="action", help="Field containing add/remove actions.")
    parser.add_argument(
        "--summary",
        type=Path,
        default=None,
        help="Optional JSON summary path. Defaults to output path with .summary.json suffix.",
    )
    parser.add_argument(
        "--center-only",
        action="store_true",
        help="Only edit pixels whose centers fall inside polygons. Default edits all touched pixels.",
    )
    return parser.parse_args()


def apply_manual_edits(
    *,
    base_mask: Path,
    edits: Path,
    output: Path,
    layer: str | None = None,
    action_field: str = "action",
    all_touched: bool = True,
    summary_path: Path | None = None,
) -> dict[str, object]:
    base_mask = Path(base_mask).expanduser().resolve()
    edits = Path(edits).expanduser().resolve()
    output = Path(output).expanduser().resolve()
    summary_path = output.with_suffix(".summary.json") if summary_path is None else Path(summary_path).expanduser().resolve()

    with rasterio.open(base_mask) as src:
        base = (src.read(1, masked=True).filled(0) > 0).astype("uint8")
        profile = src.profile.copy()
        raster_crs = src.crs
        transform = src.transform
        out_shape = src.shape

    edit_gdf = gpd.read_file(edits, layer=layer)
    if edit_gdf.empty:
        cleaned = base
        action_counts: dict[str, int] = {}
    else:
        if action_field not in edit_gdf.columns:
            raise ValueError(f"Edit layer must contain an '{action_field}' field.")
        if edit_gdf.crs is None:
            raise ValueError("Edit layer has no CRS. Set it to match the base mask CRS.")
        if raster_crs is not None and edit_gdf.crs != raster_crs:
            edit_gdf = edit_gdf.to_crs(raster_crs)

        cleaned = base.copy()
        action_counts = {"add": 0, "remove": 0}
        for idx, row in edit_gdf.iterrows():
            action = normalize_action(row[action_field])
            geometry = row.geometry
            if geometry is None or geometry.is_empty:
                continue
            burn = rasterize(
                [(geometry, 1)],
                out_shape=out_shape,
                transform=transform,
                fill=0,
                dtype="uint8",
                all_touched=all_touched,
            ).astype(bool)
            if action == "add":
                cleaned[burn] = 1
            elif action == "remove":
                cleaned[burn] = 0
            else:
                raise ValueError(f"Unsupported normalized action at row {idx}: {action}")
            action_counts[action] = action_counts.get(action, 0) + 1

    profile.update(driver="GTiff", dtype="uint8", count=1, nodata=0, compress="lzw")
    output.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(output, "w", **profile) as dst:
        dst.write(cleaned.astype("uint8"), 1)

    summary = {
        "base_mask": str(base_mask),
        "edits": str(edits),
        "output": str(output),
        "action_field": action_field,
        "all_touched": all_touched,
        "actions": action_counts,
        "base_water_pixels": int(base.sum()),
        "cleaned_water_pixels": int(cleaned.sum()),
        "pixels_added": int(((cleaned == 1) & (base == 0)).sum()),
        "pixels_removed": int(((cleaned == 0) & (base == 1)).sum()),
        "changed_pixels": int((cleaned != base).sum()),
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def normalize_action(value: object) -> str:
    key = str(value).strip().lower()
    if key not in ACTION_ALIASES:
        allowed = ", ".join(sorted(ACTION_ALIASES))
        raise ValueError(f"Unsupported manual edit action '{value}'. Expected one of: {allowed}")
    return ACTION_ALIASES[key]


if __name__ == "__main__":
    main()
