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
import pandas as pd
import rasterio
from rasterio.features import rasterize, shapes
from scipy import ndimage
from shapely.geometry import Point, box, shape
from shapely.ops import unary_union

from rivgraph_centerline.raster_mask import crop_binary_mask


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
        tiny_land_remove_max_pixels=args.tiny_land_remove_max_pixels,
        tiny_land_export_min_pixels=args.tiny_land_export_min_pixels,
        tiny_land_export_max_pixels=args.tiny_land_export_max_pixels,
        tiny_land_export_path=args.tiny_land_export_path,
        diagonal_passage_export_path=args.diagonal_passage_export_path,
        keep_largest_water_component=args.keep_largest_water_component,
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
    parser.add_argument(
        "--tiny-land-remove-max-pixels",
        type=int,
        default=3,
        help="Remove main-river land components with this many pixels or fewer after manual edits. Use a negative value to disable.",
    )
    parser.add_argument(
        "--tiny-land-export-min-pixels",
        type=int,
        default=4,
        help="Minimum component size, in pixels, for the exported main-river land QA layer after cleanup.",
    )
    parser.add_argument(
        "--tiny-land-export-max-pixels",
        type=int,
        default=10,
        help="Maximum component size, in pixels, for the exported main-river land QA layer after cleanup. Use a negative value to disable export.",
    )
    parser.add_argument(
        "--tiny-land-export-path",
        type=Path,
        default=None,
        help="Optional GeoPackage path for the post-cleanup main-river small-land export.",
    )
    parser.add_argument(
        "--diagonal-passage-export-path",
        type=Path,
        default=None,
        help="Optional GeoPackage path for the post-cleanup main-river diagonal-passage diagnostic points.",
    )
    parser.add_argument(
        "--keep-largest-water-component",
        action="store_true",
        help="After manual edits, discard all 8-connected water components except the largest one.",
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
    tiny_land_remove_max_pixels: int = 3,
    tiny_land_export_min_pixels: int = 4,
    tiny_land_export_max_pixels: int = 10,
    tiny_land_export_path: Path | None = None,
    diagonal_passage_export_path: Path | None = None,
    keep_largest_water_component: bool = False,
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

    cleaned, largest_component_summary = keep_largest_water_component_only(
        cleaned,
        enabled=keep_largest_water_component,
    )
    cleaned, tiny_land_summary = cleanup_and_export_main_river_small_land(
        cleaned,
        transform=transform,
        crs=raster_crs,
        tiny_land_remove_max_pixels=tiny_land_remove_max_pixels,
        tiny_land_export_min_pixels=tiny_land_export_min_pixels,
        tiny_land_export_max_pixels=tiny_land_export_max_pixels,
        tiny_land_export_path=tiny_land_export_path,
    )
    diagonal_summary = export_main_river_diagonal_passage_points(
        cleaned,
        transform=transform,
        crs=raster_crs,
        export_path=diagonal_passage_export_path,
    )

    cleaned_cropped, cleaned_transform, crop_window = crop_binary_mask(cleaned, transform)
    profile.update(
        driver="GTiff",
        dtype="uint8",
        count=1,
        nodata=0,
        compress="lzw",
        transform=cleaned_transform,
        width=int(cleaned_cropped.shape[1]),
        height=int(cleaned_cropped.shape[0]),
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(output, "w", **profile) as dst:
        dst.write(cleaned_cropped.astype("uint8"), 1)

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
        "output_shape": [int(cleaned_cropped.shape[0]), int(cleaned_cropped.shape[1])],
        "crop_window": crop_window,
        "largest_water_component_filter": largest_component_summary,
        "main_river_small_land_cleanup": tiny_land_summary,
        "main_river_diagonal_passages": diagonal_summary,
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def apply_manual_edits_to_panels(
    *,
    source_panel_root: Path,
    output_panel_root: Path,
    edits: Path,
    layer: str | None = None,
    action_field: str = "action",
    all_touched: bool = True,
    tiny_land_remove_max_pixels: int = 3,
    tiny_land_export_min_pixels: int = 4,
    tiny_land_export_max_pixels: int = 10,
    tiny_land_export_path: Path | None = None,
    diagonal_passage_export_path: Path | None = None,
    keep_largest_water_component: bool = False,
    panel_footprints: dict[str, object] | None = None,
    overwrite: bool = False,
) -> dict[str, object]:
    """Apply vector edits and local cleanup to each buffered panel mask.

    The largest-water-component filter is intentionally local to each panel.
    Callers must ensure the intended river is the largest 8-connected component
    in every buffered processing area.
    """
    source_panel_root = Path(source_panel_root).expanduser().resolve()
    output_panel_root = Path(output_panel_root).expanduser().resolve()
    edits = Path(edits).expanduser().resolve()
    edit_gdf = gpd.read_file(edits, layer=layer)
    if action_field not in edit_gdf.columns:
        raise ValueError(f"Edit layer must contain an '{action_field}' field.")
    panel_dirs = sorted(path for path in source_panel_root.iterdir() if path.is_dir() and path.name.startswith("P"))
    if not panel_dirs:
        raise FileNotFoundError(f"No panel directories found in {source_panel_root}")

    summaries: dict[str, object] = {}
    for panel_dir in panel_dirs:
        source = panel_dir / "mask.tif"
        output = output_panel_root / panel_dir.name / "mask.tif"
        if not source.exists():
            raise FileNotFoundError(f"Panel mask does not exist: {source}")
        if output.exists() and not overwrite:
            raise FileExistsError(f"Panel output already exists: {output}")
        with rasterio.open(source) as src:
            base = (src.read(1, masked=True).filled(0) > 0).astype("uint8")
            profile, raster_crs, transform, out_shape = src.profile.copy(), src.crs, src.transform, src.shape
        local_edits = edit_gdf
        if local_edits.crs is None:
            raise ValueError("Edit layer has no CRS. Set it to match the panel masks.")
        if raster_crs is not None and local_edits.crs != raster_crs:
            local_edits = local_edits.to_crs(raster_crs)
        footprint = None if panel_footprints is None else panel_footprints.get(panel_dir.name)
        selection_geometry = box(*rasterio.transform.array_bounds(*out_shape, transform)) if footprint is None else footprint
        local_edits = local_edits.loc[local_edits.geometry.intersects(selection_geometry)]
        cleaned = base.copy()
        action_counts = {"add": 0, "remove": 0}
        for idx, row in local_edits.iterrows():
            action = normalize_action(row[action_field])
            geometry = row.geometry
            if geometry is None or geometry.is_empty:
                continue
            burn = rasterize([(geometry, 1)], out_shape=out_shape, transform=transform, fill=0, dtype="uint8", all_touched=all_touched).astype(bool)
            cleaned[burn] = 1 if action == "add" else 0
            action_counts[action] += 1
        if footprint is not None:
            allowed = rasterize([(footprint, 1)], out_shape=out_shape, transform=transform, fill=0, dtype="uint8").astype(bool)
            cleaned[~allowed] = 0
        cleaned, largest = keep_largest_water_component_only(cleaned, enabled=keep_largest_water_component)
        qa_dir = output.parent / "manual_edit_qa"
        panel_small_land = qa_dir / "main_river_small_land.gpkg" if tiny_land_export_path is not None else None
        panel_diagonal = qa_dir / "diagonal_passages.gpkg" if diagonal_passage_export_path is not None else None
        cleaned, tiny_land = cleanup_and_export_main_river_small_land(
            cleaned, transform=transform, crs=raster_crs,
            tiny_land_remove_max_pixels=tiny_land_remove_max_pixels,
            tiny_land_export_min_pixels=tiny_land_export_min_pixels,
            tiny_land_export_max_pixels=tiny_land_export_max_pixels,
            tiny_land_export_path=panel_small_land,
        )
        diagonal = export_main_river_diagonal_passage_points(
            cleaned, transform=transform, crs=raster_crs, export_path=panel_diagonal,
        )
        cropped, cropped_transform, crop_window = crop_binary_mask(cleaned, transform)
        profile.update(driver="GTiff", dtype="uint8", count=1, nodata=0, compress="lzw", transform=cropped_transform, width=int(cropped.shape[1]), height=int(cropped.shape[0]))
        output.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(output, "w", **profile) as dst:
            dst.write(cropped.astype("uint8"), 1)
        summaries[panel_dir.name] = {
            "source_mask": str(source), "output_mask": str(output), "intersecting_edit_count": int(len(local_edits)),
            "actions": action_counts, "base_water_pixels": int(base.sum()), "cleaned_water_pixels": int(cleaned.sum()),
            "largest_water_component_filter": largest, "main_river_small_land_cleanup": tiny_land,
            "main_river_diagonal_passages": diagonal, "crop_window": crop_window,
        }
    small_land_summary = _merge_panel_diagnostic_layers(
        panel_root=output_panel_root, panel_footprints=panel_footprints, source_name="main_river_small_land.gpkg",
        source_layer="main_river_small_land", output_path=tiny_land_export_path, output_layer="main_river_small_land",
        empty_factory=_empty_small_land_gdf,
    )
    diagonal_summary = _merge_panel_diagnostic_layers(
        panel_root=output_panel_root, panel_footprints=panel_footprints, source_name="diagonal_passages.gpkg",
        source_layer="main_river_diagonal_passages", output_path=diagonal_passage_export_path,
        output_layer="main_river_diagonal_passages", empty_factory=_empty_diagonal_passage_gdf,
    )
    return {
        "source_panel_root": str(source_panel_root), "output_panel_root": str(output_panel_root), "edits": str(edits),
        "local_largest_component_filter": keep_largest_water_component, "panels": summaries,
        "main_river_small_land": small_land_summary, "main_river_diagonal_passages": diagonal_summary,
    }


def _merge_panel_diagnostic_layers(
    *,
    panel_root: Path,
    panel_footprints: dict[str, object] | None,
    source_name: str,
    source_layer: str,
    output_path: Path | None,
    output_layer: str,
    empty_factory: object,
) -> dict[str, object]:
    """Merge panel diagnostics, assigning overlap features to their core."""
    if output_path is None:
        return {"enabled": False, "count": 0, "output_path": None}
    records: list[gpd.GeoDataFrame] = []
    crs = None
    for panel_dir in sorted(path for path in panel_root.iterdir() if path.is_dir() and path.name.startswith("P")):
        if crs is None:
            with rasterio.open(panel_dir / "mask.tif") as panel_mask:
                crs = panel_mask.crs
        source = panel_dir / "manual_edit_qa" / source_name
        if not source.exists():
            continue
        frame = gpd.read_file(source, layer=source_layer)
        crs = frame.crs if frame.crs is not None else crs
        core = None if panel_footprints is None else panel_footprints.get(panel_dir.name)
        if core is not None and not frame.empty:
            frame = frame.loc[frame.geometry.representative_point().map(core.covers)].copy()
        if not frame.empty:
            frame["panel_id"] = panel_dir.name
            records.append(frame)
    if records:
        merged = gpd.GeoDataFrame(pd.concat(records, ignore_index=True), geometry="geometry", crs=records[0].crs)
    else:
        merged = empty_factory(crs)
        merged["panel_id"] = pd.Series(dtype="object")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        output_path.unlink()
    merged.to_file(output_path, layer=output_layer, driver="GPKG")
    return {"enabled": True, "count": int(len(merged)), "output_path": str(output_path)}


def keep_largest_water_component_only(
    cleaned: np.ndarray,
    *,
    enabled: bool,
) -> tuple[np.ndarray, dict[str, object]]:
    """Optionally retain only the largest 8-connected water component."""
    water = cleaned.astype(bool)
    water_pixels_before = int(water.sum())
    if not enabled:
        return cleaned, {
            "enabled": False,
            "water_pixels_before": water_pixels_before,
            "water_pixels_after": water_pixels_before,
            "removed_water_pixels": 0,
            "removed_component_count": 0,
            "retained_component_label": None,
            "retained_component_pixels": None,
        }

    labels, n_components = ndimage.label(water, structure=np.ones((3, 3), dtype=int))
    if n_components == 0:
        return cleaned, {
            "enabled": True,
            "water_pixels_before": 0,
            "water_pixels_after": 0,
            "removed_water_pixels": 0,
            "removed_component_count": 0,
            "retained_component_label": None,
            "retained_component_pixels": 0,
        }

    sizes = np.bincount(labels.ravel())[1:]
    retained_label = int(np.argmax(sizes) + 1)
    retained = labels == retained_label
    retained_pixels = int(sizes[retained_label - 1])
    return retained.astype("uint8"), {
        "enabled": True,
        "water_pixels_before": water_pixels_before,
        "water_pixels_after": retained_pixels,
        "removed_water_pixels": water_pixels_before - retained_pixels,
        "removed_component_count": int(n_components - 1),
        "retained_component_label": retained_label,
        "retained_component_pixels": retained_pixels,
    }


def cleanup_and_export_main_river_small_land(
    cleaned: np.ndarray,
    *,
    transform: object,
    crs: object,
    tiny_land_remove_max_pixels: int,
    tiny_land_export_min_pixels: int,
    tiny_land_export_max_pixels: int,
    tiny_land_export_path: Path | None,
) -> tuple[np.ndarray, dict[str, object]]:
    cleaned = cleaned.copy()
    structure = np.ones((3, 3), dtype=int)
    water = cleaned.astype(bool)
    water_labels, n_water = ndimage.label(water, structure=structure)
    if n_water == 0:
        return cleaned, {
            "enabled": True,
            "remove_max_pixels": tiny_land_remove_max_pixels,
            "export_min_pixels": tiny_land_export_min_pixels,
            "export_max_pixels": tiny_land_export_max_pixels,
            "reason": "no_water_components",
            "removed_component_count": 0,
            "removed_pixel_count": 0,
            "export_component_count": 0,
            "export_path": str(tiny_land_export_path) if tiny_land_export_path is not None else None,
        }

    water_sizes = np.bincount(water_labels.ravel())[1:]
    main_water_label = int(np.argmax(water_sizes) + 1)
    main_water = water_labels == main_water_label

    removed_component_count = 0
    removed_pixel_count = 0
    if tiny_land_remove_max_pixels >= 0:
        land = ~water
        land_labels, land_sizes, border_labels, adjacent_labels = identify_main_river_land_components(
            land=land,
            main_water=main_water,
            structure=structure,
        )
        remove_labels = [
            label_id
            for label_id in adjacent_labels
            if label_id not in border_labels and int(land_sizes[label_id - 1]) <= tiny_land_remove_max_pixels
        ]
        if remove_labels:
            remove_mask = np.isin(land_labels, remove_labels)
            removed_component_count = len(remove_labels)
            removed_pixel_count = int(remove_mask.sum())
            cleaned[remove_mask] = 1

    export_component_count = 0
    export_path_str: str | None = None
    if tiny_land_export_path is not None and tiny_land_export_max_pixels >= 0 and tiny_land_export_max_pixels >= tiny_land_export_min_pixels:
        export_gdf = extract_main_river_small_land_polygons(
            cleaned,
            transform=transform,
            crs=crs,
            min_pixels=tiny_land_export_min_pixels,
            max_pixels=tiny_land_export_max_pixels,
        )
        tiny_land_export_path.parent.mkdir(parents=True, exist_ok=True)
        if tiny_land_export_path.exists():
            tiny_land_export_path.unlink()
        export_gdf.to_file(tiny_land_export_path, layer="main_river_small_land", driver="GPKG")
        export_component_count = int(len(export_gdf))
        export_path_str = str(tiny_land_export_path)

    return cleaned, {
        "enabled": True,
        "main_water_label": main_water_label,
        "main_water_pixels": int(water_sizes[main_water_label - 1]),
        "remove_max_pixels": tiny_land_remove_max_pixels,
        "export_min_pixels": tiny_land_export_min_pixels,
        "export_max_pixels": tiny_land_export_max_pixels,
        "removed_component_count": removed_component_count,
        "removed_pixel_count": removed_pixel_count,
        "export_component_count": export_component_count,
        "export_path": export_path_str,
    }


def identify_main_river_land_components(
    *,
    land: np.ndarray,
    main_water: np.ndarray,
    structure: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, set[int], list[int]]:
    land_labels, n_land = ndimage.label(land, structure=structure)
    if n_land == 0:
        return land_labels, np.array([], dtype=int), set(), []
    land_sizes = np.bincount(land_labels.ravel())[1:]
    border_labels = set(
        int(v)
        for v in np.unique(
            np.concatenate([land_labels[0, :], land_labels[-1, :], land_labels[:, 0], land_labels[:, -1]])
        )
        if int(v) > 0
    )
    main_water_dilated = ndimage.binary_dilation(main_water, structure=structure)
    adjacent_labels = [
        int(v)
        for v in np.unique(land_labels[main_water_dilated & land])
        if int(v) > 0
    ]
    return land_labels, land_sizes, border_labels, adjacent_labels


def extract_main_river_small_land_polygons(
    cleaned: np.ndarray,
    *,
    transform: object,
    crs: object,
    min_pixels: int,
    max_pixels: int,
) -> gpd.GeoDataFrame:
    structure = np.ones((3, 3), dtype=int)
    water = cleaned.astype(bool)
    water_labels, n_water = ndimage.label(water, structure=structure)
    if n_water == 0:
        return _empty_small_land_gdf(crs)
    water_sizes = np.bincount(water_labels.ravel())[1:]
    main_water_label = int(np.argmax(water_sizes) + 1)
    main_water = water_labels == main_water_label
    land = ~water
    land_labels, land_sizes, border_labels, adjacent_labels = identify_main_river_land_components(
        land=land,
        main_water=main_water,
        structure=structure,
    )
    selected = [
        label_id
        for label_id in adjacent_labels
        if label_id not in border_labels and min_pixels <= int(land_sizes[label_id - 1]) <= max_pixels
    ]
    if not selected:
        return _empty_small_land_gdf(crs)

    selected_mask = np.isin(land_labels, selected)
    selected_labels = np.where(selected_mask, land_labels, 0).astype(np.int32)
    parts: list[dict[str, object]] = []
    for geom_mapping, value in shapes(selected_labels, mask=selected_mask, transform=transform):
        label_id = int(value)
        if label_id <= 0:
            continue
        parts.append({"component_id": label_id, "geometry": shape(geom_mapping)})

    parts_gdf = gpd.GeoDataFrame(parts, geometry="geometry", crs=crs)
    pixel_area = abs(float(transform.a)) * abs(float(transform.e))
    records: list[dict[str, object]] = []
    for component_id, group in parts_gdf.groupby("component_id", sort=True):
        geom = unary_union(group.geometry.tolist())
        size_pixels = int(land_sizes[int(component_id) - 1])
        records.append(
            {
                "component_id": int(component_id),
                "size_pixels": size_pixels,
                "size_m2": float(size_pixels * pixel_area),
                "geometry": geom,
            }
        )
    return gpd.GeoDataFrame(records, geometry="geometry", crs=crs).sort_values(
        ["size_pixels", "component_id"], kind="mergesort"
    ).reset_index(drop=True)


def export_main_river_diagonal_passage_points(
    cleaned: np.ndarray,
    *,
    transform: object,
    crs: object,
    export_path: Path | None,
) -> dict[str, object]:
    if export_path is None:
        return {
            "enabled": False,
            "count": 0,
            "export_path": None,
        }

    structure = np.ones((3, 3), dtype=int)
    water = cleaned.astype(bool)
    water_labels, n_water = ndimage.label(water, structure=structure)
    if n_water == 0:
        empty = _empty_diagonal_passage_gdf(crs)
        export_path.parent.mkdir(parents=True, exist_ok=True)
        if export_path.exists():
            export_path.unlink()
        empty.to_file(export_path, layer="main_river_diagonal_passages", driver="GPKG")
        return {
            "enabled": True,
            "count": 0,
            "main_water_label": None,
            "export_path": str(export_path),
        }

    water_sizes = np.bincount(water_labels.ravel())[1:]
    main_water_label = int(np.argmax(water_sizes) + 1)
    main_water = water_labels == main_water_label
    land = ~water

    land_nw = land[:-1, :-1]
    land_ne = land[:-1, 1:]
    land_sw = land[1:, :-1]
    land_se = land[1:, 1:]
    water_nw = main_water[:-1, :-1]
    water_ne = main_water[:-1, 1:]
    water_sw = main_water[1:, :-1]
    water_se = main_water[1:, 1:]

    pattern_nw_se = land_nw & land_se & (~land_ne) & (~land_sw) & water_ne & water_sw
    pattern_ne_sw = land_ne & land_sw & (~land_nw) & (~land_se) & water_nw & water_se

    records: list[dict[str, object]] = []
    for row, col in np.argwhere(pattern_nw_se):
        x, y = transform * (col + 1, row + 1)
        records.append(
            {
                "spot_id": len(records) + 1,
                "row": int(row),
                "col": int(col),
                "pattern_type": "land_nw_se",
                "geometry": Point(float(x), float(y)),
            }
        )
    for row, col in np.argwhere(pattern_ne_sw):
        x, y = transform * (col + 1, row + 1)
        records.append(
            {
                "spot_id": len(records) + 1,
                "row": int(row),
                "col": int(col),
                "pattern_type": "land_ne_sw",
                "geometry": Point(float(x), float(y)),
            }
        )

    gdf = gpd.GeoDataFrame(records, geometry="geometry", crs=crs) if records else _empty_diagonal_passage_gdf(crs)
    export_path.parent.mkdir(parents=True, exist_ok=True)
    if export_path.exists():
        export_path.unlink()
    gdf.to_file(export_path, layer="main_river_diagonal_passages", driver="GPKG")
    return {
        "enabled": True,
        "count": int(len(gdf)),
        "main_water_label": main_water_label,
        "main_water_pixels": int(water_sizes[main_water_label - 1]),
        "export_path": str(export_path),
    }


def _empty_small_land_gdf(crs: object) -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(
        {
            "component_id": pd.Series(dtype="int64"),
            "size_pixels": pd.Series(dtype="int64"),
            "size_m2": pd.Series(dtype="float64"),
        },
        geometry=gpd.GeoSeries([], crs=crs),
        crs=crs,
    )


def _empty_diagonal_passage_gdf(crs: object) -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(
        {
            "spot_id": pd.Series(dtype="int64"),
            "row": pd.Series(dtype="int64"),
            "col": pd.Series(dtype="int64"),
            "pattern_type": pd.Series(dtype="string"),
        },
        geometry=gpd.GeoSeries([], crs=crs),
        crs=crs,
    )


def normalize_action(value: object) -> str:
    key = str(value).strip().lower()
    if key not in ACTION_ALIASES:
        allowed = ", ".join(sorted(ACTION_ALIASES))
        raise ValueError(f"Unsupported manual edit action '{value}'. Expected one of: {allowed}")
    return ACTION_ALIASES[key]


if __name__ == "__main__":
    main()
