"""Create and validate editable panel layouts for large RivGraph masks."""

from __future__ import annotations

import json
from pathlib import Path

import geopandas as gpd
import rasterio
from rasterio.features import rasterize
from rasterio.windows import from_bounds
from shapely.geometry import LineString, box
from shapely.ops import polygonize, unary_union

from rivgraph_centerline.raster_mask import crop_binary_mask


PANEL_BREAK_LAYER = "panel_breaks"


def create_panel_breaks(
    *, mask_path: Path, output_path: Path, rows: int = 4, columns: int = 4, overwrite: bool = False
) -> dict[str, object]:
    """Create evenly spaced editable break transects covering a mask extent."""
    if rows < 2 or columns < 2:
        raise ValueError("rows and columns must both be at least 2.")
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Panel-break file already exists: {output_path}")
    with rasterio.open(mask_path) as src:
        bounds, crs = src.bounds, src.crs
    records: list[dict[str, object]] = []
    for index in range(1, columns):
        fraction = index / columns
        x = bounds.left + (bounds.right - bounds.left) * fraction
        records.append({"break_id": f"V{index}", "orientation": "vertical", "active": True,
                        "notes": "Provisional divider; move to a simple single-link reach.",
                        "geometry": LineString([(x, bounds.bottom), (x, bounds.top)])})
    for index in range(1, rows):
        fraction = index / rows
        y = bounds.bottom + (bounds.top - bounds.bottom) * fraction
        records.append({"break_id": f"H{index}", "orientation": "horizontal", "active": True,
                        "notes": "Provisional divider; move to a simple single-link reach.",
                        "geometry": LineString([(bounds.left, y), (bounds.right, y)])})
    output_path.parent.mkdir(parents=True, exist_ok=True)
    gpd.GeoDataFrame(records, geometry="geometry", crs=crs).to_file(output_path, layer=PANEL_BREAK_LAYER, driver="GPKG")
    return {"output": str(output_path), "layer": PANEL_BREAK_LAYER, "break_count": len(records), "rows": rows, "columns": columns}


def plan_panels(
    *, mask_path: Path, breaks_path: Path, output_path: Path, overlap_m: float = 5000, max_panel_pixels: int = 160_000_000, overwrite: bool = False
) -> dict[str, object]:
    """Derive irregular core and buffered processing polygons from edited breaks."""
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Panel plan already exists: {output_path}")
    with rasterio.open(mask_path) as src:
        bounds, crs, transform = src.bounds, src.crs, src.transform
    extent = box(bounds.left, bounds.bottom, bounds.right, bounds.top)
    breaks = gpd.read_file(breaks_path, layer=PANEL_BREAK_LAYER).to_crs(crs)
    active = breaks.loc[breaks["active"].astype(bool)].copy()
    lines = [geom.intersection(extent) for geom in active.geometry if geom is not None and not geom.is_empty]
    polygons = list(polygonize(unary_union([extent.boundary, *lines])))
    if len(polygons) < 2:
        raise ValueError("Active break lines do not partition the mask extent into multiple panels.")
    polygons.sort(key=lambda geom: (-geom.centroid.y, geom.centroid.x))
    pixel_area = abs(transform.a * transform.e)
    records = []
    for index, core in enumerate(polygons, start=1):
        processing = core.buffer(overlap_m).intersection(extent)
        estimated_pixels = int(round(processing.area / pixel_area))
        records.append({"panel_id": f"P{index:03d}", "estimated_processing_pixels": estimated_pixels,
                        "within_pixel_budget": estimated_pixels <= max_panel_pixels, "geometry": core})
    cores = gpd.GeoDataFrame(records, geometry="geometry", crs=crs)
    processing = cores.copy()
    processing["geometry"] = [geom.buffer(overlap_m).intersection(extent) for geom in cores.geometry]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cores.to_file(output_path, layer="panel_cores", driver="GPKG")
    processing.to_file(output_path, layer="processing_areas", driver="GPKG")
    summary = {"mask": str(mask_path), "breaks": str(breaks_path), "output": str(output_path), "panel_count": len(cores), "overlap_m": overlap_m,
               "max_panel_pixels": max_panel_pixels, "over_budget_panel_ids": cores.loc[~cores["within_pixel_budget"], "panel_id"].tolist()}
    output_path.with_suffix(".json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def prepare_panel_masks(*, mask_path: Path, plan_path: Path, output_dir: Path, panel_ids: set[str] | None = None, overwrite: bool = False) -> dict[str, object]:
    """Clip buffered processing areas to independently runnable binary masks."""
    cores = gpd.read_file(plan_path, layer="panel_cores")
    processing = gpd.read_file(plan_path, layer="processing_areas")
    selected = processing if panel_ids is None else processing.loc[processing["panel_id"].isin(panel_ids)]
    missing = set() if panel_ids is None else panel_ids.difference(set(processing["panel_id"]))
    if missing:
        raise ValueError(f"Unknown panel_id(s): {sorted(missing)}")
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest: list[dict[str, object]] = []
    with rasterio.open(mask_path) as src:
        for record in selected.itertuples(index=False):
            panel_dir = output_dir / record.panel_id
            panel_mask = panel_dir / "mask.tif"
            if panel_mask.exists() and not overwrite:
                raise FileExistsError(f"Panel mask already exists: {panel_mask}")
            geometry = record.geometry
            window = from_bounds(*geometry.bounds, transform=src.transform).round_offsets().round_lengths()
            window = window.intersection(rasterio.windows.Window(0, 0, src.width, src.height))
            data = src.read(1, window=window)
            local_transform = src.window_transform(window)
            keep = rasterize([(geometry, 1)], out_shape=data.shape, transform=local_transform, fill=0, dtype="uint8").astype(bool)
            clipped = ((data > 0) & keep).astype("uint8")
            cropped, cropped_transform, crop_window = crop_binary_mask(clipped, local_transform)
            profile = src.profile.copy()
            profile.update(driver="GTiff", dtype="uint8", count=1, nodata=0, compress="lzw", height=cropped.shape[0], width=cropped.shape[1], transform=cropped_transform)
            panel_dir.mkdir(parents=True, exist_ok=True)
            with rasterio.open(panel_mask, "w", **profile) as dst:
                dst.write(cropped, 1)
            core = cores.loc[cores["panel_id"] == record.panel_id].geometry.iloc[0]
            summary = {"panel_id": record.panel_id, "mask": str(panel_mask), "source_mask": str(mask_path), "water_pixels": int(cropped.sum()),
                       "shape": [int(cropped.shape[0]), int(cropped.shape[1])], "crop_window": crop_window,
                       "core_bounds": list(core.bounds), "processing_bounds": list(geometry.bounds)}
            (panel_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
            manifest.append(summary)
    manifest_path = output_dir / "panel_mask_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return {"output_dir": str(output_dir), "manifest": str(manifest_path), "prepared_panel_ids": [item["panel_id"] for item in manifest]}
