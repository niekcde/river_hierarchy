from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.features import rasterize
from rasterio.transform import rowcol
from rasterio.transform import xy as raster_xy
from scipy.ndimage import binary_dilation, find_objects, label
from shapely.geometry import LineString, Point


def _example_prefix(cleaned_mask_path: Path) -> str:
    stem = cleaned_mask_path.stem
    if stem.endswith("_cleaned"):
        return stem[: -len("_cleaned")]
    return stem


def _load_single_centerline(centerline_path: Path) -> tuple[gpd.GeoDataFrame, LineString]:
    centerline = gpd.read_file(centerline_path)
    if len(centerline) != 1:
        raise ValueError(f"Expected exactly one centerline feature, found {len(centerline)} in {centerline_path}.")
    geometry = centerline.geometry.iloc[0]
    if geometry.geom_type != "LineString":
        raise ValueError(f"Expected a LineString centerline, found {geometry.geom_type} in {centerline_path}.")
    return centerline, geometry


def _main_water_component(water_mask: np.ndarray, transform: object, centerline: LineString) -> tuple[int, np.ndarray]:
    labels, _ = label(water_mask.astype(np.uint8), structure=np.ones((3, 3), dtype=np.uint8))
    coords = list(centerline.coords)
    midpoint = Point(coords[len(coords) // 2])
    row, col = rowcol(transform, midpoint.x, midpoint.y)
    if 0 <= row < labels.shape[0] and 0 <= col < labels.shape[1] and labels[row, col] > 0:
        component_id = int(labels[row, col])
    else:
        values, counts = np.unique(labels[labels > 0], return_counts=True)
        if len(values) == 0:
            raise ValueError("No connected water component was found in the cleaned mask.")
        component_id = int(values[np.argmax(counts)])
    return component_id, labels == component_id


def _nearest_point(points: np.ndarray, target_xy: np.ndarray) -> tuple[np.ndarray, float]:
    deltas = points - target_xy[None, :]
    squared = np.sum(deltas * deltas, axis=1)
    index = int(np.argmin(squared))
    return points[index], float(np.sqrt(squared[index]))


def _nearest_border_target(point: Point, bounds: object, pixel_length: float) -> tuple[str, Point, float]:
    epsilon = float(pixel_length) * 1.5
    candidates = [
        ("left", Point(bounds.left - epsilon, point.y), abs(point.x - bounds.left)),
        ("right", Point(bounds.right + epsilon, point.y), abs(bounds.right - point.x)),
        ("bottom", Point(point.x, bounds.bottom - epsilon), abs(point.y - bounds.bottom)),
        ("top", Point(point.x, bounds.top + epsilon), abs(bounds.top - point.y)),
    ]
    return min(candidates, key=lambda item: item[2])


def build_boundary_component_mask(
    *,
    cleaned_mask_path: str | Path,
    centerline_path: str | Path,
    output_dir: str | Path,
    extension_buffer_pixels: int = 1,
) -> dict[str, str]:
    cleaned_mask_path = Path(cleaned_mask_path)
    centerline_path = Path(centerline_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    prefix = _example_prefix(cleaned_mask_path)

    extensions_path = output_dir / f"{prefix}_centerline_border_extensions_snapped.gpkg"
    buffered_extension_mask_path = output_dir / f"{prefix}_border_extensions_snapped_buffered_mask.tif"
    extended_water_mask_path = output_dir / f"{prefix}_extended_water_to_border_mask_snapped_buffered.tif"
    labels_path = output_dir / f"{prefix}_full_extent_land_water_labels_with_snapped_buffered_border_cut.tif"
    summary_path = output_dir / f"{prefix}_full_extent_land_component_summary_with_snapped_buffered_border_cut.csv"

    with rasterio.open(cleaned_mask_path) as src:
        raw = src.read(1)
        profile = src.profile.copy()
        transform = src.transform
        bounds = src.bounds

    water_mask = raw > 0
    pixel_length = float(max(abs(float(transform.a)), abs(float(transform.e))))

    centerline_gdf, centerline = _load_single_centerline(centerline_path)
    component_id, main_mask = _main_water_component(water_mask, transform, centerline)
    main_rows, main_cols = np.nonzero(main_mask)
    main_xs, main_ys = raster_xy(transform, main_rows, main_cols, offset="center")
    main_coords = np.column_stack([main_xs, main_ys]).astype(float)

    endpoints = [Point(centerline.coords[0]), Point(centerline.coords[-1])]
    extension_records: list[dict[str, object]] = []
    shapes: list[tuple[LineString, int]] = []
    for endpoint_id, endpoint in enumerate(endpoints, start=1):
        endpoint_xy = np.asarray([endpoint.x, endpoint.y], dtype=float)
        snapped_xy, snap_distance = _nearest_point(main_coords, endpoint_xy)
        side, border_point, border_distance = _nearest_border_target(Point(*snapped_xy), bounds, pixel_length)
        line = LineString([tuple(snapped_xy), (border_point.x, border_point.y)])
        extension_records.append(
            {
                "endpoint_id": endpoint_id,
                "centerline_endpoint_x": float(endpoint.x),
                "centerline_endpoint_y": float(endpoint.y),
                "snap_x": float(snapped_xy[0]),
                "snap_y": float(snapped_xy[1]),
                "snap_distance_to_centerline_endpoint": float(snap_distance),
                "border_side": side,
                "border_x": float(border_point.x),
                "border_y": float(border_point.y),
                "distance_to_border_from_snap": float(border_distance),
                "geometry": line,
            }
        )
        shapes.append((line, 1))

    extensions_gdf = gpd.GeoDataFrame(extension_records, geometry="geometry", crs=centerline_gdf.crs)
    extensions_gdf.to_file(extensions_path, driver="GPKG")

    extension_mask = rasterize(
        shapes,
        out_shape=water_mask.shape,
        transform=transform,
        fill=0,
        default_value=1,
        all_touched=True,
        dtype="uint8",
    ).astype(bool)
    if extension_buffer_pixels > 0:
        structure = np.ones((2 * extension_buffer_pixels + 1, 2 * extension_buffer_pixels + 1), dtype=bool)
        extension_mask = binary_dilation(extension_mask, structure=structure)

    extended_water = water_mask | extension_mask
    land_mask = ~extended_water
    structure4 = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.uint8)
    land_labels, _ = label(land_mask.astype(np.uint8), structure=structure4)

    final = np.ones(raw.shape, dtype=np.int32)
    final[land_mask] = land_labels[land_mask].astype(np.int32) + 1

    mask_profile = profile.copy()
    mask_profile.update(dtype="uint8", count=1, nodata=0, compress="lzw")
    with rasterio.open(buffered_extension_mask_path, "w", **mask_profile) as dst:
        dst.write(extension_mask.astype("uint8"), 1)
    with rasterio.open(extended_water_mask_path, "w", **mask_profile) as dst:
        dst.write(extended_water.astype("uint8"), 1)

    labels_profile = profile.copy()
    labels_profile.update(dtype="int32", count=1, nodata=0, compress="lzw")
    with rasterio.open(labels_path, "w", **labels_profile) as dst:
        dst.write(final, 1)

    component_objects = find_objects(land_labels)
    summary_records: list[dict[str, object]] = []
    for component_index, component_slice in enumerate(component_objects, start=1):
        if component_slice is None:
            continue
        component_mask = land_labels[component_slice] == component_index
        rows, cols = np.nonzero(component_mask)
        if len(rows) == 0:
            continue
        absolute_rows = rows + component_slice[0].start
        absolute_cols = cols + component_slice[1].start
        xs, ys = raster_xy(transform, absolute_rows, absolute_cols, offset="center")
        summary_records.append(
            {
                "land_component_id": int(component_index),
                "raster_value": int(component_index + 1),
                "n_pixels": int(len(rows)),
                "row_min": int(absolute_rows.min()),
                "row_max": int(absolute_rows.max()),
                "col_min": int(absolute_cols.min()),
                "col_max": int(absolute_cols.max()),
                "x_center_mean": float(np.mean(xs)),
                "y_center_mean": float(np.mean(ys)),
            }
        )

    summary = pd.DataFrame.from_records(summary_records).sort_values("land_component_id").reset_index(drop=True)
    summary.to_csv(summary_path, index=False)

    return {
        "main_component_id": str(component_id),
        "extensions": str(extensions_path.resolve()),
        "buffered_extension_mask": str(buffered_extension_mask_path.resolve()),
        "extended_water_mask": str(extended_water_mask_path.resolve()),
        "labels": str(labels_path.resolve()),
        "summary": str(summary_path.resolve()),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Create a full-extent land/water label mask from a cleaned binary mask, "
            "using snapped centerline endpoint-to-border extensions as a cut barrier."
        )
    )
    parser.add_argument("--cleaned-mask", required=True, help="Path to the cleaned binary water mask.")
    parser.add_argument("--centerline", required=True, help="Path to the single-edge reference centerline.")
    parser.add_argument("--output-dir", required=True, help="Directory for output rasters and summaries.")
    parser.add_argument(
        "--extension-buffer-pixels",
        type=int,
        default=1,
        help="Number of raster pixels used to buffer the endpoint-to-border cut lines.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    outputs = build_boundary_component_mask(
        cleaned_mask_path=args.cleaned_mask,
        centerline_path=args.centerline,
        output_dir=args.output_dir,
        extension_buffer_pixels=args.extension_buffer_pixels,
    )
    for key, value in outputs.items():
        print(f"{key}: {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
