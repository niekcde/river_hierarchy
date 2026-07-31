from __future__ import annotations

import argparse
from itertools import combinations
import json
from pathlib import Path
from typing import Sequence

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.features import rasterize
from rasterio.transform import rowcol
from rasterio.transform import xy as raster_xy
from rasterio.windows import Window, from_bounds
from scipy.ndimage import binary_dilation, find_objects, label
from shapely.geometry import LineString, Point


class _ComponentUnion:
    """Disk-label component union helper for the panelwise boundary builder."""

    def __init__(self) -> None:
        self._parent: dict[int, int] = {}

    def add(self, value: int) -> None:
        self._parent.setdefault(int(value), int(value))

    def find(self, value: int) -> int:
        value = int(value)
        parent = self._parent[value]
        if parent != value:
            self._parent[value] = self.find(parent)
        return self._parent[value]

    def union(self, left: int, right: int) -> None:
        self.add(left); self.add(right)
        left_root, right_root = self.find(left), self.find(right)
        if left_root != right_root:
            self._parent[max(left_root, right_root)] = min(left_root, right_root)


def _outward_window(bounds: tuple[float, float, float, float], transform: object) -> Window:
    """Return an integer raster window that fully contains geographic bounds."""
    raw = from_bounds(*bounds, transform=transform)
    tolerance = 1e-6
    col_start = int(np.floor(float(raw.col_off) + tolerance))
    row_start = int(np.floor(float(raw.row_off) + tolerance))
    col_stop = int(np.ceil(float(raw.col_off + raw.width) - tolerance))
    row_stop = int(np.ceil(float(raw.row_off + raw.height) - tolerance))
    return Window(col_start, row_start, col_stop - col_start, row_stop - row_start)


def _label_panel_land(water_mask: np.ndarray) -> tuple[np.ndarray, int]:
    """Label dry components in one panel using the legacy four-neighbour rule."""
    structure4 = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.uint8)
    labels, count = label((~water_mask).astype(np.uint8), structure=structure4)
    return labels.astype(np.int32, copy=False), int(count)


def _union_overlapping_panel_labels(
    components: _ComponentUnion,
    left_labels: np.ndarray,
    right_labels: np.ndarray,
    *,
    left_offset: int,
    right_offset: int,
) -> None:
    """Union dry components that occupy the same pixels in aligned panel overlap."""
    if left_labels.shape != right_labels.shape:
        raise ValueError("Aligned panel-overlap label arrays must have the same shape.")
    shared = (left_labels > 0) & (right_labels > 0)
    if not np.any(shared):
        return
    pairs = np.unique(
        np.column_stack(
            [left_labels[shared].astype(np.int64) + left_offset, right_labels[shared].astype(np.int64) + right_offset]
        ),
        axis=0,
    )
    for left, right in pairs:
        components.union(int(left), int(right))


def _resolved_component_ids(components: _ComponentUnion) -> dict[int, int]:
    """Map union roots to deterministic positive global land-component IDs."""
    roots = sorted({components.find(value) for value in components._parent})
    root_ids = {root: index for index, root in enumerate(roots, start=1)}
    return {value: root_ids[components.find(value)] for value in components._parent}


def _write_panel_land_labels(
    mask_path: Path,
    output_path: Path,
    *,
    border_cut_lines: Sequence[LineString] = (),
    extension_buffer_pixels: int = 1,
    tile_size: int = 1024,
) -> int:
    """Persist compact panel-local labels using bounded-memory raster tiles."""
    if tile_size <= 0:
        raise ValueError("tile_size must be positive.")
    raw_path = output_path.with_name(f"{output_path.stem}_raw{output_path.suffix}")
    components = _ComponentUnion()
    next_label = 0
    with rasterio.open(mask_path) as source:
        profile = source.profile.copy()
        raw_profile = profile.copy()
        raw_profile.update(dtype="int32", count=1, nodata=0, compress="lzw")
        if not raw_profile.get("tiled", False):
            raw_profile.pop("blockxsize", None)
            raw_profile.pop("blockysize", None)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(raw_path, "w+", **raw_profile) as raw:
            for row_off in range(0, source.height, tile_size):
                height = min(tile_size, source.height - row_off)
                for col_off in range(0, source.width, tile_size):
                    width = min(tile_size, source.width - col_off)
                    window = Window(col_off, row_off, width, height)
                    halo = max(int(extension_buffer_pixels), 0)
                    expanded = Window(
                        max(col_off - halo, 0),
                        max(row_off - halo, 0),
                        min(col_off + width + halo, source.width) - max(col_off - halo, 0),
                        min(row_off + height + halo, source.height) - max(row_off - halo, 0),
                    )
                    water = source.read(1, window=expanded) > 0
                    if border_cut_lines:
                        cut = rasterize(
                            [(line, 1) for line in border_cut_lines],
                            out_shape=(int(expanded.height), int(expanded.width)),
                            transform=source.window_transform(expanded),
                            dtype="uint8",
                            all_touched=True,
                        ).astype(bool)
                        if extension_buffer_pixels > 0:
                            cut = binary_dilation(
                                cut,
                                structure=np.ones(
                                    (2 * extension_buffer_pixels + 1, 2 * extension_buffer_pixels + 1),
                                    dtype=bool,
                                ),
                            )
                        water |= cut
                    local_row = int(row_off - expanded.row_off)
                    local_col = int(col_off - expanded.col_off)
                    water = water[local_row : local_row + height, local_col : local_col + width]
                    tile_labels, count = _label_panel_land(water)
                    if count:
                        positive = tile_labels > 0
                        tile_labels[positive] += next_label
                        for component_id in range(next_label + 1, next_label + count + 1):
                            components.add(component_id)
                        next_label += count
                    if row_off > 0:
                        above = raw.read(1, window=Window(col_off, row_off - 1, width, 1))[0]
                        current = tile_labels[0]
                        for left, right in np.unique(
                            np.column_stack([above[(above > 0) & (current > 0)], current[(above > 0) & (current > 0)]]),
                            axis=0,
                        ):
                            components.union(int(left), int(right))
                    if col_off > 0:
                        left_edge = raw.read(1, window=Window(col_off - 1, row_off, 1, height))[:, 0]
                        current = tile_labels[:, 0]
                        for left, right in np.unique(
                            np.column_stack(
                                [left_edge[(left_edge > 0) & (current > 0)], current[(left_edge > 0) & (current > 0)]]
                            ),
                            axis=0,
                        ):
                            components.union(int(left), int(right))
                    raw.write(tile_labels, 1, window=window)

    resolved = _resolved_component_ids(components)
    profile.update(dtype="int32", count=1, nodata=0, compress="lzw")
    if not profile.get("tiled", False):
        profile.pop("blockxsize", None)
        profile.pop("blockysize", None)
    with rasterio.open(raw_path) as raw, rasterio.open(output_path, "w", **profile) as destination:
        for _, window in raw.block_windows(1):
            values = raw.read(1, window=window)
            compact = np.zeros(values.shape, dtype=np.int32)
            for value in np.unique(values[values > 0]):
                compact[values == value] = resolved[int(value)]
            destination.write(compact, 1, window=window)
    raw_path.unlink()
    return len(set(resolved.values()))


def _write_processing_area_mask(
    *,
    cleaned_mask_path: Path,
    processing_bounds: tuple[float, float, float, float],
    output_path: Path,
    tile_size: int = 1024,
) -> None:
    """Stream one full processing-area mask from the base raster."""
    with rasterio.open(cleaned_mask_path) as source:
        source_window = _outward_window(processing_bounds, source.transform).intersection(
            Window(0, 0, source.width, source.height)
        )
        profile = source.profile.copy()
        profile.update(
            height=int(source_window.height),
            width=int(source_window.width),
            transform=source.window_transform(source_window),
            dtype="uint8",
            count=1,
            nodata=0,
            compress="lzw",
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(output_path, "w", **profile) as destination:
            for row_delta in range(0, int(source_window.height), tile_size):
                height = min(tile_size, int(source_window.height) - row_delta)
                for col_delta in range(0, int(source_window.width), tile_size):
                    width = min(tile_size, int(source_window.width) - col_delta)
                    source_tile = Window(
                        source_window.col_off + col_delta,
                        source_window.row_off + row_delta,
                        width,
                        height,
                    )
                    destination.write(
                        source.read(1, window=source_tile).astype("uint8", copy=False),
                        1,
                        window=Window(col_delta, row_delta, width, height),
                    )


def _materialize_processing_area_masks(
    *,
    cleaned_mask_path: Path,
    processing_areas: gpd.GeoDataFrame,
    output_dir: Path,
    tile_size: int = 1024,
) -> dict[str, Path]:
    """Create aligned, full processing-area masks with dry fill outside water crops."""
    masks: dict[str, Path] = {}
    for row in processing_areas.sort_values("panel_id").itertuples(index=False):
        panel_id = str(row.panel_id)
        path = output_dir / f"{panel_id}_processing_mask.tif"
        _write_processing_area_mask(
            cleaned_mask_path=cleaned_mask_path,
            processing_bounds=row.geometry.bounds,
            output_path=path,
            tile_size=tile_size,
        )
        masks[panel_id] = path
    return masks


def _union_panel_label_overlap(
    components: _ComponentUnion,
    left_path: Path,
    right_path: Path,
    *,
    left_offset: int,
    right_offset: int,
    tile_size: int = 1024,
) -> None:
    """Union coincident land labels by streaming bounded overlap tiles."""
    if tile_size <= 0:
        raise ValueError("tile_size must be positive.")
    with rasterio.open(left_path) as left, rasterio.open(right_path) as right:
        west, south = max(left.bounds.left, right.bounds.left), max(left.bounds.bottom, right.bounds.bottom)
        east, north = min(left.bounds.right, right.bounds.right), min(left.bounds.top, right.bounds.top)
        if west >= east or south >= north:
            return
        left_window = from_bounds(west, south, east, north, transform=left.transform).round_offsets().round_lengths()
        right_window = from_bounds(west, south, east, north, transform=right.transform).round_offsets().round_lengths()
        if (int(left_window.height), int(left_window.width)) != (
            int(right_window.height),
            int(right_window.width),
        ):
            raise ValueError("Panel label overlaps are not aligned to the same raster grid.")
        for row_delta in range(0, int(left_window.height), tile_size):
            height = min(tile_size, int(left_window.height) - row_delta)
            for col_delta in range(0, int(left_window.width), tile_size):
                width = min(tile_size, int(left_window.width) - col_delta)
                left_tile = Window(left_window.col_off + col_delta, left_window.row_off + row_delta, width, height)
                right_tile = Window(right_window.col_off + col_delta, right_window.row_off + row_delta, width, height)
                _union_overlapping_panel_labels(
                    components,
                    left.read(1, window=left_tile),
                    right.read(1, window=right_tile),
                    left_offset=left_offset,
                    right_offset=right_offset,
                )


def _resolve_panel_component_ids(
    panel_masks: dict[str, Path],
    work_dir: Path,
    *,
    border_cut_lines: Sequence[LineString] = (),
    tile_size: int = 1024,
    overlap_pairs: Sequence[tuple[str, str]] | None = None,
) -> tuple[dict[str, Path], dict[str, int], dict[int, int]]:
    """Build disk-backed local labels and resolve their IDs across all overlaps."""
    labels: dict[str, Path] = {}
    offsets: dict[str, int] = {}
    components = _ComponentUnion()
    next_offset = 0
    for panel_id, mask_path in sorted(panel_masks.items()):
        label_path = work_dir / f"{panel_id}_local_land_labels.tif"
        count = _write_panel_land_labels(
            mask_path,
            label_path,
            border_cut_lines=border_cut_lines,
            tile_size=tile_size,
        )
        offsets[panel_id] = next_offset
        for local_id in range(1, count + 1):
            components.add(next_offset + local_id)
        next_offset += count
        labels[panel_id] = label_path
    pairs = combinations(sorted(labels), 2) if overlap_pairs is None else overlap_pairs
    for left_id, right_id in pairs:
        _union_panel_label_overlap(
            components, labels[left_id], labels[right_id],
            left_offset=offsets[left_id], right_offset=offsets[right_id],
            tile_size=tile_size,
        )
    return labels, offsets, _resolved_component_ids(components)


def _apply_resolved_ids(local_labels: np.ndarray, *, offset: int, resolved_ids: dict[int, int]) -> np.ndarray:
    """Convert one local label array to final land/water raster values."""
    result = np.ones(local_labels.shape, dtype=np.int32)
    local_values = np.unique(local_labels[local_labels > 0])
    for value in local_values:
        result[local_labels == value] = resolved_ids[int(offset + value)] + 1
    return result


def _write_resolved_label_cores(
    *,
    output_path: Path,
    profile: dict[str, object],
    panel_cores: dict[str, tuple[Path, int, Window, Window, object]],
    resolved_ids: dict[int, int],
    tile_size: int = 1024,
) -> None:
    """Stream resolved panel core labels into one full-extent output raster."""
    if tile_size <= 0:
        raise ValueError("tile_size must be positive.")
    output_profile = dict(profile)
    output_profile.update(dtype="int32", count=1, nodata=0, compress="lzw")
    if not output_profile.get("tiled", False):
        output_profile.pop("blockxsize", None)
        output_profile.pop("blockysize", None)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(output_path, "w+", **output_profile) as destination:
        for panel_id in sorted(panel_cores):
            label_path, offset, source_window, destination_window, core_geometry = panel_cores[panel_id]
            if (int(source_window.height), int(source_window.width)) != (
                int(destination_window.height),
                int(destination_window.width),
            ):
                raise ValueError(f"Panel {panel_id} core windows are not aligned.")
            with rasterio.open(label_path) as source:
                for row_delta in range(0, int(source_window.height), tile_size):
                    height = min(tile_size, int(source_window.height) - row_delta)
                    for col_delta in range(0, int(source_window.width), tile_size):
                        width = min(tile_size, int(source_window.width) - col_delta)
                        source_tile = Window(
                            source_window.col_off + col_delta,
                            source_window.row_off + row_delta,
                            width,
                            height,
                        )
                        destination_tile = Window(
                            destination_window.col_off + col_delta,
                            destination_window.row_off + row_delta,
                            width,
                            height,
                        )
                        local = source.read(1, window=source_tile)
                        resolved = _apply_resolved_ids(local, offset=offset, resolved_ids=resolved_ids)
                        core_mask = rasterize(
                            [(core_geometry, 1)],
                            out_shape=(height, width),
                            transform=rasterio.windows.transform(destination_tile, destination.transform),
                            fill=0,
                            dtype="uint8",
                        ).astype(bool)
                        existing = destination.read(1, window=destination_tile)
                        existing[core_mask] = resolved[core_mask]
                        destination.write(existing, 1, window=destination_tile)


def build_panelized_boundary_component_labels(
    *,
    cleaned_mask_path: str | Path,
    centerline_path: str | Path,
    panel_plan_path: str | Path,
    panel_root: str | Path,
    output_dir: str | Path,
    tile_size: int = 1024,
) -> dict[str, str]:
    """Build global component labels from disk-backed panel labels and core ownership."""
    import geopandas as gpd

    cleaned_mask_path, centerline_path, panel_plan_path, panel_root, output_dir = (
        Path(cleaned_mask_path), Path(centerline_path), Path(panel_plan_path), Path(panel_root), Path(output_dir)
    )
    cores = gpd.read_file(panel_plan_path, layer="panel_cores").set_index("panel_id")
    source_panel_masks = {str(panel_id): panel_root / str(panel_id) / "mask.tif" for panel_id in cores.index}
    missing = [str(path) for path in source_panel_masks.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Panel masks do not exist: {missing}")
    centerline_gdf, centerline = _load_single_centerline(centerline_path)
    with rasterio.open(cleaned_mask_path) as base:
        pixel_length = float(max(abs(float(base.transform.a)), abs(float(base.transform.e))))
        cut_lines = _panelized_border_cut_lines(source_panel_masks, centerline, base.bounds, pixel_length)
    work_dir = output_dir / "panel_local_labels"
    processing_areas = gpd.read_file(panel_plan_path, layer="processing_areas")
    panel_masks = _materialize_processing_area_masks(
        cleaned_mask_path=cleaned_mask_path,
        processing_areas=processing_areas,
        output_dir=work_dir / "processing_masks",
        tile_size=tile_size,
    )
    indexed_processing = processing_areas.set_index("panel_id")
    overlap_pairs = [
        (left_id, right_id)
        for left_id, right_id in combinations(sorted(panel_masks), 2)
        if indexed_processing.loc[left_id].geometry.intersects(indexed_processing.loc[right_id].geometry)
    ]
    labels, offsets, resolved = _resolve_panel_component_ids(
        panel_masks,
        work_dir,
        border_cut_lines=cut_lines,
        tile_size=tile_size,
        overlap_pairs=overlap_pairs,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    extensions_path = output_dir / f"{_example_prefix(cleaned_mask_path)}_centerline_border_extensions_snapped.gpkg"
    gpd.GeoDataFrame(
        {"endpoint_id": list(range(1, len(cut_lines) + 1))}, geometry=cut_lines, crs=centerline_gdf.crs
    ).to_file(extensions_path, driver="GPKG")
    with rasterio.open(cleaned_mask_path) as base:
        profile = base.profile.copy()
        panel_cores: dict[str, tuple[Path, int, Window, Window, object]] = {}
        for panel_id, core in cores.geometry.items():
            destination_window = _outward_window(core.bounds, base.transform)
            with rasterio.open(labels[str(panel_id)]) as local:
                source_window = _outward_window(core.bounds, local.transform)
            panel_cores[str(panel_id)] = (
                labels[str(panel_id)],
                offsets[str(panel_id)],
                source_window,
                destination_window,
                core,
            )
    prefix = _example_prefix(cleaned_mask_path)
    labels_path = output_dir / f"{prefix}_full_extent_land_water_labels_with_snapped_buffered_border_cut.tif"
    _write_resolved_label_cores(
        output_path=labels_path,
        profile=profile,
        panel_cores=panel_cores,
        resolved_ids=resolved,
        tile_size=tile_size,
    )
    summary_path = output_dir / f"{prefix}_full_extent_land_component_summary_with_snapped_buffered_border_cut.csv"
    summary = pd.DataFrame(
        [{"land_component_id": component_id, "raster_value": component_id + 1} for component_id in sorted(set(resolved.values()))]
    )
    summary.to_csv(summary_path, index=False)
    manifest_path = output_dir / f"{prefix}_panelized_boundary_component_manifest.json"
    manifest_path.write_text(json.dumps({
        "algorithm": "panelwise_local_labels_overlap_union",
        "source_mask": str(cleaned_mask_path.resolve()),
        "centerline": str(centerline_path.resolve()),
        "panel_plan": str(panel_plan_path.resolve()),
        "panel_root": str(panel_root.resolve()),
        "panel_count": len(source_panel_masks),
        "overlap_pair_count": len(overlap_pairs),
        "tile_size": int(tile_size),
        "labels": str(labels_path.resolve()),
        "extensions": str(extensions_path.resolve()),
        "summary": str(summary_path.resolve()),
    }, indent=2), encoding="utf-8")
    return {"labels": str(labels_path.resolve()), "extensions": str(extensions_path.resolve()), "summary": str(summary_path.resolve()), "local_labels_dir": str(work_dir.resolve()), "manifest": str(manifest_path.resolve())}


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


def _nearest_mask_pixel_xy(mask: np.ndarray, transform: object, target_xy: np.ndarray) -> tuple[np.ndarray, float]:
    """Find the nearest true raster cell without materializing all cell coordinates."""
    best_xy: np.ndarray | None = None
    best_squared = float("inf")
    # A small row block bounds temporary coordinate arrays even for very large masks.
    for row_start in range(0, mask.shape[0], 64):
        rows, cols = np.nonzero(mask[row_start : row_start + 64])
        if len(rows) == 0:
            continue
        rows = rows + row_start
        xs, ys = raster_xy(transform, rows, cols, offset="center")
        coords = np.column_stack([xs, ys]).astype(float, copy=False)
        squared = np.sum((coords - target_xy[None, :]) ** 2, axis=1)
        index = int(np.argmin(squared))
        if float(squared[index]) < best_squared:
            best_squared = float(squared[index])
            best_xy = coords[index].copy()
    if best_xy is None:
        raise ValueError("The selected main-water component contains no pixels.")
    return best_xy, float(np.sqrt(best_squared))


def _nearest_water_pixel_across_panels(panel_masks: dict[str, Path], target_xy: np.ndarray) -> tuple[np.ndarray, float]:
    """Find the nearest water pixel by streaming panel blocks, not a full raster."""
    best_xy: np.ndarray | None = None
    best_squared = float("inf")
    for mask_path in panel_masks.values():
        with rasterio.open(mask_path) as source:
            for _, window in source.block_windows(1):
                water = source.read(1, window=window) > 0
                rows, cols = np.nonzero(water)
                if len(rows) == 0:
                    continue
                rows = rows + int(window.row_off)
                cols = cols + int(window.col_off)
                xs, ys = raster_xy(source.transform, rows, cols, offset="center")
                coords = np.column_stack([xs, ys]).astype(float, copy=False)
                squared = np.sum((coords - target_xy[None, :]) ** 2, axis=1)
                index = int(np.argmin(squared))
                if float(squared[index]) < best_squared:
                    best_squared = float(squared[index])
                    best_xy = coords[index].copy()
    if best_xy is None:
        raise ValueError("No water pixel was found across panel masks.")
    return best_xy, float(np.sqrt(best_squared))


def _panelized_border_cut_lines(panel_masks: dict[str, Path], centerline: LineString, bounds: object, pixel_length: float) -> list[LineString]:
    """Construct the legacy endpoint-to-border cuts using streamed panel water pixels."""
    lines: list[LineString] = []
    for endpoint in (Point(centerline.coords[0]), Point(centerline.coords[-1])):
        snapped_xy, _ = _nearest_water_pixel_across_panels(
            panel_masks, np.asarray([endpoint.x, endpoint.y], dtype=float)
        )
        _, border, _ = _nearest_border_target(Point(*snapped_xy), bounds, pixel_length)
        lines.append(LineString([tuple(snapped_xy), (border.x, border.y)]))
    return lines


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

    endpoints = [Point(centerline.coords[0]), Point(centerline.coords[-1])]
    extension_records: list[dict[str, object]] = []
    shapes: list[tuple[LineString, int]] = []
    for endpoint_id, endpoint in enumerate(endpoints, start=1):
        endpoint_xy = np.asarray([endpoint.x, endpoint.y], dtype=float)
        snapped_xy, snap_distance = _nearest_mask_pixel_xy(main_mask, transform, endpoint_xy)
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
