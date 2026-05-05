from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd
import rasterio
from rasterio.features import rasterize, shapes
from rasterio.transform import xy as raster_xy
from scipy.ndimage import binary_fill_holes
from shapely.geometry import GeometryCollection, LineString, MultiLineString, Point, shape as shapely_shape
from shapely.ops import linemerge, substring, unary_union

if __package__ is None or __package__ == "":
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib"))

from rivgraph.classes import river

from hierarchy_level_definition.metrics.unit_metrics import (
    edge_width_from_attrs,
    edge_width_samples_from_attrs,
)
from hierarchy_level_definition.graph_building.directed_network_checks import (
    build_directed_graph,
    validate_single_inlet_single_outlet,
)
from hierarchy_level_definition.unit_detection.bifurcation_confluence_units import (
    StructuralUnit,
    analyze_network,
)
from network_variants.sword_matching import match_variant_nodes_to_sword


DEFAULT_FOOTPRINT_WIDTH_FIELD = "wid_adj"
WIDTH_FALLBACK_FIELDS = ("wid_adj", "wid", "wid_med")
DEFAULT_FOOTPRINT_BUFFER_SCALE = 0.5
DEFAULT_TRANSECT_SCALE = 1.5
DEFAULT_MIN_TRANSECT_PIXELS = 5.0
DEFAULT_WIDTH_QUANTILES = (0.05, 0.5, 0.95)
RIVGRAPH_WIDTH_TRIM_MULTIPLIER = 1.1
DEFAULT_CORE_OVERLAP_FRACTION_THRESHOLD = 0.05
DEFAULT_UNCHANGED_CORE_OVERLAP_FRACTION = 0.8


@dataclass
class NetworkVariantOutputs:
    collapse_components: pd.DataFrame
    edit_geometries: gpd.GeoDataFrame
    node_match: pd.DataFrame
    node_sword_match: pd.DataFrame
    link_match: pd.DataFrame
    link_lineage: pd.DataFrame
    link_width_families: pd.DataFrame
    link_width_samples: pd.DataFrame
    enriched_links: gpd.GeoDataFrame
    directed_links: gpd.GeoDataFrame
    directed_nodes: gpd.GeoDataFrame
    output_dir: Path
    collapsed_mask_path: Path
    rivgraph_links_path: Path
    rivgraph_nodes_path: Path
    directed_links_path: Path | None
    directed_nodes_path: Path | None
    rivgraph_centerline_path: Path | None
    sword_reaches_path: Path | None
    sword_nodes_path: Path | None
    manifest: dict[str, Any]


def _git_revision() -> str | None:
    repo_root = Path(__file__).resolve().parents[1]
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return None
    revision = result.stdout.strip()
    return revision or None


def _infer_example_id(path: str | Path) -> str:
    stem = Path(path).stem
    for suffix in (
        "_cleaned",
        "_binary_projected",
        "directed_links",
        "_links",
        "_nodes",
        "_mask",
    ):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    return stem or Path(path).stem


def _default_variant_id(group_label: str | None, unit_ids: Sequence[int] | None) -> str:
    if group_label:
        return f"group_{group_label}"
    if unit_ids:
        joined = "_".join(str(int(unit_id)) for unit_id in unit_ids)
        return f"units_{joined}"
    return "variant"


def _default_output_dir(cleaned_mask_path: str | Path, variant_id: str, example_id: str | None = None) -> Path:
    example_id = str(example_id) if example_id is not None else _infer_example_id(cleaned_mask_path)
    return Path(__file__).resolve().parent / "outputs" / example_id / variant_id


def _parse_int_list(value: Any) -> list[int]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set, np.ndarray, pd.Series)):
        result: list[int] = []
        for item in value:
            result.extend(_parse_int_list(item))
        return result
    text = str(value).strip()
    if not text:
        return []
    return [int(part.strip()) for part in text.split(",") if part.strip()]


def _parse_node_pair(value: Any) -> tuple[int, int]:
    parts = _parse_int_list(value)
    if len(parts) != 2:
        raise ValueError(f"Expected exactly two node ids in id_nodes, got {value!r}.")
    return int(parts[0]), int(parts[1])


def _serialize_int_list(values: Iterable[int]) -> str:
    return ",".join(str(int(value)) for value in values)


def _serialize_float_array(values: Sequence[float]) -> str:
    if not values:
        return ""
    return ",".join(f"{float(value):.6f}" for value in values)


def _parse_float_list(value: Any) -> list[float]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, np.ndarray, pd.Series)):
        return [float(item) for item in value]
    text = str(value).strip()
    if not text:
        return []
    return [float(part.strip()) for part in text.split(",") if part.strip()]


def _ensure_unique_preserve_order(values: Iterable[int]) -> list[int]:
    seen: set[int] = set()
    ordered: list[int] = []
    for value in values:
        numeric = int(value)
        if numeric in seen:
            continue
        seen.add(numeric)
        ordered.append(numeric)
    return ordered


def _geometry_endpoints(geometry: Any) -> tuple[Point, Point]:
    if geometry is None or geometry.is_empty:
        raise ValueError("Cannot determine endpoints for empty geometry.")
    if isinstance(geometry, LineString):
        coords = list(geometry.coords)
        return Point(coords[0]), Point(coords[-1])
    if isinstance(geometry, MultiLineString):
        parts = list(geometry.geoms)
        first_coords = list(parts[0].coords)
        last_coords = list(parts[-1].coords)
        return Point(first_coords[0]), Point(last_coords[-1])
    if hasattr(geometry, "boundary") and geometry.boundary is not None:
        boundary = geometry.boundary
        if hasattr(boundary, "geoms") and len(boundary.geoms) >= 2:
            return boundary.geoms[0], boundary.geoms[-1]
    raise TypeError(f"Unsupported linear geometry type: {geometry.geom_type}")


def _geometry_node_order(
    geometry: Any,
    node_a: int,
    node_b: int,
    node_geom_a: Any,
    node_geom_b: Any,
) -> tuple[int, int]:
    start_point, end_point = _geometry_endpoints(geometry)
    score_ab = start_point.distance(node_geom_a) + end_point.distance(node_geom_b)
    score_ba = start_point.distance(node_geom_b) + end_point.distance(node_geom_a)
    if score_ab <= score_ba:
        return node_a, node_b
    return node_b, node_a


def _reverse_linear_geometry(geometry: Any) -> Any:
    if geometry is None or geometry.is_empty:
        return geometry
    if isinstance(geometry, LineString):
        return LineString(list(geometry.coords)[::-1])
    if isinstance(geometry, MultiLineString):
        reversed_parts = [LineString(list(part.coords)[::-1]) for part in reversed(list(geometry.geoms))]
        return MultiLineString(reversed_parts)
    if hasattr(geometry, "reverse"):
        return geometry.reverse()
    raise TypeError(f"Unsupported linear geometry type: {geometry.geom_type}")


def _normalize_linear_geometry(geometry: Any) -> LineString:
    if geometry is None or geometry.is_empty:
        return LineString()
    if isinstance(geometry, LineString):
        return geometry
    if isinstance(geometry, MultiLineString):
        merged = linemerge(geometry)
        if isinstance(merged, LineString):
            return merged
        if isinstance(merged, MultiLineString):
            longest = max(merged.geoms, key=lambda part: part.length, default=None)
            return LineString(list(longest.coords)) if longest is not None else LineString()
    raise TypeError(f"Unsupported linear geometry type: {geometry.geom_type}")


def _core_trim_distance(length: float, match_tolerance: float) -> float:
    if not math.isfinite(length) or length <= 0:
        return 0.0
    return max(0.0, min(float(match_tolerance), float(length) * 0.25))


def _trim_linear_geometry(geometry: Any, trim_distance: float) -> LineString:
    line = _normalize_linear_geometry(geometry)
    length = float(line.length)
    if length <= 0 or trim_distance <= 0:
        return line
    if trim_distance * 2.0 >= length:
        return line
    trimmed = substring(line, trim_distance, length - trim_distance)
    if isinstance(trimmed, Point):
        return LineString()
    if isinstance(trimmed, LineString):
        return trimmed
    if isinstance(trimmed, MultiLineString):
        merged = linemerge(trimmed)
        if isinstance(merged, LineString):
            return merged
        if isinstance(merged, MultiLineString):
            longest = max(merged.geoms, key=lambda part: part.length, default=None)
            return LineString(list(longest.coords)) if longest is not None else LineString()
    return LineString()


def _read_selection_table(workflow_output_dir: str | Path) -> pd.DataFrame:
    output_dir = Path(workflow_output_dir)
    candidates = [
        output_dir / "selected_groups.csv",
        output_dir / "ordered_group_partitions.csv",
        output_dir / "merge_tree.csv",
    ]
    for path in candidates:
        if path.exists():
            return pd.read_csv(path)
    raise FileNotFoundError(
        f"Could not find a selection table in {output_dir}. Expected one of: "
        f"{', '.join(str(path.name) for path in candidates)}."
    )


def resolve_selected_unit_ids(
    *,
    workflow_output_dir: str | Path | None = None,
    group_label: str | None = None,
    unit_ids: Sequence[int] | None = None,
) -> list[int]:
    if unit_ids:
        return _ensure_unique_preserve_order(int(unit_id) for unit_id in unit_ids)

    if group_label is None:
        raise ValueError("Provide either unit_ids or group_label.")

    if workflow_output_dir is None:
        raise ValueError("workflow_output_dir is required when selecting by group_label.")

    selection = _read_selection_table(workflow_output_dir)
    if "group_label" not in selection.columns or "unit_ids" not in selection.columns:
        raise ValueError(
            f"Selection table in {workflow_output_dir} does not contain required columns "
            "'group_label' and 'unit_ids'."
        )

    row = selection.loc[selection["group_label"].astype(str) == str(group_label)]
    if row.empty:
        raise ValueError(f"Could not find group_label={group_label!r} in {workflow_output_dir}.")

    return _ensure_unique_preserve_order(_parse_int_list(row["unit_ids"].iloc[0]))


def _componentize_units(selected_unit_ids: Sequence[int], units_by_id: Mapping[int, StructuralUnit]) -> list[dict[str, Any]]:
    selected = [units_by_id[int(unit_id)] for unit_id in selected_unit_ids]
    graph = nx.Graph()
    for unit in selected:
        graph.add_node(unit.unit_id)

    for index, left in enumerate(selected):
        for right in selected[index + 1 :]:
            if left.node_set & right.node_set:
                graph.add_edge(left.unit_id, right.unit_id)

    components: list[dict[str, Any]] = []
    for index, component_unit_ids in enumerate(
        sorted(
            (sorted(component) for component in nx.connected_components(graph)),
            key=lambda unit_ids: (min(unit_ids), len(unit_ids)),
        ),
        start=1,
    ):
        component_units = [units_by_id[unit_id] for unit_id in component_unit_ids]
        link_ids = sorted({edge_key for unit in component_units for _, _, edge_key in unit.edge_set})
        node_ids = sorted({node_id for unit in component_units for node_id in unit.node_set})
        bubble_ids = sorted(
            {
                int(unit.compound_bubble_id)
                for unit in component_units
                if unit.compound_bubble_id is not None and not pd.isna(unit.compound_bubble_id)
            }
        )
        components.append(
            {
                "component_id": f"C{index}",
                "unit_ids": component_unit_ids,
                "link_ids": link_ids,
                "node_ids": node_ids,
                "compound_bubble_ids": bubble_ids,
                "n_units": len(component_unit_ids),
                "n_links": len(link_ids),
                "n_nodes": len(node_ids),
            }
        )
    return components


def _width_from_row(
    row: Mapping[str, Any],
    *,
    preferred_field: str = DEFAULT_FOOTPRINT_WIDTH_FIELD,
) -> float:
    if preferred_field in row:
        value = row.get(preferred_field)
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            numeric = float("nan")
        if math.isfinite(numeric) and numeric > 0:
            return numeric
    return edge_width_from_attrs(row)


def _mask_to_union_geometry(mask: np.ndarray, transform: Any) -> GeometryCollection | LineString | MultiLineString:
    geometries = [
        shapely_shape(geometry)
        for geometry, value in shapes(mask.astype(np.uint8), mask=mask.astype(bool), transform=transform)
        if int(value) == 1
    ]
    if not geometries:
        return GeometryCollection()
    return unary_union(geometries)


def _iter_linear_segments(geometry: Any) -> list[LineString]:
    if geometry.is_empty:
        return []
    if isinstance(geometry, LineString):
        return [geometry]
    if isinstance(geometry, MultiLineString):
        return list(geometry.geoms)
    if isinstance(geometry, GeometryCollection):
        segments: list[LineString] = []
        for part in geometry.geoms:
            segments.extend(_iter_linear_segments(part))
        return segments
    return []


def _segment_length_and_count(geometry: Any) -> tuple[float, int]:
    segments = _iter_linear_segments(geometry)
    return float(sum(segment.length for segment in segments)), len(segments)


def _unravel_link_pixels(flat_indices: Sequence[int], shape: tuple[int, int], transform: Any) -> tuple[np.ndarray, np.ndarray]:
    rows, cols = np.unravel_index(np.asarray(flat_indices, dtype=int), shape)
    xs, ys = raster_xy(transform, rows, cols, offset="center")
    return np.asarray(xs, dtype=float), np.asarray(ys, dtype=float)


def _distances_along_link(xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    if len(xs) <= 1:
        return np.asarray([0.0], dtype=float)
    step = np.sqrt(np.diff(xs) ** 2 + np.diff(ys) ** 2)
    return np.insert(np.cumsum(step), 0, 0.0)


def _trim_slice(total_widths: np.ndarray, distances: np.ndarray) -> slice:
    if total_widths.size == 0 or distances.size == 0:
        return slice(0, 0)

    revdistances = np.insert(np.cumsum(np.flipud(np.diff(distances))), 0, 0.0)
    start_index = int(np.argmin(np.abs(distances - total_widths[0] / 2 * RIVGRAPH_WIDTH_TRIM_MULTIPLIER)))
    end_index = int(
        len(distances)
        - np.argmin(np.abs(revdistances - total_widths[-1] / 2 * RIVGRAPH_WIDTH_TRIM_MULTIPLIER))
        - 1
    )

    if start_index >= end_index:
        return slice(0, len(total_widths))
    return slice(start_index, end_index)


def _compute_family_stats(values: np.ndarray, trim_slice: slice) -> dict[str, float]:
    trimmed = values[trim_slice]
    if trimmed.size == 0:
        trimmed = values
    if trimmed.size == 0:
        return {
            "adj": float("nan"),
            "med": float("nan"),
            "p05": float("nan"),
            "p50": float("nan"),
            "p95": float("nan"),
            "mean": float("nan"),
        }
    quantiles = np.quantile(trimmed, DEFAULT_WIDTH_QUANTILES)
    return {
        "adj": float(np.mean(trimmed)),
        "med": float(np.median(trimmed)),
        "p05": float(quantiles[0]),
        "p50": float(quantiles[1]),
        "p95": float(quantiles[2]),
        "mean": float(np.mean(trimmed)),
    }


def _build_transect(
    xs: np.ndarray,
    ys: np.ndarray,
    widths_for_length: np.ndarray,
    *,
    index: int,
    pixel_length: float,
    transect_scale: float,
    min_transect_pixels: float,
) -> LineString:
    if len(xs) == 1:
        dx, dy = 1.0, 0.0
    else:
        prev_index = max(index - 1, 0)
        next_index = min(index + 1, len(xs) - 1)
        dx = float(xs[next_index] - xs[prev_index])
        dy = float(ys[next_index] - ys[prev_index])
        if dx == 0.0 and dy == 0.0:
            dx, dy = 1.0, 0.0

    norm = math.hypot(dx, dy)
    nx_unit = -dy / norm
    ny_unit = dx / norm

    width_value = widths_for_length[min(index, len(widths_for_length) - 1)] if len(widths_for_length) else pixel_length
    half_length = max(float(width_value) * transect_scale, pixel_length * min_transect_pixels)
    return LineString(
        [
            (float(xs[index] - nx_unit * half_length), float(ys[index] - ny_unit * half_length)),
            (float(xs[index] + nx_unit * half_length), float(ys[index] + ny_unit * half_length)),
        ]
    )


def compute_width_families(
    regenerated_links: gpd.GeoDataFrame,
    *,
    collapsed_mask_path: str | Path,
    wet_reference_mask_path: str | Path,
    transect_scale: float = DEFAULT_TRANSECT_SCALE,
    min_transect_pixels: float = DEFAULT_MIN_TRANSECT_PIXELS,
) -> tuple[gpd.GeoDataFrame, pd.DataFrame]:
    with rasterio.open(collapsed_mask_path) as collapsed_ds:
        collapsed_mask = collapsed_ds.read(1) > 0
        collapsed_transform = collapsed_ds.transform
        collapsed_shape = collapsed_ds.shape
        collapsed_pixlen = float(abs(collapsed_transform.a))

    with rasterio.open(wet_reference_mask_path) as wet_ds:
        wet_mask = wet_ds.read(1) > 0
        wet_transform = wet_ds.transform

    if collapsed_shape != wet_mask.shape or collapsed_transform != wet_transform:
        raise ValueError("collapsed_mask_path and wet_reference_mask_path must share shape and transform.")

    collapsed_water = _mask_to_union_geometry(collapsed_mask, collapsed_transform)
    wet_reference_water = _mask_to_union_geometry(wet_mask, wet_transform)

    enriched = regenerated_links.copy()
    sample_records: list[dict[str, Any]] = []
    summary_records: list[dict[str, Any]] = []

    for row in regenerated_links.itertuples(index=False):
        attrs = row._asdict()
        link_id = int(attrs["id_link"])
        flat_indices = _parse_int_list(attrs.get("idx_link"))
        if not flat_indices:
            continue

        total_widths = edge_width_samples_from_attrs({"wid_pix": attrs.get("wid_pix")}, pixel_width_fields=["wid_pix"])
        if total_widths.size == 0:
            continue

        xs, ys = _unravel_link_pixels(flat_indices, collapsed_shape, collapsed_transform)
        distances = _distances_along_link(xs, ys)
        wet_widths: list[float] = []
        total_sampled_widths: list[float] = []
        dry_widths: list[float] = []
        n_wet_threads_values: list[int] = []

        for index in range(len(flat_indices)):
            transect = _build_transect(
                xs,
                ys,
                total_widths,
                index=index,
                pixel_length=collapsed_pixlen,
                transect_scale=transect_scale,
                min_transect_pixels=min_transect_pixels,
            )
            total_intersection = collapsed_water.intersection(transect)
            wet_intersection = wet_reference_water.intersection(transect)
            total_width, _ = _segment_length_and_count(total_intersection)
            wet_width, wet_threads = _segment_length_and_count(wet_intersection)
            total_width = max(total_width, 0.0)
            wet_width = max(wet_width, 0.0)
            dry_width = max(total_width - wet_width, 0.0)

            total_sampled_widths.append(total_width)
            wet_widths.append(wet_width)
            dry_widths.append(dry_width)
            n_wet_threads_values.append(int(wet_threads))

            sample_records.append(
                {
                    "id_link": link_id,
                    "sample_index": index,
                    "x": float(xs[index]),
                    "y": float(ys[index]),
                    "width_total": total_width,
                    "width_wet": wet_width,
                    "width_dry": dry_width,
                    "wet_fraction": float(wet_width / total_width) if total_width > 0 else float("nan"),
                    "dry_fraction": float(dry_width / total_width) if total_width > 0 else float("nan"),
                    "n_wet_threads": int(wet_threads),
                }
            )

        total_array = np.asarray(total_sampled_widths, dtype=float)
        wet_array = np.asarray(wet_widths, dtype=float)
        dry_array = np.asarray(dry_widths, dtype=float)
        threads_array = np.asarray(n_wet_threads_values, dtype=int)
        trim_slice = _trim_slice(total_array, distances)

        total_stats = _compute_family_stats(total_array, trim_slice)
        wet_stats = _compute_family_stats(wet_array, trim_slice)
        dry_stats = _compute_family_stats(dry_array, trim_slice)

        trimmed_threads = threads_array[trim_slice] if threads_array[trim_slice].size else threads_array
        n_wet_threads_mean = float(np.mean(trimmed_threads)) if trimmed_threads.size else float("nan")
        n_wet_threads_max = int(np.max(trimmed_threads)) if trimmed_threads.size else 0

        wet_fraction_adj = float(wet_stats["adj"] / total_stats["adj"]) if total_stats["adj"] > 0 else float("nan")
        dry_fraction_adj = float(dry_stats["adj"] / total_stats["adj"]) if total_stats["adj"] > 0 else float("nan")

        summary_records.append(
            {
                "id_link": link_id,
                "wid_pix_total": _serialize_float_array(total_sampled_widths),
                "wid_pix_wet": _serialize_float_array(wet_widths),
                "wid_pix_dry": _serialize_float_array(dry_widths),
                "wid_adj_total": total_stats["adj"],
                "wid_med_total": total_stats["med"],
                "wid_p05_total": total_stats["p05"],
                "wid_p50_total": total_stats["p50"],
                "wid_p95_total": total_stats["p95"],
                "wid_mean_total": total_stats["mean"],
                "wid_adj_wet": wet_stats["adj"],
                "wid_med_wet": wet_stats["med"],
                "wid_p05_wet": wet_stats["p05"],
                "wid_p50_wet": wet_stats["p50"],
                "wid_p95_wet": wet_stats["p95"],
                "wid_mean_wet": wet_stats["mean"],
                "wid_adj_dry": dry_stats["adj"],
                "wid_med_dry": dry_stats["med"],
                "wid_p05_dry": dry_stats["p05"],
                "wid_p50_dry": dry_stats["p50"],
                "wid_p95_dry": dry_stats["p95"],
                "wid_mean_dry": dry_stats["mean"],
                "wet_fraction_adj": wet_fraction_adj,
                "dry_fraction_adj": dry_fraction_adj,
                "n_wet_threads_mean": n_wet_threads_mean,
                "n_wet_threads_max": n_wet_threads_max,
            }
        )

    summary = pd.DataFrame.from_records(summary_records).sort_values("id_link", kind="mergesort").reset_index(drop=True)
    samples = pd.DataFrame.from_records(sample_records).sort_values(["id_link", "sample_index"], kind="mergesort").reset_index(drop=True)
    enriched = enriched.merge(summary, on="id_link", how="left", validate="1:1")
    return enriched, samples


def _component_footprint_geometry(
    reviewed_links: gpd.GeoDataFrame,
    link_ids: Sequence[int],
    *,
    preferred_width_field: str,
    buffer_scale: float,
) -> Any:
    selected = reviewed_links.loc[reviewed_links["id_link"].astype(int).isin([int(link_id) for link_id in link_ids])].copy()
    if selected.empty:
        return GeometryCollection()

    buffered = []
    for row in selected.itertuples(index=False):
        attrs = row._asdict()
        width = _width_from_row(attrs, preferred_field=preferred_width_field)
        if not math.isfinite(width) or width <= 0:
            continue
        buffered.append(attrs["geometry"].buffer(float(width) * buffer_scale))

    if not buffered:
        return GeometryCollection()
    return unary_union(buffered)


def _vectorize_mask(mask: np.ndarray, transform: Any) -> list[Any]:
    return [
        shapely_shape(geometry)
        for geometry, value in shapes(mask.astype(np.uint8), mask=mask.astype(bool), transform=transform)
        if int(value) == 1
    ]


def _apply_collapse_components_to_mask(
    parent_mask_path: str | Path,
    components: list[dict[str, Any]],
    reviewed_links: gpd.GeoDataFrame,
    *,
    preferred_width_field: str,
    buffer_scale: float,
    all_touched: bool = True,
    allow_noop: bool = False,
) -> tuple[np.ndarray, Mapping[str, Any], pd.DataFrame, gpd.GeoDataFrame]:
    with rasterio.open(parent_mask_path) as src:
        parent_mask = src.read(1) > 0
        profile = src.profile.copy()
        transform = src.transform
        crs = src.crs
        pixel_area = abs(transform.a * transform.e)

    add_union = np.zeros(parent_mask.shape, dtype=bool)
    geometry_records: list[dict[str, Any]] = []
    component_records: list[dict[str, Any]] = []

    for component in components:
        footprint = _component_footprint_geometry(
            reviewed_links,
            component["link_ids"],
            preferred_width_field=preferred_width_field,
            buffer_scale=buffer_scale,
        )
        footprint_mask = rasterize(
            [(footprint, 1)],
            out_shape=parent_mask.shape,
            transform=transform,
            fill=0,
            all_touched=all_touched,
            dtype="uint8",
        ).astype(bool)
        component_water = parent_mask & footprint_mask
        component_filled = binary_fill_holes(component_water)
        component_add = component_filled & (~component_water)
        add_union |= component_add

        footprint_area = float(footprint.area) if not footprint.is_empty else 0.0
        added_pixels = int(component_add.sum())
        added_area = float(added_pixels * pixel_area)

        component_records.append(
            {
                "component_id": component["component_id"],
                "unit_ids": _serialize_int_list(component["unit_ids"]),
                "link_ids": _serialize_int_list(component["link_ids"]),
                "node_ids": _serialize_int_list(component["node_ids"]),
                "compound_bubble_ids": _serialize_int_list(component["compound_bubble_ids"]),
                "n_units": int(component["n_units"]),
                "n_links": int(component["n_links"]),
                "n_nodes": int(component["n_nodes"]),
                "footprint_area": footprint_area,
                "added_pixels": added_pixels,
                "added_area": added_area,
            }
        )

        if not footprint.is_empty:
            geometry_records.append(
                {
                    "component_id": component["component_id"],
                    "geometry_role": "footprint",
                    "action": None,
                    "unit_ids": _serialize_int_list(component["unit_ids"]),
                    "geometry": footprint,
                }
            )

        for geometry in _vectorize_mask(component_add, transform):
            geometry_records.append(
                {
                    "component_id": component["component_id"],
                    "geometry_role": "edit",
                    "action": "add",
                    "unit_ids": _serialize_int_list(component["unit_ids"]),
                    "geometry": geometry,
                }
            )

    total_added_pixels = int(add_union.sum())
    if total_added_pixels == 0 and not allow_noop:
        raise ValueError(
            "The selected units produced no enclosed dry pixels to fill. "
            "Choose a different selection or pass allow_noop=True."
        )

    collapsed_mask = (parent_mask | add_union).astype(np.uint8)
    summary = {
        "base_water_pixels": int(parent_mask.sum()),
        "collapsed_water_pixels": int(collapsed_mask.sum()),
        "pixels_added": total_added_pixels,
        "pixels_removed": 0,
        "changed_pixels": total_added_pixels,
        "pixel_area": float(pixel_area),
    }
    component_frame = pd.DataFrame.from_records(component_records).sort_values("component_id", kind="mergesort").reset_index(drop=True)
    edit_geometries = gpd.GeoDataFrame(geometry_records, geometry="geometry", crs=crs)
    return collapsed_mask, summary, component_frame, edit_geometries


def _write_mask(path: str | Path, mask: np.ndarray, profile: Mapping[str, Any]) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_profile = dict(profile)
    write_profile.update(dtype="uint8", count=1, compress="lzw")
    with rasterio.open(output_path, "w", **write_profile) as dst:
        dst.write(mask.astype(np.uint8), 1)
    return output_path


def _pixel_length_from_transform(transform: Any) -> float:
    return float(max(abs(float(transform.a)), abs(float(transform.e))))


def _type_io_label(*, is_inlet: bool, is_outlet: bool) -> str:
    if is_inlet and is_outlet:
        return "both"
    if is_inlet:
        return "inlet"
    if is_outlet:
        return "outlet"
    return "neither"


def _infer_parent_node_order(
    parent_links: gpd.GeoDataFrame,
    parent_nodes: gpd.GeoDataFrame,
) -> tuple[dict[int, int], nx.MultiDiGraph, Any]:
    parent_graph = build_directed_graph(parent_links, parent_nodes, prefer_explicit_direction=True)
    report = validate_single_inlet_single_outlet(parent_graph)
    if not report.is_valid:
        raise ValueError(f"Parent reviewed graph failed validation: {report.issues}")

    if nx.is_directed_acyclic_graph(parent_graph):
        ordered_nodes = list(nx.topological_sort(parent_graph))
    else:
        source = report.source_nodes[0]
        source_distance = nx.single_source_shortest_path_length(nx.DiGraph(parent_graph), source)
        ordered_nodes = [
            node_id
            for node_id, _ in sorted(
                source_distance.items(),
                key=lambda item: (item[1], int(item[0])),
            )
        ]
    return {int(node_id): index for index, node_id in enumerate(ordered_nodes)}, parent_graph, report


def _match_child_nodes_to_parent_nodes(
    parent_nodes: gpd.GeoDataFrame,
    child_nodes: gpd.GeoDataFrame,
    *,
    match_tolerance: float,
    parent_node_order: Mapping[int, int],
) -> pd.DataFrame:
    parent = parent_nodes.copy()
    child = child_nodes.copy()
    parent["id_node"] = parent["id_node"].astype(int)
    child["id_node"] = child["id_node"].astype(int)
    parent_indexed = parent.set_index("id_node", drop=False)
    parent_sindex = parent.sindex

    records: list[dict[str, Any]] = []
    for row in child.itertuples(index=False):
        child_id = int(row.id_node)
        child_geom = row.geometry
        child_is_inlet = bool(getattr(row, "is_inlet", False))
        child_is_outlet = bool(getattr(row, "is_outlet", False))

        candidate_idx = list(parent_sindex.query(child_geom.buffer(match_tolerance), predicate="intersects"))
        if not candidate_idx:
            candidate_idx = list(range(len(parent)))

        candidates: list[tuple[int, float, int]] = []
        for candidate in candidate_idx:
            parent_row = parent.iloc[candidate]
            parent_id = int(parent_row["id_node"])
            distance = float(child_geom.distance(parent_row.geometry))
            role_mismatch = int(
                child_is_inlet != bool(parent_row.get("is_inlet", False))
                or child_is_outlet != bool(parent_row.get("is_outlet", False))
            )
            candidates.append((parent_id, distance, role_mismatch))

        candidates.sort(key=lambda item: (item[2], item[1], item[0]))
        parent_id, distance, role_mismatch = candidates[0]
        is_match = distance <= match_tolerance or (role_mismatch == 0 and distance <= match_tolerance * 2.0)

        parent_row = parent_indexed.loc[parent_id]
        records.append(
            {
                "child_id_node": child_id,
                "matched_parent_node_id": parent_id if is_match else pd.NA,
                "match_distance": distance,
                "match_within_tolerance": bool(is_match),
                "parent_node_order": parent_node_order.get(parent_id) if is_match else pd.NA,
                "child_is_inlet_raw": child_is_inlet,
                "child_is_outlet_raw": child_is_outlet,
                "parent_is_inlet": bool(parent_row.get("is_inlet", False)) if is_match else False,
                "parent_is_outlet": bool(parent_row.get("is_outlet", False)) if is_match else False,
            }
        )

    return pd.DataFrame.from_records(records).sort_values("child_id_node", kind="mergesort").reset_index(drop=True)


def _match_child_links_to_parent_links(
    parent_links: gpd.GeoDataFrame,
    child_links: gpd.GeoDataFrame,
    *,
    match_tolerance: float,
) -> pd.DataFrame:
    parent = parent_links.copy()
    child = child_links.copy()
    parent["id_link"] = parent["id_link"].astype(int)
    child["id_link"] = child["id_link"].astype(int)
    parent_sindex = parent.sindex

    records: list[dict[str, Any]] = []
    for row in child.itertuples(index=False):
        child_id = int(row.id_link)
        child_geom = row.geometry
        child_length = float(child_geom.length)
        child_trim_distance = _core_trim_distance(child_length, match_tolerance)
        child_core_geom = _trim_linear_geometry(child_geom, child_trim_distance)
        child_core_length = float(child_core_geom.length)
        candidates: list[dict[str, Any]] = []

        candidate_idx = list(parent_sindex.query(child_geom.buffer(match_tolerance), predicate="intersects"))
        for candidate in candidate_idx:
            parent_row = parent.iloc[candidate]
            parent_geom = parent_row.geometry
            parent_length = float(parent_geom.length)
            parent_trim_distance = _core_trim_distance(parent_length, match_tolerance)
            parent_core_geom = _trim_linear_geometry(parent_geom, parent_trim_distance)
            parent_core_length = float(parent_core_geom.length)
            child_overlap = float(child_geom.intersection(parent_geom.buffer(match_tolerance)).length)
            parent_overlap = float(parent_geom.intersection(child_geom.buffer(match_tolerance)).length)
            distance = float(child_geom.distance(parent_geom))
            child_fraction = child_overlap / child_length if child_length > 0 else 0.0
            parent_fraction = parent_overlap / parent_length if parent_length > 0 else 0.0
            child_core_overlap = float(child_core_geom.intersection(parent_core_geom.buffer(match_tolerance)).length)
            parent_core_overlap = float(parent_core_geom.intersection(child_core_geom.buffer(match_tolerance)).length)
            child_core_fraction = child_core_overlap / child_core_length if child_core_length > 0 else 0.0
            parent_core_fraction = parent_core_overlap / parent_core_length if parent_core_length > 0 else 0.0

            strong_overlap = (
                child_overlap >= max(match_tolerance * 2.0, child_length * 0.05)
                or child_fraction >= 0.05
                or parent_overlap >= max(match_tolerance * 2.0, parent_length * 0.05)
                or parent_fraction >= 0.05
            )
            if not strong_overlap and distance > match_tolerance:
                continue

            has_core_overlap = (
                child_core_overlap >= max(match_tolerance, child_core_length * DEFAULT_CORE_OVERLAP_FRACTION_THRESHOLD)
                or child_core_fraction >= DEFAULT_CORE_OVERLAP_FRACTION_THRESHOLD
                or parent_core_overlap >= max(match_tolerance, parent_core_length * DEFAULT_CORE_OVERLAP_FRACTION_THRESHOLD)
                or parent_core_fraction >= DEFAULT_CORE_OVERLAP_FRACTION_THRESHOLD
            )
            candidate_class = "core_overlap" if has_core_overlap else "touch_only"

            candidates.append(
                {
                    "child_id_link": child_id,
                    "parent_id_link": int(parent_row["id_link"]),
                    "parent_id_us_node": int(parent_row["id_us_node"]),
                    "parent_id_ds_node": int(parent_row["id_ds_node"]),
                    "child_length": child_length,
                    "parent_length": parent_length,
                    "child_overlap_length": child_overlap,
                    "child_overlap_fraction": child_fraction,
                    "parent_overlap_length": parent_overlap,
                    "parent_overlap_fraction": parent_fraction,
                    "child_trim_distance": child_trim_distance,
                    "parent_trim_distance": parent_trim_distance,
                    "child_core_length": child_core_length,
                    "parent_core_length": parent_core_length,
                    "child_core_overlap_length": child_core_overlap,
                    "child_core_overlap_fraction": child_core_fraction,
                    "parent_core_overlap_length": parent_core_overlap,
                    "parent_core_overlap_fraction": parent_core_fraction,
                    "distance": distance,
                    "candidate_class": candidate_class,
                }
            )

        if not candidates:
            continue

        candidates.sort(
            key=lambda item: (
                item["candidate_class"] != "core_overlap",
                -item["child_core_overlap_length"],
                -item["parent_core_overlap_length"],
                -item["child_overlap_length"],
                -item["parent_overlap_length"],
                item["distance"],
                item["parent_id_link"],
            )
        )
        for rank, candidate in enumerate(candidates, start=1):
            candidate["candidate_rank"] = rank
            candidate["is_dominant"] = rank == 1
            records.append(candidate)

    return pd.DataFrame.from_records(records).sort_values(
        ["child_id_link", "candidate_rank", "parent_id_link"],
        kind="mergesort",
    ).reset_index(drop=True)


def _resolve_overlap_lineage_group(ordered: pd.DataFrame) -> dict[str, Any]:
    core = ordered.loc[ordered["candidate_class"] == "core_overlap"].copy()
    touch = ordered.loc[ordered["candidate_class"] == "touch_only"].copy()
    candidate_ids = [int(value) for value in ordered["parent_id_link"]]

    if core.empty:
        primary = ordered.iloc[[0]].copy()
        lineage_type = "touch_fallback_1to1"
    else:
        primary = core.copy()
        if len(core) == 1 and touch.shape[0] == ordered.shape[0] - 1:
            top = core.iloc[0]
            if (
                float(top["child_core_overlap_fraction"]) >= DEFAULT_UNCHANGED_CORE_OVERLAP_FRACTION
                and float(top["parent_core_overlap_fraction"]) >= DEFAULT_UNCHANGED_CORE_OVERLAP_FRACTION
            ):
                lineage_type = "unchanged_1to1"
            else:
                lineage_type = "resolved_1to1"
        elif len(core) == 1:
            lineage_type = "resolved_1to1"
        else:
            lineage_type = "collapsed_many_to_one"

    dominant = primary.iloc[0]
    primary_ids = [int(value) for value in primary["parent_id_link"]]
    secondary_ids = primary_ids[1:]
    touch_ids = [int(value) for value in touch["parent_id_link"]]
    return {
        "lineage_type": lineage_type,
        "dominant_parent_link_id": int(dominant["parent_id_link"]),
        "matched_parent_link_ids": _serialize_int_list(primary_ids),
        "primary_parent_link_ids": _serialize_int_list(primary_ids),
        "secondary_parent_link_ids": _serialize_int_list(secondary_ids),
        "touch_parent_link_ids": _serialize_int_list(touch_ids),
        "candidate_parent_link_ids": _serialize_int_list(candidate_ids),
        "matched_parent_link_count": int(len(primary_ids)),
        "primary_parent_link_count": int(len(primary_ids)),
        "secondary_parent_link_count": int(len(secondary_ids)),
        "touch_parent_link_count": int(len(touch_ids)),
        "dominant_parent_overlap_fraction": float(dominant["child_overlap_fraction"]),
        "dominant_parent_overlap_length": float(dominant["child_overlap_length"]),
        "dominant_parent_core_overlap_fraction": float(dominant["child_core_overlap_fraction"]),
        "dominant_parent_core_overlap_length": float(dominant["child_core_overlap_length"]),
        "matched_parent_overlap_fraction": float(dominant["child_overlap_fraction"]),
        "matched_parent_overlap_length": float(dominant["child_overlap_length"]),
        "matched_parent_core_overlap_fraction": float(dominant["child_core_overlap_fraction"]),
        "matched_parent_core_overlap_length": float(dominant["child_core_overlap_length"]),
        "lineage_method": "overlap_fallback",
        "matched_parent_us_node": pd.NA,
        "matched_parent_ds_node": pd.NA,
        "matched_parent_node_path": "",
    }


def _compute_matched_set_overlap_metrics(
    *,
    child_geometry: Any,
    parent_link_ids: Sequence[int],
    parent_edge_by_link_id: Mapping[int, Mapping[str, Any]],
    match_tolerance: float,
) -> dict[str, float]:
    if child_geometry is None or child_geometry.is_empty or not parent_link_ids:
        return {
            "matched_parent_overlap_fraction": float("nan"),
            "matched_parent_overlap_length": float("nan"),
            "matched_parent_core_overlap_fraction": float("nan"),
            "matched_parent_core_overlap_length": float("nan"),
        }

    child_line = _normalize_linear_geometry(child_geometry)
    child_length = float(child_line.length)
    child_trim_distance = _core_trim_distance(child_length, match_tolerance)
    child_core_line = _trim_linear_geometry(child_line, child_trim_distance)
    child_core_length = float(child_core_line.length)

    parent_geometries = []
    parent_core_geometries = []
    for parent_link_id in parent_link_ids:
        attrs = parent_edge_by_link_id.get(int(parent_link_id))
        if not attrs:
            continue
        parent_geometry = attrs.get("geometry")
        if parent_geometry is None or parent_geometry.is_empty:
            continue
        parent_line = _normalize_linear_geometry(parent_geometry)
        parent_geometries.append(parent_line.buffer(match_tolerance))
        parent_core_line = _trim_linear_geometry(parent_line, _core_trim_distance(float(parent_line.length), match_tolerance))
        if not parent_core_line.is_empty:
            parent_core_geometries.append(parent_core_line.buffer(match_tolerance))

    if not parent_geometries:
        return {
            "matched_parent_overlap_fraction": float("nan"),
            "matched_parent_overlap_length": float("nan"),
            "matched_parent_core_overlap_fraction": float("nan"),
            "matched_parent_core_overlap_length": float("nan"),
        }

    overlap_union = unary_union(parent_geometries)
    matched_overlap_length = float(child_line.intersection(overlap_union).length)
    matched_overlap_fraction = matched_overlap_length / child_length if child_length > 0 else float("nan")

    if parent_core_geometries and not child_core_line.is_empty and child_core_length > 0:
        overlap_core_union = unary_union(parent_core_geometries)
        matched_core_overlap_length = float(child_core_line.intersection(overlap_core_union).length)
        matched_core_overlap_fraction = (
            matched_core_overlap_length / child_core_length if child_core_length > 0 else float("nan")
        )
    else:
        matched_core_overlap_length = float("nan")
        matched_core_overlap_fraction = float("nan")

    return {
        "matched_parent_overlap_fraction": matched_overlap_fraction,
        "matched_parent_overlap_length": matched_overlap_length,
        "matched_parent_core_overlap_fraction": matched_core_overlap_fraction,
        "matched_parent_core_overlap_length": matched_core_overlap_length,
    }


def _resolve_link_lineage(
    *,
    parent_graph: nx.MultiDiGraph,
    directed_child_links: gpd.GeoDataFrame,
    node_match: pd.DataFrame,
    link_match: pd.DataFrame,
    match_tolerance: float,
) -> pd.DataFrame:
    if directed_child_links.empty:
        return pd.DataFrame(
            columns=[
                "id_link",
                "lineage_type",
                "dominant_parent_link_id",
                "matched_parent_link_ids",
                "primary_parent_link_ids",
                "secondary_parent_link_ids",
                "touch_parent_link_ids",
                "candidate_parent_link_ids",
                "matched_parent_link_count",
                "primary_parent_link_count",
                "secondary_parent_link_count",
                "touch_parent_link_count",
                "dominant_parent_overlap_fraction",
                "dominant_parent_overlap_length",
                "dominant_parent_core_overlap_fraction",
                "dominant_parent_core_overlap_length",
                "matched_parent_overlap_fraction",
                "matched_parent_overlap_length",
                "matched_parent_core_overlap_fraction",
                "matched_parent_core_overlap_length",
                "lineage_method",
                "matched_parent_us_node",
                "matched_parent_ds_node",
                "matched_parent_node_path",
            ]
        )

    if link_match.empty:
        return pd.DataFrame(
            {
                "id_link": directed_child_links["id_link"].astype(int),
                "lineage_type": "unmatched",
                "dominant_parent_link_id": pd.NA,
                "matched_parent_link_ids": "",
                "primary_parent_link_ids": "",
                "secondary_parent_link_ids": "",
                "touch_parent_link_ids": "",
                "candidate_parent_link_ids": "",
                "matched_parent_link_count": 0,
                "primary_parent_link_count": 0,
                "secondary_parent_link_count": 0,
                "touch_parent_link_count": 0,
                "dominant_parent_overlap_fraction": float("nan"),
                "dominant_parent_overlap_length": float("nan"),
                "dominant_parent_core_overlap_fraction": float("nan"),
                "dominant_parent_core_overlap_length": float("nan"),
                "matched_parent_overlap_fraction": float("nan"),
                "matched_parent_overlap_length": float("nan"),
                "matched_parent_core_overlap_fraction": float("nan"),
                "matched_parent_core_overlap_length": float("nan"),
                "lineage_method": "unmatched",
                "matched_parent_us_node": pd.NA,
                "matched_parent_ds_node": pd.NA,
                "matched_parent_node_path": "",
            }
        )

    node_match_indexed = node_match.set_index("child_id_node", drop=False)
    match_groups = {int(child_id): frame.sort_values("candidate_rank", kind="mergesort").reset_index(drop=True) for child_id, frame in link_match.groupby("child_id_link", sort=False)}
    parent_simple = nx.DiGraph(parent_graph)
    parent_edge_by_link_id: dict[int, Mapping[str, Any]] = {
        int(link_id): attrs
        for _, _, link_id, attrs in parent_graph.edges(keys=True, data=True)
    }
    records: list[dict[str, Any]] = []
    for row in directed_child_links.itertuples(index=False):
        child_id = int(row.id_link)
        ordered = match_groups.get(child_id)
        if ordered is None or ordered.empty:
            continue

        fallback = _resolve_overlap_lineage_group(ordered)
        path_lineage = False

        up_parent = pd.NA
        ds_parent = pd.NA
        parent_node_path_text = ""

        try:
            up_parent = int(node_match_indexed.loc[int(row.id_us_node), "matched_parent_node_id"])
            ds_parent = int(node_match_indexed.loc[int(row.id_ds_node), "matched_parent_node_id"])
        except Exception:
            up_parent = pd.NA
            ds_parent = pd.NA

        if up_parent is not pd.NA and ds_parent is not pd.NA and not pd.isna(up_parent) and not pd.isna(ds_parent):
            try:
                parent_node_path = nx.shortest_path(parent_simple, int(up_parent), int(ds_parent))
                parent_node_path_text = _serialize_int_list(parent_node_path)
                primary_ids: list[int] = []
                for upstream, downstream in zip(parent_node_path[:-1], parent_node_path[1:]):
                    if parent_graph.has_edge(upstream, downstream):
                        primary_ids.extend(sorted(int(key) for key in parent_graph[upstream][downstream].keys()))
                primary_ids = _ensure_unique_preserve_order(primary_ids)
                if primary_ids:
                    path_lineage = True
                    ordered_indexed = ordered.set_index("parent_id_link", drop=False)
                    primary_rows = ordered_indexed.loc[
                        [parent_id for parent_id in primary_ids if parent_id in ordered_indexed.index]
                    ].copy()
                    if primary_rows.empty:
                        path_lineage = False
                    else:
                        primary_rows = primary_rows.sort_values(
                            ["child_core_overlap_length", "child_overlap_length", "candidate_rank"],
                            ascending=[False, False, True],
                            kind="mergesort",
                        ).reset_index(drop=True)
                        dominant = primary_rows.iloc[0]
                        matched_metrics = _compute_matched_set_overlap_metrics(
                            child_geometry=row.geometry,
                            parent_link_ids=primary_ids,
                            parent_edge_by_link_id=parent_edge_by_link_id,
                            match_tolerance=match_tolerance,
                        )
                        touch_ids = [
                            int(parent_id)
                            for parent_id in ordered["parent_id_link"]
                            if int(parent_id) not in set(primary_ids)
                        ]
                        lineage_type = "unchanged_1to1" if len(primary_ids) == 1 else "collapsed_many_to_one"
                        records.append(
                            {
                                "id_link": child_id,
                                "lineage_type": lineage_type,
                                "dominant_parent_link_id": int(dominant["parent_id_link"]),
                                "matched_parent_link_ids": _serialize_int_list(primary_ids),
                                "primary_parent_link_ids": _serialize_int_list(primary_ids),
                                "secondary_parent_link_ids": "",
                                "touch_parent_link_ids": _serialize_int_list(touch_ids),
                                "candidate_parent_link_ids": _serialize_int_list(
                                    _ensure_unique_preserve_order(
                                        list(primary_ids) + [int(value) for value in ordered["parent_id_link"]]
                                    )
                                ),
                                "matched_parent_link_count": int(len(primary_ids)),
                                "primary_parent_link_count": int(len(primary_ids)),
                                "secondary_parent_link_count": 0,
                                "touch_parent_link_count": int(len(touch_ids)),
                                "dominant_parent_overlap_fraction": float(dominant["child_overlap_fraction"]),
                                "dominant_parent_overlap_length": float(dominant["child_overlap_length"]),
                                "dominant_parent_core_overlap_fraction": float(dominant["child_core_overlap_fraction"]),
                                "dominant_parent_core_overlap_length": float(dominant["child_core_overlap_length"]),
                                **matched_metrics,
                                "lineage_method": "matched_parent_node_path",
                                "matched_parent_us_node": int(up_parent),
                                "matched_parent_ds_node": int(ds_parent),
                                "matched_parent_node_path": parent_node_path_text,
                            }
                        )
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                path_lineage = False

        if path_lineage:
            continue

        fallback["id_link"] = child_id
        fallback["matched_parent_us_node"] = up_parent
        fallback["matched_parent_ds_node"] = ds_parent
        fallback["matched_parent_node_path"] = parent_node_path_text
        records.append(fallback)

    return pd.DataFrame.from_records(records).sort_values("id_link", kind="mergesort").reset_index(drop=True)


def _build_progression_distance_maps(
    *,
    child_links: gpd.GeoDataFrame,
    child_nodes: gpd.GeoDataFrame,
    orientation_by_link: Mapping[int, tuple[int, int]],
    child_inlet_nodes: set[int],
    child_outlet_nodes: set[int],
) -> tuple[dict[int, float], dict[int, float]]:
    if not orientation_by_link:
        return {}, {}

    length_lookup = {
        int(row.id_link): float(row.len) if pd.notna(row.len) else float(row.geometry.length)
        for row in child_links.itertuples(index=False)
    }

    partial_graph = nx.DiGraph()
    partial_graph.add_nodes_from(int(node_id) for node_id in child_nodes["id_node"])
    for link_id, (upstream, downstream) in orientation_by_link.items():
        weight = length_lookup.get(int(link_id), 1.0)
        if partial_graph.has_edge(upstream, downstream):
            partial_graph[upstream][downstream]["weight"] = min(
                float(partial_graph[upstream][downstream]["weight"]),
                float(weight),
            )
        else:
            partial_graph.add_edge(upstream, downstream, weight=float(weight))

    source_distance: dict[int, float] = {}
    if len(child_inlet_nodes) == 1:
        inlet_node = next(iter(child_inlet_nodes))
        source_distance = {
            int(node_id): float(distance)
            for node_id, distance in nx.single_source_dijkstra_path_length(
                partial_graph,
                inlet_node,
                weight="weight",
            ).items()
        }

    sink_distance: dict[int, float] = {}
    if len(child_outlet_nodes) == 1:
        outlet_node = next(iter(child_outlet_nodes))
        sink_distance = {
            int(node_id): float(distance)
            for node_id, distance in nx.single_source_dijkstra_path_length(
                partial_graph.reverse(copy=True),
                outlet_node,
                weight="weight",
            ).items()
        }

    return source_distance, sink_distance


def _resolve_link_by_progression(
    *,
    node_a: int,
    node_b: int,
    source_distance: Mapping[int, float],
    sink_distance: Mapping[int, float],
    geometry_order: tuple[int, int] | None = None,
    allow_geometry_fallback: bool = False,
) -> tuple[tuple[int, int], str] | None:
    source_based: tuple[int, int] | None = None
    if node_a in source_distance and node_b in source_distance:
        distance_a = float(source_distance[node_a])
        distance_b = float(source_distance[node_b])
        if not math.isclose(distance_a, distance_b, rel_tol=1e-12, abs_tol=1e-9):
            source_based = (node_a, node_b) if distance_a < distance_b else (node_b, node_a)

    sink_based: tuple[int, int] | None = None
    if node_a in sink_distance and node_b in sink_distance:
        distance_a = float(sink_distance[node_a])
        distance_b = float(sink_distance[node_b])
        if not math.isclose(distance_a, distance_b, rel_tol=1e-12, abs_tol=1e-9):
            sink_based = (node_a, node_b) if distance_a > distance_b else (node_b, node_a)

    if source_based is not None and sink_based is not None:
        if source_based == sink_based:
            return source_based, "global_progression_rule"
        return None
    if source_based is not None:
        return source_based, "global_progression_rule"
    if sink_based is not None:
        return sink_based, "global_progression_rule"
    if allow_geometry_fallback and geometry_order is not None:
        return geometry_order, "geometry_order_fallback"
    return None


def _resolve_single_remaining_link(
    *,
    link_id: int,
    edge_nodes: Mapping[int, tuple[int, int]],
    geometry_order_by_link: Mapping[int, tuple[int, int]],
    orientation_by_link: Mapping[int, tuple[int, int]],
    child_nodes: gpd.GeoDataFrame,
    child_inlet_nodes: set[int],
    child_outlet_nodes: set[int],
    child_links: gpd.GeoDataFrame,
) -> tuple[tuple[int, int], str] | None:
    node_a, node_b = edge_nodes[int(link_id)]
    geometry_order = geometry_order_by_link.get(int(link_id))
    source_distance, sink_distance = _build_progression_distance_maps(
        child_links=child_links,
        child_nodes=child_nodes,
        orientation_by_link=orientation_by_link,
        child_inlet_nodes=child_inlet_nodes,
        child_outlet_nodes=child_outlet_nodes,
    )

    valid_candidates: list[tuple[tuple[int, int], str]] = []
    for candidate in ((node_a, node_b), (node_b, node_a)):
        child_graph = nx.MultiDiGraph()
        for node_row in child_nodes.itertuples(index=False):
            child_graph.add_node(int(node_row.id_node), **node_row._asdict())
        for oriented_link_id, (upstream, downstream) in orientation_by_link.items():
            child_graph.add_edge(upstream, downstream, key=int(oriented_link_id), id_link=int(oriented_link_id))
        child_graph.add_edge(candidate[0], candidate[1], key=int(link_id), id_link=int(link_id))
        report = validate_single_inlet_single_outlet(child_graph, require_flag_match=False)
        if report.is_valid:
            valid_candidates.append((candidate, "singleton_validation_rule"))

    if not valid_candidates:
        return None
    if len(valid_candidates) == 1:
        return valid_candidates[0]

    resolved = _resolve_link_by_progression(
        node_a=node_a,
        node_b=node_b,
        source_distance=source_distance,
        sink_distance=sink_distance,
        geometry_order=geometry_order,
        allow_geometry_fallback=True,
    )
    if resolved is not None:
        return resolved

    if geometry_order is not None:
        return geometry_order, "geometry_order_fallback"
    return valid_candidates[0]


def _assign_variant_directions(
    *,
    child_links: gpd.GeoDataFrame,
    child_nodes: gpd.GeoDataFrame,
    parent_links: gpd.GeoDataFrame,
    parent_nodes: gpd.GeoDataFrame,
    node_match: pd.DataFrame,
    link_match: pd.DataFrame,
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, Mapping[str, Any]]:
    child_links = child_links.copy()
    child_nodes = child_nodes.copy()
    child_nodes["id_node"] = child_nodes["id_node"].astype(int)
    child_links["id_link"] = child_links["id_link"].astype(int)

    child_node_geoms = child_nodes.set_index("id_node")[child_nodes.geometry.name].to_dict()
    parent_node_geoms = parent_nodes.set_index("id_node")[parent_nodes.geometry.name].to_dict()
    link_match_groups = {int(child_id): frame.copy() for child_id, frame in link_match.groupby("child_id_link", sort=False)}

    child_inlet_nodes = {
        int(node_id) for node_id in child_nodes.loc[child_nodes["is_inlet"].fillna(False), "id_node"]
    }
    child_outlet_nodes = {
        int(node_id) for node_id in child_nodes.loc[child_nodes["is_outlet"].fillna(False), "id_node"]
    }

    parent_order_lookup: dict[int, int] = {}
    for row in node_match.itertuples(index=False):
        if row.matched_parent_node_id is pd.NA or pd.isna(row.matched_parent_node_id):
            continue
        if row.parent_node_order is pd.NA or pd.isna(row.parent_node_order):
            continue
        parent_order_lookup[int(row.child_id_node)] = int(row.parent_node_order)

    orientation_by_link: dict[int, tuple[int, int]] = {}
    orientation_source: dict[int, str] = {}
    orientation_score_forward: dict[int, float] = {}
    orientation_score_reverse: dict[int, float] = {}
    edge_nodes: dict[int, tuple[int, int]] = {}
    geometry_order_by_link: dict[int, tuple[int, int]] = {}
    unresolved_links: list[int] = []

    for row in child_links.itertuples(index=False):
        link_id = int(row.id_link)
        node_a, node_b = _parse_node_pair(row.id_nodes)
        edge_nodes[link_id] = (node_a, node_b)
        start_node, end_node = _geometry_node_order(
            row.geometry,
            node_a,
            node_b,
            child_node_geoms[node_a],
            child_node_geoms[node_b],
        )
        geometry_order_by_link[link_id] = (start_node, end_node)
        start_point, end_point = _geometry_endpoints(row.geometry)
        matches = link_match_groups.get(link_id)

        oriented_pair: tuple[int, int] | None = None
        direction_method = ""

        if matches is not None and not matches.empty:
            core_matches = matches.loc[matches["candidate_class"] == "core_overlap"].copy()
            if not core_matches.empty:
                matches = core_matches
            score_start_end = 0.0
            score_end_start = 0.0
            for match_row in matches.itertuples(index=False):
                weight = float(match_row.child_overlap_length)
                parent_upstream = parent_node_geoms[int(match_row.parent_id_us_node)]
                parent_downstream = parent_node_geoms[int(match_row.parent_id_ds_node)]
                score_start_end += weight * (
                    start_point.distance(parent_upstream) + end_point.distance(parent_downstream)
                )
                score_end_start += weight * (
                    end_point.distance(parent_upstream) + start_point.distance(parent_downstream)
                )
            orientation_score_forward[link_id] = score_start_end
            orientation_score_reverse[link_id] = score_end_start
            if not math.isclose(score_start_end, score_end_start, rel_tol=1e-12, abs_tol=1e-9):
                oriented_pair = (
                    (start_node, end_node)
                    if score_start_end < score_end_start
                    else (end_node, start_node)
                )
                direction_method = "matched_parent_endpoints"

        if oriented_pair is None and node_a in parent_order_lookup and node_b in parent_order_lookup:
            order_a = parent_order_lookup[node_a]
            order_b = parent_order_lookup[node_b]
            if order_a != order_b:
                oriented_pair = (node_a, node_b) if order_a < order_b else (node_b, node_a)
                direction_method = "matched_parent_node_order"

        if oriented_pair is None:
            unresolved_links.append(link_id)
        else:
            orientation_by_link[link_id] = oriented_pair
            orientation_source[link_id] = direction_method

    incident_links: dict[int, list[int]] = {int(node_id): [] for node_id in child_nodes["id_node"]}
    for link_id, (node_a, node_b) in edge_nodes.items():
        incident_links[node_a].append(link_id)
        incident_links[node_b].append(link_id)

    changed = True
    while changed and unresolved_links:
        changed = False

        for link_id in list(unresolved_links):
            node_a, node_b = edge_nodes[link_id]
            if node_a in child_inlet_nodes:
                orientation_by_link[link_id] = (node_a, node_b)
                orientation_source[link_id] = "child_inlet_rule"
                unresolved_links.remove(link_id)
                changed = True
                continue
            if node_b in child_inlet_nodes:
                orientation_by_link[link_id] = (node_b, node_a)
                orientation_source[link_id] = "child_inlet_rule"
                unresolved_links.remove(link_id)
                changed = True
                continue
            if node_a in child_outlet_nodes:
                orientation_by_link[link_id] = (node_b, node_a)
                orientation_source[link_id] = "child_outlet_rule"
                unresolved_links.remove(link_id)
                changed = True
                continue
            if node_b in child_outlet_nodes:
                orientation_by_link[link_id] = (node_a, node_b)
                orientation_source[link_id] = "child_outlet_rule"
                unresolved_links.remove(link_id)
                changed = True
                continue

        if changed or not unresolved_links:
            continue

        for node_id in child_nodes["id_node"]:
            node_id = int(node_id)
            if node_id in child_inlet_nodes or node_id in child_outlet_nodes:
                continue
            candidate_links = [link_id for link_id in incident_links[node_id] if link_id in unresolved_links]
            if len(candidate_links) != 1:
                continue
            incoming = 0
            outgoing = 0
            for link_id in incident_links[node_id]:
                if link_id in unresolved_links:
                    continue
                upstream, downstream = orientation_by_link[link_id]
                if downstream == node_id:
                    incoming += 1
                if upstream == node_id:
                    outgoing += 1
            link_id = candidate_links[0]
            node_a, node_b = edge_nodes[link_id]
            other = node_b if node_a == node_id else node_a

            if incoming > 0 and outgoing == 0:
                orientation_by_link[link_id] = (node_id, other)
                orientation_source[link_id] = "local_balance_rule"
                unresolved_links.remove(link_id)
                changed = True
                break
            if outgoing > 0 and incoming == 0:
                orientation_by_link[link_id] = (other, node_id)
                orientation_source[link_id] = "local_balance_rule"
                unresolved_links.remove(link_id)
                changed = True
                break

        if changed or not unresolved_links:
            continue

        source_distance, sink_distance = _build_progression_distance_maps(
            child_links=child_links,
            child_nodes=child_nodes,
            orientation_by_link=orientation_by_link,
            child_inlet_nodes=child_inlet_nodes,
            child_outlet_nodes=child_outlet_nodes,
        )
        for link_id in list(unresolved_links):
            node_a, node_b = edge_nodes[link_id]
            resolved = _resolve_link_by_progression(
                node_a=node_a,
                node_b=node_b,
                source_distance=source_distance,
                sink_distance=sink_distance,
                geometry_order=geometry_order_by_link.get(link_id),
                allow_geometry_fallback=False,
            )
            if resolved is None:
                continue
            oriented_pair, method = resolved
            orientation_by_link[link_id] = oriented_pair
            orientation_source[link_id] = method
            unresolved_links.remove(link_id)
            changed = True

    if unresolved_links:
        if len(unresolved_links) == 1:
            link_id = int(unresolved_links[0])
            resolved = _resolve_single_remaining_link(
                link_id=link_id,
                edge_nodes=edge_nodes,
                geometry_order_by_link=geometry_order_by_link,
                orientation_by_link=orientation_by_link,
                child_nodes=child_nodes,
                child_inlet_nodes=child_inlet_nodes,
                child_outlet_nodes=child_outlet_nodes,
                child_links=child_links,
            )
            if resolved is not None:
                oriented_pair, method = resolved
                orientation_by_link[link_id] = oriented_pair
                orientation_source[link_id] = method
                unresolved_links = []

    if unresolved_links:
        raise ValueError(
            "Could not orient all regenerated links from parent matching/topology rules. "
            f"Unresolved id_link values: {sorted(int(link_id) for link_id in unresolved_links)}"
        )

    directed_links = child_links.copy()
    directed_links["raw_id_nodes"] = directed_links["id_nodes"].astype(str)
    directed_links["raw_is_inlet"] = directed_links["is_inlet"]
    directed_links["raw_is_outlet"] = directed_links["is_outlet"]

    child_graph = nx.MultiDiGraph()
    for node_row in child_nodes.itertuples(index=False):
        child_graph.add_node(int(node_row.id_node), **node_row._asdict())
    for row in directed_links.itertuples(index=False):
        upstream, downstream = orientation_by_link[int(row.id_link)]
        child_graph.add_edge(upstream, downstream, key=int(row.id_link), id_link=int(row.id_link))

    report = validate_single_inlet_single_outlet(child_graph, require_flag_match=False)
    if not report.is_valid:
        raise ValueError(f"Directed regenerated graph failed validation: {report.issues}")

    source_node = int(report.source_nodes[0])
    sink_node = int(report.sink_nodes[0])
    upstream_nodes: list[int] = []
    downstream_nodes: list[int] = []
    direction_sources: list[str] = []
    for row in directed_links.itertuples(index=False):
        upstream, downstream = orientation_by_link[int(row.id_link)]
        upstream_nodes.append(upstream)
        downstream_nodes.append(downstream)
        direction_sources.append(orientation_source[int(row.id_link)])

    directed_links["id_us_node"] = upstream_nodes
    directed_links["id_ds_node"] = downstream_nodes
    directed_links["id_nodes"] = [f"{int(upstream)}, {int(downstream)}" for upstream, downstream in zip(upstream_nodes, downstream_nodes)]
    directed_links["is_inlet"] = directed_links["id_us_node"].astype(int) == source_node
    directed_links["is_outlet"] = directed_links["id_ds_node"].astype(int) == sink_node
    directed_links["type_io"] = [
        _type_io_label(is_inlet=bool(is_inlet), is_outlet=bool(is_outlet))
        for is_inlet, is_outlet in zip(directed_links["is_inlet"], directed_links["is_outlet"])
    ]
    directed_links["direction_assignment_method"] = direction_sources
    directed_links["direction_score_forward"] = [
        orientation_score_forward.get(int(link_id), float("nan")) for link_id in directed_links["id_link"]
    ]
    directed_links["direction_score_reverse"] = [
        orientation_score_reverse.get(int(link_id), float("nan")) for link_id in directed_links["id_link"]
    ]
    reverse_flags = []
    for row in directed_links.itertuples(index=False):
        link_id = int(row.id_link)
        reverse_flag = geometry_order_by_link[link_id] != orientation_by_link[link_id]
        reverse_flags.append(reverse_flag)
    directed_links["geometry_reversed_to_match_flow"] = reverse_flags
    if any(reverse_flags):
        geom_col = directed_links.geometry.name
        directed_links[geom_col] = [
            _reverse_linear_geometry(geometry) if reverse_flag else geometry
            for geometry, reverse_flag in zip(directed_links[geom_col], directed_links["geometry_reversed_to_match_flow"])
        ]
        if "idx_link" in directed_links.columns:
            reversed_idx = []
            for value, reverse_flag in zip(directed_links["idx_link"], directed_links["geometry_reversed_to_match_flow"]):
                idx_values = _parse_int_list(value)
                if reverse_flag:
                    idx_values = list(reversed(idx_values))
                reversed_idx.append(_serialize_int_list(idx_values))
            directed_links["idx_link"] = reversed_idx
        if "wid_pix" in directed_links.columns:
            reversed_wid_pix = []
            for value, reverse_flag in zip(directed_links["wid_pix"], directed_links["geometry_reversed_to_match_flow"]):
                sample_values = _parse_float_list(value)
                if reverse_flag:
                    sample_values = list(reversed(sample_values))
                reversed_wid_pix.append(_serialize_float_array(sample_values))
            directed_links["wid_pix"] = reversed_wid_pix

    directed_nodes = child_nodes.copy()
    directed_nodes["raw_is_inlet"] = directed_nodes["is_inlet"]
    directed_nodes["raw_is_outlet"] = directed_nodes["is_outlet"]
    directed_nodes = directed_nodes.merge(
        node_match.rename(columns={"child_id_node": "id_node"}),
        on="id_node",
        how="left",
        validate="1:1",
    )

    id_links_values: list[str] = []
    n_links_values: list[int] = []
    in_degree_values: list[int] = []
    out_degree_values: list[int] = []
    for node_id in directed_nodes["id_node"].astype(int):
        incident = sorted(int(link_id) for _, _, link_id in child_graph.in_edges(node_id, keys=True)) + sorted(
            int(link_id) for _, _, link_id in child_graph.out_edges(node_id, keys=True)
        )
        id_links_values.append(_serialize_int_list(incident))
        n_links_values.append(int(len(incident)))
        in_degree_values.append(int(child_graph.in_degree(node_id)))
        out_degree_values.append(int(child_graph.out_degree(node_id)))

    directed_nodes["id_links"] = id_links_values
    directed_nodes["n_links"] = n_links_values
    directed_nodes["in_degree"] = in_degree_values
    directed_nodes["out_degree"] = out_degree_values
    directed_nodes["is_inlet"] = directed_nodes["id_node"].astype(int) == source_node
    directed_nodes["is_outlet"] = directed_nodes["id_node"].astype(int) == sink_node
    directed_nodes["type_io"] = [
        _type_io_label(is_inlet=bool(is_inlet), is_outlet=bool(is_outlet))
        for is_inlet, is_outlet in zip(directed_nodes["is_inlet"], directed_nodes["is_outlet"])
    ]

    summary = {
        "source_node": source_node,
        "sink_node": sink_node,
        "n_oriented_links": int(len(directed_links)),
        "n_node_matches": int(node_match["match_within_tolerance"].fillna(False).sum()),
        "n_link_matches": int(link_match["child_id_link"].nunique()),
        "validation_report": report.to_dict(),
    }
    return directed_links, directed_nodes, summary


def _run_rivgraph(
    *,
    run_name: str,
    collapsed_mask_path: str | Path,
    rivgraph_output_dir: str | Path,
    exit_sides: str,
    single_thread: bool = False,
    export_sword: bool = True,
    verbose: bool = False,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Path | None]:
    output_dir = Path(rivgraph_output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    network = river(
        run_name,
        str(collapsed_mask_path),
        results_folder=str(output_dir),
        exit_sides=exit_sides,
        verbose=verbose,
        single_thread=single_thread,
    )
    network.compute_network()
    network.prune_network()
    network.compute_distance_transform()
    network.compute_link_width_and_length()
    network.compute_centerline()
    network.to_geotiff("skeleton")
    network.to_geovectors(export="network", ftype="gpkg", metadata=metadata)
    network.to_geovectors(export="centerline", ftype="gpkg", metadata=metadata)
    if export_sword:
        network.to_geovectors(export="sword", ftype="gpkg", metadata=metadata)
    network.save_network()

    return {
        "links": Path(network.paths["links"]) if "links" in network.paths else None,
        "nodes": Path(network.paths["nodes"]) if "nodes" in network.paths else None,
        "centerline": Path(network.paths["centerline"]) if "centerline" in network.paths else None,
        "skeleton": Path(network.paths["Iskel"]) if "Iskel" in network.paths else None,
        "distance": Path(network.paths["Idist"]) if "Idist" in network.paths else None,
        "network_pickle": Path(network.paths["network_pickle"]) if "network_pickle" in network.paths else None,
        "sword_reaches": Path(network.paths["reaches_sword"]) if "reaches_sword" in network.paths else None,
        "sword_nodes": Path(network.paths["nodes_sword"]) if "nodes_sword" in network.paths else None,
        "log": Path(network.paths["log"]) if "log" in network.paths else None,
    }


def generate_network_variant(
    *,
    cleaned_mask_path: str | Path,
    reviewed_links_path: str | Path,
    reviewed_nodes_path: str | Path,
    exit_sides: str,
    workflow_output_dir: str | Path | None = None,
    group_label: str | None = None,
    unit_ids: Sequence[int] | None = None,
    variant_id: str | None = None,
    example_id: str | None = None,
    output_dir: str | Path | None = None,
    wet_reference_mask_path: str | Path | None = None,
    preferred_width_field: str = DEFAULT_FOOTPRINT_WIDTH_FIELD,
    footprint_buffer_scale: float = DEFAULT_FOOTPRINT_BUFFER_SCALE,
    all_touched: bool = True,
    allow_noop: bool = False,
    single_thread: bool = False,
    export_sword: bool = True,
    transect_scale: float = DEFAULT_TRANSECT_SCALE,
    min_transect_pixels: float = DEFAULT_MIN_TRANSECT_PIXELS,
    match_tolerance: float | None = None,
    sword_node_source_path: str | Path | None = None,
    sword_wse_field: str | None = None,
    sword_match_tolerance: float | None = None,
    max_path_cutoff: int = 100,
    max_paths: int = 5000,
    verbose_rivgraph: bool = False,
) -> NetworkVariantOutputs:
    resolved_unit_ids = resolve_selected_unit_ids(
        workflow_output_dir=workflow_output_dir,
        group_label=group_label,
        unit_ids=unit_ids,
    )
    if not resolved_unit_ids:
        raise ValueError("No unit_ids were resolved for the requested selection.")

    variant_id = variant_id or _default_variant_id(group_label, resolved_unit_ids)
    example_id = str(example_id) if example_id is not None else _infer_example_id(cleaned_mask_path)
    output_path = (
        Path(output_dir)
        if output_dir is not None
        else _default_output_dir(cleaned_mask_path, variant_id, example_id=example_id)
    )
    summary_dir = output_path / "summary"
    mask_dir = output_path / "mask"
    rivgraph_dir = output_path / "rivgraph"
    matching_dir = output_path / "matching"
    directed_dir = output_path / "directed"
    width_dir = output_path / "widths"
    for directory in (summary_dir, mask_dir, rivgraph_dir, matching_dir, directed_dir, width_dir):
        directory.mkdir(parents=True, exist_ok=True)

    wet_reference_mask_path = Path(wet_reference_mask_path) if wet_reference_mask_path is not None else Path(cleaned_mask_path)

    _, units, _ = analyze_network(
        reviewed_links_path,
        reviewed_nodes_path,
        max_path_cutoff=max_path_cutoff,
        max_paths=max_paths,
    )
    units_by_id = {unit.unit_id: unit for unit in units}
    missing_unit_ids = sorted(unit_id for unit_id in resolved_unit_ids if unit_id not in units_by_id)
    if missing_unit_ids:
        raise ValueError(f"The following selected unit_ids were not found in the reviewed graph: {missing_unit_ids}")

    components = _componentize_units(resolved_unit_ids, units_by_id)
    reviewed_links = gpd.read_file(reviewed_links_path)
    reviewed_nodes = gpd.read_file(reviewed_nodes_path)
    with rasterio.open(cleaned_mask_path) as src:
        parent_profile = src.profile.copy()
        pixel_length = _pixel_length_from_transform(src.transform)

    if match_tolerance is None:
        match_tolerance = pixel_length * 1.25

    collapsed_mask, mask_summary, collapse_components, edit_geometries = _apply_collapse_components_to_mask(
        cleaned_mask_path,
        components,
        reviewed_links,
        preferred_width_field=preferred_width_field,
        buffer_scale=footprint_buffer_scale,
        all_touched=all_touched,
        allow_noop=allow_noop,
    )
    collapse_components["example_id"] = example_id
    collapse_components["variant_id"] = variant_id
    if group_label is not None:
        collapse_components["group_label"] = str(group_label)

    collapsed_mask_path = _write_mask(mask_dir / f"{example_id}__{variant_id}_collapsed.tif", collapsed_mask, parent_profile)
    edit_geometries.to_file(mask_dir / "collapse_edit_geometries.gpkg", driver="GPKG")
    collapse_components.to_csv(summary_dir / "collapse_components.csv", index=False)

    metadata = {
        "example_id": example_id,
        "variant_id": variant_id,
        "group_label": group_label,
        "selected_unit_ids": _serialize_int_list(resolved_unit_ids),
    }
    rivgraph_name = f"{example_id}__{variant_id}"
    rivgraph_paths = _run_rivgraph(
        run_name=rivgraph_name,
        collapsed_mask_path=collapsed_mask_path,
        rivgraph_output_dir=rivgraph_dir,
        exit_sides=exit_sides,
        single_thread=single_thread,
        export_sword=export_sword,
        verbose=verbose_rivgraph,
        metadata=metadata,
    )

    if rivgraph_paths["links"] is None or rivgraph_paths["nodes"] is None:
        raise RuntimeError("RivGraph did not produce links/nodes outputs for the regenerated variant.")

    raw_links = gpd.read_file(rivgraph_paths["links"])
    raw_nodes = gpd.read_file(rivgraph_paths["nodes"])
    enriched_links, link_width_samples = compute_width_families(
        raw_links,
        collapsed_mask_path=collapsed_mask_path,
        wet_reference_mask_path=wet_reference_mask_path,
        transect_scale=transect_scale,
        min_transect_pixels=min_transect_pixels,
    )
    parent_node_order, _, _ = _infer_parent_node_order(reviewed_links, reviewed_nodes)
    node_match = _match_child_nodes_to_parent_nodes(
        reviewed_nodes,
        raw_nodes,
        match_tolerance=match_tolerance,
        parent_node_order=parent_node_order,
    )
    link_match = _match_child_links_to_parent_links(
        reviewed_links,
        raw_links,
        match_tolerance=match_tolerance,
    )
    _, parent_graph, _ = _infer_parent_node_order(reviewed_links, reviewed_nodes)
    directed_links, directed_nodes, direction_summary = _assign_variant_directions(
        child_links=raw_links,
        child_nodes=raw_nodes,
        parent_links=reviewed_links,
        parent_nodes=reviewed_nodes,
        node_match=node_match,
        link_match=link_match,
    )
    directed_nodes, node_sword_match = match_variant_nodes_to_sword(
        directed_nodes=directed_nodes,
        parent_nodes=reviewed_nodes,
        node_match=node_match,
        sword_node_source_path=sword_node_source_path,
        sword_wse_field=sword_wse_field,
        sword_match_tolerance=sword_match_tolerance,
    )
    link_lineage = _resolve_link_lineage(
        parent_graph=parent_graph,
        directed_child_links=directed_links,
        node_match=node_match,
        link_match=link_match,
        match_tolerance=float(match_tolerance),
    )
    directed_links = directed_links.drop(
        columns=[column for column in link_lineage.columns if column in directed_links.columns and column != "id_link"],
        errors="ignore",
    ).merge(link_lineage, on="id_link", how="left", validate="1:1")

    width_columns = [column for column in enriched_links.columns if column not in raw_links.columns or column == enriched_links.geometry.name]
    directed_enriched_links = directed_links.copy()
    for column in width_columns:
        if column == enriched_links.geometry.name:
            continue
        directed_enriched_links[column] = enriched_links[column].values
    if "geometry_reversed_to_match_flow" in directed_enriched_links.columns and directed_enriched_links["geometry_reversed_to_match_flow"].any():
        for column in ("wid_pix_total", "wid_pix_wet", "wid_pix_dry"):
            if column not in directed_enriched_links.columns:
                continue
            reversed_values = []
            for value, reverse_flag in zip(directed_enriched_links[column], directed_enriched_links["geometry_reversed_to_match_flow"]):
                samples = _parse_float_list(value)
                if reverse_flag:
                    samples = list(reversed(samples))
                reversed_values.append(_serialize_float_array(samples))
            directed_enriched_links[column] = reversed_values

    link_width_families = pd.DataFrame(directed_enriched_links.drop(columns=directed_enriched_links.geometry.name))
    link_width_families.to_csv(width_dir / "link_width_families.csv", index=False)
    link_width_samples.to_csv(width_dir / "link_width_samples.csv", index=False)
    directed_enriched_links.to_file(width_dir / "links_with_width_families.gpkg", driver="GPKG")

    link_match.to_csv(matching_dir / "link_match.csv", index=False)
    link_lineage.to_csv(matching_dir / "link_lineage.csv", index=False)
    node_match.to_csv(matching_dir / "node_match.csv", index=False)
    node_sword_match.to_csv(matching_dir / "node_sword_match.csv", index=False)
    directed_links.to_file(directed_dir / f"{rivgraph_name}_directed_links.gpkg", driver="GPKG")
    directed_nodes.to_file(directed_dir / f"{rivgraph_name}_directed_nodes.gpkg", driver="GPKG")
    with (directed_dir / "direction_validation_report.json").open("w", encoding="utf-8") as handle:
        json.dump(direction_summary["validation_report"], handle, indent=2)

    manifest = {
        "example_id": example_id,
        "variant_id": variant_id,
        "selection_source": "group_label" if group_label is not None else "unit_ids",
        "selected_unit_ids": resolved_unit_ids,
        "group_label": group_label,
        "n_components": int(len(collapse_components)),
        "component_ids": collapse_components["component_id"].tolist(),
        "exit_sides": str(exit_sides),
        "preferred_width_field": preferred_width_field,
        "footprint_buffer_scale": float(footprint_buffer_scale),
        "transect_scale": float(transect_scale),
        "min_transect_pixels": float(min_transect_pixels),
        "match_tolerance": float(match_tolerance),
        "all_touched": bool(all_touched),
        "allow_noop": bool(allow_noop),
        "export_sword": bool(export_sword),
        "ready_for_graph_matching": True,
        "ready_for_rapid": False,
        "scope": [
            "collapsed mask regeneration",
            "RivGraph rerun",
            "total/wet/dry width-family computation",
            "parent-child graph matching",
            "directed variant graph assignment",
            "SWORD node matching/propagation",
        ],
        "not_yet_implemented": [
            "RAPID handoff package",
        ],
        "source_paths": {
            "cleaned_mask": str(Path(cleaned_mask_path).resolve()),
            "wet_reference_mask": str(Path(wet_reference_mask_path).resolve()),
            "reviewed_links": str(Path(reviewed_links_path).resolve()),
            "reviewed_nodes": str(Path(reviewed_nodes_path).resolve()),
            "workflow_output_dir": str(Path(workflow_output_dir).resolve()) if workflow_output_dir is not None else None,
            "sword_node_source": str(Path(sword_node_source_path).resolve()) if sword_node_source_path is not None else None,
        },
        "mask_summary": mask_summary,
        "sword_matching": {
            "wse_field": sword_wse_field,
            "match_tolerance": sword_match_tolerance,
            "n_matched_nodes": int(node_sword_match["sword_node_id"].notna().sum()) if not node_sword_match.empty else 0,
            "n_propagated_matches": int(node_sword_match["sword_match_from_parent"].fillna(False).sum()) if not node_sword_match.empty else 0,
        },
        "output_paths": {
            "output_dir": str(output_path.resolve()),
            "collapsed_mask": str(collapsed_mask_path.resolve()),
            "collapse_components": str((summary_dir / "collapse_components.csv").resolve()),
            "edit_geometries": str((mask_dir / "collapse_edit_geometries.gpkg").resolve()),
            "link_width_families": str((width_dir / "link_width_families.csv").resolve()),
            "link_width_samples": str((width_dir / "link_width_samples.csv").resolve()),
            "links_with_width_families": str((width_dir / "links_with_width_families.gpkg").resolve()),
            "node_match": str((matching_dir / "node_match.csv").resolve()),
            "node_sword_match": str((matching_dir / "node_sword_match.csv").resolve()),
            "link_match": str((matching_dir / "link_match.csv").resolve()),
            "link_lineage": str((matching_dir / "link_lineage.csv").resolve()),
            "directed_links": str((directed_dir / f"{rivgraph_name}_directed_links.gpkg").resolve()),
            "directed_nodes": str((directed_dir / f"{rivgraph_name}_directed_nodes.gpkg").resolve()),
            "direction_validation_report": str((directed_dir / "direction_validation_report.json").resolve()),
            "rivgraph_links": str(rivgraph_paths["links"].resolve()) if rivgraph_paths["links"] is not None else None,
            "rivgraph_nodes": str(rivgraph_paths["nodes"].resolve()) if rivgraph_paths["nodes"] is not None else None,
            "rivgraph_centerline": str(rivgraph_paths["centerline"].resolve()) if rivgraph_paths["centerline"] is not None else None,
            "sword_reaches": str(rivgraph_paths["sword_reaches"].resolve()) if rivgraph_paths["sword_reaches"] is not None else None,
            "sword_nodes": str(rivgraph_paths["sword_nodes"].resolve()) if rivgraph_paths["sword_nodes"] is not None else None,
        },
        "direction_summary": direction_summary,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "code_version": _git_revision(),
    }
    with (summary_dir / "variant_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)

    return NetworkVariantOutputs(
        collapse_components=collapse_components,
        edit_geometries=edit_geometries,
        node_match=node_match,
        node_sword_match=node_sword_match,
        link_match=link_match,
        link_lineage=link_lineage,
        link_width_families=link_width_families,
        link_width_samples=link_width_samples,
        enriched_links=directed_enriched_links,
        directed_links=directed_links,
        directed_nodes=directed_nodes,
        output_dir=output_path,
        collapsed_mask_path=collapsed_mask_path,
        rivgraph_links_path=Path(rivgraph_paths["links"]),
        rivgraph_nodes_path=Path(rivgraph_paths["nodes"]),
        directed_links_path=directed_dir / f"{rivgraph_name}_directed_links.gpkg",
        directed_nodes_path=directed_dir / f"{rivgraph_name}_directed_nodes.gpkg",
        rivgraph_centerline_path=Path(rivgraph_paths["centerline"]) if rivgraph_paths["centerline"] is not None else None,
        sword_reaches_path=Path(rivgraph_paths["sword_reaches"]) if rivgraph_paths["sword_reaches"] is not None else None,
        sword_nodes_path=Path(rivgraph_paths["sword_nodes"]) if rivgraph_paths["sword_nodes"] is not None else None,
        manifest=manifest,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a collapsed network variant from selected hierarchy units/groups: "
            "build collapse edit geometries, write a collapsed mask, rerun RivGraph, "
            "and compute total/wet/dry width families on regenerated links."
        )
    )
    parser.add_argument("--cleaned-mask", required=True, help="Parent cleaned mask GeoTIFF.")
    parser.add_argument("--reviewed-links", required=True, help="Reviewed/directed links GeoPackage.")
    parser.add_argument("--reviewed-nodes", required=True, help="Reviewed/directed nodes GeoPackage.")
    parser.add_argument("--exit-sides", required=True, help="River exit sides for RivGraph, e.g. NS or EW.")
    parser.add_argument("--workflow-output-dir", default=None, help="Hierarchy workflow output directory, required when selecting by group_label.")
    parser.add_argument("--group-label", default=None, help="Group label from selected_groups.csv or ordered_group_partitions.csv.")
    parser.add_argument("--unit-ids", nargs="*", type=int, default=None, help="Explicit unit IDs to collapse.")
    parser.add_argument("--variant-id", default=None, help="Optional output variant identifier.")
    parser.add_argument("--example-id", default=None, help="Optional stable example identifier used for regenerated file naming.")
    parser.add_argument("--output-dir", default=None, help="Optional output directory. Defaults to network_variants/outputs/<example>/<variant>/")
    parser.add_argument("--wet-reference-mask", default=None, help="Optional mask used for wet-width sampling. Defaults to the parent cleaned mask.")
    parser.add_argument("--preferred-width-field", default=DEFAULT_FOOTPRINT_WIDTH_FIELD, help="Preferred reviewed-link width field for footprint buffering.")
    parser.add_argument("--footprint-buffer-scale", type=float, default=DEFAULT_FOOTPRINT_BUFFER_SCALE, help="Multiplier applied to footprint width / 2 buffering.")
    parser.add_argument("--disable-all-touched", action="store_true", help="Disable all_touched rasterization when creating collapse footprints.")
    parser.add_argument("--allow-noop", action="store_true", help="Allow variants that produce no added pixels.")
    parser.add_argument("--single-thread", action="store_true", help="Pass single_thread=True to the RivGraph river class.")
    parser.add_argument("--disable-sword-export", action="store_true", help="Skip SWORD-style RivGraph exports.")
    parser.add_argument("--transect-scale", type=float, default=DEFAULT_TRANSECT_SCALE, help="Multiplier for transect half-length relative to local total width.")
    parser.add_argument("--min-transect-pixels", type=float, default=DEFAULT_MIN_TRANSECT_PIXELS, help="Minimum transect half-length in raster pixels.")
    parser.add_argument("--match-tolerance", type=float, default=None, help="Optional spatial tolerance for parent-child graph matching. Defaults to 1.25 raster pixels.")
    parser.add_argument("--sword-node-source", default=None, help="Optional SWORD node source file or parquet directory used for node matching.")
    parser.add_argument("--sword-wse-field", default=None, help="Optional WSE field name in the SWORD node source. Defaults to automatic detection.")
    parser.add_argument("--sword-match-tolerance", type=float, default=None, help="Optional maximum SWORD node-match distance in CRS units/meters after reprojection.")
    parser.add_argument("--max-path-cutoff", type=int, default=100, help="Unit-detection max simple-path cutoff when rehydrating units.")
    parser.add_argument("--max-paths", type=int, default=5000, help="Unit-detection maximum number of simple paths when rehydrating units.")
    parser.add_argument("--verbose-rivgraph", action="store_true", help="Print RivGraph progress to stdout.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    results = generate_network_variant(
        cleaned_mask_path=args.cleaned_mask,
        reviewed_links_path=args.reviewed_links,
        reviewed_nodes_path=args.reviewed_nodes,
        exit_sides=args.exit_sides,
        workflow_output_dir=args.workflow_output_dir,
        group_label=args.group_label,
        unit_ids=args.unit_ids,
        variant_id=args.variant_id,
        example_id=args.example_id,
        output_dir=args.output_dir,
        wet_reference_mask_path=args.wet_reference_mask,
        preferred_width_field=args.preferred_width_field,
        footprint_buffer_scale=args.footprint_buffer_scale,
        all_touched=not args.disable_all_touched,
        allow_noop=args.allow_noop,
        single_thread=args.single_thread,
        export_sword=not args.disable_sword_export,
        transect_scale=args.transect_scale,
        min_transect_pixels=args.min_transect_pixels,
        match_tolerance=args.match_tolerance,
        sword_node_source_path=args.sword_node_source,
        sword_wse_field=args.sword_wse_field,
        sword_match_tolerance=args.sword_match_tolerance,
        max_path_cutoff=args.max_path_cutoff,
        max_paths=args.max_paths,
        verbose_rivgraph=args.verbose_rivgraph,
    )

    print(f"Wrote variant outputs to {results.output_dir}")
    print(f"Components: {len(results.collapse_components)}")
    print(f"Regenerated links: {len(results.enriched_links)}")
    print(f"Collapsed mask: {results.collapsed_mask_path}")
    print(f"RivGraph links: {results.rivgraph_links_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
