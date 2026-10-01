from __future__ import annotations

import argparse
import json
import itertools
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Collection, Iterable, Mapping, Sequence

import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd
import rasterio
from rasterio.features import rasterize, shapes
from rasterio.transform import xy as raster_xy
from scipy.ndimage import binary_fill_holes, label as label_connected_components
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve
from shapely.geometry import GeometryCollection, LineString, MultiLineString, Point, shape as shapely_shape
from shapely.ops import linemerge, polygonize, substring, unary_union

if __package__ is None or __package__ == "":
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib"))

from rivgraph.classes import river
from rivgraph_compat import rivgraph_centerline_compat

from hierarchy_level_definition.metrics.unit_metrics import (
    edge_width_from_attrs,
    edge_width_samples_from_attrs,
)
from hierarchy_level_definition.graph_building.directed_network_checks import (
    ValidationReport,
    build_degree_frame,
    build_directed_graph,
    validate_single_inlet_single_outlet,
)
from hierarchy_level_definition.unit_detection.bifurcation_confluence_units import (
    StructuralUnit,
    analyze_network,
)
from network_variants.build_boundary_component_mask import build_boundary_component_mask
from network_variants.sword_matching import match_variant_nodes_to_sword
from network_variants.routing_width_application import (
    apply_routing_width_model,
    write_routing_width_audit,
)
from network_variants.width_samples import read_width_samples
from network_variants.width_two_bank import build_state_bank_labels, compute_two_bank_width_families
from network_variants.width_product_inheritance import inherit_unchanged_link_width_products


DEFAULT_FOOTPRINT_WIDTH_FIELD = "wid_adj_routing"
WIDTH_FALLBACK_FIELDS = ("wid_adj_routing", "wid_adj_wet", "wid_adj", "wid", "wid_med")
DEFAULT_FOOTPRINT_BUFFER_SCALE = 0.5
DEFAULT_TRANSECT_SCALE = 1.5
DEFAULT_MIN_TRANSECT_PIXELS = 5.0
DEFAULT_WIDTH_QUANTILES = (0.05, 0.5, 0.95)
RIVGRAPH_WIDTH_TRIM_MULTIPLIER = 1.1
DEFAULT_CORE_OVERLAP_FRACTION_THRESHOLD = 0.05
DEFAULT_UNCHANGED_CORE_OVERLAP_FRACTION = 0.8
SEGMENTED_STUB_BAD_OPPOSITION_COS_THRESHOLD = -0.5
SEGMENTED_STUB_BAD_OPPOSITION_FRACTION_THRESHOLD = 0.5
SEGMENTED_STUB_BAD_OPPOSITION_MARGIN_THRESHOLD = 0.25
SEGMENTED_STUB_WIDTH_RATIO_THRESHOLD = 1.25
SEGMENTED_STUB_WIDTH_INHERITANCE_FIELDS = (
    "wid_adj_total",
    "wid_med_total",
    "wid_p05_total",
    "wid_p50_total",
    "wid_p95_total",
    "wid_adj_wet",
    "wid_med_wet",
    "wid_p05_wet",
    "wid_p50_wet",
    "wid_p95_wet",
    "wid_adj_dry",
    "wid_med_dry",
    "wid_p05_dry",
    "wid_p50_dry",
    "wid_p95_dry",
    "wet_fraction_adj",
    "dry_fraction_adj",
    "n_wet_threads_mean",
    "n_wet_threads_max",
    "n_channels_mean",
    "n_channels_max",
)
WIDTH_SCOPE_BASE_LABEL = 1
WIDTH_SCOPE_COMPONENT_LABEL_START = 101
MAX_LOCAL_DIRECTION_REPAIR_LINKS = 10
COLLAPSE_CORRIDOR_MODES = ("realized", "anchored")


def _sanitize_rivgraph_topology_metadata(links: dict[str, object], nodes: dict[str, object]) -> None:
    """Drop stale provisional references left behind by RivGraph pruning."""

    valid_link_ids = set(links.get("id", []))
    valid_node_ids = set(nodes.get("id", []))

    if "parallels" in links:
        cleaned_parallels = []
        for pair in links["parallels"]:
            kept = [lid for lid in pair if lid in valid_link_ids]
            if len(kept) >= 2:
                cleaned_parallels.append(kept)
        links["parallels"] = cleaned_parallels

    if "arts" in links:
        cleaned_arts = []
        for triad in links["arts"]:
            kept = [lid for lid in triad if lid in valid_link_ids]
            if len(kept) >= 2:
                cleaned_arts.append(kept)
        links["arts"] = cleaned_arts

    if "link_conn" in links:
        links["link_conn"] = [[lid for lid in conn if lid in valid_link_ids] for conn in links["link_conn"]]

    for key in ("inlets", "outlets", "arts"):
        if key in nodes:
            nodes[key] = [nid for nid in nodes[key] if nid in valid_node_ids]

    if "super_apex" in nodes and nodes["super_apex"] not in valid_node_ids:
        del nodes["super_apex"]


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
    width_scope_mask_path: Path | None
    rivgraph_links_path: Path
    rivgraph_nodes_path: Path
    directed_links_path: Path | None
    directed_nodes_path: Path | None
    rivgraph_centerline_path: Path | None
    sword_reaches_path: Path | None
    sword_nodes_path: Path | None
    manifest: dict[str, Any]
    bank_labels_path: Path | None = None
    boundary_component_labels_path: Path | None = None
    origin_mask_path: Path | None = None
    rivgraph_distance_path: Path | None = None


@dataclass
class AnchoredGraphTransformation:
    """Graph products from the shared realized-to-anchored transformation.

    This contract deliberately contains no raster or serialization concerns.
    Full-raster and local-patch execution can therefore apply the exact same
    scientific anchoring operation once each has produced a valid realized
    full-network graph and its parent-matching products.
    """

    directed_links: gpd.GeoDataFrame
    directed_nodes: gpd.GeoDataFrame
    node_match: pd.DataFrame
    link_match: pd.DataFrame
    link_lineage: pd.DataFrame
    anchor_segmentation_summary: dict[str, Any]
    direction_summary: dict[str, Any]


class VariantDirectionValidationError(RuntimeError):
    """Raised after writing variant outputs when directed-graph validation fails."""

    def __init__(self, outputs: NetworkVariantOutputs, issues: Sequence[str]) -> None:
        self.outputs = outputs
        self.issues = [str(issue) for issue in issues]
        super().__init__(f"Directed regenerated graph failed validation: {self.issues}")


def _remap_copied_output_path(
    value: str | Path | None,
    *,
    source_output_dir: Path,
    target_output_dir: Path,
) -> Path | None:
    if value is None:
        return None
    path = Path(value)
    try:
        relative = path.resolve().relative_to(source_output_dir.resolve())
    except ValueError:
        return None
    return target_output_dir / relative


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


def _infer_reference_centerline_path(
    *,
    cleaned_mask_path: str | Path,
    reviewed_links_path: str | Path,
) -> Path:
    cleaned_mask_path = Path(cleaned_mask_path)
    reviewed_links_path = Path(reviewed_links_path)
    example_id = _infer_example_id(cleaned_mask_path)
    candidate_paths = [
        reviewed_links_path.with_name(f"{_infer_example_id(reviewed_links_path)}_centerline.gpkg"),
        cleaned_mask_path.parent.parent / "rivgraph" / f"{example_id}_centerline.gpkg",
    ]
    for path in candidate_paths:
        if path.exists():
            return path
    raise FileNotFoundError(
        "Could not resolve the reference single-edge centerline needed for boundary-mask creation. "
        f"Tried: {[str(path) for path in candidate_paths]}"
    )


def _ensure_boundary_component_mask(
    *,
    cleaned_mask_path: str | Path,
    reviewed_links_path: str | Path,
    output_dir: str | Path,
    boundary_component_labels_path: str | Path | None = None,
) -> dict[str, str]:
    cleaned_mask_path = Path(cleaned_mask_path)
    output_dir = Path(output_dir)
    if boundary_component_labels_path is not None:
        path = Path(boundary_component_labels_path)
        if not path.exists():
            raise FileNotFoundError(f"Boundary component label mask not found: {path}")
        prefix = _infer_example_id(cleaned_mask_path)
        return {
            "labels": str(path.resolve()),
            "summary": str((output_dir / f"{prefix}_full_extent_land_component_summary_with_snapped_buffered_border_cut.csv").resolve()),
        }

    prefix = _infer_example_id(cleaned_mask_path)
    expected_labels_path = output_dir / f"{prefix}_full_extent_land_water_labels_with_snapped_buffered_border_cut.tif"
    expected_summary_path = output_dir / f"{prefix}_full_extent_land_component_summary_with_snapped_buffered_border_cut.csv"
    if expected_labels_path.exists():
        return {
            "labels": str(expected_labels_path.resolve()),
            "summary": str(expected_summary_path.resolve()),
        }

    centerline_path = _infer_reference_centerline_path(
        cleaned_mask_path=cleaned_mask_path,
        reviewed_links_path=reviewed_links_path,
    )
    return build_boundary_component_mask(
        cleaned_mask_path=cleaned_mask_path,
        centerline_path=centerline_path,
        output_dir=output_dir,
        extension_buffer_pixels=1,
    )


def _collapsed_selection_label(unit_ids: Sequence[int] | None, group_label: str | None) -> str:
    unique_unit_ids = _ensure_unique_preserve_order(unit_ids or [])
    if group_label:
        safe_group_label = re.sub(r"[^0-9A-Za-z_-]+", "_", str(group_label)).strip("_") or "group"
        return f"collapsed_group_{safe_group_label}"
    if not unique_unit_ids:
        return "base_state"
    if len(unique_unit_ids) == 1:
        return f"collapsed_unit_{unique_unit_ids[0]}"
    return "collapsed_units_" + "_".join(str(unit_id) for unit_id in unique_unit_ids)


def _collapsed_selection_metadata(unit_ids: Sequence[int] | None, group_label: str | None) -> dict[str, Any]:
    unique_unit_ids = _ensure_unique_preserve_order(unit_ids or [])
    return {
        "collapsed_selection_type": "group" if group_label else ("unit" if unique_unit_ids else "base"),
        "collapsed_selection_label": _collapsed_selection_label(unique_unit_ids, group_label),
        "collapsed_unit_ids": _serialize_int_list(unique_unit_ids),
        "collapsed_group_label": str(group_label) if group_label is not None else "",
        "collapsed_unit_count": int(len(unique_unit_ids)),
    }


def _apply_collapsed_selection_metadata(
    frame: pd.DataFrame | gpd.GeoDataFrame,
    *,
    unit_ids: Sequence[int] | None,
    group_label: str | None,
) -> pd.DataFrame | gpd.GeoDataFrame:
    result = frame.copy()
    metadata = _collapsed_selection_metadata(unit_ids, group_label)
    for column, value in metadata.items():
        result[column] = value
    return result


def _parse_int_list(value: Any) -> list[int]:
    if value is None:
        return []
    try:
        if pd.isna(value):
            return []
    except TypeError:
        pass
    if isinstance(value, (list, tuple, set, np.ndarray, pd.Series)):
        result: list[int] = []
        for item in value:
            result.extend(_parse_int_list(item))
        return result
    text = str(value).strip()
    if not text or text.lower() in {"<na>", "nan", "none"}:
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


def _segment_projection_bounds(segment: LineString, transect: LineString) -> tuple[float, float]:
    distances = [float(transect.project(Point(x, y))) for x, y in segment.coords]
    return min(distances), max(distances)


def _interval_distance_to_point(start: float, end: float, point_distance: float) -> float:
    if point_distance < start:
        return start - point_distance
    if point_distance > end:
        return point_distance - end
    return 0.0


def _cap_wet_thread_geometry(
    wet_geometry: Any,
    *,
    transect: LineString,
    center_point: Point,
    max_threads: int,
) -> tuple[float, float, int, int, bool]:
    segments = _iter_linear_segments(wet_geometry)
    raw_threads = len(segments)
    raw_width = float(sum(segment.length for segment in segments))

    if max_threads <= 0 or raw_threads <= max_threads:
        return raw_width, raw_width, raw_threads, raw_threads, False

    center_distance = float(transect.project(center_point))
    ranked: list[tuple[float, float, float, LineString]] = []
    for segment in segments:
        start, end = _segment_projection_bounds(segment, transect)
        midpoint = 0.5 * (start + end)
        ranked.append(
            (
                _interval_distance_to_point(start, end, center_distance),
                abs(midpoint - center_distance),
                midpoint,
                segment,
            )
        )

    selected = sorted(ranked, key=lambda item: (item[0], item[1], item[2]))[:max_threads]
    selected_segments = [item[3] for item in selected]
    selected_bounds = [_segment_projection_bounds(segment, transect) for segment in selected_segments]
    wet_width = float(sum(segment.length for segment in selected_segments))
    total_width = float(max(end for _, end in selected_bounds) - min(start for start, _ in selected_bounds))
    total_width = max(total_width, wet_width)
    return total_width, wet_width, max_threads, raw_threads, True


def _select_segment_containing_point(
    geometry: Any,
    point: Point,
    *,
    tolerance: float = 1e-6,
) -> tuple[Any, float]:
    segments = _iter_linear_segments(geometry)
    if not segments:
        return GeometryCollection(), 0.0

    containing = [segment for segment in segments if segment.distance(point) <= tolerance]
    if containing:
        chosen = max(containing, key=lambda segment: float(segment.length))
        return chosen, float(chosen.length)

    chosen = min(segments, key=lambda segment: float(segment.distance(point)))
    return chosen, float(chosen.length)


def _mask_value_geometries(mask: np.ndarray, transform: Any) -> dict[int, Any]:
    grouped: dict[int, list[Any]] = {}
    for geometry, value in shapes(mask.astype(np.int32), mask=(mask > 0), transform=transform):
        label = int(value)
        if label <= 0:
            continue
        grouped.setdefault(label, []).append(shapely_shape(geometry))

    return {label: unary_union(geometries) for label, geometries in grouped.items()}


def _unravel_link_pixels(
    flat_indices: Sequence[int],
    shape: tuple[int, int],
    transform: Any,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rows, cols = np.unravel_index(np.asarray(flat_indices, dtype=int), shape)
    xs, ys = raster_xy(transform, rows, cols, offset="center")
    return (
        np.asarray(rows, dtype=int),
        np.asarray(cols, dtype=int),
        np.asarray(xs, dtype=float),
        np.asarray(ys, dtype=float),
    )


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


def _selection_uses_full_width_transects(regenerated_links: gpd.GeoDataFrame) -> bool:
    if "collapsed_unit_count" not in regenerated_links.columns:
        return False
    counts = pd.to_numeric(regenerated_links["collapsed_unit_count"], errors="coerce").fillna(0)
    return bool((counts > 1).any())


def _units_require_full_width_transects(
    *,
    resolved_unit_ids: Sequence[int],
    units_by_id: Mapping[int, StructuralUnit],
) -> bool:
    unique_unit_ids = _ensure_unique_preserve_order(int(unit_id) for unit_id in resolved_unit_ids)
    if len(unique_unit_ids) > 1:
        return True

    for unit_id in unique_unit_ids:
        unit = units_by_id[int(unit_id)]
        if bool(unit.is_compound):
            return True
        if str(unit.unit_class) == "compound_or_nested_complex":
            return True
    return False


def _build_width_scope_thread_caps(
    *,
    collapse_components: pd.DataFrame | None,
    units_by_id: Mapping[int, StructuralUnit],
) -> dict[int, int]:
    caps: dict[int, int] = {WIDTH_SCOPE_BASE_LABEL: 1}
    if collapse_components is None or collapse_components.empty:
        return caps

    if "width_scope_label" not in collapse_components.columns or "unit_ids" not in collapse_components.columns:
        return caps

    for row in collapse_components.itertuples(index=False):
        scope_label = int(getattr(row, "width_scope_label", 0) or 0)
        if scope_label <= 0:
            continue
        unit_ids = _parse_int_list(getattr(row, "unit_ids", ""))
        if len(unit_ids) != 1:
            continue
        unit = units_by_id.get(int(unit_ids[0]))
        if unit is None:
            continue
        if bool(unit.is_compound):
            continue
        if str(unit.unit_class) != "simple_bifurcation_confluence_pair":
            continue
        if int(getattr(unit, "n_paths", 0) or 0) != 2:
            continue
        caps[scope_label] = 2
    return caps


def _local_width_corridor(
    row: Mapping[str, Any],
    *,
    pixel_length: float,
) -> Any:
    geometry = row.get("geometry")
    if geometry is None or geometry.is_empty:
        return GeometryCollection()

    local_width = _width_from_row(
        row,
        preferred_field=DEFAULT_FOOTPRINT_WIDTH_FIELD,
    )
    radius = max(float(local_width) * DEFAULT_FOOTPRINT_BUFFER_SCALE, pixel_length)
    if not math.isfinite(radius) or radius <= 0:
        radius = pixel_length
    return geometry.buffer(radius)


def compute_width_families(
    regenerated_links: gpd.GeoDataFrame,
    *,
    collapsed_mask_path: str | Path,
    wet_reference_mask_path: str | Path,
    transect_scale: float = DEFAULT_TRANSECT_SCALE,
    min_transect_pixels: float = DEFAULT_MIN_TRANSECT_PIXELS,
    use_full_width_transects: bool | None = None,
    width_scope_mask_path: str | Path | None = None,
    width_scope_thread_caps: Mapping[int, int] | None = None,
    origin_mask_path: str | Path | None = None,
    rivgraph_distance_path: str | Path | None = None,
    bank_labels_path: str | Path | None = None,
    bank_candidate_neighbors_max: int = 2048,
) -> tuple[gpd.GeoDataFrame, pd.DataFrame]:
    resolved_origin_mask = origin_mask_path or width_scope_mask_path
    enriched, samples = compute_two_bank_width_families(
        regenerated_links,
        collapsed_mask_path=collapsed_mask_path,
        wet_reference_mask_path=wet_reference_mask_path,
        origin_mask_path=resolved_origin_mask,
        rivgraph_distance_path=rivgraph_distance_path,
        bank_labels_path=bank_labels_path,
        bank_candidate_neighbors_max=bank_candidate_neighbors_max,
    )
    return enriched, samples


def _compute_changed_link_width_products(
    directed_links: gpd.GeoDataFrame,
    *,
    parent_links: gpd.GeoDataFrame,
    parent_width_samples: pd.DataFrame,
    link_lineage: pd.DataFrame,
    width_runner: Callable[..., tuple[gpd.GeoDataFrame, pd.DataFrame]] = compute_width_families,
    **width_kwargs: Any,
) -> tuple[gpd.GeoDataFrame, pd.DataFrame, list[int], dict[str, Any]]:
    """Measure only links that cannot safely inherit their parent width product.

    Lineage plus exact topological equality are resolved before raster-backed
    width measurement.  Unchanged one-to-one links receive the complete parent
    width fields and samples; only the remaining child links are passed to the
    expensive width runner.  The current full-raster context is deliberately
    retained so this optimization does not alter width semantics.
    """

    assembly_started = time.perf_counter()
    classified_links, inherited_samples, classification_summary = (
        inherit_unchanged_link_width_products(
            child_links=directed_links,
            child_width_samples=pd.DataFrame(),
            parent_links=parent_links,
            parent_width_samples=parent_width_samples,
            link_lineage=link_lineage,
        )
    )
    changed_link_ids = classified_links.loc[
        classified_links["width_inheritance_status"].eq("changed_requires_evaluation"),
        "id_link",
    ].astype(int).tolist()
    changed_id_set = set(changed_link_ids)
    changed_links = directed_links.loc[
        directed_links["id_link"].astype(int).isin(changed_id_set)
    ].copy()

    measurement_started = time.perf_counter()
    if changed_links.empty:
        measured_links = directed_links.copy()
        changed_samples = pd.DataFrame()
    else:
        changed_enriched, changed_samples = width_runner(changed_links, **width_kwargs)
        measured_links = _merge_width_families_onto_links(directed_links, changed_enriched)
    measurement_seconds = time.perf_counter() - measurement_started

    sample_frames = [frame for frame in (changed_samples, inherited_samples) if not frame.empty]
    combined_samples = (
        pd.concat(sample_frames, ignore_index=True, sort=False)
        if sample_frames
        else pd.DataFrame()
    )
    enriched_links, combined_samples, inheritance_summary = (
        inherit_unchanged_link_width_products(
            child_links=measured_links,
            child_width_samples=combined_samples,
            parent_links=parent_links,
            parent_width_samples=parent_width_samples,
            link_lineage=link_lineage,
        )
    )
    inheritance_summary.update(
        {
            "measurement_scope": "changed_links_only_full_raster_context",
            "measured_link_count": int(len(changed_link_ids)),
            "measured_link_ids": sorted(changed_link_ids),
            "skipped_measurement_link_count": int(
                classification_summary["inherited_link_count"]
            ),
            "measurement_seconds": round(float(measurement_seconds), 6),
            "assembly_seconds": round(float(time.perf_counter() - assembly_started), 6),
        }
    )
    return enriched_links, combined_samples, changed_link_ids, inheritance_summary


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


def _selected_link_interior_geometry(
    reviewed_links: gpd.GeoDataFrame,
    link_ids: Sequence[int],
) -> Any:
    """Return the polygonal interior explicitly enclosed by selected links.

    Collapse footprints are intentionally wider than the selected centerlines.
    At a shared link that buffer may also enclose a neighbouring unit.  The
    selected link cycles, rather than the buffer, therefore define ownership of
    candidate dry components.
    """
    selected_ids = {int(value) for value in link_ids}
    selected = reviewed_links.loc[
        reviewed_links["id_link"].astype(int).isin(selected_ids)
    ]
    linework = [geometry for geometry in selected.geometry if geometry is not None and not geometry.is_empty]
    polygons = list(polygonize(unary_union(linework))) if linework else []
    return unary_union(polygons) if polygons else GeometryCollection()


def _owned_hole_pixels(
    candidate_add: np.ndarray,
    *,
    selected_interior: Any,
    transform: Any,
    minimum_overlap_fraction: float = 0.5,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    """Partition proposed hole fills into selected and neighbouring components."""
    if not np.any(candidate_add):
        empty = np.zeros(candidate_add.shape, dtype=bool)
        return empty, empty.copy(), []
    if selected_interior is None or selected_interior.is_empty:
        raise ValueError(
            "Selected collapse links do not form a polygonal interior; refusing "
            "unconstrained hole filling."
        )
    interior_mask = rasterize(
        [(selected_interior, 1)],
        out_shape=candidate_add.shape,
        transform=transform,
        fill=0,
        all_touched=False,
        dtype="uint8",
    ).astype(bool)
    labels, count = label_connected_components(
        candidate_add,
        structure=np.ones((3, 3), dtype=np.uint8),
    )
    accepted = np.zeros(candidate_add.shape, dtype=bool)
    rejected = np.zeros(candidate_add.shape, dtype=bool)
    audit: list[dict[str, Any]] = []
    for candidate_id in range(1, int(count) + 1):
        pixels = labels == candidate_id
        pixel_count = int(np.count_nonzero(pixels))
        overlap_count = int(np.count_nonzero(pixels & interior_mask))
        overlap_fraction = float(overlap_count / pixel_count) if pixel_count else 0.0
        is_owned = bool(overlap_fraction >= float(minimum_overlap_fraction))
        (accepted if is_owned else rejected)[pixels] = True
        audit.append(
            {
                "candidate_component_id": int(candidate_id),
                "candidate_pixels": pixel_count,
                "selected_interior_overlap_pixels": overlap_count,
                "selected_interior_overlap_fraction": overlap_fraction,
                "ownership_status": "accepted_selected_unit" if is_owned else "rejected_neighbouring_unit",
            }
        )
    return accepted, rejected, audit


def _apply_collapse_components_to_mask(
    parent_mask_path: str | Path,
    components: list[dict[str, Any]],
    reviewed_links: gpd.GeoDataFrame,
    *,
    preferred_width_field: str,
    buffer_scale: float,
    all_touched: bool = True,
    allow_noop: bool = False,
) -> tuple[np.ndarray, np.ndarray, Mapping[str, Any], pd.DataFrame, gpd.GeoDataFrame]:
    with rasterio.open(parent_mask_path) as src:
        parent_mask = src.read(1) > 0
        profile = src.profile.copy()
        transform = src.transform
        crs = src.crs
        pixel_area = abs(transform.a * transform.e)

    origin_mask = np.where(parent_mask, WIDTH_SCOPE_BASE_LABEL, 0).astype(np.uint16)
    add_union = np.zeros(parent_mask.shape, dtype=bool)
    geometry_records: list[dict[str, Any]] = []
    component_records: list[dict[str, Any]] = []

    if not components:
        collapsed_mask = parent_mask.astype(bool)
        summary = {
            "base_water_pixels": int(parent_mask.sum()),
            "collapsed_water_pixels": int(collapsed_mask.sum()),
            "pixels_added": 0,
            "pixels_removed": 0,
            "changed_pixels": 0,
            "pixel_area": float(pixel_area),
            "width_scope_labels": [WIDTH_SCOPE_BASE_LABEL],
        }
        component_frame = pd.DataFrame.from_records(component_records)
        edit_geometries = gpd.GeoDataFrame(geometry_records, geometry="geometry", crs=crs)
        summary["origin_labels"] = [WIDTH_SCOPE_BASE_LABEL]
        return collapsed_mask, origin_mask, summary, component_frame, edit_geometries

    for index, component in enumerate(components, start=1):
        scope_label = int(WIDTH_SCOPE_COMPONENT_LABEL_START + index - 1)
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
        candidate_filled = binary_fill_holes(component_water)
        candidate_add = candidate_filled & (~component_water)
        selected_interior = _selected_link_interior_geometry(
            reviewed_links,
            component["link_ids"],
        )
        component_add, rejected_add, ownership_audit = _owned_hole_pixels(
            candidate_add,
            selected_interior=selected_interior,
            transform=transform,
        )
        component_filled = component_water | component_add
        add_union |= component_add
        origin_mask[component_add] = scope_label

        footprint_area = float(footprint.area) if not footprint.is_empty else 0.0
        added_pixels = int(component_add.sum())
        added_area = float(added_pixels * pixel_area)
        scope_pixels = int(component_filled.sum())
        scope_area = float(scope_pixels * pixel_area)

        component_records.append(
            {
                "component_id": component["component_id"],
                "origin_label": scope_label,
                "width_scope_label": scope_label,
                "unit_ids": _serialize_int_list(component["unit_ids"]),
                "link_ids": _serialize_int_list(component["link_ids"]),
                "node_ids": _serialize_int_list(component["node_ids"]),
                "compound_bubble_ids": _serialize_int_list(component["compound_bubble_ids"]),
                "n_units": int(component["n_units"]),
                "n_links": int(component["n_links"]),
                "n_nodes": int(component["n_nodes"]),
                "footprint_area": footprint_area,
                "origin_scope_pixels": scope_pixels,
                "origin_scope_area": scope_area,
                "width_scope_pixels": scope_pixels,
                "width_scope_area": scope_area,
                "added_pixels": added_pixels,
                "added_area": added_area,
                "candidate_added_pixels": int(candidate_add.sum()),
                "rejected_neighbour_pixels": int(rejected_add.sum()),
                "rejected_neighbour_area": float(rejected_add.sum() * pixel_area),
                "component_ownership_audit": json.dumps(ownership_audit, sort_keys=True),
            }
        )

        if not footprint.is_empty:
            geometry_records.append(
                {
                    "component_id": component["component_id"],
                    "geometry_role": "footprint",
                    "action": None,
                    "width_scope_label": scope_label,
                    "unit_ids": _serialize_int_list(component["unit_ids"]),
                    "geometry": footprint,
                }
            )

        for geometry in _vectorize_mask(component_filled, transform):
            geometry_records.append(
                {
                    "component_id": component["component_id"],
                    "geometry_role": "collapse_domain",
                    "action": "label",
                    "width_scope_label": scope_label,
                    "unit_ids": _serialize_int_list(component["unit_ids"]),
                    "geometry": geometry,
                }
            )

        for geometry in _vectorize_mask(component_add, transform):
            geometry_records.append(
                {
                    "component_id": component["component_id"],
                    "geometry_role": "edit",
                    "action": "add",
                    "width_scope_label": scope_label,
                    "unit_ids": _serialize_int_list(component["unit_ids"]),
                    "geometry": geometry,
                }
            )

        for geometry in _vectorize_mask(rejected_add, transform):
            geometry_records.append(
                {
                    "component_id": component["component_id"],
                    "geometry_role": "candidate_edit_rejected",
                    "action": "reject",
                    "width_scope_label": scope_label,
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
        "width_scope_labels": [int(WIDTH_SCOPE_BASE_LABEL)] + [int(record["width_scope_label"]) for record in component_records],
        "origin_labels": [int(WIDTH_SCOPE_BASE_LABEL)] + [int(record["origin_label"]) for record in component_records],
    }
    component_frame = pd.DataFrame.from_records(component_records).sort_values("component_id", kind="mergesort").reset_index(drop=True)
    edit_geometries = gpd.GeoDataFrame(geometry_records, geometry="geometry", crs=crs)
    return collapsed_mask, origin_mask, summary, component_frame, edit_geometries


def _write_mask(path: str | Path, mask: np.ndarray, profile: Mapping[str, Any]) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_profile = dict(profile)
    write_dtype = "uint8" if mask.dtype == np.bool_ else np.dtype(mask.dtype).name
    write_profile.update(dtype=write_dtype, count=1, compress="lzw")
    with rasterio.open(output_path, "w", **write_profile) as dst:
        dst.write(mask.astype(write_dtype), 1)
    return output_path


def _write_state_bank_labels(
    *,
    boundary_component_labels_path: str | Path,
    origin_mask: np.ndarray,
    output_path: str | Path,
    profile: Mapping[str, Any],
) -> tuple[Path, list[int]]:
    with rasterio.open(boundary_component_labels_path) as src:
        boundary_component_labels = src.read(1).astype(np.int32)
        boundary_transform = src.transform
        boundary_shape = src.shape

    expected_shape = (int(profile["height"]), int(profile["width"]))
    expected_transform = profile["transform"]
    if boundary_shape != expected_shape or boundary_transform != expected_transform:
        raise ValueError("Boundary component label mask must share shape and transform with the state masks.")

    bank_labels, removed_labels = build_state_bank_labels(boundary_component_labels, origin_mask)
    bank_labels_path = _write_mask(output_path, bank_labels.astype(np.int32), profile)
    return bank_labels_path, removed_labels


def _build_selection_masks(
    *,
    cleaned_mask_path: str | Path,
    reviewed_links: gpd.GeoDataFrame,
    resolved_unit_ids: Sequence[int],
    units_by_id: Mapping[int, StructuralUnit],
    preferred_width_field: str,
    footprint_buffer_scale: float,
    all_touched: bool = True,
    allow_noop: bool = False,
) -> tuple[np.ndarray, np.ndarray, Mapping[str, Any], pd.DataFrame, gpd.GeoDataFrame]:
    components = _componentize_units(resolved_unit_ids, units_by_id) if resolved_unit_ids else []
    return _apply_collapse_components_to_mask(
        cleaned_mask_path,
        components,
        reviewed_links,
        preferred_width_field=preferred_width_field,
        buffer_scale=footprint_buffer_scale,
        all_touched=all_touched,
        allow_noop=allow_noop,
    )


def _merge_width_families_onto_links(
    base_links: gpd.GeoDataFrame,
    enriched_links: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    width_columns = [column for column in enriched_links.columns if column not in base_links.columns or column == enriched_links.geometry.name]
    merged = base_links.copy()
    if "id_link" in enriched_links.columns:
        enriched_lookup = enriched_links.drop(columns=enriched_links.geometry.name, errors="ignore").copy()
        merged = merged.merge(
            enriched_lookup[[column for column in enriched_lookup.columns if column == "id_link" or column in width_columns]],
            on="id_link",
            how="left",
            validate="1:1",
            suffixes=("", "_width"),
        )
    else:
        for column in width_columns:
            if column == enriched_links.geometry.name:
                continue
            if column in enriched_links.columns:
                merged[column] = enriched_links[column].values
    return merged


def _text_has_value(value: Any) -> bool:
    if value is None:
        return False
    try:
        if pd.isna(value):
            return False
    except TypeError:
        pass
    return bool(str(value).strip())


def _fraction_bad_opposition_by_link(
    link_width_samples: pd.DataFrame,
    *,
    threshold: float = SEGMENTED_STUB_BAD_OPPOSITION_COS_THRESHOLD,
) -> dict[int, float]:
    if link_width_samples.empty or "id_link" not in link_width_samples.columns or "qc_opposition_cos" not in link_width_samples.columns:
        return {}

    qc = pd.to_numeric(link_width_samples["qc_opposition_cos"], errors="coerce")
    valid = qc.notna()
    if "width_invalid" in link_width_samples.columns:
        valid &= ~link_width_samples["width_invalid"].fillna(False).astype(bool)

    frame = pd.DataFrame(
        {
            "id_link": pd.to_numeric(link_width_samples["id_link"], errors="coerce"),
            "qc_opposition_cos": qc,
        }
    ).loc[valid]
    frame = frame.dropna(subset=["id_link", "qc_opposition_cos"]).copy()
    if frame.empty:
        return {}

    fractions = (
        frame.groupby(frame["id_link"].astype(int), sort=False)["qc_opposition_cos"]
        .apply(lambda values: float(np.mean(values.to_numpy(dtype=float) > float(threshold))))
        .to_dict()
    )
    return {int(link_id): float(value) for link_id, value in fractions.items()}


def _is_touch_fallback_lineage(lineage_type: Any) -> bool:
    return str(lineage_type or "").startswith("touch_fallback")


def _is_segmented_stub_candidate(row: pd.Series) -> bool:
    if not _is_touch_fallback_lineage(row.get("lineage_type")):
        return False
    if _text_has_value(row.get("matched_parent_node_path")):
        return False
    core_overlap = pd.to_numeric(pd.Series([row.get("dominant_parent_core_overlap_fraction")]), errors="coerce").iloc[0]
    return not np.isfinite(core_overlap) or float(core_overlap) < float(DEFAULT_CORE_OVERLAP_FRACTION_THRESHOLD)


def _is_stable_family_representative(row: pd.Series) -> bool:
    if _is_touch_fallback_lineage(row.get("lineage_type")):
        return False
    if _text_has_value(row.get("matched_parent_node_path")):
        return True
    core_overlap = pd.to_numeric(pd.Series([row.get("dominant_parent_core_overlap_fraction")]), errors="coerce").iloc[0]
    return bool(np.isfinite(core_overlap) and float(core_overlap) >= float(DEFAULT_CORE_OVERLAP_FRACTION_THRESHOLD))


def _select_segmented_stub_family_representative(
    *,
    stub_row: pd.Series,
    adjacency_by_node: Mapping[int, list[int]],
    stable_link_ids: set[int],
) -> int | None:
    endpoint_candidates: list[list[int]] = []
    for node_column in ("id_us_node", "id_ds_node"):
        node_id = pd.to_numeric(pd.Series([stub_row.get(node_column)]), errors="coerce").iloc[0]
        if not np.isfinite(node_id):
            continue
        adjacent = [
            int(link_id)
            for link_id in adjacency_by_node.get(int(node_id), [])
            if int(link_id) != int(stub_row["id_link"]) and int(link_id) in stable_link_ids
        ]
        adjacent = list(dict.fromkeys(adjacent))
        if adjacent:
            endpoint_candidates.append(adjacent)

    if len(endpoint_candidates) != 1:
        return None
    if len(endpoint_candidates[0]) != 1:
        return None
    return int(endpoint_candidates[0][0])


def _apply_segmented_stub_width_inheritance(
    *,
    directed_links: gpd.GeoDataFrame,
    enriched_links: gpd.GeoDataFrame,
    link_width_samples: pd.DataFrame,
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, dict[str, Any]]:
    adjusted_links = directed_links.copy()
    adjusted_enriched = enriched_links.copy()

    metadata_defaults: dict[str, Any] = {
        "family_width_inheritance_candidate": False,
        "family_width_inheritance_applied": False,
        "family_representative_link_id": pd.Series(pd.NA, index=adjusted_links.index, dtype="Int64"),
        "family_width_inherited_from_link": pd.Series(pd.NA, index=adjusted_links.index, dtype="Int64"),
        "family_width_inheritance_reason": "",
        "family_stub_bad_opposition_fraction": np.nan,
        "family_representative_bad_opposition_fraction": np.nan,
        "family_stub_to_representative_wet_ratio": np.nan,
        "family_stub_to_representative_total_ratio": np.nan,
        "family_width_sample_provenance": "measured_link_samples_used_directly",
    }
    for column, default in metadata_defaults.items():
        if isinstance(default, pd.Series):
            adjusted_links[column] = default
        else:
            adjusted_links[column] = default
    for column, default in metadata_defaults.items():
        if isinstance(default, pd.Series):
            adjusted_enriched[column] = default.reindex(adjusted_enriched.index)
        else:
            adjusted_enriched[column] = default

    if adjusted_links.empty or adjusted_enriched.empty or link_width_samples.empty:
        return adjusted_links, adjusted_enriched, {"applied_count": 0, "applied_link_ids": []}

    if "id_link" not in adjusted_links.columns or "id_link" not in adjusted_enriched.columns:
        return adjusted_links, adjusted_enriched, {"applied_count": 0, "applied_link_ids": []}

    lineage_required = {"lineage_type", "dominant_parent_core_overlap_fraction", "matched_parent_node_path", "id_us_node", "id_ds_node"}
    if not lineage_required.issubset(adjusted_links.columns):
        return adjusted_links, adjusted_enriched, {"applied_count": 0, "applied_link_ids": []}

    bad_fraction_by_link = _fraction_bad_opposition_by_link(link_width_samples)
    if not bad_fraction_by_link:
        return adjusted_links, adjusted_enriched, {"applied_count": 0, "applied_link_ids": []}

    links_by_id = adjusted_links.set_index(pd.to_numeric(adjusted_links["id_link"], errors="coerce").astype(int), drop=False)
    enriched_by_id = adjusted_enriched.set_index(pd.to_numeric(adjusted_enriched["id_link"], errors="coerce").astype(int), drop=False)
    stable_link_ids = {
        int(link_id)
        for link_id, row in links_by_id.iterrows()
        if _is_stable_family_representative(row)
    }
    adjacency_by_node: dict[int, list[int]] = {}
    for row in adjusted_links[["id_link", "id_us_node", "id_ds_node"]].itertuples(index=False):
        link_id = int(row.id_link)
        for node_id in (row.id_us_node, row.id_ds_node):
            numeric_node = pd.to_numeric(pd.Series([node_id]), errors="coerce").iloc[0]
            if not np.isfinite(numeric_node):
                continue
            adjacency_by_node.setdefault(int(numeric_node), []).append(link_id)

    applied_records: list[dict[str, Any]] = []
    for link_id, stub_row in links_by_id.iterrows():
        if not _is_segmented_stub_candidate(stub_row):
            continue
        adjusted_links.loc[adjusted_links["id_link"].astype(int) == int(link_id), "family_width_inheritance_candidate"] = True

        representative_link_id = _select_segmented_stub_family_representative(
            stub_row=stub_row,
            adjacency_by_node=adjacency_by_node,
            stable_link_ids=stable_link_ids,
        )
        if representative_link_id is None or representative_link_id not in enriched_by_id.index:
            adjusted_links.loc[adjusted_links["id_link"].astype(int) == int(link_id), "family_width_inheritance_reason"] = "no_unique_stable_representative"
            continue

        stub_mask = adjusted_links["id_link"].astype(int) == int(link_id)
        adjusted_links.loc[stub_mask, "family_representative_link_id"] = int(representative_link_id)

        stub_bad_fraction = float(bad_fraction_by_link.get(int(link_id), float("nan")))
        representative_bad_fraction = float(bad_fraction_by_link.get(int(representative_link_id), float("nan")))
        adjusted_links.loc[stub_mask, "family_stub_bad_opposition_fraction"] = stub_bad_fraction
        adjusted_links.loc[stub_mask, "family_representative_bad_opposition_fraction"] = representative_bad_fraction

        if not np.isfinite(stub_bad_fraction) or not np.isfinite(representative_bad_fraction):
            adjusted_links.loc[stub_mask, "family_width_inheritance_reason"] = "missing_opposition_qc"
            continue

        stub_enriched = enriched_by_id.loc[int(link_id)]
        representative_enriched = enriched_by_id.loc[int(representative_link_id)]
        stub_wet = pd.to_numeric(pd.Series([stub_enriched.get("wid_adj_wet")]), errors="coerce").iloc[0]
        rep_wet = pd.to_numeric(pd.Series([representative_enriched.get("wid_adj_wet")]), errors="coerce").iloc[0]
        stub_total = pd.to_numeric(pd.Series([stub_enriched.get("wid_adj_total")]), errors="coerce").iloc[0]
        rep_total = pd.to_numeric(pd.Series([representative_enriched.get("wid_adj_total")]), errors="coerce").iloc[0]
        wet_ratio = float(stub_wet / rep_wet) if np.isfinite(stub_wet) and np.isfinite(rep_wet) and rep_wet > 0 else float("nan")
        total_ratio = float(stub_total / rep_total) if np.isfinite(stub_total) and np.isfinite(rep_total) and rep_total > 0 else float("nan")
        adjusted_links.loc[stub_mask, "family_stub_to_representative_wet_ratio"] = wet_ratio
        adjusted_links.loc[stub_mask, "family_stub_to_representative_total_ratio"] = total_ratio

        stub_channels = pd.to_numeric(pd.Series([stub_enriched.get("n_channels_max")]), errors="coerce").iloc[0]
        rep_channels = pd.to_numeric(pd.Series([representative_enriched.get("n_channels_max")]), errors="coerce").iloc[0]
        width_inflated = bool(
            (np.isfinite(wet_ratio) and wet_ratio > float(SEGMENTED_STUB_WIDTH_RATIO_THRESHOLD))
            or (np.isfinite(total_ratio) and total_ratio > float(SEGMENTED_STUB_WIDTH_RATIO_THRESHOLD))
            or (np.isfinite(stub_channels) and np.isfinite(rep_channels) and float(stub_channels) > float(rep_channels))
        )
        bad_enough = bool(
            stub_bad_fraction >= float(SEGMENTED_STUB_BAD_OPPOSITION_FRACTION_THRESHOLD)
            and (stub_bad_fraction - representative_bad_fraction) >= float(SEGMENTED_STUB_BAD_OPPOSITION_MARGIN_THRESHOLD)
        )
        if not bad_enough:
            adjusted_links.loc[stub_mask, "family_width_inheritance_reason"] = "stub_opposition_qc_not_bad_enough"
            continue
        if not width_inflated:
            adjusted_links.loc[stub_mask, "family_width_inheritance_reason"] = "stub_not_inflated_vs_representative"
            continue

        for column in SEGMENTED_STUB_WIDTH_INHERITANCE_FIELDS:
            if column in adjusted_enriched.columns and column in representative_enriched.index:
                adjusted_enriched.loc[adjusted_enriched["id_link"].astype(int) == int(link_id), column] = representative_enriched[column]

        enriched_stub_mask = adjusted_enriched["id_link"].astype(int) == int(link_id)
        adjusted_enriched.loc[
            enriched_stub_mask, "family_width_inheritance_candidate"
        ] = True
        adjusted_enriched.loc[
            enriched_stub_mask, "family_width_inheritance_applied"
        ] = True
        adjusted_enriched.loc[
            enriched_stub_mask, "family_representative_link_id"
        ] = int(representative_link_id)
        adjusted_enriched.loc[
            enriched_stub_mask, "family_width_inherited_from_link"
        ] = int(representative_link_id)
        adjusted_enriched.loc[
            enriched_stub_mask, "family_width_inheritance_reason"
        ] = "inherited_from_segmented_family_representative"
        adjusted_enriched.loc[
            enriched_stub_mask, "family_width_sample_provenance"
        ] = "measured_link_samples_retained__final_width_family_inherited_from_representative"

        adjusted_links.loc[stub_mask, "family_width_inheritance_applied"] = True
        adjusted_links.loc[stub_mask, "family_width_inherited_from_link"] = int(representative_link_id)
        adjusted_links.loc[stub_mask, "family_width_inheritance_reason"] = "inherited_from_segmented_family_representative"
        applied_records.append(
            {
                "id_link": int(link_id),
                "representative_link_id": int(representative_link_id),
                "stub_bad_opposition_fraction": stub_bad_fraction,
                "representative_bad_opposition_fraction": representative_bad_fraction,
                "wet_ratio": wet_ratio,
                "total_ratio": total_ratio,
            }
        )

    summary = {
        "applied_count": int(len(applied_records)),
        "applied_link_ids": [int(record["id_link"]) for record in applied_records],
        "records": applied_records,
        "criteria": {
            "bad_opposition_cos_threshold": float(SEGMENTED_STUB_BAD_OPPOSITION_COS_THRESHOLD),
            "bad_opposition_fraction_threshold": float(SEGMENTED_STUB_BAD_OPPOSITION_FRACTION_THRESHOLD),
            "bad_opposition_fraction_margin": float(SEGMENTED_STUB_BAD_OPPOSITION_MARGIN_THRESHOLD),
            "width_ratio_threshold": float(SEGMENTED_STUB_WIDTH_RATIO_THRESHOLD),
        },
    }
    return adjusted_links, adjusted_enriched, summary


def _write_width_family_outputs(
    *,
    width_dir: Path,
    link_width_samples: pd.DataFrame,
    directed_enriched_links: gpd.GeoDataFrame,
) -> pd.DataFrame:
    directed_enriched_links = _normalize_width_boolean_fields(directed_enriched_links)
    link_width_families = pd.DataFrame(directed_enriched_links.drop(columns=directed_enriched_links.geometry.name))
    width_dir.mkdir(parents=True, exist_ok=True)
    link_width_families.to_csv(width_dir / "link_width_families.csv", index=False)
    link_width_samples.to_csv(width_dir / "link_width_samples.csv", index=False)
    directed_enriched_links.to_file(width_dir / "links_with_width_families.gpkg", driver="GPKG")
    return link_width_families


WIDTH_BOOLEAN_FIELDS = (
    "width_reviewed_flag",
    "width_update_applied",
    "manual_width_review_candidate",
    "manual_width_review_requires_correction",
    "width_model_candidate",
    "split_sensitive",
    "width_model_excluded",
    "routing_context_uses_endpoint",
    "comparison_flag",
    "width_correction_applied",
    "width_correction_flag",
    "dry_corridor_meaningful",
    "measured_wet_exceeds_total_corridor",
    "routing_exceeds_total_corridor",
    "routing_corridor_consistency_flag",
    "width_model_evaluation_scope",
)


def _normalize_width_boolean_fields(
    links: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    """Keep inherited boolean provenance typed through GeoPackage writes."""
    result = links.copy()

    def normalize(value: Any) -> Any:
        if pd.isna(value):
            return pd.NA
        if isinstance(value, (bool, np.bool_)):
            return bool(value)
        if isinstance(value, (int, float, np.integer, np.floating)) and float(value) in {0.0, 1.0}:
            return bool(value)
        text = str(value).strip().lower()
        if text in {"true", "1"}:
            return True
        if text in {"false", "0"}:
            return False
        raise ValueError(f"Cannot normalize boolean width provenance value {value!r}.")

    for field in WIDTH_BOOLEAN_FIELDS:
        if field not in result.columns:
            continue
        normalized = result[field].map(normalize)
        result[field] = normalized.astype("boolean" if normalized.isna().any() else bool)
    return result


def _apply_width_sample_flow_alignment(
    directed_enriched_links: gpd.GeoDataFrame,
    *,
    collapse_corridor_mode: str,
) -> gpd.GeoDataFrame:
    aligned = directed_enriched_links.copy()
    if collapse_corridor_mode == "anchored":
        return aligned
    if "geometry_reversed_to_match_flow" not in aligned.columns or not aligned["geometry_reversed_to_match_flow"].any():
        return aligned

    for column in ("wid_pix_total", "wid_pix_wet", "wid_pix_dry"):
        if column not in aligned.columns:
            continue
        reversed_values = []
        for value, reverse_flag in zip(aligned[column], aligned["geometry_reversed_to_match_flow"]):
            samples = _parse_float_list(value)
            if reverse_flag:
                samples = list(reversed(samples))
            reversed_values.append(_serialize_float_array(samples))
        aligned[column] = reversed_values
    return aligned


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


def _resolve_canonical_child_boundary_nodes(
    *,
    parent_nodes: gpd.GeoDataFrame,
    node_match: pd.DataFrame,
    topology_source_nodes: Collection[int] = (),
    topology_sink_nodes: Collection[int] = (),
) -> tuple[set[int], set[int], list[str]]:
    issues: list[str] = []
    if node_match.empty or "matched_parent_node_id" not in node_match.columns:
        return set(), set(), issues

    if parent_nodes.empty or not {"id_node", "is_inlet", "is_outlet"}.issubset(parent_nodes.columns):
        return set(), set(), issues

    parent_lookup = parent_nodes.set_index("id_node", drop=False)[["is_inlet", "is_outlet"]].to_dict("index")
    matches = node_match.copy()
    matches["child_id_node"] = pd.to_numeric(matches["child_id_node"], errors="coerce")
    matches["matched_parent_node_id"] = pd.to_numeric(matches["matched_parent_node_id"], errors="coerce")
    matches = matches.dropna(subset=["child_id_node", "matched_parent_node_id"]).copy()
    if matches.empty:
        return set(), set(), issues

    matches["child_id_node"] = matches["child_id_node"].astype(int)
    matches["matched_parent_node_id"] = matches["matched_parent_node_id"].astype(int)
    if "match_distance" in matches.columns:
        matches["match_distance"] = pd.to_numeric(matches["match_distance"], errors="coerce")
    else:
        matches["match_distance"] = np.nan
    if "parent_node_order" in matches.columns:
        matches["parent_node_order"] = pd.to_numeric(matches["parent_node_order"], errors="coerce")
    else:
        matches["parent_node_order"] = np.nan

    canonical_inlet_nodes: set[int] = set()
    canonical_outlet_nodes: set[int] = set()
    topology_source_node_set = {int(node_id) for node_id in topology_source_nodes}
    topology_sink_node_set = {int(node_id) for node_id in topology_sink_nodes}
    for parent_node_id, flags in parent_lookup.items():
        is_inlet = bool(flags.get("is_inlet", False))
        is_outlet = bool(flags.get("is_outlet", False))
        if not is_inlet and not is_outlet:
            continue

        candidates = matches.loc[matches["matched_parent_node_id"] == int(parent_node_id)].copy()
        if candidates.empty:
            fallback_candidates = matches.iloc[0:0].copy()
            if is_inlet and topology_source_node_set:
                fallback_candidates = matches.loc[matches["child_id_node"].isin(topology_source_node_set)].copy()
                if not fallback_candidates.empty:
                    fallback_candidates = fallback_candidates.sort_values(
                        ["parent_node_order", "match_distance", "child_id_node"],
                        kind="mergesort",
                        na_position="last",
                    )
            if is_outlet and topology_sink_node_set:
                fallback_candidates = matches.loc[matches["child_id_node"].isin(topology_sink_node_set)].copy()
                if not fallback_candidates.empty:
                    fallback_candidates = fallback_candidates.sort_values(
                        ["parent_node_order", "match_distance", "child_id_node"],
                        ascending=[False, True, True],
                        kind="mergesort",
                        na_position="last",
                    )
            if fallback_candidates.empty:
                issues.append(f"No child node match was found for canonical parent boundary node {int(parent_node_id)}.")
                continue
            candidates = fallback_candidates

        if is_inlet and "child_is_inlet_raw" in candidates.columns:
            preferred = candidates.loc[candidates["child_is_inlet_raw"].fillna(False)]
            if not preferred.empty:
                candidates = preferred.copy()
        if is_outlet and "child_is_outlet_raw" in candidates.columns:
            preferred = candidates.loc[candidates["child_is_outlet_raw"].fillna(False)]
            if not preferred.empty:
                candidates = preferred.copy()

        candidates = candidates.sort_values(
            ["match_distance", "child_id_node"],
            kind="mergesort",
            na_position="last",
        )
        child_node_id = int(candidates.iloc[0]["child_id_node"])
        if is_inlet:
            canonical_inlet_nodes.add(child_node_id)
        if is_outlet:
            canonical_outlet_nodes.add(child_node_id)

    return canonical_inlet_nodes, canonical_outlet_nodes, issues


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
                        resolved_primary_ids = list(primary_ids)
                        lineage_method = "matched_parent_node_path"
                        exact_parent_ids = []
                        for parent_id in primary_ids:
                            parent_geometry = parent_edge_by_link_id.get(int(parent_id), {}).get("geometry")
                            if parent_geometry is not None and row.geometry.equals(parent_geometry):
                                exact_parent_ids.append(int(parent_id))
                        # A node path through parallel edges includes every
                        # edge key between the same two nodes. If exactly one
                        # of those edges is geometrically identical to the
                        # child, that equality is definitive one-to-one
                        # evidence and must override the ambiguous node path.
                        if len(primary_ids) > 1 and len(exact_parent_ids) == 1:
                            resolved_primary_ids = exact_parent_ids
                            dominant = primary_rows.loc[
                                primary_rows["parent_id_link"].astype(int).eq(exact_parent_ids[0])
                            ].iloc[0]
                            lineage_method = "exact_geometry_override_parent_node_path"
                        matched_metrics = _compute_matched_set_overlap_metrics(
                            child_geometry=row.geometry,
                            parent_link_ids=resolved_primary_ids,
                            parent_edge_by_link_id=parent_edge_by_link_id,
                            match_tolerance=match_tolerance,
                        )
                        touch_ids = [
                            int(parent_id)
                            for parent_id in ordered["parent_id_link"]
                            if int(parent_id) not in set(resolved_primary_ids)
                        ]
                        lineage_type = (
                            "unchanged_1to1"
                            if len(resolved_primary_ids) == 1
                            else "collapsed_many_to_one"
                        )
                        records.append(
                            {
                                "id_link": child_id,
                                "lineage_type": lineage_type,
                                "dominant_parent_link_id": int(dominant["parent_id_link"]),
                                "matched_parent_link_ids": _serialize_int_list(resolved_primary_ids),
                                "primary_parent_link_ids": _serialize_int_list(resolved_primary_ids),
                                "secondary_parent_link_ids": "",
                                "touch_parent_link_ids": _serialize_int_list(touch_ids),
                                "candidate_parent_link_ids": _serialize_int_list(
                                    _ensure_unique_preserve_order(
                                        list(primary_ids) + [int(value) for value in ordered["parent_id_link"]]
                                    )
                                ),
                                "matched_parent_link_count": int(len(resolved_primary_ids)),
                                "primary_parent_link_count": int(len(resolved_primary_ids)),
                                "secondary_parent_link_count": 0,
                                "touch_parent_link_count": int(len(touch_ids)),
                                "dominant_parent_overlap_fraction": float(dominant["child_overlap_fraction"]),
                                "dominant_parent_overlap_length": float(dominant["child_overlap_length"]),
                                "dominant_parent_core_overlap_fraction": float(dominant["child_core_overlap_fraction"]),
                                "dominant_parent_core_overlap_length": float(dominant["child_core_overlap_length"]),
                                **matched_metrics,
                                "lineage_method": lineage_method,
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


def _selection_anchor_nodes(
    *,
    resolved_unit_ids: Sequence[int],
    units_by_id: Mapping[int, StructuralUnit],
) -> tuple[int | None, int | None]:
    selected_units = [units_by_id[int(unit_id)] for unit_id in resolved_unit_ids if int(unit_id) in units_by_id]
    if not selected_units:
        return None, None

    selection_graph = nx.MultiDiGraph()
    for unit in selected_units:
        for upstream, downstream, edge_key in unit.edge_set:
            selection_graph.add_edge(int(upstream), int(downstream), key=int(edge_key))

    if selection_graph.number_of_edges() > 0:
        source_nodes = sorted(
            int(node_id)
            for node_id in selection_graph.nodes
            if selection_graph.in_degree(node_id) == 0 and selection_graph.out_degree(node_id) > 0
        )
        sink_nodes = sorted(
            int(node_id)
            for node_id in selection_graph.nodes
            if selection_graph.out_degree(node_id) == 0 and selection_graph.in_degree(node_id) > 0
        )
        if len(source_nodes) == 1 and len(sink_nodes) == 1:
            return source_nodes[0], sink_nodes[0]

    if len(selected_units) == 1:
        unit = selected_units[0]
        return int(unit.bifurcation), int(unit.confluence)

    upstream_candidates = sorted(int(unit.bifurcation) for unit in selected_units)
    downstream_candidates = sorted(int(unit.confluence) for unit in selected_units)
    if upstream_candidates and downstream_candidates:
        return upstream_candidates[0], downstream_candidates[-1]
    return None, None


def _selection_edge_pairs(
    *,
    resolved_unit_ids: Sequence[int],
    units_by_id: Mapping[int, StructuralUnit],
) -> set[tuple[int, int]]:
    edge_pairs: set[tuple[int, int]] = set()
    for unit_id in resolved_unit_ids:
        unit = units_by_id.get(int(unit_id))
        if unit is None:
            continue
        for upstream, downstream, edge_key in unit.edge_set:
            _ = edge_key
            edge_pairs.add((int(upstream), int(downstream)))
    return edge_pairs


def _parent_interval_lengths(
    parent_graph: nx.MultiDiGraph,
    parent_node_path: Sequence[int],
) -> list[float]:
    lengths: list[float] = []
    for upstream, downstream in zip(parent_node_path[:-1], parent_node_path[1:]):
        if not parent_graph.has_edge(int(upstream), int(downstream)):
            return []
        edge_lengths: list[float] = []
        for edge_key, attrs in parent_graph[int(upstream)][int(downstream)].items():
            _ = edge_key
            # Anchor-interface placement must use the same physical-link
            # length convention as the hierarchy and equivalent-resistance
            # workflows. ``len_adj`` is only a legacy fallback.
            value = attrs.get("len")
            if value is None or pd.isna(value):
                value = attrs.get("len_adj")
            if value is None or pd.isna(value):
                geometry = attrs.get("geometry")
                value = float(geometry.length) if geometry is not None else 0.0
            edge_lengths.append(max(float(value), 0.0))
        if not edge_lengths:
            return []
        lengths.append(float(np.mean(edge_lengths)))
    return lengths


def _split_line_geometry(
    geometry: LineString,
    cut_distances: Sequence[float],
) -> tuple[list[LineString], list[Point]]:
    ordered_distances = sorted(
        {
            max(0.0, min(float(distance), float(geometry.length)))
            for distance in cut_distances
            if float(distance) > 0.0 and float(distance) < float(geometry.length)
        }
    )
    if not ordered_distances:
        return [LineString(geometry)], []

    points = [geometry.interpolate(distance) for distance in ordered_distances]
    boundaries = [0.0] + ordered_distances + [float(geometry.length)]
    segments: list[LineString] = []
    for start_distance, end_distance in zip(boundaries[:-1], boundaries[1:]):
        segment = substring(geometry, start_distance, end_distance)
        if segment.is_empty:
            continue
        if isinstance(segment, LineString):
            segments.append(segment)
            continue
        merged = linemerge(segment)
        if isinstance(merged, LineString):
            segments.append(merged)
        elif isinstance(merged, MultiLineString):
            coords: list[tuple[float, float]] = []
            for part in merged.geoms:
                part_coords = list(part.coords)
                if not coords:
                    coords.extend(part_coords)
                elif coords[-1] == part_coords[0]:
                    coords.extend(part_coords[1:])
                else:
                    coords.extend(part_coords)
            segments.append(LineString(coords))
        else:
            raise TypeError(f"Unexpected split geometry type: {type(segment)!r}")
    return segments, points


def _split_ordered_samples(
    values: Sequence[Any],
    *,
    cut_fractions: Sequence[float],
) -> list[list[Any]]:
    n_segments = len(cut_fractions) + 1
    if n_segments <= 1:
        return [list(values)]
    if not values:
        return [[] for _ in range(n_segments)]
    if len(values) == 1:
        sample = list(values)
        return [sample.copy() for _ in range(n_segments)]

    max_point_index = len(values) - 1
    cut_indices: list[int] = []
    previous_index = 0
    for position, fraction in enumerate(cut_fractions, start=1):
        raw_index = int(round(float(fraction) * max_point_index))
        remaining_cuts = len(cut_fractions) - position
        lower_bound = previous_index + 1
        upper_bound = max_point_index - max(remaining_cuts, 0)
        cut_index = max(lower_bound, min(raw_index, upper_bound))
        cut_indices.append(cut_index)
        previous_index = cut_index

    segments: list[list[Any]] = []
    start_index = 0
    for cut_index in cut_indices:
        segments.append(list(values[start_index : cut_index + 1]))
        start_index = cut_index
    segments.append(list(values[start_index:]))
    return segments


def _split_ordered_geometry_samples(
    values: Sequence[Any],
    *,
    geometry: LineString,
    cut_distances: Sequence[float],
) -> list[list[Any]]:
    """Split vertex-aligned samples at their nearest along-line cut vertices."""
    coordinates = np.asarray(geometry.coords, dtype=float)
    if len(values) != len(coordinates) or len(coordinates) < 2:
        total_length = float(geometry.length)
        cut_fractions = [
            max(0.0, min(float(distance) / total_length, 1.0))
            if total_length > 0
            else 0.5
            for distance in cut_distances
        ]
        return _split_ordered_samples(values, cut_fractions=cut_fractions)

    step_lengths = np.linalg.norm(np.diff(coordinates, axis=0), axis=1)
    sample_distances = np.concatenate(([0.0], np.cumsum(step_lengths)))
    ordered_cuts = sorted(
        {
            max(0.0, min(float(distance), float(sample_distances[-1])))
            for distance in cut_distances
        }
    )
    cut_fractions = [
        float(np.argmin(np.abs(sample_distances - distance))) / float(len(values) - 1)
        for distance in ordered_cuts
    ]
    return _split_ordered_samples(values, cut_fractions=cut_fractions)


def _split_serialized_link_samples(
    row_data: Mapping[str, Any],
    *,
    geometry: LineString,
    cut_distances: Sequence[float],
) -> dict[str, Any]:
    idx_link_segments = _split_ordered_geometry_samples(
        _parse_int_list(row_data.get("idx_link")),
        geometry=geometry,
        cut_distances=cut_distances,
    )
    wid_pix_segments = _split_ordered_geometry_samples(
        _parse_float_list(row_data.get("wid_pix")),
        geometry=geometry,
        cut_distances=cut_distances,
    )
    return {
        "idx_link": [_serialize_int_list(segment) for segment in idx_link_segments],
        "wid_pix": [_serialize_float_array(segment) for segment in wid_pix_segments],
        "wid_pix_values": wid_pix_segments,
    }


def _refresh_segmented_legacy_width_fields(
    row_data: Mapping[str, Any],
    *,
    geometry: LineString,
    width_values: Sequence[float],
) -> dict[str, Any]:
    """Recompute RivGraph legacy width/length summaries for one split link."""
    updated = dict(row_data)
    widths = np.asarray(width_values, dtype=float)
    valid_widths = np.isfinite(widths) & (widths > 0)
    if widths.size == 0 or not valid_widths.any():
        return updated
    if not valid_widths.all():
        fallback_width = float(np.median(widths[valid_widths]))
        widths = np.where(valid_widths, widths, fallback_width)

    coordinates = np.asarray(geometry.coords, dtype=float)
    if len(coordinates) == len(widths):
        step_lengths = np.linalg.norm(np.diff(coordinates, axis=0), axis=1)
        distances = np.concatenate(([0.0], np.cumsum(step_lengths)))
        sample_points = coordinates
    else:
        distances = np.linspace(0.0, float(geometry.length), len(widths))
        sample_points = np.asarray(
            [
                geometry.interpolate(float(distance)).coords[0]
                for distance in distances
            ],
            dtype=float,
        )

    total_distance = float(distances[-1]) if len(distances) else 0.0
    reverse_distances = total_distance - distances[::-1]
    start_index = int(
        np.argmin(
            np.abs(
                distances
                - widths[0] / 2.0 * RIVGRAPH_WIDTH_TRIM_MULTIPLIER
            )
        )
    )
    end_offset = int(
        np.argmin(
            np.abs(
                reverse_distances
                - widths[-1] / 2.0 * RIVGRAPH_WIDTH_TRIM_MULTIPLIER
            )
        )
    )
    end_index = len(distances) - end_offset - 1

    if start_index >= end_index:
        adjusted_widths = widths
        adjusted_length = total_distance
        start_point = sample_points[0]
        end_point = sample_points[-1]
    else:
        adjusted_widths = widths[start_index:end_index]
        adjusted_length = float(distances[end_index] - distances[start_index])
        start_point = sample_points[start_index]
        end_point = sample_points[end_index]

    positive_steps = np.diff(distances)
    positive_steps = positive_steps[positive_steps > 0]
    pixel_length = float(np.median(positive_steps)) if positive_steps.size else 0.0
    chord = float(np.linalg.norm(end_point - start_point))

    updated["wid"] = max(pixel_length, float(np.mean(widths)))
    updated["wid_adj"] = max(pixel_length, float(np.mean(adjusted_widths)))
    updated["wid_med"] = float(np.median(adjusted_widths))
    updated["len_adj"] = max(pixel_length, adjusted_length)
    updated["sinuosity"] = adjusted_length / chord if chord > 0 else 1.0
    return updated


def _selected_parent_subpaths(
    parent_node_path: Sequence[int],
    *,
    selection_edge_pairs: Collection[tuple[int, int]],
) -> list[list[int]]:
    if len(parent_node_path) < 2 or not selection_edge_pairs:
        return []

    runs: list[list[int]] = []
    current_run: list[int] = []
    for upstream, downstream in zip(parent_node_path[:-1], parent_node_path[1:]):
        if (int(upstream), int(downstream)) in selection_edge_pairs:
            if not current_run:
                current_run = [int(upstream), int(downstream)]
            elif int(current_run[-1]) == int(upstream):
                current_run.append(int(downstream))
            else:
                runs.append(current_run)
                current_run = [int(upstream), int(downstream)]
        elif current_run:
            runs.append(current_run)
            current_run = []
    if current_run:
        runs.append(current_run)
    return runs


def _boundary_overlap_cut_targets(
    *,
    parent_graph: nx.MultiDiGraph,
    parent_node_geometries: Mapping[int, Point],
    directed_links: gpd.GeoDataFrame,
    node_match: pd.DataFrame,
    link_match: pd.DataFrame,
    selection_edge_pairs: Collection[tuple[int, int]],
    nominal_upstream_node: int,
    nominal_downstream_node: int,
    parent_overlap_threshold: float = 0.95,
) -> dict[int, list[dict[str, Any]]]:
    if link_match.empty or not selection_edge_pairs:
        return {}

    matched_parent_node_ids = set(
        pd.to_numeric(node_match.get("matched_parent_node_id"), errors="coerce").dropna().astype(int).tolist()
    )
    child_geometries = {
        int(row.id_link): row.geometry
        for row in directed_links.itertuples(index=False)
    }

    selected_node_ids = {
        int(node_id)
        for upstream, downstream in selection_edge_pairs
        for node_id in (int(upstream), int(downstream))
    }

    boundary_specs: list[tuple[str, int, list[int]]] = []
    seen_specs: set[tuple[str, int, tuple[int, ...]]] = set()
    for upstream, downstream, key in parent_graph.edges(keys=True):
        parent_edge = (int(upstream), int(downstream))
        if parent_edge in selection_edge_pairs:
            continue
        upstream_in = int(upstream) in selected_node_ids
        downstream_in = int(downstream) in selected_node_ids
        if upstream_in == downstream_in:
            continue
        if upstream_in:
            role = "selected_upstream"
            boundary_node_id = int(upstream)
        else:
            role = "selected_downstream"
            boundary_node_id = int(downstream)
        spec_key = (role, boundary_node_id, (int(key),))
        if spec_key in seen_specs:
            continue
        seen_specs.add(spec_key)
        boundary_specs.append((role, boundary_node_id, [int(key)]))

    targets: dict[int, list[dict[str, Any]]] = {}
    for role, boundary_node_id, candidate_parent_links in boundary_specs:
        if boundary_node_id in matched_parent_node_ids:
            continue
        if not candidate_parent_links:
            continue
        boundary_point = parent_node_geometries.get(int(boundary_node_id))
        if boundary_point is None or boundary_point.is_empty:
            continue

        rows = link_match.loc[
            pd.to_numeric(link_match["parent_id_link"], errors="coerce").isin(candidate_parent_links)
            & pd.to_numeric(link_match["parent_overlap_fraction"], errors="coerce").ge(parent_overlap_threshold)
        ].copy()
        if rows.empty:
            continue

        rows["candidate_rank"] = pd.to_numeric(rows["candidate_rank"], errors="coerce")
        rows["child_overlap_fraction"] = pd.to_numeric(rows["child_overlap_fraction"], errors="coerce")
        rows["boundary_point_distance"] = [
            float(boundary_point.distance(child_geometries[int(child_id)]))
            for child_id in rows["child_id_link"]
        ]
        rows = rows.sort_values(
            ["boundary_point_distance", "candidate_rank", "child_overlap_fraction", "parent_id_link"],
            ascending=[True, True, False, True],
            kind="mergesort",
        ).reset_index(drop=True)
        selected = rows.iloc[0]
        child_id_link = int(selected["child_id_link"])
        child_geometry = child_geometries[child_id_link]
        cut_distance = float(child_geometry.project(boundary_point))
        if cut_distance <= 0.0 or cut_distance >= float(child_geometry.length):
            continue
        targets.setdefault(child_id_link, []).append(
            {
                "parent_node_id": int(boundary_node_id),
                "cut_distance": cut_distance,
                "role": role,
                "source_parent_link_id": int(selected["parent_id_link"]),
            }
        )

    return targets


def _segment_parent_paths(
    parent_node_path: Sequence[int],
    internal_cut_nodes: Sequence[int],
) -> list[list[int]]:
    if not parent_node_path:
        return []
    ordered_cuts = [int(node_id) for node_id in internal_cut_nodes if int(node_id) in parent_node_path[1:-1]]
    if not ordered_cuts:
        return [[int(node_id) for node_id in parent_node_path]]

    segments: list[list[int]] = []
    start_index = 0
    for cut_node in ordered_cuts:
        cut_index = parent_node_path.index(int(cut_node))
        segments.append([int(node_id) for node_id in parent_node_path[start_index : cut_index + 1]])
        start_index = cut_index
    segments.append([int(node_id) for node_id in parent_node_path[start_index:]])
    return segments


def _segment_contains_nominal_interval(
    *,
    segment_parent_path: Sequence[int],
    nominal_upstream_node: int,
    nominal_downstream_node: int,
) -> bool:
    if not segment_parent_path:
        return False
    if nominal_upstream_node in segment_parent_path and nominal_downstream_node in segment_parent_path:
        return (
            segment_parent_path.index(int(nominal_upstream_node))
            < segment_parent_path.index(int(nominal_downstream_node))
        )
    if nominal_upstream_node in segment_parent_path and nominal_upstream_node != int(segment_parent_path[-1]):
        return True
    if nominal_downstream_node in segment_parent_path and nominal_downstream_node != int(segment_parent_path[0]):
        return True
    return False


def _line_length_fields_updated(
    row_data: Mapping[str, Any],
    geometry: LineString,
) -> dict[str, Any]:
    updated = dict(row_data)
    segment_length = float(geometry.length)
    chord = float(Point(geometry.coords[0]).distance(Point(geometry.coords[-1])))
    for column in ("len", "len_adj", "link_length_m", "parent_link_length_m"):
        if column in updated:
            updated[column] = segment_length
    if "sinuosity" in updated:
        updated["sinuosity"] = segment_length / chord if chord > 0 else 1.0
    if "n_nodes" in updated:
        updated["n_nodes"] = 2
    return updated


def _refresh_explicit_directed_nodes(
    *,
    directed_links: gpd.GeoDataFrame,
    directed_nodes: gpd.GeoDataFrame,
    node_geometry_overrides: Mapping[int, Point] | None = None,
) -> tuple[gpd.GeoDataFrame, dict[str, Any]]:
    result = directed_nodes.copy()
    geometry_lookup = {
        int(row.id_node): row.geometry
        for row in result.itertuples(index=False)
    }
    if node_geometry_overrides:
        geometry_lookup.update({int(node_id): geometry for node_id, geometry in node_geometry_overrides.items()})

    endpoint_node_ids = sorted(
        {
            int(node_id)
            for node_id in pd.concat(
                [
                    pd.to_numeric(directed_links["id_us_node"], errors="coerce"),
                    pd.to_numeric(directed_links["id_ds_node"], errors="coerce"),
                ],
                ignore_index=True,
            ).dropna().astype(int).tolist()
        }
    )
    result = result.loc[pd.to_numeric(result["id_node"], errors="coerce").isin(endpoint_node_ids)].copy()
    existing_ids = set(pd.to_numeric(result["id_node"], errors="coerce").dropna().astype(int).tolist())
    missing_ids = [node_id for node_id in endpoint_node_ids if node_id not in existing_ids]
    if missing_ids:
        template = {column: pd.NA for column in result.columns}
        if result.empty:
            template["schema_rg"] = "rg-v1"
        elif "schema_rg" in result.columns:
            non_null_schema = result["schema_rg"].dropna()
            template["schema_rg"] = non_null_schema.iloc[0] if not non_null_schema.empty else "rg-v1"
        missing_rows: list[dict[str, Any]] = []
        for node_id in missing_ids:
            row = dict(template)
            row["id_node"] = int(node_id)
            row["geometry"] = geometry_lookup[int(node_id)]
            missing_rows.append(row)
        result = pd.concat(
            [result, gpd.GeoDataFrame(missing_rows, geometry="geometry", crs=directed_nodes.crs)],
            ignore_index=True,
        )

    result["id_node"] = pd.to_numeric(result["id_node"], errors="coerce").astype(int)
    result["geometry"] = [geometry_lookup[int(node_id)] for node_id in result["id_node"].tolist()]
    result = result.sort_values("id_node", kind="mergesort").reset_index(drop=True)
    if "idx_node" in result.columns:
        result["idx_node"] = np.arange(len(result), dtype=int)

    graph = build_directed_graph(directed_links, result, prefer_explicit_direction=True)
    report = validate_single_inlet_single_outlet(graph, require_flag_match=False)
    source_nodes = sorted(int(node_id) for node_id in report.source_nodes)
    sink_nodes = sorted(int(node_id) for node_id in report.sink_nodes)
    source_node_set = set(source_nodes)
    sink_node_set = set(sink_nodes)

    id_links_values: list[str] = []
    n_links_values: list[int] = []
    in_degree_values: list[int] = []
    out_degree_values: list[int] = []
    for node_id in result["id_node"].astype(int):
        incident = sorted(int(link_id) for _, _, link_id in graph.in_edges(node_id, keys=True)) + sorted(
            int(link_id) for _, _, link_id in graph.out_edges(node_id, keys=True)
        )
        id_links_values.append(_serialize_int_list(incident))
        n_links_values.append(int(len(incident)))
        in_degree_values.append(int(graph.in_degree(node_id)))
        out_degree_values.append(int(graph.out_degree(node_id)))

    result["id_links"] = id_links_values
    result["n_links"] = n_links_values
    result["in_degree"] = in_degree_values
    result["out_degree"] = out_degree_values
    result["is_inlet"] = result["id_node"].astype(int).isin(source_node_set)
    result["is_outlet"] = result["id_node"].astype(int).isin(sink_node_set)
    result["type_io"] = [
        _type_io_label(is_inlet=bool(is_inlet), is_outlet=bool(is_outlet))
        for is_inlet, is_outlet in zip(result["is_inlet"], result["is_outlet"])
    ]

    summary = {
        "is_valid": bool(report.is_valid),
        "source_nodes": source_nodes,
        "sink_nodes": sink_nodes,
        "validation_report": report.to_dict(),
        "issues": list(report.issues),
    }
    return result, summary


def _segment_collapsed_links_to_anchor_nodes(
    *,
    parent_graph: nx.MultiDiGraph,
    parent_nodes: gpd.GeoDataFrame,
    parent_node_order: Mapping[int, int],
    directed_links: gpd.GeoDataFrame,
    directed_nodes: gpd.GeoDataFrame,
    node_match: pd.DataFrame,
    link_match: pd.DataFrame,
    link_lineage: pd.DataFrame,
    selection_edge_pairs: Collection[tuple[int, int]],
    nominal_upstream_node: int | None,
    nominal_downstream_node: int | None,
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, pd.DataFrame, dict[str, Any]]:
    if nominal_upstream_node is None or nominal_downstream_node is None or not selection_edge_pairs:
        return directed_links, directed_nodes, node_match, {"n_anchor_splits": 0, "mode_applied": False}

    lineage_by_link = link_lineage.set_index("id_link", drop=False) if not link_lineage.empty else pd.DataFrame()
    parent_node_flags = (
        parent_nodes.set_index("id_node", drop=False)[["is_inlet", "is_outlet"]].to_dict("index")
        if not parent_nodes.empty and {"id_node", "is_inlet", "is_outlet"}.issubset(parent_nodes.columns)
        else {}
    )
    parent_node_geometries = (
        parent_nodes.set_index("id_node", drop=False)["geometry"].to_dict()
        if not parent_nodes.empty and {"id_node", "geometry"}.issubset(parent_nodes.columns)
        else {}
    )
    geometry_overrides: dict[int, Point] = {}
    updated_rows: list[dict[str, Any]] = []
    next_link_id = int(pd.to_numeric(directed_links["id_link"], errors="coerce").dropna().astype(int).max()) + 1
    next_node_id = int(pd.to_numeric(directed_nodes["id_node"], errors="coerce").dropna().astype(int).max()) + 1
    node_match_rows = node_match.to_dict("records")
    n_anchor_splits = 0
    n_new_nodes = 0
    n_new_links = 0
    boundary_cut_targets = _boundary_overlap_cut_targets(
        parent_graph=parent_graph,
        parent_node_geometries=parent_node_geometries,
        directed_links=directed_links,
        node_match=node_match,
        link_match=link_match,
        selection_edge_pairs=selection_edge_pairs,
        nominal_upstream_node=int(nominal_upstream_node),
        nominal_downstream_node=int(nominal_downstream_node),
    )

    for row in directed_links.itertuples(index=False):
        row_data = row._asdict()
        link_id = int(row_data["id_link"])
        boundary_specs = sorted(
            boundary_cut_targets.get(link_id, []),
            key=lambda item: float(item["cut_distance"]),
        )
        if (lineage_by_link.empty or link_id not in lineage_by_link.index) and not boundary_specs:
            updated_rows.append(dict(row_data))
            continue

        lineage_row = lineage_by_link.loc[link_id] if not lineage_by_link.empty and link_id in lineage_by_link.index else None
        parent_node_path = _parse_int_list(lineage_row.get("matched_parent_node_path", "")) if lineage_row is not None else []
        if len(parent_node_path) < 3 and not boundary_specs:
            updated_rows.append(dict(row_data))
            continue

        candidate_parent_runs = _selected_parent_subpaths(
            parent_node_path,
            selection_edge_pairs=selection_edge_pairs,
        )

        selected_parent_path: list[int] | None = None
        internal_cut_nodes: list[int] = []
        use_boundary_overlap_fallback = False
        if candidate_parent_runs:
            selected_parent_path = max(
                candidate_parent_runs,
                key=lambda path: (
                    nominal_upstream_node in path and nominal_downstream_node in path,
                    nominal_upstream_node in path or nominal_downstream_node in path,
                    len(path),
                ),
            )
            if len(selected_parent_path) >= 2:
                internal_cut_nodes = [
                    int(node_id)
                    for node_id in (int(selected_parent_path[0]), int(selected_parent_path[-1]))
                    if int(node_id) in parent_node_path[1:-1]
                ]
                internal_cut_nodes = [
                    node_id for node_id in parent_node_path[1:-1] if int(node_id) in set(internal_cut_nodes)
                ]
        if not internal_cut_nodes and boundary_specs:
            use_boundary_overlap_fallback = True
        elif not internal_cut_nodes:
            updated_rows.append(dict(row_data))
            continue

        projected_cut_distances: list[float] = []
        fallback_cut_distances: list[float] = []
        cut_parent_nodes: list[int] = []
        cut_roles: list[str] = []
        if use_boundary_overlap_fallback:
            cut_distances = [float(spec["cut_distance"]) for spec in boundary_specs]
            cut_parent_nodes = [int(spec["parent_node_id"]) for spec in boundary_specs]
            cut_roles = [str(spec["role"]) for spec in boundary_specs]
        else:
            interval_lengths = _parent_interval_lengths(parent_graph, parent_node_path)
            if not interval_lengths:
                updated_rows.append(dict(row_data))
                continue

            cumulative_lengths = [0.0]
            for interval_length in interval_lengths:
                cumulative_lengths.append(cumulative_lengths[-1] + float(interval_length))
            total_length = cumulative_lengths[-1]
            if total_length <= 0:
                updated_rows.append(dict(row_data))
                continue

            for node_id in internal_cut_nodes:
                cut_index = parent_node_path.index(int(node_id))
                fraction = cumulative_lengths[cut_index] / total_length
                if fraction <= 0.0 or fraction >= 1.0:
                    continue
                fallback_cut_distances.append(float(row.geometry.length) * fraction)
                parent_geometry = parent_node_geometries.get(int(node_id))
                if parent_geometry is None or parent_geometry.is_empty:
                    projected_cut_distances.append(float(row.geometry.length) * fraction)
                else:
                    projected_cut_distances.append(float(row.geometry.project(parent_geometry)))
                cut_parent_nodes.append(int(node_id))
            if not fallback_cut_distances:
                updated_rows.append(dict(row_data))
                continue

            cut_distances = projected_cut_distances
            if any(
                distance <= 0.0 or distance >= float(row.geometry.length)
                for distance in cut_distances
            ) or any(
                next_distance <= current_distance
                for current_distance, next_distance in zip(cut_distances[:-1], cut_distances[1:])
            ):
                cut_distances = fallback_cut_distances

        segments, points = _split_line_geometry(row.geometry, cut_distances)
        if len(segments) <= 1:
            updated_rows.append(dict(row_data))
            continue

        if use_boundary_overlap_fallback:
            parent_path_segments = [[] for _ in segments]
        else:
            parent_path_segments = _segment_parent_paths(parent_node_path, cut_parent_nodes)
            if len(parent_path_segments) != len(segments):
                updated_rows.append(dict(row_data))
                continue

        split_samples = _split_serialized_link_samples(
            row_data,
            geometry=row.geometry,
            cut_distances=cut_distances,
        )
        idx_link_segments = split_samples["idx_link"]
        wid_pix_segments = split_samples["wid_pix"]
        wid_pix_value_segments = split_samples["wid_pix_values"]

        split_node_ids: list[int] = []
        for point, parent_node_id in zip(points, cut_parent_nodes):
            new_node_id = next_node_id
            next_node_id += 1
            geometry_overrides[new_node_id] = point
            split_node_ids.append(new_node_id)
            n_new_nodes += 1
            flags = parent_node_flags.get(int(parent_node_id), {})
            node_match_row = {column: pd.NA for column in node_match.columns}
            node_match_row["child_id_node"] = int(new_node_id)
            node_match_row["matched_parent_node_id"] = int(parent_node_id)
            if "match_distance" in node_match_row:
                node_match_row["match_distance"] = 0.0
            if "match_within_tolerance" in node_match_row:
                node_match_row["match_within_tolerance"] = True
            if "parent_node_order" in node_match_row:
                node_match_row["parent_node_order"] = parent_node_order.get(int(parent_node_id), pd.NA)
            if "child_is_inlet_raw" in node_match_row:
                node_match_row["child_is_inlet_raw"] = False
            if "child_is_outlet_raw" in node_match_row:
                node_match_row["child_is_outlet_raw"] = False
            if "parent_is_inlet" in node_match_row:
                node_match_row["parent_is_inlet"] = bool(flags.get("is_inlet", False))
            if "parent_is_outlet" in node_match_row:
                node_match_row["parent_is_outlet"] = bool(flags.get("is_outlet", False))
            node_match_rows.append(node_match_row)

        segment_node_ids = [int(row.id_us_node)] + split_node_ids + [int(row.id_ds_node)]
        keep_index: int | None = None
        if use_boundary_overlap_fallback:
            roles = cut_roles
            if roles == ["selected_upstream"]:
                keep_index = 0
            elif roles == ["selected_downstream"]:
                keep_index = 1
            elif len(roles) == 2:
                keep_index = 1
        else:
            for index, segment_parent_path in enumerate(parent_path_segments):
                if [int(node_id) for node_id in segment_parent_path] == [int(node_id) for node_id in selected_parent_path]:
                    keep_index = index
                    break
        if keep_index is None:
            updated_rows.append(dict(row_data))
            continue

        for index, (segment_geometry, segment_parent_path) in enumerate(zip(segments, parent_path_segments)):
            new_row = _line_length_fields_updated(row_data, segment_geometry)
            new_row["geometry"] = segment_geometry
            # Preserve the source link's pre-anchor provenance verbatim.
            # Splitting a changed realized link at scientific interfaces does
            # not prove that any resulting piece is an unchanged parent link.
            # The local-patch adapter revalidates both eligibility and parent
            # identity after segmentation.
            if "idx_link" in new_row and index < len(idx_link_segments):
                new_row["idx_link"] = idx_link_segments[index]
            if "wid_pix" in new_row and index < len(wid_pix_segments):
                new_row["wid_pix"] = wid_pix_segments[index]
            if index < len(wid_pix_value_segments):
                new_row = _refresh_segmented_legacy_width_fields(
                    new_row,
                    geometry=segment_geometry,
                    width_values=wid_pix_value_segments[index],
                )
            new_row["id_us_node"] = int(segment_node_ids[index])
            new_row["id_ds_node"] = int(segment_node_ids[index + 1])
            new_row["id_nodes"] = f"{int(segment_node_ids[index])}, {int(segment_node_ids[index + 1])}"
            if "raw_id_nodes" in new_row:
                new_row["raw_id_nodes"] = new_row["id_nodes"]
            segment_is_inlet = bool(row_data.get("is_inlet", False)) and index == 0
            segment_is_outlet = bool(row_data.get("is_outlet", False)) and index == (len(segments) - 1)
            if "is_inlet" in new_row:
                new_row["is_inlet"] = segment_is_inlet
            if "is_outlet" in new_row:
                new_row["is_outlet"] = segment_is_outlet
            if "type_io" in new_row:
                new_row["type_io"] = _type_io_label(is_inlet=segment_is_inlet, is_outlet=segment_is_outlet)
            new_row["id_link"] = int(link_id) if index == keep_index else int(next_link_id)
            if index != keep_index:
                next_link_id += 1
                n_new_links += 1
            updated_rows.append(new_row)
        n_anchor_splits += 1

    if n_anchor_splits == 0:
        return directed_links, directed_nodes, node_match, {"n_anchor_splits": 0, "mode_applied": False}

    updated_links = gpd.GeoDataFrame(updated_rows, geometry="geometry", crs=directed_links.crs)
    updated_links["id_link"] = pd.to_numeric(updated_links["id_link"], errors="coerce").astype(int)
    updated_links = updated_links.sort_values("id_link", kind="mergesort").reset_index(drop=True)

    updated_node_match = pd.DataFrame.from_records(node_match_rows, columns=node_match.columns if not node_match.empty else None)
    if not updated_node_match.empty and "child_id_node" in updated_node_match.columns:
        updated_node_match["child_id_node"] = pd.to_numeric(updated_node_match["child_id_node"], errors="coerce").astype(int)
        if "matched_parent_node_id" in updated_node_match.columns:
            updated_node_match["matched_parent_node_id"] = pd.to_numeric(
                updated_node_match["matched_parent_node_id"],
                errors="coerce",
            ).astype("Int64")
        updated_node_match = updated_node_match.sort_values(
            ["child_id_node", "matched_parent_node_id"],
            kind="mergesort",
        ).reset_index(drop=True)

    updated_nodes, node_summary = _refresh_explicit_directed_nodes(
        directed_links=updated_links,
        directed_nodes=directed_nodes,
        node_geometry_overrides=geometry_overrides,
    )
    summary = {
        "mode_applied": True,
        "n_anchor_splits": int(n_anchor_splits),
        "n_new_nodes": int(n_new_nodes),
        "n_new_links": int(n_new_links),
        "nominal_upstream_node": int(nominal_upstream_node),
        "nominal_downstream_node": int(nominal_downstream_node),
        "validation_report": node_summary["validation_report"],
        "issues": node_summary["issues"],
    }
    return updated_links, updated_nodes, updated_node_match, summary


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


def _build_oriented_child_graph(
    *,
    child_nodes: gpd.GeoDataFrame,
    orientation_by_link: Mapping[int, tuple[int, int]],
) -> nx.MultiDiGraph:
    graph = nx.MultiDiGraph()
    for node_row in child_nodes.itertuples(index=False):
        graph.add_node(int(node_row.id_node), **node_row._asdict())
    for link_id, (upstream, downstream) in orientation_by_link.items():
        graph.add_edge(int(upstream), int(downstream), key=int(link_id), id_link=int(link_id))
    return graph


def _boundary_harmonic_orientation(
    *,
    edge_nodes: Mapping[int, tuple[int, int]],
    child_links: gpd.GeoDataFrame,
    child_nodes: gpd.GeoDataFrame,
    child_inlet_nodes: set[int],
    child_outlet_nodes: set[int],
    allow_additional_topological_terminals: bool = False,
    fixed_orientation_by_link: Mapping[int, tuple[int, int]] | None = None,
    fixed_node_potential_by_node: Mapping[int, float] | None = None,
) -> dict[int, tuple[int, int]]:
    """Orient a connected local graph down a cut-constrained harmonic potential.

    ``fixed_orientation_by_link`` supplies authoritative directions for links
    inherited unchanged from a parent graph.  They participate in the
    potential and reachability scaffold but are never reoriented.  This is
    required for a local replacement assembled from alternating inherited and
    regenerated segments: solving only the regenerated subset can hide the
    true source or sink attachment.
    """
    prescribed_sources = {int(value) for value in child_inlet_nodes}
    prescribed_sinks = {int(value) for value in child_outlet_nodes}
    sources = set(prescribed_sources)
    sinks = set(prescribed_sinks)
    # An open/interconnected selected unit may use one scientific interface
    # node for both incoming and outgoing external continuations. Such a node
    # is not a Dirichlet endpoint of the local potential problem: assigning it
    # both 1 and 0 is contradictory. Keep it as an internal harmonic node and
    # constrain the solve with the remaining role-exclusive interfaces.
    shared_interfaces = sources.intersection(sinks)
    sources.difference_update(shared_interfaces)
    sinks.difference_update(shared_interfaces)
    if not sources or not sinks:
        raise ValueError(
            "Boundary-potential direction fallback requires at least one prescribed "
            "role-exclusive inlet and outlet."
        )

    node_ids = sorted(int(value) for value in child_nodes["id_node"])
    graph = nx.Graph()
    graph.add_nodes_from(node_ids)
    links_by_id = child_links.set_index(child_links["id_link"].astype(int), drop=False)
    conductance: dict[tuple[int, int], float] = {}
    for link_id, (node_a, node_b) in edge_nodes.items():
        length = float(links_by_id.loc[int(link_id)].geometry.length)
        weight = 1.0 / max(length, 1e-9)
        pair = tuple(sorted((int(node_a), int(node_b))))
        conductance[pair] = conductance.get(pair, 0.0) + weight
        graph.add_edge(*pair)
    if not nx.is_connected(graph) or not sources.union(sinks).issubset(graph.nodes):
        raise ValueError(
            "Boundary-potential direction fallback requires one connected graph "
            "containing every prescribed boundary."
        )

    supplied_node_potential = {
        int(node_id): float(value)
        for node_id, value in dict(fixed_node_potential_by_node or {}).items()
        if int(node_id) in graph
    }
    if supplied_node_potential:
        missing_prescribed = sorted(
            prescribed_sources.union(prescribed_sinks).difference(
                supplied_node_potential
            )
        )
        if missing_prescribed:
            raise ValueError(
                "Parent-order boundary potential is missing prescribed interfaces: "
                f"{missing_prescribed}."
            )
        boundary_values = supplied_node_potential
    else:
        boundary_values = {node_id: 1.0 for node_id in sources}
        boundary_values.update({node_id: 0.0 for node_id in sinks})
    internal = [node_id for node_id in node_ids if node_id not in boundary_values]
    potential = dict(boundary_values)
    if internal:
        index = {node_id: idx for idx, node_id in enumerate(internal)}
        matrix = lil_matrix((len(internal), len(internal)), dtype=float)
        right = np.zeros(len(internal), dtype=float)
        adjacency: dict[int, dict[int, float]] = {node_id: {} for node_id in node_ids}
        for (node_a, node_b), weight in conductance.items():
            adjacency[node_a][node_b] = adjacency[node_a].get(node_b, 0.0) + weight
            adjacency[node_b][node_a] = adjacency[node_b].get(node_a, 0.0) + weight
        for node_id in internal:
            row = index[node_id]
            for neighbor, weight in adjacency[node_id].items():
                matrix[row, row] += weight
                if neighbor in index:
                    matrix[row, index[neighbor]] -= weight
                else:
                    right[row] += weight * boundary_values[neighbor]
        solution = np.asarray(spsolve(matrix.tocsr(), right), dtype=float)
        if not np.isfinite(solution).all():
            raise ValueError("Boundary-potential direction fallback produced non-finite values.")
        potential.update({node_id: float(solution[index[node_id]]) for node_id in internal})

    fixed_orientation = {
        int(link_id): (int(nodes[0]), int(nodes[1]))
        for link_id, nodes in dict(fixed_orientation_by_link or {}).items()
    }
    unknown_fixed_ids = sorted(set(fixed_orientation).difference(edge_nodes))
    if unknown_fixed_ids:
        raise ValueError(
            "Fixed boundary-potential orientations reference absent links: "
            f"{unknown_fixed_ids}."
        )
    for link_id, endpoints in fixed_orientation.items():
        if set(endpoints) != set(map(int, edge_nodes[int(link_id)])):
            raise ValueError(
                "Fixed boundary-potential orientation endpoints disagree with "
                f"link {link_id}: fixed={endpoints}, graph={edge_nodes[int(link_id)]}."
            )

    orientation: dict[int, tuple[int, int]] = {}
    tie_tolerance = 1e-12
    distance_to_sink = nx.multi_source_dijkstra_path_length(
        graph, sinks, weight=None
    )
    distance_from_source = nx.multi_source_dijkstra_path_length(
        graph, sources, weight=None
    )
    for link_id, (node_a, node_b) in edge_nodes.items():
        node_a, node_b = int(node_a), int(node_b)
        if int(link_id) in fixed_orientation:
            orientation[int(link_id)] = fixed_orientation[int(link_id)]
            continue
        difference = float(potential[node_a] - potential[node_b])
        if difference > tie_tolerance:
            orientation[int(link_id)] = (node_a, node_b)
        elif difference < -tie_tolerance:
            orientation[int(link_id)] = (node_b, node_a)
        else:
            # Equal-potential connectors occur particularly between multiple
            # prescribed interfaces. Node-ID order is acyclic but can point a
            # source-source connector away from the downstream network. Use
            # cut-to-cut progression first, then a deterministic ID tie-break.
            # Greater sink distance and smaller source distance are upstream.
            rank_a = (
                int(distance_to_sink[node_a]),
                -int(distance_from_source[node_a]),
                -int(node_a),
            )
            rank_b = (
                int(distance_to_sink[node_b]),
                -int(distance_from_source[node_b]),
                -int(node_b),
            )
            orientation[int(link_id)] = (
                (node_a, node_b) if rank_a > rank_b else (node_b, node_a)
            )

    directed = _build_oriented_child_graph(
        child_nodes=child_nodes, orientation_by_link=orientation
    )
    if shared_interfaces or allow_additional_topological_terminals:
        weak_component_count = nx.number_weakly_connected_components(directed)
        unreachable_sources = sorted(
            source
            for source in prescribed_sources
            if not any(
                sink in directed and nx.has_path(directed, source, sink)
                for sink in prescribed_sinks
            )
        )
        unreachable_sinks = sorted(
            sink
            for sink in prescribed_sinks
            if not any(
                source in directed and nx.has_path(directed, source, sink)
                for source in prescribed_sources
            )
        )
        shared_issues = []
        if weak_component_count != 1:
            shared_issues.append(
                f"Expected one weak component, found {weak_component_count}."
            )
        if unreachable_sources:
            shared_issues.append(
                f"Unreachable prescribed sources: {unreachable_sources}."
            )
        if unreachable_sinks:
            shared_issues.append(
                f"Unreachable prescribed sinks: {unreachable_sinks}."
            )
        report_issues = shared_issues
    else:
        report = _validate_variant_direction_boundaries(
            directed,
            child_inlet_nodes=sources,
            child_outlet_nodes=sinks,
            require_flag_match=True,
        )
        report_issues = list(report.issues)
    if report_issues or not nx.is_directed_acyclic_graph(nx.DiGraph(directed)):
        raise ValueError(
            "Boundary-potential direction fallback did not produce a valid cut-to-cut DAG: "
            f"{report_issues}"
        )
    return orientation


def _validate_variant_direction_boundaries(
    graph: nx.MultiDiGraph,
    *,
    child_inlet_nodes: Collection[int],
    child_outlet_nodes: Collection[int],
    require_flag_match: bool,
) -> ValidationReport:
    """Validate either a closed child or an open multi-interface child DAG.

    A local crop of a member of a compound unit can legitimately have several
    anchored inflow and/or outflow interfaces.  Those prescribed interfaces,
    rather than a global one-inlet/one-outlet rule, are authoritative until the
    local graph is merged back into the globally closed parent network.
    """
    expected_sources = sorted({int(value) for value in child_inlet_nodes})
    expected_sinks = sorted({int(value) for value in child_outlet_nodes})
    if len(expected_sources) == 1 and len(expected_sinks) == 1:
        return validate_single_inlet_single_outlet(
            graph, require_flag_match=require_flag_match
        )

    degree_frame = build_degree_frame(graph)
    weak_components = [
        sorted(int(node) for node in component)
        for component in nx.weakly_connected_components(graph)
    ]
    source_nodes = sorted(
        int(node)
        for node in degree_frame.loc[degree_frame["is_source"], "node_id"]
    )
    sink_nodes = sorted(
        int(node)
        for node in degree_frame.loc[degree_frame["is_sink"], "node_id"]
    )
    isolated_nodes = sorted(
        int(node)
        for node in degree_frame.loc[degree_frame["is_isolated"], "node_id"]
    )
    flagged_inlets = sorted(
        int(node)
        for node in degree_frame.loc[degree_frame["flag_is_inlet"], "node_id"]
    )
    flagged_outlets = sorted(
        int(node)
        for node in degree_frame.loc[degree_frame["flag_is_outlet"], "node_id"]
    )
    issues: list[str] = []
    if len(weak_components) != 1:
        issues.append(
            f"Expected one weakly connected component, found {len(weak_components)}."
        )
    if isolated_nodes:
        issues.append(
            "Found isolated node(s) with zero in-degree and zero out-degree: "
            f"{isolated_nodes}."
        )
    if source_nodes != expected_sources:
        issues.append(
            f"Expected prescribed local source nodes {expected_sources}, found {source_nodes}."
        )
    if sink_nodes != expected_sinks:
        issues.append(
            f"Expected prescribed local sink nodes {expected_sinks}, found {sink_nodes}."
        )
    if require_flag_match and flagged_inlets != expected_sources:
        issues.append(
            f"Expected inlet flags {expected_sources}, found {flagged_inlets}."
        )
    if require_flag_match and flagged_outlets != expected_sinks:
        issues.append(
            f"Expected outlet flags {expected_sinks}, found {flagged_outlets}."
        )
    return ValidationReport(
        n_nodes=graph.number_of_nodes(),
        n_edges=graph.number_of_edges(),
        n_weak_components=len(weak_components),
        weak_components=weak_components,
        source_nodes=source_nodes,
        sink_nodes=sink_nodes,
        isolated_nodes=isolated_nodes,
        flagged_inlets=flagged_inlets,
        flagged_outlets=flagged_outlets,
        issues=issues,
    )


def _link_length_lookup(child_links: gpd.GeoDataFrame) -> dict[int, float]:
    return {
        int(row.id_link): float(row.len) if pd.notna(row.len) else float(row.geometry.length)
        for row in child_links.itertuples(index=False)
    }


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


def _link_components(
    link_ids: Sequence[int],
    incident_links: Mapping[int, Sequence[int]],
) -> list[list[int]]:
    link_set = {int(link_id) for link_id in link_ids}
    if not link_set:
        return []

    graph = nx.Graph()
    graph.add_nodes_from(link_set)
    for links_at_node in incident_links.values():
        local = sorted(int(link_id) for link_id in links_at_node if int(link_id) in link_set)
        if len(local) < 2:
            continue
        graph.add_edges_from((local[0], other) for other in local[1:])

    return [sorted(int(link_id) for link_id in component) for component in nx.connected_components(graph)]


def _expand_candidate_link_ids(
    candidate_link_ids: Sequence[int],
    *,
    edge_nodes: Mapping[int, tuple[int, int]],
    incident_links: Mapping[int, Sequence[int]],
) -> list[int]:
    expanded = {int(link_id) for link_id in candidate_link_ids}
    touched_nodes: set[int] = set()
    for link_id in expanded:
        touched_nodes.update(int(node_id) for node_id in edge_nodes[int(link_id)])

    for node_id in touched_nodes:
        expanded.update(int(link_id) for link_id in incident_links.get(int(node_id), ()))
    return sorted(expanded)


def _local_direction_evidence_penalty(
    *,
    link_id: int,
    candidate: tuple[int, int],
    edge_nodes: Mapping[int, tuple[int, int]],
    geometry_order_by_link: Mapping[int, tuple[int, int]],
    orientation_score_forward: Mapping[int, float],
    orientation_score_reverse: Mapping[int, float],
    parent_order_lookup: Mapping[int, int],
    source_distance: Mapping[int, float],
    sink_distance: Mapping[int, float],
    child_inlet_nodes: set[int],
    child_outlet_nodes: set[int],
) -> float:
    node_a, node_b = edge_nodes[int(link_id)]
    penalty = 0.0

    if node_a in child_inlet_nodes and candidate != (node_a, node_b):
        penalty += 1.0e12
    if node_b in child_inlet_nodes and candidate != (node_b, node_a):
        penalty += 1.0e12
    if node_a in child_outlet_nodes and candidate != (node_b, node_a):
        penalty += 1.0e12
    if node_b in child_outlet_nodes and candidate != (node_a, node_b):
        penalty += 1.0e12

    geometry_order = geometry_order_by_link.get(int(link_id))
    forward_score = orientation_score_forward.get(int(link_id))
    reverse_score = orientation_score_reverse.get(int(link_id))
    if (
        geometry_order is not None
        and forward_score is not None
        and reverse_score is not None
        and math.isfinite(float(forward_score))
        and math.isfinite(float(reverse_score))
        and not math.isclose(float(forward_score), float(reverse_score), rel_tol=1e-12, abs_tol=1e-9)
    ):
        preferred = geometry_order if float(forward_score) < float(reverse_score) else (geometry_order[1], geometry_order[0])
        if candidate != preferred:
            penalty += abs(float(forward_score) - float(reverse_score))

    if node_a in parent_order_lookup and node_b in parent_order_lookup:
        order_a = int(parent_order_lookup[node_a])
        order_b = int(parent_order_lookup[node_b])
        if order_a != order_b:
            preferred = (node_a, node_b) if order_a < order_b else (node_b, node_a)
            if candidate != preferred:
                penalty += 1.0e6 + 1.0e3 * abs(order_a - order_b)

    progression = _resolve_link_by_progression(
        node_a=node_a,
        node_b=node_b,
        source_distance=source_distance,
        sink_distance=sink_distance,
        geometry_order=None,
        allow_geometry_fallback=False,
    )
    if progression is not None and candidate != progression[0]:
        penalty += 1.0e4

    if geometry_order is not None and candidate != geometry_order:
        penalty += 1.0

    return penalty


def _search_local_direction_repair(
    *,
    candidate_link_ids: Sequence[int],
    edge_nodes: Mapping[int, tuple[int, int]],
    incident_links: Mapping[int, Sequence[int]],
    current_orientation_by_link: Mapping[int, tuple[int, int]],
    child_links: gpd.GeoDataFrame,
    child_nodes: gpd.GeoDataFrame,
    geometry_order_by_link: Mapping[int, tuple[int, int]],
    orientation_score_forward: Mapping[int, float],
    orientation_score_reverse: Mapping[int, float],
    parent_order_lookup: Mapping[int, int],
    child_inlet_nodes: set[int],
    child_outlet_nodes: set[int],
    require_acyclic: bool = True,
    require_flag_match: bool = False,
    max_candidate_links: int = MAX_LOCAL_DIRECTION_REPAIR_LINKS,
) -> dict[int, tuple[int, int]] | None:
    candidate_links = sorted(int(link_id) for link_id in candidate_link_ids)
    if not candidate_links:
        return None
    if len(candidate_links) > max_candidate_links:
        return None

    fixed_orientation = {
        int(link_id): (int(upstream), int(downstream))
        for link_id, (upstream, downstream) in current_orientation_by_link.items()
        if int(link_id) not in set(candidate_links)
    }
    source_distance, sink_distance = _build_progression_distance_maps(
        child_links=child_links,
        child_nodes=child_nodes,
        orientation_by_link=fixed_orientation,
        child_inlet_nodes=child_inlet_nodes,
        child_outlet_nodes=child_outlet_nodes,
    )
    link_lengths = _link_length_lookup(child_links)

    best_assignment: dict[int, tuple[int, int]] | None = None
    best_score: tuple[float, int, float, tuple[int, ...]] | None = None
    for bit_pattern in itertools.product((0, 1), repeat=len(candidate_links)):
        proposal = fixed_orientation.copy()
        changed_links: list[int] = []
        total_penalty = 0.0
        total_flipped_length = 0.0
        signature: list[int] = []
        for link_id, bit in zip(candidate_links, bit_pattern):
            node_a, node_b = edge_nodes[int(link_id)]
            candidate = (node_a, node_b) if bit == 0 else (node_b, node_a)
            proposal[int(link_id)] = candidate
            total_penalty += _local_direction_evidence_penalty(
                link_id=int(link_id),
                candidate=candidate,
                edge_nodes=edge_nodes,
                geometry_order_by_link=geometry_order_by_link,
                orientation_score_forward=orientation_score_forward,
                orientation_score_reverse=orientation_score_reverse,
                parent_order_lookup=parent_order_lookup,
                source_distance=source_distance,
                sink_distance=sink_distance,
                child_inlet_nodes=child_inlet_nodes,
                child_outlet_nodes=child_outlet_nodes,
            )
            current = current_orientation_by_link.get(int(link_id))
            if current is not None and candidate != current:
                changed_links.append(int(link_id))
                total_flipped_length += float(link_lengths.get(int(link_id), 0.0))
            signature.extend(candidate)

        candidate_graph = _build_oriented_child_graph(
            child_nodes=child_nodes,
            orientation_by_link=proposal,
        )
        report = validate_single_inlet_single_outlet(
            candidate_graph, require_flag_match=require_flag_match
        )
        if not report.is_valid:
            continue
        if require_acyclic and not nx.is_directed_acyclic_graph(nx.DiGraph(candidate_graph)):
            continue

        score = (
            float(total_penalty),
            int(len(changed_links)),
            float(total_flipped_length),
            tuple(int(value) for value in signature),
        )
        if best_score is None or score < best_score:
            best_assignment = {int(link_id): proposal[int(link_id)] for link_id in candidate_links}
            best_score = score

    return best_assignment


def _repair_unresolved_link_components(
    *,
    unresolved_links: Sequence[int],
    edge_nodes: Mapping[int, tuple[int, int]],
    incident_links: Mapping[int, Sequence[int]],
    orientation_by_link: Mapping[int, tuple[int, int]],
    child_links: gpd.GeoDataFrame,
    child_nodes: gpd.GeoDataFrame,
    geometry_order_by_link: Mapping[int, tuple[int, int]],
    orientation_score_forward: Mapping[int, float],
    orientation_score_reverse: Mapping[int, float],
    parent_order_lookup: Mapping[int, int],
    child_inlet_nodes: set[int],
    child_outlet_nodes: set[int],
    require_flag_match: bool = False,
) -> dict[int, tuple[int, int]]:
    repaired: dict[int, tuple[int, int]] = {}
    for component in _link_components(unresolved_links, incident_links):
        candidate_links = list(component)
        assignment = _search_local_direction_repair(
            candidate_link_ids=candidate_links,
            edge_nodes=edge_nodes,
            incident_links=incident_links,
            current_orientation_by_link={**orientation_by_link, **repaired},
            child_links=child_links,
            child_nodes=child_nodes,
            geometry_order_by_link=geometry_order_by_link,
            orientation_score_forward=orientation_score_forward,
            orientation_score_reverse=orientation_score_reverse,
            parent_order_lookup=parent_order_lookup,
            child_inlet_nodes=child_inlet_nodes,
            child_outlet_nodes=child_outlet_nodes,
            require_acyclic=True,
            require_flag_match=require_flag_match,
        )
        if assignment is None:
            expanded = _expand_candidate_link_ids(
                candidate_links,
                edge_nodes=edge_nodes,
                incident_links=incident_links,
            )
            assignment = _search_local_direction_repair(
                candidate_link_ids=expanded,
                edge_nodes=edge_nodes,
                incident_links=incident_links,
                current_orientation_by_link={**orientation_by_link, **repaired},
                child_links=child_links,
                child_nodes=child_nodes,
                geometry_order_by_link=geometry_order_by_link,
                orientation_score_forward=orientation_score_forward,
                orientation_score_reverse=orientation_score_reverse,
                parent_order_lookup=parent_order_lookup,
                child_inlet_nodes=child_inlet_nodes,
                child_outlet_nodes=child_outlet_nodes,
                require_acyclic=True,
                require_flag_match=require_flag_match,
            )
        if assignment is None:
            continue
        for link_id in component:
            if int(link_id) in assignment:
                repaired[int(link_id)] = assignment[int(link_id)]
    return repaired


def _repair_invalid_directionality(
    *,
    orientation_by_link: Mapping[int, tuple[int, int]],
    edge_nodes: Mapping[int, tuple[int, int]],
    incident_links: Mapping[int, Sequence[int]],
    child_links: gpd.GeoDataFrame,
    child_nodes: gpd.GeoDataFrame,
    geometry_order_by_link: Mapping[int, tuple[int, int]],
    orientation_score_forward: Mapping[int, float],
    orientation_score_reverse: Mapping[int, float],
    parent_order_lookup: Mapping[int, int],
    child_inlet_nodes: set[int],
    child_outlet_nodes: set[int],
    require_flag_match: bool = False,
) -> dict[int, tuple[int, int]]:
    repaired = {
        int(link_id): (int(upstream), int(downstream))
        for link_id, (upstream, downstream) in orientation_by_link.items()
    }

    for _ in range(3):
        graph = _build_oriented_child_graph(
            child_nodes=child_nodes,
            orientation_by_link=repaired,
        )
        report = validate_single_inlet_single_outlet(
            graph, require_flag_match=require_flag_match
        )
        require_acyclic = nx.is_directed_acyclic_graph(nx.DiGraph(graph))
        if report.is_valid and (not require_acyclic or nx.is_directed_acyclic_graph(nx.DiGraph(graph))):
            return repaired

        bad_nodes = sorted(set(int(node_id) for node_id in (report.source_nodes + report.sink_nodes)))
        if not bad_nodes:
            break
        seed_links = sorted(
            {
                int(link_id)
                for node_id in bad_nodes
                for link_id in incident_links.get(int(node_id), ())
            }
        )
        if not seed_links:
            break

        components = _link_components(seed_links, incident_links)
        changed = False
        for component in components:
            assignment = _search_local_direction_repair(
                candidate_link_ids=component,
                edge_nodes=edge_nodes,
                incident_links=incident_links,
                current_orientation_by_link=repaired,
                child_links=child_links,
                child_nodes=child_nodes,
                geometry_order_by_link=geometry_order_by_link,
                orientation_score_forward=orientation_score_forward,
                orientation_score_reverse=orientation_score_reverse,
                parent_order_lookup=parent_order_lookup,
                child_inlet_nodes=child_inlet_nodes,
                child_outlet_nodes=child_outlet_nodes,
                require_acyclic=require_acyclic,
                require_flag_match=require_flag_match,
            )
            if assignment is None:
                expanded = _expand_candidate_link_ids(
                    component,
                    edge_nodes=edge_nodes,
                    incident_links=incident_links,
                )
                assignment = _search_local_direction_repair(
                    candidate_link_ids=expanded,
                    edge_nodes=edge_nodes,
                    incident_links=incident_links,
                    current_orientation_by_link=repaired,
                    child_links=child_links,
                    child_nodes=child_nodes,
                    geometry_order_by_link=geometry_order_by_link,
                    orientation_score_forward=orientation_score_forward,
                    orientation_score_reverse=orientation_score_reverse,
                    parent_order_lookup=parent_order_lookup,
                    child_inlet_nodes=child_inlet_nodes,
                    child_outlet_nodes=child_outlet_nodes,
                    require_acyclic=require_acyclic,
                    require_flag_match=require_flag_match,
                )
            if assignment is None:
                continue
            for link_id, oriented_pair in assignment.items():
                if repaired.get(int(link_id)) != oriented_pair:
                    repaired[int(link_id)] = oriented_pair
                    changed = True

        if not changed:
            break

    return repaired


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
    require_flag_match: bool = False,
    allow_width_continuity_tiebreak: bool = False,
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
        report = validate_single_inlet_single_outlet(
            child_graph, require_flag_match=require_flag_match
        )
        if report.is_valid:
            valid_candidates.append((candidate, "singleton_validation_rule"))

    if len(valid_candidates) == 1:
        return valid_candidates[0]

    if allow_width_continuity_tiebreak:
        # This is deliberately a last-resort rule. It is only called after
        # parent matching and exact component repair leave one link unresolved
        # and both directions pass the graph-validity gate. Use one width
        # semantic shared by the unresolved link and every incident oriented
        # link; never mix routing, wet, and legacy widths in one balance.
        incident_link_ids = {
            int(candidate_link_id)
            for candidate_link_id, endpoints in edge_nodes.items()
            if int(candidate_link_id) == int(link_id)
            or int(node_a) in map(int, endpoints)
            or int(node_b) in map(int, endpoints)
        }
        width_by_link: dict[int, float] | None = None
        width_field: str | None = None
        indexed_links = child_links.set_index(
            child_links["id_link"].astype(int), drop=False
        )
        for candidate_field in (
            "wid_adj_routing",
            "wid_adj_wet",
            "wid_adj",
            "wid",
        ):
            if candidate_field not in indexed_links.columns:
                continue
            candidate_widths: dict[int, float] = {}
            for candidate_link_id in incident_link_ids:
                if candidate_link_id not in indexed_links.index:
                    break
                value = pd.to_numeric(
                    pd.Series([
                        indexed_links.loc[candidate_link_id, candidate_field]
                    ]),
                    errors="coerce",
                ).iloc[0]
                if not np.isfinite(value) or float(value) <= 0.0:
                    break
                candidate_widths[candidate_link_id] = float(value)
            if len(candidate_widths) == len(incident_link_ids):
                width_by_link = candidate_widths
                width_field = candidate_field
                break

        if width_by_link is not None and width_field is not None:
            width_candidate_pool = (
                valid_candidates
                if valid_candidates
                else [
                    ((int(node_a), int(node_b)), ""),
                    ((int(node_b), int(node_a)), ""),
                ]
            )
            scored: list[tuple[float, tuple[int, int]]] = []
            for candidate, _ in width_candidate_pool:
                trial_orientation = {
                    **{int(key): tuple(map(int, value)) for key, value in orientation_by_link.items()},
                    int(link_id): tuple(map(int, candidate)),
                }
                node_scores: list[float] = []
                for node_id in (int(node_a), int(node_b)):
                    incoming_width = sum(
                        width_by_link[candidate_link_id]
                        for candidate_link_id, (_, downstream) in trial_orientation.items()
                        if candidate_link_id in width_by_link
                        and int(downstream) == node_id
                    )
                    outgoing_width = sum(
                        width_by_link[candidate_link_id]
                        for candidate_link_id, (upstream, _) in trial_orientation.items()
                        if candidate_link_id in width_by_link
                        and int(upstream) == node_id
                    )
                    if incoming_width > 0.0 and outgoing_width > 0.0:
                        node_scores.append(
                            abs(math.log(incoming_width / outgoing_width))
                        )
                if len(node_scores) == 2:
                    scored.append((float(sum(node_scores)), candidate))
            if len(scored) == 2:
                scored.sort(key=lambda item: item[0])
                best_score, best_candidate = scored[0]
                second_score = scored[1][0]
                if (
                    best_score + 0.05 < second_score
                    and best_score <= second_score * 0.9
                ):
                    return (
                        best_candidate,
                        (
                            f"singleton_width_continuity_{width_field}"
                            if valid_candidates
                            else f"singleton_width_continuity_provisional_{width_field}"
                        ),
                    )

    if not valid_candidates:
        return None

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


def _repair_global_progression_from_parent_reachability(
    *,
    orientation_by_link: Mapping[int, tuple[int, int]],
    orientation_source: Mapping[int, str],
    edge_nodes: Mapping[int, tuple[int, int]],
    incident_links: Mapping[int, Sequence[int]],
    child_nodes: gpd.GeoDataFrame,
    parent_links: gpd.GeoDataFrame,
    node_match: pd.DataFrame,
    child_inlet_nodes: set[int],
    child_outlet_nodes: set[int],
    require_flag_match: bool,
) -> tuple[dict[int, tuple[int, int]], dict[int, str], list[dict[str, Any]]]:
    """Repair weak global-progression directions from regional reachability.

    Parent-link overlap is intentionally not used here: changed links can have
    tiny, misleading overlaps with several parent links.  Instead, stable
    one-to-one node matches define a local frontier around each weakly oriented
    link.  A reversal is accepted only when it strictly reduces disagreement
    with parent reachability without introducing a new disagreement, while the
    complete child graph remains a valid DAG with the prescribed boundaries.
    """

    stable = node_match.copy()
    if "match_within_tolerance" in stable.columns:
        stable = stable.loc[stable["match_within_tolerance"].fillna(False)]
    stable = stable.loc[stable["matched_parent_node_id"].notna()].copy()
    if stable.empty:
        return dict(orientation_by_link), dict(orientation_source), []
    stable["child_id_node"] = pd.to_numeric(
        stable["child_id_node"], errors="coerce"
    ).astype("Int64")
    stable["matched_parent_node_id"] = pd.to_numeric(
        stable["matched_parent_node_id"], errors="coerce"
    ).astype("Int64")
    stable = stable.dropna(subset=["child_id_node", "matched_parent_node_id"])
    stable = stable.loc[
        ~stable["child_id_node"].duplicated(keep=False)
        & ~stable["matched_parent_node_id"].duplicated(keep=False)
    ]
    child_to_parent = {
        int(row.child_id_node): int(row.matched_parent_node_id)
        for row in stable.itertuples(index=False)
    }
    if len(child_to_parent) < 2:
        return dict(orientation_by_link), dict(orientation_source), []

    parent_graph = nx.DiGraph()
    for row in parent_links.itertuples(index=False):
        if pd.isna(row.id_us_node) or pd.isna(row.id_ds_node):
            continue
        parent_graph.add_edge(int(row.id_us_node), int(row.id_ds_node))

    repaired = {
        int(link_id): tuple(map(int, endpoints))
        for link_id, endpoints in orientation_by_link.items()
    }
    repaired_source = {
        int(link_id): str(source) for link_id, source in orientation_source.items()
    }
    audit: list[dict[str, Any]] = []

    def frontier_for(link_id: int) -> list[int]:
        queue = list(edge_nodes[int(link_id)])
        visited: set[int] = set()
        frontier: set[int] = set()
        while queue:
            node_id = int(queue.pop())
            if node_id in visited:
                continue
            visited.add(node_id)
            if node_id in child_to_parent:
                frontier.add(node_id)
                continue
            for adjacent_link_id in incident_links.get(node_id, ()):
                node_a, node_b = edge_nodes[int(adjacent_link_id)]
                queue.append(int(node_b if int(node_a) == node_id else node_a))
        return sorted(frontier)

    def mismatch_counts(
        graph: nx.DiGraph, frontier: Sequence[int]
    ) -> tuple[int, int]:
        lost = 0
        extra = 0
        for child_a in frontier:
            for child_b in frontier:
                if int(child_a) == int(child_b):
                    continue
                parent_a = child_to_parent[int(child_a)]
                parent_b = child_to_parent[int(child_b)]
                expected = (
                    parent_a in parent_graph
                    and parent_b in parent_graph
                    and nx.has_path(parent_graph, parent_a, parent_b)
                )
                observed = nx.has_path(graph, int(child_a), int(child_b))
                lost += int(expected and not observed)
                extra += int(observed and not expected)
        return int(lost), int(extra)

    candidates = sorted(
        int(link_id)
        for link_id, source in repaired_source.items()
        if source == "global_progression_rule"
    )
    for link_id in candidates:
        frontier = frontier_for(link_id)
        if len(frontier) < 2:
            continue
        current_graph = nx.DiGraph(
            _build_oriented_child_graph(
                child_nodes=child_nodes,
                orientation_by_link=repaired,
            )
        )
        current_lost, current_extra = mismatch_counts(current_graph, frontier)

        upstream, downstream = repaired[link_id]
        trial = dict(repaired)
        trial[link_id] = (int(downstream), int(upstream))
        trial_multigraph = _build_oriented_child_graph(
            child_nodes=child_nodes,
            orientation_by_link=trial,
        )
        trial_graph = nx.DiGraph(trial_multigraph)
        trial_report = _validate_variant_direction_boundaries(
            trial_multigraph,
            child_inlet_nodes=child_inlet_nodes,
            child_outlet_nodes=child_outlet_nodes,
            require_flag_match=require_flag_match,
        )
        if not trial_report.is_valid or not nx.is_directed_acyclic_graph(trial_graph):
            continue
        trial_lost, trial_extra = mismatch_counts(trial_graph, frontier)
        dominates = (
            trial_lost <= current_lost
            and trial_extra <= current_extra
            and (trial_lost < current_lost or trial_extra < current_extra)
        )
        if not dominates:
            continue
        repaired = trial
        repaired_source[link_id] = "regional_parent_reachability_rule"
        audit.append(
            {
                "id_link": int(link_id),
                "frontier_child_nodes": [int(value) for value in frontier],
                "frontier_parent_nodes": [
                    int(child_to_parent[value]) for value in frontier
                ],
                "current_lost_parent_paths": int(current_lost),
                "current_extra_child_paths": int(current_extra),
                "repaired_lost_parent_paths": int(trial_lost),
                "repaired_extra_child_paths": int(trial_extra),
            }
        )

    return repaired, repaired_source, audit


def _assign_variant_directions(
    *,
    child_links: gpd.GeoDataFrame,
    child_nodes: gpd.GeoDataFrame,
    parent_links: gpd.GeoDataFrame,
    parent_nodes: gpd.GeoDataFrame,
    node_match: pd.DataFrame,
    link_match: pd.DataFrame,
    canonicalize_parent_boundaries: bool = True,
    require_child_boundary_flag_match: bool = False,
    allow_boundary_potential_fallback: bool = False,
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
            # A touch-only candidate is adjacency evidence, not lineage.  It
            # must never override the direction of a genuinely changed local
            # connector merely because both geometries meet near one parent
            # node.  Only a positive core-overlap match can transfer parent
            # endpoint direction; otherwise local topology resolves the link.
            if not core_matches.empty:
                score_start_end = 0.0
                score_end_start = 0.0
                for match_row in core_matches.itertuples(index=False):
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
                if not math.isclose(
                    score_start_end,
                    score_end_start,
                    rel_tol=1e-12,
                    abs_tol=1e-9,
                ):
                    oriented_pair = (
                        (start_node, end_node)
                        if score_start_end < score_end_start
                        else (end_node, start_node)
                    )
                    direction_method = "matched_parent_core_endpoints"

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

    used_boundary_potential_fallback = False
    if unresolved_links and allow_boundary_potential_fallback:
        orientation_by_link = _boundary_harmonic_orientation(
            edge_nodes=edge_nodes,
            child_links=child_links,
            child_nodes=child_nodes,
            child_inlet_nodes=child_inlet_nodes,
            child_outlet_nodes=child_outlet_nodes,
        )
        orientation_source = {
            int(link_id): "boundary_harmonic_fallback" for link_id in orientation_by_link
        }
        unresolved_links = []
        used_boundary_potential_fallback = True

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
                require_flag_match=require_child_boundary_flag_match,
            )
            if resolved is not None:
                oriented_pair, method = resolved
                orientation_by_link[link_id] = oriented_pair
                orientation_source[link_id] = method
                unresolved_links = []

    if unresolved_links:
        repaired_orientations = _repair_unresolved_link_components(
            unresolved_links=unresolved_links,
            edge_nodes=edge_nodes,
            incident_links=incident_links,
            orientation_by_link=orientation_by_link,
            child_links=child_links,
            child_nodes=child_nodes,
            geometry_order_by_link=geometry_order_by_link,
            orientation_score_forward=orientation_score_forward,
            orientation_score_reverse=orientation_score_reverse,
            parent_order_lookup=parent_order_lookup,
            child_inlet_nodes=child_inlet_nodes,
            child_outlet_nodes=child_outlet_nodes,
            require_flag_match=require_child_boundary_flag_match,
        )
        for link_id, oriented_pair in repaired_orientations.items():
            orientation_by_link[int(link_id)] = oriented_pair
            orientation_source[int(link_id)] = "local_exact_repair"
        unresolved_links = [
            int(link_id) for link_id in unresolved_links if int(link_id) not in repaired_orientations
        ]

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
            require_flag_match=require_child_boundary_flag_match,
            allow_width_continuity_tiebreak=True,
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

    child_graph = _build_oriented_child_graph(
        child_nodes=child_nodes,
        orientation_by_link=orientation_by_link,
    )

    report = _validate_variant_direction_boundaries(
        child_graph,
        child_inlet_nodes=child_inlet_nodes,
        child_outlet_nodes=child_outlet_nodes,
        require_flag_match=require_child_boundary_flag_match,
    )
    if not report.is_valid or not nx.is_directed_acyclic_graph(nx.DiGraph(child_graph)):
        repaired_orientation = _repair_invalid_directionality(
            orientation_by_link=orientation_by_link,
            edge_nodes=edge_nodes,
            incident_links=incident_links,
            child_links=child_links,
            child_nodes=child_nodes,
            geometry_order_by_link=geometry_order_by_link,
            orientation_score_forward=orientation_score_forward,
            orientation_score_reverse=orientation_score_reverse,
            parent_order_lookup=parent_order_lookup,
            child_inlet_nodes=child_inlet_nodes,
            child_outlet_nodes=child_outlet_nodes,
            require_flag_match=require_child_boundary_flag_match,
        )
        for link_id, oriented_pair in repaired_orientation.items():
            if orientation_by_link.get(int(link_id)) != oriented_pair:
                orientation_source[int(link_id)] = "local_exact_repair"
        orientation_by_link = repaired_orientation
        child_graph = _build_oriented_child_graph(
            child_nodes=child_nodes,
            orientation_by_link=orientation_by_link,
        )
        report = _validate_variant_direction_boundaries(
            child_graph,
            child_inlet_nodes=child_inlet_nodes,
            child_outlet_nodes=child_outlet_nodes,
            require_flag_match=require_child_boundary_flag_match,
        )

    if (
        allow_boundary_potential_fallback
        and (not report.is_valid or not nx.is_directed_acyclic_graph(nx.DiGraph(child_graph)))
    ):
        orientation_by_link = _boundary_harmonic_orientation(
            edge_nodes=edge_nodes,
            child_links=child_links,
            child_nodes=child_nodes,
            child_inlet_nodes=child_inlet_nodes,
            child_outlet_nodes=child_outlet_nodes,
        )
        orientation_source = {
            int(link_id): "boundary_harmonic_fallback" for link_id in orientation_by_link
        }
        used_boundary_potential_fallback = True
        child_graph = _build_oriented_child_graph(
            child_nodes=child_nodes,
            orientation_by_link=orientation_by_link,
        )
        report = _validate_variant_direction_boundaries(
            child_graph,
            child_inlet_nodes=child_inlet_nodes,
            child_outlet_nodes=child_outlet_nodes,
            require_flag_match=require_child_boundary_flag_match,
        )

    regional_reachability_repairs: list[dict[str, Any]] = []
    if report.is_valid and nx.is_directed_acyclic_graph(nx.DiGraph(child_graph)):
        (
            orientation_by_link,
            orientation_source,
            regional_reachability_repairs,
        ) = _repair_global_progression_from_parent_reachability(
            orientation_by_link=orientation_by_link,
            orientation_source=orientation_source,
            edge_nodes=edge_nodes,
            incident_links=incident_links,
            child_nodes=child_nodes,
            parent_links=parent_links,
            node_match=node_match,
            child_inlet_nodes=child_inlet_nodes,
            child_outlet_nodes=child_outlet_nodes,
            require_flag_match=require_child_boundary_flag_match,
        )
        if regional_reachability_repairs:
            child_graph = _build_oriented_child_graph(
                child_nodes=child_nodes,
                orientation_by_link=orientation_by_link,
            )
            report = _validate_variant_direction_boundaries(
                child_graph,
                child_inlet_nodes=child_inlet_nodes,
                child_outlet_nodes=child_outlet_nodes,
                require_flag_match=require_child_boundary_flag_match,
            )

    topology_source_nodes = sorted(int(node_id) for node_id in report.source_nodes)
    topology_sink_nodes = sorted(int(node_id) for node_id in report.sink_nodes)
    if canonicalize_parent_boundaries:
        canonical_source_nodes, canonical_sink_nodes, canonical_issues = _resolve_canonical_child_boundary_nodes(
            parent_nodes=parent_nodes,
            node_match=node_match,
            topology_source_nodes=topology_source_nodes,
            topology_sink_nodes=topology_sink_nodes,
        )
    else:
        canonical_source_nodes, canonical_sink_nodes, canonical_issues = set(), set(), []
    source_nodes = (
        sorted(int(node_id) for node_id in canonical_source_nodes)
        if canonical_source_nodes
        else topology_source_nodes
    )
    sink_nodes = (
        sorted(int(node_id) for node_id in canonical_sink_nodes)
        if canonical_sink_nodes
        else topology_sink_nodes
    )
    source_node = source_nodes[0] if len(source_nodes) == 1 else None
    sink_node = sink_nodes[0] if len(sink_nodes) == 1 else None
    source_node_set = set(source_nodes)
    sink_node_set = set(sink_nodes)
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
    directed_links["is_inlet"] = directed_links["id_us_node"].astype(int).isin(source_node_set)
    directed_links["is_outlet"] = directed_links["id_ds_node"].astype(int).isin(sink_node_set)
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
    directed_nodes["is_inlet"] = directed_nodes["id_node"].astype(int).isin(source_node_set)
    directed_nodes["is_outlet"] = directed_nodes["id_node"].astype(int).isin(sink_node_set)
    directed_nodes["type_io"] = [
        _type_io_label(is_inlet=bool(is_inlet), is_outlet=bool(is_outlet))
        for is_inlet, is_outlet in zip(directed_nodes["is_inlet"], directed_nodes["is_outlet"])
    ]

    summary = {
        "is_valid": bool(report.is_valid),
        "source_node": source_node,
        "sink_node": sink_node,
        "source_nodes": source_nodes,
        "sink_nodes": sink_nodes,
        "n_oriented_links": int(len(directed_links)),
        "n_node_matches": int(node_match["match_within_tolerance"].fillna(False).sum()),
        "n_link_matches": int(link_match["child_id_link"].nunique()),
        "issues": list(report.issues) + canonical_issues,
        "validation_report": report.to_dict(),
        "topology_source_nodes": topology_source_nodes,
        "topology_sink_nodes": topology_sink_nodes,
        "canonical_source_nodes": sorted(int(node_id) for node_id in canonical_source_nodes),
        "canonical_sink_nodes": sorted(int(node_id) for node_id in canonical_sink_nodes),
        "canonicalize_parent_boundaries": bool(canonicalize_parent_boundaries),
        "require_child_boundary_flag_match": bool(require_child_boundary_flag_match),
        "allow_boundary_potential_fallback": bool(allow_boundary_potential_fallback),
        "used_boundary_potential_fallback": bool(used_boundary_potential_fallback),
        "n_regional_parent_reachability_repairs": int(
            len(regional_reachability_repairs)
        ),
        "regional_parent_reachability_repairs": regional_reachability_repairs,
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
    import rivgraph.rivers.river_utils as ru

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
    network.links, network.nodes = ru.prune_river(
        network.links,
        network.nodes,
        network.exit_sides,
        network.Iskel,
    )
    _sanitize_rivgraph_topology_metadata(network.links, network.nodes)
    network.finalize_ids()
    network.compute_distance_transform()
    network.compute_link_width_and_length()
    with rivgraph_centerline_compat():
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
    selection_type: str | None = None,
    compound_bubble_id: int | None = None,
    bubble_root_unit_id: int | None = None,
    variant_id: str | None = None,
    example_id: str | None = None,
    output_dir: str | Path | None = None,
    collapse_corridor_mode: str = "realized",
    wet_reference_mask_path: str | Path | None = None,
    boundary_component_labels_path: str | Path | None = None,
    boundary_component_output_dir: str | Path | None = None,
    bank_candidate_neighbors_max: int = 2048,
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
    sword_example_station_source_path: str | Path | None = None,
    sword_station_match_source_path: str | Path | None = None,
    sword_reach_buffer_steps: int = 2,
    max_path_cutoff: int = 100,
    max_paths: int = 5000,
    verbose_rivgraph: bool = False,
    exclude_closed_two_link_replacement: bool = False,
) -> NetworkVariantOutputs:
    if collapse_corridor_mode not in COLLAPSE_CORRIDOR_MODES:
        raise ValueError(
            f"Unsupported collapse_corridor_mode {collapse_corridor_mode!r}. "
            f"Expected one of {COLLAPSE_CORRIDOR_MODES}."
        )
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
    boundary_output_dir = Path(boundary_component_output_dir) if boundary_component_output_dir is not None else (output_path / "boundary_components")
    boundary_outputs = _ensure_boundary_component_mask(
        cleaned_mask_path=cleaned_mask_path,
        reviewed_links_path=reviewed_links_path,
        output_dir=boundary_output_dir,
        boundary_component_labels_path=boundary_component_labels_path,
    )
    resolved_boundary_component_labels_path = Path(boundary_outputs["labels"])

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

    reviewed_links = gpd.read_file(reviewed_links_path)
    reviewed_nodes = gpd.read_file(reviewed_nodes_path)
    with rasterio.open(cleaned_mask_path) as src:
        parent_profile = src.profile.copy()
        pixel_length = _pixel_length_from_transform(src.transform)

    if match_tolerance is None:
        match_tolerance = pixel_length * 1.25

    collapsed_mask, origin_mask, mask_summary, collapse_components, edit_geometries = _build_selection_masks(
        cleaned_mask_path=cleaned_mask_path,
        reviewed_links=reviewed_links,
        resolved_unit_ids=resolved_unit_ids,
        units_by_id=units_by_id,
        preferred_width_field=preferred_width_field,
        footprint_buffer_scale=footprint_buffer_scale,
        all_touched=all_touched,
        allow_noop=allow_noop,
    )
    selection_metadata = _collapsed_selection_metadata(resolved_unit_ids, group_label)
    collapse_components = _apply_collapsed_selection_metadata(
        collapse_components,
        unit_ids=resolved_unit_ids,
        group_label=group_label,
    )
    collapse_components["example_id"] = example_id
    collapse_components["variant_id"] = variant_id
    if group_label is not None:
        collapse_components["group_label"] = str(group_label)
    edit_geometries = _apply_collapsed_selection_metadata(
        edit_geometries,
        unit_ids=resolved_unit_ids,
        group_label=group_label,
    )
    edit_geometries["example_id"] = example_id
    edit_geometries["variant_id"] = variant_id

    collapsed_mask_path = _write_mask(mask_dir / f"{example_id}__{variant_id}_collapsed.tif", collapsed_mask, parent_profile)
    origin_mask_path = _write_mask(mask_dir / "origin_mask.tif", origin_mask, parent_profile)
    bank_labels_path, removed_bank_labels = _write_state_bank_labels(
        boundary_component_labels_path=resolved_boundary_component_labels_path,
        origin_mask=origin_mask,
        output_path=mask_dir / "bank_labels.tif",
        profile=parent_profile,
    )
    edit_geometries.to_file(mask_dir / "collapse_edit_geometries.gpkg", driver="GPKG")
    collapse_components.to_csv(summary_dir / "collapse_components.csv", index=False)

    metadata = {
        "example_id": example_id,
        "variant_id": variant_id,
        "group_label": group_label,
        "selected_unit_ids": _serialize_int_list(resolved_unit_ids),
        "collapse_corridor_mode": collapse_corridor_mode,
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
    raw_links = _apply_collapsed_selection_metadata(
        raw_links,
        unit_ids=resolved_unit_ids,
        group_label=group_label,
    )
    raw_nodes = _apply_collapsed_selection_metadata(
        raw_nodes,
        unit_ids=resolved_unit_ids,
        group_label=group_label,
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
    link_lineage = _resolve_link_lineage(
        parent_graph=parent_graph,
        directed_child_links=directed_links,
        node_match=node_match,
        link_match=link_match,
        match_tolerance=float(match_tolerance),
    )
    anchor_segmentation_summary: dict[str, Any] | None = None
    if collapse_corridor_mode == "anchored":
        nominal_upstream_node, nominal_downstream_node = _selection_anchor_nodes(
            resolved_unit_ids=resolved_unit_ids,
            units_by_id=units_by_id,
        )
        selection_edge_pairs = _selection_edge_pairs(
            resolved_unit_ids=resolved_unit_ids,
            units_by_id=units_by_id,
        )
        directed_links, directed_nodes, node_match, anchor_segmentation_summary = _segment_collapsed_links_to_anchor_nodes(
            parent_graph=parent_graph,
            parent_nodes=reviewed_nodes,
            parent_node_order=parent_node_order,
            directed_links=directed_links,
            directed_nodes=directed_nodes,
            node_match=node_match,
            link_match=link_match,
            link_lineage=link_lineage,
            selection_edge_pairs=selection_edge_pairs,
            nominal_upstream_node=nominal_upstream_node,
            nominal_downstream_node=nominal_downstream_node,
        )
        link_match = _match_child_links_to_parent_links(
            reviewed_links,
            directed_links,
            match_tolerance=match_tolerance,
        )
        link_lineage = _resolve_link_lineage(
            parent_graph=parent_graph,
            directed_child_links=directed_links,
            node_match=node_match,
            link_match=link_match,
            match_tolerance=float(match_tolerance),
        )
        direction_summary = {
            "is_valid": bool(anchor_segmentation_summary.get("validation_report", {}).get("is_valid", direction_summary.get("is_valid", False))),
            "source_node": anchor_segmentation_summary.get("validation_report", {}).get("source_nodes", [direction_summary.get("source_node")])[0]
            if anchor_segmentation_summary.get("validation_report", {}).get("source_nodes")
            else direction_summary.get("source_node"),
            "sink_node": anchor_segmentation_summary.get("validation_report", {}).get("sink_nodes", [direction_summary.get("sink_node")])[0]
            if anchor_segmentation_summary.get("validation_report", {}).get("sink_nodes")
            else direction_summary.get("sink_node"),
            "source_nodes": anchor_segmentation_summary.get("validation_report", {}).get("source_nodes", direction_summary.get("source_nodes", [])),
            "sink_nodes": anchor_segmentation_summary.get("validation_report", {}).get("sink_nodes", direction_summary.get("sink_nodes", [])),
            "n_oriented_links": int(len(directed_links)),
            "n_node_matches": int(node_match["match_within_tolerance"].fillna(False).sum()) if "match_within_tolerance" in node_match.columns else 0,
            "n_link_matches": int(link_match["child_id_link"].nunique()) if "child_id_link" in link_match.columns else 0,
            "issues": list(anchor_segmentation_summary.get("issues", [])),
            "validation_report": anchor_segmentation_summary.get("validation_report", direction_summary.get("validation_report", {})),
        }

    directed_nodes, node_sword_match, sword_match_summary = match_variant_nodes_to_sword(
        directed_nodes=directed_nodes,
        parent_nodes=reviewed_nodes,
        node_match=node_match,
        sword_node_source_path=sword_node_source_path,
        sword_wse_field=sword_wse_field,
        sword_match_tolerance=sword_match_tolerance,
        example_id=example_id,
        sword_example_station_source_path=sword_example_station_source_path,
        sword_station_match_source_path=sword_station_match_source_path,
        sword_reach_buffer_steps=sword_reach_buffer_steps,
    )
    node_match = _apply_collapsed_selection_metadata(
        node_match,
        unit_ids=resolved_unit_ids,
        group_label=group_label,
    )
    link_match = _apply_collapsed_selection_metadata(
        link_match,
        unit_ids=resolved_unit_ids,
        group_label=group_label,
    )
    directed_links = _apply_collapsed_selection_metadata(
        directed_links,
        unit_ids=resolved_unit_ids,
        group_label=group_label,
    )
    directed_nodes = _apply_collapsed_selection_metadata(
        directed_nodes,
        unit_ids=resolved_unit_ids,
        group_label=group_label,
    )
    node_sword_match = _apply_collapsed_selection_metadata(
        node_sword_match,
        unit_ids=resolved_unit_ids,
        group_label=group_label,
    )
    link_lineage = _apply_collapsed_selection_metadata(
        link_lineage,
        unit_ids=resolved_unit_ids,
        group_label=group_label,
    )
    directed_links = directed_links.drop(
        columns=[column for column in link_lineage.columns if column in directed_links.columns and column != "id_link"],
        errors="ignore",
    ).merge(link_lineage, on="id_link", how="left", validate="1:1")

    try:
        parent_width_samples = read_width_samples(Path(reviewed_links_path).parent)
    except FileNotFoundError:
        parent_width_samples = pd.DataFrame()
    (
        directed_enriched_links,
        link_width_samples,
        changed_width_link_ids,
        width_product_inheritance,
    ) = _compute_changed_link_width_products(
        directed_links,
        parent_links=reviewed_links,
        parent_width_samples=parent_width_samples,
        link_lineage=link_lineage,
        collapsed_mask_path=collapsed_mask_path,
        wet_reference_mask_path=wet_reference_mask_path,
        origin_mask_path=origin_mask_path,
        rivgraph_distance_path=rivgraph_paths.get("distance"),
        bank_labels_path=bank_labels_path,
        bank_candidate_neighbors_max=bank_candidate_neighbors_max,
    )

    if collapse_corridor_mode == "anchored":
        (
            segmented_metadata_links,
            directed_enriched_links,
            segmented_stub_inheritance_summary,
        ) = _apply_segmented_stub_width_inheritance(
            directed_links=directed_links,
            enriched_links=directed_enriched_links,
            link_width_samples=link_width_samples,
        )
        for column in segmented_metadata_links.columns:
            if column.startswith("family_"):
                directed_enriched_links[column] = segmented_metadata_links[column].to_numpy()
    else:
        segmented_stub_inheritance_summary = {"applied_count": 0, "applied_link_ids": [], "records": []}

    directed_enriched_links = _apply_width_sample_flow_alignment(
        directed_enriched_links,
        collapse_corridor_mode=collapse_corridor_mode,
    )
    link_width_samples = _apply_collapsed_selection_metadata(
        link_width_samples,
        unit_ids=resolved_unit_ids,
        group_label=group_label,
    )
    link_width_samples["example_id"] = example_id
    link_width_samples["variant_id"] = variant_id
    excluded_routing_width_link_ids: list[int] = []
    closed_replacement_audit = pd.DataFrame()
    if exclude_closed_two_link_replacement:
        from network_variants.width_trim_selection import resolve_closed_replacement_exclusions

        excluded_routing_width_link_ids, closed_replacement_audit = resolve_closed_replacement_exclusions(
            directed_enriched_links,
            reviewed_links,
            changed_width_link_ids,
        )
        if not closed_replacement_audit.empty:
            decision_columns = [
                column for column in closed_replacement_audit.columns
                if column != "id_link" and column not in directed_enriched_links.columns
            ]
            directed_enriched_links = directed_enriched_links.merge(
                closed_replacement_audit[["id_link", *decision_columns]],
                on="id_link",
                how="left",
                validate="1:1",
            )
    directed_enriched_links, routing_width_audit, routing_width_application = apply_routing_width_model(
        directed_enriched_links,
        link_width_samples,
        directed_nodes,
        example_id=example_id,
        variant_id=variant_id,
        apply_known_base_exclusions=False,
        evaluation_link_ids=changed_width_link_ids,
        excluded_evaluation_link_ids=excluded_routing_width_link_ids,
        exclusion_reason="closed_two_link_replacement",
    )
    from network_variants.width_trim_selection import build_parent_child_width_diagnostics

    parent_child_diagnostics = build_parent_child_width_diagnostics(
        directed_enriched_links,
        reviewed_links,
    )
    if not parent_child_diagnostics.empty:
        diagnostic_columns = [
            column for column in parent_child_diagnostics.columns
            if column != "id_link" and column not in directed_enriched_links.columns
        ]
        directed_enriched_links = directed_enriched_links.merge(
            parent_child_diagnostics[["id_link", *diagnostic_columns]],
            on="id_link",
            how="left",
            validate="1:1",
        )
        if not routing_width_audit.empty:
            routing_width_audit = routing_width_audit.merge(
                parent_child_diagnostics,
                on="id_link",
                how="left",
                validate="1:1",
            )
    directed_enriched_links.loc[
        directed_enriched_links["id_link"].astype(int).isin(changed_width_link_ids),
        "width_inheritance_status",
    ] = "changed_evaluated"
    routing_width_paths = write_routing_width_audit(
        routing_width_audit,
        routing_width_application,
        output_dir=width_dir,
    )

    link_width_families = _write_width_family_outputs(
        width_dir=width_dir,
        link_width_samples=link_width_samples,
        directed_enriched_links=directed_enriched_links,
    )

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
        "selection_type": selection_type,
        "compound_bubble_id": compound_bubble_id,
        "bubble_root_unit_id": bubble_root_unit_id,
        "selected_unit_ids": resolved_unit_ids,
        "group_label": group_label,
        "collapsed_selection": selection_metadata,
        "collapse_corridor_mode": collapse_corridor_mode,
        "n_components": int(len(collapse_components)),
        "component_ids": collapse_components["component_id"].tolist(),
        "exit_sides": str(exit_sides),
        "preferred_width_field": preferred_width_field,
        "footprint_buffer_scale": float(footprint_buffer_scale),
        "transect_scale": float(transect_scale),
        "min_transect_pixels": float(min_transect_pixels),
        "match_tolerance": float(match_tolerance),
        "bank_candidate_neighbors_max": int(bank_candidate_neighbors_max),
        "all_touched": bool(all_touched),
        "allow_noop": bool(allow_noop),
        "export_sword": bool(export_sword),
        "ready_for_graph_matching": True,
        "ready_for_rapid": bool(direction_summary.get("is_valid", False)),
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
            "boundary_component_labels": str(resolved_boundary_component_labels_path.resolve()),
            "reviewed_links": str(Path(reviewed_links_path).resolve()),
            "reviewed_nodes": str(Path(reviewed_nodes_path).resolve()),
            "workflow_output_dir": str(Path(workflow_output_dir).resolve()) if workflow_output_dir is not None else None,
            "sword_node_source": str(Path(sword_node_source_path).resolve()) if sword_node_source_path is not None else None,
            "sword_example_station_source": (
                str(Path(sword_example_station_source_path).resolve())
                if sword_example_station_source_path is not None
                else sword_match_summary.get("example_station_source")
            ),
            "sword_station_match_source": (
                str(Path(sword_station_match_source_path).resolve())
                if sword_station_match_source_path is not None
                else sword_match_summary.get("station_match_source")
            ),
        },
        "mask_summary": mask_summary,
        "bank_label_summary": {
            "removed_land_labels": removed_bank_labels,
            "n_removed_land_labels": int(len(removed_bank_labels)),
        },
        "segmented_stub_inheritance": segmented_stub_inheritance_summary,
        "width_product_inheritance": width_product_inheritance,
        "routing_width_application": routing_width_application,
        "anchor_segmentation": anchor_segmentation_summary,
        "sword_matching": {
            "wse_field": sword_wse_field,
            "match_tolerance": sword_match_tolerance,
            "reach_buffer_steps": int(sword_reach_buffer_steps),
            "n_matched_nodes": int(node_sword_match["sword_node_id"].notna().sum()) if not node_sword_match.empty else 0,
            "n_propagated_matches": int(node_sword_match["sword_match_from_parent"].fillna(False).sum()) if not node_sword_match.empty else 0,
            "candidate_scope": sword_match_summary.get("scope"),
            "candidate_region": sword_match_summary.get("candidate_region"),
            "candidate_reach_count": sword_match_summary.get("candidate_reach_count"),
            "candidate_reach_ids": sword_match_summary.get("candidate_reach_ids", []),
            "upstream_station_key": sword_match_summary.get("upstream_station_key"),
            "downstream_station_key": sword_match_summary.get("downstream_station_key"),
            "upstream_reach_id": sword_match_summary.get("upstream_reach_id"),
            "downstream_reach_id": sword_match_summary.get("downstream_reach_id"),
            "filter_reason": sword_match_summary.get("reason"),
        },
        "output_paths": {
            "output_dir": str(output_path.resolve()),
            "collapsed_mask": str(collapsed_mask_path.resolve()),
            "origin_mask": str(origin_mask_path.resolve()),
            "width_scope_mask": str(origin_mask_path.resolve()),
            "bank_labels": str(bank_labels_path.resolve()),
            "boundary_component_labels": str(resolved_boundary_component_labels_path.resolve()),
            "collapse_components": str((summary_dir / "collapse_components.csv").resolve()),
            "edit_geometries": str((mask_dir / "collapse_edit_geometries.gpkg").resolve()),
            "link_width_families": str((width_dir / "link_width_families.csv").resolve()),
            "link_width_samples": str((width_dir / "link_width_samples.csv").resolve()),
            "links_with_width_families": str((width_dir / "links_with_width_families.gpkg").resolve()),
            "routing_width_application": routing_width_paths["link_audit"],
            "routing_width_application_manifest": routing_width_paths["manifest"],
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
            "rivgraph_distance": str(rivgraph_paths["distance"].resolve()) if rivgraph_paths["distance"] is not None else None,
            "sword_reaches": str(rivgraph_paths["sword_reaches"].resolve()) if rivgraph_paths["sword_reaches"] is not None else None,
            "sword_nodes": str(rivgraph_paths["sword_nodes"].resolve()) if rivgraph_paths["sword_nodes"] is not None else None,
        },
        "direction_summary": direction_summary,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "code_version": _git_revision(),
    }
    with (summary_dir / "variant_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)

    results = NetworkVariantOutputs(
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
        width_scope_mask_path=origin_mask_path,
        bank_labels_path=bank_labels_path,
        boundary_component_labels_path=resolved_boundary_component_labels_path,
        rivgraph_links_path=Path(rivgraph_paths["links"]),
        rivgraph_nodes_path=Path(rivgraph_paths["nodes"]),
        directed_links_path=directed_dir / f"{rivgraph_name}_directed_links.gpkg",
        directed_nodes_path=directed_dir / f"{rivgraph_name}_directed_nodes.gpkg",
        rivgraph_centerline_path=Path(rivgraph_paths["centerline"]) if rivgraph_paths["centerline"] is not None else None,
        sword_reaches_path=Path(rivgraph_paths["sword_reaches"]) if rivgraph_paths["sword_reaches"] is not None else None,
        sword_nodes_path=Path(rivgraph_paths["sword_nodes"]) if rivgraph_paths["sword_nodes"] is not None else None,
        manifest=manifest,
        origin_mask_path=origin_mask_path,
        rivgraph_distance_path=Path(rivgraph_paths["distance"]) if rivgraph_paths["distance"] is not None else None,
    )
    if not bool(direction_summary.get("is_valid", False)):
        raise VariantDirectionValidationError(results, direction_summary.get("issues", []))
    return results


def derive_anchored_graph_from_realized(
    *,
    parent_links: gpd.GeoDataFrame,
    parent_nodes: gpd.GeoDataFrame,
    realized_links: gpd.GeoDataFrame,
    realized_nodes: gpd.GeoDataFrame,
    realized_node_match: pd.DataFrame,
    realized_link_match: pd.DataFrame,
    realized_link_lineage: pd.DataFrame,
    selected_unit_ids: Sequence[int],
    units_by_id: Mapping[int, StructuralUnit],
    match_tolerance: float,
) -> AnchoredGraphTransformation:
    """Apply the established scientific anchoring transform to a realized graph.

    The function is intentionally independent of how the realized graph was
    generated.  It preserves the existing full-raster algorithm exactly:
    determine the selected parent interval, split/project collapsed child
    links at the parent anchor nodes, then rebuild link matching and lineage.
    Raster-derived width products are handled by the caller after this graph
    transformation.
    """

    resolved_unit_ids = [int(unit_id) for unit_id in selected_unit_ids]
    missing_unit_ids = sorted(
        unit_id for unit_id in resolved_unit_ids if unit_id not in units_by_id
    )
    if missing_unit_ids:
        raise ValueError(
            "The following selected unit_ids were not found in the reviewed graph: "
            f"{missing_unit_ids}"
        )

    parent_node_order, parent_graph, _ = _infer_parent_node_order(
        parent_links,
        parent_nodes,
    )
    nominal_upstream_node, nominal_downstream_node = _selection_anchor_nodes(
        resolved_unit_ids=resolved_unit_ids,
        units_by_id=units_by_id,
    )
    selection_edge_pairs = _selection_edge_pairs(
        resolved_unit_ids=resolved_unit_ids,
        units_by_id=units_by_id,
    )
    (
        directed_links,
        directed_nodes,
        node_match,
        anchor_segmentation_summary,
    ) = _segment_collapsed_links_to_anchor_nodes(
        parent_graph=parent_graph,
        parent_nodes=parent_nodes,
        parent_node_order=parent_node_order,
        directed_links=realized_links,
        directed_nodes=realized_nodes,
        node_match=realized_node_match,
        link_match=realized_link_match,
        link_lineage=realized_link_lineage,
        selection_edge_pairs=selection_edge_pairs,
        nominal_upstream_node=nominal_upstream_node,
        nominal_downstream_node=nominal_downstream_node,
    )
    link_match = _match_child_links_to_parent_links(
        parent_links,
        directed_links,
        match_tolerance=float(match_tolerance),
    )
    link_lineage = _resolve_link_lineage(
        parent_graph=parent_graph,
        directed_child_links=directed_links,
        node_match=node_match,
        link_match=link_match,
        match_tolerance=float(match_tolerance),
    )
    validation_report = anchor_segmentation_summary.get("validation_report", {})
    direction_summary = {
        "is_valid": bool(validation_report.get("is_valid", False)),
        "source_node": validation_report.get("source_nodes", [None])[0]
        if validation_report.get("source_nodes")
        else None,
        "sink_node": validation_report.get("sink_nodes", [None])[0]
        if validation_report.get("sink_nodes")
        else None,
        "source_nodes": validation_report.get("source_nodes", []),
        "sink_nodes": validation_report.get("sink_nodes", []),
        "n_oriented_links": int(len(directed_links)),
        "n_node_matches": int(
            node_match["match_within_tolerance"].fillna(False).sum()
        )
        if "match_within_tolerance" in node_match.columns
        else 0,
        "n_link_matches": int(link_match["child_id_link"].nunique())
        if "child_id_link" in link_match.columns
        else 0,
        "issues": list(anchor_segmentation_summary.get("issues", [])),
        "validation_report": validation_report,
    }
    return AnchoredGraphTransformation(
        directed_links=directed_links,
        directed_nodes=directed_nodes,
        node_match=node_match,
        link_match=link_match,
        link_lineage=link_lineage,
        anchor_segmentation_summary=anchor_segmentation_summary,
        direction_summary=direction_summary,
    )


def derive_anchored_variant_from_realized(
    *,
    source_output_dir: str | Path,
    reviewed_links_path: str | Path,
    reviewed_nodes_path: str | Path,
    output_dir: str | Path,
    example_id: str | None = None,
    variant_id: str | None = None,
    wet_reference_mask_path: str | Path | None = None,
    boundary_component_labels_path: str | Path | None = None,
    boundary_component_output_dir: str | Path | None = None,
    bank_candidate_neighbors_max: int = 2048,
    sword_node_source_path: str | Path | None = None,
    sword_wse_field: str | None = None,
    sword_match_tolerance: float | None = None,
    sword_example_station_source_path: str | Path | None = None,
    sword_station_match_source_path: str | Path | None = None,
    sword_reach_buffer_steps: int = 2,
    max_path_cutoff: int = 100,
    max_paths: int = 5000,
    transect_scale: float = DEFAULT_TRANSECT_SCALE,
    min_transect_pixels: float = DEFAULT_MIN_TRANSECT_PIXELS,
    match_tolerance: float | None = None,
) -> NetworkVariantOutputs:
    derive_started = time.perf_counter()
    stage_timings: dict[str, float] = {}
    source_output_path = Path(source_output_dir)
    output_path = Path(output_dir)
    source_manifest_path = source_output_path / "summary" / "variant_manifest.json"
    if not source_manifest_path.exists():
        raise FileNotFoundError(f"Realized variant manifest not found: {source_manifest_path}")

    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    panelized_execution = str(source_manifest.get("execution_mode", "")) == "panelized"
    selected_unit_ids = [int(unit_id) for unit_id in source_manifest.get("selected_unit_ids", [])]
    if not selected_unit_ids:
        raise ValueError("Realized variant manifest does not contain selected_unit_ids for anchored derivation.")

    example_id = str(example_id) if example_id is not None else str(source_manifest.get("example_id"))
    variant_id = str(variant_id) if variant_id is not None else str(source_manifest.get("variant_id"))

    copy_started = time.perf_counter()
    if output_path.exists():
        shutil.rmtree(output_path)
    shutil.copytree(source_output_path, output_path)
    stage_timings["copy_realized_variant_seconds"] = time.perf_counter() - copy_started
    setup_started = time.perf_counter()

    output_paths = {
        key: _remap_copied_output_path(
            value,
            source_output_dir=source_output_path,
            target_output_dir=output_path,
        )
        for key, value in dict(source_manifest.get("output_paths", {})).items()
    }

    collapsed_mask_path = output_paths.get("collapsed_mask")
    if collapsed_mask_path is None or not collapsed_mask_path.exists():
        source_collapsed_mask = source_manifest.get("output_paths", {}).get("collapsed_mask")
        if source_collapsed_mask:
            copied_mask = output_path / "mask" / Path(str(source_collapsed_mask)).name
            if copied_mask.exists():
                collapsed_mask_path = copied_mask
    if collapsed_mask_path is None or not collapsed_mask_path.exists():
        raise FileNotFoundError("Could not resolve copied collapsed mask path for anchored derivation.")
    if wet_reference_mask_path is None:
        source_wet_reference = source_manifest.get("source_paths", {}).get("wet_reference_mask")
        wet_reference_mask_path = source_wet_reference if source_wet_reference is not None else collapsed_mask_path
    origin_mask_path = output_paths.get("origin_mask") or output_paths.get("width_scope_mask")
    rivgraph_distance_path = output_paths.get("rivgraph_distance")
    source_cleaned_mask = source_manifest.get("source_paths", {}).get("cleaned_mask")
    if source_cleaned_mask is None and panelized_execution:
        source_cleaned_mask = (
            source_manifest.get("source_paths", {}).get("wet_reference_mask")
            or collapsed_mask_path
        )
    if source_cleaned_mask is None:
        raise FileNotFoundError("Could not resolve source cleaned mask path for anchored derivation.")
    boundary_output_dir = Path(boundary_component_output_dir) if boundary_component_output_dir is not None else (output_path / "boundary_components")
    resolved_boundary_component_labels = (
        boundary_component_labels_path
        or source_manifest.get("output_paths", {}).get("boundary_component_labels")
        or source_manifest.get("source_paths", {}).get("boundary_component_labels")
    )
    boundary_outputs = _ensure_boundary_component_mask(
        cleaned_mask_path=source_cleaned_mask,
        reviewed_links_path=reviewed_links_path,
        output_dir=boundary_output_dir,
        boundary_component_labels_path=resolved_boundary_component_labels,
    )
    resolved_boundary_component_labels_path = Path(boundary_outputs["labels"])

    directed_links_path = output_paths.get("directed_links")
    directed_nodes_path = output_paths.get("directed_nodes")
    if directed_links_path is None or directed_nodes_path is None:
        raise FileNotFoundError("Could not resolve copied directed graph paths for anchored derivation.")

    matching_dir = output_path / "matching"
    width_dir = output_path / "widths"
    summary_dir = output_path / "summary"
    directed_dir = output_path / "directed"
    mask_dir = output_path / "mask"

    node_match_path = output_paths.get("node_match")
    link_match_path = output_paths.get("link_match")
    link_lineage_path = output_paths.get("link_lineage")
    if node_match_path is None or link_match_path is None or link_lineage_path is None:
        raise FileNotFoundError("Could not resolve copied matching paths for anchored derivation.")

    directed_links = gpd.read_file(directed_links_path)
    directed_nodes = gpd.read_file(directed_nodes_path)
    realized_directed_links = directed_links.copy()
    realized_width_links_path = output_paths.get("links_with_width_families")
    realized_width_samples_path = output_paths.get("link_width_samples")
    if realized_width_samples_path is None or not realized_width_samples_path.exists():
        compact_width_samples_path = width_dir / "link_width_samples.parquet"
        if compact_width_samples_path.exists():
            realized_width_samples_path = compact_width_samples_path
    if realized_width_links_path is None or not realized_width_links_path.exists():
        raise FileNotFoundError(
            "Anchored derivation requires the realized links-with-width-families product."
        )
    if realized_width_samples_path is None or not realized_width_samples_path.exists():
        raise FileNotFoundError(
            "Anchored derivation requires the realized link-width samples."
        )
    realized_width_links = gpd.read_file(realized_width_links_path)
    if realized_width_samples_path.suffix.lower() == ".parquet":
        realized_width_samples = pd.read_parquet(realized_width_samples_path)
    else:
        realized_width_samples = pd.read_csv(realized_width_samples_path)
    node_match = pd.read_csv(node_match_path)
    link_match = pd.read_csv(link_match_path)
    link_lineage = pd.read_csv(link_lineage_path)
    collapse_components = pd.read_csv(summary_dir / "collapse_components.csv")
    edit_geometries = gpd.read_file(mask_dir / "collapse_edit_geometries.gpkg")

    reviewed_links = gpd.read_file(reviewed_links_path)
    reviewed_nodes = gpd.read_file(reviewed_nodes_path)
    try:
        parent_width_samples = read_width_samples(Path(reviewed_links_path).parent)
    except FileNotFoundError:
        parent_width_samples = None
    with rasterio.open(collapsed_mask_path) as src:
        pixel_length = _pixel_length_from_transform(src.transform)
        collapsed_profile = src.profile.copy()
    if match_tolerance is None:
        match_tolerance = pixel_length * 1.25

    _, units, _ = analyze_network(
        reviewed_links_path,
        reviewed_nodes_path,
        max_path_cutoff=max_path_cutoff,
        max_paths=max_paths,
    )
    units_by_id = {unit.unit_id: unit for unit in units}
    missing_unit_ids = sorted(unit_id for unit_id in selected_unit_ids if unit_id not in units_by_id)
    if missing_unit_ids:
        raise ValueError(f"The following selected unit_ids were not found in the reviewed graph: {missing_unit_ids}")
    if origin_mask_path is None or not origin_mask_path.exists():
        _, origin_mask, _, collapse_components, edit_geometries = _build_selection_masks(
            cleaned_mask_path=source_cleaned_mask,
            reviewed_links=reviewed_links,
            resolved_unit_ids=selected_unit_ids,
            units_by_id=units_by_id,
            preferred_width_field=str(source_manifest.get("preferred_width_field", DEFAULT_FOOTPRINT_WIDTH_FIELD)),
            footprint_buffer_scale=float(source_manifest.get("footprint_buffer_scale", DEFAULT_FOOTPRINT_BUFFER_SCALE)),
            all_touched=bool(source_manifest.get("all_touched", True)),
            allow_noop=True,
        )
        origin_mask_path = _write_mask(mask_dir / "origin_mask.tif", origin_mask, collapsed_profile)
        collapse_components.to_csv(summary_dir / "collapse_components.csv", index=False)
        edit_geometries.to_file(mask_dir / "collapse_edit_geometries.gpkg", driver="GPKG")
    bank_labels_path = mask_dir / "bank_labels.tif"
    if panelized_execution:
        from network_variants.panelized_raster import write_state_bank_labels_windowed

        removed_bank_labels = write_state_bank_labels_windowed(
            boundary_component_labels_path=resolved_boundary_component_labels_path,
            origin_mask_path=origin_mask_path,
            output_path=bank_labels_path,
        )
    else:
        with rasterio.open(origin_mask_path) as origin_src:
            origin_mask_array = origin_src.read(1).astype(np.uint16)
        bank_labels_path, removed_bank_labels = _write_state_bank_labels(
            boundary_component_labels_path=resolved_boundary_component_labels_path,
            origin_mask=origin_mask_array,
            output_path=bank_labels_path,
            profile=collapsed_profile,
        )
    stage_timings["load_and_bank_setup_seconds"] = time.perf_counter() - setup_started

    segmentation_started = time.perf_counter()
    anchored_graph = derive_anchored_graph_from_realized(
        parent_links=reviewed_links,
        parent_nodes=reviewed_nodes,
        realized_links=directed_links,
        realized_nodes=directed_nodes,
        realized_node_match=node_match,
        realized_link_match=link_match,
        realized_link_lineage=link_lineage,
        selected_unit_ids=selected_unit_ids,
        units_by_id=units_by_id,
        match_tolerance=float(match_tolerance),
    )
    directed_links = anchored_graph.directed_links
    directed_nodes = anchored_graph.directed_nodes
    node_match = anchored_graph.node_match
    link_match = anchored_graph.link_match
    link_lineage = anchored_graph.link_lineage
    anchor_segmentation_summary = anchored_graph.anchor_segmentation_summary
    direction_summary = anchored_graph.direction_summary
    stage_timings["anchor_segmentation_and_matching_seconds"] = (
        time.perf_counter() - segmentation_started
    )

    sword_started = time.perf_counter()
    effective_sword_wse_field = sword_wse_field or source_manifest.get("sword_matching", {}).get("wse_field")
    directed_nodes, node_sword_match, sword_match_summary = match_variant_nodes_to_sword(
        directed_nodes=directed_nodes,
        parent_nodes=reviewed_nodes,
        node_match=node_match,
        sword_node_source_path=sword_node_source_path,
        sword_wse_field=effective_sword_wse_field,
        sword_match_tolerance=sword_match_tolerance,
        example_id=example_id,
        sword_example_station_source_path=sword_example_station_source_path,
        sword_station_match_source_path=sword_station_match_source_path,
        sword_reach_buffer_steps=sword_reach_buffer_steps,
    )
    stage_timings["sword_matching_seconds"] = time.perf_counter() - sword_started

    group_label = source_manifest.get("group_label")
    node_match = _apply_collapsed_selection_metadata(
        node_match,
        unit_ids=selected_unit_ids,
        group_label=group_label,
    )
    link_match = _apply_collapsed_selection_metadata(
        link_match,
        unit_ids=selected_unit_ids,
        group_label=group_label,
    )
    directed_links = _apply_collapsed_selection_metadata(
        directed_links,
        unit_ids=selected_unit_ids,
        group_label=group_label,
    )
    directed_nodes = _apply_collapsed_selection_metadata(
        directed_nodes,
        unit_ids=selected_unit_ids,
        group_label=group_label,
    )
    node_sword_match = _apply_collapsed_selection_metadata(
        node_sword_match,
        unit_ids=selected_unit_ids,
        group_label=group_label,
    )
    link_lineage = _apply_collapsed_selection_metadata(
        link_lineage,
        unit_ids=selected_unit_ids,
        group_label=group_label,
    )
    directed_links = directed_links.drop(
        columns=[column for column in link_lineage.columns if column in directed_links.columns and column != "id_link"],
        errors="ignore",
    ).merge(link_lineage, on="id_link", how="left", validate="1:1")

    width_patch_started = time.perf_counter()
    from network_variants.panelized_raster import compute_segmented_width_patch

    secondary_width_kwargs: dict[str, object] = {}
    if parent_width_samples is not None:
        secondary_width_kwargs = {
            "secondary_source_links": reviewed_links,
            "secondary_source_width_links": reviewed_links,
            "secondary_source_width_samples": parent_width_samples,
        }
    enriched_links, anchored_link_width_samples, anchored_width_patch_summary = (
        compute_segmented_width_patch(
            segmented_links=directed_links,
            source_links=realized_directed_links,
            source_width_links=realized_width_links,
            source_width_samples=realized_width_samples,
            bank_labels_path=bank_labels_path,
            origin_mask_path=origin_mask_path,
            bank_candidate_neighbors_max=bank_candidate_neighbors_max,
            **secondary_width_kwargs,
        )
    )
    stage_timings["targeted_width_patch_seconds"] = time.perf_counter() - width_patch_started
    width_finalize_started = time.perf_counter()
    link_width_samples = _apply_collapsed_selection_metadata(
        anchored_link_width_samples,
        unit_ids=selected_unit_ids,
        group_label=group_label,
    )
    directed_links, enriched_links, segmented_stub_inheritance_summary = _apply_segmented_stub_width_inheritance(
        directed_links=directed_links,
        enriched_links=enriched_links,
        link_width_samples=link_width_samples,
    )

    directed_enriched_links = _merge_width_families_onto_links(directed_links, enriched_links)
    parent_width_samples_for_inheritance = (
        parent_width_samples if parent_width_samples is not None else pd.DataFrame()
    )
    directed_enriched_links, link_width_samples, width_product_inheritance = (
        inherit_unchanged_link_width_products(
            child_links=directed_enriched_links,
            child_width_samples=link_width_samples,
            parent_links=reviewed_links,
            parent_width_samples=parent_width_samples_for_inheritance,
            link_lineage=link_lineage,
        )
    )
    link_width_samples = _apply_collapsed_selection_metadata(
        link_width_samples,
        unit_ids=selected_unit_ids,
        group_label=group_label,
    )
    link_width_samples["example_id"] = example_id
    link_width_samples["variant_id"] = variant_id
    changed_width_link_ids = directed_enriched_links.loc[
        directed_enriched_links["width_inheritance_status"].eq("changed_requires_evaluation"),
        "id_link",
    ].astype(int).tolist()
    directed_enriched_links, routing_width_audit, routing_width_application = apply_routing_width_model(
        directed_enriched_links,
        link_width_samples,
        directed_nodes,
        example_id=example_id,
        variant_id=variant_id,
        apply_known_base_exclusions=False,
        evaluation_link_ids=changed_width_link_ids,
    )
    directed_enriched_links.loc[
        directed_enriched_links["id_link"].astype(int).isin(changed_width_link_ids),
        "width_inheritance_status",
    ] = "changed_evaluated"
    routing_width_paths = write_routing_width_audit(
        routing_width_audit,
        routing_width_application,
        output_dir=width_dir,
    )
    link_width_families = _write_width_family_outputs(
        width_dir=width_dir,
        link_width_samples=link_width_samples,
        directed_enriched_links=directed_enriched_links,
    )
    stage_timings["width_inheritance_and_routing_seconds"] = (
        time.perf_counter() - width_finalize_started
    )

    serialization_started = time.perf_counter()
    link_match.to_csv(matching_dir / "link_match.csv", index=False)
    link_lineage.to_csv(matching_dir / "link_lineage.csv", index=False)
    node_match.to_csv(matching_dir / "node_match.csv", index=False)
    node_sword_match.to_csv(matching_dir / "node_sword_match.csv", index=False)
    directed_links.to_file(directed_links_path, driver="GPKG")
    directed_nodes.to_file(directed_nodes_path, driver="GPKG")
    with (directed_dir / "direction_validation_report.json").open("w", encoding="utf-8") as handle:
        json.dump(direction_summary["validation_report"], handle, indent=2)
    stage_timings["variant_output_serialization_seconds"] = (
        time.perf_counter() - serialization_started
    )
    stage_timings["anchored_variant_total_seconds"] = time.perf_counter() - derive_started

    manifest = dict(source_manifest)
    manifest["collapse_corridor_mode"] = "anchored"
    manifest["anchor_segmentation"] = anchor_segmentation_summary
    width_scope_description = "targeted anchored-link width patch with exact unchanged-link reuse"
    manifest["scope"] = [
        "collapsed mask reuse from realized variant",
        "anchored corridor segmentation",
        "parent-child graph rematching",
        width_scope_description,
        "SWORD node matching/propagation",
    ]
    manifest["segmented_stub_inheritance"] = segmented_stub_inheritance_summary
    manifest["width_product_inheritance"] = width_product_inheritance
    manifest["routing_width_application"] = routing_width_application
    manifest["anchored_stage_timings"] = {
        key: round(float(value), 6) for key, value in stage_timings.items()
    }
    if anchored_width_patch_summary is not None:
        manifest["anchored_width_patch"] = anchored_width_patch_summary
    manifest["ready_for_rapid"] = bool(direction_summary.get("is_valid", False))
    manifest["bank_candidate_neighbors_max"] = int(bank_candidate_neighbors_max)
    manifest["source_paths"] = dict(manifest.get("source_paths", {}))
    manifest["source_paths"]["reviewed_links"] = str(Path(reviewed_links_path).resolve())
    manifest["source_paths"]["reviewed_nodes"] = str(Path(reviewed_nodes_path).resolve())
    manifest["source_paths"]["wet_reference_mask"] = str(Path(wet_reference_mask_path).resolve())
    manifest["source_paths"]["realized_variant_output_dir"] = str(source_output_path.resolve())
    manifest["source_paths"]["boundary_component_labels"] = str(resolved_boundary_component_labels_path.resolve())
    manifest["bank_label_summary"] = {
        "removed_land_labels": removed_bank_labels,
        "n_removed_land_labels": int(len(removed_bank_labels)),
    }
    manifest["sword_matching"] = {
        "wse_field": effective_sword_wse_field,
        "match_tolerance": sword_match_tolerance,
        "reach_buffer_steps": int(sword_reach_buffer_steps),
        "n_matched_nodes": int(node_sword_match["sword_node_id"].notna().sum()) if not node_sword_match.empty else 0,
        "n_propagated_matches": int(node_sword_match["sword_match_from_parent"].fillna(False).sum()) if not node_sword_match.empty else 0,
        "candidate_scope": sword_match_summary.get("scope"),
        "candidate_region": sword_match_summary.get("candidate_region"),
        "candidate_reach_count": sword_match_summary.get("candidate_reach_count"),
        "candidate_reach_ids": sword_match_summary.get("candidate_reach_ids", []),
        "upstream_station_key": sword_match_summary.get("upstream_station_key"),
        "downstream_station_key": sword_match_summary.get("downstream_station_key"),
        "upstream_reach_id": sword_match_summary.get("upstream_reach_id"),
        "downstream_reach_id": sword_match_summary.get("downstream_reach_id"),
        "filter_reason": sword_match_summary.get("reason"),
    }
    manifest["output_paths"] = {
        key: str(path.resolve()) if path is not None else None
        for key, path in output_paths.items()
    }
    manifest["output_paths"]["origin_mask"] = str(Path(origin_mask_path).resolve()) if origin_mask_path is not None else None
    manifest["output_paths"]["width_scope_mask"] = str(Path(origin_mask_path).resolve()) if origin_mask_path is not None else None
    manifest["output_paths"]["bank_labels"] = str(bank_labels_path.resolve())
    manifest["output_paths"]["boundary_component_labels"] = str(resolved_boundary_component_labels_path.resolve())
    manifest["output_paths"]["routing_width_application"] = routing_width_paths["link_audit"]
    manifest["output_paths"]["routing_width_application_manifest"] = routing_width_paths["manifest"]
    manifest["direction_summary"] = direction_summary
    manifest["generated_at"] = datetime.now(timezone.utc).isoformat()
    manifest["code_version"] = _git_revision()
    with (summary_dir / "variant_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)

    rivgraph_links_path = output_paths.get("rivgraph_links")
    rivgraph_nodes_path = output_paths.get("rivgraph_nodes")
    if rivgraph_links_path is None or rivgraph_nodes_path is None:
        raise FileNotFoundError("Could not resolve copied RivGraph link/node paths for anchored derivation.")

    results = NetworkVariantOutputs(
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
        width_scope_mask_path=origin_mask_path,
        bank_labels_path=bank_labels_path,
        boundary_component_labels_path=resolved_boundary_component_labels_path,
        rivgraph_links_path=rivgraph_links_path,
        rivgraph_nodes_path=rivgraph_nodes_path,
        directed_links_path=directed_links_path,
        directed_nodes_path=directed_nodes_path,
        rivgraph_centerline_path=output_paths.get("rivgraph_centerline"),
        sword_reaches_path=output_paths.get("sword_reaches"),
        sword_nodes_path=output_paths.get("sword_nodes"),
        manifest=manifest,
        origin_mask_path=origin_mask_path,
        rivgraph_distance_path=rivgraph_distance_path,
    )
    if not bool(direction_summary.get("is_valid", False)):
        raise VariantDirectionValidationError(results, direction_summary.get("issues", []))
    return results


def refresh_variant_width_scope_and_families(
    *,
    variant_output_dir: str | Path,
    reviewed_links_path: str | Path | None = None,
    reviewed_nodes_path: str | Path | None = None,
    cleaned_mask_path: str | Path | None = None,
    wet_reference_mask_path: str | Path | None = None,
    boundary_component_labels_path: str | Path | None = None,
    boundary_component_output_dir: str | Path | None = None,
    bank_candidate_neighbors_max: int | None = None,
    preferred_width_field: str | None = None,
    footprint_buffer_scale: float | None = None,
    transect_scale: float = DEFAULT_TRANSECT_SCALE,
    min_transect_pixels: float = DEFAULT_MIN_TRANSECT_PIXELS,
    max_path_cutoff: int = 100,
    max_paths: int = 5000,
) -> dict[str, Path]:
    output_path = Path(variant_output_dir)
    manifest_path = output_path / "summary" / "variant_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Variant manifest not found: {manifest_path}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    source_paths = dict(manifest.get("source_paths", {}))
    output_paths = dict(manifest.get("output_paths", {}))
    selected_unit_ids = [int(unit_id) for unit_id in manifest.get("selected_unit_ids", [])]
    group_label = manifest.get("group_label")
    collapse_corridor_mode = str(manifest.get("collapse_corridor_mode", "realized"))

    resolved_reviewed_links = Path(reviewed_links_path) if reviewed_links_path is not None else Path(source_paths["reviewed_links"])
    resolved_reviewed_nodes = Path(reviewed_nodes_path) if reviewed_nodes_path is not None else Path(source_paths["reviewed_nodes"])
    resolved_cleaned_mask = Path(cleaned_mask_path) if cleaned_mask_path is not None else Path(source_paths["cleaned_mask"])
    resolved_wet_reference_mask = (
        Path(wet_reference_mask_path)
        if wet_reference_mask_path is not None
        else Path(source_paths.get("wet_reference_mask") or output_paths["collapsed_mask"])
    )
    boundary_output_dir = Path(boundary_component_output_dir) if boundary_component_output_dir is not None else (output_path / "boundary_components")
    resolved_boundary_component_labels = (
        boundary_component_labels_path
        or output_paths.get("boundary_component_labels")
        or source_paths.get("boundary_component_labels")
    )
    boundary_outputs = _ensure_boundary_component_mask(
        cleaned_mask_path=resolved_cleaned_mask,
        reviewed_links_path=resolved_reviewed_links,
        output_dir=boundary_output_dir,
        boundary_component_labels_path=resolved_boundary_component_labels,
    )
    resolved_boundary_component_labels_path = Path(boundary_outputs["labels"])
    collapsed_mask_path = Path(output_paths["collapsed_mask"])
    directed_links_path = Path(output_paths["directed_links"])
    width_dir = output_path / "widths"
    mask_dir = output_path / "mask"
    summary_dir = output_path / "summary"

    reviewed_links = gpd.read_file(resolved_reviewed_links)
    reviewed_nodes = gpd.read_file(resolved_reviewed_nodes)
    directed_links = gpd.read_file(directed_links_path)

    _, units, _ = analyze_network(
        resolved_reviewed_links,
        resolved_reviewed_nodes,
        max_path_cutoff=max_path_cutoff,
        max_paths=max_paths,
    )
    units_by_id = {unit.unit_id: unit for unit in units}
    missing_unit_ids = sorted(unit_id for unit_id in selected_unit_ids if unit_id not in units_by_id)
    if missing_unit_ids:
        raise ValueError(f"The following selected unit_ids were not found in the reviewed graph: {missing_unit_ids}")

    collapsed_mask, origin_mask, mask_summary, collapse_components, edit_geometries = _build_selection_masks(
        cleaned_mask_path=resolved_cleaned_mask,
        reviewed_links=reviewed_links,
        resolved_unit_ids=selected_unit_ids,
        units_by_id=units_by_id,
        preferred_width_field=str(preferred_width_field or manifest.get("preferred_width_field", DEFAULT_FOOTPRINT_WIDTH_FIELD)),
        footprint_buffer_scale=float(
            footprint_buffer_scale if footprint_buffer_scale is not None else manifest.get("footprint_buffer_scale", DEFAULT_FOOTPRINT_BUFFER_SCALE)
        ),
        all_touched=bool(manifest.get("all_touched", True)),
        allow_noop=bool(manifest.get("allow_noop", False) or not selected_unit_ids),
    )

    with rasterio.open(collapsed_mask_path) as src:
        collapsed_profile = src.profile.copy()
    origin_mask_path = _write_mask(mask_dir / "origin_mask.tif", origin_mask, collapsed_profile)
    bank_labels_path, removed_bank_labels = _write_state_bank_labels(
        boundary_component_labels_path=resolved_boundary_component_labels_path,
        origin_mask=origin_mask,
        output_path=mask_dir / "bank_labels.tif",
        profile=collapsed_profile,
    )

    collapse_components = _apply_collapsed_selection_metadata(
        collapse_components,
        unit_ids=selected_unit_ids,
        group_label=group_label,
    )
    edit_geometries = _apply_collapsed_selection_metadata(
        edit_geometries,
        unit_ids=selected_unit_ids,
        group_label=group_label,
    )
    collapse_components.to_csv(summary_dir / "collapse_components.csv", index=False)
    edit_geometries.to_file(mask_dir / "collapse_edit_geometries.gpkg", driver="GPKG")
    enriched_links, link_width_samples = compute_width_families(
        directed_links,
        collapsed_mask_path=collapsed_mask_path,
        wet_reference_mask_path=resolved_wet_reference_mask,
        origin_mask_path=origin_mask_path,
        rivgraph_distance_path=output_paths.get("rivgraph_distance"),
        bank_labels_path=bank_labels_path,
        bank_candidate_neighbors_max=int(bank_candidate_neighbors_max or manifest.get("bank_candidate_neighbors_max", 2048)),
    )
    link_width_samples = _apply_collapsed_selection_metadata(
        link_width_samples,
        unit_ids=selected_unit_ids,
        group_label=group_label,
    )
    directed_enriched_links = _merge_width_families_onto_links(directed_links, enriched_links)
    directed_enriched_links = _apply_width_sample_flow_alignment(
        directed_enriched_links,
        collapse_corridor_mode=collapse_corridor_mode,
    )
    _write_width_family_outputs(
        width_dir=width_dir,
        link_width_samples=link_width_samples,
        directed_enriched_links=directed_enriched_links,
    )

    manifest["preferred_width_field"] = str(preferred_width_field or manifest.get("preferred_width_field", DEFAULT_FOOTPRINT_WIDTH_FIELD))
    manifest["footprint_buffer_scale"] = float(
        footprint_buffer_scale if footprint_buffer_scale is not None else manifest.get("footprint_buffer_scale", DEFAULT_FOOTPRINT_BUFFER_SCALE)
    )
    manifest["transect_scale"] = float(transect_scale)
    manifest["min_transect_pixels"] = float(min_transect_pixels)
    manifest["bank_candidate_neighbors_max"] = int(bank_candidate_neighbors_max or manifest.get("bank_candidate_neighbors_max", 2048))
    manifest["mask_summary"] = mask_summary
    manifest["source_paths"] = source_paths
    manifest["source_paths"]["cleaned_mask"] = str(resolved_cleaned_mask.resolve())
    manifest["source_paths"]["wet_reference_mask"] = str(resolved_wet_reference_mask.resolve())
    manifest["source_paths"]["reviewed_links"] = str(resolved_reviewed_links.resolve())
    manifest["source_paths"]["reviewed_nodes"] = str(resolved_reviewed_nodes.resolve())
    manifest["source_paths"]["boundary_component_labels"] = str(resolved_boundary_component_labels_path.resolve())
    manifest["output_paths"] = output_paths
    manifest["output_paths"]["origin_mask"] = str(origin_mask_path.resolve())
    manifest["output_paths"]["width_scope_mask"] = str(origin_mask_path.resolve())
    manifest["output_paths"]["bank_labels"] = str(bank_labels_path.resolve())
    manifest["output_paths"]["boundary_component_labels"] = str(resolved_boundary_component_labels_path.resolve())
    manifest["output_paths"]["collapse_components"] = str((summary_dir / "collapse_components.csv").resolve())
    manifest["output_paths"]["edit_geometries"] = str((mask_dir / "collapse_edit_geometries.gpkg").resolve())
    manifest["output_paths"]["link_width_families"] = str((width_dir / "link_width_families.csv").resolve())
    manifest["output_paths"]["link_width_samples"] = str((width_dir / "link_width_samples.csv").resolve())
    manifest["output_paths"]["links_with_width_families"] = str((width_dir / "links_with_width_families.gpkg").resolve())
    manifest["width_scope_patch"] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "mode": "two_bank_dt_origin_mask_bank_labels",
        "removed_land_labels": removed_bank_labels,
    }
    manifest["generated_at"] = datetime.now(timezone.utc).isoformat()
    manifest["code_version"] = _git_revision()
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    return {
        "variant_output_dir": output_path.resolve(),
        "origin_mask": origin_mask_path.resolve(),
        "width_scope_mask": origin_mask_path.resolve(),
        "bank_labels": bank_labels_path.resolve(),
        "link_width_families": (width_dir / "link_width_families.csv").resolve(),
        "link_width_samples": (width_dir / "link_width_samples.csv").resolve(),
        "links_with_width_families": (width_dir / "links_with_width_families.gpkg").resolve(),
    }


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
    parser.add_argument(
        "--collapse-corridor-mode",
        choices=COLLAPSE_CORRIDOR_MODES,
        default="realized",
        help="Whether to keep the full realized collapsed corridor or split it back to the nominal selection anchors.",
    )
    parser.add_argument("--wet-reference-mask", default=None, help="Optional mask used for wet-width sampling. Defaults to the parent cleaned mask.")
    parser.add_argument("--preferred-width-field", default=DEFAULT_FOOTPRINT_WIDTH_FIELD, help="Preferred reviewed-link width field for footprint buffering.")
    parser.add_argument("--footprint-buffer-scale", type=float, default=DEFAULT_FOOTPRINT_BUFFER_SCALE, help="Multiplier applied to footprint width / 2 buffering.")
    parser.add_argument("--disable-all-touched", action="store_true", help="Disable all_touched rasterization when creating collapse footprints.")
    parser.add_argument("--allow-noop", action="store_true", help="Allow variants that produce no added pixels.")
    parser.add_argument("--single-thread", action="store_true", help="Pass single_thread=True to the RivGraph river class.")
    parser.add_argument("--disable-sword-export", action="store_true", help="Skip SWORD-style RivGraph exports.")
    parser.add_argument("--transect-scale", type=float, default=DEFAULT_TRANSECT_SCALE, help="Multiplier for transect half-length relative to local total width.")
    parser.add_argument("--min-transect-pixels", type=float, default=DEFAULT_MIN_TRANSECT_PIXELS, help="Minimum transect half-length in raster pixels.")
    parser.add_argument(
        "--bank-candidate-neighbors-max",
        type=int,
        default=2048,
        help="Maximum number of nearest boundary pixels to query adaptively when seeking left/right bank candidates.",
    )
    parser.add_argument("--match-tolerance", type=float, default=None, help="Optional spatial tolerance for parent-child graph matching. Defaults to 1.25 raster pixels.")
    parser.add_argument("--sword-node-source", default=None, help="Optional SWORD node source file or parquet directory used for node matching.")
    parser.add_argument("--sword-wse-field", default=None, help="Optional WSE field name in the SWORD node source. Defaults to automatic detection.")
    parser.add_argument("--sword-match-tolerance", type=float, default=None, help="Optional maximum SWORD node-match distance in CRS units/meters after reprojection.")
    parser.add_argument("--sword-example-stations-source", default=None, help="Optional example-station source used to derive example-specific SWORD reach corridors. Defaults to the repo hierarchy_examples_filtered_subdaily_manual_updates_final.gpkg when present.")
    parser.add_argument("--sword-station-match-source", default=None, help="Optional station-to-SWORD match source used to derive example-specific SWORD reach corridors. Defaults to the repo selected_event_stations_same_main_path.gpkg when present.")
    parser.add_argument("--sword-reach-buffer-steps", type=int, default=2, help="Number of reach steps to extend upstream and downstream beyond the example corridor endpoints when constraining SWORD node candidates.")
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
        collapse_corridor_mode=args.collapse_corridor_mode,
        wet_reference_mask_path=args.wet_reference_mask,
        preferred_width_field=args.preferred_width_field,
        footprint_buffer_scale=args.footprint_buffer_scale,
        all_touched=not args.disable_all_touched,
        allow_noop=args.allow_noop,
        single_thread=args.single_thread,
        export_sword=not args.disable_sword_export,
        transect_scale=args.transect_scale,
        min_transect_pixels=args.min_transect_pixels,
        bank_candidate_neighbors_max=args.bank_candidate_neighbors_max,
        match_tolerance=args.match_tolerance,
        sword_node_source_path=args.sword_node_source,
        sword_wse_field=args.sword_wse_field,
        sword_match_tolerance=args.sword_match_tolerance,
        sword_example_station_source_path=args.sword_example_stations_source,
        sword_station_match_source_path=args.sword_station_match_source,
        sword_reach_buffer_steps=args.sword_reach_buffer_steps,
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
