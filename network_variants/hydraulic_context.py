from __future__ import annotations

import json
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any

import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd
from shapely.geometry import LineString, Point
from shapely.ops import linemerge, unary_union


HYDRAULIC_CONTEXT_MANIFEST = "hydraulic_context_manifest.json"


@dataclass(frozen=True, slots=True)
class SlopeConfig:
    min_slope: float = 1e-6
    preferred_length_field: str = "len"
    interpolate_along_corridor: bool = True
    min_anchor_nodes_for_profile: int = 2
    slope_profile_mode: str = "network_dist"
    reference_section_summary_path: str | None = None
    reference_section_use_parent_match: bool = True
    reference_section_fallback_to_network_dist: bool = True
    fill_from_nearest_valid_link: bool = True
    max_slope_for_k: float | None = None
    section_slope_ratio_min: float | None = None
    section_slope_ratio_max: float | None = None
    use_section_slope_fallback: bool = True
    use_compound_bubble_slope_qc: bool = True
    use_compound_bubble_slope_repair: bool = True
    compound_bubble_slope_repair_flag_factor: float = 3.0
    compound_bubble_max_paths_for_qc: int = 500
    compound_bubble_path_drop_rel_tol: float = 0.1
    compound_bubble_path_drop_abs_tol_m: float = 0.05


@dataclass(frozen=True, slots=True)
class HydraulicContextResult:
    node_context: pd.DataFrame
    link_context: pd.DataFrame
    section_slope_summary: dict[str, object]


@dataclass(frozen=True, slots=True)
class StateHydraulicContextOutputs:
    output_dir: Path
    link_context_path: Path
    node_context_path: Path
    manifest_path: Path
    link_context: pd.DataFrame
    node_context: pd.DataFrame
    manifest: dict[str, object]


REQUIRED_NODE_COLUMNS = {
    "id_node",
    "sword_wse",
    "sword_wse_field",
    "sword_wse_fill_method",
}


COMPOUND_BUBBLE_QC_COLUMNS: dict[str, object] = {
    "compound_bubble_id_link": pd.Series(dtype="Int64"),
    "bubble_slope_qc_candidate": False,
    "bubble_slope_qc_valid": False,
    "bubble_slope_qc_reason": "",
    "bubble_entry_node_id": pd.Series(dtype="Int64"),
    "bubble_exit_node_id": pd.Series(dtype="Int64"),
    "bubble_reference_drop_m": np.nan,
    "bubble_reference_dist_span_m": np.nan,
    "bubble_reference_slope": np.nan,
    "bubble_reference_profile_method": "",
    "bubble_link_slope_ref": np.nan,
    "bubble_slope_repair_candidate": False,
    "bubble_slope_repair_applied": False,
    "bubble_slope_repair_method": "",
    "bubble_slope_repair_change_factor": np.nan,
    "bubble_slope_repair_large_change_flag": False,
    "bubble_n_internal_links": pd.Series(dtype="Int64"),
    "bubble_n_paths_checked": pd.Series(dtype="Int64"),
    "bubble_path_drop_min_m": np.nan,
    "bubble_path_drop_max_m": np.nan,
    "bubble_path_drop_spread_m": np.nan,
    "bubble_path_drop_max_abs_error_m": np.nan,
    "bubble_path_drop_max_rel_error": np.nan,
    "bubble_path_drop_qc_failed": False,
}


def _load_unit_summary_if_exists(hierarchy_output_dir: str | Path | None) -> pd.DataFrame | None:
    if hierarchy_output_dir is None:
        return None
    path = Path(hierarchy_output_dir)
    unit_summary_path = path / "unit_summary.csv"
    if not unit_summary_path.exists():
        return None
    return pd.read_csv(unit_summary_path)


def _infer_reference_section_summary_path(*paths: Path) -> Path | None:
    for path in paths:
        try:
            resolved = Path(path).expanduser().resolve()
        except FileNotFoundError:
            continue
        for parent in (resolved.parent, *resolved.parents):
            candidate = parent / "reference_section" / "reference_section_summary.json"
            if candidate.exists():
                return candidate.resolve()
    return None


def _select_variant_lengths(
    links: gpd.GeoDataFrame,
    *,
    preferred_length_field: str,
) -> tuple[pd.Series, pd.Series]:
    geometry_length = pd.Series(np.nan, index=links.index, dtype=float)
    if links.geometry.name in links.columns and links.geometry.notna().any():
        if links.crs is not None and not links.crs.is_geographic:
            geometry_length = links.geometry.length.astype(float)

    field_length = pd.Series(np.nan, index=links.index, dtype=float)
    if preferred_length_field in links.columns:
        field_length = pd.to_numeric(links[preferred_length_field], errors="coerce")

    use_geometry = geometry_length.gt(0)
    lengths = geometry_length.where(use_geometry, field_length)
    length_source = pd.Series(
        np.where(use_geometry, "geometry", preferred_length_field),
        index=links.index,
        dtype="object",
    )

    if lengths.isna().any() or (lengths <= 0).any():
        bad = links.loc[lengths.isna() | lengths.le(0), "id_link"].tolist() if "id_link" in links.columns else []
        raise ValueError(
            "Could not resolve a positive variant length for all links."
            f" Problem id_link values: {bad[:10]}"
        )
    return lengths.astype(float), length_source


def _compute_network_dist_out(
    links: gpd.GeoDataFrame,
    nodes: gpd.GeoDataFrame,
    *,
    lengths: pd.Series,
) -> pd.DataFrame:
    frame = links[["id_link", "id_us_node", "id_ds_node"]].copy()
    frame["id_link"] = pd.to_numeric(frame["id_link"], errors="coerce").astype(int)
    frame["id_us_node"] = pd.to_numeric(frame["id_us_node"], errors="coerce").astype(int)
    frame["id_ds_node"] = pd.to_numeric(frame["id_ds_node"], errors="coerce").astype(int)
    frame["link_length_m"] = pd.to_numeric(lengths, errors="coerce").astype(float).to_numpy()

    node_ids = pd.to_numeric(nodes["id_node"], errors="coerce").dropna().astype(int).tolist()
    if not node_ids:
        return pd.DataFrame(columns=["id_node", "node_network_dist_out_m", "node_network_dist_source"])

    graph = nx.DiGraph()
    graph.add_nodes_from(node_ids)
    successors_by_node: dict[int, list[tuple[int, float]]] = {}
    for row in frame.itertuples(index=False):
        us = int(row.id_us_node)
        ds = int(row.id_ds_node)
        length = float(row.link_length_m)
        graph.add_edge(us, ds)
        successors_by_node.setdefault(us, []).append((ds, length))

    result = pd.DataFrame({"id_node": pd.Series(node_ids, dtype="Int64")})
    if not nx.is_directed_acyclic_graph(graph):
        result["node_network_dist_out_m"] = np.nan
        result["node_network_dist_source"] = "cycle_detected"
        return result

    if "is_outlet" in nodes.columns and nodes["is_outlet"].fillna(False).any():
        outlet_nodes = set(
            pd.to_numeric(
                nodes.loc[nodes["is_outlet"].fillna(False), "id_node"],
                errors="coerce",
            ).dropna().astype(int).tolist()
        )
    else:
        outlet_nodes = {int(node_id) for node_id in graph.nodes if graph.out_degree(node_id) == 0}
    if not outlet_nodes:
        outlet_nodes = {int(node_id) for node_id in graph.nodes if graph.out_degree(node_id) == 0}

    dist_out: dict[int, float] = {}
    for node_id in reversed(list(nx.topological_sort(graph))):
        downstream_options = [
            float(length) + float(dist_out.get(int(ds), 0.0))
            for ds, length in successors_by_node.get(int(node_id), [])
        ]
        if downstream_options:
            dist_out[int(node_id)] = float(max(downstream_options))
        else:
            dist_out[int(node_id)] = 0.0 if int(node_id) in outlet_nodes or graph.out_degree(node_id) == 0 else float("nan")

    result["node_network_dist_out_m"] = result["id_node"].map(
        lambda value: dist_out.get(int(value), float("nan")) if pd.notna(value) else float("nan")
    ).astype(float)
    result["node_network_dist_source"] = "graph_longest_path_to_outlet"
    return result


def _fit_monotone_increasing_profile(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    if len(x) != len(y):
        raise ValueError("x and y must have the same length for monotone profile fitting.")
    if len(x) == 0:
        return np.asarray([], dtype=float)
    if len(x) == 1:
        return np.asarray(y, dtype=float)

    blocks: list[dict[str, float | int]] = []
    for index, value in enumerate(y):
        blocks.append(
            {
                "start": index,
                "end": index,
                "weight": 1.0,
                "sum": float(value),
            }
        )
        while len(blocks) >= 2:
            right = blocks[-1]
            left = blocks[-2]
            left_mean = float(left["sum"]) / float(left["weight"])
            right_mean = float(right["sum"]) / float(right["weight"])
            if left_mean <= right_mean:
                break
            merged = {
                "start": int(left["start"]),
                "end": int(right["end"]),
                "weight": float(left["weight"]) + float(right["weight"]),
                "sum": float(left["sum"]) + float(right["sum"]),
            }
            blocks = blocks[:-2] + [merged]

    fitted = np.empty(len(y), dtype=float)
    for block in blocks:
        mean_value = float(block["sum"]) / float(block["weight"])
        fitted[int(block["start"]) : int(block["end"]) + 1] = mean_value
    return fitted


def _interp_linear_extrapolation(
    xp: np.ndarray,
    yp: np.ndarray,
    x: np.ndarray,
) -> np.ndarray:
    if len(xp) != len(yp):
        raise ValueError("xp and yp must have the same length.")
    if len(xp) == 0:
        return np.full(len(x), np.nan, dtype=float)
    if len(xp) == 1:
        return np.full(len(x), float(yp[0]), dtype=float)

    xp = np.asarray(xp, dtype=float)
    yp = np.asarray(yp, dtype=float)
    x = np.asarray(x, dtype=float)
    interpolated = np.interp(x, xp, yp)

    left_slope = float((yp[1] - yp[0]) / (xp[1] - xp[0])) if xp[1] != xp[0] else 0.0
    right_slope = float((yp[-1] - yp[-2]) / (xp[-1] - xp[-2])) if xp[-1] != xp[-2] else 0.0

    left_mask = x < xp[0]
    right_mask = x > xp[-1]
    if left_mask.any():
        interpolated[left_mask] = yp[0] + (x[left_mask] - xp[0]) * left_slope
    if right_mask.any():
        interpolated[right_mask] = yp[-1] + (x[right_mask] - xp[-1]) * right_slope
    return interpolated


def _preferred_profile_coordinate_column(frame: pd.DataFrame) -> str | None:
    for column in ("node_reference_section_dist_out_m", "node_network_dist_out_m", "node_sword_dist_out"):
        if column in frame.columns:
            series = pd.to_numeric(frame[column], errors="coerce")
            if series.notna().any():
                return column
    return None


def _apply_family_representative_slope_inheritance(
    link_context: pd.DataFrame,
    links: gpd.GeoDataFrame,
) -> tuple[pd.DataFrame, dict[str, object]]:
    result = link_context.copy()
    result["family_slope_inheritance_applied"] = False
    result["family_slope_inherited_from_link"] = pd.Series(pd.NA, index=result.index, dtype="Int64")
    result["family_slope_inheritance_reason"] = ""

    required_columns = {"id_link", "family_width_inheritance_applied", "family_width_inherited_from_link"}
    if result.empty or links.empty or not required_columns.issubset(links.columns):
        return result, {"applied_count": 0, "applied_link_ids": []}

    inheritance = links.loc[
        links["family_width_inheritance_applied"].fillna(False).astype(bool),
        ["id_link", "family_width_inherited_from_link"],
    ].copy()
    if inheritance.empty:
        return result, {"applied_count": 0, "applied_link_ids": []}

    inheritance["id_link"] = pd.to_numeric(inheritance["id_link"], errors="coerce").astype("Int64")
    inheritance["family_width_inherited_from_link"] = pd.to_numeric(
        inheritance["family_width_inherited_from_link"],
        errors="coerce",
    ).astype("Int64")
    inheritance = inheritance.dropna(subset=["id_link", "family_width_inherited_from_link"]).copy()
    if inheritance.empty:
        return result, {"applied_count": 0, "applied_link_ids": []}

    link_index_by_id = {
        int(link_id): int(index)
        for index, link_id in result["id_link"].items()
        if pd.notna(link_id)
    }
    applied_link_ids: list[int] = []
    for row in inheritance.itertuples(index=False):
        stub_link_id = int(row.id_link)
        representative_link_id = int(row.family_width_inherited_from_link)
        stub_index = link_index_by_id.get(stub_link_id)
        representative_index = link_index_by_id.get(representative_link_id)
        if stub_index is None or representative_index is None:
            continue

        result.at[stub_index, "slope_used"] = float(result.at[representative_index, "slope_used"])
        result.at[stub_index, "slope_source_method"] = "family_representative_link"
        result.at[stub_index, "slope_adjusted"] = True
        result.at[stub_index, "slope_reason"] = "inherited_from_segmented_family_representative"
        if "slope_minimum_applied" in result.columns:
            result.at[stub_index, "slope_minimum_applied"] = bool(result.at[representative_index, "slope_minimum_applied"])
        result.at[stub_index, "family_slope_inheritance_applied"] = True
        result.at[stub_index, "family_slope_inherited_from_link"] = representative_link_id
        result.at[stub_index, "family_slope_inheritance_reason"] = "inherited_from_segmented_family_representative"
        applied_link_ids.append(stub_link_id)

    return result, {
        "applied_count": int(len(applied_link_ids)),
        "applied_link_ids": [int(link_id) for link_id in applied_link_ids],
    }


def _normalize_slope_profile_mode(mode: str | None) -> str:
    normalized = str(mode or "network_dist").strip().lower()
    if normalized not in {"network_dist", "reference_section"}:
        raise ValueError(
            f"Unsupported slope_profile_mode '{mode}'. Expected one of: network_dist, reference_section."
        )
    return normalized


def _single_reference_line(centerline: gpd.GeoDataFrame) -> LineString:
    geometries = [geometry for geometry in centerline.geometry.tolist() if geometry is not None and not geometry.is_empty]
    if not geometries:
        raise ValueError("Reference-section centerline does not contain any valid geometry.")
    if len(geometries) == 1 and isinstance(geometries[0], LineString):
        line = geometries[0]
    else:
        merged = linemerge(unary_union(geometries))
        if isinstance(merged, LineString):
            line = merged
        elif hasattr(merged, "geoms"):
            line = max((geometry for geometry in merged.geoms if isinstance(geometry, LineString)), key=lambda geometry: geometry.length)
        else:
            raise ValueError("Could not resolve a LineString from the reference-section centerline geometry.")
    if line.length <= 0:
        raise ValueError("Reference-section centerline has zero length.")
    return line


def _resolve_reference_section_sword_nodes_path(
    summary: dict[str, Any],
    *,
    centerline_path: Path,
) -> Path | None:
    paths = summary.get("paths", {}) or {}
    direct_path = str(paths.get("sword_nodes_path", "") or "").strip()
    if direct_path:
        candidate = Path(direct_path).expanduser().resolve()
        if candidate.exists():
            return candidate

    variant_manifest_path = str(paths.get("variant_manifest_path", "") or "").strip()
    if variant_manifest_path:
        manifest_candidate = Path(variant_manifest_path).expanduser().resolve()
        if manifest_candidate.exists():
            try:
                variant_manifest = json.loads(manifest_candidate.read_text(encoding="utf-8"))
            except Exception:
                variant_manifest = {}
            for container_key in ("output_paths", "paths"):
                container = variant_manifest.get(container_key, {}) or {}
                sword_nodes_path = str(container.get("sword_nodes", "") or container.get("sword_nodes_path", "") or "").strip()
                if not sword_nodes_path:
                    continue
                candidate = Path(sword_nodes_path).expanduser().resolve()
                if candidate.exists():
                    return candidate

    for pattern in ("*_SWORD_nodes.gpkg", "*SWORD_nodes.gpkg"):
        matches = sorted(centerline_path.parent.glob(pattern))
        if matches:
            return matches[0].resolve()
    return None


def _point_like(geometry: Any) -> Point | None:
    if geometry is None or getattr(geometry, "is_empty", True):
        return None
    if isinstance(geometry, Point):
        return geometry
    centroid = geometry.centroid
    if centroid is None or centroid.is_empty:
        return None
    return centroid


@lru_cache(maxsize=8)
def _load_reference_section_support(summary_path_str: str) -> dict[str, Any]:
    summary_path = Path(summary_path_str).expanduser().resolve()
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    paths = summary.get("paths", {}) or {}
    centerline_path = Path(str(paths.get("centerline_path", "") or "")).expanduser().resolve()
    directed_nodes_path = Path(str(paths.get("directed_nodes_path", "") or "")).expanduser().resolve()
    if not centerline_path.exists():
        raise FileNotFoundError(f"Reference-section centerline was not found: {centerline_path}")
    if not directed_nodes_path.exists():
        raise FileNotFoundError(f"Reference-section directed nodes were not found: {directed_nodes_path}")

    centerline = gpd.read_file(centerline_path)
    line = _single_reference_line(centerline)
    reference_nodes = gpd.read_file(directed_nodes_path)
    if reference_nodes.empty:
        raise ValueError(f"Reference-section directed nodes are empty: {directed_nodes_path}")
    if "is_outlet" in reference_nodes.columns and reference_nodes["is_outlet"].fillna(False).any():
        outlet_geometry = _point_like(reference_nodes.loc[reference_nodes["is_outlet"].fillna(False), reference_nodes.geometry.name].iloc[0])
    else:
        outlet_geometry = _point_like(reference_nodes.geometry.iloc[-1])
    if outlet_geometry is None:
        raise ValueError("Could not resolve an outlet geometry for the reference section.")
    outlet_proj = float(line.project(outlet_geometry))
    outlet_at_start = outlet_proj <= float(line.length - outlet_proj)
    sword_nodes_path = _resolve_reference_section_sword_nodes_path(
        summary,
        centerline_path=centerline_path,
    )
    reference_sword_nodes = pd.DataFrame()
    if sword_nodes_path is not None and sword_nodes_path.exists():
        sword_nodes = gpd.read_file(sword_nodes_path)
        if not sword_nodes.empty:
            if sword_nodes.crs is not None and centerline.crs is not None and sword_nodes.crs != centerline.crs:
                sword_nodes = sword_nodes.to_crs(centerline.crs)
            sword_dist_out: list[float] = []
            for geometry in sword_nodes.geometry.tolist():
                point = _point_like(geometry)
                if point is None:
                    sword_dist_out.append(float("nan"))
                    continue
                station = float(line.project(point))
                sword_dist_out.append(station if outlet_at_start else float(line.length - station))
            reference_sword_nodes = pd.DataFrame(sword_nodes.drop(columns=sword_nodes.geometry.name, errors="ignore"))
            reference_sword_nodes["reference_section_dist_out_m"] = pd.to_numeric(pd.Series(sword_dist_out), errors="coerce")

    experiment_dir = summary_path.parent.parent
    base_node_dist_out: dict[int, float] = {}
    registry_path = experiment_dir / "state_registry.csv"
    if registry_path.exists():
        registry = pd.read_csv(registry_path)
        if "state_role" in registry.columns and "directed_nodes_path" in registry.columns:
            base_rows = registry.loc[registry["state_role"].astype(str).eq("base")]
            if not base_rows.empty:
                base_nodes_path = Path(str(base_rows.iloc[0]["directed_nodes_path"])).expanduser().resolve()
                if base_nodes_path.exists():
                    base_nodes = gpd.read_file(base_nodes_path)
                    if base_nodes.crs is not None and centerline.crs is not None and base_nodes.crs != centerline.crs:
                        base_nodes = base_nodes.to_crs(centerline.crs)
                    for row in base_nodes.itertuples(index=False):
                        point = _point_like(row.geometry)
                        if point is None:
                            continue
                        station = float(line.project(point))
                        dist_out = station if outlet_at_start else float(line.length - station)
                        base_node_dist_out[int(row.id_node)] = dist_out

    return {
        "summary_path": str(summary_path),
        "centerline_path": str(centerline_path),
        "centerline_crs": centerline.crs.to_string() if centerline.crs is not None else "",
        "line": line,
        "line_length_m": float(line.length),
        "outlet_at_start": bool(outlet_at_start),
        "base_node_dist_out": base_node_dist_out,
        "reference_sword_nodes_path": str(sword_nodes_path) if sword_nodes_path is not None else "",
        "reference_sword_nodes": reference_sword_nodes,
    }


def _project_nodes_to_reference_section(
    nodes: gpd.GeoDataFrame,
    *,
    support: dict[str, Any],
    config: SlopeConfig,
) -> tuple[pd.Series, pd.Series]:
    node_dist_out = pd.Series(np.nan, index=nodes.index, dtype=float)
    node_source = pd.Series("", index=nodes.index, dtype="object")
    line: LineString = support["line"]
    outlet_at_start = bool(support["outlet_at_start"])
    centerline_crs = support.get("centerline_crs", "") or None
    base_node_dist_out: dict[int, float] = support.get("base_node_dist_out", {}) or {}

    if bool(config.reference_section_use_parent_match) and base_node_dist_out:
        if "matched_parent_node_id" in nodes.columns:
            matched_parent = pd.to_numeric(nodes["matched_parent_node_id"], errors="coerce")
            match_valid = nodes["match_within_tolerance"].fillna(False).astype(bool) if "match_within_tolerance" in nodes.columns else matched_parent.notna()
            for index, parent_value, is_valid in zip(nodes.index, matched_parent.tolist(), match_valid.tolist()):
                if not is_valid or pd.isna(parent_value):
                    continue
                parent_id = int(parent_value)
                if parent_id not in base_node_dist_out:
                    continue
                node_dist_out.at[index] = float(base_node_dist_out[parent_id])
                node_source.at[index] = "matched_parent_reference_section"

    nodes_for_projection = nodes
    if centerline_crs and nodes.crs is not None and str(nodes.crs) != str(centerline_crs):
        nodes_for_projection = nodes.to_crs(centerline_crs)

    for index, geometry in zip(nodes_for_projection.index, nodes_for_projection.geometry.tolist()):
        if pd.notna(node_dist_out.at[index]):
            continue
        point = _point_like(geometry)
        if point is None:
            continue
        station = float(line.project(point))
        node_dist_out.at[index] = station if outlet_at_start else float(line.length - station)
        node_source.at[index] = "geometry_projection_reference_section"

    unresolved = node_source.eq("")
    node_source.loc[unresolved] = "missing_reference_section_projection"
    return node_dist_out.astype(float), node_source


def _requested_node_wse_field(node_attrs: pd.DataFrame) -> str:
    requested_fields = [
        str(value).strip()
        for value in node_attrs.loc[
            node_attrs["node_wse_fill_method"].eq("requested_field"),
            "node_wse_field",
        ].dropna().astype(str).tolist()
        if str(value).strip()
    ]
    return requested_fields[0] if requested_fields else ""


@lru_cache(maxsize=32)
def _load_reference_section_profile(
    summary_path_str: str,
    requested_wse_field: str,
    min_anchor_nodes: int,
) -> dict[str, Any]:
    support = _load_reference_section_support(summary_path_str)
    reference_sword_nodes = support.get("reference_sword_nodes")
    if not isinstance(reference_sword_nodes, pd.DataFrame) or reference_sword_nodes.empty:
        return {}

    candidate_columns = [
        requested_wse_field,
        "sword_wse",
        "wse_obs_p50",
        "wse",
    ]
    wse_column = next(
        (
            column
            for column in candidate_columns
            if column
            and column in reference_sword_nodes.columns
            and pd.to_numeric(reference_sword_nodes[column], errors="coerce").notna().any()
        ),
        "",
    )
    if not wse_column:
        return {}

    fill_method_column = next(
        (
            column
            for column in ("sword_wse_fill_method", "wse_fill_method")
            if column in reference_sword_nodes.columns
        ),
        "",
    )

    anchors = pd.DataFrame()
    profile_method = ""
    if fill_method_column:
        requested_mask = reference_sword_nodes[fill_method_column].astype(str).eq("requested_field")
        anchors = reference_sword_nodes.loc[
            requested_mask,
            ["reference_section_dist_out_m", wse_column],
        ].copy()
        profile_method = "reference_section_monotone_requested_field_sword_nodes"

    anchors["reference_section_dist_out_m"] = pd.to_numeric(
        anchors.get("reference_section_dist_out_m"),
        errors="coerce",
    )
    anchors[wse_column] = pd.to_numeric(anchors.get(wse_column), errors="coerce")
    anchors = anchors.dropna(subset=["reference_section_dist_out_m", wse_column]).copy()

    if len(anchors) < int(min_anchor_nodes):
        anchors = reference_sword_nodes.loc[
            pd.to_numeric(reference_sword_nodes.get("reference_section_dist_out_m"), errors="coerce").notna()
            & pd.to_numeric(reference_sword_nodes.get(wse_column), errors="coerce").notna(),
            ["reference_section_dist_out_m", wse_column],
        ].copy()
        anchors["reference_section_dist_out_m"] = pd.to_numeric(
            anchors["reference_section_dist_out_m"],
            errors="coerce",
        )
        anchors[wse_column] = pd.to_numeric(anchors[wse_column], errors="coerce")
        anchors = anchors.dropna(subset=["reference_section_dist_out_m", wse_column]).copy()
        profile_method = "reference_section_monotone_all_sword_nodes"

    if len(anchors) < int(min_anchor_nodes):
        return {}

    anchors = (
        anchors.groupby("reference_section_dist_out_m", as_index=False)[wse_column]
        .mean()
        .sort_values("reference_section_dist_out_m")
    )
    xp = anchors["reference_section_dist_out_m"].to_numpy(dtype=float)
    yp = anchors[wse_column].to_numpy(dtype=float)
    fitted = _fit_monotone_increasing_profile(xp, yp)
    return {
        "xp": xp.tolist(),
        "fitted": fitted.tolist(),
        "profile_method": profile_method,
        "anchor_count": int(len(xp)),
        "wse_column": wse_column,
    }


def _apply_reference_section_profile(
    node_attrs: pd.DataFrame,
    *,
    nodes: gpd.GeoDataFrame,
    config: SlopeConfig,
) -> pd.DataFrame:
    if not config.reference_section_summary_path:
        return node_attrs

    support = _load_reference_section_support(str(config.reference_section_summary_path))
    result = node_attrs.copy()
    result["rapid_wse_for_slope_network_dist"] = result["rapid_wse_for_slope"].astype(float)
    result["rapid_wse_source_method_network_dist"] = result["rapid_wse_source_method"].astype(str)
    result["rapid_wse_interpolated_network_dist"] = result["rapid_wse_interpolated"].astype(bool)
    result["rapid_wse_anchor_count_network_dist"] = pd.to_numeric(result["rapid_wse_anchor_count"], errors="coerce").fillna(0).astype(int)
    result["rapid_wse_profile_residual_network_dist"] = pd.to_numeric(result["rapid_wse_profile_residual"], errors="coerce")

    node_reference_dist_out, node_reference_source = _project_nodes_to_reference_section(
        nodes,
        support=support,
        config=config,
    )
    result["node_reference_section_dist_out_m"] = node_reference_dist_out
    result["node_reference_section_dist_source"] = node_reference_source

    coord = pd.to_numeric(result["node_reference_section_dist_out_m"], errors="coerce")
    requested_anchor_mask = pd.Series(False, index=result.index, dtype=bool)
    anchors = pd.DataFrame()
    profile_method = ""
    profile = _load_reference_section_profile(
        str(config.reference_section_summary_path),
        _requested_node_wse_field(result),
        int(config.min_anchor_nodes_for_profile),
    )
    if profile:
        xp = np.asarray(profile["xp"], dtype=float)
        fitted = np.asarray(profile["fitted"], dtype=float)
        profile_method = str(profile["profile_method"])
        target_mask = coord.notna()
        eval_coord = coord.loc[target_mask].to_numpy(dtype=float)
        profile_values = _interp_linear_extrapolation(xp, fitted, eval_coord)
        result.loc[target_mask, "rapid_wse_for_slope"] = profile_values
        result.loc[target_mask, "rapid_wse_source_method"] = profile_method
        result.loc[target_mask, "rapid_wse_interpolated"] = True
        result.loc[target_mask, "rapid_wse_anchor_count"] = int(profile["anchor_count"])
        result.loc[target_mask, "rapid_wse_profile_residual"] = (
            result.loc[target_mask, "node_wse_raw"].astype(float)
            - result.loc[target_mask, "rapid_wse_for_slope"].astype(float)
        )
        return result

    if anchors.empty:
        requested_anchor_mask = (
            result["node_wse_fill_method"].eq("requested_field")
            & result["node_wse_raw"].notna()
            & coord.notna()
        )
        anchors = result.loc[requested_anchor_mask, ["node_reference_section_dist_out_m", "node_wse_raw"]].copy()
        anchors = anchors.rename(
            columns={
                "node_reference_section_dist_out_m": "reference_section_dist_out_m",
                "node_wse_raw": "anchor_wse",
            }
        )
        profile_method = "reference_section_monotone_requested_field_nodes"
        if len(anchors) < int(config.min_anchor_nodes_for_profile):
            available_mask = result["node_wse_raw"].notna() & coord.notna()
            anchors = result.loc[available_mask, ["node_reference_section_dist_out_m", "node_wse_raw"]].copy()
            anchors = anchors.rename(
                columns={
                    "node_reference_section_dist_out_m": "reference_section_dist_out_m",
                    "node_wse_raw": "anchor_wse",
                }
            )
            profile_method = "reference_section_monotone_all_available_nodes"

    anchors["reference_section_dist_out_m"] = pd.to_numeric(anchors["reference_section_dist_out_m"], errors="coerce")
    anchors["anchor_wse"] = pd.to_numeric(anchors["anchor_wse"], errors="coerce")
    anchors = anchors.dropna(subset=["reference_section_dist_out_m", "anchor_wse"]).copy()
    if anchors.empty:
        return result

    anchors = (
        anchors.groupby("reference_section_dist_out_m", as_index=False)["anchor_wse"]
        .mean()
        .sort_values("reference_section_dist_out_m")
    )
    xp = anchors["reference_section_dist_out_m"].to_numpy(dtype=float)
    yp = anchors["anchor_wse"].to_numpy(dtype=float)
    if len(xp) < int(config.min_anchor_nodes_for_profile):
        return result

    fitted = _fit_monotone_increasing_profile(xp, yp)
    target_mask = coord.notna()
    eval_coord = coord.loc[target_mask].to_numpy(dtype=float)
    profile_values = _interp_linear_extrapolation(xp, fitted, eval_coord)
    result.loc[target_mask, "rapid_wse_for_slope"] = profile_values
    result.loc[target_mask, "rapid_wse_source_method"] = profile_method
    if "sword_nodes" in profile_method:
        result.loc[target_mask, "rapid_wse_interpolated"] = True
    else:
        result.loc[target_mask, "rapid_wse_interpolated"] = ~requested_anchor_mask.loc[target_mask]
    result.loc[target_mask, "rapid_wse_anchor_count"] = int(len(xp))
    result.loc[target_mask, "rapid_wse_profile_residual"] = (
        result.loc[target_mask, "node_wse_raw"].astype(float)
        - result.loc[target_mask, "rapid_wse_for_slope"].astype(float)
    )
    return result


def _load_corridor_profile_anchors(
    node_attrs: pd.DataFrame,
    *,
    config: SlopeConfig,
) -> tuple[pd.DataFrame, str] | None:
    source_col = "node_sword_source_file"
    reach_col = "node_sword_reach_id"
    region_col = "node_sword_region"
    if source_col not in node_attrs.columns or reach_col not in node_attrs.columns:
        return None

    source_paths = [
        str(value).strip()
        for value in node_attrs[source_col].dropna().astype(str).tolist()
        if str(value).strip()
    ]
    unique_source_paths = sorted(dict.fromkeys(source_paths))
    if len(unique_source_paths) != 1:
        return None

    reach_ids = [
        int(value)
        for value in pd.to_numeric(node_attrs[reach_col], errors="coerce").dropna().astype(int).tolist()
    ]
    unique_reach_ids = sorted(dict.fromkeys(reach_ids))
    if not unique_reach_ids:
        return None

    regions = None
    if region_col in node_attrs.columns:
        region_values = [
            str(value).strip().lower()
            for value in node_attrs[region_col].dropna().astype(str).tolist()
            if str(value).strip()
        ]
        unique_regions = sorted(dict.fromkeys(region_values))
        if unique_regions:
            regions = unique_regions

    requested_wse_field = None
    if "node_wse_field" in node_attrs.columns:
        requested_fields = [
            str(value).strip()
            for value in node_attrs.loc[
                node_attrs["node_wse_fill_method"].eq("requested_field"),
                "node_wse_field",
            ].dropna().astype(str).tolist()
            if str(value).strip()
        ]
        unique_fields = sorted(dict.fromkeys(requested_fields))
        if unique_fields:
            requested_wse_field = unique_fields[0]

    try:
        from network_variants.sword_matching import load_sword_node_reference
    except ModuleNotFoundError:
        return None

    sword_nodes = load_sword_node_reference(
        unique_source_paths[0],
        sword_wse_field=requested_wse_field,
        regions=regions,
        reach_ids=unique_reach_ids,
    )
    if sword_nodes.empty:
        return None

    source = sword_nodes.rename(
        columns={
            "sword_dist_out": "corridor_dist_out",
            "sword_wse": "corridor_wse_raw",
            "sword_wse_fill_method": "corridor_wse_fill_method",
        }
    ).copy()
    source["corridor_dist_out"] = pd.to_numeric(source["corridor_dist_out"], errors="coerce")
    source["corridor_wse_raw"] = pd.to_numeric(source["corridor_wse_raw"], errors="coerce")
    requested_anchor_mask = (
        source["corridor_wse_fill_method"].eq("requested_field")
        & source["corridor_wse_raw"].notna()
        & source["corridor_dist_out"].notna()
    )
    anchors = source.loc[requested_anchor_mask, ["corridor_dist_out", "corridor_wse_raw"]].copy()
    profile_method = "profile_monotone_corridor_requested_field_nodes"
    if len(anchors) < int(config.min_anchor_nodes_for_profile):
        anchors = source.loc[
            source["corridor_wse_raw"].notna() & source["corridor_dist_out"].notna(),
            ["corridor_dist_out", "corridor_wse_raw"],
        ].copy()
        profile_method = "profile_monotone_corridor_all_available_nodes"
    anchors = anchors.dropna(subset=["corridor_dist_out", "corridor_wse_raw"]).copy()
    if len(anchors) < int(config.min_anchor_nodes_for_profile):
        return None
    return anchors, profile_method


def _load_reference_section_corridor_sword_anchors(
    node_attrs: pd.DataFrame,
    *,
    support: dict[str, Any],
) -> pd.DataFrame:
    source_col = "node_sword_source_file"
    reach_col = "node_sword_reach_id"
    region_col = "node_sword_region"
    field_col = "node_wse_field"
    fill_col = "node_wse_fill_method"
    if source_col not in node_attrs.columns or reach_col not in node_attrs.columns:
        return pd.DataFrame()

    source_paths = [
        str(value).strip()
        for value in node_attrs[source_col].dropna().astype(str).tolist()
        if str(value).strip()
    ]
    unique_source_paths = sorted(dict.fromkeys(source_paths))
    unique_reach_ids = [
        int(value)
        for value in pd.to_numeric(node_attrs[reach_col], errors="coerce").dropna().astype(int).tolist()
    ]
    unique_reach_ids = sorted(dict.fromkeys(unique_reach_ids))
    if not unique_source_paths or not unique_reach_ids:
        return pd.DataFrame()

    requested_fields = [
        str(value).strip()
        for value in node_attrs.loc[
            node_attrs.get(fill_col, pd.Series("", index=node_attrs.index)).astype(str).eq("requested_field"),
            field_col,
        ].dropna().astype(str).tolist()
        if str(value).strip()
    ]
    requested_wse_field = requested_fields[0] if requested_fields else None
    regions = None
    if region_col in node_attrs.columns:
        region_values = [
            str(value).strip().lower()
            for value in node_attrs[region_col].dropna().astype(str).tolist()
            if str(value).strip()
        ]
        unique_regions = sorted(dict.fromkeys(region_values))
        if unique_regions:
            regions = unique_regions

    try:
        from network_variants.sword_matching import load_sword_node_reference
    except ModuleNotFoundError:
        return pd.DataFrame()

    loaded_frames: list[gpd.GeoDataFrame] = []
    for source_path in unique_source_paths:
        try:
            source = load_sword_node_reference(
                source_path,
                sword_wse_field=requested_wse_field,
                regions=regions,
                reach_ids=unique_reach_ids,
            )
        except Exception:
            continue
        if source.empty:
            continue
        loaded_frames.append(source)
    if not loaded_frames:
        return pd.DataFrame()

    anchors = pd.concat(loaded_frames, ignore_index=True)
    if {"sword_source_file", "sword_node_id"}.issubset(anchors.columns):
        anchors = anchors.drop_duplicates(subset=["sword_source_file", "sword_node_id"], keep="first").copy()
    elif "sword_node_id" in anchors.columns:
        anchors = anchors.drop_duplicates(subset=["sword_node_id"], keep="first").copy()
    if anchors.empty:
        return pd.DataFrame()

    centerline_crs = support.get("centerline_crs", "") or None
    if centerline_crs and getattr(anchors, "crs", None) is not None and str(anchors.crs) != str(centerline_crs):
        anchors = anchors.to_crs(centerline_crs)
    line: LineString = support["line"]
    outlet_at_start = bool(support["outlet_at_start"])
    dist_out: list[float] = []
    for geometry in anchors.geometry.tolist():
        point = _point_like(geometry)
        if point is None:
            dist_out.append(float("nan"))
            continue
        station = float(line.project(point))
        dist_out.append(station if outlet_at_start else float(line.length - station))
    anchors = anchors.copy()
    anchors["reference_section_dist_out_m"] = pd.to_numeric(pd.Series(dist_out), errors="coerce")
    return pd.DataFrame(anchors)


def _resolve_rapid_wse_for_slope(
    nodes: gpd.GeoDataFrame,
    *,
    config: SlopeConfig,
    network_dist_out: pd.DataFrame | None = None,
) -> pd.DataFrame:
    optional_columns = [
        "sword_wse_fallback_used",
        "sword_node_id",
        "sword_reach_id",
        "sword_region",
        "sword_dist_out",
        "sword_source_file",
    ]
    columns = [
        column
        for column in ["id_node", "sword_wse", "sword_wse_field", "sword_wse_fill_method"] + optional_columns
        if column in nodes.columns
    ]
    node_attrs = (
        nodes[columns]
        .copy()
        .rename(
            columns={
                "sword_wse": "node_wse_raw",
                "sword_wse_field": "node_wse_field",
                "sword_wse_fill_method": "node_wse_fill_method",
                "sword_wse_fallback_used": "node_wse_fallback_used",
                "sword_node_id": "node_sword_node_id",
                "sword_reach_id": "node_sword_reach_id",
                "sword_region": "node_sword_region",
                "sword_dist_out": "node_sword_dist_out",
                "sword_source_file": "node_sword_source_file",
            }
        )
    )
    node_attrs["id_node"] = pd.to_numeric(node_attrs["id_node"], errors="coerce").astype("Int64")
    node_attrs["node_wse_raw"] = pd.to_numeric(node_attrs["node_wse_raw"], errors="coerce")
    if "node_sword_dist_out" in node_attrs.columns:
        node_attrs["node_sword_dist_out"] = pd.to_numeric(node_attrs["node_sword_dist_out"], errors="coerce")

    if network_dist_out is not None and not network_dist_out.empty:
        node_attrs = node_attrs.merge(network_dist_out, on="id_node", how="left")
    else:
        node_attrs["node_network_dist_out_m"] = pd.to_numeric(node_attrs.get("node_sword_dist_out"), errors="coerce")
        node_attrs["node_network_dist_source"] = np.where(
            node_attrs["node_network_dist_out_m"].notna(),
            "sword_dist_out_fallback",
            "missing",
        )

    # Three distinct head concepts are intentionally kept separate:
    # - sword_wse / node_wse_raw: observed or matched node elevation context
    # - rapid_wse_for_slope: monotone/interpolated profile used only for slope derivation
    # - normalized solver heads for equivalent resistance are computed elsewhere
    node_attrs["rapid_wse_for_slope"] = node_attrs["node_wse_raw"]
    node_attrs["rapid_wse_source_method"] = node_attrs["node_wse_fill_method"].fillna("unresolved").astype("object")
    node_attrs["rapid_wse_interpolated"] = False
    node_attrs["rapid_wse_anchor_count"] = 0
    node_attrs["rapid_wse_profile_residual"] = np.nan

    profile_mode = _normalize_slope_profile_mode(config.slope_profile_mode)
    if not config.interpolate_along_corridor or "node_network_dist_out_m" not in node_attrs.columns:
        return node_attrs

    coord = pd.to_numeric(node_attrs["node_network_dist_out_m"], errors="coerce")
    requested_anchor_mask = pd.Series(False, index=node_attrs.index)
    corridor_anchors = _load_corridor_profile_anchors(node_attrs, config=config)
    profile_method = ""
    if corridor_anchors is not None:
        anchors, profile_method = corridor_anchors
        anchors = (
            anchors.groupby("corridor_dist_out", as_index=False)["corridor_wse_raw"]
            .mean()
            .sort_values("corridor_dist_out")
        )
        xp = anchors["corridor_dist_out"].to_numpy(dtype=float)
        yp = anchors["corridor_wse_raw"].to_numpy(dtype=float)
        state_coord_min = float(coord.min())
        state_coord_max = float(coord.max())
        if not np.isfinite(state_coord_min) or not np.isfinite(state_coord_max) or state_coord_max <= state_coord_min:
            corridor_anchors = None
    if corridor_anchors is None:
        requested_anchor_mask = (
            node_attrs["node_wse_fill_method"].eq("requested_field")
            & node_attrs["node_wse_raw"].notna()
            & coord.notna()
        )
        anchors = node_attrs.loc[
            requested_anchor_mask,
            ["id_node", "node_network_dist_out_m", "node_wse_raw"],
        ].copy()
        profile_method = "profile_monotone_requested_field_nodes"
        if len(anchors) < int(config.min_anchor_nodes_for_profile):
            available_mask = node_attrs["node_wse_raw"].notna() & coord.notna()
            anchors = node_attrs.loc[
                available_mask,
                ["id_node", "node_network_dist_out_m", "node_wse_raw"],
            ].copy()
            profile_method = "profile_monotone_all_available_nodes"

        anchors["node_network_dist_out_m"] = pd.to_numeric(anchors["node_network_dist_out_m"], errors="coerce")
        anchors["node_wse_raw"] = pd.to_numeric(anchors["node_wse_raw"], errors="coerce")
        anchors = anchors.dropna(subset=["node_network_dist_out_m", "node_wse_raw"]).copy()
        if anchors.empty:
            return node_attrs

        anchors = (
            anchors.groupby("node_network_dist_out_m", as_index=False)["node_wse_raw"]
            .mean()
            .sort_values("node_network_dist_out_m")
        )
        xp = anchors["node_network_dist_out_m"].to_numpy(dtype=float)
        yp = anchors["node_wse_raw"].to_numpy(dtype=float)
    node_attrs["rapid_wse_anchor_count"] = int(len(xp))
    if len(xp) < int(config.min_anchor_nodes_for_profile):
        return node_attrs

    fitted = _fit_monotone_increasing_profile(xp, yp)
    target_mask = coord.notna()
    eval_coord = coord.loc[target_mask].to_numpy(dtype=float)
    if corridor_anchors is not None:
        sword_coord_min = float(np.min(xp))
        sword_coord_max = float(np.max(xp))
        eval_coord = sword_coord_min + (
            (eval_coord - state_coord_min) / (state_coord_max - state_coord_min)
        ) * (sword_coord_max - sword_coord_min)
    profile_values = _interp_linear_extrapolation(
        xp,
        fitted,
        eval_coord,
    )
    node_attrs.loc[target_mask, "rapid_wse_for_slope"] = profile_values
    node_attrs.loc[target_mask, "rapid_wse_source_method"] = profile_method
    node_attrs.loc[target_mask, "rapid_wse_interpolated"] = ~requested_anchor_mask.loc[target_mask]
    node_attrs.loc[target_mask, "rapid_wse_profile_residual"] = (
        node_attrs.loc[target_mask, "node_wse_raw"].astype(float)
        - node_attrs.loc[target_mask, "rapid_wse_for_slope"].astype(float)
    )

    if profile_mode != "reference_section":
        return node_attrs

    try:
        return _apply_reference_section_profile(
            node_attrs,
            nodes=nodes,
            config=config,
        )
    except Exception:
        if not bool(config.reference_section_fallback_to_network_dist):
            raise
        return node_attrs


def _fill_invalid_slopes_from_neighbors(
    frame: pd.DataFrame,
    links: gpd.GeoDataFrame,
    *,
    config: SlopeConfig,
    invalid_mask: pd.Series,
    slope_column: str = "raw_slope",
) -> pd.DataFrame:
    frame = frame.copy()
    frame["slope_neighbor_source_link_id"] = pd.NA
    frame["slope_neighbor_distance"] = pd.NA
    frame["slope_neighbor_direction"] = ""
    frame["slope_neighbor_value"] = np.nan
    if not config.fill_from_nearest_valid_link or not invalid_mask.any():
        return frame

    link_ids = pd.to_numeric(frame["id_link"], errors="coerce").astype(int)
    us_nodes = pd.to_numeric(frame["id_us_node"], errors="coerce").astype(int)
    ds_nodes = pd.to_numeric(frame["id_ds_node"], errors="coerce").astype(int)
    frame_index = frame.index

    outgoing_by_node: dict[int, list[int]] = {}
    incoming_by_node: dict[int, list[int]] = {}
    for link_id, us_node, ds_node in zip(link_ids, us_nodes, ds_nodes):
        outgoing_by_node.setdefault(int(us_node), []).append(int(link_id))
        incoming_by_node.setdefault(int(ds_node), []).append(int(link_id))

    successors: dict[int, list[int]] = {}
    predecessors: dict[int, list[int]] = {}
    for link_id, us_node, ds_node in zip(link_ids, us_nodes, ds_nodes):
        successors[int(link_id)] = [candidate for candidate in outgoing_by_node.get(int(ds_node), []) if candidate != int(link_id)]
        predecessors[int(link_id)] = [candidate for candidate in incoming_by_node.get(int(us_node), []) if candidate != int(link_id)]

    row_index_by_link = dict(zip(link_ids, frame_index))
    slope_by_link = dict(zip(link_ids, pd.to_numeric(frame[slope_column], errors="coerce")))
    valid_links = {
        int(link_id)
        for link_id, raw_slope in slope_by_link.items()
        if pd.notna(raw_slope) and not bool(invalid_mask.loc[row_index_by_link[int(link_id)]])
    }

    centroid_by_link: dict[int, object] = {}
    if links.geometry.name in links.columns and links.geometry.notna().any():
        centroids = links.geometry.centroid
        centroid_by_link = dict(zip(pd.to_numeric(links["id_link"], errors="coerce").astype(int), centroids))

    def _nearest_valid(start_link: int, direction: str) -> tuple[int | None, int | None]:
        graph = successors if direction == "downstream" else predecessors
        visited = {int(start_link)}
        frontier = [int(start_link)]
        distance = 0
        while frontier:
            distance += 1
            next_frontier: list[int] = []
            candidates: list[int] = []
            for current in frontier:
                for neighbor in graph.get(int(current), []):
                    if neighbor in visited:
                        continue
                    visited.add(neighbor)
                    if neighbor in valid_links:
                        candidates.append(neighbor)
                    next_frontier.append(neighbor)
            if candidates:
                return distance, min(candidates)
            frontier = next_frontier
        return None, None

    def _choose_candidate(
        start_link: int,
        upstream_result: tuple[int | None, int | None],
        downstream_result: tuple[int | None, int | None],
    ) -> tuple[int | None, int | None, str]:
        candidate_rows: list[tuple[int, int, str]] = []
        up_distance, up_link = upstream_result
        if up_distance is not None and up_link is not None:
            candidate_rows.append((int(up_distance), int(up_link), "upstream"))
        down_distance, down_link = downstream_result
        if down_distance is not None and down_link is not None:
            candidate_rows.append((int(down_distance), int(down_link), "downstream"))
        if not candidate_rows:
            return None, None, ""
        min_distance = min(row[0] for row in candidate_rows)
        candidate_rows = [row for row in candidate_rows if row[0] == min_distance]
        if len(candidate_rows) == 1:
            distance, link_id, direction = candidate_rows[0]
            return link_id, distance, direction

        start_centroid = centroid_by_link.get(int(start_link))
        if start_centroid is not None:
            candidate_rows = sorted(
                candidate_rows,
                key=lambda row: (
                    float(start_centroid.distance(centroid_by_link.get(int(row[1]), start_centroid))),
                    row[1],
                ),
            )
        else:
            candidate_rows = sorted(candidate_rows, key=lambda row: row[1])
        distance, link_id, direction = candidate_rows[0]
        return link_id, distance, direction

    for link_id in link_ids[invalid_mask.to_numpy()]:
        upstream_result = _nearest_valid(int(link_id), "upstream")
        downstream_result = _nearest_valid(int(link_id), "downstream")
        neighbor_link, neighbor_distance, neighbor_direction = _choose_candidate(
            int(link_id),
            upstream_result,
            downstream_result,
        )
        if neighbor_link is None:
            continue
        neighbor_row_index = row_index_by_link[int(neighbor_link)]
        row_index = row_index_by_link[int(link_id)]
        frame.at[row_index, "slope_neighbor_source_link_id"] = int(neighbor_link)
        frame.at[row_index, "slope_neighbor_distance"] = int(neighbor_distance)
        frame.at[row_index, "slope_neighbor_direction"] = neighbor_direction
        frame.at[row_index, "slope_neighbor_value"] = float(frame.at[neighbor_row_index, slope_column])
    return frame


def _compute_section_slope_reference(
    node_attrs: pd.DataFrame,
    *,
    config: SlopeConfig,
) -> tuple[float, str, int]:
    coordinate_column = _preferred_profile_coordinate_column(node_attrs)
    if coordinate_column is None:
        return float("nan"), "missing_dist_out", 0

    anchors = node_attrs.loc[
        node_attrs["rapid_wse_for_slope"].notna()
        & pd.to_numeric(node_attrs[coordinate_column], errors="coerce").notna()
    ].copy()
    if anchors.empty:
        return float("nan"), "missing_anchor_wse", 0

    anchors = (
        anchors.groupby(coordinate_column, as_index=False)["rapid_wse_for_slope"]
        .mean()
        .sort_values(coordinate_column)
    )
    if len(anchors) < int(config.min_anchor_nodes_for_profile):
        return float("nan"), "insufficient_anchor_nodes", int(len(anchors))

    downstream = anchors.iloc[0]
    upstream = anchors.iloc[-1]
    delta_dist = float(upstream[coordinate_column] - downstream[coordinate_column])
    delta_wse = float(upstream["rapid_wse_for_slope"] - downstream["rapid_wse_for_slope"])
    if delta_dist <= 0 or delta_wse <= 0:
        return float("nan"), "invalid_anchor_profile", int(len(anchors))

    if coordinate_column == "node_reference_section_dist_out_m":
        source_method = "profile_reference_section_dist_out"
    elif coordinate_column == "node_network_dist_out_m":
        source_method = "profile_network_dist_out"
    else:
        source_method = "profile_sword_dist_out"
    return float(delta_wse / delta_dist), source_method, int(len(anchors))


def _initialize_compound_bubble_qc_columns(
    frame: pd.DataFrame,
    *,
    default_reason: str,
) -> pd.DataFrame:
    result = frame.copy()
    for column, default in COMPOUND_BUBBLE_QC_COLUMNS.items():
        if isinstance(default, pd.Series):
            result[column] = default.reindex(result.index)
        else:
            result[column] = default
    result["bubble_slope_qc_reason"] = str(default_reason)
    return result


def _parse_unit_node_ids(value: object) -> set[int]:
    if value is None or pd.isna(value):
        return set()
    text = str(value).strip()
    if not text:
        return set()
    node_ids: set[int] = set()
    for token in text.split(","):
        token = token.strip()
        if not token:
            continue
        try:
            node_ids.add(int(token))
        except ValueError:
            continue
    return node_ids


def _build_bubble_linear_reference_profile(
    *,
    internal_node_ids: set[int],
    node_lookup: pd.DataFrame,
    entry_node: int,
    exit_node: int,
    entry_wse: float,
    exit_wse: float,
    entry_dist: float,
    exit_dist: float,
    coordinate_column: str,
) -> tuple[dict[int, float] | None, str]:
    reference_drop = float(entry_wse - exit_wse)
    reference_dist_span = float(entry_dist - exit_dist)
    if reference_drop <= 0 or reference_dist_span <= 0:
        return None, ""

    node_profile: dict[int, float] = {}
    for node_id in sorted(internal_node_ids):
        if node_id not in node_lookup.index:
            return None, ""
        node_dist = float(node_lookup.loc[node_id].get(coordinate_column, np.nan))
        if not np.isfinite(node_dist):
            return None, ""
        fraction = (node_dist - exit_dist) / reference_dist_span
        node_profile[int(node_id)] = float(exit_wse + fraction * reference_drop)

    node_profile[int(entry_node)] = float(entry_wse)
    node_profile[int(exit_node)] = float(exit_wse)
    if coordinate_column == "node_reference_section_dist_out_m":
        return node_profile, "bubble_local_linear_reference_section_dist_out"
    if coordinate_column == "node_sword_dist_out":
        return node_profile, "bubble_local_linear_sword_dist_out"
    return node_profile, "bubble_local_linear_network_dist_out"


def _build_bubble_reference_slopes(
    *,
    bubble_links: pd.DataFrame,
    node_profile: dict[int, float],
) -> tuple[dict[int, float], bool]:
    ref_slopes: dict[int, float] = {}
    for row in bubble_links.itertuples(index=False):
        numerator = float(node_profile[int(row.id_us_node)] - node_profile[int(row.id_ds_node)])
        denominator = float(row.link_length_m)
        ref_slope = numerator / denominator if denominator > 0 else float("nan")
        if not np.isfinite(ref_slope) or ref_slope <= 0:
            return {}, False
        ref_slopes[int(row.id_link)] = float(ref_slope)
    return ref_slopes, True


def _project_bubble_link_drops(
    *,
    bubble_links: pd.DataFrame,
    path_link_sequences: list[list[int]],
    reference_drop: float,
    min_slope: float,
) -> tuple[dict[int, float] | None, dict[int, float] | None]:
    if not path_link_sequences:
        return None, None

    link_ids = [int(value) for value in bubble_links["id_link"].tolist()]
    link_lengths = {
        int(row.id_link): float(row.link_length_m)
        for row in bubble_links.itertuples(index=False)
    }
    prior_slopes = {
        int(row.id_link): float(row.slope_used)
        for row in bubble_links.itertuples(index=False)
    }
    min_drops = {
        link_id: float(min_slope) * max(float(link_lengths[link_id]), 1.0)
        for link_id in link_ids
    }
    prior_drops = {
        link_id: max(float(prior_slopes[link_id]) * float(link_lengths[link_id]), float(min_drops[link_id]))
        for link_id in link_ids
    }

    A = np.zeros((len(path_link_sequences), len(link_ids)), dtype=float)
    link_index = {link_id: idx for idx, link_id in enumerate(link_ids)}
    for row_index, path in enumerate(path_link_sequences):
        for link_id in path:
            A[row_index, link_index[int(link_id)]] = 1.0
    b = np.full(len(path_link_sequences), float(reference_drop), dtype=float)
    p = np.asarray([prior_drops[link_id] for link_id in link_ids], dtype=float)
    w_inv = np.diag(np.maximum(p, 1e-9) ** 2)
    residual = A @ p - b
    try:
        lagrange = np.linalg.pinv(A @ w_inv @ A.T) @ residual
    except np.linalg.LinAlgError:
        return None, None
    projected = p - w_inv @ A.T @ lagrange
    if not np.isfinite(projected).all():
        return None, None
    if np.any(projected <= np.asarray([min_drops[link_id] for link_id in link_ids], dtype=float)):
        return None, None

    drop_map = {link_id: float(projected[link_index[link_id]]) for link_id in link_ids}
    slope_map = {
        link_id: float(drop_map[link_id] / float(link_lengths[link_id]))
        for link_id in link_ids
    }
    if not all(np.isfinite(value) and value > 0 for value in slope_map.values()):
        return None, None
    return drop_map, slope_map


def _apply_compound_bubble_qc(
    frame: pd.DataFrame,
    *,
    node_attrs: pd.DataFrame,
    unit_summary: pd.DataFrame | None,
    config: SlopeConfig,
) -> pd.DataFrame:
    if not bool(config.use_compound_bubble_slope_qc):
        return _initialize_compound_bubble_qc_columns(frame, default_reason="disabled")
    if unit_summary is None or unit_summary.empty:
        return _initialize_compound_bubble_qc_columns(frame, default_reason="unavailable_unit_summary")

    required_columns = {"compound_bubble_id", "in_compound_bubble", "unit_node_ids"}
    if not required_columns.issubset(unit_summary.columns):
        return _initialize_compound_bubble_qc_columns(frame, default_reason="unit_summary_missing_columns")

    result = _initialize_compound_bubble_qc_columns(frame, default_reason="not_in_compound_bubble")
    result["id_link"] = pd.to_numeric(result["id_link"], errors="coerce").astype(int)
    result["id_us_node"] = pd.to_numeric(result["id_us_node"], errors="coerce").astype(int)
    result["id_ds_node"] = pd.to_numeric(result["id_ds_node"], errors="coerce").astype(int)
    link_index_by_id = dict(zip(result["id_link"], result.index))

    node_lookup = node_attrs.copy()
    node_lookup["id_node"] = pd.to_numeric(node_lookup["id_node"], errors="coerce").astype("Int64")
    node_lookup = node_lookup.dropna(subset=["id_node"]).copy()
    node_lookup["id_node"] = node_lookup["id_node"].astype(int)
    node_lookup = node_lookup.set_index("id_node", drop=False)
    profile_coordinate_column = _preferred_profile_coordinate_column(node_lookup.reset_index(drop=True))

    working = unit_summary.copy()
    working["compound_bubble_id"] = pd.to_numeric(working["compound_bubble_id"], errors="coerce").astype("Int64")
    working["in_compound_bubble"] = working["in_compound_bubble"].fillna(False).astype(bool)
    working = working.loc[working["compound_bubble_id"].notna() & working["in_compound_bubble"]].copy()
    if working.empty:
        return result

    bubble_nodes: dict[int, set[int]] = {}
    bubble_candidate_indices: dict[int, list[int]] = {}
    link_to_candidate_bubbles: dict[int, set[int]] = {}
    for bubble_id, group in working.groupby("compound_bubble_id", sort=True):
        bubble_numeric = int(bubble_id)
        node_ids: set[int] = set()
        for value in group["unit_node_ids"]:
            node_ids.update(_parse_unit_node_ids(value))
        if not node_ids:
            continue
        bubble_nodes[bubble_numeric] = node_ids
        candidate_mask = result["id_us_node"].isin(node_ids) & result["id_ds_node"].isin(node_ids)
        candidate_indices = result.index[candidate_mask].tolist()
        bubble_candidate_indices[bubble_numeric] = candidate_indices
        for index in candidate_indices:
            link_to_candidate_bubbles.setdefault(int(index), set()).add(bubble_numeric)

    for bubble_numeric, node_ids in bubble_nodes.items():
        candidate_indices = bubble_candidate_indices.get(bubble_numeric, [])
        if not candidate_indices:
            continue

        result.loc[candidate_indices, "bubble_slope_qc_candidate"] = True
        ambiguous_indices = [index for index in candidate_indices if len(link_to_candidate_bubbles.get(int(index), set())) > 1]
        unique_candidate_indices = [index for index in candidate_indices if index not in ambiguous_indices]
        if unique_candidate_indices:
            result.loc[unique_candidate_indices, "compound_bubble_id_link"] = bubble_numeric
            result.loc[unique_candidate_indices, "bubble_n_internal_links"] = int(len(unique_candidate_indices))
        if ambiguous_indices:
            result.loc[ambiguous_indices, "bubble_slope_qc_reason"] = "ambiguous_link_assignment"
            result.loc[ambiguous_indices, "bubble_n_internal_links"] = int(len(candidate_indices))
        if not unique_candidate_indices:
            continue
        if ambiguous_indices:
            result.loc[unique_candidate_indices, "bubble_slope_qc_reason"] = "ambiguous_link_assignment"
            continue

        bubble_links = result.loc[unique_candidate_indices].copy()
        internal_node_ids = set(bubble_links["id_us_node"].tolist()) | set(bubble_links["id_ds_node"].tolist())
        incoming_boundary_nodes = sorted(
            {
                int(row.id_ds_node)
                for row in result.itertuples(index=False)
                if int(row.id_ds_node) in internal_node_ids and int(row.id_us_node) not in internal_node_ids
            }
        )
        outgoing_boundary_nodes = sorted(
            {
                int(row.id_us_node)
                for row in result.itertuples(index=False)
                if int(row.id_us_node) in internal_node_ids and int(row.id_ds_node) not in internal_node_ids
            }
        )
        if len(incoming_boundary_nodes) != 1:
            reason = "multiple_entry_nodes" if len(incoming_boundary_nodes) > 1 else "missing_entry_node"
            result.loc[unique_candidate_indices, "bubble_slope_qc_reason"] = reason
            continue
        if len(outgoing_boundary_nodes) != 1:
            reason = "multiple_exit_nodes" if len(outgoing_boundary_nodes) > 1 else "missing_exit_node"
            result.loc[unique_candidate_indices, "bubble_slope_qc_reason"] = reason
            continue

        entry_node = int(incoming_boundary_nodes[0])
        exit_node = int(outgoing_boundary_nodes[0])
        result.loc[unique_candidate_indices, "bubble_entry_node_id"] = entry_node
        result.loc[unique_candidate_indices, "bubble_exit_node_id"] = exit_node

        internal_graph = nx.MultiDiGraph()
        internal_graph.add_nodes_from(sorted(internal_node_ids))
        for row in bubble_links.itertuples(index=False):
            internal_graph.add_edge(int(row.id_us_node), int(row.id_ds_node), key=int(row.id_link), id_link=int(row.id_link))
        if not nx.is_directed_acyclic_graph(internal_graph):
            result.loc[unique_candidate_indices, "bubble_slope_qc_reason"] = "cycle_detected"
            continue
        if not nx.has_path(internal_graph, entry_node, exit_node):
            result.loc[unique_candidate_indices, "bubble_slope_qc_reason"] = "no_entry_exit_path"
            continue

        if entry_node not in node_lookup.index or exit_node not in node_lookup.index:
            result.loc[unique_candidate_indices, "bubble_slope_qc_reason"] = "missing_reference_nodes"
            continue
        entry_meta = node_lookup.loc[entry_node]
        exit_meta = node_lookup.loc[exit_node]
        entry_wse = float(entry_meta.get("rapid_wse_for_slope", np.nan))
        exit_wse = float(exit_meta.get("rapid_wse_for_slope", np.nan))
        entry_dist = float(entry_meta.get(profile_coordinate_column, np.nan)) if profile_coordinate_column is not None else float("nan")
        exit_dist = float(exit_meta.get(profile_coordinate_column, np.nan)) if profile_coordinate_column is not None else float("nan")
        if not all(np.isfinite([entry_wse, exit_wse, entry_dist, exit_dist])):
            result.loc[unique_candidate_indices, "bubble_slope_qc_reason"] = "missing_reference_profile"
            continue
        reference_drop = float(entry_wse - exit_wse)
        reference_dist_span = float(entry_dist - exit_dist)
        if reference_drop <= 0:
            result.loc[unique_candidate_indices, "bubble_slope_qc_reason"] = "nonpositive_reference_drop"
            continue
        if reference_dist_span <= 0:
            result.loc[unique_candidate_indices, "bubble_slope_qc_reason"] = "nonpositive_reference_dist_span"
            continue

        path_link_sequences: list[list[int]] = []
        max_paths = int(config.compound_bubble_max_paths_for_qc)
        path_limit_exceeded = False
        for path_index, edge_path in enumerate(nx.all_simple_edge_paths(internal_graph, entry_node, exit_node), start=1):
            if path_index > max_paths:
                path_limit_exceeded = True
                break
            path_link_sequences.append([int(link_key) for _, _, link_key in edge_path])
        if path_limit_exceeded:
            result.loc[unique_candidate_indices, "bubble_slope_qc_reason"] = "path_limit_exceeded"
            result.loc[unique_candidate_indices, "bubble_n_paths_checked"] = int(max_paths)
            continue
        if not path_link_sequences:
            result.loc[unique_candidate_indices, "bubble_slope_qc_reason"] = "no_entry_exit_path"
            continue

        profile_node_ids = {
            int(node_id)
            for node_id in internal_node_ids
            if int(node_id) in node_lookup.index and np.isfinite(float(node_lookup.loc[int(node_id)].get("rapid_wse_for_slope", np.nan)))
        }
        if not internal_node_ids.issubset(profile_node_ids):
            result.loc[unique_candidate_indices, "bubble_slope_qc_reason"] = "missing_internal_dist_out"
            continue
        node_profile = {
            int(node_id): float(node_lookup.loc[int(node_id), "rapid_wse_for_slope"])
            for node_id in sorted(internal_node_ids)
        }
        ref_slopes, valid_reference_slopes = _build_bubble_reference_slopes(
            bubble_links=bubble_links,
            node_profile=node_profile,
        )
        if profile_coordinate_column == "node_reference_section_dist_out_m":
            reference_profile_method = "state_profile_reference_section_dist_out"
        elif profile_coordinate_column == "node_sword_dist_out":
            reference_profile_method = "state_profile_sword_dist_out"
        else:
            reference_profile_method = "state_profile_network_dist_out"
        repair_candidate = False
        repair_method = ""
        if not valid_reference_slopes:
            repair_candidate = True
            projected_drops, projected_slopes = _project_bubble_link_drops(
                bubble_links=bubble_links,
                path_link_sequences=path_link_sequences,
                reference_drop=reference_drop,
                min_slope=float(config.min_slope),
            )
            if projected_slopes is not None and projected_drops is not None:
                ref_slopes = projected_slopes
                valid_reference_slopes = True
                reference_profile_method = "bubble_weighted_path_drop_projection"
                repair_method = "bubble_weighted_path_drop_projection"
            else:
                linear_profile, linear_method = _build_bubble_linear_reference_profile(
                    internal_node_ids=internal_node_ids,
                    node_lookup=node_lookup,
                    entry_node=entry_node,
                    exit_node=exit_node,
                    entry_wse=entry_wse,
                    exit_wse=exit_wse,
                    entry_dist=entry_dist,
                    exit_dist=exit_dist,
                    coordinate_column=str(profile_coordinate_column or "node_network_dist_out_m"),
                )
                if linear_profile is None:
                    result.loc[unique_candidate_indices, "bubble_slope_qc_reason"] = "invalid_reference_profile"
                    result.loc[unique_candidate_indices, "bubble_slope_repair_candidate"] = True
                    continue
                ref_slopes, valid_reference_slopes = _build_bubble_reference_slopes(
                    bubble_links=bubble_links,
                    node_profile=linear_profile,
                )
                if not valid_reference_slopes:
                    result.loc[unique_candidate_indices, "bubble_slope_qc_reason"] = "invalid_reference_profile"
                    result.loc[unique_candidate_indices, "bubble_slope_repair_candidate"] = True
                    continue
                node_profile = linear_profile
                reference_profile_method = linear_method
                repair_method = linear_method

        result.loc[unique_candidate_indices, "bubble_slope_repair_candidate"] = repair_candidate
        if repair_candidate and bool(config.use_compound_bubble_slope_repair):
            result.loc[unique_candidate_indices, "bubble_slope_repair_applied"] = True
            result.loc[unique_candidate_indices, "bubble_slope_repair_method"] = repair_method
            for row_index, row in bubble_links.iterrows():
                prior_slope = float(row["slope_used"])
                repaired_slope = float(ref_slopes[int(row["id_link"])])
                result.at[int(row_index), "slope_used"] = repaired_slope
                result.at[int(row_index), "slope_source_method"] = "compound_bubble_profile"
                result.at[int(row_index), "slope_adjusted"] = True
                result.at[int(row_index), "slope_reason"] = "filled_from_compound_bubble_profile"
                result.at[int(row_index), "slope_minimum_applied"] = False
                if np.isfinite(prior_slope) and prior_slope > 0:
                    factor = float(repaired_slope / prior_slope)
                    result.at[int(row_index), "bubble_slope_repair_change_factor"] = factor
                    result.at[int(row_index), "bubble_slope_repair_large_change_flag"] = bool(
                        factor >= float(config.compound_bubble_slope_repair_flag_factor)
                    )

        path_drops: list[float] = []
        for path in path_link_sequences:
            drop = 0.0
            for link_key in path:
                row_index = link_index_by_id.get(int(link_key))
                if row_index is None:
                    drop = float("nan")
                    break
                slope_value = float(result.at[row_index, "slope_used"])
                length_value = float(result.at[row_index, "link_length_m"])
                drop += slope_value * length_value
            path_drops.append(float(drop))

        path_drop_array = np.asarray(path_drops, dtype=float)
        if not np.isfinite(path_drop_array).all():
            result.loc[unique_candidate_indices, "bubble_slope_qc_reason"] = "nonfinite_path_drop"
            continue
        abs_errors = np.abs(path_drop_array - reference_drop)
        rel_errors = abs_errors / reference_drop
        max_abs_error = float(abs_errors.max()) if len(abs_errors) else float("nan")
        max_rel_error = float(rel_errors.max()) if len(rel_errors) else float("nan")
        qc_failed = bool(
            max_abs_error > float(config.compound_bubble_path_drop_abs_tol_m)
            or max_rel_error > float(config.compound_bubble_path_drop_rel_tol)
        )

        result.loc[unique_candidate_indices, "bubble_slope_qc_valid"] = True
        result.loc[unique_candidate_indices, "bubble_slope_qc_reason"] = "path_drop_mismatch" if qc_failed else "ok"
        result.loc[unique_candidate_indices, "bubble_reference_drop_m"] = reference_drop
        result.loc[unique_candidate_indices, "bubble_reference_dist_span_m"] = reference_dist_span
        result.loc[unique_candidate_indices, "bubble_reference_slope"] = float(reference_drop / reference_dist_span)
        result.loc[unique_candidate_indices, "bubble_reference_profile_method"] = reference_profile_method
        result.loc[unique_candidate_indices, "bubble_n_paths_checked"] = int(len(path_drop_array))
        result.loc[unique_candidate_indices, "bubble_path_drop_min_m"] = float(path_drop_array.min())
        result.loc[unique_candidate_indices, "bubble_path_drop_max_m"] = float(path_drop_array.max())
        result.loc[unique_candidate_indices, "bubble_path_drop_spread_m"] = float(path_drop_array.max() - path_drop_array.min())
        result.loc[unique_candidate_indices, "bubble_path_drop_max_abs_error_m"] = max_abs_error
        result.loc[unique_candidate_indices, "bubble_path_drop_max_rel_error"] = max_rel_error
        result.loc[unique_candidate_indices, "bubble_path_drop_qc_failed"] = qc_failed
        for row_index, row in bubble_links.iterrows():
            result.at[int(row_index), "bubble_link_slope_ref"] = float(ref_slopes[int(row["id_link"])])

    return result


def compute_hydraulic_context(
    links: gpd.GeoDataFrame,
    nodes: gpd.GeoDataFrame,
    *,
    unit_summary: pd.DataFrame | None = None,
    config: SlopeConfig | None = None,
) -> HydraulicContextResult:
    """Compute node profile context and link slope context for a realized network state."""
    config = config or SlopeConfig()
    profile_mode = _normalize_slope_profile_mode(config.slope_profile_mode)
    if config.max_slope_for_k is not None and config.max_slope_for_k <= 0:
        raise ValueError("Slope outlier upper bound must be positive when provided.")
    if config.section_slope_ratio_min is not None and config.section_slope_ratio_min <= 0:
        raise ValueError("Section-slope ratio minimum must be positive when provided.")
    if config.section_slope_ratio_max is not None and config.section_slope_ratio_max <= 0:
        raise ValueError("Section-slope ratio maximum must be positive when provided.")
    if config.compound_bubble_max_paths_for_qc <= 0:
        raise ValueError("Compound-bubble max paths for QC must be positive.")
    if config.compound_bubble_path_drop_rel_tol < 0:
        raise ValueError("Compound-bubble relative path-drop tolerance cannot be negative.")
    if config.compound_bubble_path_drop_abs_tol_m < 0:
        raise ValueError("Compound-bubble absolute path-drop tolerance cannot be negative.")
    if (
        config.section_slope_ratio_min is not None
        and config.section_slope_ratio_max is not None
        and config.section_slope_ratio_min > config.section_slope_ratio_max
    ):
        raise ValueError("Section-slope ratio minimum cannot exceed the maximum.")
    if profile_mode == "reference_section" and not config.reference_section_summary_path and not bool(config.reference_section_fallback_to_network_dist):
        raise ValueError(
            "slope_profile_mode='reference_section' requires reference_section_summary_path "
            "unless reference_section_fallback_to_network_dist is enabled."
        )
    missing = sorted(REQUIRED_NODE_COLUMNS.difference(nodes.columns))
    if missing:
        raise ValueError(
            "Directed nodes are missing required SWORD WSE columns: "
            + ", ".join(missing)
        )
    for column in ("id_link", "id_us_node", "id_ds_node"):
        if column not in links.columns:
            raise ValueError(f"Directed links are missing required column '{column}'.")

    lengths, length_source = _select_variant_lengths(
        links,
        preferred_length_field=config.preferred_length_field,
    )

    network_dist_out = _compute_network_dist_out(links, nodes, lengths=lengths)
    node_attrs = _resolve_rapid_wse_for_slope(
        nodes,
        config=config,
        network_dist_out=network_dist_out,
    )
    section_slope_ref, section_slope_source_method, section_slope_anchor_count = _compute_section_slope_reference(
        node_attrs,
        config=config,
    )

    frame = links[["id_link", "id_us_node", "id_ds_node"]].copy()
    frame["link_length_m"] = lengths
    frame["link_length_source"] = length_source
    frame["id_us_node"] = pd.to_numeric(frame["id_us_node"], errors="coerce").astype("Int64")
    frame["id_ds_node"] = pd.to_numeric(frame["id_ds_node"], errors="coerce").astype("Int64")

    us = node_attrs.rename(
        columns={
            "id_node": "id_us_node",
            "node_wse_raw": "wse_us_raw",
            "rapid_wse_for_slope": "wse_us",
            "node_wse_field": "wse_us_field",
            "node_wse_fill_method": "wse_us_fill_method",
            "node_wse_fallback_used": "wse_us_fallback_used",
            "node_sword_node_id": "sword_us_node_id",
            "node_sword_reach_id": "sword_us_reach_id",
            "node_sword_dist_out": "sword_us_dist_out",
            "node_network_dist_out_m": "network_us_dist_out_m",
            "node_network_dist_source": "network_us_dist_source",
            "rapid_wse_source_method": "wse_us_source_method",
            "rapid_wse_interpolated": "wse_us_interpolated",
            "rapid_wse_anchor_count": "wse_us_anchor_count",
            "rapid_wse_profile_residual": "wse_us_profile_residual",
        }
    )
    ds = node_attrs.rename(
        columns={
            "id_node": "id_ds_node",
            "node_wse_raw": "wse_ds_raw",
            "rapid_wse_for_slope": "wse_ds",
            "node_wse_field": "wse_ds_field",
            "node_wse_fill_method": "wse_ds_fill_method",
            "node_wse_fallback_used": "wse_ds_fallback_used",
            "node_sword_node_id": "sword_ds_node_id",
            "node_sword_reach_id": "sword_ds_reach_id",
            "node_sword_dist_out": "sword_ds_dist_out",
            "node_network_dist_out_m": "network_ds_dist_out_m",
            "node_network_dist_source": "network_ds_dist_source",
            "rapid_wse_source_method": "wse_ds_source_method",
            "rapid_wse_interpolated": "wse_ds_interpolated",
            "rapid_wse_anchor_count": "wse_ds_anchor_count",
            "rapid_wse_profile_residual": "wse_ds_profile_residual",
        }
    )

    frame = frame.merge(us, on="id_us_node", how="left").merge(ds, on="id_ds_node", how="left")
    frame["raw_slope"] = (
        pd.to_numeric(frame["wse_us"], errors="coerce")
        - pd.to_numeric(frame["wse_ds"], errors="coerce")
    ) / frame["link_length_m"]
    frame["slope_local_raw"] = pd.to_numeric(frame["raw_slope"], errors="coerce")
    frame["slope_section_ref"] = float(section_slope_ref)
    frame["slope_section_source_method"] = section_slope_source_method
    frame["slope_section_anchor_count"] = int(section_slope_anchor_count)

    missing_wse = frame["wse_us"].isna() | frame["wse_ds"].isna()
    invalid_raw = frame["slope_local_raw"].isna() | frame["slope_local_raw"].lt(float(config.min_slope))
    above_max = (
        frame["slope_local_raw"].gt(float(config.max_slope_for_k))
        if config.max_slope_for_k is not None
        else pd.Series(False, index=frame.index)
    )
    section_slope_available = np.isfinite(section_slope_ref) and section_slope_ref >= float(config.min_slope)
    slope_ratio = (
        frame["slope_local_raw"] / float(section_slope_ref)
        if section_slope_available
        else pd.Series(np.nan, index=frame.index)
    )
    below_ratio = (
        slope_ratio.lt(float(config.section_slope_ratio_min))
        if section_slope_available and config.section_slope_ratio_min is not None
        else pd.Series(False, index=frame.index)
    )
    above_ratio = (
        slope_ratio.gt(float(config.section_slope_ratio_max))
        if section_slope_available and config.section_slope_ratio_max is not None
        else pd.Series(False, index=frame.index)
    )

    frame["slope_outlier_reason"] = np.select(
        [
            missing_wse,
            invalid_raw,
            above_max,
            below_ratio,
            above_ratio,
        ],
        [
            "missing_wse",
            "below_minimum_raw",
            "above_maximum_raw",
            "below_section_ratio",
            "above_section_ratio",
        ],
        default="ok",
    )
    frame["slope_outlier_flag"] = frame["slope_outlier_reason"].ne("ok")
    frame["slope_section_ratio"] = slope_ratio.astype(float)

    frame = _fill_invalid_slopes_from_neighbors(
        frame,
        links,
        config=config,
        invalid_mask=frame["slope_outlier_flag"],
        slope_column="slope_local_raw",
    )

    use_neighbor = frame["slope_outlier_flag"] & frame["slope_neighbor_value"].notna()
    use_section = (
        frame["slope_outlier_flag"]
        & ~use_neighbor
        & bool(config.use_section_slope_fallback)
        & section_slope_available
    )
    use_minimum = frame["slope_outlier_flag"] & ~use_neighbor & ~use_section

    frame["slope_source_method"] = "raw_wse"
    frame["slope_used"] = frame["slope_local_raw"].astype(float)
    frame.loc[use_neighbor, "slope_source_method"] = "nearest_valid_link"
    frame.loc[use_neighbor, "slope_used"] = frame.loc[use_neighbor, "slope_neighbor_value"].astype(float)
    frame.loc[use_section, "slope_source_method"] = "section_slope"
    frame.loc[use_section, "slope_used"] = float(section_slope_ref)
    frame.loc[use_minimum, "slope_source_method"] = "minimum_slope"
    frame.loc[use_minimum, "slope_used"] = float(config.min_slope)

    frame["slope_adjusted"] = frame["slope_source_method"].ne("raw_wse")
    frame["slope_reason"] = np.select(
        [
            use_neighbor,
            use_section,
            use_minimum,
        ],
        [
            "filled_from_neighbor_link",
            "filled_from_section_slope",
            frame["slope_outlier_reason"],
        ],
        default="ok",
    )
    frame["slope_minimum_applied"] = frame["slope_used"].eq(config.min_slope)
    frame = _apply_compound_bubble_qc(
        frame,
        node_attrs=node_attrs,
        unit_summary=unit_summary,
        config=config,
    )

    section_slope_summary = {
        "slope": float(section_slope_ref),
        "source_method": str(section_slope_source_method),
        "anchor_count": int(section_slope_anchor_count),
        "profile_mode": profile_mode,
        "reference_section_summary_path": str(config.reference_section_summary_path or ""),
    }
    return HydraulicContextResult(
        node_context=node_attrs,
        link_context=frame,
        section_slope_summary=section_slope_summary,
    )


def compute_link_slopes(
    links: gpd.GeoDataFrame,
    nodes: gpd.GeoDataFrame,
    *,
    unit_summary: pd.DataFrame | None = None,
    config: SlopeConfig | None = None,
) -> pd.DataFrame:
    """Backward-compatible wrapper that returns only the link-level hydraulic context."""
    return compute_hydraulic_context(
        links,
        nodes,
        unit_summary=unit_summary,
        config=config,
    ).link_context


def compute_section_slope_reference(
    nodes: gpd.GeoDataFrame,
    *,
    config: SlopeConfig | None = None,
) -> dict[str, object]:
    config = config or SlopeConfig()
    missing = sorted(REQUIRED_NODE_COLUMNS.difference(nodes.columns))
    if missing:
        raise ValueError(
            "Directed nodes are missing required SWORD WSE columns: "
            + ", ".join(missing)
        )
    node_attrs = _resolve_rapid_wse_for_slope(nodes, config=config, network_dist_out=None)
    slope_ref, source_method, anchor_count = _compute_section_slope_reference(
        node_attrs,
        config=config,
    )
    return {
        "slope": float(slope_ref),
        "source_method": str(source_method),
        "anchor_count": int(anchor_count),
    }


def generate_state_hydraulic_context(
    *,
    directed_links_path: str | Path,
    directed_nodes_path: str | Path,
    hierarchy_output_dir: str | Path | None,
    output_dir: str | Path,
    state_id: str | None = None,
    config: SlopeConfig | None = None,
) -> StateHydraulicContextOutputs:
    """Write state-level node and link hydraulic context artifacts.

    The generated artifacts preserve the distinction between observed/matched
    node WSE context and the monotone profile WSE used for slope derivation.
    Normalized internal heads for equivalent-resistance solves are not part of
    these artifacts and should be written separately by the solver workflow.
    """
    links_path = Path(directed_links_path).expanduser().resolve()
    nodes_path = Path(directed_nodes_path).expanduser().resolve()
    output_path = Path(output_dir).expanduser().resolve()
    output_path.mkdir(parents=True, exist_ok=True)
    resolved_config = config or SlopeConfig()
    if (
        _normalize_slope_profile_mode(resolved_config.slope_profile_mode) == "reference_section"
        and not resolved_config.reference_section_summary_path
    ):
        inferred = _infer_reference_section_summary_path(output_path, links_path, nodes_path)
        if inferred is not None:
            resolved_config = replace(resolved_config, reference_section_summary_path=str(inferred))

    links = gpd.read_file(links_path)
    nodes = gpd.read_file(nodes_path)
    unit_summary = _load_unit_summary_if_exists(hierarchy_output_dir)
    result = compute_hydraulic_context(
        links,
        nodes,
        unit_summary=unit_summary,
        config=resolved_config,
    )
    link_context, family_slope_inheritance_summary = _apply_family_representative_slope_inheritance(
        result.link_context,
        links,
    )

    link_context_path = output_path / "link_hydraulic_context.csv"
    node_context_path = output_path / "node_hydraulic_context.csv"
    manifest_path = output_path / HYDRAULIC_CONTEXT_MANIFEST

    link_context.to_csv(link_context_path, index=False)
    result.node_context.to_csv(node_context_path, index=False)

    manifest: dict[str, object] = {
        "state_id": state_id or "",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "input_paths": {
            "directed_links": str(links_path),
            "directed_nodes": str(nodes_path),
            "hierarchy_output_dir": str(Path(hierarchy_output_dir).expanduser().resolve()) if hierarchy_output_dir is not None else "",
        },
        "output_paths": {
            "link_hydraulic_context": str(link_context_path),
            "node_hydraulic_context": str(node_context_path),
        },
        "n_links": int(len(result.link_context)),
        "n_nodes": int(len(result.node_context)),
        "section_slope_summary": result.section_slope_summary,
        "head_concepts": {
            "observed_or_matched_node_wse": "sword_wse / node_wse_raw",
            "derived_profile_wse_for_slope": "rapid_wse_for_slope",
            "normalized_solver_heads": "not written here",
        },
        "family_slope_inheritance": family_slope_inheritance_summary,
        "slope_config": {
            "slope_profile_mode": _normalize_slope_profile_mode(resolved_config.slope_profile_mode),
            "reference_section_summary_path": str(resolved_config.reference_section_summary_path or ""),
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return StateHydraulicContextOutputs(
        output_dir=output_path,
        link_context_path=link_context_path,
        node_context_path=node_context_path,
        manifest_path=manifest_path,
        link_context=link_context,
        node_context=result.node_context,
        manifest=manifest,
    )


__all__ = [
    "HYDRAULIC_CONTEXT_MANIFEST",
    "HydraulicContextResult",
    "SlopeConfig",
    "StateHydraulicContextOutputs",
    "compute_hydraulic_context",
    "compute_link_slopes",
    "compute_section_slope_reference",
    "generate_state_hydraulic_context",
]
