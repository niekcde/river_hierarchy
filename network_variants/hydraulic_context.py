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
from scipy.optimize import minimize
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

COLLAPSED_PARENT_PROFILE_COLUMNS: dict[str, object] = {
    "collapsed_profile_repair_candidate": False,
    "collapsed_profile_repair_applied": False,
    "collapsed_profile_repair_reason": "",
    "collapsed_parent_us_node": pd.Series(dtype="Int64"),
    "collapsed_parent_ds_node": pd.Series(dtype="Int64"),
    "collapsed_reference_drop_m": np.nan,
    "collapsed_reference_slope": np.nan,
    "collapsed_profile_change_factor": np.nan,
}


def _safe_float(value: object) -> float | None:
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(numeric):
        return None
    return float(numeric)


def _load_unit_summary_if_exists(hierarchy_output_dir: str | Path | None) -> pd.DataFrame | None:
    if hierarchy_output_dir is None:
        return None
    path = Path(hierarchy_output_dir)
    unit_summary_path = path / "unit_summary.csv"
    if not unit_summary_path.exists():
        return None
    return pd.read_csv(unit_summary_path)


def _infer_state_dir_from_links_path(links_path: Path) -> Path | None:
    resolved = Path(links_path).expanduser().resolve()
    for parent in resolved.parents:
        if parent.parent.name == "states":
            return parent
    return None


def _parse_int_list(value: object) -> list[int]:
    if value is None or pd.isna(value):
        return []
    text = str(value).strip()
    if not text:
        return []
    return [int(float(part.strip())) for part in text.split(",") if part.strip()]


def _compute_base_node_profile_lookup(
    *,
    state_dir: Path,
    config: SlopeConfig,
) -> pd.DataFrame | None:
    experiment_dir = state_dir.parent.parent
    base_state_dir = experiment_dir / "states" / "S000_base"
    base_node_context_path = base_state_dir / "hydraulic_context" / "node_hydraulic_context.csv"
    if base_node_context_path.exists():
        base_node_context = pd.read_csv(base_node_context_path)
        if {"id_node", "rapid_wse_for_slope"}.issubset(base_node_context.columns):
            base_node_context["id_node"] = pd.to_numeric(base_node_context["id_node"], errors="coerce").astype("Int64")
            base_node_context["rapid_wse_for_slope"] = pd.to_numeric(
                base_node_context["rapid_wse_for_slope"], errors="coerce"
            )
            return base_node_context[["id_node", "rapid_wse_for_slope"]].dropna(subset=["id_node"])

    directed_dir = base_state_dir / "variant" / "directed"
    base_links_candidates = sorted(directed_dir.glob("*_directed_links.gpkg"))
    base_nodes_candidates = sorted(directed_dir.glob("*_directed_nodes.gpkg"))
    if not base_links_candidates or not base_nodes_candidates:
        return None

    base_links = gpd.read_file(base_links_candidates[0])
    base_nodes = gpd.read_file(base_nodes_candidates[0])
    base_unit_summary = _load_unit_summary_if_exists(base_state_dir / "hierarchy")
    base_result = compute_hydraulic_context(
        base_links,
        base_nodes,
        unit_summary=base_unit_summary,
        config=config,
    )
    lookup = base_result.node_context[["id_node", "rapid_wse_for_slope"]].copy()
    lookup["id_node"] = pd.to_numeric(lookup["id_node"], errors="coerce").astype("Int64")
    lookup["rapid_wse_for_slope"] = pd.to_numeric(lookup["rapid_wse_for_slope"], errors="coerce")
    return lookup.dropna(subset=["id_node"])


def _solve_base_component_heads(
    base_component_links: pd.DataFrame,
    *,
    base_node_context: pd.DataFrame,
) -> tuple[dict[int, float], int | None]:
    if base_component_links.empty:
        return {}, None

    node_ids = sorted(
        set(pd.to_numeric(base_component_links["id_us_node"], errors="coerce").dropna().astype(int))
        | set(pd.to_numeric(base_component_links["id_ds_node"], errors="coerce").dropna().astype(int))
    )
    if not node_ids:
        return {}, None

    base_node_lookup = base_node_context.copy()
    base_node_lookup["id_node"] = pd.to_numeric(base_node_lookup["id_node"], errors="coerce").astype("Int64")
    base_node_lookup["rapid_wse_for_slope"] = pd.to_numeric(
        base_node_lookup["rapid_wse_for_slope"], errors="coerce"
    )
    base_node_lookup = base_node_lookup.dropna(subset=["id_node"]).drop_duplicates(subset=["id_node"]).set_index("id_node")

    anchor_candidates = [
        (int(node_id), float(base_node_lookup.loc[int(node_id), "rapid_wse_for_slope"]))
        for node_id in node_ids
        if int(node_id) in base_node_lookup.index
    ]
    if not anchor_candidates:
        return {}, None

    anchor_node, anchor_head = min(anchor_candidates, key=lambda item: item[1])
    node_index = {node_id: index for index, node_id in enumerate(node_ids)}
    equations: list[np.ndarray] = []
    rhs: list[float] = []
    for row in base_component_links.itertuples(index=False):
        us_node = int(row.id_us_node)
        ds_node = int(row.id_ds_node)
        drop = float(row.link_length_m) * float(row.slope_used)
        coeff = np.zeros(len(node_ids), dtype=float)
        coeff[node_index[us_node]] = 1.0
        coeff[node_index[ds_node]] = -1.0
        equations.append(coeff)
        rhs.append(drop)

    anchor_coeff = np.zeros(len(node_ids), dtype=float)
    anchor_coeff[node_index[anchor_node]] = 1.0
    equations.append(anchor_coeff)
    rhs.append(anchor_head)

    matrix = np.vstack(equations)
    target = np.asarray(rhs, dtype=float)
    solution, *_ = np.linalg.lstsq(matrix, target, rcond=None)
    return {node_id: float(solution[node_index[node_id]]) for node_id in node_ids}, anchor_node


def _build_collapsed_unit_boundary_analysis(
    link_context: pd.DataFrame,
    *,
    links_path: Path,
) -> list[dict[str, object]]:
    state_dir = _infer_state_dir_from_links_path(links_path)
    if state_dir is None or state_dir.name == "S000_base":
        return []

    collapse_components_path = state_dir / "variant" / "summary" / "collapse_components.csv"
    node_match_path = state_dir / "variant" / "matching" / "node_match.csv"
    experiment_dir = state_dir.parent.parent
    base_state_dir = experiment_dir / "states" / "S000_base"
    base_link_context_path = base_state_dir / "hydraulic_context" / "link_hydraulic_context.csv"
    base_node_context_path = base_state_dir / "hydraulic_context" / "node_hydraulic_context.csv"
    if not (
        collapse_components_path.exists()
        and node_match_path.exists()
        and base_link_context_path.exists()
        and base_node_context_path.exists()
    ):
        return []

    collapse_components = pd.read_csv(collapse_components_path)
    node_match = pd.read_csv(node_match_path)
    base_link_context = pd.read_csv(base_link_context_path)
    base_node_context = pd.read_csv(base_node_context_path)

    if not {"child_id_node", "matched_parent_node_id"}.issubset(node_match.columns):
        return []

    node_match["child_id_node"] = pd.to_numeric(node_match["child_id_node"], errors="coerce").astype("Int64")
    node_match["matched_parent_node_id"] = pd.to_numeric(node_match["matched_parent_node_id"], errors="coerce").astype("Int64")
    if "match_within_tolerance" in node_match.columns:
        node_match = node_match.loc[node_match["match_within_tolerance"].fillna(False).astype(bool)].copy()

    variant_graph = nx.DiGraph()
    for row in link_context.itertuples(index=False):
        if pd.isna(getattr(row, "id_us_node", np.nan)) or pd.isna(getattr(row, "id_ds_node", np.nan)):
            continue
        variant_graph.add_edge(
            int(row.id_us_node),
            int(row.id_ds_node),
            id_link=int(row.id_link),
            link_length_m=float(row.link_length_m),
            slope_used=float(row.slope_used),
        )

    analyses: list[dict[str, object]] = []
    for row in collapse_components.itertuples(index=False):
        base_unit_nodes = sorted(set(_parse_int_list(getattr(row, "node_ids", ""))))
        base_unit_links = sorted(set(_parse_int_list(getattr(row, "link_ids", ""))))
        if not base_unit_nodes or not base_unit_links:
            continue

        matched_rows = node_match.loc[node_match["matched_parent_node_id"].isin(base_unit_nodes)].copy()
        if matched_rows.empty:
            continue

        variant_boundary_pairs = sorted(
            {
                (int(child_id), int(parent_id))
                for child_id, parent_id in zip(
                    matched_rows["child_id_node"].tolist(),
                    matched_rows["matched_parent_node_id"].tolist(),
                    strict=False,
                )
                if pd.notna(child_id) and pd.notna(parent_id)
            }
        )
        if len(variant_boundary_pairs) < 2:
            continue

        variant_boundary_nodes = sorted({child_id for child_id, _ in variant_boundary_pairs})
        variant_path_records: list[dict[str, object]] = []
        variant_local_link_ids: set[int] = set()
        boundary_predecessors = {node_id: set() for node_id in variant_boundary_nodes}
        boundary_successors = {node_id: set() for node_id in variant_boundary_nodes}
        cutoff = max(2, min(len(variant_graph.nodes), 40))
        for source in variant_boundary_nodes:
            for target in variant_boundary_nodes:
                if source == target or source not in variant_graph or target not in variant_graph:
                    continue
                if not nx.has_path(variant_graph, source, target):
                    continue
                for node_path in nx.all_simple_paths(variant_graph, source, target, cutoff=cutoff):
                    link_ids = [
                        int(variant_graph[us_node][ds_node]["id_link"])
                        for us_node, ds_node in zip(node_path[:-1], node_path[1:], strict=False)
                    ]
                    variant_local_link_ids.update(link_ids)
                    boundary_predecessors[target].add(source)
                    boundary_successors[source].add(target)
                    variant_path_records.append(
                        {
                            "source_boundary_node": int(source),
                            "target_boundary_node": int(target),
                            "node_path": [int(node_id) for node_id in node_path],
                            "link_path": link_ids,
                        }
                    )

        if not variant_path_records:
            continue

        variant_entry_nodes = sorted(
            node_id for node_id in variant_boundary_nodes if not boundary_predecessors[node_id]
        )
        variant_exit_nodes = sorted(
            node_id for node_id in variant_boundary_nodes if not boundary_successors[node_id]
        )

        base_component_links = base_link_context.loc[
            pd.to_numeric(base_link_context["id_link"], errors="coerce").isin(base_unit_links)
        ].copy()
        if base_component_links.empty:
            continue
        for column in ("id_link", "id_us_node", "id_ds_node"):
            base_component_links[column] = pd.to_numeric(base_component_links[column], errors="coerce").astype("Int64")
        for column in ("link_length_m", "slope_used"):
            base_component_links[column] = pd.to_numeric(base_component_links[column], errors="coerce")
        base_component_links = base_component_links.dropna(
            subset=["id_link", "id_us_node", "id_ds_node", "link_length_m", "slope_used"]
        ).copy()
        if base_component_links.empty:
            continue

        base_graph = nx.DiGraph()
        for base_link in base_component_links.itertuples(index=False):
            drop = float(base_link.link_length_m) * float(base_link.slope_used)
            base_graph.add_edge(
                int(base_link.id_us_node),
                int(base_link.id_ds_node),
                id_link=int(base_link.id_link),
                link_length_m=float(base_link.link_length_m),
                slope_used=float(base_link.slope_used),
                drop_m=drop,
            )

        base_path_drop_records: list[dict[str, object]] = []
        base_drop_consistency: dict[str, dict[str, float | int | bool | None]] = {}
        for source in base_unit_nodes:
            for target in base_unit_nodes:
                if source == target or source not in base_graph or target not in base_graph:
                    continue
                if not nx.has_path(base_graph, source, target):
                    continue
                pair_key = f"{source}->{target}"
                pair_drops: list[float] = []
                for node_path in nx.all_simple_paths(base_graph, source, target, cutoff=max(2, len(base_unit_nodes))):
                    link_ids = [
                        int(base_graph[us_node][ds_node]["id_link"])
                        for us_node, ds_node in zip(node_path[:-1], node_path[1:], strict=False)
                    ]
                    drop_m = float(
                        sum(base_graph[us_node][ds_node]["drop_m"] for us_node, ds_node in zip(node_path[:-1], node_path[1:], strict=False))
                    )
                    pair_drops.append(drop_m)
                    base_path_drop_records.append(
                        {
                            "source_base_node": int(source),
                            "target_base_node": int(target),
                            "node_path": [int(node_id) for node_id in node_path],
                            "link_path": link_ids,
                            "drop_m": drop_m,
                        }
                    )
                if pair_drops:
                    drop_min = float(min(pair_drops))
                    drop_max = float(max(pair_drops))
                    base_drop_consistency[pair_key] = {
                        "n_paths": int(len(pair_drops)),
                        "drop_min_m": drop_min,
                        "drop_max_m": drop_max,
                        "drop_mean_m": float(np.mean(pair_drops)),
                        "drop_spread_m": float(drop_max - drop_min),
                        "is_consistent": bool(np.isclose(drop_min, drop_max, rtol=1e-6, atol=1e-9)),
                    }

        solved_base_heads, anchor_node = _solve_base_component_heads(
            base_component_links,
            base_node_context=base_node_context,
        )
        variant_boundary_heads = [
            {
                "variant_node": int(child_id),
                "base_node": int(parent_id),
                "head_m": solved_base_heads.get(int(parent_id)),
            }
            for child_id, parent_id in variant_boundary_pairs
        ]

        analyses.append(
            {
                "component_id": str(getattr(row, "component_id", "")),
                "collapsed_selection_label": str(getattr(row, "collapsed_selection_label", "")),
                "base_unit_nodes": [int(node_id) for node_id in base_unit_nodes],
                "base_unit_links": [int(link_id) for link_id in base_unit_links],
                "variant_boundary_node_matches": [
                    {"variant_node": int(child_id), "base_node": int(parent_id)}
                    for child_id, parent_id in variant_boundary_pairs
                ],
                "variant_boundary_nodes": [int(node_id) for node_id in variant_boundary_nodes],
                "variant_entry_nodes": [int(node_id) for node_id in variant_entry_nodes],
                "variant_exit_nodes": [int(node_id) for node_id in variant_exit_nodes],
                "variant_local_link_ids": sorted(int(link_id) for link_id in variant_local_link_ids),
                "variant_paths": variant_path_records,
                "base_path_drop_consistency": base_drop_consistency,
                "base_head_anchor_node": int(anchor_node) if anchor_node is not None else None,
                "base_boundary_heads": [
                    {"base_node": int(node_id), "head_m": float(head_value)}
                    for node_id, head_value in sorted(solved_base_heads.items())
                    if node_id in base_unit_nodes
                ],
                "variant_boundary_heads": variant_boundary_heads,
            }
        )

    return analyses


def _apply_collapsed_parent_profile_slope_repair(
    link_context: pd.DataFrame,
    *,
    links_path: Path,
    config: SlopeConfig,
) -> pd.DataFrame:
    result = link_context.copy()
    for column, default in COLLAPSED_PARENT_PROFILE_COLUMNS.items():
        if column not in result.columns:
            result[column] = default

    state_dir = _infer_state_dir_from_links_path(links_path)
    if state_dir is None or state_dir.name == "S000_base":
        result["collapsed_profile_repair_reason"] = np.where(
            result["collapsed_profile_repair_reason"].eq(""),
            "base_or_unknown_state",
            result["collapsed_profile_repair_reason"],
        )
        return result

    lineage_path = state_dir / "variant" / "matching" / "link_lineage.csv"
    if not lineage_path.exists():
        result["collapsed_profile_repair_reason"] = np.where(
            result["collapsed_profile_repair_reason"].eq(""),
            "missing_lineage",
            result["collapsed_profile_repair_reason"],
        )
        return result

    lineage = pd.read_csv(lineage_path)
    required = {"id_link", "lineage_type", "matched_parent_us_node", "matched_parent_ds_node", "matched_parent_link_ids"}
    if not required.issubset(lineage.columns):
        result["collapsed_profile_repair_reason"] = np.where(
            result["collapsed_profile_repair_reason"].eq(""),
            "missing_lineage_columns",
            result["collapsed_profile_repair_reason"],
        )
        return result

    lineage = lineage[list(required)].copy()
    lineage["id_link"] = pd.to_numeric(lineage["id_link"], errors="coerce").astype("Int64")
    for column in ("matched_parent_us_node", "matched_parent_ds_node"):
        lineage[column] = pd.to_numeric(lineage[column], errors="coerce").astype("Int64")
    lineage["_matched_parent_link_ids"] = lineage["matched_parent_link_ids"].map(_parse_int_list)

    collapsed_component_links: set[int] = set()
    collapse_components_path = state_dir / "variant" / "summary" / "collapse_components.csv"
    if collapse_components_path.exists():
        collapse_components = pd.read_csv(collapse_components_path)
        if "link_ids" in collapse_components.columns:
            for value in collapse_components["link_ids"].tolist():
                collapsed_component_links.update(_parse_int_list(value))

    result = result.merge(lineage, on="id_link", how="left", suffixes=("", "_lineage"))
    result["_collapsed_parent_link_overlap"] = result["_matched_parent_link_ids"].map(
        lambda ids: bool(set(ids).intersection(collapsed_component_links))
    )
    collapsed_mask = (
        result["lineage_type"].astype(str).eq("collapsed_many_to_one")
        & result["_collapsed_parent_link_overlap"]
        & result["matched_parent_us_node"].notna()
        & result["matched_parent_ds_node"].notna()
        & pd.to_numeric(result["link_length_m"], errors="coerce").gt(0)
    )
    result.loc[collapsed_mask, "collapsed_profile_repair_candidate"] = True
    if not collapsed_mask.any():
        result.loc[result["collapsed_profile_repair_reason"].eq(""), "collapsed_profile_repair_reason"] = "no_collapsed_candidates"
        return result

    base_lookup = _compute_base_node_profile_lookup(state_dir=state_dir, config=config)
    if base_lookup is None or base_lookup.empty:
        result.loc[collapsed_mask, "collapsed_profile_repair_reason"] = "missing_base_profile"
        return result

    base_lookup = base_lookup.drop_duplicates(subset=["id_node"]).set_index("id_node")
    for row_index, row in result.loc[collapsed_mask].iterrows():
        us_node = int(row["matched_parent_us_node"])
        ds_node = int(row["matched_parent_ds_node"])
        result.at[row_index, "collapsed_parent_us_node"] = us_node
        result.at[row_index, "collapsed_parent_ds_node"] = ds_node
        if us_node not in base_lookup.index or ds_node not in base_lookup.index:
            result.at[row_index, "collapsed_profile_repair_reason"] = "missing_base_parent_nodes"
            continue

        us_wse = float(base_lookup.loc[us_node, "rapid_wse_for_slope"])
        ds_wse = float(base_lookup.loc[ds_node, "rapid_wse_for_slope"])
        link_length = float(row["link_length_m"])
        reference_drop = float(us_wse - ds_wse)
        if not np.isfinite(reference_drop):
            result.at[row_index, "collapsed_profile_repair_reason"] = "nonfinite_reference_drop"
            continue
        if reference_drop <= 0:
            result.at[row_index, "collapsed_profile_repair_reason"] = "nonpositive_reference_drop"
            continue
        if not np.isfinite(link_length) or link_length <= 0:
            result.at[row_index, "collapsed_profile_repair_reason"] = "invalid_link_length"
            continue

        repaired_slope = max(reference_drop / link_length, float(config.min_slope))
        prior_slope = float(row["slope_used"]) if pd.notna(row["slope_used"]) else float("nan")
        result.at[row_index, "collapsed_reference_drop_m"] = reference_drop
        result.at[row_index, "collapsed_reference_slope"] = repaired_slope
        result.at[row_index, "slope_used"] = repaired_slope
        result.at[row_index, "slope_source_method"] = "collapsed_parent_profile"
        result.at[row_index, "slope_adjusted"] = True
        result.at[row_index, "slope_reason"] = "filled_from_collapsed_parent_profile"
        result.at[row_index, "slope_minimum_applied"] = bool(repaired_slope == float(config.min_slope))
        result.at[row_index, "collapsed_profile_repair_applied"] = True
        result.at[row_index, "collapsed_profile_repair_reason"] = "ok"
        if np.isfinite(prior_slope) and prior_slope > 0:
            result.at[row_index, "collapsed_profile_change_factor"] = float(repaired_slope / prior_slope)

    return result


def _apply_collapsed_unit_head_solver(
    link_context: pd.DataFrame,
    *,
    collapsed_unit_boundary_analysis: list[dict[str, object]],
    min_slope: float = 1e-6,
) -> tuple[pd.DataFrame, dict[str, object]]:
    result = link_context.copy()
    for column, default in COLLAPSED_PARENT_PROFILE_COLUMNS.items():
        if column not in result.columns:
            result[column] = default

    if not collapsed_unit_boundary_analysis:
        result.loc[result["collapsed_profile_repair_reason"].eq(""), "collapsed_profile_repair_reason"] = "no_unit_boundary_analysis"
        return result, {"candidate_link_ids": [], "applied_link_ids": [], "solved_component_ids": []}

    index_by_link_id = {
        int(link_id): int(row_index)
        for row_index, link_id in result["id_link"].items()
        if pd.notna(link_id)
    }
    candidate_link_ids: set[int] = set()
    applied_link_ids: set[int] = set()
    solved_component_ids: list[str] = []
    topology_conflict_component_ids: list[str] = []
    topology_conflict_link_ids: set[int] = set()
    topology_context_link_ids: set[int] = set()
    path_relative_component_ids: list[str] = []
    path_relative_failed_components: list[dict[str, str]] = []

    for component in collapsed_unit_boundary_analysis:
        component_id = str(component.get("component_id", "") or "")
        variant_local_link_ids = [
            int(link_id)
            for link_id in component.get("variant_local_link_ids", []) or []
            if int(link_id) in index_by_link_id
        ]
        if not variant_local_link_ids:
            continue
        candidate_link_ids.update(variant_local_link_ids)

        boundary_heads = {
            int(item["variant_node"]): float(item["head_m"])
            for item in component.get("variant_boundary_heads", []) or []
            if item.get("head_m") is not None
        }
        if len(boundary_heads) < 2:
            for link_id in variant_local_link_ids:
                result.at[index_by_link_id[link_id], "collapsed_profile_repair_reason"] = "insufficient_boundary_heads"
            continue

        local_links = result.loc[
            pd.to_numeric(result["id_link"], errors="coerce").isin(variant_local_link_ids)
        ].copy()
        if local_links.empty:
            continue
        local_links["id_link"] = pd.to_numeric(local_links["id_link"], errors="coerce").astype("Int64")
        local_links["id_us_node"] = pd.to_numeric(local_links["id_us_node"], errors="coerce").astype("Int64")
        local_links["id_ds_node"] = pd.to_numeric(local_links["id_ds_node"], errors="coerce").astype("Int64")
        local_links["link_length_m"] = pd.to_numeric(local_links["link_length_m"], errors="coerce")
        local_links = local_links.dropna(subset=["id_link", "id_us_node", "id_ds_node", "link_length_m"]).copy()
        if local_links.empty:
            continue

        # A collapse can move the effective split past a former base junction.
        # In that case a child edge is correctly directed in the new graph while
        # its matched parent endpoints occur in reverse order.  Hard-copying both
        # parent heads then makes a positive child drop impossible.  Solve the
        # changed component together with its immediate incident links, using
        # exact parent slopes on the unchanged continuations as strong targets.
        direction_match = local_links.get(
            "slope_parent_direction_matches",
            pd.Series(pd.NA, index=local_links.index, dtype="boolean"),
        )
        raw_local_slope = pd.to_numeric(local_links.get("raw_slope", np.nan), errors="coerce")
        conflict_rows = local_links.loc[direction_match.eq(False) & raw_local_slope.lt(float(min_slope))].copy()
        if not conflict_rows.empty:
            local_node_set = set(local_links["id_us_node"].astype(int)) | set(local_links["id_ds_node"].astype(int))
            all_us = pd.to_numeric(result["id_us_node"], errors="coerce")
            all_ds = pd.to_numeric(result["id_ds_node"], errors="coerce")
            neighborhood = result.loc[all_us.isin(local_node_set) | all_ds.isin(local_node_set)].copy()
            for column in ("id_link", "id_us_node", "id_ds_node"):
                neighborhood[column] = pd.to_numeric(neighborhood[column], errors="coerce").astype("Int64")
            neighborhood["link_length_m"] = pd.to_numeric(neighborhood["link_length_m"], errors="coerce")
            neighborhood = neighborhood.dropna(subset=["id_link", "id_us_node", "id_ds_node", "link_length_m"])
            neighborhood = neighborhood.loc[neighborhood["link_length_m"].gt(0)].copy()

            continuation_target_by_conflict: dict[int, tuple[int, float]] = {}
            for conflict in conflict_rows.itertuples(index=False):
                continuation = neighborhood.loc[
                    neighborhood["id_us_node"].astype(int).eq(int(conflict.id_ds_node))
                    & ~neighborhood["id_link"].astype(int).isin(variant_local_link_ids)
                    & neighborhood.get(
                        "slope_parent_inheritance_applied",
                        pd.Series(False, index=neighborhood.index),
                    ).fillna(False).astype(bool)
                ].copy()
                continuation["_target"] = pd.to_numeric(continuation.get("slope_parent_value"), errors="coerce")
                continuation = continuation.loc[continuation["_target"].gt(0)].copy()
                if "dominant_parent_overlap_fraction" in continuation.columns:
                    continuation = continuation.sort_values(
                        ["dominant_parent_overlap_fraction", "id_link"],
                        ascending=[False, True],
                        na_position="last",
                    )
                else:
                    continuation = continuation.sort_values("id_link")
                if not continuation.empty:
                    selected = continuation.iloc[0]
                    continuation_target_by_conflict[int(conflict.id_link)] = (
                        int(selected["id_link"]),
                        float(selected["_target"]),
                    )

            if continuation_target_by_conflict:
                neighborhood_node_ids = sorted(
                    set(neighborhood["id_us_node"].astype(int))
                    | set(neighborhood["id_ds_node"].astype(int))
                )
                node_head_values: dict[int, list[float]] = {node_id: [] for node_id in neighborhood_node_ids}
                for row in neighborhood.itertuples(index=False):
                    us_head = _safe_float(getattr(row, "wse_us", None))
                    ds_head = _safe_float(getattr(row, "wse_ds", None))
                    if us_head is not None:
                        node_head_values[int(row.id_us_node)].append(us_head)
                    if ds_head is not None:
                        node_head_values[int(row.id_ds_node)].append(ds_head)
                node_head_prior = {
                    node_id: float(np.mean(values))
                    for node_id, values in node_head_values.items()
                    if values
                }
                outer_nodes = [node_id for node_id in neighborhood_node_ids if node_id not in local_node_set]
                anchor_candidates = [node_id for node_id in outer_nodes if node_id in node_head_prior]
                if not anchor_candidates:
                    anchor_candidates = [node_id for node_id in neighborhood_node_ids if node_id in node_head_prior]
                anchor_node = min(anchor_candidates, key=lambda node_id: node_head_prior[node_id]) if anchor_candidates else None
                variable_nodes = [node_id for node_id in neighborhood_node_ids if node_id != anchor_node]
                variable_index = {node_id: index for index, node_id in enumerate(variable_nodes)}
                fixed_heads = {anchor_node: node_head_prior[anchor_node]} if anchor_node is not None else {}

                if anchor_node is not None and all(node_id in node_head_prior for node_id in variable_nodes):
                    section_values = pd.to_numeric(neighborhood.get("slope_section_ref", np.nan), errors="coerce")
                    positive_sections = section_values.loc[section_values.gt(0)]
                    slope_scale_default = (
                        float(positive_sections.median())
                        if not positive_sections.empty
                        else max(float(min_slope), 1e-4)
                    )
                    edge_specs: list[tuple[int, int, int, float, float, float, str]] = []
                    for row in neighborhood.itertuples(index=False):
                        link_id = int(row.id_link)
                        target_source = "child_preinheritance"
                        target = _safe_float(getattr(row, "slope_parent_value", None))
                        inherited = bool(getattr(row, "slope_parent_inheritance_applied", False))
                        if link_id in continuation_target_by_conflict:
                            _, target = continuation_target_by_conflict[link_id]
                            weight = 100.0
                            target_source = "downstream_unchanged_parent_continuation"
                        elif target is not None and target > 0:
                            direction_matches = getattr(row, "slope_parent_direction_matches", pd.NA)
                            if inherited:
                                weight = 100.0
                                target_source = "unchanged_parent"
                            elif pd.notna(direction_matches) and bool(direction_matches):
                                weight = 50.0 if link_id not in variant_local_link_ids else 25.0
                                target_source = "high_overlap_parent_target"
                            else:
                                target = None
                                weight = 10.0 if link_id in variant_local_link_ids else 25.0
                        else:
                            target = _safe_float(getattr(row, "slope_preinheritance_value", None))
                            if target is None or target < float(min_slope):
                                target = _safe_float(getattr(row, "slope_section_ref", None))
                                target_source = "section_reference"
                            weight = 10.0 if link_id in variant_local_link_ids else 25.0
                        if target is None or target < float(min_slope):
                            target = max(float(min_slope), slope_scale_default)
                            target_source = "minimum_or_section_reference"
                        edge_specs.append(
                            (
                                link_id,
                                int(row.id_us_node),
                                int(row.id_ds_node),
                                float(row.link_length_m),
                                float(target),
                                float(weight),
                                target_source,
                            )
                        )

                    # Initialize from the target drops rather than the conflicting
                    # child profile heads.  This keeps SLSQP well scaled and gives
                    # an exact solution immediately when the one-hop graph is a
                    # tree; loops are reconciled by least squares.
                    all_node_index = {
                        node_id: index for index, node_id in enumerate(neighborhood_node_ids)
                    }
                    equations: list[np.ndarray] = []
                    rhs: list[float] = []
                    for _, us_node, ds_node, length, target, _, _ in edge_specs:
                        coeff = np.zeros(len(neighborhood_node_ids), dtype=float)
                        coeff[all_node_index[us_node]] = 1.0
                        coeff[all_node_index[ds_node]] = -1.0
                        equations.append(coeff)
                        rhs.append(float(target) * length)
                    anchor_coeff = np.zeros(len(neighborhood_node_ids), dtype=float)
                    anchor_coeff[all_node_index[int(anchor_node)]] = 1.0
                    equations.append(anchor_coeff)
                    rhs.append(float(fixed_heads[int(anchor_node)]))
                    initial_solution, *_ = np.linalg.lstsq(
                        np.vstack(equations),
                        np.asarray(rhs, dtype=float),
                        rcond=None,
                    )
                    x0 = np.asarray(
                        [initial_solution[all_node_index[node_id]] for node_id in variable_nodes],
                        dtype=float,
                    )

                    def local_head(node_id: int, x: np.ndarray) -> float:
                        if node_id in variable_index:
                            return float(x[variable_index[node_id]])
                        return float(fixed_heads[node_id])

                    conflict_upstream_nodes = {
                        int(row.id_us_node) for row in conflict_rows.itertuples(index=False)
                    }

                    def local_objective(x: np.ndarray) -> float:
                        slope_penalty = 0.0
                        for _, us_node, ds_node, length, target, weight, _ in edge_specs:
                            slope = (local_head(us_node, x) - local_head(ds_node, x)) / length
                            scale = max(abs(target), slope_scale_default, float(min_slope))
                            slope_penalty += weight * ((slope - target) / scale) ** 2
                        head_penalty = 0.0
                        for node_id in variable_nodes:
                            # Profile heads stabilize the absolute placement but
                            # remain deliberately weak; slope targets determine
                            # the internally consistent local head differences.
                            weight = 0.0001 if node_id in conflict_upstream_nodes else 0.001
                            head_penalty += weight * ((local_head(node_id, x) - node_head_prior[node_id]) / 0.25) ** 2
                        return float(slope_penalty + head_penalty)

                    constraints = [
                        {
                            "type": "ineq",
                            "fun": lambda x, us_node=us_node, ds_node=ds_node, length=length: float(
                                local_head(us_node, x)
                                - local_head(ds_node, x)
                                - float(min_slope) * length
                            ),
                        }
                        for _, us_node, ds_node, length, _, _, _ in edge_specs
                    ]
                    initial_feasible = all(
                        local_head(us_node, x0) - local_head(ds_node, x0)
                        >= float(min_slope) * length - 1e-9
                        for _, us_node, ds_node, length, _, _, _ in edge_specs
                    )
                    if initial_feasible:
                        solved_x = x0
                        local_success = True
                    else:
                        try:
                            local_solution = minimize(
                                local_objective,
                                x0,
                                method="SLSQP",
                                constraints=constraints,
                                options={"ftol": 1e-12, "maxiter": 5000},
                            )
                        except Exception:
                            local_solution = None
                        local_success = bool(local_solution is not None and local_solution.success)
                        solved_x = np.asarray(local_solution.x, dtype=float) if local_success else x0

                    if local_success:
                        solved_component_ids.append(component_id)
                        topology_conflict_component_ids.append(component_id)
                        x = np.asarray(solved_x, dtype=float)
                        for link_id, us_node, ds_node, length, target, _, target_source in edge_specs:
                            row_index = index_by_link_id.get(link_id)
                            if row_index is None:
                                continue
                            solved_us_head = local_head(us_node, x)
                            solved_ds_head = local_head(ds_node, x)
                            solved_drop = solved_us_head - solved_ds_head
                            solved_slope = max(solved_drop / length, float(min_slope))
                            prior_slope = _safe_float(result.at[row_index, "slope_used"])
                            result.at[row_index, "collapsed_profile_repair_candidate"] = True
                            result.at[row_index, "collapsed_profile_repair_applied"] = True
                            result.at[row_index, "collapsed_profile_repair_reason"] = "topology_conflict_local_solve"
                            result.at[row_index, "collapsed_reference_drop_m"] = solved_drop
                            result.at[row_index, "collapsed_reference_slope"] = solved_slope
                            result.at[row_index, "slope_used"] = solved_slope
                            result.at[row_index, "slope_source_method"] = "topology_conflict_local_head_solver"
                            result.at[row_index, "slope_adjusted"] = True
                            result.at[row_index, "slope_reason"] = f"target:{target_source}"
                            result.at[row_index, "slope_minimum_applied"] = bool(
                                np.isclose(solved_slope, float(min_slope))
                            )
                            result.at[row_index, "topology_conflict_target_slope"] = target
                            result.at[row_index, "topology_conflict_target_source"] = target_source
                            result.at[row_index, "topology_optimized_head_us"] = solved_us_head
                            result.at[row_index, "topology_optimized_head_ds"] = solved_ds_head
                            if np.isfinite(prior_slope or np.nan) and float(prior_slope) > 0:
                                result.at[row_index, "collapsed_profile_change_factor"] = solved_slope / float(prior_slope)
                            applied_link_ids.add(link_id)
                            if link_id in variant_local_link_ids:
                                topology_conflict_link_ids.add(link_id)
                            else:
                                topology_context_link_ids.add(link_id)
                        continue

        local_node_ids = sorted(
            set(local_links["id_us_node"].astype(int).tolist())
            | set(local_links["id_ds_node"].astype(int).tolist())
        )
        internal_nodes = [node_id for node_id in local_node_ids if node_id not in boundary_heads]
        internal_index = {node_id: index for index, node_id in enumerate(internal_nodes)}

        edges: list[tuple[int, int, int, float]] = [
            (
                int(row.id_link),
                int(row.id_us_node),
                int(row.id_ds_node),
                float(row.link_length_m),
            )
            for row in local_links.itertuples(index=False)
        ]

        if not edges:
            continue

        local_by_link = local_links.set_index("id_link", drop=False)
        occurrence_records: list[dict[str, object]] = []
        invalid_path_reason = ""
        for path_number, path in enumerate(component.get("variant_paths", []) or [], start=1):
            path_link_ids = [
                int(link_id)
                for link_id in path.get("link_path", []) or []
                if int(link_id) in local_by_link.index
            ]
            if not path_link_ids:
                continue
            source_node = int(path["source_boundary_node"])
            target_node = int(path["target_boundary_node"])
            if source_node not in boundary_heads or target_node not in boundary_heads:
                invalid_path_reason = "path_boundary_head_unavailable"
                break
            path_length = float(
                pd.to_numeric(local_by_link.loc[path_link_ids, "link_length_m"], errors="coerce").sum()
            )
            target_drop = float(boundary_heads[source_node] - boundary_heads[target_node])
            target_slope = target_drop / path_length if path_length > 0 else float("nan")
            if not np.isfinite(target_slope) or target_slope < float(min_slope):
                invalid_path_reason = (
                    f"nonpositive_or_below_minimum_path_target:{source_node}->{target_node}"
                )
                break
            path_id = f"P{path_number}:{source_node}->{target_node}"
            for link_id in path_link_ids:
                link_length = float(local_by_link.loc[link_id, "link_length_m"])
                occurrence_records.append(
                    {
                        "path_id": path_id,
                        "link_id": int(link_id),
                        "path_length_m": path_length,
                        "target_drop_m": target_drop,
                        "target_slope": target_slope,
                        "weight": link_length / path_length,
                    }
                )

        if invalid_path_reason or not occurrence_records:
            reason = invalid_path_reason or "missing_variant_path_targets"
            for link_id in variant_local_link_ids:
                result.at[index_by_link_id[link_id], "collapsed_profile_repair_reason"] = reason
            path_relative_failed_components.append({"component_id": component_id, "reason": reason})
            continue

        occurrences = pd.DataFrame.from_records(occurrence_records)

        def head(node_id: int, x: np.ndarray) -> float:
            if node_id in boundary_heads:
                return float(boundary_heads[node_id])
            return float(x[internal_index[node_id]])

        def objective(x: np.ndarray) -> float:
            return float(
                sum(
                    float(row.weight)
                    * (
                        (
                            (
                                head(int(local_by_link.loc[int(row.link_id), "id_us_node"]), x)
                                - head(int(local_by_link.loc[int(row.link_id), "id_ds_node"]), x)
                            )
                            / float(local_by_link.loc[int(row.link_id), "link_length_m"])
                            / float(row.target_slope)
                        )
                        - 1.0
                    )
                    ** 2
                    for row in occurrences.itertuples(index=False)
                )
            )

        # Recover a topology-consistent starting point from the current positive
        # link drops while retaining the hard base-derived boundary heads.
        current_heads = dict(boundary_heads)
        unresolved = set(internal_nodes)
        for _ in range(len(local_node_ids) + 1):
            changed = False
            for link_id, us_node, ds_node, link_length in edges:
                row_index = index_by_link_id[link_id]
                current_slope = _safe_float(result.at[row_index, "slope_used"])
                if current_slope is None or current_slope < float(min_slope):
                    current_slope = float(min_slope)
                drop = current_slope * link_length
                if us_node in current_heads and ds_node not in current_heads:
                    current_heads[ds_node] = current_heads[us_node] - drop
                    unresolved.discard(ds_node)
                    changed = True
                elif ds_node in current_heads and us_node not in current_heads:
                    current_heads[us_node] = current_heads[ds_node] + drop
                    unresolved.discard(us_node)
                    changed = True
            if not changed:
                break
        if unresolved:
            reason = "unresolved_internal_head_initialization"
            for link_id in variant_local_link_ids:
                result.at[index_by_link_id[link_id], "collapsed_profile_repair_reason"] = reason
            path_relative_failed_components.append({"component_id": component_id, "reason": reason})
            continue
        x0 = np.asarray([current_heads[node_id] for node_id in internal_nodes], dtype=float)
        objective_before = objective(x0)

        constraints = [
            {
                "type": "ineq",
                "fun": lambda x, us_node=us_node, ds_node=ds_node, link_length=link_length: float(
                    head(us_node, x)
                    - head(ds_node, x)
                    - float(min_slope) * link_length
                ),
            }
            for _, us_node, ds_node, link_length in edges
        ]

        if internal_nodes:
            try:
                solution = minimize(
                    objective,
                    x0,
                    method="SLSQP",
                    constraints=constraints,
                    options={"ftol": 1e-12, "maxiter": 5000},
                )
            except Exception:
                solution = None
            solution_success = bool(solution is not None and solution.success)
            solved_values = np.asarray(solution.x, dtype=float) if solution_success else x0
        else:
            solution = None
            solved_values = np.asarray([], dtype=float)
            solution_success = all(
                head(us_node, solved_values) - head(ds_node, solved_values)
                >= float(min_slope) * link_length - 1e-9
                for _, us_node, ds_node, link_length in edges
            )

        if not solution_success:
            reason = "solver_failed"
            if solution is not None and solution.message:
                reason = f"solver_failed:{solution.message}"
            for link_id in variant_local_link_ids:
                result.at[index_by_link_id[link_id], "collapsed_profile_repair_reason"] = reason
            path_relative_failed_components.append({"component_id": component_id, "reason": reason})
            continue

        solved_component_ids.append(component_id)
        path_relative_component_ids.append(component_id)
        x = solved_values
        objective_after = objective(x)
        for link_id, us_node, ds_node, link_length in edges:
            row_index = index_by_link_id.get(int(link_id))
            if row_index is None:
                continue
            candidate_link_ids.add(int(link_id))
            reference_drop = float(head(us_node, x) - head(ds_node, x))
            repaired_slope = (
                max(float(reference_drop / link_length), float(min_slope))
                if link_length > 0
                else float("nan")
            )
            prior_slope = _safe_float(result.at[row_index, "slope_used"])
            result.at[row_index, "collapsed_profile_repair_candidate"] = True
            result.at[row_index, "collapsed_profile_repair_applied"] = True
            result.at[row_index, "collapsed_profile_repair_reason"] = "ok"
            result.at[row_index, "collapsed_reference_drop_m"] = reference_drop
            result.at[row_index, "collapsed_reference_slope"] = repaired_slope
            result.at[row_index, "slope_used"] = repaired_slope
            result.at[row_index, "slope_source_method"] = "collapsed_unit_path_relative_solver"
            result.at[row_index, "slope_adjusted"] = True
            result.at[row_index, "slope_reason"] = "filled_from_collapsed_unit_path_relative_solver"
            result.at[row_index, "slope_minimum_applied"] = bool(
                np.isclose(repaired_slope, float(min_slope))
            )
            targets = occurrences.loc[occurrences["link_id"].eq(int(link_id))]
            result.at[row_index, "path_relative_target_paths"] = ",".join(
                targets["path_id"].astype(str).tolist()
            )
            result.at[row_index, "path_relative_target_slopes"] = ",".join(
                f"{float(value):.12g}" for value in targets["target_slope"].tolist()
            )
            result.at[row_index, "path_relative_objective_before"] = objective_before
            result.at[row_index, "path_relative_objective_after"] = objective_after
            result.at[row_index, "path_relative_optimized_head_us"] = head(us_node, x)
            result.at[row_index, "path_relative_optimized_head_ds"] = head(ds_node, x)
            if np.isfinite(prior_slope) and prior_slope > 0:
                result.at[row_index, "collapsed_profile_change_factor"] = float(repaired_slope / prior_slope)
            applied_link_ids.add(int(link_id))

    untouched_mask = (
        result["collapsed_profile_repair_candidate"].fillna(False).astype(bool)
        & result["collapsed_profile_repair_reason"].eq("")
    )
    result.loc[untouched_mask, "collapsed_profile_repair_reason"] = "not_applied"
    return result, {
        "candidate_link_ids": sorted(int(link_id) for link_id in candidate_link_ids),
        "applied_link_ids": sorted(int(link_id) for link_id in applied_link_ids),
        "solved_component_ids": sorted(component_id for component_id in solved_component_ids if component_id),
        "topology_conflict_component_ids": sorted(
            component_id for component_id in topology_conflict_component_ids if component_id
        ),
        "topology_conflict_link_ids": sorted(topology_conflict_link_ids),
        "topology_context_link_ids": sorted(topology_context_link_ids),
        "path_relative_component_ids": sorted(
            component_id for component_id in path_relative_component_ids if component_id
        ),
        "path_relative_failed_components": path_relative_failed_components,
    }


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


def _apply_parent_slope_inheritance(
    link_context: pd.DataFrame,
    links: gpd.GeoDataFrame,
    *,
    parent_link_context: pd.DataFrame | None,
    excluded_link_ids: set[int] | None = None,
    overlap_threshold: float = 0.999,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Reuse final parent slopes for strict, direction-consistent 1:1 links.

    Child-state WSE profiles are still retained in the raw/profile columns.  The
    inherited value replaces only the final routing slope and records both the
    child pre-inheritance value and the parent provenance.  Links participating
    in a collapsed local component are excluded because their heads/slopes must
    be solved together even when geometry matching labels one segment 1:1.
    """

    result = link_context.copy()
    result["slope_parent_inheritance_candidate"] = False
    result["slope_parent_inheritance_applied"] = False
    result["slope_inherited_from_parent_link"] = pd.Series(pd.NA, index=result.index, dtype="Int64")
    result["slope_parent_inheritance_reason"] = ""
    result["slope_parent_value"] = np.nan
    result["slope_preinheritance_value"] = pd.to_numeric(result.get("slope_used"), errors="coerce")
    result["slope_parent_source_method"] = ""
    result["slope_parent_reason"] = ""
    result["slope_parent_direction_matches"] = pd.Series(pd.NA, index=result.index, dtype="boolean")

    required_link_columns = {
        "id_link",
        "lineage_type",
        "dominant_parent_link_id",
        "dominant_parent_overlap_fraction",
        "matched_parent_us_node",
        "matched_parent_ds_node",
    }
    required_parent_columns = {"id_link", "id_us_node", "id_ds_node", "slope_used"}
    if parent_link_context is None or parent_link_context.empty:
        result["slope_parent_inheritance_reason"] = "parent_context_unavailable"
        return result, {
            "applied_count": 0,
            "applied_link_ids": [],
            "excluded_local_link_ids": sorted(int(value) for value in (excluded_link_ids or set())),
            "reason": "parent_context_unavailable",
        }
    if not required_link_columns.issubset(links.columns):
        result["slope_parent_inheritance_reason"] = "child_lineage_columns_unavailable"
        return result, {
            "applied_count": 0,
            "applied_link_ids": [],
            "excluded_local_link_ids": sorted(int(value) for value in (excluded_link_ids or set())),
            "reason": "child_lineage_columns_unavailable",
        }
    if not required_parent_columns.issubset(parent_link_context.columns):
        result["slope_parent_inheritance_reason"] = "parent_slope_columns_unavailable"
        return result, {
            "applied_count": 0,
            "applied_link_ids": [],
            "excluded_local_link_ids": sorted(int(value) for value in (excluded_link_ids or set())),
            "reason": "parent_slope_columns_unavailable",
        }

    child = links[list(required_link_columns)].copy()
    for column in (
        "id_link",
        "dominant_parent_link_id",
        "matched_parent_us_node",
        "matched_parent_ds_node",
    ):
        child[column] = pd.to_numeric(child[column], errors="coerce").astype("Int64")
    child["dominant_parent_overlap_fraction"] = pd.to_numeric(
        child["dominant_parent_overlap_fraction"], errors="coerce"
    )

    parent_columns = ["id_link", "id_us_node", "id_ds_node", "slope_used"]
    for optional_column in ("slope_source_method", "slope_reason", "slope_minimum_applied"):
        if optional_column in parent_link_context.columns:
            parent_columns.append(optional_column)
    parent = parent_link_context[parent_columns].copy().rename(
        columns={
            "id_link": "parent_id_link",
            "id_us_node": "parent_id_us_node",
            "id_ds_node": "parent_id_ds_node",
            "slope_used": "parent_slope_used",
            "slope_source_method": "parent_slope_source_method",
            "slope_reason": "parent_slope_reason",
            "slope_minimum_applied": "parent_slope_minimum_applied",
        }
    )
    for column in ("parent_id_link", "parent_id_us_node", "parent_id_ds_node"):
        parent[column] = pd.to_numeric(parent[column], errors="coerce").astype("Int64")
    parent["parent_slope_used"] = pd.to_numeric(parent["parent_slope_used"], errors="coerce")
    parent = parent.dropna(subset=["parent_id_link"]).drop_duplicates(subset=["parent_id_link"])

    inheritance = child.merge(
        parent,
        left_on="dominant_parent_link_id",
        right_on="parent_id_link",
        how="left",
    ).set_index("id_link", drop=False)
    excluded = {int(value) for value in (excluded_link_ids or set())}
    result_index_by_link = {
        int(link_id): row_index
        for row_index, link_id in result["id_link"].items()
        if pd.notna(link_id)
    }
    applied_link_ids: list[int] = []
    candidate_link_ids: list[int] = []
    direction_mismatch_link_ids: list[int] = []

    for child_link_id, row in inheritance.iterrows():
        if pd.isna(child_link_id) or int(child_link_id) not in result_index_by_link:
            continue
        link_id = int(child_link_id)
        result_index = result_index_by_link[link_id]
        if str(row.get("lineage_type", "")) != "unchanged_1to1":
            result.at[result_index, "slope_parent_inheritance_reason"] = "not_unchanged_1to1"
            continue
        overlap = _safe_float(row.get("dominant_parent_overlap_fraction"))
        if overlap is None or overlap < 0.95:
            result.at[result_index, "slope_parent_inheritance_reason"] = "below_parent_target_overlap_threshold"
            continue

        direction_values = [
            row.get("matched_parent_us_node"),
            row.get("matched_parent_ds_node"),
            row.get("parent_id_us_node"),
            row.get("parent_id_ds_node"),
        ]
        if any(pd.isna(value) for value in direction_values):
            result.at[result_index, "slope_parent_inheritance_reason"] = "missing_parent_direction"
            continue
        direction_matches = (
            int(row["matched_parent_us_node"]) == int(row["parent_id_us_node"])
            and int(row["matched_parent_ds_node"]) == int(row["parent_id_ds_node"])
        )
        result.at[result_index, "slope_parent_direction_matches"] = bool(direction_matches)
        parent_slope = _safe_float(row.get("parent_slope_used"))
        if parent_slope is not None and parent_slope > 0:
            result.at[result_index, "slope_parent_value"] = parent_slope
            result.at[result_index, "slope_parent_source_method"] = str(
                row.get("parent_slope_source_method", "") or ""
            )
            result.at[result_index, "slope_parent_reason"] = str(row.get("parent_slope_reason", "") or "")
        if not direction_matches:
            result.at[result_index, "slope_parent_inheritance_reason"] = (
                "collapsed_local_parent_direction_mismatch"
                if link_id in excluded
                else "parent_direction_mismatch"
            )
            direction_mismatch_link_ids.append(link_id)
            continue
        if overlap < float(overlap_threshold):
            result.at[result_index, "slope_parent_inheritance_reason"] = "below_overlap_threshold_parent_target_available"
            continue
        result.at[result_index, "slope_parent_inheritance_candidate"] = True
        candidate_link_ids.append(link_id)
        if link_id in excluded:
            result.at[result_index, "slope_parent_inheritance_reason"] = "collapsed_local_recalculation"
            continue

        if parent_slope is None or parent_slope <= 0:
            result.at[result_index, "slope_parent_inheritance_reason"] = "invalid_parent_slope"
            continue

        parent_link_id = int(row["parent_id_link"])
        result.at[result_index, "slope_used"] = parent_slope
        result.at[result_index, "slope_source_method"] = "parent_unchanged_1to1"
        result.at[result_index, "slope_reason"] = "inherited_from_parent_unchanged_1to1"
        result.at[result_index, "slope_adjusted"] = True
        if "slope_minimum_applied" in result.columns:
            inherited_minimum = row.get("parent_slope_minimum_applied", False)
            result.at[result_index, "slope_minimum_applied"] = bool(inherited_minimum) if pd.notna(inherited_minimum) else False
        result.at[result_index, "slope_parent_inheritance_applied"] = True
        result.at[result_index, "slope_inherited_from_parent_link"] = parent_link_id
        result.at[result_index, "slope_parent_inheritance_reason"] = "inherited_from_parent_unchanged_1to1"
        applied_link_ids.append(link_id)

    untouched = result["slope_parent_inheritance_reason"].eq("")
    result.loc[untouched, "slope_parent_inheritance_reason"] = "lineage_record_unavailable"
    return result, {
        "applied_count": int(len(applied_link_ids)),
        "candidate_count": int(len(candidate_link_ids)),
        "applied_link_ids": sorted(applied_link_ids),
        "candidate_link_ids": sorted(candidate_link_ids),
        "excluded_local_link_ids": sorted(excluded),
        "direction_mismatch_link_ids": sorted(direction_mismatch_link_ids),
        "overlap_threshold": float(overlap_threshold),
        "reason": "ok",
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
    parent_link_context_path: str | Path | None = None,
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
    collapsed_unit_boundary_analysis = _build_collapsed_unit_boundary_analysis(
        result.link_context,
        links_path=links_path,
    )
    collapsed_local_link_ids = {
        int(link_id)
        for component in collapsed_unit_boundary_analysis
        for link_id in component.get("variant_local_link_ids", []) or []
    }
    resolved_parent_link_context_path = (
        Path(parent_link_context_path).expanduser().resolve()
        if parent_link_context_path is not None
        else None
    )
    parent_link_context = (
        pd.read_csv(resolved_parent_link_context_path)
        if resolved_parent_link_context_path is not None and resolved_parent_link_context_path.exists()
        else None
    )
    link_context, parent_slope_inheritance_summary = _apply_parent_slope_inheritance(
        result.link_context,
        links,
        parent_link_context=parent_link_context,
        excluded_link_ids=collapsed_local_link_ids,
    )
    link_context, family_slope_inheritance_summary = _apply_family_representative_slope_inheritance(
        link_context,
        links,
    )
    link_context, collapsed_unit_head_solver_summary = _apply_collapsed_unit_head_solver(
        link_context,
        collapsed_unit_boundary_analysis=collapsed_unit_boundary_analysis,
        min_slope=resolved_config.min_slope,
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
            "parent_link_hydraulic_context": str(resolved_parent_link_context_path) if resolved_parent_link_context_path is not None else "",
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
        "parent_slope_inheritance": parent_slope_inheritance_summary,
        "collapsed_unit_head_solver": collapsed_unit_head_solver_summary,
        "collapsed_unit_boundary_analysis": collapsed_unit_boundary_analysis,
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
