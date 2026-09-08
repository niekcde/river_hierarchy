from __future__ import annotations

from dataclasses import asdict, dataclass
from os import PathLike
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import networkx as nx

from network_variants.reference_context import ResolvedKbContext, resolve_kb_context

from hierarchy_level_definition.metrics.equivalent_resistance import (
    compute_K_chezy,
    compute_K_manning,
    compute_geometry,
    solve_bubble_equivalent_K,
)
from hierarchy_level_definition.unit_detection.bifurcation_confluence_units import (
    StructuralUnit,
    annotate_unit_context,
    load_network,
)


WIDTH_FIELD_CANDIDATES = (
    "wid_adj_routing",
    "wid_adj_wet",
    "wid_adj",
    "wid",
    "wid_med",
)
# Prefer the physical centreline length. ``len_adj`` is retained solely as a
# compatibility fallback for legacy products that lack ``len``.
LENGTH_FIELD_CANDIDATES = ("len", "len_adj", "link_length_m")
KB_FIELD_CANDIDATES = ("k_b", "kb", "rapid_kb_value")
MANNING_N_FIELD_CANDIDATES = ("n", "manning_n", "rapid_n_manning")
CHEZY_CZ_FIELD_CANDIDATES = ("C_z", "Cz", "chezy_cz")
NODE_HEAD_FIELD_CANDIDATES = ("sword_wse", "H", "head", "wse")


def resolve_length_field_candidates(preferred_field: str | None) -> tuple[str, ...]:
    if preferred_field is None:
        return tuple(LENGTH_FIELD_CANDIDATES)
    normalized = str(preferred_field).strip()
    if normalized not in LENGTH_FIELD_CANDIDATES:
        raise ValueError(
            f"Unsupported equivalent-resistance length field {preferred_field!r}. "
            f"Expected one of {list(LENGTH_FIELD_CANDIDATES)}."
        )
    return (normalized,) + tuple(candidate for candidate in LENGTH_FIELD_CANDIDATES if candidate != normalized)


@dataclass(frozen=True, slots=True)
class BubbleEquivalentResistanceConfig:
    resistance_law: str = "manning"
    geometry_method: str = "wide"
    kb_mode: str = "fixed"
    kb_value: float = 20.0
    reference_kb_summary: Mapping[str, Any] | None = None
    width_field_candidates: tuple[str, ...] = WIDTH_FIELD_CANDIDATES
    length_field_candidates: tuple[str, ...] = LENGTH_FIELD_CANDIDATES
    kb_field_candidates: tuple[str, ...] = KB_FIELD_CANDIDATES
    manning_n_field_candidates: tuple[str, ...] = MANNING_N_FIELD_CANDIDATES
    chezy_cz_field_candidates: tuple[str, ...] = CHEZY_CZ_FIELD_CANDIDATES
    default_manning_n: float = 0.035
    default_chezy_cz: float = 22.0
    gravity_mps2: float = 9.81
    use_node_head_field_as_initial_guess: bool = True
    smooth: bool = True
    smooth_eps: float = 1e-10
    solver_backend: str = "dense_legacy"
    sparse_min_internal_nodes: int = 64
    solver_cache_dir: str | None = None
    conditional_residual_relative_tolerance: float = 0.01
    extended_residual_relative_allowance: float = 0.015
    conditional_k_relative_tolerance: float = 0.02


def _resolve_first_present(columns: Sequence[str], candidates: Sequence[str]) -> str | None:
    column_set = set(columns)
    for candidate in candidates:
        if candidate in column_set:
            return candidate
    return None


def _resolve_numeric_series(
    frame: pd.DataFrame,
    *,
    candidates: Sequence[str],
    field_name: str,
    default_value: float | None = None,
) -> tuple[pd.Series, str]:
    column = _resolve_first_present(frame.columns, candidates)
    if column is not None:
        values = pd.to_numeric(frame[column], errors="coerce")
        if default_value is None:
            if values.isna().any():
                raise ValueError(f"Could not resolve finite values for '{field_name}' from column '{column}'.")
            return values.astype(float), column
        values = values.fillna(float(default_value))
        return values.astype(float), column
    if default_value is None:
        raise ValueError(
            f"Could not resolve '{field_name}'. Tried columns: {list(candidates)}"
        )
    return pd.Series(float(default_value), index=frame.index, dtype=float), f"default:{field_name}"


def _canonicalize_links(
    links: pd.DataFrame,
    *,
    length_field_candidates: Sequence[str] = LENGTH_FIELD_CANDIDATES,
) -> pd.DataFrame:
    frame = links.copy()
    if "id_link" not in frame.columns:
        raise ValueError("Equivalent-resistance link inputs require an 'id_link' column.")

    if "id_us_node" in frame.columns and "id_ds_node" in frame.columns:
        frame["id_us_node"] = pd.to_numeric(frame["id_us_node"], errors="coerce").astype("Int64")
        frame["id_ds_node"] = pd.to_numeric(frame["id_ds_node"], errors="coerce").astype("Int64")
    elif "id_nodes" in frame.columns:
        endpoints = frame["id_nodes"].astype(str).str.split(",", expand=True)
        if endpoints.shape[1] != 2:
            raise ValueError("Could not parse 'id_nodes' into upstream/downstream endpoints.")
        frame["id_us_node"] = pd.to_numeric(endpoints[0].str.strip(), errors="coerce").astype("Int64")
        frame["id_ds_node"] = pd.to_numeric(endpoints[1].str.strip(), errors="coerce").astype("Int64")
    else:
        raise ValueError("Equivalent-resistance link inputs require endpoint columns 'id_us_node'/'id_ds_node' or 'id_nodes'.")

    if frame["id_us_node"].isna().any() or frame["id_ds_node"].isna().any():
        raise ValueError("Link endpoint columns contain missing values.")

    frame["id_link"] = pd.to_numeric(frame["id_link"], errors="coerce").astype("Int64")
    if frame["id_link"].isna().any():
        raise ValueError("Link id_link values must be numeric.")

    if "geometry" in frame.columns:
        geometry_length = frame["geometry"].map(lambda geom: float(geom.length) if geom is not None else np.nan)
    else:
        geometry_length = pd.Series(np.nan, index=frame.index, dtype=float)

    length_values, length_source = _resolve_numeric_series(
        frame.assign(_geometry_length=geometry_length),
        candidates=tuple(length_field_candidates) + ("_geometry_length",),
        field_name="edge length",
    )
    frame["L"] = length_values.astype(float)
    frame["length_source"] = length_source
    return frame


def _canonicalize_nodes(nodes: pd.DataFrame) -> pd.DataFrame:
    frame = nodes.copy()
    node_id_column = _resolve_first_present(frame.columns, ("id_node", "node_id", "id"))
    if node_id_column is None:
        raise ValueError("Equivalent-resistance node inputs require an id_node/node_id column.")
    frame["node_id"] = pd.to_numeric(frame[node_id_column], errors="coerce").astype("Int64")
    if frame["node_id"].isna().any():
        raise ValueError("Node identifiers must be numeric.")
    return frame


def _resolved_kb_context(config: BubbleEquivalentResistanceConfig) -> ResolvedKbContext:
    return resolve_kb_context(
        kb_mode=config.kb_mode,
        kb_value=config.kb_value,
        reference_kb_summary=config.reference_kb_summary,
    )


def prepare_bubble_resistance_edges(
    links: pd.DataFrame,
    *,
    config: BubbleEquivalentResistanceConfig | None = None,
) -> tuple[pd.DataFrame, ResolvedKbContext]:
    """Compute geometry-derived edge resistance values for bubble reduction."""

    config = config or BubbleEquivalentResistanceConfig()
    normalized_law = str(config.resistance_law or "manning").strip().lower()
    if normalized_law not in {"manning", "chezy"}:
        raise ValueError("resistance_law must be either 'manning' or 'chezy'.")

    frame = _canonicalize_links(links, length_field_candidates=config.length_field_candidates)
    resolved_kb = _resolved_kb_context(config)

    width_values, width_source = _resolve_numeric_series(
        frame,
        candidates=config.width_field_candidates,
        field_name="edge width",
    )
    kb_values, kb_source = _resolve_numeric_series(
        frame,
        candidates=config.kb_field_candidates,
        field_name="width-depth ratio",
        default_value=resolved_kb.kb_value,
    )

    geometry = compute_geometry(width_values.to_numpy(dtype=float), kb_values.to_numpy(dtype=float), method=config.geometry_method)
    prepared = frame.copy()
    prepared["edge_id"] = prepared["id_link"].astype(int)
    prepared["u"] = prepared["id_us_node"].astype(int)
    prepared["v"] = prepared["id_ds_node"].astype(int)
    prepared["w"] = width_values.astype(float)
    prepared["width_source"] = width_source
    prepared["k_b"] = kb_values.astype(float)
    prepared["kb_source"] = kb_source
    prepared["d"] = np.asarray(geometry["d"], dtype=float)
    prepared["A"] = np.asarray(geometry["A"], dtype=float)
    prepared["R_h"] = np.asarray(geometry["R_h"], dtype=float)
    prepared["geometry_method"] = str(geometry["method"])
    prepared["kb_mode"] = resolved_kb.kb_mode
    prepared["kb_source_method"] = resolved_kb.kb_source_method

    if normalized_law == "manning":
        n_values, n_source = _resolve_numeric_series(
            prepared,
            candidates=config.manning_n_field_candidates,
            field_name="Manning n",
            default_value=config.default_manning_n,
        )
        prepared["n"] = n_values.astype(float)
        prepared["n_source"] = n_source
        prepared["C_z"] = np.nan
        prepared["C_z_source"] = ""
        prepared["K_e"] = compute_K_manning(
            prepared["L"].to_numpy(dtype=float),
            prepared["w"].to_numpy(dtype=float),
            prepared["n"].to_numpy(dtype=float),
            prepared["k_b"].to_numpy(dtype=float),
            geometry_method=config.geometry_method,
        )
    else:
        cz_values, cz_source = _resolve_numeric_series(
            prepared,
            candidates=config.chezy_cz_field_candidates,
            field_name="Chezy Cz",
            default_value=config.default_chezy_cz,
        )
        prepared["n"] = np.nan
        prepared["n_source"] = ""
        prepared["C_z"] = cz_values.astype(float)
        prepared["C_z_source"] = cz_source
        prepared["K_e"] = compute_K_chezy(
            prepared["L"].to_numpy(dtype=float),
            prepared["w"].to_numpy(dtype=float),
            prepared["C_z"].to_numpy(dtype=float),
            prepared["k_b"].to_numpy(dtype=float),
            geometry_method=config.geometry_method,
            g=config.gravity_mps2,
        )

    prepared["G_e"] = 1.0 / prepared["K_e"].astype(float)
    prepared["resistance_law"] = normalized_law
    prepared["solver_cache_edge_alias"] = prepared["edge_id"].map(int).astype(object)
    prepared["solver_cache_u_alias"] = prepared["u"].map(int).astype(object)
    prepared["solver_cache_v_alias"] = prepared["v"].map(int).astype(object)
    lineage_columns = {
        "lineage_type",
        "dominant_parent_link_id",
        "matched_parent_us_node",
        "matched_parent_ds_node",
    }
    if lineage_columns.issubset(prepared.columns):
        exact = prepared["lineage_type"].astype(str).eq("unchanged_1to1")
        parent_edge = pd.to_numeric(
            prepared["dominant_parent_link_id"], errors="coerce"
        )
        parent_u = pd.to_numeric(prepared["matched_parent_us_node"], errors="coerce")
        parent_v = pd.to_numeric(prepared["matched_parent_ds_node"], errors="coerce")
        exact &= parent_edge.notna() & parent_u.notna() & parent_v.notna()
        prepared.loc[exact, "solver_cache_edge_alias"] = parent_edge.loc[exact].astype(int)

        node_alias_candidates: dict[int, set[int]] = {}
        for row in prepared.loc[exact].itertuples(index=False):
            node_alias_candidates.setdefault(int(row.u), set()).add(
                int(row.matched_parent_us_node)
            )
            node_alias_candidates.setdefault(int(row.v), set()).add(
                int(row.matched_parent_ds_node)
            )
        stable_node_alias = {
            child_node: next(iter(parent_nodes))
            for child_node, parent_nodes in node_alias_candidates.items()
            if len(parent_nodes) == 1
        }
        prepared["solver_cache_u_alias"] = prepared["u"].map(
            lambda node: stable_node_alias.get(int(node), f"local:{int(node)}")
        )
        prepared["solver_cache_v_alias"] = prepared["v"].map(
            lambda node: stable_node_alias.get(int(node), f"local:{int(node)}")
        )
        prepared.loc[~exact, "solver_cache_edge_alias"] = prepared.loc[
            ~exact, "edge_id"
        ].map(lambda edge_id: f"local:{int(edge_id)}")
    return prepared, resolved_kb


def _empty_bubble_summary_frame() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "compound_bubble_id",
            "bubble_root_unit_id",
            "upstream_boundary_node",
            "downstream_boundary_node",
            "declared_root_upstream_boundary_node",
            "declared_root_downstream_boundary_node",
            "assembled_boundaries_differ_from_root",
            "n_member_units",
            "member_unit_ids",
            "n_closed_domain_connector_edges",
            "closed_domain_connector_edge_ids",
            "n_nodes",
            "n_edges",
            "solver_success",
            "solver_method",
            "solver_algorithm_version",
            "solver_backend_requested",
            "solver_backend_resolved",
            "solver_optimizer_success",
            "solver_solution_stage",
            "energy_stage_success",
            "energy_stage_message",
            "energy_stage_nit",
            "energy_stage_nfev",
            "energy_stage_residual_norm",
            "energy_stage_residual_relative",
            "polish_attempted",
            "polish_success",
            "polish_message",
            "polish_nfev",
            "polish_residual_norm",
            "polish_residual_relative",
            "sparse_min_internal_nodes",
            "solver_cache_hit",
            "solver_cache_key",
            "solver_cache_path",
            "solver_message",
            "solver_nfev",
            "solver_initialization",
            "solver_strict_success",
            "solver_conditional_acceptance",
            "solver_acceptance_tier",
            "candidate_K_relative_difference",
            "conditional_residual_relative_tolerance",
            "extended_residual_relative_allowance",
            "conditional_K_relative_tolerance",
            "primary_K_eq",
            "neutral_fallback_K_eq",
            "primary_solver_success",
            "primary_solver_nfev",
            "primary_residual_relative",
            "neutral_fallback_attempted",
            "neutral_fallback_solver_success",
            "neutral_fallback_solver_nfev",
            "neutral_fallback_residual_relative",
            "K_eq",
            "G_eq",
            "Q_unit",
            "residual_norm",
            "residual_relative",
            "resistance_law",
            "geometry_method",
            "kb_mode",
            "kb_value",
            "kb_source_method",
        ]
    )


def _empty_bubble_edge_frame() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "compound_bubble_id",
            "bubble_root_unit_id",
            "edge_id",
            "u",
            "v",
            "L",
            "w",
            "width_source",
            "k_b",
            "kb_source",
            "d",
            "A",
            "R_h",
            "n",
            "C_z",
            "K_e",
            "G_e",
            "q_unit",
            "dh",
        ]
    )


def _empty_bubble_node_frame() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "compound_bubble_id",
            "bubble_root_unit_id",
            "node_id",
            "head_norm",
            "is_boundary",
            "is_upstream_boundary",
            "is_downstream_boundary",
            "head_source",
            "continuity_residual",
        ]
    )


def _bubble_groups(units: Sequence[StructuralUnit]) -> list[tuple[int, list[StructuralUnit], StructuralUnit]]:
    grouped: dict[int, list[StructuralUnit]] = {}
    for unit in units:
        if unit.compound_bubble_id is None:
            continue
        grouped.setdefault(int(unit.compound_bubble_id), []).append(unit)

    bubble_specs: list[tuple[int, list[StructuralUnit], StructuralUnit]] = []
    for bubble_id, members in grouped.items():
        members = sorted(members, key=lambda unit: unit.unit_id)
        root_candidates = [unit for unit in members if unit.compound_bubble_role == "bubble_root"]
        root_unit = root_candidates[0] if root_candidates else members[0]
        bubble_specs.append((bubble_id, members, root_unit))
    return sorted(bubble_specs, key=lambda item: item[0])


def _validated_assembled_bubble_terminals(
    bubble_edges: pd.DataFrame,
    *,
    bubble_id: int,
) -> tuple[int, int]:
    graph = nx.DiGraph()
    graph.add_edges_from(
        (int(row.u), int(row.v)) for row in bubble_edges.itertuples(index=False)
    )
    sources = [int(node) for node in graph if graph.in_degree(node) == 0]
    sinks = [int(node) for node in graph if graph.out_degree(node) == 0]
    if len(sources) != 1 or len(sinks) != 1:
        raise ValueError(
            f"Assembled bubble {bubble_id} requires one directed source and sink; "
            f"found sources={sources}, sinks={sinks}."
        )
    source_node = sources[0]
    sink_node = sinks[0]
    if not nx.is_directed_acyclic_graph(graph):
        raise ValueError(f"Assembled bubble {bubble_id} is not a directed acyclic graph.")
    between_nodes = (
        (set(nx.descendants(graph, source_node)) | {source_node})
        & (set(nx.ancestors(graph, sink_node)) | {sink_node})
    )
    excluded = sorted(set(graph.nodes) - between_nodes)
    if excluded:
        raise ValueError(
            f"Assembled bubble {bubble_id} contains nodes outside its complete "
            f"source-to-sink domain: {excluded[:10]}."
        )
    return source_node, sink_node


def _complete_compound_bubble_edge_keys(
    prepared_edges: pd.DataFrame,
    members: Sequence[StructuralUnit],
) -> tuple[list[tuple[int, int, int]], list[int]]:
    """Return the smallest closed directed domain containing all member edges.

    Edge-overlap components can be connected through ordinary single-channel
    links that are not members of any detected structural unit. Omitting those
    connectors makes a scientifically closed compound appear to have several
    sinks. Singleton bubbles already carry their complete unit edge set and do
    not require domain expansion.
    """
    selected_keys = {
        (int(u), int(v), int(key))
        for unit in members
        for u, v, key in unit.edge_set
    }
    if len(members) <= 1 or not selected_keys:
        return sorted(selected_keys), []

    graph = nx.DiGraph()
    graph.add_edges_from(
        (int(row.u), int(row.v)) for row in prepared_edges.itertuples(index=False)
    )
    if not nx.is_directed_acyclic_graph(graph):
        raise ValueError("Complete compound-bubble domain requires a directed acyclic graph.")

    selected_nodes = {node for u, v, _ in selected_keys for node in (u, v)}
    common_ancestors = set(graph.nodes)
    common_descendants = set(graph.nodes)
    for node_id in selected_nodes:
        common_ancestors.intersection_update(nx.ancestors(graph, node_id) | {node_id})
        common_descendants.intersection_update(nx.descendants(graph, node_id) | {node_id})
    if not common_ancestors or not common_descendants:
        raise ValueError("Compound-bubble member edges have no common closed hydraulic domain.")

    node_order = {int(node): index for index, node in enumerate(nx.topological_sort(graph))}
    source_node = max(common_ancestors, key=lambda node: node_order[int(node)])
    sink_node = min(common_descendants, key=lambda node: node_order[int(node)])
    if any(
        int(node) != int(source_node)
        and not nx.has_path(graph, int(node), int(source_node))
        for node in common_ancestors
    ):
        raise ValueError("Compound-bubble domain has no unique latest common upstream boundary.")
    if any(
        int(node) != int(sink_node)
        and not nx.has_path(graph, int(sink_node), int(node))
        for node in common_descendants
    ):
        raise ValueError("Compound-bubble domain has no unique earliest common downstream boundary.")

    between_nodes = (
        (set(nx.descendants(graph, source_node)) | {source_node})
        & (set(nx.ancestors(graph, sink_node)) | {sink_node})
    )
    domain = prepared_edges.loc[
        prepared_edges["u"].astype(int).isin(between_nodes)
        & prepared_edges["v"].astype(int).isin(between_nodes)
    ]
    domain_keys = {
        (int(row.u), int(row.v), int(row.edge_id))
        for row in domain.itertuples(index=False)
    }
    missing_selected = sorted(selected_keys.difference(domain_keys))
    if missing_selected:
        raise ValueError(
            "Closed compound-bubble domain omitted selected member edges: "
            f"{missing_selected[:10]}."
        )
    connector_ids = sorted(key for _, _, key in domain_keys.difference(selected_keys))
    return sorted(domain_keys), connector_ids


def _select_stable_solver_attempt(
    primary: Mapping[str, Any],
    fallback: Mapping[str, Any],
    *,
    config: BubbleEquivalentResistanceConfig,
) -> tuple[dict[str, Any], str]:
    """Select the best numerical attempt and audit conditional acceptance.

    The nonlinear solver's native success criterion remains the strict result.
    A near-converged result is accepted only when both initializations produce
    a stable equivalent K.  This prevents a small residual alone from admitting
    a numerically ambiguous hydraulic reduction.
    """

    primary = dict(primary)
    fallback = dict(fallback)
    primary_success = bool(primary.get("solver_success", False))
    fallback_success = bool(fallback.get("solver_success", False))
    primary_residual = float(primary.get("residual_relative", np.nan))
    fallback_residual = float(fallback.get("residual_relative", np.nan))

    if fallback_success and not primary_success:
        selected = fallback
        selected_label = "neutral_internal_heads_fallback"
    elif primary_success and not fallback_success:
        selected = primary
        selected_label = "available_node_heads"
    elif np.isfinite(fallback_residual) and (
        not np.isfinite(primary_residual) or fallback_residual < primary_residual
    ):
        selected = fallback
        selected_label = "neutral_internal_heads_fallback"
    else:
        selected = primary
        selected_label = "available_node_heads"

    primary_k = float(primary.get("K_eq", np.nan))
    fallback_k = float(fallback.get("K_eq", np.nan))
    if (
        np.isfinite(primary_k)
        and np.isfinite(fallback_k)
        and primary_k > 0.0
        and fallback_k > 0.0
    ):
        k_relative_difference = abs(primary_k - fallback_k) / max(
            abs(primary_k), abs(fallback_k)
        )
    else:
        k_relative_difference = np.nan

    strict_success = bool(selected.get("solver_success", False))
    selected_residual = float(selected.get("residual_relative", np.nan))
    stable_k = bool(
        np.isfinite(k_relative_difference)
        and k_relative_difference <= config.conditional_k_relative_tolerance
    )
    conditional_success = bool(
        not strict_success
        and stable_k
        and np.isfinite(selected_residual)
        and selected_residual
        <= config.extended_residual_relative_allowance
    )
    if strict_success:
        acceptance_tier = "strict"
    elif conditional_success and (
        selected_residual <= config.conditional_residual_relative_tolerance
    ):
        acceptance_tier = "conditional_stable"
    elif conditional_success:
        acceptance_tier = "extended_conditional_stable"
    else:
        acceptance_tier = "rejected"

    selected["solver_success"] = bool(strict_success or conditional_success)
    selected.update(
        {
            "solver_strict_success": strict_success,
            "solver_conditional_acceptance": conditional_success,
            "solver_acceptance_tier": acceptance_tier,
            "candidate_K_relative_difference": float(k_relative_difference),
            "conditional_residual_relative_tolerance": float(
                config.conditional_residual_relative_tolerance
            ),
            "extended_residual_relative_allowance": float(
                config.extended_residual_relative_allowance
            ),
            "conditional_K_relative_tolerance": float(
                config.conditional_k_relative_tolerance
            ),
            "primary_K_eq": primary_k,
            "neutral_fallback_K_eq": fallback_k,
        }
    )
    return selected, selected_label


def _mask_failed_solver_values(selected: dict[str, Any]) -> None:
    selected["K_eq"] = np.nan
    selected["G_eq"] = np.nan
    selected["Q_unit"] = np.nan
    edge_flows = selected["edge_flows"].copy()
    edge_flows[["q_unit", "dh"]] = np.nan
    selected["edge_flows"] = edge_flows
    node_heads = selected["node_heads"].copy()
    for column in ("head_norm", "continuity_residual"):
        if column in node_heads.columns:
            node_heads[column] = np.nan
    selected["node_heads"] = node_heads


def _solve_with_neutral_head_fallback(
    *,
    bubble_nodes: pd.DataFrame,
    bubble_edges: pd.DataFrame,
    upstream_node: int,
    downstream_node: int,
    config: BubbleEquivalentResistanceConfig,
) -> dict[str, Any]:
    optional_guess = bubble_nodes if config.use_node_head_field_as_initial_guess else None
    primary_initialization = (
        "available_node_heads"
        if optional_guess is not None
        else "solver_default_initialization"
    )
    primary = solve_bubble_equivalent_K(
        bubble_nodes,
        bubble_edges,
        upstream_node=int(upstream_node),
        downstream_node=int(downstream_node),
        optional_H_guess=optional_guess,
        smooth=config.smooth,
        eps=config.smooth_eps,
        solver_backend=config.solver_backend,
        sparse_min_internal_nodes=config.sparse_min_internal_nodes,
        solver_cache_dir=config.solver_cache_dir,
    )
    primary = dict(primary)
    primary_success = bool(primary["solver_success"])
    primary_residual = float(primary["residual_relative"])
    primary_nfev = int(primary["solver_nfev"])
    if primary_success:
        primary.update(
            {
                "solver_strict_success": True,
                "solver_conditional_acceptance": False,
                "solver_acceptance_tier": "strict",
                "candidate_K_relative_difference": np.nan,
                "conditional_residual_relative_tolerance": float(
                    config.conditional_residual_relative_tolerance
                ),
                "extended_residual_relative_allowance": float(
                    config.extended_residual_relative_allowance
                ),
                "conditional_K_relative_tolerance": float(
                    config.conditional_k_relative_tolerance
                ),
                "primary_K_eq": float(primary.get("K_eq", np.nan)),
                "neutral_fallback_K_eq": np.nan,
                "solver_initialization": primary_initialization,
                "primary_solver_success": True,
                "primary_solver_nfev": primary_nfev,
                "primary_residual_relative": primary_residual,
                "neutral_fallback_attempted": False,
                "neutral_fallback_solver_success": pd.NA,
                "neutral_fallback_solver_nfev": pd.NA,
                "neutral_fallback_residual_relative": np.nan,
            }
        )
        return primary

    node_ids = pd.to_numeric(bubble_nodes["node_id"], errors="coerce").dropna().astype(int)
    neutral_guess = {int(node_id): 0.5 for node_id in node_ids}
    neutral_guess[int(upstream_node)] = 1.0
    neutral_guess[int(downstream_node)] = 0.0
    fallback = solve_bubble_equivalent_K(
        bubble_nodes,
        bubble_edges,
        upstream_node=int(upstream_node),
        downstream_node=int(downstream_node),
        optional_H_guess=neutral_guess,
        smooth=config.smooth,
        eps=config.smooth_eps,
        solver_backend=config.solver_backend,
        sparse_min_internal_nodes=config.sparse_min_internal_nodes,
        solver_cache_dir=config.solver_cache_dir,
    )
    fallback = dict(fallback)
    fallback_success = bool(fallback["solver_success"])
    fallback_residual = float(fallback["residual_relative"])
    fallback_nfev = int(fallback["solver_nfev"])
    selected, selected_initialization = _select_stable_solver_attempt(
        primary,
        fallback,
        config=config,
    )
    if selected_initialization == "available_node_heads":
        selected_initialization = primary_initialization
    selected.update(
        {
            "solver_initialization": selected_initialization,
            "primary_solver_success": False,
            "primary_solver_nfev": primary_nfev,
            "primary_residual_relative": primary_residual,
            "neutral_fallback_attempted": True,
            "neutral_fallback_solver_success": fallback_success,
            "neutral_fallback_solver_nfev": fallback_nfev,
            "neutral_fallback_residual_relative": fallback_residual,
        }
    )
    if not bool(selected["solver_success"]):
        _mask_failed_solver_values(selected)
        selected["solver_message"] = (
            f"Primary solve failed ({primary.get('solver_message', '')}); "
            f"neutral fallback failed ({fallback.get('solver_message', '')})."
        )
    return selected


def compute_bubble_equivalent_metrics_from_units(
    links: pd.DataFrame,
    nodes: pd.DataFrame,
    units: Sequence[StructuralUnit],
    *,
    config: BubbleEquivalentResistanceConfig | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Reduce structural-unit bubbles to equivalent geometry-informed resistance.

    The equivalent resistance predictor is computed from geometry-derived edge
    resistance only. Model discharge Q, hydrograph outputs, and routed peak
    timing must not be used in the predictor calculation.
    """

    if not units:
        return _empty_bubble_summary_frame(), _empty_bubble_edge_frame(), _empty_bubble_node_frame()

    config = config or BubbleEquivalentResistanceConfig()
    prepared_edges, resolved_kb = prepare_bubble_resistance_edges(links, config=config)
    prepared_nodes = _canonicalize_nodes(nodes)

    annotate_unit_context(list(units))
    bubble_specs = _bubble_groups(units)
    if not bubble_specs:
        return _empty_bubble_summary_frame(), _empty_bubble_edge_frame(), _empty_bubble_node_frame()

    edge_lookup = prepared_edges.set_index(["u", "v", "edge_id"], drop=False)
    bubble_rows: list[dict[str, Any]] = []
    edge_rows: list[pd.DataFrame] = []
    node_rows: list[pd.DataFrame] = []

    for bubble_id, members, root_unit in bubble_specs:
        bubble_edge_keys, connector_edge_ids = _complete_compound_bubble_edge_keys(
            prepared_edges,
            members,
        )
        missing_keys = [key for key in bubble_edge_keys if key not in edge_lookup.index]
        if missing_keys:
            raise ValueError(
                f"Could not resolve link rows for bubble {bubble_id}. Missing edge keys: {missing_keys[:10]}"
            )

        bubble_edges = edge_lookup.loc[bubble_edge_keys].copy()
        if isinstance(bubble_edges, pd.Series):
            bubble_edges = bubble_edges.to_frame().T
        bubble_edges = bubble_edges.reset_index(drop=True)
        bubble_node_ids = sorted({int(node_id) for unit in members for node_id in unit.node_set} | set(bubble_edges["u"]) | set(bubble_edges["v"]))
        bubble_nodes = prepared_nodes.loc[prepared_nodes["node_id"].isin(bubble_node_ids)].copy()
        if len(bubble_nodes) < len(bubble_node_ids):
            missing_node_ids = sorted(set(bubble_node_ids).difference(set(bubble_nodes["node_id"].astype(int).tolist())))
            bubble_nodes = pd.concat(
                [bubble_nodes, pd.DataFrame({"node_id": pd.Series(missing_node_ids, dtype="Int64")})],
                ignore_index=True,
            )

        assembled_upstream_node, assembled_downstream_node = (
            _validated_assembled_bubble_terminals(
                bubble_edges,
                bubble_id=bubble_id,
            )
        )
        declared_upstream_node = int(root_unit.bifurcation)
        declared_downstream_node = int(root_unit.confluence)
        solve_result = _solve_with_neutral_head_fallback(
            bubble_nodes=bubble_nodes,
            bubble_edges=bubble_edges[
                [
                    "edge_id",
                    "u",
                    "v",
                    "K_e",
                    "solver_cache_edge_alias",
                    "solver_cache_u_alias",
                    "solver_cache_v_alias",
                ]
            ],
            upstream_node=assembled_upstream_node,
            downstream_node=assembled_downstream_node,
            config=config,
        )

        bubble_rows.append(
            {
                "compound_bubble_id": bubble_id,
                "bubble_root_unit_id": int(root_unit.unit_id),
                "upstream_boundary_node": assembled_upstream_node,
                "downstream_boundary_node": assembled_downstream_node,
                "declared_root_upstream_boundary_node": declared_upstream_node,
                "declared_root_downstream_boundary_node": declared_downstream_node,
                "assembled_boundaries_differ_from_root": bool(
                    assembled_upstream_node != declared_upstream_node
                    or assembled_downstream_node != declared_downstream_node
                ),
                "n_member_units": int(len(members)),
                "member_unit_ids": ",".join(str(unit.unit_id) for unit in members),
                "n_closed_domain_connector_edges": int(len(connector_edge_ids)),
                "closed_domain_connector_edge_ids": ",".join(
                    str(edge_id) for edge_id in connector_edge_ids
                ),
                "n_nodes": int(solve_result["n_nodes"]),
                "n_edges": int(solve_result["n_edges"]),
                "solver_success": bool(solve_result["solver_success"]),
                "solver_method": solve_result["solver_method"],
                "solver_algorithm_version": solve_result.get(
                    "solver_algorithm_version", ""
                ),
                "solver_backend_requested": solve_result.get(
                    "solver_backend_requested", config.solver_backend
                ),
                "solver_backend_resolved": solve_result.get(
                    "solver_backend_resolved", ""
                ),
                "solver_optimizer_success": bool(
                    solve_result.get("solver_optimizer_success", False)
                ),
                "solver_solution_stage": solve_result.get(
                    "solver_solution_stage", ""
                ),
                "energy_stage_success": solve_result.get("energy_stage_success"),
                "energy_stage_message": solve_result.get("energy_stage_message", ""),
                "energy_stage_nit": solve_result.get("energy_stage_nit", 0),
                "energy_stage_nfev": solve_result.get("energy_stage_nfev", 0),
                "energy_stage_residual_norm": solve_result.get(
                    "energy_stage_residual_norm", np.nan
                ),
                "energy_stage_residual_relative": solve_result.get(
                    "energy_stage_residual_relative", np.nan
                ),
                "polish_attempted": bool(solve_result.get("polish_attempted", False)),
                "polish_success": solve_result.get("polish_success"),
                "polish_message": solve_result.get("polish_message", ""),
                "polish_nfev": solve_result.get("polish_nfev", 0),
                "polish_residual_norm": solve_result.get(
                    "polish_residual_norm", np.nan
                ),
                "polish_residual_relative": solve_result.get(
                    "polish_residual_relative", np.nan
                ),
                "sparse_min_internal_nodes": solve_result.get(
                    "sparse_min_internal_nodes", config.sparse_min_internal_nodes
                ),
                "solver_cache_hit": bool(solve_result.get("solver_cache_hit", False)),
                "solver_cache_key": solve_result.get("solver_cache_key", ""),
                "solver_cache_path": solve_result.get("solver_cache_path", ""),
                "solver_message": solve_result["solver_message"],
                "solver_nfev": int(solve_result["solver_nfev"]),
                "solver_initialization": solve_result["solver_initialization"],
                "solver_strict_success": solve_result["solver_strict_success"],
                "solver_conditional_acceptance": solve_result[
                    "solver_conditional_acceptance"
                ],
                "solver_acceptance_tier": solve_result["solver_acceptance_tier"],
                "candidate_K_relative_difference": solve_result[
                    "candidate_K_relative_difference"
                ],
                "conditional_residual_relative_tolerance": solve_result[
                    "conditional_residual_relative_tolerance"
                ],
                "extended_residual_relative_allowance": solve_result[
                    "extended_residual_relative_allowance"
                ],
                "conditional_K_relative_tolerance": solve_result[
                    "conditional_K_relative_tolerance"
                ],
                "primary_K_eq": solve_result["primary_K_eq"],
                "neutral_fallback_K_eq": solve_result["neutral_fallback_K_eq"],
                "primary_solver_success": solve_result["primary_solver_success"],
                "primary_solver_nfev": solve_result["primary_solver_nfev"],
                "primary_residual_relative": solve_result["primary_residual_relative"],
                "neutral_fallback_attempted": solve_result["neutral_fallback_attempted"],
                "neutral_fallback_solver_success": solve_result[
                    "neutral_fallback_solver_success"
                ],
                "neutral_fallback_solver_nfev": solve_result[
                    "neutral_fallback_solver_nfev"
                ],
                "neutral_fallback_residual_relative": solve_result[
                    "neutral_fallback_residual_relative"
                ],
                "K_eq": float(solve_result["K_eq"]),
                "G_eq": float(solve_result["G_eq"]),
                "Q_unit": float(solve_result["Q_unit"]),
                "residual_norm": float(solve_result["residual_norm"]),
                "residual_relative": float(solve_result["residual_relative"]),
                "resistance_law": str(config.resistance_law),
                "geometry_method": str(config.geometry_method),
                "kb_mode": resolved_kb.kb_mode,
                "kb_value": float(resolved_kb.kb_value),
                "kb_source_method": resolved_kb.kb_source_method,
            }
        )

        edge_result = bubble_edges.merge(
            solve_result["edge_flows"][["edge_id", "q_unit", "dh"]],
            on="edge_id",
            how="left",
        )
        edge_result.insert(0, "compound_bubble_id", bubble_id)
        edge_result.insert(1, "bubble_root_unit_id", int(root_unit.unit_id))
        edge_rows.append(
            edge_result[
                [
                    "compound_bubble_id",
                    "bubble_root_unit_id",
                    "edge_id",
                    "u",
                    "v",
                    "L",
                    "w",
                    "width_source",
                    "k_b",
                    "kb_source",
                    "d",
                    "A",
                    "R_h",
                    "n",
                    "C_z",
                    "K_e",
                    "G_e",
                    "q_unit",
                    "dh",
                ]
            ].copy()
        )

        node_result = bubble_nodes.merge(
            solve_result["node_heads"],
            on="node_id",
            how="right",
        )
        node_result.insert(0, "compound_bubble_id", bubble_id)
        node_result.insert(1, "bubble_root_unit_id", int(root_unit.unit_id))
        head_guess_field = _resolve_first_present(node_result.columns, NODE_HEAD_FIELD_CANDIDATES)
        if head_guess_field is not None:
            node_result["head_guess_source_field"] = head_guess_field
            node_result["head_guess_value"] = pd.to_numeric(node_result[head_guess_field], errors="coerce")
        else:
            node_result["head_guess_source_field"] = ""
            node_result["head_guess_value"] = np.nan
        node_rows.append(node_result)

    bubble_summary = pd.DataFrame.from_records(bubble_rows).sort_values("compound_bubble_id").reset_index(drop=True)
    bubble_summary.attrs["equivalent_resistance_config"] = asdict(config)
    bubble_summary.attrs["resolved_kb_context"] = resolved_kb.to_dict()

    bubble_edges = (
        pd.concat(edge_rows, ignore_index=True)
        if edge_rows
        else _empty_bubble_edge_frame()
    )
    bubble_edges = bubble_edges.sort_values(["compound_bubble_id", "edge_id"], kind="mergesort").reset_index(drop=True)
    bubble_edges.attrs["equivalent_resistance_config"] = asdict(config)
    bubble_edges.attrs["resolved_kb_context"] = resolved_kb.to_dict()

    bubble_nodes = (
        pd.concat(node_rows, ignore_index=True)
        if node_rows
        else _empty_bubble_node_frame()
    )
    preferred_columns = [
        "compound_bubble_id",
        "bubble_root_unit_id",
        "node_id",
        "head_norm",
        "is_boundary",
        "is_upstream_boundary",
        "is_downstream_boundary",
        "head_source",
        "continuity_residual",
        "head_guess_source_field",
        "head_guess_value",
    ]
    remaining_columns = [column for column in bubble_nodes.columns if column not in preferred_columns]
    bubble_nodes = bubble_nodes[preferred_columns + remaining_columns].sort_values(
        ["compound_bubble_id", "node_id"],
        kind="mergesort",
    ).reset_index(drop=True)
    bubble_nodes.attrs["equivalent_resistance_config"] = asdict(config)
    bubble_nodes.attrs["resolved_kb_context"] = resolved_kb.to_dict()

    return bubble_summary, bubble_edges, bubble_nodes


def compute_bubble_equivalent_metrics(
    links_path: str | PathLike[str],
    nodes_path: str | PathLike[str],
    *,
    max_path_cutoff: int = 100,
    max_paths: int = 5000,
    debug: bool = False,
    config: BubbleEquivalentResistanceConfig | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    links, nodes = load_network(links_path, nodes_path)
    from hierarchy_level_definition.unit_detection.bifurcation_confluence_units import analyze_network

    _, units, _ = analyze_network(
        links_path,
        nodes_path,
        max_path_cutoff=max_path_cutoff,
        max_paths=max_paths,
        debug=debug,
    )
    return compute_bubble_equivalent_metrics_from_units(
        links,
        nodes,
        units,
        config=config,
    )


__all__ = [
    "BubbleEquivalentResistanceConfig",
    "compute_bubble_equivalent_metrics",
    "compute_bubble_equivalent_metrics_from_units",
    "prepare_bubble_resistance_edges",
]
