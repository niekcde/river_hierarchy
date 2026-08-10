from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd

from hierarchy_level_definition.metrics import BubbleEquivalentResistanceConfig
from hierarchy_level_definition.metrics.bubble_equivalent_metrics import (
    prepare_bubble_resistance_edges,
)
from hierarchy_level_definition.metrics.equivalent_resistance import (
    solve_bubble_equivalent_K,
)
from network_variants.equivalent_resistance_experiment import (
    _compute_equivalent_resistance,
)


@dataclass(frozen=True)
class HydraulicDomain:
    domain_type: str
    source_node: int
    sink_node: int
    links: pd.DataFrame
    selected_unit_ids: tuple[int, ...]
    compound_bubble_id: int | None
    selected_unit_is_closed: bool
    selected_unit_violating_link_ids: tuple[int, ...]


def _int_values(values: pd.Series | Sequence[Any]) -> tuple[int, ...]:
    numeric = pd.to_numeric(pd.Series(values), errors="coerce").dropna().astype(int)
    return tuple(dict.fromkeys(numeric.tolist()))


def extract_complete_directed_domain(
    full_links: pd.DataFrame,
    source_node: int,
    sink_node: int,
) -> pd.DataFrame:
    """Return every directed edge lying on a source-to-sink path."""
    source_node = int(source_node)
    sink_node = int(sink_node)
    graph = nx.DiGraph()
    graph.add_edges_from(
        (int(row.id_us_node), int(row.id_ds_node))
        for row in full_links.itertuples(index=False)
    )
    if source_node not in graph or sink_node not in graph:
        raise ValueError(
            f"Hydraulic-domain anchors {source_node}->{sink_node} are absent from the network."
        )
    if not nx.has_path(graph, source_node, sink_node):
        raise ValueError(
            f"No directed path exists between hydraulic-domain anchors "
            f"{source_node}->{sink_node}."
        )
    domain_nodes = (
        (set(nx.descendants(graph, source_node)) | {source_node})
        & (set(nx.ancestors(graph, sink_node)) | {sink_node})
    )
    return full_links.loc[
        pd.to_numeric(full_links["id_us_node"], errors="coerce").isin(domain_nodes)
        & pd.to_numeric(full_links["id_ds_node"], errors="coerce").isin(domain_nodes)
    ].copy()


def audit_hydraulic_domain_closure(
    domain_links: pd.DataFrame,
    full_links: pd.DataFrame,
    source_node: int,
    sink_node: int,
) -> dict[str, Any]:
    """Check whether excluded edges exchange flow with a two-terminal domain."""
    source_node = int(source_node)
    sink_node = int(sink_node)
    domain_link_ids = set(_int_values(domain_links["id_link"]))
    domain_nodes = set(_int_values(domain_links["id_us_node"])) | set(
        _int_values(domain_links["id_ds_node"])
    )
    internal_nodes = domain_nodes - {source_node, sink_node}
    violations: list[dict[str, Any]] = []
    excluded = full_links.loc[
        ~pd.to_numeric(full_links["id_link"], errors="coerce").isin(domain_link_ids)
    ]
    for row in excluded.itertuples(index=False):
        link_id = int(row.id_link)
        upstream = int(row.id_us_node)
        downstream = int(row.id_ds_node)
        reason: str | None = None
        if upstream in internal_nodes or downstream in internal_nodes:
            reason = "external_edge_touches_internal_node"
        elif upstream == source_node:
            reason = "external_outflow_from_source"
        elif downstream == sink_node:
            reason = "external_inflow_to_sink"
        if reason is not None:
            violations.append(
                {
                    "id_link": link_id,
                    "id_us_node": upstream,
                    "id_ds_node": downstream,
                    "reason": reason,
                }
            )
    return {
        "is_closed": not violations,
        "source_node": source_node,
        "sink_node": sink_node,
        "n_links": len(domain_link_ids),
        "n_nodes": len(domain_nodes),
        "violations": pd.DataFrame.from_records(violations),
    }


def _unique_compound_terminals(
    bubble_edge_resistance: pd.DataFrame,
    compound_bubble_id: int,
) -> tuple[int, int]:
    edges = bubble_edge_resistance.loc[
        pd.to_numeric(
            bubble_edge_resistance["compound_bubble_id"], errors="coerce"
        ).eq(int(compound_bubble_id))
    ]
    if edges.empty:
        raise ValueError(
            f"No resistance edges exist for compound bubble {compound_bubble_id}."
        )
    graph = nx.DiGraph()
    graph.add_edges_from(
        (int(row.u), int(row.v)) for row in edges.itertuples(index=False)
    )
    sources = [int(node) for node in graph if graph.in_degree(node) == 0]
    sinks = [int(node) for node in graph if graph.out_degree(node) == 0]
    if len(sources) != 1 or len(sinks) != 1:
        raise ValueError(
            f"Compound bubble {compound_bubble_id} does not have unique directed "
            f"terminals (sources={sources}, sinks={sinks})."
        )
    return sources[0], sinks[0]


def select_smallest_closed_parent_domain(
    *,
    full_links: pd.DataFrame,
    unit_summary: pd.DataFrame,
    unit_link_membership: pd.DataFrame,
    bubble_edge_resistance: pd.DataFrame,
    selected_unit_ids: Sequence[int],
    selected_compound_bubble_id: int | None = None,
) -> HydraulicDomain:
    selected_unit_ids = _int_values(selected_unit_ids)
    if not selected_unit_ids:
        raise ValueError("A selected hydraulic domain requires at least one unit ID.")

    units = unit_summary.loc[
        pd.to_numeric(unit_summary["unit_id"], errors="coerce").isin(selected_unit_ids)
    ].copy()
    if len(units) != len(selected_unit_ids):
        found = set(_int_values(units["unit_id"]))
        missing = sorted(set(selected_unit_ids) - found)
        raise ValueError(f"Selected units are absent from the parent hierarchy: {missing}.")

    selected_link_ids = _int_values(
        unit_link_membership.loc[
            pd.to_numeric(unit_link_membership["unit_id"], errors="coerce").isin(
                selected_unit_ids
            ),
            "id_link",
        ]
    )
    selected_links = full_links.loc[
        pd.to_numeric(full_links["id_link"], errors="coerce").isin(selected_link_ids)
    ].copy()
    if selected_links.empty:
        raise ValueError(
            f"No parent links were resolved for selected units {selected_unit_ids}."
        )

    if selected_compound_bubble_id is not None:
        selected_source, selected_sink = _unique_compound_terminals(
            bubble_edge_resistance, int(selected_compound_bubble_id)
        )
    elif len(selected_unit_ids) == 1:
        selected_source = int(units.iloc[0]["bifurcation"])
        selected_sink = int(units.iloc[0]["confluence"])
    else:
        compound_ids = _int_values(units.get("compound_bubble_id", pd.Series(dtype=float)))
        if len(compound_ids) != 1:
            raise ValueError(
                "A multi-unit selection requires one containing compound bubble."
            )
        selected_compound_bubble_id = compound_ids[0]
        selected_source, selected_sink = _unique_compound_terminals(
            bubble_edge_resistance, selected_compound_bubble_id
        )

    selected_audit = audit_hydraulic_domain_closure(
        selected_links, full_links, selected_source, selected_sink
    )
    violating_link_ids = tuple(
        _int_values(selected_audit["violations"].get("id_link", pd.Series(dtype=float)))
    )
    if selected_audit["is_closed"]:
        return HydraulicDomain(
            domain_type=(
                "closed_compound_unit"
                if selected_compound_bubble_id is not None or len(selected_unit_ids) > 1
                else "closed_unit"
            ),
            source_node=selected_source,
            sink_node=selected_sink,
            links=selected_links,
            selected_unit_ids=selected_unit_ids,
            compound_bubble_id=(
                int(selected_compound_bubble_id)
                if selected_compound_bubble_id is not None
                else None
            ),
            selected_unit_is_closed=True,
            selected_unit_violating_link_ids=(),
        )

    compound_ids = _int_values(units.get("compound_bubble_id", pd.Series(dtype=float)))
    if selected_compound_bubble_id is not None:
        compound_ids = (int(selected_compound_bubble_id),)
    if len(compound_ids) != 1:
        raise ValueError(
            f"Selected units {selected_unit_ids} are not hydraulically closed and do "
            "not resolve to one containing compound bubble."
        )
    compound_bubble_id = compound_ids[0]
    source_node, sink_node = _unique_compound_terminals(
        bubble_edge_resistance, compound_bubble_id
    )
    compound_links = extract_complete_directed_domain(
        full_links, source_node, sink_node
    )
    compound_audit = audit_hydraulic_domain_closure(
        compound_links, full_links, source_node, sink_node
    )
    if not compound_audit["is_closed"]:
        compound_violations = _int_values(
            compound_audit["violations"].get("id_link", pd.Series(dtype=float))
        )
        raise ValueError(
            f"Containing compound bubble {compound_bubble_id} is not hydraulically "
            f"closed; violating links={compound_violations}."
        )
    return HydraulicDomain(
        domain_type="containing_compound_context",
        source_node=source_node,
        sink_node=sink_node,
        links=compound_links,
        selected_unit_ids=selected_unit_ids,
        compound_bubble_id=compound_bubble_id,
        selected_unit_is_closed=False,
        selected_unit_violating_link_ids=violating_link_ids,
    )


def load_stable_parent_to_child_node_map(
    node_match_path: str | Path,
) -> dict[int, int]:
    node_match = pd.read_csv(node_match_path)
    required = {"child_id_node", "matched_parent_node_id"}
    if not required.issubset(node_match.columns):
        raise ValueError(
            f"Node-match table is missing columns: {sorted(required - set(node_match.columns))}."
        )
    child = pd.to_numeric(node_match["child_id_node"], errors="coerce")
    parent = pd.to_numeric(node_match["matched_parent_node_id"], errors="coerce")
    valid = child.notna() & parent.notna()
    if "match_within_tolerance" in node_match.columns:
        valid &= node_match["match_within_tolerance"].fillna(False).astype(bool)
    candidates = node_match.loc[valid].copy()
    candidates["_child"] = child.loc[valid].astype(int)
    candidates["_parent"] = parent.loc[valid].astype(int)
    sort_columns = ["_parent"]
    if "match_distance" in candidates.columns:
        sort_columns.append("match_distance")
    sort_columns.append("_child")
    candidates = candidates.sort_values(sort_columns, kind="mergesort")
    candidates = candidates.drop_duplicates("_parent", keep="first")
    return dict(zip(candidates["_parent"], candidates["_child"]))


def load_parent_to_child_anchor_map(
    *,
    node_match_path: str | Path,
    child_links: pd.DataFrame,
    link_lineage_path: str | Path | None = None,
) -> tuple[dict[int, int], dict[int, str]]:
    """Resolve parent anchors by stable nodes, then directed link lineage.

    Collapse can remove a parent boundary node even though the new collapsed
    link retains that physical boundary as one of its endpoints. Link lineage
    records those parent upstream/downstream endpoints and provides the
    topology-preserving fallback when point matching has no stable row.
    """
    resolved = load_stable_parent_to_child_node_map(node_match_path)
    methods = {parent: "stable_node_match" for parent in resolved}
    if link_lineage_path is None or not Path(link_lineage_path).exists():
        return resolved, methods

    lineage = pd.read_csv(link_lineage_path)
    required = {"id_link", "matched_parent_us_node", "matched_parent_ds_node"}
    if not required.issubset(lineage.columns):
        return resolved, methods
    endpoints = child_links[["id_link", "id_us_node", "id_ds_node"]].copy()
    endpoints["id_link"] = pd.to_numeric(endpoints["id_link"], errors="coerce")
    lineage["id_link"] = pd.to_numeric(lineage["id_link"], errors="coerce")
    joined = lineage.merge(endpoints, on="id_link", how="inner")

    candidates: dict[int, list[int]] = {}
    for row in joined.itertuples(index=False):
        for parent_value, child_value in (
            (row.matched_parent_us_node, row.id_us_node),
            (row.matched_parent_ds_node, row.id_ds_node),
        ):
            if pd.isna(parent_value) or pd.isna(child_value):
                continue
            candidates.setdefault(int(parent_value), []).append(int(child_value))

    for parent_node, child_candidates in candidates.items():
        if parent_node in resolved:
            continue
        counts = pd.Series(child_candidates, dtype=int).value_counts()
        if counts.empty:
            continue
        highest_count = int(counts.iloc[0])
        winners = sorted(int(node) for node in counts.loc[counts.eq(highest_count)].index)
        if len(winners) != 1:
            continue
        resolved[parent_node] = winners[0]
        methods[parent_node] = "directed_link_lineage_endpoint"
    return resolved, methods


def build_matched_child_domain(
    *,
    parent_domain: HydraulicDomain,
    child_links: pd.DataFrame,
    parent_to_child_node: Mapping[int, int],
) -> HydraulicDomain:
    missing = [
        node
        for node in (parent_domain.source_node, parent_domain.sink_node)
        if int(node) not in parent_to_child_node
    ]
    if missing:
        raise ValueError(
            f"No stable child-node match exists for parent hydraulic anchors {missing}."
        )
    source_node = int(parent_to_child_node[parent_domain.source_node])
    sink_node = int(parent_to_child_node[parent_domain.sink_node])
    links = extract_complete_directed_domain(child_links, source_node, sink_node)
    audit = audit_hydraulic_domain_closure(
        links, child_links, source_node, sink_node
    )
    if not audit["is_closed"]:
        violating = _int_values(
            audit["violations"].get("id_link", pd.Series(dtype=float))
        )
        raise ValueError(
            f"Matched child hydraulic domain {source_node}->{sink_node} is not closed; "
            f"violating links={violating}."
        )
    return HydraulicDomain(
        domain_type=f"matched_{parent_domain.domain_type}",
        source_node=source_node,
        sink_node=sink_node,
        links=links,
        selected_unit_ids=parent_domain.selected_unit_ids,
        compound_bubble_id=parent_domain.compound_bubble_id,
        selected_unit_is_closed=parent_domain.selected_unit_is_closed,
        selected_unit_violating_link_ids=parent_domain.selected_unit_violating_link_ids,
    )


def solve_hydraulic_domain(
    *,
    domain: HydraulicDomain,
    nodes: pd.DataFrame,
    config: BubbleEquivalentResistanceConfig,
) -> tuple[dict[str, Any], Mapping[str, Any] | None]:
    """Solve with available heads first and a neutral internal-head fallback."""
    node_ids = set(_int_values(domain.links["id_us_node"])) | set(
        _int_values(domain.links["id_ds_node"])
    )
    node_subset = nodes.loc[
        pd.to_numeric(nodes["id_node"], errors="coerce").isin(node_ids)
    ].copy()
    primary, resolved_kb = _compute_equivalent_resistance(
        domain.links,
        node_subset,
        upstream_node=domain.source_node,
        downstream_node=domain.sink_node,
        config=config,
    )
    primary = dict(primary)
    primary_success = bool(primary.get("solver_success", False))
    primary_residual = primary.get("residual_relative", np.nan)
    if primary_success:
        primary.update(
            {
                "initialization_used": "available_node_heads",
                "primary_solver_success": True,
                "primary_residual_relative": primary_residual,
            }
        )
        return primary, resolved_kb

    prepared_edges, fallback_kb = prepare_bubble_resistance_edges(
        domain.links, config=config
    )
    incident_nodes = set(prepared_edges["u"]) | set(prepared_edges["v"])
    fallback_nodes = node_subset.loc[
        pd.to_numeric(node_subset["id_node"], errors="coerce").isin(incident_nodes)
    ]
    neutral_guess = {int(node): 0.5 for node in incident_nodes}
    neutral_guess[domain.source_node] = 1.0
    neutral_guess[domain.sink_node] = 0.0
    fallback = solve_bubble_equivalent_K(
        fallback_nodes,
        prepared_edges,
        upstream_node=domain.source_node,
        downstream_node=domain.sink_node,
        optional_H_guess=neutral_guess,
        smooth=config.smooth,
        eps=config.smooth_eps,
        solver_backend=config.solver_backend,
        sparse_min_internal_nodes=config.sparse_min_internal_nodes,
        solver_cache_dir=config.solver_cache_dir,
    )
    fallback = dict(fallback)
    fallback.update(
        {
            "initialization_used": "neutral_internal_heads_fallback",
            "primary_solver_success": False,
            "primary_residual_relative": primary_residual,
        }
    )
    return fallback, fallback_kb.to_dict()


def read_network(path: str | Path) -> gpd.GeoDataFrame:
    return gpd.read_file(path)


__all__ = [
    "HydraulicDomain",
    "audit_hydraulic_domain_closure",
    "build_matched_child_domain",
    "extract_complete_directed_domain",
    "load_stable_parent_to_child_node_map",
    "load_parent_to_child_anchor_map",
    "read_network",
    "select_smallest_closed_parent_domain",
    "solve_hydraulic_domain",
]
