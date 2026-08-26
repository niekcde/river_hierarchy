"""Phase-5 conservative merge of a raw local RivGraph into its parent graph.

The cut barriers are computational boundaries only.  The raw graph is first
trimmed to the exact selected-unit interfaces; only those scientific boundary
nodes are then matched to the parent.  Temporary cut terminals must never be
forced onto parent nodes because doing so can create duplicate matches or make
valid multi-terminal computational context appear ambiguous.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd
import rasterio
from shapely.geometry import LineString, MultiLineString, Point
from shapely.ops import linemerge, substring

from hierarchy_level_definition.graph_building.directed_network_checks import (
    build_directed_graph,
    parse_id_nodes,
    validate_single_inlet_single_outlet,
)
from network_variants.local_patch import LocalPatchContract
from network_variants.local_patch_raster import LocalPatchRasterResult
from network_variants.local_patch_variant import (
    LocalPatchVariantResult,
    validate_cut_terminal_graph,
)
from network_variants.variant_generation import (
    _assign_variant_directions,
    _boundary_harmonic_orientation,
    _infer_parent_node_order,
    _match_child_links_to_parent_links,
    _match_child_nodes_to_parent_nodes,
)


LOCAL_PATCH_MERGE_VERSION = "topology-local-patch-merge-v16"


@dataclass(frozen=True, slots=True)
class LocalPatchMergeResult:
    status: str
    output_paths: Mapping[str, str]
    summary: Mapping[str, Any]
    merge_version: str = LOCAL_PATCH_MERGE_VERSION

    def to_manifest(self) -> dict[str, Any]:
        return {
            "merge_version": self.merge_version,
            "status": self.status,
            "output_paths": dict(self.output_paths),
            "summary": dict(self.summary),
        }


def _linear_geometry(geometry):
    if isinstance(geometry, LineString):
        return geometry
    if isinstance(geometry, MultiLineString):
        merged = linemerge(geometry)
        if isinstance(merged, LineString):
            return merged
    raise ValueError("Local merge requires one continuous LineString per link.")


def _geometry_equivalence_metrics(
    first_geometry,
    second_geometry,
    *,
    tolerance_m: float,
) -> tuple[float, float, float]:
    """Measure symmetric geometric agreement for a possible duplicate link."""
    first = _linear_geometry(first_geometry)
    second = _linear_geometry(second_geometry)
    tolerance = max(float(tolerance_m), 1e-9)
    first_length = max(float(first.length), 1e-9)
    second_length = max(float(second.length), 1e-9)
    return (
        float(first.hausdorff_distance(second)),
        float(first.intersection(second.buffer(tolerance)).length / first_length),
        float(second.intersection(first.buffer(tolerance)).length / second_length),
    )


def _geometries_are_equivalent(
    first_geometry,
    second_geometry,
    *,
    tolerance_m: float,
    minimum_overlap_fraction: float,
) -> tuple[bool, float, float, float]:
    hausdorff, first_coverage, second_coverage = _geometry_equivalence_metrics(
        first_geometry,
        second_geometry,
        tolerance_m=tolerance_m,
    )
    equivalent = bool(
        hausdorff <= max(float(tolerance_m), 1e-9)
        and first_coverage >= float(minimum_overlap_fraction)
        and second_coverage >= float(minimum_overlap_fraction)
    )
    return equivalent, hausdorff, first_coverage, second_coverage


def _prune_raw_local_to_cut_corridor(
    links: gpd.GeoDataFrame,
    nodes: gpd.GeoDataFrame,
    *,
    source_node_id: int,
    sink_node_id: int,
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, dict[str, Any]]:
    """Remove dangling raw branches that cannot lie on a cut-to-cut path."""
    graph = nx.MultiGraph()
    for row in links.itertuples(index=False):
        node_a, node_b = parse_id_nodes(row.id_nodes)
        graph.add_edge(int(node_a), int(node_b), key=int(row.id_link))
    source_node_id, sink_node_id = int(source_node_id), int(sink_node_id)
    if source_node_id not in graph or sink_node_id not in graph:
        raise ValueError("Cut corridor terminals are absent from the raw local graph.")
    if not nx.has_path(graph, source_node_id, sink_node_id):
        raise ValueError("Raw local graph has no undirected path between its cut terminals.")

    queue = [
        int(node_id)
        for node_id in graph.nodes
        if int(node_id) not in {source_node_id, sink_node_id} and graph.degree(node_id) <= 1
    ]
    removed_node_ids: list[int] = []
    while queue:
        node_id = int(queue.pop())
        if (
            node_id not in graph
            or node_id in {source_node_id, sink_node_id}
            or graph.degree(node_id) > 1
        ):
            continue
        neighbors = [int(value) for value in graph.neighbors(node_id)]
        graph.remove_node(node_id)
        removed_node_ids.append(node_id)
        for neighbor in neighbors:
            if (
                neighbor in graph
                and neighbor not in {source_node_id, sink_node_id}
                and graph.degree(neighbor) <= 1
            ):
                queue.append(neighbor)

    if source_node_id not in graph or sink_node_id not in graph or not nx.has_path(
        graph, source_node_id, sink_node_id
    ):
        raise ValueError("Dangling-branch pruning removed the prescribed cut-to-cut path.")
    retained_node_ids = {int(value) for value in nx.node_connected_component(graph, source_node_id)}
    retained_link_ids = {
        int(key)
        for node_a, node_b, key in graph.edges(keys=True)
        if int(node_a) in retained_node_ids and int(node_b) in retained_node_ids
    }
    removed_link_ids = sorted(
        set(links["id_link"].astype(int)).difference(retained_link_ids)
    )
    pruned_links = links.loc[links["id_link"].astype(int).isin(retained_link_ids)].copy()
    pruned_nodes = nodes.loc[nodes["id_node"].astype(int).isin(retained_node_ids)].copy()
    return pruned_links, pruned_nodes, {
        "method": "iterative_nonboundary_degree_one_pruning",
        "source_node_id": source_node_id,
        "sink_node_id": sink_node_id,
        "removed_node_ids": sorted(set(removed_node_ids).union(
            set(nodes["id_node"].astype(int)).difference(retained_node_ids)
        )),
        "removed_link_ids": removed_link_ids,
        "retained_node_count": int(len(pruned_nodes)),
        "retained_link_count": int(len(pruned_links)),
    }


def _boundary_bindings(
    local_links: gpd.GeoDataFrame,
    local_nodes: gpd.GeoDataFrame,
    cuts: gpd.GeoDataFrame,
    contract: LocalPatchContract,
    *,
    tolerance: float,
    global_boundary_roles: Mapping[str, int] | None = None,
) -> pd.DataFrame:
    degree: dict[int, int] = {int(value): 0 for value in local_nodes["id_node"]}
    for value in local_links["id_nodes"]:
        node_a, node_b = parse_id_nodes(value)
        degree[node_a] = degree.get(node_a, 0) + 1
        degree[node_b] = degree.get(node_b, 0) + 1
    terminals = local_nodes.loc[local_nodes["id_node"].astype(int).map(degree) == 1].copy()
    if terminals.empty:
        raise ValueError("Raw local graph has no degree-one cut terminals.")

    expected = {
        "upstream": int(contract.source_splice_anchor_node_id),
        "downstream": int(contract.sink_splice_anchor_node_id),
    }
    records: list[dict[str, Any]] = []
    used: set[int] = set()
    global_boundary_roles = dict(global_boundary_roles or {})
    for role, anchor_id in expected.items():
        role_cuts = cuts.loc[cuts["role"].astype(str) == role]
        if len(role_cuts) != 1:
            raise ValueError(f"Expected exactly one {role} cut, found {len(role_cuts)}.")
        cut = role_cuts.iloc[0].geometry
        candidates = []
        for row in terminals.itertuples(index=False):
            raw_id = int(row.id_node)
            if raw_id in used:
                continue
            distance = float(row.geometry.distance(cut))
            if distance <= tolerance:
                role_matches = (
                    bool(getattr(row, "is_inlet", False))
                    if role == "upstream"
                    else bool(getattr(row, "is_outlet", False))
                )
                candidates.append((not role_matches, distance, raw_id, role_matches))
        candidates.sort()
        role_candidates = [candidate for candidate in candidates if candidate[3]]
        if len(role_candidates) == 1:
            selected_candidate = role_candidates[0]
            binding_method = "cut_proximity_and_raw_boundary_role"
        elif len(candidates) == 1:
            selected_candidate = candidates[0]
            binding_method = "unique_cut_proximity"
        else:
            # At a true global source/sink, skeletonization may retreat well
            # inside a broad channel even though the bank-to-bank cut itself is
            # correct.  Only there, bind the unique compatible raw boundary
            # terminal within one local cut width.  Internal interfaces retain
            # the strict fixed proximity rule above.
            extended_limit = max(float(tolerance), float(cut.length))
            role_terminals = []
            if int(global_boundary_roles.get(role, -1)) == anchor_id:
                for row in terminals.itertuples(index=False):
                    raw_id = int(row.id_node)
                    if raw_id in used:
                        continue
                    role_matches = (
                        bool(getattr(row, "is_inlet", False))
                        if role == "upstream"
                        else bool(getattr(row, "is_outlet", False))
                    )
                    distance = float(row.geometry.distance(cut))
                    if role_matches and distance <= extended_limit:
                        role_terminals.append((False, distance, raw_id, True))
            if len(role_terminals) == 1:
                selected_candidate = role_terminals[0]
                binding_method = "unique_global_boundary_role_extended_tolerance"
            else:
                raise ValueError(
                    f"{role.capitalize()} cut has {len(candidates)} local terminal candidates "
                    f"within {tolerance:.3f} m; merge is ambiguous."
                )
        _, distance, raw_id, role_matches = selected_candidate
        used.add(raw_id)
        records.append(
            {
                "child_id_node": raw_id,
                "cut_role": role,
                "forced_parent_node_id": anchor_id,
                "cut_distance_m": distance,
                "raw_boundary_role_matches": bool(role_matches),
                "binding_method": binding_method,
            }
        )
    return pd.DataFrame.from_records(records)


def _force_boundary_node_matches(
    node_match: pd.DataFrame,
    bindings: pd.DataFrame,
    parent_order: Mapping[int, int],
    child_links: gpd.GeoDataFrame,
) -> pd.DataFrame:
    result = node_match.copy()
    result["spatial_candidate_parent_node_id"] = result[
        "matched_parent_node_id"
    ].where(result["match_within_tolerance"].fillna(False))
    result["match_method"] = np.where(
        result["match_within_tolerance"].fillna(False), "spatial_parent_match", "new_local_node"
    )
    result["cut_role"] = ""
    result["cut_distance_m"] = np.nan
    result["raw_boundary_role_matches"] = False
    result["binding_method"] = ""
    result["duplicate_resolution_method"] = ""
    for binding in bindings.itertuples(index=False):
        mask = result["child_id_node"].astype(int) == int(binding.child_id_node)
        if int(mask.sum()) != 1:
            raise ValueError(f"Cut terminal {binding.child_id_node} is absent from node matches.")
        parent_id = int(binding.forced_parent_node_id)
        result.loc[mask, "matched_parent_node_id"] = parent_id
        result.loc[mask, "match_within_tolerance"] = True
        result.loc[mask, "parent_node_order"] = parent_order[parent_id]
        result.loc[mask, "match_method"] = str(
            getattr(binding, "match_method", "forced_cut_anchor")
        )
        existing_roles = {
            value
            for value in str(result.loc[mask, "cut_role"].iloc[0]).split(",")
            if value
        }
        existing_roles.add(str(binding.cut_role))
        result.loc[mask, "cut_role"] = ",".join(sorted(existing_roles))
        result.loc[mask, "cut_distance_m"] = float(binding.cut_distance_m)
        result.loc[mask, "raw_boundary_role_matches"] = bool(binding.raw_boundary_role_matches)
        result.loc[mask, "binding_method"] = str(binding.binding_method)

    degree: dict[int, int] = {}
    for value in child_links["id_nodes"]:
        node_a, node_b = parse_id_nodes(value)
        degree[int(node_a)] = degree.get(int(node_a), 0) + 1
        degree[int(node_b)] = degree.get(int(node_b), 0) + 1

    # Exact scientific splice bindings are authoritative. A nearby internal
    # junction may be a legitimate new node created by the collapse; it must
    # not inherit the same parent ID merely because it lies within the spatial
    # tolerance. Demote only non-boundary, non-terminal internal candidates.
    matched = result.loc[result["match_within_tolerance"].fillna(False)].copy()
    duplicated_parent_ids = sorted(
        matched.loc[
            matched["matched_parent_node_id"].dropna().astype(int).duplicated(keep=False),
            "matched_parent_node_id",
        ].dropna().astype(int).unique()
    )
    for parent_id in duplicated_parent_ids:
        matched_parent_numeric = pd.to_numeric(
            result["matched_parent_node_id"], errors="coerce"
        )
        group_mask = (
            result["match_within_tolerance"].fillna(False)
            & matched_parent_numeric.eq(parent_id)
        )
        group_indices = result.index[group_mask].tolist()
        forced_indices = result.index[
            group_mask & result["match_method"].eq("forced_anchored_boundary")
        ].tolist()
        if len(forced_indices) == 1:
            keeper_index = forced_indices[0]
            resolution_method = "keep_forced_anchored_boundary"
        elif len(forced_indices) > 1:
            continue
        else:
            terminal_indices = [
                index
                for index in group_indices
                if degree.get(int(result.loc[index, "child_id_node"]), 0) <= 1
            ]
            if len(terminal_indices) == 1:
                keeper_index = terminal_indices[0]
                resolution_method = "keep_unique_external_terminal"
            else:
                keeper_index = min(
                    group_indices,
                    key=lambda index: (
                        float(result.loc[index, "match_distance"]),
                        int(result.loc[index, "child_id_node"]),
                    ),
                )
                resolution_method = "keep_nearest_parent_candidate"
        demote_mask = group_mask & result.index.to_series().ne(keeper_index)
        demote_ids = result.loc[demote_mask, "child_id_node"].astype(int)
        if demote_ids.empty or any(degree.get(int(node_id), 0) < 2 for node_id in demote_ids):
            continue
        if result.loc[demote_mask, "cut_role"].astype(str).str.len().gt(0).any():
            continue
        result.loc[demote_mask, "matched_parent_node_id"] = np.nan
        result.loc[demote_mask, "match_within_tolerance"] = False
        result.loc[demote_mask, "parent_node_order"] = np.nan
        result.loc[demote_mask, "match_method"] = (
            "new_internal_node_near_anchored_boundary"
        )
        result.loc[demote_mask, "duplicate_resolution_method"] = resolution_method
        result.loc[keeper_index, "duplicate_resolution_method"] = resolution_method

    matched = result.loc[result["match_within_tolerance"].fillna(False)].copy()
    duplicate = matched["matched_parent_node_id"].dropna().astype(int).duplicated(keep=False)
    if duplicate.any():
        values = sorted(matched.loc[duplicate, "matched_parent_node_id"].astype(int).unique())
        raise ValueError(f"Multiple local nodes map to the same parent node: {values}.")
    return result


def _selected_parent_node_ids(
    parent_links: gpd.GeoDataFrame,
    selected_parent_link_ids: Iterable[int],
) -> set[int]:
    """Return every parent node touched by the scientifically selected links.

    These nodes describe topology inside or on the edge of the selected unit.
    During the local-realized splice they are location candidates only: shared
    anchoring, which runs afterwards, is the authoritative stage that decides
    which of them survive as child interfaces.
    """
    selected_ids = {int(value) for value in selected_parent_link_ids}
    if not selected_ids:
        return set()
    selected = parent_links.loc[parent_links["id_link"].astype(int).isin(selected_ids)]
    return set(selected["id_us_node"].astype(int)).union(
        selected["id_ds_node"].astype(int)
    )


def _apply_realized_node_identity_policy(
    node_match: pd.DataFrame,
    *,
    merge_stage: str,
    protected_selected_parent_node_ids: set[int],
    identity_tolerance_m: float,
) -> pd.DataFrame:
    """Separate spatial proximity from accepted parent-node identity.

    The realized splice may use a generous spatial tolerance to associate raw
    geometry with its parent context.  That does not authorize moving a raw
    node onto a selected-unit node.  Selected topology remains child-native
    until the shared realized-to-anchored transformation projects and splits
    the scientific interfaces.  Outer computational cut bindings remain hard.
    """
    result = node_match.copy()
    result["identity_candidate_parent_node_id"] = result[
        "matched_parent_node_id"
    ].where(result["match_within_tolerance"].fillna(False))
    result["identity_accepted"] = False
    result["identity_parent_node_id"] = np.nan
    result["identity_decision"] = "new_local_node_no_spatial_candidate"
    result["identity_tolerance_m"] = float(identity_tolerance_m)

    for index, row in result.iterrows():
        candidate = row.get("identity_candidate_parent_node_id")
        if pd.isna(candidate):
            continue
        parent_id = int(candidate)
        cut_role = str(row.get("cut_role", "") or "")
        forced = bool(cut_role) or str(row.get("match_method", "")).startswith("forced_")
        distance = pd.to_numeric(pd.Series([row.get("match_distance")]), errors="coerce").iloc[0]

        if forced:
            accepted = True
            decision = "forced_outer_computational_boundary"
        elif (
            str(merge_stage) == "local_realized"
            and parent_id in protected_selected_parent_node_ids
        ):
            accepted = False
            decision = "defer_selected_topology_to_shared_anchoring"
        elif (
            str(merge_stage) == "local_realized"
            and (pd.isna(distance) or float(distance) > float(identity_tolerance_m))
        ):
            accepted = False
            decision = "spatial_candidate_outside_realized_identity_tolerance"
        else:
            accepted = True
            decision = "accepted_parent_node_identity"

        result.loc[index, "identity_accepted"] = bool(accepted)
        result.loc[index, "identity_decision"] = decision
        if accepted:
            result.loc[index, "identity_parent_node_id"] = parent_id
    return result


def _eligible_same_endpoint_connector_ids(
    connector_ids: Iterable[int],
    *,
    merge_stage: str,
    selected_parent_link_ids: set[int],
) -> list[int]:
    """Exclude selected-unit connectors from realized reconstruction.

    Reconstructing an unselected context connector can repair a compressed
    outer splice. Reconstructing a selected connector would reintroduce parent
    morphology before the scientific anchoring stage and is never allowed.
    """
    return [
        int(parent_id)
        for parent_id in connector_ids
        if not (
            str(merge_stage) == "local_realized"
            and int(parent_id) in selected_parent_link_ids
        )
    ]


def _trim_to_anchored_boundaries(
    directed_links: gpd.GeoDataFrame,
    directed_nodes: gpd.GeoDataFrame,
    parent_nodes: gpd.GeoDataFrame,
    contract: LocalPatchContract,
    *,
    parent_links: gpd.GeoDataFrame | None = None,
    tolerance: float,
    junction_snap_tolerance_m: float | None = None,
    prune_undirected_context: bool = True,
    upstream_boundary_node_ids: tuple[int, ...] | None = None,
    downstream_boundary_node_ids: tuple[int, ...] | None = None,
    selected_parent_link_ids: tuple[int, ...] | None = None,
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, pd.DataFrame]:
    """Clip computational cut context to prescribed parent boundaries.

    The default boundary/link sets retain the historical scientific anchoring
    behaviour.  The local-realized assembler supplies the outer computational
    interfaces instead, allowing this geometry allocator to be reused without
    applying the scientific realized-to-anchored transformation prematurely.
    """
    links = directed_links.copy()
    nodes = directed_nodes.copy()
    # This stage is deliberately geometry-first: raw RivGraph products may
    # carry id_us_node/id_ds_node columns whose values are still null because
    # direction has not been assigned.  Column presence therefore does not
    # imply usable directed endpoints.  Rebuild a complete temporary endpoint
    # pair from the authoritative undirected id_nodes field for every row;
    # parent-guided direction is assigned only after anchored trimming.
    unordered_endpoints = [parse_id_nodes(value) for value in links["id_nodes"]]
    links["id_us_node"] = [int(values[0]) for values in unordered_endpoints]
    links["id_ds_node"] = [int(values[1]) for values in unordered_endpoints]
    parent_index = parent_nodes.set_index(parent_nodes["id_node"].astype(int), drop=False)
    next_node_id = int(nodes["id_node"].astype(int).max()) + 1
    next_link_id = int(links["id_link"].astype(int).max()) + 1
    boundary_records: list[dict[str, Any]] = []

    upstream_boundaries = tuple(
        int(value)
        for value in (
            upstream_boundary_node_ids
            if upstream_boundary_node_ids is not None
            else contract.anchored_upstream_boundary_node_ids
        )
    )
    downstream_boundaries = tuple(
        int(value)
        for value in (
            downstream_boundary_node_ids
            if downstream_boundary_node_ids is not None
            else contract.anchored_downstream_boundary_node_ids
        )
    )
    boundary_specs = [
        ("upstream", contract.upstream_collar, int(parent_id))
        for parent_id in upstream_boundaries
    ] + [
        ("downstream", contract.downstream_collar, int(parent_id))
        for parent_id in downstream_boundaries
    ]
    # A valid open/interconnected unit can use one parent node as both an
    # upstream and downstream scientific interface. Allocate that geometry
    # once, then retain both roles in the binding table.
    specs_by_parent: dict[int, list[tuple[str, LocalPatchCollar]]] = {}
    for role, collar, parent_id in boundary_specs:
        specs_by_parent.setdefault(int(parent_id), []).append((str(role), collar))

    junction_snap_tolerance = float(
        junction_snap_tolerance_m
        if junction_snap_tolerance_m is not None
        else max(1e-9, tolerance * 0.5)
    )
    degree: dict[int, int] = {int(value): 0 for value in nodes["id_node"]}
    for value in links["id_nodes"]:
        endpoint_a, endpoint_b = parse_id_nodes(value)
        degree[int(endpoint_a)] = degree.get(int(endpoint_a), 0) + 1
        degree[int(endpoint_b)] = degree.get(int(endpoint_b), 0) + 1

    # Reserve strong existing-node matches globally before modifying any
    # geometry. This prevents upstream-first greedy allocation from stealing a
    # junction that is a much better match for a downstream parent boundary.
    preferred_node_candidates: list[tuple[float, int, int]] = []
    for parent_id in specs_by_parent:
        target = parent_index.loc[int(parent_id)].geometry
        node_distances = nodes.geometry.distance(target)
        link_distances = links.geometry.distance(target)
        if node_distances.empty:
            continue
        node_index = node_distances.idxmin()
        node_id = int(nodes.loc[node_index, "id_node"])
        node_distance = float(node_distances.loc[node_index])
        link_distance = float(link_distances.min()) if not link_distances.empty else float("inf")
        if (
            node_distance <= tolerance
            and node_distance <= link_distance + junction_snap_tolerance
        ):
            preferred_node_candidates.append((node_distance, int(parent_id), node_id))
    preferred_node_candidates.sort()
    reserved_node_by_parent: dict[int, int] = {}
    reserved_parent_by_node: dict[int, int] = {}
    for _, parent_id, node_id in preferred_node_candidates:
        if parent_id in reserved_node_by_parent or node_id in reserved_parent_by_node:
            continue
        reserved_node_by_parent[parent_id] = node_id
        reserved_parent_by_node[node_id] = parent_id

    allocated_child_by_parent: dict[int, int] = {}
    allocation_method_by_parent: dict[int, str] = {}

    selected_parent_links = None
    if parent_links is not None:
        selected_ids = set(
            int(value)
            for value in (
                selected_parent_link_ids
                if selected_parent_link_ids is not None
                else contract.anchored_replacement_link_ids
            )
        )
        selected_parent_links = parent_links.loc[
            parent_links["id_link"].astype(int).isin(selected_ids)
        ].copy()

    def _selected_incident_links(parent_id: int) -> gpd.GeoDataFrame | None:
        if selected_parent_links is None or selected_parent_links.empty:
            return None
        return selected_parent_links.loc[
            selected_parent_links["id_us_node"].astype(int).eq(int(parent_id))
            | selected_parent_links["id_ds_node"].astype(int).eq(int(parent_id))
        ].copy()

    def _candidate_overlap_score(link_geometry, parent_id: int) -> float:
        """Prefer the raw branch representing the selected parent interface."""
        incident = _selected_incident_links(parent_id)
        if incident is None or incident.empty:
            return 0.0
        raw = _linear_geometry(link_geometry)
        raw_length = max(float(raw.length), 1e-9)
        best = 0.0
        for parent_geometry in incident.geometry:
            parent_line = _linear_geometry(parent_geometry)
            parent_length = max(float(parent_line.length), 1e-9)
            raw_coverage = float(
                raw.intersection(parent_line.buffer(tolerance)).length / raw_length
            )
            parent_coverage = float(
                parent_line.intersection(raw.buffer(tolerance)).length / parent_length
            )
            best = max(best, min(raw_coverage, parent_coverage))
        return float(best)

    def _selected_parent_connector(
        first_parent_id: int,
        second_parent_id: int,
    ) -> pd.Series | None:
        if selected_parent_links is None or selected_parent_links.empty:
            return None
        endpoints = {int(first_parent_id), int(second_parent_id)}
        candidates = selected_parent_links.loc[
            selected_parent_links.apply(
                lambda row: {
                    int(row["id_us_node"]), int(row["id_ds_node"])
                } == endpoints,
                axis=1,
            )
        ]
        if len(candidates) != 1:
            return None
        return candidates.iloc[0]

    for parent_id in sorted(
        specs_by_parent,
        key=lambda value: (
            0 if value in reserved_node_by_parent else 1,
            float(parent_index.loc[int(value)].geometry.x),
            float(parent_index.loc[int(value)].geometry.y),
            int(value),
        ),
    ):
        parent_id = int(parent_id)
        target = parent_index.loc[parent_id].geometry
        reserved_node_id = reserved_node_by_parent.get(parent_id)
        if reserved_node_id is not None:
            node_index = nodes.index[nodes["id_node"].astype(int).eq(reserved_node_id)][0]
            nodes.loc[node_index, nodes.geometry.name] = target
            raw_boundary_id = int(reserved_node_id)
            method = "existing_local_node_at_anchored_boundary"
        else:
            link_distances = links.geometry.distance(target)
            eligible_indices = link_distances.index[
                link_distances.le(float(tolerance))
            ].tolist()
            candidate_rows = sorted(
                (
                    -_candidate_overlap_score(links.loc[index].geometry, parent_id),
                    float(link_distances.loc[index]),
                    int(links.loc[index, "id_link"]),
                    index,
                )
                for index in eligible_indices
            )
            ordered = link_distances.sort_values()
            if ordered.empty or float(ordered.iloc[0]) > tolerance:
                roles = ",".join(role for role, _ in specs_by_parent[parent_id])
                raise ValueError(
                    f"No local link or node lies within {tolerance:.3f} m of the "
                    f"{roles} anchored boundary node {parent_id}."
                )
            best_overlap = -float(candidate_rows[0][0])
            if best_overlap > 0.0:
                tied_indices = [
                    index
                    for negative_overlap, distance, _, index in candidate_rows
                    if abs((-negative_overlap) - best_overlap) <= 1e-9
                    and abs(distance - float(candidate_rows[0][1])) <= 1e-6
                ]
            else:
                tied_indices = ordered.index[
                    ordered.sub(float(ordered.iloc[0])).abs().le(1e-6)
                ].tolist()
            if len(tied_indices) > 1:
                tied_endpoint_sets = [
                    set(parse_id_nodes(links.loc[index, "id_nodes"]))
                    for index in tied_indices
                ]
                shared_endpoints = set.intersection(*tied_endpoint_sets)
                available_shared = [
                    int(node_id)
                    for node_id in sorted(shared_endpoints)
                    if int(node_id) not in reserved_parent_by_node
                    and int(node_id) not in allocated_child_by_parent.values()
                ]
                node_index_by_id = nodes.set_index(nodes["id_node"].astype(int), drop=False)
                acceptable_shared = [
                    node_id
                    for node_id in available_shared
                    if float(node_index_by_id.loc[node_id].geometry.distance(target))
                    <= float(ordered.iloc[0]) + junction_snap_tolerance
                ]
                if len(acceptable_shared) == 1:
                    raw_boundary_id = int(acceptable_shared[0])
                    node_index = nodes.index[nodes["id_node"].astype(int).eq(raw_boundary_id)][0]
                    nodes.loc[node_index, nodes.geometry.name] = target
                    method = "shared_local_junction_at_anchored_boundary"
                    allocated_child_by_parent[parent_id] = raw_boundary_id
                    allocation_method_by_parent[parent_id] = method
                    continue
                roles = ",".join(role for role, _ in specs_by_parent[parent_id])
                raise ValueError(
                    f"Multiple local links are equally close to the {roles} anchored "
                    f"boundary node {parent_id}."
                )
            link_index = tied_indices[0]
            row = links.loc[link_index].copy()
            line = _linear_geometry(row.geometry)
            endpoint_a, endpoint_b = parse_id_nodes(row["id_nodes"])
            node_geometry = nodes.set_index(nodes["id_node"].astype(int)).geometry
            line_start = Point(line.coords[0])
            if line_start.distance(node_geometry.loc[int(endpoint_a)]) <= line_start.distance(
                node_geometry.loc[int(endpoint_b)]
            ):
                geometry_start_node, geometry_end_node = int(endpoint_a), int(endpoint_b)
            else:
                geometry_start_node, geometry_end_node = int(endpoint_b), int(endpoint_a)
            measure = float(line.project(target))
            endpoint_margin = min(tolerance * 0.25, float(line.length) * 0.05)
            near_endpoint = (
                measure <= endpoint_margin
                or measure >= float(line.length) - endpoint_margin
            )
            endpoint_id = int(
                geometry_start_node
                if measure <= endpoint_margin
                else geometry_end_node
            )
            endpoint_owner = reserved_parent_by_node.get(endpoint_id)
            endpoint_available = endpoint_owner in (None, parent_id) and endpoint_id not in (
                set(allocated_child_by_parent.values()) - {allocated_child_by_parent.get(parent_id)}
            )
            if near_endpoint and endpoint_available:
                node_index = nodes.index[nodes["id_node"].astype(int).eq(endpoint_id)][0]
                nodes.loc[node_index, nodes.geometry.name] = target
                raw_boundary_id = endpoint_id
                method = "existing_link_endpoint_at_anchored_boundary"
            else:
                split_epsilon = max(1e-7, float(line.length) * 1e-9)
                if measure <= split_epsilon or measure >= float(line.length) - split_epsilon:
                    allocated_owner = next(
                        (
                            int(owner_parent_id)
                            for owner_parent_id, owner_child_id in allocated_child_by_parent.items()
                            if int(owner_child_id) == endpoint_id
                        ),
                        None,
                    )
                    endpoint_owner = reserved_parent_by_node.get(
                        endpoint_id, allocated_owner
                    )
                    connector = (
                        _selected_parent_connector(int(endpoint_owner), parent_id)
                        if endpoint_owner is not None and int(endpoint_owner) != parent_id
                        else None
                    )
                    if connector is None:
                        raise ValueError(
                            f"Anchored boundary node {parent_id} projects onto already assigned "
                            f"local boundary node {endpoint_id}; no selected parent connector "
                            "supports a distinct anchored allocation."
                        )

                    # RivGraph can compress two nearby parent junctions into
                    # one raw node. Re-expand that junction using the selected
                    # parent topology: move the correct raw selected branch to
                    # an exact new boundary node and retain the short parent
                    # connector between the two distinct interfaces.
                    raw_boundary_id = next_node_id
                    next_node_id += 1
                    node_values = {column: None for column in nodes.columns}
                    node_values.update(
                        {"id_node": raw_boundary_id, nodes.geometry.name: target}
                    )
                    nodes = gpd.GeoDataFrame(
                        pd.concat(
                            [
                                nodes,
                                gpd.GeoDataFrame(
                                    [node_values],
                                    geometry=nodes.geometry.name,
                                    crs=nodes.crs,
                                ),
                            ],
                            ignore_index=True,
                        ),
                        geometry=nodes.geometry.name,
                        crs=nodes.crs,
                    )

                    coordinates = list(line.coords)
                    if endpoint_id == geometry_start_node:
                        coordinates[0] = (float(target.x), float(target.y))
                        geometry_start_node = raw_boundary_id
                    else:
                        coordinates[-1] = (float(target.x), float(target.y))
                        geometry_end_node = raw_boundary_id
                    row["id_us_node"] = int(geometry_start_node)
                    row["id_ds_node"] = int(geometry_end_node)
                    row["id_nodes"] = f"{geometry_start_node}, {geometry_end_node}"
                    row[links.geometry.name] = LineString(coordinates)
                    for field in ("idx_link", "wid_pix"):
                        if field in links.columns:
                            row[field] = None
                    links.loc[link_index] = row

                    connector_row = row.copy()
                    connector_row["id_link"] = next_link_id
                    next_link_id += 1
                    connector_parent_us = int(connector["id_us_node"])
                    owner_child_id = int(endpoint_id)
                    if connector_parent_us == int(endpoint_owner):
                        connector_us, connector_ds = owner_child_id, raw_boundary_id
                    else:
                        connector_us, connector_ds = raw_boundary_id, owner_child_id
                    connector_row["id_us_node"] = connector_us
                    connector_row["id_ds_node"] = connector_ds
                    connector_row["id_nodes"] = f"{connector_us}, {connector_ds}"
                    connector_row[links.geometry.name] = connector.geometry
                    for field in ("idx_link", "wid_pix"):
                        if field in links.columns:
                            connector_row[field] = None
                    links = gpd.GeoDataFrame(
                        pd.concat(
                            [
                                links,
                                gpd.GeoDataFrame(
                                    [connector_row],
                                    geometry=links.geometry.name,
                                    crs=links.crs,
                                ),
                            ],
                            ignore_index=True,
                        ),
                        geometry=links.geometry.name,
                        crs=links.crs,
                    )
                    method = "expand_compressed_selected_parent_junction"
                    allocated_child_by_parent[parent_id] = raw_boundary_id
                    allocation_method_by_parent[parent_id] = method
                    continue
                raw_boundary_id = next_node_id
                next_node_id += 1
                node_values = {column: None for column in nodes.columns}
                node_values.update({"id_node": raw_boundary_id, nodes.geometry.name: target})
                nodes = gpd.GeoDataFrame(
                    pd.concat(
                        [nodes, gpd.GeoDataFrame([node_values], geometry=nodes.geometry.name, crs=nodes.crs)],
                        ignore_index=True,
                    ),
                    geometry=nodes.geometry.name,
                    crs=nodes.crs,
                )
                first_geometry = substring(line, 0.0, measure)
                second_geometry = substring(line, measure, float(line.length))
                first = row.copy()
                second = row.copy()
                first["id_us_node"] = geometry_start_node
                first["id_ds_node"] = raw_boundary_id
                first["id_nodes"] = f"{geometry_start_node}, {raw_boundary_id}"
                first[links.geometry.name] = first_geometry
                second["id_link"] = next_link_id
                next_link_id += 1
                second["id_us_node"] = raw_boundary_id
                second["id_ds_node"] = geometry_end_node
                second["id_nodes"] = f"{raw_boundary_id}, {geometry_end_node}"
                second[links.geometry.name] = second_geometry
                for field in ("idx_link", "wid_pix"):
                    if field in links.columns:
                        first[field] = None
                        second[field] = None
                links.loc[link_index] = first
                links = gpd.GeoDataFrame(
                    pd.concat(
                        [links, gpd.GeoDataFrame([second], geometry=links.geometry.name, crs=links.crs)],
                        ignore_index=True,
                    ),
                    geometry=links.geometry.name,
                    crs=links.crs,
                )
                method = "split_local_link_at_anchored_boundary"
        allocated_child_by_parent[parent_id] = int(raw_boundary_id)
        allocation_method_by_parent[parent_id] = method

    child_to_parent: dict[int, int] = {}
    for parent_id, child_id in allocated_child_by_parent.items():
        existing_parent = child_to_parent.get(int(child_id))
        if existing_parent is not None and existing_parent != int(parent_id):
            raise ValueError(
                "Distinct anchored parent boundaries cannot share one local boundary node."
            )
        child_to_parent[int(child_id)] = int(parent_id)
        for role, _ in specs_by_parent[parent_id]:
            boundary_records.append(
                {
                    "child_id_node": int(child_id),
                    "cut_role": str(role),
                    "forced_parent_node_id": int(parent_id),
                    "cut_distance_m": 0.0,
                    "raw_boundary_role_matches": True,
                    "binding_method": allocation_method_by_parent[parent_id],
                    "match_method": "forced_anchored_boundary",
                }
            )

    source_ids = {
        int(record["child_id_node"])
        for record in boundary_records if record["cut_role"] == "upstream"
    }
    sink_ids = {
        int(record["child_id_node"])
        for record in boundary_records if record["cut_role"] == "downstream"
    }
    if not prune_undirected_context:
        return (
            links.reset_index(drop=True),
            nodes.reset_index(drop=True),
            pd.DataFrame(boundary_records),
        )

    # Geometry is authoritative at this stage. RivGraph's local direction can
    # be wrong when the crop orientation differs from the global network, so
    # retain the anchored corridor on undirected topology and direct it only
    # after the exact scientific boundaries have been allocated.
    graph = nx.MultiGraph()
    for row in links.itertuples(index=False):
        node_a, node_b = parse_id_nodes(row.id_nodes)
        graph.add_edge(int(node_a), int(node_b), key=int(row.id_link))
    if not all(
        source_id in graph
        and any(sink_id in graph and nx.has_path(graph, source_id, sink_id) for sink_id in sink_ids)
        for source_id in source_ids
    ):
        raise ValueError("Anchored local boundaries are not connected geometrically.")
    protected = source_ids | sink_ids
    queue = [int(node) for node in graph if node not in protected and graph.degree(node) <= 1]
    while queue:
        node = int(queue.pop())
        if node not in graph or node in protected or graph.degree(node) > 1:
            continue
        neighbours = [int(value) for value in graph.neighbors(node)]
        graph.remove_node(node)
        for neighbour in neighbours:
            if neighbour in graph and neighbour not in protected and graph.degree(neighbour) <= 1:
                queue.append(neighbour)
    retained_nodes: set[int] = set()
    for source_id in source_ids:
        if source_id in graph:
            retained_nodes.update(int(value) for value in nx.node_connected_component(graph, source_id))
    links = links.loc[
        links["id_us_node"].astype(int).isin(retained_nodes)
        & links["id_ds_node"].astype(int).isin(retained_nodes)
    ].copy()
    used_nodes = set(links["id_us_node"].astype(int)).union(links["id_ds_node"].astype(int))
    nodes = nodes.loc[nodes["id_node"].astype(int).isin(used_nodes)].copy()
    return links.reset_index(drop=True), nodes.reset_index(drop=True), pd.DataFrame(boundary_records)


def _prune_directed_to_anchored_interval(
    links: gpd.GeoDataFrame,
    nodes: gpd.GeoDataFrame,
    *,
    source_node_ids: set[int],
    sink_node_ids: set[int],
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, dict[str, Any]]:
    """Retain links on a directed path between any anchored interface pair."""
    graph = nx.MultiDiGraph()
    for row in links.itertuples(index=False):
        graph.add_edge(
            int(row.id_us_node),
            int(row.id_ds_node),
            key=int(row.id_link),
        )
    sources = {int(value) for value in source_node_ids}
    sinks = {int(value) for value in sink_node_ids}
    missing = sorted(sources.union(sinks).difference(graph.nodes))
    if missing:
        raise ValueError(f"Anchored interval nodes are absent from the directed graph: {missing}.")

    forward: set[int] = set()
    for source in sources:
        forward.add(source)
        forward.update(int(value) for value in nx.descendants(graph, source))
    reverse_graph = graph.reverse(copy=False)
    reverse: set[int] = set()
    for sink in sinks:
        reverse.add(sink)
        reverse.update(int(value) for value in nx.descendants(reverse_graph, sink))
    retained_nodes = forward.intersection(reverse)
    if not sources.union(sinks).issubset(retained_nodes):
        raise ValueError(
            "No directed anchored interval connects every selected interface."
        )
    retained_links = {
        int(key)
        for node_a, node_b, key in graph.edges(keys=True)
        if int(node_a) in retained_nodes and int(node_b) in retained_nodes
    }
    result_links = links.loc[
        links["id_link"].astype(int).isin(retained_links)
    ].copy()
    used_nodes = set(result_links["id_us_node"].astype(int)).union(
        result_links["id_ds_node"].astype(int)
    )
    result_nodes = nodes.loc[nodes["id_node"].astype(int).isin(used_nodes)].copy()
    return (
        result_links.reset_index(drop=True),
        result_nodes.reset_index(drop=True),
        {
            "method": "directed_union_between_anchored_interfaces",
            "source_node_ids": sorted(sources),
            "sink_node_ids": sorted(sinks),
            "retained_node_count": int(len(result_nodes)),
            "retained_link_count": int(len(result_links)),
        },
    )


def _snap_and_orient_geometry(
    geometry,
    raw_us: Point,
    raw_ds: Point,
    final_us: Point,
    final_ds: Point,
    *,
    same_endpoint_fallback_geometry=None,
    same_endpoint_reconstruction_limit_m: float | None = None,
):
    line = _linear_geometry(geometry)
    coordinates = list(line.coords)
    forward = Point(coordinates[0]).distance(raw_us) + Point(coordinates[-1]).distance(raw_ds)
    reverse = Point(coordinates[-1]).distance(raw_us) + Point(coordinates[0]).distance(raw_ds)
    if reverse < forward:
        coordinates.reverse()
    line = LineString(coordinates)
    us_measure = float(line.project(final_us))
    ds_measure = float(line.project(final_ds))
    projection_epsilon = max(1e-9, float(line.length) * 1e-10)
    if us_measure > ds_measure + projection_epsilon:
        # Raw RivGraph direction is only a computational-crop hint. Once both
        # endpoints have authoritative parent matches, reverse the local
        # geometry to that parent flow order instead of rejecting an otherwise
        # valid projected splice.
        line = LineString(list(line.coords)[::-1])
        us_measure = float(line.project(final_us))
        ds_measure = float(line.project(final_ds))
    if us_measure >= ds_measure - projection_epsilon:
        fallback = (
            _linear_geometry(same_endpoint_fallback_geometry)
            if same_endpoint_fallback_geometry is not None
            else None
        )
        limit = (
            float(same_endpoint_reconstruction_limit_m)
            if same_endpoint_reconstruction_limit_m is not None
            else None
        )
        connector_length = float(final_us.distance(final_ds))
        collapsed_at_endpoint = (
            min(us_measure, float(line.length) - us_measure) <= projection_epsilon
            and min(ds_measure, float(line.length) - ds_measure) <= projection_epsilon
        )
        fallback_endpoints_match = False
        if fallback is not None:
            fallback_coordinates = list(fallback.coords)
            fallback_start = Point(fallback_coordinates[0])
            fallback_end = Point(fallback_coordinates[-1])
            fallback_endpoints_match = min(
                fallback_start.distance(final_us) + fallback_end.distance(final_ds),
                fallback_start.distance(final_ds) + fallback_end.distance(final_us),
            ) <= max(projection_epsilon, 1e-6)
        if (
            fallback is not None
            and limit is not None
            and limit > 0.0
            and connector_length > projection_epsilon
            and connector_length <= limit
            and float(fallback.length) <= limit
            and collapsed_at_endpoint
            and fallback_endpoints_match
        ):
            coordinates = list(fallback.coords)
            direct = Point(coordinates[0]).distance(final_us) + Point(
                coordinates[-1]
            ).distance(final_ds)
            reverse = Point(coordinates[-1]).distance(final_us) + Point(
                coordinates[0]
            ).distance(final_ds)
            if reverse < direct:
                coordinates.reverse()
            coordinates[0] = (float(final_us.x), float(final_us.y))
            coordinates[-1] = (float(final_ds.x), float(final_ds.y))
            return LineString(coordinates)
        raise ValueError(
            "Anchored boundary-node projections collapse to one indistinguishable position."
        )
    coordinates = list(substring(line, us_measure, ds_measure).coords)
    coordinates[0] = (float(final_us.x), float(final_us.y))
    coordinates[-1] = (float(final_ds.x), float(final_ds.y))
    return LineString(coordinates)


def _empty_like(columns: list[str]) -> dict[str, Any]:
    return {column: None for column in columns}


def _serialize_ids(values) -> str:
    return ", ".join(str(int(value)) for value in values)


def _refresh_topology(
    links: gpd.GeoDataFrame,
    nodes: gpd.GeoDataFrame,
    *,
    source_id: int,
    sink_id: int,
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    links = links.copy()
    nodes = nodes.copy()
    links["id_nodes"] = [
        f"{int(us)}, {int(ds)}" for us, ds in zip(links["id_us_node"], links["id_ds_node"])
    ]
    incident: dict[int, list[int]] = {int(value): [] for value in nodes["id_node"]}
    incoming: dict[int, int] = {int(value): 0 for value in nodes["id_node"]}
    outgoing: dict[int, int] = {int(value): 0 for value in nodes["id_node"]}
    for row in links.itertuples(index=False):
        link_id, us, ds = int(row.id_link), int(row.id_us_node), int(row.id_ds_node)
        incident.setdefault(us, []).append(link_id)
        incident.setdefault(ds, []).append(link_id)
        outgoing[us] = outgoing.get(us, 0) + 1
        incoming[ds] = incoming.get(ds, 0) + 1
    if "link_conn" in links.columns:
        link_connections: list[str] = []
        for row in links.itertuples(index=False):
            link_id, us, ds = int(row.id_link), int(row.id_us_node), int(row.id_ds_node)
            connected = sorted(
                (set(incident.get(us, ())) | set(incident.get(ds, ()))) - {link_id}
            )
            link_connections.append(_serialize_ids(connected))
        links["link_conn"] = link_connections
    links["is_inlet"] = links["id_us_node"].astype(int) == source_id
    links["is_outlet"] = links["id_ds_node"].astype(int) == sink_id
    links["type_io"] = np.select(
        [links["is_inlet"] & links["is_outlet"], links["is_inlet"], links["is_outlet"]],
        ["both", "inlet", "outlet"], default="internal",
    )
    nodes["id_links"] = [_serialize_ids(sorted(incident[int(value)])) for value in nodes["id_node"]]
    nodes["n_links"] = [len(incident[int(value)]) for value in nodes["id_node"]]
    nodes["in_degree"] = [incoming.get(int(value), 0) for value in nodes["id_node"]]
    nodes["out_degree"] = [outgoing.get(int(value), 0) for value in nodes["id_node"]]
    nodes["is_inlet"] = nodes["id_node"].astype(int) == source_id
    nodes["is_outlet"] = nodes["id_node"].astype(int) == sink_id
    nodes["type_io"] = np.select(
        [nodes["is_inlet"] & nodes["is_outlet"], nodes["is_inlet"], nodes["is_outlet"]],
        ["both", "inlet", "outlet"], default="internal",
    )
    return links, nodes


def _orient_inserted_components_from_parent_order(
    links: gpd.GeoDataFrame,
    nodes: gpd.GeoDataFrame,
    *,
    inserted_link_ids: set[int],
    fixed_scaffold_link_ids: set[int] | None = None,
    parent_node_order: Mapping[int, int],
    protected_source_node_ids: set[int],
    protected_sink_node_ids: set[int],
) -> tuple[gpd.GeoDataFrame, dict[str, Any]]:
    """Orient new merged components from authoritative parent topology.

    Raw local direction is only provisional. Once exact parent boundary nodes
    are known, each connected inserted component is directed between its most
    upstream and downstream parent attachment using a harmonic potential. This
    handles shared-role interfaces without allowing the local crop orientation
    to reverse an otherwise valid parent corridor.
    """
    result = links.copy()
    available_ids = set(result["id_link"].astype(int))
    inserted_link_ids = set(map(int, inserted_link_ids)).intersection(available_ids)
    fixed_scaffold_link_ids = set(
        map(int, fixed_scaffold_link_ids or set())
    ).intersection(available_ids).difference(inserted_link_ids)
    inserted = result.loc[result["id_link"].astype(int).isin(inserted_link_ids)].copy()
    if inserted.empty:
        return result, {
            "method": "parent_order_harmonic_inserted_components",
            "component_count": 0,
            "oriented_component_count": 0,
            "reoriented_link_count": 0,
        }

    orientation_scope_ids = inserted_link_ids.union(fixed_scaffold_link_ids)
    orientation_scope = result.loc[
        result["id_link"].astype(int).isin(orientation_scope_ids)
    ].copy()
    graph = nx.MultiGraph()
    for row in orientation_scope.itertuples(index=False):
        graph.add_edge(int(row.id_us_node), int(row.id_ds_node), key=int(row.id_link))

    def _has_fixed_incidence(node_id: int) -> bool:
        return any(
            int(key) in fixed_scaffold_link_ids
            for _, _, key in graph.edges(int(node_id), keys=True)
        )
    protected_source_attachments = {
        int(node_id)
        for node_id in graph.nodes
        if int(node_id) in protected_source_node_ids
    }
    protected_sink_attachments = {
        int(node_id)
        for node_id in graph.nodes
        if int(node_id) in protected_sink_node_ids
    }
    protected_attachments = (
        protected_source_attachments | protected_sink_attachments
    )
    removed_one_attachment_block_link_ids: set[int] = set()
    # A leaf-pruning pass cannot remove a pendant loop: every node in that
    # loop has undirected degree two, although the whole block touches the
    # scientific corridor at only one articulation.  Such a block can never
    # receive an acyclic orientation without introducing an artificial local
    # source or sink.  Retain precisely the block-cut-tree subtree connecting
    # the protected scientific interfaces.
    simple_graph = nx.Graph(graph)
    retained_pairs: set[tuple[int, int]] = set()
    for connected_nodes in nx.connected_components(simple_graph):
        connected_nodes = {int(value) for value in connected_nodes}
        component_attachments = connected_nodes.intersection(protected_attachments)
        if len(component_attachments) < 2:
            # The existing single-attachment component pass below removes the
            # entire component and records that distinct audit category.
            retained_pairs.update(
                tuple(sorted((int(node_a), int(node_b))))
                for node_a, node_b in simple_graph.subgraph(connected_nodes).edges
            )
            continue
        component_graph = simple_graph.subgraph(connected_nodes).copy()
        block_nodes = [
            {int(value) for value in component}
            for component in nx.biconnected_components(component_graph)
        ]
        articulations = {
            int(value) for value in nx.articulation_points(component_graph)
        }
        block_tree = nx.Graph()
        for block_index, block in enumerate(block_nodes):
            block_ref = ("block", int(block_index))
            block_tree.add_node(block_ref)
            for node_id in block.intersection(articulations):
                block_tree.add_edge(
                    block_ref, ("articulation", int(node_id))
                )
        terminal_refs: list[tuple[str, int]] = []
        for node_id in sorted(component_attachments):
            if node_id in articulations:
                terminal_refs.append(("articulation", int(node_id)))
                continue
            containing = [
                index for index, block in enumerate(block_nodes) if node_id in block
            ]
            if len(containing) != 1:
                raise ValueError(
                    "Could not place protected local interface node "
                    f"{node_id} in the inserted block-cut tree."
                )
            terminal_refs.append(("block", int(containing[0])))
        unique_terminal_refs = list(dict.fromkeys(terminal_refs))
        retained_tree_nodes: set[tuple[str, int]] = {unique_terminal_refs[0]}
        for terminal_ref in unique_terminal_refs[1:]:
            retained_tree_nodes.update(
                nx.shortest_path(block_tree, unique_terminal_refs[0], terminal_ref)
            )
        retained_block_indexes = {
            int(value)
            for kind, value in retained_tree_nodes
            if kind == "block"
        }
        retained_pairs.update(
            tuple(sorted((int(node_a), int(node_b))))
            for block_index in retained_block_indexes
            for node_a, node_b in component_graph.subgraph(
                block_nodes[block_index]
            ).edges
        )
    if retained_pairs:
        removed_one_attachment_block_link_ids = {
            int(key)
            for node_a, node_b, key in graph.edges(keys=True)
            if int(key) in inserted_link_ids
            if tuple(sorted((int(node_a), int(node_b)))) not in retained_pairs
        }
        if removed_one_attachment_block_link_ids:
            result = result.loc[
                ~result["id_link"].astype(int).isin(
                    removed_one_attachment_block_link_ids
                )
            ].copy()
            inserted = inserted.loc[
                ~inserted["id_link"].astype(int).isin(
                    removed_one_attachment_block_link_ids
                )
            ].copy()
            graph.remove_edges_from(
                [
                    (node_a, node_b, key)
                    for node_a, node_b, key in graph.edges(keys=True)
                    if int(key) in removed_one_attachment_block_link_ids
                ]
            )
            graph.remove_nodes_from(list(nx.isolates(graph)))
    queue = [
        int(node_id)
        for node_id in graph.nodes
        if int(node_id) not in protected_attachments
        and graph.degree(node_id) <= 1
        and not _has_fixed_incidence(int(node_id))
    ]
    removed_dangling_link_ids: set[int] = set()
    while queue:
        node_id = int(queue.pop())
        if (
            node_id not in graph
            or node_id in protected_attachments
            or graph.degree(node_id) > 1
            or _has_fixed_incidence(node_id)
        ):
            continue
        neighbours = [int(value) for value in graph.neighbors(node_id)]
        for _, _, key in list(graph.edges(node_id, keys=True)):
            if int(key) in inserted_link_ids:
                removed_dangling_link_ids.add(int(key))
        graph.remove_node(node_id)
        for neighbour in neighbours:
            if (
                neighbour in graph
                and neighbour not in protected_attachments
                and graph.degree(neighbour) <= 1
                and not _has_fixed_incidence(neighbour)
            ):
                queue.append(neighbour)
    if removed_dangling_link_ids:
        result = result.loc[
            ~result["id_link"].astype(int).isin(removed_dangling_link_ids)
        ].copy()
        inserted = inserted.loc[
            ~inserted["id_link"].astype(int).isin(removed_dangling_link_ids)
        ].copy()
    if inserted.empty:
        return result, {
            "method": "parent_order_harmonic_inserted_components",
            "component_count": 0,
            "oriented_component_count": 0,
            "reoriented_link_count": 0,
            "removed_one_attachment_block_link_ids": sorted(
                removed_one_attachment_block_link_ids
            ),
            "removed_dangling_link_ids": sorted(removed_dangling_link_ids),
        }
    removed_single_attachment_link_ids: set[int] = set()
    for component_nodes in list(nx.connected_components(graph)):
        component_nodes = {int(value) for value in component_nodes}
        attachments = component_nodes.intersection(protected_attachments)
        if len(attachments) >= 2:
            continue
        removed_single_attachment_link_ids.update(
            int(key)
            for node_a, node_b, key in graph.edges(keys=True)
            if int(key) in inserted_link_ids
            if int(node_a) in component_nodes and int(node_b) in component_nodes
        )
    if removed_single_attachment_link_ids:
        result = result.loc[
            ~result["id_link"].astype(int).isin(removed_single_attachment_link_ids)
        ].copy()
        inserted = inserted.loc[
            ~inserted["id_link"].astype(int).isin(removed_single_attachment_link_ids)
        ].copy()
        graph.remove_edges_from([
            (node_a, node_b, key)
            for node_a, node_b, key in graph.edges(keys=True)
            if int(key) in removed_single_attachment_link_ids
        ])
        graph.remove_nodes_from(list(nx.isolates(graph)))
    if inserted.empty:
        return result, {
            "method": "parent_order_harmonic_inserted_components",
            "component_count": 0,
            "oriented_component_count": 0,
            "reoriented_link_count": 0,
            "removed_dangling_link_ids": sorted(removed_dangling_link_ids),
            "removed_single_attachment_link_ids": sorted(
                removed_single_attachment_link_ids
            ),
        }
    node_index = nodes.set_index(nodes["id_node"].astype(int), drop=False)
    reoriented = 0
    oriented_components = 0
    parent_order_retry_components: list[dict[str, Any]] = []
    skipped_components: list[dict[str, Any]] = []
    for component_nodes in nx.connected_components(graph):
        component_nodes = {int(value) for value in component_nodes}
        component_link_ids = {
            int(row.id_link)
            for row in inserted.itertuples(index=False)
            if int(row.id_us_node) in component_nodes
            and int(row.id_ds_node) in component_nodes
        }
        if not component_link_ids:
            continue
        attachments = sorted(
            component_nodes.intersection(protected_attachments),
            key=lambda node_id: (int(parent_node_order[node_id]), int(node_id)),
        )
        if len(attachments) < 2:
            skipped_components.append({
                "link_ids": sorted(component_link_ids),
                "parent_attachment_node_ids": attachments,
                "reason": "fewer_than_two_parent_attachments",
            })
            continue
        sources = component_nodes.intersection(protected_source_attachments)
        sinks = component_nodes.intersection(protected_sink_attachments)
        if not (sources - sinks) or not (sinks - sources):
            # A degenerate component whose available interfaces all share one
            # role needs a deterministic last-resort ordering. Normal selected
            # domains use the explicit contract roles above.
            upstream_order = min(
                int(parent_node_order[node_id]) for node_id in attachments
            )
            downstream_order = max(
                int(parent_node_order[node_id]) for node_id in attachments
            )
            if upstream_order == downstream_order:
                skipped_components.append({
                    "link_ids": sorted(component_link_ids),
                    "parent_attachment_node_ids": attachments,
                    "reason": "parent_attachments_have_equal_order",
                })
                continue
            sources = {
                node_id for node_id in attachments
                if int(parent_node_order[node_id]) == upstream_order
            }
            sinks = {
                node_id for node_id in attachments
                if int(parent_node_order[node_id]) == downstream_order
            }
        component_scope_link_ids = {
            int(key)
            for node_a, node_b, key in graph.edges(keys=True)
            if int(node_a) in component_nodes and int(node_b) in component_nodes
        }
        component_links = result.loc[
            result["id_link"].astype(int).isin(component_scope_link_ids)
        ].copy()
        component_node_frame = node_index.loc[sorted(component_nodes)].copy()
        component_node_frame["is_inlet"] = component_node_frame["id_node"].astype(int).isin(sources)
        component_node_frame["is_outlet"] = component_node_frame["id_node"].astype(int).isin(sinks)
        edge_nodes = {
            int(row.id_link): (int(row.id_us_node), int(row.id_ds_node))
            for row in component_links.itertuples(index=False)
        }
        fixed_orientation = {
            int(row.id_link): (int(row.id_us_node), int(row.id_ds_node))
            for row in component_links.itertuples(index=False)
            if int(row.id_link) in fixed_scaffold_link_ids
        }
        try:
            orientation = _boundary_harmonic_orientation(
                edge_nodes=edge_nodes,
                child_links=component_links,
                child_nodes=component_node_frame,
                child_inlet_nodes=sources,
                child_outlet_nodes=sinks,
                allow_additional_topological_terminals=True,
                fixed_orientation_by_link=fixed_orientation,
            )
        except ValueError as error:
            # The unchanged parent links above are already frozen to their
            # authoritative directions. The unconstrained harmonic field can
            # nevertheless point an adjacent regenerated connector against
            # that scaffold and create a directed cycle. Preserve the current
            # behaviour for every successful component; only retry a failed
            # DAG with stable parent-node order as additional Dirichlet data.
            if (
                "did not produce a valid cut-to-cut DAG" not in str(error)
                or not fixed_orientation
            ):
                raise
            prescribed = set(map(int, sources)).union(map(int, sinks))
            stable_parent_nodes = {
                int(node_id)
                for node_id in component_nodes
                if int(node_id) in parent_node_order
            }
            if not prescribed.issubset(stable_parent_nodes):
                raise
            maximum_parent_order = max(
                int(value) for value in parent_node_order.values()
            )
            denominator = float(max(maximum_parent_order, 1))
            fixed_node_potential = {
                int(node_id): 1.0
                - float(parent_node_order[int(node_id)]) / denominator
                for node_id in stable_parent_nodes
            }
            orientation = _boundary_harmonic_orientation(
                edge_nodes=edge_nodes,
                child_links=component_links,
                child_nodes=component_node_frame,
                child_inlet_nodes=sources,
                child_outlet_nodes=sinks,
                allow_additional_topological_terminals=True,
                fixed_orientation_by_link=fixed_orientation,
                fixed_node_potential_by_node=fixed_node_potential,
            )
            parent_order_retry_components.append({
                "link_ids": sorted(component_link_ids),
                "fixed_scaffold_link_ids": sorted(fixed_orientation),
                "constrained_parent_node_ids": sorted(fixed_node_potential),
                "initial_error": str(error),
            })
        for link_id, (upstream, downstream) in orientation.items():
            if int(link_id) not in component_link_ids:
                continue
            mask = result["id_link"].astype(int).eq(int(link_id))
            current = result.loc[mask].iloc[0]
            if (
                int(current["id_us_node"]) != int(upstream)
                or int(current["id_ds_node"]) != int(downstream)
            ):
                geometry = _linear_geometry(current.geometry)
                result.loc[mask, result.geometry.name] = gpd.GeoSeries(
                    [LineString(list(geometry.coords)[::-1])],
                    index=result.index[mask],
                    crs=result.crs,
                )
                reoriented += 1
            result.loc[mask, "id_us_node"] = int(upstream)
            result.loc[mask, "id_ds_node"] = int(downstream)
        oriented_components += 1
    return result, {
        "method": "parent_order_harmonic_hybrid_replacement_scaffold",
        "component_count": int(nx.number_connected_components(graph)),
        "oriented_component_count": int(oriented_components),
        "reoriented_link_count": int(reoriented),
        "fixed_scaffold_link_ids": sorted(fixed_scaffold_link_ids),
        "parent_order_retry_component_count": int(
            len(parent_order_retry_components)
        ),
        "parent_order_retry_components": parent_order_retry_components,
        "removed_one_attachment_block_link_ids": sorted(
            removed_one_attachment_block_link_ids
        ),
        "removed_dangling_link_ids": sorted(removed_dangling_link_ids),
        "removed_single_attachment_link_ids": sorted(
            removed_single_attachment_link_ids
        ),
        "skipped_components": skipped_components,
    }


def _remove_post_direction_reverse_parent_duplicates(
    links: gpd.GeoDataFrame,
    *,
    inserted_link_ids: set[int],
    retained_parent_link_ids: set[int],
    geometry_tolerance_m: float,
    minimum_overlap_fraction: float,
) -> tuple[gpd.GeoDataFrame, dict[int, int], list[dict[str, Any]]]:
    """Remove inserted links that become reverse parent duplicates.

    The authoritative direction pass occurs after the first conservative
    parent/local match. It can therefore reveal a reverse duplicate that was
    not visible during that earlier match. Endpoint equality alone is not
    sufficient: distinct parallel channels may share nodes. Both directed
    reversal and near-complete bidirectional geometric coverage are required.
    """
    result = links.copy()
    by_id = result.set_index(result["id_link"].astype(int), drop=False)
    retained_by_reverse_pair: dict[tuple[int, int], list[int]] = {}
    for link_id in sorted(retained_parent_link_ids.intersection(by_id.index)):
        row = by_id.loc[int(link_id)]
        retained_by_reverse_pair.setdefault(
            (int(row["id_ds_node"]), int(row["id_us_node"])), []
        ).append(int(link_id))

    removed: dict[int, int] = {}
    audit: list[dict[str, Any]] = []
    for inserted_id in sorted(inserted_link_ids.intersection(by_id.index)):
        inserted_row = by_id.loc[int(inserted_id)]
        candidates = retained_by_reverse_pair.get(
            (int(inserted_row["id_us_node"]), int(inserted_row["id_ds_node"])),
            [],
        )
        matches: list[tuple[float, int, float, float]] = []
        for parent_id in candidates:
            (
                equivalent,
                hausdorff,
                inserted_coverage,
                parent_coverage,
            ) = _geometries_are_equivalent(
                inserted_row.geometry,
                by_id.loc[int(parent_id)].geometry,
                tolerance_m=geometry_tolerance_m,
                minimum_overlap_fraction=minimum_overlap_fraction,
            )
            if equivalent:
                matches.append(
                    (hausdorff, int(parent_id), inserted_coverage, parent_coverage)
                )
        if not matches:
            continue
        hausdorff, parent_id, inserted_coverage, parent_coverage = min(matches)
        removed[int(inserted_id)] = int(parent_id)
        audit.append(
            {
                "inserted_link_id": int(inserted_id),
                "retained_parent_link_id": int(parent_id),
                "hausdorff_distance_m": float(hausdorff),
                "inserted_coverage_fraction": float(inserted_coverage),
                "parent_coverage_fraction": float(parent_coverage),
            }
        )
    if removed:
        result = result.loc[
            ~result["id_link"].astype(int).isin(removed)
        ].copy()
    return result, removed, audit


def _validate_anchored_replacement_incidence(
    links: gpd.GeoDataFrame,
    *,
    replacement_link_ids: set[int],
    upstream_boundary_node_ids: set[int],
    downstream_boundary_node_ids: set[int],
) -> list[dict[str, Any]]:
    """Require the replacement graph to enter and leave its parent interfaces.

    The complete river graph can still have one global source and sink while a
    local replacement silently loses one boundary connection.  Validate the
    selected-domain incidence itself: upstream interfaces require an outgoing
    replacement edge and downstream interfaces require an incoming one.  A
    shared-role node requires both.
    """
    available_ids = set(links["id_link"].astype(int))
    replacement_ids = set(map(int, replacement_link_ids)).intersection(available_ids)
    replacement = links.loc[
        links["id_link"].astype(int).isin(replacement_ids)
    ]
    boundary_ids = sorted(
        set(map(int, upstream_boundary_node_ids)).union(
            map(int, downstream_boundary_node_ids)
        )
    )
    audit: list[dict[str, Any]] = []
    issues: list[str] = []
    for node_id in boundary_ids:
        incoming = sorted(
            replacement.loc[
                replacement["id_ds_node"].astype(int).eq(int(node_id)),
                "id_link",
            ].astype(int)
        )
        outgoing = sorted(
            replacement.loc[
                replacement["id_us_node"].astype(int).eq(int(node_id)),
                "id_link",
            ].astype(int)
        )
        requires_outgoing = int(node_id) in upstream_boundary_node_ids
        requires_incoming = int(node_id) in downstream_boundary_node_ids
        incoming_valid = bool(incoming) if requires_incoming else True
        outgoing_valid = bool(outgoing) if requires_outgoing else True
        audit.append(
            {
                "parent_boundary_node_id": int(node_id),
                "requires_replacement_incoming": bool(requires_incoming),
                "requires_replacement_outgoing": bool(requires_outgoing),
                "replacement_incoming_link_ids": incoming,
                "replacement_outgoing_link_ids": outgoing,
                "incidence_valid": bool(incoming_valid and outgoing_valid),
            }
        )
        if not incoming_valid:
            issues.append(f"downstream boundary {node_id} has no incoming replacement link")
        if not outgoing_valid:
            issues.append(f"upstream boundary {node_id} has no outgoing replacement link")
    if issues:
        raise ValueError(
            "Merged local-patch replacement violates anchored boundary incidence: "
            + "; ".join(issues)
            + "."
        )
    return audit


def _undirected_cycle_rank(links: gpd.GeoDataFrame) -> int:
    """Return the independent-cycle count of a possibly parallel graph."""
    graph = nx.MultiGraph()
    for row in links.itertuples(index=False):
        graph.add_edge(
            int(row.id_us_node),
            int(row.id_ds_node),
            key=int(row.id_link),
        )
    if graph.number_of_edges() == 0:
        return 0
    return int(
        graph.number_of_edges()
        - graph.number_of_nodes()
        + nx.number_connected_components(graph)
    )


def _expected_cycle_reduction(
    variant_result: LocalPatchVariantResult,
    *,
    parent_links: gpd.GeoDataFrame | None = None,
    selected_link_ids: Sequence[int] | None = None,
) -> int | None:
    """Return the independent-cycle count removed by the selected topology.

    The scientific expectation comes from the selected parent subgraph, not
    from the number of raster land components filled.  A compound or nested
    selection can contain several filled components that represent fewer
    independent graph cycles.  The raster audit remains a legacy fallback for
    callers that do not have the parent topology available.
    """
    if parent_links is not None and selected_link_ids is not None:
        selected_ids = {int(value) for value in selected_link_ids}
        selected = parent_links.loc[
            pd.to_numeric(parent_links["id_link"], errors="coerce").isin(selected_ids)
        ]
        if len(selected) != len(selected_ids):
            present = set(pd.to_numeric(selected["id_link"], errors="coerce").dropna().astype(int))
            missing = sorted(selected_ids.difference(present))
            raise ValueError(
                "Could not derive selected-parent cycle rank because parent links "
                f"are missing: {missing}."
            )
        return _undirected_cycle_rank(selected)

    path_value = variant_result.output_paths.get("collapse_components")
    if not path_value:
        return None
    path = Path(path_value)
    if not path.exists():
        return None
    components = pd.read_csv(path)
    if "component_ownership_audit" not in components.columns:
        return None
    accepted_component_ids: set[tuple[str, int]] = set()
    for row_index, value in components["component_ownership_audit"].items():
        if pd.isna(value) or not str(value).strip():
            continue
        records = json.loads(str(value))
        for record in records:
            if str(record.get("ownership_status", "")) != "accepted_selected_unit":
                continue
            accepted_component_ids.add(
                (str(row_index), int(record["candidate_component_id"]))
            )
    return int(len(accepted_component_ids))


def merge_local_patch_variant(
    *,
    contract: LocalPatchContract,
    raster_result: LocalPatchRasterResult,
    variant_result: LocalPatchVariantResult,
    parent_links_path: str | Path,
    parent_nodes_path: str | Path,
    output_dir: str | Path,
    match_tolerance_m: float | None = None,
    boundary_match_tolerance_m: float | None = None,
    unchanged_overlap_fraction: float = 0.95,
    replacement_parent_link_ids: tuple[int, ...] | None = None,
    upstream_boundary_node_ids: tuple[int, ...] | None = None,
    downstream_boundary_node_ids: tuple[int, ...] | None = None,
    merge_stage: str = "scientific_anchored",
) -> LocalPatchMergeResult:
    """Conservatively replace a prescribed contract domain in the parent graph.

    Defaults preserve the historical scientific anchored merge.  Explicit
    replacement and boundary sets are used by the standalone local-realized
    assembler, whose computational splice must occur before scientific
    anchoring.
    """
    if variant_result.status != "complete_raw_local_variant":
        raise ValueError("Phase-5 merge requires a complete raw local variant.")
    parent_links = gpd.read_file(parent_links_path)
    parent_nodes = gpd.read_file(parent_nodes_path)
    local_links = gpd.read_file(variant_result.output_paths["raw_rivgraph_links"])
    local_nodes = gpd.read_file(variant_result.output_paths["raw_rivgraph_nodes"])
    if parent_links.crs is None or parent_nodes.crs is None or local_links.crs is None:
        raise ValueError("Parent and local graph products require a CRS.")
    local_links = local_links.to_crs(parent_links.crs)
    local_nodes = local_nodes.to_crs(parent_nodes.crs)
    cuts = gpd.read_file(raster_result.output_paths["main_bank_cuts"]).to_crs(parent_nodes.crs)
    with rasterio.open(variant_result.output_paths["local_collapsed_mask"]) as source:
        pixel_size = float(max(abs(source.transform.a), abs(source.transform.e)))
    # RivGraph terminals are centerline pixels beside the multi-pixel cut
    # barrier, rather than points on the vector cut itself.  Five pixels
    # accommodates that deterministic raster offset while remaining far below
    # the separation between the two cut interfaces.
    tolerance = float(match_tolerance_m or pixel_size * 5.0)
    boundary_tolerance = float(
        boundary_match_tolerance_m
        if boundary_match_tolerance_m is not None
        else max(tolerance, pixel_size * 8.0)
    )
    if boundary_tolerance < tolerance:
        raise ValueError(
            "boundary_match_tolerance_m cannot be smaller than match_tolerance_m."
        )

    replacement_ids = set(
        int(value)
        for value in (
            replacement_parent_link_ids
            if replacement_parent_link_ids is not None
            else contract.anchored_replacement_link_ids
        )
    )
    prescribed_upstream_boundary_ids = tuple(
        int(value)
        for value in (
            upstream_boundary_node_ids
            if upstream_boundary_node_ids is not None
            else contract.anchored_upstream_boundary_node_ids
        )
    )
    prescribed_downstream_boundary_ids = tuple(
        int(value)
        for value in (
            downstream_boundary_node_ids
            if downstream_boundary_node_ids is not None
            else contract.anchored_downstream_boundary_node_ids
        )
    )
    if not replacement_ids:
        raise ValueError("A local merge requires at least one replacement parent link.")
    missing_replacement_ids = sorted(
        replacement_ids.difference(parent_links["id_link"].astype(int))
    )
    if missing_replacement_ids:
        raise ValueError(
            "Replacement parent links are absent from the parent graph: "
            f"{missing_replacement_ids}."
        )
    parent_node_ids = set(parent_nodes["id_node"].astype(int))
    missing_boundary_ids = sorted(
        set(prescribed_upstream_boundary_ids)
        .union(prescribed_downstream_boundary_ids)
        .difference(parent_node_ids)
    )
    if missing_boundary_ids:
        raise ValueError(
            "Prescribed local merge boundaries are absent from the parent graph: "
            f"{missing_boundary_ids}."
        )

    parent_order, _, parent_report = _infer_parent_node_order(parent_links, parent_nodes)
    raw_cut_binding = validate_cut_terminal_graph(
        local_links,
        local_nodes,
        cuts,
        tolerance_m=boundary_tolerance,
        require_directed_path=False,
    )
    cut_source_id = int(raw_cut_binding["upstream_terminal_node_id"])
    cut_sink_id = int(raw_cut_binding["downstream_terminal_node_id"])
    local_links, local_nodes, cut_corridor_pruning = _prune_raw_local_to_cut_corridor(
        local_links,
        local_nodes,
        source_node_id=cut_source_id,
        sink_node_id=cut_sink_id,
    )
    local_nodes = local_nodes.copy()
    local_nodes["rivgraph_is_inlet"] = local_nodes["is_inlet"].fillna(False).astype(bool)
    local_nodes["rivgraph_is_outlet"] = local_nodes["is_outlet"].fillna(False).astype(bool)
    local_nodes["is_inlet"] = local_nodes["id_node"].astype(int).eq(cut_source_id)
    local_nodes["is_outlet"] = local_nodes["id_node"].astype(int).eq(cut_sink_id)
    local_nodes["type_io"] = np.select(
        [local_nodes["is_inlet"], local_nodes["is_outlet"]],
        ["inlet", "outlet"],
        default="neither",
    )
    # Allocate and trim the scientific geometry before interpreting direction.
    # RivGraph's local inlet/outlet choice is deliberately not authoritative.
    trimmed_local_links, trimmed_local_nodes, bindings = _trim_to_anchored_boundaries(
        local_links,
        local_nodes,
        parent_nodes,
        contract,
        parent_links=parent_links,
        tolerance=boundary_tolerance,
        junction_snap_tolerance_m=pixel_size * 0.75,
        prune_undirected_context=True,
        upstream_boundary_node_ids=prescribed_upstream_boundary_ids,
        downstream_boundary_node_ids=prescribed_downstream_boundary_ids,
        selected_parent_link_ids=tuple(sorted(replacement_ids)),
    )
    anchored_source_ids = set(
        bindings.loc[bindings["cut_role"].eq("upstream"), "child_id_node"].astype(int)
    )
    anchored_sink_ids = set(
        bindings.loc[bindings["cut_role"].eq("downstream"), "child_id_node"].astype(int)
    )
    trimmed_local_nodes = trimmed_local_nodes.copy()
    # Computational cut context has now been removed geometrically. Direction
    # hints must therefore be attached to the exact scientific interfaces,
    # not to the former outer cut terminals (which are no longer present).
    trimmed_local_nodes["is_inlet"] = trimmed_local_nodes["id_node"].astype(int).isin(
        anchored_source_ids
    )
    trimmed_local_nodes["is_outlet"] = trimmed_local_nodes["id_node"].astype(int).isin(
        anchored_sink_ids
    )
    trimmed_local_nodes["type_io"] = np.select(
        [trimmed_local_nodes["is_inlet"], trimmed_local_nodes["is_outlet"]],
        ["inlet", "outlet"],
        default="neither",
    )
    provisional_node_match = _match_child_nodes_to_parent_nodes(
        parent_nodes,
        trimmed_local_nodes,
        match_tolerance=tolerance,
        parent_node_order=parent_order,
    )
    provisional_node_match = _force_boundary_node_matches(
        provisional_node_match,
        bindings,
        parent_order,
        trimmed_local_links,
    )
    provisional_link_match = _match_child_links_to_parent_links(
        parent_links,
        trimmed_local_links,
        match_tolerance=tolerance,
    )
    if "len" not in trimmed_local_links.columns:
        trimmed_local_links = trimmed_local_links.copy()
        trimmed_local_links["len"] = trimmed_local_links.geometry.length.astype(float)
    try:
        (
            directed_local_links,
            directed_local_nodes,
            direction_summary,
        ) = _assign_variant_directions(
            child_links=trimmed_local_links,
            child_nodes=trimmed_local_nodes,
            parent_links=parent_links,
            parent_nodes=parent_nodes,
            node_match=provisional_node_match,
            link_match=provisional_link_match,
            canonicalize_parent_boundaries=False,
            require_child_boundary_flag_match=True,
            allow_boundary_potential_fallback=True,
        )
    except ValueError as error:
        # Direction inside the raw crop is only a hint. Multi-interface and
        # shared-interface domains can be impossible to validate before the
        # unchanged parent continuations are restored. Keep the already
        # validated undirected anchored geometry and let the authoritative
        # post-merge parent-order pass orient it. The final merged graph still
        # has to pass the strict global DAG, source/sink, and anchored-interface
        # reachability gates below, so this does not weaken acceptance.
        directed_local_links = trimmed_local_links.copy()
        directed_local_nodes = trimmed_local_nodes.copy()
        direction_summary = {
            "method": "deferred_provisional_direction_after_geometry_validation",
            "provisional_direction_error_type": type(error).__name__,
            "provisional_direction_error": str(error),
            "is_valid": False,
            "issues": [str(error)],
        }
    anchored_interval_summary = {
        "method": "undirected_geometry_interval_before_parent_merge",
        "retained_link_count": int(len(directed_local_links)),
        "retained_node_count": int(len(directed_local_nodes)),
        "source_node_ids": sorted(anchored_source_ids),
        "sink_node_ids": sorted(anchored_sink_ids),
        "direction_validation_deferred_until_after_parent_merge": True,
    }
    direction_summary = {
        **dict(direction_summary),
        "method": "geometry_first_anchored_trim_merge_then_final_direction_validation",
        "raw_cut_binding": raw_cut_binding,
        "cut_corridor_pruning": cut_corridor_pruning,
        "anchored_interval_pruning": anchored_interval_summary,
        "anchored_source_node_ids": sorted(anchored_source_ids),
        "anchored_sink_node_ids": sorted(anchored_sink_ids),
    }
    node_match = _match_child_nodes_to_parent_nodes(
        parent_nodes,
        directed_local_nodes,
        match_tolerance=tolerance,
        parent_node_order=parent_order,
    )
    node_match = _force_boundary_node_matches(
        node_match, bindings, parent_order, directed_local_links
    )
    selected_parent_ids = {int(value) for value in contract.selected_link_ids}
    protected_selected_node_ids = _selected_parent_node_ids(
        parent_links,
        selected_parent_ids,
    )
    realized_identity_tolerance = float(
        match_tolerance_m
        if match_tolerance_m is not None
        else pixel_size * 1.25
    )
    node_match = _apply_realized_node_identity_policy(
        node_match,
        merge_stage=str(merge_stage),
        protected_selected_parent_node_ids=protected_selected_node_ids,
        identity_tolerance_m=realized_identity_tolerance,
    )
    link_match = _match_child_links_to_parent_links(
        parent_links, directed_local_links, match_tolerance=tolerance
    )
    if link_match.empty:
        link_match = pd.DataFrame(columns=["child_id_link", "candidate_class"])

    max_parent_node = int(parent_nodes["id_node"].astype(int).max())
    next_node = max_parent_node + 1
    node_id_map: dict[int, int] = {}
    for row in node_match.sort_values("child_id_node").itertuples(index=False):
        raw_id = int(row.child_id_node)
        if bool(row.identity_accepted) and not pd.isna(row.identity_parent_node_id):
            node_id_map[raw_id] = int(row.identity_parent_node_id)
        else:
            node_id_map[raw_id] = next_node
            next_node += 1
    node_match["final_node_id"] = node_match["child_id_node"].astype(int).map(node_id_map)
    node_match["final_node_method"] = node_match["identity_decision"]

    parent_node_by_id = parent_nodes.set_index(parent_nodes["id_node"].astype(int), drop=False)
    parent_link_by_id = parent_links.set_index(parent_links["id_link"].astype(int), drop=False)
    parent_connector_ids_by_pair: dict[frozenset[int], list[int]] = {}
    for parent_row in parent_links.itertuples(index=False):
        parent_connector_ids_by_pair.setdefault(
            frozenset((int(parent_row.id_us_node), int(parent_row.id_ds_node))),
            [],
        ).append(int(parent_row.id_link))
    raw_node_by_id = directed_local_nodes.set_index(directed_local_nodes["id_node"].astype(int), drop=False)
    matches_by_child = (
        {int(key): value.copy() for key, value in link_match.groupby("child_id_link")}
        if not link_match.empty else {}
    )
    strict_parent_by_raw: dict[int, int] = {}
    candidate_parent_by_raw: dict[int, int] = {}
    candidate_parent_endpoints_by_raw: dict[int, tuple[int, int]] = {}
    candidate_overlap_ok_by_raw: dict[int, bool] = {}
    mapped_endpoints_by_raw: dict[int, tuple[int, int]] = {}
    audit_records: list[dict[str, Any]] = []
    for row in directed_local_links.itertuples(index=False):
        raw_id = int(row.id_link)
        mapped_us = node_id_map[int(row.id_us_node)]
        mapped_ds = node_id_map[int(row.id_ds_node)]
        frame = matches_by_child.get(raw_id)
        candidate = None
        if frame is not None:
            domain = frame.loc[frame["parent_id_link"].astype(int).isin(replacement_ids)]
            if not domain.empty:
                candidate = domain.sort_values("candidate_rank").iloc[0]
        strict = False
        reason = "no_parent_candidate"
        mapped_endpoints_by_raw[raw_id] = (int(mapped_us), int(mapped_ds))
        if candidate is not None:
            parent_id = int(candidate["parent_id_link"])
            candidate_parent_by_raw[raw_id] = parent_id
            candidate_parent_endpoints_by_raw[raw_id] = (
                int(candidate["parent_id_us_node"]),
                int(candidate["parent_id_ds_node"]),
            )
            endpoints_match = (
                mapped_us == int(candidate["parent_id_us_node"])
                and mapped_ds == int(candidate["parent_id_ds_node"])
            )
            overlaps_match = (
                float(candidate["child_core_overlap_fraction"]) >= unchanged_overlap_fraction
                and float(candidate["parent_core_overlap_fraction"]) >= unchanged_overlap_fraction
            )
            candidate_overlap_ok_by_raw[raw_id] = bool(overlaps_match)
            strict = endpoints_match and overlaps_match
            reason = "strict_unchanged" if strict else "changed_geometry_or_topology"
            if strict:
                strict_parent_by_raw[raw_id] = parent_id
        audit_records.append({
            "raw_local_link_id": raw_id,
            "candidate_parent_link_id": int(candidate["parent_id_link"]) if candidate is not None else pd.NA,
            "mapped_us_node": mapped_us,
            "mapped_ds_node": mapped_ds,
            "child_core_overlap_fraction": float(candidate["child_core_overlap_fraction"]) if candidate is not None else np.nan,
            "parent_core_overlap_fraction": float(candidate["parent_core_overlap_fraction"]) if candidate is not None else np.nan,
            "merge_decision": "retain_parent" if strict else "insert_local",
            "decision_reason": reason,
            "bounded_same_endpoint_reconstruction": False,
            "same_endpoint_parent_connector_id": pd.NA,
            "same_endpoint_reconstruction_limit_m": np.nan,
            "same_endpoint_selected_connector_rejected": False,
            "realized_lineage_class": (
                "unchanged_parent"
                if strict
                else (
                    "partial_parent_overlap_interface"
                    if candidate is not None
                    and (
                        float(candidate["child_core_overlap_fraction"]) > 0.0
                        or float(candidate["parent_core_overlap_fraction"]) > 0.0
                    )
                    else "changed_local"
                )
            ),
        })

    retained_context = set(strict_parent_by_raw.values())
    duplicate_parent_by_raw: dict[int, int] = {}
    for raw_id, parent_id in candidate_parent_by_raw.items():
        if raw_id in strict_parent_by_raw or parent_id not in retained_context:
            continue
        mapped_pair = mapped_endpoints_by_raw[raw_id]
        parent_pair = candidate_parent_endpoints_by_raw[raw_id]
        if (
            candidate_overlap_ok_by_raw.get(raw_id, False)
            and set(mapped_pair) == set(parent_pair)
        ):
            duplicate_parent_by_raw[raw_id] = parent_id
            for record in audit_records:
                if int(record["raw_local_link_id"]) == raw_id:
                    record["merge_decision"] = "discard_duplicate_parent_geometry"
                    record["decision_reason"] = (
                        "same_parent_core_and_endpoints_already_retained"
                    )
                    break
    outside = parent_links.loc[
        ~parent_links["id_link"].astype(int).isin(replacement_ids - retained_context)
    ].copy()
    retained_reverse_lookup: dict[tuple[int, int], list[int]] = {}
    for parent_row in outside.itertuples(index=False):
        retained_reverse_lookup.setdefault(
            (int(parent_row.id_ds_node), int(parent_row.id_us_node)), []
        ).append(int(parent_row.id_link))
    outside_by_id = outside.set_index(outside["id_link"].astype(int), drop=False)
    max_parent_link = int(parent_links["id_link"].astype(int).max())
    next_link = max_parent_link + 1
    inserted_rows: list[dict[str, Any]] = []
    final_link_by_raw: dict[int, int] = {}
    pre_direction_duplicate_audit: list[dict[str, Any]] = []
    for row in directed_local_links.itertuples(index=False):
        raw_id = int(row.id_link)
        if raw_id in strict_parent_by_raw:
            final_link_by_raw[raw_id] = strict_parent_by_raw[raw_id]
            continue
        if raw_id in duplicate_parent_by_raw:
            final_link_by_raw[raw_id] = duplicate_parent_by_raw[raw_id]
            continue
        final_id = next_link
        next_link += 1
        final_link_by_raw[raw_id] = final_id
        values = _empty_like(list(parent_links.columns))
        raw_values = row._asdict()
        for column in values:
            if column in raw_values and column != parent_links.geometry.name:
                values[column] = raw_values[column]
        raw_us, raw_ds = int(row.id_us_node), int(row.id_ds_node)
        final_us, final_ds = node_id_map[raw_us], node_id_map[raw_ds]
        final_us_geom = parent_node_by_id.loc[final_us].geometry if final_us in parent_node_by_id.index else raw_node_by_id.loc[raw_us].geometry
        final_ds_geom = parent_node_by_id.loc[final_ds].geometry if final_ds in parent_node_by_id.index else raw_node_by_id.loc[raw_ds].geometry
        same_endpoint_fallback_geometry = None
        same_endpoint_fallback_parent_id: int | None = None
        direct_parent_connector_ids = parent_connector_ids_by_pair.get(
            frozenset((int(final_us), int(final_ds))), []
        )
        eligible_direct_parent_connector_ids = _eligible_same_endpoint_connector_ids(
            direct_parent_connector_ids,
            merge_stage=str(merge_stage),
            selected_parent_link_ids=selected_parent_ids,
        )
        if (
            str(merge_stage) == "local_realized"
            and direct_parent_connector_ids
            and not eligible_direct_parent_connector_ids
        ):
            for record in audit_records:
                if int(record["raw_local_link_id"]) == raw_id:
                    record["same_endpoint_selected_connector_rejected"] = True
                    break
        if len(eligible_direct_parent_connector_ids) == 1:
            # The matched raw branch can lie wholly beyond a very short
            # scientific interface, causing both exact boundary nodes to
            # project onto one raw endpoint. In that exceptional case the
            # unique direct parent connector is more authoritative than the
            # highest-overlap raw-link candidate.
            same_endpoint_fallback_geometry = parent_link_by_id.loc[
                int(eligible_direct_parent_connector_ids[0])
            ].geometry
            same_endpoint_fallback_parent_id = int(
                eligible_direct_parent_connector_ids[0]
            )
        elif candidate is not None:
            candidate_parent_id = int(candidate["parent_id_link"])
            candidate_endpoints = {
                int(candidate["parent_id_us_node"]),
                int(candidate["parent_id_ds_node"]),
            }
            if (
                candidate_parent_id in parent_link_by_id.index
                and not (
                    str(merge_stage) == "local_realized"
                    and candidate_parent_id in selected_parent_ids
                )
                and {int(final_us), int(final_ds)} == candidate_endpoints
            ):
                same_endpoint_fallback_geometry = parent_link_by_id.loc[
                    candidate_parent_id
                ].geometry
                same_endpoint_fallback_parent_id = int(candidate_parent_id)
        try:
            inserted_geometry = _snap_and_orient_geometry(
                row.geometry,
                raw_node_by_id.loc[raw_us].geometry,
                raw_node_by_id.loc[raw_ds].geometry,
                final_us_geom,
                final_ds_geom,
            )
        except ValueError as error:
            if "projections collapse to one indistinguishable position" not in str(error):
                raise
            inserted_geometry = _snap_and_orient_geometry(
                row.geometry,
                raw_node_by_id.loc[raw_us].geometry,
                raw_node_by_id.loc[raw_ds].geometry,
                final_us_geom,
                final_ds_geom,
                same_endpoint_fallback_geometry=same_endpoint_fallback_geometry,
                same_endpoint_reconstruction_limit_m=tolerance,
            )
            for record in audit_records:
                if int(record["raw_local_link_id"]) == raw_id:
                    record["bounded_same_endpoint_reconstruction"] = True
                    record["same_endpoint_parent_connector_id"] = (
                        same_endpoint_fallback_parent_id
                    )
                    record["same_endpoint_reconstruction_limit_m"] = float(
                        tolerance
                    )
                    break
        reverse_parent_candidates = retained_reverse_lookup.get(
            (int(final_us), int(final_ds)), []
        )
        equivalent_reverse_candidates: list[
            tuple[float, int, float, float]
        ] = []
        for retained_parent_id in reverse_parent_candidates:
            (
                equivalent,
                hausdorff,
                inserted_coverage,
                parent_coverage,
            ) = _geometries_are_equivalent(
                inserted_geometry,
                outside_by_id.loc[int(retained_parent_id)].geometry,
                tolerance_m=tolerance,
                minimum_overlap_fraction=unchanged_overlap_fraction,
            )
            if equivalent:
                equivalent_reverse_candidates.append(
                    (
                        float(hausdorff),
                        int(retained_parent_id),
                        float(inserted_coverage),
                        float(parent_coverage),
                    )
                )
        if equivalent_reverse_candidates:
            (
                hausdorff,
                retained_parent_id,
                inserted_coverage,
                parent_coverage,
            ) = min(equivalent_reverse_candidates)
            duplicate_parent_by_raw[raw_id] = int(retained_parent_id)
            final_link_by_raw[raw_id] = int(retained_parent_id)
            pre_direction_duplicate_audit.append(
                {
                    "raw_local_link_id": int(raw_id),
                    "retained_parent_link_id": int(retained_parent_id),
                    "hausdorff_distance_m": float(hausdorff),
                    "inserted_coverage_fraction": float(inserted_coverage),
                    "parent_coverage_fraction": float(parent_coverage),
                }
            )
            for record in audit_records:
                if int(record["raw_local_link_id"]) == raw_id:
                    record["merge_decision"] = "discard_reverse_duplicate_parent_geometry"
                    record["decision_reason"] = (
                        "reverse_endpoints_and_geometry_equivalent_to_retained_parent"
                    )
                    record["candidate_parent_link_id"] = int(retained_parent_id)
                    break
            continue
        values.update({
            "id_link": final_id,
            "id_us_node": final_us,
            "id_ds_node": final_ds,
            "id_nodes": f"{final_us}, {final_ds}",
            parent_links.geometry.name: inserted_geometry,
        })
        if "idx_link" in values:
            values["idx_link"] = None
        inserted_rows.append(values)

    if inserted_rows:
        inserted = gpd.GeoDataFrame(
            inserted_rows, geometry=parent_links.geometry.name, crs=parent_links.crs
        )
    else:
        inserted = gpd.GeoDataFrame(
            {column: pd.Series(dtype=parent_links[column].dtype) for column in parent_links.columns},
            geometry=parent_links.geometry.name,
            crs=parent_links.crs,
        )
    parent_boolean_columns = [
        column
        for column in parent_links.columns
        if column != parent_links.geometry.name and pd.api.types.is_bool_dtype(parent_links[column].dtype)
    ]
    for column in parent_boolean_columns:
        if column not in inserted.columns:
            inserted[column] = False
        else:
            inserted[column] = inserted[column].map(
                lambda value: False if pd.isna(value) else bool(value)
            ).astype(bool)
    all_null_inserted_columns = [
        column
        for column in inserted.columns
        if column != inserted.geometry.name
        and column not in {"id_link", "id_us_node", "id_ds_node", "id_nodes"}
        and inserted[column].isna().all()
    ]
    if all_null_inserted_columns:
        inserted = inserted.drop(columns=all_null_inserted_columns)
    merged_links = gpd.GeoDataFrame(
        pd.concat([outside, inserted], ignore_index=True),
        geometry=parent_links.geometry.name,
        crs=parent_links.crs,
    ).sort_values("id_link").reset_index(drop=True)
    for column in parent_boolean_columns:
        merged_links[column] = merged_links[column].map(
            lambda value: False if pd.isna(value) else bool(value)
        ).astype(bool)

    used_node_ids = set(merged_links["id_us_node"].astype(int)).union(
        merged_links["id_ds_node"].astype(int)
    )
    node_rows: list[dict[str, Any]] = []
    for node_id in sorted(used_node_ids):
        if node_id in parent_node_by_id.index:
            node_rows.append(parent_node_by_id.loc[node_id].to_dict())
        else:
            raw_id = next(key for key, value in node_id_map.items() if value == node_id)
            values = _empty_like(list(parent_nodes.columns))
            raw_values = raw_node_by_id.loc[raw_id].to_dict()
            for column in values:
                if column in raw_values:
                    values[column] = raw_values[column]
            values["id_node"] = node_id
            node_rows.append(values)
    merged_nodes = gpd.GeoDataFrame(node_rows, geometry=parent_nodes.geometry.name, crs=parent_nodes.crs)
    retained_selected_scaffold_ids = retained_context.intersection(replacement_ids)
    merged_links, final_direction_summary = _orient_inserted_components_from_parent_order(
        merged_links,
        merged_nodes,
        inserted_link_ids=set(inserted["id_link"].astype(int)),
        fixed_scaffold_link_ids=retained_selected_scaffold_ids,
        parent_node_order=parent_order,
        protected_source_node_ids={
            *prescribed_upstream_boundary_ids
        },
        protected_sink_node_ids={
            *prescribed_downstream_boundary_ids
        },
    )
    (
        merged_links,
        post_direction_duplicate_parent_by_inserted,
        post_direction_duplicate_audit,
    ) = _remove_post_direction_reverse_parent_duplicates(
        merged_links,
        inserted_link_ids=set(inserted["id_link"].astype(int)),
        retained_parent_link_ids=set(outside["id_link"].astype(int)),
        geometry_tolerance_m=tolerance,
        minimum_overlap_fraction=unchanged_overlap_fraction,
    )
    if post_direction_duplicate_parent_by_inserted:
        raw_by_final = {
            int(final_id): int(raw_id)
            for raw_id, final_id in final_link_by_raw.items()
        }
        for inserted_id, parent_id in post_direction_duplicate_parent_by_inserted.items():
            raw_id = raw_by_final.get(int(inserted_id))
            if raw_id is None:
                continue
            duplicate_parent_by_raw[int(raw_id)] = int(parent_id)
            final_link_by_raw[int(raw_id)] = int(parent_id)
            for record in audit_records:
                if int(record["raw_local_link_id"]) == int(raw_id):
                    record["merge_decision"] = (
                        "discard_post_direction_reverse_parent_duplicate"
                    )
                    record["decision_reason"] = (
                        "final_orientation_revealed_geometry_matched_reverse_parent"
                    )
                    record["candidate_parent_link_id"] = int(parent_id)
                    break
        inserted = inserted.loc[
            ~inserted["id_link"].astype(int).isin(
                post_direction_duplicate_parent_by_inserted
            )
        ].copy()
    final_direction_summary = {
        **dict(final_direction_summary),
        "post_direction_reverse_parent_duplicates": post_direction_duplicate_audit,
        "post_direction_reverse_parent_duplicate_count": int(
            len(post_direction_duplicate_parent_by_inserted)
        ),
    }
    final_replacement_link_ids = retained_selected_scaffold_ids.union(
        set(inserted["id_link"].astype(int))
    ).intersection(set(merged_links["id_link"].astype(int)))
    anchored_incidence_audit = _validate_anchored_replacement_incidence(
        merged_links,
        replacement_link_ids=final_replacement_link_ids,
        upstream_boundary_node_ids=set(
            prescribed_upstream_boundary_ids
        ),
        downstream_boundary_node_ids=set(
            prescribed_downstream_boundary_ids
        ),
    )
    final_used_node_ids = set(merged_links["id_us_node"].astype(int)).union(
        merged_links["id_ds_node"].astype(int)
    )
    merged_nodes = merged_nodes.loc[
        merged_nodes["id_node"].astype(int).isin(final_used_node_ids)
    ].copy()
    source_id, sink_id = int(parent_report.source_nodes[0]), int(parent_report.sink_nodes[0])
    merged_links, merged_nodes = _refresh_topology(
        merged_links, merged_nodes, source_id=source_id, sink_id=sink_id
    )
    merged_graph = build_directed_graph(merged_links, merged_nodes)
    report = validate_single_inlet_single_outlet(merged_graph)
    direction_validation_deferred = str(merge_stage) == "local_realized"
    if not direction_validation_deferred and not report.is_valid:
        raise ValueError(f"Merged local-patch graph failed validation: {report.issues}")
    if direction_validation_deferred:
        if not nx.is_weakly_connected(merged_graph):
            raise ValueError(
                "Computational realized splice is not weakly connected before "
                "canonical direction assignment."
            )
        isolated = sorted(
            int(node_id)
            for node_id in merged_graph.nodes
            if merged_graph.degree(node_id) == 0
        )
        if isolated:
            raise ValueError(
                "Computational realized splice contains isolated nodes before "
                f"canonical direction assignment: {isolated}."
            )
    elif not nx.is_directed_acyclic_graph(merged_graph):
        cycle = nx.find_cycle(merged_graph, orientation="original")
        cycle_summary = [
            [int(edge[0]), int(edge[1]), int(edge[2]) if len(edge) > 3 else None]
            for edge in cycle
        ]
        raise ValueError(
            "Merged local-patch graph is not acyclic; directed cycle edges: "
            f"{cycle_summary}."
        )
    parent_cycle_rank = _undirected_cycle_rank(parent_links)
    merged_cycle_rank = _undirected_cycle_rank(merged_links)
    observed_cycle_reduction = int(parent_cycle_rank - merged_cycle_rank)
    expected_cycle_reduction = _expected_cycle_reduction(
        variant_result,
        parent_links=parent_links,
        selected_link_ids=contract.selected_link_ids,
    )
    if (
        expected_cycle_reduction is not None
        and observed_cycle_reduction != int(expected_cycle_reduction)
        and not direction_validation_deferred
    ):
        raise ValueError(
            "Local-patch collapse topology is inconsistent with the raster edit: "
            f"selected-parent cycle rank={expected_cycle_reduction}, "
            f"observed cycle-rank reduction={observed_cycle_reduction}, "
            f"parent cycle rank={parent_cycle_rank}, "
            f"merged cycle rank={merged_cycle_rank}."
        )
    final_source_ids = {
        int(value) for value in prescribed_upstream_boundary_ids
    }
    final_sink_ids = {
        int(value) for value in prescribed_downstream_boundary_ids
    }
    unreachable_sources = sorted(
        source_id
        for source_id in final_source_ids
        if source_id not in merged_graph
        or not any(
            sink_id in merged_graph
            and nx.has_path(merged_graph, source_id, sink_id)
            for sink_id in final_sink_ids
        )
    )
    unreachable_sinks = sorted(
        sink_id
        for sink_id in final_sink_ids
        if sink_id not in merged_graph
        or not any(
            source_id in merged_graph
            and nx.has_path(merged_graph, source_id, sink_id)
            for source_id in final_source_ids
        )
    )
    if (
        not direction_validation_deferred
        and (unreachable_sources or unreachable_sinks)
    ):
        raise ValueError(
            "Final merged parent/local graph does not connect every anchored interface; "
            f"unreachable upstream={unreachable_sources}, "
            f"unreachable downstream={unreachable_sinks}."
        )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    links_path = output_dir / "merged_links.gpkg"
    nodes_path = output_dir / "merged_nodes.gpkg"
    node_audit_path = output_dir / "node_merge_audit.csv"
    link_audit_path = output_dir / "link_merge_audit.csv"
    manifest_path = output_dir / "local_patch_merge_manifest.json"
    merged_links.to_file(links_path, driver="GPKG")
    merged_nodes.to_file(nodes_path, driver="GPKG")
    node_match.to_csv(node_audit_path, index=False)
    pd.DataFrame.from_records(audit_records).assign(
        final_link_id=lambda frame: frame["raw_local_link_id"].map(final_link_by_raw)
    ).to_csv(link_audit_path, index=False)
    output_paths = {
        "merged_links": str(links_path.resolve()),
        "merged_nodes": str(nodes_path.resolve()),
        "node_merge_audit": str(node_audit_path.resolve()),
        "link_merge_audit": str(link_audit_path.resolve()),
        "manifest": str(manifest_path.resolve()),
    }
    summary = {
        "merge_stage": str(merge_stage),
        "selected_unit_ids": list(contract.selected_unit_ids),
        "replacement_parent_link_ids": sorted(replacement_ids),
        "retained_parent_link_count": int(len(outside)),
        "strict_context_link_count": int(len(retained_context)),
        "inserted_local_link_count": int(len(inserted)),
        "new_local_node_count": int(sum(value > max_parent_node for value in node_id_map.values())),
        "match_tolerance_m": tolerance,
        "realized_node_identity_tolerance_m": realized_identity_tolerance,
        "protected_selected_parent_node_ids": sorted(protected_selected_node_ids),
        "cut_terminal_tolerance_m": boundary_tolerance,
        "raw_cut_terminal_bindings": [],
        "matching_order": "provisional_direction_hints_then_trim_then_authoritative_parent_match",
        "direction_summary": dict(direction_summary),
        "final_inserted_direction_summary": final_direction_summary,
        "pre_direction_reverse_parent_duplicates": pre_direction_duplicate_audit,
        "preserved_boundary_interface_link_ids": [],
        "anchored_replacement_incidence": anchored_incidence_audit,
        "final_replacement_link_ids": sorted(final_replacement_link_ids),
        "validation_report": report.to_dict(),
        "is_directed_acyclic": bool(nx.is_directed_acyclic_graph(merged_graph)),
        "direction_validation_deferred_to_realized_assembler": bool(
            direction_validation_deferred
        ),
        "collapse_topology_validation": {
            "method": "selected_parent_subgraph_cycle_rank_reduction",
            "parent_cycle_rank": int(parent_cycle_rank),
            "merged_cycle_rank": int(merged_cycle_rank),
            "observed_cycle_reduction": int(observed_cycle_reduction),
            "expected_cycle_reduction": (
                int(expected_cycle_reduction)
                if expected_cycle_reduction is not None
                else None
            ),
            "passed": bool(
                expected_cycle_reduction is None
                or observed_cycle_reduction == int(expected_cycle_reduction)
            ),
        },
        "discarded_duplicate_parent_link_count": int(len(duplicate_parent_by_raw)),
        "discarded_duplicate_parent_link_ids": sorted(duplicate_parent_by_raw),
        "final_anchored_interface_reachability": {
            "upstream_parent_node_ids": sorted(final_source_ids),
            "downstream_parent_node_ids": sorted(final_sink_ids),
            "all_interfaces_connected": True,
        },
    }
    result = LocalPatchMergeResult("complete_conservative_merge", output_paths, summary)
    manifest_path.write_text(json.dumps(result.to_manifest(), indent=2, sort_keys=True) + "\n")
    return result


__all__ = [
    "LOCAL_PATCH_MERGE_VERSION",
    "LocalPatchMergeResult",
    "_prune_raw_local_to_cut_corridor",
    "merge_local_patch_variant",
]
