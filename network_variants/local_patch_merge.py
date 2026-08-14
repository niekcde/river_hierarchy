"""Phase-5 conservative merge of a raw local RivGraph into its parent graph.

The cut barriers are interfaces, not new scientific boundaries.  A terminal
created by each cut is therefore bound to the corresponding parent splice
anchor before local directions are inferred.  Parent links outside the local
replacement domain are retained verbatim; a local link is allowed to retain a
parent ID only after a strict geometry, endpoint, and one-to-one match.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping

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
from network_variants.local_patch_variant import LocalPatchVariantResult
from network_variants.variant_generation import (
    _assign_variant_directions,
    _infer_parent_node_order,
    _match_child_links_to_parent_links,
    _match_child_nodes_to_parent_nodes,
)


LOCAL_PATCH_MERGE_VERSION = "topology-local-patch-merge-v2"


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
) -> pd.DataFrame:
    result = node_match.copy()
    result["match_method"] = np.where(
        result["match_within_tolerance"].fillna(False), "spatial_parent_match", "new_local_node"
    )
    result["cut_role"] = ""
    result["cut_distance_m"] = np.nan
    result["raw_boundary_role_matches"] = False
    result["binding_method"] = ""
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
        result.loc[mask, "cut_role"] = str(binding.cut_role)
        result.loc[mask, "cut_distance_m"] = float(binding.cut_distance_m)
        result.loc[mask, "raw_boundary_role_matches"] = bool(binding.raw_boundary_role_matches)
        result.loc[mask, "binding_method"] = str(binding.binding_method)

    matched = result.loc[result["match_within_tolerance"].fillna(False)].copy()
    duplicate = matched["matched_parent_node_id"].dropna().astype(int).duplicated(keep=False)
    if duplicate.any():
        values = sorted(matched.loc[duplicate, "matched_parent_node_id"].astype(int).unique())
        raise ValueError(f"Multiple local nodes map to the same parent node: {values}.")
    return result


def _trim_to_anchored_boundaries(
    directed_links: gpd.GeoDataFrame,
    directed_nodes: gpd.GeoDataFrame,
    parent_nodes: gpd.GeoDataFrame,
    contract: LocalPatchContract,
    *,
    tolerance: float,
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, pd.DataFrame]:
    """Clip computational cut context to the anchored inner boundaries."""
    links = directed_links.copy()
    nodes = directed_nodes.copy()
    parent_index = parent_nodes.set_index(parent_nodes["id_node"].astype(int), drop=False)
    next_node_id = int(nodes["id_node"].astype(int).max()) + 1
    next_link_id = int(links["id_link"].astype(int).max()) + 1
    boundary_records: list[dict[str, Any]] = []

    boundary_specs = [
        ("upstream", contract.upstream_collar, int(parent_id))
        for parent_id in contract.anchored_upstream_boundary_node_ids
    ] + [
        ("downstream", contract.downstream_collar, int(parent_id))
        for parent_id in contract.anchored_downstream_boundary_node_ids
    ]
    for role, collar, parent_id in boundary_specs:
        parent_id = int(parent_id)
        target = parent_index.loc[parent_id].geometry
        node_distances = nodes.geometry.distance(target)
        nearest_node_index = node_distances.idxmin()
        nearest_node_distance = float(node_distances.loc[nearest_node_index])
        if nearest_node_distance <= tolerance:
            raw_boundary_id = int(nodes.loc[nearest_node_index, "id_node"])
            nodes.loc[nearest_node_index, nodes.geometry.name] = target
            method = "existing_local_node_at_anchored_boundary"
        else:
            link_distances = links.geometry.distance(target)
            ordered = link_distances.sort_values()
            if ordered.empty or float(ordered.iloc[0]) > tolerance:
                raise ValueError(
                    f"No local link or node lies within {tolerance:.3f} m of the "
                    f"{role} anchored boundary node {parent_id}."
                )
            if len(ordered) > 1 and float(ordered.iloc[1] - ordered.iloc[0]) <= 1e-6:
                raise ValueError(
                    f"Multiple local links are equally close to the {role} anchored "
                    f"boundary node {parent_id}."
                )
            link_index = ordered.index[0]
            row = links.loc[link_index].copy()
            line = _linear_geometry(row.geometry)
            measure = float(line.project(target))
            endpoint_margin = min(tolerance * 0.25, float(line.length) * 0.05)
            if measure <= endpoint_margin or measure >= float(line.length) - endpoint_margin:
                endpoint_id = int(row["id_us_node"] if measure <= endpoint_margin else row["id_ds_node"])
                node_index = nodes.index[nodes["id_node"].astype(int).eq(endpoint_id)][0]
                nodes.loc[node_index, nodes.geometry.name] = target
                raw_boundary_id = endpoint_id
                method = "existing_link_endpoint_at_anchored_boundary"
            else:
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
                first["id_ds_node"] = raw_boundary_id
                first["id_nodes"] = f"{int(first['id_us_node'])}, {raw_boundary_id}"
                first[links.geometry.name] = first_geometry
                second["id_link"] = next_link_id
                next_link_id += 1
                second["id_us_node"] = raw_boundary_id
                second["id_nodes"] = f"{raw_boundary_id}, {int(second['id_ds_node'])}"
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
        boundary_records.append(
            {
                "child_id_node": raw_boundary_id,
                "cut_role": role,
                "forced_parent_node_id": parent_id,
                "cut_distance_m": 0.0,
                "raw_boundary_role_matches": True,
                "binding_method": method,
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
    graph = nx.MultiDiGraph()
    for row in links.itertuples(index=False):
        graph.add_edge(int(row.id_us_node), int(row.id_ds_node), key=int(row.id_link))
    forward_nodes: set[int] = set()
    reverse_nodes: set[int] = set()
    for source_id in source_ids:
        if source_id in graph:
            forward_nodes.update({source_id, *nx.descendants(graph, source_id)})
    for sink_id in sink_ids:
        if sink_id in graph:
            reverse_nodes.update({sink_id, *nx.ancestors(graph, sink_id)})
    retained_nodes = forward_nodes & reverse_nodes
    if not retained_nodes or not all(
        any(nx.has_path(graph, source_id, sink_id) for sink_id in sink_ids)
        for source_id in source_ids
    ):
        raise ValueError("Anchored local boundaries are not connected in flow direction.")
    links = links.loc[
        links["id_us_node"].astype(int).isin(retained_nodes)
        & links["id_ds_node"].astype(int).isin(retained_nodes)
    ].copy()
    used_nodes = set(links["id_us_node"].astype(int)).union(links["id_ds_node"].astype(int))
    nodes = nodes.loc[nodes["id_node"].astype(int).isin(used_nodes)].copy()
    return links.reset_index(drop=True), nodes.reset_index(drop=True), pd.DataFrame(boundary_records)


def _snap_and_orient_geometry(geometry, raw_us: Point, raw_ds: Point, final_us: Point, final_ds: Point):
    line = _linear_geometry(geometry)
    coordinates = list(line.coords)
    forward = Point(coordinates[0]).distance(raw_us) + Point(coordinates[-1]).distance(raw_ds)
    reverse = Point(coordinates[-1]).distance(raw_us) + Point(coordinates[0]).distance(raw_ds)
    if reverse < forward:
        coordinates.reverse()
    line = LineString(coordinates)
    us_measure = float(line.project(final_us))
    ds_measure = float(line.project(final_ds))
    if us_measure >= ds_measure:
        raise ValueError(
            "Anchored boundary-node projections do not occur in upstream-to-downstream order."
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


def merge_local_patch_variant(
    *,
    contract: LocalPatchContract,
    raster_result: LocalPatchRasterResult,
    variant_result: LocalPatchVariantResult,
    parent_links_path: str | Path,
    parent_nodes_path: str | Path,
    output_dir: str | Path,
    match_tolerance_m: float | None = None,
    unchanged_overlap_fraction: float = 0.95,
) -> LocalPatchMergeResult:
    """Conservatively replace the contract domain in the parent graph."""
    if variant_result.status != "complete_raw_local_variant":
        raise ValueError("Phase-5 merge requires a complete raw local variant.")
    parent_links = gpd.read_file(parent_links_path)
    parent_nodes = gpd.read_file(parent_nodes_path)
    local_links = gpd.read_file(variant_result.output_paths["raw_rivgraph_links"])
    local_nodes = gpd.read_file(variant_result.output_paths["raw_rivgraph_nodes"])
    cuts = gpd.read_file(raster_result.output_paths["main_bank_cuts"])
    if parent_links.crs is None or parent_nodes.crs is None or local_links.crs is None:
        raise ValueError("Parent and local graph products require a CRS.")
    local_links = local_links.to_crs(parent_links.crs)
    local_nodes = local_nodes.to_crs(parent_nodes.crs)
    cuts = cuts.to_crs(parent_nodes.crs)
    with rasterio.open(variant_result.output_paths["local_collapsed_mask"]) as source:
        pixel_size = float(max(abs(source.transform.a), abs(source.transform.e)))
    # RivGraph terminals are centerline pixels beside the multi-pixel cut
    # barrier, rather than points on the vector cut itself.  Five pixels
    # accommodates that deterministic raster offset while remaining far below
    # the separation between the two cut interfaces.
    tolerance = float(match_tolerance_m or pixel_size * 5.0)
    boundary_tolerance = max(tolerance, pixel_size * 8.0)

    parent_order, _, parent_report = _infer_parent_node_order(parent_links, parent_nodes)
    global_boundary_roles = {
        "upstream": int(parent_report.source_nodes[0]),
        "downstream": int(parent_report.sink_nodes[0]),
    }
    cut_bindings = _boundary_bindings(
        local_links,
        local_nodes,
        cuts,
        contract,
        tolerance=boundary_tolerance,
        global_boundary_roles=global_boundary_roles,
    )
    node_match = _match_child_nodes_to_parent_nodes(
        parent_nodes, local_nodes, match_tolerance=tolerance, parent_node_order=parent_order
    )
    node_match = _force_boundary_node_matches(node_match, cut_bindings, parent_order)
    link_match = _match_child_links_to_parent_links(
        parent_links, local_links, match_tolerance=tolerance
    )
    if link_match.empty:
        link_match = pd.DataFrame(columns=["child_id_link", "candidate_class"])
    directed_local_links, directed_local_nodes, direction_summary = _assign_variant_directions(
        child_links=local_links,
        child_nodes=local_nodes,
        parent_links=parent_links,
        parent_nodes=parent_nodes,
        node_match=node_match,
        link_match=link_match,
    )
    directed_local_links, directed_local_nodes, bindings = _trim_to_anchored_boundaries(
        directed_local_links,
        directed_local_nodes,
        parent_nodes,
        contract,
        tolerance=boundary_tolerance,
    )
    node_match = _match_child_nodes_to_parent_nodes(
        parent_nodes,
        directed_local_nodes,
        match_tolerance=tolerance,
        parent_node_order=parent_order,
    )
    node_match = _force_boundary_node_matches(node_match, bindings, parent_order)
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
        if bool(row.match_within_tolerance) and not pd.isna(row.matched_parent_node_id):
            node_id_map[raw_id] = int(row.matched_parent_node_id)
        else:
            node_id_map[raw_id] = next_node
            next_node += 1
    node_match["final_node_id"] = node_match["child_id_node"].astype(int).map(node_id_map)
    node_match["final_node_method"] = np.where(
        node_match["match_method"].eq("forced_anchored_boundary"),
        "trim_to_anchored_inner_boundary",
        node_match["match_method"],
    )

    parent_node_by_id = parent_nodes.set_index(parent_nodes["id_node"].astype(int), drop=False)
    raw_node_by_id = directed_local_nodes.set_index(directed_local_nodes["id_node"].astype(int), drop=False)
    replacement_ids = set(int(value) for value in contract.anchored_replacement_link_ids)
    matches_by_child = (
        {int(key): value.copy() for key, value in link_match.groupby("child_id_link")}
        if not link_match.empty else {}
    )
    dominant_parent_count: dict[int, int] = {}
    for frame in matches_by_child.values():
        domain = frame.loc[frame["parent_id_link"].astype(int).isin(replacement_ids)]
        if not domain.empty:
            parent_id = int(domain.sort_values("candidate_rank").iloc[0]["parent_id_link"])
            dominant_parent_count[parent_id] = dominant_parent_count.get(parent_id, 0) + 1

    strict_parent_by_raw: dict[int, int] = {}
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
        if candidate is not None:
            parent_id = int(candidate["parent_id_link"])
            endpoints_match = (
                mapped_us == int(candidate["parent_id_us_node"])
                and mapped_ds == int(candidate["parent_id_ds_node"])
            )
            overlaps_match = (
                float(candidate["child_core_overlap_fraction"]) >= unchanged_overlap_fraction
                and float(candidate["parent_core_overlap_fraction"]) >= unchanged_overlap_fraction
            )
            unique = dominant_parent_count.get(parent_id, 0) == 1
            strict = endpoints_match and overlaps_match and unique
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
        })

    retained_context = set(strict_parent_by_raw.values())
    outside = parent_links.loc[
        ~parent_links["id_link"].astype(int).isin(replacement_ids - retained_context)
    ].copy()
    max_parent_link = int(parent_links["id_link"].astype(int).max())
    next_link = max_parent_link + 1
    inserted_rows: list[dict[str, Any]] = []
    final_link_by_raw: dict[int, int] = {}
    for row in directed_local_links.itertuples(index=False):
        raw_id = int(row.id_link)
        if raw_id in strict_parent_by_raw:
            final_link_by_raw[raw_id] = strict_parent_by_raw[raw_id]
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
        values.update({
            "id_link": final_id,
            "id_us_node": final_us,
            "id_ds_node": final_ds,
            "id_nodes": f"{final_us}, {final_ds}",
            parent_links.geometry.name: _snap_and_orient_geometry(
                row.geometry, raw_node_by_id.loc[raw_us].geometry, raw_node_by_id.loc[raw_ds].geometry,
                final_us_geom, final_ds_geom,
            ),
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
        if column != inserted.geometry.name and inserted[column].isna().all()
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
    source_id, sink_id = int(parent_report.source_nodes[0]), int(parent_report.sink_nodes[0])
    merged_links, merged_nodes = _refresh_topology(
        merged_links, merged_nodes, source_id=source_id, sink_id=sink_id
    )
    report = validate_single_inlet_single_outlet(build_directed_graph(merged_links, merged_nodes))
    if not report.is_valid:
        raise ValueError(f"Merged local-patch graph failed validation: {report.issues}")

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
        "selected_unit_ids": list(contract.selected_unit_ids),
        "replacement_parent_link_ids": sorted(replacement_ids),
        "retained_parent_link_count": int(len(outside)),
        "strict_context_link_count": int(len(retained_context)),
        "inserted_local_link_count": int(len(inserted)),
        "new_local_node_count": int(sum(value > max_parent_node for value in node_id_map.values())),
        "match_tolerance_m": tolerance,
        "cut_terminal_tolerance_m": boundary_tolerance,
        "raw_cut_terminal_bindings": cut_bindings.to_dict(orient="records"),
        "direction_summary": dict(direction_summary),
        "validation_report": report.to_dict(),
    }
    result = LocalPatchMergeResult("complete_conservative_merge", output_paths, summary)
    manifest_path.write_text(json.dumps(result.to_manifest(), indent=2, sort_keys=True) + "\n")
    return result


__all__ = [
    "LOCAL_PATCH_MERGE_VERSION",
    "LocalPatchMergeResult",
    "merge_local_patch_variant",
]
