"""Shared scientific anchoring for a locally assembled realized child.

Phase 3 produces a complete full-network realized graph at the outer
computational splice interfaces.  This module converts that graph to the
canonical matching/lineage contract and invokes the same graph-only
realized-to-anchored transformation used by conventional full-raster runs.
It intentionally does not calculate widths or publish a production variant.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Collection, Mapping, Sequence

import geopandas as gpd
import networkx as nx
import pandas as pd
from shapely.geometry import LineString, MultiLineString, Point
from shapely.ops import linemerge, substring

from hierarchy_level_definition.graph_building.directed_network_checks import (
    build_directed_graph,
    validate_single_inlet_single_outlet,
)
from hierarchy_level_definition.unit_detection.bifurcation_confluence_units import (
    StructuralUnit,
)
from network_variants.local_patch_realized import LocalPatchRealizedResult
from network_variants.local_patch_merge import _refresh_topology
from network_variants.variant_generation import (
    _infer_parent_node_order,
    _match_child_links_to_parent_links,
    _match_child_nodes_to_parent_nodes,
    _resolve_link_lineage,
    derive_anchored_graph_from_realized,
)


LOCAL_PATCH_ANCHORED_ADAPTER_VERSION = "local-realized-shared-anchoring-v6"
COMPRESSED_INTERFACE_MAX_TOLERANCE_MULTIPLIER = 5.0
PRE_ANCHOR_CANONICALIZATION_MIN_OVERLAP_FRACTION = 0.90


@dataclass(frozen=True, slots=True)
class LocalPatchAnchoredResult:
    """Persisted graph products from shared scientific anchoring."""

    status: str
    output_paths: Mapping[str, str]
    summary: Mapping[str, Any]
    adapter_version: str = LOCAL_PATCH_ANCHORED_ADAPTER_VERSION

    def to_manifest(self) -> dict[str, Any]:
        return {
            "adapter_version": self.adapter_version,
            "status": self.status,
            "output_paths": dict(self.output_paths),
            "summary": dict(self.summary),
        }


def _canonicalize_unchanged_parent_links(
    *,
    anchored_links: gpd.GeoDataFrame,
    anchored_nodes: gpd.GeoDataFrame,
    anchored_lineage: pd.DataFrame,
    parent_links: gpd.GeoDataFrame,
    parent_nodes: gpd.GeoDataFrame,
    source_id: int,
    sink_id: int,
    match_tolerance_m: float,
    canonicalizable_child_link_ids: Collection[int] | None = None,
    canonicalizable_parent_by_child_link_id: Mapping[int, int] | None = None,
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, dict[str, Any]]:
    """Use exact parent geometry without changing established child topology.

    Strict one-to-one links that were already proven unchanged before the
    anchoring transform inherit the complete parent line. Their child endpoint
    nodes are then placed on the corresponding exact parent endpoints. A
    post-segmentation one-to-one match is insufficient evidence because a
    changed collapsed link can resemble one surviving parent branch after it
    is cut at the scientific interfaces.

    ``canonicalizable_child_link_ids`` carries that authoritative pre-anchor
    provenance. Links outside the set retain their raw collapsed geometry,
    except for the separately bounded reversed short-interface repair below.
    ``None`` preserves the standalone helper's historical all-eligible
    behaviour for callers that do not supply provenance.
    """
    match_tolerance = float(match_tolerance_m)
    if match_tolerance <= 0:
        raise ValueError("match_tolerance_m must be positive.")
    compressed_interface_distance_limit = float(
        COMPRESSED_INTERFACE_MAX_TOLERANCE_MULTIPLIER * match_tolerance
    )
    links = anchored_links.copy()
    nodes = anchored_nodes.copy()
    parent_by_id = parent_links.set_index(
        parent_links["id_link"].astype(int), drop=False
    )
    parent_node_by_id = parent_nodes.set_index(
        parent_nodes["id_node"].astype(int), drop=False
    )
    node_by_id = nodes.set_index(nodes["id_node"].astype(int), drop=False)
    canonicalizable_ids = (
        None
        if canonicalizable_child_link_ids is None
        else {int(value) for value in canonicalizable_child_link_ids}
    )
    canonicalizable_parent = (
        None
        if canonicalizable_parent_by_child_link_id is None
        else {
            int(child_id): int(parent_id)
            for child_id, parent_id in canonicalizable_parent_by_child_link_id.items()
        }
    )
    records: list[dict[str, int]] = []
    provenance_excluded_records: list[dict[str, Any]] = []
    reversed_parent_connectors: list[dict[str, Any]] = []
    node_targets: dict[int, list[Point]] = {}
    lineage_by_id = anchored_lineage.set_index(
        anchored_lineage["id_link"].astype(int), drop=False
    )
    unchanged = anchored_lineage.loc[
        anchored_lineage["lineage_type"].astype(str).eq("unchanged_1to1")
        & anchored_lineage["dominant_parent_link_id"].notna()
    ]
    for lineage_row in unchanged.itertuples(index=False):
        child_id = int(lineage_row.id_link)
        parent_id = int(lineage_row.dominant_parent_link_id)
        if parent_id not in parent_by_id.index:
            raise ValueError(
                f"Unchanged lineage references absent parent link {parent_id}."
            )
        child_index = links.index[links["id_link"].astype(int).eq(child_id)]
        if len(child_index) != 1:
            raise ValueError(
                f"Expected one anchored child row for unchanged link {child_id}."
            )
        index = child_index[0]
        parent_row = parent_by_id.loc[parent_id]
        parent_us = int(parent_row["id_us_node"])
        parent_ds = int(parent_row["id_ds_node"])
        if parent_us not in parent_node_by_id.index or parent_ds not in parent_node_by_id.index:
            raise ValueError(
                f"Parent link {parent_id} references absent endpoint nodes."
            )
        parent_us_point = parent_node_by_id.loc[parent_us].geometry
        parent_ds_point = parent_node_by_id.loc[parent_ds].geometry
        old_child_us = int(links.at[index, "id_us_node"])
        old_child_ds = int(links.at[index, "id_ds_node"])
        old_us_point = node_by_id.loc[old_child_us].geometry
        old_ds_point = node_by_id.loc[old_child_ds].geometry
        direct_distance = old_us_point.distance(parent_us_point) + old_ds_point.distance(parent_ds_point)
        reverse_distance = old_us_point.distance(parent_ds_point) + old_ds_point.distance(parent_us_point)
        provenance_parent_id = (
            None
            if canonicalizable_parent is None
            else canonicalizable_parent.get(child_id)
        )
        eligible_by_source = canonicalizable_ids is None or child_id in canonicalizable_ids
        parent_identity_preserved = (
            canonicalizable_parent is None or provenance_parent_id == parent_id
        )
        general_canonicalization_allowed = bool(
            eligible_by_source and parent_identity_preserved
        )
        reversed_short_interface_candidate = reverse_distance < direct_distance
        if not general_canonicalization_allowed and not reversed_short_interface_candidate:
            provenance_excluded_records.append(
                {
                    "child_id_link": child_id,
                    "parent_id_link": parent_id,
                    "reason": (
                        "pre_anchor_parent_identity_changed"
                        if eligible_by_source and not parent_identity_preserved
                        else "not_genuinely_unchanged_before_anchoring"
                    ),
                    "pre_anchor_parent_id_link": provenance_parent_id,
                }
            )
            continue
        child_us, child_ds = (
            (old_child_us, old_child_ds)
            if direct_distance <= reverse_distance
            else (old_child_ds, old_child_us)
        )
        child_us_point_before = (
            old_us_point if child_us == old_child_us else old_ds_point
        )
        child_ds_point_before = (
            old_ds_point if child_ds == old_child_ds else old_us_point
        )
        parent_us_match_distance = float(
            child_us_point_before.distance(parent_us_point)
        )
        parent_ds_match_distance = float(
            child_ds_point_before.distance(parent_ds_point)
        )

        parent_geometry = parent_row.geometry
        if isinstance(parent_geometry, MultiLineString):
            parent_geometry = linemerge(parent_geometry)
        if not isinstance(parent_geometry, LineString):
            raise ValueError(
                f"Cannot canonicalize non-LineString parent link {parent_id}."
            )
        if not general_canonicalization_allowed:
            # The compressed-interface repair is an explicitly bounded
            # exception to pre-anchor provenance. A reversed-looking strict
            # match alone is not enough to activate it. If its complete
            # topology and distance contract is absent, preserve the changed
            # raw segment and let the final graph-validation gate decide.
            repair_rejection_reason: str | None = None
            endpoint_match_distances = (
                parent_us_match_distance,
                parent_ds_match_distance,
            )
            if any(distance > match_tolerance for distance in endpoint_match_distances):
                repair_rejection_reason = "unstable_parent_endpoint_matches"
            elif float(parent_geometry.length) > compressed_interface_distance_limit:
                repair_rejection_reason = "parent_connector_exceeds_distance_limit"
            else:
                companion_candidates: list[int] = []
                for candidate_row in links.itertuples(index=False):
                    candidate_id = int(candidate_row.id_link)
                    if candidate_id == child_id or candidate_id not in lineage_by_id.index:
                        continue
                    candidate_lineage = lineage_by_id.loc[candidate_id]
                    if str(candidate_lineage.get("lineage_type", "")) == "unchanged_1to1":
                        continue
                    dominant_parent = candidate_lineage.get("dominant_parent_link_id")
                    if pd.isna(dominant_parent) or int(dominant_parent) != parent_id:
                        continue
                    if child_us in (
                        int(candidate_row.id_us_node),
                        int(candidate_row.id_ds_node),
                    ):
                        companion_candidates.append(candidate_id)
                if len(companion_candidates) != 1:
                    repair_rejection_reason = (
                        "requires_exactly_one_non_strict_continuation;found="
                        + ",".join(map(str, sorted(companion_candidates)))
                    )
                else:
                    candidate_id = int(companion_candidates[0])
                    candidate = links.loc[
                        links["id_link"].astype(int).eq(candidate_id)
                    ].iloc[0]
                    candidate_geometry = candidate.geometry
                    if isinstance(candidate_geometry, MultiLineString):
                        candidate_geometry = linemerge(candidate_geometry)
                    if not isinstance(candidate_geometry, LineString):
                        repair_rejection_reason = "continuation_is_not_linestring"
                    else:
                        start = Point(candidate_geometry.coords[0])
                        end = Point(candidate_geometry.coords[-1])
                        upstream_at_start = (
                            start.distance(child_us_point_before)
                            <= end.distance(child_us_point_before)
                        )
                        measure = float(candidate_geometry.project(parent_ds_point))
                        projected_trim_distance = float(
                            measure
                            if upstream_at_start
                            else candidate_geometry.length - measure
                        )
                        target_offset = float(candidate_geometry.distance(parent_ds_point))
                        if projected_trim_distance > compressed_interface_distance_limit:
                            repair_rejection_reason = "continuation_trim_exceeds_distance_limit"
                        elif target_offset > compressed_interface_distance_limit:
                            repair_rejection_reason = "continuation_attachment_exceeds_distance_limit"
            if repair_rejection_reason is not None:
                provenance_excluded_records.append(
                    {
                        "child_id_link": child_id,
                        "parent_id_link": parent_id,
                        "reason": "bounded_interface_repair_not_applicable",
                        "repair_rejection_reason": repair_rejection_reason,
                    }
                )
                continue
        parent_coordinates = list(parent_geometry.coords)
        if Point(parent_coordinates[0]).distance(parent_us_point) > Point(parent_coordinates[-1]).distance(parent_us_point):
            parent_coordinates.reverse()
        links.at[index, links.geometry.name] = LineString(parent_coordinates)
        links.at[index, "id_us_node"] = int(child_us)
        links.at[index, "id_ds_node"] = int(child_ds)
        if "id_nodes" in links.columns:
            links.at[index, "id_nodes"] = f"{int(child_us)}, {int(child_ds)}"
        if "geometry_reversed_to_match_flow" in links.columns:
            links.at[index, "geometry_reversed_to_match_flow"] = False
        node_targets.setdefault(child_us, []).append(parent_us_point)
        node_targets.setdefault(child_ds, []).append(parent_ds_point)
        if (child_us, child_ds) != (old_child_us, old_child_ds):
            reversed_parent_connectors.append(
                {
                    "child_id_link": child_id,
                    "parent_id_link": parent_id,
                    "parent_us_node": parent_us,
                    "parent_ds_node": parent_ds,
                    "child_us_node": child_us,
                    "child_ds_node": child_ds,
                    "parent_us_match_distance_m": parent_us_match_distance,
                    "parent_ds_match_distance_m": parent_ds_match_distance,
                }
            )
        records.append(
            {
                "child_id_link": child_id,
                "parent_id_link": parent_id,
                "canonicalization_basis": (
                    "pre_anchor_unchanged_context"
                    if general_canonicalization_allowed
                    else "bounded_reversed_short_interface_candidate"
                ),
            }
        )

    moved_nodes: dict[int, tuple[Point, Point]] = {}
    for child_node_id, targets in node_targets.items():
        target = targets[0]
        if any(candidate.distance(target) > 1e-7 for candidate in targets[1:]):
            raise ValueError(
                "Unchanged parent links propose conflicting exact locations for "
                f"child node {child_node_id}."
            )
        node_index = nodes.index[nodes["id_node"].astype(int).eq(child_node_id)]
        if len(node_index) != 1:
            raise ValueError(f"Expected one anchored child node {child_node_id}.")
        old_point = nodes.at[node_index[0], nodes.geometry.name]
        if old_point.distance(target) > 1e-7:
            moved_nodes[child_node_id] = (old_point, target)
            nodes.at[node_index[0], nodes.geometry.name] = target

    for index, row in links.iterrows():
        upstream = int(row["id_us_node"])
        downstream = int(row["id_ds_node"])
        if upstream not in moved_nodes and downstream not in moved_nodes:
            continue
        geometry = row.geometry
        if isinstance(geometry, MultiLineString):
            geometry = linemerge(geometry)
        if not isinstance(geometry, LineString):
            raise ValueError(
                f"Cannot align endpoints of non-LineString child link {int(row['id_link'])}."
            )
        coordinates = list(geometry.coords)
        start = Point(coordinates[0])
        end = Point(coordinates[-1])
        for node_id in (upstream, downstream):
            if node_id not in moved_nodes:
                continue
            old_point, target = moved_nodes[node_id]
            if start.distance(old_point) <= end.distance(old_point):
                coordinates[0] = (float(target.x), float(target.y))
                start = target
            else:
                coordinates[-1] = (float(target.x), float(target.y))
                end = target
        links.at[index, links.geometry.name] = LineString(coordinates)

    # RivGraph can compress two nearby scientific interfaces into one local
    # junction. In that case the shared anchoring transform may represent the
    # short parent connector twice: once as an exact one-to-one connector and
    # once as the beginning of the collapsed continuation. When the exact
    # connector was reversed above, move the unique non-strict companion from
    # the connector's upstream interface to its downstream interface and trim
    # away the duplicated connector portion. This preserves both exact parent
    # direction and the selected-unit continuation.
    companion_records: list[dict[str, Any]] = []
    for correction in reversed_parent_connectors:
        strict_child_id = int(correction["child_id_link"])
        parent_id = int(correction["parent_id_link"])
        child_us = int(correction["child_us_node"])
        child_ds = int(correction["child_ds_node"])
        endpoint_match_distances = (
            float(correction["parent_us_match_distance_m"]),
            float(correction["parent_ds_match_distance_m"]),
        )
        if any(distance > match_tolerance for distance in endpoint_match_distances):
            raise ValueError(
                "A reversed strict parent connector does not have stable endpoint "
                "matches: parent link "
                f"{parent_id} has distances {endpoint_match_distances} m, exceeding "
                f"the {match_tolerance:.6f} m match tolerance."
            )
        parent_geometry = parent_by_id.loc[parent_id].geometry
        if isinstance(parent_geometry, MultiLineString):
            parent_geometry = linemerge(parent_geometry)
        if not isinstance(parent_geometry, LineString):
            raise ValueError(
                f"Cannot measure non-LineString parent connector {parent_id}."
            )
        parent_connector_length = float(parent_geometry.length)
        if parent_connector_length > compressed_interface_distance_limit:
            raise ValueError(
                "A reversed strict parent connector exceeds the bounded compressed-"
                "interface correction distance: parent link "
                f"{parent_id} is {parent_connector_length:.6f} m, exceeding "
                f"{compressed_interface_distance_limit:.6f} m "
                f"({COMPRESSED_INTERFACE_MAX_TOLERANCE_MULTIPLIER:g} matching tolerances)."
            )
        candidates: list[int] = []
        for row in links.itertuples(index=False):
            candidate_id = int(row.id_link)
            if candidate_id == strict_child_id or candidate_id not in lineage_by_id.index:
                continue
            lineage_row = lineage_by_id.loc[candidate_id]
            if str(lineage_row.get("lineage_type", "")) == "unchanged_1to1":
                continue
            dominant_parent = lineage_row.get("dominant_parent_link_id")
            if pd.isna(dominant_parent) or int(dominant_parent) != parent_id:
                continue
            if child_us in (int(row.id_us_node), int(row.id_ds_node)):
                candidates.append(candidate_id)
        if len(candidates) != 1:
            raise ValueError(
                "A reversed strict parent connector requires exactly one "
                "non-strict collapsed continuation at its upstream interface; "
                f"parent link {parent_id} found candidates {sorted(candidates)}."
            )
        candidate_id = int(candidates[0])
        candidate_index = links.index[links["id_link"].astype(int).eq(candidate_id)][0]
        candidate = links.loc[candidate_index].copy()
        geometry = candidate.geometry
        if isinstance(geometry, MultiLineString):
            geometry = linemerge(geometry)
        if not isinstance(geometry, LineString):
            raise ValueError(
                f"Cannot trim non-LineString collapsed continuation {candidate_id}."
            )
        child_us_point = nodes.loc[
            nodes["id_node"].astype(int).eq(child_us), nodes.geometry.name
        ].iloc[0]
        child_ds_point = nodes.loc[
            nodes["id_node"].astype(int).eq(child_ds), nodes.geometry.name
        ].iloc[0]
        start = Point(geometry.coords[0])
        end = Point(geometry.coords[-1])
        upstream_at_start = start.distance(child_us_point) <= end.distance(child_us_point)
        measure = float(geometry.project(child_ds_point))
        projected_trim_distance = float(
            measure if upstream_at_start else geometry.length - measure
        )
        target_offset = float(geometry.distance(child_ds_point))
        bounded_distances = {
            "parent_connector_length_m": parent_connector_length,
            "projected_trim_distance_m": projected_trim_distance,
            "target_offset_m": target_offset,
        }
        excessive_distances = {
            name: distance
            for name, distance in bounded_distances.items()
            if distance > compressed_interface_distance_limit
        }
        if excessive_distances:
            raise ValueError(
                "A reversed strict parent connector requires an unbounded compressed-"
                "interface correction: parent link "
                f"{parent_id} has {excessive_distances}, exceeding "
                f"{compressed_interface_distance_limit:.6f} m "
                f"({COMPRESSED_INTERFACE_MAX_TOLERANCE_MULTIPLIER:g} matching tolerances)."
            )
        if upstream_at_start:
            trimmed = substring(geometry, measure, float(geometry.length))
        else:
            trimmed = substring(geometry, 0.0, measure)
        if not isinstance(trimmed, LineString) or trimmed.length <= 0:
            raise ValueError(
                f"Collapsed continuation {candidate_id} cannot be trimmed at parent connector {parent_id}."
            )
        coordinates = list(trimmed.coords)
        if upstream_at_start:
            coordinates[0] = (float(child_ds_point.x), float(child_ds_point.y))
        else:
            coordinates[-1] = (float(child_ds_point.x), float(child_ds_point.y))
        candidate[links.geometry.name] = LineString(coordinates)
        if int(candidate["id_us_node"]) == child_us:
            candidate["id_us_node"] = child_ds
        elif int(candidate["id_ds_node"]) == child_us:
            candidate["id_ds_node"] = child_ds
        else:
            raise ValueError(
                f"Collapsed continuation {candidate_id} lost connector endpoint {child_us}."
            )
        if "id_nodes" in links.columns:
            candidate["id_nodes"] = (
                f"{int(candidate['id_us_node'])}, {int(candidate['id_ds_node'])}"
            )
        for field in ("idx_link", "wid_pix"):
            if field in links.columns:
                candidate[field] = None
        for field in ("len", "len_adj"):
            if field in links.columns:
                candidate[field] = float(candidate.geometry.length)
        links.loc[candidate_index] = candidate
        companion_records.append(
            {
                **correction,
                "continuation_child_id_link": candidate_id,
                "parent_connector_length_m": parent_connector_length,
                "distance_limit_m": compressed_interface_distance_limit,
                "distance_limit_tolerance_multiplier": float(
                    COMPRESSED_INTERFACE_MAX_TOLERANCE_MULTIPLIER
                ),
                "projected_trim_distance_m": projected_trim_distance,
                "target_offset_m": target_offset,
                "geometry_length_change_m": float(
                    candidate.geometry.length - geometry.length
                ),
            }
        )
    links, nodes = _refresh_topology(
        links,
        nodes,
        source_id=int(source_id),
        sink_id=int(sink_id),
    )
    graph = build_directed_graph(links, nodes, prefer_explicit_direction=True)
    report = validate_single_inlet_single_outlet(graph)
    is_dag = nx.is_directed_acyclic_graph(graph)
    if not report.is_valid or not is_dag:
        issues = list(report.issues)
        if not is_dag:
            issues.append("Canonicalized anchored graph is not acyclic.")
        raise ValueError(
            "Exact parent canonicalization invalidated the anchored graph: "
            f"{issues}"
        )
    return links, nodes, {
        "method": "pre_anchor_provenance_scoped_parent_geometry_with_bounded_interface_repair",
        "canonicalized_link_count": int(len(records)),
        "pre_anchor_eligible_link_count": (
            None if canonicalizable_ids is None else int(len(canonicalizable_ids))
        ),
        "post_anchor_matches_excluded_by_provenance_count": int(
            len(provenance_excluded_records)
        ),
        "post_anchor_matches_excluded_by_provenance": provenance_excluded_records,
        "renamed_link_count": 0,
        "canonicalized_node_count": int(len(moved_nodes)),
        "maximum_node_move_m": float(
            max((old.distance(new) for old, new in moved_nodes.values()), default=0.0)
        ),
        "parent_direction_correction_count": int(len(reversed_parent_connectors)),
        "compressed_interface_continuation_corrections": companion_records,
        "compressed_interface_distance_limit_m": compressed_interface_distance_limit,
        "compressed_interface_distance_limit_tolerance_multiplier": float(
            COMPRESSED_INTERFACE_MAX_TOLERANCE_MULTIPLIER
        ),
        "records": records,
        "validation_report": {**report.to_dict(), "is_dag": bool(is_dag)},
    }


def anchor_local_patch_realized_graph(
    *,
    realized_result: LocalPatchRealizedResult,
    parent_links_path: str | Path,
    parent_nodes_path: str | Path,
    selected_unit_ids: Sequence[int],
    units_by_id: Mapping[int, StructuralUnit],
    output_dir: str | Path,
    match_tolerance_m: float,
) -> LocalPatchAnchoredResult:
    """Apply the canonical realized-to-anchored graph transform.

    Matching and lineage are deliberately recomputed from the assembled
    full-network realized graph.  The Phase-3 partial-overlap classifications
    remain a splice audit and are not substituted for canonical lineage.
    """
    if realized_result.status != "complete_local_realized_graph":
        raise ValueError(
            "Shared local anchoring requires a complete local realized graph."
        )
    tolerance = float(match_tolerance_m)
    if tolerance <= 0:
        raise ValueError("match_tolerance_m must be positive.")
    selected_ids = tuple(dict.fromkeys(int(value) for value in selected_unit_ids))
    if not selected_ids:
        raise ValueError("Shared local anchoring requires selected unit IDs.")

    parent_links = gpd.read_file(parent_links_path)
    parent_nodes = gpd.read_file(parent_nodes_path)
    realized_links = gpd.read_file(realized_result.output_paths["realized_links"])
    realized_nodes = gpd.read_file(realized_result.output_paths["realized_nodes"])
    if parent_links.crs is None or parent_nodes.crs is None or realized_links.crs is None:
        raise ValueError("Parent and realized graph products require a CRS.")
    realized_links = realized_links.to_crs(parent_links.crs)
    realized_nodes = realized_nodes.to_crs(parent_nodes.crs)

    parent_order, parent_graph, parent_report = _infer_parent_node_order(
        parent_links,
        parent_nodes,
    )
    realized_node_match = _match_child_nodes_to_parent_nodes(
        parent_nodes,
        realized_nodes,
        match_tolerance=tolerance,
        parent_node_order=parent_order,
    )
    realized_link_match = _match_child_links_to_parent_links(
        parent_links,
        realized_links,
        match_tolerance=tolerance,
    )
    realized_link_lineage = _resolve_link_lineage(
        parent_graph=parent_graph,
        directed_child_links=realized_links,
        node_match=realized_node_match,
        link_match=realized_link_match,
        match_tolerance=tolerance,
    )

    # Carry authoritative pre-anchor provenance through graph segmentation.
    # Segmentation copies source row fields to every resulting piece, allowing
    # context links to remain exact parent products without mistaking a changed
    # collapsed segment for an unchanged link merely because it resembles one
    # parent branch after the cut.
    selected_parent_link_ids = {
        int(edge_key)
        for unit_id in selected_ids
        for _upstream, _downstream, edge_key in units_by_id[int(unit_id)].edge_set
    }
    realized_lineage_by_id = realized_link_lineage.set_index(
        realized_link_lineage["id_link"].astype(int), drop=False
    )
    realized_links = realized_links.copy()
    realized_links["local_patch_source_realized_link_id"] = realized_links[
        "id_link"
    ].astype(int)
    pre_anchor_parent_ids: list[object] = []
    pre_anchor_overlap_fractions: list[float] = []
    pre_anchor_canonicalizable: list[bool] = []
    for link_id in realized_links["id_link"].astype(int):
        if link_id not in realized_lineage_by_id.index:
            pre_anchor_parent_ids.append(pd.NA)
            pre_anchor_overlap_fractions.append(float("nan"))
            pre_anchor_canonicalizable.append(False)
            continue
        lineage_row = realized_lineage_by_id.loc[link_id]
        parent_value = lineage_row.get("dominant_parent_link_id", pd.NA)
        parent_id = None if pd.isna(parent_value) else int(parent_value)
        overlap_value = pd.to_numeric(
            pd.Series([lineage_row.get("dominant_parent_overlap_fraction")]),
            errors="coerce",
        ).iloc[0]
        overlap_fraction = (
            float(overlap_value) if pd.notna(overlap_value) else float("nan")
        )
        pre_anchor_parent_ids.append(pd.NA if parent_id is None else parent_id)
        pre_anchor_overlap_fractions.append(overlap_fraction)
        pre_anchor_canonicalizable.append(
            bool(
                str(lineage_row.get("lineage_type", "")) == "unchanged_1to1"
                and parent_id is not None
                and parent_id not in selected_parent_link_ids
                and overlap_fraction
                >= PRE_ANCHOR_CANONICALIZATION_MIN_OVERLAP_FRACTION
            )
        )
    realized_links["local_patch_pre_anchor_parent_link_id"] = pd.array(
        pre_anchor_parent_ids, dtype="Int64"
    )
    realized_links["local_patch_pre_anchor_overlap_fraction"] = (
        pre_anchor_overlap_fractions
    )
    realized_links["local_patch_pre_anchor_canonicalizable"] = (
        pre_anchor_canonicalizable
    )

    transformed = derive_anchored_graph_from_realized(
        parent_links=parent_links,
        parent_nodes=parent_nodes,
        realized_links=realized_links,
        realized_nodes=realized_nodes,
        realized_node_match=realized_node_match,
        realized_link_match=realized_link_match,
        realized_link_lineage=realized_link_lineage,
        selected_unit_ids=selected_ids,
        units_by_id=units_by_id,
        match_tolerance=tolerance,
    )
    canonicalizable_child_link_ids = set(
        transformed.directed_links.loc[
            transformed.directed_links[
                "local_patch_pre_anchor_canonicalizable"
            ].fillna(False).astype(bool),
            "id_link",
        ].astype(int)
    )
    canonicalizable_parent_by_child_link_id = {
        int(row.id_link): int(row.local_patch_pre_anchor_parent_link_id)
        for row in transformed.directed_links.itertuples(index=False)
        if bool(getattr(row, "local_patch_pre_anchor_canonicalizable", False))
        and pd.notna(getattr(row, "local_patch_pre_anchor_parent_link_id", pd.NA))
    }
    direction_summary = dict(transformed.direction_summary)
    if not bool(direction_summary.get("is_valid", False)):
        # When the computational-domain boundary is already the scientific
        # boundary, shared segmentation is intentionally a no-op and may not
        # emit its own validation payload. Validate that unchanged graph
        # directly; non-empty transformation issues still remain fatal.
        no_op_graph = build_directed_graph(
            transformed.directed_links,
            transformed.directed_nodes,
            prefer_explicit_direction=True,
        )
        no_op_report = validate_single_inlet_single_outlet(no_op_graph)
        no_op_is_dag = nx.is_directed_acyclic_graph(no_op_graph)
        if (
            direction_summary.get("issues")
            or not no_op_report.is_valid
            or not no_op_is_dag
        ):
            raise ValueError(
                "Shared realized-to-anchored transformation produced an invalid graph: "
                f"{direction_summary.get('issues', [])}."
            )
        direction_summary = {
            **direction_summary,
            "is_valid": True,
            "source_node": int(no_op_report.source_nodes[0]),
            "sink_node": int(no_op_report.sink_nodes[0]),
            "source_nodes": list(no_op_report.source_nodes),
            "sink_nodes": list(no_op_report.sink_nodes),
            "issues": [],
            "validation_report": {
                **no_op_report.to_dict(),
                "is_dag": True,
            },
            "validation_method": "direct_validation_after_no_op_segmentation",
        }
    anchored_links, anchored_nodes, canonicalization_summary = (
        _canonicalize_unchanged_parent_links(
            anchored_links=transformed.directed_links,
            anchored_nodes=transformed.directed_nodes,
            anchored_lineage=transformed.link_lineage,
            parent_links=parent_links,
            parent_nodes=parent_nodes,
            source_id=int(parent_report.source_nodes[0]),
            sink_id=int(parent_report.sink_nodes[0]),
            match_tolerance_m=tolerance,
            canonicalizable_child_link_ids=canonicalizable_child_link_ids,
            canonicalizable_parent_by_child_link_id=(
                canonicalizable_parent_by_child_link_id
            ),
        )
    )
    internal_provenance_columns = [
        "local_patch_source_realized_link_id",
        "local_patch_pre_anchor_parent_link_id",
        "local_patch_pre_anchor_overlap_fraction",
        "local_patch_pre_anchor_canonicalizable",
    ]
    anchored_links = anchored_links.drop(
        columns=[
            column for column in internal_provenance_columns if column in anchored_links.columns
        ],
        errors="ignore",
    )
    anchored_node_match = _match_child_nodes_to_parent_nodes(
        parent_nodes,
        anchored_nodes,
        match_tolerance=tolerance,
        parent_node_order=parent_order,
    )
    anchored_link_match = _match_child_links_to_parent_links(
        parent_links,
        anchored_links,
        match_tolerance=tolerance,
    )
    anchored_lineage = _resolve_link_lineage(
        parent_graph=parent_graph,
        directed_child_links=anchored_links,
        node_match=anchored_node_match,
        link_match=anchored_link_match,
        match_tolerance=tolerance,
    )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    links_path = output_dir / "anchored_links.gpkg"
    nodes_path = output_dir / "anchored_nodes.gpkg"
    realized_node_match_path = output_dir / "realized_node_match.csv"
    realized_link_match_path = output_dir / "realized_link_match.csv"
    realized_lineage_path = output_dir / "realized_link_lineage.csv"
    anchored_node_match_path = output_dir / "anchored_node_match.csv"
    anchored_link_match_path = output_dir / "anchored_link_match.csv"
    anchored_lineage_path = output_dir / "anchored_link_lineage.csv"
    manifest_path = output_dir / "local_shared_anchoring_manifest.json"

    anchored_links.to_file(links_path, driver="GPKG")
    anchored_nodes.to_file(nodes_path, driver="GPKG")
    realized_node_match.to_csv(realized_node_match_path, index=False)
    realized_link_match.to_csv(realized_link_match_path, index=False)
    realized_link_lineage.to_csv(realized_lineage_path, index=False)
    anchored_node_match.to_csv(anchored_node_match_path, index=False)
    anchored_link_match.to_csv(anchored_link_match_path, index=False)
    anchored_lineage.to_csv(anchored_lineage_path, index=False)

    output_paths = {
        "anchored_links": str(links_path.resolve()),
        "anchored_nodes": str(nodes_path.resolve()),
        "realized_node_match": str(realized_node_match_path.resolve()),
        "realized_link_match": str(realized_link_match_path.resolve()),
        "realized_link_lineage": str(realized_lineage_path.resolve()),
        "anchored_node_match": str(anchored_node_match_path.resolve()),
        "anchored_link_match": str(anchored_link_match_path.resolve()),
        "anchored_link_lineage": str(anchored_lineage_path.resolve()),
        "manifest": str(manifest_path.resolve()),
    }
    summary = {
        "selected_unit_ids": list(selected_ids),
        "source_realized_manifest": realized_result.output_paths.get("manifest"),
        "match_tolerance_m": tolerance,
        "scientific_anchoring_applied": True,
        "anchoring_implementation": "derive_anchored_graph_from_realized",
        "realized_link_count": int(len(realized_links)),
        "anchored_link_count": int(len(anchored_links)),
        "realized_node_count": int(len(realized_nodes)),
        "anchored_node_count": int(len(anchored_nodes)),
        "unchanged_parent_canonicalization": canonicalization_summary,
        "anchor_segmentation_summary": transformed.anchor_segmentation_summary,
        "direction_summary": direction_summary,
    }
    result = LocalPatchAnchoredResult(
        status="complete_shared_anchored_graph",
        output_paths=output_paths,
        summary=summary,
    )
    manifest_path.write_text(
        json.dumps(result.to_manifest(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result


__all__ = [
    "LOCAL_PATCH_ANCHORED_ADAPTER_VERSION",
    "LocalPatchAnchoredResult",
    "anchor_local_patch_realized_graph",
]
