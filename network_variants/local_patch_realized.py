"""Assemble a full-network realized child from a cropped RivGraph result.

The computational crop and its bank-to-bank cuts are implementation details.
This module restores the unchanged parent graph outside the outer splice
interfaces and inserts the locally regenerated graph inside that interval.  It
does *not* apply scientific anchor segmentation; the shared
``derive_anchored_graph_from_realized`` transformation owns that later step.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping

import geopandas as gpd
import networkx as nx
import pandas as pd
import rasterio
from shapely.geometry import LineString, MultiLineString, Point
from shapely.ops import linemerge, substring

from network_variants.local_patch import LocalPatchContract
from network_variants.local_patch_merge import (
    _refresh_topology,
    merge_local_patch_variant,
)
from network_variants.local_patch_raster import LocalPatchRasterResult
from network_variants.local_patch_variant import LocalPatchVariantResult
from network_variants.variant_generation import (
    _assign_variant_directions,
    _infer_parent_node_order,
    _match_child_links_to_parent_links,
    _match_child_nodes_to_parent_nodes,
)
from hierarchy_level_definition.graph_building.directed_network_checks import (
    build_directed_graph,
    validate_single_inlet_single_outlet,
)


LOCAL_PATCH_REALIZED_ASSEMBLY_VERSION = "topology-local-realized-assembly-v4"


@dataclass(frozen=True, slots=True)
class LocalPatchRealizedResult:
    """Persisted full-network realized graph and its splice/lineage audit."""

    status: str
    output_paths: Mapping[str, str]
    summary: Mapping[str, Any]
    assembly_version: str = LOCAL_PATCH_REALIZED_ASSEMBLY_VERSION

    def to_manifest(self) -> dict[str, Any]:
        return {
            "assembly_version": self.assembly_version,
            "status": self.status,
            "output_paths": dict(self.output_paths),
            "summary": dict(self.summary),
        }


def _line(geometry) -> LineString:
    if isinstance(geometry, LineString):
        return geometry
    if isinstance(geometry, MultiLineString):
        merged = linemerge(geometry)
        if isinstance(merged, LineString):
            return merged
    raise ValueError("Computational splice requires continuous LineString links.")


def _cut_center(cut_geometry) -> Point:
    line = _line(cut_geometry)
    coordinates = list(line.coords)
    if len(coordinates) >= 3:
        coordinate = coordinates[len(coordinates) // 2]
        return Point(float(coordinate[0]), float(coordinate[1]))
    return line.interpolate(float(line.length) * 0.5)


def _prepare_computational_splice_parent(
    *,
    parent_links: gpd.GeoDataFrame,
    parent_nodes: gpd.GeoDataFrame,
    cuts: gpd.GeoDataFrame,
    contract: LocalPatchContract,
    boundary_hint_points: Mapping[str, Point] | None = None,
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, tuple[int, ...], tuple[int, ...], tuple[int, ...], dict[str, Any]]:
    """Insert exact parent nodes at the two raster-cut centreline crossings."""
    links = parent_links.copy()
    nodes = parent_nodes.copy()
    next_node_id = int(nodes["id_node"].astype(int).max()) + 1
    next_link_id = int(links["id_link"].astype(int).max()) + 1
    split_original_ids: set[int] = set()
    boundary_ids: dict[str, int] = {}
    split_audit: list[dict[str, Any]] = []

    for role, inner_boundary_id in (
        ("upstream", int(contract.upstream_collar.domain_boundary_node_id)),
        ("downstream", int(contract.downstream_collar.domain_boundary_node_id)),
    ):
        role_cuts = cuts.loc[cuts["role"].astype(str).eq(role)]
        if len(role_cuts) != 1:
            raise ValueError(f"Expected exactly one {role} computational cut.")
        cut_center = _cut_center(role_cuts.iloc[0].geometry)
        target_point = (boundary_hint_points or {}).get(role, cut_center)
        target_point = Point(float(target_point.x), float(target_point.y))

        graph = nx.MultiDiGraph()
        for row in links.itertuples(index=False):
            graph.add_edge(
                int(row.id_us_node), int(row.id_ds_node), key=int(row.id_link)
            )
        if inner_boundary_id not in graph:
            raise ValueError(
                f"The {role} inner boundary node {inner_boundary_id} is absent "
                "from the parent graph."
            )
        if role == "upstream":
            eligible_nodes = {inner_boundary_id}.union(
                int(value) for value in nx.ancestors(graph, inner_boundary_id)
            )
        else:
            eligible_nodes = {inner_boundary_id}.union(
                int(value) for value in nx.descendants(graph, inner_boundary_id)
            )
        eligible = links.loc[
            links["id_us_node"].astype(int).isin(eligible_nodes)
            & links["id_ds_node"].astype(int).isin(eligible_nodes)
        ].copy()
        if eligible.empty:
            raise ValueError(f"No parent links lie on the {role} side of the domain.")
        distances = eligible.geometry.distance(target_point)
        link_index = distances.idxmin()
        link_distance = float(distances.loc[link_index])
        row = links.loc[link_index].copy()
        original_link_id = int(row["id_link"])
        line = _line(row.geometry)
        measure = float(line.project(target_point))
        projected = line.interpolate(measure)

        node_index = nodes.set_index(nodes["id_node"].astype(int), drop=False)
        upstream_id = int(row["id_us_node"])
        downstream_id = int(row["id_ds_node"])
        start = Point(line.coords[0])
        if start.distance(node_index.loc[upstream_id].geometry) > start.distance(
            node_index.loc[downstream_id].geometry
        ):
            line = LineString(list(line.coords)[::-1])
            measure = float(line.project(projected))
        endpoint_tolerance = max(float(line.length) * 1e-9, 1e-7)
        if measure <= endpoint_tolerance:
            boundary_node_id = upstream_id
            method = "existing_parent_upstream_endpoint"
        elif measure >= float(line.length) - endpoint_tolerance:
            boundary_node_id = downstream_id
            method = "existing_parent_downstream_endpoint"
        else:
            boundary_node_id = next_node_id
            next_node_id += 1
            node_values = {column: None for column in nodes.columns}
            node_values.update(
                {"id_node": boundary_node_id, nodes.geometry.name: projected}
            )
            nodes = gpd.GeoDataFrame(
                pd.concat(
                    [
                        nodes,
                        gpd.GeoDataFrame(
                            [node_values], geometry=nodes.geometry.name, crs=nodes.crs
                        ),
                    ],
                    ignore_index=True,
                ),
                geometry=nodes.geometry.name,
                crs=nodes.crs,
            )
            upstream_geometry = substring(line, 0.0, measure)
            downstream_geometry = substring(line, measure, float(line.length))
            outer_is_upstream = role == "upstream"
            outer_geometry = upstream_geometry if outer_is_upstream else downstream_geometry
            inner_geometry = downstream_geometry if outer_is_upstream else upstream_geometry
            outer_us = upstream_id if outer_is_upstream else boundary_node_id
            outer_ds = boundary_node_id if outer_is_upstream else downstream_id
            inner_us = boundary_node_id if outer_is_upstream else upstream_id
            inner_ds = downstream_id if outer_is_upstream else boundary_node_id

            outer = row.copy()
            outer["id_us_node"], outer["id_ds_node"] = outer_us, outer_ds
            outer["id_nodes"] = f"{outer_us}, {outer_ds}"
            outer[links.geometry.name] = outer_geometry
            inner = row.copy()
            inner["id_link"] = next_link_id
            next_link_id += 1
            inner["id_us_node"], inner["id_ds_node"] = inner_us, inner_ds
            inner["id_nodes"] = f"{inner_us}, {inner_ds}"
            inner[links.geometry.name] = inner_geometry
            for candidate in (outer, inner):
                for field in ("idx_link", "wid_pix"):
                    if field in links.columns:
                        candidate[field] = None
                for field in ("len", "len_adj"):
                    if field in links.columns:
                        candidate[field] = float(candidate[links.geometry.name].length)
            links.loc[link_index] = outer
            links = gpd.GeoDataFrame(
                pd.concat(
                    [
                        links,
                        gpd.GeoDataFrame(
                            [inner], geometry=links.geometry.name, crs=links.crs
                        ),
                    ],
                    ignore_index=True,
                ),
                geometry=links.geometry.name,
                crs=links.crs,
            )
            split_original_ids.add(original_link_id)
            method = "split_parent_link_at_computational_cut"
        boundary_ids[role] = int(boundary_node_id)
        split_audit.append(
            {
                "role": role,
                "original_parent_link_id": original_link_id,
                "computational_boundary_node_id": int(boundary_node_id),
                "cut_to_parent_link_distance_m": link_distance,
                "split_method": method,
                "projected_x": float(projected.x),
                "projected_y": float(projected.y),
                "raw_terminal_hint_x": float(target_point.x),
                "raw_terminal_hint_y": float(target_point.y),
                "cut_center_x": float(cut_center.x),
                "cut_center_y": float(cut_center.y),
            }
        )

    graph = nx.MultiDiGraph()
    for row in links.itertuples(index=False):
        graph.add_edge(
            int(row.id_us_node), int(row.id_ds_node), key=int(row.id_link)
        )
    source_id = int(boundary_ids["upstream"])
    sink_id = int(boundary_ids["downstream"])
    if not nx.has_path(graph, source_id, sink_id):
        raise ValueError(
            "The computational cut boundaries are not connected in parent flow direction."
        )
    forward = {source_id}.union(int(value) for value in nx.descendants(graph, source_id))
    reverse_graph = graph.reverse(copy=False)
    reverse = {sink_id}.union(int(value) for value in nx.descendants(reverse_graph, sink_id))
    interval_nodes = forward.intersection(reverse)
    replacement_ids = tuple(
        sorted(
            int(key)
            for node_a, node_b, key in graph.edges(keys=True)
            if int(node_a) in interval_nodes and int(node_b) in interval_nodes
        )
    )
    if not set(contract.core_link_ids).issubset(replacement_ids):
        raise ValueError(
            "The computational splice interval does not contain every planned core link."
        )
    return (
        links.reset_index(drop=True),
        nodes.reset_index(drop=True),
        replacement_ids,
        (source_id,),
        (sink_id,),
        {
            "method": "split_parent_at_actual_cut_then_directed_interval",
            "cut_splits": split_audit,
            "split_original_parent_link_ids": sorted(split_original_ids),
            "replacement_parent_link_ids": list(replacement_ids),
        },
    )


def _directed_line(geometry, upstream_point: Point, downstream_point: Point) -> LineString:
    line = _line(geometry)
    coordinates = list(line.coords)
    forward = Point(coordinates[0]).distance(upstream_point) + Point(
        coordinates[-1]
    ).distance(downstream_point)
    reverse = Point(coordinates[-1]).distance(upstream_point) + Point(
        coordinates[0]
    ).distance(downstream_point)
    if reverse < forward:
        coordinates.reverse()
    return LineString(coordinates)


def _dissolve_computational_cut_interfaces(
    *,
    links: gpd.GeoDataFrame,
    nodes: gpd.GeoDataFrame,
    splice_summary: Mapping[str, Any],
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, dict[str, Any]]:
    """Remove temporary degree-two cut nodes from the realized topology."""
    result_links = links.copy()
    result_nodes = nodes.copy()
    audit: list[dict[str, Any]] = []
    for split in splice_summary["cut_splits"]:
        if split["split_method"] != "split_parent_link_at_computational_cut":
            continue
        role = str(split["role"])
        boundary_id = int(split["computational_boundary_node_id"])
        original_stub_id = int(split["original_parent_link_id"])
        incoming = result_links.loc[
            result_links["id_ds_node"].astype(int).eq(boundary_id)
        ]
        outgoing = result_links.loc[
            result_links["id_us_node"].astype(int).eq(boundary_id)
        ]
        if len(incoming) != 1 or len(outgoing) != 1:
            raise ValueError(
                f"Computational {role} boundary node {boundary_id} is not a "
                "simple directed degree-two interface."
            )
        first = incoming.iloc[0].copy()
        second = outgoing.iloc[0].copy()
        incident_ids = {int(first["id_link"]), int(second["id_link"])}
        if original_stub_id not in incident_ids:
            raise ValueError(
                f"Computational {role} interface does not contain expected parent "
                f"stub link {original_stub_id}."
            )
        local = second.copy() if int(first["id_link"]) == original_stub_id else first.copy()
        local_id = int(local["id_link"])
        node_index = result_nodes.set_index(
            result_nodes["id_node"].astype(int), drop=False
        )
        first_line = _directed_line(
            first.geometry,
            node_index.loc[int(first["id_us_node"])].geometry,
            node_index.loc[int(first["id_ds_node"])].geometry,
        )
        second_line = _directed_line(
            second.geometry,
            node_index.loc[int(second["id_us_node"])].geometry,
            node_index.loc[int(second["id_ds_node"])].geometry,
        )
        coordinates = list(first_line.coords)
        second_coordinates = list(second_line.coords)
        if Point(coordinates[-1]).distance(Point(second_coordinates[0])) > 1e-6:
            raise ValueError(
                f"Computational {role} interface geometries do not meet at node "
                f"{boundary_id}."
            )
        coordinates.extend(second_coordinates[1:])
        combined_geometry = LineString(coordinates)
        local["id_us_node"] = int(first["id_us_node"])
        local["id_ds_node"] = int(second["id_ds_node"])
        local["id_nodes"] = f"{int(first['id_us_node'])}, {int(second['id_ds_node'])}"
        local[result_links.geometry.name] = combined_geometry
        for field in ("len", "len_adj"):
            if field in result_links.columns:
                local[field] = float(combined_geometry.length)
        for field in ("idx_link", "wid_pix"):
            if field in result_links.columns:
                local[field] = None
        result_links = result_links.loc[
            ~result_links["id_link"].astype(int).isin(incident_ids)
        ].copy()
        result_links = gpd.GeoDataFrame(
            pd.concat(
                [
                    result_links,
                    gpd.GeoDataFrame(
                        [local], geometry=result_links.geometry.name, crs=result_links.crs
                    ),
                ],
                ignore_index=True,
            ),
            geometry=result_links.geometry.name,
            crs=result_links.crs,
        )
        result_nodes = result_nodes.loc[
            ~result_nodes["id_node"].astype(int).eq(boundary_id)
        ].copy()
        audit.append(
            {
                "role": role,
                "removed_computational_node_id": boundary_id,
                "removed_parent_stub_link_id": original_stub_id,
                "retained_local_link_id": local_id,
                "final_upstream_node_id": int(local["id_us_node"]),
                "final_downstream_node_id": int(local["id_ds_node"]),
            }
        )
    return (
        result_links.sort_values("id_link").reset_index(drop=True),
        result_nodes.sort_values("id_node").reset_index(drop=True),
        {
            "method": "dissolve_temporary_cut_node_into_parent_stub_and_local_link",
            "interfaces": audit,
            "dissolved_interface_count": int(len(audit)),
        },
    )
def _assert_parent_outside_splice_is_exact(
    *,
    parent_links: gpd.GeoDataFrame,
    realized_links: gpd.GeoDataFrame,
    replacement_link_ids: set[int],
    interface_split_parent_link_ids: set[int] | None = None,
) -> dict[str, Any]:
    """Require every parent link outside the computational domain unchanged."""
    parent_id = pd.to_numeric(parent_links["id_link"], errors="raise").astype(int)
    realized_id = pd.to_numeric(realized_links["id_link"], errors="raise").astype(int)
    split_ids = set(interface_split_parent_link_ids or ())
    expected_ids = set(parent_id).difference(replacement_link_ids).difference(split_ids)
    realized_index = realized_links.set_index(realized_id, drop=False)
    missing_ids = sorted(expected_ids.difference(realized_index.index.astype(int)))
    changed_ids: list[int] = []
    if not missing_ids:
        parent_index = parent_links.set_index(parent_id, drop=False)
        for link_id in sorted(expected_ids):
            parent_row = parent_index.loc[int(link_id)]
            child_row = realized_index.loc[int(link_id)]
            if (
                int(parent_row["id_us_node"]) != int(child_row["id_us_node"])
                or int(parent_row["id_ds_node"]) != int(child_row["id_ds_node"])
                or not parent_row.geometry.equals_exact(child_row.geometry, 0.0)
            ):
                changed_ids.append(int(link_id))
    if missing_ids or changed_ids:
        raise ValueError(
            "Local realized assembly changed parent links outside its computational "
            f"splice domain; missing={missing_ids}, changed={changed_ids}."
        )
    return {
        "method": "exact_parent_id_endpoints_and_geometry",
        "checked_parent_link_count": int(len(expected_ids)),
        "missing_parent_link_ids": missing_ids,
        "changed_parent_link_ids": changed_ids,
        "interface_split_parent_link_ids": sorted(split_ids),
        "passed": True,
    }


def _computational_domain_boundaries(
    contract: LocalPatchContract,
    parent_links: gpd.GeoDataFrame,
) -> tuple[int, int]:
    """Return the unique directed source/sink of the closed raster domain."""
    core_ids = set(int(value) for value in contract.core_link_ids)
    core = parent_links.loc[parent_links["id_link"].astype(int).isin(core_ids)]
    if len(core) != len(core_ids):
        missing = sorted(core_ids.difference(core["id_link"].astype(int)))
        raise ValueError(f"Computational-domain parent links are missing: {missing}.")
    graph = nx.MultiDiGraph()
    for row in core.itertuples(index=False):
        graph.add_edge(
            int(row.id_us_node), int(row.id_ds_node), key=int(row.id_link)
        )
    sources = sorted(
        int(node_id)
        for node_id in graph.nodes
        if graph.in_degree(node_id) == 0 and graph.out_degree(node_id) > 0
    )
    sinks = sorted(
        int(node_id)
        for node_id in graph.nodes
        if graph.out_degree(node_id) == 0 and graph.in_degree(node_id) > 0
    )
    if len(sources) != 1 or len(sinks) != 1:
        raise ValueError(
            "Local realized assembly requires a closed computational domain with "
            f"one source and sink; sources={sources}, sinks={sinks}."
        )
    if not nx.has_path(graph, sources[0], sinks[0]):
        raise ValueError("Computational-domain source and sink are not connected.")
    return sources[0], sinks[0]


def assemble_local_patch_realized_graph(
    *,
    contract: LocalPatchContract,
    raster_result: LocalPatchRasterResult,
    variant_result: LocalPatchVariantResult,
    parent_links_path: str | Path,
    parent_nodes_path: str | Path,
    output_dir: str | Path,
    match_tolerance_m: float | None = None,
    unchanged_overlap_fraction: float = 0.95,
    boundary_match_tolerance_pixels: float = 8.0,
    expected_cycle_reduction_override: int | None = None,
) -> LocalPatchRealizedResult:
    """Splice the raw local graph at the closed computational-domain boundary.

    The 1,000 m raster cuts are context barriers, not graph interfaces. Parent
    links outside ``contract.core_link_ids`` remain byte-for-byte equivalent
    in topology and geometry. Links inside that domain are
    classified as unchanged parent, partial parent-overlap interface, or
    changed local geometry in the link audit.  Scientific anchored boundary
    IDs are deliberately ignored here.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    parent_links = gpd.read_file(parent_links_path)
    parent_nodes = gpd.read_file(parent_nodes_path)
    if parent_links.crs is None or parent_nodes.crs is None:
        raise ValueError("Parent graph requires a CRS.")
    if float(boundary_match_tolerance_pixels) < 8.0:
        raise ValueError("boundary_match_tolerance_pixels must be at least 8.")
    with rasterio.open(variant_result.output_paths["local_collapsed_mask"]) as source:
        terminal_tolerance = float(
            max(abs(source.transform.a), abs(source.transform.e))
        ) * float(boundary_match_tolerance_pixels)
    domain_source_id, domain_sink_id = _computational_domain_boundaries(
        contract,
        parent_links,
    )
    replacement_ids = tuple(int(value) for value in contract.core_link_ids)
    upstream_ids = (int(domain_source_id),)
    downstream_ids = (int(domain_sink_id),)

    merge_result = merge_local_patch_variant(
        contract=contract,
        raster_result=raster_result,
        variant_result=variant_result,
        parent_links_path=parent_links_path,
        parent_nodes_path=parent_nodes_path,
        output_dir=output_dir / "splice",
        match_tolerance_m=match_tolerance_m,
        unchanged_overlap_fraction=unchanged_overlap_fraction,
        replacement_parent_link_ids=replacement_ids,
        upstream_boundary_node_ids=upstream_ids,
        downstream_boundary_node_ids=downstream_ids,
        merge_stage="local_realized",
        boundary_match_tolerance_m=terminal_tolerance,
        expected_cycle_reduction_override=expected_cycle_reduction_override,
    )

    realized_links = gpd.read_file(merge_result.output_paths["merged_links"])
    realized_nodes = gpd.read_file(merge_result.output_paths["merged_nodes"])
    splice_links = realized_links.copy()
    splice_nodes = realized_nodes.copy()
    parent_order, parent_graph, parent_report = _infer_parent_node_order(
        parent_links,
        parent_nodes,
    )
    with rasterio.open(variant_result.output_paths["local_collapsed_mask"]) as source:
        canonical_match_tolerance = float(
            match_tolerance_m
            or max(abs(source.transform.a), abs(source.transform.e)) * 1.25
        )
    canonical_node_match = _match_child_nodes_to_parent_nodes(
        parent_nodes,
        realized_nodes,
        match_tolerance=canonical_match_tolerance,
        parent_node_order=parent_order,
    )
    canonical_link_match = _match_child_links_to_parent_links(
        parent_links,
        realized_links,
        match_tolerance=canonical_match_tolerance,
    )
    source_id = int(parent_report.source_nodes[0])
    sink_id = int(parent_report.sink_nodes[0])
    splice_links, splice_nodes = _refresh_topology(
        splice_links,
        splice_nodes,
        source_id=source_id,
        sink_id=sink_id,
    )
    splice_graph = build_directed_graph(
        splice_links, splice_nodes, prefer_explicit_direction=True
    )
    splice_report = validate_single_inlet_single_outlet(splice_graph)
    splice_is_dag = nx.is_directed_acyclic_graph(splice_graph)

    canonical_direction_error: str | None = None
    try:
        realized_links, realized_nodes, canonical_direction_summary = (
            _assign_variant_directions(
                child_links=realized_links,
                child_nodes=realized_nodes,
                parent_links=parent_links,
                parent_nodes=parent_nodes,
                node_match=canonical_node_match,
                link_match=canonical_link_match,
            )
        )
        realized_links, realized_nodes = _refresh_topology(
            realized_links,
            realized_nodes,
            source_id=source_id,
            sink_id=sink_id,
        )
    except ValueError as error:
        canonical_direction_error = str(error)
        if not splice_report.is_valid or not splice_is_dag:
            raise
        realized_links = splice_links
        realized_nodes = splice_nodes
        canonical_direction_summary = {
            "method": "preserved_valid_component_direction_after_canonical_error",
            "canonical_direction_error": str(error),
            "is_valid": True,
        }
    realized_graph = build_directed_graph(
        realized_links,
        realized_nodes,
        prefer_explicit_direction=True,
    )
    realized_report = validate_single_inlet_single_outlet(realized_graph)
    realized_is_dag = nx.is_directed_acyclic_graph(realized_graph)
    if (
        (not realized_report.is_valid or not realized_is_dag)
        and splice_report.is_valid
        and splice_is_dag
    ):
        canonical_direction_error = (
            "Canonical parent-fragment direction produced invalid whole-graph "
            f"topology: {list(realized_report.issues)}; is_dag={realized_is_dag}."
        )
        realized_links = splice_links
        realized_nodes = splice_nodes
        realized_graph = splice_graph
        realized_report = splice_report
        realized_is_dag = splice_is_dag
        canonical_direction_summary = {
            **dict(canonical_direction_summary),
            "method": "preserved_valid_component_direction_after_fragment_conflict",
            "canonical_direction_error": canonical_direction_error,
            "is_valid": True,
        }
    if not realized_report.is_valid or not realized_is_dag:
        issues = list(realized_report.issues)
        if not realized_is_dag:
            issues.append("Realized graph is not a directed acyclic graph.")
        raise ValueError(
            "Dissolving computational cut interfaces produced an invalid realized "
            f"graph: {issues}"
        )
    # These are the canonical realized products consumed by shared anchoring.
    # Overwrite the intermediate splice files only after topology validation.
    realized_links.to_file(merge_result.output_paths["merged_links"], driver="GPKG")
    realized_nodes.to_file(merge_result.output_paths["merged_nodes"], driver="GPKG")
    outside_validation = _assert_parent_outside_splice_is_exact(
        parent_links=parent_links,
        realized_links=realized_links,
        replacement_link_ids=set(replacement_ids),
    )
    link_audit = pd.read_csv(merge_result.output_paths["link_merge_audit"])
    lineage_counts = {
        str(key): int(value)
        for key, value in link_audit["realized_lineage_class"].value_counts().items()
    }

    manifest_path = output_dir / "local_realized_assembly_manifest.json"
    output_paths = {
        "realized_links": str(Path(merge_result.output_paths["merged_links"]).resolve()),
        "realized_nodes": str(Path(merge_result.output_paths["merged_nodes"]).resolve()),
        "node_splice_audit": str(
            Path(merge_result.output_paths["node_merge_audit"]).resolve()
        ),
        "link_lineage_audit": str(
            Path(merge_result.output_paths["link_merge_audit"]).resolve()
        ),
        "splice_manifest": str(Path(merge_result.output_paths["manifest"]).resolve()),
        "manifest": str(manifest_path.resolve()),
    }
    summary = {
        "selected_unit_ids": list(contract.selected_unit_ids),
        "computational_replacement_link_ids": sorted(replacement_ids),
        "outer_upstream_boundary_node_ids": list(upstream_ids),
        "outer_downstream_boundary_node_ids": list(downstream_ids),
        "scientific_anchoring_applied": False,
        "lineage_class_counts": lineage_counts,
        "outside_parent_validation": outside_validation,
        "computational_domain_splice": {
            "method": "trim_raster_context_to_closed_domain_then_restore_parent_outside",
            "domain_source_node_id": int(domain_source_id),
            "domain_sink_node_id": int(domain_sink_id),
            "replacement_parent_link_ids": sorted(replacement_ids),
        },
        "realized_graph_validation": {
            **realized_report.to_dict(),
            "is_dag": True,
        },
        "canonical_direction_assignment": dict(canonical_direction_summary),
        "precanonical_splice_direction_validation": {
            **splice_report.to_dict(),
            "is_dag": bool(splice_is_dag),
        },
        "canonical_direction_rejected": bool(canonical_direction_error),
        "canonical_direction_rejection_reason": canonical_direction_error,
        "canonical_direction_match_tolerance_m": canonical_match_tolerance,
        "computational_boundary_match_tolerance_m": terminal_tolerance,
        "computational_boundary_match_tolerance_pixels": float(
            boundary_match_tolerance_pixels
        ),
        "splice_summary": dict(merge_result.summary),
    }
    result = LocalPatchRealizedResult(
        status="complete_local_realized_graph",
        output_paths=output_paths,
        summary=summary,
    )
    manifest_path.write_text(
        json.dumps(result.to_manifest(), indent=2, sort_keys=True) + "\n"
    )
    return result


__all__ = [
    "LOCAL_PATCH_REALIZED_ASSEMBLY_VERSION",
    "LocalPatchRealizedResult",
    "assemble_local_patch_realized_graph",
]
