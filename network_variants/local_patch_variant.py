"""Phase-4 local collapse editing and raw RivGraph materialization.

This module deliberately stops before parent/local graph merging.  It consumes
an accepted Phase-3 crop, applies the requested collapse inside that bounded
raster, preserves both bank-cut barriers, and retains raw local RivGraph
outputs for the Phase-5 conservative merge.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Callable, Mapping

import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd
import rasterio
from shapely.geometry import LineString, Point

from hierarchy_level_definition.unit_detection.bifurcation_confluence_units import (
    analyze_network,
)
from network_variants.local_patch import LocalPatchContract
from network_variants.local_patch_raster import LocalPatchRasterResult
from network_variants.variant_generation import (
    _apply_collapse_components_to_mask,
    _componentize_units,
    _run_rivgraph,
    _write_mask,
)
from hierarchy_level_definition.graph_building.directed_network_checks import (
    parse_id_nodes,
)


LOCAL_PATCH_VARIANT_VERSION = "topology-local-patch-variant-v6"


def _parse_rivgraph_flat_indices(value: object, *, field: str) -> np.ndarray:
    """Parse a RivGraph index field without assuming its serialized format."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return np.asarray([], dtype=np.int64)
    if isinstance(value, (int, np.integer)):
        return np.asarray([int(value)], dtype=np.int64)
    if isinstance(value, (float, np.floating)) and float(value).is_integer():
        return np.asarray([int(value)], dtype=np.int64)
    values = re.findall(r"-?\d+", str(value))
    if not values:
        return np.asarray([], dtype=np.int64)
    result = np.asarray([int(item) for item in values], dtype=np.int64)
    if np.any(result < 0):
        raise ValueError(f"{field} contains negative raster indices.")
    return result


def _normalize_rivgraph_geometries_to_skeleton(
    links_path: str | Path,
    nodes_path: str | Path,
    skeleton_path: str | Path,
    *,
    reference_mask_path: str | Path | None = None,
) -> dict[str, Any]:
    """Rebuild exported vectors from the raster grid that owns their indices.

    RivGraph's ``idx_link`` and ``idx_node`` values are flattened against its
    internal skeleton. In some cropped runs that skeleton has a second crop and
    therefore differs from the supplied local mask. Reinterpreting those
    indices with the local-mask width creates kilometre-scale zigzags. The
    skeleton transform and shape are the authoritative coordinate contract.
    """
    links_path = Path(links_path)
    nodes_path = Path(nodes_path)
    skeleton_path = Path(skeleton_path)
    if not skeleton_path.exists():
        raise FileNotFoundError(f"Local RivGraph skeleton is missing: {skeleton_path}")

    with rasterio.open(skeleton_path) as skeleton:
        skeleton_shape = (int(skeleton.height), int(skeleton.width))
        skeleton_transform = skeleton.transform
        skeleton_crs = skeleton.crs
    if skeleton_crs is None:
        raise ValueError("Local RivGraph skeleton has no CRS.")

    reference_shape: tuple[int, int] | None = None
    if reference_mask_path is not None:
        with rasterio.open(reference_mask_path) as reference:
            reference_shape = (int(reference.height), int(reference.width))

    links = gpd.read_file(links_path).to_crs(skeleton_crs)
    nodes = gpd.read_file(nodes_path).to_crs(skeleton_crs)
    if "idx_link" not in links.columns or "idx_node" not in nodes.columns:
        raise ValueError(
            "Raw local RivGraph vectors require idx_link and idx_node for grid normalization."
        )

    normalized_link_count = 0
    link_geometries = []
    raster_size = int(np.prod(skeleton_shape))
    for row in links.itertuples(index=False):
        indices = _parse_rivgraph_flat_indices(row.idx_link, field="idx_link")
        if indices.size < 2:
            raise ValueError(f"RivGraph link {int(row.id_link)} has fewer than two pixels.")
        if int(indices.max()) >= raster_size:
            raise ValueError(
                f"RivGraph link {int(row.id_link)} index {int(indices.max())} exceeds "
                f"skeleton size {raster_size}."
            )
        rows, cols = np.unravel_index(indices, skeleton_shape)
        xs, ys = rasterio.transform.xy(
            skeleton_transform, rows, cols, offset="center"
        )
        coordinates: list[tuple[float, float]] = []
        for x, y in zip(xs, ys):
            coordinate = (float(x), float(y))
            if not coordinates or coordinate != coordinates[-1]:
                coordinates.append(coordinate)
        if len(coordinates) < 2:
            raise ValueError(
                f"RivGraph link {int(row.id_link)} collapses to one coordinate."
            )
        link_geometries.append(LineString(coordinates))
        normalized_link_count += 1

    normalized_node_count = 0
    node_geometries = []
    for row in nodes.itertuples(index=False):
        indices = _parse_rivgraph_flat_indices(row.idx_node, field="idx_node")
        if indices.size != 1:
            raise ValueError(
                f"RivGraph node {int(row.id_node)} must have exactly one raster index."
            )
        index = int(indices[0])
        if index >= raster_size:
            raise ValueError(
                f"RivGraph node {int(row.id_node)} index {index} exceeds skeleton size "
                f"{raster_size}."
            )
        row_index, col_index = np.unravel_index(index, skeleton_shape)
        x, y = rasterio.transform.xy(
            skeleton_transform, int(row_index), int(col_index), offset="center"
        )
        node_geometries.append(Point(float(x), float(y)))
        normalized_node_count += 1

    links = links.set_geometry(gpd.GeoSeries(link_geometries, index=links.index, crs=skeleton_crs))
    nodes = nodes.set_geometry(gpd.GeoSeries(node_geometries, index=nodes.index, crs=skeleton_crs))
    temporary_links = links_path.with_name(f"{links_path.stem}_normalized_tmp.gpkg")
    temporary_nodes = nodes_path.with_name(f"{nodes_path.stem}_normalized_tmp.gpkg")
    temporary_links.unlink(missing_ok=True)
    temporary_nodes.unlink(missing_ok=True)
    links.to_file(temporary_links, layer=links_path.stem, driver="GPKG")
    nodes.to_file(temporary_nodes, layer=nodes_path.stem, driver="GPKG")
    temporary_links.replace(links_path)
    temporary_nodes.replace(nodes_path)

    return {
        "method": "rebuild_vectors_from_authoritative_skeleton_indices",
        "skeleton_shape": list(skeleton_shape),
        "reference_mask_shape": list(reference_shape) if reference_shape is not None else None,
        "skeleton_differs_from_reference_mask": bool(
            reference_shape is not None and skeleton_shape != reference_shape
        ),
        "normalized_link_count": normalized_link_count,
        "normalized_node_count": normalized_node_count,
    }


def _local_rivgraph_run_name(contract: LocalPatchContract) -> str:
    """Return a short deterministic name while manifests retain the full IDs."""
    unit_ids = tuple(int(value) for value in contract.selected_unit_ids)
    candidate = f"{contract.example_id}__local_{'_'.join(map(str, unit_ids))}"
    if len(candidate) <= 120:
        return candidate
    digest = hashlib.sha1(
        ",".join(map(str, unit_ids)).encode("utf-8")
    ).hexdigest()[:10]
    selection = str(contract.selection_type).replace("_", "-")
    identifier = (
        f"bubble-{int(contract.selected_compound_bubble_id)}"
        if contract.selected_compound_bubble_id is not None
        else f"n-{len(unit_ids)}"
    )
    return f"{contract.example_id}__local_{selection}_{identifier}_{digest}"


def _side_distance(
    side: str,
    *,
    x: float,
    y: float,
    bounds: tuple[float, float, float, float],
) -> float:
    minx, miny, maxx, maxy = bounds
    return {
        "W": abs(float(x) - float(minx)),
        "E": abs(float(maxx) - float(x)),
        "S": abs(float(y) - float(miny)),
        "N": abs(float(maxy) - float(y)),
    }[str(side)]


def derive_local_exit_side_candidates(
    raster_result: LocalPatchRasterResult,
    *,
    maximum_candidates: int = 4,
) -> tuple[str, ...]:
    """Return a bounded, deterministic list of ordered RivGraph exit pairs.

    The dominant cut-to-cut orientation remains primary. Alternate pairs are
    ordered by how closely their source and sink sides approach the matching
    cut anchors. This includes mixed-axis candidates for curved or nested
    reaches without trying every cardinal permutation.
    """
    primary, _ = derive_local_exit_sides(raster_result)
    attempt = raster_result.selected_attempt
    if attempt is None:
        raise ValueError("Local exit-side candidates require a selected raster attempt.")
    cuts = {str(cut.role): cut for cut in attempt.bank_cuts}
    source, sink = cuts["upstream"], cuts["downstream"]
    bounds = tuple(float(value) for value in attempt.bounds)
    alternatives: list[tuple[float, str]] = []
    for source_side in "NSEW":
        for sink_side in "NSEW":
            if source_side == sink_side:
                continue
            pair = f"{source_side}{sink_side}"
            if pair == primary:
                continue
            score = _side_distance(
                source_side,
                x=float(source.anchor_x),
                y=float(source.anchor_y),
                bounds=bounds,
            ) + _side_distance(
                sink_side,
                x=float(sink.anchor_x),
                y=float(sink.anchor_y),
                bounds=bounds,
            )
            alternatives.append((float(score), pair))
    alternatives.sort(key=lambda item: (item[0], item[1]))
    limit = max(1, int(maximum_candidates))
    return tuple([primary, *[pair for _, pair in alternatives[: limit - 1]]])


def derive_local_exit_sides(
    raster_result: LocalPatchRasterResult,
) -> tuple[str, dict[str, Any]]:
    """Derive ordered RivGraph exit sides from cut-to-cut flow displacement.

    The cut anchors are the intersections of the accepted bank cuts with the
    reference centerline.  Their upstream-to-downstream displacement is
    therefore independent of the rectangular crop shape and halo.  RivGraph's
    first side is the upstream/source side and its second side is the
    downstream/sink side.
    """
    attempt = raster_result.selected_attempt
    if not raster_result.is_safe or attempt is None:
        raise ValueError("Local exit sides require a safe selected raster attempt.")

    cuts_by_role = {str(cut.role): cut for cut in attempt.bank_cuts}
    if set(cuts_by_role) != {"upstream", "downstream"}:
        raise ValueError(
            "Local exit-side derivation requires exactly one upstream and one downstream cut."
        )

    source = cuts_by_role["upstream"]
    sink = cuts_by_role["downstream"]
    source_x, source_y = float(source.anchor_x), float(source.anchor_y)
    sink_x, sink_y = float(sink.anchor_x), float(sink.anchor_y)
    delta_x = sink_x - source_x
    delta_y = sink_y - source_y
    if np.hypot(delta_x, delta_y) <= 1e-9:
        raise ValueError("Upstream and downstream cut centres are coincident.")

    if abs(delta_x) >= abs(delta_y):
        dominant_axis = "horizontal"
        source_side, sink_side = (("W", "E") if delta_x > 0 else ("E", "W"))
    else:
        dominant_axis = "vertical"
        source_side, sink_side = (("S", "N") if delta_y > 0 else ("N", "S"))
    local_exit_sides = f"{source_side}{sink_side}"
    diagnostics = {
        "method": "dominant_cut_to_cut_reference_displacement",
        "local_exit_sides": local_exit_sides,
        "source_role": "upstream",
        "sink_role": "downstream",
        "source_cut_anchor": [source_x, source_y],
        "sink_cut_anchor": [sink_x, sink_y],
        "cut_to_cut_delta_x_m": float(delta_x),
        "cut_to_cut_delta_y_m": float(delta_y),
        "cut_to_cut_distance_m": float(np.hypot(delta_x, delta_y)),
        "dominant_axis": dominant_axis,
    }
    return local_exit_sides, diagnostics


def validate_cut_terminal_graph(
    links: gpd.GeoDataFrame,
    nodes: gpd.GeoDataFrame,
    cuts: gpd.GeoDataFrame,
    *,
    tolerance_m: float,
    require_directed_path: bool,
) -> dict[str, Any]:
    """Assign source/sink terminals using role, relative position, and topology.

    Computational cuts are soft interfaces. The pixel-scaled tolerance is
    preferred but not absolute: a terminal beyond it is accepted when it is
    substantially closer to its intended cut than to the opposite cut and is
    within one quarter of the cut-to-cut separation.
    """
    if links.crs is None or nodes.crs is None or cuts.crs is None:
        raise ValueError("Raw local graph and bank cuts require a CRS.")
    links = links.to_crs(cuts.crs)
    nodes = nodes.to_crs(cuts.crs)
    cuts_by_role = {
        str(row.role): row.geometry for row in cuts.itertuples(index=False)
    }
    if set(cuts_by_role) != {"upstream", "downstream"}:
        raise ValueError("Raw terminal validation requires one upstream and one downstream cut.")
    if "is_inlet" not in nodes.columns or "is_outlet" not in nodes.columns:
        raise ValueError("Raw RivGraph nodes lack inlet/outlet role columns.")

    inlet_nodes = nodes.loc[nodes["is_inlet"].fillna(False).astype(bool)].copy()
    outlet_nodes = nodes.loc[nodes["is_outlet"].fillna(False).astype(bool)].copy()
    role_specs = (
        ("upstream", "inlet", inlet_nodes),
        ("downstream", "outlet", outlet_nodes),
    )
    selected: dict[str, int] = {}
    distances: dict[str, float] = {}
    selection_methods: dict[str, str] = {}
    candidate_records: dict[str, list[dict[str, Any]]] = {}
    cut_separation = float(
        cuts_by_role["upstream"].distance(cuts_by_role["downstream"])
    )
    relative_limit = max(float(tolerance_m), cut_separation * 0.25)
    for cut_role, terminal_role, candidates in role_specs:
        if candidates.empty:
            raise ValueError(f"Raw local RivGraph has no node marked as {terminal_role}.")
        opposite_role = "downstream" if cut_role == "upstream" else "upstream"
        intended_distances = candidates.geometry.distance(cuts_by_role[cut_role])
        opposite_distances = candidates.geometry.distance(cuts_by_role[opposite_role])
        records: list[dict[str, Any]] = []
        for index in candidates.index:
            intended = float(intended_distances.loc[index])
            opposite = float(opposite_distances.loc[index])
            within_fixed = intended <= float(tolerance_m)
            relatively_assigned = (
                intended < opposite
                and intended <= relative_limit
                and (opposite <= 0.0 or intended / opposite <= 0.5)
            )
            records.append(
                {
                    "index": index,
                    "node_id": int(candidates.loc[index, "id_node"]),
                    "intended_cut_distance_m": intended,
                    "opposite_cut_distance_m": opposite,
                    "within_fixed_tolerance": bool(within_fixed),
                    "relative_assignment_valid": bool(relatively_assigned),
                }
            )
        valid = [
            record
            for record in records
            if record["within_fixed_tolerance"] or record["relative_assignment_valid"]
        ]
        if not valid:
            nearest = float(intended_distances.min())
            raise ValueError(
                f"Raw local RivGraph has no unambiguous {terminal_role} terminal for "
                f"the {cut_role} cut (fixed_tolerance={float(tolerance_m):.3f} m, "
                f"relative_limit={relative_limit:.3f} m, nearest={nearest:.3f} m)."
            )
        valid.sort(
            key=lambda record: (
                not record["within_fixed_tolerance"],
                record["intended_cut_distance_m"],
                record["node_id"],
            )
        )
        chosen = valid[0]
        selected[cut_role] = int(chosen["node_id"])
        distances[cut_role] = float(chosen["intended_cut_distance_m"])
        selection_methods[cut_role] = (
            "nearest_role_terminal_within_fixed_tolerance"
            if chosen["within_fixed_tolerance"]
            else "relative_cut_proximity"
        )
        candidate_records[cut_role] = [
            {key: value for key, value in record.items() if key != "index"}
            for record in records
        ]
    if selected["upstream"] == selected["downstream"]:
        raise ValueError("The same raw local node was assigned to both bank cuts.")

    graph = nx.DiGraph() if require_directed_path else nx.Graph()
    for value in links["id_nodes"]:
        upstream_id, downstream_id = parse_id_nodes(value)
        graph.add_edge(int(upstream_id), int(downstream_id))
    if not nx.has_path(graph, selected["upstream"], selected["downstream"]):
        raise ValueError(
            "Raw local RivGraph has no "
            f"{'directed' if require_directed_path else 'undirected'} path from the "
            "upstream-cut terminal to the downstream-cut terminal."
        )
    path_key = "directed_cut_to_cut_path" if require_directed_path else "undirected_cut_to_cut_path"
    return {
        "method": (
            "explicit_directed_cut_role_terminal_validation"
            if require_directed_path
            else "explicit_undirected_cut_role_terminal_validation"
        ),
        "tolerance_m": float(tolerance_m),
        "upstream_terminal_node_id": selected["upstream"],
        "downstream_terminal_node_id": selected["downstream"],
        "upstream_terminal_distance_m": distances["upstream"],
        "downstream_terminal_distance_m": distances["downstream"],
        "upstream_terminal_selection_method": selection_methods["upstream"],
        "downstream_terminal_selection_method": selection_methods["downstream"],
        "cut_to_cut_separation_m": cut_separation,
        "relative_terminal_distance_limit_m": relative_limit,
        "terminal_candidates": candidate_records,
        path_key: True,
    }


def validate_raw_cut_terminals(
    links_path: str | Path,
    nodes_path: str | Path,
    cuts_path: str | Path,
    *,
    tolerance_m: float,
) -> dict[str, Any]:
    """Validate raw RivGraph topology without treating ``id_nodes`` as directed."""
    return validate_cut_terminal_graph(
        gpd.read_file(links_path),
        gpd.read_file(nodes_path),
        gpd.read_file(cuts_path),
        tolerance_m=tolerance_m,
        require_directed_path=False,
    )


@dataclass(frozen=True, slots=True)
class LocalPatchVariantResult:
    status: str
    selected_unit_ids: tuple[int, ...]
    output_paths: Mapping[str, str]
    collapse_summary: Mapping[str, Any]
    variant_version: str = LOCAL_PATCH_VARIANT_VERSION

    def to_manifest(self) -> dict[str, Any]:
        return {
            "variant_version": self.variant_version,
            "status": self.status,
            "selected_unit_ids": list(self.selected_unit_ids),
            "output_paths": dict(self.output_paths),
            "collapse_summary": dict(self.collapse_summary),
        }


def materialize_local_patch_variant(
    *,
    contract: LocalPatchContract,
    raster_result: LocalPatchRasterResult,
    reviewed_links_path: str | Path,
    reviewed_nodes_path: str | Path,
    exit_sides: str,
    output_dir: str | Path,
    preferred_width_field: str = "wid_adj_routing",
    footprint_buffer_scale: float = 0.5,
    all_touched: bool = True,
    allow_noop: bool = False,
    single_thread: bool = False,
    verbose_rivgraph: bool = False,
    max_path_cutoff: int = 100,
    max_paths: int = 5000,
    local_exit_sides_override: str | None = None,
    network_analyzer: Callable[..., tuple[Any, list[Any], Any]] = analyze_network,
    rivgraph_runner: Callable[..., Mapping[str, Path | None]] = _run_rivgraph,
) -> LocalPatchVariantResult:
    """Apply one selected collapse inside a safe crop and run RivGraph locally."""
    if not raster_result.is_safe or raster_result.selected_attempt is None:
        raise ValueError("Local collapse execution requires a safe Phase-3 raster result.")
    required_paths = {
        "local_base_mask",
        "main_bank_cut_barrier",
        "boundary_component_labels",
        "main_bank_cuts",
    }
    missing = sorted(required_paths - set(raster_result.output_paths))
    if missing:
        raise ValueError(f"Phase-3 raster result is missing outputs: {missing}.")

    output_dir = Path(output_dir)
    mask_dir = output_dir / "mask"
    rivgraph_dir = output_dir / "rivgraph"
    summary_dir = output_dir / "summary"
    for directory in (mask_dir, rivgraph_dir, summary_dir):
        directory.mkdir(parents=True, exist_ok=True)

    reviewed_links_path = Path(reviewed_links_path)
    reviewed_nodes_path = Path(reviewed_nodes_path)
    _, units, _ = network_analyzer(
        reviewed_links_path,
        reviewed_nodes_path,
        max_path_cutoff=max_path_cutoff,
        max_paths=max_paths,
    )
    units_by_id = {int(unit.unit_id): unit for unit in units}
    missing_units = sorted(
        int(unit_id)
        for unit_id in contract.selected_unit_ids
        if int(unit_id) not in units_by_id
    )
    if missing_units:
        raise ValueError(f"Selected units are missing from the reviewed graph: {missing_units}.")
    components = _componentize_units(contract.selected_unit_ids, units_by_id)
    reviewed_links = gpd.read_file(reviewed_links_path)

    local_base_mask_path = Path(raster_result.output_paths["local_base_mask"])
    (
        collapsed_mask,
        origin_mask,
        collapse_summary,
        collapse_components,
        edit_geometries,
    ) = _apply_collapse_components_to_mask(
        local_base_mask_path,
        components,
        reviewed_links,
        preferred_width_field=preferred_width_field,
        buffer_scale=footprint_buffer_scale,
        all_touched=all_touched,
        allow_noop=allow_noop,
    )

    barrier_path = Path(raster_result.output_paths["main_bank_cut_barrier"])
    with rasterio.open(local_base_mask_path) as source:
        profile = source.profile.copy()
        local_transform = source.transform
        local_crs = source.crs
    with rasterio.open(barrier_path) as source:
        if source.transform != local_transform or source.crs != local_crs:
            raise ValueError("The Phase-3 cut barrier is not aligned with the local base mask.")
        cut_barrier = source.read(1) > 0
    if cut_barrier.shape != collapsed_mask.shape:
        raise ValueError("The Phase-3 cut barrier has the wrong raster shape.")

    recleared_pixels = int(np.count_nonzero(collapsed_mask & cut_barrier))
    collapsed_mask = np.asarray(collapsed_mask, dtype=np.uint8)
    collapsed_mask[cut_barrier] = 0
    origin_mask = np.asarray(origin_mask)
    origin_mask[cut_barrier] = 0
    collapse_summary = dict(collapse_summary)
    collapse_summary.update(
        {
            "cut_barrier_pixels_recleared": recleared_pixels,
            "collapsed_water_pixels_after_cut_enforcement": int(collapsed_mask.sum()),
            "local_patch_contract_version": contract.contract_version,
            "local_patch_raster_version": raster_result.planner_version,
        }
    )

    collapsed_mask_path = _write_mask(mask_dir / "local_collapsed_mask.tif", collapsed_mask, profile)
    origin_mask_path = _write_mask(mask_dir / "local_origin_mask.tif", origin_mask, profile)
    collapse_components_path = summary_dir / "collapse_components.csv"
    collapse_components.to_csv(collapse_components_path, index=False)
    edit_geometries_path = mask_dir / "collapse_edit_geometries.gpkg"
    if edit_geometries.empty:
        edit_geometries_path = None
    else:
        edit_geometries.to_file(edit_geometries_path, driver="GPKG")

    global_exit_sides = str(exit_sides).upper()
    derived_exit_sides, local_exit_side_assignment = derive_local_exit_sides(raster_result)
    local_exit_sides = str(local_exit_sides_override or derived_exit_sides).upper()
    if (
        len(local_exit_sides) != 2
        or local_exit_sides[0] == local_exit_sides[1]
        or any(side not in "NSEW" for side in local_exit_sides)
    ):
        raise ValueError(
            "local_exit_sides_override must contain two distinct sides from N, S, E, W."
        )
    local_exit_side_assignment = {
        **dict(local_exit_side_assignment),
        "derived_local_exit_sides": derived_exit_sides,
        "local_exit_sides": local_exit_sides,
        "override_applied": bool(local_exit_sides_override is not None),
    }
    run_name = _local_rivgraph_run_name(contract)
    collapse_summary.update(
        {
            "global_exit_sides": global_exit_sides,
            "local_exit_sides": local_exit_sides,
            "local_exit_side_assignment": local_exit_side_assignment,
            "local_rivgraph_run_name": run_name,
        }
    )

    rivgraph_paths = rivgraph_runner(
        run_name=run_name,
        collapsed_mask_path=collapsed_mask_path,
        rivgraph_output_dir=rivgraph_dir,
        exit_sides=local_exit_sides,
        single_thread=single_thread,
        export_sword=False,
        verbose=verbose_rivgraph,
        metadata={
            "example_id": contract.example_id,
            "selected_unit_ids": ",".join(map(str, contract.selected_unit_ids)),
            "execution_mode": "local_patch",
            "global_exit_sides": global_exit_sides,
            "local_exit_sides": local_exit_sides,
            "local_exit_side_method": local_exit_side_assignment["method"],
        },
    )
    if rivgraph_paths.get("links") is None or rivgraph_paths.get("nodes") is None:
        raise RuntimeError("Local RivGraph did not produce links and nodes.")
    if rivgraph_paths.get("skeleton") is None:
        raise RuntimeError("Local RivGraph did not produce its authoritative skeleton raster.")
    collapse_summary["raw_rivgraph_geometry_normalization"] = (
        _normalize_rivgraph_geometries_to_skeleton(
            rivgraph_paths["links"],
            rivgraph_paths["nodes"],
            rivgraph_paths["skeleton"],
            reference_mask_path=collapsed_mask_path,
        )
    )
    cut_terminal_validation = validate_raw_cut_terminals(
        rivgraph_paths["links"],
        rivgraph_paths["nodes"],
        raster_result.output_paths["main_bank_cuts"],
        tolerance_m=float(max(abs(local_transform.a), abs(local_transform.e))) * 8.0,
    )
    collapse_summary["raw_cut_terminal_validation"] = cut_terminal_validation

    output_paths: dict[str, str] = {
        "local_base_mask": str(local_base_mask_path.resolve()),
        "local_collapsed_mask": str(collapsed_mask_path.resolve()),
        "local_origin_mask": str(origin_mask_path.resolve()),
        "main_bank_cut_barrier": str(barrier_path.resolve()),
        "main_bank_cuts": str(Path(raster_result.output_paths["main_bank_cuts"]).resolve()),
        "boundary_component_labels": str(
            Path(raster_result.output_paths["boundary_component_labels"]).resolve()
        ),
        "collapse_components": str(collapse_components_path.resolve()),
    }
    if edit_geometries_path is not None:
        output_paths["collapse_edit_geometries"] = str(edit_geometries_path.resolve())
    for key, value in rivgraph_paths.items():
        if value is not None:
            output_paths[f"raw_rivgraph_{key}"] = str(Path(value).resolve())

    manifest_path = output_dir / "local_patch_variant_manifest.json"
    output_paths["manifest"] = str(manifest_path.resolve())
    result = LocalPatchVariantResult(
        status="complete_raw_local_variant",
        selected_unit_ids=tuple(int(value) for value in contract.selected_unit_ids),
        output_paths=output_paths,
        collapse_summary=collapse_summary,
    )
    manifest_path.write_text(
        json.dumps(result.to_manifest(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result


__all__ = [
    "LOCAL_PATCH_VARIANT_VERSION",
    "LocalPatchVariantResult",
    "_normalize_rivgraph_geometries_to_skeleton",
    "derive_local_exit_side_candidates",
    "derive_local_exit_sides",
    "materialize_local_patch_variant",
    "validate_cut_terminal_graph",
    "validate_raw_cut_terminals",
]
