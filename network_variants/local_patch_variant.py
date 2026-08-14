"""Phase-4 local collapse editing and raw RivGraph materialization.

This module deliberately stops before parent/local graph merging.  It consumes
an accepted Phase-3 crop, applies the requested collapse inside that bounded
raster, preserves both bank-cut barriers, and retains raw local RivGraph
outputs for the Phase-5 conservative merge.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Callable, Mapping

import geopandas as gpd
import numpy as np
import rasterio

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


LOCAL_PATCH_VARIANT_VERSION = "topology-local-patch-variant-v1"


def derive_local_exit_sides(
    raster_result: LocalPatchRasterResult,
) -> tuple[str, dict[str, Any]]:
    """Derive ordered RivGraph exit sides from the accepted source/sink cuts.

    RivGraph requires two distinct cardinal sides, ordered upstream first.  A
    curved local reach can place both cuts nearest the same crop edge, so the
    sides are selected jointly by minimizing their total distance to two
    distinct edges rather than selecting each side independently.
    """
    attempt = raster_result.selected_attempt
    if not raster_result.is_safe or attempt is None:
        raise ValueError("Local exit sides require a safe selected raster attempt.")

    cuts_by_role = {str(cut.role): cut for cut in attempt.bank_cuts}
    if set(cuts_by_role) != {"upstream", "downstream"}:
        raise ValueError(
            "Local exit-side derivation requires exactly one upstream and one downstream cut."
        )

    minx, miny, maxx, maxy = (float(value) for value in attempt.bounds)
    if not maxx > minx or not maxy > miny:
        raise ValueError(f"Local raster attempt has invalid bounds: {attempt.bounds}.")

    def distances(role: str) -> dict[str, float]:
        cut = cuts_by_role[role]
        x = float(cut.anchor_x)
        y = float(cut.anchor_y)
        if x < minx or x > maxx or y < miny or y > maxy:
            raise ValueError(f"The {role} cut centre lies outside the selected crop bounds.")
        return {
            "N": float(maxy - y),
            "S": float(y - miny),
            "E": float(maxx - x),
            "W": float(x - minx),
        }

    source_distances = distances("upstream")
    sink_distances = distances("downstream")
    # The rank only resolves exact geometric ties deterministically.  It does
    # not encode a preferred global flow direction.
    side_rank = {side: rank for rank, side in enumerate(("N", "S", "E", "W"))}
    candidates = [
        (
            float(source_distances[source_side] + sink_distances[sink_side]),
            side_rank[source_side],
            side_rank[sink_side],
            source_side,
            sink_side,
        )
        for source_side in "NSEW"
        for sink_side in "NSEW"
        if source_side != sink_side
    ]
    _, _, _, source_side, sink_side = min(candidates)
    local_exit_sides = f"{source_side}{sink_side}"
    diagnostics = {
        "method": "minimum_joint_cut_distance_to_distinct_crop_sides",
        "local_exit_sides": local_exit_sides,
        "source_role": "upstream",
        "sink_role": "downstream",
        "source_cut_anchor": [
            float(cuts_by_role["upstream"].anchor_x),
            float(cuts_by_role["upstream"].anchor_y),
        ],
        "sink_cut_anchor": [
            float(cuts_by_role["downstream"].anchor_x),
            float(cuts_by_role["downstream"].anchor_y),
        ],
        "crop_bounds": [minx, miny, maxx, maxy],
        "source_side_distances_m": source_distances,
        "sink_side_distances_m": sink_distances,
        "joint_distance_m": float(
            source_distances[source_side] + sink_distances[sink_side]
        ),
    }
    return local_exit_sides, diagnostics


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
    local_exit_sides, local_exit_side_assignment = derive_local_exit_sides(raster_result)
    collapse_summary.update(
        {
            "global_exit_sides": global_exit_sides,
            "local_exit_sides": local_exit_sides,
            "local_exit_side_assignment": local_exit_side_assignment,
        }
    )

    run_name = f"{contract.example_id}__local_{'_'.join(map(str, contract.selected_unit_ids))}"
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
    "derive_local_exit_sides",
    "materialize_local_patch_variant",
]
