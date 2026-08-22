from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio

repo_root = Path(__file__).resolve().parents[1]
rapid_src = repo_root / "RAPID" / "src"
if str(rapid_src) not in sys.path:
    sys.path.insert(0, str(rapid_src))
rivgraph_src = repo_root / "rivgraph_centerline" / "src"
if str(rivgraph_src) not in sys.path:
    sys.path.insert(0, str(rivgraph_src))

if __package__ is None or __package__ == "":
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

from hierarchy_level_definition.run_unit_workflow import (
    UnitWorkflowOutputs,
    run_unit_workflow,
    write_unit_workflow_outputs,
)
from hierarchy_level_definition.unit_detection.bifurcation_confluence_units import (
    DIRECTED_GRAPH_SCHEMA_VERSION,
)
from hierarchy_level_definition.metrics import (
    BubbleEquivalentResistanceConfig,
    LENGTH_FIELD_CANDIDATES,
    compare_states,
    resolve_length_field_candidates,
)
from network_variants.hydraulic_context import (
    SlopeConfig,
    StateHydraulicContextOutputs,
    generate_state_hydraulic_context,
)
from network_variants.equivalent_transition_domain import (
    build_matched_child_domain,
    load_parent_to_child_anchor_map,
    read_network,
    select_smallest_closed_parent_domain,
    solve_hydraulic_domain,
)
from network_variants.sword_matching import match_variant_nodes_to_sword
from network_variants.variant_generation import (
    COLLAPSE_CORRIDOR_MODES,
    NetworkVariantOutputs,
    VariantDirectionValidationError,
    _apply_width_sample_flow_alignment,
    _apply_collapsed_selection_metadata,
    _assign_variant_directions,
    _collapsed_selection_metadata,
    _component_footprint_geometry,
    _ensure_boundary_component_mask,
    _infer_parent_node_order,
    _match_child_links_to_parent_links,
    _match_child_nodes_to_parent_nodes,
    _merge_width_families_onto_links,
    _resolve_link_lineage,
    _write_width_family_outputs,
    _write_state_bank_labels,
    compute_width_families,
    derive_anchored_variant_from_realized,
    generate_network_variant,
)
from network_variants.build_boundary_component_mask import build_panelized_boundary_component_labels
from network_variants.base_width_review_application import attach_permanent_base_review_provenance
from network_variants.routing_width_application import (
    apply_routing_width_model,
    write_routing_width_audit,
)
from network_variants.width_product_inheritance import inherit_unchanged_link_width_products
from network_variants.panelized_raster import (
    compute_panelized_base_width_families,
    compute_panelized_child_width_patch,
    materialize_panelized_child_state,
    write_panelized_virtual_support_rasters,
    write_state_bank_labels_windowed,
    write_state_origin_mask_windowed,
)
from network_variants.local_patch_execution import (
    LocalPatchFallbackRequested,
    materialize_local_patch_network_variant,
)
from rivgraph_centerline.workflow import run_rivgraph_for_run
from rapid_tools.forcing import ForcingConfig, prepare_forcing_table
from rapid_tools.reference_section import (
    REFERENCE_SECTION_VARIANT_ID,
    compute_reference_section_kb,
)


EXPERIMENT_MODES = (
    "independent-units",
    "compound-bubbles",
    "sequential-units",
    "sequential-groups",
)
EXECUTION_MODES = ("full_raster", "panelized", "local_patch")
COLLAPSE_CORRIDOR_MODE_CHOICES = COLLAPSE_CORRIDOR_MODES + ("both",)


@dataclass(frozen=True)
class ExecutionBackendPlan:
    """Resolved base, primary-child, and fallback execution backends."""

    requested_modes: tuple[str, ...]
    base_backend: str
    child_backend: str
    fallback_backend: str | None

    @property
    def is_hybrid_panelized_local_patch(self) -> bool:
        return self.base_backend == "panelized" and self.child_backend == "local_patch"

    def as_manifest(self) -> dict[str, object]:
        return {
            "requested_modes": list(self.requested_modes),
            "base_backend": self.base_backend,
            "child_backend": self.child_backend,
            "fallback_backend": self.fallback_backend,
            "hybrid_panelized_local_patch": self.is_hybrid_panelized_local_patch,
        }


def resolve_execution_backend_plan(
    execution_mode: str | Sequence[str],
) -> ExecutionBackendPlan:
    """Resolve backward-compatible CLI modes into explicit execution roles."""
    raw_modes = (
        [execution_mode]
        if isinstance(execution_mode, str)
        else [str(mode) for mode in execution_mode]
    )
    modes: list[str] = []
    for mode in raw_modes:
        normalized = str(mode).strip()
        if not normalized or normalized in modes:
            continue
        if normalized not in EXECUTION_MODES:
            raise ValueError(
                f"Unsupported execution mode {normalized!r}. Expected values from "
                f"{EXECUTION_MODES}."
            )
        modes.append(normalized)
    if not modes:
        raise ValueError("At least one execution mode is required.")

    mode_set = set(modes)
    if mode_set == {"panelized", "local_patch"}:
        return ExecutionBackendPlan(
            requested_modes=("panelized", "local_patch"),
            base_backend="panelized",
            child_backend="local_patch",
            fallback_backend="panelized",
        )
    if len(mode_set) != 1:
        raise ValueError(
            "Unsupported execution-mode combination. The only combined mode is "
            "'panelized local_patch'."
        )

    mode = modes[0]
    if mode == "local_patch":
        return ExecutionBackendPlan(
            requested_modes=(mode,),
            base_backend="full_raster",
            child_backend="local_patch",
            fallback_backend="full_raster",
        )
    return ExecutionBackendPlan(
        requested_modes=(mode,),
        base_backend=mode,
        child_backend=mode,
        fallback_backend=None,
    )


@dataclass
class ExperimentSelection:
    selection_type: str
    selected_unit_ids: list[int]
    selected_group_label: str | None
    selected_group_size: int
    selected_rank_start: int | None
    selected_rank_end: int | None
    selected_collapse_order: int | None
    selected_collapse_priority_score: float | None
    state_suffix: str
    compound_bubble_id: int | None = None
    bubble_root_unit_id: int | None = None
    total_unit_nodes_selected: int | None = None

    @property
    def is_compound_state(self) -> bool:
        return self.selection_type == "compound_bubble"


@dataclass
class ExperimentStateContext:
    state_id: str
    parent_state_id: str | None
    depth: int
    state_dir: Path
    hierarchy_output_dir: Path
    cleaned_mask_path: Path
    reviewed_links_path: Path
    reviewed_nodes_path: Path
    workflow: UnitWorkflowOutputs | None
    analysis_links_path: Path | None = None
    variant_output_dir: Path | None = None
    hydraulic_context_output_dir: Path | None = None
    equivalent_transition_output_dir: Path | None = None
    equivalent_transition_comparison_path: Path | None = None
    equivalent_transition_summary_path: Path | None = None
    equivalent_transition_summary: dict[str, Any] | None = None
    selection: ExperimentSelection | None = None
    hierarchy_deferred: bool = False


@dataclass
class CollapseExperimentOutputs:
    state_registry: pd.DataFrame
    transition_registry: pd.DataFrame
    equivalent_transition_registry: pd.DataFrame
    output_dir: Path
    manifest: dict[str, Any]


@dataclass
class BaseStateVariantOutputs:
    output_dir: Path
    collapsed_mask_path: Path
    directed_links_path: Path
    directed_nodes_path: Path
    bank_labels_path: Path | None = None
    boundary_component_labels_path: Path | None = None


@dataclass
class PanelizedChildVariantOutputs:
    """Minimal state-output contract consumed by the shared child builder."""
    output_dir: Path
    collapsed_mask_path: Path
    directed_links_path: Path
    directed_nodes_path: Path
    panelized_result: dict[str, Any]


def _variant_analysis_links_path(
    *,
    variant_output_dir: str | Path,
    directed_links_path: str | Path,
) -> Path:
    """Prefer the directed link product carrying the complete width families."""
    width_family_links = (
        Path(variant_output_dir) / "widths" / "links_with_width_families.gpkg"
    )
    if width_family_links.exists():
        return width_family_links
    return Path(directed_links_path)


def _finalize_panelized_child_result(
    *,
    panelized_result: Mapping[str, Any],
    state_dir: Path,
    parent_mask_path: Path,
    parent_links_path: Path,
    parent_nodes_path: Path,
    parent_merged_links_path: Path,
    boundary_component_labels_path: Path,
    footprint: Any,
    unit_ids: Sequence[int],
    selected_link_ids: Sequence[int],
    selected_node_ids: Sequence[int],
    group_label: str | None,
    selection_type: str | None,
    compound_bubble_id: int | None,
    bubble_root_unit_id: int | None,
    example_id: str,
    variant_id: str,
    collapse_corridor_mode: str,
    bank_candidate_neighbors_max: int,
    sword_node_source_path: str | Path | None,
    sword_wse_field: str | None,
    sword_match_tolerance: float | None,
    sword_example_station_source_path: str | Path | None,
    sword_station_match_source_path: str | Path | None,
    sword_reach_buffer_steps: int,
) -> NetworkVariantOutputs:
    """Finalize a panelized patch into the regular child-variant contract."""
    merged = panelized_result.get("merged", {})
    raw_links_path = Path(str(merged["directed_links"]))
    raw_nodes_path = Path(str(merged["directed_nodes"]))
    if not raw_links_path.exists() or not raw_nodes_path.exists():
        raise FileNotFoundError("Panelized child merge did not produce directed links and nodes.")
    collapsed_mask = Path(str(panelized_result["collapsed_mask"]))
    if not collapsed_mask.exists():
        raise FileNotFoundError("Panelized child merge did not produce a collapsed state mask.")

    variant_dir = state_dir / "variant"
    summary_dir = variant_dir / "summary"
    mask_dir = variant_dir / "mask"
    matching_dir = variant_dir / "matching"
    directed_dir = variant_dir / "directed"
    width_dir = variant_dir / "widths"
    rivgraph_dir = variant_dir / "rivgraph"
    for directory in (summary_dir, mask_dir, matching_dir, directed_dir, width_dir, rivgraph_dir):
        directory.mkdir(parents=True, exist_ok=True)

    run_name = f"{example_id}__{variant_id}"
    mask_publication = dict(panelized_result.get("mask_publication", {}))
    virtual_support_manifest: dict[str, object] | None = None
    if mask_publication.get("mode") == "virtual_patch":
        collapsed_mask_path = collapsed_mask
        overlay_path = mask_publication.get("overlay_path")
        if not overlay_path:
            raise ValueError("Panelized virtual mask publication is missing its overlay path.")
        (
            origin_mask_path,
            bank_labels_path,
            removed_bank_labels,
            virtual_support_manifest,
        ) = write_panelized_virtual_support_rasters(
            base_mask_path=parent_mask_path,
            collapsed_patch_path=Path(str(overlay_path)),
            boundary_component_labels_path=boundary_component_labels_path,
            output_dir=mask_dir,
        )
    else:
        collapsed_mask_path = mask_dir / f"{run_name}_collapsed.tif"
        shutil.copy2(collapsed_mask, collapsed_mask_path)
        origin_mask_path = write_state_origin_mask_windowed(
            base_mask_path=parent_mask_path,
            collapsed_mask_path=collapsed_mask_path,
            output_path=mask_dir / "origin_mask.tif",
        )
        bank_labels_path = mask_dir / "bank_labels.tif"
        removed_bank_labels = write_state_bank_labels_windowed(
            boundary_component_labels_path=boundary_component_labels_path,
            origin_mask_path=origin_mask_path,
            output_path=bank_labels_path,
        )

    raw_links = _normalize_numeric_id_columns(gpd.read_file(raw_links_path))
    raw_nodes = _normalize_numeric_id_columns(gpd.read_file(raw_nodes_path))
    parent_links = _normalize_numeric_id_columns(gpd.read_file(parent_links_path))
    parent_nodes = _normalize_numeric_id_columns(gpd.read_file(parent_nodes_path))
    parent_variant_dir = parent_links_path.parent.parent
    parent_width_links = gpd.read_file(parent_variant_dir / "widths" / "links_with_width_families.gpkg")
    parent_width_samples = pd.read_csv(parent_variant_dir / "widths" / "link_width_samples.csv")
    child_source_samples = pd.read_csv(str(merged["width_samples"]))
    child_merged_links = gpd.read_file(
        Path(str(merged["core_dir"])) / "merged_links.gpkg",
        layer="merged_links",
    )
    parent_merged_links = gpd.read_file(parent_merged_links_path, layer="merged_links")
    enriched_raw_links, link_width_samples, width_patch_summary = compute_panelized_child_width_patch(
        child_links=raw_links,
        child_source_samples=child_source_samples,
        child_merged_links=child_merged_links,
        parent_merged_links=parent_merged_links,
        parent_width_links=parent_width_links,
        parent_width_samples=parent_width_samples,
        affected_panel_ids=list(panelized_result["panel_plan"]["affected"]),
        bank_labels_path=bank_labels_path,
        origin_mask_path=origin_mask_path,
        bank_candidate_neighbors_max=bank_candidate_neighbors_max,
    )

    with rasterio.open(collapsed_mask_path) as source:
        pixel_length = float(max(abs(source.transform.a), abs(source.transform.e)))
    match_tolerance = pixel_length * 1.25
    parent_node_order, parent_graph, _ = _infer_parent_node_order(parent_links, parent_nodes)
    node_match = _match_child_nodes_to_parent_nodes(
        parent_nodes,
        raw_nodes,
        match_tolerance=match_tolerance,
        parent_node_order=parent_node_order,
    )
    link_match = _match_child_links_to_parent_links(
        parent_links,
        raw_links,
        match_tolerance=match_tolerance,
    )
    directed_links, directed_nodes, direction_summary = _assign_variant_directions(
        child_links=raw_links,
        child_nodes=raw_nodes,
        parent_links=parent_links,
        parent_nodes=parent_nodes,
        node_match=node_match,
        link_match=link_match,
    )
    link_lineage = _resolve_link_lineage(
        parent_graph=parent_graph,
        directed_child_links=directed_links,
        node_match=node_match,
        link_match=link_match,
        match_tolerance=match_tolerance,
    )
    directed_nodes, node_sword_match, sword_match_summary = match_variant_nodes_to_sword(
        directed_nodes=directed_nodes,
        parent_nodes=parent_nodes,
        node_match=node_match,
        sword_node_source_path=sword_node_source_path,
        sword_wse_field=sword_wse_field,
        sword_match_tolerance=sword_match_tolerance,
        example_id=example_id,
        sword_example_station_source_path=sword_example_station_source_path,
        sword_station_match_source_path=sword_station_match_source_path,
        sword_reach_buffer_steps=sword_reach_buffer_steps,
    )

    frames = (node_match, link_match, link_lineage, directed_links, directed_nodes, node_sword_match)
    frames = tuple(
        _apply_collapsed_selection_metadata(frame, unit_ids=unit_ids, group_label=group_label)
        for frame in frames
    )
    node_match, link_match, link_lineage, directed_links, directed_nodes, node_sword_match = frames
    link_width_samples = _apply_collapsed_selection_metadata(
        link_width_samples,
        unit_ids=unit_ids,
        group_label=group_label,
    )
    for frame in (
        node_match,
        link_match,
        link_lineage,
        directed_links,
        directed_nodes,
        node_sword_match,
        link_width_samples,
    ):
        frame["example_id"] = example_id
        frame["variant_id"] = variant_id
    directed_links = directed_links.drop(
        columns=[
            column
            for column in link_lineage.columns
            if column in directed_links.columns and column != "id_link"
        ],
        errors="ignore",
    ).merge(link_lineage, on="id_link", how="left", validate="1:1")
    directed_enriched_links = _merge_width_families_onto_links(directed_links, enriched_raw_links)
    directed_enriched_links = _apply_width_sample_flow_alignment(
        directed_enriched_links,
        collapse_corridor_mode=collapse_corridor_mode,
    )
    directed_enriched_links, link_width_samples, width_product_inheritance = (
        inherit_unchanged_link_width_products(
            child_links=directed_enriched_links,
            child_width_samples=link_width_samples,
            parent_links=parent_width_links,
            parent_width_samples=parent_width_samples,
            link_lineage=link_lineage,
        )
    )
    link_width_samples = _apply_collapsed_selection_metadata(
        link_width_samples,
        unit_ids=unit_ids,
        group_label=group_label,
    )
    link_width_samples["example_id"] = example_id
    link_width_samples["variant_id"] = variant_id
    changed_width_link_ids = directed_enriched_links.loc[
        directed_enriched_links["width_inheritance_status"].eq("changed_requires_evaluation"),
        "id_link",
    ].astype(int).tolist()
    directed_enriched_links, routing_width_audit, routing_width_application = apply_routing_width_model(
        directed_enriched_links,
        link_width_samples,
        directed_nodes,
        example_id=example_id,
        variant_id=variant_id,
        apply_known_base_exclusions=False,
        evaluation_link_ids=changed_width_link_ids,
    )
    directed_enriched_links.loc[
        directed_enriched_links["id_link"].astype(int).isin(changed_width_link_ids),
        "width_inheritance_status",
    ] = "changed_evaluated"
    routing_width_paths = write_routing_width_audit(
        routing_width_audit,
        routing_width_application,
        output_dir=width_dir,
    )
    link_width_families = _write_width_family_outputs(
        width_dir=width_dir,
        link_width_samples=link_width_samples,
        directed_enriched_links=directed_enriched_links,
    )

    selection_metadata = _collapsed_selection_metadata(unit_ids, group_label)
    added_pixels = int(
        sum(int(edit.get("added_pixels", 0)) for edit in panelized_result.get("edits", {}).values())
    )
    pixel_area = pixel_length * pixel_length
    collapse_components = pd.DataFrame(
        [
            {
                "component_id": "C1",
                "unit_ids": _serialize_int_list(unit_ids),
                "link_ids": _serialize_int_list(selected_link_ids),
                "node_ids": _serialize_int_list(selected_node_ids),
                "compound_bubble_ids": (
                    str(int(compound_bubble_id)) if compound_bubble_id is not None else ""
                ),
                "n_units": len(unit_ids),
                "n_links": len(selected_link_ids),
                "n_nodes": len(selected_node_ids),
                "footprint_area": float(footprint.area),
                "added_pixels": added_pixels,
                "added_area": float(added_pixels * pixel_area),
            }
        ]
    )
    collapse_components = _apply_collapsed_selection_metadata(
        collapse_components,
        unit_ids=unit_ids,
        group_label=group_label,
    )
    collapse_components["example_id"] = example_id
    collapse_components["variant_id"] = variant_id
    edit_geometries = gpd.GeoDataFrame(
        {
            "component_id": ["C1"],
            "geometry_role": ["collapse_footprint"],
            "action": ["fill"],
            "unit_ids": [_serialize_int_list(unit_ids)],
        },
        geometry=[footprint],
        crs=parent_links.crs,
    )
    edit_geometries = _apply_collapsed_selection_metadata(
        edit_geometries,
        unit_ids=unit_ids,
        group_label=group_label,
    )
    edit_geometries["example_id"] = example_id
    edit_geometries["variant_id"] = variant_id

    collapse_components.to_csv(summary_dir / "collapse_components.csv", index=False)
    edit_geometries.to_file(mask_dir / "collapse_edit_geometries.gpkg", driver="GPKG")
    node_match.to_csv(matching_dir / "node_match.csv", index=False)
    node_sword_match.to_csv(matching_dir / "node_sword_match.csv", index=False)
    link_match.to_csv(matching_dir / "link_match.csv", index=False)
    link_lineage.to_csv(matching_dir / "link_lineage.csv", index=False)
    directed_links_path = directed_dir / f"{run_name}_directed_links.gpkg"
    directed_nodes_path = directed_dir / f"{run_name}_directed_nodes.gpkg"
    directed_links.to_file(directed_links_path, driver="GPKG")
    directed_nodes.to_file(directed_nodes_path, driver="GPKG")
    validation_report = dict(direction_summary.get("validation_report", {}))
    with (directed_dir / "direction_validation_report.json").open("w", encoding="utf-8") as handle:
        json.dump(validation_report, handle, indent=2)

    rivgraph_links_path = rivgraph_dir / f"{run_name}_links.gpkg"
    rivgraph_nodes_path = rivgraph_dir / f"{run_name}_nodes.gpkg"
    raw_links.to_file(rivgraph_links_path, driver="GPKG")
    raw_nodes.to_file(rivgraph_nodes_path, driver="GPKG")
    child_source_samples.to_csv(rivgraph_dir / f"{run_name}_link_width_samples.csv", index=False)
    source_centerline = raw_links_path.with_name(f"{example_id}_centerline.gpkg")
    rivgraph_centerline_path = rivgraph_dir / f"{run_name}_centerline.gpkg"
    if source_centerline.exists():
        shutil.copy2(source_centerline, rivgraph_centerline_path)
    else:
        rivgraph_centerline_path = None

    manifest = {
        "example_id": example_id,
        "variant_id": variant_id,
        "execution_mode": "panelized",
        "selection_source": "group_label" if group_label is not None else "unit_ids",
        "selection_type": selection_type,
        "compound_bubble_id": compound_bubble_id,
        "bubble_root_unit_id": bubble_root_unit_id,
        "selected_unit_ids": [int(unit_id) for unit_id in unit_ids],
        "group_label": group_label,
        "collapsed_selection": selection_metadata,
        "collapse_corridor_mode": collapse_corridor_mode,
        "n_components": 1,
        "component_ids": ["C1"],
        "ready_for_graph_matching": True,
        "ready_for_rapid": bool(direction_summary.get("is_valid", False)),
        "mask_summary": {
            "pixels_added": added_pixels,
            "pixels_removed": 0,
            "changed_pixels": added_pixels,
        },
        "panelized_patch": {
            **width_patch_summary,
            "panel_plan": panelized_result["panel_plan"],
            "workspace": panelized_result.get("workspace", {}),
            "mask_publication": mask_publication,
            "virtual_support_rasters": virtual_support_manifest,
        },
        "width_product_inheritance": width_product_inheritance,
        "routing_width_application": routing_width_application,
        "bank_label_summary": {
            "removed_land_labels": removed_bank_labels,
            "n_removed_land_labels": int(len(removed_bank_labels)),
        },
        "sword_matching": sword_match_summary,
        "source_paths": {
            "cleaned_mask": str(parent_mask_path.resolve()),
            "wet_reference_mask": str(parent_mask_path.resolve()),
            "reviewed_links": str(parent_links_path.resolve()),
            "reviewed_nodes": str(parent_nodes_path.resolve()),
            "boundary_component_labels": str(
                boundary_component_labels_path.resolve()
            ),
        },
        "output_paths": {
            "output_dir": str(variant_dir.resolve()),
            "collapsed_mask": str(collapsed_mask_path.resolve()),
            "origin_mask": str(origin_mask_path.resolve()),
            "width_scope_mask": str(origin_mask_path.resolve()),
            "bank_labels": str(bank_labels_path.resolve()),
            "boundary_component_labels": str(boundary_component_labels_path.resolve()),
            "collapse_components": str((summary_dir / "collapse_components.csv").resolve()),
            "edit_geometries": str((mask_dir / "collapse_edit_geometries.gpkg").resolve()),
            "link_width_families": str((width_dir / "link_width_families.csv").resolve()),
            "link_width_samples": str((width_dir / "link_width_samples.csv").resolve()),
            "links_with_width_families": str((width_dir / "links_with_width_families.gpkg").resolve()),
            "routing_width_application": routing_width_paths["link_audit"],
            "routing_width_application_manifest": routing_width_paths["manifest"],
            "node_match": str((matching_dir / "node_match.csv").resolve()),
            "node_sword_match": str((matching_dir / "node_sword_match.csv").resolve()),
            "link_match": str((matching_dir / "link_match.csv").resolve()),
            "link_lineage": str((matching_dir / "link_lineage.csv").resolve()),
            "directed_links": str(directed_links_path.resolve()),
            "directed_nodes": str(directed_nodes_path.resolve()),
            "direction_validation_report": str(
                (directed_dir / "direction_validation_report.json").resolve()
            ),
            "rivgraph_links": str(rivgraph_links_path.resolve()),
            "rivgraph_nodes": str(rivgraph_nodes_path.resolve()),
        },
        "direction_summary": direction_summary,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "code_version": _git_revision(),
    }
    with (summary_dir / "variant_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)

    return NetworkVariantOutputs(
        collapse_components=collapse_components,
        edit_geometries=edit_geometries,
        node_match=node_match,
        node_sword_match=node_sword_match,
        link_match=link_match,
        link_lineage=link_lineage,
        link_width_families=link_width_families,
        link_width_samples=link_width_samples,
        enriched_links=directed_enriched_links,
        directed_links=directed_links,
        directed_nodes=directed_nodes,
        output_dir=variant_dir,
        collapsed_mask_path=collapsed_mask_path,
        width_scope_mask_path=origin_mask_path,
        bank_labels_path=bank_labels_path,
        boundary_component_labels_path=boundary_component_labels_path,
        origin_mask_path=origin_mask_path,
        rivgraph_links_path=rivgraph_links_path,
        rivgraph_nodes_path=rivgraph_nodes_path,
        directed_links_path=directed_links_path,
        directed_nodes_path=directed_nodes_path,
        rivgraph_centerline_path=rivgraph_centerline_path,
        sword_reaches_path=None,
        sword_nodes_path=None,
        manifest=manifest,
    )


def _git_revision() -> str | None:
    repo_root = Path(__file__).resolve().parents[1]
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return None
    revision = result.stdout.strip()
    return revision or None


def _infer_example_id(path: str | Path) -> str:
    stem = Path(path).stem
    for suffix in (
        "_cleaned",
        "_binary_projected",
        "directed_links",
        "_links",
        "_nodes",
        "_mask",
    ):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    return stem or Path(path).stem


def _default_experiment_id(cleaned_mask_path: str | Path, mode: str) -> str:
    return f"{_infer_example_id(cleaned_mask_path)}_{mode.replace('-', '_')}"


def _default_output_dir(cleaned_mask_path: str | Path, experiment_id: str) -> Path:
    example_id = _infer_example_id(cleaned_mask_path)
    return Path(__file__).resolve().parent / "experiments" / example_id / experiment_id


def _suffix_experiment_id_for_collapse_mode(experiment_id: str, collapse_corridor_mode: str) -> str:
    suffix = f"__{collapse_corridor_mode}"
    return experiment_id if experiment_id.endswith(suffix) else f"{experiment_id}{suffix}"


def _suffix_output_dir_for_collapse_mode(output_dir: str | Path, collapse_corridor_mode: str) -> Path:
    path = Path(output_dir)
    suffix = f"__{collapse_corridor_mode}"
    if path.name.endswith(suffix):
        return path
    return path.with_name(f"{path.name}{suffix}")


def _copy_boundary_component_artifacts(
    *,
    source_dir: str | Path,
    output_dir: str | Path,
    example_id: str,
) -> Path:
    """Copy a complete boundary-component product and localize manifest paths."""
    source = Path(source_dir).resolve()
    destination = Path(output_dir).resolve()
    if source == destination:
        raise ValueError(
            "Boundary-component reuse source must be outside the experiment output "
            f"directory that will be reset: {source}"
        )
    if not source.is_dir():
        raise FileNotFoundError(f"Boundary-component artifact directory does not exist: {source}")

    labels_name = (
        f"{example_id}_full_extent_land_water_labels_with_snapped_buffered_border_cut.tif"
    )
    if not (source / labels_name).exists():
        raise FileNotFoundError(
            "Boundary-component reuse source does not contain the expected labels: "
            f"{source / labels_name}"
        )
    _copytree_transactional(source, destination)
    labels_path = destination / labels_name

    for manifest_path in destination.glob("*boundary_component_manifest.json"):
        with manifest_path.open("r", encoding="utf-8") as handle:
            manifest = json.load(handle)
        for key in ("labels", "extensions", "summary"):
            value = manifest.get(key)
            if not value:
                continue
            local_path = destination / Path(str(value)).name
            if local_path.exists():
                manifest[key] = str(local_path.resolve())
        with manifest_path.open("w", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2)

    return labels_path


def _infer_boundary_component_artifact_source_dir(
    base_state_variant_source_dir: str | Path,
) -> Path | None:
    """Locate experiment-level boundary artifacts beside a reused base variant."""
    source = Path(base_state_variant_source_dir).resolve()
    for ancestor in source.parents:
        if ancestor.name != "states":
            continue
        candidate = ancestor.parent / "boundary_components"
        return candidate if candidate.is_dir() else None
    return None


def _supports_realized_reuse_speedup(mode: str) -> bool:
    return mode in {"independent-units", "compound-bubbles"}


def _make_realized_reuse_variant_runner(
    *,
    realized_experiment_output_dir: Path,
    fallback_variant_runner: Callable[..., NetworkVariantOutputs],
) -> Callable[..., NetworkVariantOutputs]:
    def runner(**kwargs: Any) -> NetworkVariantOutputs:
        source_output_dir = realized_experiment_output_dir / "states" / str(kwargs["variant_id"]) / "variant"
        if not source_output_dir.exists():
            return fallback_variant_runner(**kwargs)
        return derive_anchored_variant_from_realized(
            source_output_dir=source_output_dir,
            reviewed_links_path=kwargs["reviewed_links_path"],
            reviewed_nodes_path=kwargs["reviewed_nodes_path"],
            output_dir=kwargs["output_dir"],
            example_id=kwargs.get("example_id"),
            variant_id=kwargs.get("variant_id"),
            boundary_component_labels_path=kwargs.get("boundary_component_labels_path"),
            boundary_component_output_dir=kwargs.get("boundary_component_output_dir"),
            sword_node_source_path=kwargs.get("sword_node_source_path"),
            sword_wse_field=kwargs.get("sword_wse_field"),
            sword_match_tolerance=kwargs.get("sword_match_tolerance"),
            sword_example_station_source_path=kwargs.get("sword_example_station_source_path"),
            sword_station_match_source_path=kwargs.get("sword_station_match_source_path"),
            sword_reach_buffer_steps=int(kwargs.get("sword_reach_buffer_steps", 2)),
            max_path_cutoff=int(kwargs.get("max_path_cutoff", 100)),
            max_paths=int(kwargs.get("max_paths", 5000)),
            transect_scale=float(kwargs.get("transect_scale", 1.5)),
            min_transect_pixels=float(kwargs.get("min_transect_pixels", 5.0)),
            bank_candidate_neighbors_max=int(kwargs.get("bank_candidate_neighbors_max", 2048)),
            match_tolerance=kwargs.get("match_tolerance"),
        )

    return runner


def _make_realized_seeded_anchored_variant_runner(
    *,
    fallback_variant_runner: Callable[..., NetworkVariantOutputs],
) -> Callable[..., NetworkVariantOutputs]:
    def runner(**kwargs: Any) -> NetworkVariantOutputs:
        final_output_dir = Path(kwargs["output_dir"])
        temp_parent = final_output_dir.parent
        temp_parent.mkdir(parents=True, exist_ok=True)
        prefix = f"{kwargs.get('variant_id', 'variant')}__realized_seed_"
        with tempfile.TemporaryDirectory(prefix=prefix, dir=temp_parent) as temp_dir:
            realized_kwargs = dict(kwargs)
            realized_kwargs["collapse_corridor_mode"] = "realized"
            realized_kwargs["output_dir"] = str(Path(temp_dir) / "variant")
            realized_outputs = fallback_variant_runner(**realized_kwargs)
            return derive_anchored_variant_from_realized(
                source_output_dir=realized_outputs.output_dir,
                reviewed_links_path=kwargs["reviewed_links_path"],
                reviewed_nodes_path=kwargs["reviewed_nodes_path"],
                output_dir=final_output_dir,
                example_id=kwargs.get("example_id"),
                variant_id=kwargs.get("variant_id"),
                boundary_component_labels_path=kwargs.get("boundary_component_labels_path"),
                boundary_component_output_dir=kwargs.get("boundary_component_output_dir"),
                sword_node_source_path=kwargs.get("sword_node_source_path"),
                sword_wse_field=kwargs.get("sword_wse_field"),
                sword_match_tolerance=kwargs.get("sword_match_tolerance"),
                sword_example_station_source_path=kwargs.get("sword_example_station_source_path"),
                sword_station_match_source_path=kwargs.get("sword_station_match_source_path"),
                sword_reach_buffer_steps=int(kwargs.get("sword_reach_buffer_steps", 2)),
                max_path_cutoff=int(kwargs.get("max_path_cutoff", 100)),
                max_paths=int(kwargs.get("max_paths", 5000)),
                transect_scale=float(kwargs.get("transect_scale", 1.5)),
                min_transect_pixels=float(kwargs.get("min_transect_pixels", 5.0)),
                bank_candidate_neighbors_max=int(kwargs.get("bank_candidate_neighbors_max", 2048)),
                match_tolerance=kwargs.get("match_tolerance"),
            )

    return runner


def _serialize_int_list(values: Sequence[int]) -> str:
    return ",".join(str(int(value)) for value in values)


def _parse_int_list(value: Any) -> list[int]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set, pd.Series)):
        result: list[int] = []
        for item in value:
            result.extend(_parse_int_list(item))
        return result
    text = str(value).strip()
    if not text:
        return []
    return [int(part.strip()) for part in text.split(",") if part.strip()]


def _sanitize_label(value: str) -> str:
    cleaned = "".join(char if char.isalnum() or char in {"_", "-"} else "_" for char in value.strip())
    return cleaned or "selection"


def _selection_label(selection: ExperimentSelection | None) -> str:
    if selection is None:
        return "base_state"
    return _collapsed_selection_metadata(selection.selected_unit_ids, selection.selected_group_label)["collapsed_selection_label"]


def _optional_int(value: Any) -> int | None:
    if value is None or value is pd.NA or pd.isna(value):
        return None
    return int(value)


def _optional_float(value: Any) -> float | None:
    if value is None or value is pd.NA or pd.isna(value):
        return None
    return float(value)


def _workflow_output_path(context: ExperimentStateContext, filename: str) -> str:
    if context.hierarchy_deferred or context.workflow is None:
        return ""
    return str((context.hierarchy_output_dir / filename).resolve())


def _count_equivalent_solver_success(frame: pd.DataFrame) -> int:
    if frame.empty or "solver_success" not in frame.columns:
        return 0
    return int(frame["solver_success"].fillna(False).astype(bool).sum())


def _record_int_or_zero(record: Mapping[str, Any], key: str) -> int:
    value = record.get(key, 0)
    if value is None or value is pd.NA or pd.isna(value):
        return 0
    return int(value)


def _needs_equivalent_reference_kb_prepass(
    config: BubbleEquivalentResistanceConfig | None,
) -> bool:
    if config is None:
        return False
    return (
        str(config.kb_mode or "fixed").strip() == "based_reference_section"
        and config.reference_kb_summary is None
    )


def _resolve_equivalent_reference_kb_summary(
    *,
    experiment_dir: Path,
    equivalent_resistance_config: BubbleEquivalentResistanceConfig,
    equivalent_forcing_path: str | Path | None,
    equivalent_forcing_config: ForcingConfig | None,
    equivalent_kb_model_path: str | Path | None,
    equivalent_kb_width_sample_field: str,
    equivalent_kb_width_percentile: float | None,
    equivalent_forcing_table_runner: Callable[..., tuple[pd.DataFrame, dict[str, object]]],
    equivalent_reference_kb_summary_runner: Callable[..., dict[str, Any]],
    reference_section_variant_builder: Callable[..., Mapping[str, Any]] | None = None,
) -> tuple[BubbleEquivalentResistanceConfig, dict[str, Any] | None, dict[str, object] | None]:
    if not _needs_equivalent_reference_kb_prepass(equivalent_resistance_config):
        return (
            equivalent_resistance_config,
            (
                dict(equivalent_resistance_config.reference_kb_summary)
                if equivalent_resistance_config.reference_kb_summary is not None
                else None
            ),
            None,
        )

    if equivalent_forcing_path is None:
        raise ValueError(
            "Equivalent-resistance kb_mode='based_reference_section' requires "
            "equivalent_forcing_path to be provided."
        )

    forcing_config = equivalent_forcing_config or ForcingConfig()
    forcing_table, forcing_metadata = equivalent_forcing_table_runner(
        equivalent_forcing_path,
        config=forcing_config,
    )
    reference_kwargs: dict[str, Any] = {
        "forcing": forcing_table,
        "width_sample_field": equivalent_kb_width_sample_field,
        "width_percentile": equivalent_kb_width_percentile,
        "model_path": equivalent_kb_model_path,
    }
    if reference_section_variant_builder is not None:
        reference_kwargs["variant_builder"] = reference_section_variant_builder
    reference_kb_summary = equivalent_reference_kb_summary_runner(
        experiment_dir,
        **reference_kwargs,
    )
    resolved_config = replace(
        equivalent_resistance_config,
        reference_kb_summary=dict(reference_kb_summary),
    )
    return resolved_config, dict(reference_kb_summary), dict(forcing_metadata)


def _match_equivalent_bubble_rows(
    parent_frame: pd.DataFrame,
    child_frame: pd.DataFrame,
) -> pd.DataFrame:
    parent = parent_frame.copy().reset_index(drop=True)
    child = child_frame.copy().reset_index(drop=True)
    if parent.empty or child.empty:
        return pd.DataFrame()

    parent["_parent_row_id"] = range(len(parent))
    child["_child_row_id"] = range(len(child))
    candidate_keys: list[tuple[str, tuple[str, ...], tuple[str, ...]]] = [
        (
            "matched_parent_upstream_boundary_node__matched_parent_downstream_boundary_node",
            ("upstream_boundary_node", "downstream_boundary_node"),
            ("matched_parent_upstream_boundary_node", "matched_parent_downstream_boundary_node"),
        ),
        (
            "upstream_boundary_node__downstream_boundary_node",
            ("upstream_boundary_node", "downstream_boundary_node"),
            ("upstream_boundary_node", "downstream_boundary_node"),
        ),
        ("bubble_root_unit_id", ("bubble_root_unit_id",), ("bubble_root_unit_id",)),
        ("compound_bubble_id", ("compound_bubble_id",), ("compound_bubble_id",)),
    ]

    matched_frames: list[pd.DataFrame] = []
    parent_remaining = parent.copy()
    child_remaining = child.copy()

    for match_method, parent_key_columns, child_key_columns in candidate_keys:
        if parent_remaining.empty or child_remaining.empty:
            break
        if not set(parent_key_columns).issubset(parent_remaining.columns) or not set(child_key_columns).issubset(child_remaining.columns):
            continue

        parent_unique = parent_remaining.dropna(subset=list(parent_key_columns)).copy()
        child_unique = child_remaining.dropna(subset=list(child_key_columns)).copy()
        if parent_unique.empty or child_unique.empty:
            continue

        parent_unique = parent_unique.loc[~parent_unique.duplicated(list(parent_key_columns), keep=False)].copy()
        child_unique = child_unique.loc[~child_unique.duplicated(list(child_key_columns), keep=False)].copy()
        if parent_unique.empty or child_unique.empty:
            continue

        matched = parent_unique.merge(
            child_unique,
            left_on=list(parent_key_columns),
            right_on=list(child_key_columns),
            how="inner",
            suffixes=("_before", "_after"),
        )
        if matched.empty:
            continue

        matched["match_method"] = match_method
        matched_frames.append(matched)
        parent_remaining = parent_remaining.loc[~parent_remaining["_parent_row_id"].isin(matched["_parent_row_id"])].copy()
        child_remaining = child_remaining.loc[~child_remaining["_child_row_id"].isin(matched["_child_row_id"])].copy()

    if not matched_frames:
        return pd.DataFrame()
    return pd.concat(matched_frames, ignore_index=True)


def _load_child_to_parent_node_id_map(child_context: ExperimentStateContext) -> dict[int, int]:
    variant_output_dir = child_context.variant_output_dir
    if variant_output_dir is None:
        return {}
    node_match_path = Path(variant_output_dir) / "matching" / "node_match.csv"
    if not node_match_path.exists():
        return {}

    node_match = pd.read_csv(node_match_path)
    required = {"child_id_node", "matched_parent_node_id"}
    if not required.issubset(node_match.columns):
        return {}

    child_ids = pd.to_numeric(node_match["child_id_node"], errors="coerce")
    parent_ids = pd.to_numeric(node_match["matched_parent_node_id"], errors="coerce")
    valid = child_ids.notna() & parent_ids.notna()
    if "match_within_tolerance" in node_match.columns:
        valid &= node_match["match_within_tolerance"].fillna(False).astype(bool)
    if not valid.any():
        return {}

    return dict(zip(child_ids.loc[valid].astype(int), parent_ids.loc[valid].astype(int)))


def _augment_child_equivalent_frame_for_parent_matching(
    child_frame: pd.DataFrame,
    child_context: ExperimentStateContext,
) -> pd.DataFrame:
    result = child_frame.copy()
    node_id_map = _load_child_to_parent_node_id_map(child_context)
    if not node_id_map:
        return result

    for boundary_column in ("upstream_boundary_node", "downstream_boundary_node"):
        if boundary_column not in result.columns:
            continue
        mapped_column = f"matched_parent_{boundary_column}"
        boundary_ids = pd.to_numeric(result[boundary_column], errors="coerce")
        result[mapped_column] = boundary_ids.map(node_id_map)
    return result


def _empty_equivalent_transition_frame() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "transition_id",
            "parent_state_id",
            "child_state_id",
            "match_method",
            "comparison_scope",
            "selected_domain_status",
            "selected_domain_error",
            "hydraulic_domain_type",
            "selected_unit_ids",
            "selected_unit_is_closed",
            "selected_unit_violating_link_ids",
            "parent_compound_bubble_id",
            "child_compound_bubble_id",
            "parent_bubble_root_unit_id",
            "child_bubble_root_unit_id",
            "parent_upstream_boundary_node",
            "parent_downstream_boundary_node",
            "child_upstream_boundary_node",
            "child_downstream_boundary_node",
            "child_upstream_anchor_match_method",
            "child_downstream_anchor_match_method",
            "parent_n_member_units",
            "child_n_member_units",
            "parent_domain_n_links",
            "child_domain_n_links",
            "parent_solver_success",
            "child_solver_success",
            "parent_solver_method",
            "child_solver_method",
            "parent_solver_backend_requested",
            "child_solver_backend_requested",
            "parent_solver_backend_resolved",
            "child_solver_backend_resolved",
            "parent_solver_nfev",
            "child_solver_nfev",
            "parent_solver_solution_stage",
            "child_solver_solution_stage",
            "parent_energy_stage_residual_relative",
            "child_energy_stage_residual_relative",
            "parent_polish_residual_relative",
            "child_polish_residual_relative",
            "parent_solver_cache_hit",
            "child_solver_cache_hit",
            "parent_solver_cache_key",
            "child_solver_cache_key",
            "parent_solver_cache_path",
            "child_solver_cache_path",
            "parent_solver_initialization",
            "child_solver_initialization",
            "parent_primary_solver_success",
            "child_primary_solver_success",
            "parent_primary_residual_relative",
            "child_primary_residual_relative",
            "K_before",
            "K_after",
            "G_before",
            "G_after",
            "Q_unit_before",
            "Q_unit_after",
            "residual_norm_before",
            "residual_norm_after",
            "residual_relative_before",
            "residual_relative_after",
            "Lambda_K",
            "Lambda_G",
            "percent_change_K",
            "percent_change_G",
        ]
    )


def _equivalent_transition_frame(records: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    if not records:
        return _empty_equivalent_transition_frame()
    return pd.DataFrame.from_records(records)


def _finite_comparison_statistic(
    frame: pd.DataFrame,
    column: str,
    statistic: str,
) -> float:
    if frame.empty or column not in frame.columns:
        return float("nan")
    values = pd.to_numeric(frame[column], errors="coerce").dropna()
    if values.empty:
        return float("nan")
    if statistic == "mean":
        return float(values.mean())
    if statistic == "median":
        return float(values.median())
    raise ValueError(f"Unsupported comparison statistic: {statistic}.")


def _selected_hydraulic_domain_comparison(
    *,
    transition_id: str,
    parent_context: ExperimentStateContext,
    child_context: ExperimentStateContext,
    equivalent_resistance_config: BubbleEquivalentResistanceConfig | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate the selected collapse in its smallest closed hydraulic domain."""
    selection = child_context.selection
    if (
        selection is None
        or parent_context.workflow is None
        or equivalent_resistance_config is None
        or parent_context.analysis_links_path is None
        or child_context.analysis_links_path is None
        or parent_context.variant_output_dir is None
        or child_context.variant_output_dir is None
    ):
        return pd.DataFrame(), pd.DataFrame()

    membership_path = (
        Path(parent_context.variant_output_dir)
        / "summary"
        / "unit_link_membership.csv"
    )
    node_match_path = (
        Path(child_context.variant_output_dir) / "matching" / "node_match.csv"
    )
    if not membership_path.exists() or not node_match_path.exists():
        return pd.DataFrame(), pd.DataFrame()

    parent_links = read_network(parent_context.analysis_links_path)
    child_links = read_network(child_context.analysis_links_path)
    parent_nodes = read_network(parent_context.reviewed_nodes_path)
    child_nodes = read_network(child_context.reviewed_nodes_path)
    membership = pd.read_csv(membership_path)

    parent_domain = select_smallest_closed_parent_domain(
        full_links=parent_links,
        unit_summary=parent_context.workflow.unit_summary,
        unit_link_membership=membership,
        bubble_edge_resistance=parent_context.workflow.bubble_edge_resistance,
        selected_unit_ids=selection.selected_unit_ids,
        selected_compound_bubble_id=selection.compound_bubble_id,
    )
    parent_to_child, anchor_match_methods = load_parent_to_child_anchor_map(
        node_match_path=node_match_path,
        child_links=child_links,
        link_lineage_path=(
            Path(child_context.variant_output_dir)
            / "matching"
            / "link_lineage.csv"
        ),
    )
    child_domain = build_matched_child_domain(
        parent_domain=parent_domain,
        child_links=child_links,
        parent_to_child_node=parent_to_child,
    )
    parent_solve, _ = solve_hydraulic_domain(
        domain=parent_domain,
        nodes=parent_nodes,
        config=equivalent_resistance_config,
    )
    child_solve, _ = solve_hydraulic_domain(
        domain=child_domain,
        nodes=child_nodes,
        config=equivalent_resistance_config,
    )

    parent_k = float(parent_solve.get("K_eq", np.nan))
    child_k = float(child_solve.get("K_eq", np.nan))
    comparison_metrics = {
        "Lambda_K": np.nan,
        "Lambda_G": np.nan,
        "percent_change_K": np.nan,
        "percent_change_G": np.nan,
    }
    if np.isfinite(parent_k) and parent_k > 0 and np.isfinite(child_k) and child_k > 0:
        resolved = compare_states(parent_k, child_k)
        comparison_metrics = {
            key: float(resolved[key])
            for key in comparison_metrics
        }

    compound_bubble_id = parent_domain.compound_bubble_id
    if compound_bubble_id is None:
        member_count = len(parent_domain.selected_unit_ids)
    else:
        compound_ids = pd.to_numeric(
            parent_context.workflow.unit_summary.get(
                "compound_bubble_id", pd.Series(dtype=float)
            ),
            errors="coerce",
        )
        member_count = int(compound_ids.eq(compound_bubble_id).sum())
    selected_unit_ids_text = ",".join(map(str, parent_domain.selected_unit_ids))
    violating_link_ids_text = ",".join(
        map(str, parent_domain.selected_unit_violating_link_ids)
    )
    record = {
        "transition_id": transition_id,
        "parent_state_id": parent_context.state_id,
        "child_state_id": child_context.state_id,
        "match_method": "smallest_closed_hydraulic_domain__matched_parent_anchors",
        "comparison_scope": "selected_hydraulic_domain",
        "selected_domain_status": "complete",
        "selected_domain_error": "",
        "hydraulic_domain_type": parent_domain.domain_type,
        "selected_unit_ids": selected_unit_ids_text,
        "selected_unit_is_closed": parent_domain.selected_unit_is_closed,
        "selected_unit_violating_link_ids": violating_link_ids_text,
        "parent_compound_bubble_id": compound_bubble_id,
        "child_compound_bubble_id": compound_bubble_id,
        "parent_bubble_root_unit_id": selection.bubble_root_unit_id,
        "child_bubble_root_unit_id": selection.bubble_root_unit_id,
        "parent_upstream_boundary_node": parent_domain.source_node,
        "parent_downstream_boundary_node": parent_domain.sink_node,
        "child_upstream_boundary_node": child_domain.source_node,
        "child_downstream_boundary_node": child_domain.sink_node,
        "child_upstream_anchor_match_method": anchor_match_methods.get(
            parent_domain.source_node, ""
        ),
        "child_downstream_anchor_match_method": anchor_match_methods.get(
            parent_domain.sink_node, ""
        ),
        "parent_n_member_units": member_count,
        "child_n_member_units": member_count,
        "parent_domain_n_links": len(parent_domain.links),
        "child_domain_n_links": len(child_domain.links),
        "parent_solver_success": bool(parent_solve.get("solver_success", False)),
        "child_solver_success": bool(child_solve.get("solver_success", False)),
        "parent_solver_method": parent_solve.get("solver_method", ""),
        "child_solver_method": child_solve.get("solver_method", ""),
        "parent_solver_backend_requested": parent_solve.get(
            "solver_backend_requested", equivalent_resistance_config.solver_backend
        ),
        "child_solver_backend_requested": child_solve.get(
            "solver_backend_requested", equivalent_resistance_config.solver_backend
        ),
        "parent_solver_backend_resolved": parent_solve.get(
            "solver_backend_resolved", ""
        ),
        "child_solver_backend_resolved": child_solve.get(
            "solver_backend_resolved", ""
        ),
        "parent_solver_nfev": parent_solve.get("solver_nfev", np.nan),
        "child_solver_nfev": child_solve.get("solver_nfev", np.nan),
        "parent_solver_solution_stage": parent_solve.get(
            "solver_solution_stage", ""
        ),
        "child_solver_solution_stage": child_solve.get(
            "solver_solution_stage", ""
        ),
        "parent_energy_stage_residual_relative": parent_solve.get(
            "energy_stage_residual_relative", np.nan
        ),
        "child_energy_stage_residual_relative": child_solve.get(
            "energy_stage_residual_relative", np.nan
        ),
        "parent_polish_residual_relative": parent_solve.get(
            "polish_residual_relative", np.nan
        ),
        "child_polish_residual_relative": child_solve.get(
            "polish_residual_relative", np.nan
        ),
        "parent_solver_cache_hit": bool(parent_solve.get("solver_cache_hit", False)),
        "child_solver_cache_hit": bool(child_solve.get("solver_cache_hit", False)),
        "parent_solver_cache_key": parent_solve.get("solver_cache_key", ""),
        "child_solver_cache_key": child_solve.get("solver_cache_key", ""),
        "parent_solver_cache_path": parent_solve.get("solver_cache_path", ""),
        "child_solver_cache_path": child_solve.get("solver_cache_path", ""),
        "parent_solver_initialization": parent_solve.get("initialization_used", ""),
        "child_solver_initialization": child_solve.get("initialization_used", ""),
        "parent_primary_solver_success": bool(
            parent_solve.get("primary_solver_success", False)
        ),
        "child_primary_solver_success": bool(
            child_solve.get("primary_solver_success", False)
        ),
        "parent_primary_residual_relative": parent_solve.get(
            "primary_residual_relative", np.nan
        ),
        "child_primary_residual_relative": child_solve.get(
            "primary_residual_relative", np.nan
        ),
        "K_before": parent_solve.get("K_eq", np.nan),
        "K_after": child_solve.get("K_eq", np.nan),
        "G_before": parent_solve.get("G_eq", np.nan),
        "G_after": child_solve.get("G_eq", np.nan),
        "Q_unit_before": parent_solve.get("Q_unit", np.nan),
        "Q_unit_after": child_solve.get("Q_unit", np.nan),
        "residual_norm_before": parent_solve.get("residual_norm", np.nan),
        "residual_norm_after": child_solve.get("residual_norm", np.nan),
        "residual_relative_before": parent_solve.get("residual_relative", np.nan),
        "residual_relative_after": child_solve.get("residual_relative", np.nan),
        **comparison_metrics,
    }

    link_audits: list[pd.DataFrame] = []
    for role, state_id, domain in (
        ("parent", parent_context.state_id, parent_domain),
        ("child", child_context.state_id, child_domain),
    ):
        available = [
            column
            for column in (
                "id_link",
                "id_us_node",
                "id_ds_node",
                "len",
                "wid_adj_routing",
                "wid_adj_wet",
                "wid_adj_total",
            )
            if column in domain.links.columns
        ]
        audit = domain.links[available].copy()
        audit.insert(0, "state_id", state_id)
        audit.insert(0, "domain_role", role)
        audit.insert(0, "transition_id", transition_id)
        link_audits.append(audit)
    return pd.DataFrame.from_records([record]), pd.concat(link_audits, ignore_index=True)


def _write_equivalent_transition_outputs(
    *,
    transition_id: str,
    parent_context: ExperimentStateContext,
    child_context: ExperimentStateContext,
    equivalent_resistance_config: BubbleEquivalentResistanceConfig | None = None,
) -> pd.DataFrame:
    parent_frame = parent_context.workflow.bubble_equivalent_metrics.copy()
    child_frame = _augment_child_equivalent_frame_for_parent_matching(
        child_context.workflow.bubble_equivalent_metrics.copy(),
        child_context,
    )
    output_dir = child_context.state_dir / "equivalent_resistance"
    output_dir.mkdir(parents=True, exist_ok=True)
    comparison_path = output_dir / "bubble_equivalent_state_comparison.csv"
    summary_path = output_dir / "bubble_equivalent_state_comparison_summary.json"

    child_context.equivalent_transition_output_dir = output_dir
    child_context.equivalent_transition_comparison_path = comparison_path
    child_context.equivalent_transition_summary_path = summary_path

    matched = _match_equivalent_bubble_rows(parent_frame, child_frame)
    comparison = _empty_equivalent_transition_frame()
    if not matched.empty:
        matched = matched.copy()
        for shared_column in (
            "compound_bubble_id",
            "bubble_root_unit_id",
            "upstream_boundary_node",
            "downstream_boundary_node",
        ):
            before_column = f"{shared_column}_before"
            after_column = f"{shared_column}_after"
            if shared_column in matched.columns:
                if before_column not in matched.columns:
                    matched[before_column] = matched[shared_column]
                if after_column not in matched.columns:
                    matched[after_column] = matched[shared_column]
        comparison = matched.rename(
            columns={
                "compound_bubble_id_before": "parent_compound_bubble_id",
                "compound_bubble_id_after": "child_compound_bubble_id",
                "bubble_root_unit_id_before": "parent_bubble_root_unit_id",
                "bubble_root_unit_id_after": "child_bubble_root_unit_id",
                "upstream_boundary_node_before": "parent_upstream_boundary_node",
                "upstream_boundary_node_after": "child_upstream_boundary_node",
                "downstream_boundary_node_before": "parent_downstream_boundary_node",
                "downstream_boundary_node_after": "child_downstream_boundary_node",
                "n_member_units_before": "parent_n_member_units",
                "n_member_units_after": "child_n_member_units",
                "solver_success_before": "parent_solver_success",
                "solver_success_after": "child_solver_success",
                "K_eq_before": "K_before",
                "K_eq_after": "K_after",
                "G_eq_before": "G_before",
                "G_eq_after": "G_after",
                "Q_unit_before": "Q_unit_before",
                "Q_unit_after": "Q_unit_after",
                "residual_norm_before": "residual_norm_before",
                "residual_norm_after": "residual_norm_after",
                "residual_relative_before": "residual_relative_before",
                "residual_relative_after": "residual_relative_after",
            }
        ).copy()
        comparison.insert(0, "transition_id", transition_id)
        comparison.insert(1, "parent_state_id", parent_context.state_id)
        comparison.insert(2, "child_state_id", child_context.state_id)
        comparison["Lambda_K"] = np.nan
        comparison["Lambda_G"] = np.nan
        comparison["percent_change_K"] = np.nan
        comparison["percent_change_G"] = np.nan
        valid_mask = (
            pd.to_numeric(comparison["K_before"], errors="coerce").gt(0)
            & pd.to_numeric(comparison["K_after"], errors="coerce").gt(0)
        )
        if valid_mask.any():
            metrics = compare_states(
                comparison.loc[valid_mask, "K_before"].to_numpy(dtype=float),
                comparison.loc[valid_mask, "K_after"].to_numpy(dtype=float),
            )
            comparison.loc[valid_mask, "Lambda_K"] = np.asarray(metrics["Lambda_K"], dtype=float)
            comparison.loc[valid_mask, "Lambda_G"] = np.asarray(metrics["Lambda_G"], dtype=float)
            comparison.loc[valid_mask, "percent_change_K"] = np.asarray(metrics["percent_change_K"], dtype=float)
            comparison.loc[valid_mask, "percent_change_G"] = np.asarray(metrics["percent_change_G"], dtype=float)
        comparison = comparison[
            [
                "transition_id",
                "parent_state_id",
                "child_state_id",
                "match_method",
                "parent_compound_bubble_id",
                "child_compound_bubble_id",
                "parent_bubble_root_unit_id",
                "child_bubble_root_unit_id",
                "parent_upstream_boundary_node",
                "parent_downstream_boundary_node",
                "child_upstream_boundary_node",
                "child_downstream_boundary_node",
                "parent_n_member_units",
                "child_n_member_units",
                "parent_solver_success",
                "child_solver_success",
                "K_before",
                "K_after",
                "G_before",
                "G_after",
                "Q_unit_before",
                "Q_unit_after",
                "residual_norm_before",
                "residual_norm_after",
                "residual_relative_before",
                "residual_relative_after",
                "Lambda_K",
                "Lambda_G",
                "percent_change_K",
                "percent_change_G",
            ]
        ].copy()

    redetected_comparison = comparison.copy()
    redetected_comparison_path = output_dir / "redetected_bubble_matches.csv"
    redetected_comparison.to_csv(redetected_comparison_path, index=False)
    selected_domain_error = ""
    try:
        selected_comparison, selected_link_audit = _selected_hydraulic_domain_comparison(
            transition_id=transition_id,
            parent_context=parent_context,
            child_context=child_context,
            equivalent_resistance_config=equivalent_resistance_config,
        )
    except Exception as exc:
        selected_domain_error = f"{type(exc).__name__}: {exc}"
        selected_comparison = pd.DataFrame.from_records(
            [
                {
                    "transition_id": transition_id,
                    "parent_state_id": parent_context.state_id,
                    "child_state_id": child_context.state_id,
                    "match_method": "smallest_closed_hydraulic_domain_failed",
                    "comparison_scope": "selected_hydraulic_domain",
                    "selected_domain_status": "failed",
                    "selected_domain_error": selected_domain_error,
                    "selected_unit_ids": (
                        ",".join(map(str, child_context.selection.selected_unit_ids))
                        if child_context.selection is not None
                        else ""
                    ),
                    "parent_solver_success": False,
                    "child_solver_success": False,
                    "K_before": np.nan,
                    "K_after": np.nan,
                    "G_before": np.nan,
                    "G_after": np.nan,
                    "Q_unit_before": np.nan,
                    "Q_unit_after": np.nan,
                    "residual_norm_before": np.nan,
                    "residual_norm_after": np.nan,
                    "residual_relative_before": np.nan,
                    "residual_relative_after": np.nan,
                    "Lambda_K": np.nan,
                    "Lambda_G": np.nan,
                    "percent_change_K": np.nan,
                    "percent_change_G": np.nan,
                }
            ]
        )
        selected_link_audit = pd.DataFrame()
    selected_link_audit_path: Path | None = None
    if not selected_comparison.empty:
        comparison = selected_comparison
        selected_link_audit_path = output_dir / "selected_hydraulic_domain_links.csv"
        selected_link_audit.to_csv(selected_link_audit_path, index=False)

    comparison.to_csv(comparison_path, index=False)
    summary = {
        "transition_id": transition_id,
        "parent_state_id": parent_context.state_id,
        "child_state_id": child_context.state_id,
        "comparison_csv": str(comparison_path.resolve()),
        "comparison_method": (
            "smallest_closed_hydraulic_domain"
            if not selected_comparison.empty
            else "redetected_bubble_matching_compatibility_fallback"
        ),
        "redetected_bubble_matches_csv": str(redetected_comparison_path.resolve()),
        "selected_hydraulic_domain_links_csv": (
            str(selected_link_audit_path.resolve())
            if selected_link_audit_path is not None
            else None
        ),
        "n_equivalent_bubbles_before": int(len(parent_frame)),
        "n_equivalent_bubbles_after": int(len(child_frame)),
        "n_equivalent_bubbles_matched": int(len(redetected_comparison)),
        "n_equivalent_bubbles_unmatched_before": int(max(len(parent_frame) - len(redetected_comparison), 0)),
        "n_equivalent_bubbles_unmatched_after": int(max(len(child_frame) - len(redetected_comparison), 0)),
        "n_selected_hydraulic_domain_comparisons": int(len(selected_comparison)),
        "selected_hydraulic_domain_status": (
            str(selected_comparison.iloc[0].get("selected_domain_status", "complete"))
            if not selected_comparison.empty
            else "not_available"
        ),
        "selected_hydraulic_domain_error": selected_domain_error or None,
        "n_equivalent_solver_success_before": _count_equivalent_solver_success(parent_frame),
        "n_equivalent_solver_success_after": _count_equivalent_solver_success(child_frame),
        "mean_lambda_k": _finite_comparison_statistic(comparison, "Lambda_K", "mean"),
        "median_lambda_k": _finite_comparison_statistic(comparison, "Lambda_K", "median"),
        "mean_percent_change_k": _finite_comparison_statistic(
            comparison, "percent_change_K", "mean"
        ),
        "median_percent_change_k": _finite_comparison_statistic(
            comparison, "percent_change_K", "median"
        ),
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    child_context.equivalent_transition_summary = summary
    return comparison


def _optimal_n_groups(group_count_summary: pd.DataFrame) -> int | None:
    if group_count_summary.empty or "is_optimal_n_groups" not in group_count_summary.columns:
        return None
    selected = group_count_summary.loc[group_count_summary["is_optimal_n_groups"]]
    if selected.empty:
        return None
    return int(selected["n_groups"].iloc[0])


def _select_unit_from_ranking(collapse_ranking: pd.DataFrame) -> ExperimentSelection | None:
    if collapse_ranking.empty or "unit_id" not in collapse_ranking.columns:
        return None

    ranking = collapse_ranking.copy()
    if "collapse_order_global" in ranking.columns:
        ranking = ranking.sort_values("collapse_order_global", kind="mergesort")
    row = ranking.iloc[0]
    unit_id = int(row["unit_id"])
    collapse_order = _optional_int(row.get("collapse_order_global"))
    return ExperimentSelection(
        selection_type="unit",
        selected_unit_ids=[unit_id],
        selected_group_label=None,
        selected_group_size=1,
        selected_rank_start=collapse_order,
        selected_rank_end=collapse_order,
        selected_collapse_order=collapse_order,
        selected_collapse_priority_score=_optional_float(row.get("collapse_priority_score")),
        state_suffix=f"unit_{unit_id}",
    )


def _select_all_units_from_ranking(
    collapse_ranking: pd.DataFrame,
    unit_metrics: pd.DataFrame,
) -> list[ExperimentSelection]:
    if not collapse_ranking.empty and "unit_id" in collapse_ranking.columns:
        ranking = collapse_ranking.copy()
        if "collapse_order_global" in ranking.columns:
            ranking = ranking.sort_values("collapse_order_global", kind="mergesort")
        return [
            ExperimentSelection(
                selection_type="unit",
                selected_unit_ids=[int(row["unit_id"])],
                selected_group_label=None,
                selected_group_size=1,
                selected_rank_start=_optional_int(row.get("collapse_order_global")),
                selected_rank_end=_optional_int(row.get("collapse_order_global")),
                selected_collapse_order=_optional_int(row.get("collapse_order_global")),
                selected_collapse_priority_score=_optional_float(row.get("collapse_priority_score")),
                state_suffix=f"unit_{int(row['unit_id'])}",
            )
            for _, row in ranking.iterrows()
        ]

    if unit_metrics.empty or "unit_id" not in unit_metrics.columns:
        return []

    ordered_unit_ids = sorted(int(unit_id) for unit_id in unit_metrics["unit_id"].tolist())
    return [
        ExperimentSelection(
            selection_type="unit",
            selected_unit_ids=[unit_id],
            selected_group_label=None,
            selected_group_size=1,
            selected_rank_start=None,
            selected_rank_end=None,
            selected_collapse_order=None,
            selected_collapse_priority_score=None,
            state_suffix=f"unit_{unit_id}",
        )
        for unit_id in ordered_unit_ids
    ]


def _select_compound_bubbles_from_workflow(workflow: UnitWorkflowOutputs) -> list[ExperimentSelection]:
    summary = workflow.unit_summary.copy()
    if summary.empty:
        return []

    fallback_columns = [
        column
        for column in (
            "unit_id",
            "compound_bubble_id",
            "in_compound_bubble",
            "compound_bubble_role",
            "unit_node_count",
        )
        if column in workflow.unit_metrics.columns and column not in summary.columns
    ]
    if fallback_columns:
        summary = summary.merge(
            workflow.unit_metrics[["unit_id"] + fallback_columns],
            on="unit_id",
            how="left",
            validate="1:1",
        )

    required = {"unit_id", "compound_bubble_id", "compound_bubble_role"}
    if not required.issubset(summary.columns):
        return []

    working = summary.copy()
    working["unit_id"] = pd.to_numeric(working["unit_id"], errors="coerce").astype("Int64")
    working["compound_bubble_id"] = pd.to_numeric(working["compound_bubble_id"], errors="coerce").astype("Int64")
    if "unit_node_count" in working.columns:
        working["unit_node_count"] = pd.to_numeric(working["unit_node_count"], errors="coerce")
    else:
        working["unit_node_count"] = pd.Series([pd.NA] * len(working), dtype="Float64")

    mask = (
        working["compound_bubble_id"].notna()
        & working["compound_bubble_role"].isin(["bubble_root", "bubble_member"])
    )
    working = working.loc[mask].copy()
    if working.empty:
        return []

    selections: list[ExperimentSelection] = []
    grouped = working.groupby("compound_bubble_id", sort=True)
    for bubble_id, group in grouped:
        unit_ids = sorted(int(unit_id) for unit_id in group["unit_id"].dropna().astype(int).tolist())
        if len(unit_ids) < 2:
            continue
        bubble_root_rows = group.loc[group["compound_bubble_role"] == "bubble_root", "unit_id"]
        if bubble_root_rows.empty:
            bubble_root_unit_id = min(unit_ids)
        else:
            bubble_root_unit_id = int(pd.to_numeric(bubble_root_rows, errors="coerce").dropna().astype(int).min())
        total_unit_nodes = (
            int(group["unit_node_count"].fillna(0).sum())
            if "unit_node_count" in group.columns
            else None
        )
        bubble_numeric = int(bubble_id)
        selections.append(
            ExperimentSelection(
                selection_type="compound_bubble",
                selected_unit_ids=unit_ids,
                selected_group_label=None,
                selected_group_size=len(unit_ids),
                selected_rank_start=None,
                selected_rank_end=None,
                selected_collapse_order=None,
                selected_collapse_priority_score=None,
                state_suffix=f"bubble_{bubble_numeric}",
                compound_bubble_id=bubble_numeric,
                bubble_root_unit_id=bubble_root_unit_id,
                total_unit_nodes_selected=total_unit_nodes,
            )
        )

    selections.sort(
        key=lambda selection: (
            -int(selection.selected_group_size),
            -int(selection.total_unit_nodes_selected or 0),
            int(selection.compound_bubble_id or 0),
        )
    )
    return selections


def _select_group_from_selected_groups(selected_groups: pd.DataFrame) -> ExperimentSelection | None:
    if selected_groups.empty or "group_label" not in selected_groups.columns or "unit_ids" not in selected_groups.columns:
        return None

    groups = selected_groups.copy()
    sort_columns = [column for column in ("rank_start", "group_index") if column in groups.columns]
    if sort_columns:
        groups = groups.sort_values(sort_columns, kind="mergesort")
    row = groups.iloc[0]
    unit_ids = _parse_int_list(row["unit_ids"])
    group_label = str(row["group_label"])
    return ExperimentSelection(
        selection_type="group",
        selected_unit_ids=unit_ids,
        selected_group_label=group_label,
        selected_group_size=len(unit_ids),
        selected_rank_start=_optional_int(row.get("rank_start")),
        selected_rank_end=_optional_int(row.get("rank_end")),
        selected_collapse_order=_optional_int(row.get("rank_start")),
        selected_collapse_priority_score=None,
        state_suffix=f"group_{_sanitize_label(group_label)}",
    )


def _state_record(context: ExperimentStateContext, mode: str) -> dict[str, Any]:
    selection = context.selection
    equivalent_transition_summary = context.equivalent_transition_summary or {}
    workflow = context.workflow
    return {
        "state_id": context.state_id,
        "parent_state_id": context.parent_state_id or "",
        "depth": int(context.depth),
        "mode": mode,
        "state_role": "base" if context.depth == 0 else "derived",
        "selection_type": selection.selection_type if selection is not None else "",
        "collapsed_selection_label": _selection_label(selection),
        "selected_unit_ids": _serialize_int_list(selection.selected_unit_ids) if selection is not None else "",
        "selected_group_label": selection.selected_group_label or "" if selection is not None else "",
        "selected_group_size": int(selection.selected_group_size) if selection is not None else 0,
        "selected_rank_start": selection.selected_rank_start if selection is not None else pd.NA,
        "selected_rank_end": selection.selected_rank_end if selection is not None else pd.NA,
        "selected_collapse_order": selection.selected_collapse_order if selection is not None else pd.NA,
        "selected_collapse_priority_score": selection.selected_collapse_priority_score if selection is not None else float("nan"),
        "is_compound_state": bool(selection.is_compound_state) if selection is not None else False,
        "compound_bubble_id": selection.compound_bubble_id if selection is not None else pd.NA,
        "bubble_root_unit_id": selection.bubble_root_unit_id if selection is not None else pd.NA,
        "total_unit_nodes_selected": selection.total_unit_nodes_selected if selection is not None else pd.NA,
        "state_dir": str(context.state_dir.resolve()),
        "variant_output_dir": str(context.variant_output_dir.resolve()) if context.variant_output_dir is not None else "",
        "hydraulic_context_output_dir": (
            str(context.hydraulic_context_output_dir.resolve())
            if context.hydraulic_context_output_dir is not None
            else ""
        ),
        "hierarchy_output_dir": str(context.hierarchy_output_dir.resolve()),
        "hierarchy_status": "deferred" if context.hierarchy_deferred else "complete",
        "bubble_equivalent_metrics_path": _workflow_output_path(context, "bubble_equivalent_metrics.csv"),
        "bubble_edge_resistance_path": _workflow_output_path(context, "bubble_edge_resistance.csv"),
        "bubble_node_heads_path": _workflow_output_path(context, "bubble_node_heads.csv"),
        "equivalent_transition_output_dir": (
            str(context.equivalent_transition_output_dir.resolve())
            if context.equivalent_transition_output_dir is not None
            else ""
        ),
        "equivalent_transition_comparison_csv": (
            str(context.equivalent_transition_comparison_path.resolve())
            if context.equivalent_transition_comparison_path is not None
            else ""
        ),
        "equivalent_transition_summary_json": (
            str(context.equivalent_transition_summary_path.resolve())
            if context.equivalent_transition_summary_path is not None
            else ""
        ),
        "cleaned_mask_path": str(context.cleaned_mask_path.resolve()),
        "directed_links_path": str(context.reviewed_links_path.resolve()),
        "analysis_links_path": str(
            (context.analysis_links_path or context.reviewed_links_path).resolve()
        ),
        "directed_nodes_path": str(context.reviewed_nodes_path.resolve()),
        "n_units_detected": int(len(workflow.unit_metrics)) if workflow is not None else pd.NA,
        "n_paths_detected": int(len(workflow.path_metrics)) if workflow is not None else pd.NA,
        "n_equivalent_bubbles": int(len(workflow.bubble_equivalent_metrics)) if workflow is not None else pd.NA,
        "n_equivalent_bubbles_solver_success": _count_equivalent_solver_success(workflow.bubble_equivalent_metrics) if workflow is not None else pd.NA,
        "n_equivalent_bubbles_solver_cache_hits": (
            int(
                workflow.bubble_equivalent_metrics.get(
                    "solver_cache_hit", pd.Series(dtype=bool)
                ).fillna(False).astype(bool).sum()
            )
            if workflow is not None
            else pd.NA
        ),
        "n_equivalent_bubbles_matched_to_parent": equivalent_transition_summary.get("n_equivalent_bubbles_matched", pd.NA),
        "n_selected_groups": int(len(workflow.selected_groups)) if workflow is not None else pd.NA,
        "optimal_n_groups": _optimal_n_groups(workflow.group_count_summary) if workflow is not None else pd.NA,
        "status": "hydraulic_complete_hierarchy_deferred" if context.hierarchy_deferred else "complete",
        "error_stage": "",
        "error": "",
    }


def _transition_record(
    *,
    transition_id: str,
    step_index: int,
    parent_context: ExperimentStateContext,
    child_context: ExperimentStateContext,
) -> dict[str, Any]:
    selection = child_context.selection
    assert selection is not None
    equivalent_transition_summary = child_context.equivalent_transition_summary or {}
    parent_workflow = parent_context.workflow
    child_workflow = child_context.workflow
    return {
        "transition_id": transition_id,
        "step_index": int(step_index),
        "parent_state_id": parent_context.state_id,
        "child_state_id": child_context.state_id,
        "depth": int(child_context.depth),
        "mode": "",
        "selection_type": selection.selection_type,
        "collapsed_selection_label": _selection_label(selection),
        "selected_unit_ids": _serialize_int_list(selection.selected_unit_ids),
        "selected_group_label": selection.selected_group_label or "",
        "selected_group_size": int(selection.selected_group_size),
        "selected_rank_start": selection.selected_rank_start,
        "selected_rank_end": selection.selected_rank_end,
        "selected_collapse_order": selection.selected_collapse_order,
        "selected_collapse_priority_score": selection.selected_collapse_priority_score,
        "is_compound_state": bool(selection.is_compound_state),
        "compound_bubble_id": selection.compound_bubble_id if selection is not None else pd.NA,
        "bubble_root_unit_id": selection.bubble_root_unit_id if selection is not None else pd.NA,
        "total_unit_nodes_selected": selection.total_unit_nodes_selected if selection is not None else pd.NA,
        "n_units_before": int(len(parent_workflow.unit_metrics)) if parent_workflow is not None else pd.NA,
        "n_units_after": int(len(child_workflow.unit_metrics)) if child_workflow is not None else pd.NA,
        "n_equivalent_bubbles_before": int(len(parent_workflow.bubble_equivalent_metrics)) if parent_workflow is not None else pd.NA,
        "n_equivalent_bubbles_after": int(len(child_workflow.bubble_equivalent_metrics)) if child_workflow is not None else pd.NA,
        "n_equivalent_bubbles_matched": equivalent_transition_summary.get("n_equivalent_bubbles_matched", 0),
        "n_equivalent_bubbles_unmatched_before": equivalent_transition_summary.get("n_equivalent_bubbles_unmatched_before", 0),
        "n_equivalent_bubbles_unmatched_after": equivalent_transition_summary.get("n_equivalent_bubbles_unmatched_after", 0),
        "mean_lambda_k": equivalent_transition_summary.get("mean_lambda_k", float("nan")),
        "median_lambda_k": equivalent_transition_summary.get("median_lambda_k", float("nan")),
        "mean_percent_change_k": equivalent_transition_summary.get("mean_percent_change_k", float("nan")),
        "median_percent_change_k": equivalent_transition_summary.get("median_percent_change_k", float("nan")),
        "parent_hierarchy_output_dir": str(parent_context.hierarchy_output_dir.resolve()),
        "child_hierarchy_output_dir": str(child_context.hierarchy_output_dir.resolve()),
        "child_hierarchy_status": "deferred" if child_context.hierarchy_deferred else "complete",
        "parent_bubble_equivalent_metrics_csv": _workflow_output_path(parent_context, "bubble_equivalent_metrics.csv"),
        "child_bubble_equivalent_metrics_csv": _workflow_output_path(child_context, "bubble_equivalent_metrics.csv"),
        "equivalent_transition_comparison_csv": (
            str(child_context.equivalent_transition_comparison_path.resolve())
            if child_context.equivalent_transition_comparison_path is not None
            else ""
        ),
        "equivalent_transition_summary_json": (
            str(child_context.equivalent_transition_summary_path.resolve())
            if child_context.equivalent_transition_summary_path is not None
            else ""
        ),
        "child_variant_output_dir": str(child_context.variant_output_dir.resolve()) if child_context.variant_output_dir is not None else "",
        "status": "hydraulic_complete_hierarchy_deferred" if child_context.hierarchy_deferred else "complete",
        "error_stage": "",
        "error": "",
    }


def _write_state_failure_artifact(state_dir: Path, payload: Mapping[str, Any]) -> Path:
    state_dir.mkdir(parents=True, exist_ok=True)
    path = state_dir / "state_failure.json"
    path.write_text(json.dumps(dict(payload), indent=2), encoding="utf-8")
    return path


def _ignore_missing_rmtree_entry(
    function: Callable[..., Any],
    path: str,
    exc: BaseException,
) -> None:
    """Ignore an entry that vanished during recursive cleanup, and nothing else."""
    del function, path
    if isinstance(exc, FileNotFoundError):
        return
    raise exc


def _remove_directory_tolerating_missing_entries(path: Path) -> None:
    if not path.exists():
        return
    try:
        shutil.rmtree(path, onexc=_ignore_missing_rmtree_entry)
    except FileNotFoundError:
        # The root itself may disappear between exists() and rmtree().
        pass


def _reset_state_dir(state_dir: Path) -> None:
    _remove_directory_tolerating_missing_entries(state_dir)
    state_dir.mkdir(parents=True, exist_ok=True)


def _reset_experiment_output_dir(output_dir: Path) -> None:
    _remove_directory_tolerating_missing_entries(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)


def _format_exception(exc: BaseException) -> str:
    message = str(exc).strip()
    if message:
        return f"{type(exc).__name__}: {message}"
    return type(exc).__name__


def _format_traceback(exc: BaseException) -> str:
    return "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)).strip()


def _failed_state_record(
    *,
    state_id: str,
    parent_state_id: str | None,
    depth: int,
    mode: str,
    state_dir: Path,
    selection: ExperimentSelection,
    error_stage: str,
    error: str,
    error_traceback: str = "",
    variant_outputs: NetworkVariantOutputs | None = None,
) -> dict[str, Any]:
    direction_report = ""
    if variant_outputs is not None:
        direction_report = str(
            Path(
                variant_outputs.manifest.get("output_paths", {}).get("direction_validation_report", "")
            ).resolve()
        ) if variant_outputs.manifest.get("output_paths", {}).get("direction_validation_report") else ""
    failure_payload = {
        "state_id": state_id,
        "parent_state_id": parent_state_id or "",
        "mode": mode,
        "selection_type": selection.selection_type,
        "selected_unit_ids": selection.selected_unit_ids,
        "selected_group_label": selection.selected_group_label,
        "error_stage": error_stage,
        "error": error,
        "error_traceback": error_traceback,
        "variant_output_dir": (
            str(variant_outputs.output_dir.resolve()) if variant_outputs is not None else ""
        ),
        "direction_validation_report": direction_report,
    }
    failure_artifact = _write_state_failure_artifact(state_dir, failure_payload)
    return {
        "state_id": state_id,
        "parent_state_id": parent_state_id or "",
        "depth": int(depth),
        "mode": mode,
        "state_role": "base" if depth == 0 else "derived",
        "selection_type": selection.selection_type,
        "collapsed_selection_label": _selection_label(selection),
        "selected_unit_ids": _serialize_int_list(selection.selected_unit_ids),
        "selected_group_label": selection.selected_group_label or "",
        "selected_group_size": int(selection.selected_group_size),
        "selected_rank_start": selection.selected_rank_start,
        "selected_rank_end": selection.selected_rank_end,
        "selected_collapse_order": selection.selected_collapse_order,
        "selected_collapse_priority_score": selection.selected_collapse_priority_score,
        "is_compound_state": bool(selection.is_compound_state),
        "compound_bubble_id": selection.compound_bubble_id if selection is not None else pd.NA,
        "bubble_root_unit_id": selection.bubble_root_unit_id if selection is not None else pd.NA,
        "total_unit_nodes_selected": selection.total_unit_nodes_selected if selection is not None else pd.NA,
        "state_dir": str(state_dir.resolve()),
        "variant_output_dir": str(variant_outputs.output_dir.resolve()) if variant_outputs is not None else "",
        "hierarchy_output_dir": "",
        "cleaned_mask_path": (
            str(variant_outputs.collapsed_mask_path.resolve()) if variant_outputs is not None else ""
        ),
        "directed_links_path": (
            str(variant_outputs.directed_links_path.resolve())
            if variant_outputs is not None and variant_outputs.directed_links_path is not None
            else ""
        ),
        "directed_nodes_path": (
            str(variant_outputs.directed_nodes_path.resolve())
            if variant_outputs is not None and variant_outputs.directed_nodes_path is not None
            else ""
        ),
        "direction_validation_report": direction_report,
        "bubble_equivalent_metrics_path": "",
        "bubble_edge_resistance_path": "",
        "bubble_node_heads_path": "",
        "equivalent_transition_output_dir": "",
        "equivalent_transition_comparison_csv": "",
        "equivalent_transition_summary_json": "",
        "n_units_detected": pd.NA,
        "n_paths_detected": pd.NA,
        "n_equivalent_bubbles": pd.NA,
        "n_equivalent_bubbles_solver_success": pd.NA,
        "n_equivalent_bubbles_matched_to_parent": pd.NA,
        "n_selected_groups": pd.NA,
        "optimal_n_groups": pd.NA,
        "status": "failed",
        "error_stage": error_stage,
        "error": error,
        "error_traceback": error_traceback,
        "failure_artifact_path": str(failure_artifact.resolve()),
    }


def _failed_transition_record(
    *,
    transition_id: str,
    step_index: int,
    parent_context: ExperimentStateContext,
    selection: ExperimentSelection,
    child_state_id: str,
    error_stage: str,
    error: str,
    error_traceback: str = "",
    variant_outputs: NetworkVariantOutputs | None = None,
) -> dict[str, Any]:
    return {
        "transition_id": transition_id,
        "step_index": int(step_index),
        "parent_state_id": parent_context.state_id,
        "child_state_id": child_state_id,
        "depth": int(parent_context.depth + 1),
        "mode": "",
        "selection_type": selection.selection_type,
        "collapsed_selection_label": _selection_label(selection),
        "selected_unit_ids": _serialize_int_list(selection.selected_unit_ids),
        "selected_group_label": selection.selected_group_label or "",
        "selected_group_size": int(selection.selected_group_size),
        "selected_rank_start": selection.selected_rank_start,
        "selected_rank_end": selection.selected_rank_end,
        "selected_collapse_order": selection.selected_collapse_order,
        "selected_collapse_priority_score": selection.selected_collapse_priority_score,
        "is_compound_state": bool(selection.is_compound_state),
        "compound_bubble_id": selection.compound_bubble_id if selection is not None else pd.NA,
        "bubble_root_unit_id": selection.bubble_root_unit_id if selection is not None else pd.NA,
        "total_unit_nodes_selected": selection.total_unit_nodes_selected if selection is not None else pd.NA,
        "n_units_before": int(len(parent_context.workflow.unit_metrics)),
        "n_units_after": pd.NA,
        "n_equivalent_bubbles_before": int(len(parent_context.workflow.bubble_equivalent_metrics)),
        "n_equivalent_bubbles_after": pd.NA,
        "n_equivalent_bubbles_matched": pd.NA,
        "n_equivalent_bubbles_unmatched_before": pd.NA,
        "n_equivalent_bubbles_unmatched_after": pd.NA,
        "mean_lambda_k": float("nan"),
        "median_lambda_k": float("nan"),
        "mean_percent_change_k": float("nan"),
        "median_percent_change_k": float("nan"),
        "parent_hierarchy_output_dir": str(parent_context.hierarchy_output_dir.resolve()),
        "child_hierarchy_output_dir": "",
        "parent_bubble_equivalent_metrics_csv": _workflow_output_path(parent_context, "bubble_equivalent_metrics.csv"),
        "child_bubble_equivalent_metrics_csv": "",
        "equivalent_transition_comparison_csv": "",
        "equivalent_transition_summary_json": "",
        "child_variant_output_dir": (
            str(variant_outputs.output_dir.resolve()) if variant_outputs is not None else ""
        ),
        "status": "failed",
        "error_stage": error_stage,
        "error": error,
        "error_traceback": error_traceback,
    }


def _write_experiment_outputs(
    output_dir: Path,
    *,
    state_records: Sequence[Mapping[str, Any]],
    transition_records: Sequence[Mapping[str, Any]],
    equivalent_transition_records: Sequence[Mapping[str, Any]],
    manifest: Mapping[str, Any],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame.from_records(state_records).to_csv(output_dir / "state_registry.csv", index=False)
    pd.DataFrame.from_records(transition_records).to_csv(output_dir / "transition_registry.csv", index=False)
    _equivalent_transition_frame(equivalent_transition_records).to_csv(
        output_dir / "equivalent_transition_registry.csv",
        index=False,
    )
    with (output_dir / "experiment_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(dict(manifest), handle, indent=2)


def _run_state_hierarchy(
    *,
    state_id: str,
    parent_state_id: str | None,
    depth: int,
    state_dir: Path,
    cleaned_mask_path: str | Path,
    reviewed_links_path: str | Path,
    reviewed_nodes_path: str | Path,
    directed_links_path: str | Path | None = None,
    selection: ExperimentSelection | None,
    unit_workflow_runner: Callable[..., UnitWorkflowOutputs],
    unit_workflow_writer: Callable[..., None],
    unit_workflow_kwargs: Mapping[str, Any],
) -> ExperimentStateContext:
    hierarchy_dir = state_dir / "hierarchy"
    workflow = unit_workflow_runner(
        reviewed_links_path,
        reviewed_nodes_path,
        **dict(unit_workflow_kwargs),
    )
    unit_workflow_writer(
        hierarchy_dir,
        workflow,
        manifest_overrides={
            "state_id": state_id,
            "parent_state_id": parent_state_id,
            "depth": int(depth),
        },
    )
    return ExperimentStateContext(
        state_id=state_id,
        parent_state_id=parent_state_id,
        depth=depth,
        state_dir=state_dir,
        hierarchy_output_dir=hierarchy_dir,
        cleaned_mask_path=Path(cleaned_mask_path),
        reviewed_links_path=Path(directed_links_path or reviewed_links_path),
        reviewed_nodes_path=Path(reviewed_nodes_path),
        workflow=workflow,
        analysis_links_path=Path(reviewed_links_path),
        selection=selection,
    )


_UNIT_WORKFLOW_FRAME_FILES: dict[str, str] = {
    "unit_summary": "unit_summary.csv",
    "unit_metrics": "unit_metrics.csv",
    "path_metrics": "path_metrics.csv",
    "hierarchy_level_metrics": "hierarchy_level_metrics.csv",
    "collapse_ranking": "collapse_ranking.csv",
    "merge_tree": "merge_tree.csv",
    "bubble_summary": "bubble_summary.csv",
    "bubble_equivalent_metrics": "bubble_equivalent_metrics.csv",
    "bubble_edge_resistance": "bubble_edge_resistance.csv",
    "bubble_node_heads": "bubble_node_heads.csv",
    "group_count_summary": "group_count_summary.csv",
    "selected_groups": "selected_groups.csv",
}


def _load_unit_workflow_outputs(output_dir: str | Path) -> UnitWorkflowOutputs:
    """Load a previously completed hierarchy bundle without recomputation."""
    hierarchy_dir = Path(output_dir).resolve()
    missing = [
        hierarchy_dir / filename
        for filename in _UNIT_WORKFLOW_FRAME_FILES.values()
        if not (hierarchy_dir / filename).exists()
    ]
    manifest_path = hierarchy_dir / "unit_workflow_manifest.json"
    if not manifest_path.exists():
        missing.append(manifest_path)
    if missing:
        raise FileNotFoundError(
            "Completed base hierarchy reuse is missing required artifacts: "
            + ", ".join(str(path) for path in missing)
        )
    frames: dict[str, pd.DataFrame] = {}
    for field, filename in _UNIT_WORKFLOW_FRAME_FILES.items():
        try:
            frames[field] = pd.read_csv(hierarchy_dir / filename)
        except pd.errors.EmptyDataError:
            # A few optional workflow products are legitimately empty and
            # older writers represented those as a zero-byte CSV. Preserve
            # that completed state instead of forcing an expensive rebuild.
            frames[field] = pd.DataFrame()
    return UnitWorkflowOutputs(**frames)


def _hierarchy_uses_current_directed_graph_schema(source_variant_dir: str | Path) -> bool:
    manifest_path = (
        Path(source_variant_dir).resolve().parent
        / "hierarchy"
        / "unit_workflow_manifest.json"
    )
    if not manifest_path.exists():
        return False
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    return manifest.get("directed_graph_schema") == DIRECTED_GRAPH_SCHEMA_VERSION


def _copy_completed_base_state_products(
    *,
    source_variant_dir: str | Path,
    destination_state_dir: Path,
) -> UnitWorkflowOutputs:
    """Copy the complete hierarchy/hydraulic base-state contract."""
    source_state_dir = Path(source_variant_dir).resolve().parent
    source_hierarchy = source_state_dir / "hierarchy"
    source_hydraulic = source_state_dir / "hydraulic_context"
    missing = _missing_completed_base_state_artifacts(source_variant_dir)
    if missing:
        raise FileNotFoundError(
            "Base-state reuse source is incomplete; refusing partial reuse: "
            + ", ".join(str(path) for path in missing)
        )

    workflow = _load_unit_workflow_outputs(source_hierarchy)
    _copytree_transactional(source_hierarchy, destination_state_dir / "hierarchy")
    _copytree_transactional(
        source_hydraulic,
        destination_state_dir / "hydraulic_context",
    )
    return workflow


def _missing_completed_base_state_artifacts(
    source_variant_dir: str | Path,
) -> list[Path]:
    source_state_dir = Path(source_variant_dir).resolve().parent
    source_hierarchy = source_state_dir / "hierarchy"
    source_hydraulic = source_state_dir / "hydraulic_context"
    required = (
        source_state_dir / "variant",
        source_hierarchy / "unit_workflow_manifest.json",
        *(source_hierarchy / filename for filename in _UNIT_WORKFLOW_FRAME_FILES.values()),
        source_hydraulic / "link_hydraulic_context.csv",
        source_hydraulic / "node_hydraulic_context.csv",
        source_hydraulic / "hydraulic_context_manifest.json",
    )
    return [path for path in required if not path.exists()]


def _copytree_transactional(source: Path, destination: Path) -> None:
    """Copy one artifact tree without exposing a partially copied product."""
    source = Path(source).resolve()
    destination = Path(destination).resolve()
    staging = destination.parent / f".{destination.name}.base-reuse-staging"
    if staging.exists():
        shutil.rmtree(staging)
    try:
        shutil.copytree(
            source,
            staging,
            ignore=shutil.ignore_patterns("._*"),
        )
        if destination.exists():
            shutil.rmtree(destination)
        staging.replace(destination)
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise


def _localize_json_value(
    value: Any,
    *,
    source_experiment_dir: Path,
    destination_experiment_dir: Path,
) -> Any:
    if isinstance(value, dict):
        return {
            key: _localize_json_value(
                item,
                source_experiment_dir=source_experiment_dir,
                destination_experiment_dir=destination_experiment_dir,
            )
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [
            _localize_json_value(
                item,
                source_experiment_dir=source_experiment_dir,
                destination_experiment_dir=destination_experiment_dir,
            )
            for item in value
        ]
    if not isinstance(value, str):
        return value
    try:
        relative = Path(value).resolve().relative_to(source_experiment_dir)
    except (OSError, ValueError):
        return value
    return str((destination_experiment_dir / relative).resolve())


def _localize_copied_base_manifests(
    *,
    source_experiment_dir: Path,
    destination_experiment_dir: Path,
    destination_state_dir: Path,
) -> list[str]:
    """Rewrite copied internal paths while retaining external provenance paths."""
    localized: list[str] = []
    roots = [
        destination_state_dir,
        destination_experiment_dir / "reference_section",
        destination_experiment_dir / "equivalent_resistance" / "solve_cache",
        destination_experiment_dir / "boundary_components",
    ]
    seen: set[Path] = set()
    for root in roots:
        if not root.exists():
            continue
        for path in sorted(root.rglob("*.json")):
            resolved = path.resolve()
            if resolved in seen or path.name.startswith("._"):
                continue
            seen.add(resolved)
            payload = json.loads(path.read_text(encoding="utf-8"))
            rewritten = _localize_json_value(
                payload,
                source_experiment_dir=source_experiment_dir,
                destination_experiment_dir=destination_experiment_dir,
            )
            if rewritten != payload:
                path.write_text(json.dumps(rewritten, indent=2) + "\n", encoding="utf-8")
                localized.append(str(path.resolve()))
    return localized


def _artifact_inventory(root: Path) -> dict[str, int]:
    return {
        str(path.relative_to(root)): int(path.stat().st_size)
        for path in sorted(root.rglob("*"))
        if path.is_file()
        and not path.name.startswith("._")
        and path.relative_to(root).parts[0]
        in {"variant", "hierarchy", "hydraulic_context"}
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_base_reuse_audit(
    *,
    source_experiment_dir: Path,
    destination_experiment_dir: Path,
    destination_state_dir: Path,
    localized_manifests: Sequence[str],
    source_manifest: Mapping[str, Any] | None,
) -> Path:
    source_state = source_experiment_dir / "states" / "S000_base"
    source_inventory = _artifact_inventory(source_state)
    destination_inventory = _artifact_inventory(destination_state_dir)
    missing = sorted(set(source_inventory).difference(destination_inventory))
    size_mismatches = sorted(
        path
        for path in set(source_inventory).intersection(destination_inventory)
        if not path.endswith(".json")
        and source_inventory[path] != destination_inventory[path]
    )
    content_mismatches = sorted(
        relative
        for relative in set(source_inventory).intersection(destination_inventory)
        if not relative.endswith(".json")
        and relative not in size_mismatches
        and _sha256(source_state / relative)
        != _sha256(destination_state_dir / relative)
    )
    if missing or size_mismatches or content_mismatches:
        raise ValueError(
            "Copied base-state inventory validation failed: "
            f"missing={missing}, size_mismatches={size_mismatches}, "
            f"content_mismatches={content_mismatches}."
        )
    audit_path = destination_state_dir / "base_state_reuse_manifest.json"
    audit = {
        "schema_version": "base-state-reuse-v1",
        "status": "complete",
        "source_experiment_dir": str(source_experiment_dir.resolve()),
        "destination_experiment_dir": str(destination_experiment_dir.resolve()),
        "source_code_version": (
            source_manifest.get("code_version") if source_manifest is not None else None
        ),
        "source_mode": source_manifest.get("mode") if source_manifest is not None else None,
        "source_stop_reason": (
            source_manifest.get("stop_reason") if source_manifest is not None else None
        ),
        "source_file_count": int(len(source_inventory)),
        "destination_file_count_before_audit": int(len(destination_inventory)),
        "localized_manifest_count": int(len(localized_manifests)),
        "localized_manifests": list(localized_manifests),
        "missing_relative_paths": missing,
        "non_json_size_mismatches": size_mismatches,
        "non_json_content_mismatches": content_mismatches,
        "non_json_content_verified": True,
        "validation_passed": True,
    }
    audit_path.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    return audit_path


def _source_experiment_dir_from_base_variant(source_variant_dir: str | Path) -> Path:
    source = Path(source_variant_dir).resolve()
    if source.name != "variant" or source.parent.name != "S000_base" or source.parent.parent.name != "states":
        raise ValueError(
            "Base-state reuse expects a canonical <experiment>/states/S000_base/variant directory: "
            f"{source}"
        )
    return source.parent.parent.parent


def _run_state_hydraulic_context(
    *,
    context: ExperimentStateContext,
    hydraulic_context_runner: Callable[..., StateHydraulicContextOutputs],
    hydraulic_context_kwargs: Mapping[str, Any] | None = None,
    parent_context: ExperimentStateContext | None = None,
) -> ExperimentStateContext:
    runner_kwargs = dict(hydraulic_context_kwargs or {})
    if parent_context is not None and parent_context.hydraulic_context_output_dir is not None:
        runner_kwargs.setdefault(
            "parent_link_context_path",
            parent_context.hydraulic_context_output_dir / "link_hydraulic_context.csv",
        )
    outputs = hydraulic_context_runner(
        directed_links_path=context.analysis_links_path or context.reviewed_links_path,
        directed_nodes_path=context.reviewed_nodes_path,
        hierarchy_output_dir=None if context.hierarchy_deferred else context.hierarchy_output_dir,
        output_dir=context.state_dir / "hydraulic_context",
        state_id=context.state_id,
        **runner_kwargs,
    )
    context.hydraulic_context_output_dir = outputs.output_dir
    return context


def _build_identity_node_match(nodes: gpd.GeoDataFrame) -> pd.DataFrame:
    ordered_nodes = nodes.copy()
    ordered_nodes["id_node"] = ordered_nodes["id_node"].astype(int)
    ordered_nodes = ordered_nodes.reset_index(drop=True)
    return pd.DataFrame(
        {
            "child_id_node": ordered_nodes["id_node"].astype(int),
            "matched_parent_node_id": ordered_nodes["id_node"].astype(int),
            "match_distance": 0.0,
            "match_within_tolerance": True,
            "parent_node_order": ordered_nodes.index.astype(int),
            "child_is_inlet_raw": ordered_nodes["is_inlet"].fillna(False).astype(bool),
            "child_is_outlet_raw": ordered_nodes["is_outlet"].fillna(False).astype(bool),
            "parent_is_inlet": ordered_nodes["is_inlet"].fillna(False).astype(bool),
            "parent_is_outlet": ordered_nodes["is_outlet"].fillna(False).astype(bool),
        }
    )


def _normalize_numeric_id_columns(
    frame: pd.DataFrame | gpd.GeoDataFrame,
    *,
    columns: Sequence[str] = ("id_node", "idx_node", "id_link", "id_us_node", "id_ds_node"),
) -> pd.DataFrame | gpd.GeoDataFrame:
    result = frame.copy()
    for column in columns:
        if column not in result.columns:
            continue
        numeric = pd.to_numeric(result[column], errors="coerce")
        non_null_original = result[column].notna()
        if numeric[non_null_original].isna().any():
            continue
        if numeric.isna().any():
            result[column] = numeric.astype("Int64")
        else:
            result[column] = numeric.astype("int64")
    return result


def _type_io_label(*, is_inlet: bool, is_outlet: bool) -> str:
    if is_inlet and is_outlet:
        return "both"
    if is_inlet:
        return "inlet"
    if is_outlet:
        return "outlet"
    return "neither"


def _ensure_seed_direction_columns(
    links: gpd.GeoDataFrame,
    nodes: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    result = links.copy()
    us_nodes = pd.to_numeric(result.get("id_us_node"), errors="coerce")
    ds_nodes = pd.to_numeric(result.get("id_ds_node"), errors="coerce")
    if us_nodes is not None and ds_nodes is not None and us_nodes.notna().all() and ds_nodes.notna().all():
        return result

    node_frame = nodes.copy()
    node_frame["id_node"] = pd.to_numeric(node_frame["id_node"], errors="coerce").astype("Int64")
    inlet_nodes = set(
        node_frame.loc[node_frame.get("is_inlet", False).fillna(False), "id_node"].dropna().astype(int).tolist()
    )
    outlet_nodes = set(
        node_frame.loc[node_frame.get("is_outlet", False).fillna(False), "id_node"].dropna().astype(int).tolist()
    )

    seeded_us: list[int] = []
    seeded_ds: list[int] = []
    seeded_id_nodes: list[str] = []
    seeded_is_inlet: list[bool] = []
    seeded_is_outlet: list[bool] = []
    seeded_type_io: list[str] = []

    for row in result.itertuples(index=False):
        pair = _parse_int_list(getattr(row, "id_nodes"))
        if len(pair) != 2:
            raise ValueError(f"Expected exactly two node ids in id_nodes, got {getattr(row, 'id_nodes')!r}.")
        us_node, ds_node = int(pair[0]), int(pair[1])
        is_inlet = us_node in inlet_nodes
        is_outlet = ds_node in outlet_nodes
        seeded_us.append(us_node)
        seeded_ds.append(ds_node)
        seeded_id_nodes.append(f"{us_node}, {ds_node}")
        seeded_is_inlet.append(is_inlet)
        seeded_is_outlet.append(is_outlet)
        seeded_type_io.append(_type_io_label(is_inlet=is_inlet, is_outlet=is_outlet))

    result["id_us_node"] = pd.Series(seeded_us, index=result.index, dtype="Int64")
    result["id_ds_node"] = pd.Series(seeded_ds, index=result.index, dtype="Int64")
    result["id_nodes"] = seeded_id_nodes
    result["is_inlet"] = seeded_is_inlet
    result["is_outlet"] = seeded_is_outlet
    result["type_io"] = seeded_type_io
    if "direction_assignment_method" not in result.columns:
        result["direction_assignment_method"] = "raw_id_nodes_seed"
    else:
        result["direction_assignment_method"] = result["direction_assignment_method"].where(
            pd.notna(result["direction_assignment_method"]) & result["direction_assignment_method"].astype(str).ne(""),
            "raw_id_nodes_seed",
        )
    if "geometry_reversed_to_match_flow" not in result.columns:
        result["geometry_reversed_to_match_flow"] = False
    return result


def _build_identity_link_match(links: gpd.GeoDataFrame) -> pd.DataFrame:
    base_links = links.copy()
    base_links["id_link"] = base_links["id_link"].astype(int)
    lengths = base_links.geometry.length.astype(float)
    return pd.DataFrame(
        {
            "child_id_link": base_links["id_link"].astype(int),
            "parent_id_link": base_links["id_link"].astype(int),
            "parent_id_us_node": pd.to_numeric(base_links.get("id_us_node"), errors="coerce").astype("Int64"),
            "parent_id_ds_node": pd.to_numeric(base_links.get("id_ds_node"), errors="coerce").astype("Int64"),
            "child_length": lengths,
            "parent_length": lengths,
            "child_overlap_length": lengths,
            "child_overlap_fraction": 1.0,
            "parent_overlap_length": lengths,
            "parent_overlap_fraction": 1.0,
            "child_trim_distance": 0.0,
            "parent_trim_distance": 0.0,
            "child_core_length": lengths,
            "parent_core_length": lengths,
            "child_core_overlap_length": lengths,
            "child_core_overlap_fraction": 1.0,
            "parent_core_overlap_length": lengths,
            "parent_core_overlap_fraction": 1.0,
            "distance": 0.0,
            "candidate_class": "core_overlap",
            "candidate_rank": 1,
            "is_dominant": True,
        }
    )


def _build_identity_link_lineage(links: gpd.GeoDataFrame) -> pd.DataFrame:
    base_links = links.copy()
    base_links["id_link"] = base_links["id_link"].astype(int)
    us_nodes = pd.to_numeric(base_links.get("id_us_node"), errors="coerce").astype("Int64")
    ds_nodes = pd.to_numeric(base_links.get("id_ds_node"), errors="coerce").astype("Int64")
    link_ids_text = base_links["id_link"].astype(str)
    return pd.DataFrame(
        {
            "id_link": base_links["id_link"].astype(int),
            "lineage_type": "base_identity_1to1",
            "dominant_parent_link_id": base_links["id_link"].astype(int),
            "matched_parent_link_ids": link_ids_text,
            "primary_parent_link_ids": link_ids_text,
            "secondary_parent_link_ids": "",
            "touch_parent_link_ids": "",
            "candidate_parent_link_ids": link_ids_text,
            "matched_parent_link_count": 1,
            "primary_parent_link_count": 1,
            "secondary_parent_link_count": 0,
            "touch_parent_link_count": 0,
            "dominant_parent_overlap_fraction": 1.0,
            "dominant_parent_overlap_length": base_links.geometry.length.astype(float),
            "dominant_parent_core_overlap_fraction": 1.0,
            "dominant_parent_core_overlap_length": base_links.geometry.length.astype(float),
            "matched_parent_overlap_fraction": 1.0,
            "matched_parent_overlap_length": base_links.geometry.length.astype(float),
            "matched_parent_core_overlap_fraction": 1.0,
            "matched_parent_core_overlap_length": base_links.geometry.length.astype(float),
            "lineage_method": "base_state_identity",
            "matched_parent_us_node": us_nodes,
            "matched_parent_ds_node": ds_nodes,
            "matched_parent_node_path": "",
        }
    )


def _serialize_unique_ints(values: pd.Series) -> str:
    unique: list[int] = []
    seen: set[int] = set()
    for value in values.tolist():
        if value is pd.NA or pd.isna(value):
            continue
        numeric = int(value)
        if numeric in seen:
            continue
        seen.add(numeric)
        unique.append(numeric)
    return _serialize_int_list(unique)


def _serialize_unique_strings(values: pd.Series) -> str:
    unique: list[str] = []
    seen: set[str] = set()
    for value in values.tolist():
        if value is pd.NA or pd.isna(value):
            continue
        text = str(value).strip()
        if not text or text in seen:
            continue
        seen.add(text)
        unique.append(text)
    return ",".join(unique)


def _build_unit_link_membership(
    *,
    path_metrics: pd.DataFrame,
    unit_metrics: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    membership_columns = [
        "unit_id",
        "id_link",
        "path_ids",
        "n_paths_through_link",
        "root_unit_id",
        "collapse_level",
        "compound_unit_id",
        "compound_bubble_id",
        "unit_class",
        "unit_topodynamic_class",
    ]
    link_summary_columns = [
        "id_link",
        "unit_ids",
        "n_units",
        "root_unit_ids",
        "collapse_levels",
        "compound_unit_ids",
        "compound_bubble_ids",
        "unit_classes",
        "unit_topodynamic_classes",
    ]
    if path_metrics.empty or unit_metrics.empty or "unit_id" not in path_metrics.columns or "id_links" not in path_metrics.columns:
        return (
            pd.DataFrame(columns=membership_columns),
            pd.DataFrame(columns=link_summary_columns),
        )

    raw_records: list[dict[str, Any]] = []
    for row in path_metrics.itertuples(index=False):
        link_ids = _parse_int_list(getattr(row, "id_links", ""))
        if not link_ids:
            continue
        for link_id in link_ids:
            raw_records.append(
                {
                    "unit_id": int(row.unit_id),
                    "id_link": int(link_id),
                    "path_id": int(row.path_id),
                }
            )
    if not raw_records:
        return (
            pd.DataFrame(columns=membership_columns),
            pd.DataFrame(columns=link_summary_columns),
        )

    raw_membership = pd.DataFrame.from_records(raw_records)
    membership = (
        raw_membership.groupby(["unit_id", "id_link"], sort=True, as_index=False)
        .agg(
            path_ids=("path_id", lambda values: _serialize_int_list(sorted({int(value) for value in values}))),
            n_paths_through_link=("path_id", "nunique"),
        )
    )

    unit_metric_columns = [
        column
        for column in (
            "unit_id",
            "root_unit_id",
            "collapse_level",
            "compound_unit_id",
            "compound_bubble_id",
            "class",
            "unit_topodynamic_class",
        )
        if column in unit_metrics.columns
    ]
    if unit_metric_columns:
        membership = membership.merge(
            unit_metrics[unit_metric_columns].rename(columns={"class": "unit_class"}),
            on="unit_id",
            how="left",
            validate="many_to_one",
        )
    for column in membership_columns:
        if column not in membership.columns:
            membership[column] = pd.NA
    membership = membership[membership_columns].sort_values(["id_link", "unit_id"], kind="mergesort").reset_index(drop=True)

    link_summary = (
        membership.groupby("id_link", sort=True, as_index=False)
        .agg(
            unit_ids=("unit_id", _serialize_unique_ints),
            n_units=("unit_id", "nunique"),
            root_unit_ids=("root_unit_id", _serialize_unique_ints),
            collapse_levels=("collapse_level", _serialize_unique_ints),
            compound_unit_ids=("compound_unit_id", _serialize_unique_ints),
            compound_bubble_ids=("compound_bubble_id", _serialize_unique_ints),
            unit_classes=("unit_class", _serialize_unique_strings),
            unit_topodynamic_classes=("unit_topodynamic_class", _serialize_unique_strings),
        )
        .sort_values("id_link", kind="mergesort")
        .reset_index(drop=True)
    )
    return membership, link_summary


def _refresh_reused_base_unit_metadata(
    *,
    variant_outputs: BaseStateVariantOutputs,
    workflow: UnitWorkflowOutputs,
) -> None:
    """Replace stale copied unit annotations without recomputing base widths."""
    membership, link_summary = _build_unit_link_membership(
        path_metrics=workflow.path_metrics,
        unit_metrics=workflow.unit_metrics,
    )
    summary_dir = Path(variant_outputs.output_dir) / "summary"
    width_dir = Path(variant_outputs.output_dir) / "widths"
    membership.to_csv(summary_dir / "unit_link_membership.csv", index=False)
    metadata_columns = [column for column in link_summary.columns if column != "id_link"]

    def refreshed(frame: pd.DataFrame) -> pd.DataFrame:
        result = frame.drop(columns=metadata_columns, errors="ignore").merge(
            link_summary,
            on="id_link",
            how="left",
            validate="1:1",
        )
        for column, default in (
            ("unit_ids", ""),
            ("n_units", 0),
            ("root_unit_ids", ""),
            ("collapse_levels", ""),
            ("compound_unit_ids", ""),
            ("compound_bubble_ids", ""),
            ("unit_classes", ""),
            ("unit_topodynamic_classes", ""),
        ):
            if column not in result.columns:
                result[column] = default
            elif column == "n_units":
                result[column] = (
                    pd.to_numeric(result[column], errors="coerce").fillna(0).astype(int)
                )
            else:
                result[column] = result[column].astype("string").fillna(default)
        if {"id_nodes", "id_us_node", "id_ds_node"}.issubset(result.columns):
            complete = result[["id_us_node", "id_ds_node"]].notna().all(axis=1)
            result.loc[complete, "id_nodes"] = [
                f"{int(upstream)}, {int(downstream)}"
                for upstream, downstream in zip(
                    result.loc[complete, "id_us_node"],
                    result.loc[complete, "id_ds_node"],
                )
            ]
        return result

    for path in (
        Path(variant_outputs.directed_links_path),
        width_dir / "links_with_width_families.gpkg",
    ):
        if not path.exists():
            continue
        frame = refreshed(gpd.read_file(path))
        path.unlink()
        gpd.GeoDataFrame(frame, geometry="geometry", crs=frame.crs).to_file(
            path,
            driver="GPKG",
        )
    family_path = width_dir / "link_width_families.csv"
    if family_path.exists():
        refreshed(pd.read_csv(family_path)).to_csv(family_path, index=False)


def _selected_unit_link_ids(path_metrics: pd.DataFrame, unit_ids: Sequence[int]) -> list[int]:
    """Resolve selected units to the reviewed-link IDs recorded by hierarchy paths."""
    selected = path_metrics.loc[path_metrics["unit_id"].astype(int).isin([int(unit_id) for unit_id in unit_ids])]
    result: set[int] = set()
    for value in selected.get("id_links", pd.Series(dtype=object)):
        result.update(int(item.strip()) for item in str(value).split(",") if item.strip())
    if not result:
        raise ValueError(f"No reviewed link IDs were found for selected unit IDs {list(unit_ids)}.")
    return sorted(result)


def _materialize_copied_base_state_variant(
    *,
    context: ExperimentStateContext,
    source_variant_dir: str | Path,
    example_id: str,
    collapse_corridor_mode: str,
    boundary_component_labels_path: str | Path | None,
) -> BaseStateVariantOutputs:
    """Copy a validated base variant into a new self-contained experiment."""
    source = Path(source_variant_dir).resolve()
    if not source.is_dir():
        raise FileNotFoundError(f"Base-state variant source does not exist: {source}")
    destination = (context.state_dir / "variant").resolve()
    if source == destination:
        raise ValueError("Base-state variant source and destination must differ.")
    _copytree_transactional(source, destination)

    source_experiment = _source_experiment_dir_from_base_variant(source)
    destination_experiment = context.state_dir.parent.parent.resolve()
    localized_manifests = _localize_copied_base_manifests(
        source_experiment_dir=source_experiment,
        destination_experiment_dir=destination_experiment,
        destination_state_dir=context.state_dir.resolve(),
    )

    manifest_path = destination / "summary" / "variant_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["collapse_corridor_mode"] = collapse_corridor_mode
        manifest["base_state_copy"] = {
            "copied": True,
            "source_variant_dir": str(source),
            "source_experiment_dir": str(source_experiment),
        }
        if boundary_component_labels_path is not None:
            local_boundary = str(Path(boundary_component_labels_path).resolve())
            manifest.setdefault("output_paths", {})["boundary_component_labels"] = local_boundary
            manifest.setdefault("source_paths", {})["boundary_component_labels"] = local_boundary
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    routing_manifest_path = destination / "widths" / "routing_width_application_manifest.json"
    if routing_manifest_path.exists():
        routing_manifest = json.loads(routing_manifest_path.read_text(encoding="utf-8"))
        local_audit = destination / "widths" / "routing_width_application.csv"
        if local_audit.exists():
            routing_manifest.setdefault("files", {})["link_audit"] = local_audit.name
        routing_manifest_path.write_text(json.dumps(routing_manifest, indent=2), encoding="utf-8")

    directed_links = next(
        (
            path
            for path in (destination / "directed").glob("*S000_base_directed_links.gpkg")
            if not path.name.startswith("._")
        ),
        None,
    )
    directed_nodes = next(
        (
            path
            for path in (destination / "directed").glob("*S000_base_directed_nodes.gpkg")
            if not path.name.startswith("._")
        ),
        None,
    )
    collapsed_mask = next(
        (
            path
            for path in (destination / "mask").glob("*S000_base_collapsed.tif")
            if not path.name.startswith("._")
        ),
        None,
    )
    if directed_links is None or directed_nodes is None or collapsed_mask is None:
        raise FileNotFoundError(
            f"Copied base variant is missing required S000_base artifacts in {destination}."
        )
    bank_labels = destination / "mask" / "bank_labels.tif"
    source_manifest_path = source_experiment / "experiment_manifest.json"
    source_manifest = (
        json.loads(source_manifest_path.read_text(encoding="utf-8"))
        if source_manifest_path.exists()
        else None
    )
    _write_base_reuse_audit(
        source_experiment_dir=source_experiment,
        destination_experiment_dir=destination_experiment,
        destination_state_dir=context.state_dir.resolve(),
        localized_manifests=localized_manifests,
        source_manifest=source_manifest,
    )
    return BaseStateVariantOutputs(
        output_dir=destination,
        collapsed_mask_path=collapsed_mask,
        directed_links_path=directed_links,
        directed_nodes_path=directed_nodes,
        bank_labels_path=bank_labels if bank_labels.exists() else None,
        boundary_component_labels_path=(
            Path(boundary_component_labels_path)
            if boundary_component_labels_path is not None
            else None
        ),
    )


def _make_panelized_child_runner(
    *,
    source_panel_root: Path,
    panel_plan_path: Path,
    panel_breaks_path: Path,
    boundary_match_plan_path: Path,
    parent_merged_links_path: Path,
    boundary_component_labels_path: Path,
    exit_sides_by_panel: Mapping[str, str],
    run_name: str,
) -> Callable[..., PanelizedChildVariantOutputs]:
    """Return the panelized child runner for unit and compound-bubble selections."""
    def runner(**kwargs: Any) -> PanelizedChildVariantOutputs:
        unit_ids = list(kwargs.get("unit_ids") or [])
        if not unit_ids:
            raise ValueError("Panelized child execution requires at least one selected unit.")
        if kwargs.get("group_label") is not None:
            raise NotImplementedError("Panelized sequential-group execution is not implemented yet.")
        links = gpd.read_file(kwargs["reviewed_links_path"])
        link_ids = _selected_unit_link_ids(pd.read_csv(kwargs["panelized_path_metrics"]), unit_ids)
        selected = links.loc[links["id_link"].astype(int).isin(link_ids)]
        if selected.empty:
            raise ValueError("Selected unit links are not present in the panelized parent state.")
        routing_widths = (
            pd.to_numeric(selected["wid_adj_routing"], errors="coerce")
            if "wid_adj_routing" in selected.columns
            else pd.Series(np.nan, index=selected.index, dtype=float)
        )
        wet_widths = (
            pd.to_numeric(selected["wid_adj_wet"], errors="coerce")
            if "wid_adj_wet" in selected.columns
            else pd.Series(np.nan, index=selected.index, dtype=float)
        )
        legacy_widths = (
            pd.to_numeric(selected["wid_adj"], errors="coerce")
            if "wid_adj" in selected.columns
            else pd.Series(np.nan, index=selected.index, dtype=float)
        )
        width = routing_widths.where(
            np.isfinite(routing_widths) & (routing_widths > 0),
            wet_widths,
        )
        width = width.where(np.isfinite(width) & (width > 0), legacy_widths).fillna(0).max()
        footprint = _component_footprint_geometry(
            links,
            link_ids,
            preferred_width_field=str(kwargs.get("preferred_width_field", "wid_adj_routing")),
            buffer_scale=float(kwargs.get("footprint_buffer_scale", 0.5)),
        )
        if footprint.is_empty:
            raise ValueError("Selected unit links did not produce a valid collapse footprint.")
        selected_node_ids = sorted(
            {
                node_id
                for value in selected.get("id_nodes", pd.Series(dtype=object))
                for node_id in _parse_int_list(value)
            }
        )
        state_root = Path(kwargs["output_dir"]) / "panelized_state"
        result = materialize_panelized_child_state(
            parent_mask_path=kwargs["cleaned_mask_path"], source_panel_root=source_panel_root,
            state_root=state_root, panel_plan_path=panel_plan_path, panel_breaks_path=panel_breaks_path,
            boundary_match_plan_path=boundary_match_plan_path, boundary_component_labels_path=boundary_component_labels_path,
            footprint=footprint, halo_distance=max(float(width), 1.0), run_name=run_name,
            exit_sides_by_panel=dict(exit_sides_by_panel), panel_runner=run_rivgraph_for_run,
            parent_directed_links_path=kwargs["reviewed_links_path"],
            materialize_full_mask=(
                str(kwargs.get("panelized_mask_publication", "materialized"))
                != "virtual_patch"
            ),
            all_touched=bool(kwargs.get("all_touched", True)),
            require_single_source_sink_path=bool(
                kwargs.get("require_single_source_sink_path", False)
            ),
        )
        return _finalize_panelized_child_result(
            panelized_result=result,
            state_dir=Path(kwargs["output_dir"]).parent,
            parent_mask_path=Path(kwargs["cleaned_mask_path"]),
            parent_links_path=Path(kwargs["reviewed_links_path"]),
            parent_nodes_path=Path(kwargs["reviewed_nodes_path"]),
            parent_merged_links_path=parent_merged_links_path,
            boundary_component_labels_path=boundary_component_labels_path,
            footprint=footprint,
            unit_ids=unit_ids,
            selected_link_ids=link_ids,
            selected_node_ids=selected_node_ids,
            group_label=kwargs.get("group_label"),
            selection_type=kwargs.get("selection_type"),
            compound_bubble_id=kwargs.get("compound_bubble_id"),
            bubble_root_unit_id=kwargs.get("bubble_root_unit_id"),
            example_id=str(kwargs.get("example_id") or run_name),
            variant_id=str(kwargs["variant_id"]),
            collapse_corridor_mode=str(kwargs.get("collapse_corridor_mode", "realized")),
            bank_candidate_neighbors_max=int(kwargs.get("bank_candidate_neighbors_max", 2048)),
            sword_node_source_path=kwargs.get("sword_node_source_path"),
            sword_wse_field=kwargs.get("sword_wse_field"),
            sword_match_tolerance=kwargs.get("sword_match_tolerance"),
            sword_example_station_source_path=kwargs.get("sword_example_station_source_path"),
            sword_station_match_source_path=kwargs.get("sword_station_match_source_path"),
            sword_reach_buffer_steps=int(kwargs.get("sword_reach_buffer_steps", 2)),
        )
    return runner


def _make_panelized_reference_section_builder(
    *,
    panelized_variant_runner: Callable[..., NetworkVariantOutputs],
    base_context: ExperimentStateContext,
    base_variant_outputs: BaseStateVariantOutputs,
    base_analysis_links_path: Path,
    example_id: str,
    exit_sides: str,
    preferred_width_field: str,
    footprint_buffer_scale: float,
    all_touched: bool,
    bank_candidate_neighbors_max: int,
    sword_node_source_path: str | Path | None,
    sword_wse_field: str | None,
    sword_match_tolerance: float | None,
    sword_example_station_source_path: str | Path | None,
    sword_station_match_source_path: str | Path | None,
    sword_reach_buffer_steps: int,
) -> Callable[..., Mapping[str, Any]]:
    """Build the all-units reference section through the panelized child path."""
    if base_context.workflow is None:
        raise ValueError("Panelized reference-section construction requires the base hierarchy.")
    unit_summary = base_context.workflow.unit_summary
    if "unit_id" not in unit_summary.columns:
        raise ValueError("Base unit summary is missing unit_id for the reference section.")
    unit_ids = sorted(
        pd.to_numeric(unit_summary["unit_id"], errors="coerce")
        .dropna()
        .astype(int)
        .unique()
        .tolist()
    )
    if not unit_ids:
        raise ValueError("No base units were available for the panelized reference section.")

    def builder(**kwargs: Any) -> Mapping[str, Any]:
        reference_paths = kwargs["reference_paths"]
        variant_dir = Path(reference_paths["variant"])
        outputs = panelized_variant_runner(
            cleaned_mask_path=base_variant_outputs.collapsed_mask_path,
            reviewed_links_path=base_analysis_links_path,
            reviewed_nodes_path=base_variant_outputs.directed_nodes_path,
            unit_ids=unit_ids,
            variant_id=REFERENCE_SECTION_VARIANT_ID,
            example_id=example_id,
            output_dir=variant_dir,
            group_label=None,
            selection_type="reference_section_all_units",
            collapse_corridor_mode="realized",
            preferred_width_field=preferred_width_field,
            footprint_buffer_scale=footprint_buffer_scale,
            all_touched=all_touched,
            bank_candidate_neighbors_max=bank_candidate_neighbors_max,
            sword_node_source_path=sword_node_source_path,
            sword_wse_field=sword_wse_field,
            sword_match_tolerance=sword_match_tolerance,
            sword_example_station_source_path=sword_example_station_source_path,
            sword_station_match_source_path=sword_station_match_source_path,
            sword_reach_buffer_steps=sword_reach_buffer_steps,
            panelized_path_metrics=base_context.hierarchy_output_dir / "path_metrics.csv",
            panelized_mask_publication="virtual_patch",
            require_single_source_sink_path=True,
        )
        return {
            "construction_backend": "panelized",
            "example_id": example_id,
            "exit_sides": exit_sides,
            "unit_ids": unit_ids,
            "variant_output_dir": outputs.output_dir,
            "directed_links_path": outputs.directed_links_path,
            "directed_nodes_path": outputs.directed_nodes_path,
            "centerline_path": outputs.rivgraph_centerline_path,
            "collapsed_mask_path": outputs.collapsed_mask_path,
            "boundary_component_labels_path": outputs.boundary_component_labels_path,
            "link_width_samples_path": outputs.output_dir / "widths" / "link_width_samples.csv",
            "manifest_path": outputs.output_dir / "summary" / "variant_manifest.json",
        }

    return builder


def _materialize_base_state_variant(
    *,
    context: ExperimentStateContext,
    example_id: str,
    exit_sides: str,
    collapse_corridor_mode: str,
    preferred_width_field: str,
    transect_scale: float,
    min_transect_pixels: float,
    bank_candidate_neighbors_max: int = 2048,
    sword_node_source_path: str | Path | None,
    sword_wse_field: str | None,
    sword_match_tolerance: float | None,
    sword_example_station_source_path: str | Path | None,
    sword_station_match_source_path: str | Path | None,
    sword_reach_buffer_steps: int,
    boundary_component_labels_path: str | Path | None = None,
    boundary_component_output_dir: str | Path | None = None,
    panelized_width_samples_path: str | Path | None = None,
    panelized_boundary_component_labels_path: str | Path | None = None,
) -> BaseStateVariantOutputs:
    variant_dir = context.state_dir / "variant"
    summary_dir = variant_dir / "summary"
    mask_dir = variant_dir / "mask"
    matching_dir = variant_dir / "matching"
    directed_dir = variant_dir / "directed"
    width_dir = variant_dir / "widths"
    for directory in (summary_dir, mask_dir, matching_dir, directed_dir, width_dir):
        directory.mkdir(parents=True, exist_ok=True)

    selection_metadata = _collapsed_selection_metadata([], None)
    base_variant_id = context.state_id
    base_run_name = f"{example_id}__{base_variant_id}"
    collapsed_mask_path = mask_dir / f"{base_run_name}_collapsed.tif"
    shutil.copy2(context.cleaned_mask_path, collapsed_mask_path)
    origin_mask_path = mask_dir / "origin_mask.tif"
    shutil.copy2(collapsed_mask_path, origin_mask_path)
    boundary_output_dir = (
        Path(boundary_component_output_dir)
        if boundary_component_output_dir is not None
        else (variant_dir / "boundary_components")
    )
    boundary_outputs = _ensure_boundary_component_mask(
        cleaned_mask_path=context.cleaned_mask_path,
        reviewed_links_path=context.reviewed_links_path,
        output_dir=boundary_output_dir,
        boundary_component_labels_path=boundary_component_labels_path,
    )
    resolved_boundary_component_labels_path = Path(boundary_outputs["labels"])

    reviewed_links = gpd.read_file(context.reviewed_links_path)
    reviewed_nodes = gpd.read_file(context.reviewed_nodes_path)
    reviewed_links = _ensure_seed_direction_columns(reviewed_links, reviewed_nodes)
    directed_links = _apply_collapsed_selection_metadata(reviewed_links, unit_ids=[], group_label=None)
    directed_nodes = _apply_collapsed_selection_metadata(reviewed_nodes, unit_ids=[], group_label=None)
    directed_links = _normalize_numeric_id_columns(directed_links)
    directed_nodes = _normalize_numeric_id_columns(directed_nodes)
    directed_links["example_id"] = example_id
    directed_links["variant_id"] = base_variant_id
    directed_nodes["example_id"] = example_id
    directed_nodes["variant_id"] = base_variant_id

    unit_link_membership, unit_link_summary = _build_unit_link_membership(
        path_metrics=context.workflow.path_metrics,
        unit_metrics=context.workflow.unit_metrics,
    )
    if not unit_link_membership.empty:
        unit_link_membership.to_csv(summary_dir / "unit_link_membership.csv", index=False)
    else:
        pd.DataFrame(
            columns=[
                "unit_id",
                "id_link",
                "path_ids",
                "n_paths_through_link",
                "root_unit_id",
                "collapse_level",
                "compound_unit_id",
                "compound_bubble_id",
                "unit_class",
                "unit_topodynamic_class",
            ]
        ).to_csv(summary_dir / "unit_link_membership.csv", index=False)

    if not unit_link_summary.empty:
        directed_links = directed_links.merge(unit_link_summary, on="id_link", how="left", validate="1:1")
    for column, default in (
        ("unit_ids", ""),
        ("n_units", 0),
        ("root_unit_ids", ""),
        ("collapse_levels", ""),
        ("compound_unit_ids", ""),
        ("compound_bubble_ids", ""),
        ("unit_classes", ""),
        ("unit_topodynamic_classes", ""),
    ):
        if column not in directed_links.columns:
            directed_links[column] = default
    directed_links["n_units"] = pd.to_numeric(directed_links["n_units"], errors="coerce").fillna(0).astype(int)
    directed_links = _normalize_numeric_id_columns(directed_links)

    with rasterio.open(collapsed_mask_path) as src:
        collapsed_profile = src.profile.copy()
        origin_mask = src.read(1).astype(np.uint16)
    bank_labels_path, removed_bank_labels = _write_state_bank_labels(
        boundary_component_labels_path=resolved_boundary_component_labels_path,
        origin_mask=origin_mask,
        output_path=mask_dir / "bank_labels.tif",
        profile=collapsed_profile,
    )

    enriched_links, link_width_samples = compute_width_families(
        directed_links,
        collapsed_mask_path=collapsed_mask_path,
        wet_reference_mask_path=collapsed_mask_path,
        origin_mask_path=origin_mask_path,
        bank_labels_path=bank_labels_path,
        bank_candidate_neighbors_max=bank_candidate_neighbors_max,
    )
    enriched_links, base_width_review_provenance = attach_permanent_base_review_provenance(
        enriched_links,
        reviewed_links_path=context.reviewed_links_path,
        reviewed_nodes_path=context.reviewed_nodes_path,
        cleaned_mask_path=context.cleaned_mask_path,
    )
    enriched_links = _apply_collapsed_selection_metadata(enriched_links, unit_ids=[], group_label=None)
    enriched_links = _normalize_numeric_id_columns(enriched_links)
    enriched_links["example_id"] = example_id
    enriched_links["variant_id"] = base_variant_id
    link_width_samples = _apply_collapsed_selection_metadata(link_width_samples, unit_ids=[], group_label=None)
    link_width_samples = _normalize_numeric_id_columns(link_width_samples)
    link_width_samples["example_id"] = example_id
    link_width_samples["variant_id"] = base_variant_id
    enriched_links, routing_width_audit, routing_width_application = apply_routing_width_model(
        enriched_links,
        link_width_samples,
        directed_nodes,
        example_id=example_id,
        variant_id=base_variant_id,
        apply_known_base_exclusions=True,
    )
    routing_width_paths = write_routing_width_audit(
        routing_width_audit,
        routing_width_application,
        output_dir=width_dir,
    )

    node_match = _apply_collapsed_selection_metadata(_build_identity_node_match(reviewed_nodes), unit_ids=[], group_label=None)
    node_match["example_id"] = example_id
    node_match["variant_id"] = base_variant_id
    link_match = _apply_collapsed_selection_metadata(_build_identity_link_match(reviewed_links), unit_ids=[], group_label=None)
    link_match["example_id"] = example_id
    link_match["variant_id"] = base_variant_id
    link_lineage = _apply_collapsed_selection_metadata(_build_identity_link_lineage(reviewed_links), unit_ids=[], group_label=None)
    link_lineage["example_id"] = example_id
    link_lineage["variant_id"] = base_variant_id

    directed_nodes, node_sword_match, sword_match_summary = match_variant_nodes_to_sword(
        directed_nodes=directed_nodes,
        parent_nodes=reviewed_nodes,
        node_match=node_match,
        sword_node_source_path=sword_node_source_path,
        sword_wse_field=sword_wse_field,
        sword_match_tolerance=sword_match_tolerance,
        example_id=example_id,
        sword_example_station_source_path=sword_example_station_source_path,
        sword_station_match_source_path=sword_station_match_source_path,
        sword_reach_buffer_steps=sword_reach_buffer_steps,
    )
    directed_nodes = _apply_collapsed_selection_metadata(directed_nodes, unit_ids=[], group_label=None)
    directed_nodes = _normalize_numeric_id_columns(directed_nodes)
    directed_nodes["example_id"] = example_id
    directed_nodes["variant_id"] = base_variant_id
    node_sword_match = _apply_collapsed_selection_metadata(node_sword_match, unit_ids=[], group_label=None)
    node_sword_match["example_id"] = example_id
    node_sword_match["variant_id"] = base_variant_id

    link_width_families = pd.DataFrame(enriched_links.drop(columns=enriched_links.geometry.name))
    directed_links_path = directed_dir / f"{base_run_name}_directed_links.gpkg"
    directed_nodes_path = directed_dir / f"{base_run_name}_directed_nodes.gpkg"
    links_with_widths_path = width_dir / "links_with_width_families.gpkg"

    empty_components = _apply_collapsed_selection_metadata(
        pd.DataFrame(
            columns=[
                "component_id",
                "unit_ids",
                "link_ids",
                "node_ids",
                "compound_bubble_ids",
                "n_units",
                "n_links",
                "n_nodes",
                "footprint_area",
                "added_pixels",
                "added_area",
            ]
        ),
        unit_ids=[],
        group_label=None,
    )
    empty_components["example_id"] = example_id
    empty_components["variant_id"] = base_variant_id
    empty_edit_geometries = gpd.GeoDataFrame(
        {
            "component_id": pd.Series(dtype="string"),
            "geometry_role": pd.Series(dtype="string"),
            "action": pd.Series(dtype="string"),
            "unit_ids": pd.Series(dtype="string"),
            "collapsed_selection_type": pd.Series(dtype="string"),
            "collapsed_selection_label": pd.Series(dtype="string"),
            "collapsed_unit_ids": pd.Series(dtype="string"),
            "collapsed_group_label": pd.Series(dtype="string"),
            "collapsed_unit_count": pd.Series(dtype="int64"),
            "example_id": pd.Series(dtype="string"),
            "variant_id": pd.Series(dtype="string"),
        },
        geometry=gpd.GeoSeries([], crs=reviewed_links.crs),
        crs=reviewed_links.crs,
    )

    empty_components.to_csv(summary_dir / "collapse_components.csv", index=False)
    empty_edit_geometries.to_file(mask_dir / "collapse_edit_geometries.gpkg", driver="GPKG")
    link_width_families.to_csv(width_dir / "link_width_families.csv", index=False)
    link_width_samples.to_csv(width_dir / "link_width_samples.csv", index=False)
    enriched_links.to_file(links_with_widths_path, driver="GPKG")
    node_match.to_csv(matching_dir / "node_match.csv", index=False)
    node_sword_match.to_csv(matching_dir / "node_sword_match.csv", index=False)
    link_match.to_csv(matching_dir / "link_match.csv", index=False)
    link_lineage.to_csv(matching_dir / "link_lineage.csv", index=False)
    directed_links.to_file(directed_links_path, driver="GPKG")
    directed_nodes.to_file(directed_nodes_path, driver="GPKG")
    with (directed_dir / "direction_validation_report.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "method": "base_state_reuse",
                "is_valid": True,
                "issues": [],
                "source_nodes": reviewed_nodes.loc[reviewed_nodes["is_inlet"].fillna(False), "id_node"].astype(int).tolist(),
                "sink_nodes": reviewed_nodes.loc[reviewed_nodes["is_outlet"].fillna(False), "id_node"].astype(int).tolist(),
            },
            handle,
            indent=2,
        )

    manifest = {
        "example_id": example_id,
        "variant_id": base_variant_id,
        "exit_sides": exit_sides,
        "collapse_corridor_mode": collapse_corridor_mode,
        "selection_source": "base_state",
        "selected_unit_ids": [],
        "group_label": None,
        "collapsed_selection": selection_metadata,
        "n_components": 0,
        "component_ids": [],
        "base_state_materialized": True,
        "mask_summary": {
            "base_water_pixels": None,
            "collapsed_water_pixels": None,
            "pixels_added": 0,
            "pixels_removed": 0,
            "changed_pixels": 0,
        },
        "sword_matching": {
            "wse_field": sword_wse_field,
            "match_tolerance": sword_match_tolerance,
            "reach_buffer_steps": int(sword_reach_buffer_steps),
            "n_matched_nodes": int(node_sword_match["sword_node_id"].notna().sum()) if not node_sword_match.empty else 0,
            "n_propagated_matches": int(node_sword_match["sword_match_from_parent"].fillna(False).sum()) if not node_sword_match.empty else 0,
            "candidate_scope": sword_match_summary.get("scope"),
            "candidate_region": sword_match_summary.get("candidate_region"),
            "candidate_reach_count": sword_match_summary.get("candidate_reach_count"),
            "candidate_reach_ids": sword_match_summary.get("candidate_reach_ids", []),
        },
        "output_paths": {
            "output_dir": str(variant_dir.resolve()),
            "collapsed_mask": str(collapsed_mask_path.resolve()),
            "origin_mask": str(origin_mask_path.resolve()),
            "width_scope_mask": str(origin_mask_path.resolve()),
            "bank_labels": str(bank_labels_path.resolve()),
            "boundary_component_labels": str(resolved_boundary_component_labels_path.resolve()),
            "collapse_components": str((summary_dir / "collapse_components.csv").resolve()),
            "unit_link_membership": str((summary_dir / "unit_link_membership.csv").resolve()),
            "edit_geometries": str((mask_dir / "collapse_edit_geometries.gpkg").resolve()),
            "link_width_families": str((width_dir / "link_width_families.csv").resolve()),
            "link_width_samples": str((width_dir / "link_width_samples.csv").resolve()),
            "links_with_width_families": str(links_with_widths_path.resolve()),
            "routing_width_application": routing_width_paths["link_audit"],
            "routing_width_application_manifest": routing_width_paths["manifest"],
            "node_match": str((matching_dir / "node_match.csv").resolve()),
            "node_sword_match": str((matching_dir / "node_sword_match.csv").resolve()),
            "link_match": str((matching_dir / "link_match.csv").resolve()),
            "link_lineage": str((matching_dir / "link_lineage.csv").resolve()),
            "directed_links": str(directed_links_path.resolve()),
            "directed_nodes": str(directed_nodes_path.resolve()),
            "direction_validation_report": str((directed_dir / "direction_validation_report.json").resolve()),
        },
        "source_paths": {
            "cleaned_mask": str(context.cleaned_mask_path.resolve()),
            "reviewed_links": str(context.reviewed_links_path.resolve()),
            "reviewed_nodes": str(context.reviewed_nodes_path.resolve()),
            "boundary_component_labels": str(resolved_boundary_component_labels_path.resolve()),
        },
        "bank_label_summary": {
            "removed_land_labels": removed_bank_labels,
            "n_removed_land_labels": int(len(removed_bank_labels)),
        },
        "bank_candidate_neighbors_max": int(bank_candidate_neighbors_max),
        "base_width_review_provenance": base_width_review_provenance,
        "routing_width_application": routing_width_application,
    }
    with (summary_dir / "variant_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)

    return BaseStateVariantOutputs(
        output_dir=variant_dir,
        collapsed_mask_path=collapsed_mask_path,
        directed_links_path=directed_links_path,
        directed_nodes_path=directed_nodes_path,
        bank_labels_path=bank_labels_path,
        boundary_component_labels_path=resolved_boundary_component_labels_path,
    )


def _materialize_panelized_base_state_variant(
    *,
    context: ExperimentStateContext,
    example_id: str,
    exit_sides: str,
    collapse_corridor_mode: str,
    panelized_width_samples_path: str | Path | None = None,
    boundary_component_labels_path: str | Path | None = None,
    boundary_component_output_dir: str | Path | None = None,
    sword_node_source_path: str | Path | None = None,
    sword_wse_field: str | None = None,
    sword_match_tolerance: float | None = None,
    sword_example_station_source_path: str | Path | None = None,
    sword_station_match_source_path: str | Path | None = None,
    sword_reach_buffer_steps: int = 2,
    bank_candidate_neighbors_max: int = 2048,
    **_: Any,
) -> BaseStateVariantOutputs:
    """Materialize a schema-compatible S000_base from the panelized handoff."""
    if panelized_width_samples_path is None:
        candidate = context.reviewed_links_path.with_name(
            f"{_infer_example_id(context.reviewed_links_path)}_link_width_samples.csv"
        )
        panelized_width_samples_path = candidate
    sample_path = Path(panelized_width_samples_path)
    if not sample_path.exists():
        raise FileNotFoundError(f"Panelized width samples do not exist: {sample_path}")

    # Keep the standard experiment-level boundary-component product.  The
    # canonical panelized handoff supplies the required single centerline.
    boundary_output_dir = Path(boundary_component_output_dir) if boundary_component_output_dir else context.state_dir.parent.parent / "boundary_components"
    if boundary_component_labels_path is not None:
        boundary_outputs = _ensure_boundary_component_mask(
            cleaned_mask_path=context.cleaned_mask_path,
            reviewed_links_path=context.reviewed_links_path,
            output_dir=boundary_output_dir,
            boundary_component_labels_path=boundary_component_labels_path,
        )
    else:
        run_root = context.reviewed_links_path.resolve().parent.parent
        boundary_outputs = build_panelized_boundary_component_labels(
            cleaned_mask_path=context.cleaned_mask_path,
            centerline_path=context.reviewed_links_path.with_name(f"{example_id}_centerline.gpkg"),
            panel_plan_path=run_root / "panels" / "panel_plan.gpkg",
            panel_root=run_root / "panels",
            output_dir=boundary_output_dir,
        )
    resolved_boundary_labels = Path(boundary_outputs["labels"])

    variant_dir = context.state_dir / "variant"
    summary_dir, mask_dir, matching_dir, directed_dir, width_dir = (
        variant_dir / "summary", variant_dir / "mask", variant_dir / "matching", variant_dir / "directed", variant_dir / "widths"
    )
    for directory in (summary_dir, mask_dir, matching_dir, directed_dir, width_dir):
        directory.mkdir(parents=True, exist_ok=True)
    base_run_name = f"{example_id}__{context.state_id}"
    collapsed_mask_path = mask_dir / f"{base_run_name}_collapsed.tif"
    origin_mask_path = mask_dir / "origin_mask.tif"
    shutil.copy2(context.cleaned_mask_path, collapsed_mask_path)
    shutil.copy2(context.cleaned_mask_path, origin_mask_path)

    base_variant_id = context.state_id
    selection_metadata = _collapsed_selection_metadata([], None)
    reviewed_nodes = _normalize_numeric_id_columns(gpd.read_file(context.reviewed_nodes_path))
    reviewed_links = _normalize_numeric_id_columns(
        _ensure_seed_direction_columns(gpd.read_file(context.reviewed_links_path), reviewed_nodes)
    )
    directed_links = _apply_collapsed_selection_metadata(reviewed_links, unit_ids=[], group_label=None)
    directed_nodes = _apply_collapsed_selection_metadata(reviewed_nodes, unit_ids=[], group_label=None)
    for frame in (directed_links, directed_nodes):
        frame["example_id"] = example_id
        frame["variant_id"] = base_variant_id

    unit_link_membership, unit_link_summary = _build_unit_link_membership(
        path_metrics=context.workflow.path_metrics,
        unit_metrics=context.workflow.unit_metrics,
    )
    unit_link_membership.to_csv(summary_dir / "unit_link_membership.csv", index=False)
    if not unit_link_summary.empty:
        directed_links = directed_links.merge(unit_link_summary, on="id_link", how="left", validate="1:1")
    for column, default in (
        ("unit_ids", ""),
        ("n_units", 0),
        ("root_unit_ids", ""),
        ("collapse_levels", ""),
        ("compound_unit_ids", ""),
        ("compound_bubble_ids", ""),
        ("unit_classes", ""),
        ("unit_topodynamic_classes", ""),
    ):
        if column not in directed_links.columns:
            directed_links[column] = default
        elif column != "n_units":
            directed_links[column] = directed_links[column].astype("string").fillna(default)
    directed_links["n_units"] = pd.to_numeric(directed_links["n_units"], errors="coerce").fillna(0).astype(int)
    directed_links = _normalize_numeric_id_columns(directed_links)

    bank_labels_path = mask_dir / "bank_labels.tif"
    removed_bank_labels = write_state_bank_labels_windowed(
        boundary_component_labels_path=resolved_boundary_labels,
        origin_mask_path=origin_mask_path,
        output_path=bank_labels_path,
    )

    source_width_samples = pd.read_csv(sample_path)
    source_width_samples.to_csv(width_dir / "panelized_source_width_samples.csv", index=False)
    enriched_links, link_width_samples = compute_panelized_base_width_families(
        directed_links,
        source_width_samples,
        bank_labels_path=bank_labels_path,
        origin_mask_path=origin_mask_path,
        bank_candidate_neighbors_max=bank_candidate_neighbors_max,
    )
    enriched_links, base_width_review_provenance = attach_permanent_base_review_provenance(
        enriched_links,
        reviewed_links_path=context.reviewed_links_path,
        reviewed_nodes_path=context.reviewed_nodes_path,
        cleaned_mask_path=context.cleaned_mask_path,
    )
    enriched_links = _apply_collapsed_selection_metadata(enriched_links, unit_ids=[], group_label=None)
    enriched_links = _normalize_numeric_id_columns(enriched_links)
    enriched_links["example_id"] = example_id
    enriched_links["variant_id"] = base_variant_id
    link_width_samples = _apply_collapsed_selection_metadata(link_width_samples, unit_ids=[], group_label=None)
    link_width_samples = _normalize_numeric_id_columns(link_width_samples)
    link_width_samples["example_id"] = example_id
    link_width_samples["variant_id"] = base_variant_id
    enriched_links, routing_width_audit, routing_width_application = apply_routing_width_model(
        enriched_links,
        link_width_samples,
        directed_nodes,
        example_id=example_id,
        variant_id=base_variant_id,
        apply_known_base_exclusions=True,
    )
    routing_width_paths = write_routing_width_audit(
        routing_width_audit,
        routing_width_application,
        output_dir=width_dir,
    )

    node_match = _apply_collapsed_selection_metadata(
        _build_identity_node_match(reviewed_nodes), unit_ids=[], group_label=None
    )
    link_match = _apply_collapsed_selection_metadata(
        _build_identity_link_match(reviewed_links), unit_ids=[], group_label=None
    )
    link_lineage = _apply_collapsed_selection_metadata(
        _build_identity_link_lineage(reviewed_links), unit_ids=[], group_label=None
    )
    for frame in (node_match, link_match, link_lineage):
        frame["example_id"] = example_id
        frame["variant_id"] = base_variant_id

    directed_nodes, node_sword_match, sword_match_summary = match_variant_nodes_to_sword(
        directed_nodes=directed_nodes,
        parent_nodes=reviewed_nodes,
        node_match=node_match,
        sword_node_source_path=sword_node_source_path,
        sword_wse_field=sword_wse_field,
        sword_match_tolerance=sword_match_tolerance,
        example_id=example_id,
        sword_example_station_source_path=sword_example_station_source_path,
        sword_station_match_source_path=sword_station_match_source_path,
        sword_reach_buffer_steps=sword_reach_buffer_steps,
    )
    directed_nodes = _apply_collapsed_selection_metadata(directed_nodes, unit_ids=[], group_label=None)
    directed_nodes = _normalize_numeric_id_columns(directed_nodes)
    directed_nodes["example_id"] = example_id
    directed_nodes["variant_id"] = base_variant_id
    node_sword_match = _apply_collapsed_selection_metadata(node_sword_match, unit_ids=[], group_label=None)
    node_sword_match["example_id"] = example_id
    node_sword_match["variant_id"] = base_variant_id

    directed_links_path = directed_dir / f"{base_run_name}_directed_links.gpkg"
    directed_nodes_path = directed_dir / f"{base_run_name}_directed_nodes.gpkg"
    links_with_widths_path = width_dir / "links_with_width_families.gpkg"

    empty_components = _apply_collapsed_selection_metadata(
        pd.DataFrame(
            columns=[
                "component_id",
                "unit_ids",
                "link_ids",
                "node_ids",
                "compound_bubble_ids",
                "n_units",
                "n_links",
                "n_nodes",
                "footprint_area",
                "added_pixels",
                "added_area",
            ]
        ),
        unit_ids=[],
        group_label=None,
    )
    empty_components["example_id"] = example_id
    empty_components["variant_id"] = base_variant_id
    empty_edit_geometries = gpd.GeoDataFrame(
        {
            "component_id": pd.Series(dtype="string"),
            "geometry_role": pd.Series(dtype="string"),
            "action": pd.Series(dtype="string"),
            "unit_ids": pd.Series(dtype="string"),
            "collapsed_selection_type": pd.Series(dtype="string"),
            "collapsed_selection_label": pd.Series(dtype="string"),
            "collapsed_unit_ids": pd.Series(dtype="string"),
            "collapsed_group_label": pd.Series(dtype="string"),
            "collapsed_unit_count": pd.Series(dtype="int64"),
            "example_id": pd.Series(dtype="string"),
            "variant_id": pd.Series(dtype="string"),
        },
        geometry=gpd.GeoSeries([], crs=reviewed_links.crs),
        crs=reviewed_links.crs,
    )

    empty_components.to_csv(summary_dir / "collapse_components.csv", index=False)
    empty_edit_geometries.to_file(mask_dir / "collapse_edit_geometries.gpkg", driver="GPKG")
    pd.DataFrame(enriched_links.drop(columns=enriched_links.geometry.name)).to_csv(
        width_dir / "link_width_families.csv", index=False
    )
    link_width_samples.to_csv(width_dir / "link_width_samples.csv", index=False)
    enriched_links.to_file(links_with_widths_path, driver="GPKG")
    node_match.to_csv(matching_dir / "node_match.csv", index=False)
    node_sword_match.to_csv(matching_dir / "node_sword_match.csv", index=False)
    link_match.to_csv(matching_dir / "link_match.csv", index=False)
    link_lineage.to_csv(matching_dir / "link_lineage.csv", index=False)
    directed_links.to_file(directed_links_path, driver="GPKG")
    directed_nodes.to_file(directed_nodes_path, driver="GPKG")
    with (directed_dir / "direction_validation_report.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "method": "panelized_base_reuse",
                "is_valid": True,
                "issues": [],
                "source_nodes": reviewed_nodes.loc[
                    reviewed_nodes["is_inlet"].fillna(False), "id_node"
                ].astype(int).tolist(),
                "sink_nodes": reviewed_nodes.loc[
                    reviewed_nodes["is_outlet"].fillna(False), "id_node"
                ].astype(int).tolist(),
            },
            handle,
            indent=2,
        )

    manifest = {
        "example_id": example_id,
        "variant_id": base_variant_id,
        "execution_mode": "panelized",
        "exit_sides": exit_sides,
        "collapse_corridor_mode": collapse_corridor_mode,
        "selection_source": "base_state",
        "selected_unit_ids": [],
        "group_label": None,
        "collapsed_selection": selection_metadata,
        "n_components": 0,
        "component_ids": [],
        "base_state_materialized": True,
        "mask_summary": {
            "base_water_pixels": None,
            "collapsed_water_pixels": None,
            "pixels_added": 0,
            "pixels_removed": 0,
            "changed_pixels": 0,
        },
        "sword_matching": {
            "wse_field": sword_wse_field,
            "match_tolerance": sword_match_tolerance,
            "reach_buffer_steps": int(sword_reach_buffer_steps),
            "n_matched_nodes": int(node_sword_match["sword_node_id"].notna().sum())
            if not node_sword_match.empty
            else 0,
            "n_propagated_matches": int(node_sword_match["sword_match_from_parent"].fillna(False).sum())
            if not node_sword_match.empty
            else 0,
            "candidate_scope": sword_match_summary.get("scope"),
            "candidate_region": sword_match_summary.get("candidate_region"),
            "candidate_reach_count": sword_match_summary.get("candidate_reach_count"),
            "candidate_reach_ids": sword_match_summary.get("candidate_reach_ids", []),
        },
        "output_paths": {
            "output_dir": str(variant_dir.resolve()),
            "collapsed_mask": str(collapsed_mask_path.resolve()),
            "origin_mask": str(origin_mask_path.resolve()),
            "width_scope_mask": str(origin_mask_path.resolve()),
            "bank_labels": str(bank_labels_path.resolve()),
            "boundary_component_labels": str(resolved_boundary_labels.resolve()),
            "collapse_components": str((summary_dir / "collapse_components.csv").resolve()),
            "unit_link_membership": str((summary_dir / "unit_link_membership.csv").resolve()),
            "edit_geometries": str((mask_dir / "collapse_edit_geometries.gpkg").resolve()),
            "link_width_families": str((width_dir / "link_width_families.csv").resolve()),
            "link_width_samples": str((width_dir / "link_width_samples.csv").resolve()),
            "links_with_width_families": str(links_with_widths_path.resolve()),
            "routing_width_application": routing_width_paths["link_audit"],
            "routing_width_application_manifest": routing_width_paths["manifest"],
            "panelized_source_width_samples": str(
                (width_dir / "panelized_source_width_samples.csv").resolve()
            ),
            "node_match": str((matching_dir / "node_match.csv").resolve()),
            "node_sword_match": str((matching_dir / "node_sword_match.csv").resolve()),
            "link_match": str((matching_dir / "link_match.csv").resolve()),
            "link_lineage": str((matching_dir / "link_lineage.csv").resolve()),
            "directed_links": str(directed_links_path.resolve()),
            "directed_nodes": str(directed_nodes_path.resolve()),
            "direction_validation_report": str(
                (directed_dir / "direction_validation_report.json").resolve()
            ),
        },
        "source_paths": {
            "cleaned_mask": str(context.cleaned_mask_path.resolve()),
            "reviewed_links": str(context.reviewed_links_path.resolve()),
            "reviewed_nodes": str(context.reviewed_nodes_path.resolve()),
            "panelized_width_samples": str(sample_path.resolve()),
            "boundary_component_labels": str(resolved_boundary_labels.resolve()),
        },
        "bank_label_summary": {
            "removed_land_labels": removed_bank_labels,
            "n_removed_land_labels": int(len(removed_bank_labels)),
        },
        "bank_candidate_neighbors_max": int(bank_candidate_neighbors_max),
        "base_width_review_provenance": base_width_review_provenance,
        "routing_width_application": routing_width_application,
    }
    with (summary_dir / "variant_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    return BaseStateVariantOutputs(
        output_dir=variant_dir,
        collapsed_mask_path=collapsed_mask_path,
        directed_links_path=directed_links_path,
        directed_nodes_path=directed_nodes_path,
        bank_labels_path=bank_labels_path,
        boundary_component_labels_path=resolved_boundary_labels,
    )


def _make_local_patch_child_runner(
    *,
    fallback_variant_runner: Callable[..., NetworkVariantOutputs] = generate_network_variant,
    local_variant_runner: Callable[..., NetworkVariantOutputs] = materialize_local_patch_network_variant,
    fallback_execution_mode: str = "full_raster",
) -> Callable[..., NetworkVariantOutputs]:
    """Use a bounded local patch with an explicit per-child fallback backend."""
    if fallback_execution_mode not in {"full_raster", "panelized"}:
        raise ValueError(
            "Local-patch fallback execution must be 'full_raster' or 'panelized', "
            f"not {fallback_execution_mode!r}."
        )

    def run(**kwargs: Any) -> NetworkVariantOutputs:
        output_dir = Path(kwargs["output_dir"])
        attempt_dir = output_dir / "local_patch_attempt"
        try:
            return local_variant_runner(**kwargs)
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception as exc:
            if isinstance(exc, LocalPatchFallbackRequested):
                stage = exc.stage
                code = exc.code
                reason = exc.reason
                source_audit = str(exc.audit_path.resolve())
            else:
                stage = "execution"
                code = type(exc).__name__
                reason = str(exc)
                source_audit = None
            attempt_dir.mkdir(parents=True, exist_ok=True)
            fallback_audit = {
                "schema_version": "local-patch-fallback-v2",
                "execution_mode_requested": "local_patch",
                "execution_mode_used": fallback_execution_mode,
                "fallback_stage": stage,
                "fallback_code": code,
                "fallback_reason": reason,
                "source_audit": source_audit,
                "generated_at": datetime.now(timezone.utc).isoformat(),
            }
            fallback_path = attempt_dir / f"fallback_to_{fallback_execution_mode}.json"
            fallback_path.write_text(
                json.dumps(fallback_audit, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            # A late local failure may have started publishing canonical files.
            # Remove only those generated directories; retain the attempt audit.
            for name in ("summary", "mask", "rivgraph", "matching", "directed", "widths"):
                target = output_dir / name
                if target.exists():
                    shutil.rmtree(target)
            fallback_kwargs = dict(kwargs)
            fallback_kwargs.pop("local_patch_reference_centerline_path", None)
            outputs = fallback_variant_runner(**fallback_kwargs)
            outputs.manifest.update(fallback_audit)
            outputs.manifest["local_patch_fallback_audit"] = str(fallback_path.resolve())
            manifest_path = outputs.output_dir / "summary" / "variant_manifest.json"
            manifest_path.write_text(
                json.dumps(outputs.manifest, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            return outputs

    return run


def _build_child_state(
    *,
    parent_context: ExperimentStateContext,
    selection: ExperimentSelection,
    state_id: str,
    state_dir: Path,
    exit_sides: str,
    variant_runner: Callable[..., NetworkVariantOutputs],
    variant_kwargs: Mapping[str, Any],
    unit_workflow_runner: Callable[..., UnitWorkflowOutputs],
    unit_workflow_writer: Callable[..., None],
    unit_workflow_kwargs: Mapping[str, Any],
    hydraulic_context_runner: Callable[..., StateHydraulicContextOutputs],
    hydraulic_context_kwargs: Mapping[str, Any] | None = None,
    defer_hierarchy: bool = False,
) -> ExperimentStateContext:
    state_started = time.perf_counter()
    stage_timings: dict[str, float | str] = {
        "schema_version": "collapse-state-stage-timings-v1",
        "state_id": state_id,
    }
    _reset_state_dir(state_dir)
    variant_dir = state_dir / "variant"
    variant_call_kwargs = dict(variant_kwargs)
    variant_call_kwargs.update(
        {
            "cleaned_mask_path": parent_context.cleaned_mask_path,
            "reviewed_links_path": (
                parent_context.analysis_links_path
                or parent_context.reviewed_links_path
            ),
            "reviewed_nodes_path": parent_context.reviewed_nodes_path,
            "exit_sides": exit_sides,
            "variant_id": state_id,
            "output_dir": variant_dir,
            "selection_type": selection.selection_type,
            "compound_bubble_id": selection.compound_bubble_id,
            "bubble_root_unit_id": selection.bubble_root_unit_id,
        }
    )
    if selection.selection_type == "group":
        variant_call_kwargs["workflow_output_dir"] = parent_context.hierarchy_output_dir
        variant_call_kwargs["group_label"] = selection.selected_group_label
        variant_call_kwargs["unit_ids"] = None
    else:
        variant_call_kwargs["workflow_output_dir"] = None
        variant_call_kwargs["group_label"] = None
        variant_call_kwargs["unit_ids"] = list(selection.selected_unit_ids)

    variant_started = time.perf_counter()
    variant_outputs = variant_runner(**variant_call_kwargs)
    stage_timings["variant_generation_seconds"] = time.perf_counter() - variant_started

    analysis_links_path = _variant_analysis_links_path(
        variant_output_dir=variant_outputs.output_dir,
        directed_links_path=variant_outputs.directed_links_path,
    )
    if defer_hierarchy:
        child_context = ExperimentStateContext(
            state_id=state_id,
            parent_state_id=parent_context.state_id,
            depth=parent_context.depth + 1,
            state_dir=state_dir,
            hierarchy_output_dir=state_dir / "hierarchy",
            cleaned_mask_path=Path(variant_outputs.collapsed_mask_path),
            reviewed_links_path=Path(variant_outputs.directed_links_path),
            reviewed_nodes_path=Path(variant_outputs.directed_nodes_path),
            workflow=None,
            analysis_links_path=analysis_links_path,
            selection=selection,
            hierarchy_deferred=True,
        )
        stage_timings["hierarchy_seconds"] = 0.0
        stage_timings["hierarchy_status"] = "deferred"
    else:
        hierarchy_started = time.perf_counter()
        child_context = _run_state_hierarchy(
            state_id=state_id,
            parent_state_id=parent_context.state_id,
            depth=parent_context.depth + 1,
            state_dir=state_dir,
            cleaned_mask_path=variant_outputs.collapsed_mask_path,
            reviewed_links_path=analysis_links_path,
            reviewed_nodes_path=variant_outputs.directed_nodes_path,
            directed_links_path=variant_outputs.directed_links_path,
            selection=selection,
            unit_workflow_runner=unit_workflow_runner,
            unit_workflow_writer=unit_workflow_writer,
            unit_workflow_kwargs=unit_workflow_kwargs,
        )
        stage_timings["hierarchy_seconds"] = time.perf_counter() - hierarchy_started
        stage_timings["hierarchy_status"] = "complete"
    child_context.variant_output_dir = variant_outputs.output_dir
    hydraulic_started = time.perf_counter()
    child_context = _run_state_hydraulic_context(
        context=child_context,
        hydraulic_context_runner=hydraulic_context_runner,
        hydraulic_context_kwargs=hydraulic_context_kwargs,
        parent_context=parent_context,
    )
    stage_timings["hydraulic_context_seconds"] = time.perf_counter() - hydraulic_started
    stage_timings["state_total_seconds"] = time.perf_counter() - state_started
    (state_dir / "stage_timings.json").write_text(
        json.dumps(
            {
                key: round(float(value), 6) if isinstance(value, float) else value
                for key, value in stage_timings.items()
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return child_context


def run_collapse_experiment(
    mode: str,
    *,
    cleaned_mask_path: str | Path,
    reviewed_links_path: str | Path,
    reviewed_nodes_path: str | Path,
    exit_sides: str,
    experiment_id: str | None = None,
    output_dir: str | Path | None = None,
    collapse_corridor_mode: str = "realized",
    max_steps: int | None = None,
    unit_workflow_runner: Callable[..., UnitWorkflowOutputs] = run_unit_workflow,
    unit_workflow_writer: Callable[..., None] = write_unit_workflow_outputs,
    variant_runner: Callable[..., NetworkVariantOutputs] = generate_network_variant,
    base_state_variant_materializer: Callable[..., BaseStateVariantOutputs] = _materialize_base_state_variant,
    state_hydraulic_context_runner: Callable[..., StateHydraulicContextOutputs] = generate_state_hydraulic_context,
    max_path_cutoff: int = 100,
    max_paths: int = 5000,
    pixel_width_fields: Sequence[str] | None = None,
    pixel_width_percentiles: Sequence[float] | None = None,
    use_pixel_widths_for_extremes: bool = True,
    debug_hierarchy: bool = False,
    preferred_width_field: str = "wid_adj_routing",
    footprint_buffer_scale: float = 0.5,
    all_touched: bool = True,
    allow_noop: bool = False,
    single_thread: bool = False,
    export_sword: bool = True,
    transect_scale: float = 1.5,
    min_transect_pixels: float = 5.0,
    bank_candidate_neighbors_max: int = 2048,
    match_tolerance: float | None = None,
    sword_node_source_path: str | Path | None = None,
    sword_wse_field: str | None = None,
    sword_match_tolerance: float | None = None,
    sword_example_station_source_path: str | Path | None = None,
    sword_station_match_source_path: str | Path | None = None,
    sword_reach_buffer_steps: int = 2,
    verbose_rivgraph: bool = False,
    state_hydraulic_context_kwargs: Mapping[str, Any] | None = None,
    defer_child_hierarchy: bool = False,
    equivalent_resistance_config: BubbleEquivalentResistanceConfig | None = None,
    equivalent_forcing_path: str | Path | None = None,
    equivalent_forcing_config: ForcingConfig | None = None,
    equivalent_kb_model_path: str | Path | None = None,
    equivalent_kb_width_sample_field: str = "wid_adj_routing",
    equivalent_kb_width_percentile: float | None = None,
    base_only: bool = False,
    requested_unit_ids: Sequence[int] | None = None,
    requested_compound_bubble_ids: Sequence[int] | None = None,
    execution_mode: str | Sequence[str] = "full_raster",
    panelized_width_samples_path: str | Path | None = None,
    panelized_boundary_component_labels_path: str | Path | None = None,
    boundary_component_artifact_source_dir: str | Path | None = None,
    base_state_variant_source_dir: str | Path | None = None,
    equivalent_forcing_table_runner: Callable[..., tuple[pd.DataFrame, dict[str, object]]] = prepare_forcing_table,
    equivalent_reference_kb_summary_runner: Callable[..., dict[str, Any]] = compute_reference_section_kb,
) -> CollapseExperimentOutputs:
    execution_plan = resolve_execution_backend_plan(execution_mode)
    if execution_plan.base_backend == "panelized":
        base_state_variant_materializer = _materialize_panelized_base_state_variant
    if mode not in EXPERIMENT_MODES:
        raise ValueError(f"Unsupported mode {mode!r}. Expected one of {EXPERIMENT_MODES}.")
    if requested_compound_bubble_ids is not None and mode != "compound-bubbles":
        raise ValueError("--compound-bubble-id is supported only with compound-bubbles.")
    if execution_plan.child_backend == "local_patch":
        if mode not in {"independent-units", "compound-bubbles"}:
            raise ValueError(
                "Local-patch execution currently supports only independent-units and "
                "compound-bubbles."
            )
        if collapse_corridor_mode != "anchored":
            raise ValueError(
                "Local-patch execution currently requires collapse_corridor_mode='anchored'."
            )
    if defer_child_hierarchy and mode not in {"independent-units", "compound-bubbles"}:
        raise ValueError(
            "Deferred child hierarchy is currently supported only for independent-units "
            "and compound-bubbles; sequential modes require each child hierarchy to select "
            "the next collapse."
        )
    if collapse_corridor_mode not in COLLAPSE_CORRIDOR_MODES:
        raise ValueError(
            f"Unsupported collapse_corridor_mode {collapse_corridor_mode!r}. "
            f"Expected one of {COLLAPSE_CORRIDOR_MODES}."
        )

    root_example_id = _infer_example_id(cleaned_mask_path)
    experiment_id = experiment_id or _default_experiment_id(cleaned_mask_path, mode)
    output_path = Path(output_dir) if output_dir is not None else _default_output_dir(cleaned_mask_path, experiment_id)
    boundary_component_output_dir = output_path / "boundary_components"
    if (
        base_state_variant_source_dir is not None
        and boundary_component_artifact_source_dir is None
        and panelized_boundary_component_labels_path is None
    ):
        inferred_boundary_source = _infer_boundary_component_artifact_source_dir(
            base_state_variant_source_dir
        )
        if inferred_boundary_source is None:
            raise FileNotFoundError(
                "Base-state reuse could not find the source experiment's "
                "boundary_components directory. Pass "
                "--panelized-boundary-component-labels explicitly or reuse a canonical "
                "<experiment>/states/S000_base/variant directory."
            )
        boundary_component_artifact_source_dir = inferred_boundary_source
    if (
        boundary_component_artifact_source_dir is not None
        and Path(boundary_component_artifact_source_dir).resolve()
        == boundary_component_output_dir.resolve()
    ):
        raise ValueError(
            "Boundary-component reuse source must be outside the experiment output "
            f"directory that will be reset: {boundary_component_output_dir.resolve()}"
        )
    if (
        base_state_variant_source_dir is not None
        and Path(base_state_variant_source_dir).resolve()
        == (output_path / "states" / "S000_base" / "variant").resolve()
    ):
        raise ValueError(
            "Base-state reuse source must be outside the experiment output directory "
            f"that will be reset: {Path(base_state_variant_source_dir).resolve()}"
        )
    reused_base_experiment_dir: Path | None = None
    reused_base_manifest: dict[str, Any] | None = None
    base_reuse_compatibility_validated = False
    base_reuse_hierarchy_schema_current = True
    if base_state_variant_source_dir is not None:
        reused_base_experiment_dir = _source_experiment_dir_from_base_variant(
            base_state_variant_source_dir
        )
        source_manifest_path = reused_base_experiment_dir / "experiment_manifest.json"
        if not source_manifest_path.exists():
            raise FileNotFoundError(
                "Base-state reuse requires the source experiment manifest: "
                f"{source_manifest_path}"
            )
        reused_base_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
        base_reuse_hierarchy_schema_current = (
            _hierarchy_uses_current_directed_graph_schema(
                base_state_variant_source_dir
            )
        )
        missing_base_artifacts = _missing_completed_base_state_artifacts(
            base_state_variant_source_dir
        )
        if missing_base_artifacts:
            raise FileNotFoundError(
                "Base-state reuse source is incomplete; refusing partial reuse: "
                + ", ".join(str(path) for path in missing_base_artifacts)
            )
        source_roots = reused_base_manifest.get("root_paths", {})
        requested_roots = {
            "cleaned_mask": Path(cleaned_mask_path).resolve(),
            "reviewed_links": Path(reviewed_links_path).resolve(),
            "reviewed_nodes": Path(reviewed_nodes_path).resolve(),
        }
        mismatches: list[str] = []
        for key, requested_path in requested_roots.items():
            source_value = source_roots.get(key)
            if not source_value:
                mismatches.append(f"{key}: source manifest value is missing")
            elif Path(str(source_value)).resolve() != requested_path:
                mismatches.append(
                    f"{key}: source={Path(str(source_value)).resolve()} requested={requested_path}"
                )
        if str(reused_base_manifest.get("example_id")) != str(root_example_id):
            mismatches.append(
                "example_id: "
                f"source={reused_base_manifest.get('example_id')!r} requested={root_example_id!r}"
            )
        if str(reused_base_manifest.get("exit_sides")) != str(exit_sides):
            mismatches.append(
                "exit_sides: "
                f"source={reused_base_manifest.get('exit_sides')!r} requested={exit_sides!r}"
            )
        if int(reused_base_manifest.get("n_failed_states", 0) or 0) != 0:
            mismatches.append(
                f"n_failed_states: source={reused_base_manifest.get('n_failed_states')!r} requested=0"
            )
        if int(reused_base_manifest.get("n_complete_states", 0) or 0) < 1:
            mismatches.append("source experiment has no completed base state")

        source_options = reused_base_manifest.get("options", {})
        source_hierarchy = source_options.get("hierarchy", {})
        requested_hierarchy = {
            "max_path_cutoff": max_path_cutoff,
            "max_paths": max_paths,
            "pixel_width_fields": (
                list(pixel_width_fields) if pixel_width_fields is not None else None
            ),
            "pixel_width_percentiles": (
                list(pixel_width_percentiles)
                if pixel_width_percentiles is not None
                else None
            ),
            "use_pixel_widths_for_extremes": bool(use_pixel_widths_for_extremes),
        }
        for key, requested_value in requested_hierarchy.items():
            if key in source_hierarchy and source_hierarchy.get(key) != requested_value:
                mismatches.append(
                    f"hierarchy.{key}: source={source_hierarchy.get(key)!r} "
                    f"requested={requested_value!r}"
                )

        source_variant = source_options.get("variant", {})
        requested_variant = {
            "exit_sides": exit_sides,
            "preferred_width_field": preferred_width_field,
            "transect_scale": transect_scale,
            "min_transect_pixels": min_transect_pixels,
            "bank_candidate_neighbors_max": bank_candidate_neighbors_max,
            "sword_wse_field": sword_wse_field,
            "sword_match_tolerance": sword_match_tolerance,
            "sword_reach_buffer_steps": sword_reach_buffer_steps,
        }
        for key, requested_value in requested_variant.items():
            if key in source_variant and source_variant.get(key) != requested_value:
                mismatches.append(
                    f"variant.{key}: source={source_variant.get(key)!r} "
                    f"requested={requested_value!r}"
                )

        if equivalent_resistance_config is not None:
            source_equivalent_config = source_hierarchy.get(
                "equivalent_resistance_config"
            )
            if source_equivalent_config is None:
                mismatches.append(
                    "equivalent resistance: source base has no compatible requested configuration"
                )
            else:
                comparison_fields = (
                    "resistance_law",
                    "geometry_method",
                    "kb_mode",
                    "default_chezy_cz",
                    "solver_backend",
                    "sparse_min_internal_nodes",
                )
                current_config = asdict(equivalent_resistance_config)
                for field in comparison_fields:
                    if source_equivalent_config.get(field) != current_config.get(field):
                        mismatches.append(
                            f"equivalent_resistance_config.{field}: "
                            f"source={source_equivalent_config.get(field)!r} "
                            f"requested={current_config.get(field)!r}"
                        )
                source_width_field = source_hierarchy.get(
                    "equivalent_kb_width_sample_field"
                )
                source_width_percentile = source_hierarchy.get(
                    "equivalent_kb_width_percentile"
                )
                if source_width_field != equivalent_kb_width_sample_field:
                    mismatches.append(
                        "equivalent_kb_width_sample_field: "
                        f"source={source_width_field!r} requested={equivalent_kb_width_sample_field!r}"
                    )
                if source_width_percentile != equivalent_kb_width_percentile:
                    mismatches.append(
                        "equivalent_kb_width_percentile: "
                        f"source={source_width_percentile!r} requested={equivalent_kb_width_percentile!r}"
                    )
        if mismatches:
            raise ValueError(
                "Base-state source is incompatible with the requested experiment: "
                + "; ".join(mismatches)
            )
        base_reuse_compatibility_validated = True

    # Do not reset an existing destination until every source compatibility
    # check above has passed.
    _reset_experiment_output_dir(output_path)
    states_dir = output_path / "states"
    states_dir.mkdir(parents=True, exist_ok=True)
    if boundary_component_artifact_source_dir is not None:
        boundary_component_labels_path: Path | None = _copy_boundary_component_artifacts(
            source_dir=boundary_component_artifact_source_dir,
            output_dir=boundary_component_output_dir,
            example_id=root_example_id,
        )
    else:
        boundary_component_labels_path = (
            Path(panelized_boundary_component_labels_path)
            if execution_plan.base_backend == "panelized"
            and panelized_boundary_component_labels_path is not None
            else None
        )
    if reused_base_experiment_dir is not None:
        source_reference = reused_base_experiment_dir / "reference_section"
        source_reference_complete = (
            (source_reference / "reference_section_summary.json").is_file()
            and (source_reference / "reference_kb_summary.json").is_file()
        )
        if source_reference_complete:
            _copytree_transactional(source_reference, output_path / "reference_section")
        source_solver_cache = (
            reused_base_experiment_dir / "equivalent_resistance" / "solve_cache"
        )
        if source_solver_cache.is_dir():
            _copytree_transactional(
                source_solver_cache,
                output_path / "equivalent_resistance" / "solve_cache",
            )

    active_equivalent_resistance_config = equivalent_resistance_config
    if (
        active_equivalent_resistance_config is not None
        and active_equivalent_resistance_config.reference_kb_summary is None
        and reused_base_manifest is not None
    ):
        source_hierarchy_options = reused_base_manifest.get("options", {}).get("hierarchy", {})
        source_reference_summary = source_hierarchy_options.get("equivalent_reference_kb_summary")
        if source_reference_summary is not None:
            localized_reference_summary = dict(source_reference_summary)
            local_reference_summary_path = (
                output_path / "reference_section" / "reference_section_summary.json"
            )
            if local_reference_summary_path.exists():
                localized_reference_summary["reference_section_summary_path"] = str(
                    local_reference_summary_path.resolve()
                )
            active_equivalent_resistance_config = replace(
                active_equivalent_resistance_config,
                reference_kb_summary=localized_reference_summary,
            )
    if (
        active_equivalent_resistance_config is not None
        and active_equivalent_resistance_config.solver_cache_dir is None
    ):
        active_equivalent_resistance_config = replace(
            active_equivalent_resistance_config,
            solver_cache_dir=str(
                (output_path / "equivalent_resistance" / "solve_cache").resolve()
            ),
        )
    base_equivalent_resistance_config = active_equivalent_resistance_config
    if _needs_equivalent_reference_kb_prepass(active_equivalent_resistance_config):
        base_equivalent_resistance_config = None

    unit_workflow_kwargs = {
        "max_path_cutoff": max_path_cutoff,
        "max_paths": max_paths,
        "pixel_width_fields": pixel_width_fields,
        "pixel_width_percentiles": pixel_width_percentiles,
        "use_pixel_widths_for_extremes": use_pixel_widths_for_extremes,
        "equivalent_resistance_config": base_equivalent_resistance_config,
        "debug": debug_hierarchy,
    }
    variant_kwargs = {
        "example_id": root_example_id,
        "collapse_corridor_mode": collapse_corridor_mode,
        "boundary_component_labels_path": boundary_component_labels_path,
        "boundary_component_output_dir": boundary_component_output_dir,
        "preferred_width_field": preferred_width_field,
        "footprint_buffer_scale": footprint_buffer_scale,
        "all_touched": all_touched,
        "allow_noop": allow_noop,
        "single_thread": single_thread,
        "export_sword": export_sword,
        "transect_scale": transect_scale,
        "min_transect_pixels": min_transect_pixels,
        "bank_candidate_neighbors_max": bank_candidate_neighbors_max,
        "match_tolerance": match_tolerance,
        "sword_node_source_path": sword_node_source_path,
        "sword_wse_field": sword_wse_field,
        "sword_match_tolerance": sword_match_tolerance,
        "sword_example_station_source_path": sword_example_station_source_path,
        "sword_station_match_source_path": sword_station_match_source_path,
        "sword_reach_buffer_steps": sword_reach_buffer_steps,
        "max_path_cutoff": max_path_cutoff,
        "max_paths": max_paths,
        "verbose_rivgraph": verbose_rivgraph,
    }

    base_state_id = "S000_base"
    _reset_state_dir(states_dir / base_state_id)
    reused_base_workflow = (
        _copy_completed_base_state_products(
            source_variant_dir=base_state_variant_source_dir,
            destination_state_dir=states_dir / base_state_id,
        )
        if base_state_variant_source_dir is not None
        else None
    )
    if reused_base_workflow is not None:
        base_context = ExperimentStateContext(
            state_id=base_state_id,
            parent_state_id=None,
            depth=0,
            state_dir=states_dir / base_state_id,
            hierarchy_output_dir=states_dir / base_state_id / "hierarchy",
            cleaned_mask_path=Path(cleaned_mask_path),
            reviewed_links_path=Path(reviewed_links_path),
            reviewed_nodes_path=Path(reviewed_nodes_path),
            workflow=reused_base_workflow,
            analysis_links_path=Path(reviewed_links_path),
            hydraulic_context_output_dir=states_dir / base_state_id / "hydraulic_context",
            selection=None,
        )
    else:
        base_context = _run_state_hierarchy(
            state_id=base_state_id,
            parent_state_id=None,
            depth=0,
            state_dir=states_dir / base_state_id,
            cleaned_mask_path=cleaned_mask_path,
            reviewed_links_path=reviewed_links_path,
            reviewed_nodes_path=reviewed_nodes_path,
            selection=None,
            unit_workflow_runner=unit_workflow_runner,
            unit_workflow_writer=unit_workflow_writer,
            unit_workflow_kwargs=unit_workflow_kwargs,
        )
    if base_state_variant_source_dir is not None:
        base_variant_outputs = _materialize_copied_base_state_variant(
            context=base_context,
            source_variant_dir=base_state_variant_source_dir,
            example_id=root_example_id,
            collapse_corridor_mode=collapse_corridor_mode,
            boundary_component_labels_path=boundary_component_labels_path,
        )
    else:
        base_variant_outputs = base_state_variant_materializer(
            context=base_context,
            example_id=root_example_id,
            exit_sides=exit_sides,
            collapse_corridor_mode=collapse_corridor_mode,
            preferred_width_field=preferred_width_field,
            transect_scale=transect_scale,
            min_transect_pixels=min_transect_pixels,
            sword_node_source_path=sword_node_source_path,
            sword_wse_field=sword_wse_field,
            sword_match_tolerance=sword_match_tolerance,
            sword_example_station_source_path=sword_example_station_source_path,
            sword_station_match_source_path=sword_station_match_source_path,
            sword_reach_buffer_steps=sword_reach_buffer_steps,
            boundary_component_labels_path=boundary_component_labels_path,
            boundary_component_output_dir=boundary_component_output_dir,
            bank_candidate_neighbors_max=bank_candidate_neighbors_max,
            panelized_width_samples_path=panelized_width_samples_path,
        )
    if base_variant_outputs.boundary_component_labels_path is not None:
        boundary_component_labels_path = base_variant_outputs.boundary_component_labels_path
        variant_kwargs["boundary_component_labels_path"] = boundary_component_labels_path
    base_analysis_links_path = _variant_analysis_links_path(
        variant_output_dir=base_variant_outputs.output_dir,
        directed_links_path=base_variant_outputs.directed_links_path,
    )
    if reused_base_workflow is not None and not base_reuse_hierarchy_schema_current:
        base_context = _run_state_hierarchy(
            state_id=base_state_id,
            parent_state_id=None,
            depth=0,
            state_dir=states_dir / base_state_id,
            cleaned_mask_path=base_variant_outputs.collapsed_mask_path,
            reviewed_links_path=base_analysis_links_path,
            reviewed_nodes_path=base_variant_outputs.directed_nodes_path,
            directed_links_path=base_variant_outputs.directed_links_path,
            selection=None,
            unit_workflow_runner=unit_workflow_runner,
            unit_workflow_writer=unit_workflow_writer,
            unit_workflow_kwargs=unit_workflow_kwargs,
        )
        _refresh_reused_base_unit_metadata(
            variant_outputs=base_variant_outputs,
            workflow=base_context.workflow,
        )
        reused_base_workflow = None
    elif reused_base_workflow is not None:
        base_context.cleaned_mask_path = base_variant_outputs.collapsed_mask_path
        base_context.reviewed_links_path = base_variant_outputs.directed_links_path
        base_context.reviewed_nodes_path = base_variant_outputs.directed_nodes_path
        base_context.analysis_links_path = base_analysis_links_path
    elif base_analysis_links_path != Path(base_variant_outputs.directed_links_path):
        base_context = _run_state_hierarchy(
            state_id=base_state_id,
            parent_state_id=None,
            depth=0,
            state_dir=states_dir / base_state_id,
            cleaned_mask_path=base_variant_outputs.collapsed_mask_path,
            reviewed_links_path=base_analysis_links_path,
            reviewed_nodes_path=base_variant_outputs.directed_nodes_path,
            directed_links_path=base_variant_outputs.directed_links_path,
            selection=None,
            unit_workflow_runner=unit_workflow_runner,
            unit_workflow_writer=unit_workflow_writer,
            unit_workflow_kwargs=unit_workflow_kwargs,
        )
    else:
        base_context.cleaned_mask_path = base_variant_outputs.collapsed_mask_path
        base_context.reviewed_links_path = base_variant_outputs.directed_links_path
        base_context.reviewed_nodes_path = base_variant_outputs.directed_nodes_path
        base_context.analysis_links_path = base_analysis_links_path
    base_context.variant_output_dir = base_variant_outputs.output_dir
    if reused_base_workflow is None:
        base_context = _run_state_hydraulic_context(
            context=base_context,
            hydraulic_context_runner=state_hydraulic_context_runner,
            hydraulic_context_kwargs=state_hydraulic_context_kwargs,
        )
    panelized_child_runner: Callable[..., NetworkVariantOutputs] | None = None
    reference_section_variant_builder: Callable[..., Mapping[str, Any]] | None = None
    needs_panelized_child_runner = (
        execution_plan.child_backend == "panelized"
        or execution_plan.fallback_backend == "panelized"
    )
    if needs_panelized_child_runner and variant_runner is generate_network_variant:
        child_boundary_labels = Path(panelized_boundary_component_labels_path) if panelized_boundary_component_labels_path else boundary_component_labels_path
        if child_boundary_labels is None:
            raise ValueError("Panelized child execution could not resolve base boundary-component labels.")
        run_root = Path(reviewed_links_path).resolve().parent.parent
        exits = json.loads((run_root / "panel_matching" / "panel_exit_sides.json").read_text())
        panelized_child_runner = _make_panelized_child_runner(
            source_panel_root=run_root / "panels", panel_plan_path=run_root / "panels" / "panel_plan.gpkg",
            panel_breaks_path=run_root / "manual_edits" / f"{root_example_id}_panel_breaks.gpkg",
            boundary_match_plan_path=run_root / "panel_matching" / "boundary_match_plan.json",
            parent_merged_links_path=run_root / "merged_network" / "core_boundary" / "merged_links.gpkg",
            boundary_component_labels_path=child_boundary_labels,
            exit_sides_by_panel={key: value["exit_sides"] for key, value in exits.items()},
            run_name=root_example_id,
        )
        variant_kwargs["panelized_path_metrics"] = base_context.hierarchy_output_dir / "path_metrics.csv"
        if execution_plan.base_backend == "panelized":
            reference_section_variant_builder = _make_panelized_reference_section_builder(
                panelized_variant_runner=panelized_child_runner,
                base_context=base_context,
                base_variant_outputs=base_variant_outputs,
                base_analysis_links_path=base_analysis_links_path,
                example_id=root_example_id,
                exit_sides=exit_sides,
                preferred_width_field=preferred_width_field,
                footprint_buffer_scale=footprint_buffer_scale,
                all_touched=all_touched,
                bank_candidate_neighbors_max=bank_candidate_neighbors_max,
                sword_node_source_path=sword_node_source_path,
                sword_wse_field=sword_wse_field,
                sword_match_tolerance=sword_match_tolerance,
                sword_example_station_source_path=sword_example_station_source_path,
                sword_station_match_source_path=sword_station_match_source_path,
                sword_reach_buffer_steps=sword_reach_buffer_steps,
            )
    if (
        execution_plan.child_backend == "panelized"
        and panelized_child_runner is not None
        and variant_runner is generate_network_variant
    ):
        variant_runner = panelized_child_runner
    elif execution_plan.child_backend == "local_patch" and not base_only and variant_runner is generate_network_variant:
        reference_centerline_path = Path(reviewed_links_path).with_name(
            f"{root_example_id}_centerline.gpkg"
        )
        if not reference_centerline_path.exists():
            raise FileNotFoundError(
                "Local-patch execution requires the run-level single centerline beside "
                f"the reviewed links: {reference_centerline_path}"
        )
        variant_kwargs["local_patch_reference_centerline_path"] = reference_centerline_path
        if execution_plan.is_hybrid_panelized_local_patch:
            variant_kwargs["local_patch_mask_publication"] = "virtual_patch"
            variant_kwargs["panelized_mask_publication"] = "virtual_patch"
        fallback_runner = (
            panelized_child_runner
            if execution_plan.fallback_backend == "panelized"
            else generate_network_variant
        )
        if fallback_runner is None:
            raise RuntimeError("Local-patch execution could not resolve its fallback backend.")
        variant_runner = _make_local_patch_child_runner(
            fallback_variant_runner=fallback_runner,
            fallback_execution_mode=str(execution_plan.fallback_backend),
        )

    state_records: list[dict[str, Any]] = [_state_record(base_context, mode)]
    transition_records: list[dict[str, Any]] = []
    equivalent_transition_records: list[dict[str, Any]] = []
    equivalent_reference_kb_summary: dict[str, Any] | None = (
        dict(active_equivalent_resistance_config.reference_kb_summary)
        if active_equivalent_resistance_config is not None
        and active_equivalent_resistance_config.reference_kb_summary is not None
        else None
    )
    equivalent_forcing_metadata: dict[str, object] | None = None
    reference_kb_prepass_required = _needs_equivalent_reference_kb_prepass(
        active_equivalent_resistance_config
    )
    stop_reason = (
        "reference_kb_prepass_pending"
        if reference_kb_prepass_required
        else "base_state_only"
    )
    terminal_state_id = base_context.state_id

    def write_metadata() -> None:
        n_complete_states = sum(1 for record in state_records if record.get("status") == "complete")
        n_hierarchy_deferred_states = sum(
            1
            for record in state_records
            if record.get("status") == "hydraulic_complete_hierarchy_deferred"
        )
        n_failed_states = sum(1 for record in state_records if record.get("status") == "failed")
        n_compound_states = sum(1 for record in state_records if bool(record.get("is_compound_state", False)))
        n_equivalent_transition_rows = len(equivalent_transition_records)
        n_equivalent_transitions_with_matches = sum(
            1
            for record in transition_records
            if _record_int_or_zero(record, "n_equivalent_bubbles_matched") > 0
        )
        manifest = {
            "experiment_id": experiment_id,
            "mode": mode,
            "example_id": _infer_example_id(cleaned_mask_path),
            "root_example_id": root_example_id,
            "exit_sides": exit_sides,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "code_version": _git_revision(),
            "execution_backends": execution_plan.as_manifest(),
            "root_paths": {
                "cleaned_mask": str(Path(cleaned_mask_path).resolve()),
                "reviewed_links": str(Path(reviewed_links_path).resolve()),
                "reviewed_nodes": str(Path(reviewed_nodes_path).resolve()),
                "boundary_component_labels": (
                    str(boundary_component_labels_path.resolve())
                    if boundary_component_labels_path is not None
                    else None
                ),
            },
            "max_steps": max_steps,
            "n_states": len(state_records),
            "n_complete_states": n_complete_states,
            "n_hierarchy_deferred_states": n_hierarchy_deferred_states,
            "n_failed_states": n_failed_states,
            "n_compound_states": n_compound_states,
            "n_transitions": len(transition_records),
            "n_equivalent_transition_rows": n_equivalent_transition_rows,
            "n_equivalent_transitions_with_matches": n_equivalent_transitions_with_matches,
            "terminal_state_id": terminal_state_id,
            "reference_kb_prepass": {
                "required": bool(reference_kb_prepass_required),
                "complete": bool(
                    not reference_kb_prepass_required
                    or equivalent_reference_kb_summary is not None
                ),
                "construction_backend": (
                    execution_plan.base_backend
                    if reference_kb_prepass_required
                    else None
                ),
            },
            "stop_reason": stop_reason,
            "base_state_reuse": {
                "requested": base_state_variant_source_dir is not None,
                "source_experiment_dir": (
                    str(reused_base_experiment_dir)
                    if reused_base_experiment_dir is not None
                    else None
                ),
                "compatibility_validated": bool(base_reuse_compatibility_validated),
                "variant_reused": base_state_variant_source_dir is not None,
                "hierarchy_reused": reused_base_workflow is not None,
                "hydraulic_context_reused": reused_base_workflow is not None,
                "hierarchy_schema_current": bool(
                    base_reuse_hierarchy_schema_current
                ),
                "hierarchy_recomputed_for_schema": bool(
                    base_state_variant_source_dir is not None
                    and not base_reuse_hierarchy_schema_current
                ),
                "boundary_components_reused": boundary_component_artifact_source_dir is not None,
                "solver_cache_seeded": bool(
                    reused_base_experiment_dir is not None
                    and (
                        reused_base_experiment_dir
                        / "equivalent_resistance"
                        / "solve_cache"
                    ).is_dir()
                ),
                "audit_manifest": (
                    str(
                        (
                            output_path
                            / "states"
                            / "S000_base"
                            / "base_state_reuse_manifest.json"
                        ).resolve()
                    )
                    if base_state_variant_source_dir is not None
                    else None
                ),
            },
            "options": {
                "hierarchy": {
                    "defer_child_hierarchy": bool(defer_child_hierarchy),
                    "max_path_cutoff": max_path_cutoff,
                    "max_paths": max_paths,
                    "pixel_width_fields": list(pixel_width_fields) if pixel_width_fields is not None else None,
                    "pixel_width_percentiles": list(pixel_width_percentiles) if pixel_width_percentiles is not None else None,
                    "use_pixel_widths_for_extremes": bool(use_pixel_widths_for_extremes),
                    "equivalent_resistance_config": (
                        asdict(active_equivalent_resistance_config)
                        if active_equivalent_resistance_config is not None
                        else None
                    ),
                    "equivalent_reference_kb_summary": equivalent_reference_kb_summary,
                    "equivalent_forcing_path": (
                        str(Path(equivalent_forcing_path).expanduser().resolve())
                        if equivalent_forcing_path is not None
                        else None
                    ),
                    "equivalent_forcing_config": (
                        asdict(equivalent_forcing_config)
                        if equivalent_forcing_config is not None
                        else None
                    ),
                    "equivalent_forcing_metadata": equivalent_forcing_metadata,
                    "equivalent_kb_model_path": (
                        str(Path(equivalent_kb_model_path).expanduser().resolve())
                        if equivalent_kb_model_path is not None
                        else None
                    ),
                    "equivalent_kb_width_sample_field": equivalent_kb_width_sample_field,
                    "equivalent_kb_width_percentile": (
                        float(equivalent_kb_width_percentile)
                        if equivalent_kb_width_percentile is not None
                        else None
                    ),
                    "debug_hierarchy": bool(debug_hierarchy),
                },
            "variant": {
                "execution_backends": execution_plan.as_manifest(),
                "base_only": bool(base_only),
                "requested_unit_ids": (
                    [int(unit_id) for unit_id in requested_unit_ids]
                    if requested_unit_ids is not None
                    else None
                ),
                "requested_compound_bubble_ids": (
                    [int(bubble_id) for bubble_id in requested_compound_bubble_ids]
                    if requested_compound_bubble_ids is not None
                    else None
                ),
                "collapse_corridor_mode": collapse_corridor_mode,
                    "exit_sides": exit_sides,
                    "preferred_width_field": preferred_width_field,
                    "footprint_buffer_scale": float(footprint_buffer_scale),
                    "all_touched": bool(all_touched),
                    "allow_noop": bool(allow_noop),
                    "single_thread": bool(single_thread),
                    "export_sword": bool(export_sword),
                    "transect_scale": float(transect_scale),
                    "min_transect_pixels": float(min_transect_pixels),
                    "bank_candidate_neighbors_max": int(bank_candidate_neighbors_max),
                    "match_tolerance": match_tolerance,
                    "sword_node_source_path": str(Path(sword_node_source_path).resolve()) if sword_node_source_path is not None else None,
                    "sword_wse_field": sword_wse_field,
                    "sword_match_tolerance": sword_match_tolerance,
                    "sword_example_station_source_path": (
                        str(Path(sword_example_station_source_path).resolve())
                        if sword_example_station_source_path is not None
                        else None
                    ),
                    "sword_station_match_source_path": (
                        str(Path(sword_station_match_source_path).resolve())
                        if sword_station_match_source_path is not None
                        else None
                    ),
                    "sword_reach_buffer_steps": int(sword_reach_buffer_steps),
                    "verbose_rivgraph": bool(verbose_rivgraph),
                },
            },
        }
        _write_experiment_outputs(
            output_path,
            state_records=state_records,
            transition_records=transition_records,
            equivalent_transition_records=equivalent_transition_records,
            manifest=manifest,
        )

    write_metadata()

    if _needs_equivalent_reference_kb_prepass(active_equivalent_resistance_config):
        assert active_equivalent_resistance_config is not None
        active_equivalent_resistance_config, equivalent_reference_kb_summary, equivalent_forcing_metadata = (
            _resolve_equivalent_reference_kb_summary(
                experiment_dir=output_path,
                equivalent_resistance_config=active_equivalent_resistance_config,
                equivalent_forcing_path=equivalent_forcing_path,
                equivalent_forcing_config=equivalent_forcing_config,
                equivalent_kb_model_path=equivalent_kb_model_path,
                equivalent_kb_width_sample_field=equivalent_kb_width_sample_field,
                equivalent_kb_width_percentile=equivalent_kb_width_percentile,
                equivalent_forcing_table_runner=equivalent_forcing_table_runner,
                equivalent_reference_kb_summary_runner=equivalent_reference_kb_summary_runner,
                reference_section_variant_builder=reference_section_variant_builder,
            )
        )
        unit_workflow_kwargs["equivalent_resistance_config"] = active_equivalent_resistance_config
        base_context.workflow = unit_workflow_runner(
            base_context.analysis_links_path or base_context.reviewed_links_path,
            base_context.reviewed_nodes_path,
            **dict(unit_workflow_kwargs),
        )
        unit_workflow_writer(
            base_context.hierarchy_output_dir,
            base_context.workflow,
            manifest_overrides={
                "state_id": base_context.state_id,
                "parent_state_id": base_context.parent_state_id,
                "depth": int(base_context.depth),
            },
        )
        state_records[0] = _state_record(base_context, mode)
        stop_reason = "base_state_only"
        write_metadata()

    if base_only:
        stop_reason = "base_only_requested"
        terminal_state_id = base_context.state_id
        write_metadata()
        return CollapseExperimentOutputs(
            state_registry=pd.DataFrame.from_records(state_records),
            transition_registry=pd.DataFrame.from_records(transition_records),
            equivalent_transition_registry=_equivalent_transition_frame(equivalent_transition_records),
            output_dir=output_path,
            manifest=json.loads((output_path / "experiment_manifest.json").read_text(encoding="utf-8")),
        )

    if mode in {"independent-units", "compound-bubbles"}:
        if mode == "compound-bubbles":
            if requested_unit_ids is not None:
                raise ValueError("--unit-id is currently supported only with independent-units.")
            selections = _select_compound_bubbles_from_workflow(base_context.workflow)
            if requested_compound_bubble_ids is not None:
                requested_bubble_ids = {
                    int(bubble_id) for bubble_id in requested_compound_bubble_ids
                }
                available_bubble_ids = {
                    int(selection.compound_bubble_id)
                    for selection in selections
                    if selection.compound_bubble_id is not None
                }
                unknown_bubble_ids = sorted(requested_bubble_ids - available_bubble_ids)
                if unknown_bubble_ids:
                    raise ValueError(
                        "Requested compound bubble ID(s) are not present in the base state: "
                        f"{unknown_bubble_ids}."
                    )
                selections = [
                    selection
                    for selection in selections
                    if selection.compound_bubble_id in requested_bubble_ids
                ]
        else:
            selections = _select_all_units_from_ranking(base_context.workflow.collapse_ranking, base_context.workflow.unit_metrics)
            if requested_unit_ids is not None:
                requested_ids = {int(unit_id) for unit_id in requested_unit_ids}
                available_ids = {selection.selected_unit_ids[0] for selection in selections}
                unknown_ids = sorted(requested_ids - available_ids)
                if unknown_ids:
                    raise ValueError(
                        "Requested unit ID(s) are not present in the base-state collapse ranking: "
                        f"{unknown_ids}."
                    )
                selections = [
                    selection
                    for selection in selections
                    if selection.selected_unit_ids[0] in requested_ids
                ]
        if not selections:
            stop_reason = "no_compound_bubbles_in_base_state" if mode == "compound-bubbles" else "no_units_in_base_state"
            write_metadata()
            return CollapseExperimentOutputs(
                state_registry=pd.DataFrame.from_records(state_records),
                transition_registry=pd.DataFrame.from_records(transition_records),
                equivalent_transition_registry=_equivalent_transition_frame(equivalent_transition_records),
                output_dir=output_path,
                manifest=json.loads((output_path / "experiment_manifest.json").read_text(encoding="utf-8")),
            )

        for index, selection in enumerate(selections, start=1):
            prefix = "CB" if mode == "compound-bubbles" else "S"
            state_id = f"{prefix}{index:03d}_{selection.state_suffix}"
            state_dir = states_dir / state_id
            try:
                child_context = _build_child_state(
                    parent_context=base_context,
                    selection=selection,
                    state_id=state_id,
                    state_dir=state_dir,
                    exit_sides=exit_sides,
                    variant_runner=variant_runner,
                    variant_kwargs=variant_kwargs,
                    unit_workflow_runner=unit_workflow_runner,
                    unit_workflow_writer=unit_workflow_writer,
                    unit_workflow_kwargs=unit_workflow_kwargs,
                    hydraulic_context_runner=state_hydraulic_context_runner,
                    hydraulic_context_kwargs=state_hydraulic_context_kwargs,
                    defer_hierarchy=defer_child_hierarchy,
                )
            except VariantDirectionValidationError as exc:
                error_text = _format_exception(exc)
                error_traceback = _format_traceback(exc)
                state_records.append(
                    _failed_state_record(
                        state_id=state_id,
                        parent_state_id=base_context.state_id,
                        depth=base_context.depth + 1,
                        mode=mode,
                        state_dir=state_dir,
                        selection=selection,
                        error_stage="direction_validation",
                        error=error_text,
                        error_traceback=error_traceback,
                        variant_outputs=exc.outputs,
                    )
                )
                transition = _failed_transition_record(
                    transition_id=f"T{index:03d}",
                    step_index=index,
                    parent_context=base_context,
                    selection=selection,
                    child_state_id=state_id,
                    error_stage="direction_validation",
                    error=error_text,
                    error_traceback=error_traceback,
                    variant_outputs=exc.outputs,
                )
                transition["mode"] = mode
                transition_records.append(transition)
                terminal_state_id = state_id
                write_metadata()
                continue
            except Exception as exc:
                error_text = _format_exception(exc)
                error_traceback = _format_traceback(exc)
                state_records.append(
                    _failed_state_record(
                        state_id=state_id,
                        parent_state_id=base_context.state_id,
                        depth=base_context.depth + 1,
                        mode=mode,
                        state_dir=state_dir,
                        selection=selection,
                        error_stage="child_state_build",
                        error=error_text,
                        error_traceback=error_traceback,
                    )
                )
                transition = _failed_transition_record(
                    transition_id=f"T{index:03d}",
                    step_index=index,
                    parent_context=base_context,
                    selection=selection,
                    child_state_id=state_id,
                    error_stage="child_state_build",
                    error=error_text,
                    error_traceback=error_traceback,
                )
                transition["mode"] = mode
                transition_records.append(transition)
                terminal_state_id = state_id
                write_metadata()
                continue

            transition_id = f"T{index:03d}"
            if not child_context.hierarchy_deferred:
                equivalent_transition = _write_equivalent_transition_outputs(
                    transition_id=transition_id,
                    parent_context=base_context,
                    child_context=child_context,
                    equivalent_resistance_config=(
                        active_equivalent_resistance_config
                        or BubbleEquivalentResistanceConfig()
                    ),
                )
                if not equivalent_transition.empty:
                    equivalent_transition_records.extend(equivalent_transition.to_dict("records"))
            state_records.append(_state_record(child_context, mode))
            transition = _transition_record(
                transition_id=transition_id,
                step_index=index,
                parent_context=base_context,
                child_context=child_context,
            )
            transition["mode"] = mode
            transition_records.append(transition)
            terminal_state_id = child_context.state_id
            write_metadata()

        stop_reason = "all_compound_bubbles_processed" if mode == "compound-bubbles" else "all_base_units_processed"
        write_metadata()
        return CollapseExperimentOutputs(
            state_registry=pd.DataFrame.from_records(state_records),
            transition_registry=pd.DataFrame.from_records(transition_records),
            equivalent_transition_registry=_equivalent_transition_frame(equivalent_transition_records),
            output_dir=output_path,
            manifest=json.loads((output_path / "experiment_manifest.json").read_text(encoding="utf-8")),
        )

    current_context = base_context
    step_index = 0

    while True:
        if max_steps is not None and step_index >= max_steps:
            stop_reason = "max_steps_reached"
            break

        if mode == "sequential-units":
            selection = _select_unit_from_ranking(current_context.workflow.collapse_ranking)
            if selection is None:
                stop_reason = "no_units_remaining"
                break
        else:
            selection = _select_group_from_selected_groups(current_context.workflow.selected_groups)
            if selection is None:
                selection = _select_unit_from_ranking(current_context.workflow.collapse_ranking)
                if selection is None:
                    stop_reason = "no_units_remaining"
                    break

        step_index += 1
        state_id = f"S{step_index:03d}_{selection.state_suffix}"
        state_dir = states_dir / state_id
        try:
            child_context = _build_child_state(
                parent_context=current_context,
                selection=selection,
                state_id=state_id,
                state_dir=state_dir,
                exit_sides=exit_sides,
                variant_runner=variant_runner,
                variant_kwargs=variant_kwargs,
                unit_workflow_runner=unit_workflow_runner,
                unit_workflow_writer=unit_workflow_writer,
                unit_workflow_kwargs=unit_workflow_kwargs,
                hydraulic_context_runner=state_hydraulic_context_runner,
                hydraulic_context_kwargs=state_hydraulic_context_kwargs,
                defer_hierarchy=False,
            )
        except VariantDirectionValidationError as exc:
            error_text = _format_exception(exc)
            error_traceback = _format_traceback(exc)
            state_records.append(
                _failed_state_record(
                    state_id=state_id,
                    parent_state_id=current_context.state_id,
                    depth=current_context.depth + 1,
                    mode=mode,
                    state_dir=state_dir,
                    selection=selection,
                    error_stage="direction_validation",
                    error=error_text,
                    error_traceback=error_traceback,
                    variant_outputs=exc.outputs,
                )
            )
            transition = _failed_transition_record(
                transition_id=f"T{step_index:03d}",
                step_index=step_index,
                parent_context=current_context,
                selection=selection,
                child_state_id=state_id,
                error_stage="direction_validation",
                error=error_text,
                error_traceback=error_traceback,
                variant_outputs=exc.outputs,
            )
            transition["mode"] = mode
            transition_records.append(transition)
            terminal_state_id = state_id
            stop_reason = "state_failed"
            write_metadata()
            break
        except Exception as exc:
            error_text = _format_exception(exc)
            error_traceback = _format_traceback(exc)
            state_records.append(
                _failed_state_record(
                    state_id=state_id,
                    parent_state_id=current_context.state_id,
                    depth=current_context.depth + 1,
                    mode=mode,
                    state_dir=state_dir,
                    selection=selection,
                    error_stage="child_state_build",
                    error=error_text,
                    error_traceback=error_traceback,
                )
            )
            transition = _failed_transition_record(
                transition_id=f"T{step_index:03d}",
                step_index=step_index,
                parent_context=current_context,
                selection=selection,
                child_state_id=state_id,
                error_stage="child_state_build",
                error=error_text,
                error_traceback=error_traceback,
            )
            transition["mode"] = mode
            transition_records.append(transition)
            terminal_state_id = state_id
            stop_reason = "state_failed"
            write_metadata()
            break

        transition_id = f"T{step_index:03d}"
        equivalent_transition = _write_equivalent_transition_outputs(
            transition_id=transition_id,
            parent_context=current_context,
            child_context=child_context,
            equivalent_resistance_config=(
                active_equivalent_resistance_config
                or BubbleEquivalentResistanceConfig()
            ),
        )
        if not equivalent_transition.empty:
            equivalent_transition_records.extend(equivalent_transition.to_dict("records"))
        state_records.append(_state_record(child_context, mode))
        transition = _transition_record(
            transition_id=transition_id,
            step_index=step_index,
            parent_context=current_context,
            child_context=child_context,
        )
        transition["mode"] = mode
        transition_records.append(transition)
        current_context = child_context
        terminal_state_id = child_context.state_id
        write_metadata()

    write_metadata()
    return CollapseExperimentOutputs(
        state_registry=pd.DataFrame.from_records(state_records),
        transition_registry=pd.DataFrame.from_records(transition_records),
        equivalent_transition_registry=_equivalent_transition_frame(equivalent_transition_records),
        output_dir=output_path,
        manifest=json.loads((output_path / "experiment_manifest.json").read_text(encoding="utf-8")),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run one of the collapse experiment workflows over a graph-like river network: "
            "independent base-unit variants, whole-compound-bubble variants, "
            "adaptive sequential unit collapse, or adaptive sequential group collapse."
        )
    )
    parser.add_argument("mode", choices=EXPERIMENT_MODES, help="Experiment mode to run.")
    parser.add_argument("--cleaned-mask", required=True, help="Base cleaned mask GeoTIFF.")
    parser.add_argument("--reviewed-links", required=True, help="Base reviewed/directed links GeoPackage.")
    parser.add_argument("--reviewed-nodes", required=True, help="Base reviewed/directed nodes GeoPackage.")
    parser.add_argument("--exit-sides", required=True, help="River exit sides for RivGraph, e.g. NS or EW.")
    parser.add_argument("--experiment-id", default=None, help="Optional experiment identifier.")
    parser.add_argument("--output-dir", default=None, help="Optional experiment output directory.")
    parser.add_argument(
        "--collapse-corridor-mode",
        choices=COLLAPSE_CORRIDOR_MODE_CHOICES,
        default="realized",
        help=(
            "How to materialize collapsed corridors: keep the full realized corridor, "
            "split it back to the nominal selection anchors, or run both experiment variants."
        ),
    )
    parser.add_argument("--max-steps", type=int, default=None, help="Optional maximum number of sequential collapse steps.")
    parser.add_argument(
        "--unit-id",
        dest="requested_unit_ids",
        action="append",
        type=int,
        default=None,
        help="Optional independent-units filter; repeat to run several unit IDs. Omit to run all units.",
    )
    parser.add_argument(
        "--compound-bubble-id",
        dest="requested_compound_bubble_ids",
        action="append",
        type=int,
        default=None,
        help=(
            "Optional compound-bubbles filter; repeat to run several bubble IDs. "
            "Omit to run all compound bubbles."
        ),
    )
    parser.add_argument("--max-path-cutoff", type=int, default=100, help="Maximum edge-count cutoff for unit path enumeration.")
    parser.add_argument("--max-paths", type=int, default=5000, help="Maximum number of simple paths per unit.")
    parser.add_argument(
        "--pixel-width-fields",
        nargs="*",
        default=None,
        help="Optional candidate field names containing per-link width samples.",
    )
    parser.add_argument(
        "--pixel-width-percentiles",
        nargs="*",
        type=float,
        default=None,
        help="Optional percentile set for width diagnostics, e.g. 5 50 95.",
    )
    parser.add_argument(
        "--disable-pixel-width-extremes",
        action="store_true",
        help="Force path width min/max diagnostics to use representative widths instead of pixel samples.",
    )
    parser.add_argument(
        "--compute-equivalent-resistance",
        action="store_true",
        help="Also compute geometry-informed equivalent resistance/conductance for each state hierarchy.",
    )
    parser.add_argument(
        "--equivalent-resistance-law",
        choices=("manning", "chezy"),
        default="manning",
        help="Resistance law for equivalent-resistance edge K values.",
    )
    parser.add_argument(
        "--equivalent-geometry-method",
        choices=("wide", "rectangular"),
        default="wide",
        help="Hydraulic geometry method for equivalent-resistance edge K values.",
    )
    parser.add_argument(
        "--equivalent-kb-mode",
        choices=("fixed", "based_reference_section"),
        default="fixed",
        help="How to select kb for the equivalent-resistance workflow.",
    )
    parser.add_argument(
        "--equivalent-kb-value",
        type=float,
        default=20.0,
        help="Fixed width-depth ratio used when no per-edge kb field is available.",
    )
    parser.add_argument(
        "--equivalent-kb-model-path",
        default=None,
        help="Optional local path to the BASED model .ubj file when equivalent-kb-mode=based_reference_section.",
    )
    parser.add_argument(
        "--equivalent-kb-width-sample-field",
        default="wid_adj_routing",
        help="Reference-section width field used when equivalent-kb-mode=based_reference_section. Summary width fields on the reference link are preferred over raw sample percentiles when available.",
    )
    parser.add_argument(
        "--equivalent-kb-width-percentile",
        type=float,
        default=None,
        help=(
            "Reference-section width percentile used only when the selected equivalent kb width "
            "field is read from raw sample widths. Omit it for summary fields such as wid_adj_routing."
        ),
    )
    parser.add_argument(
        "--equivalent-length-field",
        choices=LENGTH_FIELD_CANDIDATES,
        default=LENGTH_FIELD_CANDIDATES[0],
        help="Preferred link-length field for equivalent-resistance solves.",
    )
    parser.add_argument(
        "--equivalent-manning-n",
        type=float,
        default=0.035,
        help="Default Manning n used when the links do not provide one.",
    )
    parser.add_argument(
        "--equivalent-chezy-cz",
        type=float,
        default=22.0,
        help="Default Chezy Cz used when the links do not provide one.",
    )
    parser.add_argument(
        "--equivalent-solver-backend",
        choices=(
            "dense_legacy", "sparse", "sparse_least_squares",
            "energy_sparse", "auto",
        ),
        default="dense_legacy",
        help=(
            "Equivalent-resistance solver backend. dense_legacy preserves the "
            "previous dense-SVD path; sparse/sparse_least_squares use LSMR; "
            "energy_sparse uses convex energy minimization plus LSMR polishing; "
            "auto uses energy_sparse for larger domains."
        ),
    )
    parser.add_argument(
        "--equivalent-sparse-min-internal-nodes",
        type=int,
        default=64,
        help="Internal-node threshold at which auto selects energy_sparse.",
    )
    parser.add_argument(
        "--equivalent-forcing-path",
        default=None,
        help="CSV or parquet file with time/discharge forcing used when equivalent-kb-mode=based_reference_section.",
    )
    parser.add_argument(
        "--equivalent-forcing-station-key",
        default=None,
        help="Explicit station_key to extract from a multi-station forcing source for based_reference_section kb.",
    )
    parser.add_argument(
        "--equivalent-station-key-column",
        default="station_key",
        help="Station-key column in the equivalent-resistance forcing file.",
    )
    parser.add_argument(
        "--equivalent-forcing-start-time",
        default=None,
        help="Optional inclusive UTC truncation start for equivalent-resistance forcing normalization.",
    )
    parser.add_argument(
        "--equivalent-forcing-end-time",
        default=None,
        help="Optional inclusive UTC truncation end for equivalent-resistance forcing normalization.",
    )
    equivalent_forcing_resample_group = parser.add_mutually_exclusive_group()
    equivalent_forcing_resample_group.add_argument(
        "--equivalent-forcing-resample-minutes",
        type=int,
        default=None,
        help="Optional normalized forcing interval in minutes for equivalent-resistance kb preparation.",
    )
    equivalent_forcing_resample_group.add_argument(
        "--equivalent-forcing-resample-seconds",
        type=int,
        default=None,
        help="Optional normalized forcing interval in seconds for equivalent-resistance kb preparation.",
    )
    parser.add_argument(
        "--equivalent-forcing-output-cache-dir",
        default=None,
        help="Optional cache directory for normalized equivalent-resistance forcing tables.",
    )
    parser.add_argument(
        "--equivalent-time-column",
        default="time",
        help="Timestamp column in the equivalent-resistance forcing file.",
    )
    parser.add_argument(
        "--equivalent-discharge-column",
        default="discharge",
        help="Discharge column in the equivalent-resistance forcing file.",
    )
    parser.add_argument("--debug-hierarchy", action="store_true", help="Enable hierarchy/unit-detection debug output.")
    parser.add_argument(
        "--preferred-width-field",
        default="wid_adj_routing",
        help="Preferred reviewed-link width field for collapse-footprint buffering.",
    )
    parser.add_argument("--footprint-buffer-scale", type=float, default=0.5, help="Multiplier applied to footprint width / 2 buffering.")
    parser.add_argument("--disable-all-touched", action="store_true", help="Disable all_touched rasterization when creating collapse footprints.")
    parser.add_argument("--allow-noop", action="store_true", help="Allow variants that produce no added pixels.")
    parser.add_argument("--single-thread", action="store_true", help="Pass single_thread=True to the RivGraph river class.")
    parser.add_argument("--disable-sword-export", action="store_true", help="Skip SWORD-style RivGraph exports.")
    parser.add_argument("--transect-scale", type=float, default=1.5, help="Multiplier for transect half-length relative to local total width.")
    parser.add_argument("--min-transect-pixels", type=float, default=5.0, help="Minimum transect half-length in raster pixels.")
    parser.add_argument(
        "--bank-candidate-neighbors-max",
        type=int,
        default=2048,
        help="Maximum number of nearest boundary pixels to query adaptively when seeking left/right bank candidates.",
    )
    parser.add_argument("--match-tolerance", type=float, default=None, help="Optional spatial tolerance for parent-child graph matching.")
    parser.add_argument("--sword-node-source", default=None, help="Optional SWORD node source file or parquet directory used for node matching.")
    parser.add_argument("--sword-wse-field", default=None, help="Optional WSE field name in the SWORD node source. Defaults to automatic detection.")
    parser.add_argument("--sword-match-tolerance", type=float, default=None, help="Optional maximum SWORD node-match distance in CRS units/meters after reprojection.")
    parser.add_argument("--sword-example-stations-source", default=None, help="Optional example-station source used to derive example-specific SWORD reach corridors. Defaults to the repo hierarchy_examples_filtered_subdaily_manual_updates_final.gpkg when present.")
    parser.add_argument("--sword-station-match-source", default=None, help="Optional station-to-SWORD match source used to derive example-specific SWORD reach corridors. Defaults to the repo selected_event_stations_same_main_path.gpkg when present.")
    parser.add_argument("--sword-reach-buffer-steps", type=int, default=2, help="Number of reach steps to extend upstream and downstream beyond the example corridor endpoints when constraining SWORD node candidates.")
    parser.add_argument(
        "--slope-profile-mode",
        choices=("network_dist", "reference_section"),
        default="network_dist",
        help="How state hydraulic context should derive the monotone head profile during variant creation.",
    )
    parser.add_argument(
        "--reference-section-summary-path",
        default=None,
        help="Optional explicit reference_section_summary.json path used during state hydraulic-context generation.",
    )
    parser.add_argument("--min-slope", type=float, default=1e-6, help="Minimum positive slope used in state hydraulic-context generation.")
    parser.add_argument("--max-slope-for-k", type=float, default=None, help="Optional upper bound for local slopes used in state hydraulic-context generation.")
    parser.add_argument("--section-slope-ratio-min", type=float, default=None, help="Optional lower bound on local_slope / section_slope for state hydraulic-context QC.")
    parser.add_argument("--section-slope-ratio-max", type=float, default=None, help="Optional upper bound on local_slope / section_slope for state hydraulic-context QC.")
    parser.add_argument("--disable-section-slope-fallback", action="store_true", help="Disable section-slope fallback during state hydraulic-context generation.")
    parser.add_argument("--disable-compound-bubble-slope-qc", action="store_true", help="Disable compound-bubble path-drop QC during state hydraulic-context generation.")
    parser.add_argument("--disable-compound-bubble-slope-repair", action="store_true", help="Disable compound-bubble slope repair during state hydraulic-context generation.")
    parser.add_argument("--verbose-rivgraph", action="store_true", help="Print RivGraph progress to stdout.")
    parser.add_argument(
        "--base-only",
        action="store_true",
        help=(
            "Materialize only the S000_base state, including its directed variant and hydraulic context, "
            "then stop before building derived unit/group states."
        ),
    )
    parser.add_argument(
        "--defer-child-hierarchy",
        action="store_true",
        help=(
            "For independent-unit or compound-bubble runs, finish child variant and hydraulic "
            "artifacts without rebuilding each complete child hierarchy. Hierarchy-dependent "
            "comparison/equivalent outputs are marked deferred and can be finalized later."
        ),
    )
    parser.add_argument(
        "--execution-mode",
        choices=EXECUTION_MODES,
        nargs="+",
        default="full_raster",
        help=(
            "Use full-raster processing, panelized processing, or the bounded local-patch "
            "child workflow. Pass 'panelized local_patch' to use a panelized base, local-patch "
            "children, and a panelized per-child fallback."
        ),
    )
    parser.add_argument(
        "--base-state-source",
        default=None,
        help=(
            "Optional completed experiment directory (or S000_base/variant directory) whose "
            "base variant should be copied instead of recomputing base widths. When the source "
            "also contains complete hierarchy and hydraulic-context bundles, those products, "
            "boundary components, reference section, and solver cache are reused as well."
        ),
    )
    parser.add_argument(
        "--anchored-source-experiment",
        default=None,
        help=(
            "Optional completed realized experiment directory used to build only the matching "
            "anchored experiment. This skips realized-state regeneration, reuses its boundary "
            "components and base state, and requires --collapse-corridor-mode anchored."
        ),
    )
    parser.add_argument("--panelized-width-samples", default=None, help="Optional merged panelized width-samples CSV; defaults beside --reviewed-links.")
    parser.add_argument("--panelized-boundary-component-labels", default=None, help="Required global boundary-component labels for panelized child states.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        execution_plan = resolve_execution_backend_plan(args.execution_mode)
    except ValueError as exc:
        parser.error(str(exc))
    equivalent_resistance_config = None
    equivalent_forcing_config = None
    if args.compute_equivalent_resistance:
        equivalent_resistance_config = BubbleEquivalentResistanceConfig(
            resistance_law=args.equivalent_resistance_law,
            geometry_method=args.equivalent_geometry_method,
            kb_mode=args.equivalent_kb_mode,
            kb_value=args.equivalent_kb_value,
            length_field_candidates=resolve_length_field_candidates(args.equivalent_length_field),
            default_manning_n=args.equivalent_manning_n,
            default_chezy_cz=args.equivalent_chezy_cz,
            solver_backend=args.equivalent_solver_backend,
            sparse_min_internal_nodes=args.equivalent_sparse_min_internal_nodes,
        )
        if args.equivalent_kb_mode == "based_reference_section":
            equivalent_forcing_config = ForcingConfig(
                time_column=args.equivalent_time_column,
                discharge_column=args.equivalent_discharge_column,
                station_key_column=args.equivalent_station_key_column,
                station_key=args.equivalent_forcing_station_key,
                start_time=args.equivalent_forcing_start_time,
                end_time=args.equivalent_forcing_end_time,
                resample_minutes=args.equivalent_forcing_resample_minutes,
                resample_seconds=args.equivalent_forcing_resample_seconds,
                output_cache_dir=args.equivalent_forcing_output_cache_dir,
            )

    run_kwargs = {
        "cleaned_mask_path": args.cleaned_mask,
        "reviewed_links_path": args.reviewed_links,
        "reviewed_nodes_path": args.reviewed_nodes,
        "exit_sides": args.exit_sides,
        "max_steps": args.max_steps,
        "max_path_cutoff": args.max_path_cutoff,
        "max_paths": args.max_paths,
        "pixel_width_fields": args.pixel_width_fields,
        "pixel_width_percentiles": args.pixel_width_percentiles,
        "use_pixel_widths_for_extremes": not args.disable_pixel_width_extremes,
        "debug_hierarchy": args.debug_hierarchy,
        "preferred_width_field": args.preferred_width_field,
        "footprint_buffer_scale": args.footprint_buffer_scale,
        "all_touched": not args.disable_all_touched,
        "allow_noop": args.allow_noop,
        "single_thread": args.single_thread,
        "export_sword": not args.disable_sword_export,
        "transect_scale": args.transect_scale,
        "min_transect_pixels": args.min_transect_pixels,
        "bank_candidate_neighbors_max": args.bank_candidate_neighbors_max,
        "match_tolerance": args.match_tolerance,
        "sword_node_source_path": args.sword_node_source,
        "sword_wse_field": args.sword_wse_field,
        "sword_match_tolerance": args.sword_match_tolerance,
        "sword_example_station_source_path": args.sword_example_stations_source,
        "sword_station_match_source_path": args.sword_station_match_source,
        "sword_reach_buffer_steps": args.sword_reach_buffer_steps,
        "verbose_rivgraph": args.verbose_rivgraph,
        "state_hydraulic_context_kwargs": {
            "config": SlopeConfig(
                min_slope=args.min_slope,
                slope_profile_mode=args.slope_profile_mode,
                reference_section_summary_path=args.reference_section_summary_path,
                max_slope_for_k=args.max_slope_for_k,
                section_slope_ratio_min=args.section_slope_ratio_min,
                section_slope_ratio_max=args.section_slope_ratio_max,
                use_section_slope_fallback=not args.disable_section_slope_fallback,
                use_compound_bubble_slope_qc=not args.disable_compound_bubble_slope_qc,
                use_compound_bubble_slope_repair=not args.disable_compound_bubble_slope_repair,
            )
        },
        "equivalent_resistance_config": equivalent_resistance_config,
        "equivalent_forcing_path": args.equivalent_forcing_path,
        "equivalent_forcing_config": equivalent_forcing_config,
        "equivalent_kb_model_path": args.equivalent_kb_model_path,
        "equivalent_kb_width_sample_field": args.equivalent_kb_width_sample_field,
        "equivalent_kb_width_percentile": args.equivalent_kb_width_percentile,
        "base_only": args.base_only,
        "defer_child_hierarchy": args.defer_child_hierarchy,
        "requested_unit_ids": args.requested_unit_ids,
        "requested_compound_bubble_ids": args.requested_compound_bubble_ids,
        "execution_mode": execution_plan.requested_modes,
        "panelized_width_samples_path": args.panelized_width_samples,
        "panelized_boundary_component_labels_path": args.panelized_boundary_component_labels,
    }
    if args.base_state_source is not None:
        base_source = Path(args.base_state_source).resolve()
        if base_source.name != "variant":
            base_source = base_source / "states" / "S000_base" / "variant"
        run_kwargs["base_state_variant_source_dir"] = str(base_source)
    if args.panelized_boundary_component_labels is not None:
        run_kwargs["boundary_component_artifact_source_dir"] = str(
            Path(args.panelized_boundary_component_labels).resolve().parent
        )

    anchored_source_experiment: Path | None = None
    if execution_plan.child_backend == "local_patch" and args.collapse_corridor_mode != "anchored":
        parser.error("--execution-mode local_patch currently requires --collapse-corridor-mode anchored.")
    if args.anchored_source_experiment is not None:
        if args.collapse_corridor_mode != "anchored":
            parser.error("--anchored-source-experiment requires --collapse-corridor-mode anchored.")
        if not _supports_realized_reuse_speedup(args.mode):
            parser.error(
                "--anchored-source-experiment is supported only for independent-units and compound-bubbles."
            )
        anchored_source_experiment = Path(args.anchored_source_experiment).resolve()
        required_source_paths = (
            anchored_source_experiment / "states" / "S000_base" / "variant",
            anchored_source_experiment / "boundary_components",
        )
        missing_source_paths = [path for path in required_source_paths if not path.exists()]
        if missing_source_paths:
            parser.error(
                "--anchored-source-experiment is missing required realized artifacts: "
                + ", ".join(str(path) for path in missing_source_paths)
            )

    persist_realized_and_anchored = (
        args.collapse_corridor_mode in {"both", "anchored"}
        and anchored_source_experiment is None
        and execution_plan.child_backend != "local_patch"
    )
    requested_modes = (
        ("anchored",)
        if anchored_source_experiment is not None
        else (COLLAPSE_CORRIDOR_MODES if persist_realized_and_anchored else (args.collapse_corridor_mode,))
    )
    suffix_mode_outputs = (
        persist_realized_and_anchored
        or anchored_source_experiment is not None
        or execution_plan.child_backend == "local_patch"
    )
    realized_output_dir_for_reuse: Path | None = anchored_source_experiment
    for collapse_corridor_mode in requested_modes:
        experiment_id = args.experiment_id
        output_dir = args.output_dir
        current_run_kwargs = dict(run_kwargs)
        if suffix_mode_outputs:
            experiment_id = _suffix_experiment_id_for_collapse_mode(
                experiment_id or _default_experiment_id(args.cleaned_mask, args.mode),
                collapse_corridor_mode,
            )
            output_dir = (
                _suffix_output_dir_for_collapse_mode(args.output_dir, collapse_corridor_mode)
                if args.output_dir is not None
                else None
            )
        if (
            collapse_corridor_mode == "anchored"
            and realized_output_dir_for_reuse is not None
            and _supports_realized_reuse_speedup(args.mode)
        ):
            current_run_kwargs["boundary_component_artifact_source_dir"] = str(
                (realized_output_dir_for_reuse / "boundary_components").resolve()
            )
            current_run_kwargs["base_state_variant_source_dir"] = str(
                (realized_output_dir_for_reuse / "states" / "S000_base" / "variant").resolve()
            )
            if execution_plan.child_backend != "local_patch":
                current_run_kwargs["variant_runner"] = _make_realized_reuse_variant_runner(
                    realized_experiment_output_dir=realized_output_dir_for_reuse,
                    fallback_variant_runner=generate_network_variant,
                )
        results = run_collapse_experiment(
            args.mode,
            experiment_id=experiment_id,
            output_dir=output_dir,
            collapse_corridor_mode=collapse_corridor_mode,
            **current_run_kwargs,
        )
        if collapse_corridor_mode == "realized":
            realized_output_dir_for_reuse = Path(results.output_dir)

        print(f"Wrote experiment outputs to {results.output_dir}")
        print(f"Collapse corridor mode: {collapse_corridor_mode}")
        print(f"States: {len(results.state_registry)}")
        print(f"Transitions: {len(results.transition_registry)}")
        print(f"Equivalent transition rows: {len(results.equivalent_transition_registry)}")
        print(f"Mode: {results.manifest.get('mode')}")
        print(f"Stop reason: {results.manifest.get('stop_reason')}")
        print(f"Terminal state: {results.manifest.get('terminal_state_id')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
