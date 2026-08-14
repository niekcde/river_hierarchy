"""Phase-6B hierarchy and hydraulic products for a merged local patch.

Phase 6A creates a complete merged link table while recalculating raster-based
width products only for changed links.  This module deliberately starts from
that complete table.  It rebuilds topology-derived hierarchy tables, delegates
slope/head work to the normal child-state hydraulic workflow, and points the
equivalent-resistance solver at the parent experiment cache.  Consequently:

* unchanged 1-to-1 links retain their parent slopes and hydraulic QC fields;
* changed topology is handled by the same head/slope solvers as a full state;
* unchanged hydraulic subgraphs can hit the existing resistance cache; and
* no full-raster operation is introduced by Phase 6B.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields, replace
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Callable, Mapping

import geopandas as gpd
import pandas as pd

from hierarchy_level_definition.metrics.bubble_equivalent_metrics import (
    BubbleEquivalentResistanceConfig,
)
from hierarchy_level_definition.run_unit_workflow import (
    UnitWorkflowOutputs,
    run_unit_workflow,
    write_unit_workflow_outputs,
)
from network_variants.hydraulic_context import (
    SlopeConfig,
    StateHydraulicContextOutputs,
    generate_state_hydraulic_context,
)
from network_variants.local_patch_merge import LocalPatchMergeResult
from network_variants.local_patch_products import LocalPatchProductResult


LOCAL_PATCH_HYDRAULICS_VERSION = "topology-local-patch-hydraulics-v1"


@dataclass(frozen=True, slots=True)
class LocalPatchHydraulicResult:
    status: str
    output_paths: Mapping[str, str]
    summary: Mapping[str, Any]
    hydraulics_version: str = LOCAL_PATCH_HYDRAULICS_VERSION

    def to_manifest(self) -> dict[str, Any]:
        return {
            "hydraulics_version": self.hydraulics_version,
            "status": self.status,
            "output_paths": dict(self.output_paths),
            "summary": dict(self.summary),
        }


def _read_hierarchy_options(experiment_dir: Path) -> dict[str, Any]:
    manifest_path = experiment_dir / "experiment_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Experiment manifest does not exist: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    options = manifest.get("options", {}).get("hierarchy", {})
    if not isinstance(options, dict):
        raise ValueError(f"Invalid hierarchy options in {manifest_path}.")
    return dict(options)


def _equivalent_config_from_options(
    options: Mapping[str, Any],
    *,
    solver_cache_dir: Path,
) -> BubbleEquivalentResistanceConfig | None:
    raw = options.get("equivalent_resistance_config")
    if not isinstance(raw, Mapping):
        return None
    accepted = {field.name for field in fields(BubbleEquivalentResistanceConfig)}
    values = {key: value for key, value in raw.items() if key in accepted}
    for key in (
        "width_field_candidates",
        "length_field_candidates",
        "kb_field_candidates",
        "manning_n_field_candidates",
        "chezy_cz_field_candidates",
    ):
        if key in values and values[key] is not None:
            values[key] = tuple(values[key])
    values["solver_cache_dir"] = str(solver_cache_dir.resolve())
    return BubbleEquivalentResistanceConfig(**values)


def _workflow_kwargs(
    options: Mapping[str, Any],
    *,
    equivalent_config: BubbleEquivalentResistanceConfig | None,
) -> dict[str, Any]:
    return {
        "max_path_cutoff": int(options.get("max_path_cutoff", 100)),
        "max_paths": int(options.get("max_paths", 5000)),
        "pixel_width_fields": options.get("pixel_width_fields"),
        "pixel_width_percentiles": options.get("pixel_width_percentiles"),
        "use_pixel_widths_for_extremes": bool(
            options.get("use_pixel_widths_for_extremes", True)
        ),
        "equivalent_resistance_config": equivalent_config,
        "debug": bool(options.get("debug_hierarchy", False)),
    }


def _resolved_slope_config(
    config: SlopeConfig,
    *,
    experiment_dir: Path,
) -> SlopeConfig:
    """Replace stale machine-specific reference paths with local artifacts."""
    if config.slope_profile_mode != "reference_section":
        return config
    local_summary = experiment_dir / "reference_section" / "reference_section_summary.json"
    if local_summary.exists():
        return replace(config, reference_section_summary_path=str(local_summary.resolve()))
    return config


def _hydraulic_nodes_path(
    nodes_path: Path,
    *,
    output_dir: Path,
    sword_node_source_path: str | Path | None,
) -> Path:
    """Relocate persisted SWORD paths without mutating the merged node product."""
    if sword_node_source_path is None:
        return nodes_path
    nodes = gpd.read_file(nodes_path)
    if "sword_source_file" not in nodes.columns:
        return nodes_path
    requested = Path(sword_node_source_path).expanduser().resolve()
    original = [
        Path(str(value))
        for value in nodes["sword_source_file"].dropna().astype(str).unique()
        if str(value).strip()
    ]
    source_name = original[0].name if original else ""
    resolved_source = requested / source_name if requested.is_dir() else requested
    if not resolved_source.exists():
        raise FileNotFoundError(f"Replacement SWORD node source was not found: {resolved_source}")
    nodes["sword_source_file"] = str(resolved_source)
    relocated = output_dir / "inputs" / "nodes_with_local_sword_source.gpkg"
    relocated.parent.mkdir(parents=True, exist_ok=True)
    nodes.to_file(relocated, driver="GPKG")
    return relocated


def _final_node_match_path(node_merge_audit_path: Path, *, output_dir: Path) -> Path:
    """Translate Phase-5 raw local node IDs to final merged child node IDs."""
    matches = pd.read_csv(node_merge_audit_path)
    required = {"final_node_id", "matched_parent_node_id"}
    if not required.issubset(matches.columns):
        raise ValueError(
            f"Node merge audit lacks final match columns {sorted(required)}: "
            f"{node_merge_audit_path}"
        )
    matches["child_id_node"] = pd.to_numeric(
        matches["final_node_id"], errors="coerce"
    ).astype("Int64")
    matches["matched_parent_node_id"] = pd.to_numeric(
        matches["matched_parent_node_id"], errors="coerce"
    ).astype("Int64")
    if "match_within_tolerance" not in matches.columns:
        matches["match_within_tolerance"] = matches["matched_parent_node_id"].notna()
    if "match_distance" not in matches.columns:
        matches["match_distance"] = float("inf")
    matches = matches.dropna(subset=["child_id_node"]).copy()
    matches = matches.sort_values(
        ["child_id_node", "match_within_tolerance", "match_distance"],
        ascending=[True, False, True],
        kind="mergesort",
    ).drop_duplicates("child_id_node", keep="first")
    path = output_dir / "inputs" / "final_node_match.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    matches.to_csv(path, index=False)
    return path


def materialize_local_patch_hydraulic_products(
    *,
    products_result: LocalPatchProductResult,
    merge_result: LocalPatchMergeResult,
    collapse_components_path: str | Path,
    experiment_dir: str | Path,
    parent_state_dir: str | Path,
    output_dir: str | Path,
    state_id: str,
    parent_state_id: str = "S000_base",
    depth: int = 1,
    slope_config: SlopeConfig | None = None,
    compute_equivalent_resistance: bool = True,
    equivalent_solver_cache_dir: str | Path | None = None,
    sword_node_source_path: str | Path | None = None,
    unit_workflow_runner: Callable[..., UnitWorkflowOutputs] = run_unit_workflow,
    unit_workflow_writer: Callable[..., None] = write_unit_workflow_outputs,
    hydraulic_context_runner: Callable[..., StateHydraulicContextOutputs] = generate_state_hydraulic_context,
) -> LocalPatchHydraulicResult:
    """Create complete hierarchy, slope/head, and resistance products.

    The caller supplies ``slope_config`` explicitly because older experiment
    manifests did not preserve every slope CLI option.  The default is kept for
    synthetic tests and new experiments whose defaults were used.
    """
    if products_result.status != "complete_width_length_products":
        raise ValueError("Phase-6B requires complete Phase-6A products.")
    if merge_result.status != "complete_conservative_merge":
        raise ValueError("Phase-6B requires a complete conservative merge.")

    experiment_dir = Path(experiment_dir).expanduser().resolve()
    parent_state_dir = Path(parent_state_dir).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    hierarchy_dir = output_dir / "hierarchy"
    hydraulic_dir = output_dir / "hydraulic_context"
    hierarchy_dir.mkdir(parents=True, exist_ok=True)
    hydraulic_dir.mkdir(parents=True, exist_ok=True)

    links_path = Path(products_result.output_paths["links_with_width_families"]).resolve()
    nodes_path = Path(merge_result.output_paths["merged_nodes"]).resolve()
    hydraulic_nodes_path = _hydraulic_nodes_path(
        nodes_path,
        output_dir=output_dir,
        sword_node_source_path=sword_node_source_path,
    )
    parent_context_path = parent_state_dir / "hydraulic_context" / "link_hydraulic_context.csv"
    parent_node_context_path = parent_state_dir / "hydraulic_context" / "node_hydraulic_context.csv"
    collapse_components_path = Path(collapse_components_path).expanduser().resolve()
    node_merge_audit_path = Path(merge_result.output_paths["node_merge_audit"]).resolve()
    if not parent_context_path.exists():
        raise FileNotFoundError(
            "Parent hydraulic context is required for unchanged-link inheritance: "
            f"{parent_context_path}"
        )
    for required_path in (
        parent_node_context_path,
        collapse_components_path,
        node_merge_audit_path,
    ):
        if not required_path.exists():
            raise FileNotFoundError(f"Required Phase-6B context does not exist: {required_path}")
    node_match_path = _final_node_match_path(node_merge_audit_path, output_dir=output_dir)

    options = _read_hierarchy_options(experiment_dir)
    solver_cache_dir = (
        Path(equivalent_solver_cache_dir).expanduser().resolve()
        if equivalent_solver_cache_dir is not None
        else experiment_dir / "equivalent_resistance" / "solve_cache"
    )
    equivalent_config = (
        _equivalent_config_from_options(options, solver_cache_dir=solver_cache_dir)
        if compute_equivalent_resistance
        else None
    )
    workflow_kwargs = _workflow_kwargs(options, equivalent_config=equivalent_config)
    workflow = unit_workflow_runner(links_path, nodes_path, **workflow_kwargs)
    unit_workflow_writer(
        hierarchy_dir,
        workflow,
        manifest_overrides={
            "state_id": str(state_id),
            "parent_state_id": str(parent_state_id),
            "depth": int(depth),
            "local_patch_phase": "6B",
        },
    )

    resolved_slope = _resolved_slope_config(
        slope_config or SlopeConfig(),
        experiment_dir=experiment_dir,
    )
    hydraulic = hydraulic_context_runner(
        directed_links_path=links_path,
        directed_nodes_path=hydraulic_nodes_path,
        hierarchy_output_dir=hierarchy_dir,
        output_dir=hydraulic_dir,
        state_id=str(state_id),
        config=resolved_slope,
        parent_link_context_path=parent_context_path,
        collapse_components_path=collapse_components_path,
        node_match_path=node_match_path,
        base_link_context_path=parent_context_path,
        base_node_context_path=parent_node_context_path,
    )

    output_paths = {
        "hierarchy": str(hierarchy_dir),
        "hydraulic_context": str(hydraulic_dir),
        "link_hydraulic_context": str(Path(hydraulic.link_context_path).resolve()),
        "node_hydraulic_context": str(Path(hydraulic.node_context_path).resolve()),
        "hydraulic_context_manifest": str(Path(hydraulic.manifest_path).resolve()),
    }
    manifest_path = output_dir / "local_patch_hydraulics_manifest.json"
    output_paths["manifest"] = str(manifest_path)
    summary = {
        "state_id": str(state_id),
        "parent_state_id": str(parent_state_id),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "links_path": str(links_path),
        "nodes_path": str(nodes_path),
        "hydraulic_nodes_path": str(hydraulic_nodes_path),
        "parent_link_hydraulic_context": str(parent_context_path),
        "parent_node_hydraulic_context": str(parent_node_context_path),
        "collapse_components_path": str(collapse_components_path),
        "node_match_path": str(node_match_path),
        "node_merge_audit_path": str(node_merge_audit_path),
        "equivalent_resistance_enabled": equivalent_config is not None,
        "equivalent_solver_cache_dir": (
            str(solver_cache_dir) if equivalent_config is not None else None
        ),
        "equivalent_solver_backend": (
            equivalent_config.solver_backend if equivalent_config is not None else None
        ),
        "slope_config": asdict(resolved_slope),
        "recalculated_width_link_ids": list(
            products_result.summary.get("recalculated_link_ids", [])
        ),
    }
    result = LocalPatchHydraulicResult(
        status="complete_hierarchy_hydraulic_products",
        output_paths=output_paths,
        summary=summary,
    )
    manifest_path.write_text(
        json.dumps(result.to_manifest(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result


__all__ = [
    "LOCAL_PATCH_HYDRAULICS_VERSION",
    "LocalPatchHydraulicResult",
    "materialize_local_patch_hydraulic_products",
]
