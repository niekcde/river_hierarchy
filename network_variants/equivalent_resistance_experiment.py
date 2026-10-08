from __future__ import annotations

import argparse
from collections import deque
import json
import sys
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
import tempfile
from typing import Any, Callable, Mapping, Sequence

import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd

repo_root = Path(__file__).resolve().parents[1]
rapid_src = repo_root / "RAPID" / "src"
if str(rapid_src) not in sys.path:
    sys.path.insert(0, str(rapid_src))

if __package__ is None or __package__ == "":
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

from hierarchy_level_definition.metrics import (
    BubbleEquivalentResistanceConfig,
    LENGTH_FIELD_CANDIDATES,
    compare_states,
    compute_bubble_equivalent_metrics,
    prepare_bubble_resistance_edges,
    resolve_length_field_candidates,
)
from hierarchy_level_definition.metrics.bubble_equivalent_metrics import (
    solve_bubble_with_config,
)
from rapid_tools.forcing import ForcingConfig, prepare_forcing_table
from rapid_tools.reference_section import compute_reference_section_kb


SUPPORTED_MODES = ("independent-units", "compound-bubbles")
NODE_HEAD_FIELD_CANDIDATES = ("H", "head", "sword_wse", "wse", "h")
CHANGED_LINEAGE_TYPES = {"collapsed_many_to_one", "resolved_1to1", "touch_fallback_1to1", "unmatched"}


@dataclass(frozen=True, slots=True)
class EquivalentResistanceExperimentOutputs:
    base_nominal_targets: pd.DataFrame
    child_realized_objects: pd.DataFrame
    matched_base_corridors: pd.DataFrame
    comparison: pd.DataFrame
    output_dir: Path
    manifest: dict[str, Any]


@dataclass(frozen=True, slots=True)
class StateHierarchyRecomputeOutputs:
    """Outputs from an equivalent-resistance-only state hierarchy refresh."""

    audit: pd.DataFrame
    state_registry: pd.DataFrame
    n_states_recomputed: int
    audit_csv_path: Path
    summary_json_path: Path


def _atomic_write_csv(frame: pd.DataFrame, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
        delete=False,
        mode="w",
        encoding="utf-8",
        newline="",
    ) as handle:
        temporary = Path(handle.name)
        frame.to_csv(handle, index=False)
    try:
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_write_json(payload: Mapping[str, Any], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
        delete=False,
        mode="w",
        encoding="utf-8",
    ) as handle:
        temporary = Path(handle.name)
        json.dump(dict(payload), handle, indent=2)
        handle.write("\n")
    try:
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


def _parse_int_list(value: Any) -> list[int]:
    if value is None:
        return []
    if isinstance(value, (int, np.integer)):
        return [int(value)]
    if isinstance(value, (float, np.floating)):
        if pd.isna(value):
            return []
        return [int(value)]
    if isinstance(value, (list, tuple, set, np.ndarray, pd.Series)):
        result: list[int] = []
        for item in value:
            result.extend(_parse_int_list(item))
        return result
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return []
    result: list[int] = []
    for part in text.split(","):
        token = part.strip()
        if not token:
            continue
        result.append(int(float(token)))
    return result


def _serialize_int_list(values: Sequence[int]) -> str:
    return ",".join(str(int(value)) for value in values)


def _optional_int(value: Any) -> int | None:
    if value is None or value is pd.NA or pd.isna(value):
        return None
    return int(value)


def _optional_path(value: Any) -> Path | None:
    if value is None or value is pd.NA or pd.isna(value):
        return None
    text = str(value).strip()
    if not text:
        return None
    return Path(text).expanduser().resolve()


def _infer_mode(experiment_dir: Path, registry: pd.DataFrame) -> str:
    if "mode" in registry.columns:
        values = registry["mode"].dropna().astype(str).str.strip()
        values = values[values.ne("")]
        if not values.empty:
            return str(values.iloc[0])
    manifest_path = experiment_dir / "experiment_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        mode = str(manifest.get("mode", "") or "").strip()
        if mode:
            return mode
    raise ValueError("Could not resolve experiment mode from state_registry.csv or experiment_manifest.json.")


def _empty_base_nominal_targets_frame() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "state_id",
            "target_type",
            "target_id",
            "selected_unit_ids",
            "base_state_id",
            "nominal_upstream_node",
            "nominal_downstream_node",
            "nominal_base_link_ids",
            "nominal_base_node_ids",
            "nominal_n_links",
            "nominal_n_nodes",
            "K_eq_nominal_base",
            "G_eq_nominal_base",
            "Q_unit_nominal_base",
            "solver_success_nominal",
            "solver_method_nominal",
            "solver_message_nominal",
            "solver_nfev_nominal",
            "residual_norm_nominal",
            "residual_relative_nominal",
            "resistance_law",
            "geometry_method",
            "kb_mode",
            "kb_value",
            "kb_source_method",
        ]
    )


def _empty_child_realized_objects_frame() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "state_id",
            "target_type",
            "target_id",
            "selected_unit_ids",
            "child_object_id",
            "child_object_type",
            "child_link_ids",
            "child_node_ids",
            "child_n_links",
            "child_n_nodes",
            "child_upstream_node",
            "child_downstream_node",
            "child_boundary_match_upstream_parent_node",
            "child_boundary_match_downstream_parent_node",
            "matched_parent_node_path",
            "matched_parent_link_ids",
            "boundaries_match_nominal_target",
            "boundary_relation",
            "K_variant_realized",
            "G_variant_realized",
            "Q_unit_variant",
            "solver_success_variant",
            "solver_method_variant",
            "solver_message_variant",
            "solver_nfev_variant",
            "residual_norm_variant",
            "residual_relative_variant",
        ]
    )


def _empty_matched_base_corridors_frame() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "state_id",
            "child_object_id",
            "target_type",
            "target_id",
            "matched_base_upstream_node",
            "matched_base_downstream_node",
            "matched_base_link_ids",
            "matched_base_node_ids",
            "matched_base_node_path",
            "matched_base_n_links",
            "matched_base_n_nodes",
            "K_eq_matched_base",
            "G_eq_matched_base",
            "Q_unit_matched_base",
            "solver_success_matched",
            "solver_method_matched",
            "solver_message_matched",
            "solver_nfev_matched",
            "residual_norm_matched",
            "residual_relative_matched",
            "boundaries_match_nominal_target",
            "boundary_relation",
        ]
    )


def _empty_comparison_frame() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "state_id",
            "child_object_id",
            "target_type",
            "target_id",
            "K_eq_nominal_base",
            "K_eq_matched_base",
            "K_variant_realized",
            "Lambda_nominal",
            "Lambda_matched",
            "Gamma_nominal",
            "Gamma_matched",
            "boundaries_match_nominal_target",
            "boundary_relation",
            "comparison_valid_for_nominal",
            "comparison_valid_for_matched",
        ]
    )


def _normalize_numeric_id_columns(
    frame: pd.DataFrame | gpd.GeoDataFrame,
    *,
    columns: Sequence[str] = ("id_node", "idx_node", "id_link", "id_us_node", "id_ds_node"),
) -> pd.DataFrame | gpd.GeoDataFrame:
    result = frame.copy()
    for column in columns:
        if column in result.columns:
            result[column] = pd.to_numeric(result[column], errors="coerce")
            if result[column].notna().all():
                result[column] = result[column].astype(int)
    return result


def _load_variant_links(
    *,
    directed_links_path: Path,
    variant_output_dir: Path,
    analysis_links_path: Path | None = None,
) -> gpd.GeoDataFrame:
    if analysis_links_path is not None and analysis_links_path.exists():
        return _normalize_numeric_id_columns(gpd.read_file(analysis_links_path))
    widths_links_path = variant_output_dir / "widths" / "links_with_width_families.gpkg"
    width_families_csv_path = variant_output_dir / "widths" / "link_width_families.csv"

    if widths_links_path.exists():
        return _normalize_numeric_id_columns(gpd.read_file(widths_links_path))

    directed_links = _normalize_numeric_id_columns(gpd.read_file(directed_links_path))
    if not width_families_csv_path.exists():
        return directed_links

    width_families = pd.read_csv(width_families_csv_path)
    width_families = _normalize_numeric_id_columns(width_families)
    if "id_link" not in width_families.columns:
        return directed_links

    geometry_name = directed_links.geometry.name
    merge_columns = [column for column in width_families.columns if column != geometry_name]
    base_columns = set(directed_links.columns)
    width_lookup = width_families[merge_columns].copy()
    override_columns = [column for column in merge_columns if column != "id_link" and column in base_columns]
    if override_columns:
        directed_links = directed_links.drop(columns=override_columns, errors="ignore")
    merged = directed_links.merge(width_lookup, on="id_link", how="left", validate="1:1")
    return _normalize_numeric_id_columns(merged)


def _load_base_state_context(
    base_row: pd.Series,
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, pd.DataFrame, pd.DataFrame]:
    directed_links_path = _optional_path(base_row.get("directed_links_path"))
    directed_nodes_path = _optional_path(base_row.get("directed_nodes_path"))
    hierarchy_output_dir = _optional_path(base_row.get("hierarchy_output_dir"))
    variant_output_dir = _optional_path(base_row.get("variant_output_dir"))
    if directed_links_path is None or directed_nodes_path is None or hierarchy_output_dir is None or variant_output_dir is None:
        raise ValueError("Base state row is missing required directed-links/nodes or hierarchy paths.")

    unit_summary_path = hierarchy_output_dir / "unit_summary.csv"
    unit_link_membership_path = variant_output_dir / "summary" / "unit_link_membership.csv"
    if not unit_summary_path.exists():
        raise FileNotFoundError(f"Base unit summary was not found: {unit_summary_path}")
    if not unit_link_membership_path.exists():
        raise FileNotFoundError(f"Base unit-link membership was not found: {unit_link_membership_path}")

    base_links = _load_variant_links(
        directed_links_path=directed_links_path,
        variant_output_dir=variant_output_dir,
    )
    base_nodes = _normalize_numeric_id_columns(gpd.read_file(directed_nodes_path))
    unit_summary = pd.read_csv(unit_summary_path)
    unit_link_membership = pd.read_csv(unit_link_membership_path)
    for frame in (unit_summary, unit_link_membership):
        for column in ("unit_id", "id_link", "bifurcation", "confluence"):
            if column in frame.columns:
                frame[column] = pd.to_numeric(frame[column], errors="coerce")
                if frame[column].notna().all():
                    frame[column] = frame[column].astype(int)
    return base_links, base_nodes, unit_summary, unit_link_membership


def _resolve_equivalent_reference_kb_summary(
    *,
    experiment_dir: Path,
    equivalent_resistance_config: BubbleEquivalentResistanceConfig,
    forcing_path: str | Path | None,
    forcing_config: ForcingConfig | None,
    kb_model_path: str | Path | None,
    kb_width_sample_field: str,
    kb_width_percentile: float | None,
    forcing_table_runner: Callable[..., tuple[pd.DataFrame, dict[str, object]]],
    reference_kb_summary_runner: Callable[..., dict[str, Any]],
) -> tuple[BubbleEquivalentResistanceConfig, dict[str, Any] | None, dict[str, object] | None]:
    if str(equivalent_resistance_config.kb_mode or "fixed").strip() != "based_reference_section":
        return equivalent_resistance_config, (
            dict(equivalent_resistance_config.reference_kb_summary)
            if equivalent_resistance_config.reference_kb_summary is not None
            else None
        ), None
    if equivalent_resistance_config.reference_kb_summary is not None:
        return equivalent_resistance_config, dict(equivalent_resistance_config.reference_kb_summary), None
    if forcing_path is None:
        raise ValueError(
            "Equivalent-resistance kb_mode='based_reference_section' requires forcing_path to be provided."
        )

    forcing, forcing_metadata = forcing_table_runner(forcing_path, config=forcing_config or ForcingConfig())
    reference_kb_summary = reference_kb_summary_runner(
        experiment_dir,
        forcing=forcing,
        width_sample_field=kb_width_sample_field,
        width_percentile=kb_width_percentile,
        model_path=kb_model_path,
    )
    config = replace(equivalent_resistance_config, reference_kb_summary=dict(reference_kb_summary))
    return config, dict(reference_kb_summary), dict(forcing_metadata)


def load_manifest_equivalent_resistance_config(
    experiment_dir: str | Path,
) -> tuple[BubbleEquivalentResistanceConfig, dict[str, Any]]:
    """Recover the exact state-hierarchy resistance configuration from a collapse manifest."""

    experiment_path = Path(experiment_dir).expanduser().resolve()
    manifest_path = experiment_path / "experiment_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Experiment manifest was not found: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    hierarchy_options = manifest.get("options", {}).get("hierarchy", {})
    payload = hierarchy_options.get("equivalent_resistance_config")
    if not isinstance(payload, Mapping) or not payload:
        raise ValueError(
            "The experiment manifest does not contain options.hierarchy."
            "equivalent_resistance_config; state hierarchy resistance cannot be "
            "recomputed safely without its original configuration."
        )

    normalized = dict(payload)
    tuple_fields = (
        "width_field_candidates",
        "length_field_candidates",
        "kb_field_candidates",
        "manning_n_field_candidates",
        "chezy_cz_field_candidates",
        "solver_continuation_eps_schedule",
    )
    for field_name in tuple_fields:
        if field_name in normalized and normalized[field_name] is not None:
            normalized[field_name] = tuple(normalized[field_name])
    config = BubbleEquivalentResistanceConfig(**normalized)
    return config, hierarchy_options


def _read_optional_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _audit_state_recompute(
    *,
    state_id: str,
    old_metrics: pd.DataFrame,
    new_metrics: pd.DataFrame,
) -> pd.DataFrame:
    audit_columns = (
        "compound_bubble_id",
        "bubble_root_unit_id",
        "upstream_boundary_node",
        "downstream_boundary_node",
        "declared_root_upstream_boundary_node",
        "declared_root_downstream_boundary_node",
        "assembled_boundaries_differ_from_root",
        "solver_success",
        "solver_initialization",
        "neutral_fallback_attempted",
        "neutral_fallback_solver_success",
        "K_eq",
        "G_eq",
        "Q_unit",
        "residual_relative",
    )

    def selected(frame: pd.DataFrame, prefix: str) -> pd.DataFrame:
        available = [column for column in audit_columns if column in frame.columns]
        result = frame[available].copy()
        if "compound_bubble_id" not in result.columns:
            result["compound_bubble_id"] = pd.Series(dtype="Int64")
        return result.rename(
            columns={
                column: f"{prefix}_{column}"
                for column in result.columns
                if column != "compound_bubble_id"
            }
        )

    old_selected = selected(old_metrics, "old")
    new_selected = selected(new_metrics, "new")
    audit = old_selected.merge(
        new_selected,
        on="compound_bubble_id",
        how="outer",
        validate="1:1",
        indicator=True,
    )
    audit.insert(0, "state_id", state_id)

    def numeric_column(column: str) -> pd.Series:
        if column not in audit.columns:
            return pd.Series(np.nan, index=audit.index, dtype=float)
        return pd.to_numeric(audit[column], errors="coerce")

    old_up = numeric_column("old_upstream_boundary_node")
    old_down = numeric_column("old_downstream_boundary_node")
    new_up = numeric_column("new_upstream_boundary_node")
    new_down = numeric_column("new_downstream_boundary_node")
    audit["boundary_changed"] = ~(
        old_up.eq(new_up) & old_down.eq(new_down)
    )
    old_k = numeric_column("old_K_eq")
    new_k = numeric_column("new_K_eq")
    audit["K_changed"] = ~np.isclose(
        old_k.to_numpy(dtype=float),
        new_k.to_numpy(dtype=float),
        rtol=1e-10,
        atol=0.0,
        equal_nan=True,
    )
    return audit


def recompute_state_hierarchy_equivalent_resistance(
    experiment_dir: str | Path,
    *,
    equivalent_resistance_config: BubbleEquivalentResistanceConfig | None = None,
    max_path_cutoff: int | None = None,
    max_paths: int | None = None,
    debug: bool = False,
    metrics_runner: Callable[..., tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]] = compute_bubble_equivalent_metrics,
) -> StateHierarchyRecomputeOutputs:
    """Recompute only per-state hierarchy equivalent resistance artifacts.

    Masks, RivGraph products, widths, slopes, unit metrics, and collapse states
    are read-only. The three equivalent-resistance CSVs and their manifest/count
    metadata are replaced atomically.
    """

    experiment_path = Path(experiment_dir).expanduser().resolve()
    registry_path = experiment_path / "state_registry.csv"
    if not registry_path.exists():
        raise FileNotFoundError(f"State registry was not found: {registry_path}")
    registry = pd.read_csv(registry_path)
    manifest_config, hierarchy_options = load_manifest_equivalent_resistance_config(
        experiment_path
    )
    config = equivalent_resistance_config or manifest_config
    resolved_max_path_cutoff = int(
        max_path_cutoff
        if max_path_cutoff is not None
        else hierarchy_options.get("max_path_cutoff", 100)
    )
    resolved_max_paths = int(
        max_paths
        if max_paths is not None
        else hierarchy_options.get("max_paths", 5000)
    )

    audit_frames: list[pd.DataFrame] = []
    state_summaries: list[dict[str, Any]] = []
    completed = registry["status"].fillna("complete").astype(str).eq("complete")
    if "hierarchy_status" in registry.columns:
        completed &= registry["hierarchy_status"].fillna("complete").astype(str).eq("complete")

    for index, row in registry.loc[completed].iterrows():
        state_id = str(row["state_id"])
        links_path = _optional_path(row.get("analysis_links_path"))
        nodes_path = _optional_path(row.get("directed_nodes_path"))
        hierarchy_dir = _optional_path(row.get("hierarchy_output_dir"))
        if links_path is None or nodes_path is None or hierarchy_dir is None:
            raise ValueError(
                f"State {state_id} is missing analysis_links_path, directed_nodes_path, "
                "or hierarchy_output_dir."
            )
        if not links_path.exists() or not nodes_path.exists():
            raise FileNotFoundError(
                f"State {state_id} inputs were not found: links={links_path}, nodes={nodes_path}"
            )

        metrics_path = hierarchy_dir / "bubble_equivalent_metrics.csv"
        edges_path = hierarchy_dir / "bubble_edge_resistance.csv"
        heads_path = hierarchy_dir / "bubble_node_heads.csv"
        old_metrics = _read_optional_csv(metrics_path)
        new_metrics, new_edges, new_heads = metrics_runner(
            links_path,
            nodes_path,
            max_path_cutoff=resolved_max_path_cutoff,
            max_paths=resolved_max_paths,
            debug=debug,
            config=config,
        )
        state_audit = _audit_state_recompute(
            state_id=state_id,
            old_metrics=old_metrics,
            new_metrics=new_metrics,
        )
        audit_frames.append(state_audit)

        _atomic_write_csv(new_metrics, metrics_path)
        _atomic_write_csv(new_edges, edges_path)
        _atomic_write_csv(new_heads, heads_path)

        workflow_manifest_path = hierarchy_dir / "unit_workflow_manifest.json"
        workflow_manifest: dict[str, Any] = {}
        if workflow_manifest_path.exists():
            workflow_manifest = json.loads(
                workflow_manifest_path.read_text(encoding="utf-8")
            )
        recomputed_at = datetime.now(timezone.utc).isoformat()
        workflow_manifest.update(
            {
                "n_equivalent_bubbles": int(len(new_metrics)),
                "n_equivalent_edges": int(len(new_edges)),
                "n_equivalent_nodes": int(len(new_heads)),
                "equivalent_resistance_config": asdict(config),
                "resolved_kb_context": dict(
                    new_metrics.attrs.get("resolved_kb_context", {})
                ),
                "equivalent_resistance_recomputed_at": recomputed_at,
                "equivalent_resistance_recompute_scope": "state_hierarchy_only",
            }
        )
        _atomic_write_json(workflow_manifest, workflow_manifest_path)

        solver_success = (
            int(new_metrics["solver_success"].fillna(False).astype(bool).sum())
            if "solver_success" in new_metrics.columns
            else 0
        )
        registry.loc[index, "n_equivalent_bubbles"] = int(len(new_metrics))
        registry.loc[index, "n_equivalent_bubbles_solver_success"] = solver_success
        state_summaries.append(
            {
                "state_id": state_id,
                "n_old_rows": int(len(old_metrics)),
                "n_new_rows": int(len(new_metrics)),
                "n_solver_success": solver_success,
                "n_solver_failed": int(len(new_metrics) - solver_success),
                "n_boundary_changed": int(state_audit["boundary_changed"].sum()),
                "n_K_changed": int(state_audit["K_changed"].sum()),
            }
        )

    audit = (
        pd.concat(audit_frames, ignore_index=True)
        if audit_frames
        else pd.DataFrame()
    )
    report_dir = experiment_path / "equivalent_resistance"
    audit_csv_path = report_dir / "state_hierarchy_recompute_audit.csv"
    summary_json_path = report_dir / "state_hierarchy_recompute_summary.json"
    _atomic_write_csv(audit, audit_csv_path)
    _atomic_write_csv(registry, registry_path)
    summary = {
        "experiment_dir": str(experiment_path),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "scope": "state_hierarchy_equivalent_resistance_only",
        "max_path_cutoff": resolved_max_path_cutoff,
        "max_paths": resolved_max_paths,
        "equivalent_resistance_config": asdict(config),
        "n_states_recomputed": int(len(state_summaries)),
        "n_rows_recomputed": int(len(audit)),
        "n_boundaries_changed": int(audit["boundary_changed"].sum()) if not audit.empty else 0,
        "n_K_values_changed": int(audit["K_changed"].sum()) if not audit.empty else 0,
        "n_solver_failed_after": int(
            sum(item["n_solver_failed"] for item in state_summaries)
        ),
        "state_summaries": state_summaries,
        "audit_csv": str(audit_csv_path),
    }
    _atomic_write_json(summary, summary_json_path)
    return StateHierarchyRecomputeOutputs(
        audit=audit,
        state_registry=registry,
        n_states_recomputed=len(state_summaries),
        audit_csv_path=audit_csv_path,
        summary_json_path=summary_json_path,
    )


def _incident_node_ids(links: pd.DataFrame) -> list[int]:
    nodes = pd.unique(pd.concat([links["id_us_node"], links["id_ds_node"]], ignore_index=True))
    return [int(node_id) for node_id in nodes.tolist()]


def _compute_equivalent_resistance(
    links: gpd.GeoDataFrame | pd.DataFrame,
    nodes: gpd.GeoDataFrame | pd.DataFrame,
    *,
    upstream_node: int | None,
    downstream_node: int | None,
    config: BubbleEquivalentResistanceConfig,
) -> tuple[dict[str, Any], Mapping[str, Any] | None]:
    if upstream_node is None or downstream_node is None:
        return {
            "solver_success": False,
            "solver_method": "",
            "solver_backend_requested": config.solver_backend,
            "solver_backend_resolved": "",
            "solver_message": "Could not resolve boundary nodes for equivalent-resistance solve.",
            "solver_nfev": np.nan,
            "K_eq": np.nan,
            "G_eq": np.nan,
            "Q_unit": np.nan,
            "residual_norm": np.nan,
            "residual_relative": np.nan,
        }, None
    try:
        prepared_edges, resolved_kb = prepare_bubble_resistance_edges(links, config=config)
        node_subset = nodes.loc[pd.to_numeric(nodes["id_node"], errors="coerce").isin(_incident_node_ids(prepared_edges))]
        optional_head_guess = (
            node_subset
            if any(column in node_subset.columns for column in NODE_HEAD_FIELD_CANDIDATES)
            else None
        )
        result = solve_bubble_with_config(
            node_subset,
            prepared_edges,
            upstream_node=int(upstream_node),
            downstream_node=int(downstream_node),
            optional_H_guess=optional_head_guess,
            config=config,
        )
        return result, resolved_kb.to_dict()
    except Exception as exc:
        return {
            "solver_success": False,
            "solver_method": "",
            "solver_backend_requested": config.solver_backend,
            "solver_backend_resolved": "",
            "solver_message": str(exc),
            "solver_nfev": np.nan,
            "K_eq": np.nan,
            "G_eq": np.nan,
            "Q_unit": np.nan,
            "residual_norm": np.nan,
            "residual_relative": np.nan,
        }, None


def _build_boundary_graph(links: pd.DataFrame) -> nx.MultiDiGraph:
    graph = nx.MultiDiGraph()
    for row in links.itertuples(index=False):
        graph.add_edge(int(row.id_us_node), int(row.id_ds_node), key=int(row.id_link), id_link=int(row.id_link))
    return graph


def _resolve_component_boundaries(links: pd.DataFrame) -> tuple[int | None, int | None]:
    if links.empty:
        return None, None
    if len(links) == 1:
        row = links.iloc[0]
        return int(row["id_us_node"]), int(row["id_ds_node"])
    graph = _build_boundary_graph(links)
    sources = [int(node) for node in graph.nodes if graph.in_degree(node) == 0 and graph.out_degree(node) > 0]
    sinks = [int(node) for node in graph.nodes if graph.out_degree(node) == 0 and graph.in_degree(node) > 0]
    if len(sources) == 1 and len(sinks) == 1:
        return sources[0], sinks[0]
    return None, None


def _resolve_linear_node_path(links: pd.DataFrame, *, upstream_node: int | None, downstream_node: int | None) -> list[int]:
    if upstream_node is None or downstream_node is None or links.empty:
        return []
    graph = nx.DiGraph()
    for row in links.itertuples(index=False):
        graph.add_edge(int(row.id_us_node), int(row.id_ds_node))
    if not graph.has_node(int(upstream_node)) or not graph.has_node(int(downstream_node)):
        return []
    if not nx.has_path(graph, int(upstream_node), int(downstream_node)):
        return []
    try:
        return [int(node_id) for node_id in nx.shortest_path(graph, int(upstream_node), int(downstream_node))]
    except nx.NetworkXNoPath:
        return []


def _classify_child_object_type(links: pd.DataFrame) -> str:
    if len(links) == 1:
        return "single_link"
    graph = _build_boundary_graph(links)
    max_in = max((graph.in_degree(node) for node in graph.nodes), default=0)
    max_out = max((graph.out_degree(node) for node in graph.nodes), default=0)
    return "linear_corridor" if max_in <= 1 and max_out <= 1 else "branched_corridor"


def _classify_boundary_relation(
    *,
    nominal_upstream_node: int | None,
    nominal_downstream_node: int | None,
    matched_node_path: Sequence[int],
) -> tuple[bool, str]:
    if nominal_upstream_node is None or nominal_downstream_node is None or not matched_node_path:
        return False, "ambiguous"
    if nominal_upstream_node not in matched_node_path or nominal_downstream_node not in matched_node_path:
        return False, "shifted"
    idx_up = matched_node_path.index(int(nominal_upstream_node))
    idx_down = matched_node_path.index(int(nominal_downstream_node))
    if idx_up >= idx_down:
        return False, "shifted"
    expanded_upstream = idx_up > 0
    expanded_downstream = idx_down < len(matched_node_path) - 1
    if not expanded_upstream and not expanded_downstream:
        return True, "exact_match"
    if expanded_upstream and expanded_downstream:
        return False, "expanded_both"
    if expanded_upstream:
        return False, "expanded_upstream"
    return False, "expanded_downstream"


def _build_nominal_targets(
    *,
    registry: pd.DataFrame,
    mode: str,
    base_state_id: str,
    base_links: gpd.GeoDataFrame,
    base_nodes: gpd.GeoDataFrame,
    unit_summary: pd.DataFrame,
    unit_link_membership: pd.DataFrame,
    config: BubbleEquivalentResistanceConfig,
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    derived_rows = registry.loc[
        registry["state_role"].astype(str).eq("derived")
        & registry["status"].fillna("complete").astype(str).eq("complete")
    ].copy()

    for row in derived_rows.itertuples(index=False):
        selected_unit_ids = _parse_int_list(getattr(row, "selected_unit_ids", ""))
        if not selected_unit_ids:
            continue

        if mode == "compound-bubbles":
            target_type = "compound_bubble"
            target_id = _optional_int(getattr(row, "compound_bubble_id", pd.NA))
            root_unit_id = _optional_int(getattr(row, "bubble_root_unit_id", pd.NA))
            if root_unit_id is None:
                root_unit_id = selected_unit_ids[0]
        else:
            target_type = "unit"
            target_id = int(selected_unit_ids[0])
            root_unit_id = int(selected_unit_ids[0])

        nominal_row = unit_summary.loc[pd.to_numeric(unit_summary["unit_id"], errors="coerce").eq(int(root_unit_id))]
        if nominal_row.empty:
            continue
        nominal_unit = nominal_row.iloc[0]
        nominal_upstream_node = _optional_int(nominal_unit.get("bifurcation"))
        nominal_downstream_node = _optional_int(nominal_unit.get("confluence"))

        link_rows = unit_link_membership.loc[
            pd.to_numeric(unit_link_membership["unit_id"], errors="coerce").isin([int(unit_id) for unit_id in selected_unit_ids])
        ].copy()
        nominal_link_ids = sorted(pd.to_numeric(link_rows["id_link"], errors="coerce").dropna().astype(int).unique().tolist())
        if not nominal_link_ids:
            continue

        nominal_links = base_links.loc[pd.to_numeric(base_links["id_link"], errors="coerce").isin(nominal_link_ids)].copy()
        nominal_node_ids = _incident_node_ids(nominal_links)
        nominal_nodes = base_nodes.loc[pd.to_numeric(base_nodes["id_node"], errors="coerce").isin(nominal_node_ids)].copy()
        solve, resolved_kb = _compute_equivalent_resistance(
            nominal_links,
            nominal_nodes,
            upstream_node=nominal_upstream_node,
            downstream_node=nominal_downstream_node,
            config=config,
        )
        records.append(
            {
                "state_id": str(row.state_id),
                "target_type": target_type,
                "target_id": target_id,
                "selected_unit_ids": _serialize_int_list(selected_unit_ids),
                "base_state_id": base_state_id,
                "nominal_upstream_node": nominal_upstream_node,
                "nominal_downstream_node": nominal_downstream_node,
                "nominal_base_link_ids": _serialize_int_list(nominal_link_ids),
                "nominal_base_node_ids": _serialize_int_list(nominal_node_ids),
                "nominal_n_links": int(len(nominal_link_ids)),
                "nominal_n_nodes": int(len(nominal_node_ids)),
                "K_eq_nominal_base": solve["K_eq"],
                "G_eq_nominal_base": solve["G_eq"],
                "Q_unit_nominal_base": solve["Q_unit"],
                "solver_success_nominal": bool(solve["solver_success"]),
                "solver_method_nominal": str(solve["solver_method"]),
                "solver_message_nominal": str(solve["solver_message"]),
                "solver_nfev_nominal": solve["solver_nfev"],
                "residual_norm_nominal": solve["residual_norm"],
                "residual_relative_nominal": solve["residual_relative"],
                "resistance_law": config.resistance_law,
                "geometry_method": config.geometry_method,
                "kb_mode": resolved_kb.get("kb_mode") if resolved_kb is not None else config.kb_mode,
                "kb_value": resolved_kb.get("kb_value") if resolved_kb is not None else config.kb_value,
                "kb_source_method": resolved_kb.get("kb_source_method", "") if resolved_kb is not None else "",
            }
        )
    if not records:
        return _empty_base_nominal_targets_frame()
    return pd.DataFrame.from_records(records).sort_values(["state_id", "target_type", "target_id"], kind="mergesort").reset_index(drop=True)


def _parse_lineage_lists(lineage: pd.DataFrame) -> pd.DataFrame:
    result = lineage.copy()
    result["id_link"] = pd.to_numeric(result["id_link"], errors="coerce").astype(int)
    result["_matched_parent_link_ids"] = result.get("matched_parent_link_ids", "").map(_parse_int_list)
    result["_matched_parent_node_path"] = result.get("matched_parent_node_path", "").map(_parse_int_list)
    if "lineage_type" not in result.columns:
        result["lineage_type"] = ""
    result["_is_changed_lineage"] = result["lineage_type"].astype(str).str.strip().isin(CHANGED_LINEAGE_TYPES)
    return result


def _coerce_bool(value: Any) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if value is None or value is pd.NA:
        return False
    if isinstance(value, (int, np.integer, float, np.floating)):
        if pd.isna(value):
            return False
        return bool(value)
    text = str(value).strip().lower()
    if not text or text == "nan":
        return False
    return text in {"1", "true", "t", "yes", "y"}


def _parse_node_match(node_match: pd.DataFrame) -> pd.DataFrame:
    result = node_match.copy()
    for column in ("child_id_node", "matched_parent_node_id", "parent_node_order"):
        if column in result.columns:
            result[column] = pd.to_numeric(result[column], errors="coerce")
            if result[column].notna().all():
                result[column] = result[column].astype(int)
    if "match_within_tolerance" not in result.columns:
        result["match_within_tolerance"] = False
    result["_is_stable_match"] = result["match_within_tolerance"].map(_coerce_bool) & result.get(
        "matched_parent_node_id",
        pd.Series(index=result.index, dtype=float),
    ).notna()
    return result


def _build_simple_digraph(links: pd.DataFrame) -> nx.DiGraph:
    graph = nx.DiGraph()
    for row in links.itertuples(index=False):
        graph.add_edge(int(row.id_us_node), int(row.id_ds_node))
    return graph


def _build_simple_graph(links: pd.DataFrame) -> nx.Graph:
    graph = nx.Graph()
    for row in links.itertuples(index=False):
        graph.add_edge(int(row.id_us_node), int(row.id_ds_node))
    return graph


def _extract_links_between_anchors(
    links: gpd.GeoDataFrame | pd.DataFrame,
    *,
    upstream_node: int | None,
    downstream_node: int | None,
) -> gpd.GeoDataFrame | pd.DataFrame:
    if upstream_node is None or downstream_node is None or links.empty:
        return links.iloc[0:0].copy()
    graph = _build_simple_digraph(links)
    upstream = int(upstream_node)
    downstream = int(downstream_node)
    if not graph.has_node(upstream) or not graph.has_node(downstream) or not nx.has_path(graph, upstream, downstream):
        return links.iloc[0:0].copy()
    upstream_reachable = nx.descendants(graph, upstream) | {upstream}
    downstream_ancestors = nx.ancestors(graph, downstream) | {downstream}
    node_set = upstream_reachable.intersection(downstream_ancestors)
    if not node_set:
        return links.iloc[0:0].copy()
    mask = (
        pd.to_numeric(links["id_us_node"], errors="coerce").isin(node_set)
        & pd.to_numeric(links["id_ds_node"], errors="coerce").isin(node_set)
    )
    return links.loc[mask].copy()


def _build_changed_link_candidates(
    *,
    child_links: gpd.GeoDataFrame,
    lineage: pd.DataFrame,
    node_match: pd.DataFrame,
) -> tuple[gpd.GeoDataFrame, set[int], dict[int, int]]:
    lineage_columns = ["id_link", "_matched_parent_link_ids", "_matched_parent_node_path", "_is_changed_lineage", "lineage_type"]
    lineage_merge = lineage.loc[:, [column for column in lineage_columns if column in lineage.columns]].copy()
    result = child_links.merge(lineage_merge, on="id_link", how="left")
    if "_matched_parent_link_ids" not in result.columns:
        result["_matched_parent_link_ids"] = [[] for _ in range(len(result))]
    result["_matched_parent_link_ids"] = result["_matched_parent_link_ids"].map(_parse_int_list)
    if "_matched_parent_node_path" not in result.columns:
        result["_matched_parent_node_path"] = [[] for _ in range(len(result))]
    result["_matched_parent_node_path"] = result["_matched_parent_node_path"].map(_parse_int_list)
    if "_is_changed_lineage" not in result.columns:
        result["_is_changed_lineage"] = False
    result["_is_changed_lineage"] = result["_is_changed_lineage"].fillna(False).map(_coerce_bool)

    stable_match_by_child = {
        int(child_id): int(parent_id)
        for child_id, parent_id, is_stable in zip(
            pd.to_numeric(node_match.get("child_id_node"), errors="coerce"),
            pd.to_numeric(node_match.get("matched_parent_node_id"), errors="coerce"),
            node_match.get("_is_stable_match", pd.Series(dtype=bool)).map(_coerce_bool),
            strict=False,
        )
        if pd.notna(child_id) and pd.notna(parent_id) and is_stable
    }
    unstable_child_nodes = {
        int(child_id)
        for child_id, is_stable in zip(
            pd.to_numeric(node_match.get("child_id_node"), errors="coerce"),
            node_match.get("_is_stable_match", pd.Series(dtype=bool)).map(_coerce_bool),
            strict=False,
        )
        if pd.notna(child_id) and not is_stable
    }

    result["_touches_unstable_node"] = (
        pd.to_numeric(result["id_us_node"], errors="coerce").isin(unstable_child_nodes)
        | pd.to_numeric(result["id_ds_node"], errors="coerce").isin(unstable_child_nodes)
    )
    result["_candidate_changed"] = result["_is_changed_lineage"] | result["_touches_unstable_node"]
    return result, unstable_child_nodes, stable_match_by_child


def _build_parent_to_child_stable_matches(node_match: pd.DataFrame) -> dict[int, list[int]]:
    mapping: dict[int, list[int]] = {}
    if node_match.empty:
        return mapping
    stable_rows = node_match.loc[node_match["_is_stable_match"].fillna(False).map(_coerce_bool)].copy()
    if stable_rows.empty:
        return mapping
    stable_rows = stable_rows.assign(
        _child_id_node=pd.to_numeric(stable_rows["child_id_node"], errors="coerce"),
        _matched_parent_node_id=pd.to_numeric(stable_rows["matched_parent_node_id"], errors="coerce"),
        _parent_node_order=pd.to_numeric(stable_rows.get("parent_node_order"), errors="coerce"),
    )
    stable_rows = stable_rows.sort_values(
        by=["_matched_parent_node_id", "_parent_node_order", "_child_id_node"],
        kind="mergesort",
    )
    for row in stable_rows[["_child_id_node", "_matched_parent_node_id"]].itertuples(index=False, name=None):
        child_id_node, matched_parent_node_id = row
        if pd.isna(child_id_node) or pd.isna(matched_parent_node_id):
            continue
        mapping.setdefault(int(matched_parent_node_id), []).append(int(child_id_node))
    return mapping


def _resolve_target_parent_node_ids(target: Any) -> set[int]:
    node_ids = set(_parse_int_list(getattr(target, "nominal_base_node_ids", "")))
    upstream_node = _optional_int(getattr(target, "nominal_upstream_node", pd.NA))
    downstream_node = _optional_int(getattr(target, "nominal_downstream_node", pd.NA))
    if upstream_node is not None:
        node_ids.add(int(upstream_node))
    if downstream_node is not None:
        node_ids.add(int(downstream_node))
    return node_ids


def _find_nearest_parent_anchor_from_missing(
    graph: nx.DiGraph,
    *,
    missing_parent_nodes: set[int],
    stable_parent_nodes: set[int],
    direction: str,
) -> int | None:
    if not missing_parent_nodes or not stable_parent_nodes:
        return None
    visited = {int(node_id) for node_id in missing_parent_nodes}
    frontier: deque[int] = deque(sorted(visited))
    while frontier:
        current = int(frontier.popleft())
        neighbors = (
            sorted(int(node_id) for node_id in graph.predecessors(current))
            if direction == "upstream"
            else sorted(int(node_id) for node_id in graph.successors(current))
        )
        for neighbor in neighbors:
            if neighbor in visited:
                continue
            if neighbor in stable_parent_nodes and neighbor not in missing_parent_nodes:
                return neighbor
            visited.add(neighbor)
            frontier.append(neighbor)
    return None


def _select_child_anchor_pair(
    *,
    child_links: gpd.GeoDataFrame,
    parent_to_child_matches: Mapping[int, Sequence[int]],
    parent_upstream_anchor: int | None,
    parent_downstream_anchor: int | None,
) -> tuple[int | None, int | None, gpd.GeoDataFrame]:
    empty = child_links.iloc[0:0].copy()
    if parent_upstream_anchor is None or parent_downstream_anchor is None:
        return None, None, empty
    upstream_candidates = [int(node_id) for node_id in parent_to_child_matches.get(int(parent_upstream_anchor), [])]
    downstream_candidates = [int(node_id) for node_id in parent_to_child_matches.get(int(parent_downstream_anchor), [])]
    if not upstream_candidates or not downstream_candidates:
        return None, None, empty
    graph = _build_simple_digraph(child_links)
    best: tuple[float, float, int, int, gpd.GeoDataFrame] | None = None
    for child_upstream in upstream_candidates:
        for child_downstream in downstream_candidates:
            if child_upstream == child_downstream:
                continue
            if child_upstream not in graph or child_downstream not in graph or not nx.has_path(graph, child_upstream, child_downstream):
                continue
            corridor_links = _extract_links_between_anchors(
                child_links,
                upstream_node=child_upstream,
                downstream_node=child_downstream,
            )
            if corridor_links.empty:
                continue
            try:
                shortest_length = float(nx.shortest_path_length(graph, child_upstream, child_downstream))
            except nx.NetworkXNoPath:
                continue
            n_links = float(len(corridor_links))
            candidate = (shortest_length, n_links, int(child_upstream), int(child_downstream), corridor_links)
            if best is None or candidate[:4] < best[:4]:
                best = candidate
    if best is None:
        return None, None, empty
    return best[2], best[3], best[4]


def _component_parent_node_ids(
    component_lineage: pd.DataFrame,
    component_node_ids: set[int],
    node_match: pd.DataFrame,
) -> set[int]:
    parent_node_ids: set[int] = set()
    if not component_lineage.empty:
        for values in component_lineage.get("_matched_parent_node_path", pd.Series(dtype=object)).tolist():
            parent_node_ids.update(_parse_int_list(values))
        for column in ("matched_parent_us_node", "matched_parent_ds_node"):
            if column in component_lineage.columns:
                parent_node_ids.update(
                    int(value)
                    for value in pd.to_numeric(component_lineage[column], errors="coerce").dropna().astype(int).tolist()
                )
    if not node_match.empty:
        matches = node_match.loc[
            node_match["_is_stable_match"].fillna(False).map(_coerce_bool)
            & pd.to_numeric(node_match["child_id_node"], errors="coerce").isin(component_node_ids)
        ]
        parent_node_ids.update(
            int(value)
            for value in pd.to_numeric(matches["matched_parent_node_id"], errors="coerce").dropna().astype(int).tolist()
        )
    return parent_node_ids


def _min_graph_distance(
    graph: nx.Graph,
    *,
    source_nodes: set[int],
    target_node: int | None,
) -> float:
    if target_node is None or target_node not in graph or not source_nodes:
        return float("inf")
    best = float("inf")
    for source in source_nodes:
        if source not in graph:
            continue
        try:
            best = min(best, float(nx.shortest_path_length(graph, source, int(target_node))))
        except nx.NetworkXNoPath:
            continue
    return best


def _score_changed_component(
    *,
    component_links: gpd.GeoDataFrame,
    component_lineage: pd.DataFrame,
    component_node_ids: set[int],
    target: Any,
    target_link_ids: set[int],
    node_match: pd.DataFrame,
    unstable_child_nodes: set[int],
    base_node_graph: nx.Graph,
) -> tuple[float, float, float, float, float, float]:
    matched_parent_link_ids = {
        int(parent_link_id)
        for ids in component_lineage.get("_matched_parent_link_ids", pd.Series(dtype=object)).tolist()
        for parent_link_id in _parse_int_list(ids)
    }
    parent_node_ids = _component_parent_node_ids(component_lineage, component_node_ids, node_match)
    boundary_hits = float(
        int(_optional_int(target.nominal_upstream_node) in parent_node_ids)
        + int(_optional_int(target.nominal_downstream_node) in parent_node_ids)
    )
    link_overlap = float(len(matched_parent_link_ids.intersection(target_link_ids)))
    distance_up = _min_graph_distance(
        base_node_graph,
        source_nodes=parent_node_ids,
        target_node=_optional_int(target.nominal_upstream_node),
    )
    distance_down = _min_graph_distance(
        base_node_graph,
        source_nodes=parent_node_ids,
        target_node=_optional_int(target.nominal_downstream_node),
    )
    total_distance = distance_up + distance_down
    distance_score = -total_distance if np.isfinite(total_distance) else -1e9
    unstable_touch_count = float(len(component_node_ids.intersection(unstable_child_nodes)))
    changed_link_count = float(component_lineage.get("_is_changed_lineage", pd.Series(dtype=bool)).fillna(False).map(_coerce_bool).sum())
    total_length = float(pd.to_numeric(component_links.get("len"), errors="coerce").fillna(0.0).sum())
    return (
        boundary_hits,
        link_overlap,
        distance_score,
        unstable_touch_count,
        changed_link_count,
        total_length,
    )


def _reconstruct_upstream_path(prev: dict[int, int], start_node: int, anchor_node: int) -> list[int]:
    path = [int(anchor_node)]
    current = int(anchor_node)
    while current != int(start_node):
        current = int(prev[current])
        path.append(current)
    return path


def _reconstruct_downstream_path(prev: dict[int, int], start_node: int, anchor_node: int) -> list[int]:
    path = [int(anchor_node)]
    current = int(anchor_node)
    while current != int(start_node):
        current = int(prev[current])
        path.append(current)
    return list(reversed(path))


def _find_stable_anchor(
    *,
    graph: nx.DiGraph,
    start_node: int | None,
    direction: str,
    component_node_ids: set[int],
    stable_match_by_child: Mapping[int, int],
) -> tuple[int | None, int | None]:
    if start_node is None:
        return None, None
    start = int(start_node)
    if start in stable_match_by_child:
        return start, int(stable_match_by_child[start])

    frontier: deque[int] = deque([start])
    prev: dict[int, int] = {}
    visited = {start}
    while frontier:
        current = int(frontier.popleft())
        if direction == "upstream":
            neighbors = [int(node_id) for node_id in graph.predecessors(current)]
        else:
            neighbors = [int(node_id) for node_id in graph.successors(current)]
        for neighbor in neighbors:
            if neighbor in visited:
                continue
            if current != start and neighbor in component_node_ids:
                continue
            visited.add(neighbor)
            prev[neighbor] = current
            if neighbor in stable_match_by_child:
                return neighbor, int(stable_match_by_child[neighbor])
            frontier.append(neighbor)
    return None, None


def _resolve_child_objects(
    *,
    registry: pd.DataFrame,
    base_links: gpd.GeoDataFrame,
    nominal_targets: pd.DataFrame,
    config: BubbleEquivalentResistanceConfig,
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    targets_by_state = {str(row.state_id): row for row in nominal_targets.itertuples(index=False)}
    base_node_graph = _build_simple_graph(base_links)
    base_digraph = _build_simple_digraph(base_links)
    for row in registry.itertuples(index=False):
        state_id = str(row.state_id)
        if state_id not in targets_by_state:
            continue
        if str(getattr(row, "status", "complete") or "complete") != "complete":
            continue

        target = targets_by_state[state_id]
        target_link_ids = set(_parse_int_list(target.nominal_base_link_ids))
        child_links_path = _optional_path(getattr(row, "directed_links_path", None))
        child_nodes_path = _optional_path(getattr(row, "directed_nodes_path", None))
        variant_output_dir = _optional_path(getattr(row, "variant_output_dir", None))
        if child_links_path is None or child_nodes_path is None or variant_output_dir is None:
            continue

        lineage_path = _optional_path(getattr(row, "link_lineage_path", None))
        if lineage_path is None:
            lineage_path = variant_output_dir / "matching" / "link_lineage.csv"
        node_match_path = _optional_path(getattr(row, "node_match_path", None))
        if node_match_path is None:
            node_match_path = variant_output_dir / "matching" / "node_match.csv"

        analysis_links_path = _optional_path(getattr(row, "analysis_links_path", None))
        child_links = _load_variant_links(
            directed_links_path=child_links_path,
            variant_output_dir=variant_output_dir,
            analysis_links_path=analysis_links_path,
        )
        child_nodes = _normalize_numeric_id_columns(gpd.read_file(child_nodes_path))
        if not lineage_path.exists():
            continue
        lineage = _parse_lineage_lists(pd.read_csv(lineage_path))
        node_match = _parse_node_match(pd.read_csv(node_match_path)) if node_match_path.exists() else pd.DataFrame()
        target_parent_node_ids = _resolve_target_parent_node_ids(target)
        stable_parent_to_child = _build_parent_to_child_stable_matches(node_match)
        stable_parent_node_ids = set(stable_parent_to_child)
        missing_target_parent_nodes = target_parent_node_ids.difference(stable_parent_node_ids)
        nominal_upstream_node = _optional_int(target.nominal_upstream_node)
        nominal_downstream_node = _optional_int(target.nominal_downstream_node)

        # Prefer the target's own boundaries whenever both survived matching.
        # A collapse of a compound/nested unit commonly removes its internal
        # nodes while retaining its inlet and outlet. Treating those internal
        # removals as a reason to replace nominal endpoints yields a small,
        # non-comparable residual corridor rather than the full target.
        if (
            nominal_upstream_node is not None
            and nominal_downstream_node is not None
            and nominal_upstream_node in stable_parent_node_ids
            and nominal_downstream_node in stable_parent_node_ids
        ):
            parent_upstream_anchor = nominal_upstream_node
            parent_downstream_anchor = nominal_downstream_node
        elif missing_target_parent_nodes:
            parent_upstream_anchor = _find_nearest_parent_anchor_from_missing(
                base_digraph,
                missing_parent_nodes=missing_target_parent_nodes,
                stable_parent_nodes=stable_parent_node_ids,
                direction="upstream",
            )
            parent_downstream_anchor = _find_nearest_parent_anchor_from_missing(
                base_digraph,
                missing_parent_nodes=missing_target_parent_nodes,
                stable_parent_nodes=stable_parent_node_ids,
                direction="downstream",
            )

        else:
            parent_upstream_anchor = None
            parent_downstream_anchor = None

        if parent_upstream_anchor is not None and parent_downstream_anchor is not None:
            child_upstream_node, child_downstream_node, corridor_links = _select_child_anchor_pair(
                child_links=child_links,
                parent_to_child_matches=stable_parent_to_child,
                parent_upstream_anchor=parent_upstream_anchor,
                parent_downstream_anchor=parent_downstream_anchor,
            )
            matched_parent_links = _extract_links_between_anchors(
                base_links,
                upstream_node=parent_upstream_anchor,
                downstream_node=parent_downstream_anchor,
            )
            if (
                child_upstream_node is not None
                and child_downstream_node is not None
                and not corridor_links.empty
                and not matched_parent_links.empty
            ):
                corridor_link_ids = sorted(pd.to_numeric(corridor_links["id_link"], errors="coerce").dropna().astype(int).tolist())
                corridor_node_ids = _incident_node_ids(corridor_links)
                corridor_nodes = child_nodes.loc[pd.to_numeric(child_nodes["id_node"], errors="coerce").isin(corridor_node_ids)].copy()
                matched_parent_link_ids = sorted(
                    pd.to_numeric(matched_parent_links["id_link"], errors="coerce").dropna().astype(int).tolist()
                )
                matched_parent_node_path = _resolve_linear_node_path(
                    matched_parent_links,
                    upstream_node=parent_upstream_anchor,
                    downstream_node=parent_downstream_anchor,
                )
                boundaries_match_nominal_target, boundary_relation = _classify_boundary_relation(
                    nominal_upstream_node=_optional_int(target.nominal_upstream_node),
                    nominal_downstream_node=_optional_int(target.nominal_downstream_node),
                    matched_node_path=matched_parent_node_path,
                )
                solve, _ = _compute_equivalent_resistance(
                    corridor_links,
                    corridor_nodes,
                    upstream_node=child_upstream_node,
                    downstream_node=child_downstream_node,
                    config=config,
                )
                records.append(
                    {
                        "state_id": state_id,
                        "target_type": target.target_type,
                        "target_id": target.target_id,
                        "selected_unit_ids": target.selected_unit_ids,
                        "child_object_id": f"{state_id}__obj_001",
                        "child_object_type": _classify_child_object_type(corridor_links),
                        "child_link_ids": _serialize_int_list(corridor_link_ids),
                        "child_node_ids": _serialize_int_list(corridor_node_ids),
                        "child_n_links": int(len(corridor_link_ids)),
                        "child_n_nodes": int(len(corridor_node_ids)),
                        "child_upstream_node": child_upstream_node,
                        "child_downstream_node": child_downstream_node,
                        "child_boundary_match_upstream_parent_node": parent_upstream_anchor if parent_upstream_anchor is not None else np.nan,
                        "child_boundary_match_downstream_parent_node": parent_downstream_anchor if parent_downstream_anchor is not None else np.nan,
                        "matched_parent_node_path": _serialize_int_list(matched_parent_node_path),
                        "matched_parent_link_ids": _serialize_int_list(matched_parent_link_ids),
                        "boundaries_match_nominal_target": bool(boundaries_match_nominal_target),
                        "boundary_relation": boundary_relation,
                        "K_variant_realized": solve["K_eq"],
                        "G_variant_realized": solve["G_eq"],
                        "Q_unit_variant": solve["Q_unit"],
                        "solver_success_variant": bool(solve["solver_success"]),
                        "solver_method_variant": str(solve["solver_method"]),
                        "solver_message_variant": str(solve["solver_message"]),
                        "solver_nfev_variant": solve["solver_nfev"],
                        "residual_norm_variant": solve["residual_norm"],
                        "residual_relative_variant": solve["residual_relative"],
                    }
                )
                continue

        candidate_links, unstable_child_nodes, stable_match_by_child = _build_changed_link_candidates(
            child_links=child_links,
            lineage=lineage,
            node_match=node_match,
        )
        candidate_links = candidate_links.loc[candidate_links["_candidate_changed"]].copy()
        if candidate_links.empty:
            lineage_overlap = lineage.loc[lineage["_matched_parent_link_ids"].map(lambda ids: bool(target_link_ids.intersection(ids)))]
            candidate_child_link_ids = sorted(lineage_overlap["id_link"].astype(int).tolist())
            candidate_links = child_links.loc[pd.to_numeric(child_links["id_link"], errors="coerce").isin(candidate_child_link_ids)].copy()
            if candidate_links.empty:
                continue
            candidate_links["_matched_parent_link_ids"] = candidate_links.get("_matched_parent_link_ids", [[] for _ in range(len(candidate_links))])
            candidate_links["_matched_parent_node_path"] = candidate_links.get("_matched_parent_node_path", [[] for _ in range(len(candidate_links))])
            candidate_links["_is_changed_lineage"] = True

        candidate_graph = _build_simple_graph(candidate_links)
        if candidate_graph.number_of_edges() == 0:
            continue

        best_component_links: gpd.GeoDataFrame | None = None
        best_component_score: tuple[float, float, float, float, float, float] | None = None
        for component_nodes in nx.connected_components(candidate_graph):
            component_node_ids = {int(node_id) for node_id in component_nodes}
            component_links = candidate_links.loc[
                pd.to_numeric(candidate_links["id_us_node"], errors="coerce").isin(component_node_ids)
                & pd.to_numeric(candidate_links["id_ds_node"], errors="coerce").isin(component_node_ids)
            ].copy()
            if component_links.empty:
                continue
            component_lineage = lineage.loc[lineage["id_link"].isin(pd.to_numeric(component_links["id_link"], errors="coerce"))].copy()
            component_score = _score_changed_component(
                component_links=component_links,
                component_lineage=component_lineage,
                component_node_ids=component_node_ids,
                target=target,
                target_link_ids=target_link_ids,
                node_match=node_match,
                unstable_child_nodes=unstable_child_nodes,
                base_node_graph=base_node_graph,
            )
            if best_component_score is None or component_score > best_component_score:
                best_component_score = component_score
                best_component_links = component_links

        if best_component_links is None or best_component_links.empty:
            continue

        component_link_ids = sorted(pd.to_numeric(best_component_links["id_link"], errors="coerce").dropna().astype(int).tolist())
        component_node_ids = set(_incident_node_ids(best_component_links))
        component_upstream_node, component_downstream_node = _resolve_component_boundaries(best_component_links)
        child_graph = _build_simple_digraph(child_links)
        child_upstream_node, parent_upstream_anchor = _find_stable_anchor(
            graph=child_graph,
            start_node=component_upstream_node,
            direction="upstream",
            component_node_ids=component_node_ids,
            stable_match_by_child=stable_match_by_child,
        )
        child_downstream_node, parent_downstream_anchor = _find_stable_anchor(
            graph=child_graph,
            start_node=component_downstream_node,
            direction="downstream",
            component_node_ids=component_node_ids,
            stable_match_by_child=stable_match_by_child,
        )
        if child_upstream_node is None or child_downstream_node is None:
            child_upstream_node = component_upstream_node
            child_downstream_node = component_downstream_node
        corridor_links = _extract_links_between_anchors(
            child_links,
            upstream_node=child_upstream_node,
            downstream_node=child_downstream_node,
        )
        if corridor_links.empty:
            corridor_links = best_component_links.copy()

        corridor_link_ids = sorted(pd.to_numeric(corridor_links["id_link"], errors="coerce").dropna().astype(int).tolist())
        corridor_node_ids = _incident_node_ids(corridor_links)
        corridor_nodes = child_nodes.loc[pd.to_numeric(child_nodes["id_node"], errors="coerce").isin(corridor_node_ids)].copy()
        corridor_lineage = lineage.loc[lineage["id_link"].isin(corridor_link_ids)].copy()
        matched_parent_link_ids = sorted(
            {
                int(parent_link_id)
                for ids in corridor_lineage.get("_matched_parent_link_ids", pd.Series(dtype=object)).tolist()
                for parent_link_id in _parse_int_list(ids)
            }
        )
        matched_parent_links = _extract_links_between_anchors(
            base_links,
            upstream_node=parent_upstream_anchor,
            downstream_node=parent_downstream_anchor,
        )
        if matched_parent_links.empty and matched_parent_link_ids:
            matched_parent_links = base_links.loc[
                pd.to_numeric(base_links["id_link"], errors="coerce").isin(matched_parent_link_ids)
            ].copy()
        matched_parent_node_path = _resolve_linear_node_path(
            matched_parent_links,
            upstream_node=parent_upstream_anchor,
            downstream_node=parent_downstream_anchor,
        )
        boundaries_match_nominal_target, boundary_relation = _classify_boundary_relation(
            nominal_upstream_node=_optional_int(target.nominal_upstream_node),
            nominal_downstream_node=_optional_int(target.nominal_downstream_node),
            matched_node_path=matched_parent_node_path,
        )
        solve, _ = _compute_equivalent_resistance(
            corridor_links,
            corridor_nodes,
            upstream_node=child_upstream_node,
            downstream_node=child_downstream_node,
            config=config,
        )
        records.append(
            {
                "state_id": state_id,
                "target_type": target.target_type,
                "target_id": target.target_id,
                "selected_unit_ids": target.selected_unit_ids,
                "child_object_id": f"{state_id}__obj_001",
                "child_object_type": _classify_child_object_type(corridor_links),
                "child_link_ids": _serialize_int_list(corridor_link_ids),
                "child_node_ids": _serialize_int_list(corridor_node_ids),
                "child_n_links": int(len(corridor_link_ids)),
                "child_n_nodes": int(len(corridor_node_ids)),
                "child_upstream_node": child_upstream_node,
                "child_downstream_node": child_downstream_node,
                "child_boundary_match_upstream_parent_node": parent_upstream_anchor if parent_upstream_anchor is not None else np.nan,
                "child_boundary_match_downstream_parent_node": parent_downstream_anchor if parent_downstream_anchor is not None else np.nan,
                "matched_parent_node_path": _serialize_int_list(matched_parent_node_path),
                "matched_parent_link_ids": _serialize_int_list(matched_parent_link_ids),
                "boundaries_match_nominal_target": bool(boundaries_match_nominal_target),
                "boundary_relation": boundary_relation,
                "K_variant_realized": solve["K_eq"],
                "G_variant_realized": solve["G_eq"],
                "Q_unit_variant": solve["Q_unit"],
                "solver_success_variant": bool(solve["solver_success"]),
                "solver_method_variant": str(solve["solver_method"]),
                "solver_message_variant": str(solve["solver_message"]),
                "solver_nfev_variant": solve["solver_nfev"],
                "residual_norm_variant": solve["residual_norm"],
                "residual_relative_variant": solve["residual_relative"],
            }
        )
    if not records:
        return _empty_child_realized_objects_frame()
    return pd.DataFrame.from_records(records).sort_values(["state_id", "child_object_id"], kind="mergesort").reset_index(drop=True)


def _build_matched_base_corridors(
    *,
    base_links: gpd.GeoDataFrame,
    base_nodes: gpd.GeoDataFrame,
    child_realized_objects: pd.DataFrame,
    config: BubbleEquivalentResistanceConfig,
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for row in child_realized_objects.itertuples(index=False):
        matched_base_link_ids = _parse_int_list(getattr(row, "matched_parent_link_ids"))
        matched_base_upstream_node = _optional_int(getattr(row, "child_boundary_match_upstream_parent_node", pd.NA))
        matched_base_downstream_node = _optional_int(getattr(row, "child_boundary_match_downstream_parent_node", pd.NA))
        matched_links = _extract_links_between_anchors(
            base_links,
            upstream_node=matched_base_upstream_node,
            downstream_node=matched_base_downstream_node,
        )
        if matched_links.empty and matched_base_link_ids:
            matched_links = base_links.loc[pd.to_numeric(base_links["id_link"], errors="coerce").isin(matched_base_link_ids)].copy()
        if matched_links.empty:
            continue
        if matched_base_upstream_node is None or matched_base_downstream_node is None:
            matched_base_upstream_node, matched_base_downstream_node = _resolve_component_boundaries(matched_links)
        matched_base_link_ids = sorted(pd.to_numeric(matched_links["id_link"], errors="coerce").dropna().astype(int).tolist())
        matched_base_node_path = _resolve_linear_node_path(
            matched_links,
            upstream_node=matched_base_upstream_node,
            downstream_node=matched_base_downstream_node,
        )
        matched_node_ids = _incident_node_ids(matched_links)
        matched_nodes = base_nodes.loc[pd.to_numeric(base_nodes["id_node"], errors="coerce").isin(matched_node_ids)].copy()
        solve, _ = _compute_equivalent_resistance(
            matched_links,
            matched_nodes,
            upstream_node=matched_base_upstream_node,
            downstream_node=matched_base_downstream_node,
            config=config,
        )
        records.append(
            {
                "state_id": row.state_id,
                "child_object_id": row.child_object_id,
                "target_type": row.target_type,
                "target_id": row.target_id,
                "matched_base_upstream_node": matched_base_upstream_node,
                "matched_base_downstream_node": matched_base_downstream_node,
                "matched_base_link_ids": _serialize_int_list(matched_base_link_ids),
                "matched_base_node_ids": _serialize_int_list(matched_node_ids),
                "matched_base_node_path": _serialize_int_list(matched_base_node_path),
                "matched_base_n_links": int(len(matched_base_link_ids)),
                "matched_base_n_nodes": int(len(matched_node_ids)),
                "K_eq_matched_base": solve["K_eq"],
                "G_eq_matched_base": solve["G_eq"],
                "Q_unit_matched_base": solve["Q_unit"],
                "solver_success_matched": bool(solve["solver_success"]),
                "solver_method_matched": str(solve["solver_method"]),
                "solver_message_matched": str(solve["solver_message"]),
                "solver_nfev_matched": solve["solver_nfev"],
                "residual_norm_matched": solve["residual_norm"],
                "residual_relative_matched": solve["residual_relative"],
                "boundaries_match_nominal_target": bool(row.boundaries_match_nominal_target),
                "boundary_relation": row.boundary_relation,
            }
        )
    if not records:
        return _empty_matched_base_corridors_frame()
    return pd.DataFrame.from_records(records).sort_values(["state_id", "child_object_id"], kind="mergesort").reset_index(drop=True)


def _valid_positive(value: Any) -> bool:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return False
    return np.isfinite(numeric) and numeric > 0


def _build_comparison_frame(
    *,
    nominal_targets: pd.DataFrame,
    child_realized_objects: pd.DataFrame,
    matched_base_corridors: pd.DataFrame,
) -> pd.DataFrame:
    if nominal_targets.empty or child_realized_objects.empty or matched_base_corridors.empty:
        return _empty_comparison_frame()
    records: list[dict[str, Any]] = []
    nominal_by_state = {str(row.state_id): row for row in nominal_targets.itertuples(index=False)}
    matched_by_object = {str(row.child_object_id): row for row in matched_base_corridors.itertuples(index=False)}

    for child_row in child_realized_objects.itertuples(index=False):
        nominal = nominal_by_state.get(str(child_row.state_id))
        matched = matched_by_object.get(str(child_row.child_object_id))
        if nominal is None or matched is None:
            continue
        lambda_nominal = np.nan
        lambda_matched = np.nan
        gamma_nominal = np.nan
        gamma_matched = np.nan
        valid_nominal = bool(child_row.boundaries_match_nominal_target) and _valid_positive(child_row.K_variant_realized) and _valid_positive(nominal.K_eq_nominal_base)
        valid_matched = _valid_positive(child_row.K_variant_realized) and _valid_positive(matched.K_eq_matched_base)
        if valid_nominal:
            metrics = compare_states(nominal.K_eq_nominal_base, child_row.K_variant_realized)
            lambda_nominal = float(metrics["Lambda_K"])
            gamma_nominal = float(metrics["Lambda_G"])
        if valid_matched:
            metrics = compare_states(matched.K_eq_matched_base, child_row.K_variant_realized)
            lambda_matched = float(metrics["Lambda_K"])
            gamma_matched = float(metrics["Lambda_G"])
        records.append(
            {
                "state_id": child_row.state_id,
                "child_object_id": child_row.child_object_id,
                "target_type": child_row.target_type,
                "target_id": child_row.target_id,
                "K_eq_nominal_base": nominal.K_eq_nominal_base,
                "K_eq_matched_base": matched.K_eq_matched_base,
                "K_variant_realized": child_row.K_variant_realized,
                "Lambda_nominal": lambda_nominal,
                "Lambda_matched": lambda_matched,
                "Gamma_nominal": gamma_nominal,
                "Gamma_matched": gamma_matched,
                "boundaries_match_nominal_target": bool(child_row.boundaries_match_nominal_target),
                "boundary_relation": child_row.boundary_relation,
                "comparison_valid_for_nominal": bool(valid_nominal),
                "comparison_valid_for_matched": bool(valid_matched),
            }
        )
    if not records:
        return _empty_comparison_frame()
    return pd.DataFrame.from_records(records).sort_values(["state_id", "child_object_id"], kind="mergesort").reset_index(drop=True)


def _selected_domain_ids(domain: pd.DataFrame, role: str, column: str) -> list[int]:
    if domain.empty or "domain_role" not in domain.columns or column not in domain.columns:
        return []
    values = pd.to_numeric(
        domain.loc[domain["domain_role"].astype(str).eq(role), column],
        errors="coerce",
    )
    return sorted(values.dropna().astype(int).unique().tolist())


def _preserve_residual_corridor_diagnostics(
    frame: pd.DataFrame,
    columns: Sequence[str],
) -> pd.DataFrame:
    """Retain superseded isolated-corridor values as named diagnostics."""
    result = frame.copy()
    for column in columns:
        if column in result.columns:
            result[f"residual_corridor_{column}_diagnostic"] = result[column]
    return result


def _apply_selected_hydraulic_domain_aggregates(
    *,
    experiment_path: Path,
    nominal_targets: pd.DataFrame,
    child_realized_objects: pd.DataFrame,
    matched_base_corridors: pd.DataFrame,
    comparison: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, int]:
    """Make the state-transition hydraulic domain authoritative in root tables."""
    transition_path = experiment_path / "equivalent_transition_registry.csv"
    if not transition_path.exists():
        return child_realized_objects, matched_base_corridors, comparison, 0
    transitions = pd.read_csv(transition_path)
    required = {
        "transition_id",
        "child_state_id",
        "selected_domain_status",
        "parent_solver_success",
        "child_solver_success",
        "K_before",
        "K_after",
        "Lambda_K",
    }
    if transitions.empty or not required.issubset(transitions.columns):
        return child_realized_objects, matched_base_corridors, comparison, 0
    transitions = transitions.loc[
        transitions["selected_domain_status"].astype(str).eq("complete")
        & transitions["parent_solver_success"].fillna(False).astype(bool)
        & transitions["child_solver_success"].fillna(False).astype(bool)
    ].copy()
    if transitions.empty:
        return child_realized_objects, matched_base_corridors, comparison, 0

    child = _preserve_residual_corridor_diagnostics(
        child_realized_objects,
        ("child_link_ids", "child_node_ids", "K_variant_realized", "G_variant_realized", "Q_unit_variant"),
    )
    matched = _preserve_residual_corridor_diagnostics(
        matched_base_corridors,
        ("matched_base_link_ids", "matched_base_node_ids", "K_eq_matched_base", "G_eq_matched_base", "Q_unit_matched_base"),
    )
    compared = _preserve_residual_corridor_diagnostics(
        comparison,
        ("K_eq_matched_base", "K_variant_realized", "Lambda_matched", "Gamma_matched", "boundary_relation"),
    )
    nominal_by_state = {
        str(row.state_id): row for row in nominal_targets.itertuples(index=False)
    }

    def ensure_row(frame: pd.DataFrame, record: Mapping[str, Any]) -> tuple[pd.DataFrame, int]:
        state_id = str(record["state_id"])
        indices = frame.index[frame["state_id"].astype(str).eq(state_id)].tolist() if "state_id" in frame else []
        if indices:
            return frame, int(indices[0])
        frame = pd.concat([frame, pd.DataFrame([dict(record)])], ignore_index=True, sort=False)
        return frame, int(frame.index[-1])

    applied = 0
    for transition in transitions.itertuples(index=False):
        state_id = str(transition.child_state_id)
        domain_path = experiment_path / "states" / state_id / "equivalent_resistance" / "selected_hydraulic_domain_links.csv"
        if not domain_path.exists():
            continue
        domain = pd.read_csv(domain_path)
        parent_link_ids = _selected_domain_ids(domain, "parent", "id_link")
        child_link_ids = _selected_domain_ids(domain, "child", "id_link")
        parent_node_ids = sorted(set(_selected_domain_ids(domain, "parent", "id_us_node") + _selected_domain_ids(domain, "parent", "id_ds_node")))
        child_node_ids = sorted(set(_selected_domain_ids(domain, "child", "id_us_node") + _selected_domain_ids(domain, "child", "id_ds_node")))
        if not parent_link_ids or not child_link_ids:
            continue

        nominal = nominal_by_state.get(state_id)
        target_type = getattr(nominal, "target_type", "") if nominal is not None else ""
        target_id = getattr(nominal, "target_id", np.nan) if nominal is not None else np.nan
        selected_unit_ids = getattr(transition, "selected_unit_ids", "")
        child_object_id = f"{state_id}__selected_hydraulic_domain"
        nominal_upstream = _optional_int(getattr(nominal, "nominal_upstream_node", pd.NA)) if nominal is not None else None
        nominal_downstream = _optional_int(getattr(nominal, "nominal_downstream_node", pd.NA)) if nominal is not None else None
        parent_upstream = _optional_int(getattr(transition, "parent_upstream_boundary_node", pd.NA))
        parent_downstream = _optional_int(getattr(transition, "parent_downstream_boundary_node", pd.NA))
        child_upstream = _optional_int(getattr(transition, "child_upstream_boundary_node", pd.NA))
        child_downstream = _optional_int(getattr(transition, "child_downstream_boundary_node", pd.NA))
        exact_nominal = nominal_upstream is not None and nominal_downstream is not None and parent_upstream == nominal_upstream and parent_downstream == nominal_downstream
        boundary_relation = "exact_match" if exact_nominal else "selected_hydraulic_domain"

        child_record = {
            "state_id": state_id, "target_type": target_type, "target_id": target_id,
            "selected_unit_ids": selected_unit_ids, "child_object_id": child_object_id,
            "child_object_type": str(getattr(transition, "hydraulic_domain_type", "selected_hydraulic_domain")),
            "child_link_ids": _serialize_int_list(child_link_ids), "child_node_ids": _serialize_int_list(child_node_ids),
            "child_n_links": len(child_link_ids), "child_n_nodes": len(child_node_ids),
            "child_upstream_node": child_upstream, "child_downstream_node": child_downstream,
            "child_boundary_match_upstream_parent_node": parent_upstream,
            "child_boundary_match_downstream_parent_node": parent_downstream,
            "matched_parent_node_path": "", "matched_parent_link_ids": _serialize_int_list(parent_link_ids),
            "boundaries_match_nominal_target": bool(exact_nominal), "boundary_relation": boundary_relation,
            "K_variant_realized": getattr(transition, "K_after"), "G_variant_realized": getattr(transition, "G_after", np.nan),
            "Q_unit_variant": getattr(transition, "Q_unit_after", np.nan), "solver_success_variant": True,
            "solver_method_variant": str(getattr(transition, "child_solver_method", "")),
            "solver_message_variant": "authoritative selected hydraulic-domain transition",
            "solver_nfev_variant": getattr(transition, "child_solver_nfev", np.nan),
            "residual_norm_variant": getattr(transition, "residual_norm_after", np.nan),
            "residual_relative_variant": getattr(transition, "residual_relative_after", np.nan),
            "comparison_scope": "selected_hydraulic_domain", "authoritative_transition_id": str(transition.transition_id),
        }
        child, child_index = ensure_row(child, child_record)
        for key, value in child_record.items():
            child.loc[child_index, key] = value

        matched_record = {
            "state_id": state_id, "child_object_id": child_object_id, "target_type": target_type, "target_id": target_id,
            "matched_base_upstream_node": parent_upstream, "matched_base_downstream_node": parent_downstream,
            "matched_base_link_ids": _serialize_int_list(parent_link_ids), "matched_base_node_ids": _serialize_int_list(parent_node_ids),
            "matched_base_node_path": "", "matched_base_n_links": len(parent_link_ids), "matched_base_n_nodes": len(parent_node_ids),
            "K_eq_matched_base": getattr(transition, "K_before"), "G_eq_matched_base": getattr(transition, "G_before", np.nan),
            "Q_unit_matched_base": getattr(transition, "Q_unit_before", np.nan), "solver_success_matched": True,
            "solver_method_matched": str(getattr(transition, "parent_solver_method", "")),
            "solver_message_matched": "authoritative selected hydraulic-domain transition",
            "solver_nfev_matched": getattr(transition, "parent_solver_nfev", np.nan),
            "residual_norm_matched": getattr(transition, "residual_norm_before", np.nan),
            "residual_relative_matched": getattr(transition, "residual_relative_before", np.nan),
            "boundaries_match_nominal_target": bool(exact_nominal), "boundary_relation": boundary_relation,
            "comparison_scope": "selected_hydraulic_domain", "authoritative_transition_id": str(transition.transition_id),
        }
        matched, matched_index = ensure_row(matched, matched_record)
        for key, value in matched_record.items():
            matched.loc[matched_index, key] = value

        compared_record = {
            "state_id": state_id, "child_object_id": child_object_id, "target_type": target_type, "target_id": target_id,
            "K_eq_nominal_base": getattr(nominal, "K_eq_nominal_base", np.nan) if nominal is not None else np.nan,
            "K_eq_matched_base": getattr(transition, "K_before"), "K_variant_realized": getattr(transition, "K_after"),
            "Lambda_nominal": np.nan, "Lambda_matched": getattr(transition, "Lambda_K"), "Gamma_nominal": np.nan,
            "Gamma_matched": getattr(transition, "Lambda_G", np.nan), "boundaries_match_nominal_target": bool(exact_nominal),
            "boundary_relation": boundary_relation, "comparison_valid_for_nominal": False,
            "comparison_valid_for_matched": True, "comparison_scope": "selected_hydraulic_domain",
            "hydraulic_domain_type": str(getattr(transition, "hydraulic_domain_type", "")),
            "authoritative_transition_id": str(transition.transition_id),
        }
        compared, compared_index = ensure_row(compared, compared_record)
        for key, value in compared_record.items():
            compared.loc[compared_index, key] = value
        applied += 1

    sort_columns = ["state_id", "child_object_id"]
    return (
        child.sort_values(sort_columns, kind="mergesort").reset_index(drop=True),
        matched.sort_values(sort_columns, kind="mergesort").reset_index(drop=True),
        compared.sort_values(sort_columns, kind="mergesort").reset_index(drop=True),
        applied,
    )


def write_equivalent_resistance_outputs(
    output_dir: str | Path,
    *,
    base_nominal_targets: pd.DataFrame,
    child_realized_objects: pd.DataFrame,
    matched_base_corridors: pd.DataFrame,
    comparison: pd.DataFrame,
    manifest: Mapping[str, Any],
) -> None:
    output_path = Path(output_dir).expanduser().resolve()
    output_path.mkdir(parents=True, exist_ok=True)
    base_nominal_targets.to_csv(output_path / "base_nominal_targets.csv", index=False)
    child_realized_objects.to_csv(output_path / "child_realized_objects.csv", index=False)
    matched_base_corridors.to_csv(output_path / "matched_base_corridors.csv", index=False)
    comparison.to_csv(output_path / "equivalent_resistance_comparison.csv", index=False)
    (output_path / "equivalent_resistance_manifest.json").write_text(json.dumps(dict(manifest), indent=2), encoding="utf-8")


def run_equivalent_resistance_experiment(
    experiment_dir: str | Path,
    *,
    output_dir: str | Path | None = None,
    equivalent_resistance_config: BubbleEquivalentResistanceConfig | None = None,
    forcing_path: str | Path | None = None,
    forcing_config: ForcingConfig | None = None,
    kb_model_path: str | Path | None = None,
    kb_width_sample_field: str = "wid_adj_routing",
    kb_width_percentile: float | None = None,
    forcing_table_runner: Callable[..., tuple[pd.DataFrame, dict[str, object]]] = prepare_forcing_table,
    reference_kb_summary_runner: Callable[..., dict[str, Any]] = compute_reference_section_kb,
) -> EquivalentResistanceExperimentOutputs:
    experiment_path = Path(experiment_dir).expanduser().resolve()
    registry_path = experiment_path / "state_registry.csv"
    if not registry_path.exists():
        raise FileNotFoundError(f"State registry was not found: {registry_path}")
    registry = pd.read_csv(registry_path)
    mode = _infer_mode(experiment_path, registry)
    if mode not in SUPPORTED_MODES:
        raise ValueError(f"Unsupported experiment mode {mode!r}. Expected one of {SUPPORTED_MODES}.")

    config = equivalent_resistance_config or BubbleEquivalentResistanceConfig()
    if config.depth_width_exponent is not None and (experiment_path / "equivalent_transition_registry.csv").exists():
        stored_config, _ = load_manifest_equivalent_resistance_config(experiment_path)
        if stored_config.depth_width_exponent != config.depth_width_exponent:
            raise ValueError(
                "Selected-domain transition results use a different depth-width policy. "
                "Refresh those transitions before rebuilding aggregate resistance tables; "
                "otherwise historical values would overwrite the new calculations."
            )
    config, reference_kb_summary, forcing_metadata = _resolve_equivalent_reference_kb_summary(
        experiment_dir=experiment_path,
        equivalent_resistance_config=config,
        forcing_path=forcing_path,
        forcing_config=forcing_config,
        kb_model_path=kb_model_path,
        kb_width_sample_field=kb_width_sample_field,
        kb_width_percentile=kb_width_percentile,
        forcing_table_runner=forcing_table_runner,
        reference_kb_summary_runner=reference_kb_summary_runner,
    )

    base_rows = registry.loc[
        registry["state_role"].astype(str).eq("base")
        & registry["status"].fillna("complete").astype(str).eq("complete")
    ]
    if base_rows.empty:
        raise ValueError("Could not resolve a complete base state row from the experiment state registry.")
    base_row = base_rows.iloc[0]
    base_links, base_nodes, unit_summary, unit_link_membership = _load_base_state_context(base_row)

    base_nominal_targets = _build_nominal_targets(
        registry=registry,
        mode=mode,
        base_state_id=str(base_row["state_id"]),
        base_links=base_links,
        base_nodes=base_nodes,
        unit_summary=unit_summary,
        unit_link_membership=unit_link_membership,
        config=config,
    )
    child_realized_objects = _resolve_child_objects(
        registry=registry,
        base_links=base_links,
        nominal_targets=base_nominal_targets,
        config=config,
    )
    matched_base_corridors = _build_matched_base_corridors(
        base_links=base_links,
        base_nodes=base_nodes,
        child_realized_objects=child_realized_objects,
        config=config,
    )
    comparison = _build_comparison_frame(
        nominal_targets=base_nominal_targets,
        child_realized_objects=child_realized_objects,
        matched_base_corridors=matched_base_corridors,
    )
    child_realized_objects, matched_base_corridors, comparison, n_authoritative_transition_aggregates = (
        _apply_selected_hydraulic_domain_aggregates(
            experiment_path=experiment_path,
            nominal_targets=base_nominal_targets,
            child_realized_objects=child_realized_objects,
            matched_base_corridors=matched_base_corridors,
            comparison=comparison,
        )
    )

    output_path = (
        Path(output_dir).expanduser().resolve()
        if output_dir is not None
        else (experiment_path / "equivalent_resistance").resolve()
    )
    manifest = {
        "experiment_dir": str(experiment_path),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "mode": mode,
        "equivalent_resistance_config": asdict(config),
        "reference_kb_summary": reference_kb_summary,
        "forcing_path": str(Path(forcing_path).expanduser().resolve()) if forcing_path is not None else None,
        "forcing_config": asdict(forcing_config) if forcing_config is not None else None,
        "forcing_metadata": forcing_metadata,
        "n_nominal_targets": int(len(base_nominal_targets)),
        "n_child_realized_objects": int(len(child_realized_objects)),
        "n_matched_base_corridors": int(len(matched_base_corridors)),
        "n_comparisons": int(len(comparison)),
        "n_valid_matched_comparisons": int(comparison["comparison_valid_for_matched"].fillna(False).astype(bool).sum()) if not comparison.empty else 0,
        "n_valid_nominal_comparisons": int(comparison["comparison_valid_for_nominal"].fillna(False).astype(bool).sum()) if not comparison.empty else 0,
        "n_authoritative_transition_aggregates": int(n_authoritative_transition_aggregates),
    }
    write_equivalent_resistance_outputs(
        output_path,
        base_nominal_targets=base_nominal_targets,
        child_realized_objects=child_realized_objects,
        matched_base_corridors=matched_base_corridors,
        comparison=comparison,
        manifest=manifest,
    )
    return EquivalentResistanceExperimentOutputs(
        base_nominal_targets=base_nominal_targets,
        child_realized_objects=child_realized_objects,
        matched_base_corridors=matched_base_corridors,
        comparison=comparison,
        output_dir=output_path,
        manifest=manifest,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Recompute optional per-state hierarchy equivalent resistance and "
            "compute experiment-level comparisons between nominal base targets "
            "and realized collapsed variants."
        )
    )
    parser.add_argument("experiment_dir", help="Path to a network_variants experiment directory.")
    parser.add_argument("--output-dir", default=None, help="Optional output directory. Defaults to <experiment>/equivalent_resistance.")
    parser.add_argument(
        "--recompute-state-hierarchy",
        action="store_true",
        help=(
            "Recompute only the three equivalent-resistance hierarchy CSVs for "
            "every completed state before refreshing experiment comparisons. "
            "Masks, RivGraph, widths, slopes, and other hierarchy metrics are not rerun."
        ),
    )
    parser.add_argument(
        "--reuse-manifest-config",
        action="store_true",
        help=(
            "Use the exact equivalent-resistance configuration stored in the "
            "collapse experiment manifest. Recommended for regenerating an existing experiment."
        ),
    )
    parser.add_argument(
        "--equivalent-resistance-law",
        choices=("manning", "chezy"),
        default="manning",
        help="Resistance law used for edge K values.",
    )
    parser.add_argument(
        "--equivalent-geometry-method",
        choices=("wide", "rectangular"),
        default="wide",
        help="Hydraulic geometry method used for edge K values.",
    )
    parser.add_argument(
        "--equivalent-kb-mode",
        choices=("fixed", "based_reference_section"),
        default="fixed",
        help="How to resolve kb for the equivalent-resistance workflow.",
    )
    parser.add_argument("--equivalent-depth-width-exponent", type=float, default=None, help="Anchored local depth exponent; requires stored BASED reference width and depth.")
    parser.add_argument("--equivalent-kb-value", type=float, default=20.0, help="Fixed kb scalar when equivalent-kb-mode=fixed.")
    parser.add_argument(
        "--equivalent-kb-model-path",
        default=None,
        help="Optional local BASED model path when equivalent-kb-mode=based_reference_section.",
    )
    parser.add_argument(
        "--equivalent-kb-width-sample-field",
        default="wid_adj_routing",
        help="Reference-section width field used for based_reference_section kb. Summary width fields on the reference link are preferred over raw sample percentiles when available.",
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
    parser.add_argument("--equivalent-manning-n", type=float, default=0.035, help="Default Manning n used when links do not provide one.")
    parser.add_argument("--equivalent-chezy-cz", type=float, default=22.0, help="Default Chezy Cz used when links do not provide one.")
    parser.add_argument(
        "--equivalent-solver-backend",
        choices=(
            "dense_legacy", "sparse", "sparse_least_squares",
            "energy_sparse", "auto",
        ),
        default=None,
        help=(
            "Override the manifest solver backend. Omit to retain the manifest "
            "backend; new non-manifest runs default to dense_legacy."
        ),
    )
    parser.add_argument(
        "--equivalent-sparse-min-internal-nodes",
        type=int,
        default=None,
        help="Override the internal-node threshold used by auto.",
    )
    parser.add_argument(
        "--equivalent-continuation-eps",
        nargs="+",
        type=float,
        default=None,
        help=(
            "Strictly decreasing smoothing-epsilon schedule for an optional "
            "equivalent-resistance continuation solve. Omit to retain the direct solver."
        ),
    )
    parser.add_argument(
        "--equivalent-energy-max-iterations",
        type=int,
        default=None,
        help="Override energy-stage iterations for direct or continuation solves.",
    )
    parser.add_argument(
        "--equivalent-polish-max-function-evaluations",
        type=int,
        default=None,
        help="Override sparse-polish evaluations for direct or continuation solves.",
    )
    parser.add_argument("--equivalent-forcing-path", default=None, help="Forcing CSV/parquet used when equivalent-kb-mode=based_reference_section.")
    parser.add_argument("--equivalent-forcing-station-key", default=None, help="Station key used when equivalent-kb-mode=based_reference_section.")
    parser.add_argument("--equivalent-station-key-column", default="station_key", help="Station-key column in the forcing file.")
    parser.add_argument("--equivalent-forcing-start-time", default=None, help="Optional forcing truncation start time.")
    parser.add_argument("--equivalent-forcing-end-time", default=None, help="Optional forcing truncation end time.")
    parser.add_argument(
        "--equivalent-forcing-resample-minutes",
        type=float,
        default=None,
        help="Optional forcing resample interval in minutes. Fractional values are allowed if they resolve to whole seconds.",
    )
    parser.add_argument(
        "--equivalent-forcing-resample-seconds",
        type=int,
        default=None,
        help="Optional forcing resample interval in seconds. Prefer this when the admissible RAPID timestep is not a whole number of minutes.",
    )
    parser.add_argument("--equivalent-forcing-output-cache-dir", default=None, help="Optional forcing cache directory.")
    parser.add_argument("--equivalent-time-column", default="time", help="Timestamp column in the forcing file.")
    parser.add_argument("--equivalent-discharge-column", default="discharge", help="Discharge column in the forcing file.")
    parser.add_argument(
        "--equivalent-length-field",
        choices=LENGTH_FIELD_CANDIDATES,
        default=LENGTH_FIELD_CANDIDATES[0],
        help="Preferred link-length field for equivalent-resistance solves.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    manifest_config = None
    if args.reuse_manifest_config or args.recompute_state_hierarchy:
        manifest_config, _ = load_manifest_equivalent_resistance_config(
            args.experiment_dir
        )
    forcing_config = None
    if manifest_config is None and args.equivalent_kb_mode == "based_reference_section":
        forcing_config = ForcingConfig(
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
    equivalent_resistance_config = manifest_config or BubbleEquivalentResistanceConfig(
        resistance_law=args.equivalent_resistance_law,
        geometry_method=args.equivalent_geometry_method,
        kb_mode=args.equivalent_kb_mode,
        kb_value=args.equivalent_kb_value,
        length_field_candidates=resolve_length_field_candidates(args.equivalent_length_field),
        default_manning_n=args.equivalent_manning_n,
        default_chezy_cz=args.equivalent_chezy_cz,
        solver_backend=args.equivalent_solver_backend or "dense_legacy",
        sparse_min_internal_nodes=(
            args.equivalent_sparse_min_internal_nodes
            if args.equivalent_sparse_min_internal_nodes is not None
            else 64
        ),
    )
    if args.equivalent_depth_width_exponent is not None:
        equivalent_resistance_config = replace(equivalent_resistance_config, depth_width_exponent=args.equivalent_depth_width_exponent)
    if args.equivalent_solver_backend is not None:
        equivalent_resistance_config = replace(
            equivalent_resistance_config,
            solver_backend=args.equivalent_solver_backend,
        )
    if args.equivalent_sparse_min_internal_nodes is not None:
        equivalent_resistance_config = replace(
            equivalent_resistance_config,
            sparse_min_internal_nodes=args.equivalent_sparse_min_internal_nodes,
        )
    if args.equivalent_continuation_eps is not None:
        equivalent_resistance_config = replace(
            equivalent_resistance_config,
            solver_continuation_eps_schedule=tuple(args.equivalent_continuation_eps),
        )
    if args.equivalent_energy_max_iterations is not None:
        equivalent_resistance_config = replace(
            equivalent_resistance_config,
            solver_energy_max_iterations=args.equivalent_energy_max_iterations,
        )
    if args.equivalent_polish_max_function_evaluations is not None:
        equivalent_resistance_config = replace(
            equivalent_resistance_config,
            solver_polish_max_function_evaluations=args.equivalent_polish_max_function_evaluations,
        )
    if equivalent_resistance_config.solver_cache_dir is None:
        equivalent_resistance_config = replace(
            equivalent_resistance_config,
            solver_cache_dir=str(
                (
                    Path(args.experiment_dir).expanduser().resolve()
                    / "equivalent_resistance"
                    / "solve_cache"
                ).resolve()
            ),
        )
    state_results = None
    if args.recompute_state_hierarchy:
        state_results = recompute_state_hierarchy_equivalent_resistance(
            args.experiment_dir,
            equivalent_resistance_config=equivalent_resistance_config,
        )
    results = run_equivalent_resistance_experiment(
        args.experiment_dir,
        output_dir=args.output_dir,
        equivalent_resistance_config=equivalent_resistance_config,
        forcing_path=args.equivalent_forcing_path,
        forcing_config=forcing_config,
        kb_model_path=args.equivalent_kb_model_path,
        kb_width_sample_field=args.equivalent_kb_width_sample_field,
        kb_width_percentile=args.equivalent_kb_width_percentile,
    )
    if state_results is not None:
        print(f"Recomputed state hierarchy resistance for {state_results.n_states_recomputed} states")
        print(f"State hierarchy rows: {len(state_results.audit)}")
        print(f"State hierarchy audit: {state_results.audit_csv_path}")
        print(f"State hierarchy summary: {state_results.summary_json_path}")
    print(f"Wrote equivalent-resistance outputs to {results.output_dir}")
    print(f"Nominal targets: {len(results.base_nominal_targets)}")
    print(f"Child realized objects: {len(results.child_realized_objects)}")
    print(f"Matched base corridors: {len(results.matched_base_corridors)}")
    print(f"Comparisons: {len(results.comparison)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
