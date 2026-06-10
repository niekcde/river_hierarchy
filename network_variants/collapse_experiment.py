from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import traceback
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import geopandas as gpd
import numpy as np
import pandas as pd

repo_root = Path(__file__).resolve().parents[1]
rapid_src = repo_root / "RAPID" / "src"
if str(rapid_src) not in sys.path:
    sys.path.insert(0, str(rapid_src))

if __package__ is None or __package__ == "":
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

from hierarchy_level_definition.run_unit_workflow import (
    UnitWorkflowOutputs,
    run_unit_workflow,
    write_unit_workflow_outputs,
)
from hierarchy_level_definition.metrics import BubbleEquivalentResistanceConfig, compare_states
from network_variants.hydraulic_context import (
    StateHydraulicContextOutputs,
    generate_state_hydraulic_context,
)
from network_variants.sword_matching import match_variant_nodes_to_sword
from network_variants.variant_generation import (
    COLLAPSE_CORRIDOR_MODES,
    NetworkVariantOutputs,
    VariantDirectionValidationError,
    _apply_collapsed_selection_metadata,
    _collapsed_selection_metadata,
    compute_width_families,
    generate_network_variant,
)
from rapid_tools.forcing import ForcingConfig, prepare_forcing_table
from rapid_tools.reference_section import compute_reference_section_kb


EXPERIMENT_MODES = (
    "independent-units",
    "compound-bubbles",
    "sequential-units",
    "sequential-groups",
)
COLLAPSE_CORRIDOR_MODE_CHOICES = COLLAPSE_CORRIDOR_MODES + ("both",)


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
    workflow: UnitWorkflowOutputs
    variant_output_dir: Path | None = None
    hydraulic_context_output_dir: Path | None = None
    equivalent_transition_output_dir: Path | None = None
    equivalent_transition_comparison_path: Path | None = None
    equivalent_transition_summary_path: Path | None = None
    equivalent_transition_summary: dict[str, Any] | None = None
    selection: ExperimentSelection | None = None


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
    equivalent_kb_width_percentile: float,
    equivalent_forcing_table_runner: Callable[..., tuple[pd.DataFrame, dict[str, object]]],
    equivalent_reference_kb_summary_runner: Callable[..., dict[str, Any]],
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
    reference_kb_summary = equivalent_reference_kb_summary_runner(
        experiment_dir,
        forcing=forcing_table,
        width_sample_field=equivalent_kb_width_sample_field,
        width_percentile=equivalent_kb_width_percentile,
        model_path=equivalent_kb_model_path,
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
    candidate_keys = [
        ("upstream_boundary_node", "downstream_boundary_node"),
        ("bubble_root_unit_id",),
        ("compound_bubble_id",),
    ]

    matched_frames: list[pd.DataFrame] = []
    parent_remaining = parent.copy()
    child_remaining = child.copy()

    for key_columns in candidate_keys:
        if parent_remaining.empty or child_remaining.empty:
            break
        if not set(key_columns).issubset(parent_remaining.columns) or not set(key_columns).issubset(child_remaining.columns):
            continue

        parent_unique = parent_remaining.dropna(subset=list(key_columns)).copy()
        child_unique = child_remaining.dropna(subset=list(key_columns)).copy()
        if parent_unique.empty or child_unique.empty:
            continue

        parent_unique = parent_unique.loc[~parent_unique.duplicated(list(key_columns), keep=False)].copy()
        child_unique = child_unique.loc[~child_unique.duplicated(list(key_columns), keep=False)].copy()
        if parent_unique.empty or child_unique.empty:
            continue

        matched = parent_unique.merge(
            child_unique,
            on=list(key_columns),
            how="inner",
            suffixes=("_before", "_after"),
        )
        if matched.empty:
            continue

        matched["match_method"] = "__".join(key_columns)
        matched_frames.append(matched)
        parent_remaining = parent_remaining.loc[~parent_remaining["_parent_row_id"].isin(matched["_parent_row_id"])].copy()
        child_remaining = child_remaining.loc[~child_remaining["_child_row_id"].isin(matched["_child_row_id"])].copy()

    if not matched_frames:
        return pd.DataFrame()
    return pd.concat(matched_frames, ignore_index=True)


def _empty_equivalent_transition_frame() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
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
    )


def _equivalent_transition_frame(records: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    if not records:
        return _empty_equivalent_transition_frame()
    return pd.DataFrame.from_records(records)


def _write_equivalent_transition_outputs(
    *,
    transition_id: str,
    parent_context: ExperimentStateContext,
    child_context: ExperimentStateContext,
) -> pd.DataFrame:
    parent_frame = parent_context.workflow.bubble_equivalent_metrics.copy()
    child_frame = child_context.workflow.bubble_equivalent_metrics.copy()
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

    comparison.to_csv(comparison_path, index=False)
    summary = {
        "transition_id": transition_id,
        "parent_state_id": parent_context.state_id,
        "child_state_id": child_context.state_id,
        "comparison_csv": str(comparison_path.resolve()),
        "n_equivalent_bubbles_before": int(len(parent_frame)),
        "n_equivalent_bubbles_after": int(len(child_frame)),
        "n_equivalent_bubbles_matched": int(len(comparison)),
        "n_equivalent_bubbles_unmatched_before": int(max(len(parent_frame) - len(comparison), 0)),
        "n_equivalent_bubbles_unmatched_after": int(max(len(child_frame) - len(comparison), 0)),
        "n_equivalent_solver_success_before": _count_equivalent_solver_success(parent_frame),
        "n_equivalent_solver_success_after": _count_equivalent_solver_success(child_frame),
        "mean_lambda_k": float(comparison["Lambda_K"].mean()) if not comparison.empty else float("nan"),
        "median_lambda_k": float(comparison["Lambda_K"].median()) if not comparison.empty else float("nan"),
        "mean_percent_change_k": float(comparison["percent_change_K"].mean()) if not comparison.empty else float("nan"),
        "median_percent_change_k": float(comparison["percent_change_K"].median()) if not comparison.empty else float("nan"),
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
        "directed_nodes_path": str(context.reviewed_nodes_path.resolve()),
        "n_units_detected": int(len(context.workflow.unit_metrics)),
        "n_paths_detected": int(len(context.workflow.path_metrics)),
        "n_equivalent_bubbles": int(len(context.workflow.bubble_equivalent_metrics)),
        "n_equivalent_bubbles_solver_success": _count_equivalent_solver_success(context.workflow.bubble_equivalent_metrics),
        "n_equivalent_bubbles_matched_to_parent": equivalent_transition_summary.get("n_equivalent_bubbles_matched", pd.NA),
        "n_selected_groups": int(len(context.workflow.selected_groups)),
        "optimal_n_groups": _optimal_n_groups(context.workflow.group_count_summary),
        "status": "complete",
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
        "n_units_before": int(len(parent_context.workflow.unit_metrics)),
        "n_units_after": int(len(child_context.workflow.unit_metrics)),
        "n_equivalent_bubbles_before": int(len(parent_context.workflow.bubble_equivalent_metrics)),
        "n_equivalent_bubbles_after": int(len(child_context.workflow.bubble_equivalent_metrics)),
        "n_equivalent_bubbles_matched": equivalent_transition_summary.get("n_equivalent_bubbles_matched", 0),
        "n_equivalent_bubbles_unmatched_before": equivalent_transition_summary.get("n_equivalent_bubbles_unmatched_before", 0),
        "n_equivalent_bubbles_unmatched_after": equivalent_transition_summary.get("n_equivalent_bubbles_unmatched_after", 0),
        "mean_lambda_k": equivalent_transition_summary.get("mean_lambda_k", float("nan")),
        "median_lambda_k": equivalent_transition_summary.get("median_lambda_k", float("nan")),
        "mean_percent_change_k": equivalent_transition_summary.get("mean_percent_change_k", float("nan")),
        "median_percent_change_k": equivalent_transition_summary.get("median_percent_change_k", float("nan")),
        "parent_hierarchy_output_dir": str(parent_context.hierarchy_output_dir.resolve()),
        "child_hierarchy_output_dir": str(child_context.hierarchy_output_dir.resolve()),
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
        "status": "complete",
        "error_stage": "",
        "error": "",
    }


def _write_state_failure_artifact(state_dir: Path, payload: Mapping[str, Any]) -> Path:
    state_dir.mkdir(parents=True, exist_ok=True)
    path = state_dir / "state_failure.json"
    path.write_text(json.dumps(dict(payload), indent=2), encoding="utf-8")
    return path


def _reset_state_dir(state_dir: Path) -> None:
    if state_dir.exists():
        shutil.rmtree(state_dir)
    state_dir.mkdir(parents=True, exist_ok=True)


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
        reviewed_links_path=Path(reviewed_links_path),
        reviewed_nodes_path=Path(reviewed_nodes_path),
        workflow=workflow,
        selection=selection,
    )


def _run_state_hydraulic_context(
    *,
    context: ExperimentStateContext,
    hydraulic_context_runner: Callable[..., StateHydraulicContextOutputs],
    hydraulic_context_kwargs: Mapping[str, Any] | None = None,
) -> ExperimentStateContext:
    outputs = hydraulic_context_runner(
        directed_links_path=context.reviewed_links_path,
        directed_nodes_path=context.reviewed_nodes_path,
        hierarchy_output_dir=context.hierarchy_output_dir,
        output_dir=context.state_dir / "hydraulic_context",
        state_id=context.state_id,
        **dict(hydraulic_context_kwargs or {}),
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


def _materialize_base_state_variant(
    *,
    context: ExperimentStateContext,
    example_id: str,
    exit_sides: str,
    collapse_corridor_mode: str,
    preferred_width_field: str,
    transect_scale: float,
    min_transect_pixels: float,
    sword_node_source_path: str | Path | None,
    sword_wse_field: str | None,
    sword_match_tolerance: float | None,
    sword_example_station_source_path: str | Path | None,
    sword_station_match_source_path: str | Path | None,
    sword_reach_buffer_steps: int,
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

    reviewed_links = gpd.read_file(context.reviewed_links_path)
    reviewed_nodes = gpd.read_file(context.reviewed_nodes_path)
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

    enriched_links, link_width_samples = compute_width_families(
        directed_links,
        collapsed_mask_path=collapsed_mask_path,
        wet_reference_mask_path=collapsed_mask_path,
        transect_scale=transect_scale,
        min_transect_pixels=min_transect_pixels,
    )
    enriched_links = _apply_collapsed_selection_metadata(enriched_links, unit_ids=[], group_label=None)
    enriched_links = _normalize_numeric_id_columns(enriched_links)
    enriched_links["example_id"] = example_id
    enriched_links["variant_id"] = base_variant_id
    link_width_samples = _apply_collapsed_selection_metadata(link_width_samples, unit_ids=[], group_label=None)
    link_width_samples = _normalize_numeric_id_columns(link_width_samples)
    link_width_samples["example_id"] = example_id
    link_width_samples["variant_id"] = base_variant_id

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
            "collapse_components": str((summary_dir / "collapse_components.csv").resolve()),
            "unit_link_membership": str((summary_dir / "unit_link_membership.csv").resolve()),
            "edit_geometries": str((mask_dir / "collapse_edit_geometries.gpkg").resolve()),
            "link_width_families": str((width_dir / "link_width_families.csv").resolve()),
            "link_width_samples": str((width_dir / "link_width_samples.csv").resolve()),
            "links_with_width_families": str(links_with_widths_path.resolve()),
            "node_match": str((matching_dir / "node_match.csv").resolve()),
            "node_sword_match": str((matching_dir / "node_sword_match.csv").resolve()),
            "link_match": str((matching_dir / "link_match.csv").resolve()),
            "link_lineage": str((matching_dir / "link_lineage.csv").resolve()),
            "directed_links": str(directed_links_path.resolve()),
            "directed_nodes": str(directed_nodes_path.resolve()),
            "direction_validation_report": str((directed_dir / "direction_validation_report.json").resolve()),
        },
    }
    with (summary_dir / "variant_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)

    return BaseStateVariantOutputs(
        output_dir=variant_dir,
        collapsed_mask_path=collapsed_mask_path,
        directed_links_path=directed_links_path,
        directed_nodes_path=directed_nodes_path,
    )


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
) -> ExperimentStateContext:
    _reset_state_dir(state_dir)
    variant_dir = state_dir / "variant"
    variant_call_kwargs = dict(variant_kwargs)
    variant_call_kwargs.update(
        {
            "cleaned_mask_path": parent_context.cleaned_mask_path,
            "reviewed_links_path": parent_context.reviewed_links_path,
            "reviewed_nodes_path": parent_context.reviewed_nodes_path,
            "exit_sides": exit_sides,
            "variant_id": state_id,
            "output_dir": variant_dir,
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

    variant_outputs = variant_runner(**variant_call_kwargs)

    child_context = _run_state_hierarchy(
        state_id=state_id,
        parent_state_id=parent_context.state_id,
        depth=parent_context.depth + 1,
        state_dir=state_dir,
        cleaned_mask_path=variant_outputs.collapsed_mask_path,
        reviewed_links_path=variant_outputs.directed_links_path,
        reviewed_nodes_path=variant_outputs.directed_nodes_path,
        selection=selection,
        unit_workflow_runner=unit_workflow_runner,
        unit_workflow_writer=unit_workflow_writer,
        unit_workflow_kwargs=unit_workflow_kwargs,
    )
    child_context.variant_output_dir = variant_outputs.output_dir
    child_context = _run_state_hydraulic_context(
        context=child_context,
        hydraulic_context_runner=hydraulic_context_runner,
        hydraulic_context_kwargs=hydraulic_context_kwargs,
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
    preferred_width_field: str = "wid_adj",
    footprint_buffer_scale: float = 0.5,
    all_touched: bool = True,
    allow_noop: bool = False,
    single_thread: bool = False,
    export_sword: bool = True,
    transect_scale: float = 1.5,
    min_transect_pixels: float = 5.0,
    match_tolerance: float | None = None,
    sword_node_source_path: str | Path | None = None,
    sword_wse_field: str | None = None,
    sword_match_tolerance: float | None = None,
    sword_example_station_source_path: str | Path | None = None,
    sword_station_match_source_path: str | Path | None = None,
    sword_reach_buffer_steps: int = 2,
    verbose_rivgraph: bool = False,
    state_hydraulic_context_kwargs: Mapping[str, Any] | None = None,
    equivalent_resistance_config: BubbleEquivalentResistanceConfig | None = None,
    equivalent_forcing_path: str | Path | None = None,
    equivalent_forcing_config: ForcingConfig | None = None,
    equivalent_kb_model_path: str | Path | None = None,
    equivalent_kb_width_sample_field: str = "width_wet",
    equivalent_kb_width_percentile: float = 90.0,
    equivalent_forcing_table_runner: Callable[..., tuple[pd.DataFrame, dict[str, object]]] = prepare_forcing_table,
    equivalent_reference_kb_summary_runner: Callable[..., dict[str, Any]] = compute_reference_section_kb,
) -> CollapseExperimentOutputs:
    if mode not in EXPERIMENT_MODES:
        raise ValueError(f"Unsupported mode {mode!r}. Expected one of {EXPERIMENT_MODES}.")
    if collapse_corridor_mode not in COLLAPSE_CORRIDOR_MODES:
        raise ValueError(
            f"Unsupported collapse_corridor_mode {collapse_corridor_mode!r}. "
            f"Expected one of {COLLAPSE_CORRIDOR_MODES}."
        )

    root_example_id = _infer_example_id(cleaned_mask_path)
    experiment_id = experiment_id or _default_experiment_id(cleaned_mask_path, mode)
    output_path = Path(output_dir) if output_dir is not None else _default_output_dir(cleaned_mask_path, experiment_id)
    states_dir = output_path / "states"
    states_dir.mkdir(parents=True, exist_ok=True)

    active_equivalent_resistance_config = equivalent_resistance_config
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
        "preferred_width_field": preferred_width_field,
        "footprint_buffer_scale": footprint_buffer_scale,
        "all_touched": all_touched,
        "allow_noop": allow_noop,
        "single_thread": single_thread,
        "export_sword": export_sword,
        "transect_scale": transect_scale,
        "min_transect_pixels": min_transect_pixels,
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
    )
    base_context.variant_output_dir = base_variant_outputs.output_dir
    base_context.cleaned_mask_path = base_variant_outputs.collapsed_mask_path
    base_context.reviewed_links_path = base_variant_outputs.directed_links_path
    base_context.reviewed_nodes_path = base_variant_outputs.directed_nodes_path
    base_context = _run_state_hydraulic_context(
        context=base_context,
        hydraulic_context_runner=state_hydraulic_context_runner,
        hydraulic_context_kwargs=state_hydraulic_context_kwargs,
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
    stop_reason = "base_state_only"
    terminal_state_id = base_context.state_id

    def write_metadata() -> None:
        n_complete_states = sum(1 for record in state_records if record.get("status") == "complete")
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
            "root_paths": {
                "cleaned_mask": str(Path(cleaned_mask_path).resolve()),
                "reviewed_links": str(Path(reviewed_links_path).resolve()),
                "reviewed_nodes": str(Path(reviewed_nodes_path).resolve()),
            },
            "max_steps": max_steps,
            "n_states": len(state_records),
            "n_complete_states": n_complete_states,
            "n_failed_states": n_failed_states,
            "n_compound_states": n_compound_states,
            "n_transitions": len(transition_records),
            "n_equivalent_transition_rows": n_equivalent_transition_rows,
            "n_equivalent_transitions_with_matches": n_equivalent_transitions_with_matches,
            "terminal_state_id": terminal_state_id,
            "stop_reason": stop_reason,
            "options": {
                "hierarchy": {
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
                    "equivalent_kb_width_percentile": float(equivalent_kb_width_percentile),
                    "debug_hierarchy": bool(debug_hierarchy),
                },
                "variant": {
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
            )
        )
        unit_workflow_kwargs["equivalent_resistance_config"] = active_equivalent_resistance_config
        base_context.workflow = unit_workflow_runner(
            base_context.reviewed_links_path,
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
        write_metadata()

    if mode in {"independent-units", "compound-bubbles"}:
        if mode == "compound-bubbles":
            selections = _select_compound_bubbles_from_workflow(base_context.workflow)
        else:
            selections = _select_all_units_from_ranking(base_context.workflow.collapse_ranking, base_context.workflow.unit_metrics)
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
            equivalent_transition = _write_equivalent_transition_outputs(
                transition_id=transition_id,
                parent_context=base_context,
                child_context=child_context,
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
        default="width_wet",
        help="Reference-section width sample field used when equivalent-kb-mode=based_reference_section.",
    )
    parser.add_argument(
        "--equivalent-kb-width-percentile",
        type=float,
        default=90.0,
        help="Reference-section width percentile used when equivalent-kb-mode=based_reference_section.",
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
    parser.add_argument(
        "--equivalent-forcing-resample-minutes",
        type=int,
        default=None,
        help="Optional normalized forcing interval in minutes for equivalent-resistance kb preparation.",
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
    parser.add_argument("--preferred-width-field", default="wid_adj", help="Preferred reviewed-link width field for collapse-footprint buffering.")
    parser.add_argument("--footprint-buffer-scale", type=float, default=0.5, help="Multiplier applied to footprint width / 2 buffering.")
    parser.add_argument("--disable-all-touched", action="store_true", help="Disable all_touched rasterization when creating collapse footprints.")
    parser.add_argument("--allow-noop", action="store_true", help="Allow variants that produce no added pixels.")
    parser.add_argument("--single-thread", action="store_true", help="Pass single_thread=True to the RivGraph river class.")
    parser.add_argument("--disable-sword-export", action="store_true", help="Skip SWORD-style RivGraph exports.")
    parser.add_argument("--transect-scale", type=float, default=1.5, help="Multiplier for transect half-length relative to local total width.")
    parser.add_argument("--min-transect-pixels", type=float, default=5.0, help="Minimum transect half-length in raster pixels.")
    parser.add_argument("--match-tolerance", type=float, default=None, help="Optional spatial tolerance for parent-child graph matching.")
    parser.add_argument("--sword-node-source", default=None, help="Optional SWORD node source file or parquet directory used for node matching.")
    parser.add_argument("--sword-wse-field", default=None, help="Optional WSE field name in the SWORD node source. Defaults to automatic detection.")
    parser.add_argument("--sword-match-tolerance", type=float, default=None, help="Optional maximum SWORD node-match distance in CRS units/meters after reprojection.")
    parser.add_argument("--sword-example-stations-source", default=None, help="Optional example-station source used to derive example-specific SWORD reach corridors. Defaults to the repo hierarchy_examples_filtered_subdaily_manual_updates_final.gpkg when present.")
    parser.add_argument("--sword-station-match-source", default=None, help="Optional station-to-SWORD match source used to derive example-specific SWORD reach corridors. Defaults to the repo selected_event_stations_same_main_path.gpkg when present.")
    parser.add_argument("--sword-reach-buffer-steps", type=int, default=2, help="Number of reach steps to extend upstream and downstream beyond the example corridor endpoints when constraining SWORD node candidates.")
    parser.add_argument("--verbose-rivgraph", action="store_true", help="Print RivGraph progress to stdout.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    equivalent_resistance_config = None
    equivalent_forcing_config = None
    if args.compute_equivalent_resistance:
        equivalent_resistance_config = BubbleEquivalentResistanceConfig(
            resistance_law=args.equivalent_resistance_law,
            geometry_method=args.equivalent_geometry_method,
            kb_mode=args.equivalent_kb_mode,
            kb_value=args.equivalent_kb_value,
            default_manning_n=args.equivalent_manning_n,
            default_chezy_cz=args.equivalent_chezy_cz,
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
        "match_tolerance": args.match_tolerance,
        "sword_node_source_path": args.sword_node_source,
        "sword_wse_field": args.sword_wse_field,
        "sword_match_tolerance": args.sword_match_tolerance,
        "sword_example_station_source_path": args.sword_example_stations_source,
        "sword_station_match_source_path": args.sword_station_match_source,
        "sword_reach_buffer_steps": args.sword_reach_buffer_steps,
        "verbose_rivgraph": args.verbose_rivgraph,
        "equivalent_resistance_config": equivalent_resistance_config,
        "equivalent_forcing_path": args.equivalent_forcing_path,
        "equivalent_forcing_config": equivalent_forcing_config,
        "equivalent_kb_model_path": args.equivalent_kb_model_path,
        "equivalent_kb_width_sample_field": args.equivalent_kb_width_sample_field,
        "equivalent_kb_width_percentile": args.equivalent_kb_width_percentile,
    }

    requested_modes = COLLAPSE_CORRIDOR_MODES if args.collapse_corridor_mode == "both" else (args.collapse_corridor_mode,)
    for collapse_corridor_mode in requested_modes:
        experiment_id = args.experiment_id
        output_dir = args.output_dir
        if args.collapse_corridor_mode == "both":
            experiment_id = _suffix_experiment_id_for_collapse_mode(
                experiment_id or _default_experiment_id(args.cleaned_mask, args.mode),
                collapse_corridor_mode,
            )
            output_dir = (
                _suffix_output_dir_for_collapse_mode(args.output_dir, collapse_corridor_mode)
                if args.output_dir is not None
                else None
            )
        results = run_collapse_experiment(
            args.mode,
            experiment_id=experiment_id,
            output_dir=output_dir,
            collapse_corridor_mode=collapse_corridor_mode,
            **run_kwargs,
        )

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
