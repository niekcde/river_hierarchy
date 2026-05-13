from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import geopandas as gpd
import pandas as pd

if __package__ is None or __package__ == "":
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

from hierarchy_level_definition.run_unit_workflow import (
    UnitWorkflowOutputs,
    run_unit_workflow,
    write_unit_workflow_outputs,
)
from network_variants.sword_matching import match_variant_nodes_to_sword
from network_variants.variant_generation import (
    NetworkVariantOutputs,
    VariantDirectionValidationError,
    _apply_collapsed_selection_metadata,
    _collapsed_selection_metadata,
    compute_width_families,
    generate_network_variant,
)


EXPERIMENT_MODES = (
    "independent-units",
    "sequential-units",
    "sequential-groups",
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
    selection: ExperimentSelection | None = None


@dataclass
class CollapseExperimentOutputs:
    state_registry: pd.DataFrame
    transition_registry: pd.DataFrame
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
        "state_dir": str(context.state_dir.resolve()),
        "variant_output_dir": str(context.variant_output_dir.resolve()) if context.variant_output_dir is not None else "",
        "hierarchy_output_dir": str(context.hierarchy_output_dir.resolve()),
        "cleaned_mask_path": str(context.cleaned_mask_path.resolve()),
        "directed_links_path": str(context.reviewed_links_path.resolve()),
        "directed_nodes_path": str(context.reviewed_nodes_path.resolve()),
        "n_units_detected": int(len(context.workflow.unit_metrics)),
        "n_paths_detected": int(len(context.workflow.path_metrics)),
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
        "n_units_before": int(len(parent_context.workflow.unit_metrics)),
        "n_units_after": int(len(child_context.workflow.unit_metrics)),
        "parent_hierarchy_output_dir": str(parent_context.hierarchy_output_dir.resolve()),
        "child_hierarchy_output_dir": str(child_context.hierarchy_output_dir.resolve()),
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
        "n_units_detected": pd.NA,
        "n_paths_detected": pd.NA,
        "n_selected_groups": pd.NA,
        "optimal_n_groups": pd.NA,
        "status": "failed",
        "error_stage": error_stage,
        "error": error,
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
        "n_units_before": int(len(parent_context.workflow.unit_metrics)),
        "n_units_after": pd.NA,
        "parent_hierarchy_output_dir": str(parent_context.hierarchy_output_dir.resolve()),
        "child_hierarchy_output_dir": "",
        "child_variant_output_dir": (
            str(variant_outputs.output_dir.resolve()) if variant_outputs is not None else ""
        ),
        "status": "failed",
        "error_stage": error_stage,
        "error": error,
    }


def _write_experiment_outputs(
    output_dir: Path,
    *,
    state_records: Sequence[Mapping[str, Any]],
    transition_records: Sequence[Mapping[str, Any]],
    manifest: Mapping[str, Any],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame.from_records(state_records).to_csv(output_dir / "state_registry.csv", index=False)
    pd.DataFrame.from_records(transition_records).to_csv(output_dir / "transition_registry.csv", index=False)
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
) -> ExperimentStateContext:
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
    max_steps: int | None = None,
    unit_workflow_runner: Callable[..., UnitWorkflowOutputs] = run_unit_workflow,
    unit_workflow_writer: Callable[..., None] = write_unit_workflow_outputs,
    variant_runner: Callable[..., NetworkVariantOutputs] = generate_network_variant,
    base_state_variant_materializer: Callable[..., BaseStateVariantOutputs] = _materialize_base_state_variant,
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
) -> CollapseExperimentOutputs:
    if mode not in EXPERIMENT_MODES:
        raise ValueError(f"Unsupported mode {mode!r}. Expected one of {EXPERIMENT_MODES}.")

    root_example_id = _infer_example_id(cleaned_mask_path)
    experiment_id = experiment_id or _default_experiment_id(cleaned_mask_path, mode)
    output_path = Path(output_dir) if output_dir is not None else _default_output_dir(cleaned_mask_path, experiment_id)
    states_dir = output_path / "states"
    states_dir.mkdir(parents=True, exist_ok=True)

    unit_workflow_kwargs = {
        "max_path_cutoff": max_path_cutoff,
        "max_paths": max_paths,
        "pixel_width_fields": pixel_width_fields,
        "pixel_width_percentiles": pixel_width_percentiles,
        "use_pixel_widths_for_extremes": use_pixel_widths_for_extremes,
        "debug": debug_hierarchy,
    }
    variant_kwargs = {
        "example_id": root_example_id,
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

    state_records: list[dict[str, Any]] = [_state_record(base_context, mode)]
    transition_records: list[dict[str, Any]] = []
    stop_reason = "base_state_only"
    terminal_state_id = base_context.state_id

    def write_metadata() -> None:
        n_complete_states = sum(1 for record in state_records if record.get("status") == "complete")
        n_failed_states = sum(1 for record in state_records if record.get("status") == "failed")
        manifest = {
            "experiment_id": experiment_id,
            "mode": mode,
            "example_id": _infer_example_id(cleaned_mask_path),
            "root_example_id": root_example_id,
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
            "n_transitions": len(transition_records),
            "terminal_state_id": terminal_state_id,
            "stop_reason": stop_reason,
            "options": {
                "hierarchy": {
                    "max_path_cutoff": max_path_cutoff,
                    "max_paths": max_paths,
                    "pixel_width_fields": list(pixel_width_fields) if pixel_width_fields is not None else None,
                    "pixel_width_percentiles": list(pixel_width_percentiles) if pixel_width_percentiles is not None else None,
                    "use_pixel_widths_for_extremes": bool(use_pixel_widths_for_extremes),
                    "debug_hierarchy": bool(debug_hierarchy),
                },
                "variant": {
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
            manifest=manifest,
        )

    write_metadata()

    if mode == "independent-units":
        selections = _select_all_units_from_ranking(base_context.workflow.collapse_ranking, base_context.workflow.unit_metrics)
        if not selections:
            stop_reason = "no_units_in_base_state"
            write_metadata()
            return CollapseExperimentOutputs(
                state_registry=pd.DataFrame.from_records(state_records),
                transition_registry=pd.DataFrame.from_records(transition_records),
                output_dir=output_path,
                manifest=json.loads((output_path / "experiment_manifest.json").read_text(encoding="utf-8")),
            )

        for index, selection in enumerate(selections, start=1):
            state_id = f"S{index:03d}_{selection.state_suffix}"
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
                )
            except VariantDirectionValidationError as exc:
                state_records.append(
                    _failed_state_record(
                        state_id=state_id,
                        parent_state_id=base_context.state_id,
                        depth=base_context.depth + 1,
                        mode=mode,
                        state_dir=state_dir,
                        selection=selection,
                        error_stage="direction_validation",
                        error=str(exc),
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
                    error=str(exc),
                    variant_outputs=exc.outputs,
                )
                transition["mode"] = mode
                transition_records.append(transition)
                terminal_state_id = state_id
                write_metadata()
                continue
            except Exception as exc:
                state_records.append(
                    _failed_state_record(
                        state_id=state_id,
                        parent_state_id=base_context.state_id,
                        depth=base_context.depth + 1,
                        mode=mode,
                        state_dir=state_dir,
                        selection=selection,
                        error_stage="child_state_build",
                        error=str(exc),
                    )
                )
                transition = _failed_transition_record(
                    transition_id=f"T{index:03d}",
                    step_index=index,
                    parent_context=base_context,
                    selection=selection,
                    child_state_id=state_id,
                    error_stage="child_state_build",
                    error=str(exc),
                )
                transition["mode"] = mode
                transition_records.append(transition)
                terminal_state_id = state_id
                write_metadata()
                continue

            state_records.append(_state_record(child_context, mode))
            transition = _transition_record(
                transition_id=f"T{index:03d}",
                step_index=index,
                parent_context=base_context,
                child_context=child_context,
            )
            transition["mode"] = mode
            transition_records.append(transition)
            terminal_state_id = child_context.state_id
            write_metadata()

        stop_reason = "all_base_units_processed"
        write_metadata()
        return CollapseExperimentOutputs(
            state_registry=pd.DataFrame.from_records(state_records),
            transition_registry=pd.DataFrame.from_records(transition_records),
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
            )
        except VariantDirectionValidationError as exc:
            state_records.append(
                _failed_state_record(
                    state_id=state_id,
                    parent_state_id=current_context.state_id,
                    depth=current_context.depth + 1,
                    mode=mode,
                    state_dir=state_dir,
                    selection=selection,
                    error_stage="direction_validation",
                    error=str(exc),
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
                error=str(exc),
                variant_outputs=exc.outputs,
            )
            transition["mode"] = mode
            transition_records.append(transition)
            terminal_state_id = state_id
            stop_reason = "state_failed"
            write_metadata()
            break
        except Exception as exc:
            state_records.append(
                _failed_state_record(
                    state_id=state_id,
                    parent_state_id=current_context.state_id,
                    depth=current_context.depth + 1,
                    mode=mode,
                    state_dir=state_dir,
                    selection=selection,
                    error_stage="child_state_build",
                    error=str(exc),
                )
            )
            transition = _failed_transition_record(
                transition_id=f"T{step_index:03d}",
                step_index=step_index,
                parent_context=current_context,
                selection=selection,
                child_state_id=state_id,
                error_stage="child_state_build",
                error=str(exc),
            )
            transition["mode"] = mode
            transition_records.append(transition)
            terminal_state_id = state_id
            stop_reason = "state_failed"
            write_metadata()
            break

        state_records.append(_state_record(child_context, mode))
        transition = _transition_record(
            transition_id=f"T{step_index:03d}",
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
        output_dir=output_path,
        manifest=json.loads((output_path / "experiment_manifest.json").read_text(encoding="utf-8")),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run one of the collapse experiment workflows over a graph-like river network: "
            "independent base-unit variants, adaptive sequential unit collapse, or adaptive sequential group collapse."
        )
    )
    parser.add_argument("mode", choices=EXPERIMENT_MODES, help="Experiment mode to run.")
    parser.add_argument("--cleaned-mask", required=True, help="Base cleaned mask GeoTIFF.")
    parser.add_argument("--reviewed-links", required=True, help="Base reviewed/directed links GeoPackage.")
    parser.add_argument("--reviewed-nodes", required=True, help="Base reviewed/directed nodes GeoPackage.")
    parser.add_argument("--exit-sides", required=True, help="River exit sides for RivGraph, e.g. NS or EW.")
    parser.add_argument("--experiment-id", default=None, help="Optional experiment identifier.")
    parser.add_argument("--output-dir", default=None, help="Optional experiment output directory.")
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

    results = run_collapse_experiment(
        args.mode,
        cleaned_mask_path=args.cleaned_mask,
        reviewed_links_path=args.reviewed_links,
        reviewed_nodes_path=args.reviewed_nodes,
        exit_sides=args.exit_sides,
        experiment_id=args.experiment_id,
        output_dir=args.output_dir,
        max_steps=args.max_steps,
        max_path_cutoff=args.max_path_cutoff,
        max_paths=args.max_paths,
        pixel_width_fields=args.pixel_width_fields,
        pixel_width_percentiles=args.pixel_width_percentiles,
        use_pixel_widths_for_extremes=not args.disable_pixel_width_extremes,
        debug_hierarchy=args.debug_hierarchy,
        preferred_width_field=args.preferred_width_field,
        footprint_buffer_scale=args.footprint_buffer_scale,
        all_touched=not args.disable_all_touched,
        allow_noop=args.allow_noop,
        single_thread=args.single_thread,
        export_sword=not args.disable_sword_export,
        transect_scale=args.transect_scale,
        min_transect_pixels=args.min_transect_pixels,
        match_tolerance=args.match_tolerance,
        sword_node_source_path=args.sword_node_source,
        sword_wse_field=args.sword_wse_field,
        sword_match_tolerance=args.sword_match_tolerance,
        sword_example_station_source_path=args.sword_example_stations_source,
        sword_station_match_source_path=args.sword_station_match_source,
        sword_reach_buffer_steps=args.sword_reach_buffer_steps,
        verbose_rivgraph=args.verbose_rivgraph,
    )

    print(f"Wrote experiment outputs to {results.output_dir}")
    print(f"States: {len(results.state_registry)}")
    print(f"Transitions: {len(results.transition_registry)}")
    print(f"Mode: {results.manifest.get('mode')}")
    print(f"Stop reason: {results.manifest.get('stop_reason')}")
    print(f"Terminal state: {results.manifest.get('terminal_state_id')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
