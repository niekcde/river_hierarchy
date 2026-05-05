from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

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
from network_variants.variant_generation import NetworkVariantOutputs, generate_network_variant


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

    state_records: list[dict[str, Any]] = [_state_record(base_context, mode)]
    transition_records: list[dict[str, Any]] = []
    stop_reason = "base_state_only"
    terminal_state_id = base_context.state_id

    def write_metadata() -> None:
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
            child_context = _build_child_state(
                parent_context=base_context,
                selection=selection,
                state_id=state_id,
                state_dir=states_dir / state_id,
                exit_sides=exit_sides,
                variant_runner=variant_runner,
                variant_kwargs=variant_kwargs,
                unit_workflow_runner=unit_workflow_runner,
                unit_workflow_writer=unit_workflow_writer,
                unit_workflow_kwargs=unit_workflow_kwargs,
            )
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
        child_context = _build_child_state(
            parent_context=current_context,
            selection=selection,
            state_id=state_id,
            state_dir=states_dir / state_id,
            exit_sides=exit_sides,
            variant_runner=variant_runner,
            variant_kwargs=variant_kwargs,
            unit_workflow_runner=unit_workflow_runner,
            unit_workflow_writer=unit_workflow_writer,
            unit_workflow_kwargs=unit_workflow_kwargs,
        )
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
