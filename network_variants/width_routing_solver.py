from __future__ import annotations

import argparse
import json
import subprocess
from collections import defaultdict, deque
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from network_variants.width_junction_audit import ENDPOINTS_FILENAME
from network_variants.width_junction_hybrid_audit import (
    DEFAULT_JUNCTION_AUDIT_DIR,
    DEFAULT_WINDOW_ID,
)
from network_variants.width_qc_final_model import (
    DEFAULT_OUTPUT_DIR as DEFAULT_FINAL_MODEL_DIR,
    EXCLUDED_FILENAME,
    FINAL_MODEL_VERSION,
)
from network_variants.width_qc_model import PREDICTIONS_FILENAME
from network_variants.width_review_archive import _atomic_write_csv, _atomic_write_json, _sha256


SOLVER_VERSION = "width-routing-solver-v1"
DEFAULT_MIN_SCALE_FACTOR = 0.25
DEFAULT_MAX_SCALE_FACTOR = 4.0
DEFAULT_SELECTED_MODEL = "logistic_regression"
DEFAULT_ENDPOINTS_PATH = DEFAULT_JUNCTION_AUDIT_DIR / ENDPOINTS_FILENAME
DEFAULT_PREDICTIONS_PATH = DEFAULT_FINAL_MODEL_DIR / "model" / PREDICTIONS_FILENAME
DEFAULT_EXCLUDED_PATH = DEFAULT_FINAL_MODEL_DIR / EXCLUDED_FILENAME
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "output" / "width_routing_solver"

LINKS_FILENAME = "width_routing_solver_links_v1.csv"
JUNCTIONS_FILENAME = "width_routing_solver_junctions_v1.csv"
SUMMARY_FILENAME = "width_routing_solver_summary_v1.csv"
REJECTED_FILENAME = "width_routing_solver_rejected_corrections_v1.csv"
MANIFEST_FILENAME = "width_routing_solver_manifest_v1.json"


def _git_revision() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[1],
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip() or None
    except (OSError, subprocess.CalledProcessError):
        return None


def prepare_hybrid_solver_links(
    endpoint_widths: pd.DataFrame,
    *,
    length_over_width_threshold: float = 10.0,
    require_nonoverlap: bool = True,
) -> pd.DataFrame:
    required = {
        "example_id",
        "id_link",
        "id_us_node",
        "id_ds_node",
        "whole_link_mean_wet_width",
        "width_near_us_node_wet",
        "width_near_ds_node_wet",
        "length_over_wet_width",
        "endpoint_windows_nonoverlapping",
    }
    missing = sorted(required.difference(endpoint_widths.columns))
    if missing:
        raise ValueError(f"Endpoint-width data is missing solver fields: {missing}")
    links = endpoint_widths.copy()
    links["example_id"] = links["example_id"].astype("string").str.zfill(2)
    for column in ("id_link", "id_us_node", "id_ds_node"):
        links[column] = pd.to_numeric(links[column], errors="raise").astype(int)
    for column in (
        "whole_link_mean_wet_width",
        "width_near_us_node_wet",
        "width_near_ds_node_wet",
        "length_over_wet_width",
    ):
        links[column] = pd.to_numeric(links[column], errors="coerce")
    eligible = links["length_over_wet_width"].ge(float(length_over_width_threshold))
    if require_nonoverlap:
        eligible &= links["endpoint_windows_nonoverlapping"].astype("boolean").fillna(False)
    eligible &= (
        links["width_near_us_node_wet"].notna()
        & links["width_near_ds_node_wet"].notna()
    )
    links["wid_adj_wet"] = links["whole_link_mean_wet_width"]
    links["routing_context_width_us"] = np.where(
        eligible, links["width_near_us_node_wet"], links["wid_adj_wet"]
    )
    links["routing_context_width_ds"] = np.where(
        eligible, links["width_near_ds_node_wet"], links["wid_adj_wet"]
    )
    links["routing_context_uses_endpoint"] = eligible
    return links[
        [
            "example_id",
            "id_link",
            "id_us_node",
            "id_ds_node",
            "wid_adj_wet",
            "routing_context_width_us",
            "routing_context_width_ds",
            "routing_context_uses_endpoint",
            "length_over_wet_width",
            "endpoint_windows_nonoverlapping",
        ]
    ]


def _topological_nodes(links: pd.DataFrame) -> tuple[list[int], set[int]]:
    nodes = set(links["id_us_node"]).union(links["id_ds_node"])
    successors: dict[int, set[int]] = defaultdict(set)
    indegree = {int(node): 0 for node in nodes}
    for row in links.itertuples(index=False):
        source, target = int(row.id_us_node), int(row.id_ds_node)
        if target not in successors[source]:
            successors[source].add(target)
            indegree[target] += 1
    queue = deque(sorted(node for node, degree in indegree.items() if degree == 0))
    order: list[int] = []
    while queue:
        node = queue.popleft()
        order.append(node)
        for target in sorted(successors.get(node, ())):
            indegree[target] -= 1
            if indegree[target] == 0:
                queue.append(target)
    cyclic = {node for node, degree in indegree.items() if degree > 0}
    return order, cyclic


def _ids(values: pd.Series) -> str:
    return ",".join(str(int(value)) for value in sorted(values.astype(int)))


def solve_routing_widths(
    links: pd.DataFrame,
    decisions: pd.DataFrame,
    *,
    min_scale_factor: float = DEFAULT_MIN_SCALE_FACTOR,
    max_scale_factor: float = DEFAULT_MAX_SCALE_FACTOR,
    model_version: str = FINAL_MODEL_VERSION,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not 0 < min_scale_factor <= 1 <= max_scale_factor:
        raise ValueError("Solver scale bounds must satisfy 0 < minimum <= 1 <= maximum.")
    required_links = {
        "id_link",
        "id_us_node",
        "id_ds_node",
        "wid_adj_wet",
        "routing_context_width_us",
        "routing_context_width_ds",
    }
    missing = sorted(required_links.difference(links.columns))
    if missing:
        raise ValueError(f"Solver links are missing fields: {missing}")
    required_decisions = {"id_link", "width_model_prediction"}
    missing = sorted(required_decisions.difference(decisions.columns))
    if missing:
        raise ValueError(f"Solver decisions are missing fields: {missing}")

    frame = links.copy()
    for column in ("id_link", "id_us_node", "id_ds_node"):
        frame[column] = pd.to_numeric(frame[column], errors="raise").astype(int)
    for column in ("wid_adj_wet", "routing_context_width_us", "routing_context_width_ds"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    if "routing_context_uses_endpoint" not in frame:
        frame["routing_context_uses_endpoint"] = False
    if "length_over_wet_width" not in frame:
        frame["length_over_wet_width"] = np.nan
    if frame.duplicated("id_link").any():
        raise ValueError("Solver links contain duplicate link IDs.")
    invalid_width = (
        frame[["wid_adj_wet", "routing_context_width_us", "routing_context_width_ds"]]
        .le(0)
        .any(axis=1)
        | frame[["wid_adj_wet", "routing_context_width_us", "routing_context_width_ds"]]
        .isna()
        .any(axis=1)
    )
    if invalid_width.any():
        raise ValueError(
            "Solver requires positive raw and endpoint-context widths; invalid links: "
            f"{frame.loc[invalid_width, 'id_link'].tolist()[:10]}"
        )

    choice = decisions.copy()
    choice["id_link"] = pd.to_numeric(choice["id_link"], errors="raise").astype(int)
    if choice.duplicated("id_link").any():
        raise ValueError("Solver decisions contain duplicate link IDs.")
    keep = ["id_link", "width_model_prediction"]
    for optional in ("width_model_probability", "decision_status"):
        if optional in choice.columns:
            keep.append(optional)
    frame = frame.merge(choice[keep], on="id_link", how="left", validate="1:1")
    prediction = pd.to_numeric(frame["width_model_prediction"], errors="coerce")
    if not bool(prediction.dropna().isin([0, 1]).all()):
        raise ValueError("width_model_prediction must contain only 0, 1, or missing values.")
    frame["width_model_prediction"] = prediction.astype("Int64")
    if "width_model_probability" not in frame:
        frame["width_model_probability"] = np.nan
    if "decision_status" not in frame:
        frame["decision_status"] = pd.NA

    frame["wid_adj_routing"] = frame["wid_adj_wet"]
    frame["width_qc_decision"] = "accept"
    frame["width_qc_reason"] = np.where(
        frame["width_model_prediction"].eq(0).fillna(False),
        "model_accept",
        "not_model_candidate",
    )
    ambiguous = frame["decision_status"].eq("ambiguous_excluded").fillna(False)
    frame.loc[ambiguous, "width_qc_decision"] = "retain_raw_flag"
    frame.loc[ambiguous, "width_qc_reason"] = "ambiguous_excluded_from_final_training"
    frame["width_qc_flag"] = ambiguous
    frame["width_correction_applied"] = False
    frame["width_correction_source"] = pd.NA
    frame["width_algorithm_version"] = SOLVER_VERSION
    frame["width_model_version"] = model_version
    frame["width_correction_scale_factor"] = 1.0
    frame["width_correction_context_target"] = np.nan
    frame["width_correction_candidate_context_target"] = np.nan
    frame["width_correction_candidate_routing_width"] = np.nan
    frame["width_correction_candidate_scale_factor"] = np.nan
    frame["width_correction_scale_bound_violation"] = ""
    frame["width_correction_min_scale_factor"] = float(min_scale_factor)
    frame["width_correction_max_scale_factor"] = float(max_scale_factor)

    frame = frame.reset_index(drop=True)
    incoming_by_node = frame.groupby("id_ds_node").groups
    outgoing_by_node = frame.groupby("id_us_node").groups
    order, cyclic_nodes = _topological_nodes(frame)
    junction_rows: list[dict[str, Any]] = []

    for node_id in order:
        incoming_index = list(incoming_by_node.get(node_id, []))
        outgoing_index = list(outgoing_by_node.get(node_id, []))
        if not outgoing_index:
            continue
        outgoing = frame.loc[outgoing_index]
        flagged_index = outgoing.index[
            outgoing["width_model_prediction"].eq(1).fillna(False)
        ].tolist()
        incoming_context = float(
            (
                frame.loc[incoming_index, "routing_context_width_ds"]
                * frame.loc[incoming_index, "width_correction_scale_factor"]
            ).sum()
        ) if incoming_index else np.nan
        outgoing_before = float(outgoing["routing_context_width_us"].sum())
        before_ratio = outgoing_before / incoming_context if incoming_context > 0 else np.nan
        status = "no_flagged_outgoing"
        reason = ""
        proposed_context: dict[int, float] = {}
        accepted_context = np.nan
        remaining = np.nan
        candidate_scale_min = np.nan
        candidate_scale_max = np.nan
        scale_bound_violation = ""

        if flagged_index:
            accepted_index = [index for index in outgoing_index if index not in flagged_index]
            accepted_context = float(frame.loc[accepted_index, "routing_context_width_us"].sum())
            remaining = incoming_context - accepted_context if np.isfinite(incoming_context) else np.nan
            weights = frame.loc[flagged_index, "routing_context_width_us"]
            if np.isfinite(remaining) and np.isfinite(weights.sum()) and weights.sum() > 0:
                candidate_targets = remaining * weights / float(weights.sum())
                candidate_scales = candidate_targets / weights
                candidate_scale_min = float(candidate_scales.min())
                candidate_scale_max = float(candidate_scales.max())
                for index, candidate_target in candidate_targets.items():
                    candidate_scale = float(candidate_scales.loc[index])
                    frame.at[index, "width_correction_candidate_context_target"] = float(
                        candidate_target
                    )
                    frame.at[index, "width_correction_candidate_scale_factor"] = candidate_scale
                    frame.at[index, "width_correction_candidate_routing_width"] = float(
                        frame.at[index, "wid_adj_wet"]
                    ) * candidate_scale
            if not incoming_index or not np.isfinite(incoming_context):
                status, reason = "retain_raw_flag", "no_incoming_width_at_source_node"
            elif not np.isfinite(remaining) or remaining <= 0:
                status, reason = "retain_raw_flag", "nonpositive_width_remaining_after_accepted_outgoing"
                scale_bound_violation = "nonpositive_remaining_width"
            elif not np.isfinite(weights.sum()) or weights.sum() <= 0:
                status, reason = "retain_raw_flag", "invalid_flagged_outgoing_allocation_weights"
            else:
                targets = candidate_targets
                scales = candidate_scales
                unsafe = scales.lt(float(min_scale_factor)) | scales.gt(float(max_scale_factor))
                if unsafe.any():
                    status, reason = "retain_raw_flag", "joint_allocation_outside_scale_bounds"
                    below = bool(scales.lt(float(min_scale_factor)).any())
                    above = bool(scales.gt(float(max_scale_factor)).any())
                    scale_bound_violation = (
                        "below_min_and_above_max"
                        if below and above
                        else "below_min"
                        if below
                        else "above_max"
                    )
                else:
                    status, reason = "correct", "joint_upstream_continuity_allocation"
                    for index, target in targets.items():
                        proposed_context[int(index)] = float(target)
                        scale = float(scales.loc[index])
                        frame.at[index, "wid_adj_routing"] = float(frame.at[index, "wid_adj_wet"]) * scale
                        frame.at[index, "width_correction_scale_factor"] = scale
                        frame.at[index, "width_correction_context_target"] = float(target)
                        frame.at[index, "width_qc_decision"] = "correct"
                        frame.at[index, "width_qc_reason"] = reason
                        frame.at[index, "width_correction_applied"] = not np.isclose(scale, 1.0)
                        frame.at[index, "width_correction_source"] = "joint_upstream_continuity"
            if status == "retain_raw_flag":
                frame.loc[flagged_index, "width_qc_decision"] = status
                frame.loc[flagged_index, "width_qc_reason"] = reason
                frame.loc[flagged_index, "width_qc_flag"] = True
                frame.loc[
                    flagged_index, "width_correction_scale_bound_violation"
                ] = scale_bound_violation

        outgoing_after = float(
            (
                frame.loc[outgoing_index, "routing_context_width_us"]
                * frame.loc[outgoing_index, "width_correction_scale_factor"]
            ).sum()
        )
        after_ratio = outgoing_after / incoming_context if incoming_context > 0 else np.nan
        junction_rows.append(
            {
                "id_node": int(node_id),
                "incoming_link_ids": _ids(frame.loc[incoming_index, "id_link"]) if incoming_index else "",
                "outgoing_link_ids": _ids(frame.loc[outgoing_index, "id_link"]),
                "flagged_outgoing_link_ids": _ids(frame.loc[flagged_index, "id_link"]) if flagged_index else "",
                "flagged_outgoing_count": int(len(flagged_index)),
                "incoming_context_width_sum": incoming_context,
                "accepted_outgoing_context_width_sum": accepted_context,
                "remaining_context_width_for_flagged": remaining,
                "candidate_scale_factor_min": candidate_scale_min,
                "candidate_scale_factor_max": candidate_scale_max,
                "scale_bound_violation": scale_bound_violation,
                "outgoing_context_width_sum_before": outgoing_before,
                "outgoing_context_width_sum_after": outgoing_after,
                "continuity_ratio_before": before_ratio,
                "continuity_abs_deviation_before": abs(before_ratio - 1) if np.isfinite(before_ratio) else np.nan,
                "continuity_ratio_after": after_ratio,
                "continuity_abs_deviation_after": abs(after_ratio - 1) if np.isfinite(after_ratio) else np.nan,
                "solver_status": status,
                "solver_reason": reason,
            }
        )

    if cyclic_nodes:
        cyclic_outgoing = frame["id_us_node"].isin(cyclic_nodes) & frame[
            "width_model_prediction"
        ].eq(1).fillna(False)
        frame.loc[cyclic_outgoing, "width_qc_decision"] = "retain_raw_flag"
        frame.loc[cyclic_outgoing, "width_qc_reason"] = "directed_topology_cycle"
        frame.loc[cyclic_outgoing, "width_qc_flag"] = True

    frame["width_correction_absolute_change"] = frame["wid_adj_routing"] - frame["wid_adj_wet"]
    frame["width_correction_absolute_magnitude"] = frame["width_correction_absolute_change"].abs()
    frame["width_correction_relative_change"] = (
        frame["width_correction_absolute_change"] / frame["wid_adj_wet"]
    )
    junctions = pd.DataFrame(junction_rows)
    return frame.sort_values("id_link", kind="mergesort").reset_index(drop=True), junctions


def build_rejected_correction_report(
    links: pd.DataFrame,
    junctions: pd.DataFrame,
) -> pd.DataFrame:
    rejected = links.loc[
        links["width_model_prediction"].eq(1)
        & links["width_qc_decision"].eq("retain_raw_flag")
    ].copy()
    junction_columns = [
        "example_id",
        "id_node",
        "incoming_link_ids",
        "outgoing_link_ids",
        "flagged_outgoing_link_ids",
        "flagged_outgoing_count",
        "incoming_context_width_sum",
        "accepted_outgoing_context_width_sum",
        "remaining_context_width_for_flagged",
        "outgoing_context_width_sum_before",
        "outgoing_context_width_sum_after",
        "continuity_ratio_before",
        "continuity_abs_deviation_before",
        "continuity_ratio_after",
        "continuity_abs_deviation_after",
        "candidate_scale_factor_min",
        "candidate_scale_factor_max",
        "scale_bound_violation",
        "solver_status",
        "solver_reason",
    ]
    missing = sorted(set(junction_columns).difference(junctions.columns))
    if missing:
        raise ValueError(f"Junction results are missing rejected-report fields: {missing}")
    rejected = rejected.merge(
        junctions[junction_columns],
        left_on=["example_id", "id_us_node"],
        right_on=["example_id", "id_node"],
        how="left",
        validate="m:1",
        suffixes=("", "_junction"),
    )
    columns = [
        "example_id",
        "id_link",
        "id_us_node",
        "id_ds_node",
        "width_model_probability",
        "width_model_prediction",
        "wid_adj_wet",
        "wid_adj_routing",
        "width_correction_candidate_routing_width",
        "width_correction_candidate_context_target",
        "width_correction_candidate_scale_factor",
        "width_correction_min_scale_factor",
        "width_correction_max_scale_factor",
        "width_correction_scale_bound_violation",
        "routing_context_width_us",
        "routing_context_width_ds",
        "routing_context_uses_endpoint",
        "length_over_wet_width",
        "incoming_link_ids",
        "outgoing_link_ids",
        "flagged_outgoing_link_ids",
        "flagged_outgoing_count",
        "incoming_context_width_sum",
        "accepted_outgoing_context_width_sum",
        "remaining_context_width_for_flagged",
        "outgoing_context_width_sum_before",
        "outgoing_context_width_sum_after",
        "continuity_ratio_before",
        "continuity_abs_deviation_before",
        "continuity_ratio_after",
        "continuity_abs_deviation_after",
        "candidate_scale_factor_min",
        "candidate_scale_factor_max",
        "scale_bound_violation",
        "width_qc_decision",
        "width_qc_reason",
        "solver_status",
        "solver_reason",
    ]
    missing = sorted(set(columns).difference(rejected.columns))
    if missing:
        raise ValueError(f"Rejected correction report is missing fields: {missing}")
    return rejected[columns].sort_values(
        ["width_qc_reason", "width_model_probability", "example_id", "id_link"],
        ascending=[True, False, True, True],
        kind="mergesort",
    ).reset_index(drop=True)


def build_solver_summary(links: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    groups: list[tuple[str, pd.DataFrame]] = [("ALL", links)]
    if "example_id" in links:
        groups.extend((str(example), values) for example, values in links.groupby("example_id", sort=True))
    for example_id, values in groups:
        predicted = values["width_model_prediction"].eq(1)
        applied = values["width_correction_applied"].astype(bool)
        flagged = values["width_qc_flag"].astype(bool)
        magnitudes = values.loc[applied, "width_correction_absolute_magnitude"]
        relative = values.loc[applied, "width_correction_relative_change"].abs()
        rows.append(
            {
                "example_id": example_id,
                "all_links": int(len(values)),
                "model_candidates": int(values["width_model_prediction"].notna().sum()),
                "predicted_corrections": int(predicted.sum()),
                "corrections_applied": int(applied.sum()),
                "retained_raw_with_flag": int(flagged.sum()),
                "ambiguous_excluded": int(values["decision_status"].eq("ambiguous_excluded").sum()),
                "median_absolute_width_change": float(magnitudes.median()) if len(magnitudes) else np.nan,
                "p95_absolute_width_change": float(magnitudes.quantile(0.95)) if len(magnitudes) else np.nan,
                "median_absolute_relative_change": float(relative.median()) if len(relative) else np.nan,
                "p95_absolute_relative_change": float(relative.quantile(0.95)) if len(relative) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def run_solver_evaluation(
    *,
    endpoints_path: str | Path = DEFAULT_ENDPOINTS_PATH,
    predictions_path: str | Path = DEFAULT_PREDICTIONS_PATH,
    excluded_path: str | Path = DEFAULT_EXCLUDED_PATH,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    selected_model: str = DEFAULT_SELECTED_MODEL,
    window_id: str = DEFAULT_WINDOW_ID,
    min_scale_factor: float = DEFAULT_MIN_SCALE_FACTOR,
    max_scale_factor: float = DEFAULT_MAX_SCALE_FACTOR,
) -> dict[str, Any]:
    endpoints_path = Path(endpoints_path).expanduser().resolve()
    predictions_path = Path(predictions_path).expanduser().resolve()
    excluded_path = Path(excluded_path).expanduser().resolve()
    output = Path(output_dir).expanduser().resolve()
    for path in (endpoints_path, predictions_path, excluded_path):
        if not path.exists():
            raise FileNotFoundError(path)
    endpoints = pd.read_csv(endpoints_path, dtype={"example_id": "string"})
    endpoints = endpoints.loc[endpoints["window_id"].eq(window_id)].copy()
    solver_links = prepare_hybrid_solver_links(endpoints)
    predictions = pd.read_csv(predictions_path, dtype={"example_id": "string"})
    predictions = predictions.loc[predictions["width_model_name"].eq(selected_model)].copy()
    if predictions.empty:
        raise ValueError(f"Final predictions contain no rows for model {selected_model!r}.")
    if set(predictions["width_model_version"].dropna().astype(str)) != {FINAL_MODEL_VERSION}:
        raise ValueError("Selected predictions do not use the frozen final model version.")
    predictions["example_id"] = predictions["example_id"].astype("string").str.zfill(2)
    predictions["id_link"] = pd.to_numeric(predictions["id_link"], errors="raise").astype(int)
    if predictions.duplicated(["example_id", "id_link"]).any():
        raise ValueError("Selected final predictions contain duplicate example/link keys.")
    excluded = pd.read_csv(excluded_path, dtype={"example_id": "string"})
    excluded["example_id"] = excluded["example_id"].astype("string").str.zfill(2)
    excluded["id_link"] = pd.to_numeric(excluded["id_link"], errors="raise").astype(int)
    if excluded.duplicated(["example_id", "id_link"]).any():
        raise ValueError("Excluded cases contain duplicate example/link keys.")

    link_outputs: list[pd.DataFrame] = []
    junction_outputs: list[pd.DataFrame] = []
    for example_id, example_links in solver_links.groupby("example_id", sort=True):
        decisions = predictions.loc[
            predictions["example_id"].eq(example_id),
            ["id_link", "width_model_prediction", "width_model_probability"],
        ].copy()
        decisions["decision_status"] = "final_held_out_prediction"
        ambiguous = excluded.loc[excluded["example_id"].eq(example_id), ["id_link"]].copy()
        ambiguous["width_model_prediction"] = pd.NA
        ambiguous["width_model_probability"] = np.nan
        ambiguous["decision_status"] = "ambiguous_excluded"
        decisions = pd.concat([decisions, ambiguous], ignore_index=True)
        solved, junctions = solve_routing_widths(
            example_links.drop(columns="example_id"),
            decisions,
            min_scale_factor=min_scale_factor,
            max_scale_factor=max_scale_factor,
        )
        solved.insert(0, "example_id", example_id)
        junctions.insert(0, "example_id", example_id)
        link_outputs.append(solved)
        junction_outputs.append(junctions)
    solved_links = pd.concat(link_outputs, ignore_index=True)
    solved_junctions = pd.concat(junction_outputs, ignore_index=True)
    summary = build_solver_summary(solved_links)
    rejected = build_rejected_correction_report(solved_links, solved_junctions)

    output.mkdir(parents=True, exist_ok=True)
    output_paths = {
        "links": output / LINKS_FILENAME,
        "junctions": output / JUNCTIONS_FILENAME,
        "summary": output / SUMMARY_FILENAME,
        "rejected_corrections": output / REJECTED_FILENAME,
        "manifest": output / MANIFEST_FILENAME,
    }
    _atomic_write_csv(solved_links, output_paths["links"])
    _atomic_write_csv(solved_junctions, output_paths["junctions"])
    _atomic_write_csv(summary, output_paths["summary"])
    _atomic_write_csv(rejected, output_paths["rejected_corrections"])
    manifest = {
        "schema_version": SOLVER_VERSION,
        "code_revision": _git_revision(),
        "production_authorized": False,
        "evaluation_only": True,
        "selected_model": selected_model,
        "model_version": FINAL_MODEL_VERSION,
        "window_id": window_id,
        "context_rule": "lw_ge_10_nonoverlap",
        "min_scale_factor": float(min_scale_factor),
        "max_scale_factor": float(max_scale_factor),
        "inputs": {
            "endpoint_widths": str(endpoints_path),
            "predictions": str(predictions_path),
            "excluded_cases": str(excluded_path),
        },
        "input_sha256": {
            "endpoint_widths": _sha256(endpoints_path),
            "predictions": _sha256(predictions_path),
            "excluded_cases": _sha256(excluded_path),
        },
        "counts": summary.loc[summary["example_id"].eq("ALL")].iloc[0].to_dict(),
        "files": {key: str(path) for key, path in output_paths.items() if key != "manifest"},
        "file_sha256": {
            key: _sha256(path) for key, path in output_paths.items() if key != "manifest"
        },
    }
    _atomic_write_json(manifest, output_paths["manifest"])
    return {"output_dir": str(output), "manifest": str(output_paths["manifest"]), **manifest["counts"]}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate joint upstream-continuity routing-width corrections.")
    parser.add_argument("--endpoint-widths", default=str(DEFAULT_ENDPOINTS_PATH))
    parser.add_argument("--predictions", default=str(DEFAULT_PREDICTIONS_PATH))
    parser.add_argument("--excluded-cases", default=str(DEFAULT_EXCLUDED_PATH))
    parser.add_argument("--selected-model", default=DEFAULT_SELECTED_MODEL)
    parser.add_argument("--window-id", default=DEFAULT_WINDOW_ID)
    parser.add_argument("--min-scale-factor", type=float, default=DEFAULT_MIN_SCALE_FACTOR)
    parser.add_argument("--max-scale-factor", type=float, default=DEFAULT_MAX_SCALE_FACTOR)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = run_solver_evaluation(
        endpoints_path=args.endpoint_widths,
        predictions_path=args.predictions,
        excluded_path=args.excluded_cases,
        output_dir=args.output_dir,
        selected_model=args.selected_model,
        window_id=args.window_id,
        min_scale_factor=args.min_scale_factor,
        max_scale_factor=args.max_scale_factor,
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
