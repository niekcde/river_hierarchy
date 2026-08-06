from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from network_variants.adaptive_routing_solver import (
    AdaptiveSolverConfig,
    adaptive_evidence_parameters,
    solve_adaptive_two_ended,
)

from network_variants.width_review_archive import _atomic_write_csv, _atomic_write_json, _sha256


COMPARISON_VERSION = "width-solver-comparison-v1"
DEFAULT_SOLVER_DIR = Path(__file__).resolve().parent / "output" / "width_routing_solver"
DEFAULT_FINAL_MODEL_DIR = Path(__file__).resolve().parent / "output" / "width_qc_final_model"
DEFAULT_FINAL_MODEL_MANIFEST = DEFAULT_FINAL_MODEL_DIR / "width_qc_final_workflow_manifest_v1.json"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "output" / "width_routing_solver_comparison"
LINKS_FILENAME = "width_solver_comparison_links_v1.csv"
SUMMARY_FILENAME = "width_solver_comparison_summary_v1.csv"
CASES_FILENAME = "width_solver_comparison_nominated_cases_v1.csv"
MANIFEST_FILENAME = "width_solver_comparison_manifest_v1.json"


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


def _node_deviations(frame: pd.DataFrame, scales: np.ndarray) -> dict[int, float]:
    us_context = frame["routing_context_width_us"].to_numpy(float) * scales
    ds_context = frame["routing_context_width_ds"].to_numpy(float) * scales
    result: dict[int, float] = {}
    for node in sorted(set(frame["id_us_node"]).union(frame["id_ds_node"])):
        incoming = float(ds_context[frame["id_ds_node"].to_numpy(int) == int(node)].sum())
        outgoing = float(us_context[frame["id_us_node"].to_numpy(int) == int(node)].sum())
        result[int(node)] = abs(outgoing / incoming - 1.0) if incoming > 0 else float("nan")
    return result


def _evaluate_method(
    frame: pd.DataFrame,
    *,
    method: str,
    scales: pd.Series,
    predicted_mask: pd.Series,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    evaluated = frame.copy()
    evaluated["comparison_method"] = method
    evaluated["comparison_scale_factor"] = pd.to_numeric(scales, errors="coerce").fillna(1.0)
    evaluated["comparison_routing_width"] = (
        evaluated["wid_adj_wet"].astype(float) * evaluated["comparison_scale_factor"]
    )
    for column in (
        "upstream_deviation_before",
        "upstream_deviation_after",
        "downstream_deviation_before",
        "downstream_deviation_after",
    ):
        evaluated[column] = np.nan
    # Node IDs are only unique within an example. Evaluate every network
    # independently so equal numeric node IDs from different rivers never mix.
    for _, example in evaluated.groupby("example_id", sort=False):
        positions = example.index
        raw_scales = np.ones(len(example), dtype=float)
        method_scales = example["comparison_scale_factor"].to_numpy(float)
        before = _node_deviations(example, raw_scales)
        after = _node_deviations(example, method_scales)
        evaluated.loc[positions, "upstream_deviation_before"] = example["id_us_node"].map(before).to_numpy()
        evaluated.loc[positions, "upstream_deviation_after"] = example["id_us_node"].map(after).to_numpy()
        evaluated.loc[positions, "downstream_deviation_before"] = example["id_ds_node"].map(before).to_numpy()
        evaluated.loc[positions, "downstream_deviation_after"] = example["id_ds_node"].map(after).to_numpy()
    candidates = evaluated.loc[predicted_mask].copy()
    valid_upstream = candidates[["upstream_deviation_before", "upstream_deviation_after"]].dropna()
    valid_downstream = candidates[["downstream_deviation_before", "downstream_deviation_after"]].dropna()
    changed = ~np.isclose(candidates["comparison_scale_factor"], 1.0)
    summary = {
        "comparison_method": method,
        "predicted_corrections": int(len(candidates)),
        "corrections_changed": int(changed.sum()),
        "normal_bound_violations": int(
            (
                candidates["comparison_scale_factor"].lt(0.25)
                | candidates["comparison_scale_factor"].gt(4.0)
            ).sum()
        ),
        "nonpositive_widths": int(candidates["comparison_routing_width"].le(0).sum()),
        "median_scale": float(candidates.loc[changed, "comparison_scale_factor"].median()) if changed.any() else np.nan,
        "p05_scale": float(candidates.loc[changed, "comparison_scale_factor"].quantile(0.05)) if changed.any() else np.nan,
        "p95_scale": float(candidates.loc[changed, "comparison_scale_factor"].quantile(0.95)) if changed.any() else np.nan,
        "median_upstream_deviation_before": float(valid_upstream["upstream_deviation_before"].median()),
        "median_upstream_deviation_after": float(valid_upstream["upstream_deviation_after"].median()),
        "p95_upstream_deviation_after": float(valid_upstream["upstream_deviation_after"].quantile(0.95)),
        "upstream_improved": int(
            (valid_upstream["upstream_deviation_after"] < valid_upstream["upstream_deviation_before"] - 1e-9).sum()
        ),
        "upstream_worsened": int(
            (valid_upstream["upstream_deviation_after"] > valid_upstream["upstream_deviation_before"] + 1e-9).sum()
        ),
        "median_downstream_deviation_before": float(valid_downstream["downstream_deviation_before"].median()),
        "median_downstream_deviation_after": float(valid_downstream["downstream_deviation_after"].median()),
        "p95_downstream_deviation_after": float(valid_downstream["downstream_deviation_after"].quantile(0.95)),
        "downstream_improved": int(
            (valid_downstream["downstream_deviation_after"] < valid_downstream["downstream_deviation_before"] - 1e-9).sum()
        ),
        "downstream_worsened": int(
            (valid_downstream["downstream_deviation_after"] > valid_downstream["downstream_deviation_before"] + 1e-9).sum()
        ),
        "downstream_deviation_gt_1": int(valid_downstream["downstream_deviation_after"].gt(1.0).sum()),
    }
    return candidates, summary


def run_solver_comparison(
    *,
    solver_links_path: str | Path = DEFAULT_SOLVER_DIR / "width_routing_solver_links_v1.csv",
    final_dataset_path: str | Path = DEFAULT_FINAL_MODEL_DIR / "width_qc_final_training_data_v1.csv",
    final_model_manifest_path: str | Path = DEFAULT_FINAL_MODEL_MANIFEST,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    upstream_weights: Sequence[float] = (2.0, 3.0, 5.0),
) -> dict[str, Any]:
    solver_links_path = Path(solver_links_path).expanduser().resolve()
    final_dataset_path = Path(final_dataset_path).expanduser().resolve()
    final_model_manifest_path = Path(final_model_manifest_path).expanduser().resolve()
    output = Path(output_dir).expanduser().resolve()
    final_model_manifest = json.loads(final_model_manifest_path.read_text(encoding="utf-8"))
    selected_threshold = float(final_model_manifest["selected_threshold"])
    links = pd.read_csv(solver_links_path, dtype={"example_id": "string"})
    dataset = pd.read_csv(final_dataset_path, dtype={"example_id": "string"})
    for frame in (links, dataset):
        frame["example_id"] = frame["example_id"].astype("string").str.zfill(2)
        frame["id_link"] = pd.to_numeric(frame["id_link"], errors="raise").astype(int)
    evidence_columns = [
        "example_id",
        "id_link",
        "manual_decision",
        "opposition_cos_trimmed_mean",
        "opposition_cos_trimmed_median",
        "upstream_continuity_abs_deviation",
        "downstream_continuity_abs_deviation",
        "legacy_width_update_scale_factor",
        "legacy_width_update_target_context_width",
    ]
    links = links.merge(
        dataset[[column for column in evidence_columns if column in dataset.columns]],
        on=["example_id", "id_link"],
        how="left",
        validate="1:1",
    )
    links["width_model_threshold"] = selected_threshold
    predicted = links["width_model_prediction"].eq(1).fillna(False)

    outputs: list[pd.DataFrame] = []
    summaries: list[dict[str, Any]] = []
    current_scales = pd.to_numeric(links["width_correction_scale_factor"], errors="coerce").fillna(1.0)
    current_rows, current_summary = _evaluate_method(
        links, method="current_upstream_exact", scales=current_scales, predicted_mask=predicted
    )
    outputs.append(current_rows)
    summaries.append(current_summary)

    candidate = pd.to_numeric(links["width_correction_candidate_scale_factor"], errors="coerce")
    capped_scales = candidate.where(predicted, 1.0).fillna(1.0).clip(lower=0.25, upper=4.0)
    capped_rows, capped_summary = _evaluate_method(
        links, method="hard_capped_partial", scales=capped_scales, predicted_mask=predicted
    )
    outputs.append(capped_rows)
    summaries.append(capped_summary)

    adaptive_diagnostics: dict[str, Any] = {}
    for weight in upstream_weights:
        method = f"adaptive_two_ended_w{weight:g}"
        method_scales = pd.Series(1.0, index=links.index, dtype=float)
        method_evidence: list[pd.DataFrame] = []
        method_runs: list[dict[str, Any]] = []
        for example_id, example_links in links.groupby("example_id", sort=True):
            config = AdaptiveSolverConfig(upstream_weight=float(weight))
            solved, diagnostics = solve_adaptive_two_ended(example_links, config=config)
            original_index = example_links.index
            method_scales.loc[original_index] = solved["comparison_scale_factor"].to_numpy(float)
            evidence_fields = [
                "id_link",
                "model_confidence_strength",
                "upstream_deviation_strength",
                "opposition_badness",
                "adaptive_relaxation_strength",
                "adaptive_min_scale",
                "adaptive_max_scale",
                "adaptive_raw_penalty_weight",
                "comparison_status",
                "comparison_flag",
            ]
            evidence_frame = solved[evidence_fields].copy()
            evidence_frame.insert(0, "example_id", example_id)
            method_evidence.append(evidence_frame)
            method_runs.append({"example_id": example_id, **diagnostics})
        rows, summary = _evaluate_method(
            links, method=method, scales=method_scales, predicted_mask=predicted
        )
        evidence_all = pd.concat(method_evidence, ignore_index=True)
        rows = rows.merge(evidence_all, on=["example_id", "id_link"], how="left", validate="1:1")
        outputs.append(rows)
        summaries.append(summary)
        adaptive_diagnostics[method] = method_runs

    legacy_scales = pd.to_numeric(links["legacy_width_update_scale_factor"], errors="coerce")
    legacy_available = predicted & legacy_scales.notna() & legacy_scales.gt(0)
    legacy_method_scales = legacy_scales.where(legacy_available, 1.0).fillna(1.0)
    legacy_rows, legacy_summary = _evaluate_method(
        links, method="legacy_two_ended_available", scales=legacy_method_scales, predicted_mask=predicted
    )
    legacy_summary["legacy_scale_available"] = int(legacy_available.sum())
    outputs.append(legacy_rows)
    summaries.append(legacy_summary)

    comparison_links = pd.concat(outputs, ignore_index=True)
    summary_frame = pd.DataFrame(summaries)
    nominated = comparison_links.loc[
        (
            comparison_links["example_id"].eq("37")
            & comparison_links["id_link"].isin([143, 241, 304])
        )
        | (comparison_links["example_id"].eq("22") & comparison_links["id_link"].eq(231))
    ].copy()
    output.mkdir(parents=True, exist_ok=True)
    links_path = output / LINKS_FILENAME
    summary_path = output / SUMMARY_FILENAME
    cases_path = output / CASES_FILENAME
    manifest_path = output / MANIFEST_FILENAME
    _atomic_write_csv(comparison_links, links_path)
    _atomic_write_csv(summary_frame, summary_path)
    _atomic_write_csv(nominated, cases_path)
    manifest = {
        "schema_version": COMPARISON_VERSION,
        "code_revision": _git_revision(),
        "production_authorized": False,
        "purpose": "Offline solver-method comparison; no production solver artifacts are modified.",
        "inputs": {
            "solver_links": str(solver_links_path),
            "solver_links_sha256": _sha256(solver_links_path),
            "final_dataset": str(final_dataset_path),
            "final_dataset_sha256": _sha256(final_dataset_path),
            "final_model_manifest": str(final_model_manifest_path),
            "final_model_manifest_sha256": _sha256(final_model_manifest_path),
            "selected_model_threshold": selected_threshold,
        },
        "methods": [str(value) for value in summary_frame["comparison_method"]],
        "adaptive_contract": {
            "upstream_weights": [float(value) for value in upstream_weights],
            "downstream_weight": 1.0,
            "raw_penalty": 0.20,
            "normal_scale_bounds": [0.25, 4.0],
            "hard_adaptive_scale_bounds": [0.05, 8.0],
            "evidence": [
                "width_model_probability",
                "upstream_continuity_abs_deviation",
                "opposition_cos_trimmed_mean",
            ],
            "accepted_siblings": "fixed",
            "continuity": "soft robust log-ratio loss at upstream and downstream nodes",
        },
        "adaptive_diagnostics": adaptive_diagnostics,
        "files": {
            "links": str(links_path),
            "summary": str(summary_path),
            "nominated_cases": str(cases_path),
        },
        "file_sha256": {
            "links": _sha256(links_path),
            "summary": _sha256(summary_path),
            "nominated_cases": _sha256(cases_path),
        },
    }
    _atomic_write_json(manifest, manifest_path)
    return {
        "output_dir": str(output),
        "manifest": str(manifest_path),
        "predicted_corrections": int(predicted.sum()),
        "summary": summary_frame.to_dict("records"),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare routing-width solvers without changing production outputs.")
    parser.add_argument("--solver-links", default=str(DEFAULT_SOLVER_DIR / "width_routing_solver_links_v1.csv"))
    parser.add_argument("--final-dataset", default=str(DEFAULT_FINAL_MODEL_DIR / "width_qc_final_training_data_v1.csv"))
    parser.add_argument("--final-model-manifest", default=str(DEFAULT_FINAL_MODEL_MANIFEST))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--upstream-weight", type=float, action="append", dest="upstream_weights")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = run_solver_comparison(
        solver_links_path=args.solver_links,
        final_dataset_path=args.final_dataset,
        final_model_manifest_path=args.final_model_manifest,
        output_dir=args.output_dir,
        upstream_weights=tuple(args.upstream_weights or (2.0, 3.0, 5.0)),
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
