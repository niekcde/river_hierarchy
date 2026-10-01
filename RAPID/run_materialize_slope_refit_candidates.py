from __future__ import annotations

"""Materialise reviewed base slope-refit candidates as isolated artifacts.

This is deliberately a batch *candidate* runner. It never opens an existing
hydraulic-context directory for writing and does not invoke resistance or
RAPID. The lambda registry below is the reviewed decision table from
``notebooks/slope_investigation.ipynb``.
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
if str(ROOT.parent) not in sys.path:
    sys.path.insert(0, str(ROOT.parent))

import pandas as pd

from network_variants.reference_profile_candidate import generate_reference_profile_candidate_context
from rapid_tools.reference_profile import (
    ReferenceProfileConfig,
    apply_manual_linear_profile_bridge,
    fit_reference_profile,
    write_reference_profile_artifacts,
)
from rapid_tools.reference_profile_inputs import build_reference_profile_inputs


SLOPE_REFIT_SCHEMA_VERSION = "slope-refit-candidate-batch-v4"

# Recorded from the accepted 16-example notebook selection table. Values are
# materialised directly: rerunning selection is unnecessary and would discard
# the reviewed upper-tail convergence decision for 12 examples.
REVIEWED_LAMBDA_BY_EXAMPLE: dict[str, float] = {
    "02": 1.995262e11,
    "03": 7.943282e5,
    "05": 7.943282e5,
    "11": 3.981072e5,
    "13": 1.995262e11,
    "15": 3.981072e5,
    "21": 1.995262e11,
    "22": 1.995262e11,
    "37": 1.995262e11,
    "41": 1.995262e11,
    "45": 1.995262e11,
    "46": 1.995262e11,
    "48": 1.995262e11,
    "54": 1.995262e11,
    "55": 1.995262e11,
    "56": 1.995262e11,
}

# A profile-level correction accepted after reviewing the weakly supported
# Example 46 370--410 km reach. The endpoints retain their fitted heads; only
# the intervening profile rows are replaced by their continuous straight line.
REVIEWED_MANUAL_PROFILE_BRIDGES_BY_EXAMPLE: dict[str, tuple[tuple[str, float, float], ...]] = {
    "46": (("manual_profile_bridge_46_370_410", 369970.16393422784, 410002.52962018247),),
}


def _canonical_example_id(value: str) -> str:
    numeric = int(str(value).strip())
    result = f"{numeric:02d}"
    if result not in REVIEWED_LAMBDA_BY_EXAMPLE:
        raise ValueError(f"Example {result} is not in the reviewed slope-refit population.")
    return result


def _required_profile_files(profile_dir: Path) -> list[Path]:
    return [
        profile_dir / "reference_profile.parquet",
        profile_dir / "reference_profile_observations.parquet",
        profile_dir / "reference_profile_acquisition_offsets.parquet",
        profile_dir / "reference_profile_cv.csv",
        profile_dir / "reference_profile_selection.json",
        profile_dir / "reference_profile_audit.json",
    ]


def _required_candidate_files(candidate_dir: Path) -> list[Path]:
    return [
        candidate_dir / "candidate_link_hydraulic_context.parquet",
        candidate_dir / "candidate_node_hydraulic_context.parquet",
        candidate_dir / "candidate_domain_head_loss_audit.csv",
        candidate_dir / "candidate_link_ownership_audit.csv",
        candidate_dir / "candidate_context_manifest.json",
    ]


def _complete(paths: list[Path]) -> bool:
    return all(path.exists() and path.stat().st_size > 0 for path in paths)


def _write_fixed_profile(
    experiment_dir: Path,
    profile_dir: Path,
    *,
    sword_node_source: str | Path,
    raw_observations_path: str | Path,
    lambda_value: float,
    manual_bridges: tuple[tuple[str, float, float], ...] = (),
) -> None:
    inputs = build_reference_profile_inputs(
        experiment_dir,
        sword_node_source=sword_node_source,
        raw_observations_path=raw_observations_path,
    )
    config = ReferenceProfileConfig()
    fit = fit_reference_profile(
        inputs.static_profile,
        inputs.observations,
        endpoint_distances_m=inputs.endpoint_distances_m,
        lambda_value=lambda_value,
        config=config,
    )
    manual_adjustments: list[dict[str, object]] = []
    for adjustment_id, start_distance_m, end_distance_m in manual_bridges:
        fit, adjustment = apply_manual_linear_profile_bridge(
            fit,
            adjustment_id=adjustment_id,
            start_distance_m=start_distance_m,
            end_distance_m=end_distance_m,
        )
        manual_adjustments.append(adjustment)
    cv = pd.DataFrame(
        [{
            "lambda_value": lambda_value,
            "cv_mean_mse_m2": pd.NA,
            "cv_sd_mse_m2": pd.NA,
            "cv_se_mse_m2": pd.NA,
            "n_acquisitions": int(fit.summary["n_acquisitions"]),
        }]
    )
    selection = {
        "selection_rule": (
            "accepted_notebook_fixed_lambda_plus_reviewed_manual_profile_bridge"
            if manual_adjustments
            else "accepted_notebook_fixed_lambda"
        ),
        "selected_lambda": lambda_value,
        "best_lambda": lambda_value,
        "one_standard_error_threshold_m2": None,
        "selected_at_grid_upper_edge": False,
        "lambda_grid_count": 1,
        "cv_reused_from_notebook": True,
        "reviewed_population": "direction-corrected-16-example-batch",
    }
    write_reference_profile_artifacts(
        fit,
        cv,
        selection,
        profile_dir,
        config=config,
        input_metadata=inputs.metadata,
        manual_adjustments=manual_adjustments,
    )


def materialize_candidates(
    output_root: str | Path,
    *,
    experiment_output_root: str | Path,
    sword_node_source: str | Path,
    raw_observations_path: str | Path,
    example_ids: list[str],
    execute: bool,
    resume: bool,
    reuse_profile_root: str | Path | None = None,
    apply_reviewed_manual_bridges: bool = False,
    checkpoint_csv: Path | None = None,
) -> pd.DataFrame:
    root = Path(output_root).expanduser().resolve()
    experiment_root = Path(experiment_output_root).expanduser().resolve()
    reusable_root = (
        Path(reuse_profile_root).expanduser().resolve()
        if reuse_profile_root is not None
        else None
    )
    manual_bridges_by_example = (
        REVIEWED_MANUAL_PROFILE_BRIDGES_BY_EXAMPLE if apply_reviewed_manual_bridges else {}
    )
    rows: list[dict[str, object]] = []

    def checkpoint() -> None:
        if checkpoint_csv is None:
            return
        current = pd.DataFrame(rows)
        if checkpoint_csv.exists():
            previous = pd.read_csv(checkpoint_csv)
            if "example_id" in previous.columns:
                previous["example_id"] = previous["example_id"].map(_canonical_example_id)
                current["example_id"] = current["example_id"].map(_canonical_example_id)
                current = pd.concat([previous, current], ignore_index=True, sort=False)
                current = current.drop_duplicates(subset="example_id", keep="last")
            else:
                current = pd.concat([previous, current], ignore_index=True, sort=False)
        current["example_id"] = current["example_id"].astype(str).str.zfill(2)
        current.sort_values("example_id").to_csv(checkpoint_csv, index=False)
    for example_id in example_ids:
        experiment_dir = (experiment_root / f"{example_id}_indep__anchored").resolve()
        manual_bridges = manual_bridges_by_example.get(example_id, ())
        profile_dir = (
            root / "profiles" / example_id
            if manual_bridges
            else (
                reusable_root / "profiles" / example_id
                if reusable_root is not None
                else root / "profiles" / example_id
            )
        )
        candidate_dir = root / "candidate_contexts" / example_id
        profile_complete = _complete(_required_profile_files(profile_dir))
        candidate_complete = _complete(_required_candidate_files(candidate_dir))
        status = "planned"
        if not experiment_dir.exists():
            status = "missing_experiment"
        elif profile_complete and candidate_complete:
            status = "already_complete"
        elif profile_dir.exists() and not profile_complete:
            status = "incomplete_existing_profile_requires_manual_review"
        elif candidate_dir.exists() and not candidate_complete:
            status = "incomplete_existing_candidate_requires_manual_review"
        elif profile_complete and not candidate_dir.exists():
            status = "profile_complete_candidate_planned" if resume else "profile_complete_requires_resume"
        row = {
            "example_id": example_id,
            "reviewed_lambda": REVIEWED_LAMBDA_BY_EXAMPLE[example_id],
            "experiment_dir": str(experiment_dir),
            "profile_dir": str(profile_dir),
            "candidate_context_dir": str(candidate_dir),
            "profile_complete_before": profile_complete,
            "candidate_complete_before": candidate_complete,
            "manual_profile_bridges": json.dumps(manual_bridges),
            "status": status,
            "action": "",
            "error": "",
        }
        if execute and status in {"planned", "profile_complete_candidate_planned"}:
            try:
                print(f"Materialising candidate: {example_id}", flush=True)
                if not profile_complete:
                    if reusable_root is not None and not manual_bridges:
                        raise FileNotFoundError(
                            f"Reusable frozen profile is incomplete for example {example_id}: {profile_dir}"
                        )
                    _write_fixed_profile(
                        experiment_dir,
                        profile_dir,
                        sword_node_source=sword_node_source,
                        raw_observations_path=raw_observations_path,
                        lambda_value=REVIEWED_LAMBDA_BY_EXAMPLE[example_id],
                        manual_bridges=manual_bridges,
                    )
                    row["action"] = "profile_materialised"
                if not candidate_complete:
                    generate_reference_profile_candidate_context(experiment_dir, profile_dir, candidate_dir)
                    row["action"] = "profile_and_candidate_materialised" if row["action"] else "candidate_materialised"
                row["status"] = "complete"
            except Exception as exc:
                row["status"] = "failed"
                row["error"] = f"{type(exc).__name__}: {exc}"
        rows.append(row)
        checkpoint()
    if checkpoint_csv is not None and checkpoint_csv.exists():
        accumulated = pd.read_csv(checkpoint_csv)
        accumulated["example_id"] = accumulated["example_id"].map(_canonical_example_id)
        return accumulated.drop_duplicates(subset="example_id", keep="last").sort_values("example_id").reset_index(drop=True)
    return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Materialise reviewed slope-refit profiles and isolated base candidate contexts. "
            "Dry-run is the default; use --execute after reviewing the manifest."
        )
    )
    parser.add_argument("output_root", help="New root for all immutable candidate artifacts.")
    parser.add_argument(
        "--experiment-output-root",
        default="/Volumes/PhD/river_hierarchy/network_variants/output",
    )
    parser.add_argument("--sword-node-source", required=True)
    parser.add_argument(
        "--raw-observations-path",
        default=str(ROOT.parent / "data_validation" / "swot_sword_17c" / "target_observations.parquet"),
    )
    parser.add_argument("--examples", nargs="+", help="Optional reviewed example IDs; default is all 16.")
    parser.add_argument(
        "--reuse-profile-root",
        help=(
            "Existing immutable candidate root whose frozen profiles are reused. "
            "Only candidate contexts are written under OUTPUT_ROOT; profile fitting is skipped."
        ),
    )
    parser.add_argument(
        "--apply-reviewed-manual-bridges",
        action="store_true",
        help=(
            "Apply the reviewed profile-level bridge registry. A bridged example is "
            "refit under OUTPUT_ROOT while other examples may reuse frozen profiles."
        ),
    )
    parser.add_argument("--execute", action="store_true", help="Write profile and candidate artifacts.")
    parser.add_argument("--resume", action="store_true", help="Continue only an incomplete candidate output root.")
    args = parser.parse_args(argv)
    examples = [_canonical_example_id(value) for value in (args.examples or list(REVIEWED_LAMBDA_BY_EXAMPLE))]
    if len(examples) != len(set(examples)):
        parser.error("--examples contains a duplicate ID.")
    root = Path(args.output_root).expanduser().resolve()
    checkpoint_csv = None
    if args.execute:
        root.mkdir(parents=True, exist_ok=True)
        checkpoint_csv = root / "slope_refit_candidate_batch_manifest.csv"
        (root / "slope_refit_candidate_batch_manifest.json").write_text(
            json.dumps(
                {
                    "schema_version": SLOPE_REFIT_SCHEMA_VERSION,
                    "generated_at": datetime.now(timezone.utc).isoformat(),
                    "execution_requested": True,
                    "run_status": "in_progress",
                    "reviewed_lambda_by_example": REVIEWED_LAMBDA_BY_EXAMPLE,
                    "reuse_profile_root": str(Path(args.reuse_profile_root).expanduser().resolve()) if args.reuse_profile_root else None,
                    "reviewed_manual_profile_bridges_applied": bool(args.apply_reviewed_manual_bridges),
                    "manual_profile_bridge_registry": REVIEWED_MANUAL_PROFILE_BRIDGES_BY_EXAMPLE if args.apply_reviewed_manual_bridges else {},
                    "manifest_csv": str(checkpoint_csv),
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
    report = materialize_candidates(
        args.output_root,
        experiment_output_root=args.experiment_output_root,
        sword_node_source=args.sword_node_source,
        raw_observations_path=args.raw_observations_path,
        example_ids=examples,
        execute=args.execute,
        resume=args.resume,
        reuse_profile_root=args.reuse_profile_root,
        apply_reviewed_manual_bridges=args.apply_reviewed_manual_bridges,
        checkpoint_csv=checkpoint_csv,
    )
    if args.execute:
        manifest_path = root / "slope_refit_candidate_batch_manifest.csv"
        metadata_path = root / "slope_refit_candidate_batch_manifest.json"
        report.to_csv(manifest_path, index=False)
        metadata_path.write_text(
            json.dumps(
                {
                    "schema_version": SLOPE_REFIT_SCHEMA_VERSION,
                    "generated_at": datetime.now(timezone.utc).isoformat(),
                    "execution_requested": True,
                    "run_status": "completed",
                    "reviewed_lambda_by_example": REVIEWED_LAMBDA_BY_EXAMPLE,
                    "reuse_profile_root": str(Path(args.reuse_profile_root).expanduser().resolve()) if args.reuse_profile_root else None,
                    "reviewed_manual_profile_bridges_applied": bool(args.apply_reviewed_manual_bridges),
                    "manual_profile_bridge_registry": REVIEWED_MANUAL_PROFILE_BRIDGES_BY_EXAMPLE if args.apply_reviewed_manual_bridges else {},
                    "manifest_csv": str(manifest_path),
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
    print(report.to_string(index=False))
    print(f"Complete: {int(report['status'].eq('complete').sum())}; already complete: {int(report['status'].eq('already_complete').sum())}; failures: {int(report['status'].eq('failed').sum())}")
    return 0 if not report["status"].isin([
        "failed",
        "missing_experiment",
        "incomplete_existing_profile_requires_manual_review",
        "incomplete_existing_candidate_requires_manual_review",
        "profile_complete_requires_resume",
    ]).any() else 1


if __name__ == "__main__":
    raise SystemExit(main())
