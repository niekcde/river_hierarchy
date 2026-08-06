from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from network_variants.width_junction_hybrid_audit import (
    COMPARISONS_FILENAME as HYBRID_COMPARISONS_FILENAME,
    DEFAULT_OUTPUT_DIR as DEFAULT_HYBRID_OUTPUT_DIR,
    DEFAULT_REVIEW_MANIFEST_PATH,
    DEFAULT_REVIEW_PATH,
    MANIFEST_FILENAME as HYBRID_MANIFEST_FILENAME,
    build_effective_label_frame,
)
from network_variants.width_qc_model import (
    DEFAULT_DATASET_PATH,
    DEFAULT_MAX_SPLIT_FALSE_CORRECTION_RATE,
    DEFAULT_MODEL_OUTPUT_DIR,
    FEATURE_COLUMNS,
    MANIFEST_FILENAME as ORIGINAL_MODEL_MANIFEST_FILENAME,
    PREDICTIONS_FILENAME as ORIGINAL_PREDICTIONS_FILENAME,
    classification_metrics,
    run_leave_one_example_out,
    write_model_development_outputs,
)
from network_variants.width_review_archive import _atomic_write_csv, _atomic_write_json, _sha256


FINAL_MODEL_VERSION = "width-qc-final-hybrid-v1"
FINAL_DATASET_SCHEMA_VERSION = "width-qc-final-training-v1"
FINAL_WORKFLOW_SCHEMA_VERSION = "width-qc-final-workflow-v1"
FROZEN_REVIEW_REBASE_FILENAME = "width_qc_disagreement_review_rebase_v1.json"
FROZEN_HYBRID_VARIANT = "lw_ge_10_nonoverlap"
HYBRID_FEATURE = "upstream_hybrid_continuity_abs_deviation"
FINAL_FEATURE_COLUMNS = (*FEATURE_COLUMNS, HYBRID_FEATURE)

DEFAULT_HYBRID_COMPARISONS_PATH = DEFAULT_HYBRID_OUTPUT_DIR / HYBRID_COMPARISONS_FILENAME
DEFAULT_HYBRID_MANIFEST_PATH = DEFAULT_HYBRID_OUTPUT_DIR / HYBRID_MANIFEST_FILENAME
DEFAULT_ORIGINAL_PREDICTIONS_PATH = DEFAULT_MODEL_OUTPUT_DIR / ORIGINAL_PREDICTIONS_FILENAME
DEFAULT_ORIGINAL_MODEL_MANIFEST_PATH = DEFAULT_MODEL_OUTPUT_DIR / ORIGINAL_MODEL_MANIFEST_FILENAME
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "output" / "width_qc_final_model"

FINAL_DATASET_FILENAME = "width_qc_final_training_data_v1.csv"
EXCLUDED_FILENAME = "width_qc_final_excluded_cases_v1.csv"
FINAL_DATASET_MANIFEST_FILENAME = "width_qc_final_training_manifest_v1.json"
COMPARISON_FILENAME = "width_qc_old_new_prediction_comparison_v1.csv"
DECISION_SUMMARY_FILENAME = "width_qc_old_new_decision_summary_v1.csv"
VALIDATION_COMPARISON_FILENAME = "width_qc_old_new_validation_comparison_v1.csv"
WORKFLOW_MANIFEST_FILENAME = "width_qc_final_workflow_manifest_v1.json"


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


def _load_manifest(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _validate_hashed_file(
    *,
    file_path: Path,
    manifest: dict[str, Any],
    hash_key: str,
    label: str,
) -> None:
    expected = (manifest.get("file_sha256", {}) or {}).get(hash_key)
    if not expected:
        raise ValueError(f"{label} manifest does not record the {hash_key!r} hash.")
    if _sha256(file_path) != expected:
        raise ValueError(f"{label} does not match its source manifest.")


def build_final_training_dataset(
    dataset: pd.DataFrame,
    review: pd.DataFrame,
    hybrid_comparisons: pd.DataFrame,
    *,
    hybrid_variant: str = FROZEN_HYBRID_VARIANT,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    source = dataset.copy()
    source["example_id"] = source["example_id"].astype("string").str.zfill(2)
    source["id_link"] = pd.to_numeric(source["id_link"], errors="raise").astype(int)
    if source.duplicated(["example_id", "id_link"]).any():
        raise ValueError("Source training dataset contains duplicate example/link keys.")
    missing_features = sorted(set(FEATURE_COLUMNS).difference(source.columns))
    if missing_features:
        raise ValueError(f"Source training dataset is missing model features: {missing_features}")

    labels = build_effective_label_frame(source, review)[
        [
            "example_id",
            "id_link",
            "adjudication",
            "adjudicated_requires_correction",
            "include_in_training",
            "review_comment",
            "effective_requires_correction",
            "effective_label_source",
            "in_completed_review",
        ]
    ]
    hybrid = hybrid_comparisons.copy()
    hybrid["example_id"] = hybrid["example_id"].astype("string").str.zfill(2)
    hybrid["id_link"] = pd.to_numeric(hybrid["id_link"], errors="raise").astype(int)
    hybrid = hybrid.loc[hybrid["hybrid_variant_id"].eq(hybrid_variant)].copy()
    if hybrid.duplicated(["example_id", "id_link"]).any():
        raise ValueError(f"Hybrid audit contains duplicate rows for {hybrid_variant!r}.")
    if len(hybrid) != len(source):
        raise ValueError(
            f"Expected one {hybrid_variant!r} row for each source link; found {len(hybrid)} "
            f"for {len(source)} source rows."
        )
    hybrid = hybrid[
        [
            "example_id",
            "id_link",
            "hybrid_continuity_ratio",
            "hybrid_continuity_abs_deviation",
            "incoming_endpoint_link_count",
            "outgoing_endpoint_link_count",
        ]
    ].rename(
        columns={
            "hybrid_continuity_ratio": "upstream_hybrid_continuity_ratio",
            "hybrid_continuity_abs_deviation": HYBRID_FEATURE,
            "incoming_endpoint_link_count": "upstream_hybrid_incoming_endpoint_link_count",
            "outgoing_endpoint_link_count": "upstream_hybrid_outgoing_endpoint_link_count",
        }
    )

    frame = source.merge(labels, on=["example_id", "id_link"], how="left", validate="1:1")
    frame = frame.merge(hybrid, on=["example_id", "id_link"], how="left", validate="1:1")
    if frame[HYBRID_FEATURE].isna().any():
        missing = frame.loc[frame[HYBRID_FEATURE].isna(), ["example_id", "id_link"]]
        raise ValueError(f"Final dataset has missing frozen hybrid values: {missing.to_dict('records')[:10]}")

    frame["original_manual_decision"] = frame["manual_decision"]
    frame["original_requires_correction"] = pd.to_numeric(
        frame["requires_correction"], errors="raise"
    ).astype(int)
    frame["final_dataset_schema_version"] = FINAL_DATASET_SCHEMA_VERSION
    frame["final_label_changed_by_adjudication"] = (
        frame["effective_requires_correction"].notna()
        & frame["effective_requires_correction"].astype("Float64").ne(
            frame["original_requires_correction"].astype(float)
        )
    )
    frame["frozen_hybrid_variant"] = hybrid_variant
    frame["upstream_hybrid_uses_endpoint"] = (
        frame["upstream_hybrid_incoming_endpoint_link_count"].fillna(0)
        + frame["upstream_hybrid_outgoing_endpoint_link_count"].fillna(0)
    ).gt(0)

    excluded = frame.loc[frame["effective_requires_correction"].isna()].copy()
    included = frame.loc[frame["effective_requires_correction"].notna()].copy()
    included["requires_correction"] = included["effective_requires_correction"].astype(int)
    included["manual_decision"] = included["requires_correction"].map({0: "good", 1: "bad"})
    for feature in FINAL_FEATURE_COLUMNS:
        included[feature] = pd.to_numeric(included[feature], errors="coerce")
    if included[list(FINAL_FEATURE_COLUMNS)].isna().all(axis=0).any():
        empty = included[list(FINAL_FEATURE_COLUMNS)].columns[
            included[list(FINAL_FEATURE_COLUMNS)].isna().all()
        ].tolist()
        raise ValueError(f"Final model features are entirely missing: {empty}")
    return (
        included.sort_values(["example_id", "id_link"], kind="mergesort").reset_index(drop=True),
        excluded.sort_values(["example_id", "id_link"], kind="mergesort").reset_index(drop=True),
    )


def build_old_new_prediction_comparison(
    final_dataset: pd.DataFrame,
    old_predictions: pd.DataFrame,
    new_predictions: pd.DataFrame,
) -> pd.DataFrame:
    keys = ["example_id", "id_link", "width_model_name"]
    old = old_predictions.copy()
    new = new_predictions.copy()
    for frame in (old, new):
        frame["example_id"] = frame["example_id"].astype("string").str.zfill(2)
        frame["id_link"] = pd.to_numeric(frame["id_link"], errors="raise").astype(int)
    if old.duplicated(keys).any() or new.duplicated(keys).any():
        raise ValueError("Old or final held-out predictions contain duplicate model/example/link keys.")

    old = old[
        [
            *keys,
            "width_model_probability",
            "width_model_prediction",
            "width_model_threshold",
            "width_model_version",
        ]
    ].rename(
        columns={
            "width_model_probability": "old_probability",
            "width_model_prediction": "old_prediction",
            "width_model_threshold": "old_threshold",
            "width_model_version": "old_model_version",
        }
    )
    new = new[
        [
            *keys,
            "requires_correction",
            "width_model_probability",
            "width_model_prediction",
            "width_model_threshold",
            "width_model_version",
            "routing_impact_score",
        ]
    ].rename(
        columns={
            "requires_correction": "final_requires_correction",
            "width_model_probability": "new_probability",
            "width_model_prediction": "new_prediction",
            "width_model_threshold": "new_threshold",
            "width_model_version": "new_model_version",
        }
    )
    comparison = new.merge(old, on=keys, how="left", validate="1:1")
    if comparison["old_probability"].isna().any():
        raise ValueError("Original held-out predictions do not cover every included final-model row.")

    context = final_dataset[
        [
            "example_id",
            "id_link",
            "original_manual_decision",
            "original_requires_correction",
            "effective_label_source",
            "adjudication",
            "final_label_changed_by_adjudication",
            "split_sensitive",
            "raw_wet_width",
            "upstream_continuity_abs_deviation",
            HYBRID_FEATURE,
            "upstream_hybrid_uses_endpoint",
        ]
    ]
    comparison = comparison.merge(context, on=["example_id", "id_link"], validate="m:1")
    comparison["probability_change"] = comparison["new_probability"] - comparison["old_probability"]
    comparison["absolute_probability_change"] = comparison["probability_change"].abs()
    comparison["decision_changed"] = comparison["new_prediction"].ne(comparison["old_prediction"])
    comparison["new_prediction_at_old_threshold"] = comparison["new_probability"].ge(
        comparison["old_threshold"]
    ).astype(int)
    comparison["changed_at_fixed_old_threshold"] = comparison[
        "new_prediction_at_old_threshold"
    ].ne(comparison["old_prediction"])
    comparison["changed_by_final_threshold_after_refit"] = comparison["new_prediction"].ne(
        comparison["new_prediction_at_old_threshold"]
    )
    comparison["decision_change"] = np.select(
        [
            comparison["old_prediction"].eq(0) & comparison["new_prediction"].eq(1),
            comparison["old_prediction"].eq(1) & comparison["new_prediction"].eq(0),
        ],
        ["accept_to_correct", "correct_to_accept"],
        default="unchanged",
    )
    comparison["continuity_feature_change"] = (
        comparison[HYBRID_FEATURE] - comparison["upstream_continuity_abs_deviation"]
    )
    comparison["numeric_correction_status"] = np.where(
        comparison["new_prediction"].eq(1),
        "pending_joint_upstream_continuity_solver",
        "not_required",
    )
    return comparison.sort_values(
        ["width_model_name", "absolute_probability_change", "example_id", "id_link"],
        ascending=[True, False, True, True],
        kind="mergesort",
    ).reset_index(drop=True)


def build_decision_summary(comparison: pd.DataFrame, *, selected_model: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for model_name, model_rows in comparison.groupby("width_model_name", sort=True):
        groups: list[tuple[str, str, pd.DataFrame]] = [("all", "ALL", model_rows)]
        groups.extend(
            ("example", str(example_id), example_rows)
            for example_id, example_rows in model_rows.groupby("example_id", sort=True)
        )
        for scope, example_id, values in groups:
            rows.append(
                {
                    "width_model_name": model_name,
                    "selected_model": model_name == selected_model,
                    "scope": scope,
                    "example_id": example_id,
                    "rows": int(len(values)),
                    "old_predicted_corrections": int(values["old_prediction"].sum()),
                    "new_predicted_corrections": int(values["new_prediction"].sum()),
                    "decision_changes": int(values["decision_changed"].sum()),
                    "changes_at_fixed_old_threshold": int(
                        values["changed_at_fixed_old_threshold"].sum()
                    ),
                    "changes_from_final_threshold_after_refit": int(
                        values["changed_by_final_threshold_after_refit"].sum()
                    ),
                    "accept_to_correct": int(values["decision_change"].eq("accept_to_correct").sum()),
                    "correct_to_accept": int(values["decision_change"].eq("correct_to_accept").sum()),
                    "label_changes_from_adjudication": int(
                        values[["example_id", "id_link", "final_label_changed_by_adjudication"]]
                        .drop_duplicates(["example_id", "id_link"])["final_label_changed_by_adjudication"]
                        .sum()
                    ),
                    "mean_probability_change": float(values["probability_change"].mean()),
                    "median_absolute_probability_change": float(
                        values["absolute_probability_change"].median()
                    ),
                    "p95_absolute_probability_change": float(
                        values["absolute_probability_change"].quantile(0.95)
                    ),
                }
            )
    return pd.DataFrame(rows).sort_values(
        ["selected_model", "width_model_name", "scope", "example_id"],
        ascending=[False, True, True, True],
        kind="mergesort",
    ).reset_index(drop=True)


def build_validation_comparison(comparison: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for model_name, values in comparison.groupby("width_model_name", sort=True):
        labels = values["final_requires_correction"].astype(int).to_numpy()
        split_sensitive = values["split_sensitive"].fillna(0).astype(bool).to_numpy()
        for version, probability_column, threshold_column in (
            ("original_model_on_final_labels", "old_probability", "old_threshold"),
            ("final_hybrid_model", "new_probability", "new_threshold"),
        ):
            thresholds = pd.to_numeric(values[threshold_column], errors="raise").unique()
            if len(thresholds) != 1:
                raise ValueError(f"Model {model_name} has inconsistent {threshold_column} values.")
            metrics = classification_metrics(
                labels,
                pd.to_numeric(values[probability_column], errors="raise").to_numpy(dtype=float),
                threshold=float(thresholds[0]),
                split_sensitive=split_sensitive,
            )
            rows.append({"width_model_name": model_name, "comparison_version": version, **metrics})
    return pd.DataFrame(rows).sort_values(
        ["width_model_name", "comparison_version"], kind="mergesort"
    ).reset_index(drop=True)


def run_final_model_workflow(
    *,
    dataset_path: str | Path = DEFAULT_DATASET_PATH,
    review_path: str | Path = DEFAULT_REVIEW_PATH,
    review_manifest_path: str | Path = DEFAULT_REVIEW_MANIFEST_PATH,
    hybrid_comparisons_path: str | Path = DEFAULT_HYBRID_COMPARISONS_PATH,
    hybrid_manifest_path: str | Path = DEFAULT_HYBRID_MANIFEST_PATH,
    original_predictions_path: str | Path = DEFAULT_ORIGINAL_PREDICTIONS_PATH,
    original_model_manifest_path: str | Path = DEFAULT_ORIGINAL_MODEL_MANIFEST_PATH,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    max_split_false_correction_rate: float = DEFAULT_MAX_SPLIT_FALSE_CORRECTION_RATE,
) -> dict[str, Any]:
    paths = {
        "dataset": Path(dataset_path).expanduser().resolve(),
        "review": Path(review_path).expanduser().resolve(),
        "review_manifest": Path(review_manifest_path).expanduser().resolve(),
        "hybrid_comparisons": Path(hybrid_comparisons_path).expanduser().resolve(),
        "hybrid_manifest": Path(hybrid_manifest_path).expanduser().resolve(),
        "original_predictions": Path(original_predictions_path).expanduser().resolve(),
        "original_model_manifest": Path(original_model_manifest_path).expanduser().resolve(),
    }
    for path in paths.values():
        if not path.exists():
            raise FileNotFoundError(path)
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)

    review_manifest = _load_manifest(paths["review_manifest"])
    hybrid_manifest = _load_manifest(paths["hybrid_manifest"])
    original_model_manifest = _load_manifest(paths["original_model_manifest"])
    _validate_hashed_file(
        file_path=paths["review"], manifest=review_manifest, hash_key="review", label="Review CSV"
    )
    _validate_hashed_file(
        file_path=paths["hybrid_comparisons"],
        manifest=hybrid_manifest,
        hash_key="comparisons",
        label="Hybrid comparisons",
    )
    _validate_hashed_file(
        file_path=paths["original_predictions"],
        manifest=original_model_manifest,
        hash_key="predictions",
        label="Original predictions",
    )
    dataset_hash = _sha256(paths["dataset"])
    original_review_dataset_hash = (review_manifest.get("inputs", {}) or {}).get("dataset_sha256")
    frozen_review_rebase_path: Path | None = None
    if original_review_dataset_hash != dataset_hash:
        frozen_review_rebase_path = paths["review_manifest"].parent / FROZEN_REVIEW_REBASE_FILENAME
        if not frozen_review_rebase_path.exists():
            raise ValueError("Completed review and source training dataset hashes disagree.")
        frozen_review_rebase = _load_manifest(frozen_review_rebase_path)
        if frozen_review_rebase.get("status") != "validated_frozen_adjudication_reuse":
            raise ValueError("Frozen adjudication rebase is not validated.")
        if (frozen_review_rebase.get("original_inputs", {}) or {}).get(
            "dataset_sha256"
        ) != original_review_dataset_hash:
            raise ValueError("Frozen adjudication rebase does not originate from the reviewed dataset.")
        if (frozen_review_rebase.get("current_inputs", {}) or {}).get(
            "dataset_sha256"
        ) != dataset_hash:
            raise ValueError("Frozen adjudication rebase does not authorize the current dataset.")
        human_review = frozen_review_rebase.get("human_review", {}) or {}
        if human_review.get("review_sha256") != _sha256(paths["review"]):
            raise ValueError("Frozen adjudication rebase refers to a different human-review CSV.")
        if human_review.get("manifest_sha256") != _sha256(paths["review_manifest"]):
            raise ValueError("Frozen adjudication rebase refers to a different human-review manifest.")
    if (hybrid_manifest.get("inputs", {}) or {}).get("dataset_sha256") != dataset_hash:
        raise ValueError("Hybrid audit and source training dataset hashes disagree.")
    if int((review_manifest.get("counts", {}) or {}).get("unreviewed", -1)) != 0:
        raise ValueError("Completed disagreement review is not complete.")

    dataset = pd.read_csv(paths["dataset"], dtype={"example_id": "string"})
    review = pd.read_csv(paths["review"], dtype={"example_id": "string"})
    hybrid = pd.read_csv(paths["hybrid_comparisons"], dtype={"example_id": "string"})
    final_dataset, excluded = build_final_training_dataset(dataset, review, hybrid)

    final_dataset_path = output / FINAL_DATASET_FILENAME
    excluded_path = output / EXCLUDED_FILENAME
    dataset_manifest_path = output / FINAL_DATASET_MANIFEST_FILENAME
    _atomic_write_csv(final_dataset, final_dataset_path)
    _atomic_write_csv(excluded, excluded_path)
    final_dataset_manifest = {
        "schema_version": FINAL_DATASET_SCHEMA_VERSION,
        "dataset_id": _sha256(final_dataset_path),
        "dataset_path": str(final_dataset_path),
        "dataset_sha256": _sha256(final_dataset_path),
        "row_count": int(len(final_dataset)),
        "excluded_row_count": int(len(excluded)),
        "label_changes_from_adjudication": int(final_dataset["final_label_changed_by_adjudication"].sum()),
        "feature_columns": list(FINAL_FEATURE_COLUMNS),
        "frozen_hybrid_variant": FROZEN_HYBRID_VARIANT,
        "source_sha256": {key: _sha256(path) for key, path in paths.items()},
        "excluded_path": str(excluded_path),
        "excluded_sha256": _sha256(excluded_path),
    }
    _atomic_write_json(final_dataset_manifest, dataset_manifest_path)

    result = run_leave_one_example_out(
        final_dataset,
        max_split_false_correction_rate=max_split_false_correction_rate,
        feature_columns=FINAL_FEATURE_COLUMNS,
        model_version=FINAL_MODEL_VERSION,
    )
    model_outputs = write_model_development_outputs(
        result,
        dataset_path=final_dataset_path,
        dataset_manifest_path=dataset_manifest_path,
        output_dir=output / "model",
        feature_columns=FINAL_FEATURE_COLUMNS,
        model_version=FINAL_MODEL_VERSION,
    )
    selected_model = result.summary["provisional_candidate_model"]
    old_predictions = pd.read_csv(paths["original_predictions"], dtype={"example_id": "string"})
    comparison = build_old_new_prediction_comparison(
        final_dataset, old_predictions, result.predictions
    )
    decision_summary = build_decision_summary(comparison, selected_model=selected_model)
    validation_comparison = build_validation_comparison(comparison)
    comparison_path = output / COMPARISON_FILENAME
    decision_summary_path = output / DECISION_SUMMARY_FILENAME
    validation_comparison_path = output / VALIDATION_COMPARISON_FILENAME
    _atomic_write_csv(comparison, comparison_path)
    _atomic_write_csv(decision_summary, decision_summary_path)
    _atomic_write_csv(validation_comparison, validation_comparison_path)

    selected = comparison.loc[comparison["width_model_name"].eq(selected_model)]
    workflow_manifest_path = output / WORKFLOW_MANIFEST_FILENAME
    workflow_manifest = {
        "schema_version": FINAL_WORKFLOW_SCHEMA_VERSION,
        "model_version": FINAL_MODEL_VERSION,
        "code_revision": _git_revision(),
        "production_authorized": False,
        "selection_frozen": True,
        "selected_model": selected_model,
        "selection_rule": result.summary["provisional_selection_reason"],
        "selected_threshold": float(
            result.summary["candidate_models"][selected_model]["recommended_threshold"]
        ),
        "frozen_feature_columns": list(FINAL_FEATURE_COLUMNS),
        "frozen_hybrid_rule": {
            "variant_id": FROZEN_HYBRID_VARIANT,
            "endpoint_window": "median valid trimmed two-bank wet samples within clamp(10% link length, 250 m, 1000 m)",
            "eligibility": "individual adjacent link length/wet-width >= 10 and endpoint windows non-overlapping",
            "incoming_endpoint": "downstream endpoint",
            "outgoing_endpoint": "upstream endpoint",
            "fallback": "whole-link wid_adj_wet",
        },
        "label_contract": {
            "resolved_review": "use adjudicated_requires_correction",
            "nonqueue": "retain original manual label",
            "ambiguous_or_problem": "exclude",
            "no_further_feedback_review": True,
        },
        "counts": {
            "source_rows": int(len(dataset)),
            "final_training_rows": int(len(final_dataset)),
            "excluded_rows": int(len(excluded)),
            "label_changes_from_adjudication": int(final_dataset["final_label_changed_by_adjudication"].sum()),
            "selected_model_old_predicted_corrections": int(selected["old_prediction"].sum()),
            "selected_model_new_predicted_corrections": int(selected["new_prediction"].sum()),
            "selected_model_decision_changes": int(selected["decision_changed"].sum()),
            "selected_model_changes_at_fixed_old_threshold": int(
                selected["changed_at_fixed_old_threshold"].sum()
            ),
            "selected_model_changes_from_final_threshold_after_refit": int(
                selected["changed_by_final_threshold_after_refit"].sum()
            ),
            "selected_model_accept_to_correct": int(selected["decision_change"].eq("accept_to_correct").sum()),
            "selected_model_correct_to_accept": int(selected["decision_change"].eq("correct_to_accept").sum()),
        },
        "correction_contract": {
            "classification_complete": True,
            "numeric_correction_complete": False,
            "reason": "Numeric routing-width changes require the separate joint upstream-continuity solver, especially when multiple outgoing links are flagged.",
        },
        "inputs": {key: str(path) for key, path in paths.items()},
        "input_sha256": {key: _sha256(path) for key, path in paths.items()},
        "frozen_review_rebase": str(frozen_review_rebase_path)
        if frozen_review_rebase_path is not None
        else None,
        "frozen_review_rebase_sha256": _sha256(frozen_review_rebase_path)
        if frozen_review_rebase_path is not None
        else None,
        "files": {
            "final_dataset": str(final_dataset_path),
            "excluded_cases": str(excluded_path),
            "final_dataset_manifest": str(dataset_manifest_path),
            "model_manifest": str(model_outputs["manifest"]),
            "prediction_comparison": str(comparison_path),
            "decision_summary": str(decision_summary_path),
            "validation_comparison": str(validation_comparison_path),
        },
    }
    workflow_manifest["file_sha256"] = {
        key: _sha256(Path(path)) for key, path in workflow_manifest["files"].items()
    }
    _atomic_write_json(workflow_manifest, workflow_manifest_path)
    return {
        "output_dir": str(output),
        "manifest": str(workflow_manifest_path),
        "selected_model": selected_model,
        "selected_threshold": workflow_manifest["selected_threshold"],
        **workflow_manifest["counts"],
        "production_authorized": False,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fit the frozen adjudicated width-QC model with conservative hybrid junction continuity."
    )
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET_PATH))
    parser.add_argument("--review", default=str(DEFAULT_REVIEW_PATH))
    parser.add_argument("--review-manifest", default=str(DEFAULT_REVIEW_MANIFEST_PATH))
    parser.add_argument("--hybrid-comparisons", default=str(DEFAULT_HYBRID_COMPARISONS_PATH))
    parser.add_argument("--hybrid-manifest", default=str(DEFAULT_HYBRID_MANIFEST_PATH))
    parser.add_argument("--original-predictions", default=str(DEFAULT_ORIGINAL_PREDICTIONS_PATH))
    parser.add_argument("--original-model-manifest", default=str(DEFAULT_ORIGINAL_MODEL_MANIFEST_PATH))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument(
        "--max-split-sensitive-false-correction-rate",
        type=float,
        default=DEFAULT_MAX_SPLIT_FALSE_CORRECTION_RATE,
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not 0 <= args.max_split_sensitive_false_correction_rate <= 1:
        raise ValueError("--max-split-sensitive-false-correction-rate must be between 0 and 1.")
    result = run_final_model_workflow(
        dataset_path=args.dataset,
        review_path=args.review,
        review_manifest_path=args.review_manifest,
        hybrid_comparisons_path=args.hybrid_comparisons,
        hybrid_manifest_path=args.hybrid_manifest,
        original_predictions_path=args.original_predictions,
        original_model_manifest_path=args.original_model_manifest,
        output_dir=args.output_dir,
        max_split_false_correction_rate=args.max_split_sensitive_false_correction_rate,
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
