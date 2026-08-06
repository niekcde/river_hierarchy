from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import BaseEstimator
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from network_variants.width_review_archive import _atomic_write_csv, _atomic_write_json, _sha256
from network_variants.width_review_dataset import (
    DATASET_FILENAME,
    DATASET_SCHEMA_VERSION,
    DEFAULT_OUTPUT_DIR as DEFAULT_DATASET_OUTPUT_DIR,
    MANIFEST_FILENAME as DATASET_MANIFEST_FILENAME,
)


MODEL_VERSION = "width-qc-candidates-v1"
DEFAULT_DATASET_PATH = DEFAULT_DATASET_OUTPUT_DIR / DATASET_FILENAME
DEFAULT_DATASET_MANIFEST_PATH = DEFAULT_DATASET_OUTPUT_DIR / DATASET_MANIFEST_FILENAME
DEFAULT_MODEL_OUTPUT_DIR = Path(__file__).resolve().parent / "output" / "width_qc_model"
DEFAULT_MAX_SPLIT_FALSE_CORRECTION_RATE = 0.05

FEATURE_COLUMNS = (
    "length_over_raw_wet_width",
    "split_sensitive",
    "opposition_cos_trimmed_mean",
    "opposition_cos_trimmed_median",
    "invalid_sample_fraction",
    "upstream_continuity_ratio",
    "upstream_continuity_abs_deviation",
)

PREDICTIONS_FILENAME = "width_qc_loeo_predictions_v1.csv"
PER_EXAMPLE_FILENAME = "width_qc_loeo_per_example_metrics_v1.csv"
THRESHOLDS_FILENAME = "width_qc_threshold_metrics_v1.csv"
CALIBRATION_FILENAME = "width_qc_calibration_v1.csv"
SUMMARY_FILENAME = "width_qc_model_validation_v1.json"
MANIFEST_FILENAME = "width_qc_model_manifest_v1.json"


@dataclass(frozen=True)
class ModelSpecification:
    name: str
    factory: Callable[[], Pipeline]
    role: str


@dataclass(frozen=True)
class ModelDevelopmentResult:
    predictions: pd.DataFrame
    per_example_metrics: pd.DataFrame
    threshold_metrics: pd.DataFrame
    calibration: pd.DataFrame
    summary: dict[str, Any]
    fitted_models: dict[str, Pipeline]


def _pipeline(estimator: BaseEstimator) -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", estimator),
        ]
    )


def model_specifications() -> tuple[ModelSpecification, ...]:
    return (
        ModelSpecification(
            "logistic_regression",
            lambda: _pipeline(LogisticRegression(C=1.0, max_iter=2000, random_state=0)),
            "preferred_interpretable_baseline",
        ),
        ModelSpecification(
            "linear_svm",
            lambda: _pipeline(SVC(C=1.0, kernel="linear", probability=True, random_state=0)),
            "linear_margin_comparison",
        ),
        ModelSpecification(
            "rbf_svm",
            lambda: _pipeline(SVC(C=1.0, kernel="rbf", gamma="scale", probability=True, random_state=0)),
            "exploratory_nonlinear_comparison",
        ),
    )


def load_model_dataset(dataset_path: str | Path = DEFAULT_DATASET_PATH) -> pd.DataFrame:
    path = Path(dataset_path).expanduser().resolve()
    frame = pd.read_csv(path, dtype={"example_id": "string"})
    required = {
        "example_id",
        "id_link",
        "manual_decision",
        "requires_correction",
        "split_sensitive",
        "raw_wet_width",
        *FEATURE_COLUMNS,
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Training dataset is missing required fields: {missing}")
    if frame.empty:
        raise ValueError("Training dataset is empty.")
    frame["example_id"] = frame["example_id"].astype("string").str.zfill(2)
    frame["id_link"] = pd.to_numeric(frame["id_link"], errors="raise").astype(int)
    frame["requires_correction"] = pd.to_numeric(frame["requires_correction"], errors="raise").astype(int)
    if not set(frame["requires_correction"].unique()).issubset({0, 1}):
        raise ValueError("requires_correction must contain only 0/1 values.")
    expected = frame["manual_decision"].astype("string").str.lower().map({"good": 0, "bad": 1})
    if expected.isna().any() or not np.array_equal(expected.astype(int), frame["requires_correction"]):
        raise ValueError("manual_decision and requires_correction disagree.")
    if frame.duplicated(["example_id", "id_link"]).any():
        raise ValueError("Training dataset contains duplicate example_id/id_link keys.")
    for column in FEATURE_COLUMNS:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    if frame[list(FEATURE_COLUMNS)].isna().all(axis=0).any():
        empty = frame[list(FEATURE_COLUMNS)].columns[frame[list(FEATURE_COLUMNS)].isna().all()].tolist()
        raise ValueError(f"Model features are entirely missing: {empty}")
    return frame.sort_values(["example_id", "id_link"], kind="mergesort").reset_index(drop=True)


def _safe_auc(y_true: np.ndarray, probability: np.ndarray) -> float | None:
    return float(roc_auc_score(y_true, probability)) if len(np.unique(y_true)) == 2 else None


def _safe_average_precision(y_true: np.ndarray, probability: np.ndarray) -> float | None:
    return float(average_precision_score(y_true, probability)) if np.any(y_true == 1) else None


def classification_metrics(
    y_true: np.ndarray,
    probability: np.ndarray,
    *,
    threshold: float,
    split_sensitive: np.ndarray,
) -> dict[str, Any]:
    prediction = probability >= float(threshold)
    tn, fp, fn, tp = confusion_matrix(y_true, prediction.astype(int), labels=[0, 1]).ravel()
    good_count = int(tn + fp)
    split_good = (y_true == 0) & split_sensitive
    split_false_corrections = int(np.count_nonzero(prediction & split_good))
    split_good_count = int(np.count_nonzero(split_good))
    split_rows = split_sensitive
    split_y = y_true[split_rows]
    split_probability = probability[split_rows]
    split_prediction = prediction[split_rows]
    split_tn, split_fp, split_fn, split_tp = confusion_matrix(
        split_y, split_prediction.astype(int), labels=[0, 1]
    ).ravel()
    return {
        "threshold": float(threshold),
        "rows": int(len(y_true)),
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
        "precision": float(precision_score(y_true, prediction, zero_division=0)),
        "recall": float(recall_score(y_true, prediction, zero_division=0)),
        "false_correction_rate": float(fp / good_count) if good_count else None,
        "split_sensitive_good_count": split_good_count,
        "split_sensitive_false_corrections": split_false_corrections,
        "split_sensitive_false_correction_rate": (
            float(split_false_corrections / split_good_count) if split_good_count else None
        ),
        "split_sensitive_true_negative": int(split_tn),
        "split_sensitive_false_positive": int(split_fp),
        "split_sensitive_false_negative": int(split_fn),
        "split_sensitive_true_positive": int(split_tp),
        "split_sensitive_precision": (
            float(precision_score(split_y, split_prediction, zero_division=0)) if len(split_y) else None
        ),
        "split_sensitive_recall": (
            float(recall_score(split_y, split_prediction, zero_division=0)) if len(split_y) else None
        ),
        "roc_auc": _safe_auc(y_true, probability),
        "average_precision": _safe_average_precision(y_true, probability),
        "brier_score": float(brier_score_loss(y_true, probability)),
        "split_sensitive_roc_auc": _safe_auc(split_y, split_probability) if len(split_y) else None,
        "split_sensitive_average_precision": (
            _safe_average_precision(split_y, split_probability) if len(split_y) else None
        ),
    }


def threshold_tradeoff_table(
    y_true: np.ndarray,
    probability: np.ndarray,
    split_sensitive: np.ndarray,
    *,
    model_name: str,
) -> pd.DataFrame:
    thresholds = sorted(set(np.round(np.linspace(0.05, 0.95, 19), 4).tolist() + [0.975, 0.99, 1.0]))
    rows = []
    for threshold in thresholds:
        metrics = classification_metrics(
            y_true,
            probability,
            threshold=threshold,
            split_sensitive=split_sensitive,
        )
        rows.append({"model_name": model_name, **metrics})
    return pd.DataFrame(rows)


def select_conservative_threshold(
    threshold_table: pd.DataFrame,
    *,
    max_split_false_correction_rate: float,
) -> tuple[float, dict[str, Any]]:
    risk = pd.to_numeric(threshold_table["split_sensitive_false_correction_rate"], errors="coerce")
    eligible = threshold_table.loc[risk.fillna(0.0).le(float(max_split_false_correction_rate))].copy()
    if eligible.empty:
        selected = threshold_table.sort_values("threshold", ascending=False, kind="mergesort").iloc[0]
        constraint_met = False
    else:
        eligible = eligible.sort_values(
            ["recall", "precision", "threshold"],
            ascending=[False, False, False],
            kind="mergesort",
        )
        selected = eligible.iloc[0]
        constraint_met = True
    return float(selected["threshold"]), {
        "constraint_met": constraint_met,
        "max_split_sensitive_false_correction_rate": float(max_split_false_correction_rate),
        "selection_rule": (
            "maximize recall, then precision, then threshold among thresholds whose false-correction "
            "rate on manually good split-sensitive links does not exceed the configured maximum"
        ),
    }


def calibration_table(y_true: np.ndarray, probability: np.ndarray, *, model_name: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    edges = np.linspace(0.0, 1.0, 11)
    bins = np.minimum(np.digitize(probability, edges[1:-1], right=False), 9)
    for bin_index in range(10):
        selected = bins == bin_index
        if not np.any(selected):
            rows.append(
                {
                    "model_name": model_name,
                    "probability_bin": bin_index,
                    "lower": float(edges[bin_index]),
                    "upper": float(edges[bin_index + 1]),
                    "rows": 0,
                    "mean_probability": np.nan,
                    "observed_correction_rate": np.nan,
                }
            )
            continue
        rows.append(
            {
                "model_name": model_name,
                "probability_bin": bin_index,
                "lower": float(edges[bin_index]),
                "upper": float(edges[bin_index + 1]),
                "rows": int(np.count_nonzero(selected)),
                "mean_probability": float(np.mean(probability[selected])),
                "observed_correction_rate": float(np.mean(y_true[selected])),
            }
        )
    return pd.DataFrame(rows)


def _predict_probability(model: Pipeline, x: pd.DataFrame) -> np.ndarray:
    classes = model.named_steps["model"].classes_
    positive_index = int(np.flatnonzero(classes == 1)[0])
    return np.asarray(model.predict_proba(x)[:, positive_index], dtype=float)


def full_fit_diagnostics(
    model: Pipeline,
    *,
    feature_columns: Sequence[str] = FEATURE_COLUMNS,
) -> dict[str, Any]:
    imputer = model.named_steps["imputer"]
    scaler = model.named_steps["scaler"]
    estimator = model.named_steps["model"]
    diagnostics: dict[str, Any] = {
        "imputation_medians": {
            feature: float(value) for feature, value in zip(feature_columns, imputer.statistics_)
        },
        "standardization_means": {
            feature: float(value) for feature, value in zip(feature_columns, scaler.mean_)
        },
        "standardization_scales": {
            feature: float(value) for feature, value in zip(feature_columns, scaler.scale_)
        },
        "estimator_parameters": {
            key: value
            for key, value in estimator.get_params(deep=False).items()
            if value is None or isinstance(value, (str, int, float, bool))
        },
    }
    if hasattr(estimator, "coef_"):
        coefficients = np.asarray(estimator.coef_[0], dtype=float)
        diagnostics["standardized_coefficients"] = {
            feature: float(value) for feature, value in zip(feature_columns, coefficients)
        }
        diagnostics["intercept"] = float(np.asarray(estimator.intercept_).ravel()[0])
        if isinstance(estimator, LogisticRegression):
            diagnostics["standardized_odds_ratios"] = {
                feature: float(np.exp(value)) for feature, value in zip(feature_columns, coefficients)
            }
    return diagnostics


def run_leave_one_example_out(
    frame: pd.DataFrame,
    *,
    specifications: Sequence[ModelSpecification] | None = None,
    max_split_false_correction_rate: float = DEFAULT_MAX_SPLIT_FALSE_CORRECTION_RATE,
    feature_columns: Sequence[str] = FEATURE_COLUMNS,
    model_version: str = MODEL_VERSION,
) -> ModelDevelopmentResult:
    specs = tuple(specifications or model_specifications())
    features = tuple(feature_columns)
    if not features or len(set(features)) != len(features):
        raise ValueError("Model feature columns must be non-empty and unique.")
    missing_features = sorted(set(features).difference(frame.columns))
    if missing_features:
        raise ValueError(f"Model frame is missing feature columns: {missing_features}")
    groups = frame["example_id"].astype(str)
    examples = sorted(groups.unique())
    if len(examples) < 3:
        raise ValueError("Leave-one-example-out validation requires at least three examples.")
    x = frame.loc[:, features]
    y = frame["requires_correction"].astype(int).to_numpy()
    split_sensitive = frame["split_sensitive"].fillna(0).astype(bool).to_numpy()

    prediction_frames: list[pd.DataFrame] = []
    threshold_frames: list[pd.DataFrame] = []
    calibration_frames: list[pd.DataFrame] = []
    per_example_rows: list[dict[str, Any]] = []
    aggregate_summaries: dict[str, Any] = {}
    fitted_models: dict[str, Pipeline] = {}

    for spec in specs:
        probabilities = np.full(len(frame), np.nan, dtype=float)
        for held_example in examples:
            test_mask = groups.eq(held_example).to_numpy()
            train_mask = ~test_mask
            if len(np.unique(y[train_mask])) != 2:
                raise ValueError(f"Training fold excluding {held_example} does not contain both labels.")
            model = spec.factory()
            model.fit(x.loc[train_mask], y[train_mask])
            probabilities[test_mask] = _predict_probability(model, x.loc[test_mask])
        if not np.isfinite(probabilities).all():
            raise RuntimeError(f"Model {spec.name} did not produce one held-out probability per row.")

        thresholds = threshold_tradeoff_table(
            y, probabilities, split_sensitive, model_name=spec.name
        )
        recommended_threshold, threshold_rule = select_conservative_threshold(
            thresholds,
            max_split_false_correction_rate=max_split_false_correction_rate,
        )
        prediction = probabilities >= recommended_threshold
        routing_impact = (
            frame["upstream_continuity_abs_deviation"].fillna(0).to_numpy(dtype=float)
            * frame["raw_wet_width"].fillna(0).to_numpy(dtype=float)
            * np.where(split_sensitive, 2.0, 1.0)
        )
        model_predictions = frame[
            ["example_id", "id_link", "manual_decision", "requires_correction", "split_sensitive"]
        ].copy()
        model_predictions["width_model_probability"] = probabilities
        model_predictions["width_model_prediction"] = prediction.astype(int)
        model_predictions["width_model_threshold"] = recommended_threshold
        model_predictions["width_model_name"] = spec.name
        model_predictions["width_model_version"] = model_version
        model_predictions["manual_model_disagreement"] = prediction.astype(int) != y
        model_predictions["routing_impact_score"] = routing_impact
        prediction_frames.append(model_predictions)
        threshold_frames.append(thresholds)
        calibration_frames.append(calibration_table(y, probabilities, model_name=spec.name))

        aggregate = classification_metrics(
            y,
            probabilities,
            threshold=recommended_threshold,
            split_sensitive=split_sensitive,
        )
        aggregate_summaries[spec.name] = {
            "role": spec.role,
            "recommended_threshold": recommended_threshold,
            "threshold_rule": threshold_rule,
            "metrics": aggregate,
        }
        for held_example in examples:
            mask = groups.eq(held_example).to_numpy()
            metrics = classification_metrics(
                y[mask],
                probabilities[mask],
                threshold=recommended_threshold,
                split_sensitive=split_sensitive[mask],
            )
            per_example_rows.append(
                {"model_name": spec.name, "held_out_example": held_example, **metrics}
            )

        fitted = spec.factory()
        fitted.fit(x, y)
        fitted_models[spec.name] = fitted
        aggregate_summaries[spec.name]["full_fit_diagnostics"] = full_fit_diagnostics(
            fitted,
            feature_columns=features,
        )

    predictions = pd.concat(prediction_frames, ignore_index=True).sort_values(
        ["width_model_name", "example_id", "id_link"], kind="mergesort"
    ).reset_index(drop=True)
    per_example = pd.DataFrame(per_example_rows).sort_values(
        ["model_name", "held_out_example"], kind="mergesort"
    ).reset_index(drop=True)
    threshold_table = pd.concat(threshold_frames, ignore_index=True)
    calibration = pd.concat(calibration_frames, ignore_index=True)

    best_ap = max(
        float(details["metrics"]["average_precision"])
        for details in aggregate_summaries.values()
        if details["metrics"]["average_precision"] is not None
    )
    logistic_ap = float(aggregate_summaries["logistic_regression"]["metrics"]["average_precision"])
    if logistic_ap >= best_ap - 0.02:
        provisional_model = "logistic_regression"
        selection_reason = "preferred interpretable baseline is within 0.02 average precision of the best candidate"
    else:
        provisional_model = max(
            aggregate_summaries,
            key=lambda name: float(aggregate_summaries[name]["metrics"]["average_precision"]),
        )
        selection_reason = "candidate has materially higher leave-one-example-out average precision"

    selected_predictions = predictions.loc[predictions["width_model_name"] == provisional_model]
    false_positives = selected_predictions.loc[
        (selected_predictions["requires_correction"] == 0)
        & (selected_predictions["width_model_prediction"] == 1)
    ].nlargest(20, "routing_impact_score")
    false_negatives = selected_predictions.loc[
        (selected_predictions["requires_correction"] == 1)
        & (selected_predictions["width_model_prediction"] == 0)
    ].nlargest(20, "routing_impact_score")

    summary = {
        "model_version": model_version,
        "validation_design": "leave_one_complete_example_out",
        "preprocessing_contract": "median imputation and standard scaling fitted inside every training fold",
        "feature_columns": list(features),
        "example_count": int(len(examples)),
        "row_count": int(len(frame)),
        "candidate_models": aggregate_summaries,
        "provisional_candidate_model": provisional_model,
        "provisional_selection_reason": selection_reason,
        "production_authorized": False,
        "approval_note": "Model and threshold require disagreement review before production use.",
        "highest_routing_impact_false_positives": false_positives.to_dict("records"),
        "highest_routing_impact_false_negatives": false_negatives.to_dict("records"),
    }
    return ModelDevelopmentResult(
        predictions=predictions,
        per_example_metrics=per_example,
        threshold_metrics=threshold_table,
        calibration=calibration,
        summary=summary,
        fitted_models=fitted_models,
    )


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


def _atomic_joblib_dump(payload: object, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
    try:
        joblib.dump(payload, temporary)
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


def write_model_development_outputs(
    result: ModelDevelopmentResult,
    *,
    dataset_path: str | Path,
    output_dir: str | Path = DEFAULT_MODEL_OUTPUT_DIR,
    dataset_manifest_path: str | Path | None = DEFAULT_DATASET_MANIFEST_PATH,
    feature_columns: Sequence[str] = FEATURE_COLUMNS,
    model_version: str = MODEL_VERSION,
) -> dict[str, Any]:
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    dataset = Path(dataset_path).expanduser().resolve()
    dataset_hash = _sha256(dataset)
    features = tuple(feature_columns)
    if list(result.summary.get("feature_columns", [])) != list(features):
        raise ValueError("Requested output feature columns disagree with the fitted model result.")
    if result.summary.get("model_version") != model_version:
        raise ValueError("Requested output model version disagrees with the fitted model result.")
    dataset_manifest: dict[str, Any] = {}
    if dataset_manifest_path is not None:
        candidate = Path(dataset_manifest_path).expanduser().resolve()
        if candidate.exists():
            dataset_manifest = json.loads(candidate.read_text(encoding="utf-8"))
            expected_hash = dataset_manifest.get("dataset_sha256")
            if expected_hash and expected_hash != dataset_hash:
                raise ValueError("Training dataset hash does not match its Phase 4 manifest.")

    paths = {
        "predictions": output / PREDICTIONS_FILENAME,
        "per_example_metrics": output / PER_EXAMPLE_FILENAME,
        "threshold_metrics": output / THRESHOLDS_FILENAME,
        "calibration": output / CALIBRATION_FILENAME,
        "summary": output / SUMMARY_FILENAME,
        "manifest": output / MANIFEST_FILENAME,
    }
    _atomic_write_csv(result.predictions, paths["predictions"])
    _atomic_write_csv(result.per_example_metrics, paths["per_example_metrics"])
    _atomic_write_csv(result.threshold_metrics, paths["threshold_metrics"])
    _atomic_write_csv(result.calibration, paths["calibration"])

    summary = dict(result.summary)
    summary["dataset_schema_version"] = DATASET_SCHEMA_VERSION
    summary["dataset_id"] = dataset_manifest.get("dataset_id")
    summary["dataset_sha256"] = dataset_hash
    summary["code_revision"] = _git_revision()
    summary["software_versions"] = {
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit_learn": sklearn.__version__,
        "joblib": joblib.__version__,
    }
    _atomic_write_json(summary, paths["summary"])

    bundle_paths: dict[str, Path] = {}
    for name, model in result.fitted_models.items():
        bundle_path = output / f"width_qc_{name}_candidate_v1.joblib"
        bundle = {
            "pipeline": model,
            "model_name": name,
            "model_version": model_version,
            "production_authorized": False,
            "feature_columns": list(features),
            "threshold": float(result.summary["candidate_models"][name]["recommended_threshold"]),
            "dataset_sha256": dataset_hash,
            "dataset_id": dataset_manifest.get("dataset_id"),
            "validation_summary": result.summary["candidate_models"][name],
            "full_fit_diagnostics": result.summary["candidate_models"][name]["full_fit_diagnostics"],
            "model_parameters": model.named_steps["model"].get_params(deep=False),
            "preprocessing_parameters": {
                "imputer": model.named_steps["imputer"].get_params(deep=False),
                "scaler": model.named_steps["scaler"].get_params(deep=False),
            },
            "code_revision": summary["code_revision"],
            "software_versions": summary["software_versions"],
        }
        _atomic_joblib_dump(bundle, bundle_path)
        bundle_paths[name] = bundle_path

    manifest = {
        "model_version": model_version,
        "production_authorized": False,
        "dataset_path": str(dataset),
        "dataset_sha256": dataset_hash,
        "dataset_id": dataset_manifest.get("dataset_id"),
        "feature_columns": list(features),
        "provisional_candidate_model": result.summary["provisional_candidate_model"],
        "files": {key: str(path) for key, path in paths.items() if key != "manifest"},
        "file_sha256": {key: _sha256(path) for key, path in paths.items() if key != "manifest"},
        "candidate_bundles": {name: str(path) for name, path in bundle_paths.items()},
        "candidate_bundle_sha256": {name: _sha256(path) for name, path in bundle_paths.items()},
    }
    _atomic_write_json(manifest, paths["manifest"])
    return {
        "output_dir": str(output),
        "manifest": str(paths["manifest"]),
        "summary": str(paths["summary"]),
        "provisional_candidate_model": result.summary["provisional_candidate_model"],
        "production_authorized": False,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare candidate supervised width-QC classifiers with leave-one-example-out validation."
    )
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET_PATH))
    parser.add_argument("--dataset-manifest", default=str(DEFAULT_DATASET_MANIFEST_PATH))
    parser.add_argument("--output-dir", default=str(DEFAULT_MODEL_OUTPUT_DIR))
    parser.add_argument(
        "--max-split-sensitive-false-correction-rate",
        type=float,
        default=DEFAULT_MAX_SPLIT_FALSE_CORRECTION_RATE,
        help="Maximum provisional false-correction rate among manually good split-sensitive links.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not 0.0 <= args.max_split_sensitive_false_correction_rate <= 1.0:
        raise ValueError("--max-split-sensitive-false-correction-rate must be between 0 and 1.")
    frame = load_model_dataset(args.dataset)
    result = run_leave_one_example_out(
        frame,
        max_split_false_correction_rate=args.max_split_sensitive_false_correction_rate,
    )
    outputs = write_model_development_outputs(
        result,
        dataset_path=args.dataset,
        output_dir=args.output_dir,
        dataset_manifest_path=args.dataset_manifest,
    )
    print(json.dumps(outputs, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
