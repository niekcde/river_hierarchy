from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


DEFAULT_REVIEW_ROOT = Path(__file__).resolve().parent / "output"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "output" / "width_review_model"
DEFAULT_REVIEW_CSV_NAME = "width_review_decisions.csv"
LABEL_COLUMN = "review_decision"
POSITIVE_LABEL = "bad"
NEGATIVE_LABEL = "good"
DEFAULT_FEATURE_COLUMNS = (
    "qc_mean_opposition_cos_trimmed_from_samples",
    "qc_median_opposition_cos_trimmed_from_samples",
    "us_signed_dev",
    "ds_signed_dev",
    "split_sensitive",
    "n_trimmed_qc_samples",
)


@dataclass(frozen=True)
class ReviewDataset:
    frame: pd.DataFrame
    review_csv_paths: tuple[Path, ...]


def discover_review_csvs(review_root: str | Path = DEFAULT_REVIEW_ROOT) -> list[Path]:
    root = Path(review_root).expanduser().resolve()
    return sorted(root.rglob(DEFAULT_REVIEW_CSV_NAME))


def infer_example_id_from_path(path: Path) -> str:
    parts = path.parts
    for part in parts:
        if "_indep" in part or "_compound" in part or "_sequential" in part:
            return part.split("_", 1)[0]
    return "unknown"


def load_review_dataset(review_csv_paths: list[str | Path]) -> ReviewDataset:
    frames: list[pd.DataFrame] = []
    resolved_paths: list[Path] = []
    for item in review_csv_paths:
        path = Path(item).expanduser().resolve()
        frame = pd.read_csv(path)
        frame["review_csv_path"] = str(path)
        if "example_id" not in frame.columns:
            frame["example_id"] = infer_example_id_from_path(path)
        if "variant_id" not in frame.columns:
            frame["variant_id"] = "S000_base"
        frames.append(frame)
        resolved_paths.append(path)
    if not frames:
        return ReviewDataset(frame=pd.DataFrame(), review_csv_paths=tuple())

    combined = pd.concat(frames, ignore_index=True)
    combined[LABEL_COLUMN] = combined[LABEL_COLUMN].astype("string").str.lower()
    combined = combined.loc[combined[LABEL_COLUMN].isin([NEGATIVE_LABEL, POSITIVE_LABEL])].copy()
    combined["split_sensitive"] = pd.to_numeric(combined.get("split_sensitive"), errors="coerce").fillna(0).astype(int)
    for column in DEFAULT_FEATURE_COLUMNS:
        if column == "split_sensitive":
            continue
        combined[column] = pd.to_numeric(combined.get(column), errors="coerce")
    combined["target"] = (combined[LABEL_COLUMN] == POSITIVE_LABEL).astype(int)
    return ReviewDataset(frame=combined.reset_index(drop=True), review_csv_paths=tuple(resolved_paths))


def build_training_table(review_root: str | Path = DEFAULT_REVIEW_ROOT) -> ReviewDataset:
    return load_review_dataset(discover_review_csvs(review_root))


def summarize_dataset(frame: pd.DataFrame, *, feature_columns: tuple[str, ...] = DEFAULT_FEATURE_COLUMNS) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "n_rows": int(len(frame)),
        "n_examples": int(frame["example_id"].nunique()) if "example_id" in frame.columns and not frame.empty else 0,
        "label_counts": frame[LABEL_COLUMN].value_counts().to_dict() if LABEL_COLUMN in frame.columns else {},
    }
    if frame.empty:
        summary["feature_means_by_label"] = {}
        return summary

    means = (
        frame.groupby(LABEL_COLUMN)[list(feature_columns)]
        .mean(numeric_only=True)
        .round(6)
        .replace({np.nan: None})
        .to_dict(orient="index")
    )
    summary["feature_means_by_label"] = means
    return summary


def _require_sklearn():
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.svm import LinearSVC
    except Exception as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            "scikit-learn is required to fit the width review model. "
            "Install it in the river-hierarchy-rivgraph environment or use only the dataset export."
        ) from exc
    return LogisticRegression, Pipeline, StandardScaler, LinearSVC


def fit_linear_models(
    frame: pd.DataFrame,
    *,
    feature_columns: tuple[str, ...] = DEFAULT_FEATURE_COLUMNS,
) -> dict[str, Any]:
    LogisticRegression, Pipeline, StandardScaler, LinearSVC = _require_sklearn()
    model_frame = frame.dropna(subset=list(feature_columns) + ["target"]).copy()
    if model_frame.empty:
        raise ValueError("No fully observed labeled rows are available for model fitting.")

    x = model_frame.loc[:, feature_columns].copy()
    x["split_sensitive"] = x["split_sensitive"].astype(float)
    y = model_frame["target"].astype(int).to_numpy()

    logistic = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(random_state=0, max_iter=500)),
        ]
    )
    logistic.fit(x, y)

    svm = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("model", LinearSVC(random_state=0, max_iter=5000)),
        ]
    )
    svm.fit(x, y)

    logistic_model = logistic.named_steps["model"]
    svm_model = svm.named_steps["model"]
    logistic_score = float(logistic.score(x, y))
    svm_score = float(svm.score(x, y))
    return {
        "n_training_rows": int(len(model_frame)),
        "feature_columns": list(feature_columns),
        "logistic_regression": {
            "training_accuracy": logistic_score,
            "intercept": float(logistic_model.intercept_[0]),
            "coefficients": {feature: float(value) for feature, value in zip(feature_columns, logistic_model.coef_[0])},
        },
        "linear_svm": {
            "training_accuracy": svm_score,
            "intercept": float(svm_model.intercept_[0]),
            "coefficients": {feature: float(value) for feature, value in zip(feature_columns, svm_model.coef_[0])},
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Assemble labeled width-review examples and optionally fit simple linear classifiers."
    )
    parser.add_argument("--review-root", default=str(DEFAULT_REVIEW_ROOT), help="Root under which width_review_decisions.csv files will be discovered.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Directory for the combined training CSV and optional model summary JSON.")
    parser.add_argument(
        "--fit-model",
        action="store_true",
        help="Fit a logistic regression and linear SVM on the discovered labeled examples.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    dataset = build_training_table(args.review_root)
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    training_csv = output_dir / "width_review_training_data.csv"
    dataset.frame.to_csv(training_csv, index=False)

    payload: dict[str, Any] = {
        "review_root": str(Path(args.review_root).expanduser().resolve()),
        "n_review_csvs": len(dataset.review_csv_paths),
        "review_csv_paths": [str(path) for path in dataset.review_csv_paths],
        "training_csv": str(training_csv),
        "dataset_summary": summarize_dataset(dataset.frame),
    }
    if args.fit_model:
        payload["model_summary"] = fit_linear_models(dataset.frame)

    summary_path = output_dir / "width_review_model_summary.json"
    summary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote training data to {training_csv}")
    print(f"Wrote summary to {summary_path}")


if __name__ == "__main__":
    main()
