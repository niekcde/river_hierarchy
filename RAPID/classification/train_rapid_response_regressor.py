#!/usr/bin/env python3
"""Predict continuous RAPID lag response from collapse geometry, with grouped CV.

The target is the signed ``lag_to_inflow_peak_pct_shift_vs_base`` response.
It is never used as an input feature.  This is a real-network-only, additive
workflow and shares the exact completeness, boundary, and edge filters used by
the classifier workflow.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor

from train_rapid_impact_classifier import Config, PRIMARY_FEATURES, PRIMARY_Y_METRIC, load_and_filter


MODEL_VERSION = "rapid-response-regression-v1"
RANDOM_STATE = 20260824


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[2]
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--example-ids", nargs="+", default=["02", "03", "05", "11", "21"], metavar="ID")
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=repo_root / "RAPID" / "classification" / "results" / f"{MODEL_VERSION}-{stamp}",
    )
    return parser.parse_args()


def model_specs() -> dict[str, object]:
    return {
        "mean_response": DummyRegressor(strategy="mean"),
        "ridge": Pipeline([("scale", StandardScaler()), ("model", Ridge(alpha=1.0))]),
        "shallow_tree": DecisionTreeRegressor(max_depth=2, min_samples_leaf=8, random_state=RANDOM_STATE),
        "random_forest": RandomForestRegressor(
            n_estimators=400, max_depth=4, min_samples_leaf=6, random_state=RANDOM_STATE, n_jobs=1
        ),
        "extra_trees": ExtraTreesRegressor(
            n_estimators=400, max_depth=4, min_samples_leaf=6, random_state=RANDOM_STATE, n_jobs=1
        ),
        "hist_gradient_boosting": HistGradientBoostingRegressor(
            learning_rate=0.05, max_leaf_nodes=7, l2_regularization=1.0, random_state=RANDOM_STATE
        ),
    }


def grouped_cv(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    y = frame[PRIMARY_Y_METRIC].astype(float).to_numpy()
    groups = frame["example_id"].astype(str).to_numpy()
    score_rows: list[dict[str, object]] = []
    prediction_rows: list[pd.DataFrame] = []
    for name, estimator in model_specs().items():
        predicted = np.empty(len(frame), dtype=float)
        for held_out in sorted(np.unique(groups)):
            test = groups == held_out
            estimator.fit(frame.loc[~test, PRIMARY_FEATURES], y[~test])
            predicted[test] = estimator.predict(frame.loc[test, PRIMARY_FEATURES])
        score_rows.append(
            {
                "model": name,
                "mae_pct_points": mean_absolute_error(y, predicted),
                "rmse_pct_points": mean_squared_error(y, predicted) ** 0.5,
                "r2": r2_score(y, predicted),
            }
        )
        prediction_rows.append(
            pd.DataFrame(
                {
                    "model": name,
                    "example_id": groups,
                    "state_id": frame["state_id"].astype(str).to_numpy(),
                    "observed_lag_shift_pct": y,
                    "predicted_lag_shift_pct": predicted,
                    "residual_pct_points": y - predicted,
                }
            )
        )
    return pd.DataFrame(score_rows).sort_values("mae_pct_points"), pd.concat(prediction_rows, ignore_index=True)


def main() -> None:
    args = parse_args()
    config = Config(example_ids=tuple(str(value).zfill(2) for value in args.example_ids))
    results_dir = args.results_dir.resolve()
    if results_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing result directory: {results_dir}")
    results_dir.mkdir(parents=True)
    frame, audit = load_and_filter(config, args.output_root.resolve())
    scores, predictions = grouped_cv(frame)
    recommended = str(scores.iloc[0]["model"])
    model = model_specs()[recommended]
    model.fit(frame.loc[:, PRIMARY_FEATURES], frame[PRIMARY_Y_METRIC].astype(float))
    frame.to_csv(results_dir / "eligible_response_rows.csv", index=False)
    scores.to_csv(results_dir / "grouped_cv_regression_scores.csv", index=False)
    predictions.to_csv(results_dir / "grouped_cv_regression_predictions.csv", index=False)
    joblib.dump(
        {"model_version": MODEL_VERSION, "target": PRIMARY_Y_METRIC, "features": PRIMARY_FEATURES, "model": model},
        results_dir / "frozen_regressor.joblib",
    )
    provenance = {"python": sys.version, "platform": platform.platform(), "scikit_learn": sklearn.__version__}
    (results_dir / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    report = f"""# RAPID continuous-response prediction ({MODEL_VERSION})

The signed target is `{PRIMARY_Y_METRIC}` in percentage points. It is not a model input. This run uses {audit['eligible_rows']} eligible rows from {audit['eligible_examples']} examples and grouped leave-one-example-out validation.

## Grouped-CV model comparison

{scores.to_csv(index=False)}

The recommended model is **{recommended}**, selected by lowest grouped-CV MAE. Inspect `grouped_cv_regression_predictions.csv` in the companion prediction notebook before treating this as a production predictor.
"""
    (results_dir / "REPORT.md").write_text(report)
    print(f"Wrote {results_dir}")
    print(f"Recommended regressor: {recommended}")


if __name__ == "__main__":
    main()
