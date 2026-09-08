#!/usr/bin/env python3
"""Run one fixed, interpretable RAPID timing-response model with grouped CV.

This is intentionally separate from the broad model-selection grid. It reruns
only the preliminary candidate selected from that screen:

    Ridge(alpha=10), using len_fraction_width_weighted_log_lambda,
    equivalent_length, and topologic_complexity_score.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from search_rapid_response_predictors import add_topology_features, verify_domain_aligned_exposure
from train_rapid_impact_classifier import Config, PRIMARY_Y_METRIC, load_and_filter


MODEL_VERSION = "rapid-response-preliminary-ridge-v1"
FEATURES = (
    "len_fraction_width_weighted_log_lambda",
    "equivalent_length",
    "topologic_complexity_score",
)
MODEL_NAME = "ridge_alpha_10"
REQUIRED_SOURCE_TABLES = (
    "rapid_run_registry.csv",
    "equivalent_resistance/equivalent_resistance_comparison.csv",
    "equivalent_resistance/matched_base_corridors.csv",
    "equivalent_resistance/child_realized_objects.csv",
)


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[2]
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--example-ids", nargs="+", required=True, metavar="ID")
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=root / "RAPID" / "classification" / "results" / f"{MODEL_VERSION}-{stamp}",
    )
    return parser.parse_args()


def prototype() -> Pipeline:
    return Pipeline([("scale", StandardScaler()), ("model", Ridge(alpha=10.0))])


def nonempty_examples(example_ids: list[str], output_root: Path) -> tuple[list[str], list[dict[str, str]]]:
    """Keep only examples whose two anchored outputs have non-empty source tables."""
    usable: list[str] = []
    skipped: list[dict[str, str]] = []
    for example_id in example_ids:
        empty_paths: list[str] = []
        for suffix in ("indep__anchored", "compound_bubbles__anchored"):
            experiment = output_root / f"{example_id}_{suffix}"
            for relative_path in REQUIRED_SOURCE_TABLES:
                source = experiment / relative_path
                if not source.is_file():
                    empty_paths.append(str(source))
                    continue
                try:
                    # A one-byte newline file is not a usable CSV even though
                    # its filesystem size is non-zero.
                    preview = pd.read_csv(source, nrows=1)
                    if preview.empty:
                        empty_paths.append(str(source))
                except pd.errors.EmptyDataError:
                    empty_paths.append(str(source))
        if empty_paths:
            skipped.append({"example_id": example_id, "reason": "missing_or_empty_required_source_table", "paths": " | ".join(empty_paths)})
        else:
            usable.append(example_id)
    return usable, skipped


def evaluate(frame: pd.DataFrame) -> pd.DataFrame:
    y = frame[PRIMARY_Y_METRIC].to_numpy(dtype=float)
    groups = frame["example_id"].astype(str).to_numpy()
    prediction = np.empty(len(frame), dtype=float)
    for held_out in sorted(np.unique(groups)):
        test = groups == held_out
        model = clone(prototype())
        model.fit(frame.loc[~test, FEATURES], y[~test])
        prediction[test] = model.predict(frame.loc[test, FEATURES])
    return pd.DataFrame(
        {
            "example_id": groups,
            "state_id": frame["state_id"].astype(str).to_numpy(),
            "mode": frame["mode"].astype(str).to_numpy(),
            "observed_lag_shift_pct": y,
            "predicted_lag_shift_pct": prediction,
            "residual_pct_points": y - prediction,
        }
    )


def main() -> None:
    args = parse_args()
    results_dir = args.results_dir.resolve()
    if results_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing results: {results_dir}")
    results_dir.mkdir(parents=True)
    requested_ids = [str(value).zfill(2) for value in args.example_ids]
    usable_ids, skipped_examples = nonempty_examples(requested_ids, args.output_root.resolve())
    if len(usable_ids) < 5:
        raise ValueError(f"Fewer than five requested examples have non-empty source tables: {usable_ids}")
    if skipped_examples:
        print("Skipping examples with missing or empty required anchored outputs:")
        for skipped in skipped_examples:
            print(f"- {skipped['example_id']}: {skipped['reason']}")
    config = Config(example_ids=tuple(usable_ids))
    frame, audit = load_and_filter(config, args.output_root.resolve())
    exposure_audit = verify_domain_aligned_exposure(frame)
    frame, topology_provenance = add_topology_features(frame)
    frame = frame.dropna(subset=[*FEATURES, PRIMARY_Y_METRIC]).copy()
    if frame["example_id"].nunique() < 5:
        raise ValueError("This preliminary grouped-CV run needs at least five usable example IDs.")
    predictions = evaluate(frame)
    score = pd.DataFrame(
        [
            {
                "model": MODEL_NAME,
                "features": " + ".join(FEATURES),
                "n_features": len(FEATURES),
                "mae_pct_points": mean_absolute_error(predictions["observed_lag_shift_pct"], predictions["predicted_lag_shift_pct"]),
                "rmse_pct_points": mean_squared_error(predictions["observed_lag_shift_pct"], predictions["predicted_lag_shift_pct"]) ** 0.5,
                "r2": r2_score(predictions["observed_lag_shift_pct"], predictions["predicted_lag_shift_pct"]),
            }
        ]
    )
    fitted = prototype().fit(frame.loc[:, FEATURES], frame[PRIMARY_Y_METRIC])
    frame.to_csv(results_dir / "common_complete_case_cohort.csv", index=False)
    predictions.to_csv(results_dir / "grouped_cv_predictions.csv", index=False)
    score.to_csv(results_dir / "grouped_cv_score.csv", index=False)
    exposure_audit.to_csv(results_dir / "exposure_metric_audit.csv", index=False)
    topology_provenance.to_csv(results_dir / "topology_metric_provenance.csv", index=False)
    pd.DataFrame(skipped_examples, columns=["example_id", "reason", "paths"]).to_csv(results_dir / "skipped_requested_examples.csv", index=False)
    joblib.dump({"model": fitted, "features": FEATURES, "target": PRIMARY_Y_METRIC, "model_name": MODEL_NAME}, results_dir / "fitted_model.joblib")
    summary = score.iloc[0]
    (results_dir / "REPORT.md").write_text(
        f"# Preliminary RAPID timing-response model\n\n"
        f"Fixed model: `{MODEL_NAME}`\n\n"
        f"Features: `{', '.join(FEATURES)}`\n\n"
        f"Target: `{PRIMARY_Y_METRIC}` (signed percentage points)\n\n"
        f"Requested examples: {', '.join(requested_ids)}.\n\n"
        f"Included examples: {', '.join(usable_ids)}.\n\n"
        f"Skipped requested examples: {', '.join(item['example_id'] for item in skipped_examples) or 'none'}.\n\n"
        f"Usable cohort: {len(frame)} states across {frame['example_id'].nunique()} examples.\n\n"
        f"Leave-one-example-out CV: MAE = {summary.mae_pct_points:.4f} percentage points; "
        f"RMSE = {summary.rmse_pct_points:.4f}; R² = {summary.r2:.4f}.\n\n"
        f"The domain-aligned exposure check passed; see `exposure_metric_audit.csv`.\n"
    )
    (results_dir / "config.json").write_text(json.dumps({"model": MODEL_NAME, "features": FEATURES, "target": PRIMARY_Y_METRIC, "requested_example_ids": requested_ids, "included_example_ids": usable_ids, "skipped_examples": skipped_examples, "data_audit": audit}, indent=2) + "\n")
    print(f"Wrote {results_dir}")
    print(score.to_string(index=False))


if __name__ == "__main__":
    main()
