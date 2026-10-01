#!/usr/bin/env python3
"""Screen geometry-feature and regression-model combinations with grouped CV.

This is an exploratory model-selection grid. Every candidate is evaluated on
the same complete-case, leave-one-example-out cohort; RAPID response remains
the target and is never a predictor.
"""

from __future__ import annotations

import argparse
import itertools
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor

from train_rapid_impact_classifier import Config, PRIMARY_Y_METRIC, load_and_filter
from rapid_tools.response_diagnostics import response_scores


MODEL_VERSION = "rapid-response-grid-v1"
RANDOM_STATE = 20260825
CANDIDATE_FEATURES = (
    "len_fraction_width_weighted_log_lambda",
    "along_network_frac",
    "length_fraction",
    "width_fraction",
    "k_ratio_to_matched_base",
    "equivalent_length",
    "elongation",
    "topologic_complexity_score",
    "effective_n_paths_width",
    "dominant_width_fraction",
    "path_length_cv",
    "branching_density_by_length",
)
PEAK_MAGNITUDE_TARGET = "peak_discharge_pct_shift_vs_base"
TOPOLOGY_FEATURES = CANDIDATE_FEATURES[5:]


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[2]
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--example-ids", nargs="+", default=["02", "03", "05", "11", "13", "15", "21", "56"], metavar="ID")
    parser.add_argument("--results-dir", type=Path, default=root / "RAPID" / "classification" / "results" / f"{MODEL_VERSION}-{stamp}")
    return parser.parse_args()


def feature_sets() -> list[tuple[str, ...]]:
    # All one- through four-variable mechanisms, plus the complete decomposed
    # mechanism set. The four-variable search tests whether complementary
    # topology descriptors add information beyond the leading three-variable
    # candidates without relying only on an opaque all-feature model.
    sets = [combo for size in (1, 2, 3, 4) for combo in itertools.combinations(CANDIDATE_FEATURES, size)]
    return [*sets, CANDIDATE_FEATURES]


def model_grid() -> dict[str, object]:
    models: dict[str, object] = {}
    for alpha in (0.01, 0.1, 1.0, 10.0):
        models[f"ridge_alpha_{alpha:g}"] = Pipeline([("scale", StandardScaler()), ("model", Ridge(alpha=alpha))])
    for alpha, l1_ratio in ((0.01, 0.2), (0.01, 0.8), (0.1, 0.5)):
        models[f"elasticnet_a{alpha:g}_l1{l1_ratio:g}"] = Pipeline(
            [("scale", StandardScaler()), ("model", ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=10000, random_state=RANDOM_STATE))]
        )
    for depth, leaf in ((2, 8), (3, 12)):
        models[f"tree_d{depth}_leaf{leaf}"] = DecisionTreeRegressor(max_depth=depth, min_samples_leaf=leaf, random_state=RANDOM_STATE)
    models["random_forest_d4_leaf8"] = RandomForestRegressor(n_estimators=300, max_depth=4, min_samples_leaf=8, random_state=RANDOM_STATE, n_jobs=1)
    models["extra_trees_d4_leaf8"] = ExtraTreesRegressor(n_estimators=300, max_depth=4, min_samples_leaf=8, random_state=RANDOM_STATE, n_jobs=1)
    models["hist_gradient_boosting"] = HistGradientBoostingRegressor(learning_rate=0.05, max_leaf_nodes=7, l2_regularization=1.0, random_state=RANDOM_STATE)
    return models


def add_peak_magnitude_target(frame: pd.DataFrame, output_root: Path) -> pd.DataFrame:
    """Attach 100 * (candidate peak / mode-specific base peak - 1)."""
    rows: list[pd.DataFrame] = []
    for example_id in frame["example_id"].astype(str).unique():
        for mode, suffix in (("indep", "indep__anchored"), ("compound", "compound_bubbles__anchored")):
            path = output_root / f"{example_id}_{suffix}" / "rapid_run_registry.csv"
            registry = pd.read_csv(path)
            base = registry.loc[registry["state_id"].astype(str).eq("S000_base"), "peak_discharge_cms"]
            if base.empty:
                continue
            base_peak = pd.to_numeric(base.iloc[0], errors="coerce")
            part = registry.loc[registry["state_id"].astype(str).ne("S000_base"), ["state_id", "peak_discharge_cms"]].copy()
            part["peak_discharge_cms"] = pd.to_numeric(part["peak_discharge_cms"], errors="coerce")
            part[PEAK_MAGNITUDE_TARGET] = 100.0 * (part["peak_discharge_cms"] / base_peak - 1.0)
            part["example_id"] = example_id
            part["mode"] = mode
            rows.append(part[["example_id", "mode", "state_id", PEAK_MAGNITUDE_TARGET]])
    magnitude = pd.concat(rows, ignore_index=True)
    return frame.merge(magnitude, on=["example_id", "mode", "state_id"], how="left")


def _weighted_member_summary(unit_metrics: pd.DataFrame, member_ids: str) -> dict[str, float]:
    ids = pd.to_numeric(pd.Series(str(member_ids).split(",")), errors="coerce").dropna().astype(int)
    members = unit_metrics.loc[unit_metrics["unit_id"].isin(ids)].copy()
    weights = pd.to_numeric(members["equivalent_width"], errors="coerce").clip(lower=0)
    result: dict[str, float] = {}
    for feature in TOPOLOGY_FEATURES:
        values = pd.to_numeric(members[feature], errors="coerce")
        valid = values.notna() & weights.notna() & (weights > 0)
        result[feature] = float(np.average(values[valid], weights=weights[valid])) if valid.any() else np.nan
    return result


def add_topology_features(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Attach direct unit metrics or width-weighted member-unit bubble aggregates."""
    assembled: list[pd.DataFrame] = []
    provenance_rows: list[dict[str, object]] = []
    for experiment_dir, part in frame.groupby("experiment_dir", sort=False):
        exp_path = Path(experiment_dir)
        unit_metrics = pd.read_csv(exp_path / "states" / "S000_base" / "hierarchy" / "unit_metrics.csv")
        unit_metrics["unit_id"] = pd.to_numeric(unit_metrics["unit_id"], errors="coerce").astype("Int64")
        missing = set(TOPOLOGY_FEATURES).difference(unit_metrics.columns)
        if missing:
            raise ValueError(f"{exp_path} unit metrics missing requested fields: {sorted(missing)}")
        unit_lookup = unit_metrics.set_index("unit_id")[list(TOPOLOGY_FEATURES)]
        direct = part.loc[part["mode"].eq("indep")].copy()
        if not direct.empty:
            direct["target_id"] = pd.to_numeric(direct["target_id"], errors="coerce").astype("Int64")
            direct = direct.join(unit_lookup, on="target_id")
            direct["topology_metric_source"] = "direct_base_unit"
            assembled.append(direct)
        compound = part.loc[part["mode"].eq("compound")].copy()
        if not compound.empty:
            bubbles = pd.read_csv(exp_path / "states" / "S000_base" / "hierarchy" / "bubble_equivalent_metrics.csv")
            bubble_metrics: list[dict[str, object]] = []
            for row in bubbles.itertuples(index=False):
                summary = _weighted_member_summary(unit_metrics, row.member_unit_ids)
                bubble_metrics.append({"target_id": int(row.compound_bubble_id), **summary})
            bubble_frame = pd.DataFrame(bubble_metrics)
            compound["target_id"] = pd.to_numeric(compound["target_id"], errors="coerce").astype("Int64")
            compound = compound.merge(bubble_frame, on="target_id", how="left")
            compound["topology_metric_source"] = "equivalent_width_weighted_member_unit_mean"
            assembled.append(compound)
        provenance_rows.append({"experiment_dir": str(exp_path), "independent_source": "direct_base_unit", "compound_source": "equivalent_width_weighted_member_unit_mean"})
    return pd.concat(assembled, ignore_index=True), pd.DataFrame(provenance_rows)


def verify_domain_aligned_exposure(frame: pd.DataFrame) -> pd.DataFrame:
    """Fail fast unless the shared geometry frame used the revised exposure rule."""
    required = {
        "width_fraction",
        "width_fraction_of_total",
        "width_fraction_of_total_raw",
        "exposure_fraction_source",
        "length_fraction",
        "neg_log10_lambda",
        "len_fraction_width_weighted_log_lambda",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(
            "The geometry frame does not provide the domain-aligned exposure fields "
            f"required for this search: {missing}. Restart with the updated rapid_tools code."
        )
    exposure = pd.to_numeric(frame["width_fraction"], errors="coerce").to_numpy(dtype=float)
    aligned = pd.to_numeric(frame["width_fraction_of_total"], errors="coerce").to_numpy(dtype=float)
    expected_predictor = (
        pd.to_numeric(frame["length_fraction"], errors="coerce")
        * pd.to_numeric(frame["neg_log10_lambda"], errors="coerce")
        * pd.to_numeric(frame["width_fraction"], errors="coerce")
    ).to_numpy(dtype=float)
    observed_predictor = pd.to_numeric(frame["len_fraction_width_weighted_log_lambda"], errors="coerce").to_numpy(dtype=float)
    if not np.allclose(exposure, aligned, equal_nan=True) or not np.allclose(expected_predictor, observed_predictor, equal_nan=True):
        raise ValueError(
            "Exposure consistency check failed: width_fraction or "
            "len_fraction_width_weighted_log_lambda does not match the domain-aligned definition."
        )
    audit = frame.assign(
        _raw=pd.to_numeric(frame["width_fraction_of_total_raw"], errors="coerce"),
        _exposure=pd.to_numeric(frame["width_fraction"], errors="coerce"),
    ).groupby("exposure_fraction_source", dropna=False).agg(
        states=("state_id", "size"),
        raw_width_share_mean=("_raw", "mean"),
        exposure_factor_mean=("_exposure", "mean"),
    ).reset_index()
    audit["states_changed_from_raw"] = [
        int(((group["_raw"] - group["_exposure"]).abs() > 1e-10).sum())
        for _, group in frame.assign(
            _raw=pd.to_numeric(frame["width_fraction_of_total_raw"], errors="coerce"),
            _exposure=pd.to_numeric(frame["width_fraction"], errors="coerce"),
        ).groupby("exposure_fraction_source", dropna=False)
    ]
    return audit


def evaluate(frame: pd.DataFrame, target: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    y = frame[target].to_numpy(dtype=float)
    groups = frame["example_id"].astype(str).to_numpy()
    score_rows: list[dict[str, object]] = []
    prediction_frames: list[pd.DataFrame] = []
    for features in feature_sets():
        for model_name, prototype in model_grid().items():
            prediction = np.empty(len(frame), dtype=float)
            for held_out in sorted(np.unique(groups)):
                test = groups == held_out
                model = clone(prototype)
                model.fit(frame.loc[~test, features], y[~test])
                prediction[test] = model.predict(frame.loc[test, features])
            score_rows.append({
                "target": target,
                "model": model_name,
                "features": " + ".join(features),
                "n_features": len(features),
                "mae_pct_points": mean_absolute_error(y, prediction),
                "rmse_pct_points": mean_squared_error(y, prediction) ** 0.5,
                **response_scores(y, prediction),
            })
            prediction_frames.append(pd.DataFrame({
                "target": target, "model": model_name, "features": " + ".join(features), "example_id": groups,
                "state_id": frame["state_id"].astype(str).to_numpy(), "observed": y, "predicted": prediction,
            }))
    return pd.DataFrame(score_rows).sort_values(
        ["mae_pct_points", "n_features", "rmse_pct_points"]
    ), pd.concat(prediction_frames, ignore_index=True)


def main() -> None:
    args = parse_args()
    results_dir = args.results_dir.resolve()
    if results_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing result directory: {results_dir}")
    results_dir.mkdir(parents=True)
    config = Config(example_ids=tuple(str(value).zfill(2) for value in args.example_ids))
    frame, audit = load_and_filter(config, args.output_root.resolve())
    exposure_audit = verify_domain_aligned_exposure(frame)
    frame = add_peak_magnitude_target(frame, args.output_root.resolve())
    frame, topology_provenance = add_topology_features(frame)
    targets = (PRIMARY_Y_METRIC, PEAK_MAGNITUDE_TARGET)
    frame = frame.dropna(subset=[*CANDIDATE_FEATURES, *targets]).copy()
    if frame["example_id"].nunique() < 5:
        raise ValueError("Feature-search cohort has fewer than five usable example IDs.")
    score_frames: list[pd.DataFrame] = []
    prediction_frames: list[pd.DataFrame] = []
    for target in targets:
        scores_for_target, predictions_for_target = evaluate(frame, target)
        score_frames.append(scores_for_target)
        prediction_frames.append(predictions_for_target)
    scores = pd.concat(score_frames, ignore_index=True)
    predictions = pd.concat(prediction_frames, ignore_index=True)
    best_by_target = scores.sort_values(
        ["mae_pct_points", "n_features", "rmse_pct_points"]
    ).groupby("target", as_index=False).first()
    frozen: dict[str, object] = {}
    for row in best_by_target.itertuples(index=False):
        features = tuple(row.features.split(" + "))
        frozen[row.target] = {"model": clone(model_grid()[row.model]).fit(frame.loc[:, features], frame[row.target]), "features": features}
    scores.to_csv(results_dir / "grouped_cv_grid_scores.csv", index=False)
    predictions.to_csv(results_dir / "grouped_cv_grid_predictions.csv", index=False)
    frame.to_csv(results_dir / "common_complete_case_cohort.csv", index=False)
    topology_provenance.to_csv(results_dir / "topology_metric_provenance.csv", index=False)
    exposure_audit.to_csv(results_dir / "exposure_metric_audit.csv", index=False)
    joblib.dump({"models": frozen, "selection": "screening LOEO CV minimum MAE per target"}, results_dir / "screening_best_models.joblib")
    best_text = "\n".join(
        f"- `{row.target}`: **{row.mae_pct_points:.4f} percentage points** with `{row.model}` using `{row.features}`"
        for row in best_by_target.itertuples(index=False)
    )
    report = f"""# RAPID response-predictor grid search ({MODEL_VERSION})

The common complete-case cohort contains {len(frame)} states across {frame['example_id'].nunique()} examples after the existing exact-boundary, primary-pair, and edge-compound filters. Every candidate uses the same cohort and grouped leave-one-example-out folds.

The grid covers {len(feature_sets())} feature sets and {len(model_grid())} model settings ({len(scores)} combinations). The feature candidates are: `{', '.join(CANDIDATE_FEATURES)}`.

The domain-aligned exposure verification passed. Its per-source audit is in `exposure_metric_audit.csv`; `width_fraction` was verified equal to the shared frame's `width_fraction_of_total`, and `len_fraction_width_weighted_log_lambda` was verified to use that exposure factor.

Best screening-CV combination per target:

{best_text}

This ranking is exploratory model selection; confirm selected configurations with nested grouped CV or new examples before treating scores as final unbiased performance estimates.
"""
    (results_dir / "REPORT.md").write_text(report)
    (results_dir / "grid_config.json").write_text(json.dumps({"candidate_features": CANDIDATE_FEATURES, "models": list(model_grid()), "feature_sets": [list(item) for item in feature_sets()], "audit": audit}, indent=2) + "\n")
    print(f"Wrote {results_dir}")
    for row in best_by_target.itertuples(index=False):
        print(f"Best {row.target}: {row.model} | {row.features} | MAE={row.mae_pct_points:.4f}")


if __name__ == "__main__":
    main()
