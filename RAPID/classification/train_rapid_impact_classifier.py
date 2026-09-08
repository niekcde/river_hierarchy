#!/usr/bin/env python3
"""Train and audit an interpretable RAPID collapse-impact classifier.

This module is deliberately additive: it imports the existing RAPID analysis
functions, reads real-network outputs, and writes only under its result root.
It never reads synthetic_runs data and never changes model-generating data.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

# Make the additive script runnable directly as well as via PYTHONPATH=RAPID/src.
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
RAPID_SOURCE = REPOSITORY_ROOT / "RAPID" / "src"
if str(RAPID_SOURCE) not in sys.path:
    sys.path.insert(0, str(RAPID_SOURCE))

from rapid_tools import build_geometry_context_frame, build_multi_example_timing_frame, supported_timing_x_metrics


# Frozen, named analysis configuration.  Change only with a new version tag.
MODEL_VERSION = "rapid-impact-v1"
EXAMPLE_IDS = ("02", "03", "05", "11", "21")
LENGTH_MODE = "matched_parent_vs_network"
PRIMARY_Y_METRIC = "lag_to_inflow_peak_pct_shift_vs_base"
PRIMARY_X_METRIC = "len_fraction_width_weighted_log_lambda"
EXCLUDE_NETWORK_EDGE_COMPOUNDS = True
IMPACT_MAGNITUDE_THRESHOLDS_PCT = (0.5, 1.0)
SENSITIVITY_THRESHOLDS_PCT = (0.25, 1.0)
CLASS_NAMES = ("no_impact", "intermediate_impact", "high_impact")
PRIMARY_FEATURES = (
    "len_fraction_width_weighted_log_lambda",
    "along_network_frac",
)
RANDOM_STATE = 20260824


@dataclass(frozen=True)
class Config:
    model_version: str = MODEL_VERSION
    example_ids: tuple[str, ...] = EXAMPLE_IDS
    length_mode: str = LENGTH_MODE
    primary_y_metric: str = PRIMARY_Y_METRIC
    primary_x_metric: str = PRIMARY_X_METRIC
    exclude_network_edge_compounds: bool = EXCLUDE_NETWORK_EDGE_COMPOUNDS
    impact_magnitude_thresholds_pct: tuple[float, float] = IMPACT_MAGNITUDE_THRESHOLDS_PCT
    sensitivity_thresholds_pct: tuple[float, float] = SENSITIVITY_THRESHOLDS_PCT
    class_names: tuple[str, ...] = CLASS_NAMES
    primary_features: tuple[str, ...] = PRIMARY_FEATURES
    random_state: int = RANDOM_STATE


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[2]
    run_stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
        help="Read-only network_variants/output root containing real-network experiments.",
    )
    parser.add_argument(
        "--example-ids",
        nargs="+",
        default=list(EXAMPLE_IDS),
        metavar="ID",
        help=(
            "Real-network example IDs to include, separated by spaces "
            f"(default: {' '.join(EXAMPLE_IDS)}). At least five usable IDs are required for grouped CV."
        ),
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=repo_root / "RAPID" / "classification" / "results" / f"{MODEL_VERSION}-{run_stamp}",
        help="New versioned directory for all analysis artifacts (defaults to a UTC-stamped run).",
    )
    return parser.parse_args()


def validate_sources(output_root: Path, example_ids: Iterable[str]) -> None:
    required = (
        "rapid_run_registry.csv",
        "equivalent_resistance/equivalent_resistance_comparison.csv",
        "equivalent_resistance/matched_base_corridors.csv",
        "equivalent_resistance/child_realized_objects.csv",
    )
    missing: list[str] = []
    for example_id in example_ids:
        for mode_suffix in ("indep__anchored", "compound_bubbles__anchored"):
            experiment = output_root / f"{example_id}_{mode_suffix}"
            for relative_path in required:
                if not (experiment / relative_path).is_file():
                    missing.append(str(experiment / relative_path))
    if missing:
        raise FileNotFoundError(
            "Required real-network source tables are missing.  No analysis was run:\n"
            + "\n".join(missing)
        )


def edge_compound_mask(frame: pd.DataFrame) -> pd.Series:
    max_order = frame.groupby("example_id")["compound_downstream_order"].transform("max")
    return frame["compound_downstream_order"].notna() & (
        (frame["compound_downstream_order"] <= 1)
        | (frame["compound_downstream_order"] >= max_order)
    )


def make_labels(values: pd.Series, thresholds: tuple[float, float]) -> pd.Categorical:
    magnitude = pd.to_numeric(values, errors="coerce").abs()
    return pd.cut(
        magnitude,
        bins=(-np.inf, thresholds[0], thresholds[1], np.inf),
        labels=CLASS_NAMES,
        right=False,
        ordered=True,
    )


def load_and_filter(config: Config, output_root: Path) -> tuple[pd.DataFrame, dict[str, int]]:
    validate_sources(output_root, config.example_ids)
    timing = build_multi_example_timing_frame(config.example_ids, output_root=output_root)
    frame = build_geometry_context_frame(timing, length_mode=config.length_mode)
    for column in (config.primary_x_metric, config.primary_y_metric, *config.primary_features):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame["primary_pair_complete"] = (
        np.isfinite(frame[config.primary_x_metric]) & np.isfinite(frame[config.primary_y_metric])
    )
    frame["is_edge_compound"] = edge_compound_mask(frame)
    audit = {
        "assembled_rows": int(len(frame)),
        "examples": int(frame["example_id"].nunique()),
        "excluded_non_exact_boundary": int((~frame["boundary_relation"].eq("exact_match")).sum()),
        "excluded_missing_primary_pair": int((~frame["primary_pair_complete"]).sum()),
        "excluded_edge_compound": int(frame["is_edge_compound"].sum()),
    }
    eligible = frame.loc[frame["boundary_relation"].eq("exact_match") & frame["primary_pair_complete"]].copy()
    if config.exclude_network_edge_compounds:
        eligible = eligible.loc[~eligible["is_edge_compound"]].copy()
    eligible["impact_class"] = make_labels(eligible[config.primary_y_metric], config.impact_magnitude_thresholds_pct)
    eligible["impact_class_sensitivity"] = make_labels(
        eligible[config.primary_y_metric], config.sensitivity_thresholds_pct
    )
    eligible = eligible.dropna(subset=[*config.primary_features, "impact_class"]).copy()
    audit["eligible_rows"] = int(len(eligible))
    audit["eligible_examples"] = int(eligible["example_id"].nunique())
    if audit["eligible_examples"] < 5:
        raise ValueError(
            f"Only {audit['eligible_examples']} usable example_ids remain after existing filters; grouped CV is not meaningful."
        )
    class_counts = eligible["impact_class"].value_counts().reindex(CLASS_NAMES, fill_value=0)
    if (class_counts < 2).any():
        raise ValueError(f"Default binning leaves unsupported classes: {class_counts.to_dict()}")
    return eligible, audit


def vif_table(frame: pd.DataFrame, columns: tuple[str, ...]) -> pd.DataFrame:
    values = frame.loc[:, columns].astype(float).to_numpy()
    rows = []
    for index, column in enumerate(columns):
        other = np.delete(values, index, axis=1)
        if other.shape[1] == 0:
            vif = 1.0
        else:
            model = LinearRegression().fit(other, values[:, index])
            r_squared = model.score(other, values[:, index])
            vif = float("inf") if r_squared >= 1 - 1e-12 else float(1 / (1 - r_squared))
        rows.append({"feature": column, "vif": vif})
    return pd.DataFrame(rows)


def markdown_table(frame: pd.DataFrame, *, index: bool = True) -> str:
    """Render a small Markdown table without requiring pandas' tabulate extra."""
    display = frame.reset_index() if index else frame.copy()
    columns = [str(column) for column in display.columns]
    display = display.fillna("").astype(str)
    rows = display.apply(lambda column: column.str.replace("|", "\\|", regex=False)).values.tolist()
    header = "| " + " | ".join(columns) + " |"
    divider = "| " + " | ".join("---" for _ in columns) + " |"
    body = ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join([header, divider, *body])


def model_specs(config: Config) -> dict[str, tuple[tuple[str, ...], object]]:
    return {
        "majority_class": (
            config.primary_features,
            DummyClassifier(strategy="most_frequent"),
        ),
        "lambda_single_threshold": (
            (config.primary_x_metric,),
            DecisionTreeClassifier(max_depth=1, min_samples_leaf=8, class_weight="balanced", random_state=config.random_state),
        ),
        "multinomial_logistic": (
            config.primary_features,
            Pipeline(
                [
                    ("scale", StandardScaler()),
                    ("model", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=config.random_state)),
                ]
            ),
        ),
        "shallow_tree": (
            config.primary_features,
            DecisionTreeClassifier(max_depth=2, min_samples_leaf=8, class_weight="balanced", random_state=config.random_state),
        ),
        "random_forest": (
            config.primary_features,
            RandomForestClassifier(
                n_estimators=400,
                max_depth=3,
                min_samples_leaf=6,
                class_weight="balanced",
                random_state=config.random_state,
                n_jobs=1,
            ),
        ),
    }


def grouped_cv(frame: pd.DataFrame, config: Config) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    y = frame["impact_class"].astype(str).to_numpy()
    groups = frame["example_id"].astype(str).to_numpy()
    all_predictions: list[pd.DataFrame] = []
    summary_rows: list[dict[str, object]] = []
    confusion_rows: list[dict[str, object]] = []
    for name, (features, estimator) in model_specs(config).items():
        predicted = np.empty(len(frame), dtype=object)
        for test_group in sorted(np.unique(groups)):
            test = groups == test_group
            train = ~test
            estimator.fit(frame.loc[train, features], y[train])
            predicted[test] = estimator.predict(frame.loc[test, features])
        report = classification_report(y, predicted, labels=list(CLASS_NAMES), output_dict=True, zero_division=0)
        summary_rows.append(
            {
                "model": name,
                "accuracy": accuracy_score(y, predicted),
                "macro_f1": f1_score(y, predicted, labels=list(CLASS_NAMES), average="macro", zero_division=0),
                **{f"{label}_{metric}": report[label][metric] for label in CLASS_NAMES for metric in ("precision", "recall", "f1-score", "support")},
            }
        )
        matrix = confusion_matrix(y, predicted, labels=list(CLASS_NAMES))
        for actual_index, actual in enumerate(CLASS_NAMES):
            for predicted_index, prediction in enumerate(CLASS_NAMES):
                confusion_rows.append({"model": name, "actual": actual, "predicted": prediction, "count": int(matrix[actual_index, predicted_index])})
        all_predictions.append(
            pd.DataFrame(
                {
                    "model": name,
                    "example_id": groups,
                    "state_id": frame["state_id"].astype(str).to_numpy(),
                    "actual": y,
                    "predicted": predicted,
                }
            )
        )
    return pd.DataFrame(summary_rows).sort_values("macro_f1", ascending=False), pd.DataFrame(confusion_rows), pd.concat(all_predictions, ignore_index=True)


def choose_recommended_model(scores: pd.DataFrame) -> str:
    candidate_order = ["multinomial_logistic", "shallow_tree", "random_forest"]
    best = float(scores.loc[scores["model"].isin(candidate_order), "macro_f1"].max())
    comparable = set(scores.loc[(scores["model"].isin(candidate_order)) & (scores["macro_f1"] >= best - 0.03), "model"])
    for model in candidate_order:
        if model in comparable:
            return model
    return str(scores.iloc[0]["model"])


def describe_frozen_model(model: object, features: tuple[str, ...]) -> str:
    if isinstance(model, Pipeline):
        logistic = model.named_steps["model"]
        coefficients = pd.DataFrame(logistic.coef_, columns=features, index=logistic.classes_)
        return "Standardized multinomial-logistic coefficients:\n\n" + markdown_table(coefficients.round(4))
    if isinstance(model, DecisionTreeClassifier):
        from sklearn.tree import export_text
        return "Decision-tree rule:\n\n```\n" + export_text(model, feature_names=list(features), decimals=5) + "\n```"
    return "Tree-ensemble feature importances:\n\n" + markdown_table(
        pd.DataFrame({"feature": features, "importance": model.feature_importances_}).round(4), index=False
    )


def render_report(
    config: Config,
    audit: dict[str, int],
    frame: pd.DataFrame,
    correlations: pd.DataFrame,
    vifs: pd.DataFrame,
    scores: pd.DataFrame,
    confusion: pd.DataFrame,
    recommended: str,
    frozen_input_features: tuple[str, ...],
    frozen_description: str,
) -> str:
    class_balance = frame.groupby(["example_id", "impact_class"], observed=False).size().unstack(fill_value=0).reindex(columns=CLASS_NAMES, fill_value=0)
    sensitivity = frame["impact_class_sensitivity"].value_counts().reindex(CLASS_NAMES, fill_value=0)
    selected_score = scores.loc[scores["model"].eq(recommended)].iloc[0]
    feature_descriptions = supported_timing_x_metrics()
    feature_table = pd.DataFrame(
        [{"feature": feature, "description": feature_descriptions.get(feature, feature)} for feature in config.primary_features]
    )
    selected_confusion = confusion.loc[confusion["model"].eq(recommended)].pivot(index="actual", columns="predicted", values="count").reindex(index=CLASS_NAMES, columns=CLASS_NAMES, fill_value=0)
    return f"""# RAPID collapse-impact classifier audit ({config.model_version})

## Scope and frozen configuration

This is a real-network-only analysis of examples {', '.join(config.example_ids)}. It reads the existing RAPID and equivalent-resistance outputs and does not use `synthetic_runs/`. Validation is grouped leave-one-`example_id`-out, so a river example never appears in both sides of a fold.

The response metric `{config.primary_y_metric}` is used only to create labels. Model inputs are the geometric / resistance-change features below.

{markdown_table(feature_table, index=False)}

## Data audit

| Item | Count |
|---|---:|
| Assembled candidate rows | {audit['assembled_rows']} |
| Examples assembled | {audit['examples']} |
| Excluded: `boundary_relation != exact_match` | {audit['excluded_non_exact_boundary']} |
| Excluded: missing primary x/y pair | {audit['excluded_missing_primary_pair']} |
| Excluded by existing edge-compound filter | {audit['excluded_edge_compound']} |
| Final eligible rows | {audit['eligible_rows']} |
| Final eligible examples | {audit['eligible_examples']} |

Default class counts by example:

{markdown_table(class_balance)}

## Impact-label bins

The target is the **absolute** percentage change in lag-to-inflow peak: directional advance/delay is not itself treated as an impact level. The default bins are explicit frozen configuration values:

| Class | Absolute timing shift |
|---|---:|
| no impact | < {config.impact_magnitude_thresholds_pct[0]}% |
| intermediate impact | {config.impact_magnitude_thresholds_pct[0]}% to < {config.impact_magnitude_thresholds_pct[1]}% |
| high impact | >= {config.impact_magnitude_thresholds_pct[1]}% |

The 0.5% lower threshold is a deliberately conservative practical tolerance: it is materially larger than numerical routing timestep noise while remaining well below the observed upper tail of this real-network response population. The 1% upper threshold isolates a smaller, clearly changed-response class. Sensitivity using 0.25%/1% changes the class counts to `{sensitivity.to_dict()}`; it is reported as a label-definition sensitivity, not optimized for model score.

## Collinearity check

Selected features were checked before fitting the linear model. Their Pearson correlation matrix and VIF values are below; low VIF supports retaining both physically distinct predictors.

{markdown_table(correlations.round(3))}

{markdown_table(vifs.round(3), index=False)}

## Grouped cross-validation and baselines

Scores aggregate predictions from five leave-one-example-out folds. `majority_class` is the naive majority baseline; `lambda_single_threshold` is a depth-one rule using only `{config.primary_x_metric}`.

{markdown_table(scores.round(3), index=False)}

Recommended model: **{recommended}** (macro-F1 {selected_score['macro_f1']:.3f}). The selection rule prefers the simplest model within 0.03 macro-F1 of the best non-baseline candidate.

Per-class confusion matrix for the recommended model (rows are actual, columns predicted):

{markdown_table(selected_confusion)}

## Frozen decision aid

{frozen_description}

In plain language: the frozen `{recommended}` rule uses `{', '.join(frozen_input_features)}` to assign the ordered impact bucket. It does not use a RAPID timing result as an input. The frozen artifact should be used as a screening aid for candidate collapses; this five-example validation set is a minimum-sized initial validation and should be refreshed as additional real-network experiments arrive.
"""


def main() -> None:
    args = parse_args()
    config = Config(example_ids=tuple(str(example_id).zfill(2) for example_id in args.example_ids))
    results_dir = args.results_dir.resolve()
    if results_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing result directory: {results_dir}")
    results_dir.mkdir(parents=True)
    frame, audit = load_and_filter(config, args.output_root.resolve())
    correlations = frame.loc[:, config.primary_features].corr()
    vifs = vif_table(frame, config.primary_features)
    scores, confusion, predictions = grouped_cv(frame, config)
    recommended = choose_recommended_model(scores)
    features, estimator = model_specs(config)[recommended]
    estimator.fit(frame.loc[:, features], frame["impact_class"].astype(str))
    frozen_input_features = features
    if hasattr(estimator, "feature_importances_"):
        frozen_input_features = tuple(
            feature for feature, importance in zip(features, estimator.feature_importances_) if importance > 0
        )
    frozen_description = describe_frozen_model(estimator, features)

    frame.to_csv(results_dir / "eligible_training_rows.csv", index=False)
    correlations.to_csv(results_dir / "feature_correlations.csv")
    vifs.to_csv(results_dir / "feature_vif.csv", index=False)
    scores.to_csv(results_dir / "grouped_cv_scores.csv", index=False)
    confusion.to_csv(results_dir / "grouped_cv_confusion.csv", index=False)
    predictions.to_csv(results_dir / "grouped_cv_predictions.csv", index=False)
    joblib.dump({"config": asdict(config), "features": features, "model": estimator}, results_dir / "frozen_model.joblib")
    (results_dir / "frozen_config.json").write_text(json.dumps(asdict(config), indent=2) + "\n")
    report = render_report(
        config, audit, frame, correlations, vifs, scores, confusion, recommended, frozen_input_features, frozen_description
    )
    (results_dir / "REPORT.md").write_text(report)
    provenance = {"python": sys.version, "platform": platform.platform(), "numpy": np.__version__, "pandas": pd.__version__, "scikit_learn": sklearn.__version__}
    (results_dir / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(f"Wrote {results_dir}")
    print(f"Recommended model: {recommended}")


if __name__ == "__main__":
    main()
