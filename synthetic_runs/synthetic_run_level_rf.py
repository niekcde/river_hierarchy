from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


MERGE_KEY = ["grid_id", "network_id"]
OUTPUT_KEY = ["grid_id", "network_id"]

CANONICAL_COLUMN_ALIASES: dict[str, list[str]] = {
    "grid_id": ["grid_id"],
    "network_id": ["network_id"],
    "network_type": ["network_type", "network_type_x", "network_type_y"],
    "kb": ["kb", "kb_x", "kb_y"],
    "S_global": ["S_global", "S_global_x", "S_global_y"],
    "S_local": ["S_local", "S_local_x", "S_local_y"],
    "sinuosity": ["sinuosity", "sinuosity_x", "sinuosity_y"],
    "forcing_hours": ["forcing_hours", "forcing_hours_x", "forcing_hours_y"],
    "fall_hours": ["fall_hours", "fall_hours_x", "fall_hours_y"],
    "peak": ["peak", "peak_x", "peak_y"],
    "baseflow": ["baseflow", "baseflow_x", "baseflow_y"],
    "slope_target": ["slope_target", "slope_target_x", "slope_target_y"],
    "sinuosity_target": ["sinuosity_target", "sinuosity_target_x", "sinuosity_target_y"],
}

DEFAULT_TARGETS = [
    "fallTime_50",
    "fallTime_10",
    "fallTime_to_baseflow",
    "tau_efold_80_20",
    "time_to_peak",
    "peak_Q",
    "K_Q",
]

DESIGNED_NUMERIC_FEATURES = [
    "kb",
    "S_global",
    "S_local",
    "sinuosity",
    "forcing_hours",
    "fall_hours",
    "peak",
    "baseflow",
]

DESIGNED_CATEGORICAL_FEATURES = [
    "network_id",
    "network_type",
    "slope_target",
    "sinuosity_target",
]

EDGE_SUMMARY_AGGREGATIONS: dict[str, list[str]] = {
    "slope_local": ["mean", "std", "min", "max"],
    "sinuosity_applied": ["mean", "std", "min", "max"],
    "width": ["mean", "std", "min", "max"],
    "length": ["sum", "mean", "max"],
}

@dataclass
class RunLevelRFResult:
    target: str
    data: pd.DataFrame
    feature_columns: list[str]
    numeric_features: list[str]
    categorical_features: list[str]
    dropped_constant_features: list[str]
    metrics: dict[str, Any]
    permutation_importance: pd.DataFrame
    predictions: pd.DataFrame
    shap_importance: pd.DataFrame | None = None
    shap_values: pd.DataFrame | None = None
    model: Any | None = None

def _read_table(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    lower_name = path.name.lower()
    if lower_name.endswith((".parquet", ".pq")):
        return pd.read_parquet(path)
    if lower_name.endswith(".tsv"):
        return pd.read_csv(path, sep="\t")
    return pd.read_csv(path)


def _series_conflict_count(left: pd.Series, right: pd.Series) -> int:
    mask = left.notna() & right.notna()
    if not mask.any():
        return 0

    left_masked = left.loc[mask]
    right_masked = right.loc[mask]
    if pd.api.types.is_numeric_dtype(left_masked) and pd.api.types.is_numeric_dtype(right_masked):
        equal = np.isclose(
            left_masked.astype(float).to_numpy(),
            right_masked.astype(float).to_numpy(),
            equal_nan=True,
        )
        return int((~equal).sum())

    equal = left_masked.astype(str).to_numpy() == right_masked.astype(str).to_numpy()
    return int((~equal).sum())


def canonicalize_run_columns(
    df: pd.DataFrame,
    alias_map: dict[str, list[str]] | None = None,
    verbose: bool = True,
) -> pd.DataFrame:
    alias_map = alias_map or CANONICAL_COLUMN_ALIASES
    out = df.copy()

    for canonical, aliases in alias_map.items():
        present = [col for col in aliases if col in out.columns]
        if not present:
            continue

        source = present[0]
        if canonical not in out.columns:
            out[canonical] = out[source]

        if verbose and len(present) > 1:
            reference = out[present[0]]
            for other in present[1:]:
                n_conflicts = _series_conflict_count(reference, out[other])
                if n_conflicts:
                    print(
                        f"[warn] Column alias mismatch for '{canonical}': "
                        f"{present[0]} vs {other} differ on {n_conflicts} rows. "
                        f"Using '{present[0]}' as the canonical source."
                    )

    return out


def summarize_edge_geometry(edge_df: pd.DataFrame) -> pd.DataFrame:
    missing_keys = [col for col in MERGE_KEY if col not in edge_df.columns]
    if missing_keys:
        raise ValueError(
            f"edge_df is missing required run keys: {missing_keys}. "
            "Expected at least 'grid_id' and 'network_id'."
        )

    available_aggs = {
        col: aggregations
        for col, aggregations in EDGE_SUMMARY_AGGREGATIONS.items()
        if col in edge_df.columns
    }
    if not available_aggs:
        raise ValueError(
            "edge_df does not contain any of the supported geometry columns: "
            f"{sorted(EDGE_SUMMARY_AGGREGATIONS)}"
        )

    grouped = edge_df.groupby(MERGE_KEY, as_index=False)
    summary = grouped.size().rename(columns={"size": "edge_count"})

    agg_summary = (
        grouped.agg(available_aggs)
        .reset_index()
    )
    agg_summary.columns = [
        "_".join(str(part) for part in col if part).rstrip("_")
        if isinstance(col, tuple) else col
        for col in agg_summary.columns
    ]

    out = summary.merge(agg_summary, on=MERGE_KEY, how="left")

    for stem in ("slope_local", "sinuosity_applied", "width", "length"):
        mean_col = f"{stem}_mean"
        std_col = f"{stem}_std"
        if mean_col in out.columns and std_col in out.columns:
            denom = out[mean_col].replace(0, np.nan)
            out[f"{stem}_cv"] = out[std_col] / denom

    return out


def merge_edge_summary_features(run_df: pd.DataFrame, edge_df: pd.DataFrame) -> pd.DataFrame:
    edge_summary = summarize_edge_geometry(edge_df)
    out = run_df.merge(edge_summary, on=MERGE_KEY, how="left")

    if "slope_local_mean" in out.columns and "S_local" in out.columns:
        out["slope_local_mean_minus_S_local"] = out["slope_local_mean"] - out["S_local"]
    if "slope_local_mean" in out.columns and "S_global" in out.columns:
        out["slope_local_mean_minus_S_global"] = out["slope_local_mean"] - out["S_global"]
    if "slope_local_max" in out.columns and "S_global" in out.columns:
        denom = out["S_global"].replace(0, np.nan)
        out["slope_local_max_over_S_global"] = out["slope_local_max"] / denom
    if "sinuosity_applied_mean" in out.columns and "sinuosity" in out.columns:
        out["sinuosity_applied_mean_minus_sinuosity"] = (
            out["sinuosity_applied_mean"] - out["sinuosity"]
        )

    return out


def _realized_feature_candidates(df: pd.DataFrame) -> list[str]:
    ordered_candidates = [
        "edge_count",
        "slope_local_mean",
        "slope_local_std",
        "slope_local_min",
        "slope_local_max",
        "slope_local_cv",
        "sinuosity_applied_mean",
        "sinuosity_applied_std",
        "sinuosity_applied_min",
        "sinuosity_applied_max",
        "sinuosity_applied_cv",
        "width_mean",
        "width_std",
        "width_min",
        "width_max",
        "width_cv",
        "length_sum",
        "length_mean",
        "length_max",
        "length_cv",
        "slope_local_mean_minus_S_local",
        "slope_local_mean_minus_S_global",
        "slope_local_max_over_S_global",
        "sinuosity_applied_mean_minus_sinuosity",
    ]
    return [col for col in ordered_candidates if col in df.columns]


def _coerce_feature_types(
    df: pd.DataFrame,
    numeric_features: list[str],
    categorical_features: list[str],
) -> pd.DataFrame:
    out = df.copy()
    for col in numeric_features:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    for col in categorical_features:
        out[col] = out[col].astype("string")
    return out


def prepare_run_level_dataset(
    df_final: pd.DataFrame,
    target: str,
    edge_df: pd.DataFrame | None = None,
    feature_mode: str = "designed_plus_realized",
    include_topology: bool = True,
    extra_feature_columns: list[str] | None = None,
    include_feature_columns: list[str] | None = None,
    exclude_feature_columns: list[str] | None = None,
    verbose: bool = True,
) -> tuple[pd.DataFrame, list[str], list[str], list[str], list[str]]:
    run_df = canonicalize_run_columns(df_final, verbose=verbose)

    if target not in run_df.columns:
        raise ValueError(
            f"Target '{target}' is not present in df_final. "
            f"Available columns include: {sorted(run_df.columns.tolist())[:40]}"
        )

    if feature_mode not in {"designed", "designed_plus_realized"}:
        raise ValueError("feature_mode must be 'designed' or 'designed_plus_realized'.")

    if feature_mode == "designed_plus_realized":
        if edge_df is None:
            raise ValueError(
                "feature_mode='designed_plus_realized' requires edge_df or --edge-file "
                "so the script can compute run-level summaries of edge properties."
            )
        run_df = merge_edge_summary_features(run_df, edge_df)

    numeric_features = [col for col in DESIGNED_NUMERIC_FEATURES if col in run_df.columns]
    categorical_features = []
    if include_topology:
        categorical_features = [col for col in DESIGNED_CATEGORICAL_FEATURES if col in run_df.columns]

    if feature_mode == "designed_plus_realized":
        numeric_features.extend(_realized_feature_candidates(run_df))

    if extra_feature_columns:
        for col in extra_feature_columns:
            if col not in run_df.columns:
                raise ValueError(f"Requested extra feature column '{col}' is not present in the dataset.")
            if pd.api.types.is_numeric_dtype(run_df[col]):
                if col not in numeric_features:
                    numeric_features.append(col)
            else:
                if col not in categorical_features:
                    categorical_features.append(col)

    if include_feature_columns:
        requested_numeric: list[str] = []
        requested_categorical: list[str] = []
        for col in include_feature_columns:
            if col not in run_df.columns:
                raise ValueError(f"Requested include_feature_columns entry '{col}' is not present in the dataset.")
            if pd.api.types.is_numeric_dtype(run_df[col]):
                requested_numeric.append(col)
            else:
                requested_categorical.append(col)
        numeric_features = requested_numeric
        categorical_features = requested_categorical

    if exclude_feature_columns:
        exclude_set = set(exclude_feature_columns)
        numeric_features = [col for col in numeric_features if col not in exclude_set]
        categorical_features = [col for col in categorical_features if col not in exclude_set]

    numeric_features = list(dict.fromkeys(numeric_features))
    categorical_features = list(dict.fromkeys(categorical_features))
    feature_columns = numeric_features + categorical_features

    if not feature_columns:
        raise ValueError("No modeling features were found after applying the requested configuration.")

    output_columns = [col for col in OUTPUT_KEY if col in run_df.columns]
    model_columns = list(dict.fromkeys(output_columns + feature_columns + [target]))
    model_df = run_df[model_columns].copy()
    model_df = _coerce_feature_types(model_df, numeric_features, categorical_features)
    model_df = model_df.loc[model_df[target].notna()].copy()

    dropped_constant_features: list[str] = []
    variable_numeric: list[str] = []
    variable_categorical: list[str] = []
    for col in numeric_features:
        nunique = model_df[col].dropna().nunique()
        if nunique <= 1:
            dropped_constant_features.append(col)
        else:
            variable_numeric.append(col)

    for col in categorical_features:
        nunique = model_df[col].dropna().nunique()
        if nunique <= 1:
            dropped_constant_features.append(col)
        else:
            variable_categorical.append(col)

    selected_features = variable_numeric + variable_categorical
    if not selected_features:
        raise ValueError(
            "All candidate features were constant after filtering. "
            "Check whether the current grid search actually varies the requested inputs."
        )

    model_columns = list(dict.fromkeys(output_columns + selected_features + [target]))
    model_df = model_df[model_columns].copy()
    return model_df, selected_features, variable_numeric, variable_categorical, dropped_constant_features


def _build_model_pipeline(numeric_features: list[str], categorical_features: list[str], rf_kwargs: dict[str, Any]):
    try:
        from sklearn.compose import ColumnTransformer
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.impute import SimpleImputer
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import OneHotEncoder
    except ImportError as exc:
        raise ImportError(
            "scikit-learn is required for synthetic_run_level_rf.py. "
            "Install it in the environment used for the notebook or CLI."
        ) from exc

    transformers: list[tuple[str, Any, list[str]]] = []
    if numeric_features:
        numeric_pipe = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
            ]
        )
        transformers.append(("num", numeric_pipe, numeric_features))

    if categorical_features:
        categorical_pipe = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]
        )
        transformers.append(("cat", categorical_pipe, categorical_features))

    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")
    model = RandomForestRegressor(**rf_kwargs)
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )


def _train_test_split(
    model_df: pd.DataFrame,
    feature_columns: list[str],
    target: str,
    test_size: float,
    random_state: int,
    group_col: str | None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.DataFrame, pd.Series, pd.DataFrame, pd.Series | None]:
    try:
        from sklearn.model_selection import GroupShuffleSplit, train_test_split
    except ImportError as exc:
        raise ImportError(
            "scikit-learn is required for synthetic_run_level_rf.py. "
            "Install it in the environment used for the notebook or CLI."
        ) from exc

    X = model_df[feature_columns].copy()
    y = model_df[target]
    groups = None

    if group_col:
        if group_col not in model_df.columns:
            raise ValueError(f"group_col '{group_col}' is not present in the prepared model table.")
        groups = model_df[group_col].copy()
        splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
        train_idx, test_idx = next(splitter.split(X, y, groups=groups))
        X_train = X.iloc[train_idx].copy()
        X_test = X.iloc[test_idx].copy()
        y_train = y.iloc[train_idx].copy()
        y_test = y.iloc[test_idx].copy()
        test_keys = model_df.iloc[test_idx][OUTPUT_KEY].copy()
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=random_state,
        )
        test_keys = model_df.loc[X_test.index, OUTPUT_KEY].copy()

    return X_train, X_test, y_train, y_test, X, y, test_keys, groups


def _evaluate_regression(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float]:
    try:
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    except ImportError as exc:
        raise ImportError(
            "scikit-learn is required for synthetic_run_level_rf.py. "
            "Install it in the environment used for the notebook or CLI."
        ) from exc

    return {
        "r2": float(r2_score(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
    }


def _cross_validate_model(
    pipeline: Any,
    X: pd.DataFrame,
    y: pd.Series,
    cv_folds: int,
    random_state: int,
    groups: pd.Series | None = None,
) -> dict[str, float]:
    try:
        from sklearn.model_selection import GroupKFold, KFold, cross_validate
    except ImportError as exc:
        raise ImportError(
            "scikit-learn is required for synthetic_run_level_rf.py. "
            "Install it in the environment used for the notebook or CLI."
        ) from exc

    if cv_folds < 2:
        return {}

    if groups is not None:
        if groups.nunique(dropna=True) < cv_folds:
            print(
                f"[warn] group_col has only {groups.nunique(dropna=True)} unique groups. "
                f"Skipping grouped cross-validation with cv_folds={cv_folds}."
            )
            return {}
        splitter = GroupKFold(n_splits=cv_folds)
    else:
        splitter = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)

    scores = cross_validate(
        pipeline,
        X,
        y,
        groups=groups,
        cv=splitter,
        scoring={
            "r2": "r2",
            "rmse": "neg_root_mean_squared_error",
            "mae": "neg_mean_absolute_error",
        },
        n_jobs=1,
    )

    return {
        "cv_r2_mean": float(np.mean(scores["test_r2"])),
        "cv_r2_std": float(np.std(scores["test_r2"])),
        "cv_rmse_mean": float(-np.mean(scores["test_rmse"])),
        "cv_rmse_std": float(np.std(scores["test_rmse"])),
        "cv_mae_mean": float(-np.mean(scores["test_mae"])),
        "cv_mae_std": float(np.std(scores["test_mae"])),
    }


def _compute_permutation_importance(
    pipeline: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    random_state: int,
    n_repeats: int,
) -> pd.DataFrame:
    try:
        from sklearn.inspection import permutation_importance
    except ImportError as exc:
        raise ImportError(
            "scikit-learn is required for synthetic_run_level_rf.py. "
            "Install it in the environment used for the notebook or CLI."
        ) from exc

    perm = permutation_importance(
        pipeline,
        X_test,
        y_test,
        n_repeats=n_repeats,
        random_state=random_state,
        scoring="r2",
        n_jobs=1,
    )
    return (
        pd.DataFrame(
            {
                "feature": X_test.columns,
                "importance_mean": perm.importances_mean,
                "importance_std": perm.importances_std,
            }
        )
        .sort_values("importance_mean", ascending=False)
        .reset_index(drop=True)
    )


def _compute_shap_outputs(
    pipeline: Any,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    numeric_features: list[str],
    categorical_features: list[str],
    random_state: int,
    background_size: int,
    sample_size: int,
) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    try:
        import shap
    except ImportError:
        print("[info] SHAP is not installed. Skipping SHAP analysis.")
        return None, None

    if X_test.empty:
        return None, None

    background_raw = X_train.sample(min(len(X_train), background_size), random_state=random_state).copy()
    sample_raw = X_test.sample(min(len(X_test), sample_size), random_state=random_state).copy()
    if sample_raw.empty:
        return None, None

    preprocessor = pipeline.named_steps["preprocessor"]
    model = pipeline.named_steps["model"]

    background = np.asarray(preprocessor.transform(background_raw), dtype=float)
    sample = np.asarray(preprocessor.transform(sample_raw), dtype=float)
    transformed_feature_names = list(preprocessor.get_feature_names_out())

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(sample)
    shap_matrix = np.asarray(shap_values)
    if shap_matrix.ndim == 3:
        shap_matrix = shap_matrix[..., 0]

    original_names: list[str] = []
    for transformed_name in transformed_feature_names:
        stripped = transformed_name.split("__", 1)[1] if "__" in transformed_name else transformed_name
        if stripped in numeric_features:
            original_names.append(stripped)
            continue

        matched = None
        for feature in categorical_features:
            if stripped == feature or stripped.startswith(f"{feature}_"):
                matched = feature
                break
        original_names.append(matched or stripped)

    feature_to_indices: dict[str, list[int]] = {}
    for idx, feature in enumerate(original_names):
        feature_to_indices.setdefault(feature, []).append(idx)

    ordered_features = list(X_test.columns)
    aggregated = {}
    for feature in ordered_features:
        indices = feature_to_indices.get(feature, [])
        if not indices:
            continue
        aggregated[feature] = shap_matrix[:, indices].sum(axis=1)

    if not aggregated:
        print("[info] SHAP ran, but no transformed features could be mapped back to original feature names.")
        return None, None

    shap_detail = sample_raw.copy()
    shap_detail["prediction"] = pipeline.predict(sample_raw)
    for feature in ordered_features:
        if feature in aggregated:
            shap_detail[f"shap__{feature}"] = aggregated[feature]

    shap_importance = (
        pd.DataFrame(
            {
                "feature": list(aggregated),
                "mean_abs_shap": [float(np.mean(np.abs(values))) for values in aggregated.values()],
            }
        )
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )

    return shap_importance, shap_detail


def run_random_forest_regression(
    df_final: pd.DataFrame,
    target: str = "fallTime_50",
    edge_df: pd.DataFrame | None = None,
    feature_mode: str = "designed_plus_realized",
    include_topology: bool = True,
    extra_feature_columns: list[str] | None = None,
    include_feature_columns: list[str] | None = None,
    exclude_feature_columns: list[str] | None = None,
    test_size: float = 0.2,
    random_state: int = 42,
    n_estimators: int = 500,
    min_samples_leaf: int = 2,
    max_depth: int | None = None,
    permutation_repeats: int = 20,
    cv_folds: int = 5,
    group_col: str | None = None,
    run_shap: bool = True,
    shap_background_size: int = 200,
    shap_sample_size: int = 300,
    verbose: bool = True,
) -> RunLevelRFResult:
    model_df, selected_features, numeric_features, categorical_features, dropped_constant_features = (
        prepare_run_level_dataset(
            df_final=df_final,
            target=target,
            edge_df=edge_df,
            feature_mode=feature_mode,
            include_topology=include_topology,
            extra_feature_columns=extra_feature_columns,
            include_feature_columns=include_feature_columns,
            exclude_feature_columns=exclude_feature_columns,
            verbose=verbose,
        )
    )

    rf_kwargs = {
        "n_estimators": n_estimators,
        "random_state": random_state,
        "n_jobs": -1,
        "min_samples_leaf": min_samples_leaf,
        "max_depth": max_depth,
    }
    pipeline = _build_model_pipeline(numeric_features, categorical_features, rf_kwargs)

    X_train, X_test, y_train, y_test, X_all, y_all, test_keys, groups = _train_test_split(
        model_df=model_df,
        feature_columns=selected_features,
        target=target,
        test_size=test_size,
        random_state=random_state,
        group_col=group_col,
    )

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    metrics: dict[str, Any] = {
        "target": target,
        "feature_mode": feature_mode,
        "include_topology": include_topology,
        "n_rows": int(len(model_df)),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "group_col": group_col,
        "random_state": random_state,
    }
    metrics.update(_evaluate_regression(y_test, y_pred))
    metrics.update(_cross_validate_model(pipeline, X_all, y_all, cv_folds, random_state, groups))

    permutation_df = _compute_permutation_importance(
        pipeline=pipeline,
        X_test=X_test,
        y_test=y_test,
        random_state=random_state,
        n_repeats=permutation_repeats,
    )

    predictions = test_keys.copy()
    predictions["observed"] = y_test.to_numpy()
    predictions["predicted"] = y_pred
    predictions["residual"] = predictions["observed"] - predictions["predicted"]

    shap_importance = None
    shap_values = None
    if run_shap:
        shap_importance, shap_values = _compute_shap_outputs(
            pipeline=pipeline,
            X_train=X_train,
            X_test=X_test,
            numeric_features=numeric_features,
            categorical_features=categorical_features,
            random_state=random_state,
            background_size=shap_background_size,
            sample_size=shap_sample_size,
        )

    if verbose:
        print(f"[info] Trained Random Forest for target '{target}' on {len(model_df)} runs.")
        print(f"[info] Test metrics: R2={metrics['r2']:.3f}, RMSE={metrics['rmse']:.3f}, MAE={metrics['mae']:.3f}")
        if dropped_constant_features:
            print(f"[info] Dropped constant features: {', '.join(dropped_constant_features)}")

    return RunLevelRFResult(
        target=target,
        data=model_df,
        feature_columns=selected_features,
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        dropped_constant_features=dropped_constant_features,
        metrics=metrics,
        permutation_importance=permutation_df,
        predictions=predictions,
        shap_importance=shap_importance,
        shap_values=shap_values,
        model=pipeline,
    )


def _save_bar_plot(df: pd.DataFrame, value_col: str, output_path: Path, title: str, top_n: int = 15) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    plot_df = df.head(top_n).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, max(4, 0.4 * len(plot_df))))
    ax.barh(plot_df["feature"], plot_df[value_col], color="steelblue")
    ax.set_title(title)
    ax.set_xlabel(value_col)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def save_run_level_result(result: RunLevelRFResult, output_dir: str | Path, top_n_plot: int = 15) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    (output_dir / "metrics.json").write_text(json.dumps(result.metrics, indent=2))
    (output_dir / "selected_features.json").write_text(
        json.dumps(
            {
                "feature_columns": result.feature_columns,
                "numeric_features": result.numeric_features,
                "categorical_features": result.categorical_features,
                "dropped_constant_features": result.dropped_constant_features,
            },
            indent=2,
        )
    )
    result.permutation_importance.to_csv(output_dir / "permutation_importance.csv", index=False)
    result.predictions.to_csv(output_dir / "test_predictions.csv", index=False)
    _save_bar_plot(
        result.permutation_importance,
        value_col="importance_mean",
        output_path=output_dir / "permutation_importance.png",
        title=f"Permutation Importance: {result.target}",
        top_n=top_n_plot,
    )

    if result.shap_importance is not None:
        result.shap_importance.to_csv(output_dir / "shap_importance.csv", index=False)
        _save_bar_plot(
            result.shap_importance,
            value_col="mean_abs_shap",
            output_path=output_dir / "shap_importance.png",
            title=f"SHAP Importance: {result.target}",
            top_n=top_n_plot,
        )

    if result.shap_values is not None:
        result.shap_values.to_csv(output_dir / "shap_values_sample.csv", index=False)


def run_random_forest_from_files(
    df_final_path: str | Path,
    target: str = "fallTime_50",
    edge_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    **kwargs: Any,
) -> RunLevelRFResult:
    df_final = _read_table(df_final_path)
    edge_df = _read_table(edge_path) if edge_path else None
    result = run_random_forest_regression(
        df_final=df_final,
        target=target,
        edge_df=edge_df,
        **kwargs,
    )
    if output_dir:
        save_run_level_result(result, output_dir=output_dir)
    return result


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run a run-level Random Forest regression on synthetic sensitivity outputs. "
            "The preferred primary input is a saved df_final table; optional edge-level "
            "data can be merged in as realized geometry summaries."
        )
    )
    parser.add_argument("--df-final", required=True, help="Path to a saved df_final table (CSV/TSV/Parquet).")
    parser.add_argument(
        "--edge-file",
        default=None,
        help=(
            "Optional edge-level table (for example edge_velocity_tc.parquet or edge_full.parquet). "
            "Required when feature-mode is 'designed_plus_realized'."
        ),
    )
    parser.add_argument("--target", default="fallTime_50", help=f"Regression target. Common choices: {DEFAULT_TARGETS}")
    parser.add_argument(
        "--feature-mode",
        choices=["designed", "designed_plus_realized"],
        default="designed_plus_realized",
        help="Use only designed run inputs, or add run-level summaries of edge geometry.",
    )
    parser.add_argument(
        "--no-topology",
        action="store_true",
        help="Exclude topology labels such as network_id/network_type/slope_target/sinuosity_target.",
    )
    parser.add_argument(
        "--extra-feature",
        action="append",
        default=[],
        help="Additional column to include as a predictor. Can be passed multiple times.",
    )
    parser.add_argument(
        "--include-feature",
        action="append",
        default=[],
        help="Use only the listed predictors. Can be passed multiple times.",
    )
    parser.add_argument(
        "--exclude-feature",
        action="append",
        default=[],
        help="Remove specific predictors from the automatically selected set. Can be passed multiple times.",
    )
    parser.add_argument("--test-size", type=float, default=0.2, help="Fraction of runs reserved for testing.")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed used for splitting and RF fitting.")
    parser.add_argument("--n-estimators", type=int, default=500, help="Number of trees in the Random Forest.")
    parser.add_argument("--min-samples-leaf", type=int, default=2, help="Minimum samples allowed in each leaf.")
    parser.add_argument("--max-depth", type=int, default=None, help="Optional maximum tree depth.")
    parser.add_argument(
        "--permutation-repeats",
        type=int,
        default=20,
        help="Number of repeats for permutation importance.",
    )
    parser.add_argument("--cv-folds", type=int, default=5, help="Cross-validation folds. Use 1 to disable CV.")
    parser.add_argument(
        "--group-col",
        default=None,
        help=(
            "Optional grouping column for train/test split and cross-validation, "
            "for example 'network_id' if you want grouped validation."
        ),
    )
    parser.add_argument(
        "--skip-shap",
        action="store_true",
        help="Disable SHAP. Useful when you only want Random Forest + permutation importance.",
    )
    parser.add_argument("--shap-background-size", type=int, default=200, help="Background rows used by SHAP.")
    parser.add_argument("--shap-sample-size", type=int, default=300, help="Test rows explained by SHAP.")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Optional directory where metrics, importance tables, predictions, and plots are written.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    result = run_random_forest_from_files(
        df_final_path=args.df_final,
        edge_path=args.edge_file,
        output_dir=args.output_dir,
        target=args.target,
        feature_mode=args.feature_mode,
        include_topology=not args.no_topology,
        extra_feature_columns=args.extra_feature,
        include_feature_columns=args.include_feature,
        exclude_feature_columns=args.exclude_feature,
        test_size=args.test_size,
        random_state=args.random_state,
        n_estimators=args.n_estimators,
        min_samples_leaf=args.min_samples_leaf,
        max_depth=args.max_depth,
        permutation_repeats=args.permutation_repeats,
        cv_folds=args.cv_folds,
        group_col=args.group_col,
        run_shap=not args.skip_shap,
    )

    print("\nTop permutation importance features:")
    print(result.permutation_importance.head(10).to_string(index=False))
    if result.shap_importance is not None:
        print("\nTop SHAP features:")
        print(result.shap_importance.head(10).to_string(index=False))
    if args.output_dir:
        print(f"\nSaved outputs to: {Path(args.output_dir).resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
