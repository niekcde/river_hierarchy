"""Production-grade fitting and artifact writing for reference WSE profiles.

This module is intentionally independent from hydraulic-context mutation.  It
fits one reviewed base-example profile and writes immutable artifacts which a
later candidate-context runner can consume.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Iterable, Literal

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.linalg import eigsh


REFERENCE_PROFILE_SCHEMA_VERSION = "reference-profile-v1"


@dataclass(frozen=True, slots=True)
class ReferenceProfileConfig:
    """Numerical and selection settings fixed by the slope investigation."""

    minimum_slope_fraction: float = 0.01
    weight_mode: Literal["equal", "capped_inverse_variance"] = "equal"
    uncertainty_floor_m: float = 0.15
    cv_folds: int = 3
    max_outer_iterations: int = 100
    max_inner_iterations: int = 500
    tolerance_m: float = 1e-7


@dataclass(frozen=True, slots=True)
class ReferenceProfileFit:
    profile: pd.DataFrame
    observations: pd.DataFrame
    acquisition_offsets: pd.DataFrame
    summary: dict[str, float | int | str | bool]


def apply_manual_linear_profile_bridge(
    fit: ReferenceProfileFit,
    *,
    adjustment_id: str,
    start_distance_m: float,
    end_distance_m: float,
) -> tuple[ReferenceProfileFit, dict[str, object]]:
    """Replace one weakly supported fitted-profile interval by a linear bridge.

    The two bridge endpoint heads are taken from the *already fitted* profile,
    so the adjustment is continuous and cannot alter the profile outside the
    documented interval. This is deliberately a profile-level intervention:
    every later network-node interpolation within the interval changes
    consistently, rather than creating an untraceable per-link override.
    """
    if not adjustment_id or not str(adjustment_id).strip():
        raise ValueError("A manual profile bridge needs a non-empty adjustment_id.")
    if not np.isfinite([start_distance_m, end_distance_m]).all() or end_distance_m <= start_distance_m:
        raise ValueError("Manual profile bridge bounds must be finite and strictly increasing.")
    profile = fit.profile.copy().sort_values("reference_section_dist_out_m").reset_index(drop=True)
    x = profile["reference_section_dist_out_m"].to_numpy(float)
    start_index = int(np.argmin(np.abs(x - float(start_distance_m))))
    end_index = int(np.argmin(np.abs(x - float(end_distance_m))))
    if start_index == end_index:
        raise ValueError("Manual profile bridge bounds resolve to the same fixed profile station.")
    if start_index > end_index:
        start_index, end_index = end_index, start_index
    start_x, end_x = float(x[start_index]), float(x[end_index])
    y = profile["fitted_wse_m"].to_numpy(float).copy()
    start_head, end_head = float(y[start_index]), float(y[end_index])
    bridge_slope = (end_head - start_head) / (end_x - start_x)
    if bridge_slope <= 0.0:
        raise ValueError("Manual profile bridge must retain a positive longitudinal slope.")
    mask = np.zeros(len(profile), dtype=bool)
    mask[start_index : end_index + 1] = True
    y[mask] = start_head + bridge_slope * (x[mask] - start_x)
    if np.any(np.diff(y) <= 0.0):
        raise ValueError("Manual profile bridge would violate the strictly positive fitted profile.")
    profile["fitted_wse_m"] = y
    slopes = np.r_[np.nan, np.diff(y) / np.diff(x)]
    profile["interval_slope_m_per_m"] = slopes
    minimum_slope = float(fit.summary["minimum_slope_m_per_m"])
    profile["floor_active"] = np.r_[False, slopes[1:] <= minimum_slope + 1e-12]
    profile["manual_profile_adjustment_id"] = pd.NA
    profile.loc[mask, "manual_profile_adjustment_id"] = str(adjustment_id)
    profile["manual_profile_adjustment_active"] = mask

    observations = fit.observations.copy()
    lookup = pd.Series(y, index=x)
    observations["fitted_profile_wse_m"] = observations["reference_section_dist_out_m"].map(lookup)
    if observations["fitted_profile_wse_m"].isna().any():
        raise ValueError("Manual bridge could not map every observation back onto the fixed profile grid.")
    offsets = fit.acquisition_offsets.set_index("acquisition_id")["stage_offset_m"]
    observations["acquisition_stage_offset_m"] = observations["acquisition_id"].astype(str).map(offsets)
    observations["residual_m"] = (
        observations["wse_m"]
        - observations["fitted_profile_wse_m"]
        - observations["acquisition_stage_offset_m"]
    )
    observations["stage_aligned_wse_m"] = observations["wse_m"] - observations["acquisition_stage_offset_m"]

    summary = dict(fit.summary)
    endpoint_x = np.array([x[0], x[-1]], dtype=float)
    endpoint_y = np.interp(endpoint_x, x, y)
    summary.update(
        {
            "fitted_endpoint_head_outlet_m": float(endpoint_y[0]),
            "fitted_endpoint_head_inlet_m": float(endpoint_y[1]),
            "fitted_head_loss_m": float(endpoint_y[1] - endpoint_y[0]),
            "fitted_global_slope_m_per_m": float((endpoint_y[1] - endpoint_y[0]) / (endpoint_x[1] - endpoint_x[0])),
            "weighted_mse_m2": float(np.mean(np.square(observations["residual_m"].to_numpy(float)))),
            "weighted_rms_m": float(np.sqrt(np.mean(np.square(observations["residual_m"].to_numpy(float))))),
            "n_floor_active_intervals": int(profile["floor_active"].sum()),
        }
    )
    record = {
        "adjustment_id": str(adjustment_id),
        "adjustment_type": "manual_linear_profile_bridge",
        "requested_start_distance_m": float(start_distance_m),
        "requested_end_distance_m": float(end_distance_m),
        "resolved_start_distance_m": start_x,
        "resolved_end_distance_m": end_x,
        "start_head_m": start_head,
        "end_head_m": end_head,
        "bridge_slope_m_per_m": float(bridge_slope),
        "n_profile_rows_replaced": int(mask.sum()),
    }
    return ReferenceProfileFit(profile=profile, observations=observations, acquisition_offsets=fit.acquisition_offsets.copy(), summary=summary), record


def default_lambda_grid() -> np.ndarray:
    """Full smooth-side grid, including the reviewed upper-tail range."""
    return np.logspace(-1.0, 11.3, 42)


def _pava_nondecreasing(values: np.ndarray) -> np.ndarray:
    """Euclidean projection onto the non-decreasing cone."""
    block_values: list[float] = []
    block_weights: list[float] = []
    block_sizes: list[int] = []
    for value in np.asarray(values, dtype=float):
        block_values.append(float(value))
        block_weights.append(1.0)
        block_sizes.append(1)
        while len(block_values) > 1 and block_values[-2] > block_values[-1]:
            weight = block_weights[-2] + block_weights[-1]
            block_values[-2] = (
                block_values[-2] * block_weights[-2]
                + block_values[-1] * block_weights[-1]
            ) / weight
            block_weights[-2] = weight
            block_sizes[-2] += block_sizes[-1]
            block_values.pop()
            block_weights.pop()
            block_sizes.pop()
    return np.repeat(np.asarray(block_values, dtype=float), block_sizes)


def _linear_interpolate_extrapolate(
    query_x: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
) -> np.ndarray:
    query_x = np.asarray(query_x, dtype=float)
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) < 2 or np.any(np.diff(x) <= 0.0):
        raise ValueError("Interpolation coordinates must be strictly increasing with at least two rows.")
    result = np.interp(query_x, x, y)
    lower = query_x < x[0]
    upper = query_x > x[-1]
    if lower.any():
        result[lower] = y[0] + (query_x[lower] - x[0]) * (y[1] - y[0]) / (x[1] - x[0])
    if upper.any():
        result[upper] = y[-1] + (query_x[upper] - x[-1]) * (y[-1] - y[-2]) / (x[-1] - x[-2])
    return result


def _curvature_penalty(x_m: np.ndarray) -> sparse.csr_matrix:
    """Normalised non-uniform-grid second-derivative penalty."""
    x_m = np.asarray(x_m, dtype=float)
    if len(x_m) < 3:
        return sparse.csr_matrix((len(x_m), len(x_m)), dtype=float)
    span = float(x_m[-1] - x_m[0])
    if span <= 0.0:
        raise ValueError("Profile grid must have positive span.")
    x = (x_m - x_m[0]) / span
    h = np.diff(x)
    if np.any(h <= 0.0):
        raise ValueError("Profile grid must be strictly increasing.")
    rows: list[int] = []
    columns: list[int] = []
    values: list[float] = []
    for index in range(1, len(x) - 1):
        left, right = h[index - 1], h[index]
        weight = np.sqrt((left + right) / 2.0)
        rows.extend([index - 1] * 3)
        columns.extend([index - 1, index, index + 1])
        values.extend(
            [
                weight * 2.0 / (left * (left + right)),
                weight * -2.0 * (1.0 / left + 1.0 / right) / (left + right),
                weight * 2.0 / (right * (left + right)),
            ]
        )
    operator = sparse.csr_matrix((values, (rows, columns)), shape=(len(x) - 2, len(x)))
    penalty = (operator.T @ operator).tocsr()
    spectral_scale = float(eigsh(penalty, k=1, which="LM", return_eigenvectors=False)[0])
    return penalty / spectral_scale


def _profile_hash(profile: pd.DataFrame) -> str:
    payload = profile[["reference_section_dist_out_m", "fitted_wse_m"]].to_csv(index=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _validated_inputs(
    static_profile: pd.DataFrame,
    observations: pd.DataFrame,
    endpoint_distances_m: Iterable[float],
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    required_static = {"reference_section_dist_out_m", "static_wse_m"}
    required_observations = {"reference_section_dist_out_m", "acquisition_id", "wse_m"}
    missing_static = sorted(required_static - set(static_profile.columns))
    missing_observations = sorted(required_observations - set(observations.columns))
    if missing_static or missing_observations:
        raise ValueError(
            f"Missing static fields={missing_static}; missing observation fields={missing_observations}."
        )
    static = static_profile[["reference_section_dist_out_m", "static_wse_m"]].copy()
    static = static.apply(pd.to_numeric, errors="coerce").dropna().groupby(
        "reference_section_dist_out_m", as_index=False
    )["static_wse_m"].mean().sort_values("reference_section_dist_out_m")
    static_x = static["reference_section_dist_out_m"].to_numpy(float)
    static_y = static["static_wse_m"].to_numpy(float)
    if len(static_x) < 2 or np.any(np.diff(static_x) <= 0.0):
        raise ValueError("Static profile needs at least two unique increasing coordinates.")
    endpoints = np.sort(np.asarray(list(endpoint_distances_m), dtype=float))
    if endpoints.shape != (2,) or not np.isfinite(endpoints).all() or endpoints[1] <= endpoints[0]:
        raise ValueError("endpoint_distances_m must contain two finite increasing values.")
    fixed_x = np.unique(np.r_[static_x, endpoints])
    initial = _linear_interpolate_extrapolate(fixed_x, static_x, static_y)

    observed = observations.copy()
    observed["reference_section_dist_out_m"] = pd.to_numeric(
        observed["reference_section_dist_out_m"], errors="coerce"
    )
    observed["wse_m"] = pd.to_numeric(observed["wse_m"], errors="coerce")
    observed = observed.dropna(subset=["reference_section_dist_out_m", "wse_m", "acquisition_id"]).copy()
    if observed.empty:
        raise ValueError("No valid raw-pass observations remain.")
    index = np.searchsorted(fixed_x, observed["reference_section_dist_out_m"].to_numpy(float))
    valid = index < len(fixed_x)
    valid[valid] &= np.isclose(
        fixed_x[index[valid]], observed.loc[valid, "reference_section_dist_out_m"].to_numpy(float), rtol=0.0, atol=1e-6
    )
    if not valid.all():
        raise ValueError("Every raw observation must lie on the complete fixed SWORD corridor grid.")
    observed["_node_index"] = index.astype(int)
    return fixed_x, initial, observed.reset_index(drop=True), endpoints, static_x, static_y


def _fit(
    *,
    x_m: np.ndarray,
    node_index: np.ndarray,
    acquisition_index: np.ndarray,
    wse_m: np.ndarray,
    weights: np.ndarray,
    lambda_value: float,
    initial_profile_m: np.ndarray,
    minimum_slope_m_per_m: float,
    penalty: sparse.csr_matrix,
    config: ReferenceProfileConfig,
) -> dict[str, object]:
    n_nodes = len(x_m)
    n_acquisitions = int(acquisition_index.max()) + 1
    by_node = np.bincount(node_index, weights=weights, minlength=n_nodes)
    by_acquisition = np.bincount(acquisition_index, weights=weights, minlength=n_acquisitions)
    if np.any(by_acquisition <= 0.0):
        raise ValueError("Every acquisition must have positive weight.")
    profile = _pava_nondecreasing(initial_profile_m - minimum_slope_m_per_m * x_m) + minimum_slope_m_per_m * x_m
    offsets = np.zeros(n_acquisitions, dtype=float)
    step = 1.0 / (2.0 * float(by_node.max()) + 2.0 * float(lambda_value))
    inner_iteration = 0
    for outer_iteration in range(config.max_outer_iterations):
        profile_before = profile.copy()
        offsets_before = offsets.copy()
        adjusted_sum = np.bincount(
            node_index,
            weights=weights * (wse_m - offsets[acquisition_index]),
            minlength=n_nodes,
        )
        for inner_iteration in range(config.max_inner_iterations):
            gradient = 2.0 * (by_node * profile - adjusted_sum) + 2.0 * float(lambda_value) * (penalty @ profile)
            candidate = _pava_nondecreasing(
                profile - step * gradient - minimum_slope_m_per_m * x_m
            ) + minimum_slope_m_per_m * x_m
            if float(np.max(np.abs(candidate - profile))) <= config.tolerance_m:
                profile = candidate
                break
            profile = candidate
        offset_sum = np.bincount(
            acquisition_index,
            weights=weights * (wse_m - profile[node_index]),
            minlength=n_acquisitions,
        )
        offsets = offset_sum / by_acquisition
        offset_shift = float(offsets.mean())
        offsets -= offset_shift
        profile += offset_shift
        change = max(float(np.max(np.abs(profile - profile_before))), float(np.max(np.abs(offsets - offsets_before))))
        if change <= config.tolerance_m:
            break
    residual = wse_m - profile[node_index] - offsets[acquisition_index]
    return {
        "profile": profile,
        "offsets": offsets,
        "residual": residual,
        "weighted_mse_m2": float(np.average(np.square(residual), weights=weights)),
        "weighted_rms_m": float(np.sqrt(np.average(np.square(residual), weights=weights))),
        "outer_iterations": outer_iteration + 1,
        "inner_iterations": inner_iteration + 1,
        "n_observed_profile_nodes": int(np.count_nonzero(by_node)),
    }


def fit_reference_profile(
    static_profile: pd.DataFrame,
    observations: pd.DataFrame,
    *,
    endpoint_distances_m: Iterable[float],
    lambda_value: float,
    config: ReferenceProfileConfig = ReferenceProfileConfig(),
) -> ReferenceProfileFit:
    """Fit one fixed-lambda acquisition-stage-adjusted monotone WSE profile."""
    if lambda_value <= 0.0:
        raise ValueError("lambda_value must be positive.")
    x, initial, observed, endpoints, static_x, static_y = _validated_inputs(
        static_profile, observations, endpoint_distances_m
    )
    endpoint_static = _linear_interpolate_extrapolate(endpoints, static_x, static_y)
    reference_slope = float((endpoint_static[1] - endpoint_static[0]) / (endpoints[1] - endpoints[0]))
    if not np.isfinite(reference_slope) or reference_slope <= 0.0:
        raise ValueError("Static endpoint profile must yield a positive reference slope.")
    min_slope = float(config.minimum_slope_fraction * reference_slope)
    acquisition_order = pd.Index(sorted(observed["acquisition_id"].astype(str).unique()), name="acquisition_id")
    acquisition_lookup = {value: index for index, value in enumerate(acquisition_order)}
    acquisition_index = observed["acquisition_id"].astype(str).map(acquisition_lookup).to_numpy(int)
    if config.weight_mode == "equal":
        weights = np.ones(len(observed), dtype=float)
    elif config.weight_mode == "capped_inverse_variance":
        if "wse_uncertainty_m" not in observed.columns:
            raise ValueError("capped_inverse_variance requires wse_uncertainty_m.")
        uncertainty = np.maximum(
            pd.to_numeric(observed["wse_uncertainty_m"], errors="coerce").to_numpy(float),
            config.uncertainty_floor_m,
        )
        if not np.isfinite(uncertainty).all():
            raise ValueError("Observation uncertainty must be finite.")
        weights = 1.0 / np.square(uncertainty)
    else:
        raise ValueError(f"Unsupported weight mode: {config.weight_mode}")
    penalty = _curvature_penalty(x)
    result = _fit(
        x_m=x,
        node_index=observed["_node_index"].to_numpy(int),
        acquisition_index=acquisition_index,
        wse_m=observed["wse_m"].to_numpy(float),
        weights=weights,
        lambda_value=float(lambda_value),
        initial_profile_m=initial,
        minimum_slope_m_per_m=min_slope,
        penalty=penalty,
        config=config,
    )
    profile_values = np.asarray(result["profile"], dtype=float)
    slopes = np.r_[np.nan, np.diff(profile_values) / np.diff(x)]
    profile = pd.DataFrame(
        {
            "reference_section_dist_out_m": x,
            "static_wse_m": initial,
            "fitted_wse_m": profile_values,
            "interval_slope_m_per_m": slopes,
            "floor_active": np.r_[False, slopes[1:] <= min_slope + 1e-12],
        }
    )
    observed = observed.drop(columns="_node_index").copy()
    node_index = np.searchsorted(x, observed["reference_section_dist_out_m"].to_numpy(float))
    observed["fitted_profile_wse_m"] = profile_values[node_index]
    observed["acquisition_stage_offset_m"] = np.asarray(result["offsets"])[acquisition_index]
    observed["residual_m"] = np.asarray(result["residual"])
    observed["stage_aligned_wse_m"] = observed["wse_m"] - observed["acquisition_stage_offset_m"]
    offsets = pd.DataFrame({"acquisition_id": acquisition_order.astype(str), "stage_offset_m": result["offsets"]})
    endpoint_index = np.searchsorted(x, endpoints)
    fitted_heads = profile_values[endpoint_index]
    summary: dict[str, float | int | str | bool] = {
        "schema_version": REFERENCE_PROFILE_SCHEMA_VERSION,
        "lambda_value": float(lambda_value),
        "weight_mode": config.weight_mode,
        "n_profile_nodes": int(len(profile)),
        "n_observed_profile_nodes": int(result["n_observed_profile_nodes"]),
        "n_observations": int(len(observed)),
        "n_acquisitions": int(len(acquisition_order)),
        "reference_slope_m_per_m": reference_slope,
        "minimum_slope_m_per_m": min_slope,
        "minimum_slope_fraction": float(config.minimum_slope_fraction),
        "static_endpoint_head_outlet_m": float(endpoint_static[0]),
        "static_endpoint_head_inlet_m": float(endpoint_static[1]),
        "fitted_endpoint_head_outlet_m": float(fitted_heads[0]),
        "fitted_endpoint_head_inlet_m": float(fitted_heads[1]),
        "fitted_head_loss_m": float(fitted_heads[1] - fitted_heads[0]),
        "fitted_global_slope_m_per_m": float((fitted_heads[1] - fitted_heads[0]) / (endpoints[1] - endpoints[0])),
        "weighted_mse_m2": float(result["weighted_mse_m2"]),
        "weighted_rms_m": float(result["weighted_rms_m"]),
        "outer_iterations": int(result["outer_iterations"]),
        "inner_iterations": int(result["inner_iterations"]),
        "n_floor_active_intervals": int(profile["floor_active"].sum()),
    }
    return ReferenceProfileFit(profile=profile, observations=observed, acquisition_offsets=offsets, summary=summary)


def select_reference_profile(
    static_profile: pd.DataFrame,
    observations: pd.DataFrame,
    *,
    endpoint_distances_m: Iterable[float],
    lambda_values: Iterable[float] | None = None,
    config: ReferenceProfileConfig = ReferenceProfileConfig(),
) -> tuple[ReferenceProfileFit, pd.DataFrame, dict[str, object]]:
    """Select the smoothest one-SE profile with whole-acquisition CV."""
    values = np.unique(np.asarray(list(default_lambda_grid() if lambda_values is None else lambda_values), dtype=float))
    values = values[np.isfinite(values) & (values > 0.0)]
    if not len(values):
        raise ValueError("At least one positive lambda is required.")
    if config.cv_folds < 2:
        raise ValueError("Whole-acquisition CV requires at least two folds.")
    working = observations.copy()
    working["acquisition_id"] = working["acquisition_id"].astype(str)
    if "time" in working.columns:
        ordered = working.groupby("acquisition_id", as_index=False)["time"].min().sort_values(["time", "acquisition_id"])
    else:
        ordered = pd.DataFrame({"acquisition_id": sorted(working["acquisition_id"].astype(str).unique())})
    if len(ordered) < config.cv_folds:
        raise ValueError("Fewer acquisitions than CV folds.")
    ordered["_fold"] = np.arange(len(ordered)) % config.cv_folds
    working["_fold"] = working["acquisition_id"].astype(str).map(ordered.set_index("acquisition_id")["_fold"])
    score_rows: list[dict[str, float | int | str]] = []
    for lambda_value in values:
        for fold in range(config.cv_folds):
            train = working.loc[working["_fold"].ne(fold)].drop(columns="_fold")
            test = working.loc[working["_fold"].eq(fold)].drop(columns="_fold")
            fit = fit_reference_profile(
                static_profile, train, endpoint_distances_m=endpoint_distances_m,
                lambda_value=float(lambda_value), config=config,
            )
            x = fit.profile["reference_section_dist_out_m"].to_numpy(float)
            y = fit.profile["fitted_wse_m"].to_numpy(float)
            test = test.copy()
            fitted = _linear_interpolate_extrapolate(test["reference_section_dist_out_m"].to_numpy(float), x, y)
            residual = test["wse_m"].to_numpy(float) - fitted
            if config.weight_mode == "equal":
                weight = np.ones(len(test), dtype=float)
            else:
                uncertainty = np.maximum(pd.to_numeric(test["wse_uncertainty_m"], errors="coerce").to_numpy(float), config.uncertainty_floor_m)
                weight = 1.0 / np.square(uncertainty)
            test["_test_residual"] = residual
            test["_test_weight"] = weight
            for acquisition_id, group in test.groupby("acquisition_id", sort=False):
                residuals = group["_test_residual"].to_numpy(float)
                weights = group["_test_weight"].to_numpy(float)
                offset = float(np.average(residuals, weights=weights))
                mse = float(np.average(np.square(residuals - offset), weights=weights))
                score_rows.append({"lambda_value": float(lambda_value), "fold": fold, "acquisition_id": str(acquisition_id), "weighted_mse_m2": mse})
    scores = pd.DataFrame(score_rows)
    cv = scores.groupby("lambda_value", as_index=False).agg(
        cv_mean_mse_m2=("weighted_mse_m2", "mean"),
        cv_sd_mse_m2=("weighted_mse_m2", "std"),
        n_acquisitions=("weighted_mse_m2", "count"),
    ).sort_values("lambda_value").reset_index(drop=True)
    cv["cv_se_mse_m2"] = cv["cv_sd_mse_m2"].fillna(0.0) / np.sqrt(cv["n_acquisitions"])
    best = cv.loc[cv["cv_mean_mse_m2"].idxmin()]
    threshold = float(best["cv_mean_mse_m2"] + best["cv_se_mse_m2"])
    selected_lambda = float(cv.loc[cv["cv_mean_mse_m2"].le(threshold), "lambda_value"].max())
    fit = fit_reference_profile(
        static_profile, observations.drop(columns="_fold", errors="ignore"),
        endpoint_distances_m=endpoint_distances_m, lambda_value=selected_lambda, config=config,
    )
    selection = {
        "selection_rule": "smoothest_within_one_standard_error",
        "selected_lambda": selected_lambda,
        "best_lambda": float(best["lambda_value"]),
        "one_standard_error_threshold_m2": threshold,
        "selected_at_grid_upper_edge": bool(selected_lambda == float(cv["lambda_value"].max())),
        "lambda_grid_count": int(len(cv)),
    }
    return fit, cv, selection


def write_reference_profile_artifacts(
    fit: ReferenceProfileFit,
    cv: pd.DataFrame,
    selection: dict[str, object],
    output_dir: str | Path,
    *,
    config: ReferenceProfileConfig,
    input_metadata: dict[str, object] | None = None,
    manual_adjustments: list[dict[str, object]] | None = None,
) -> dict[str, Path]:
    """Write immutable profile products without modifying any context output."""
    root = Path(output_dir).expanduser().resolve()
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(f"Reference-profile output must be empty: {root}")
    root.mkdir(parents=True, exist_ok=False)
    profile = fit.profile.copy()
    profile_hash = _profile_hash(profile)
    profile_path = root / "reference_profile.parquet"
    observations_path = root / "reference_profile_observations.parquet"
    offsets_path = root / "reference_profile_acquisition_offsets.parquet"
    cv_path = root / "reference_profile_cv.csv"
    selection_path = root / "reference_profile_selection.json"
    audit_path = root / "reference_profile_audit.json"
    profile.to_parquet(profile_path, index=False)
    fit.observations.to_parquet(observations_path, index=False)
    fit.acquisition_offsets.to_parquet(offsets_path, index=False)
    cv.to_csv(cv_path, index=False)
    selection_payload = {
        "schema_version": REFERENCE_PROFILE_SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "config": asdict(config),
        "selection": selection,
        "profile_sha256": profile_hash,
        "inputs": input_metadata or {},
        "manual_adjustments": manual_adjustments or [],
    }
    audit_payload = {
        "schema_version": REFERENCE_PROFILE_SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "summary": fit.summary,
        "profile_sha256": profile_hash,
        "manual_adjustments": manual_adjustments or [],
    }
    selection_path.write_text(json.dumps(selection_payload, indent=2, sort_keys=True), encoding="utf-8")
    audit_path.write_text(json.dumps(audit_payload, indent=2, sort_keys=True), encoding="utf-8")
    return {
        "output_dir": root,
        "profile": profile_path,
        "observations": observations_path,
        "offsets": offsets_path,
        "cv": cv_path,
        "selection": selection_path,
        "audit": audit_path,
    }
