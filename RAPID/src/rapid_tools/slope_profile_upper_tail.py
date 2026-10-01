"""Incremental upper-tail CV continuation for the slope investigation notebook."""

from __future__ import annotations

import contextlib
import io
import re
from pathlib import Path
from typing import Any, Mapping, MutableMapping

import nbformat
import numpy as np
import pandas as pd


def _acquisition_cv_scores(
    *,
    observations: pd.DataFrame,
    fit: Any,
    namespace: Mapping[str, Any],
    lambda_value: float,
    folds: int,
) -> pd.DataFrame:
    """Score one lambda with held-out whole-acquisition cross-validation."""
    rows: list[dict[str, Any]] = []
    for fold in range(folds):
        train = observations.loc[observations["_tail_fold"].ne(fold)]
        test = observations.loc[observations["_tail_fold"].eq(fold)]
        train_ids = pd.Index(sorted(train["acquisition_id"].unique()))
        lookup = {value: index for index, value in enumerate(train_ids)}
        fitted = fit(
            namespace["stage_model_x_m"],
            train["_stage_model_node_index"].to_numpy(int),
            train["acquisition_id"].map(lookup).to_numpy(int),
            train["wse_sm"].to_numpy(float),
            train["_tail_weight"].to_numpy(float),
            float(lambda_value),
            namespace["stage_model_reference_profile_m"],
            namespace["stage_model_minimum_slope_m_per_m"],
            namespace["stage_model_penalty"],
        )
        residual0 = test["wse_sm"].to_numpy(float) - fitted["profile_m"][
            test["_stage_model_node_index"].to_numpy(int)
        ]
        test_index, _ = pd.factorize(test["acquisition_id"], sort=True)
        weight = test["_tail_weight"].to_numpy(float)
        offsets = np.bincount(test_index, weights=weight * residual0) / np.bincount(
            test_index, weights=weight
        )
        held_out = pd.DataFrame(
            {
                "acquisition_id": test["acquisition_id"].to_numpy(),
                "weight": weight,
                "residual2": np.square(residual0 - offsets[test_index]),
            }
        )
        for acquisition_id, group in held_out.groupby("acquisition_id", sort=False):
            rows.append(
                {
                    "lambda_value": float(lambda_value),
                    "acquisition_id": acquisition_id,
                    "weighted_mse_m2": float(
                        np.average(group["residual2"], weights=group["weight"])
                    ),
                }
            )
    return pd.DataFrame(rows)


def run_upper_tail_continuation(
    namespace: MutableMapping[str, Any],
    *,
    notebook_path: str | Path,
    lambdas_per_run: int = 3,
    log10_step: float = 0.3,
    max_lambda: float = 1e14,
) -> dict[str, pd.DataFrame]:
    """Append upper-tail lambdas without repeating the completed baseline search.

    ``namespace`` is the notebook's ``globals()``. The function retains each
    existing per-lambda CV summary and appends only the newly evaluated values.
    The caller can run it again; only examples whose selected lambda remains at
    the current upper edge are considered on a subsequent pass.
    """
    required = {
        "all_example_joint_cv",
        "all_example_profile_summary",
        "BATCH_COMPARISON_WEIGHT_MODE",
        "BATCH_COMPARISON_LAMBDA",
        "BATCH_JOINT_CV_FOLDS",
        "FLOOR_ACTIVITY_TOLERANCE_M_PER_M",
        "REPO_ROOT",
        "go",
    }
    missing = sorted(required - namespace.keys())
    if missing:
        raise RuntimeError(
            "Run the completed 16-example batch cell first in this kernel; "
            f"missing: {missing}"
        )
    if lambdas_per_run < 1 or log10_step <= 0 or max_lambda <= 0:
        raise ValueError("Upper-tail controls must be strictly positive.")

    summary = namespace["all_example_profile_summary"]
    if "upper_tail_status" not in namespace:
        status = summary[
            [
                "example_id",
                "joint_selected_lambda",
                "joint_selected_lambda_is_cv_grid_upper_edge",
            ]
        ].rename(
            columns={
                "joint_selected_lambda": "selected_lambda",
                "joint_selected_lambda_is_cv_grid_upper_edge": "selected_at_current_upper_edge",
            }
        )
        status["continuation_rounds"] = 0
    else:
        status = namespace["upper_tail_status"].copy()
    pending = status.loc[status["selected_at_current_upper_edge"].fillna(False)].copy()
    if pending.empty:
        empty = pd.DataFrame()
        namespace["upper_tail_status"] = status
        namespace["upper_tail_continuation_summary"] = empty
        namespace["upper_tail_continuation_profiles"] = empty
        namespace["upper_tail_continuation_failures"] = empty
        return {"summary": empty, "profiles": empty, "failures": empty, "status": status}

    notebook = nbformat.read(Path(notebook_path), as_version=4)
    source = {cell.get("id"): cell.source for cell in notebook.cells}
    required_cells = [
        "f3f2018b",
        "b3690346",
        "9a7865a3",
        "direct-riversp-profile-test",
        "raw-riversp-acquisition-audit",
        "joint-stage-profile-prototype",
    ]
    absent_cells = [cell_id for cell_id in required_cells if cell_id not in source]
    if absent_cells:
        raise RuntimeError(f"Notebook cell identifiers changed: {absent_cells}")

    reports: list[dict[str, Any]] = []
    profiles: list[pd.DataFrame] = []
    failures: list[dict[str, str]] = []
    previous_show = namespace["go"].Figure.show
    namespace["go"].Figure.show = lambda *args, **kwargs: None
    try:
        for example_id in pending["example_id"].astype(str):
            try:
                local = dict(namespace)
                local["display"] = lambda *args, **kwargs: None
                config = re.sub(
                    r"EXAMPLE_ID = '[^']+'",
                    "EXAMPLE_ID = '%s'" % example_id,
                    source["f3f2018b"],
                )
                with contextlib.redirect_stdout(io.StringIO()):
                    exec(compile(config, "<upper-tail-config>", "exec"), local)
                    for cell_id in [
                        "b3690346",
                        "9a7865a3",
                        "direct-riversp-profile-test",
                        "raw-riversp-acquisition-audit",
                    ]:
                        exec(compile(source[cell_id], f"<upper-tail-{example_id}>", "exec"), local)
                    joint_source = source["joint-stage-profile-prototype"].replace(
                        "STAGE_MODEL_WEIGHT_MODE = 'equal'",
                        "STAGE_MODEL_WEIGHT_MODE = '%s'"
                        % namespace["BATCH_COMPARISON_WEIGHT_MODE"],
                    ).replace(
                        "STAGE_MODEL_LAMBDA = 1_000.0",
                        "STAGE_MODEL_LAMBDA = %r" % namespace["BATCH_COMPARISON_LAMBDA"],
                    )
                    exec(compile(joint_source, f"<upper-tail-joint-{example_id}>", "exec"), local)

                observations = local["stage_model_observations"].copy()
                observations["_tail_weight"] = local["stage_model_weights"]
                acquisition_order = (
                    observations.groupby("acquisition_id", as_index=False)["time_str"]
                    .min()
                    .sort_values(["time_str", "acquisition_id"])
                    .reset_index(drop=True)
                )
                acquisition_order["_tail_fold"] = (
                    np.arange(len(acquisition_order)) % namespace["BATCH_JOINT_CV_FOLDS"]
                )
                observations["_tail_fold"] = observations["acquisition_id"].map(
                    acquisition_order.set_index("acquisition_id")["_tail_fold"]
                ).astype(int)
                historical = namespace["all_example_joint_cv"].loc[
                    namespace["all_example_joint_cv"]["example_id"].astype(str).eq(example_id),
                    ["lambda_value", "cv_mean_mse_m2", "cv_sd_mse_m2", "n_acquisitions"],
                ].drop_duplicates("lambda_value")
                if historical.empty:
                    raise RuntimeError("No retained CV summary.")
                upper = float(historical["lambda_value"].max())
                new_lambdas = upper * np.power(
                    10.0, log10_step * np.arange(1, lambdas_per_run + 1)
                )
                new_lambdas = new_lambdas[new_lambdas <= max_lambda * (1.0 + 1e-12)]
                if not len(new_lambdas):
                    raise RuntimeError(f"Upper-tail safety ceiling reached at {upper:g}.")
                scores = pd.concat(
                    [
                        _acquisition_cv_scores(
                            observations=observations,
                            fit=local["fit_joint_stage_monotone_profile"],
                            namespace=local,
                            lambda_value=float(value),
                            folds=namespace["BATCH_JOINT_CV_FOLDS"],
                        )
                        for value in new_lambdas
                    ],
                    ignore_index=True,
                )
                new_cv = scores.groupby("lambda_value", as_index=False).agg(
                    cv_mean_mse_m2=("weighted_mse_m2", "mean"),
                    cv_sd_mse_m2=("weighted_mse_m2", "std"),
                    n_acquisitions=("weighted_mse_m2", "count"),
                )
                cv = pd.concat([historical, new_cv], ignore_index=True).sort_values(
                    "lambda_value"
                ).reset_index(drop=True)
                cv["cv_se_mse_m2"] = cv["cv_sd_mse_m2"].fillna(0.0) / np.sqrt(
                    cv["n_acquisitions"]
                )
                best = cv.loc[cv["cv_mean_mse_m2"].idxmin()]
                threshold = float(best["cv_mean_mse_m2"] + best["cv_se_mse_m2"])
                selected = float(
                    cv.loc[cv["cv_mean_mse_m2"].le(threshold), "lambda_value"].max()
                )
                at_edge = bool(selected == float(cv["lambda_value"].max()))
                result = local["fit_joint_stage_monotone_profile"](
                    local["stage_model_x_m"],
                    local["stage_model_node_index"],
                    local["stage_model_acquisition_index"],
                    local["stage_model_wse_m"],
                    local["stage_model_weights"],
                    selected,
                    local["stage_model_reference_profile_m"],
                    local["stage_model_minimum_slope_m_per_m"],
                    local["stage_model_penalty"],
                )
                slopes = np.diff(result["profile_m"]) / np.diff(local["stage_model_x_m"])
                floor_active = slopes <= (
                    local["stage_model_minimum_slope_m_per_m"]
                    + namespace["FLOOR_ACTIVITY_TOLERANCE_M_PER_M"]
                )
                endpoint_heads = result["profile_m"][local["stage_model_endpoint_node_index"]]
                static_heads = local["endpoint_reference_heads_m"]
                fitted_drop = float(np.diff(endpoint_heads)[0])
                static_drop = float(np.diff(static_heads)[0])

                namespace["all_example_joint_cv"] = pd.concat(
                    [
                        namespace["all_example_joint_cv"].loc[
                            ~namespace["all_example_joint_cv"]["example_id"].astype(str).eq(example_id)
                        ],
                        cv.assign(
                            example_id=example_id,
                            minimum_cv_lambda=float(best["lambda_value"]),
                            one_se_threshold_m2=threshold,
                            selected_lambda=selected,
                        ),
                    ],
                    ignore_index=True,
                )
                status = pd.concat(
                    [
                        status.loc[~status["example_id"].astype(str).eq(example_id)],
                        pd.DataFrame(
                            [
                                {
                                    "example_id": example_id,
                                    "selected_lambda": selected,
                                    "selected_at_current_upper_edge": at_edge,
                                    "continuation_rounds": int(
                                        pending.loc[
                                            pending["example_id"].astype(str).eq(example_id),
                                            "continuation_rounds",
                                        ].iloc[0]
                                    )
                                    + 1,
                                }
                            ]
                        ),
                    ],
                    ignore_index=True,
                ).sort_values("example_id").reset_index(drop=True)
                profiles.append(
                    pd.DataFrame(
                        {
                            "example_id": example_id,
                            "reference_section_dist_out_m": local["stage_model_x_m"],
                            "joint_monotone_wse_m": result["profile_m"],
                        }
                    )
                )
                reports.append(
                    {
                        "example_id": example_id,
                        "previous_upper_lambda": upper,
                        "tested_upper_lambdas": ", ".join("%g" % value for value in new_lambdas),
                        "selected_lambda": selected,
                        "selected_at_current_upper_edge": at_edge,
                        "minimum_cv_lambda": float(best["lambda_value"]),
                        "selected_cv_mse_m2": float(
                            cv.loc[cv["lambda_value"].eq(selected), "cv_mean_mse_m2"].iloc[0]
                        ),
                        "one_se_threshold_m2": threshold,
                        "fitted_head_loss_m": fitted_drop,
                        "relative_head_loss_difference": (fitted_drop - static_drop) / static_drop,
                        "floor_active_length_km": float(
                            np.diff(local["stage_model_x_m"])[floor_active].sum() / 1000.0
                        ),
                    }
                )
            except Exception as error:  # Preserve other examples' continuation results.
                failures.append({"example_id": example_id, "error": f"{type(error).__name__}: {error}"})
    finally:
        namespace["go"].Figure.show = previous_show

    report_frame = pd.DataFrame(reports)
    profile_frame = pd.concat(profiles, ignore_index=True) if profiles else pd.DataFrame()
    failure_frame = pd.DataFrame(failures)
    namespace["upper_tail_status"] = status
    namespace["upper_tail_continuation_summary"] = report_frame
    namespace["upper_tail_continuation_profiles"] = profile_frame
    namespace["upper_tail_continuation_failures"] = failure_frame
    return {
        "summary": report_frame,
        "profiles": profile_frame,
        "failures": failure_frame,
        "status": status,
    }
