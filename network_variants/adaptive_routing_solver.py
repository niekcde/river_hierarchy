from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize


ADAPTIVE_ROUTING_SOLVER_VERSION = "adaptive-two-ended-routing-v1"


@dataclass(frozen=True)
class AdaptiveSolverConfig:
    """Configuration frozen after the offline solver comparison."""

    upstream_weight: float = 2.0
    downstream_weight: float = 1.0
    raw_penalty: float = 0.20
    normal_min_scale: float = 0.25
    normal_max_scale: float = 4.0
    hard_min_scale: float = 0.05
    hard_max_scale: float = 8.0
    max_iterations: int = 2000


def _clip01(values: pd.Series | np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(values, dtype=float), 0.0, 1.0)


def adaptive_evidence_parameters(
    candidates: pd.DataFrame,
    *,
    config: AdaptiveSolverConfig,
) -> pd.DataFrame:
    """Translate classifier/QC evidence into raw trust and soft safety bounds."""
    probability = pd.to_numeric(candidates["width_model_probability"], errors="coerce").fillna(0.5)
    threshold = pd.to_numeric(candidates.get("width_model_threshold", 0.4), errors="coerce")
    if not isinstance(threshold, pd.Series):
        threshold = pd.Series(float(threshold), index=candidates.index)
    threshold = threshold.fillna(0.4)
    confidence = _clip01((probability - threshold) / (1.0 - threshold).clip(lower=1e-6))

    upstream_deviation = pd.to_numeric(
        candidates["upstream_continuity_abs_deviation"], errors="coerce"
    ).fillna(0.0)
    deviation_strength = _clip01(np.log1p(upstream_deviation) / np.log(5.0))

    opposition = pd.to_numeric(candidates["opposition_cos_trimmed_mean"], errors="coerce").fillna(-1.0)
    opposition_badness = _clip01((opposition + 1.0) / 2.0)

    relaxation = _clip01(
        0.45 * deviation_strength + 0.30 * confidence + 0.25 * opposition_badness
    )
    min_scale = config.normal_min_scale * np.exp(
        np.log(config.hard_min_scale / config.normal_min_scale) * relaxation
    )
    max_scale = config.normal_max_scale * np.exp(
        np.log(config.hard_max_scale / config.normal_max_scale) * relaxation
    )
    evidence_freedom = 1.0 + 1.5 * deviation_strength + confidence + opposition_badness
    raw_penalty_weight = config.raw_penalty / evidence_freedom

    return pd.DataFrame(
        {
            "model_confidence_strength": confidence,
            "upstream_deviation_strength": deviation_strength,
            "opposition_badness": opposition_badness,
            "adaptive_relaxation_strength": relaxation,
            "adaptive_min_scale": min_scale,
            "adaptive_max_scale": max_scale,
            "adaptive_raw_penalty_weight": raw_penalty_weight,
        },
        index=candidates.index,
    )


def solve_adaptive_two_ended(
    links: pd.DataFrame,
    *,
    config: AdaptiveSolverConfig,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Jointly optimize predicted corrections using soft two-ended continuity."""
    frame = links.copy().reset_index(drop=True)
    required = {
        "id_link",
        "id_us_node",
        "id_ds_node",
        "wid_adj_wet",
        "routing_context_width_us",
        "routing_context_width_ds",
        "width_model_prediction",
        "width_model_probability",
        "upstream_continuity_abs_deviation",
        "opposition_cos_trimmed_mean",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Adaptive routing links are missing fields: {missing}")
    for column in ("id_link", "id_us_node", "id_ds_node"):
        frame[column] = pd.to_numeric(frame[column], errors="raise").astype(int)
    for column in ("wid_adj_wet", "routing_context_width_us", "routing_context_width_ds"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    invalid_width = frame[["wid_adj_wet", "routing_context_width_us", "routing_context_width_ds"]].isna().any(axis=1)
    invalid_width |= frame[["wid_adj_wet", "routing_context_width_us", "routing_context_width_ds"]].le(0).any(axis=1)
    if invalid_width.any():
        raise ValueError(
            "Adaptive routing requires finite positive measured and context widths; "
            f"invalid links={frame.loc[invalid_width, 'id_link'].astype(int).tolist()}"
        )

    candidate_mask = frame["width_model_prediction"].eq(1).fillna(False).to_numpy(bool)
    candidate_positions = np.flatnonzero(candidate_mask)
    frame["comparison_scale_factor"] = 1.0
    frame["comparison_routing_width"] = frame["wid_adj_wet"].astype(float)
    frame["comparison_status"] = "accepted_fixed"
    frame["comparison_flag"] = False
    frame["comparison_objective"] = np.nan
    for column in (
        "model_confidence_strength",
        "upstream_deviation_strength",
        "opposition_badness",
        "adaptive_relaxation_strength",
        "adaptive_min_scale",
        "adaptive_max_scale",
        "adaptive_raw_penalty_weight",
    ):
        frame[column] = np.nan
    if not len(candidate_positions):
        return frame, {
            "success": True,
            "message": "no predicted corrections",
            "iterations": 0,
            "objective": None,
            "candidate_count": 0,
            "adaptive_flag_count": 0,
        }

    candidate_frame = frame.loc[candidate_mask].copy()
    evidence = adaptive_evidence_parameters(candidate_frame, config=config)
    us_nodes = frame["id_us_node"].to_numpy(int)
    ds_nodes = frame["id_ds_node"].to_numpy(int)
    us_context = frame["routing_context_width_us"].to_numpy(float)
    ds_context = frame["routing_context_width_ds"].to_numpy(float)

    upstream_nodes = set(candidate_frame["id_us_node"].astype(int))
    downstream_nodes = set(candidate_frame["id_ds_node"].astype(int))
    relevant_nodes = sorted(upstream_nodes.union(downstream_nodes))
    incoming_positions = {node: np.flatnonzero(ds_nodes == node) for node in relevant_nodes}
    outgoing_positions = {node: np.flatnonzero(us_nodes == node) for node in relevant_nodes}
    max_deviation_by_us_node = (
        candidate_frame.groupby("id_us_node")["upstream_continuity_abs_deviation"].max().to_dict()
    )
    raw_weights = evidence["adaptive_raw_penalty_weight"].to_numpy(float)

    def objective(log_scales: np.ndarray) -> float:
        scales = np.ones(len(frame), dtype=float)
        scales[candidate_positions] = np.exp(log_scales)
        loss = float(np.sum(raw_weights * np.square(log_scales)))
        for node in relevant_nodes:
            incoming = float(np.sum(ds_context[incoming_positions[node]] * scales[incoming_positions[node]]))
            outgoing = float(np.sum(us_context[outgoing_positions[node]] * scales[outgoing_positions[node]]))
            if incoming <= 0 or outgoing <= 0:
                loss += 1e6
                continue
            log_ratio = float(np.log(outgoing / incoming))
            weight = 0.0
            if node in upstream_nodes:
                magnitude = float(max_deviation_by_us_node.get(node, 0.0))
                weight += config.upstream_weight * (1.0 + min(np.log1p(max(magnitude, 0.0)), 2.0))
            if node in downstream_nodes:
                weight += config.downstream_weight
            delta = 0.5
            absolute_ratio = abs(log_ratio)
            robust = (
                0.5 * log_ratio**2
                if absolute_ratio <= delta
                else delta * (absolute_ratio - 0.5 * delta)
            )
            loss += weight * robust
        return float(loss)

    lower = np.log(evidence["adaptive_min_scale"].to_numpy(float))
    upper = np.log(evidence["adaptive_max_scale"].to_numpy(float))
    result = minimize(
        objective,
        np.zeros(len(candidate_positions), dtype=float),
        method="L-BFGS-B",
        bounds=list(zip(lower, upper)),
        options={"maxiter": int(config.max_iterations), "ftol": 1e-12, "gtol": 1e-8},
    )
    scales = np.ones(len(frame), dtype=float)
    if result.success and np.isfinite(result.fun):
        scales[candidate_positions] = np.exp(result.x)
        status = "optimized"
    else:
        status = "unresolved_retain_raw_flag"

    frame["comparison_scale_factor"] = scales
    frame["comparison_routing_width"] = frame["wid_adj_wet"].astype(float) * scales
    for column in evidence.columns:
        frame.loc[candidate_mask, column] = evidence[column].to_numpy()
    frame.loc[candidate_mask, "comparison_status"] = status
    adaptive_flag = candidate_mask & (
        (scales < config.normal_min_scale - 1e-12)
        | (scales > config.normal_max_scale + 1e-12)
    )
    frame["comparison_flag"] = adaptive_flag
    frame.loc[adaptive_flag, "comparison_status"] = "adaptive_partial_correction_flag"
    frame["comparison_objective"] = float(result.fun) if np.isfinite(result.fun) else np.nan
    return frame, {
        "success": bool(result.success),
        "message": str(result.message),
        "iterations": int(getattr(result, "nit", 0)),
        "objective": float(result.fun) if np.isfinite(result.fun) else None,
        "candidate_count": int(len(candidate_positions)),
        "adaptive_flag_count": int(adaptive_flag.sum()),
    }
