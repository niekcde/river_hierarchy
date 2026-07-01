from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True, slots=True)
class KValueConfig:
    width_field: str = "wid_adj_wet"
    x_value: float = 0.1
    kb_value: float = 20.0
    kb_source_method: str = "fixed_scalar"
    kb_reference_discharge_cms: float | None = None
    kb_reference_width_m: float | None = None
    kb_reference_slope: float | None = None
    kb_reference_depth_m: float | None = None
    n_manning: float = 0.35
    min_width: float = 1.0
    min_effective_length_m: float | None = None
    use_celerity_capping: bool = False
    min_celerity_mps: float = 0.28
    max_celerity_mps: float = 1.524


WIDTH_FIELD_CANDIDATES = (
    "wid_adj_wet",
    "wid_adj_total",
    "wid_adj",
    "wid_wet",
    "wid",
)


def resolve_width_field(
    columns: pd.Index,
    preferred_field: str,
) -> str:
    if preferred_field in columns:
        return preferred_field
    for column in WIDTH_FIELD_CANDIDATES:
        if column in columns:
            return column
    raise ValueError(
        "Could not resolve a width field for RAPID K-value preparation. "
        f"Preferred field '{preferred_field}' was not found."
    )


def compute_k_values(
    link_frame: pd.DataFrame,
    *,
    config: KValueConfig | None = None,
) -> pd.DataFrame:
    config = config or KValueConfig()
    if config.n_manning <= 0:
        raise ValueError("RAPID K-value preparation requires a positive Manning roughness.")
    if config.kb_value <= 0:
        raise ValueError("RAPID K-value preparation requires a positive kb value.")
    if config.min_width <= 0:
        raise ValueError("RAPID K-value preparation requires a positive minimum width.")
    if config.min_effective_length_m is not None and config.min_effective_length_m <= 0:
        raise ValueError("RAPID effective-length floor must be positive when provided.")
    if config.min_celerity_mps <= 0 or config.max_celerity_mps <= 0:
        raise ValueError("RAPID celerity caps must be positive when provided.")
    if config.min_celerity_mps > config.max_celerity_mps:
        raise ValueError("RAPID celerity cap minimum cannot exceed the maximum.")

    width_field = resolve_width_field(link_frame.columns, config.width_field)

    frame = link_frame.copy()
    frame["rapid_width_source_field"] = width_field
    frame["rapid_width_m_raw"] = pd.to_numeric(frame[width_field], errors="coerce")
    frame["rapid_width_adjusted"] = frame["rapid_width_m_raw"].isna() | frame["rapid_width_m_raw"].le(0)
    frame["rapid_width_m"] = frame["rapid_width_m_raw"].where(
        ~frame["rapid_width_adjusted"],
        config.min_width,
    )
    frame["rapid_width_m"] = frame["rapid_width_m"].clip(lower=config.min_width).astype(float)

    frame["rapid_x"] = float(config.x_value)
    frame["rapid_kb_value"] = float(config.kb_value)
    frame["rapid_kb_source_method"] = str(config.kb_source_method)
    frame["rapid_kb_reference_discharge_cms"] = (
        float(config.kb_reference_discharge_cms) if config.kb_reference_discharge_cms is not None else np.nan
    )
    frame["rapid_kb_reference_width_m"] = (
        float(config.kb_reference_width_m) if config.kb_reference_width_m is not None else np.nan
    )
    frame["rapid_kb_reference_slope"] = (
        float(config.kb_reference_slope) if config.kb_reference_slope is not None else np.nan
    )
    frame["rapid_kb_reference_depth_m"] = (
        float(config.kb_reference_depth_m) if config.kb_reference_depth_m is not None else np.nan
    )
    frame["rapid_celerity_mps_raw"] = (
        (5.0 / (3.0 * float(config.n_manning)))
        * np.sqrt(frame["slope_used"].astype(float))
        * (
            (frame["rapid_width_m"].astype(float) ** (2.0 / 3.0))
            / (float(config.kb_value) ** (2.0 / 3.0))
        )
    ).astype(float)
    if config.use_celerity_capping:
        frame["rapid_celerity_mps"] = frame["rapid_celerity_mps_raw"].clip(
            lower=float(config.min_celerity_mps),
            upper=float(config.max_celerity_mps),
        )
        frame["rapid_celerity_capped"] = ~np.isclose(
            frame["rapid_celerity_mps"],
            frame["rapid_celerity_mps_raw"],
            rtol=1e-12,
            atol=1e-12,
        )
        frame["rapid_k_source_method"] = np.where(
            frame["rapid_celerity_capped"],
            "hydraulic_celerity_capped",
            "hydraulic_celerity_uncapped",
        )
    else:
        frame["rapid_celerity_mps"] = frame["rapid_celerity_mps_raw"].astype(float)
        frame["rapid_celerity_capped"] = False
        frame["rapid_k_source_method"] = "hydraulic_celerity_uncapped"

    frame["rapid_celerity_cap_enabled"] = bool(config.use_celerity_capping)
    frame["rapid_celerity_cap_min_mps"] = float(config.min_celerity_mps)
    frame["rapid_celerity_cap_max_mps"] = float(config.max_celerity_mps)
    frame["rapid_effective_length_floor_m"] = (
        float(config.min_effective_length_m) if config.min_effective_length_m is not None else np.nan
    )
    if config.min_effective_length_m is not None:
        frame["rapid_effective_length_m"] = frame["link_length_m"].astype(float).clip(
            lower=float(config.min_effective_length_m)
        )
    else:
        frame["rapid_effective_length_m"] = frame["link_length_m"].astype(float)
    frame["rapid_length_floor_applied"] = ~np.isclose(
        frame["rapid_effective_length_m"],
        frame["link_length_m"].astype(float),
        rtol=1e-12,
        atol=1e-12,
    )
    frame["rapid_k"] = (frame["rapid_effective_length_m"].astype(float) / frame["rapid_celerity_mps"]).astype(float)
    return frame


def compute_dt_bounds(
    k_values: pd.Series,
    *,
    x_value: float,
) -> tuple[float, float]:
    finite = pd.to_numeric(k_values, errors="coerce")
    finite = finite[np.isfinite(finite)]
    if finite.empty:
        raise ValueError("No finite RAPID K values were available for routing dt selection.")
    dt_min = float(np.max(2.0 * finite * x_value))
    dt_max = float(np.min(2.0 * finite * (1.0 - x_value)))
    return dt_min, dt_max


def apply_stability_effective_length_cap(
    link_frame: pd.DataFrame,
    *,
    x_value: float,
    min_celerity_mps: float,
    safety_margin_seconds: float,
) -> pd.DataFrame:
    if x_value <= 0:
        raise ValueError("Muskingum X must be positive when applying the stability length cap.")
    if min_celerity_mps <= 0:
        raise ValueError("Minimum celerity must be positive when applying the stability length cap.")
    if safety_margin_seconds < 0:
        raise ValueError("Stability length-cap safety margin must be nonnegative.")

    frame = link_frame.copy()
    current_effective_length = pd.to_numeric(frame["rapid_effective_length_m"], errors="coerce").astype(float)
    floor_length = pd.to_numeric(frame.get("rapid_effective_length_floor_m"), errors="coerce")
    floor_length = floor_length.fillna(-np.inf).astype(float)

    frame["rapid_stability_length_cap_enabled"] = True
    frame["rapid_stability_length_cap_margin_seconds"] = float(safety_margin_seconds)
    frame["rapid_effective_length_m_pre_stability_cap"] = current_effective_length
    frame["rapid_stability_length_cap_candidate"] = False
    frame["rapid_stability_length_cap_applied"] = False
    frame["rapid_stability_length_cap_triggered"] = False
    frame["rapid_stability_length_cap_target_m"] = np.nan
    frame["rapid_stability_length_cap_active_m"] = np.nan

    dt_min_before, dt_max_before = compute_dt_bounds(frame["rapid_k"], x_value=x_value)
    frame["rapid_stability_length_cap_dt_min_before_seconds"] = float(dt_min_before)
    frame["rapid_stability_length_cap_dt_max_before_seconds"] = float(dt_max_before)

    target_upper_dt = dt_max_before - float(safety_margin_seconds)
    if dt_min_before <= target_upper_dt + 1e-9 or target_upper_dt <= 0:
        frame["rapid_stability_length_cap_dt_min_after_seconds"] = float(dt_min_before)
        frame["rapid_stability_length_cap_dt_max_after_seconds"] = float(dt_max_before)
        return frame

    target_cap = float(target_upper_dt * min_celerity_mps / (2.0 * x_value))
    if not np.isfinite(target_cap) or target_cap <= 0:
        frame["rapid_stability_length_cap_dt_min_after_seconds"] = float(dt_min_before)
        frame["rapid_stability_length_cap_dt_max_after_seconds"] = float(dt_max_before)
        return frame

    active_cap = np.maximum(target_cap, floor_length)
    candidate_mask = (
        pd.to_numeric(frame["rapid_celerity_mps"], errors="coerce")
        .astype(float)
        .pipe(lambda s: np.isclose(s, float(min_celerity_mps), rtol=1e-12, atol=1e-12))
        & pd.to_numeric(frame.get("rapid_celerity_capped"), errors="coerce").fillna(False).astype(bool)
        & current_effective_length.gt(active_cap)
    )

    frame["rapid_stability_length_cap_triggered"] = True
    frame["rapid_stability_length_cap_target_m"] = float(target_cap)
    frame["rapid_stability_length_cap_active_m"] = active_cap.astype(float)
    frame["rapid_stability_length_cap_candidate"] = candidate_mask

    if candidate_mask.any():
        updated_effective_length = current_effective_length.where(~candidate_mask, active_cap)
        frame["rapid_effective_length_m"] = updated_effective_length.astype(float)
        frame["rapid_stability_length_cap_applied"] = candidate_mask
        frame["rapid_k"] = (frame["rapid_effective_length_m"].astype(float) / frame["rapid_celerity_mps"]).astype(float)
        frame["rapid_k_source_method"] = np.where(
            frame["rapid_stability_length_cap_applied"],
            frame["rapid_k_source_method"].astype(str) + "+stability_length_cap",
            frame["rapid_k_source_method"],
        )

    dt_min_after, dt_max_after = compute_dt_bounds(frame["rapid_k"], x_value=x_value)
    frame["rapid_stability_length_cap_dt_min_after_seconds"] = float(dt_min_after)
    frame["rapid_stability_length_cap_dt_max_after_seconds"] = float(dt_max_after)
    return frame


def compute_routing_dt_seconds(
    k_values: pd.Series,
    *,
    x_value: float,
    forcing_dt_seconds: int,
) -> int:
    dt_min, dt_max = compute_dt_bounds(k_values, x_value=x_value)
    if dt_min > dt_max:
        raise ValueError(
            f"No stable RAPID routing timestep exists for x={x_value}. "
            f"Computed dt_min={dt_min:.3f} > dt_max={dt_max:.3f}."
        )

    candidates: list[int] = []
    for candidate in range(1, int(forcing_dt_seconds) + 1):
        if forcing_dt_seconds % candidate != 0:
            continue
        if candidate + 1e-9 < dt_min or candidate - 1e-9 > dt_max:
            continue
        candidates.append(candidate)

    if not candidates:
        raise ValueError(
            "Could not find a routing timestep that both satisfies the Muskingum "
            f"stability bounds [{dt_min:.3f}, {dt_max:.3f}] and evenly divides "
            f"the forcing timestep {forcing_dt_seconds}."
        )

    midpoint = 0.5 * (dt_min + dt_max)
    best = min(candidates, key=lambda value: (abs(value - midpoint), -value))
    return int(best)
