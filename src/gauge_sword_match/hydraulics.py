from __future__ import annotations

import numpy as np
import pandas as pd


def compute_reference_depth(width_m: float, kb: float) -> float:
    if not np.isfinite(width_m) or not np.isfinite(kb) or width_m <= 0 or kb <= 0:
        return np.nan
    return float(width_m / kb)


def compute_reference_area(width_m: float, depth_m: float) -> float:
    if not np.isfinite(width_m) or not np.isfinite(depth_m) or width_m <= 0 or depth_m <= 0:
        return np.nan
    return float(width_m * depth_m)


def compute_reference_velocity(q0: float, area_m2: float) -> float:
    if not np.isfinite(q0) or not np.isfinite(area_m2) or q0 < 0 or area_m2 <= 0:
        return np.nan
    return float(q0 / area_m2)


def compute_froude(v0: float, depth_m: float, gravity_m_s2: float = 9.80665) -> float:
    if (
        not np.isfinite(v0)
        or not np.isfinite(depth_m)
        or not np.isfinite(gravity_m_s2)
        or depth_m <= 0
        or gravity_m_s2 <= 0
    ):
        return np.nan
    return float((v0**2) / (gravity_m_s2 * depth_m))


def compute_tplus(t0_hours: float, v0: float, slope: float, depth_m: float) -> float:
    if not np.isfinite(t0_hours) or not np.isfinite(v0) or not np.isfinite(slope) or not np.isfinite(depth_m):
        return np.nan
    if t0_hours <= 0 or v0 < 0 or slope <= 0 or depth_m <= 0:
        return np.nan
    return float((t0_hours * 3600.0 * v0 * slope) / depth_m)


def get_q0_value(row: pd.Series, method: str) -> float:
    if method == "pre_event_median":
        return float(row.get("q0_pre_event_median", np.nan))
    if method == "event_start_discharge":
        return float(row.get("q0_event_start_discharge", np.nan))
    raise ValueError(f"Unsupported q0 method: {method}")


def get_t0_hours(row: pd.Series, method: str) -> float:
    if method == "rise_t10_t90":
        return float(row.get("t0_rise_t10_t90_hours", np.nan))
    if method == "rise_start_to_peak":
        return float(row.get("t0_rise_start_to_peak_hours", np.nan))
    raise ValueError(f"Unsupported t0 method: {method}")


def classify_kinematic_candidate(
    froude: float,
    tplus: float,
    *,
    regime_tplus_min: float,
    regime_froude_t0: float,
    regime_tplus_end: float,
    regime_froude_end: float,
) -> bool | None:
    if not np.isfinite(froude) or not np.isfinite(tplus):
        return None
    if tplus < regime_tplus_min:
        return False
    if regime_tplus_end <= regime_tplus_min:
        return bool(froude <= regime_froude_t0)
    slope = (regime_froude_end - regime_froude_t0) / (regime_tplus_end - regime_tplus_min)
    froude_limit = regime_froude_t0 + slope * (tplus - regime_tplus_min)
    return bool(froude <= froude_limit)
