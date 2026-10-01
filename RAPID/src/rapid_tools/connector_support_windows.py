"""Read-only physical-support diagnostics for unassigned connector links.

The helper is deliberately diagnostic-only.  It evaluates the frozen
longitudinal profile immediately upstream and downstream of one connector;
it neither changes node heads nor writes hydraulic products.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd


DEFAULT_SUPPORT_WINDOWS_M: tuple[float, ...] = (1_000.0, 2_000.0, 5_000.0)


def audit_connector_physical_support(
    profile: pd.DataFrame,
    *,
    upstream_x_m: float,
    downstream_x_m: float,
    connector_length_m: float,
    connector_slope_m_per_m: float,
    windows_m: Iterable[float] = DEFAULT_SUPPORT_WINDOWS_M,
    factor_limit: float = 5.0,
    distance_column: str = "reference_section_dist_out_m",
    wse_column: str = "joint_monotone_wse_m",
) -> dict[str, float | bool | str]:
    """Measure profile slope in bounded windows either side of a connector.

    ``x`` is distance to outlet, hence it increases upstream.  The connector
    runs from ``upstream_x_m`` to ``downstream_x_m``.  Each side is measured
    independently over a physical window and the connector itself is excluded.
    The returned physical target is a geometric median, appropriate for the
    multiplicative factor screen.  It is a diagnostic target only.
    """
    if factor_limit <= 1.0:
        raise ValueError("factor_limit must exceed one.")
    if connector_length_m <= 0.0:
        raise ValueError("connector_length_m must be positive.")
    if upstream_x_m <= downstream_x_m:
        raise ValueError("Expected upstream_x_m to exceed downstream_x_m.")

    table = profile[[distance_column, wse_column]].dropna().sort_values(distance_column)
    x = table[distance_column].to_numpy(float)
    h = table[wse_column].to_numpy(float)
    if len(x) < 2 or np.any(np.diff(x) <= 0.0):
        raise ValueError("Profile coordinates must be unique and strictly increasing.")

    record: dict[str, float | bool | str] = {
        "profile_domain_min_x_m": float(x[0]),
        "profile_domain_max_x_m": float(x[-1]),
        "connector_profile_slope_m_per_m": float(connector_slope_m_per_m),
        "connector_length_m": float(connector_length_m),
    }
    support_slopes: list[float] = []
    expected = 0
    for window in windows_m:
        window = float(window)
        if window <= 0.0:
            raise ValueError("Support windows must be positive.")
        expected += 2
        key = f"{int(round(window))}m"
        up_end = upstream_x_m + window
        down_start = downstream_x_m - window
        up_valid = up_end <= x[-1]
        down_valid = down_start >= x[0]
        up_slope = (
            float((np.interp(up_end, x, h) - np.interp(upstream_x_m, x, h)) / window)
            if up_valid
            else np.nan
        )
        down_slope = (
            float((np.interp(downstream_x_m, x, h) - np.interp(down_start, x, h)) / window)
            if down_valid
            else np.nan
        )
        record[f"upstream_support_slope_{key}"] = up_slope
        record[f"downstream_support_slope_{key}"] = down_slope
        record[f"upstream_support_available_{key}"] = bool(up_valid)
        record[f"downstream_support_available_{key}"] = bool(down_valid)
        support_slopes.extend(value for value in (up_slope, down_slope) if np.isfinite(value) and value > 0.0)

    record["n_positive_support_slopes"] = len(support_slopes)
    record["all_physical_support_windows_available"] = len(support_slopes) == expected
    if not support_slopes:
        record.update(
            {
                "physical_support_target_slope_m_per_m": np.nan,
                "physical_support_spread_factor": np.nan,
                "connector_factor_from_physical_support": np.nan,
                "factor_limit_slope_m_per_m": np.nan,
                "head_adjustment_to_factor_limit_m": np.nan,
                "physical_support_status": "insufficient_profile_support",
            }
        )
        return record

    slopes = np.asarray(support_slopes, dtype=float)
    target = float(np.exp(np.median(np.log(slopes))))
    spread = float(slopes.max() / slopes.min())
    connector_factor = (
        float(max(connector_slope_m_per_m / target, target / connector_slope_m_per_m))
        if np.isfinite(connector_slope_m_per_m) and connector_slope_m_per_m > 0.0
        else np.inf
    )
    lower, upper = target / factor_limit, target * factor_limit
    limited_slope = float(np.clip(connector_slope_m_per_m, lower, upper))
    head_adjustment = float(abs(limited_slope - connector_slope_m_per_m) * connector_length_m)
    if len(support_slopes) != expected:
        status = "insufficient_profile_support"
    elif spread > factor_limit:
        status = "heterogeneous_physical_support"
    elif connector_factor <= factor_limit:
        status = "within_physical_factor_band"
    else:
        status = "isolated_mismatch_candidate"
    record.update(
        {
            "physical_support_target_slope_m_per_m": target,
            "physical_support_spread_factor": spread,
            "connector_factor_from_physical_support": connector_factor,
            "factor_limit_slope_m_per_m": limited_slope,
            "head_adjustment_to_factor_limit_m": head_adjustment,
            "physical_support_status": status,
        }
    )
    return record
