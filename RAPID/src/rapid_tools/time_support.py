"""Explicit time support for interval means and point-observation comparisons.

An integration step, a forcing bin, a reporting bin and a gauge timestamp are
separate concepts. These helpers do not change historical routing or labels.
All numeric times use seconds on one caller-declared UTC origin.
"""
import numpy as np


def interval_bounds_and_midpoints(starts, duration):
    starts = np.asarray(starts, dtype=float)
    if starts.ndim != 1 or not np.isfinite(starts).all() or np.any(np.diff(starts) <= 0):
        raise ValueError('Interval starts must be finite and strictly increasing.')
    if not np.isfinite(duration) or duration <= 0:
        raise ValueError('Interval duration must be positive.')
    bounds = np.column_stack((starts, starts + duration))
    return bounds, starts + duration / 2


def point_series_interval_means(times, values, edges, *, allow_endpoint_hold=False,
                                max_gap_seconds=None, max_endpoint_hold_seconds=None):
    """Integrate a piecewise-linear point series exactly over explicit bins.

    Works for irregular observations without weighting densely sampled periods
    more heavily. Interpolation across observation gaps is an explicit modelling
    assumption: max_gap_seconds rejects intersecting gaps above the limit. Endpoint
    extrapolation is rejected unless constant holding is explicitly requested.
    """
    t = np.asarray(times, dtype=float)
    q = np.asarray(values, dtype=float)
    e = np.asarray(edges, dtype=float)
    if (t.ndim != 1 or q.shape != t.shape or len(t) < 2 or
            not np.isfinite(t).all() or not np.isfinite(q).all() or np.any(np.diff(t) <= 0)):
        raise ValueError('Point series needs finite values and strictly increasing times.')
    if e.ndim != 1 or len(e) < 2 or not np.isfinite(e).all() or np.any(np.diff(e) <= 0):
        raise ValueError('Bin edges must be finite and strictly increasing.')
    if not allow_endpoint_hold and (e[0] < t[0] or e[-1] > t[-1]):
        raise ValueError('Bin edges extend beyond observed time support.')
    for limit in (max_gap_seconds, max_endpoint_hold_seconds):
        if limit is not None and (not np.isfinite(limit) or limit <= 0):
            raise ValueError('Support limits must be finite and positive.')
    if max_gap_seconds is not None:
        intersects = (t[:-1] < e[-1]) & (t[1:] > e[0])
        if np.any(intersects & (np.diff(t) > max_gap_seconds)):
            raise ValueError('Requested bins intersect an excessive observation gap.')
    if max_endpoint_hold_seconds is not None:
        if max(t[0] - e[0], e[-1] - t[-1]) > max_endpoint_hold_seconds:
            raise ValueError('Requested endpoint hold exceeds its declared limit.')
    if e[0] < t[0]:
        t = np.r_[e[0], t]; q = np.r_[q[0], q]
    if e[-1] > t[-1]:
        t = np.r_[t, e[-1]]; q = np.r_[q, q[-1]]
    widths = np.diff(t)
    slopes = np.diff(q) / widths
    cumulative = np.r_[0., np.cumsum(widths * (q[:-1] + q[1:]) / 2)]
    idx = np.clip(np.searchsorted(t, e, side='right') - 1, 0, len(t) - 2)
    offset = e - t[idx]
    integral = cumulative[idx] + q[idx] * offset + slopes[idx] * offset**2 / 2
    return np.diff(integral) / np.diff(e)


def native_outlet_at_times(native_flow, dt, sample_times, *, initial_flow):
    """Interpolate solver endpoint states at gauge point timestamps.

    native_flow[0] is at dt, NOT at zero. initial_flow is the state at zero.
    No extrapolation or inferred shift of the requested timestamps is allowed.
    """
    q = np.asarray(native_flow, dtype=float)
    s = np.asarray(sample_times, dtype=float)
    if (q.ndim != 1 or len(q) == 0 or not np.isfinite(q).all() or
            not np.isfinite(dt) or dt <= 0 or not np.isfinite(initial_flow)):
        raise ValueError('Invalid native flow or clock.')
    if s.ndim != 1 or not np.isfinite(s).all() or np.any(s < 0) or np.any(s > len(q)*dt):
        raise ValueError('Requested samples exceed native time support.')
    return np.interp(s, np.arange(len(q)+1)*dt, np.r_[initial_flow, q])


def native_outlet_interval_means(native_flow, dt, bounds, *, initial_flow):
    """Compare with interval observations using their exact, possibly gapped bounds.

    Integrates the piecewise-linear reconstruction through the initial state and
    native endpoints. This is NOT the historical arithmetic endpoint-state mean.
    Observation intervals may overlap; no extrapolation is allowed.
    """
    b = np.asarray(bounds, dtype=float)
    if (b.ndim != 2 or b.shape[1] != 2 or not np.isfinite(b).all()
            or np.any(b[:, 1] <= b[:, 0])):
        raise ValueError('Observation bounds must be finite positive-width pairs.')
    # Reuse clock/state validation, including support of all requested bounds.
    native_outlet_at_times(native_flow, dt, b.ravel(), initial_flow=initial_flow)
    if len(b) == 0:
        return np.empty(0, dtype=float)
    edges = np.unique(b)
    means = point_series_interval_means(
        np.arange(len(native_flow)+1)*dt, np.r_[initial_flow, native_flow], edges)
    integral = np.r_[0., np.cumsum(means * np.diff(edges))]
    indices = np.searchsorted(edges, b)
    return (integral[indices[:, 1]] - integral[indices[:, 0]]) / np.diff(b).ravel()
