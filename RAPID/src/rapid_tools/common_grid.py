"""One interval-mean representation for gauges, forcing and routed outlets.

All times are seconds relative to one explicit UTC origin. Raw observations stay
immutable. Gap interpolation is available for diagnostics, but is always flagged.
"""
import numpy as np

from .time_support import point_series_interval_means


def resample_points(times, values, edges, *, maximum_gap_seconds,
                    endpoint_hold_seconds=0., diagnostic_gap_fill=False):
    """Integrate point observations and return means plus support flags.

    Endpoint holding is bounded independently of interior gaps. A bin touching an
    excessive gap is rejected unless diagnostic_gap_fill is explicitly enabled.
    """
    t, e = np.asarray(times, float), np.asarray(edges, float)
    if not np.isfinite(maximum_gap_seconds) or maximum_gap_seconds <= 0:
        raise ValueError('A finite positive observation-gap limit is required.')
    if not np.isfinite(endpoint_hold_seconds) or endpoint_hold_seconds < 0:
        raise ValueError('Endpoint holding must have a finite nonnegative limit.')
    means = point_series_interval_means(
        t, values, e, allow_endpoint_hold=endpoint_hold_seconds > 0,
        max_endpoint_hold_seconds=endpoint_hold_seconds or None,
        max_gap_seconds=None if diagnostic_gap_fill else maximum_gap_seconds)
    gap = np.zeros(len(e)-1, dtype=bool)
    for i in np.flatnonzero(np.diff(t) > maximum_gap_seconds):
        gap |= (e[:-1] < t[i+1]) & (e[1:] > t[i])
    endpoint = (e[:-1] < t[0]) | (e[1:] > t[-1])
    return means, gap, endpoint


def outlet_interval_means(native, nsub, *, initial_flow):
    """Exact trapezoidal means of the linear native-state reconstruction.

    Only the outlet trace is needed. The correction converts the historical
    endpoint rectangle mean into the integral over each complete report interval.
    """
    q = np.asarray(native, float)
    if (q.ndim != 1 or not len(q) or not np.isfinite(q).all()
            or int(nsub) != nsub or nsub < 1 or len(q) % int(nsub)
            or not np.isfinite(initial_flow)):
        raise ValueError('Invalid native outlet states or interval grouping.')
    blocks = q.reshape(-1, int(nsub))
    starts = np.r_[initial_flow, blocks[:-1, -1]]
    return blocks.mean(axis=1) + (starts - blocks[:, -1]) / (2*int(nsub))


def route_interval_means(frame, flow, nsub, dt, *, initial_flow):
    """Use the existing cascade with an explicit equilibrium initial condition.

    One constant initial bin makes its first-bin steady initialization independent
    of the first *event* bin mean. That bin is removed from returned physical time.
    No native trace is written to disk and the historical router is unchanged.
    """
    from .cunge import route
    if not np.isfinite(initial_flow) or initial_flow < 0:
        raise ValueError('Initial discharge must be finite and nonnegative.')
    native, audit = route(frame, np.r_[initial_flow, flow], nsub, dt, return_native=True)
    np.testing.assert_allclose(native[:nsub], initial_flow, rtol=1e-10, atol=1e-9)
    event = native[nsub:]
    means = outlet_interval_means(event, nsub, initial_flow=initial_flow)
    return means, dict(audit, final_outlet_cms=float(event[-1]), initial_outlet_cms=float(initial_flow))
