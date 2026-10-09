"""Constant-reference Muskingum--Cunge on frozen, anchored hydraulic links.

The grid is fixed across a Manning-n search. Alpha is a spatial accuracy
setting, not a fitted hydraulic parameter or an unconditional positivity bound.
This prep entry point consumes the reviewed link frames; it does not rebuild
their geometry, reference depth, slopes or collapsed-state boundaries.
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
from scipy.io import netcdf_file
from scipy.signal import lfilter

from .k_values import compute_dt_bounds


def topology(frame):
    graph = nx.DiGraph()
    edges = {}
    for i, row in enumerate(frame.itertuples()):
        graph.add_edge(row.u, row.v)
        edges.setdefault(row.u, []).append(i)
    if not nx.is_directed_acyclic_graph(graph):
        raise ValueError("Cunge routing requires a directed acyclic network.")
    roots = [u for u in graph if graph.in_degree(u) == 0]
    ends = [u for u in graph if graph.out_degree(u) == 0]
    if len(roots) != 1 or len(ends) != 1:
        raise ValueError("Expected one inlet node and one outlet node.")
    return list(nx.topological_sort(graph)), edges, roots[0], ends[0]


def freeze_grid(frame, reference_discharge_cms, *, n_bounds=(.01, .1), alpha=2.,
                min_celerity=.05, max_celerity=5.):
    """Allocate Q once and choose exact-length cells at the fastest trial n."""
    f = frame.reset_index(drop=True).copy()
    required = ['link_length_m', 'width', 'slope', 'depth_m']
    if not np.isfinite(f[required].to_numpy()).all() or (f[required] <= 0).any().any():
        raise ValueError("Length, width, slope and depth must be finite and positive.")
    if f.id_link.duplicated().any():
        raise ValueError("Hydraulic link IDs must be unique.")
    if not (0 < n_bounds[0] <= n_bounds[1] and np.isfinite(n_bounds).all()):
        raise ValueError("Invalid Manning calibration bounds.")
    if not (np.isfinite(alpha) and alpha > 0 and np.isfinite(reference_discharge_cms)
            and reference_discharge_cms > 0 and 0 < min_celerity <= max_celerity
            and np.isfinite([min_celerity, max_celerity]).all()):
        raise ValueError("Invalid Cunge reference discharge, alpha or celerity caps.")
    order, edges, root, end = topology(f)
    shares = {root: 1.}
    fractions = np.zeros(len(f)); allocation = np.zeros(len(f))
    for u in order:
        ids = edges.get(u, [])
        width = f.iloc[ids].width.sum()
        for i in ids:
            fractions[i] = f.iloc[i].width / width
            allocation[i] = shares[u] * fractions[i]
            v = f.iloc[i].v
            shares[v] = shares.get(v, 0.) + allocation[i]
    if not np.isclose(shares[end], 1., atol=1e-12, rtol=0):
        raise ValueError("Reference-flow allocation does not conserve flow.")
    f['split_fraction'] = fractions
    f['reference_flow_fraction'] = allocation
    f['reference_discharge_cms'] = reference_discharge_cms * allocation
    f['celerity_times_n'] = (5./3.) * f.depth_m**(2./3.) * np.sqrt(f.slope)
    fast = np.clip(f.celerity_times_n / n_bounds[0], min_celerity, max_celerity)
    ld = f.reference_discharge_cms / (f.width * f.slope * fast)
    f['count'] = np.maximum(1, np.ceil(f.link_length_m / (alpha*ld))).astype(int)
    f['rapid_effective_length_m'] = f.link_length_m / f['count']
    f['cunge_alpha'] = float(alpha)
    f['grid_n_min'] = float(n_bounds[0]); f['grid_n_max'] = float(n_bounds[1])
    f['celerity_min_mps'] = float(min_celerity); f['celerity_max_mps'] = float(max_celerity)
    f['rapid_length_floor_applied'] = False
    f['rapid_effective_length_floor_m'] = np.nan
    return f


def parameters(grid, n, *, method='cunge', beta=1.):
    """Update c, K and X together; all geometry, Q and counts stay frozen."""
    if not np.isfinite(n) or not grid.grid_n_min.iloc[0] <= n <= grid.grid_n_max.iloc[0]:
        raise ValueError("Manning n is outside the frozen-grid calibration bounds.")
    if method not in {'cunge', 'fixed_x'}:
        raise ValueError("Unknown routing method.")
    if not np.isfinite(beta) or beta <= 0 or (method != 'cunge' and beta != 1.):
        raise ValueError("Positive beta is supported for Cunge only.")
    f = grid.copy()
    raw = f.celerity_times_n / n
    c = np.clip(raw, f.celerity_min_mps, f.celerity_max_mps)
    dx = f.rapid_effective_length_m
    ld = f.reference_discharge_cms / (f.width*f.slope*c)
    f['rapid_celerity_mps_raw'] = raw
    f['rapid_celerity_mps'] = c
    f['rapid_celerity_capped'] = ~np.isclose(c, raw, rtol=1e-12, atol=1e-12)
    f['rapid_k'] = dx/c
    f['rapid_x'] = .5*(1-ld/dx) if method == 'cunge' else .1
    f['rapid_x'] = .5-beta*(.5-f.rapid_x) if method == 'cunge' else f.rapid_x
    f['diffusion_multiplier'] = float(beta)
    f['hydraulic_length_scale_m'] = ld
    f['hydraulic_diffusivity_m2_s'] = f.reference_discharge_cms/(2*f.width*f.slope)
    f['routing_method'] = method
    f['n_manning'] = float(n)
    if (dx > f.cunge_alpha*ld*(1+1e-12)).any():
        raise ValueError("Frozen grid violates its alpha bound.")
    return f


def common_clock(grid, forcing_step_s, *, max_dt_s=15.):
    """One clock valid for both compared methods throughout the n range."""
    bounds = []
    for method in ['cunge', 'fixed_x']:
        f = parameters(grid, grid.grid_n_min.iloc[0], method=method)
        bounds.append(compute_dt_bounds(f.rapid_k, x_value=f.rapid_x,
                                      coefficient_policy='signed_inflow')[1])
    nsub = int(math.ceil(forcing_step_s / min(max_dt_s, .99*min(bounds))))
    return forcing_step_s/nsub, nsub


def route(frame, flow, nsub, dt, *, return_native=False):
    """Exact DAG cascade with two-time inlet and held forcing, steady start.

    Audits every native subsection. Negative flows are exposed, never clipped;
    callers must reject invalid trials before calibration scoring/publication.
    """
    flow = np.asarray(flow, dtype=float)
    if flow.ndim != 1 or len(flow) < 2 or not np.isfinite(flow).all() or (flow < 0).any():
        raise ValueError("Expected finite nonnegative inflow samples.")
    if int(nsub) != nsub or nsub < 1 or not np.isfinite(dt) or dt <= 0:
        raise ValueError("Invalid routing clock.")
    f = frame.reset_index(drop=True)
    _, upper = compute_dt_bounds(f.rapid_k, x_value=f.rapid_x, coefficient_policy='signed_inflow')
    if dt > upper*(1+1e-12):
        raise ValueError("Routing timestep violates the previous-outflow bound.")
    k = f.rapid_k.to_numpy(); x = f.rapid_x.to_numpy(); den = dt+2*k*(1-x)
    a = (dt-2*k*x)/den; b = (dt+2*k*x)/den; d = (2*k*(1-x)-dt)/den
    order, edges, root, end = topology(f)
    signals = {root: np.repeat(flow, nsub)}
    minimum = np.inf; dip = 0.
    amplitude = float(np.ptp(flow))
    nonnegative_pulse = bool(np.all(flow >= flow[0]))
    for u in order:
        sig = signals.pop(u)
        if u == end:
            return (sig if return_native else sig.reshape(-1, nsub).mean(axis=1)), dict(
                minimum_native_flow_cms=minimum, maximum_pulse_dip_fraction=-dip if nonnegative_pulse else None,
                negative_current_inflow_coefficients=int((a<0).sum()),
                negative_previous_inflow_coefficients=int((b<0).sum()))
        for i in edges[u]:
            row = f.iloc[i]
            val = sig*row.split_fraction
            eq = flow[0]*row.reference_flow_fraction
            for _ in range(int(row['count'])):
                val, _ = lfilter([a[i], b[i]], [1., -d[i]], val, zi=[(b[i]+d[i])*eq])
                minimum = min(minimum, float(val.min()))
                if nonnegative_pulse and amplitude:
                    dip = min(dip, (float(val.min())-eq)/(amplitude*row.reference_flow_fraction))
            if not np.isfinite(val).all():
                raise ValueError("Nonfinite native discharge.")
            signals[row.v] = signals.get(row.v, 0.) + val
    raise ValueError("No outlet reached.")


def expand_links(frame):
    """Expand numerical cells; preserve original link IDs and anchor nodes."""
    order, by_node, root, end = topology(frame)
    virtual = min(-1, int(min(frame.u.min(), frame.v.min()))-1)
    rows = []
    for u in order:
        for i in by_node.get(u, []):
            r = frame.iloc[i]; start = r.u
            for j in range(int(r['count'])):
                finish = r.v if j == int(r['count'])-1 else virtual
                if finish == virtual: virtual -= 1
                rows.append(dict(reach_id=len(rows)+1, original_link_id=int(r.id_link),
                                 subreach_index=j+1, subreach_count=int(r['count']),
                                 id_us_node=int(start), id_ds_node=int(finish),
                                 link_length_m=float(r.rapid_effective_length_m),
                                 rapid_k=float(r.rapid_k), rapid_x=float(r.rapid_x),
                                 width=float(r.width), inlet_fraction=float(r.split_fraction) if u == root and j == 0 else 0.,
                                 is_outlet=bool(finish == end)))
                start = finish
    return pd.DataFrame(rows)


def write_prepared_state(frame, flow, forcing_step_s, routing_dt_s, output_dir, *, metadata=None,
                         output_averaging="endpoint_rectangle"):
    """Write runnable RAPID inputs directly from an audited hydraulic frame.

    CSV mapping preserves physical anchors and within-link cell positions;
    this numerical prep does not require GIS geometry or regenerate slopes.
    """
    if output_averaging not in {"endpoint_rectangle", "linear_state_interval_mean"}:
        raise ValueError("Unknown output averaging convention.")
    if forcing_step_s != int(forcing_step_s) or forcing_step_s <= 0:
        raise ValueError("NetCDF forcing step must be a positive integer number of seconds.")
    from .engine import stp_cor
    stp_cor(forcing_step_s, routing_dt_s)
    _, upper = compute_dt_bounds(frame.rapid_k, x_value=frame.rapid_x, coefficient_policy='signed_inflow')
    if routing_dt_s > upper:
        raise ValueError("Prepared clock exceeds its coefficient bound.")
    p = Path(output_dir); p.mkdir(parents=True, exist_ok=True)
    cells = expand_links(frame)
    outgoing = {u: g for u, g in cells.groupby('id_us_node', sort=False)}
    connections = []
    for r in cells.itertuples():
        kids = outgoing.get(r.id_ds_node)
        connections.append([] if kids is None else [(int(k.reach_id), float(k.width/kids.width.sum())) for k in kids.itertuples()])
    cols = max(1, max(map(len, connections)))
    with (p/'rat_srt.csv').open('w', newline='') as h:
        writer = csv.writer(h)
        for r, kids in zip(cells.itertuples(), connections):
            writer.writerow([r.reach_id]+[str(item) for item in kids]+['(0, 0.0)']*(cols-len(kids)))
    for name, col in [('riv','reach_id'),('kfc','rapid_k'),('xfc','rapid_x')]:
        cells[[col]].to_csv(p/f'{name}.csv', header=False, index=False)
    cells.to_csv(p/'rapid_link_attributes.csv', index=False)
    frame.to_parquet(p/'hydraulic_links.parquet', index=False)
    flow = np.asarray(flow, dtype=float)
    if len(flow) < 2 or not np.isfinite(flow).all() or (flow<0).any():
        raise ValueError("Invalid external forcing.")
    times = np.arange(len(flow), dtype=np.int64)*int(forcing_step_s)
    if times[-1] > np.iinfo(np.int32).max:
        raise ValueError("Relative simulation clock exceeds NetCDF integer range.")
    pd.DataFrame(dict(time_seconds=times, q_inflow_cms=flow)).to_csv(p/'forcing_normalized.csv', index=False)
    with netcdf_file(p/'inflow.nc', 'w') as ds:
        ds.createDimension('time', len(flow)); ds.createDimension('rivid', len(cells))
        ds.createVariable('time','i4',('time',))[:] = times
        ds.createVariable('rivid','i4',('rivid',))[:] = cells.reach_id.to_numpy()
        ds.createVariable('m3_riv','f8',('time','rivid'))[:] = flow[:,None]*cells.inlet_fraction.to_numpy()[None,:]*forcing_step_s
    manifest = dict(schema_version='cunge-reference-prep-v1', state_role='base',
                    prep_config=dict(routing_method=str(frame.routing_method.iloc[0]), cunge_alpha=float(frame.cunge_alpha.iloc[0]),
                                     n_manning=float(frame.n_manning.iloc[0]), coefficient_policy='signed_inflow',
                                     boundary_mode='upstream_two_time', initial_flow_mode='steady_first_inflow',
                                     network_dtype='float64', output_averaging=output_averaging,
                                     diffusion_multiplier=float(frame.get('diffusion_multiplier', pd.Series([1.])).iloc[0]),
                                     reference_parameters_frozen=True),
                    routing=dict(routing_dt_seconds=float(routing_dt_s), forcing_dt_seconds=int(forcing_step_s)),
                    original_links=len(frame), numerical_sections=len(cells),
                    physical_length_m=float(frame.link_length_m.sum()), represented_length_m=float(cells.link_length_m.sum()),
                    outlet_reach_ids=cells.loc[cells.is_outlet,'reach_id'].tolist(),
                    geometry_scope='Numerical link mapping only; original geometry remains in source anchored state.',
                    metadata=metadata or {})
    (p/'rapid_prep_manifest.json').write_text(json.dumps(manifest, indent=2))
    return manifest
