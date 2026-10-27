"""Reconstruct current Cunge reach waves, checked against the published outlet."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import cunge
from .common_grid import outlet_interval_means


def midpoint_discharge(subreach_qout):
    """Interpolate downstream-end observations at the parent-link midpoint.

    Columns must follow upstream-to-downstream order on equal-length cells.
    A one-cell link uses its sole observation (an outlet proxy for the midpoint).
    """
    q = np.asarray(subreach_qout)
    if q.ndim != 2 or q.shape[1] == 0:
        raise ValueError('Expected at least one ordered subreach discharge column')
    positions = np.arange(1, q.shape[1] + 1) / q.shape[1]
    right = min(int(np.searchsorted(positions, 0.5)), q.shape[1] - 1)
    left = max(0, right - 1)
    if left == right:
        return q[:, right]
    weight = (0.5 - positions[left]) / (positions[right] - positions[left])
    return (1 - weight) * q[:, left] + weight * q[:, right]


def wave_centroid(time_seconds, discharge, start, end, baseline):
    """Excess-volume centroid on a fixed window, using exact linear moments.

    A local flow-scaled baseline is lowered to the local window minimum if
    needed to keep excess nonnegative. Return the baseline actually used.
    """
    from .hydrograph_response import distribution
    t, q = np.asarray(time_seconds), np.asarray(discharge)
    if not t[0] <= start < end <= t[-1]:
        raise ValueError('Centroid window is outside discharge support')
    tt = np.r_[start, t[(t > start) & (t < end)], end]
    qq = np.interp(tt, t, q)
    used = min(float(baseline), float(qq.min()))
    return distribution(tt, qq - used)['centroid_seconds'], used


def load_cunge_wave_state(experiment_dir, state_id, *, reviewed_window=None):
    import geopandas as gpd
    from shapely.ops import linemerge, substring
    from .analysis import ReachWaveStack, _LoadedWaveState

    experiment = Path(experiment_dir)
    registry = pd.read_csv(experiment / 'rapid_run_registry.csv')
    selected = registry.loc[registry.state_id.eq(state_id)]
    if len(selected) != 1:
        raise ValueError(f'Expected exactly one registered state: {state_id}')
    row = selected.iloc[0].copy()
    if row.method != 'cunge' or row.status != 'ran':
        raise ValueError('Expected a completed current Cunge run')
    parameters_path = Path(row.parameters_parquet)
    frame = pd.read_parquet(parameters_path).reset_index(drop=True)
    if reviewed_window is None:
        forcing = pd.read_parquet(row.forcing_parquet)
        outlet = pd.read_parquet(row.outlet_parquet)
        result = json.loads(Path(row.result_json).read_text())
        if result['fingerprint'] != row.fingerprint:
            raise ValueError('Result fingerprint differs from registry')
    else:
        from .hydrograph_response import ROOT, load, select_pair, digest
        states, _ = load(reviewed_window)
        example = experiment.name.split('_')[0]
        if state_id == 'S000_base':
            metric = states[states.example.eq(example) & states.state.eq(state_id)].iloc[0]
        else:
            _, metric = select_pair(states, example, state_id)
        plan = json.loads((ROOT / 'RAPID/outputs/event_window_plan_20261013/plan.json').read_text())
        case = next(c for c in plan['cases'] if c['example'] == example)
        job = next(j for j in case['jobs'] if j['state_id'] == metric.state)
        if job['fingerprint'] != row.fingerprint or digest(parameters_path) != case['source_sha256'][str(parameters_path.resolve())]:
            raise ValueError('Reviewed event parameters differ from the active registry')
        if digest(metric.outlet) != metric.outlet_sha256:
            raise ValueError('Reviewed outlet has changed')
        outlet = pd.read_parquet(metric.outlet)
        forcing = pd.read_parquet(Path(metric.outlet).parents[1] / 'forcing.parquet')
        dt = float(job['native_dt_seconds'])
        result = dict(native_dt_s=dt, nsub=round(60/dt), initial_cms=case['initial_cms'])
        row['centroid_baseline_cms'] = float(metric.baseline_cms)
        row['animation_start_utc'] = metric.rise_utc
        row['animation_end_utc'] = metric.fall_utc
        row['outlet_parquet'] = metric.outlet
        row['forcing_parquet'] = str(Path(metric.outlet).parents[1] / 'forcing.parquet')
    # Production extends the shared forcing with a constant final-value tail.
    n = len(forcing)
    if len(outlet) < n:
        raise ValueError('Published outlet is shorter than the forcing')
    for col in ['time_utc', 'interval_start_utc', 'interval_end_utc']:
        if not pd.DatetimeIndex(forcing[col]).equals(pd.DatetimeIndex(outlet[col].iloc[:n])):
            raise ValueError(f'Forcing and outlet {col} clocks differ')
    step = float(row.forcing_dt_seconds)
    starts = pd.DatetimeIndex(outlet.interval_start_utc)
    ends = pd.DatetimeIndex(outlet.interval_end_utc)
    if not (np.all(np.diff(starts.asi8) == step * 1e9)
            and np.all(ends.asi8 - starts.asi8 == step * 1e9)):
        raise ValueError('Outlet reporting intervals are not uniform')
    dt, nsub = float(result['native_dt_s']), int(result['nsub'])
    if not np.isclose(dt * nsub, step) or not np.isclose(dt, row.routing_dt_seconds):
        raise ValueError('Registered and saved routing clocks differ')
    flow = np.pad(forcing.forcing_cms.to_numpy(), (0, len(outlet)-n), mode='edge')
    initial = float(result['initial_cms'])
    cells = cunge.expand_links(frame)
    qout = np.empty((len(flow), len(cells)), dtype=float)
    columns = {(int(r.original_link_id), int(r.subreach_index)-1): j
               for j, r in enumerate(cells.itertuples())}

    def capture(i, subreach, native, equilibrium):
        column = columns[(int(frame.iloc[i].id_link), subreach)]
        qout[:, column] = outlet_interval_means(native[nsub:], nsub, initial_flow=equilibrium)

    native, audit = cunge.route(frame, np.r_[initial, flow], nsub, dt,
                                return_native=True, reach_callback=capture)
    reconstructed = outlet_interval_means(native[nsub:], nsub, initial_flow=initial)
    np.testing.assert_allclose(reconstructed, outlet.q_outlet_cms, rtol=1e-10, atol=1e-8,
                               err_msg='Reconstructed Cunge outlet differs from published results')
    if audit['minimum_native_flow_cms'] < -1e-9:
        raise ValueError('Reconstructed run has negative discharge')
    manifest = json.loads((parameters_path.parents[3] / 'hydraulic_context/hydraulic_context_manifest.json').read_text())
    geometry = gpd.read_file(manifest['input_paths']['directed_links'])
    if geometry.id_link.duplicated().any() or set(geometry.id_link) != set(frame.id_link):
        raise ValueError('Current geometry link IDs differ from routed links')
    geometry = geometry.set_index('id_link')
    for r in frame.itertuples():
        g = geometry.loc[r.id_link]
        if g.id_us_node != r.u or g.id_ds_node != r.v:
            raise ValueError('Geometry endpoints differ from routed topology')
    if geometry.crs.is_geographic:
        geometry = geometry.to_crs(geometry.estimate_utm_crs())
    lines = {}
    for link, geom in geometry.geometry.items():
        line = linemerge(geom) if geom.geom_type == 'MultiLineString' else geom
        if line.geom_type != 'LineString':
            raise ValueError(f'Link {link} does not have continuous geometry')
        lines[link] = line
    shapes = [substring(lines[r.original_link_id], (r.subreach_index-1)/r.subreach_count,
                        r.subreach_index/r.subreach_count, normalized=True) for r in cells.itertuples()]
    cells['parent_link_id'] = cells.original_link_id
    cells['reference_flow_fraction'] = cells.original_link_id.map(frame.set_index('id_link').reference_flow_fraction)
    reaches = gpd.GeoDataFrame(cells, geometry=shapes, crs=geometry.crs)
    times = (pd.DatetimeIndex(outlet.time_utc) - starts[0]).total_seconds().to_numpy()
    row['event_start_time_seconds'] = 0.0
    row['event_start_time_utc'] = starts[0].isoformat()
    peak = int(np.argmax(flow))
    row['forcing_peak_time_seconds'] = float(times[peak])
    row['wave_source'] = ('Reviewed event rerun' if reviewed_window is not None else 'Active registry run') + '; reconstructed reach waves; outlet verified'
    print(f'{state_id}: reconstructed {len(cells)} subreaches; outlet verified '
          f'(max error {np.max(np.abs(reconstructed-outlet.q_outlet_cms)):.3g} m³/s)')
    return _LoadedWaveState(experiment, registry, row,
                            ReachWaveStack(reaches, times, cells.reach_id.to_numpy(), qout), outlet)


def variant_domain_geometry(reference, target, parent_links):
    """Select variant map links inside its validated collapse domain.

    Keep the full routed waves and mover trajectories separate from this
    display-only selection. Domain metadata handles aliases and renumbered nodes.
    """
    from .current_domains import load_domain_pair

    record = target.run_row.copy()
    record['registry_path'] = str(target.experiment_path / 'rapid_run_registry.csv')
    record['example_id'] = target.experiment_path.name.split('_')[0]
    record['parent_parameters_parquet'] = reference.run_row.parameters_parquet
    _, _, domain = load_domain_pair(record)
    ids = set(domain['child_links'])
    available = set(parent_links.parent_link_id.astype(int))
    if not ids or not ids.issubset(available):
        raise ValueError('Validated variant domain links are missing from the animation geometry')
    visible = parent_links.loc[parent_links.parent_link_id.isin(ids)].copy()
    return visible, domain
