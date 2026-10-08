"""Reference-network Manning calibration using the reviewed event windows.

Isolated analysis: reads production geometry/forcing, writes only calibration
outputs. The spatial recurrence is checked against production sparse operations.
"""
from pathlib import Path
import hashlib
import json
import math
import sys
import time

import numpy as np
import pandas as pd
from openpyxl import load_workbook

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "RAPID/src"))
sys.path.insert(0, str(Path(__file__).parent))
from rapid_tools.analysis import _load_gauge_quality_sheet, _load_station_timeseries_for_observed_lag
from rapid_tools.k_values import KValueConfig, compute_k_values, compute_dt_bounds
from check_fractional_routing_clocks import cascade, sparse_benchmark
from event_buffer_tests import settle

WORKBOOK = ROOT / "RAPID/reviews/hydrograph_match_review.xlsx"
OUTPUT = ROOT / "RAPID/outputs/travel_time_calibration"
BASE_N = 0.035
DAY = 86400.0


def read_schedule(path=WORKBOOK):
    """Read the named worksheet; verify cached linked cells against their inputs."""
    path = Path(path)
    book = load_workbook(path, data_only=True, read_only=True)
    try:
        sheet = book["Routing windows"]
        rows = list(sheet.values)
        records = []
        for r in range(7, 23):
            event, stages, prior = [rows[k - 1] for k in (r, r + 21, r + 42)]
            ex = str(event[0]).zfill(2)
            if ex != str(stages[0]).zfill(2) or ex != str(prior[0]).zfill(2):
                raise ValueError(f"Mismatched example IDs in workbook row {r}")
            names = ["requested_start", "requested_end_exclusive", "first_sample", "last_sample"]
            rec = dict(example=ex, workbook_row=r,
                       **{k: pd.to_datetime(v, utc=True) for k, v in zip(names, event[1:5])})
            rec.update(q_initial=float(event[5]), q_final=float(event[6]))
            names = ["simulation_start", "event_start", "plateau_start", "settling_start",
                     "settlement_confirmed", "simulation_end"]
            rec.update({k: pd.to_datetime(v, utc=True) for k, v in zip(names, stages[1:7])})
            rec.update(old_start=pd.to_datetime(prior[1], utc=True),
                       old_end=pd.to_datetime(prior[2], utc=True), tested_states=prior[6])
            if any(pd.isna(rec[k]) for k in names):
                raise ValueError(f"Missing formula cache/date for {ex}; save the workbook in Excel")
            if not (rec['requested_start'] <= rec['first_sample'] <= rec['last_sample'] < rec['requested_end_exclusive']):
                raise ValueError(f"Invalid exclusive event bounds for {ex}")
            if rec['event_start'] != rec['first_sample'] or rec['plateau_start'] != rec['last_sample']:
                raise ValueError(f"Stale linked event dates for {ex}; recalculate and save the workbook")
            if rec['settlement_confirmed'] != rec['settling_start'] + pd.Timedelta(days=1):
                raise ValueError(f"Stale settlement formula for {ex}")
            if not (rec['simulation_start'] <= rec['first_sample'] - pd.Timedelta(days=1)
                    and rec['last_sample'] <= rec['settling_start']
                    and rec['simulation_end'] >= rec['settlement_confirmed'] + pd.Timedelta(days=1)):
                raise ValueError(f"Invalid spin-up/settlement/buffer order for {ex}")
            rec['old_days'] = (rec['old_end'] - rec['old_start']).total_seconds() / DAY
            rec['new_days'] = (rec['simulation_end'] - rec['simulation_start']).total_seconds() / DAY
            rec['duration_reduction_pct'] = 100 * (1 - rec['new_days'] / rec['old_days'])
            np.testing.assert_allclose([prior[3], prior[4], prior[5]],
                                       [rec['old_days'], rec['new_days'], rec['duration_reduction_pct']/100],
                                       rtol=1e-9, atol=1e-8)
            records.append(rec)
        frame = pd.DataFrame(records).set_index('example', drop=False)
        if frame.index.has_duplicates:
            raise ValueError("Duplicate example IDs")
        frame.attrs['workbook_sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
        return frame
    finally:
        book.close()


def peak_summary(times, values):
    """First sampled global maximum, with all exact ties exposed separately."""
    times = pd.DatetimeIndex(times)
    values = np.asarray(values, dtype=float)
    if len(values) < 3 or len(times) != len(values) or not np.isfinite(values).all():
        raise ValueError("Peak search needs at least three finite samples")
    steps = np.diff(times.asi8) / 1e9
    if (steps <= 0).any() or np.ptp(values) <= 0:
        raise ValueError("Peak search needs increasing timestamps and a nonconstant event")
    ix = np.flatnonzero(values == values.max())
    return dict(first=times[ix[0]], last=times[ix[-1]], maximum=float(values.max()),
                tied_samples=len(ix), tied_span_h=(times[ix[-1]]-times[ix[0]]).total_seconds()/3600,
                disconnected_ties=bool((np.diff(ix) > 1).any()),
                boundary=bool(ix[0] == 0 or ix[-1] == len(values)-1),
                median_step_s=float(np.median(steps)), max_gap_s=float(steps.max()))


def observed_target(example, window, quality=None):
    quality = _load_gauge_quality_sheet() if quality is None else quality
    if quality is None:
        raise ValueError("Gauge station lookup is unavailable")
    match = quality.loc[pd.to_numeric(quality.Example, errors='coerce').eq(int(example))]
    if len(match) != 1:
        raise ValueError(f"Expected one gauge pair for {example}")
    record, series, summaries = match.iloc[0], {}, {}
    for name, field in [('upstream', 'Up'), ('downstream', 'Down')]:
        g = _load_station_timeseries_for_observed_lag(str(record[field]))
        g = g.loc[g.time.ge(window.requested_start) & g.time.lt(window.requested_end_exclusive)].copy()
        series[name] = g
        summaries[name] = peak_summary(g.time, g.discharge_cms)
    up, down = summaries['upstream'], summaries['downstream']
    lag = (down['first'] - up['first']).total_seconds()/3600
    flags = []
    if up['boundary'] or down['boundary']:
        flags.append('observed maximum touches window boundary')
    if up['tied_samples'] > 1 or down['tied_samples'] > 1:
        flags.append('tied observed maximum; first maximum used')
    if up['disconnected_ties'] or down['disconnected_ties']:
        flags.append('equal maxima in separate sampled blocks')
    if any(p['max_gap_s'] > 1.5*p['median_step_s'] for p in [up, down]):
        flags.append('observation gaps exceed 1.5 median intervals')
    if lag <= 0:
        flags.append('nonpositive observed first-maximum lag')
    return dict(example=str(example).zfill(2), observed_lag_h=lag,
                upstream_peak_utc=up['first'], downstream_peak_utc=down['first'],
                upstream_last_max_utc=up['last'], downstream_last_max_utc=down['last'],
                upstream_tied_span_h=up['tied_span_h'], downstream_tied_span_h=down['tied_span_h'],
                upstream_step_minutes=up['median_step_s']/60,
                downstream_step_minutes=down['median_step_s']/60,
                timing_resolution_h=(up['median_step_s']+down['median_step_s'])/3600,
                target_usable=lag > 0 and not up['boundary'] and not down['boundary'],
                target_notes='; '.join(flags), series=series)


def candidate_frame(frame, reference, n):
    if not np.isfinite(n) or n <= 0:
        raise ValueError("Manning n must be finite and positive")
    local = frame.copy()
    local['link_length_m'] = local.rapid_effective_length_m
    local['slope_used'] = local.slope
    computed = compute_k_values(local, config=KValueConfig(
        width_field='width', n_manning=float(n), depth_width_exponent=.65,
        kb_reference_width_m=reference['reference_width_m'],
        kb_reference_depth_m=reference['reference_depth_m'], x_value=.1,
        use_celerity_capping=True, min_celerity_mps=.05, max_celerity_mps=5.,
        min_effective_length_m=200.))
    # Keep the stored physical lengths and segmentation for the recurrence.
    result = frame.copy()
    for field in ['rapid_k', 'rapid_celerity_mps', 'rapid_celerity_mps_raw', 'rapid_celerity_capped']:
        result[field] = computed[field]
    np.testing.assert_allclose(computed.rapid_depth_m, frame.depth_m, rtol=1e-12)
    return result


def load_case(example, schedule, n_bounds=(.015, .08)):
    """One base geometry; choose one safe clock for the entire proposed n range."""
    # Plotting/legacy-result dependencies are not needed by the shared target
    # and event-window readers used by alternative routing calibrations.
    from rapid_tools.hydrograph_review import base_directory, event_clock
    from rapid_tools.refreshed_analysis import DEFAULT_REFRESH_ROOT
    ex = str(example).zfill(2)
    if not 0 < n_bounds[0] <= BASE_N <= n_bounds[1]:
        raise ValueError("Positive n bounds must contain the baseline 0.035")
    window = schedule.loc[ex]
    directory = base_directory(ex)
    inventory = pd.read_csv(DEFAULT_REFRESH_ROOT/ex/'inventory.csv')
    base = inventory.loc[inventory.state.eq('S000_base') & inventory.status.eq('ready')].drop_duplicates('fingerprint')
    if len(base) != 1:
        raise ValueError(f"Expected one base geometry for {ex}")
    source = Path(base.iloc[0].frame)
    frame = pd.read_parquet(source)
    manifest = json.loads((directory/'policy_manifest.json').read_text())
    policy = manifest['prep_config']
    if (policy['n_manning'], policy['depth_width_exponent'], policy['x_value'], policy['initial_flow_mode']) != (BASE_N, .65, .1, 'steady_first_inflow'):
        raise ValueError("Calibration requires the reviewed production baseline")
    reference = manifest['reference_kb_summary']
    baseline = candidate_frame(frame, reference, BASE_N)
    np.testing.assert_allclose(baseline.rapid_k, frame.rapid_k, rtol=1e-12)
    old = pd.read_csv(directory/'run/outlet_hydrograph.csv')
    clock, clock_source = event_clock(ex, old)
    mask = clock.ge(window.requested_start) & clock.lt(window.requested_end_exclusive)
    selected = old.loc[mask].copy()
    times = pd.DatetimeIndex(clock.loc[mask])
    if len(times) < 3 or times[0] != window.first_sample or times[-1] != window.last_sample:
        raise ValueError(f"Workbook sample bounds disagree with production forcing for {ex}")
    flow = selected.q_inflow_cms.to_numpy(float)
    np.testing.assert_allclose([flow[0], flow[-1]], [window.q_initial, window.q_final], rtol=1e-10)
    step = float(np.median(np.diff(times.asi8))/1e9)
    np.testing.assert_allclose(np.diff(times.asi8)/1e9, step, atol=1e-8, rtol=0)
    for stamp in [window.simulation_start, window.simulation_end]:
        quotient = (stamp-times[0]).total_seconds()/step
        if not np.isclose(quotient, round(quotient), atol=1e-8, rtol=0):
            raise ValueError(f"Workbook schedule is off the forcing grid for {ex}")
    fastest = candidate_frame(frame, reference, n_bounds[0])
    _, upper = compute_dt_bounds(fastest.rapid_k, x_value=.1, coefficient_policy='signed_current_inflow')
    saved_dt = float(json.loads((directory/'run/Qout.audit.json').read_text())['routing_dt_seconds'])
    nsub = int(math.ceil(step/min(saved_dt, .99*upper)))
    target = observed_target(ex, window)
    forcing_peak = peak_summary(times, flow)
    signature = hashlib.sha256()
    for p in [source, directory/'policy_manifest.json', Path(__file__),
              Path(__file__).with_name('check_fractional_routing_clocks.py'),
              Path(__file__).with_name('event_buffer_tests.py'),
              ROOT/'RAPID/src/rapid_tools/k_values.py', ROOT/'RAPID/src/rapid_tools/engine.py',
              ROOT/'network_variants/reference_context.py']:
        signature.update(p.read_bytes())
    signature.update(schedule.attrs['workbook_sha256'].encode())
    signature.update(flow.tobytes())
    for g in target['series'].values():
        signature.update(pd.util.hash_pandas_object(g, index=False).values.tobytes())
    signature.update(json.dumps(dict(n_bounds=n_bounds, nsub=nsub, step=step)).encode())
    return dict(example=ex, window=window, frame=frame, reference=reference, flow=flow,
                times=times, forcing_step_s=step, nsub=nsub, routing_dt_s=step/nsub,
                saved_routing_dt_s=saved_dt, bounds=n_bounds, target=target,
                forcing_peak=forcing_peak, signature=signature.hexdigest(), clock_source=clock_source)


def run_trial(case, n, output=OUTPUT, max_extensions=6):
    """Route one n; extend the constant tail until 24 h settlement + 24 h buffer."""
    n = float(n)
    if not case['bounds'][0] <= n <= case['bounds'][1]:
        raise ValueError("Candidate is outside the n range used to choose the common clock")
    identity = hashlib.sha256(f"{case['signature']}:{n:.17g}:{max_extensions}".encode()).hexdigest()[:20]
    dest = Path(output)/case['example']/identity
    summary_path, wave_path = dest/'summary.json', dest/'hydrograph.parquet'
    if summary_path.exists() and wave_path.exists():
        return json.loads(summary_path.read_text()), pd.read_parquet(wave_path)
    f = candidate_frame(case['frame'], case['reference'], n)
    dt, step, nsub = case['routing_dt_s'], case['forcing_step_s'], case['nsub']
    _, upper = compute_dt_bounds(f.rapid_k, x_value=.1, coefficient_policy='signed_current_inflow')
    if dt > .99*upper:
        raise ValueError("Candidate violates the common routing clock bound")
    event, window = case['flow'], case['window']
    # Cross-check the actual event's first changing intervals, not just a flat plateau.
    probe = event[:min(len(event), 128)]
    sparse, _, _ = sparse_benchmark(f, probe, nsub, dt, event[0], serialized_allocations=True)
    fast, _ = cascade(f, probe, nsub, dt, event[0], serialized_allocations=True)
    parity = float(np.max(np.abs(fast-sparse)))
    if parity > 1e-8*max(1., np.ptp(event)):
        raise ValueError(f"Sparse recurrence mismatch: {parity}")
    count = int(round((window.simulation_end-window.last_sample).total_seconds()/step))
    started = time.perf_counter()
    for extension in range(max_extensions+1):
        flow = np.r_[event, np.full(count, event[-1])]
        q, audit = cascade(f, flow, nsub, dt, event[0], save_audit=True, serialized_allocations=True)
        t = np.arange(len(flow))*step
        amplitude = float(q.max()-max(event[0], event[-1]))
        if amplitude <= 0:
            raise ValueError("No positive event peak above both plateaus")
        arrival, required_end = settle(t, q, (len(event)-1)*step, event[-1], .001*amplitude)
        if np.isfinite(required_end):
            break
        count = max(count*2, count+int(math.ceil(2*DAY/step)))
    else:
        raise ValueError("Tail is still censored after extensions; trial is not scored")
    peak = peak_summary(case['times'][0]+pd.to_timedelta(t, unit='s'), q)
    if peak['boundary']:
        raise ValueError("Model peak is at a simulation boundary")
    lag = (peak['first']-case['forcing_peak']['first']).total_seconds()/3600
    target = case['target']
    # Include the declared initial plateau in the saved hydrograph. Its exact
    # equilibrium signal is propagated through serialized split allocations.
    spin = int(round((window.first_sample-window.simulation_start).total_seconds()/step))
    plateau, _ = cascade(f, np.array([event[0]]), nsub, dt, event[0], serialized_allocations=True)
    full_times = window.simulation_start+pd.to_timedelta(np.arange(spin+len(q))*step, unit='s')
    wave = pd.DataFrame(dict(time_utc=full_times, q_inflow_cms=np.r_[np.full(spin,event[0]),flow],
                             q_outlet_cms=np.r_[np.full(spin,plateau[0]),q]))
    score = abs(lag-target['observed_lag_h']) if target['target_usable'] else None
    result = dict(example=case['example'], n_manning=n, observed_lag_h=target['observed_lag_h'],
                  simulated_lag_h=lag, signed_error_h=lag-target['observed_lag_h'], abs_error_h=score,
                  target_usable=target['target_usable'], target_notes=target['target_notes'],
                  model_peak_utc=peak['first'].isoformat(), model_peak_tied_span_h=peak['tied_span_h'],
                  forcing_peak_utc=case['forcing_peak']['first'].isoformat(),
                  forcing_minus_observed_upstream_peak_h=(case['forcing_peak']['first']-target['upstream_peak_utc']).total_seconds()/3600,
                  outlet_minus_observed_downstream_peak_h=(peak['first']-target['downstream_peak_utc']).total_seconds()/3600,
                  timing_resolution_h=max(target['timing_resolution_h'], step/3600),
                  routing_dt_s=dt, forcing_step_s=step, n_segments=int(f['count'].sum()),
                  capped_segment_fraction=float(f.loc[f.rapid_celerity_capped,'count'].sum()/f['count'].sum()),
                  minimum_native_flow_cms=min(a['minimum'] for a in audit),
                  sparse_parity_max_cms=parity, tail_extensions=extension,
                  settlement_start_utc=(window.first_sample+pd.to_timedelta(arrival,unit='s')).isoformat(),
                  simulation_end_utc=wave.time_utc.iloc[-1].isoformat(),
                  simulated_days=(wave.time_utc.iloc[-1]-wave.time_utc.iloc[0]).total_seconds()/DAY,
                  routing_seconds=time.perf_counter()-started, identity=identity,
                  hydrograph_path=str(wave_path), status='event-specific candidate; not applied to production')
    dest.mkdir(parents=True, exist_ok=True)
    wave.to_parquet(wave_path, index=False)
    pd.DataFrame(audit).to_parquet(dest/'segment_audit.parquet', index=False)
    summary_path.write_text(json.dumps(result, indent=2, allow_nan=False))
    return result, wave


def calibrate(case, n_grid, output=OUTPUT, refine_points=7):
    """Grid + local refinement; preserve sampled-time ties and boundary optima."""
    grid = np.unique(np.r_[np.asarray(n_grid, float), BASE_N])
    if len(grid) < 3 or not np.isfinite(grid).all():
        raise ValueError("Supply at least three finite n candidates")
    if grid.min() < case['bounds'][0] or grid.max() > case['bounds'][1]:
        raise ValueError("Search grid exceeds the bounds used for the common clock")
    if not case['target']['target_usable']:
        raise ValueError(f"Review observed target before calibration: {case['target']['target_notes']}")
    rows = {}
    def evaluate(n):
        n = float(n)
        # linspace may represent an existing coarse value a few ulps away.
        if not any(np.isclose(n, prior, rtol=0, atol=1e-12) for prior in rows):
            rows[n], _ = run_trial(case, n, output)
    for n in grid:
        evaluate(n)
    best = min(rows, key=lambda n: (rows[n]['abs_error_h'], abs(n-BASE_N)))
    index = int(np.flatnonzero(grid == best)[0])
    if refine_points and 0 < index < len(grid)-1:
        for n in np.linspace(grid[index-1],grid[index+1],refine_points):
            evaluate(n)
    trials = pd.DataFrame(rows.values()).sort_values('n_manning').reset_index(drop=True)
    best_row = trials.assign(distance_to_baseline=abs(trials.n_manning-BASE_N)).sort_values(
        ['abs_error_h','distance_to_baseline']).iloc[0].to_dict()
    eligible = trials.loc[trials.abs_error_h <= best_row['abs_error_h']+best_row['timing_resolution_h']]
    best_row.update(search_lower=float(grid.min()), search_upper=float(grid.max()),
                    best_at_search_boundary=best_row['n_manning'] in [grid.min(),grid.max()],
                    near_best_n_values=eligible.n_manning.tolist(),
                    near_best_definition='Within one sampling-resolution tolerance of minimum error; not a confidence interval',
                    baseline_abs_error_h=float(trials.loc[trials.n_manning.eq(BASE_N),'abs_error_h'].iloc[0]))
    dest = Path(output)/case['example']/case['signature'][:20]
    dest.mkdir(parents=True, exist_ok=True)
    trials.to_csv(dest/'trials.csv', index=False)
    (dest/'best_candidate.json').write_text(json.dumps(best_row, indent=2, allow_nan=False))
    return trials, best_row
