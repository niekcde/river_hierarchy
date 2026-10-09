"""Versioned common-grid base pilots and target review, without recalibration.

Run with the scientific project Python. Outputs are compact outlet series only.
Original observed point records, target choices and hydraulic parameters are frozen.
"""
from pathlib import Path
import argparse
import hashlib
import json
import math
import shutil
import sys
import time

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT/'RAPID/src')]
from rapid_tools import cunge
from rapid_tools.common_grid import resample_points, route_interval_means
from rapid_tools.time_support import point_series_interval_means

CAL = ROOT/'RAPID/outputs/cunge_calibration_reviewed_20261006/comparison.csv'
TARGETS = ROOT/'RAPID/reviews/calibration_targets_20261006/targets.json'
OUT = ROOT/'RAPID/outputs/common_grid_validation_20261008'
VARIANTS = [('300s', 300, 0, 1), ('60s', 60, 0, 1), ('30s', 30, 0, 1),
            ('60s_phase30', 60, 30, 1), ('60s_halfdt', 60, 0, 2)]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def snapshot():
    comparison = pd.read_csv(CAL, dtype={'example': str})
    paths = [CAL, TARGETS, Path(__file__), ROOT/'RAPID/src/rapid_tools/common_grid.py',
             ROOT/'RAPID/src/rapid_tools/time_support.py', ROOT/'RAPID/src/rapid_tools/cunge.py',
             ROOT/'RAPID/src/rapid_tools/engine.py', ROOT/'RAPID/src/rapid_tools/k_values.py']
    for folder in sorted(set(comparison.output_dir)):
        paths += [Path(folder)/name for name in ['case.json', 'frozen_grid.parquet',
                  'observed_upstream.parquet', 'observed_downstream.parquet',
                  'fixed_x_best_wave.parquet', 'cunge_best_wave.parquet']]
    hashes = {str(p): sha(p) for p in paths}
    signature = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()[:12]
    dest = OUT/signature
    dest.mkdir(parents=True, exist_ok=True)
    freeze = dest/'baseline'
    freeze.mkdir(exist_ok=True)
    for p in paths:
        rel = p.relative_to(ROOT)
        target = freeze/rel
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, target)
        if sha(target) != hashes[str(p)]:
            raise ValueError(f'Frozen baseline mismatch: {target}')
    provenance = dict(signature=signature, source_hashes=hashes, production_modified=False,
        parameters='Existing calibrated n, geometry, reference flow and grid frozen',
        initialization='Equilibrium at first retained upstream point, held fixed across resolutions',
        time_support='Linear point reconstruction; exact interval integrals; UTC-midnight aligned edges',
        gaps='Flag >2 nominal sampling intervals. Linear filling for diagnostics only; not accepted final forcing.',
        endpoint='Explicit constant extension before first point <=600s; tail <=3days+300s',
        outputs='Outlet interval means only. No persisted native reach matrices.',
        excluded_hierarchy_example='11; observations frozen but no routing pilot',
        variants=VARIANTS)
    (dest/'provenance.json').write_text(json.dumps(provenance, indent=2))
    (OUT/'latest.json').write_text(json.dumps({'directory': str(dest), 'signature': signature}, indent=2))
    return comparison, dest, provenance


def observation_frame(folder, side, origin):
    frame = pd.read_parquet(folder/f'observed_{side}.parquet')
    times = pd.to_datetime(frame.time, utc=True)
    return (times-origin).dt.total_seconds().to_numpy(), frame.discharge_cms.to_numpy(float)


def gauge_on_grid(times, q, edges, gap_limit):
    # Mean outside observed bounds is unavailable, never a fabricated observation.
    bracketed = (edges[:-1] >= times[0]) & (edges[1:] <= times[-1])
    mean = np.full(len(edges)-1, np.nan)
    gap = np.zeros(len(mean), bool)
    ids = np.flatnonzero(bracketed)
    if len(ids):
        values, flags, _ = resample_points(times, q, edges[ids[0]:ids[-1]+2],
            maximum_gap_seconds=gap_limit, diagnostic_gap_fill=True)
        mean[ids] = values; gap[ids] = flags
    return mean, gap, bracketed


def peak_fields(t, q, origin, target):
    finite = np.isfinite(q)
    if not finite.any():
        return dict(peak_utc=None, peak_error_min=None, peak_cms=None)
    ids = np.flatnonzero(finite)
    i = ids[np.argmax(q[finite])]
    peak = origin+pd.to_timedelta(t[i], unit='s')
    return dict(peak_utc=peak.isoformat(), peak_error_min=(peak-pd.Timestamp(target)).total_seconds()/60,
                peak_cms=float(q[i]))


def run_case(ex, rows, target, dest):
    folder = dest/'baseline'/Path(rows.iloc[0].output_dir).relative_to(ROOT)
    grid = pd.read_parquet(folder/'frozen_grid.parquet')
    raw = pd.read_parquet(folder/'observed_upstream.parquet')
    first = pd.to_datetime(raw.time, utc=True).min()
    last = pd.to_datetime(raw.time, utc=True).max()
    origin = first.floor('5min')-pd.Timedelta(minutes=5)
    end = last.ceil('5min')+pd.Timedelta(days=3)
    duration = (end-origin).total_seconds()
    ut, uq = observation_frame(folder, 'upstream', origin)
    ot, oq = observation_frame(folder, 'downstream', origin)
    initial = float(uq[0])
    limits = {side: 2*float(target[f'{side}_sampling_minutes'])*60 for side in ['upstream', 'downstream']}
    base_dt, n30 = cunge.common_clock(grid, 30.)
    case_dir = dest/ex
    case_dir.mkdir(exist_ok=True)
    metrics, gaps = [], []
    for side, ts in [('upstream', ut), ('downstream', ot)]:
        for i in np.flatnonzero(np.diff(ts) > limits[side]):
            gaps.append(dict(example=ex, side=side, start_utc=(origin+pd.Timedelta(seconds=ts[i])).isoformat(),
                end_utc=(origin+pd.Timedelta(seconds=ts[i+1])).isoformat(), duration_min=(ts[i+1]-ts[i])/60,
                gap_limit_min=limits[side]/60))
    pd.DataFrame(gaps, columns=['example', 'side', 'start_utc', 'end_utc', 'duration_min', 'gap_limit_min']).to_csv(case_dir/'gaps.csv', index=False)
    for name, step, phase, refinement in VARIANTS:
        path = case_dir/f'{name}.parquet'
        metadata = case_dir/f'{name}.json'
        if path.exists() and metadata.exists():
            metrics.extend(json.loads(metadata.read_text()))
            continue
        edges = np.arange(0, duration+step/2, step)+phase
        mid = (edges[:-1]+edges[1:])/2
        flow, upstream_gap, extension = resample_points(ut, uq, edges,
            maximum_gap_seconds=limits['upstream'], endpoint_hold_seconds=3*86400+300,
            diagnostic_gap_fill=True)
        assert max(ut[0]-edges[0], 0) <= 600
        up, _, up_valid = gauge_on_grid(ut, uq, edges, limits['upstream'])
        obs, downstream_gap, obs_valid = gauge_on_grid(ot, oq, edges, limits['downstream'])
        direct_volume = point_series_interval_means(ut, uq, [edges[0], edges[-1]],
                                                   allow_endpoint_hold=True)[0]*(edges[-1]-edges[0])
        volume_error = float(np.sum(flow)*step-direct_volume)
        assert abs(volume_error) <= max(1e-5, abs(direct_volume)*1e-12)
        output = pd.DataFrame(dict(time_utc=origin+pd.to_timedelta(mid, unit='s'),
            interval_start_utc=origin+pd.to_timedelta(edges[:-1], unit='s'),
            interval_end_utc=origin+pd.to_timedelta(edges[1:], unit='s'),
            upstream_cms=up, downstream_cms=obs, forcing_cms=flow,
            upstream_gap=upstream_gap, downstream_gap=downstream_gap,
            forcing_endpoint_hold=extension, downstream_supported=obs_valid & ~downstream_gap))
        records = []
        for r in rows.itertuples():
            f = cunge.parameters(grid, r.n_manning, method=r.method)
            ns = n30*int(step/30)*refinement; dt = base_dt/refinement
            before = time.monotonic()
            means, audit = route_interval_means(f, flow, ns, dt, initial_flow=initial)
            assert audit['minimum_native_flow_cms'] >= -1e-9
            output[f'{r.method}_cms'] = means
            score = obs_valid & ~downstream_gap
            p, o = means[score], obs[score]
            norm = lambda v: (v-v.min())/max(np.ptp(v), 1e-12)
            amplitude = max(np.ptp(uq), 1e-12)
            peak = peak_fields(mid[obs_valid], means[obs_valid], origin, target['target_utc'])
            records.append(dict(example=ex, method=r.method, variant=name, interval_s=step, phase_s=phase,
                dt_s=dt, n_manning=r.n_manning, initial_cms=initial,
                raw_rmse_cms=float(np.sqrt(np.mean((p-o)**2))),
                normalized_shape_rmse=float(np.sqrt(np.mean((norm(p)-norm(o))**2))),
                valid_observation_bins=int(score.sum()), upstream_gap_bins=int(upstream_gap.sum()),
                downstream_gap_bins=int(downstream_gap.sum()), input_volume_m3=float(direct_volume),
                input_volume_error_m3=volume_error, tail_residual_pct_inlet_amplitude=abs(audit['final_outlet_cms']-uq[-1])/amplitude*100,
                inlet_amplitude_cms=amplitude, runtime_s=time.monotonic()-before,
                target_utc=target['target_utc'], target_rule=target['selection_rule'],
                feature_status='global maximum diagnostic only; reviewed feature target unchanged',
                observation_clock_status='MRC ambiguity unresolved' if ex in ['21', '22'] else 'unchanged source UTC',
                calibration_ready=False, **peak, **audit))
        output.to_parquet(path, index=False)
        metadata.write_text(json.dumps(records, indent=2))
        metrics.extend(records)
        print(ex, name, 'completed', flush=True)
    return metrics


def comparisons(dest, metrics):
    records = []
    for ex in sorted(metrics.example.unique()):
        reference = pd.read_parquet(dest/ex/'30s.parquet')
        origin = reference.interval_start_utc.iloc[0]
        rt = (reference.time_utc-origin).dt.total_seconds().to_numpy()
        for name, _, _, _ in VARIANTS:
            if name == '30s':
                continue
            f = pd.read_parquet(dest/ex/f'{name}.parquet')
            t = (f.time_utc-origin).dt.total_seconds().to_numpy()
            valid = reference.downstream_cms.notna().to_numpy() & (rt >= t[0]) & (rt <= t[-1])
            for method in ['cunge', 'fixed_x']:
                column = method+'_cms'
                # Diagnostic comparison of continuous curves through report means,
                # not an observation score or a claim to recover sub-bin peaks.
                residual = np.interp(rt[valid], t, f[column])-reference[column].to_numpy()[valid]
                a = metrics[(metrics.example == ex) & (metrics.method == method) & (metrics.variant == name)].iloc[0]
                b = metrics[(metrics.example == ex) & (metrics.method == method) & (metrics.variant == '30s')].iloc[0]
                records.append(dict(example=ex, method=method, variant=name,
                    report_curve_max_difference_pct_inlet_amplitude=float(np.max(abs(residual))/a.inlet_amplitude_cms*100),
                    report_curve_rms_difference_pct_inlet_amplitude=float(np.sqrt(np.mean(residual**2))/a.inlet_amplitude_cms*100),
                    global_maximum_time_difference_s=(pd.Timestamp(a.peak_utc)-pd.Timestamp(b.peak_utc)).total_seconds()))
    result = pd.DataFrame(records)
    result.to_csv(dest/'resolution_comparison.csv', index=False)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--examples', nargs='*')
    args = parser.parse_args()
    table, dest, provenance = snapshot()
    target_list = json.loads((dest/'baseline'/TARGETS.relative_to(ROOT)).read_text())['records']
    targets = {r['example']: r for r in target_list}
    records = []
    selected = sorted(set(table.example)-{'11'})
    if args.examples:
        selected = [ex for ex in selected if ex in args.examples]
    for ex in selected:
        records.extend(run_case(ex, table[table.example == ex], targets[ex], dest))
        pd.DataFrame(records).to_csv(dest/'base_metrics.csv', index=False)
    metrics = pd.DataFrame(records)
    results = comparisons(dest, metrics)
    for path, expected in provenance['source_hashes'].items():
        if sha(path) != expected:
            raise ValueError(f'Source changed during validation: {path}')
    print(results.groupby('variant').agg(max_difference_pct=('report_curve_max_difference_pct_inlet_amplitude','max'),
          largest_global_peak_shift_s=('global_maximum_time_difference_s', lambda x: abs(x).max())))
    print(dest)


if __name__ == '__main__':
    main()
