"""Frozen common-grid calibration consumption and hierarchy routing.

Run with ``python -m rapid_tools.common_grid_production --help``. Outputs are
new directories only. No fitting, source modification or legacy result deletion.
Observation gaps and ambiguous clocks remain explicit review qualifications.
"""
from pathlib import Path
import argparse
import hashlib
import json
import math
import tempfile

import numpy as np
import pandas as pd
from scipy.signal import find_peaks

from . import cunge
from .common_grid import resample_points, route_interval_means

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_BUNDLE = ROOT/'RAPID/outputs/common_grid_height_calibration_20261008/33847083c6ea'
DEFAULT_ASSETS = ROOT/'RAPID/assets/bounded_robust_slopes_20261002/child_head_reconstruction_20261003'
REVIEW = ROOT/'RAPID/outputs/cunge_reference_flow_sensitivity_20261006/provenance.json'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2))


def refined_parameters(grid, selection):
    multiplier = selection.get('subdivision_multiplier', 1)
    if int(multiplier) != multiplier or multiplier < 1:
        raise ValueError('Subdivision multiplier must be a positive integer.')
    grid = grid.copy()
    grid['count'] *= int(multiplier)
    grid['rapid_effective_length_m'] = grid.link_length_m/grid['count']
    return cunge.parameters(grid, selection['n_manning'], beta=selection['beta'])


def shared_clock(frames, steps, maximum_dt):
    """One exact divisor of all reporting clocks, bounded using adjusted X."""
    if not steps or any(int(s) != s or s <= 0 for s in steps):
        raise ValueError('Report steps must be positive integer seconds.')
    if not np.isfinite(maximum_dt) or maximum_dt <= 0:
        raise ValueError('Native timestep cap must be finite and positive.')
    quantum = math.gcd(*map(int, steps))
    upper = min(float((2*f.rapid_k*(1-f.rapid_x)).min()) for f in frames)
    return quantum/math.ceil(quantum/min(maximum_dt, .99*upper))


def model_feature(q, seconds, example, review):
    """Reviewed crest operator; multi-lobe separation is in seconds, not bins."""
    q, t = np.asarray(q), np.asarray(seconds)
    if len(q) < 2 or not np.isfinite(q).all() or not (np.diff(t) > 0).all():
        raise ValueError('Feature requires finite, chronologically ordered samples.')
    step = float(np.median(np.diff(t)))
    if not np.allclose(np.diff(t), step):
        raise ValueError('Feature requires a regular common grid.')
    maximum = float(q.max())
    top = np.flatnonzero(abs(q-maximum) <= max(np.ptp(q)*1e-10, 1e-9))
    if example == '54':
        span = (pd.Timestamp(review['crest_end_utc'])-pd.Timestamp(review['crest_start_utc'])).total_seconds()
        # Preserve the reviewed integer-minute separation at both report resolutions.
        separation = max(60., int(span/180.)*60.)
        peaks, _ = find_peaks(q, prominence=max(np.ptp(q)*1e-4, 1e-8),
                             distance=max(1, int(math.ceil(separation/step))))
        peaks = peaks[q[peaks] >= q.min()+.8*np.ptp(q)]
        best = np.sort(peaks[np.argsort(q[peaks])[-2:]])
        if len(best) == 2:
            return float(t[best].mean()), float(q[best].max()), 'two_lobe_midpoint'
        return float(t[top[0]]), maximum, 'merged_or_unresolved_two_lobes'
    if example == '48':
        return float(t[top[len(top)//2]]), maximum, 'middle_equal_maximum'
    end = 0
    while end+1 < len(top) and top[end+1] == top[end]+1:
        end += 1
    return float((t[top[0]]+t[top[end]])/2), maximum, ('exact_plateau_midpoint' if end else 'single_smoothed_crest')


class CalibrationCase:
    def __init__(self, bundle, example):
        self.bundle = Path(bundle).resolve()
        self.example = str(example).zfill(2)
        if self.example == '11':
            raise ValueError('Example 11 remains excluded.')
        self.selection = json.loads((self.bundle/self.example/'result.json').read_text())
        self.source = Path(json.loads((self.bundle/'protocol.json').read_text())['source_directory'])
        table = pd.read_csv(self.source/'baseline/RAPID/outputs/cunge_calibration_reviewed_20261006/comparison.csv', dtype={'example': str})
        row = table[table.example.eq(self.example) & table.method.eq('cunge')].iloc[0]
        self.folder = self.source/'baseline'/Path(row.output_dir).relative_to(ROOT)
        self.grid = pd.read_parquet(self.folder/'frozen_grid.parquet')
        self.meta = next(r for r in json.loads((self.source/self.example/'60s.json').read_text()) if r['method'] == 'cunge')
        self.review = next(r for r in json.loads((self.bundle/'timing_targets.json').read_text())['records'] if r['example'] == self.example)
        self.wave = pd.read_parquet(self.source/self.example/'60s.parquet')
        self.initial = self.meta['initial_cms']
        self.native_dt = self.selection.get('native_dt_s', self.meta['dt_s'])
        self.reference_q = json.loads((self.folder/'case.json').read_text())['reference_discharge_cms']
        self.paths = [self.bundle/'protocol.json', self.bundle/'height_targets.json', self.bundle/'timing_targets.json',
                      self.bundle/self.example/'result.json', self.bundle/self.example/'wave.parquet',
                      self.source/self.example/'60s.parquet', self.source/self.example/'60s.json',
                      self.folder/'frozen_grid.parquet', self.folder/'case.json',
                      self.folder/'observed_upstream.parquet', self.folder/'observed_downstream.parquet']

    def forcing(self, step, *, diagnostic_gaps=False):
        if step not in (30, 60):
            raise ValueError('Validated common report intervals are 30 and 60 seconds.')
        origin = pd.Timestamp(self.wave.interval_start_utc.iloc[0])
        end = pd.Timestamp(self.wave.interval_end_utc.iloc[-1])
        if origin.timestamp() % step or end.timestamp() % step:
            raise ValueError('Frozen bounds are not UTC grid-aligned.')
        edges = np.arange(0, (end-origin).total_seconds()+step/2, step)
        output = pd.DataFrame(dict(interval_start_utc=origin+pd.to_timedelta(edges[:-1], unit='s'),
                                   interval_end_utc=origin+pd.to_timedelta(edges[1:], unit='s'),
                                   time_utc=origin+pd.to_timedelta((edges[:-1]+edges[1:])/2, unit='s')))
        for side in ['upstream', 'downstream']:
            raw = pd.read_parquet(self.folder/f'observed_{side}.parquet')
            t = (pd.to_datetime(raw.time, utc=True)-origin).dt.total_seconds().to_numpy()
            q = raw.discharge_cms.to_numpy()
            limit = 2*float(self.review[f'{side}_sampling_minutes'])*60
            supported = (edges[:-1] >= t[0]) & (edges[1:] <= t[-1])
            values = np.full(len(edges)-1, np.nan)
            flags = np.zeros(len(values), bool)
            ids = np.flatnonzero(supported)
            if len(ids):
                values[ids], flags[ids], _ = resample_points(t, q, edges[ids[0]:ids[-1]+2],
                    maximum_gap_seconds=limit, diagnostic_gap_fill=True)
            output[side+'_cms'] = values
            output[side+'_gap'] = flags
            output[side+'_supported'] = supported & ~flags
            if side == 'upstream':
                if max(t[0]-edges[0], 0) > 600:
                    raise ValueError('Initial endpoint extension exceeds reviewed limit.')
                flow, gaps, extension = resample_points(t, q, edges, maximum_gap_seconds=limit,
                    endpoint_hold_seconds=3*86400+300, diagnostic_gap_fill=diagnostic_gaps)
                output['forcing_cms'] = flow
                output['upstream_gap'] = gaps
                output['forcing_endpoint_hold'] = extension
        if step == 60:
            for col in ['forcing_cms', 'upstream_cms', 'downstream_cms']:
                np.testing.assert_allclose(output[col], self.wave[col], rtol=1e-12, atol=1e-9, equal_nan=True)
        return output


def sparse_parity(frame, flow, step, dt, initial):
    """Independent sparse-engine verification; temporary all-reach records only."""
    from .engine import run_prepared_state
    with tempfile.TemporaryDirectory(prefix='rapid-common-grid-') as temp:
        prep = Path(temp)
        manifest = cunge.write_prepared_state(frame, np.r_[initial, flow], step, dt, prep,
            output_averaging='linear_state_interval_mean')
        target, q = run_prepared_state(prep, return_qout=True)
        outlet = q[1:, np.asarray(manifest['outlet_reach_ids'])-1].sum(axis=1)
        audit = json.loads(target.with_suffix('.audit.json').read_text())
    predicted, _ = route_interval_means(frame, flow, int(round(step/dt)), dt, initial_flow=initial)
    error = float(np.max(abs(outlet-predicted)))
    np.testing.assert_allclose(outlet, predicted, rtol=1e-10, atol=1e-8)
    if audit['minimum_substep_flow_cms'] < -1e-9:
        raise ValueError('Sparse engine produced negative native discharge.')
    return dict(max_difference_cms=error, **audit)


def exterior_check(example, parent, child, parent_fp, child_fp, paths):
    keys = ['u', 'v', 'link_length_m', 'width', 'slope', 'depth_m']
    matched = parent.merge(child, on=keys, suffixes=('_p', '_c'))
    checks = {col: int((~np.isclose(matched[col+'_p'], matched[col+'_c'], rtol=1e-10, atol=1e-8)).sum())
              for col in ['reference_discharge_cms', 'rapid_celerity_mps', 'rapid_k', 'rapid_x', 'count']}
    unchanged = not any(checks.values())
    accepted = unchanged
    if not accepted and not any(checks[k] for k in ['rapid_celerity_mps', 'rapid_k', 'count']):
        records = json.loads(REVIEW.read_text())['sources']
        for record in records:
            if record['example'] == example and record['key'] == parent_fp+'_'+child_fp:
                accepted = (paths.get(record['parent_frame']) == record['parent_sha256'] and
                            paths.get(record['child_frame']) == record['child_sha256'])
    return dict(parent_fingerprint=parent_fp, child_fingerprint=child_fp, accepted=bool(accepted),
                acceptance='unchanged' if unchanged else 'reviewed_flow_redistribution' if accepted else 'unreviewed_change',
                shared_links=len(matched), mismatches=checks)


def run_case(bundle, example, output, *, assets=DEFAULT_ASSETS, pair_limit=0, all_states=False,
             steps=(60, 30), diagnostic_gaps=False, engine_check=False):
    if pair_limit or all_states:
        inventory_path = Path(assets)/str(example).zfill(2)/'inventory.csv'
        inv = pd.read_csv(inventory_path, dtype={'example': str})
        from network_variants.reviewed_geometry import active_geometry
        active_geometry().assert_hydraulics_current(inv)
        if (~inv.status.isin(['ready', 'excluded_topology_review', 'excluded_duplicate'])).any():
            raise ValueError('Unresolved inventory statuses; review inputs before routing.')
    case = CalibrationCase(bundle, example)
    output = Path(output)/case.example
    output.mkdir(parents=True, exist_ok=False)
    paths = list(case.paths)+[Path(__file__), Path(cunge.__file__)]
    from . import common_grid, engine, time_support, k_values
    paths += [Path(m.__file__) for m in [common_grid, engine, time_support, k_values]]
    paths.append(ROOT/'RAPID/investigations/cunge_full_hierarchy_reviewed.py')
    frames = {'base': refined_parameters(case.grid, case.selection)}
    comparisons, checks = [], []
    if pair_limit or all_states:
        paths.extend([inventory_path, REVIEW])
        inv = inv[inv.status.eq('ready')]
        candidates = inv[inv.parent_state.notna() & inv.parent_state.ne(inv.state)]
        selected = []
        seen = set()
        for row in candidates.itertuples():
            parent = inv[inv.family.eq(row.family) & inv.state.eq(row.parent_state)]
            if parent.empty and row.parent_state == 'S000_base':
                parent = inv[inv.state.eq('S000_base')].drop_duplicates('fingerprint')
            if len(parent) != 1:
                raise ValueError('Ambiguous hierarchy parent.')
            p = parent.iloc[0]
            key = (p.fingerprint, row.fingerprint)
            if key in seen:
                continue
            seen.add(key)
            selected.append((p, row))
            if not all_states and len(selected) >= pair_limit:
                break
        if not selected:
            raise ValueError('No eligible parent-child transitions selected.')
        inv.to_csv(output/'source_inventory.csv', index=False)
        for p, row in selected:
            for record in [p, row]:
                if record.fingerprint not in frames:
                    path = Path(record.frame)
                    paths.append(path)
                    grid = cunge.freeze_grid(pd.read_parquet(path), case.reference_q)
                    frames[record.fingerprint] = refined_parameters(grid, case.selection)
                    if record.state == 'S000_base':
                        columns = ['u', 'v', 'link_length_m', 'width', 'depth_m', 'slope',
                                   'reference_discharge_cms', 'count', 'rapid_k', 'rapid_x']
                        pd.testing.assert_frame_equal(frames['base'][columns].reset_index(drop=True),
                            frames[record.fingerprint][columns].reset_index(drop=True))
            comparisons.append(dict(family=row.family, parent_state=row.parent_state, child_state=row.state,
                                    parent_fingerprint=p.fingerprint, child_fingerprint=row.fingerprint))
        hashes = {str(p): sha(p) for p in paths}
        for pair in comparisons:
            checks.append(exterior_check(case.example, frames[pair['parent_fingerprint']],
                frames[pair['child_fingerprint']], pair['parent_fingerprint'], pair['child_fingerprint'], hashes))
        write_json(output/'exterior_checks.json', checks)
        if not all(c['accepted'] for c in checks):
            raise ValueError('Unaudited exterior changes; no routing started. See exterior_checks.json.')
    hashes = {str(p): sha(p) for p in paths}
    code = output/'source_code'
    code.mkdir()
    for p in paths:
        if p.suffix == '.py':
            (code/p.name).write_bytes(p.read_bytes())
    dt = shared_clock(list(frames.values()), steps, case.native_dt)
    metadata = dict(example=case.example, selection=case.selection, report_intervals_seconds=list(steps),
        native_dt_seconds=dt, initialization='equilibrium at frozen initial forcing; no added antecedent',
        source_hashes=hashes, beta_and_n_inherited=True, children_calibrated=False,
        output_support='interval means of linearly reconstructed native outlet states; UTC midpoint labels',
        diagnostic_gap_fill=diagnostic_gaps, observation_clock_unresolved=case.example in ['21', '22'],
        final_scientific_acceptance=False, feature='reviewed calibration feature, not legacy first crest',
        tail='frozen diagnostic tail; no full-tail volume or settling acceptance claimed')
    write_json(output/'experiment.json', metadata)
    waves, records = {}, []
    for step in steps:
        wave = case.forcing(step, diagnostic_gaps=diagnostic_gaps)
        wave.to_parquet(output/f'forcing_{step}s.parquet', index=False)
        write_json(output/f'support_{step}s.json', dict(
            upstream_gap_bins=int(wave.upstream_gap.sum()),
            downstream_gap_bins=int(wave.downstream_gap.sum()),
            forcing_endpoint_hold_bins=int(wave.forcing_endpoint_hold.sum()),
            downstream_supported_bins=int(wave.downstream_supported.sum()),
            feature_window='frozen downstream observation extent, retaining historical diagnostic gap policy',
            scores_eligible=False if diagnostic_gaps or case.example in ['21', '22'] else True))
        valid = wave.downstream_cms.notna().to_numpy()
        t = wave.time_utc.astype('int64').to_numpy()/1e9
        for name, frame in frames.items():
            q, audit = route_interval_means(frame, wave.forcing_cms.to_numpy(), int(round(step/dt)), dt,
                                            initial_flow=case.initial)
            if audit['minimum_native_flow_cms'] < -1e-9:
                raise ValueError('Negative native flow: '+name)
            tm, height, kind = model_feature(q[valid], t[valid], case.example, case.review)
            record = dict(state=name, interval_seconds=step, feature_utc=pd.Timestamp(tm, unit='s', tz='UTC').isoformat(),
                feature_seconds=tm, peak_cms=height, feature_kind=kind,
                target_error_minutes=(tm-pd.Timestamp(case.review['target_utc']).timestamp())/60, **audit)
            if name == 'base' and step == 60:
                expected = pd.read_parquet(case.bundle/case.example/'wave.parquet').calibrated_cunge_cms.to_numpy()
                record['reviewed_wave_max_difference_cms'] = float(np.max(abs(q-expected)))
                # Shared clock may tighten for children; only same-clock replay must be bitwise-close.
                if np.isclose(dt, case.native_dt):
                    np.testing.assert_allclose(q, expected, rtol=1e-12, atol=1e-9)
            directory = output/'states'/name
            directory.mkdir(parents=True, exist_ok=True)
            frame.to_parquet(directory/'parameters.parquet', index=False)
            result = wave[['interval_start_utc', 'interval_end_utc', 'time_utc']].copy()
            result['q_outlet_cms'] = q
            result.to_parquet(directory/f'outlet_{step}s.parquet', index=False)
            waves[(name, step)] = q
            records.append(record)
            print(case.example, name, step, 'complete', flush=True)
    resolution = []
    if 60 in steps and 30 in steps:
        forcing60 = pd.read_parquet(output/'forcing_60s.parquet').forcing_cms.to_numpy()
        forcing30 = pd.read_parquet(output/'forcing_30s.parquet').forcing_cms.to_numpy().reshape(-1, 2).mean(axis=1)
        np.testing.assert_allclose(forcing60, forcing30, rtol=1e-12, atol=1e-9)
        amp = max(float(np.ptp(forcing60)), 1e-12)
        for name in frames:
            difference = waves[(name, 60)]-waves[(name, 30)].reshape(-1, 2).mean(axis=1)
            rows = {r['interval_seconds']: r for r in records if r['state'] == name}
            resolution.append(dict(state=name, max_difference_cms=float(abs(difference).max()),
                max_difference_pct_inlet_amplitude=float(abs(difference).max()/amp*100),
                feature_shift_seconds=rows[60]['feature_seconds']-rows[30]['feature_seconds'],
                feature_kind_same=rows[60]['feature_kind'] == rows[30]['feature_kind']))
    pair_results = []
    for pair in comparisons:
        deltas = {}
        for step in steps:
            p = next(r for r in records if r['state'] == pair['parent_fingerprint'] and r['interval_seconds'] == step)
            c = next(r for r in records if r['state'] == pair['child_fingerprint'] and r['interval_seconds'] == step)
            deltas[step] = c['feature_seconds']-p['feature_seconds']
            pair_results.append(dict(**pair, interval_seconds=step, delta_feature_seconds=deltas[step],
                delta_peak_cms=c['peak_cms']-p['peak_cms'], feature_kind_same=p['feature_kind'] == c['feature_kind']))
        if 60 in deltas and 30 in deltas:
            pair_results[-1]['delta_timing_resolution_difference_seconds'] = deltas[60]-deltas[30]
    if engine_check:
        # Short real forcing excerpt includes its rise/crest and a deliberately distinct initial state.
        flow = case.forcing(60, diagnostic_gaps=diagnostic_gaps).forcing_cms.to_numpy()
        center = int(np.argmax(flow))
        excerpt = flow[max(0, center-60):center+61]
        parity = {name: sparse_parity(frame, excerpt, 60, dt, case.initial) for name, frame in frames.items()}
        write_json(output/'sparse_engine_parity.json', parity)
    for path, expected in hashes.items():
        if sha(path) != expected:
            raise ValueError('Source changed during run: '+path)
    write_json(output/'results.json', records)
    write_json(output/'resolution_checks.json', resolution)
    write_json(output/'parent_child_comparison.json', pair_results)
    write_json(output/'completion.json', dict(all_requested_states_routed=True, unique_states=len(frames),
        parent_child_pairs=len(comparisons), final_scientific_acceptance=False,
        unresolved=['observation clock/gap eligibility where flagged', 'full-tail settling and volume assessment',
                    'full hierarchy batch not run' if not all_states else 'hierarchy metrics require review']))
    return records


def cli(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--calibration-bundle', type=Path, default=DEFAULT_BUNDLE)
    parser.add_argument('--examples', nargs='+', default=['03'])
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--assets-root', type=Path, default=DEFAULT_ASSETS)
    parser.add_argument('--pairs', type=int, default=0, help='First N unique reviewed transitions per example; 0 replays bases only.')
    parser.add_argument('--all-states', action='store_true')
    parser.add_argument('--steps', nargs='+', type=int, default=[60, 30])
    parser.add_argument('--diagnostic-gaps', action='store_true', help='Explicit diagnostic replay of frozen gap interpolation; never marks final acceptance.')
    parser.add_argument('--engine-check', action='store_true', help='Independent sparse-engine check on a crest excerpt for each state.')
    args = parser.parse_args(argv)
    if args.pairs < 0 or not args.steps or len(set(args.steps)) != len(args.steps) or any(s not in [30, 60] for s in args.steps):
        parser.error('Nonnegative pair count and unique 30/60-second steps required.')
    for ex in args.examples:
        run_case(args.calibration_bundle, ex, args.output_root, assets=args.assets_root, pair_limit=args.pairs,
                 all_states=args.all_states, steps=tuple(args.steps), diagnostic_gaps=args.diagnostic_gaps,
                 engine_check=args.engine_check)


if __name__ == '__main__':
    cli()
