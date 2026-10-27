"""Catalogue responses under the user-selected common linear background.

Run calculate_reviewed_responses_20261013.py first. No production outputs change.
Negative excess invalidates probability-distribution metrics; never silently clip.
"""
from pathlib import Path
import hashlib
import json
import sys

import numpy as np
import pandas as pd
from scipy.signal import find_peaks

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'RAPID/src')]
from rapid_tools.common_grid_production import model_feature
sys.path.insert(0, str(Path(__file__).parent / 'geometry_response_foundation_20261012'))
from responses import event_distribution, analytic_checks

OUT = ROOT / 'RAPID/outputs/reviewed_responses_20261013'
cases = json.loads((ROOT / 'RAPID/outputs/calibration_review/plateau_review_20261013.json').read_text())['cases']
pairs = pd.read_parquet(OUT / 'peak_responses.parquet')
target54 = next(r for r in json.loads((ROOT / 'RAPID/outputs/bounded_beta_calibration_20261009/timing_targets.json').read_text())['records'] if r['example'] == '54')
span54 = (pd.Timestamp(target54['crest_end_utc']) - pd.Timestamp(target54['crest_start_utc'])).total_seconds()
separation54 = max(60., int(span54 / 180.) * 60.)


def linear_distribution(t, q):
    """Exact integrals of the disclosed linear reconstruction of interval means."""
    h = np.diff(t)
    a, b = q[:-1], q[1:]
    mass = h * (a + b) / 2
    volume = mass.sum()
    if q.min() < 0 or volume <= 0:
        raise ValueError('Nonnegative positive-volume reconstruction required')
    mean = np.sum(t[:-1] * mass + h * h * (a / 2 + (b - a) / 3)) / volume
    u = t[:-1] - mean
    variance = np.sum(u*u*mass + 2*u*h*h*(a/2+(b-a)/3) + h**3*(a/3+(b-a)/4)) / volume
    cumulative = np.r_[0., np.cumsum(mass)]
    def quantile(p):
        i = min(np.searchsorted(cumulative, p*volume, side='right') - 1, len(h)-1)
        remaining = p*volume - cumulative[i]
        slope = (b[i] - a[i]) / h[i]
        root = np.sqrt(max(0., a[i]**2 + 2*slope*remaining))
        offset = 2*remaining/(a[i]+root) if remaining else 0.
        return t[i] + offset
    t10,t25,t50,t75,t90 = [quantile(p) for p in [.1,.25,.5,.75,.9]]
    return dict(volume_m3=float(volume), centroid_seconds=float(mean), temporal_sd_seconds=float(np.sqrt(variance)),
        t10_seconds=t10,t25_seconds=t25,t50_seconds=t50,t75_seconds=t75,t90_seconds=t90,
        volume_width80_seconds=t90-t10,volume_width50_seconds=t75-t25,
        quantile_asymmetry=((t90-t50)-(t50-t10))/(t90-t10))


def read_wave(path):
    wave = pd.read_parquet(path)
    t = pd.to_datetime(wave.time_utc, utc=True).astype('int64').to_numpy() / 1e9
    q = wave.q_outlet_cms.to_numpy()
    assert np.isfinite(q).all() and np.allclose(np.diff(t), 60)
    return t, q


def crest(t, excess, peak_index, fraction):
    level = fraction * excess[peak_index]
    missing = dict(width_seconds=np.nan, rise_seconds=np.nan, fall_seconds=np.nan)
    if level <= 0:
        return missing
    left = np.flatnonzero(excess[:peak_index] <= level)
    right = np.flatnonzero(excess[peak_index + 1:] <= level)
    if not len(left) or not len(right):
        return missing
    j, k = int(left[-1]), int(peak_index + 1 + right[0])
    tl = t[j] + (t[j + 1] - t[j]) * (level - excess[j]) / (excess[j + 1] - excess[j])
    tr = t[k - 1] + (t[k] - t[k - 1]) * (excess[k - 1] - level) / (excess[k - 1] - excess[k])
    return dict(width_seconds=tr - tl, rise_seconds=t[peak_index] - tl, fall_seconds=tr - t[peak_index])


def analyze(t, q, ex):
    case = cases[ex]
    candidate = next(c for c in case['candidates'] if c['key'] == case['default_candidate'])
    selected = next(l for l in candidate['levels'] if l['raise_pct'] == case['default_raise_pct'])
    start = pd.Timestamp(case['start']['start_utc']).timestamp()
    end = pd.Timestamp(selected['crossing_utc']).timestamp()
    q0, qf = case['start']['start_cms'], selected['plateau_cms']
    assert t[0] - 30 <= start < end <= t[-1] + 30
    # Saved discharges are interval means. Clip the boundary bins to the window;
    # use the exact mean of linear B(t) on each retained bin.
    use = (t + 30 > start) & (t - 30 < end)
    tt, qq = t[use], q[use]
    edges = np.r_[start, (tt[:-1] + tt[1:]) / 2, end] - start
    mids = (edges[:-1] + edges[1:]) / 2
    baseline = q0 + (qf - q0) * mids / (end - start)
    excess = qq - baseline
    dt = np.diff(edges)
    negative = np.minimum(excess, 0)
    positive_volume = float(np.sum(np.maximum(excess, 0) * dt))
    deficit = float(-np.sum(negative * dt))
    result = dict(interval_mean_signed_excess_volume_m3=float(np.sum(excess * dt)),
        interval_mean_negative_excess_volume_m3=deficit,
        baseline_start_utc=pd.Timestamp(start, unit='s', tz='UTC').isoformat(),
        baseline_end_utc=pd.Timestamp(end, unit='s', tz='UTC').isoformat(),
        baseline_start_cms=q0, baseline_end_cms=qf)
    # Crest crossings use linear interpolation of reported interval means, with
    # exact interpolated endpoint values. This is separate from volume integration.
    interior = (t > start) & (t < end)
    ct = np.r_[start, t[interior], end]
    cq = np.interp(ct, t, q)
    ce = cq - (q0 + (qf - q0) * (ct - start) / (end - start))
    # Insert zero crossings to integrate positive and negative portions exactly.
    change = np.flatnonzero(ce[:-1] * ce[1:] < 0)
    zeros = ct[change] - ce[change] * np.diff(ct)[change] / np.diff(ce)[change]
    zt = np.sort(np.r_[ct, zeros])
    ze = np.interp(zt, ct, ce)
    deficit = float(-np.trapz(np.minimum(ze, 0), zt))
    positive_volume = float(np.trapz(np.maximum(ze, 0), zt))
    result.update(signed_excess_volume_m3=float(np.trapz(ce, ct)),
        negative_excess_volume_m3=deficit, minimum_excess_cms=float(ce.min()),
        negative_to_positive_volume_pct=100*deficit/positive_volume if positive_volume else np.nan,
        distribution_status='negative_excess' if ce.min() < -1e-8 else 'valid')
    peak = int(np.argmax(ce))
    result.update(peak_excess_cms=float(ce[peak]), excess_peak_seconds=float(ct[peak] - start),
        endpoint_excess_cms=float(ce[-1]))
    for fraction in [.5, .9]:
        for key, value in crest(ct, ce, peak, fraction).items():
            result[f'crest{int(100 * fraction)}_{key}'] = value
    distribution = None
    if result['distribution_status'] == 'valid':
        # Only roundoff within 1e-8 m³/s is zeroed, and its volume is recorded above.
        corrected = np.maximum(ce, 0)
        moments = linear_distribution(ct - start, corrected)
        result.update({k: v for k, v in moments.items() if k != 'peak_excess_cms'})
        distribution = (ct - start, corrected, moments)
    # The user's end level: first downward crossing after this state's event peak.
    # Keep the reviewed bounded-recession limit; never substitute a later rise.
    pk = np.flatnonzero((t >= start) & (t <= end))[np.argmax(q[(t >= start) & (t <= end)])]
    stop = pd.Timestamp(selected['recession_limit_utc']).timestamp()
    inds = np.flatnonzero((t[1:] <= stop) & (np.arange(1, len(t)) > pk) &
                          (q[:-1] > qf) & (q[1:] <= qf)) + 1
    result['reviewed_end_crossing_seconds'] = np.nan
    if len(inds):
        j = inds[0]
        result['reviewed_end_crossing_seconds'] = t[j - 1] + 60 * (q[j - 1] - qf) / (q[j - 1] - q[j]) - start
    result['reviewed_end_status'] = 'crossed' if len(inds) else 'not_reached_in_reviewed_recession'
    if ex == '54':
        peaks, _ = find_peaks(q, prominence=max(np.ptp(q) * 1e-4, 1e-8), distance=int(np.ceil(separation54 / 60)))
        peaks = peaks[q[peaks] >= q.min() + .8 * np.ptp(q)]
        best = np.sort(peaks[np.argsort(q[peaks])[-2:]])
        assert len(best) == 2
        for label, index in zip(['left', 'right'], best):
            result[label + '_lobe_seconds'] = t[index] - start
            result[label + '_lobe_cms'] = q[index]
            ci = int(np.argmin(abs(ct - t[index])))
            for fraction in [.5, .9]:
                for key, value in crest(ct, ce, ci, fraction).items():
                    result[f'{label}_crest{int(fraction * 100)}_{key}'] = value
        result['center_lobe_seconds'] = (result['left_lobe_seconds'] + result['right_lobe_seconds']) / 2
    return result, distribution


def shape_distance(parent, child):
    # Exact L1 distance between standardized piecewise-linear densities.
    def standardize(d):
        edges, q, m = d
        return ((edges - m['centroid_seconds']) / m['temporal_sd_seconds'],
                q * m['temporal_sd_seconds'] / m['volume_m3'])
    pe, pq = standardize(parent)
    ce, cq = standardize(child)
    edges = np.unique(np.r_[pe, ce])
    h = np.diff(edges)
    def difference(x):
        return np.interp(x, pe, pq, left=0, right=0) - np.interp(x, ce, cq, left=0, right=0)
    d25, d75 = difference(edges[:-1] + .25*h), difference(edges[:-1] + .75*h)
    a, b = 1.5*d25-.5*d75, 1.5*d75-.5*d25
    area = h*(abs(a)+abs(b))/2
    flip = a*b < 0
    area[flip] = h[flip]*(a[flip]**2+b[flip]**2)/(2*(abs(a[flip])+abs(b[flip])))
    value = .5 * area.sum()
    assert -1e-10 <= value <= 1 + 1e-10
    return float(value)


def main():
    analytic_checks()
    triangle = linear_distribution(np.array([0.,1.,2.]), np.array([0.,1.,0.]))
    np.testing.assert_allclose([triangle['volume_m3'], triangle['centroid_seconds'], triangle['temporal_sd_seconds'], triangle['t50_seconds']], [1,1,np.sqrt(1/6),1])
    shifted = linear_distribution(np.array([10.,12.,14.]), np.array([0.,3.,0.]))
    np.testing.assert_allclose(shape_distance((np.array([0.,1.,2.]),np.array([0.,1.,0.]),triangle), (np.array([10.,12.,14.]),np.array([0.,3.,0.]),shifted)), 0, atol=1e-12)
    state_rows, response_rows, hashes = [], [], {}
    for ex, group in pairs.groupby('example_id', sort=True):
        base_path = group.iloc[0].parent_outlet_parquet
        bt, bq = read_wave(base_path)
        forcing = pd.read_parquet(group.iloc[0].forcing_parquet)
        ft = pd.to_datetime(forcing.time_utc, utc=True).astype('int64').to_numpy() / 1e9
        fq = forcing.forcing_cms.to_numpy()
        inlet_feature, inlet_peak, inlet_kind = model_feature(fq, ft, ex, target54 if ex == '54' else {})
        feature_lag = pd.Timestamp(group.iloc[0].parent_feature_utc).timestamp() - inlet_feature
        feature_valid = feature_lag > 60 and (ex != '54' or inlet_kind == 'two_lobe_midpoint')
        parent, pdist = analyze(bt, bq, ex)
        state_rows.append(dict(example_id=ex, state_id='S000_base', **parent))
        hashes[base_path] = hashlib.sha256(Path(base_path).read_bytes()).hexdigest()
        for record in group.to_dict('records'):
            t, q = read_wave(record['outlet_parquet'])
            np.testing.assert_allclose([bq.max(), q.max()], [record['parent_peak_cms'], record['child_peak_cms']], rtol=1e-10)
            assert abs(t[q.argmax()] - pd.Timestamp(record['child_peak_utc']).timestamp()) < 1e-6
            hashes[record['outlet_parquet']] = hashlib.sha256(Path(record['outlet_parquet']).read_bytes()).hexdigest()
            child, cdist = analyze(t, q, ex)
            record.update(inlet_feature_kind=inlet_kind, parent_reviewed_feature_lag_seconds=feature_lag,
                reviewed_feature_relative_change_pct=100*record['reviewed_feature_shift_minutes']*60/feature_lag if feature_valid else np.nan,
                reviewed_relative_timing_status='valid' if feature_valid else 'unmatched inlet feature or lag <= one reporting interval',
                parent_total_peak_attenuation_pct=100*(1-bq.max()/inlet_peak),
                child_total_peak_attenuation_pct=100*(1-q.max()/inlet_peak))
            state_rows.append(dict(example_id=ex, state_id=record['state_id'], fingerprint=record['fingerprint'], **child))
            for label, values in [('parent', parent), ('child', child)]:
                record.update({label + '_' + k: v for k, v in values.items()})
            for key in ['peak_excess_cms', 'signed_excess_volume_m3', 'volume_m3', 'temporal_sd_seconds',
                        'volume_width80_seconds', 'volume_width50_seconds', 'reviewed_end_crossing_seconds']:
                p, c = parent.get(key, np.nan), child.get(key, np.nan)
                record[key + '_change_pct'] = 100 * (c / p - 1) if p > 0 else np.nan
            for key in ['centroid_seconds', 't50_seconds', 'temporal_sd_seconds', 'volume_width80_seconds',
                        'volume_width50_seconds', 'quantile_asymmetry', 'excess_peak_seconds',
                        'reviewed_end_crossing_seconds']:
                record[key + '_change'] = child.get(key, np.nan) - parent.get(key, np.nan)
            for key in parent:
                if key.startswith(('crest50_', 'crest90_', 'left_', 'right_', 'center_')):
                    record[key + '_change'] = child[key] - parent[key]
            record['standardized_shape_distance'] = shape_distance(pdist, cdist) if pdist and cdist else np.nan
            record['event_distribution_status'] = 'valid' if pdist and cdist else 'negative_excess_in_parent_or_child'
            record['reviewed_duration_response_status'] = 'Fixed reviewed start to each state end crossing; not departure-to-settlement duration'
            response_rows.append(record)
        print(ex, len(group), 'pairs; parent distribution:', parent['distribution_status'],
              'negative/positive volume %:', round(parent['negative_to_positive_volume_pct'], 5), flush=True)
    states = pd.DataFrame(state_rows)
    responses = pd.DataFrame(response_rows)
    states.to_csv(OUT / 'state_metrics.csv', index=False)
    responses.to_csv(OUT / 'responses.csv', index=False)
    responses.to_parquet(OUT / 'responses.parquet', index=False)
    coverage = responses.assign(valid_distribution=responses.event_distribution_status.eq('valid')).groupby('example_id').agg(
        pairs=('state_id', 'size'), valid_distributions=('valid_distribution', 'sum'),
        peak_excess=('peak_excess_cms_change_pct', 'count'), end_crossings=('reviewed_end_crossing_seconds_change', 'count'),
        widths50=('crest50_width_seconds_change', 'count'), widths90=('crest90_width_seconds_change', 'count'))
    coverage.to_csv(OUT / 'full_coverage.csv')
    manifest_path = OUT / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    manifest.update(status='Catalogue evaluated with explicit QC; see full_coverage.csv for valid counts',
        background='User-approved shared straight line between reviewed base start/end, on common absolute-time window',
        integration='Exact integration of linear interpolation between saved one-minute interval means, with interpolated window endpoints. Interval-weighted signed volume retained as a reconstruction sensitivity diagnostic.',
        negative_excess_policy='No substantive clipping. Values below -1e-8 m3/s invalidate distribution metrics; roundoff-only zeroing is recorded in deficit diagnostics.',
        duration='Fixed reviewed start to state-specific downward end crossing is a separate diagnostic, not the historical departure-to-permanent-settlement duration',
        shape='Exact L1 of standardized linearly reconstructed densities on the selected finite window; not a full-tail claim',
        hydrograph_sha256=hashes, distribution_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(coverage.to_string())


if __name__ == '__main__':
    main()
