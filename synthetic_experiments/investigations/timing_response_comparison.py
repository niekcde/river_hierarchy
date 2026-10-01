"""Compare timing definitions on saved selected runs; do not choose by R² alone.

No routing or real-gauge results are changed. Fixed original-development models
are evaluated on the previous holdout and novel extension; no feature search.
"""
from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score
from threadpoolctl import threadpool_limits

from analysis_paths import RUN, EXTENSION
from current_synthetic_analysis import estimator

OUT = RUN / 'current_analysis/timing_response_comparison'
METRICS = {
    'lag_change_pct': ('Lag change', '%', '100 * (Tc/Tp - 1)', 'network-relative'),
    'lag_change_hours': ('Absolute timing change', 'h', '(Tc - Tp)/3600', 'model time'),
    'network_speed_change_mps': ('Network speed change', 'm/s', 'Ln * (1/Tc - 1/Tp)', 'model apparent network speed'),
    'lag_per_parent_leq_min_km': ('Timing / parent Leq', 'min/km', '(Tc-Tp)/60/(Leq_p/1000)', 'parent-domain normalization'),
    'reference_speed_change_mps': ('Reference-scaled speed change', 'm/s', '(Ln/Tref) * (Tp/Tc - 1)', 'synthetic centerline reference; not gauges'),
    'reference_lag_per_parent_leq_min_km': ('Reference timing / parent Leq', 'min/km', '(Tc/Tp-1)*Tref/60/(Leq_p/1000)', 'synthetic centerline reference; not gauges'),
    'equivalent_speed_change_pct': ('State-specific equivalent-speed index', '%', '100 * ((Leq_c/Leq_p)*(Tp/Tc)-1)', 'supplementary index; network times divided by local lengths'),
}
CORE = list(METRICS)[:4]
KEYS = ['child_state', 'outer_a_fraction', 'length_multiplier']


def responses(frame):
    """Times in seconds, lengths in metres; all denominators must be positive."""
    d = frame.copy()
    tp, tc = d.parent_lag_seconds, d.child_lag_seconds
    lp, lc = d.parent_equivalent_length, d.child_equivalent_length
    length, tref = d.network_reference_length_m, d.reference_lag_seconds
    assert np.isfinite(d[['parent_lag_seconds', 'child_lag_seconds', 'parent_equivalent_length',
                          'child_equivalent_length', 'network_reference_length_m', 'reference_lag_seconds']]).all().all()
    assert ((tp > 0) & (tc > 0) & (lp > 0) & (lc > 0) & (length > 0) & (tref > 0)).all()
    r = tc / tp - 1
    d['lag_change_pct'] = 100*r
    d['lag_change_hours'] = (tc-tp)/3600
    d['network_speed_change_mps'] = length*(1/tc-1/tp)
    d['lag_per_parent_leq_min_km'] = (tc-tp)/60/(lp/1000)
    d['reference_speed_change_mps'] = length/tref*(-r/(1+r))
    d['reference_lag_per_parent_leq_min_km'] = r*tref/60/(lp/1000)
    d['equivalent_speed_change_pct'] = 100*((lc/lp)*(tp/tc)-1)
    d['equivalent_slowness_change_pct'] = 100*((tc/tp)*(lp/lc)-1)
    d['reference_time_scale'] = tref/tp
    d['exposure_resistance'] = d.width_fraction*d.resistance_log
    assert np.isfinite(d[list(METRICS)]).all().all()
    return d


def load(root, file):
    d = pd.read_parquet(file)
    states = pd.read_csv(root/'states.csv')
    refs = states[states.state_id == 'centerline/original'][['configuration_id', 'run_id']].copy()
    assert not refs.configuration_id.duplicated().any()
    values = {rid: json.loads((root/'runs'/rid/'metrics.json').read_text())['peak_lag_seconds']
              for rid in refs.run_id.unique()}
    refs['reference_lag_seconds'] = refs.run_id.map(values)
    d = d.merge(refs.rename(columns={'run_id': 'reference_run_id'}), on='configuration_id', validate='many_to_one')
    d['network_reference_length_m'] = d.network_extent_m if 'network_extent_m' in d else d.total_length_m
    d = responses(d)
    assert np.allclose(d.lag_change_pct, d.lag_shift_pct, atol=1e-10)
    return d


def algebra_checks():
    # Pure added delay preserves absolute/local timing, not network averages.
    d = pd.DataFrame(dict(parent_lag_seconds=[100., 200., 100.], child_lag_seconds=[80., 180., 50.],
        parent_equivalent_length=[20., 20., 20.], child_equivalent_length=[20., 20., 10.],
        network_reference_length_m=[100., 200., 100.], reference_lag_seconds=[100., 200., 100.],
        width_fraction=[1., 1., 1.], resistance_log=[1., 1., 1.]))
    q = responses(d)
    assert np.allclose(q.lag_change_pct, [-20., -10., -50.])
    assert np.allclose(q.network_speed_change_mps[:2], [.25, 1/9])
    assert np.allclose(q.lag_per_parent_leq_min_km[:2], [-1000/60, -1000/60])
    assert np.allclose(q.reference_lag_per_parent_leq_min_km, q.lag_per_parent_leq_min_km)
    assert np.allclose(q.reference_speed_change_mps, q.network_speed_change_mps)
    assert np.isclose(q.equivalent_speed_change_pct.iloc[2], 0)
    assert np.allclose(q.equivalent_speed_change_pct,
                       -100*q.equivalent_slowness_change_pct/(100+q.equivalent_slowness_change_pct))
    return 'Added-delay, unit conversion, reference identity, reciprocal-speed and pure-shortening checks passed.'


def embedding(d):
    rows = []
    for key, g in d.groupby(KEYS):
        g = g.sort_values('network_extent_m')
        assert len(g) == 4 and g.network_extent_m.iloc[0] == 20000
        assert np.allclose(g.parent_equivalent_length, g.parent_equivalent_length.iloc[0])
        # Same geometry at all four extents. Weak changes remain in plots/models,
        # but are separated from relative-drift summaries to avoid 0/0 effects.
        resolved = bool((g.lag_shift_seconds.abs() > 3*g.timing_resolution_seconds).all())
        for metric in METRICS:
            y = g[metric].to_numpy(); baseline = y[0]
            drift = np.max(np.abs(y-baseline))/abs(baseline) if abs(baseline) > 1e-12 else np.nan
            rows.append(dict(zip(KEYS, key), metric=metric, resolved_all_extents=resolved,
                value_20km=y[0], value_800km=y[-1], absolute_range=float(np.ptp(y)), relative_max_drift=drift))
    groups = pd.DataFrame(rows); groups.to_csv(OUT/'embedding_groups.csv', index=False)
    result = []
    for metric, g in groups.groupby('metric'):
        for scope, h in [('all', g), ('resolved_gt3dt', g[g.resolved_all_extents])]:
            z = h.relative_max_drift.dropna()
            result.append(dict(metric=metric, scope=scope, n_groups=len(h), n_nonzero_baselines=len(z),
                median_relative_drift=z.median(), p90_relative_drift=z.quantile(.9), maximum_relative_drift=z.max()))
    pd.DataFrame(result).to_csv(OUT/'embedding_summary.csv', index=False)
    moments = []
    for key, g in d.groupby(KEYS):
        g = g.sort_values('network_extent_m')
        centroid = (g.child_centroid_seconds-g.parent_centroid_seconds).to_numpy()
        peak = g.lag_shift_seconds.to_numpy()
        for label, y in [('peak', peak), ('centroid_diagnostic', centroid)]:
            moments.append(dict(zip(KEYS, key), timing=label, value_20km_seconds=y[0],
                maximum_departure_seconds=float(np.max(abs(y-y[0]))),
                relative_max_drift=float(np.max(abs(y-y[0]))/abs(y[0])) if abs(y[0]) > 1e-12 else np.nan))
    pd.DataFrame(moments).to_csv(OUT/'centroid_embedding_diagnostic.csv', index=False)


def benchmark(original, ext):
    selection = json.loads((RUN/'current_analysis/compact_predictors/selection.json').read_text())
    six = selection['features'][selection['named_sets']['selected_shared']]
    change = ['log_equivalent_length_ratio', 'log_elongation_ratio', 'change_path_length_cv']
    context = ['upstream_shortest_length_m', 'downstream_shortest_length_m']
    sets = {'omega': ['omega'], 'components': ['resistance_log', 'length_fraction', 'width_fraction'],
        'local_resistance_product': ['exposure_resistance'], 'selected_six': six,
        'six_plus_change_no_delta_dominance': six+change,
        'six_plus_change_and_context': six+change+context,
        'six_plus_change_with_delta_dominance': six+change+['change_width_dominance']}
    train = original[original.split == 'development']; test = original[original.split == 'test']
    assert not set(train.geometry_group) & set(test.geometry_group)
    novel = ext[~ext.is_original_grid_anchor]
    scopes = {'original_holdout': test, 'extension_all': novel,
        'extension_20km': novel[novel.network_extent_m == 20000],
        'extension_embedded': novel[novel.network_extent_m > 20000]}
    rows, predictions = [], []
    with threadpool_limits(limits=1):
        for metric in METRICS:
            for label, cols in sets.items():
                model = estimator('boosted').fit(train[cols], train[metric])
                for scope, data in scopes.items():
                    pred = model.predict(data[cols])
                    q = data[['configuration_id', 'child_state', metric]].rename(columns={metric: 'observed'}).copy()
                    q['predicted'] = pred
                    for state, g in list(q.groupby('child_state')) + [('pooled', q)]:
                        rows.append(dict(metric=metric, feature_set=label, scope=scope, child_state=state,
                            n=len(g), r2=r2_score(g.observed, g.predicted) if g.observed.var(ddof=0) > 1e-20 else np.nan,
                            mae=mean_absolute_error(g.observed, g.predicted)))
                    if scope in ['original_holdout', 'extension_all']:
                        predictions.append(q.assign(metric=metric, feature_set=label, scope=scope))
            print('Benchmarked', metric, flush=True)
    scores = pd.DataFrame(rows); scores.to_csv(OUT/'predictor_scores.csv', index=False)
    pd.concat(predictions, ignore_index=True).to_parquet(OUT/'predictor_predictions.parquet', index=False)
    summary = scores[scores.child_state != 'pooled'].groupby(['metric', 'feature_set', 'scope'])[['r2', 'mae']].mean().reset_index()
    summary.to_csv(OUT/'predictor_summary.csv', index=False)
    (OUT/'feature_sets.json').write_text(json.dumps(sets, indent=2))
    return sets


def main():
    OUT.mkdir(exist_ok=True, parents=True)
    checks = algebra_checks()
    sources = [RUN/'current_analysis/compact_predictors/features.parquet',
               EXTENSION/'predictor_validation/full_features.parquet']
    original = load(RUN, sources[0]); ext = load(EXTENSION, sources[1])
    # Every overlapping configuration must reproduce every new metric.
    old = original[(original.total_length_m == 20000) & (original.total_width_m == 500) &
        (original.reference_slope == 1e-4) & (original.inner_fraction == .5) & (original.internal_location == 'central')]
    anchors = ext[ext.is_original_grid_anchor].merge(old, on=KEYS, suffixes=('_ext', '_original'), validate='one_to_one')
    assert len(anchors) == 63
    for metric in METRICS: assert np.allclose(anchors[metric+'_ext'], anchors[metric+'_original'], atol=1e-8, rtol=1e-8), metric
    original.to_parquet(OUT/'original_metrics.parquet', index=False)
    ext.to_parquet(OUT/'extension_metrics.parquet', index=False)
    embedding(ext)
    sets = benchmark(original, ext)
    records = []
    for name, d in [('original', original), ('extension', ext)]:
        for state, g in d.groupby('child_state'):
            for metric in METRICS:
                records.append(dict(dataset=name, child_state=state, metric=metric, n=len(g),
                    minimum=g[metric].min(), median=g[metric].median(), maximum=g[metric].max()))
    pd.DataFrame(records).to_csv(OUT/'response_ranges.csv', index=False)
    pd.DataFrame([dict(metric=k, label=v[0], units=v[1], formula=v[2], meaning=v[3]) for k,v in METRICS.items()]).to_csv(OUT/'metric_definitions.csv', index=False)
    provenance = dict(reference='SYN-2026-09-30-TIMING-OPTIONS', metrics=METRICS, feature_sets=sets,
        source_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
        plan_ids={root.name: json.loads((root/'manifest.json').read_text())['plan_id'] for root in [RUN, EXTENSION]},
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), algebra_checks=checks,
        n_original=len(original), n_extension=len(ext), n_verified_anchors=len(anchors),
        timing_reference='Synthetic single-centerline run for the same configuration/clock; no observed gauge calibration available.',
        validation='Fixed shallow boosted estimator and original development/holdout geometry groups. No feature search, retuning, extension fitting, or real-case reruns.',
        interpretation='Metric choice uses scientific meaning and embedding stability, not highest R2. Leq_p is width-weighted and also an input; normalization introduces mathematical coupling. Unit-normalized network timing change is not local unit celerity.',
        selection_status='Candidates compared; existing primary metric contract remains unchanged pending interpretation.')
    (OUT/'provenance.json').write_text(json.dumps(provenance, indent=2))
    print(checks); print('Saved', OUT)


if __name__ == '__main__':
    main()
