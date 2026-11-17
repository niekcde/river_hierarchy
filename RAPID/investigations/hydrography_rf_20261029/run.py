"""Geometry-only RF feature comparison, frozen compound-family nested splits."""
import os
for key in ['OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ[key] = '1'
import argparse
import fcntl
import hashlib
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
# Import the existing driver by its unique module name; its `import run` must
# resolve to the previous feature builder, not this entry-point script.
sys.path.insert(0, str(ROOT/'RAPID/investigations/fixed_manning_rf_20261029'))
import nested_three_strategies as nested
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

refit = nested.refit
DEST = refit.BASE/'hydrography_rf_compound_20261029'
OLD = refit.BASE/'fixed_manning_rf_nested_20261029'
GEOMETRY = ['id_link', 'u', 'v', 'id_us_node', 'id_ds_node', 'link_length_m', 'width', 'slope']


def descriptors(frame, selected):
    """Only whitelisted geometry can enter the descriptor calculation."""
    f = frame[GEOMETRY].copy()
    a = refit.describe(f, selected)
    edges = refit.allocation_tables(f)[1].set_index('id_link')
    z = f.set_index('id_link').loc[sorted(selected)]
    w = edges.loc[z.index, 'mean_length_contribution_m']
    assert (z.slope > 0).all()
    a['geometry_J_m'] = float((w / np.sqrt(z.slope)).sum())
    a['slope_slow'] = float(np.average(1/np.sqrt(z.slope), weights=w))
    return a


def changes(a, b):
    assert a['route_length_mean'] > 0
    return dict(
        delta_spread_km=(b['route_length_sd']-a['route_length_sd'])/1000,
        delta_spread_normalized=(b['route_length_sd']-a['route_length_sd'])/a['route_length_mean'],
        delta_entropy=b['route_entropy']-a['route_entropy'],
        parent_mean_length_km=a['route_length_mean']/1000,
        parent_spread_km=a['route_length_sd']/1000,
        parent_entropy=a['route_entropy'],
        delta_geometry_J_km=(b['geometry_J_m']-a['geometry_J_m'])/1000,
        parent_geometry_J_km=a['geometry_J_m']/1000,
    )


def prepare():
    DEST.mkdir(exist_ok=True)
    if (DEST/'design.json').exists():
        design = nested.load(DEST/'design.json')
        for p, h in design['sources'].items():
            assert refit.sha(p) == h, p
        assert refit.sha(DEST/'input.parquet') == design['input_sha256']
        fp = hashlib.sha256(json.dumps(design, sort_keys=True).encode()).hexdigest()
        saved = nested.load(DEST/'splits.json')
        assert saved['fingerprint'] == fp
        return design, fp, saved['tasks']

    old_design = refit.read(OLD/'design.json')
    old_input = refit.read(OLD/'input.parquet')
    old_splits = refit.read(OLD/'splits.json')
    assert old_splits['fingerprint'] == hashlib.sha256(json.dumps(old_design, sort_keys=True).encode()).hexdigest()
    original = old_design['panels']['geometry_9']
    # Explicitly exclude depth-informed columns and all other simulated outcomes.
    d = old_input[['example', 'state', 'cohort', 'group_id', 'edit_type',
                   'centroid_shift_hours'] + original].copy()
    records = refit.read(refit.BASE/'validation_group_inventory_20261023/records.parquet')
    rows = []
    for ex, group in records.groupby('example'):
        parent = refit.active(ex, 'S000_base')[GEOMETRY]
        for r in group.itertuples():
            child = refit.active(ex, r.state)[GEOMETRY]
            if r.cohort == 'segmented':
                lineage = refit.read(refit.EXT/f'variants/{ex}/{r.state}/lineage.csv')
                lineage = lineage[lineage.role == 'collapsed_block']
                p, c = parent, child
                pmask = set(map(int, lineage.parent_link_id))
                cmask = set(map(int, lineage.child_link_id))
            else:
                p = parent[parent.id_link.isin(r.descriptor_links)].copy()
                s, t = refit.terminal(p)
                p, _ = refit.validate_domain(parent, p)
                c = refit.extract_complete_directed_domain(child, int(s), int(t))
                c, _ = refit.validate_domain(child, c)
                pmask, cmask = set(p.id_link), set(c.id_link)
            a, b = descriptors(p, pmask), descriptors(c, cmask)
            old = d[d.example.eq(ex) & d.state.eq(r.state)].iloc[0]
            for k in ['route_length_mean', 'route_length_sd', 'width_mean', 'width_cv', 'route_entropy']:
                assert a[k] > 0, ('Historical ratio undefined', ex, r.state, k)
                np.testing.assert_allclose(np.log1p(b[k]/a[k]), old['log1p__B_ratio_'+k], rtol=1e-10, atol=1e-10)
            np.testing.assert_allclose(a['slope_slow'], old.parent_slope_slow)
            rows.append(dict(example=ex, state=r.state, **changes(a, b)))
        print('Prepared geometry', ex, len(group), flush=True)
    d = d.merge(pd.DataFrame(rows), on=['example', 'state'], validate='one_to_one', how='left')
    absolute = ['delta_spread_km' if x == 'log1p__B_ratio_route_length_sd' else
                'delta_entropy' if x == 'log1p__B_ratio_route_entropy' else x for x in original]
    normalized = ['delta_spread_normalized' if x == 'delta_spread_km' else x for x in absolute]
    panels = dict(geometry_ratios_9=original, geometry_differences_9=absolute,
                  geometry_normalized_9=normalized,
                  geometry_context_14=absolute+['parent_mean_length_km', 'parent_spread_km',
                      'parent_entropy', 'delta_geometry_J_km', 'parent_geometry_J_km'])
    predictors = sorted(set(sum(panels.values(), [])))
    assert np.isfinite(d[predictors+['centroid_shift_hours']]).all().all()
    assert len(d) == 2185 and d.group_id.nunique() == 450
    np.testing.assert_array_equal(d[['example', 'state']], old_input[['example', 'state']])
    tasks = [dict(t) for t in old_splits['tasks'] if t['strategy'] == 'compound_mixed']
    assert len(tasks) == 25
    for t in tasks:
        nested.audit_task(d, t)
    for repeat in range(5):
        assert sorted(sum([t['test_indices'] for t in tasks if t['repeat'] == repeat], [])) == list(range(len(d)))
    d.to_parquet(DEST/'input.parquet', index=False)
    dependencies = [Path(__file__), Path(nested.__file__), Path(refit.__file__),
                    Path(refit.tuning.b.__file__), Path(refit.tuning.b.engine.__file__),
                    Path(refit.tuning.b.forest.__file__),
                    ROOT/'RAPID/src/rapid_tools/current_domains.py',
                    ROOT/'RAPID/investigations/closed_unit_extension_20261019/edited_features.py',
                    ROOT/'RAPID/investigations/cap_investigation_20261020/augmented_tests.py',
                    ROOT/'RAPID/investigations/cap_investigation_20261020/ten_input_outlier_audit.py']
    design = dict(panels=panels, candidates=old_design['candidates'],
        sources=refit.HASHES | {str(p): refit.sha(p) for p in dependencies},
        input_sha256=refit.sha(DEST/'input.parquet'), versions=old_design['versions'],
        target='Fixed-Manning simulated child-minus-parent centroid shift, finite 0.5% event window; hours.',
        features='Only connectivity, length, width, slope and width-proportional allocation. No n, depth, hydraulic resistance, delta J or simulated-flow predictors. Target never used to construct features.',
        geometry_J='Ggeom=sum(f_e L_e/sqrt(S_e)); f_e is width-derived domain visit probability. Stored in km. Geometry factor of J with n/h^(2/3) omitted, not a physical travel time or full delta J.',
        domains='Identical to historical geometry benchmark: outer comparison domains for regular states; edited-region rewards on full connected graph for segmented states, connectors excluded.',
        zero_ratios='Historical nine-input panel verified finite on this dataset. Zero/zero remains undefined and is never replaced with zero. Difference panels remain defined for zero parent spread/entropy.',
        validation='Frozen historical compound-family outer AND inner splits: 5 repeats x 5 outer folds x 3 inner folds. Known-network prediction, not withheld-network generalization.',
        caveats=['Historical response/type-balanced partitions retained; exploratory feature comparison on previously examined data.',
                 '435 regular family groups plus 15 full-network singleton exceptions; full-network states may share constituents with training states.',
                 'Report local and full-network cohorts separately; repeat ranges are not confidence intervals.'],
        training=old_design['weighting'], baseline='Training-only network-weighted affine regressions on length_delta_km or delta_geometry_J_km, plus zero-shift control; no per-network intercepts.',
        jobs=100, forest_fits=1900, production_modified=False)
    fp = hashlib.sha256(json.dumps(design, sort_keys=True).encode()).hexdigest()
    refit.save(DEST/'design.json', design)
    refit.save(DEST/'splits.json', dict(fingerprint=fp, tasks=tasks))
    return design, fp, tasks


def fit_task(t, model, design, fp):
    nested.DEST = DEST
    return nested.fit_task(t, model, design, fp)


def baselines(tasks):
    d = nested.load(DEST/'input.parquet')
    rows = []
    for t in tasks:
        tr, te = d.iloc[t['train_indices']], d.iloc[t['test_indices']]
        for model, col in [('linear_length', 'length_delta_km'), ('linear_geometry_J', 'delta_geometry_J_km'), ('zero_shift', None)]:
            if col:
                fit = LinearRegression().fit(tr[[col]], tr.centroid_shift_hours,
                                            sample_weight=refit.tuning.b.engine.weights(tr))
                pred = fit.predict(te[[col]])
            else:
                pred = np.zeros(len(te))
            for r, v in zip(te.itertuples(), pred):
                rows.append(dict(strategy=t['strategy'], repeat=t['repeat'], fold=t['fold'], model=model,
                    example=r.example, state=r.state, cohort=r.cohort, edit_type=r.edit_type,
                    centroid_shift_hours=r.centroid_shift_hours, predicted_hours=float(v)))
    pd.DataFrame(rows).to_parquet(DEST/'baseline_predictions.parquet', index=False)


def summarize(design, fp, complete=False):
    rows = []
    paths = sorted((DEST/'checkpoints').glob('*/*/*/*/outer.json'))
    for p in paths:
        result = nested.load(p)
        assert result['fingerprint'] == fp
        rows.extend(result['rows'])
    predictions = pd.concat([pd.DataFrame(rows), nested.load(DEST/'baseline_predictions.parquet')], ignore_index=True)
    assert not predictions.duplicated(['repeat', 'model', 'example', 'state']).any()
    scores, delays = [], []
    for (repeat, model), g in predictions.groupby(['repeat', 'model']):
        if len(g) != 2185:
            continue
        for cohort, z in [('all', g), ('original', g[g.cohort == 'original']), ('segmented', g[g.cohort == 'segmented'])]:
            scores.append(dict(repeat=repeat, model=model, cohort=cohort,
                **nested.score(z, 'equal_cohort_network' if cohort == 'all' else 'equal_network')))
        actual = g.centroid_shift_hours*60 > 1+1e-6
        called = g.predicted_hours*60 > 1+1e-6
        tp = int((actual & called).sum())
        delays.append(dict(repeat=repeat, model=model, actual_delays=int(actual.sum()),
            detected=tp, missed=int((actual & ~called).sum()), false_delays=int((~actual & called).sum()),
            recall=tp/actual.sum(), precision=tp/called.sum() if called.any() else None))
    predictions.to_parquet(DEST/('predictions.parquet' if complete else 'partial_predictions.parquet'), index=False)
    s = pd.DataFrame(scores)
    s.to_csv(DEST/'repeat_scores.csv', index=False)
    agg = s.groupby(['model', 'cohort'])[['r2', 'mae_minutes', 'relative_error_pct']].agg(['mean', 'min', 'max', 'std'])
    agg.columns = ['_'.join(x) for x in agg.columns]
    agg.reset_index().to_csv(DEST/'summary.csv', index=False)
    pd.DataFrame(delays).to_csv(DEST/'delay_detection.csv', index=False)
    if complete:
        assert len(paths) == 100 and len(predictions) == 2185*5*7
        for path, h in design['sources'].items():
            assert refit.sha(path) == h, path
        refit.save(DEST/'completion.json', dict(complete=True, fingerprint=fp, jobs=100, forest_fits=1900,
            production_modified=False, files={str(DEST/x): refit.sha(DEST/x) for x in
                ['predictions.parquet', 'summary.csv', 'delay_detection.csv']}))


def self_check():
    # Complete parallel paths, not link-length dispersion. Splitting a route
    # into more links must not alter its length moments or route entropy.
    def frame(edges):
        f = pd.DataFrame(edges, columns=['u', 'v', 'link_length_m'])
        f['id_link'] = np.arange(len(f)); f['width'] = 30.; f['slope'] = .001
        f['id_us_node'] = f.u; f['id_ds_node'] = f.v
        return f
    p = frame([(0, 1, 10000), (0, 1, 20000)])
    split = frame([(0, 2, 5000), (2, 1, 5000), (0, 1, 20000)])
    c = frame([(0, 1, 15000)])
    a, b, s = [descriptors(f, set(f.id_link)) for f in [p, c, split]]
    np.testing.assert_allclose([a['route_length_mean'], a['route_length_sd'], a['route_entropy']], [15000, 5000, np.log(2)])
    for key in ['route_length_mean', 'route_length_sd', 'route_entropy', 'geometry_J_m']:
        np.testing.assert_allclose(a[key], s[key])
    z = changes(a, b)
    np.testing.assert_allclose([z['delta_spread_km'], z['delta_spread_normalized'], z['delta_entropy']], [-5, -1/3, -np.log(2)])
    equal = frame([(0, 1, 15000), (0, 1, 15000)])
    a = descriptors(equal, set(equal.id_link))
    assert a['route_length_sd'] == 0 and changes(a, b)['delta_spread_km'] == 0
    assert np.isfinite(list(changes(b, b).values())).all()
    # Hydraulics and simulated responses attached to input are ignored.
    altered = p.assign(depth_m=1e9, n_manning=123, centroid_shift_hours=999, reference_flow_fraction=0)
    original, altered = descriptors(p, set(p.id_link)), descriptors(altered, set(p.id_link))
    assert original == altered
    print('Geometry invariance, zero-spread and hydraulic-exclusion checks passed', flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', action='store_true')
    ap.add_argument('--workers', type=int, default=2)
    ap.add_argument('--self-check', action='store_true')
    args = ap.parse_args()
    self_check()
    if args.self_check:
        return
    design, fp, tasks = prepare()
    print('Prepared 4 geometry RF panels, 100 outer jobs, 1900 forest fits', flush=True)
    if not args.run:
        return
    with (DEST/'run.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        start = time.monotonic()
        baselines(tasks)
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(fit_task, t, model, design, fp) for t in tasks for model in design['panels']]
            for i, future in enumerate(as_completed(futures), 1):
                result = future.result()
                refit.save(DEST/'progress.json', dict(status='running', completed_jobs=i, total_jobs=100,
                    elapsed_seconds=time.monotonic()-start, last_checkpoint=result, workers=args.workers))
                print(i, '/100 complete; minutes', round((time.monotonic()-start)/60, 1), flush=True)
                if i % 5 == 0:
                    summarize(design, fp)
        summarize(design, fp, complete=True)
        refit.save(DEST/'progress.json', dict(status='complete', completed_jobs=100, total_jobs=100,
                                           elapsed_seconds=time.monotonic()-start))


if __name__ == '__main__':
    main()
