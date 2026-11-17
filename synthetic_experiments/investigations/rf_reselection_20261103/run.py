"""Checkpointed, nested grouped RF feature search; no routing or real-data fitting."""
import os
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
from pathlib import Path
import sys, json, hashlib, time, fcntl, argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'RAPID/investigations/streamlined_predictors_20261103'))
import run_comparison as base
HERE = Path(__file__).resolve().parent
OUT = HERE / 'results'
SOURCE = ROOT / 'RAPID/outputs/streamlined_predictors_20261103'
PRE = HERE.parent / 'individual_predictor_screen_20261102/results'
FOLDS = HERE.parent / 'forward_selection_20261102/results/case_folds.csv'
REQUIRED = {'length', 'width', 'slope', 'topology'}
EXCLUDED = {
    'path_parent_equivalent_width': 'Sum of overlapping complete-path widths; route-count amplification.',
    'path_delta_equivalent_width': 'Difference inherits the same summed path-width problem.',
    'path_parent_elongation': 'Divides by summed equivalent path width.',
    'path_delta_elongation': 'Difference of elongations inheriting summed path-width denominator.',
}

def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True))
    tmp.replace(path)

def category(row):
    c = row['concept']
    if c == 'Topological architecture': return 'topology'
    if c == 'Reach slope distribution': return 'slope'
    if c == 'Reach width distribution': return 'width'
    if c in ['Allocated route length distribution', 'Equally weighted route length distribution',
             'Branch distance contrasts', 'Affected domain extent', 'Equivalent path length',
             'Total channel length', 'Route elongation relative to shortest route']: return 'length'
    return 'combined'

def prepare():
    OUT.mkdir(exist_ok=True)
    files = [Path(__file__), PRE/'screened_candidates.csv', FOLDS,
             SOURCE/'synthetic_input.parquet', SOURCE/'design.json']
    hashes = {str(p): base.sha(p) for p in files}
    if (OUT/'design.json').exists():
        design = json.loads((OUT/'design.json').read_text())
        assert design['sources'] == hashes, 'Frozen source changed; do not reuse checkpoints.'
        return design
    meta = pd.read_csv(PRE/'screened_candidates.csv')
    assert len(meta) == 100 and set(EXCLUDED) <= set(meta.screen_variable)
    meta['excluded_reason'] = meta.screen_variable.map(EXCLUDED).fillna('')
    meta['category'] = meta.apply(category, axis=1)
    meta['model_column'] = meta.screen_variable.replace({
        'B_ratio_width_mean': 'log1p__B_ratio_width_mean', 'domain_size': 'domain_size_m'})
    meta.to_csv(OUT/'candidate_ledger.csv', index=False)
    pool = meta[meta.excluded_reason.eq('')]
    d = pd.read_parquet(SOURCE/'synthetic_input.parquet')
    assert len(d) == 45927 and d.groupby('example').size().eq(7).all()
    assert np.isfinite(d[pool.model_column]).all().all()
    f = pd.read_csv(FOLDS).set_index('case')
    assert set(d.case) == set(f.index)
    design = dict(reference='SYN-RF-RESELECTION-2026-11-03', sources=hashes,
        candidates=dict(zip(pool.screen_variable, pool.model_column)),
        categories=dict(zip(pool.screen_variable, pool.category)), exclusions=EXCLUDED,
        params=json.loads((SOURCE/'design.json').read_text())['params'],
        response='centroid_shift_hours', depth='fixed_manning', seed=20261014,
        n_rows=len(d), n_cases=d.case.nunique(), candidates_with_G=len(pool),
        candidates_without_G=len(pool)-1, max_predictors=10, tolerance_r2=.005,
        search='Coverage-constrained greedy additions; two same-category coordinate-swap passes at size four. Smallest path within .005 of best path. Three outer and three inner case-group folds, copied from prior search. With G means eligible, not mandatory.',
        scope='Centroid primary response only. No real targets enter this search. All 100 prior complete working classes restored before four structural exclusions; no standalone outcome cutoff.',
        caveats='Post-hoc candidate redesign informed by real-network failure; prior candidate grouping inspected full synthetic population. Nested scores conditional on this pool, not independent validation of entire discovery. Greedy search is not exhaustive. Normalized complete-path descriptors remain candidates and require real-network scrutiny.',
        transformations='Mean-width ratio uses existing log1p representation; domain extent uses metres. Other candidates retain existing definitions. Sorted feature order provides deterministic subset caching.')
    save(OUT/'design.json', design)
    return design

DATA = None
def init_worker(design):
    global DATA, DESIGN, OUTER, INNER
    DESIGN = design
    DATA = pd.read_parquet(SOURCE/'synthetic_input.parquet')
    f = pd.read_csv(FOLDS).set_index('case')
    OUTER = DATA.case.map(f.outer_fold).to_numpy()
    INNER = DATA.case.map(f.inner_label).to_numpy()

def evaluate(job):
    outer, subset = job
    subset = tuple(sorted(subset))
    key = hashlib.sha256(json.dumps([outer, subset]).encode()).hexdigest()
    file = OUT/'cache'/f'{key}.json'
    if file.exists(): return json.loads(file.read_text())
    b = base.legacy.refit.tuning.b
    b.COLUMNS = [DESIGN['candidates'][v] for v in subset]
    start = time.monotonic()
    mask = OUTER != outer
    y = DATA.loc[mask, 'centroid_shift_hours'].to_numpy()
    den = np.sum((y-y.mean())**2)
    sse = 0.; sae = 0.; folds = []
    for k in range(3):
        tr = DATA.loc[mask & (INNER != k)]
        te = DATA.loc[mask & (INNER == k)]
        fitted = b.fit(tr, DESIGN['params'], jobs=1)
        pred = b.forest.predict(fitted, te)
        assert np.isfinite(pred).all()
        err = pred-te.centroid_shift_hours.to_numpy()
        sse += float(err@err); sae += float(np.abs(err).sum())
        folds.append(dict(inner=k, n_train=len(tr), n_test=len(te),
                          kept=list(fitted['preprocessor'].kept)))
    result = dict(outer=outer, variables=list(subset), r2=1-sse/den,
        mae_minutes=60*sae/len(y), seconds=time.monotonic()-start, folds=folds)
    save(file, result)
    return result

def outer_fit(outer, variant, subset, design):
    file = OUT/f'outer_{outer}_{variant}.parquet'
    if file.exists(): return
    b = base.legacy.refit.tuning.b
    b.COLUMNS = [design['candidates'][v] for v in sorted(subset)]
    tr, te = DATA.loc[OUTER != outer], DATA.loc[OUTER == outer]
    model = b.fit(tr, design['params'], jobs=1)
    z = te[['case', 'child_state', 'centroid_shift_hours']].copy()
    z['predicted_hours'] = b.forest.predict(model, te)
    z['outer'] = outer; z['variant'] = variant
    tmp = file.with_suffix('.tmp.parquet'); z.to_parquet(tmp, index=False); tmp.replace(file)

def run(design):
    init_worker(design)
    start = time.monotonic()
    with ProcessPoolExecutor(max_workers=2, initializer=init_worker, initargs=(design,)) as executor:
        for outer in range(3):
            memory = {}
            def scores(subsets):
                keys = sorted(set(tuple(sorted(s)) for s in subsets))
                jobs = {executor.submit(evaluate, (outer, k)): k for k in keys if k not in memory}
                for future in as_completed(jobs):
                    r = future.result(); memory[jobs[future]] = r['r2']
                    save(OUT/'progress.json', dict(status='running', outer=outer,
                        current_variant=variant, current_step=step,
                        evaluated_subsets_this_outer=len(memory), remaining_current_batch=sum(not j.done() for j in jobs),
                        elapsed_seconds=time.monotonic()-start))
                return [(memory[tuple(sorted(s))], s) for s in subsets]
            for variant in ['with_G', 'without_G']:
                selected_file = OUT/f'selection_{outer}_{variant}.json'
                if selected_file.exists(): continue
                pool = sorted(v for v in design['candidates'] if variant=='with_G' or v!='delta_G_manning')
                chosen=[]; path=[]
                for step in range(1, 11):
                    missing = REQUIRED-{design['categories'][v] for v in chosen}
                    opts = [v for v in pool if v not in chosen and (not missing or design['categories'][v] in missing)]
                    trials = scores([chosen+[v] for v in opts])
                    best, chosen = min(trials, key=lambda z: (-z[0], tuple(z[1])))
                    if step == 4:
                        for sweep in range(2):
                            improved=False
                            for pos in range(4):
                                rest=chosen[:pos]+chosen[pos+1:]
                                opts=[v for v in pool if v not in rest and design['categories'][v]==design['categories'][chosen[pos]]]
                                sc, candidate=min(scores([rest+[v] for v in opts]), key=lambda z:(-z[0],tuple(z[1])))
                                if sc>best+1e-10: chosen=candidate;best=sc;improved=True
                            if not improved: break
                    if step>=4: path.append(dict(n=step, r2=best, variables=chosen.copy()))
                    save(OUT/f'path_{outer}_{variant}.json', path)
                    print(f'Outer {outer+1}/3 {variant} size {step}: inner R2 {best:.8f}; {chosen}', flush=True)
                    # Exact bound for the size rule: the first valid size cannot
                    # be displaced by any later score <= 1 if already >= .995.
                    if step==4 and best>=1-design['tolerance_r2']: break
                maximum=max(p['r2'] for p in path)
                selected=next(p for p in path if p['r2']>=maximum-design['tolerance_r2'])
                assert REQUIRED <= {design['categories'][v] for v in selected['variables']}
                outer_fit(outer, variant, selected['variables'], design)
                save(selected_file, dict(selected=selected, best_path_r2=maximum,
                    bound_stop=len(path)==1, outer=outer, variant=variant))
    summarize()
    save(OUT/'completion.json', dict(complete=True, seconds=time.monotonic()-start))
    save(OUT/'progress.json', dict(status='complete'))

def summarize():
    selected=[]; rows=[]
    for file in sorted(OUT.glob('selection_*.json')):
        r=json.loads(file.read_text())
        selected.extend(dict(outer=r['outer'], variant=r['variant'], variable=v) for v in r['selected']['variables'])
    pd.DataFrame(selected).to_csv(OUT/'selected_variables_by_fold.csv', index=False)
    p=pd.concat([pd.read_parquet(f) for f in sorted(OUT.glob('outer_*.parquet'))])
    p.to_parquet(OUT/'outer_predictions.parquet', index=False)
    for variant,z in p.groupby('variant'):
        y=z.centroid_shift_hours.to_numpy(); err=z.predicted_hours.to_numpy()-y
        rows.append(dict(variant=variant,n=len(y),r2=1-(err@err)/np.sum((y-y.mean())**2),mae_minutes=60*np.abs(err).mean()))
    pd.DataFrame(rows).to_csv(OUT/'summary.csv',index=False)
    pd.DataFrame(selected).groupby(['variant','variable']).size().rename('selected_outer_folds').to_csv(OUT/'selection_frequency.csv')

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--run',action='store_true');parser.add_argument('--benchmark',action='store_true')
    args=parser.parse_args();design=prepare()
    with (OUT/'run.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
        if args.benchmark:
            init_worker(design)
            for subset in [['B_parent_width_mean'], ['B_delta_route_length_mean','parent_slope_slow','B_ratio_width_mean','path_delta_topologic_complexity_score','delta_G_manning']]:
                print(json.dumps(evaluate((0,subset))),flush=True)
        elif args.run: run(design)
        else: print(json.dumps(design,indent=2))
