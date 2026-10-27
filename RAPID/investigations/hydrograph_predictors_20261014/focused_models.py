"""Agreed outer-domain predictors: prepare by default, fit only with --run.

Independent artifacts; reuses validated fitting primitives without changing the
earlier search or its checkpoint hashes. No adaptive variable-subset selection.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
import model_search as engine

ROOT = engine.ROOT
OUT = ROOT/'RAPID/outputs/hydrograph_focused_models_20261017'
RATIOS = ['B_ratio_route_length_mean', 'B_ratio_route_length_sd',
          'B_ratio_route_length_cv', 'B_ratio_width_mean',
          'B_ratio_width_cv', 'B_ratio_route_entropy']
CONTEXT = ['B_parent_dominance_mean', 'B_exposure', 'B_K_log_a1']
SYMBOLS = dict(zip(RATIOS+CONTEXT, [r'R_{\mu_L}', r'R_{\sigma_L}',
    r'R_{\mathrm{CV}_L}', r'R_{\bar W}', r'R_{\mathrm{CV}_W}', r'R_H',
    r'\bar d_p', r'r_{\mathrm{in},p}', r'-\log_{10}(K_c/K_p)']))
GRIDS = {
    'ridge': [dict(alpha=a) for a in [.01, .1, 1., 10., 100., 1000.]],
    'elastic_net': [dict(alpha=a, l1_ratio=r) for a in [.001, .01, .1] for r in [.2, .8]],
    'forest': [dict(max_depth=d, min_samples_leaf=l, max_features=1.)
               for d in [3, 6, None] for l in [5, 15]],
    'boosting': [dict(max_leaf_nodes=l, l2_regularization=r, learning_rate=.06)
                 for l in [7, 15] for r in [0., 1., 10.]],
}


def specifications():
    result = []
    for spread in ['sd', 'cv']:
        ratios = [RATIOS[0], f'B_ratio_route_length_{spread}', *RATIOS[3:]]
        for transform in ['raw', 'log1p']:
            columns = [c if transform == 'raw' else 'log1p__'+c for c in ratios]+CONTEXT
            for model, grid in GRIDS.items():
                result.append(dict(name=f'{spread}__{transform}__{model}', columns=columns,
                                   model=model, grid=grid, corr=None, top_k=None))
    return result


def transform(data):
    d = data.copy()
    for c in RATIOS:
        x = d[c].to_numpy(float)
        d['log1p__'+c] = np.log1p(np.where(np.isfinite(x) & (x >= 0), x, np.nan))
    return d


def prepare(out):
    data, join = engine.load_data()
    # Same resistance eligibility as the previous benchmark, on identical rows
    # for every raw/logged and SD/CV candidate. No response-driven filtering.
    reasons = []
    for r in data.itertuples():
        bad = []
        if not r.eligible:
            bad.append('previous resistance/domain quality exclusion')
        for c in RATIOS+CONTEXT:
            value = getattr(r, c)
            if not np.isfinite(value):
                bad.append(c+': undefined/nonfinite')
            elif c in RATIOS and value < 0:
                bad.append(c+': negative ratio')
        if bad:
            reasons.append(dict(example=r.example, state=r.state, reason='; '.join(bad)))
    data = transform(data)
    valid = np.isfinite(data[RATIOS+CONTEXT]).all(axis=1) & data[RATIOS].ge(0).all(axis=1)
    data['eligible'] = data.eligible & valid
    sources = [engine.SOURCE/'linked_responses.parquet', engine.SOURCE/'domain_graphs.csv',
               ROOT/'RAPID/outputs/hydrograph_response/responses.parquet',
               ROOT/'RAPID/outputs/hydrograph_response/sensitivity_responses.parquet',
               ROOT/'RAPID/outputs/hydrograph_response/manifest.json', Path(__file__), Path(engine.__file__)]
    design = dict(specifications=specifications(), targets=engine.TARGETS, primary=engine.PRIMARY,
        validation='Leave one entire example out; 3 grouped inner folds tune settings separately for each fixed specification. Equal-example training weights and inner MSE. No selection across specifications.',
        domains='Outer: C for compound-member edits, closed U otherwise, N for centerlines. Same domain boundaries before/after the same edit.',
        logarithms='Natural log(1+R) applied to the five nonnegative geometric ratios; zero maps to zero. Dominance and exposure stay raw; resistance retains -log10(Kc/Kp), alpha=1.',
        preprocessing='Training-only scaling, constants and exact duplicate removal. No relevance screening or correlation pruning. Complete common input cohort. Elastic Net may shrink coefficients to zero.',
        exclusions='Nonfinite/negative ratios or unavailable parent dominance excluded explicitly; no epsilon floors or invented dominance. Previous resistance exclusions retained for comparability.',
        versions=dict(numpy=np.__version__, pandas=pd.__version__, sklearn=engine.sklearn.__version__),
        source_hashes={str(p.relative_to(ROOT)): engine.sha(p) for p in sources})
    out.mkdir(parents=True, exist_ok=True)
    design_path = out/'design.json'
    if design_path.exists() and json.loads(design_path.read_text()) != design and any((out/'checkpoints').glob('**/*.json')):
        raise ValueError('Inputs/code/design changed. Use a fresh --output directory; checkpoints preserved.')
    engine.dump(design_path, design)
    join.to_csv(out/'join_audit.csv', index=False)
    pd.DataFrame(reasons, columns=['example','state','reason']).to_csv(out/'exclusions.csv', index=False)
    audit = []
    for c in RATIOS+CONTEXT:
        x = data.loc[data.eligible, c]
        audit.append(dict(feature=c, symbol='$'+SYMBOLS[c]+'$', n=len(x), zero=int(x.eq(0).sum()),
                          unique=x.nunique(), minimum=x.min(), maximum=x.max()))
    pd.DataFrame(audit).to_csv(out/'input_audit.csv', index=False)
    rows = []
    for name in ['partial', 'including_centerlines']:
        for scenario in ['all', 'without54']:
            d = engine.cohort(data, name, scenario)
            rows.append(dict(cohort=name, scenario=scenario, n=len(d), examples=d.example.nunique(),
                fits_per_target=d.example.nunique()*sum(3*len(s['grid'])+1 for s in specifications())))
    pd.DataFrame(rows).to_csv(out/'run_plan.csv', index=False)
    print(pd.DataFrame(rows).to_string(index=False), flush=True)
    print('Preparation complete. Fitting starts only when --run is supplied.', flush=True)
    return data, design


def summarize(out, design, data):
    fingerprint = hashlib.sha256(json.dumps(design, sort_keys=True).encode()).hexdigest()
    records = {}
    for path in sorted((out/'checkpoints').glob('*/*/*/*/*.json')):
        r = json.loads(path.read_text())
        assert r['fingerprint'] == fingerprint
        key = (r['cohort'], r['scenario'], r['target'], r['spec'])
        records.setdefault(key, []).append(r)
    predictions, settings, features = [], [], []
    completed = set()
    for (cohort, scenario, target, spec), folds in records.items():
        d = engine.cohort(data, cohort, scenario)
        if {r['held'] for r in folds} != set(d.example):
            continue
        completed.add((cohort, scenario, target))
        for r in folds:
            base = dict(cohort=cohort, scenario=scenario, response=target, pipeline=spec)
            predictions.extend(dict(**base, **p) for p in r['predictions'])
            settings.append(dict(**base, held_example=r['held'], inner_mse=r['inner_mse'], params=json.dumps(r['params'])))
            for c in r['audit']['kept']:
                features.append(dict(**base, held_example=r['held'], feature=c,
                    coefficient=r['audit'].get('standardized_coefficients', {}).get(c)))
    for cohort, scenario, target in completed:
        d = engine.cohort(data, cohort, scenario)
        for held, test in d.groupby('example'):
            train = d[d.example.ne(held)]
            mean = float(np.average(train[target], weights=engine.weights(train)))
            for label, value in [('zero', 0.), ('training_mean', mean)]:
                predictions.extend(dict(cohort=cohort, scenario=scenario, response=target, pipeline=label,
                    example=r.example, state=r.state, observed=float(getattr(r, target)), predicted=value) for r in test.itertuples())
    if not predictions:
        return
    pred = pd.DataFrame(predictions)
    assert not pred.duplicated(['cohort','scenario','response','pipeline','example','state']).any()
    pred.to_parquet(out/'predictions.parquet', index=False)
    pd.DataFrame(settings).to_csv(out/'chosen_hyperparameters.csv', index=False)
    pd.DataFrame(features).to_csv(out/'retained_features.csv', index=False)
    scores, sensitivity_scores = [], []
    sensitivity = pd.read_parquet(ROOT/'RAPID/outputs/hydrograph_response/sensitivity_responses.parquet')
    sensitivity['example'] = sensitivity.example.astype(str).str.zfill(2)
    keys = ['cohort','scenario','response','pipeline']
    for key, g in pred.groupby(keys):
        base = dict(zip(keys, key))
        for label, w in [('equal_example', engine.weights(g)), ('pair', np.ones(len(g)))]:
            scores.append(dict(**base, weighting=label, n=len(g), **engine.score(g.observed,g.predicted,w)))
        for ex, gg in g.groupby('example'):
            scores.append(dict(**base, weighting='example_'+ex, n=len(gg), **engine.score(gg.observed,gg.predicted,np.ones(len(gg)))))
        for window in ['1%','2%']:
            target = key[2]
            alt = sensitivity[sensitivity.window.eq(window)][['example','state',target]]
            z = g.merge(alt,on=['example','state'],validate='one_to_one')
            assert len(z)==len(g)
            sensitivity_scores.append(dict(**base,window=window,**engine.score(z[target],z.predicted,engine.weights(z))))
    pd.DataFrame(scores).to_csv(out/'scores.csv',index=False)
    pd.DataFrame(sensitivity_scores).to_csv(out/'sensitivity_scores.csv',index=False)


def run(data, design, out, args):
    fingerprint = hashlib.sha256(json.dumps(design, sort_keys=True).encode()).hexdigest()
    for cohort in args.cohorts:
        for scenario in args.scenarios:
            d = engine.cohort(data, cohort, scenario)
            for target in args.targets:
                for held in sorted(d.example.unique()):
                    train, test = d[d.example.ne(held)], d[d.example.eq(held)]
                    inner_groups = [dict(train=sorted(train.iloc[a].example.unique()),validation=sorted(train.iloc[b].example.unique()))
                                    for a,b in engine.inner_splits(train)]
                    for spec in design['specifications']:
                        path = out/'checkpoints'/cohort/scenario/target/held/(spec['name']+'.json')
                        if path.exists():
                            assert json.loads(path.read_text())['fingerprint'] == fingerprint
                            continue
                        best,trials = engine.choose(train,target,spec,args.jobs)
                        prediction,audit = engine.fit_predict(train,test,target,spec,best['params'],args.jobs)
                        engine.dump(path,dict(fingerprint=fingerprint,cohort=cohort,scenario=scenario,target=target,
                            held=held,spec=spec['name'],inner_groups=inner_groups,params=best['params'],inner_mse=best['mse'],
                            trials=trials,audit=audit,predictions=[dict(example=r.example,state=r.state,
                            observed=float(getattr(r,target)),predicted=float(p)) for r,p in zip(test.itertuples(),prediction)]))
                    print(cohort,scenario,target,'held',held,'all fixed specifications complete',flush=True)
                summarize(out,design,data)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',action='store_true')
    parser.add_argument('--output',type=Path,default=OUT)
    parser.add_argument('--jobs',type=int,default=1,help='CPU workers within each random forest; outer fits sequential.')
    parser.add_argument('--cohorts',nargs='+',choices=['partial','including_centerlines'],default=['partial'])
    parser.add_argument('--scenarios',nargs='+',choices=['all','without54'],default=['all','without54'])
    parser.add_argument('--targets',nargs='+',choices=engine.TARGETS,default=engine.PRIMARY)
    args=parser.parse_args()
    data,design=prepare(args.output)
    if args.run:
        run(data,design,args.output,args)


if __name__=='__main__':
    main()
