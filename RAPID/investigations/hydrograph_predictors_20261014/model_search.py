"""Prepare, then explicitly run a resumable nested-example predictor benchmark.

Default command only prepares inventories and a fit-count plan. No production writes.
"""
from pathlib import Path
import argparse
import hashlib
import json
import os
import warnings

import numpy as np
import pandas as pd
import sklearn
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.exceptions import ConvergenceWarning

ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / 'RAPID/outputs/hydrograph_predictor_scopes_20261014'
OUT = ROOT / 'RAPID/outputs/hydrograph_model_search_20261014'
PRIMARY = ['peak_shift_hours', 'peak_excess_change_pct']
TARGETS = PRIMARY + ['centroid_shift_hours', 'duration80_change_pct', 'volume_change_pct', 'asymmetry_change']
SEED = 20261014

# Explicit geometry-only allowlist, not a numeric-column selector.
# One parent descriptor and one change representation per metric. Ratios are
# used for strictly positive scales; differences for zero-prone/bounded metrics.
# Exclude junction excess = cycle rank and branch variance = route SD squared.
METRICS = {
 'total_length': ('L', r'L_{\mathrm{tot}}', 'm', 'ratio'),
 'route_length_mean': ('L', r'\mu_L', 'm', 'ratio'),
 'route_length_sd': ('L', r'\sigma_L', 'm', 'delta'),
 'route_length_cv': ('L', r'\mathrm{CV}_L', '1', 'delta'),
 'route_length_min': ('L', r'L_{\min}', 'm', 'ratio'),
 'route_length_max': ('L', r'L_{\max}', 'm', 'ratio'),
 'route_length_skew': ('L', r'\gamma_L', '1', 'delta'),
 'branch_length_contrast_mean': ('L', r'\bar c_L', 'm', 'delta'),
 'branch_length_contrast_sd': ('L', r'\sigma_{c_L}', 'm', 'delta'),
 'sinuosity_proxy': ('L', r'\mu_L/L_{\min}', '1', 'ratio'),
 'width_mean': ('W', r'\bar W', 'm', 'ratio'),
 'width_sd': ('W', r'\sigma_W', 'm', 'delta'),
 'width_cv': ('W', r'\mathrm{CV}_W', '1', 'delta'),
 'width_harmonic': ('W', r'W_H', 'm', 'ratio'),
 'area_proxy': ('W', r'A_{LW}', 'm²', 'ratio'),
 'area_elongation': ('W', r'\mu_L^2/A_{LW}', '1', 'ratio'),
 'dominance_mean': ('W', r'\bar d', '1', 'delta'),
 'dominance_sd': ('W', r'\sigma_d', '1', 'delta'),
 'effective_branches_mean': ('W', r'\bar n_{\mathrm{eff}}', '1', 'delta'),
 'effective_branches_sd': ('W', r'\sigma_{n_{\mathrm{eff}}}', '1', 'delta'),
 'split_entropy_mean': ('W', r'\bar h_b', '1', 'delta'),
 'split_entropy_sd': ('W', r'\sigma_{h_b}', '1', 'delta'),
 'split_impact_mean': ('W', r'\bar I_b', '1', 'delta'),
 'split_impact_sd': ('W', r'\sigma_{I_b}', '1', 'delta'),
 'split_impact_sum': ('W', r'\sum_b I_b', '1', 'delta'),
 'bifurcations': ('T', r'N_b', '1', 'delta'),
 'cycle_rank': ('T', r'E-V+1', '1', 'delta'),
 'bifurcation_density': ('T', r'\rho_b', 'm⁻¹', 'delta'),
 'log1p_paths': ('T', r'\ln(1+N_{\mathrm{paths}})', '1', 'delta'),
 'expected_splits': ('T', r'\mathbb E[N_b]', '1', 'delta'),
 'route_entropy': ('T', r'H_{\mathrm{route}}', '1', 'delta'),
 'split_exposure_mean': ('X', r'\bar r_b', '1', 'delta'),
 'split_exposure_sd': ('X', r'\sigma_{r_b}', '1', 'delta'),
}
CONTEXT = {
 'exposure': ('X', r'r_{\mathrm{in},p}'),
 'length_fraction': ('X', r'L_{\mathrm{tot},p}^S/L_{\mathrm{tot},p}^N'),
 'mean_length_fraction': ('L', r'\mu_{L,p}^S/\mu_{L,p}^N'),
 'exposed_length_change': ('L', r'r_{\mathrm{in},p}^S\Delta\mu_L^S/\mu_{L,p}^N'),
 'exposed_sd_change': ('L', r'r_{\mathrm{in},p}^S\Delta\sigma_L^S/\mu_{L,p}^N'),
}
GRIDS = {
 'ridge': [dict(alpha=a) for a in [0.1, 10., 100.]],
 'elastic_net': [dict(alpha=a, l1_ratio=r) for a, r in [(0.01, .2), (.03, .8), (.1, .8)]],
 'forest': [dict(max_depth=d, min_samples_leaf=l, max_features=f) for d, l, f in [(4, 8, 1.), (8, 8, .7), (None, 16, .7)]],
 'boosting': [dict(max_leaf_nodes=l, l2_regularization=r, learning_rate=.06) for l, r in [(7, 1.), (15, 5.), (7, 10.)]],
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, obj):
    """Atomic checkpoint; incomplete work is never mistaken for a completed fold."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, indent=2, default=str, allow_nan=False) + '\n')
    os.replace(tmp, path)


def weights(frame):
    w = 1 / frame.groupby('example').example.transform('size').to_numpy(float)
    return w / w.mean()


def score(y, p, w):
    y, p = np.asarray(y), np.asarray(p)
    mse = float(np.average((y-p)**2, weights=w))
    variance = float(np.average((y-np.average(y, weights=w))**2, weights=w))
    return dict(mae=float(np.average(abs(y-p), weights=w)), rmse=float(np.sqrt(mse)),
                r2=1-mse/variance if variance > 0 else None)


def catalog():
    rows = []
    for scope in ['B', 'U', 'C']:
        symbol_scope = 'D' if scope == 'B' else scope
        for metric, (family, symbol, unit, change) in METRICS.items():
            for kind in ['parent', change]:
                label = (rf'({symbol})_p^{{{symbol_scope}}}' if kind == 'parent' else
                         rf'R_{{{symbol}}}^{{{symbol_scope}}}' if kind == 'ratio' else rf'\Delta({symbol})^{{{symbol_scope}}}')
                rows.append(dict(feature=f'{scope}_{kind}_{metric}', scope=scope, family=family,
                                 symbol='$'+label+'$', unit='1' if kind == 'ratio' else unit,
                                 representation=kind, metric=metric))
        for key, (family, symbol) in CONTEXT.items():
            rows.append(dict(feature=f'{scope}_{key}', scope=scope, family=family,
                             symbol='$'+symbol.replace('^S', f'^{{{symbol_scope}}}')+'$', unit='1',
                             representation='context', metric=key))
    # C resistance is already the B resistance on every paired member edit.
    for law in ['a1', 'a065']:
        for prefix, symbol in [('K_log', 'z_K'), ('weighted_K_log', 'z_{K,w}')]:
            rows.append(dict(feature=f'B_{prefix}_{law}', scope='B', family='R',
                             symbol=rf'${symbol}^D(\alpha={"1" if law == "a1" else "0.65"})$',
                             unit='1', representation='log ratio', metric='resistance'))
    return pd.DataFrame(rows)


def panels(cat, paired=False):
    def cols(scope, families):
        return cat.loc[cat.scope.eq(scope) & cat.family.isin(list(families)), 'feature'].tolist()
    resistance = cols('B', 'R')
    if paired:
        u, c = cols('U', 'LWTX'), cols('C', 'LWTX')
        return {'U': u, 'C': c, 'U+C': u+c, 'U+R_C': u+resistance,
                'C+R_C': c+resistance, 'U+C+R_C': u+c+resistance}
    result = {'context': cols('B', 'X')}
    for family in ['L', 'W', 'T', 'R', 'LW', 'LR', 'LWT', 'LWTR']:
        result[family] = cols('B', family+'X')
    for family in 'LWTR':
        result['without_'+family] = cols('B', ''.join(x for x in 'LWTRX' if x != family))
    # Preserve exactly the old small panels, on the same new benchmark records.
    from compare_scopes import PANELS
    for key in ['R_only', 'R_context', 'R_plus_L', 'R_plus_LWT']:
        result['reference_'+key] = PANELS[key]
    return result


def specifications(cat, paired=False):
    specs = []
    for panel, columns in panels(cat, paired).items():
        for model in (['ridge'] if panel.startswith('reference_') else GRIDS):
            specs.append(dict(name=panel+'__'+model, panel=panel, model=model, columns=columns,
                              corr=None, top_k=None,
                              grid=[dict(alpha=10.)] if panel.startswith('reference_') else GRIDS[model]))
    full = 'U+C+R_C' if paired else 'LWTR'
    for model in GRIDS:
        for k in [7, 14]:
            specs.append(dict(name=f'compact{k}__{model}', panel=f'compact{k}', model=model,
                              columns=panels(cat, paired)[full], corr=.95, top_k=k, grid=GRIDS[model]))
    return specs


class Preprocess:
    """Fit exclusively on the current training fold (including compact selection)."""
    def __init__(self, columns, corr=None, top_k=None):
        self.columns, self.corr, self.top_k = columns, corr, top_k

    def fit(self, data, y):
        x = data[self.columns].replace([np.inf, -np.inf], np.nan)
        self.dropped = {}
        # Per-example mean coverage avoids a large example deciding availability.
        coverage = x.notna().groupby(data.example).mean().mean()
        keep = []
        for c in self.columns:
            if coverage[c] < .70:
                self.dropped[c] = 'training example-balanced coverage < 70%'
            elif x[c].nunique() < 2:
                self.dropped[c] = 'training constant or empty'
            else:
                keep.append(c)
        self.medians = x[keep].median()
        x = x[keep].fillna(self.medians)
        # Deterministic exact-duplicate removal on training values only.
        unique = []
        for c in keep:
            if any(np.array_equal(x[c], x[o]) for o in unique):
                self.dropped[c] = 'training duplicate'
            else:
                unique.append(c)
        x = x[unique]
        w = weights(data)
        if self.corr is not None and len(unique):
            corr = x.corr(method='spearman').abs().fillna(0)
            retained = []
            for c in unique:
                if any(corr.loc[c, o] >= self.corr for o in retained):
                    self.dropped[c] = 'training Spearman redundancy'
                else:
                    retained.append(c)
            x = x[retained]
        if self.top_k is not None and x.shape[1] > self.top_k:
            # Compact alternative only: weighted marginal relevance, not a claim
            # that variables without marginal associations cannot interact.
            xx = x.to_numpy()-np.average(x, axis=0, weights=w)
            yy = np.asarray(y)-np.average(y, weights=w)
            relevance = abs(np.sum(w[:, None]*xx*yy[:, None], axis=0)) / np.sqrt(
                np.maximum(np.sum(w[:, None]*xx**2, axis=0)*np.sum(w*yy**2), 1e-100))
            best = np.argsort(-relevance, kind='stable')[:self.top_k]
            chosen = set(x.columns[best])
            for c in x.columns:
                if c not in chosen:
                    self.dropped[c] = 'training compact relevance screen'
            x = x[[c for c in x if c in chosen]]
        self.kept = list(x.columns)
        self.scaler = StandardScaler().fit(x, sample_weight=w) if self.kept else None
        return self

    def transform(self, data):
        if not self.kept:
            return np.empty((len(data), 0))
        x = data[self.kept].replace([np.inf, -np.inf], np.nan).fillna(self.medians)
        return self.scaler.transform(x)


def estimator(name, params, jobs):
    if name == 'ridge':
        return Ridge(**params)
    if name == 'elastic_net':
        return ElasticNet(**params, max_iter=20000, tol=1e-5, random_state=SEED)
    if name == 'forest':
        return RandomForestRegressor(**params, n_estimators=200, random_state=SEED, n_jobs=jobs)
    if name == 'boosting':
        # No random row-wise early-stopping validation across shared parents.
        return HistGradientBoostingRegressor(**params, max_iter=200, min_samples_leaf=15,
                                             early_stopping=False, random_state=SEED)
    raise ValueError(name)


def fit_predict(train, test, target, spec, params, jobs=1):
    y = train[target].to_numpy(float)
    w = weights(train)
    pre = Preprocess(spec['columns'], spec['corr'], spec['top_k']).fit(train, y)
    mean = float(np.average(y, weights=w))
    sd = float(np.sqrt(np.average((y-mean)**2, weights=w)))
    audit = dict(kept=pre.kept, dropped=pre.dropped, warnings=[])
    if not pre.kept or sd < 1e-14:
        audit['constant_prediction'] = True
        return np.full(len(test), mean), audit
    model = estimator(spec['model'], params, jobs)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always', ConvergenceWarning)
        model.fit(pre.transform(train), (y-mean)/sd, sample_weight=w)
    audit['warnings'] = [str(v.message) for v in caught]
    if any(issubclass(v.category, ConvergenceWarning) for v in caught):
        raise RuntimeError('Unconverged estimator: '+str(audit['warnings']))
    prediction = model.predict(pre.transform(test))*sd+mean
    if not np.isfinite(prediction).all():
        raise ValueError('Nonfinite model predictions')
    if hasattr(model, 'coef_'):
        audit['standardized_coefficients'] = dict(zip(pre.kept, model.coef_.tolist()))
    # No impurity importance is presented as explanatory evidence.
    return prediction, audit


def inner_splits(data):
    splitter = GroupKFold(n_splits=3)
    result = list(splitter.split(data, groups=data.example))
    for train, valid in result:
        assert set(data.iloc[train].example).isdisjoint(data.iloc[valid].example)
    return result


def choose(train, target, spec, jobs):
    trials = []
    for params in spec['grid']:
        prediction = np.full(len(train), np.nan)
        for ti, vi in inner_splits(train):
            prediction[vi], _ = fit_predict(train.iloc[ti], train.iloc[vi], target, spec, params, jobs)
        assert np.isfinite(prediction).all()
        mse = float(np.average((train[target].to_numpy()-prediction)**2, weights=weights(train)))
        trials.append(dict(params=params, mse=mse))
    return min(trials, key=lambda r: r['mse']), trials


def load_data():
    import sys
    sys.path.insert(0, str(ROOT/'RAPID/src'))
    from rapid_tools.hydrograph_response import load
    _, response = load('0.5%')
    d = pd.read_parquet(SOURCE/'linked_responses.parquet')
    for frame in [d, response]:
        frame['example'] = frame.example.astype(str).str.zfill(2)
        assert not frame.duplicated(['example', 'state']).any()
    join = d[['example', 'state']].merge(response[['example', 'state']], how='outer', indicator=True, validate='one_to_one')
    assert join._merge.eq('both').all(), join[join._merge.ne('both')]
    np.testing.assert_allclose(d.set_index(['example', 'state'])[TARGETS].sort_index(),
                               response.set_index(['example', 'state'])[TARGETS].sort_index())
    assert np.isfinite(d[TARGETS]).all().all()
    graphs = pd.read_csv(SOURCE/'domain_graphs.csv', dtype={'example': str})
    uc = graphs[graphs.scope.eq('U')].merge(graphs[graphs.scope.eq('C')], on=['example', 'state'], suffixes=('_U', '_C'), validate='one_to_one')
    distinct = uc[(uc.parent_graph_U != uc.parent_graph_C) | (uc.child_graph_U != uc.child_graph_C)]
    keys = pd.MultiIndex.from_frame(distinct[['example', 'state']])
    d['paired_eligible'] = pd.MultiIndex.from_frame(d[['example', 'state']]).isin(keys)
    d['eligible'] = d.B_available & d.B_K_log_a1.notna() & d.B_K_log_a065.notna()
    return d, join


def cohort(data, name, scenario):
    d = data[data.eligible].copy()
    if name != 'including_centerlines':
        d = d[d.comparison.eq('partial collapse')]
    if name == 'paired_U_C':
        d = d[d.paired_eligible]
    if scenario == 'without54':
        d = d[d.example.ne('54')]
    return d.reset_index(drop=True)


def prepare(out):
    out.mkdir(parents=True, exist_ok=True)
    data, join = load_data()
    cat = catalog()
    assert set(cat.feature) <= set(data.columns)
    cat['finite_fraction_all_rows'] = [np.isfinite(data[c]).mean() for c in cat.feature]
    cat.to_csv(out/'feature_catalog.csv', index=False)
    join.to_csv(out/'join_audit.csv', index=False)
    data[~data.eligible][['example', 'state', 'B_available', 'B_K_valid_a1', 'B_K_valid_a065']].to_csv(out/'exclusions.csv', index=False)
    counts = []
    for c in ['partial', 'including_centerlines', 'paired_U_C']:
        specs = specifications(cat, c == 'paired_U_C')
        for scenario in ['all', 'without54']:
            d = cohort(data, c, scenario)
            # Three inner fits per parameter choice plus one outer refit per spec.
            per_target = d.example.nunique()*sum(3*len(s['grid'])+1 for s in specs)
            counts.append(dict(cohort=c, scenario=scenario, n=len(d), examples=d.example.nunique(),
                               candidate_pipelines=len(specs), fits_per_target=int(per_target)))
    pd.DataFrame(counts).to_csv(out/'run_plan.csv', index=False)
    config = dict(seed=SEED, targets=TARGETS, primary=PRIMARY, inner_folds=3,
                  objective='equal-example mean squared error in original target units',
                  validation='Nested whole-example validation; refit all/without54 separately',
                  preprocessing='Training-only balanced coverage >=.70; median imputation; constant and exact duplicate removal; weighted scaling. Compact panels additionally prune |Spearman|>=.95 and keep 7/14 by weighted marginal correlation.',
                  scopes='D=C for compound members, closed U otherwise, N for centerlines. Paired U/C compares the same edits; open U is conditional geometry only, never isolated resistance.',
                  excluded_metrics={'width_p10/width_p90':'previously questioned; omitted from this first broad pool',
                                    'junction_excess':'equals cycle rank in these single-source/sink DAGs; retain cycle rank AND bifurcation count, which differ at nonbinary junctions',
                                    'branch_length_variance_sum':'equals route length SD squared',
                                    'saved slopes/depth/celerity/n':'not audited independent geometric measurements; hydraulic-model variables excluded'},
                  full=specifications(cat), paired=specifications(cat, True),
                  source_hashes={str(p.relative_to(ROOT)): sha(p) for p in [SOURCE/'linked_responses.parquet', SOURCE/'domain_graphs.csv',
                      SOURCE/'geometry_manifest.json', SOURCE/'resistance_manifest.json', ROOT/'RAPID/outputs/hydrograph_response/manifest.json',
                      ROOT/'RAPID/outputs/hydrograph_response/responses.parquet', ROOT/'RAPID/outputs/hydrograph_response/sensitivity_responses.parquet', Path(__file__)]},
                  versions={'numpy': np.__version__, 'pandas': pd.__version__, 'sklearn': sklearn.__version__})
    # A changed preparation must not silently reuse old checkpoints.
    old = out/'design.json'
    if old.exists() and json.loads(old.read_text()) != config and any((out/'checkpoints').glob('**/*.json')):
        raise ValueError('Design/input changed: use a new --output directory; old checkpoints retained.')
    dump(old, config)
    print(pd.DataFrame(counts).to_string(index=False), flush=True)
    print('Prepared only; no benchmark models fitted.', flush=True)
    return data, cat, config


def run(data, cat, config, out, args):
    fingerprint = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    for c in args.cohorts:
        specs = config['paired' if c == 'paired_U_C' else 'full']
        for scenario in args.scenarios:
            d = cohort(data, c, scenario)
            for target in args.targets:
                for held in sorted(d.example.unique()):
                    train, test = d[d.example.ne(held)], d[d.example.eq(held)]
                    assert set(train.example).isdisjoint(test.example)
                    inner_audit = [dict(train=sorted(train.iloc[t].example.unique()), validation=sorted(train.iloc[v].example.unique())) for t, v in inner_splits(train)]
                    results = []
                    for spec in specs:
                        path = out/'checkpoints'/c/scenario/target/held/(spec['name']+'.json')
                        if path.exists():
                            result = json.loads(path.read_text())
                            assert result['fingerprint'] == fingerprint
                        else:
                            best, trials = choose(train, target, spec, args.jobs)
                            prediction, audit = fit_predict(train, test, target, spec, best['params'], args.jobs)
                            result = dict(fingerprint=fingerprint, cohort=c, scenario=scenario, target=target, held=held,
                                          spec=spec['name'], inner_mse=best['mse'], params=best['params'], trials=trials,
                                          inner_groups=inner_audit, train_examples=sorted(train.example.unique()),
                                          features=audit, predictions=[dict(example=r.example, state=r.state, observed=float(getattr(r, target)), predicted=float(p)) for r, p in zip(test.itertuples(), prediction)])
                            dump(path, result)
                        results.append(result)
                    winner = min(results, key=lambda r: r['inner_mse'])
                    print(c, scenario, target, 'held', held, 'inner-selected:', winner['spec'], flush=True)
    summarize(out, config)


def summarize(out, config):
    """Only complete outer folds enter reports; never silently score partial folds."""
    rows, choices, feature_rows = [], [], []
    groups = {}
    for path in sorted((out/'checkpoints').glob('*/*/*/*/*.json')):
        r = json.loads(path.read_text())
        key = (r['cohort'], r['scenario'], r['target'])
        groups.setdefault(key, {}).setdefault(r['held'], []).append(r)
    for (c, scenario, target), folds in groups.items():
        expected_examples = 14 if scenario == 'without54' else 15
        spec_names = {s['name'] for s in config['paired' if c == 'paired_U_C' else 'full']}
        if len(folds) != expected_examples or any({r['spec'] for r in results} != spec_names for results in folds.values()):
            print('Incomplete; not scored:', c, scenario, target, flush=True)
            continue
        for held, results in folds.items():
            winner = min(results, key=lambda r: r['inner_mse'])
            compact = min((r for r in results if r['spec'].startswith('compact')), key=lambda r: r['inner_mse'])
            choices.append(dict(cohort=c, scenario=scenario, response=target, held_example=held,
                                selected=winner['spec'], compact_selected=compact['spec']))
            for r, label in [(r, r['spec']) for r in results] + [(winner, 'nested_selected'), (compact, 'nested_compact')]:
                for p in r['predictions']:
                    rows.append(dict(cohort=c, scenario=scenario, response=target, pipeline=label, **p))
                for f in r['features']['kept']:
                    coef = r['features'].get('standardized_coefficients', {}).get(f)
                    feature_rows.append(dict(cohort=c, scenario=scenario, response=target, pipeline=label,
                                             held_example=held, feature=f, coefficient=coef,
                                             active=coef is None or abs(coef)>1e-10))
        # Baselines use the same complete cohort and no test labels for fitting.
        reference = pd.DataFrame([p for rr in folds.values() for p in rr[0]['predictions']])
        for held, test in reference.groupby('example'):
            tr = reference[reference.example.ne(held)]
            mean = float(np.average(tr.observed, weights=weights(tr)))
            for label, value in [('zero', 0.), ('training_mean', mean)]:
                for r in test.itertuples():
                    rows.append(dict(cohort=c, scenario=scenario, response=target, pipeline=label,
                                     example=r.example, state=r.state, observed=r.observed, predicted=value))
    if not rows:
        return
    pred = pd.DataFrame(rows)
    pred.to_parquet(out/'predictions.parquet', index=False)
    pd.DataFrame(choices).to_csv(out/'selected_pipelines.csv', index=False)
    pd.DataFrame(feature_rows).to_csv(out/'retained_features.csv', index=False)
    scores = []
    keys = ['cohort', 'scenario', 'response', 'pipeline']
    for key, g in pred.groupby(keys):
        for label, w in [('equal_example', weights(g)), ('pair', np.ones(len(g)))]:
            scores.append(dict(zip(keys, key), weighting=label, n=len(g), **score(g.observed, g.predicted, w)))
        for ex, gg in g.groupby('example'):
            scores.append(dict(zip(keys, key), weighting='example_'+ex, n=len(gg), **score(gg.observed, gg.predicted, np.ones(len(gg)))))
    pd.DataFrame(scores).to_csv(out/'scores.csv', index=False)
    sensitivity = pd.read_parquet(ROOT/'RAPID/outputs/hydrograph_response/sensitivity_responses.parquet')
    alt_scores = []
    for window in ['1%', '2%']:
        for key, g in pred.groupby(keys):
            target = key[2]
            alt = sensitivity[sensitivity.window.eq(window)][['example', 'state', target]].copy()
            alt['example'] = alt.example.astype(str).str.zfill(2)
            joined = g.merge(alt, on=['example', 'state'], validate='one_to_one')
            assert len(joined) == len(g)
            alt_scores.append(dict(zip(keys, key), window=window, **score(joined[target], joined.predicted, weights(joined))))
    pd.DataFrame(alt_scores).to_csv(out/'sensitivity_scores.csv', index=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', action='store_true', help='Explicitly start extensive nested model fitting.')
    parser.add_argument('--output', type=Path, default=OUT)
    parser.add_argument('--cohorts', nargs='+', choices=['partial', 'paired_U_C', 'including_centerlines'], default=['partial'])
    parser.add_argument('--scenarios', nargs='+', choices=['all', 'without54'], default=['all', 'without54'])
    parser.add_argument('--targets', nargs='+', choices=TARGETS, default=PRIMARY)
    parser.add_argument('--jobs', type=int, default=1, help='Forest workers; outer loops stay sequential. Set BLAS/OMP thread limits in shell.')
    args = parser.parse_args()
    data, cat, config = prepare(args.output)
    if args.run:
        run(data, cat, config, args.output, args)


if __name__ == '__main__':
    main()
