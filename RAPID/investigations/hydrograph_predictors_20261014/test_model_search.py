"""Small synthetic checks; these do not execute the scientific model search."""
import numpy as np
import pandas as pd
import pytest
from model_search import (Preprocess, inner_splits, weights, fit_predict, catalog,
                          specifications, cohort, score, choose)


def sample():
    rng = np.random.default_rng(17)
    x = rng.normal(size=80)
    return pd.DataFrame({'example': np.repeat(['01', '02', '03', '04', '05'], 16),
                         'x': x, 'duplicate': x, 'z': rng.normal(size=80),
                         'constant': 1., 'missing': np.nan, 'target': 2*x+rng.normal(scale=.1, size=80)})


def test_preprocessing_is_training_only_and_deduplicates():
    d = sample(); train, test = d.iloc[:64], d.iloc[64:].copy()
    pre = Preprocess(['x', 'duplicate', 'z', 'constant', 'missing']).fit(train, train.target)
    assert pre.kept == ['x', 'z']
    medians, means = pre.medians.copy(), pre.scaler.mean_.copy()
    test['x'] = 1e9
    assert pre.transform(test)[:, 0].min() > 1e8
    pd.testing.assert_series_equal(pre.medians, medians)
    np.testing.assert_array_equal(pre.scaler.mean_, means)


def test_balanced_weights_and_group_disjointness():
    d = sample().drop(index=range(10))
    totals = pd.Series(weights(d), index=d.index).groupby(d.example).sum()
    np.testing.assert_allclose(totals, totals.iloc[0])
    seen = []
    for tr, va in inner_splits(d):
        assert set(d.iloc[tr].example).isdisjoint(d.iloc[va].example)
        seen.extend(va)
    assert sorted(seen) == list(range(len(d)))


@pytest.mark.parametrize('model,params', [
    ('ridge', {'alpha': 1.}), ('elastic_net', {'alpha': .01, 'l1_ratio': .5}),
    ('forest', {'max_depth': 3, 'min_samples_leaf': 3, 'max_features': 1.}),
    ('boosting', {'max_leaf_nodes': 7, 'l2_regularization': 1., 'learning_rate': .06}),
])
def test_each_model_accepts_weights_and_separate_target(model, params):
    d = sample(); tr, te = d.iloc[:64], d.iloc[64:]
    spec = dict(columns=['x', 'z'], corr=None, top_k=None, model=model)
    p, audit = fit_predict(tr, te, 'target', spec, params)
    assert len(p) == len(te) and np.isfinite(p).all()
    assert score(te.target, p, weights(te))['mae'] < 1.6
    assert not audit['warnings']


def test_compact_selection_and_inner_tuning():
    d = sample()
    pre = Preprocess(['x', 'duplicate', 'z'], corr=.95, top_k=1).fit(d, d.target)
    assert pre.kept == ['x']
    spec = dict(columns=['x', 'z'], corr=None, top_k=1, model='ridge', grid=[{'alpha': .1}, {'alpha': 10000.}])
    best, trials = choose(d, 'target', spec, 1)
    assert best['params']['alpha'] == .1 and len(trials) == 2


def test_allowlist_and_cohort_rules():
    cat = catalog()
    assert not cat.feature.duplicated().any()
    assert not cat.feature.str.contains('peak|centroid|celerity|depth|crest|route_time').any()
    for paired in [False, True]:
        specs = specifications(cat, paired)
        assert len({s['name'] for s in specs}) == len(specs)
        assert all(set(s['columns']) <= set(cat.feature) for s in specs)
    d = pd.DataFrame({'example': ['01', '54', '02', '03'], 'eligible': [True, True, True, False],
                      'paired_eligible': [True, True, False, True],
                      'comparison': ['partial collapse', 'partial collapse', 'single centerline', 'partial collapse']})
    assert cohort(d, 'partial', 'without54').example.tolist() == ['01']
    assert cohort(d, 'paired_U_C', 'all').example.tolist() == ['01', '54']
    assert len(cohort(d, 'including_centerlines', 'all')) == 3


def test_checkpoint_resume_selection_and_reporting(tmp_path, monkeypatch):
    """Exercise orchestration/reporting with stub fits, not a scientific run."""
    import model_search as m
    from types import SimpleNamespace
    examples = [f'{i:02}' for i in range(1, 15)]+['54']
    d = pd.DataFrame([dict(example=ex, state=f's{i}', peak_shift_hours=float(i),
                           eligible=True, paired_eligible=True, comparison='partial collapse')
                      for ex in examples for i in [0, 1]])
    specs = [dict(name='compact7__ridge'), dict(name='broad__ridge')]
    calls = []
    def stub_choose(train, target, spec, jobs):
        calls.append(spec['name'])
        best = dict(params={}, mse=.1 if spec['name'].startswith('compact') else .2)
        return best, [best]
    def stub_fit(train, test, target, spec, params, jobs):
        assert set(train.example).isdisjoint(test.example)
        return np.full(len(test), train[target].mean()), dict(kept=['B_parent_total_length'], dropped={}, warnings=[])
    monkeypatch.setattr(m, 'choose', stub_choose)
    monkeypatch.setattr(m, 'fit_predict', stub_fit)
    sensitivity = pd.concat([d[['example','state','peak_shift_hours']].assign(window=w) for w in ['1%', '2%']])
    monkeypatch.setattr(m.pd, 'read_parquet', lambda path: sensitivity)
    config = dict(full=specs, paired=specs)
    args = SimpleNamespace(cohorts=['partial'], scenarios=['all'], targets=['peak_shift_hours'], jobs=1)
    m.run(d, catalog(), config, tmp_path, args)
    assert len(calls) == 30
    m.run(d, catalog(), config, tmp_path, args)
    assert len(calls) == 30  # Completed fits were not repeated.
    scores = pd.read_csv(tmp_path/'scores.csv')
    assert set(scores.pipeline) == {'compact7__ridge','broad__ridge','nested_selected','nested_compact','zero','training_mean'}
    assert scores[scores.weighting.eq('equal_example')].n.eq(30).all()
    chosen = pd.read_csv(tmp_path/'selected_pipelines.csv')
    assert chosen.selected.eq('compact7__ridge').all()
    assert len(pd.read_csv(tmp_path/'sensitivity_scores.csv')) == 12
