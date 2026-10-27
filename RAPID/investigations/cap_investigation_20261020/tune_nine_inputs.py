"""Nested grouped tuning of the nine-input forest; isolated research artifact."""
import argparse
import hashlib
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from slope_length_check import b, OUT, COLUMNS, save, sha
from augmented_tests import objective

SOURCE = OUT/'augmented_tests'
DEST = OUT/'nine_input_tuning'


def fit(train, params, jobs):
    b.COLUMNS = list(COLUMNS)
    return b.fit(train, params, jobs)


def tune(data, candidates, label, fp, jobs):
    folds = list(GroupKFold(3).split(data, groups=data.example))
    ranked = []
    for i, candidate in enumerate(candidates):
        path = DEST/'inner'/label/f'{i:02d}.json'
        trial = b.forest.checked(path, fp)
        if trial is None:
            pred = np.full(len(data), np.nan)
            groups = []
            for ti, vi in folds:
                tr, va = data.iloc[ti], data.iloc[vi]
                assert set(tr.example).isdisjoint(va.example)
                model = fit(tr, candidate['params'], jobs)
                pred[vi] = b.forest.predict(model, va)
                groups.append(dict(train=sorted(tr.example.unique()), validation=sorted(va.example.unique())))
            assert np.isfinite(pred).all()
            score, by = objective(data, pred)
            trial = dict(fingerprint=fp, index=i, name=candidate['name'], params=candidate['params'],
                         score=score, cohort_scores=by, inner_groups=groups)
            save(path, trial)
        ranked.append(trial)
        print(label, candidate['name'], 'inner score', round(trial['score'], 3), flush=True)
    return sorted(ranked, key=lambda r:(r['score'], r['index']))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', action='store_true')
    parser.add_argument('--jobs', type=int, default=4)
    args = parser.parse_args()
    DEST.mkdir(exist_ok=True)
    d = pd.read_parquet(SOURCE/'input.parquet')
    prior = json.loads((SOURCE/'design.json').read_text())
    candidates = list(prior['candidates'])
    assert len(candidates)==4
    params = candidates[0]['params']
    candidates.extend([
        dict(name='absolute_feature_sampling', params=dict(params, min_samples_split=2,
             min_samples_leaf=2, max_depth=None, max_features=.75, ccp_alpha=0.)),
        dict(name='moderate_squared', params=dict(params, criterion='squared_error',
             min_samples_split=10, min_samples_leaf=5, max_depth=10, max_features=.75, ccp_alpha=0.)),
    ])
    assert len(d)==2185 and d.example.nunique()==15 and np.isfinite(d[COLUMNS].to_numpy()).all()
    files = [SOURCE/'input.parquet', SOURCE/'design.json', OUT/'slope_length_check/predictions.parquet',
             Path(__file__), Path(b.__file__), Path(b.forest.__file__), Path(b.engine.__file__),
             Path(__file__).with_name('slope_length_check.py'), Path(__file__).with_name('augmented_tests.py')]
    design = dict(columns=COLUMNS, candidates=candidates, seed=b.engine.SEED,
        target='centroid_shift_hours', primary_window='0.5%', training_rows=len(d),
        sources={str(p.resolve()):sha(p) for p in files},
        validation='15 outer whole-network holdouts; 3 grouped inner folds tuning all nine inputs.',
        objective='Mean of original and segmented cohort floored relative errors; one-minute floor; equal network weight within each cohort.',
        training='Original+segmented, equal total weight per network. Training-only preprocessing and target standardization. No pairs.',
        controls='Report winner among original four configurations as a same-grid control, and among all six as expanded tuning. Compare with nine inputs using seven-input-selected settings.',
        final='Select six-candidate winner by 3-fold grouped validation on all data; save separate research artifact. Selection score is not an independent test score.',
        caveats=['Exploratory reused networks, not new external validation.',
                 'Only 15 independent segmented cases.', 'Peak scores evaluate centroid-trained predictions.',
                 'Subset exclusions are scoring-only diagnostics.'],
        production_modified=False)
    fp = hashlib.sha256(json.dumps(design, sort_keys=True).encode()).hexdigest()
    if (DEST/'design.json').exists():
        assert json.loads((DEST/'design.json').read_text())==design, 'Use a new output version for changed design'
    save(DEST/'design.json', design)
    print('Prepared six configurations: 270 inner fits, up to 30 outer fits, 18 final-selection fits + one research model.', flush=True)
    if not args.run:
        return
    rows, selections = [], []
    for ex in sorted(d.example.unique()):
        tr, te = d[d.example.ne(ex)].reset_index(drop=True), d[d.example.eq(ex)]
        assert tr.example.nunique()==14 and ex not in set(tr.example)
        ranked = tune(tr, candidates, ex, fp, args.jobs)
        choices = {'same_grid':next(c for c in ranked if c['index']<4), 'expanded_grid':ranked[0]}
        for label, choice in choices.items():
            path = DEST/'outer'/ex/f'{choice["index"]:02d}.json'
            saved = b.forest.checked(path, fp)
            if saved is None:
                model = fit(tr, choice['params'], args.jobs)
                pred = b.forest.predict(model, te)
                saved = dict(fingerprint=fp, rows=[
                    dict(example=r.example, state=r.state, cohort=r.cohort,
                         centroid_shift_hours=r.centroid_shift_hours, peak_shift_hours=r.peak_shift_hours,
                         predicted_hours=float(v)) for r, v in zip(te.itertuples(),pred)])
                save(path,saved)
            rows.extend(dict(variant=label,**r) for r in saved['rows'])
            selections.append(dict(held=ex,variant=label,**choice))
        save(DEST/'selections.json',selections)
        print('Held-out',ex,'complete:',{k:v['name'] for k,v in choices.items()},flush=True)
    p = pd.DataFrame(rows)
    assert len(p)==4370 and not p.duplicated(['variant','example','state']).any()
    old = pd.read_parquet(OUT/'slope_length_check/predictions.parquet')
    old = old[old.panel.eq('slope_plus_length')].drop(columns='panel')
    old['variant']='reused_settings'
    keys=['example','state','cohort','centroid_shift_hours','peak_shift_hours']
    pd.testing.assert_frame_equal(old[keys].sort_values(keys[:2]).reset_index(drop=True),
                                  d[keys].sort_values(keys[:2]).reset_index(drop=True),check_dtype=False)
    p = pd.concat([p,old],ignore_index=True)
    p.to_parquet(DEST/'predictions.parquet',index=False)
    scores=[]
    for (variant,cohort),g in p.groupby(['variant','cohort']):
        for target in ['centroid_shift_hours','peak_shift_hours']:
            for subset,z in [('all',g),('excluding22',g[g.example.ne('22')]),('excluding54',g[g.example.ne('54')])]:
                y,pr,w=z[target].to_numpy(),z.predicted_hours.to_numpy(),b.engine.weights(z)
                scores.append(dict(variant=variant,cohort=cohort,target=target,subset=subset,n=len(z),
                    relative_error_pct=b.forest.sweep.relative_score(y,pr,w,1),**b.engine.score(y*60,pr*60,w)))
    s = pd.DataFrame(scores)
    s.to_csv(DEST/'scores.csv',index=False)
    print(s.query("target=='centroid_shift_hours' and subset=='all'").to_string(index=False),flush=True)
    best = tune(d.reset_index(drop=True),candidates,'all_data',fp,args.jobs)[0]
    model = fit(d,best['params'],args.jobs)
    model['fingerprint']=fp
    path=DEST/'research_model.joblib'
    joblib.dump(model,path,compress=3)
    restored=joblib.load(path)
    np.testing.assert_allclose(b.forest.predict(model,d),b.forest.predict(restored,d),rtol=1e-12,atol=1e-12)
    save(DEST/'research_model.json',dict(selected=best,columns=COLUMNS,sha256=sha(path),
        training_rows=len(d),warning='Research artifact, no production replacement; use nested scores for performance.'))
    for source,digest in design['sources'].items():
        assert sha(source)==digest
    save(DEST/'completion.json',dict(complete=True,fingerprint=fp,production_modified=False))
    print('Completed. Separate nine-input research model saved.',flush=True)


if __name__=='__main__':
    main()
