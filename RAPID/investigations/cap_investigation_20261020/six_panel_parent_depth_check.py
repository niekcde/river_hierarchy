"""Paired fixed-settings check: add absolute parent depth to all six panels."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import six_panel_geometry_depth_resistance as prior

DEST = prior.OUT / 'six_panel_parent_depth_20261023'
PANELS = {name: list(cols) + ['parent_depth_nd'] for name, cols in prior.PANELS.items()}


def summarize(predictions):
    scores = []
    networks = []
    for (depth, panel, cohort), g in predictions.groupby(['depth_added', 'panel', 'cohort']):
        for subset, z in [('all', g), ('without54', g[g.example.ne('54')])]:
            w = 1 / z.groupby('example').example.transform('size').to_numpy()
            y = z.centroid_shift_hours.to_numpy()
            err = z.predicted_hours.to_numpy() - y
            scores.append(dict(depth_added=depth, panel=panel, cohort=cohort, subset=subset,
                n=len(z), r2=1-np.sum(w*err**2)/np.sum(w*(y-np.average(y, weights=w))**2),
                mae_minutes=60*np.average(abs(err), weights=w),
                relative_error_pct=100*np.average(abs(err)/np.maximum(abs(y), 1/60), weights=w)))
        for ex, z in g.groupby('example'):
            err = z.predicted_hours-z.centroid_shift_hours
            networks.append(dict(depth_added=depth, panel=panel, cohort=cohort, example=ex,
                relative_error_pct=100*np.mean(abs(err)/np.maximum(abs(z.centroid_shift_hours), 1/60))))
    s = pd.DataFrame(scores)
    s.to_csv(DEST/'scores.csv', index=False)
    pd.DataFrame(networks).to_csv(DEST/'network_scores.csv', index=False)
    objective = s[s.subset.eq('all')].groupby(['depth_added', 'panel']).relative_error_pct.mean()
    objective.rename('equal_cohort_objective_pct').to_csv(DEST/'objective.csv')
    paired = s[s.depth_added].merge(s[~s.depth_added], on=['panel','cohort','subset'], suffixes=('_depth','_prior'), validate='one_to_one')
    for metric in ['r2','mae_minutes','relative_error_pct']:
        paired[metric+'_difference'] = paired[metric+'_depth']-paired[metric+'_prior']
    paired.to_csv(DEST/'paired_scores.csv', index=False)
    print(s[s.subset.eq('all') & s.depth_added].to_string(index=False), flush=True)
    print(objective.to_string(), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', action='store_true')
    parser.add_argument('--jobs', type=int, default=4)
    args = parser.parse_args()
    DEST.mkdir(exist_ok=True)
    source = prior.DEST
    paths = [source/x for x in ['input.parquet','predictions.parquet','design.json','completion.json']]
    paths += [Path(__file__), Path(prior.__file__)]
    paths += [Path(m.__file__) for m in [prior.b, prior.b.engine, prior.b.forest]]
    hashes = {str(p): prior.digest(p) for p in paths}
    old_design = json.loads((source/'design.json').read_text())
    assert json.loads((source/'completion.json').read_text())['complete']
    choices = old_design['choices']
    d = pd.read_parquet(source/'input.parquet')
    assert len(d)==2185 and d.example.nunique()==15
    assert not d.duplicated(['example','state']).any()
    assert (d.parent_depth_nd>0).all()
    assert np.isfinite(d[list(dict.fromkeys(sum(PANELS.values(), [])))]).all().all()
    assert len(choices)==15 and set(c['held'] for c in choices)==set(d.example)
    design = dict(sources=hashes, panels=PANELS, choices=choices, fits=90,
        seed=prior.b.engine.SEED, validation=old_design['validation'],
        depth='S_h=sum(r_e L_e h_e)/sum(r_e L_e)/(1m); B domain for regular, edited footprint excluding connectors for segmented. Derived h=a_example W^.65, not independently observed depth. No depth ratio added.',
        production_modified=False)
    fp = hashlib.sha256(json.dumps(design, sort_keys=True).encode()).hexdigest()
    (DEST/'design.json').write_text(json.dumps(design, indent=2)+'\n')
    if not args.run:
        print('Prepared 90 fits; use --run to execute.')
        return
    rows=[]
    for panel, cols in PANELS.items():
        prior.b.COLUMNS = list(cols)
        for choice in choices:
            ex = choice['held']
            path = DEST/'folds'/panel/f'{ex}.json'
            path.parent.mkdir(parents=True, exist_ok=True)
            if path.exists():
                result=json.loads(path.read_text())
                assert result['fingerprint']==fp
            else:
                train=d[d.example.ne(ex)]
                test=d[d.example.eq(ex)]
                assert train.example.nunique()==14
                artifact=prior.b.fit(train, choice['params'], jobs=args.jobs)
                pred=prior.b.forest.predict(artifact,test)
                assert np.isfinite(pred).all()
                result=dict(fingerprint=fp,rows=[dict(example=r.example,state=r.state,
                    cohort=r.cohort,centroid_shift_hours=r.centroid_shift_hours,
                    predicted_hours=float(v),panel=panel,depth_added=True)
                    for r,v in zip(test.itertuples(),pred)])
                tmp=path.with_suffix('.tmp')
                tmp.write_text(json.dumps(result,indent=2)+'\n')
                tmp.replace(path)
                print(panel, ex, 'saved', flush=True)
            rows.extend(result['rows'])
    prior.b.COLUMNS=list(prior.COLUMNS)
    new=pd.DataFrame(rows)
    assert len(new)==6*2185 and not new.duplicated(['panel','example','state']).any()
    old=pd.read_parquet(source/'predictions.parquet').assign(depth_added=False)
    check=new.merge(old,on=['panel','example','state'],validate='one_to_one',suffixes=('_new','_old'))
    assert len(check)==len(new)==len(old)
    np.testing.assert_array_equal(check.centroid_shift_hours_new,check.centroid_shift_hours_old)
    assert (check.cohort_new==check.cohort_old).all()
    predictions=pd.concat([old,new],ignore_index=True)
    predictions.to_parquet(DEST/'predictions.parquet',index=False)
    summarize(predictions)
    for path,h in hashes.items():
        assert prior.digest(path)==h,path
    (DEST/'completion.json').write_text(json.dumps(dict(complete=True,fingerprint=fp,fits=90,
        matched_per_panel=2185,unmatched=0,production_modified=False),indent=2)+'\n')


if __name__=='__main__':
    main()
