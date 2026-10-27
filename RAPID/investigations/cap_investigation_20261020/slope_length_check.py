"""Add both slope and length delta using the existing training-only fold settings."""
import json
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd
from augmented_tests import b, OUT, BASE, save, sha

SOURCE = OUT/'augmented_tests'
DEST = OUT/'slope_length_check'
COLUMNS = BASE+['parent_slope_slow', 'length_delta_km']


def main():
    DEST.mkdir(exist_ok=True)
    assert json.loads((SOURCE/'completion.json').read_text())['complete']
    d = pd.read_parquet(SOURCE/'input.parquet')
    choices = json.loads((SOURCE/'selections.json').read_text())
    assert len(choices)==15 and len(d)==2185 and not d.duplicated(['example','state']).any()
    assert np.isfinite(d[COLUMNS].to_numpy()).all()
    files = [SOURCE/'input.parquet', SOURCE/'selections.json', SOURCE/'predictions.parquet',
             SOURCE/'design.json', Path(__file__), Path(b.__file__), Path(b.engine.__file__),
             Path(b.forest.__file__)]
    design = dict(columns=COLUMNS, target='centroid_shift_hours', networks=15,
                  sources={str(p.resolve()):sha(p) for p in files},
                  validation='Reuse baseline training-only grouped-inner selections for each outer held-network fold. No new tuning or feature selection.',
                  definition='Original seven inputs with edited size, plus parent slope context and same-domain mean-length delta / 1 km. Four length-related inputs total.',
                  caveat='Exploratory follow-up on the same 15 networks; combined feature panel not separately retuned. No production changes.')
    fp = hashlib.sha256(json.dumps(design,sort_keys=True).encode()).hexdigest()
    if (DEST/'design.json').exists():
        assert json.loads((DEST/'design.json').read_text())==design
    save(DEST/'design.json',design)
    rows=[]
    b.COLUMNS=COLUMNS
    for ex in sorted(d.example.unique()):
        selected=[c for c in choices if c['held']==ex]
        assert len(selected)==1
        train,test=d[d.example.ne(ex)],d[d.example.eq(ex)]
        assert train.example.nunique()==14 and set(train.example).isdisjoint(test.example)
        path=DEST/'folds'/f'{ex}.json'
        result=b.forest.checked(path,fp)
        if result is None:
            artifact=b.fit(train,selected[0]['params'],jobs=4)
            pred=b.forest.predict(artifact,test)
            result=dict(fingerprint=fp,selected=selected[0]['name'],rows=[
                dict(example=r.example,state=r.state,cohort=r.cohort,
                     centroid_shift_hours=r.centroid_shift_hours,peak_shift_hours=r.peak_shift_hours,
                     predicted_hours=float(v)) for r,v in zip(test.itertuples(),pred)])
            save(path,result)
        rows.extend(dict(panel='slope_plus_length',**r) for r in result['rows'])
        print(ex,'complete',flush=True)
    b.COLUMNS=BASE
    combined=pd.DataFrame(rows)
    assert len(combined)==2185 and np.isfinite(combined.predicted_hours).all()
    previous=pd.read_parquet(SOURCE/'predictions.parquet')
    previous=previous[previous.panel.isin(['frozen_reference','baseline','parent_slope','length_delta'])]
    p=pd.concat([previous,combined],ignore_index=True)
    assert not p.duplicated(['panel','example','state']).any()
    p.to_parquet(DEST/'predictions.parquet',index=False)
    scores=[]
    for (panel,cohort),g in p.groupby(['panel','cohort']):
        for target in ['centroid_shift_hours','peak_shift_hours']:
            for subset,z in [('all',g),('excluding22',g[g.example.ne('22')]),('excluding54',g[g.example.ne('54')])]:
                y,pr,w=z[target].to_numpy(),z.predicted_hours.to_numpy(),b.engine.weights(z)
                scores.append(dict(panel=panel,cohort=cohort,target=target,subset=subset,n=len(z),
                    relative_error_pct=b.forest.sweep.relative_score(y,pr,w,1),**b.engine.score(y*60,pr*60,w)))
    s=pd.DataFrame(scores)
    s.to_csv(DEST/'scores.csv',index=False)
    p['absolute_error_minutes']=(p.predicted_hours-p.centroid_shift_hours).abs()*60
    p.groupby(['panel','cohort','example']).absolute_error_minutes.mean().rename('mae_minutes').to_csv(DEST/'network_errors.csv')
    p[p.cohort.eq('segmented')].to_csv(DEST/'segmented_predictions.csv',index=False)
    for path,digest in design['sources'].items():
        assert sha(path)==digest
    save(DEST/'completion.json',dict(complete=True,fingerprint=fp,fits=15,production_modified=False))
    print(s.query("target=='centroid_shift_hours' and subset=='all'").to_string(index=False))


if __name__=='__main__':
    main()
