"""Whole-network side predictions from forests trained on other networks' partial edits."""
from pathlib import Path
import json,hashlib
import numpy as np
import pandas as pd
import drop_exposure_check as baseline
forest=baseline.forest;engine=baseline.engine
OUT=forest.ROOT/'RAPID/outputs/hydrograph_whole_network_check_20261019'
SCOPES=forest.ROOT/'RAPID/outputs/hydrograph_predictor_scopes_20261014/geometry_responses.parquet'

def main():
    d=pd.read_parquet(baseline.SOURCE/'input.parquet');s=pd.read_parquet(SCOPES)
    full=s[s.state.eq('fully_collapsed')].copy()
    assert len(full)==15 and full.example.nunique()==15 and full.B_available.all() and full.B_scope.eq('N').all()
    full['domain_size']=full.B_parent_route_length_mean/1000
    raw=[c.removeprefix('log1p__') for c in baseline.COLUMNS if c.startswith('log1p__')]
    for c in raw:full['log1p__'+c]=np.log1p(full[c])
    assert np.isfinite(full[baseline.COLUMNS].to_numpy(float)).all()
    # Verify saved scope definitions agree with the frozen inputs for all existing comparisons.
    check=d.merge(s[['example','state',*raw,'B_parent_dominance_mean','B_parent_route_length_mean']],on=['example','state'],validate='one_to_one',suffixes=('_frozen','_scope'))
    assert len(check)==len(d)
    for c in raw+['B_parent_dominance_mean','B_parent_route_length_mean']:
        np.testing.assert_allclose(check[c+'_frozen'],check[c+'_scope'],rtol=1e-12,atol=1e-12)
    responses=pd.read_parquet(forest.ROOT/'RAPID/outputs/hydrograph_response/responses.parquet')
    response_check=full.merge(responses[['example','state','centroid_shift_hours','peak_shift_hours']],on=['example','state'],validate='one_to_one',suffixes=('_scope','_current'))
    assert len(response_check)==15
    for c in ['centroid_shift_hours','peak_shift_hours']:np.testing.assert_allclose(response_check[c+'_scope'],response_check[c+'_current'])
    info=json.loads((baseline.OUT/'final_model.json').read_text());params=info['params'];OUT.mkdir(parents=True,exist_ok=True)
    sources=[Path(__file__),Path(baseline.__file__),Path(forest.__file__),Path(engine.__file__),SCOPES,baseline.SOURCE/'input.parquet',baseline.OUT/'final_model.json',baseline.OUT/'predictions.parquet',forest.ROOT/'RAPID/outputs/hydrograph_response/responses.parquet']
    design=dict(method='Side evaluation only: hold whole example out, fit frozen seven-input settings on partial edits of other examples, predict held whole-network collapse. No full-network states enter training.',columns=baseline.COLUMNS,params=params,sources={str(p):engine.sha(p) for p in sources})
    fp=hashlib.sha256(json.dumps(design,sort_keys=True).encode()).hexdigest();engine.dump(OUT/'design.json',design)
    reference=pd.read_parquet(baseline.OUT/'predictions.parquet');reference=reference[reference.variant.eq('seven_inputs')]
    records=[];support=[]
    for _,row in full.iterrows():
        ex=row.example;tr=d[d.example.ne(ex)];te=d[d.example.eq(ex)];path=OUT/'checkpoints'/f'{ex}.json';saved=forest.checked(path,fp)
        for c in baseline.COLUMNS:
            v=float(row[c]);lo=float(tr[c].min());hi=float(tr[c].max());support.append(dict(example=ex,feature=c,value=v,training_min=lo,training_max=hi,outside=v<lo-1e-12 or v>hi+1e-12))
        if saved is None:
            model=baseline.fit(tr,params)
            old=te[['example','state']].merge(reference[['example','state','predicted_hours']],on=['example','state'],validate='one_to_one')
            np.testing.assert_allclose(forest.predict(model,te),old.predicted_hours,rtol=1e-10,atol=1e-12)
            pred=float(forest.predict(model,row.to_frame().T)[0]);saved=dict(fingerprint=fp,example=ex,state=row.state,predicted_hours=pred,centroid_shift_hours=float(row.centroid_shift_hours),peak_shift_hours=float(row.peak_shift_hours));engine.dump(path,saved)
            print(ex,'saved',flush=True)
        records.append(saved)
    p=pd.DataFrame(records);p['actual_minutes']=p.centroid_shift_hours*60;p['predicted_minutes']=p.predicted_hours*60;p['absolute_error_minutes']=abs(p.actual_minutes-p.predicted_minutes)
    p.to_csv(OUT/'predictions.csv',index=False);pd.DataFrame(support).to_csv(OUT/'predictor_support.csv',index=False);full.to_parquet(OUT/'input.parquet',index=False)
    scores=[]
    for response in ['centroid_shift_hours','peak_shift_hours']:
        y=p[response].to_numpy();pred=p.predicted_hours.to_numpy();w=engine.weights(p)
        scores.append(dict(response=response,relative_error_pct=forest.sweep.relative_score(y,pred,w,1),**engine.score(y*60,pred*60,w)))
    pd.DataFrame(scores).to_csv(OUT/'scores.csv',index=False)
    print(pd.DataFrame(scores).to_string(index=False))
if __name__=='__main__':main()
