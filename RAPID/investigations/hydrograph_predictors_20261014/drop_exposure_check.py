"""Fixed-setting removal check for constant outer-domain exposure; separate artifact."""
from pathlib import Path
import json,os,hashlib
import numpy as np
import pandas as pd
import joblib
import final_forest_search as forest
from sklearn.ensemble import RandomForestRegressor

engine=forest.engine
SOURCE=forest.SOURCE
REFERENCE=forest.ROOT/'RAPID/outputs/hydrograph_closed_domain_test_20261019'
OUT=forest.ROOT/'RAPID/outputs/hydrograph_no_exposure_20261019'
COLUMNS=[c for c in forest.sweep.COLUMNS if c!='B_exposure']


def fit(train,params,jobs=4):
    y=train[forest.TARGET].to_numpy(float);w=engine.weights(train)
    pre=engine.Preprocess(COLUMNS,None,None).fit(train,y)
    mean=float(np.average(y,weights=w));sd=float(np.sqrt(np.average((y-mean)**2,weights=w)))
    if not pre.kept or sd<1e-14:raise ValueError('Degenerate training fold')
    model=RandomForestRegressor(**params,random_state=engine.SEED,n_jobs=jobs)
    model.fit(pre.transform(train),(y-mean)/sd,sample_weight=w)
    return dict(preprocessor=pre,estimator=model,target_mean=mean,target_sd=sd,columns=COLUMNS,
      params=params,target=forest.TARGET,prediction_units='hours')


def main():
    d=pd.read_parquet(SOURCE/'input.parquet')
    info=json.loads((forest.OUT/'final_model.json').read_text());params=info['params']
    ref_design=json.loads((REFERENCE/'design.json').read_text())
    assert ref_design['params']==params
    assert ref_design['sources'][str(SOURCE/'input.parquet')]==engine.sha(SOURCE/'input.parquet')
    assert np.max(abs(d.B_exposure-1))<1e-12
    assert not d.duplicated(['example','state']).any()
    reference=pd.read_parquet(REFERENCE/'predictions.parquet')
    reference=reference[reference.variant=='current'].copy()
    assert set(zip(reference.example,reference.state))==set(zip(d.example,d.state))
    OUT.mkdir(parents=True,exist_ok=True)
    files=[Path(__file__),Path(forest.__file__),Path(engine.__file__),SOURCE/'input.parquet',forest.OUT/'final_model.json',REFERENCE/'design.json',REFERENCE/'predictions.parquet']
    design=dict(columns=COLUMNS,params=params,target=forest.TARGET,seed=engine.SEED,
      method='15 leave-one-network-out fits at fixed final hyperparameters; same input rows and existing outer domains. Reuse verified eight-input fixed-setting predictions from closed-domain experiment.',
      limitation='Diagnostic with settings already selected using all networks; does not replace nested-validation performance. Removing a constant feature can change randomized split tie-breaking even with a fixed seed.',
      sources={str(p):engine.sha(p) for p in files})
    fp=hashlib.sha256(json.dumps(design,sort_keys=True).encode()).hexdigest()
    if (OUT/'design.json').exists() and list((OUT/'checkpoints').glob('*.json')) and json.loads((OUT/'design.json').read_text())!=design:raise ValueError('Design changed')
    engine.dump(OUT/'design.json',design)
    rows=[]
    for ex in sorted(d.example.unique()):
      path=OUT/'checkpoints'/f'{ex}.json';saved=forest.checked(path,fp)
      if saved is None:
        tr=d[d.example!=ex];te=d[d.example==ex]
        artifact=fit(tr,params);prediction=forest.predict(artifact,te)
        saved=dict(fingerprint=fp,held=ex,predictions=[dict(example=r.example,state=r.state,predicted_hours=float(p)) for r,p in zip(te.itertuples(),prediction)])
        engine.dump(path,saved);print(ex,'saved',flush=True)
      rows.extend(saved['predictions'])
    a=pd.DataFrame(rows).merge(d[['example','state','centroid_shift_hours','peak_shift_hours']],on=['example','state'],validate='one_to_one')
    a['variant']='seven_inputs'
    reference['variant']='eight_inputs'
    p=pd.concat([a,reference[a.columns]],ignore_index=True)
    p.to_parquet(OUT/'predictions.parquet',index=False)
    scores=[]
    for variant,g in p.groupby('variant'):
      for response in ['centroid_shift_hours','peak_shift_hours']:
       for floor in [1.,2.,5.]:
        yy=g[response]*60
        for subset,mask in [('all',np.ones(len(g),bool)),('unchanged',abs(yy)<=floor),('outside_buffer',abs(yy)>floor),('delay',yy>floor),('advance',yy < -floor),('excluding_extreme',~((g.example=='37')&(g.state=='CB001_bubble_3')))]:
          z=g[mask]
          if z.empty:continue
          y=z[response].to_numpy();pred=z.predicted_hours.to_numpy();w=engine.weights(z)
          scores.append(dict(variant=variant,response=response,floor_minutes=floor,subset=subset,n=len(z),relative_error_pct=forest.sweep.relative_score(y,pred,w,floor),**engine.score(y*60,pred*60,w)))
    pd.DataFrame(scores).to_csv(OUT/'scores.csv',index=False)
    compare=a.merge(reference[['example','state','predicted_hours']],on=['example','state'],suffixes=('_seven','_eight'),validate='one_to_one')
    compare['prediction_difference_minutes']=60*(compare.predicted_hours_seven-compare.predicted_hours_eight)
    compare.to_csv(OUT/'paired_predictions.csv',index=False)
    artifact=fit(d,params);artifact.update(fingerprint=fp,versions=info['versions'],training_rows=len(d),training_examples=sorted(d.example.unique()))
    temp=OUT/'final_model.joblib.tmp';joblib.dump(artifact,temp,compress=3);os.replace(temp,OUT/'final_model.joblib')
    restored=joblib.load(OUT/'final_model.joblib');verification=forest.serialization_check(artifact,restored,d)
    # A seven-column frame must be sufficient; no hidden dependency on exposure.
    np.testing.assert_allclose(forest.predict(restored,d[COLUMNS]),forest.predict(restored,d),atol=1e-12,rtol=1e-12)
    engine.dump(OUT/'final_model.json',dict(fingerprint=fp,columns=COLUMNS,params=params,target=forest.TARGET,output_units='hours',training_rows=len(d),training_examples=sorted(d.example.unique()),versions=info['versions'],model_sha256=engine.sha(OUT/'final_model.joblib'),serialization_verification=verification,performance_warning=design['limitation']))
    print('Seven-input model saved separately:',OUT/'final_model.joblib',flush=True)

if __name__=='__main__':main()
