"""Buffered forest gate + changed-only regression, nested held-example validation.

Prepare by default; --run starts resumable fits. No response definitions change.
"""
from pathlib import Path
import argparse
import hashlib
import json
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import relative_model_sweep as sweep

engine=sweep.engine
ROOT=engine.ROOT
SOURCE=sweep.OUT
OUT=ROOT/'RAPID/outputs/hydrograph_buffered_two_stage_20261019'
TARGET='centroid_shift_hours'
BUFFERS=[1.,2.,5.]
THRESHOLDS=[.2,.35,.5,.65,.8]
CLASSIFIERS=[dict(max_depth=6,min_samples_leaf=l,max_features=1.) for l in [5,15]]
REGRESSORS=sweep.GRIDS['forest']


def changed(y,buffer):
    return np.abs(np.asarray(y))*60>buffer


def gate(probability,prediction,threshold):
    return np.where(np.asarray(probability)>=threshold,prediction,0.)


def classifier_predict(train,test,buffer,params,jobs):
    y=changed(train[TARGET],buffer).astype(int)
    w=engine.weights(train)
    if np.unique(y).size==1:return np.full(len(test),float(y[0])),dict(constant_class=int(y[0]))
    pre=engine.Preprocess(sweep.COLUMNS,None,None).fit(train,y)
    if not pre.kept:return np.full(len(test),np.average(y,weights=w)),dict(constant_probability=True)
    model=RandomForestClassifier(n_estimators=200,random_state=engine.SEED,n_jobs=jobs,**params)
    model.fit(pre.transform(train),y,sample_weight=w)
    p=model.predict_proba(pre.transform(test))[:,list(model.classes_).index(1)]
    return p,dict(kept=pre.kept,dropped=pre.dropped)


def regressor_predict(train,test,buffer,params,jobs):
    subset=train[changed(train[TARGET],buffer)]
    if subset.empty:return np.zeros(len(test)),dict(no_changed_training_rows=True)
    p,audit=sweep.fit_predict(subset,test,TARGET,'forest',params,jobs)
    return p,dict(audit,changed_training_rows=len(subset))


def select(train,buffer,probabilities,regressions):
    """Only inner out-of-fold predictions enter selection; fixed 1-minute loss floor."""
    candidates=[]
    for ci,prob in enumerate(probabilities):
      for ri,pred in enumerate(regressions):
       for threshold in THRESHOLDS:
        p=gate(prob,pred,threshold)
        candidates.append(dict(classifier=ci,regressor=ri,threshold=threshold,
          score=sweep.relative_score(train[TARGET],p,engine.weights(train),1.)))
    return min(candidates,key=lambda x:x['score']),candidates


def fit_fold(train,test,buffer,jobs):
    probabilities=np.full((len(CLASSIFIERS),len(train)),np.nan)
    regressions=np.full((len(REGRESSORS),len(train)),np.nan)
    for ti,vi in sweep.previous.inner_splits(train,'network'):
      tr,va=train.iloc[ti],train.iloc[vi]
      assert set(tr.example).isdisjoint(va.example)
      for i,params in enumerate(CLASSIFIERS):probabilities[i,vi],_=classifier_predict(tr,va,buffer,params,jobs)
      for i,params in enumerate(REGRESSORS):regressions[i,vi],_=regressor_predict(tr,va,buffer,params,jobs)
    assert np.isfinite(probabilities).all() and np.isfinite(regressions).all()
    best,trials=select(train,buffer,probabilities,regressions)
    prob,ca=classifier_predict(train,test,buffer,CLASSIFIERS[best['classifier']],jobs)
    regression,ra=regressor_predict(train,test,buffer,REGRESSORS[best['regressor']],jobs)
    prediction=gate(prob,regression,best['threshold'])
    rows=[dict(example=r.example,state=r.state,observed_hours=float(getattr(r,TARGET)),
          predicted_hours=float(p),changed_probability=float(q),conditional_prediction_hours=float(c),
          gate_changed=bool(q>=best['threshold'])) for r,p,q,c in zip(test.itertuples(),prediction,prob,regression)]
    return dict(best=best,trials=trials,classifier_audit=ca,regressor_audit=ra,predictions=rows)


def prepare(out):
    data=pd.read_parquet(SOURCE/'input.parquet')
    assert len(data)==2170 and not data.duplicated(['example','state']).any()
    assert np.isfinite(data[sweep.COLUMNS+[TARGET,'peak_shift_hours']]).all().all()
    baseline=pd.read_parquet(SOURCE/'predictions.parquet')
    baseline=baseline[(baseline.validation=='network')&(baseline.target==TARGET)&(baseline.model=='forest')&(baseline.floor_minutes==1)]
    for scenario in ['all','without54']:
      expected=data if scenario=='all' else data[data.example!='54']
      z=baseline[baseline.scenario==scenario]
      assert not z.duplicated(['example','state']).any()
      assert set(zip(z.example,z.state))==set(zip(expected.example,expected.state))
    sources=[Path(__file__),Path(sweep.__file__),Path(engine.__file__),Path(sweep.previous.__file__),Path(sweep.focused.__file__),SOURCE/'input.parquet',SOURCE/'predictions.parquet',SOURCE/'design.json']
    design=dict(target=TARGET,columns=sweep.COLUMNS,buffers_minutes=BUFFERS,thresholds=THRESHOLDS,
      classifier_grid=CLASSIFIERS,regressor_grid=REGRESSORS,validation='Leave one whole example out; three grouped inner folds. Scenarios all and without54.',
      selection='Minimum equal-example floored relative error on inner OOF two-stage predictions; denominator max(abs(y),1 minute) for ALL buffer definitions. Exact ties use grid order.',
      stages='Changed label: abs(full centroid shift)>buffer; equality is unchanged. Forest classifier trained on all training rows, forest signed regression on changed training rows only. Gate outputs zero or conditional regression without clipping or snapping.',
      weighting='Equal-example weights independently within classifier and changed-only regressor training sets; no additional class weights. Probabilities not claimed calibrated.',
      comparison='Existing nested one-stage forest with identical features, target and 1-minute tuning floor. Conditional predictions are a diagnostic of the selected two-stage regressor, not independently tuned.',
      reporting='Centroid primary, peak transfer secondary. Evaluation floors 1/2/5 independently of training buffer. Report missed changes and false changes, and evaluation-only exclusion of 37/CB001_bubble_3.',
      limitations='Exploratory repeated use of outer results; no untouched final test. Observed response groups are diagnostics, never input predictors. No extensive fits without --run.',
      versions=json.loads((SOURCE/'design.json').read_text())['versions'],sources={str(p):engine.sha(p) for p in sources})
    fingerprint=hashlib.sha256(json.dumps(design,sort_keys=True).encode()).hexdigest()
    out.mkdir(parents=True,exist_ok=True)
    if (out/'design.json').exists() and list((out/'checkpoints').glob('**/*.json')):
      if json.loads((out/'design.json').read_text())!=design:raise ValueError('Design changed; use a fresh output directory')
    engine.dump(out/'design.json',design)
    plans=[]
    for scenario in ['all','without54']:
      cohort=data if scenario=='all' else data[data.example!='54']
      for held,tr,te in sweep.previous.outer_splits(cohort,'network'):
       for buffer in BUFFERS:
        plans.append(dict(scenario=scenario,held=held,buffer_minutes=buffer,test_rows=len(te),fits=len(sweep.previous.inner_splits(tr,'network'))*(len(CLASSIFIERS)+len(REGRESSORS))+2))
    plan=pd.DataFrame(plans);plan.to_csv(out/'run_plan.csv',index=False)
    print(f'Prepared {len(plan)} checkpoints; {plan.fits.sum()} fits; {len(CLASSIFIERS)*len(REGRESSORS)*len(THRESHOLDS)} gate/regressor settings per buffer. No fits without --run.',flush=True)
    # Evaluation-only exclusion: predictions remain exactly the existing held-network predictions.
    diagnostics=[]
    for response in [TARGET,'peak_shift_hours']:
      a=baseline[baseline.scenario=='all'].merge(data[['example','state',response]],on=['example','state'],validate='one_to_one')
      for omit in [False,True]:
        z=a[~((a.example=='37')&(a.state=='CB001_bubble_3'))] if omit else a
        diagnostics.append(dict(response=response,exclude_extreme=omit,n=len(z),**engine.score(z[response]*60,z.predicted_hours*60,engine.weights(z))))
    pd.DataFrame(diagnostics).to_csv(out/'extreme_case_diagnostic.csv',index=False)
    return data,baseline,design,fingerprint


def summarize(out,data,baseline,fingerprint):
    pieces=[]
    for path in (out/'checkpoints').glob('*/*.json'):
      c=json.loads(path.read_text());assert c['fingerprint']==fingerprint
      z=pd.DataFrame(c['predictions']);z['scenario']=c['scenario'];z['buffer_minutes']=c['buffer_minutes'];z['held']=c['held'];pieces.append(z)
    if not pieces:return
    pred=pd.concat(pieces,ignore_index=True)
    assert not pred.duplicated(['scenario','buffer_minutes','example','state']).any()
    pred=pred.merge(data[['example','state','edit_type',TARGET,'peak_shift_hours']],on=['example','state'],validate='many_to_one')
    pred=pred.merge(baseline[['scenario','example','state','predicted_hours']].rename(columns={'predicted_hours':'one_stage_hours'}),on=['scenario','example','state'],validate='many_to_one',how='left')
    assert pred.one_stage_hours.notna().all()
    np.testing.assert_allclose(pred.observed_hours,pred[TARGET])
    pred.to_parquet(out/'predictions.parquet',index=False)
    scores=[];gates=[]
    def rate(z,condition):
      return float(np.average(np.asarray(condition,float),weights=engine.weights(z))) if len(z) else None
    for (scenario,buffer),g in pred.groupby(['scenario','buffer_minutes']):
      expected=data if scenario=='all' else data[data.example!='54']
      common=dict(scenario=scenario,buffer_minutes=buffer,complete=len(g)==len(expected))
      for omit in [False,True]:
       a=g[~((g.example=='37')&(g.state=='CB001_bubble_3'))] if omit else g
       for response in [TARGET,'peak_shift_hours']:
        for floor in [1.,2.,5.]:
         y=a[response]*60
         for subset,mask in dict(all=np.ones(len(a),bool),unchanged=abs(y)<=floor,advance=y < -floor,delay=y > floor,outside_buffer=abs(y)>floor).items():
          z=a[mask]
          if z.empty:continue
          for name,column in [('two_stage','predicted_hours'),('one_stage','one_stage_hours'),('conditional_diagnostic','conditional_prediction_hours')]:
           yy=z[response].to_numpy();pp=z[column].to_numpy();w=engine.weights(z)
           scores.append(dict(**common,exclude_extreme=omit,response=response,evaluation_floor_minutes=floor,subset=subset,model=name,n=len(z),examples=z.example.nunique(),relative_error_pct=sweep.relative_score(yy,pp,w,floor),mape_pct=float(100*np.average(abs((pp-yy)/yy),weights=w)) if subset=='outside_buffer' else None,**engine.score(yy*60,pp*60,w)))
      for model in ['two_stage','one_stage']:
       # Gate decision and final numerical response class are distinct: a changed-only
       # regressor can still interpolate to a value within the unchanged buffer.
       decided=g.gate_changed if model=='two_stage' else changed(g.one_stage_hours,buffer)
       output=g.predicted_hours if model=='two_stage' else g.one_stage_hours
       actual=changed(g[TARGET],buffer)
       gates.append(dict(**common,model=model,n=len(g),false_changed_rate=rate(g[~actual],decided[~actual]),missed_changed_rate=rate(g[actual],~decided[actual]),final_missed_changed_rate=rate(g[actual],~changed(output[actual],buffer)),
         **{f'missed_over_{v:g}min':rate(g[abs(g[TARGET])*60>v],~decided[abs(g[TARGET])*60>v]) for v in [5.,15.,60.]}))
    pd.DataFrame(scores).to_csv(out/'scores.csv',index=False)
    pd.DataFrame(gates).to_csv(out/'gate_scores.csv',index=False)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',action='store_true');parser.add_argument('--jobs',type=int,default=1)
    parser.add_argument('--output',type=Path,default=OUT)
    args=parser.parse_args()
    if args.jobs<1:parser.error('--jobs must be positive')
    data,baseline,design,fp=prepare(args.output)
    if args.run:
      for scenario in ['all','without54']:
       cohort=data if scenario=='all' else data[data.example!='54']
       for held,tr,te in sweep.previous.outer_splits(cohort,'network'):
        assert set(tr.example).isdisjoint(te.example)
        for buffer in BUFFERS:
         path=args.output/'checkpoints'/scenario/f'{held}__buffer{buffer:g}.json'
         if path.exists():
          assert json.loads(path.read_text())['fingerprint']==fp
          continue
         result=fit_fold(tr,te,buffer,args.jobs)
         engine.dump(path,dict(fingerprint=fp,scenario=scenario,held=held,buffer_minutes=buffer,**result))
         print(scenario,held,'buffer',buffer,'saved',flush=True)
       summarize(args.output,data,baseline,fp)
    summarize(args.output,data,baseline,fp)

if __name__=='__main__':main()
