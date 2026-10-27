"""Broad forest hyperparameter selection, nested evaluation and final all-data fit.

Default prepares only. --run evaluates the search, selects one setting and saves
its all-data fitted artifact. Interrupted searches resume at candidate boundaries.
"""
from pathlib import Path
import argparse,hashlib,json,os
import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestRegressor
import relative_model_sweep as sweep

engine=sweep.engine
ROOT=engine.ROOT;SOURCE=sweep.OUT
OUT=ROOT/'RAPID/outputs/hydrograph_final_forest_search_20261019'
TARGET='centroid_shift_hours'
SEED=20261019
# This version differs only in the final serialization check and safe resume.
LEGACY_SEARCH_SOURCE='0411acfddc59263cd34a86c11c8ea51acfe82591c8a7fc28f1fc72a4378c7183'


def serialization_check(original,restored,frame):
    # Parallel tree reduction can change the last few floating-point bits.
    a,b=predict(original,frame),predict(restored,frame)
    np.testing.assert_allclose(a,b,rtol=1e-12,atol=1e-12,equal_nan=False)
    return dict(max_absolute_difference_hours=float(np.max(np.abs(a-b))),rtol=1e-12,atol_hours=1e-12)


def resume_design(out,design):
    path=out/'design.json'
    if not path.exists():return design
    stored=json.loads(path.read_text())
    if stored==design:return stored
    if not (list((out/'tuning').glob('**/*.json')) or (out/'final_model.joblib').exists()):return design
    key=str(Path(__file__))
    updated=json.loads(json.dumps(stored))
    updated['sources'][key]=design['sources'][key]
    if stored['sources'].get(key)!=LEGACY_SEARCH_SOURCE or updated!=design:
        raise ValueError('Existing search design differs; use a fresh --output directory')
    # Keep the original search design/fingerprint: all fits and scores are unchanged.
    # Record the new finalization implementation separately rather than rewriting provenance.
    engine.dump(out/'finalization_compatibility.json',dict(original_source=LEGACY_SEARCH_SOURCE,
      finalization_source=design['sources'][key],reason='Serialization tolerance and safe resume only; search inputs, settings and scoring identical.'))
    return stored


def candidates():
    """Fixed unique sampled designs, including the exact six earlier settings."""
    grids=[];seen=set()
    def add(p):
        key=json.dumps(p,sort_keys=True)
        if key not in seen:seen.add(key);grids.append(p)
    for p in sweep.GRIDS['forest']:
        add(dict(n_estimators=200,criterion='squared_error',min_samples_split=2,bootstrap=True,max_samples=None,ccp_alpha=0.,**p))
    rng=np.random.default_rng(SEED)
    def choose(items):return items[int(rng.integers(len(items)))]
    for criterion,total in [('squared_error',80),('absolute_error',96)]:
      while len(grids)<total:
        leaf=choose([1,2,5,10,15,25]);bootstrap=choose([True,True,True,False])
        # Split thresholds <=2*leaf are equivalent restrictions, so use only 2
        # or a genuinely stricter threshold, reducing redundant configurations.
        split=choose([2]+[v for v in [5,10,20,50,80] if v>2*leaf])
        add(dict(n_estimators=choose([200,500]),criterion=criterion,
          max_depth=choose([3,6,10,16,None]),min_samples_leaf=leaf,min_samples_split=split,
          max_features=choose([.5,.75,1.]),bootstrap=bootstrap,
          max_samples=choose([None,.5,.75]) if bootstrap else None,
          ccp_alpha=choose([0.,0.,.0001,.001,.01])))
    return grids


def fit(train,params,jobs):
    y=train[TARGET].to_numpy(float);w=engine.weights(train)
    pre=engine.Preprocess(sweep.COLUMNS,None,None).fit(train,y)
    mean=float(np.average(y,weights=w));sd=float(np.sqrt(np.average((y-mean)**2,weights=w)))
    if not pre.kept or sd<1e-14:raise ValueError('Degenerate training fold')
    model=RandomForestRegressor(**params,random_state=engine.SEED,n_jobs=jobs)
    model.fit(pre.transform(train),(y-mean)/sd,sample_weight=w)
    return dict(preprocessor=pre,estimator=model,target_mean=mean,target_sd=sd,
      columns=list(sweep.COLUMNS),target=TARGET,prediction_units='hours',params=params)


def predict(artifact,frame):
    required=artifact['columns']
    if not np.isfinite(frame[required].to_numpy(float)).all():raise ValueError('Finite engineered predictors required')
    return artifact['estimator'].predict(artifact['preprocessor'].transform(frame))*artifact['target_sd']+artifact['target_mean']


def checked(path,fp):
    if not path.exists():return None
    obj=json.loads(path.read_text())
    if obj['fingerprint']!=fp:raise ValueError(f'Incompatible checkpoint: {path}')
    return obj


def tune(train,grids,out,fp,jobs,keep_predictions=False):
    splits=sweep.previous.inner_splits(train,'network');trials=[]
    for k,params in enumerate(grids):
      path=out/f'candidate_{k:03d}.json';saved=checked(path,fp)
      if saved is not None:
        if saved['params']!=params:raise ValueError('Candidate mismatch')
        trials.append(saved);continue
      oof=np.full(len(train),np.nan)
      for ti,vi in splits:
        tr,va=train.iloc[ti],train.iloc[vi]
        assert set(tr.example).isdisjoint(va.example)
        oof[vi]=predict(fit(tr,params,jobs),va)
      assert np.isfinite(oof).all()
      y=train[TARGET].to_numpy();w=engine.weights(train)
      by=[]
      for ex in sorted(train.example.unique()):
        mask=train.example.to_numpy()==ex
        by.append(dict(example=ex,relative_error_pct=sweep.relative_score(y[mask],oof[mask],np.ones(mask.sum()),1.)))
      trial=dict(fingerprint=fp,candidate=k,params=params,
        selection_score=sweep.relative_score(y,oof,w,1.),by_example=by,
        **engine.score(y*60,oof*60,w))
      if keep_predictions:
        trial['predictions']=[dict(example=r.example,state=r.state,observed_hours=float(v),predicted_hours=float(p)) for r,v,p in zip(train.itertuples(),y,oof)]
      engine.dump(path,trial);trials.append(trial)
      if (k+1)%8==0 or k==len(grids)-1:print(out.relative_to(out.parents[1]),f'{k+1}/{len(grids)} candidates saved',flush=True)
    # Fixed order breaks exact ties; no use of outer outcomes or post-hoc exceptions.
    return min(trials,key=lambda r:(r['selection_score'],r['candidate'])),trials


def prepare(out):
    d=pd.read_parquet(SOURCE/'input.parquet')
    assert len(d)==2170 and not d.duplicated(['example','state']).any()
    assert np.isfinite(d[sweep.COLUMNS+[TARGET,'peak_shift_hours']]).all().all()
    grids=candidates()
    files=[Path(__file__),Path(sweep.__file__),Path(engine.__file__),Path(sweep.previous.__file__),Path(sweep.focused.__file__),SOURCE/'input.parquet',SOURCE/'predictions.parquet',SOURCE/'design.json']
    design=dict(columns=sweep.COLUMNS,target=TARGET,candidates=grids,candidate_seed=SEED,forest_seed=engine.SEED,
      selection='Minimum equal-example mean abs(error)/max(abs(observed shift),1 minute), using grouped 3-fold OOF predictions. Ties broken by fixed candidate order.',
      fitting='Standardized target; equal-example sample weights. Squared-error trees fit weighted means; absolute-error trees fit weighted medians. Neither directly fits percentage loss. ccp_alpha acts on standardized target.',
      validation='Outer leave-one-example-out, inner GroupKFold(3) by example. All and without54 scenarios. No row-wise CV or out-of-bag scoring for selection.',
      final_selection='Same 96 candidates and 3-fold grouped tuning on all 15 examples, then one all-data refit. Selection CV scores are not reported as unbiased performance.',
      reporting='Nested OOF metrics evaluate the tuning procedure. Final artifact is a single fitted model, not the fold ensemble. Peak timing evaluated as transfer of centroid predictions. Buffers 1,2,5 minutes and extreme-case exclusion are diagnostic only.',
      frozen='Eight agreed predictors including constant exposure; no resistance, gating or variable selection. Extreme case remains in training. Example54 remains in final training.',
      limitations='Only 15 independent examples; repeated exploratory reuse of outer results. Broad search can overfit selection data. Final model has no untouched test set. Tree averages/medians still cannot extrapolate beyond training target support.',
      docs=['https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html','https://scikit-learn.org/stable/auto_examples/model_selection/plot_nested_cross_validation_iris.html'],
      versions=dict(sklearn=sklearn.__version__,numpy=np.__version__,pandas=pd.__version__,joblib=joblib.__version__),
      sources={str(p):engine.sha(p) for p in files})
    out.mkdir(parents=True,exist_ok=True)
    existing=out/'design.json'
    design=resume_design(out,design)
    engine.dump(existing,design);fp=hashlib.sha256(json.dumps(design,sort_keys=True).encode()).hexdigest()
    pd.DataFrame([dict(candidate=i,**p) for i,p in enumerate(grids)]).to_csv(out/'candidates.csv',index=False)
    plan=[]
    for scenario in ['all','without54']:
      cohort=d if scenario=='all' else d[d.example!='54']
      for held,tr,te in sweep.previous.outer_splits(cohort,'network'):
        plan.append(dict(stage='nested',scenario=scenario,held=held,test_rows=len(te),candidate_fits=3*len(grids),winner_fits=1))
    plan.append(dict(stage='final',scenario='all',held='all_data',test_rows=0,candidate_fits=3*len(grids),winner_fits=1))
    plan=pd.DataFrame(plan);plan.to_csv(out/'run_plan.csv',index=False)
    print(f'Prepared {len(grids)} settings, {int((plan.candidate_fits+plan.winner_fits).sum())} planned fits. No fits without --run.',flush=True)
    return d,grids,fp,design


def nested(d,grids,out,fp,jobs):
    for scenario in ['all','without54']:
      cohort=d if scenario=='all' else d[d.example!='54']
      for held,tr,te in sweep.previous.outer_splits(cohort,'network'):
        path=out/'outer_predictions'/scenario/f'{held}.json'
        if checked(path,fp) is not None:continue
        best,_=tune(tr,grids,out/'tuning'/scenario/held,fp,jobs)
        prediction=predict(fit(tr,best['params'],jobs),te)
        rows=[dict(example=r.example,state=r.state,observed_hours=float(getattr(r,TARGET)),predicted_hours=float(p)) for r,p in zip(te.itertuples(),prediction)]
        engine.dump(path,dict(fingerprint=fp,scenario=scenario,held=held,best_candidate=best['candidate'],params=best['params'],inner_selection_score=best['selection_score'],predictions=rows))
        print('Nested',scenario,held,'saved',flush=True)
      summarize(d,out,fp)


def finalize(d,grids,out,fp,jobs,design):
    info=checked(out/'final_model.json',fp)
    if info is not None and (out/'final_model.joblib').exists():
      if engine.sha(out/'final_model.joblib')!=info['model_sha256']:raise ValueError('Final model checksum mismatch')
      return
    best,trials=tune(d,grids,out/'tuning'/'final'/'all_data',fp,jobs,keep_predictions=True)
    pd.DataFrame([{k:v for k,v in t.items() if k not in ['predictions','by_example','fingerprint']} for t in trials]).sort_values('selection_score').to_csv(out/'final_selection_scores.csv',index=False)
    pd.DataFrame(best['predictions']).to_parquet(out/'final_selection_oof.parquet',index=False)
    artifact=fit(d,best['params'],jobs)
    artifact.update(fingerprint=fp,versions=design['versions'],training_examples=sorted(d.example.unique()),training_rows=len(d))
    tmp=out/'final_model.joblib.tmp';joblib.dump(artifact,tmp,compress=3);os.replace(tmp,out/'final_model.joblib')
    restored=joblib.load(out/'final_model.joblib')
    verification=serialization_check(artifact,restored,d)
    engine.dump(out/'final_model.json',dict(fingerprint=fp,model_sha256=engine.sha(out/'final_model.joblib'),candidate=best['candidate'],params=best['params'],selection_cv_relative_error_pct=best['selection_score'],selection_score_warning='Used to choose this setting; not independent performance. Use nested_scores.csv.',columns=sweep.COLUMNS,target=TARGET,output_units='hours',training_rows=len(d),training_examples=sorted(d.example.unique()),versions=design['versions'],serialization_verification=verification))
    print('Final all-data model saved:',out/'final_model.joblib',flush=True)


def summarize(d,out,fp):
    baseline=pd.read_parquet(SOURCE/'predictions.parquet')
    baseline=baseline[(baseline.validation=='network')&(baseline.target==TARGET)&(baseline.floor_minutes==1)&(baseline.model=='forest')].copy()
    baseline['model']='previous_six_settings';pieces=[baseline[['scenario','model','example','state','observed_hours','predicted_hours']]];choices=[]
    for path in (out/'outer_predictions').glob('*/*.json'):
      r=checked(path,fp);a=pd.DataFrame(r['predictions']);a['scenario']=r['scenario'];a['model']='broad_search';pieces.append(a)
      choices.append(dict(scenario=r['scenario'],held=r['held'],candidate=r['best_candidate'],params=json.dumps(r['params']),inner_score=r['inner_selection_score']))
    pred=pd.concat(pieces,ignore_index=True).merge(d[['example','state',TARGET,'peak_shift_hours']],on=['example','state'],validate='many_to_one')
    assert not pred.duplicated(['scenario','model','example','state']).any()
    np.testing.assert_allclose(pred.observed_hours,pred[TARGET])
    pred.to_parquet(out/'nested_predictions.parquet',index=False)
    pd.DataFrame(choices).to_csv(out/'outer_chosen_parameters.csv',index=False)
    rows=[]
    for (scenario,model),g in pred.groupby(['scenario','model']):
      expected=len(d if scenario=='all' else d[d.example!='54'])
      for response in [TARGET,'peak_shift_hours']:
       for omit in [False,True]:
        a=g[~((g.example=='37')&(g.state=='CB001_bubble_3'))] if omit else g
        for floor in [1.,2.,5.]:
         y=a[response]*60
         for subset,mask in [('all',np.ones(len(a),bool)),('outside_buffer',abs(y)>floor),('unchanged',abs(y)<=floor),('advance',y < -floor),('delay',y>floor)]:
          z=a[mask]
          if z.empty:continue
          yy=z[response].to_numpy();pp=z.predicted_hours.to_numpy();w=engine.weights(z)
          rows.append(dict(scenario=scenario,model=model,response=response,exclude_extreme=omit,evaluation_floor_minutes=floor,subset=subset,n=len(z),complete=len(g)==expected,relative_error_pct=sweep.relative_score(yy,pp,w,floor),mape_pct=float(100*np.average(abs((pp-yy)/yy),weights=w)) if subset=='outside_buffer' else None,**engine.score(yy*60,pp*60,w)))
    pd.DataFrame(rows).to_csv(out/'nested_scores.csv',index=False)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',action='store_true');parser.add_argument('--jobs',type=int,default=1)
    parser.add_argument('--output',type=Path,default=OUT)
    args=parser.parse_args()
    if args.jobs<1:parser.error('--jobs must be positive')
    d,grids,fp,design=prepare(args.output)
    if args.run:
      nested(d,grids,args.output,fp,args.jobs)
      finalize(d,grids,args.output,fp,args.jobs,design)
    summarize(d,args.output,fp)

if __name__=='__main__':main()
