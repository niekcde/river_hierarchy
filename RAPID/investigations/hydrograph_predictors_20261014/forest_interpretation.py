"""Held-network forest interpretation and a focused resistance comparison.

Default prepares resistance plan. --interpret reconstructs saved forests only.
--run-resistance explicitly starts the nested resistance fits.
SHAP can be installed separately; --shap-path accepts an isolated dependency path.
"""
from pathlib import Path
import argparse,hashlib,json,sys
import numpy as np
import pandas as pd
import relative_model_sweep as sweep

engine=sweep.engine
ROOT=engine.ROOT;SOURCE=sweep.OUT
OUT=ROOT/'RAPID/outputs/hydrograph_forest_interpretation_20261019'
TARGET='centroid_shift_hours'
WIDTH='log1p__B_ratio_width_mean';RESISTANCE='B_K_log_a1'
PANELS={'current':sweep.COLUMNS,'plus_resistance':sweep.COLUMNS+[RESISTANCE],
        'resistance_replaces_width':[c for c in sweep.COLUMNS if c!=WIDTH]+[RESISTANCE]}
LABELS=dict(zip(sweep.COLUMNS,['R_muL','muL_Dp / 1 km','R_sigmaL','R_W','R_CVW','R_H','d_Dp','r_in']))
LABELS[RESISTANCE]='-log10(lambda_K)'
GROUPS={**{LABELS[c]:[c] for c in sweep.COLUMNS},
 'length_group':sweep.COLUMNS[:3], 'width_group':[WIDTH,'log1p__B_ratio_width_cv','B_parent_dominance_mean'],
 'length_width_group':sweep.COLUMNS[:5],
 'width_entropy_group':[WIDTH,'log1p__B_ratio_width_cv','log1p__B_ratio_route_entropy','B_parent_dominance_mean']}


def fit(train,columns,params,jobs):
    y=train[TARGET].to_numpy(float);w=engine.weights(train)
    pre=engine.Preprocess(columns,None,None).fit(train,y)
    mean=float(np.average(y,weights=w));sd=float(np.sqrt(np.average((y-mean)**2,weights=w)))
    if sd<1e-14 or not pre.kept:raise ValueError('Degenerate forest fold')
    model=engine.estimator('forest',params,jobs)
    model.fit(pre.transform(train),(y-mean)/sd,sample_weight=w)
    return pre,model,mean,sd


def predict(fitted,frame):
    pre,model,mean,sd=fitted
    return model.predict(pre.transform(frame))*sd+mean


def prepare(out):
    d=pd.read_parquet(SOURCE/'input.parquet')
    assert len(d)==2170 and not d.duplicated(['example','state']).any()
    assert np.isfinite(d[PANELS['plus_resistance']+[TARGET,'peak_shift_hours']]).all().all()
    files=[Path(__file__),Path(sweep.__file__),Path(engine.__file__),Path(sweep.previous.__file__),Path(sweep.focused.__file__),SOURCE/'input.parquet',SOURCE/'predictions.parquet',SOURCE/'design.json']
    design=dict(target=TARGET,panels=PANELS,grid=sweep.GRIDS['forest'],validation='Whole-example outer holdout, three example-grouped inner folds. All and without54 scenarios.',
      tuning='Equal-network floored relative error, fixed 1-minute denominator floor.',
      baseline='Reuse existing one-stage forest; no tuning changes. New panels tune independently within identical inner folds.',
      interpretation='Reconstruct all-scenario saved forests, verify saved predictions. 20 within-held-network permutation repeats by default; grouped columns use the same permutation. Positive loss increase means useful information. Networks weighted equally; global R2 uses fixed weighted response variance, not mean fold R2.',
      shap='Exact TreeSHAP, tree_path_dependent. Explains held-network predictions relative to each fitted forest training-path expectation. Training weights retained; not an interventional or causal claim. Mean absolute SHAP aggregated equally across examples. Pair interactions reported as twice off-diagonal entries.',
      cautions=['Within-network permutation cannot assess predictors constant within that network. Correlated-feature perturbations can be implausible; grouped results supplement individual results.', 'Small R_in variation may be floating-point noise; no retrospective changes to the fitted models.', 'SHAP magnitude describes prediction allocation, not predictive value or physical causality.', 'Exploratory use of the same outer examples; no untouched final test.'],
      sources={str(p):engine.sha(p) for p in files})
    out.mkdir(parents=True,exist_ok=True)
    old=out/'design.json'
    if old.exists() and list((out/'resistance_checkpoints').glob('**/*.json')) and json.loads(old.read_text())!=design:
        raise ValueError('Design changed; choose a fresh output directory')
    engine.dump(old,design)
    fp=hashlib.sha256(json.dumps(design,sort_keys=True).encode()).hexdigest()
    plan=[]
    for scenario in ['all','without54']:
      cohort=d if scenario=='all' else d[d.example!='54']
      for held,tr,te in sweep.previous.outer_splits(cohort,'network'):
       for panel in ['plus_resistance','resistance_replaces_width']:
        plan.append(dict(scenario=scenario,held=held,panel=panel,test_rows=len(te),fits=3*len(sweep.GRIDS['forest'])+1))
    plan=pd.DataFrame(plan);plan.to_csv(out/'resistance_run_plan.csv',index=False)
    print(f'Resistance plan: {len(plan)} checkpoints, {plan.fits.sum()} fits; no comparison fits without --run-resistance.',flush=True)
    d[sweep.COLUMNS+[RESISTANCE]].corr(method='spearman').to_csv(out/'predictor_spearman.csv')
    return d,fp


def interpret(d,out,jobs,repeats):
    import shap
    permutation=[];shap_rows=[];interaction_rows=[];support=[];baseline_rows=[]
    for held,tr,te in sweep.previous.outer_splits(d,'network'):
      checkpoint=SOURCE/'checkpoints/all/network'/held/'centroid_shift_hours__forest.json'
      selected=next(r for r in json.loads(checkpoint.read_text())['results'] if r['floor_minutes']==1)
      fitted=fit(tr,sweep.COLUMNS,selected['params'],jobs)
      pre,model,mean,sd=fitted;xt=pre.transform(te);prediction=predict(fitted,te)
      expected=pd.DataFrame(selected['predictions']).set_index('state').loc[te.state,'predicted_hours']
      np.testing.assert_allclose(prediction,expected,rtol=1e-10,atol=1e-10)
      y=te[TARGET].to_numpy(float);base_relative=sweep.relative_score(y,prediction,np.ones(len(te)),1)
      base_mse=float(np.mean((prediction-y)**2)*3600)
      baseline_rows.extend(dict(example=held,state=r.state,observed_minutes=float(v*60),predicted_minutes=float(p*60)) for r,v,p in zip(te.itertuples(),y,prediction))
      for c in sweep.COLUMNS:
        support.append(dict(example=held,feature=c,symbol=LABELS[c],train_range=float(np.ptp(tr[c])),test_range=float(np.ptp(te[c])),test_unique=int(te[c].nunique()),numerically_constant=bool(np.ptp(te[c])<1e-10)))
      for name,cols in GROUPS.items():
       indices=[pre.kept.index(c) for c in cols if c in pre.kept]
       for repeat in range(repeats):
        order=np.random.default_rng(engine.SEED+int(held)*1000+repeat).permutation(len(te))
        xp=xt.copy();xp[:,indices]=xt[order][:,indices]
        pp=model.predict(xp)*sd+mean
        permutation.append(dict(example=held,group=name,repeat=repeat,delta_relative_pp=sweep.relative_score(y,pp,np.ones(len(te)),1)-base_relative,delta_mse_minutes2=float(np.mean((pp-y)**2)*3600-base_mse)))
      explainer=shap.TreeExplainer(model,feature_perturbation='tree_path_dependent',model_output='raw')
      values=np.asarray(explainer.shap_values(xt,check_additivity=True))*sd*60
      base=float(np.asarray(explainer.expected_value).reshape(-1)[0])*sd*60+mean*60
      np.testing.assert_allclose(base+values.sum(axis=1),prediction*60,atol=1e-6)
      interactions=np.asarray(explainer.shap_interaction_values(xt))*sd*60
      np.testing.assert_allclose(interactions.sum(axis=2),values,atol=1e-6)
      for j,c in enumerate(sweep.COLUMNS):
        v=values[:,pre.kept.index(c)] if c in pre.kept else np.zeros(len(te))
        original=np.expm1(te[c].to_numpy()) if c.startswith('log1p__') else te[c].to_numpy()
        shap_rows.extend(dict(example=held,state=r.state,edit_type=r.edit_type,feature=c,symbol=LABELS[c],value=float(x),shap_minutes=float(s),base_minutes=base,predicted_minutes=float(p*60)) for r,x,s,p in zip(te.itertuples(),original,v,prediction))
      for c in ['log1p__B_ratio_route_length_mean',WIDTH]:
        if c in pre.kept and 'domain_size' in pre.kept:
          vals=2*interactions[:,pre.kept.index(c),pre.kept.index('domain_size')]
          interaction_rows.extend(dict(example=held,state=r.state,pair=LABELS[c]+' × size',interaction_minutes=float(v),ratio=float(np.expm1(getattr(r,c))),domain_size=float(r.domain_size)) for r,v in zip(te.itertuples(),vals))
      print('Interpretation saved in memory:',held,flush=True)
    p=pd.DataFrame(permutation);p.to_csv(out/'permutation_repeats.csv',index=False)
    by=p.groupby(['example','group'])[['delta_relative_pp','delta_mse_minutes2']].mean().reset_index()
    by.to_csv(out/'permutation_by_network.csv',index=False)
    # Equal-example global R2 decrease; a single fixed SST for all perturbations.
    y=d[TARGET].to_numpy()*60;w=engine.weights(d);variance=np.average((y-np.average(y,weights=w))**2,weights=w)
    summary=by.groupby('group').agg(relative_increase_pp=('delta_relative_pp','mean'),mse_increase_minutes2=('delta_mse_minutes2','mean'),networks_helped_relative=('delta_relative_pp',lambda x:int((x>0).sum())),networks_helped_mse=('delta_mse_minutes2',lambda x:int((x>0).sum())))
    summary['r2_decrease']=summary.mse_increase_minutes2/variance
    summary.to_csv(out/'permutation_summary.csv')
    s=pd.DataFrame(shap_rows);s.to_parquet(out/'shap_values.parquet',index=False)
    s.assign(abs_shap_minutes=s.shap_minutes.abs()).groupby(['example','feature','symbol']).agg(mean_abs_shap_minutes=('abs_shap_minutes','mean'),value_shap_spearman=('shap_minutes',lambda x:np.nan)).reset_index().drop(columns='value_shap_spearman').to_csv(out/'shap_by_network.csv',index=False)
    sh=pd.read_csv(out/'shap_by_network.csv');sh.groupby(['feature','symbol']).mean_abs_shap_minutes.mean().sort_values(ascending=False).to_csv(out/'shap_summary.csv')
    pd.DataFrame(interaction_rows).to_parquet(out/'shap_interactions.parquet',index=False)
    pd.DataFrame(support).to_csv(out/'predictor_support.csv',index=False)
    pd.DataFrame(baseline_rows).to_csv(out/'reconstructed_predictions.csv',index=False)
    engine.dump(out/'interpretation_manifest.json',dict(shap_version=shap.__version__,permutation_repeats=repeats,held_networks=int(d.example.nunique()),prediction_checks='All held-network predictions reproduced; SHAP and interaction additivity checked.',documentation='https://shap.readthedocs.io/en/latest/generated/shap.TreeExplainer.html'))


def run_resistance(d,out,fp,jobs):
    for scenario in ['all','without54']:
      cohort=d if scenario=='all' else d[d.example!='54']
      for held,tr,te in sweep.previous.outer_splits(cohort,'network'):
       for panel in ['plus_resistance','resistance_replaces_width']:
        path=out/'resistance_checkpoints'/scenario/f'{held}__{panel}.json'
        if path.exists():
          assert json.loads(path.read_text())['fingerprint']==fp
          continue
        trials=[];cols=PANELS[panel]
        for params in sweep.GRIDS['forest']:
          oof=np.full(len(tr),np.nan)
          for ti,vi in sweep.previous.inner_splits(tr,'network'):
            a,b=tr.iloc[ti],tr.iloc[vi]
            assert set(a.example).isdisjoint(b.example)
            oof[vi]=predict(fit(a,cols,params,jobs),b)
          assert np.isfinite(oof).all()
          trials.append(dict(params=params,score=sweep.relative_score(tr[TARGET],oof,engine.weights(tr),1)))
        best=min(trials,key=lambda a:a['score']);fitted=fit(tr,cols,best['params'],jobs);pred=predict(fitted,te)
        rows=[dict(example=r.example,state=r.state,observed_hours=float(getattr(r,TARGET)),predicted_hours=float(p)) for r,p in zip(te.itertuples(),pred)]
        engine.dump(path,dict(fingerprint=fp,scenario=scenario,held=held,panel=panel,best=best,trials=trials,predictions=rows))
        print('Resistance:',scenario,held,panel,'saved',flush=True)
    summarize_resistance(d,out,fp)


def summarize_resistance(d,out,fp):
    baseline=pd.read_parquet(SOURCE/'predictions.parquet')
    baseline=baseline[(baseline.validation=='network')&(baseline.target==TARGET)&(baseline.floor_minutes==1)&(baseline.model=='forest')].copy()
    baseline['panel']='current';pieces=[baseline[['scenario','panel','example','state','observed_hours','predicted_hours']]]
    for path in (out/'resistance_checkpoints').glob('*/*.json'):
      c=json.loads(path.read_text());assert c['fingerprint']==fp
      a=pd.DataFrame(c['predictions']);a['scenario']=c['scenario'];a['panel']=c['panel'];pieces.append(a)
    p=pd.concat(pieces,ignore_index=True).merge(d[['example','state',TARGET,'peak_shift_hours']],on=['example','state'],validate='many_to_one')
    assert not p.duplicated(['scenario','panel','example','state']).any()
    np.testing.assert_allclose(p.observed_hours,p[TARGET])
    p.to_parquet(out/'resistance_predictions.parquet',index=False);scores=[]
    for (scenario,panel),g in p.groupby(['scenario','panel']):
      expected=len(d if scenario=='all' else d[d.example!='54'])
      for response in [TARGET,'peak_shift_hours']:
       for exclude in [False,True]:
        a=g[~((g.example=='37')&(g.state=='CB001_bubble_3'))] if exclude else g
        for subset,mask in [('all',np.ones(len(a),bool)),('outside_buffer',abs(a[response])*60>1),('unchanged',abs(a[response])*60<=1),('delay',a[response]*60>1),('advance',a[response]*60 < -1)]:
          z=a[mask]
          if z.empty:continue
          yy=z[response].to_numpy();pp=z.predicted_hours.to_numpy();w=engine.weights(z)
          scores.append(dict(scenario=scenario,panel=panel,response=response,exclude_extreme=exclude,subset=subset,n=len(z),complete=len(g)==expected,relative_error_pct=sweep.relative_score(yy,pp,w,1),**engine.score(yy*60,pp*60,w)))
    pd.DataFrame(scores).to_csv(out/'resistance_scores.csv',index=False)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--interpret',action='store_true');p.add_argument('--run-resistance',action='store_true')
    p.add_argument('--shap-path',type=Path);p.add_argument('--jobs',type=int,default=1);p.add_argument('--repeats',type=int,default=20)
    p.add_argument('--output',type=Path,default=OUT);args=p.parse_args()
    if args.jobs<1 or args.repeats<1:p.error('jobs and repeats must be positive')
    if args.shap_path:sys.path.insert(0,str(args.shap_path))
    d,fp=prepare(args.output)
    if args.interpret:interpret(d,args.output,args.jobs,args.repeats)
    if args.run_resistance:run_resistance(d,args.output,fp,args.jobs)
    summarize_resistance(d,args.output,fp)

if __name__=='__main__':main()
