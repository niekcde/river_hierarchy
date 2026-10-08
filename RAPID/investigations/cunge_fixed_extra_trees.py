"""Frozen 15-feature Extra Trees, +n ablation, absolute/relative timing.

Reuses routed outputs. Four complete networks are held out in turn; no tuning.
"""
from pathlib import Path
import sys,json,hashlib
import numpy as np,pandas as pd
from sklearn.metrics import r2_score,mean_absolute_error
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT),str(ROOT/'RAPID/src'),str(Path(__file__).parent)]
from full_population_lag_search import enrich,FEATURES,model
from rapid_tools.refreshed_analysis import interpolated_peak
SOURCE=ROOT/'RAPID/outputs/cunge_fixed_comparison_20261006'
OUT=ROOT/'RAPID/outputs/cunge_fixed_extra_trees_20261006'
EXAMPLES=['02','03','05','21'];COLS=FEATURES['log_geometry'];assert len(COLS)==15

def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def score(y,p):
 mae=mean_absolute_error(y,p);zero=np.mean(abs(y))
 return dict(r2=float(r2_score(y,p)) if np.ptp(y)>0 else np.nan,mae=float(mae),zero_mae=float(zero),skill_vs_zero=float(1-mae/zero) if zero else np.nan)

def main():
 OUT.mkdir(parents=True,exist_ok=True)
 paths=[SOURCE/'paired_results.parquet',SOURCE/'provenance.json',SOURCE/'forcing_comparison.csv',Path(__file__),ROOT/'RAPID/investigations/full_population_lag_search.py']
 original=pd.read_parquet(SOURCE/'paired_results.parquet');d=enrich(original);meta=json.loads((SOURCE/'provenance.json').read_text());forcing=pd.read_csv(SOURCE/'forcing_comparison.csv',dtype={'example':str}).set_index('example')
 assert len(d)==624 and sorted(d.example_id.unique())==EXAMPLES and not d.duplicated(['example_id','fingerprint','method']).any()
 assert np.isfinite(d[COLS]).all().all()
 denominators=[]
 for src in meta['sources']:
  ex=src['example'];base=src['base_fingerprint'];root=Path(src['source']);fp=SOURCE/ex/(base+'.parquet');wp=root/'common_forcing.parquet';paths.extend([fp,wp]);w=pd.read_parquet(wp);q=pd.read_parquet(fp);t=q.time_seconds.to_numpy();inlet_peak=float(t[w.q_inflow_cms.to_numpy().argmax()])
  for method in ['cunge','fixed_calibrated','fixed_same_n']:
   tp,status=interpolated_peak(t,q[method].to_numpy());assert status=='ok';lag=(tp-inlet_peak)/3600;assert lag>0
   mask=d.example_id.eq(ex)&d.method.eq(method);d.loc[mask,'parent_lag_h']=lag;d.loc[mask,'relative_shift_pct']=100*d.loc[mask,'shift_h']/lag
   n=forcing.loc[ex,'fixed_calibrated_n'] if method=='fixed_calibrated' else forcing.loc[ex,'cunge_n'];d.loc[mask,'n_supplied']=n
   denominators.append(dict(example=ex,method=method,parent_interpolated_lag_h=lag,inlet_reference='first sampled forcing maximum',n_supplied=n))
  mask=d.example_id.eq(ex)&d.method.eq('fixed_archived');d.loc[mask,'relative_shift_pct']=d.loc[mask,'interpolated_lag_shift_pct'];d.loc[mask,'n_supplied']=forcing.loc[ex,'old_n']
  # Recover the exact archived interpolation denominator from its already
  # validated signed absolute and percentage responses, not a sampled lag.
  g=d[mask];valid=g.relative_shift_pct.abs()>1e-10;lags=100*g.loc[valid,'shift_h']/g.loc[valid,'relative_shift_pct'];assert len(lags)>0
  lag=float(lags.median());np.testing.assert_allclose(lags,lag,rtol=1e-8,atol=1e-8);d.loc[mask,'parent_lag_h']=lag
  np.testing.assert_allclose(100*g.shift_h/lag,g.relative_shift_pct,rtol=1e-8,atol=1e-8)
  denominators.append(dict(example=ex,method='fixed_archived',parent_interpolated_lag_h=lag,inlet_reference='archived first sampled forcing maximum',n_supplied=forcing.loc[ex,'old_n']))
 assert np.isfinite(d[['shift_h','relative_shift_pct','parent_lag_h','n_supplied']]).all().all()
 d[['example_id','fingerprint','state_id','method','shift_h','relative_shift_pct','parent_lag_h','n_supplied']+COLS].to_parquet(OUT/'matched_features_responses.parquet',index=False)
 pd.DataFrame(denominators).to_csv(OUT/'parent_lags_and_n.csv',index=False)
 initial_hashes={str(p):digest(p) for p in paths}
 predictions=[];folds=[]
 for method,g in d.groupby('method'):
  for response in ['shift_h','relative_shift_pct']:
   for features in ['geometry_15','geometry_15_plus_n']:
    for ex in EXAMPLES:
     train=g[g.example_id.ne(ex)];test=g[g.example_id.eq(ex)];assert len(train)+len(test)==156 and set(train.example_id).isdisjoint(test.example_id)
     cols=COLS.copy();use_n=features.endswith('plus_n') and train.n_supplied.nunique()>1
     # Constant archived n carries no information. Omit it so a useless
     # column cannot alter the Extra Trees random-number stream.
     if use_n:cols+=['n_supplied']
     fit=model('extra_trees');fit.fit(train[cols],train[response]);pred=fit.predict(test[cols])
     part=test[['example_id','fingerprint','state_id','parent_lag_h','n_supplied']].copy();part['routing']=method;part['response']=response;part['features']=features;part['observed']=test[response].to_numpy();part['predicted']=pred;part['training_mean']=float(train[response].mean());part['observed_shift_h']=test.shift_h.to_numpy();part['predicted_shift_h']=pred if response=='shift_h' else pred*test.parent_lag_h.to_numpy()/100;predictions.append(part)
     folds.append(dict(routing=method,response=response,features=features,held_out=ex,training_networks=','.join(sorted(train.example_id.unique())),n_train=len(train),n_test=len(test),n_column_used=use_n,n_training_min=float(train.n_supplied.min()),n_training_max=float(train.n_supplied.max()),n_test_value=float(test.n_supplied.iloc[0]),n_outside_training_range=bool(test.n_supplied.iloc[0]<train.n_supplied.min() or test.n_supplied.iloc[0]>train.n_supplied.max()),**score(test[response].to_numpy(),pred)))
 p=pd.concat(predictions,ignore_index=True);p.to_parquet(OUT/'held_out_predictions.parquet',index=False);pd.DataFrame(folds).to_csv(OUT/'fold_audit.csv',index=False)
 scores=[];back=[]
 for (routing,response,features),g in p.groupby(['routing','response','features']):
  for scope,gg in [('pooled',g)]+list(g.groupby('example_id')):
   scores.append(dict(routing=routing,response=response,features=features,scope=scope,n=len(gg),**score(gg.observed.to_numpy(),gg.predicted.to_numpy()),training_mean_mae=float(mean_absolute_error(gg.observed,gg.training_mean))))
   back.append(dict(routing=routing,trained_response=response,features=features,scope=scope,n=len(gg),**score(gg.observed_shift_h.to_numpy(),gg.predicted_shift_h.to_numpy())))
 s=pd.DataFrame(scores);s.to_csv(OUT/'scores.csv',index=False);pd.DataFrame(back).to_csv(OUT/'scores_in_hours.csv',index=False)
 macro=s[s.scope!='pooled'].groupby(['routing','response','features']).agg(equal_network_mae=('mae','mean'),equal_network_zero_mae=('zero_mae','mean'),equal_network_training_mean_mae=('training_mean_mae','mean')).reset_index();macro['skill_vs_zero']=1-macro.equal_network_mae/macro.equal_network_zero_mae;macro.to_csv(OUT/'equal_network_scores.csv',index=False)
 # Placebo: supplying a constant n must leave archived predictions identical.
 a=p[p.routing.eq('fixed_archived')].pivot(index=['example_id','fingerprint','response'],columns='features',values='predicted');np.testing.assert_array_equal(a.geometry_15,a.geometry_15_plus_n)
 assert all(digest(path)==v for path,v in initial_hashes.items())
 (OUT/'provenance.json').write_text(json.dumps(dict(source_hashes=initial_hashes,examples=EXAMPLES,pairs=156,geometry_features=COLS,estimator=model('extra_trees').get_params(),validation='Four leave-one-network-out folds, identical rows, no tuning; same previous Extra Trees settings. Constant n omitted for archived placebo.',relative_definition='100*(child interpolated peak - parent interpolated peak)/(parent interpolated peak - first sampled inlet peak). Method-specific known parent lag.',scope='Uses existing routed results only. No routing, calibration, target or production changes. Supplied n is already calibrated using that network gauge event; this tests response prediction conditional on known n, not ungauged estimation of n. Four independent n values cannot establish causal roughness effects.'),indent=2))
 print(s[s.scope=='pooled'].to_string(index=False));print('DONE: 64 fitted forests; source hashes unchanged; constant-n placebo exact.')
if __name__=='__main__':main()
