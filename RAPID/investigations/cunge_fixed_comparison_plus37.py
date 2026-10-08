"""Five-network extension: reuse four-network controls and add completed 37."""
from pathlib import Path
import sys,json,hashlib
import numpy as np,pandas as pd
from sklearn.metrics import r2_score,mean_absolute_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingRegressor
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT),str(ROOT/'RAPID/src'),str(Path(__file__).parent)]
from rapid_tools import cunge
from rapid_tools.k_values import compute_dt_bounds
from rapid_tools.refreshed_analysis import interpolated_peak,plateau_event_duration
from cunge_full_hierarchy_pilot import settling
from cunge_frequency_control import route_frequency
OUT=ROOT/'RAPID/outputs/cunge_fixed_comparison_plus37_20261006';OUT.mkdir(exist_ok=True,parents=True)
EXAMPLES=['02','03','05','21','37'] # frozen at analysis start; do not absorb running jobs
FEATURE_PATH=ROOT/'RAPID/validation/all_pairs_lag_20261005/features.parquet'
FEATURES=['omega','resistance_log','length_fraction','width_fraction','parent_equivalent_length','parent_elongation','parent_dominant_width_fraction','parent_path_length_cv','signed_length_width_association','domain_reference_slope','log_equivalent_length_ratio','log_elongation_ratio','change_path_length_cv','change_width_dominance','upstream_shortest_length_m','downstream_shortest_length_m']
SETS={'omega':['omega'],'components':['resistance_log','length_fraction','width_fraction'],'selected_six':['resistance_log','width_fraction','parent_equivalent_length','parent_dominant_width_fraction','signed_length_width_association','domain_reference_slope'],'selected_plus_context':['resistance_log','width_fraction','parent_equivalent_length','parent_dominant_width_fraction','signed_length_width_association','domain_reference_slope','upstream_shortest_length_m','downstream_shortest_length_m']}
Y=['shift_h','peak_excess_change_pct','event_duration_change_pct']

def metrics(q,flow,step,thresholds=None):
 t=np.arange(len(q))*step;tp,status=interpolated_peak(t,q)
 tol=thresholds if thresholds is not None else (.01*(q.max()-q[0]),.01*(q.max()-q[-1]))
 try:duration=plateau_event_duration(t,q,q[0],q[-1],*tol);ds='ok'
 except ValueError as e:duration=np.nan;ds=str(e)
 return dict(peak_s=tp,status=status,sampled_peak_s=int(q.argmax())*step,peak=float(q.max()),reference=float(max(q[0],q[-1])),duration_s=duration,duration_status=ds,tolerances=tol)

def run():
 f=pd.read_parquet(FEATURE_PATH);f=f[f.example_id.isin(EXAMPLES)].copy();assert not f.duplicated(['example_id','fingerprint']).any()
 cal=pd.read_csv(ROOT/'RAPID/outputs/cunge_calibration_alpha2/comparison.csv',dtype={'example':str})
 previous=ROOT/'RAPID/outputs/cunge_fixed_comparison_20261006'
 rows=pd.read_parquet(previous/'paired_results.parquet').to_dict('records')
 audit=pd.read_csv(previous/'routing_audit.csv',dtype={'example':str}).to_dict('records')
 provenance=json.loads((previous/'provenance.json').read_text())['sources']
 forcing_audit=pd.read_csv(previous/'forcing_comparison.csv',dtype={'example':str}).to_dict('records')
 for ex in ['02','03','05','21']:
  target=OUT/ex
  if not target.exists():target.symlink_to(previous/ex,target_is_directory=True)
 for ex in ['37']:
  p=ROOT/'RAPID/outputs/cunge_full_hierarchy_pilot'/ex
  if not (p/'completion.json').exists():p=Path('/Volumes/PhD/river_hierarchy/cunge_full_hierarchy_runs')/ex
  assert (p/'completion.json').exists()
  m=json.loads((p/'experiment.json').read_text());reg=pd.read_csv(p/'state_run_registry.csv').drop_duplicates('fingerprint');g=f[f.example_id.eq(ex)]
  assert set(g.fingerprint)==set(reg.fingerprint)-{m['base_fingerprint']}
  wave=pd.read_parquet(p/'common_forcing.parquet');flow=wave.q_inflow_cms.to_numpy();step=m['forcing_step_s'];nfixed=float(cal[(cal.example==ex)&(cal.method=='fixed_x')].iloc[0].n_manning)
  frames={};upper=[]
  for r in reg.itertuples():
   par=pd.read_parquet(Path(r.output_dir)/'parameters.parquet')
   variants={'cunge':par,'fixed_calibrated':cunge.parameters(par,nfixed,method='fixed_x'),'fixed_same_n':cunge.parameters(par,m['n_manning'],method='fixed_x')}
   frames[r.fingerprint]=variants
   upper.extend(compute_dt_bounds(v.rapid_k,x_value=v.rapid_x,coefficient_policy='signed_inflow')[1] for v in variants.values())
  ns=int(np.ceil(step/min(m['routing_dt_s'],.99*min(upper))));dt=step/ns
  allmetrics={};dest=OUT/ex;dest.mkdir(exist_ok=True)
  for number,fp in enumerate([m['base_fingerprint']]+g.fingerprint.tolist()):
   fn=dest/(fp+'.parquet');cols={};info=[]
   for method,par in frames[fp].items():
    extra={}
    if method=='cunge':
     assert np.isclose(dt,m['routing_dt_s'],rtol=0,atol=1e-12)
     q=pd.read_parquet(p/'states'/fp/'outlet_hydrograph.parquet').q_outlet_cms.to_numpy()
     saved_audit=json.loads((p/'states'/fp/'result.json').read_text())
     a={'minimum_native_flow_cms':saved_audit['minimum_substep_flow_cms']}
     extra['evaluation']='existing_full_engine'
    else:
     q,fa=route_frequency(par,flow,ns,dt,padding=4)
     doubled,fb=route_frequency(par,flow,ns,dt,padding=8)
     padding_error=float(np.max(abs(q-doubled)));assert padding_error<1e-6
     a={'minimum_native_flow_cms':np.nan}
     extra.update(evaluation='frequency_of_native_recurrence',minimum_reported_outlet_cms=float(q.min()),padding_error_cms=padding_error,**fa)
     native=OUT/'37_native_reference'/(fp+'.parquet')
     if native.exists():
      reference=pd.read_parquet(native)[method].to_numpy();error=float(np.max(abs(q-reference)));assert error<1e-6
      extra['native_reference_max_error_cms']=error
      extra['native_reference_peak_bin_shift']=int(q.argmax()-reference.argmax())
    assert np.isfinite(q).all() and q.min()>=0
    assert settling(flow,q,m['event_last_seconds'],step)
    thresholds=None if fp==m['base_fingerprint'] else allmetrics[(m['base_fingerprint'],method)]['tolerances']
    met=metrics(q,flow,step,thresholds);allmetrics[(fp,method)]=met;cols[method]=q
    entry=dict(example=ex,fingerprint=fp,method=method,minimum_native_flow_cms=a['minimum_native_flow_cms'],tail_settled=True,dt_s=dt,peak_status=met['status'],duration_status=met['duration_status'])
    entry.update(extra)
    if method=='cunge':
     saved=pd.read_parquet(p/'states'/fp/'outlet_hydrograph.parquet').q_outlet_cms.to_numpy();entry['max_clock_change_cms']=float(np.max(abs(q-saved)));entry['saved_peak_shift_s']=(q.argmax()-saved.argmax())*step
    audit.append(entry)
   pd.DataFrame(dict(time_seconds=np.arange(len(flow))*step,**cols)).to_parquet(fn,index=False)
   if number%20==0:print(ex,number+1,'/',len(g)+1,'states',flush=True)
  for r in g.itertuples():
   base=r._asdict();base.pop('Index',None)
   for method in ['cunge','fixed_calibrated','fixed_same_n']:
    a=allmetrics[(m['base_fingerprint'],method)];b=allmetrics[(r.fingerprint,method)]
    row=dict(base);row.update(method=method,shift_h=(b['peak_s']-a['peak_s'])/3600,sampled_shift_h=(b['sampled_peak_s']-a['sampled_peak_s'])/3600,peak_excess_change_pct=100*(b['peak']-a['peak'])/(a['peak']-a['reference']),event_duration_change_pct=100*(b['duration_s']/a['duration_s']-1));rows.append(row)
   row=dict(base);row.update(method='fixed_archived',shift_h=r.interpolated_peak_shift_hours,sampled_shift_h=r.peak_time_shift_hours);rows.append(row)
  old=pd.read_csv(Path(g.iloc[0].refreshed_run_dir)/'outlet_hydrograph.csv');oldq=old.q_inflow_cms.to_numpy();offset=oldq.argmax()-flow.argmax();j=np.arange(len(flow));valid=(j+offset>=0)&(j+offset<len(oldq));oldpolicy=json.loads((Path(g.iloc[0].refreshed_run_dir).parent/'policy_manifest.json').read_text())
  forcing_audit.append(dict(example=ex,old_time_origin=old.time_utc.iloc[0],new_time_origin=wave.time_utc.iloc[0],old_initial=oldq[0],new_initial=flow[0],old_peak=oldq.max(),new_peak=flow.max(),max_peak_aligned_overlap_difference_cms=float(np.max(abs(oldq[j[valid]+offset]-flow[valid]))),old_n=oldpolicy['prep_config']['n_manning'],fixed_calibrated_n=nfixed,cunge_n=m['n_manning'],old_reporting_step_s=float(old.time_seconds.iloc[1]-old.time_seconds.iloc[0]),new_reporting_step_s=step))
  provenance.append(dict(example=ex,source=str(p),base_fingerprint=m['base_fingerprint'],distinct_pairs=len(g),common_dt_s=dt,n_fixed=nfixed,n_cunge=m['n_manning'],source_completion_sha256=hashlib.sha256((p/'completion.json').read_bytes()).hexdigest()))
  pd.DataFrame(rows).to_parquet(OUT/'paired_results.parquet',index=False);pd.DataFrame(audit).to_csv(OUT/'routing_audit.csv',index=False);print(ex,'done',len(g),flush=True)
 pd.DataFrame(forcing_audit).to_csv(OUT/'forcing_comparison.csv',index=False)
 (OUT/'provenance.json').write_text(json.dumps(dict(sources=provenance,feature_source=str(FEATURE_PATH),feature_sha256=hashlib.sha256(FEATURE_PATH.read_bytes()).hexdigest(),script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),scope='Frozen completed 02/03/05/21/37; first four controls reused; archived old workflow plus common-event/common-grid fixed-X controls. First-four controls reused; 37 fixed controls use frequency response of native recurrence with bounded alias error, doubled padding and partial direct-native comparisons. Native positivity for all 37 fixed-control reaches was not re-audited; reported outlet positivity checked. Cunge uses existing full-engine audited output. No recalibration or production edits.',predictors=SETS),indent=2))

 def score(obs,pred):return dict(r2=float(r2_score(obs,pred)) if np.ptp(obs)>0 else np.nan,mae=float(mean_absolute_error(obs,pred)),zero_mae=float(abs(obs).mean()),skill_vs_zero=float(1-mean_absolute_error(obs,pred)/abs(obs).mean()) if abs(obs).mean()>0 else np.nan)
 d=pd.DataFrame(rows);corr=[];preds=[];scores=[];validity=[]
 for y in Y:
  eligible=d.groupby(['example_id','fingerprint'])[y].apply(lambda z:len(z)==4 and np.isfinite(z).all());keys=set(eligible[eligible].index);v=d[[tuple(z) in keys for z in d[['example_id','fingerprint']].to_numpy()]].copy();validity.append(dict(response=y,eligible_pairs=len(keys),excluded_pairs=len(f)-len(keys)))
  for method,z in v.groupby('method'):
   for name,gg in [('pooled',z)]+list(z.groupby('example_id')):
    for feature in FEATURES:corr.append(dict(method=method,response=y,example=name,feature=feature,n=len(gg),spearman=gg[feature].corr(gg[y],method='spearman') if gg[feature].nunique()>1 and gg[y].nunique()>1 else np.nan))
   for label,cols in SETS.items():
    for kind in ['ridge_10','boosted_7']:
     ps=[]
     for ex in EXAMPLES:
      train=z[z.example_id.ne(ex)];test=z[z.example_id.eq(ex)];fit=make_pipeline(StandardScaler(),Ridge(alpha=10)) if kind=='ridge_10' else HistGradientBoostingRegressor(max_iter=80,max_leaf_nodes=7,min_samples_leaf=20,l2_regularization=1,early_stopping=False,random_state=20261006)
      fit.fit(train[cols],train[y]);part=test[['example_id','fingerprint','state_id']].copy();part['observed']=test[y].to_numpy();part['predicted']=fit.predict(test[cols]);part['training_mean']=float(train[y].mean());ps.append(part)
     ps=pd.concat(ps,ignore_index=True);ps['method']=method;ps['response']=y;ps['feature_set']=label;ps['model']=kind;preds.append(ps)
     for ex,gg in [('pooled',ps)]+list(ps.groupby('example_id')):scores.append(dict(method=method,response=y,feature_set=label,model=kind,example=ex,n=len(gg),**score(gg.observed.to_numpy(),gg.predicted.to_numpy()),mean_baseline_mae=float(mean_absolute_error(gg.observed,gg.training_mean))))
 pd.DataFrame(corr).to_csv(OUT/'correlations.csv',index=False);pd.DataFrame(scores).to_csv(OUT/'prediction_scores.csv',index=False);pd.concat(preds).to_parquet(OUT/'held_out_predictions.parquet',index=False);pd.DataFrame(validity).to_csv(OUT/'response_coverage.csv',index=False)
 print('DONE',flush=True)
if __name__=='__main__':run()
