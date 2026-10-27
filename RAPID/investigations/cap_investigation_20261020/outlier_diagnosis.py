"""Consolidate frozen-model outlier evidence; audit active K/X without new runs."""
from pathlib import Path
import json,hashlib
import numpy as np
import pandas as pd
from slope_ratio_audit import ROOT,OUT,EXT,read,SOURCES
DEST=OUT/'outlier_diagnosis_20261023'
COLS=['log1p__B_ratio_route_length_mean','domain_size','log1p__B_ratio_route_length_sd','log1p__B_ratio_width_mean','log1p__B_ratio_width_cv','log1p__B_ratio_route_entropy','B_parent_dominance_mean','parent_slope_slow','length_delta_km']
def main():
 DEST.mkdir(exist_ok=True)
 d=read(OUT/'slope_ratio_audit_20261023/input_and_residuals.parquet');assert len(d)==2185 and d.example.nunique()==15
 rows=[];summary=[]
 for cohort,g in d.groupby('cohort'):
  g=g.copy();w=1/g.groupby('example').example.transform('size')
  g['relative_error']=g.absolute_error_h/g.centroid_shift_hours.abs().clip(lower=1/60)
  g['relative_share_pct']=100*w*g.relative_error/(w*g.relative_error).sum()
  for metric in ['sse_share_pct','relative_share_pct']:
   ordered=g.sort_values(metric,ascending=False)
   summary.append(dict(cohort=cohort,metric=metric,n=len(g),top5_share=ordered[metric].head(5).sum(),top10_share=ordered[metric].head(10).sum(),cases_to_80pct=int((ordered[metric].cumsum()<80).sum()+1)))
  g['selected_sse']=g.index.isin(g.nlargest(8,'sse_share_pct').index)
  g['selected_relative']=g.index.isin(g.nlargest(8,'relative_share_pct').index)
  rows.append(g)
 allrows=pd.concat(rows);allrows.to_parquet(DEST/'ranked_population.parquet',index=False)
 pd.DataFrame(summary).to_csv(DEST/'error_concentration.csv',index=False)
 chosen=allrows[allrows.selected_sse|allrows.selected_relative].copy()
 jobs={}
 for plan,runroot in [(ROOT/'RAPID/outputs/event_window_plan_20261013/plan.json',ROOT/'RAPID/outputs/event_window_runs_20261013'),(EXT/'routing_plan.json',EXT/'runs'),(OUT/'routing_plan.json',OUT/'runs')]:
  for case in read(plan)['cases']:
   for job in case['jobs']:
    for name in set([job['state_id'],*job.get('aliases',[])]):jobs[case['example'],name]=(job,runroot,case['source_sha256'])
 support=[];routing=[];receipts=[]
 for r in chosen.itertuples():
  tr=d[d.example.ne(r.example)];lo,hi=tr.centroid_shift_hours.min(),tr.centroid_shift_hours.max()
  outside=[c for c in COLS if getattr(r,c)<tr[c].min() or getattr(r,c)>tr[c].max()]
  extreme=tr[tr.centroid_shift_hours.le(r.centroid_shift_hours)] if r.centroid_shift_hours<0 else tr[tr.centroid_shift_hours.ge(r.centroid_shift_hours)]
  support.append(dict(example=r.example,state=r.state,outside_predictors=' | '.join(outside),training_target_min=lo,training_target_max=hi,forest_range_error_lower_bound_h=max(lo-r.centroid_shift_hours,r.centroid_shift_hours-hi,0),equally_extreme_training_rows=len(extreme),equally_extreme_training_networks=extreme.example.nunique()))
  for side,state in [('parent','S000_base'),('child',r.state)]:
   job,runroot,hashes=jobs[r.example,state];f=read(job['parameters']);assert SOURCES[job['parameters']]==hashes[job['parameters']]
   dt=job['native_dt_seconds'];K=f.rapid_k.to_numpy();X=f.rapid_x.to_numpy();c=f.rapid_celerity_mps.to_numpy();dx=f.rapid_effective_length_m.to_numpy()
   expectedX=.5-f.diffusion_multiplier.to_numpy()*f.reference_discharge_cms.to_numpy()/(2*f.width.to_numpy()*f.slope.to_numpy()*c*dx)
   assert f.routing_method.eq('cunge').all();np.testing.assert_allclose(K,dx/c,rtol=1e-10,atol=1e-9);np.testing.assert_allclose(X,expectedX,rtol=1e-10,atol=1e-9)
   previous=(2*K*(1-X)-dt)/(dt+2*K*(1-X))
   w=f.reference_flow_fraction*f.link_length_m
   route_h=float((f.reference_flow_fraction*f['count']*f.rapid_k).sum()/3600)
   routing.append(dict(example=r.example,state=r.state,side=side,n_links=len(f),dt_seconds=dt,K_min_seconds=K.min(),K_max_seconds=K.max(),X_min=X.min(),X_max=X.max(),negative_X_links=int((X<0).sum()),min_previous_outflow_coefficient=previous.min(),invalid_K_or_X=int(((K<=0)|~np.isfinite(K)|~np.isfinite(X)|(X>.5)).sum()),unstable_previous_outflow_links=int((previous<-1e-10).sum()),celerity_capped_links=int(f.rapid_celerity_capped.sum()),length_floor_links=int(f.rapid_length_floor_applied.sum()),full_route_time_hours=route_h,effective_celerity_mps=float(w.sum()/(route_h*3600)),n_manning=float(f.n_manning.iloc[0]),beta=float(f.diffusion_multiplier.iloc[0])))
   complete=read(runroot/r.example/'completion.json');receipt=read(Path(complete['final_directory'])/job['state_id']/'completion.json')
   receipts.append(dict(example=r.example,state=r.state,side=side,path=str(Path(complete['final_directory'])/job['state_id']/'completion.json'),complete=receipt['complete'],settled=receipt['settled'],minimum_native_flow_cms=receipt['audit']['minimum_native_flow_cms'],final_error_cms=receipt['final_96h_max_error_cms'],tolerance_cms=receipt['tolerance_cms']))
  print(r.example,r.state,'audited',flush=True)
 s=pd.DataFrame(support);chosen=chosen.merge(s,on=['example','state'],validate='one_to_one')
 physics=pd.DataFrame(routing);checks=pd.DataFrame(receipts)
 assert checks.complete.all() and checks.settled.all() and checks.minimum_native_flow_cms.ge(-1e-9).all()
 assert physics.invalid_K_or_X.eq(0).all() and physics.unstable_previous_outflow_links.eq(0).all()
 physics.to_csv(DEST/'routing_parameters.csv',index=False);checks.to_csv(DEST/'routing_receipts.csv',index=False)
 checks_old=read(OUT/'nine_error_audit/targeted_training_checks.csv')
 checks_old=checks_old.pivot(index=['example','state'],columns='variant',values='predicted').reset_index()
 chosen=chosen.merge(checks_old,on=['example','state'],how='left',validate='one_to_one')
 mask=chosen.baseline.notna();np.testing.assert_allclose(chosen.loc[mask,'baseline'],chosen.loc[mask,'predicted_hours'],atol=1e-12,rtol=1e-12)
 for variant in ['no_bootstrap','cohort_balanced']:
  chosen[variant+'_abs_error_improvement_h']=chosen.absolute_error_h-abs(chosen[variant]-chosen.centroid_shift_hours)
 chosen['response_within_1minute']=chosen.centroid_shift_hours.abs().le(1/60)
 chosen.to_csv(DEST/'priority_cases.csv',index=False)
 for path,h in SOURCES.items():assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==h
 (DEST/'manifest.json').write_text(json.dumps(dict(sources=SOURCES,code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),population=len(d),priority_cases=len(chosen),selection='Union of eight greatest contributions to network-weighted squared error and eight greatest contributions to network-weighted one-minute-floored relative error, separately in regular and segmented cohorts.',model_checks='Reuse prior nine-input targeted no-bootstrap/cohort-balanced diagnostics, verified against current baseline. No new fits.',routing='Active K and X algebra, positive finite K, signed-inflow previous-outflow stability at actual native dt, clipping flags, existing settled/nonnegative-flow receipts. No routing perturbations; negative X is not labelled a bug. Receipt audits are read, not newly routed or independently recomputed hydrographs.',no_production_changes=True),indent=2)+'\n')
 print(pd.DataFrame(summary).to_string(index=False))
if __name__=='__main__':main()
