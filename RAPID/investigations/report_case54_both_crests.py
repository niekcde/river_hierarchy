"""Versioned two-crest report for every completed case-54 state; no rerouting files."""
from pathlib import Path
import sys,json
import numpy as np,pandas as pd
import plotly.graph_objects as go
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT),str(ROOT/'RAPID/src')]
from rapid_tools import cunge
OUT=ROOT/'RAPID/outputs/temporal_and_target_audit_20261008';run=Path('/Volumes/PhD/river_hierarchy/cunge_full_hierarchy_runs/54');m=json.loads((run/'experiment.json').read_text());reg=pd.read_csv(run/'state_run_registry.csv');pairs=pd.read_csv(run/'parent_child_comparison.csv');records=[]
for rr in reg.drop_duplicates('fingerprint').itertuples():
 p=run/'states'/rr.fingerprint;h=pd.read_parquet(p/'outlet_hydrograph.parquet');f=pd.read_parquet(p/'parameters.parquet');q,a=cunge.route(f,h.q_inflow_cms.to_numpy(),m['nsub'],m['routing_dt_s'],return_native=True);t=np.arange(1,len(q)+1)*m['routing_dt_s'];start=pd.Timestamp(h.time_utc.iloc[0]);d=dict(state=rr.state,fingerprint=rr.fingerprint,native_mean_match_max_cms=float(np.max(abs(q.reshape(-1,m['nsub']).mean(axis=1)-h.q_outlet_cms.to_numpy()))))
 for k,stamp in [(1,'2023-06-22T14:04:30Z'),(2,'2023-06-23T00:45:00Z')]:
  ids=np.flatnonzero(abs(t-(pd.Timestamp(stamp)-start).total_seconds())<4*3600);i=int(ids[q[ids].argmax()]);valid=bool(ids[0]<i<ids[-1]);d.update({f'crest{k}_time_seconds':float(t[i]) if valid else np.nan,f'crest{k}_utc':str(start+pd.to_timedelta(t[i],unit='s')) if valid else None,f'crest{k}_q_cms':float(q[i]) if valid else np.nan,f'crest{k}_interior_peak':valid})
 d['global_crest']=1 if d['crest1_q_cms']>=d['crest2_q_cms'] else 2;d['global_peak_inside_declared_windows']=bool(min(abs(t[q.argmax()]-d[f'crest{k}_time_seconds']) for k in [1,2])<=m['routing_dt_s']);records.append(d)
 if len(records)%20==0:print('states',len(records),flush=True)
a=pd.DataFrame(records).set_index('fingerprint');a.to_csv(OUT/'54_state_crest_metrics.csv');out=[]
for rr in pairs.itertuples():
 p=a.loc[rr.parent_fingerprint];c=a.loc[rr.child_fingerprint];d=dict(family=rr.family,parent_state=rr.parent_state,child_state=rr.child_state,parent_fingerprint=rr.parent_fingerprint,child_fingerprint=rr.child_fingerprint,legacy_global_delta_h=rr.delta_lag_h,main_crest=1,main_selection_status='provisional_first_crest_user_assignment_pending',parent_global_crest=int(p.global_crest),child_global_crest=int(c.global_crest),crest_switch=bool(p.global_crest!=c.global_crest))
 for k in [1,2]:
  d[f'crest{k}_delta_minutes']=(c[f'crest{k}_time_seconds']-p[f'crest{k}_time_seconds'])/60;d[f'parent_crest{k}_utc']=p[f'crest{k}_utc'];d[f'child_crest{k}_utc']=c[f'crest{k}_utc'];d[f'parent_crest{k}_q_cms']=p[f'crest{k}_q_cms'];d[f'child_crest{k}_q_cms']=c[f'crest{k}_q_cms']
 d['main_delta_peak_hours']=d['crest1_delta_minutes']/60;d['two_crest_midpoint_delta_minutes']=(d['crest1_delta_minutes']+d['crest2_delta_minutes'])/2;out.append(d)
b=pd.DataFrame(out);b.to_csv(OUT/'54_main_and_secondary_results.csv',index=False);u=b.drop_duplicates(['parent_fingerprint','child_fingerprint']);u.to_csv(OUT/'54_unique_main_and_secondary_results.csv',index=False)
fig=go.Figure()
for k,col in [(1,'#087e8b'),(2,'#cb3d3d')]:fig.add_scatter(x=u.child_state,y=u[f'crest{k}_delta_minutes'],mode='markers',name=f'Crest {k}'+(' — provisional main' if k==1 else ' — secondary'),marker=dict(color=col,size=7))
fig.update_layout(title='54: corresponding crest timing shifts across all unique transitions<br><sup>Native routing times; main = first crest provisionally. Global maximum switches remain separately flagged.</sup>',template='plotly_white',height=650,margin=dict(t=110,b=190),legend=dict(orientation='h',y=1.10),xaxis_title='Child state',yaxis_title='Child minus parent timing (minutes)');fig.update_xaxes(tickangle=-70);fig.write_html(OUT/'54_two_crest_timing.html',include_plotlyjs=True)
(OUT/'main_crest_selection.json').write_text(json.dumps(dict(example='54',main_crest=1,status='provisional_pending_user_assignment',rationale='First chronological crest; held fixed across parents and children rather than reselecting by height.',secondary_crest=2,search_half_window_hours=4,reference_crest_utc=['2023-06-22T14:04:30Z','2023-06-23T00:45:00Z'],calibration_changed=False,other_examples='No main-crest assignments made; await user choices.'),indent=2));print('done',len(a),'states',len(u),'unique pairs','valid',bool(a.crest1_interior_peak.all() and a.crest2_interior_peak.all()),flush=True)
