"""Small post-fit antecedent test, contrasting case refits, draft height review."""
from pathlib import Path
import sys,json,hashlib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'RAPID/src')]
from rapid_tools import cunge
from rapid_tools.common_grid import route_interval_means
from rapid_tools.time_support import point_series_interval_means
DEST=Path(json.loads((ROOT/'RAPID/outputs/common_grid_validation_20261008/latest.json').read_text())['directory'])
OUT=DEST/'beta_followups';OUT.mkdir(exist_ok=True)
FREEZE=DEST/'baseline'
COMP=pd.read_csv(FREEZE/'RAPID/outputs/cunge_calibration_reviewed_20261006/comparison.csv',dtype={'example':str})
TARGETS={r['example']:r for r in json.loads((FREEZE/'RAPID/reviews/calibration_targets_20261006/targets.json').read_text())['records']}
def inputs(ex):
 r=COMP[(COMP.example==ex)&(COMP.method=='cunge')].iloc[0]
 folder=FREEZE/Path(r.output_dir).relative_to(ROOT)
 return pd.read_parquet(folder/'frozen_grid.parquet'),pd.read_parquet(DEST/ex/'60s.parquet'),next(m for m in json.loads((DEST/ex/'60s.json').read_text()) if m['method']=='cunge'),folder

def antecedent():
 grid,event,meta,_=inputs('46')
 fit=json.loads((DEST/'46_diffusion_multiplier/beta025_refit.json').read_text())['selected']
 f=cunge.parameters(grid,fit['n_manning'],method='cunge');f['rapid_x']=.5-.25*(.5-f.rapid_x)
 base=pd.read_parquet(DEST/'46_diffusion_multiplier/beta025_refitted_n_waves.parquet').beta025_refitted_n_cms.to_numpy()
 waves=event[['time_utc']].copy();waves['equilibrium_cms']=base
 valid=event.downstream_cms.notna().to_numpy();ids=np.flatnonzero(valid);target=pd.Timestamp(meta['target_utc'])
 crest=valid&(abs(event.time_utc-target)<=pd.Timedelta(hours=6)).to_numpy()
 records=[]
 for days in [0,7,30,90]:
  if days==0:q=base
  else:
   points=pd.read_parquet(DEST/'46'/f'antecedent_{days}d_points.parquet')
   origin=event.interval_start_utc.iloc[0]-pd.Timedelta(days=days)
   t=(pd.to_datetime(points.time,utc=True)-origin).dt.total_seconds().to_numpy();v=points.discharge.to_numpy()
   edges=np.arange(len(event)+days*1440+1)*60.
   flow=point_series_interval_means(t,v,edges,allow_endpoint_hold=True)
   # Overwrite the event segment with identical accepted pilot forcing explicitly.
   changed=np.flatnonzero(~np.isclose(flow[days*1440:],event.forcing_cms,rtol=1e-10,atol=1e-6))
   assert np.all(changed < 10), 'Unexpected forcing mismatch beyond initial endpoint extension'
   print('event boundary hold preserved:',days,'days;',len(changed),'initial minute bins',flush=True)
   flow[days*1440:]=event.forcing_cms
   qfull,audit=route_interval_means(f,flow,6,10.,initial_flow=float(np.interp(0,t,v)))
   assert audit['minimum_native_flow_cms']>=0
   q=qfull[days*1440:]
  waves[f'antecedent_{days}d_cms']=q
  pk=ids[np.argmax(q[ids])]
  records.append(dict(days=days,n_manning=fit['n_manning'],beta=.25,peak_cms=q[pk],peak_error_minutes=(event.time_utc.iloc[pk]-target).total_seconds()/60,max_event_difference_from_equilibrium_cms=np.max(abs(q[valid]-base[valid])),max_crest_difference_from_equilibrium_cms=np.max(abs(q[crest]-base[crest])),event_start_output_cms=q[0]))
  print('antecedent',records[-1],flush=True)
 waves.to_parquet(OUT/'46_antecedent_waves.parquet',index=False)
 pd.DataFrame(records).to_csv(OUT/'46_antecedent_metrics.csv',index=False)
 fig=go.Figure()
 q0=meta['initial_cms'];shift=q0-event.downstream_cms[valid].iloc[0]
 for days in [0,7,30,90]:
  q=waves[f'antecedent_{days}d_cms']
  fig.add_trace(go.Scatter(x=event.time_utc[valid],y=q[valid]/q0,name='Equilibrium' if days==0 else f'{days} days actual antecedent',customdata=q[valid],hovertemplate='%{x}<br>Q=%{customdata:.3f} m³/s<extra>%{fullData.name}</extra>'))
 fig.add_trace(go.Scatter(x=event.time_utc[valid],y=(event.downstream_cms[valid]+shift)/q0,name='Downstream — shifted',line=dict(color='black')))
 fig.add_vline(x=target.timestamp()*1000,line_dash='dot',line_color='purple')
 fig.update_layout(title='46: antecedent test at fixed β=0.25 and n=0.0162109375 — no refit',template='plotly_white',height=750,yaxis_title='Q / original initial forcing; downstream shifted',xaxis_title='UTC')
 fig.write_html(OUT/'46_antecedent_check.html',include_plotlyjs=True)
 print('30 versus 90 event/crest differences',np.max(abs(waves.antecedent_30d_cms[valid]-waves.antecedent_90d_cms[valid])),np.max(abs(waves.antecedent_30d_cms[crest]-waves.antecedent_90d_cms[crest])),flush=True)

def other_cases():
 examples=['02','05','55','56'];records=[];trials=[]
 fig=make_subplots(rows=2,cols=2,subplot_titles=examples,vertical_spacing=.15)
 for pos,ex in enumerate(examples):
  grid,data,meta,_=inputs(ex);valid=data.downstream_cms.notna().to_numpy();ids=np.flatnonzero(valid);obs=data.downstream_cms.to_numpy();target=pd.Timestamp(TARGETS[ex]['target_utc']);q0=meta['initial_cms'];dt=meta['dt_s']
  output=data[['time_utc']].copy();plotcurves=[('Original Cunge',data.cunge_cms.to_numpy(),'#ce3d45')]
  for beta in [1.,.25]:
   cache={}
   def run(n):
    if n in cache:return cache[n]
    f=cunge.parameters(grid,n,method='cunge');f['rapid_x']=.5-beta*(.5-f.rapid_x)
    q,a=route_interval_means(f,data.forcing_cms.to_numpy(),int(round(60/dt)),dt,initial_flow=q0)
    assert a['minimum_native_flow_cms']>=0
    pk=ids[np.argmax(q[ids])];e=(data.time_utc.iloc[pk]-target).total_seconds()/60
    cache[n]=(e,q)
    trials.append(dict(example=ex,beta=beta,n=n,peak_error_minutes=e,peak_cms=q[pk]))
    return e,q
   ns=np.unique(np.r_[np.linspace(grid.grid_n_min.iloc[0],grid.grid_n_max.iloc[0],9),meta['n_manning']])
   for n in ns:run(float(n))
   for left,right in zip(ns[:-1],ns[1:]):
    left,right=float(left),float(right)
    if run(left)[0]*run(right)[0]>=0:continue
    for _ in range(18):
     mid=(left+right)/2;e,q=run(mid)
     if abs(e)<=.5:break
     if run(left)[0]*e<=0:right=mid
     else:left=mid
   best=min(cache,key=lambda n:abs(cache[n][0]));err,q=cache[best];pk=ids[np.argmax(q[ids])]
   qdown=float(np.max(obs[ids]));qin=float(data.upstream_cms[valid].max())
   rec=dict(example=ex,beta=beta,n_manning=best,peak_error_minutes=err,peak_cms=q[pk],observed_peak_cms=qdown,upstream_peak_cms=qin,height_error_pct=100*(q[pk]/qdown-1),raw_supported_rmse_cms=np.sqrt(np.mean((q[valid]-obs[valid])**2)),timing_fit_reached_one_minute=abs(err)<=1.)
   records.append(rec);output[f'beta_{beta:g}_refitted_cms']=q
   plotcurves.append((f'β={beta:g}, n refitted',q,'#dc8a16' if beta==1 else '#25965a'))
   print('other',rec,flush=True)
  shift=q0-obs[ids[0]]
  plotcurves+=[('Upstream',data.upstream_cms.to_numpy(),'#2580a5'),('Downstream — shifted',obs+shift,'#17212b')]
  for name,q,color in plotcurves:
   fig.add_trace(go.Scatter(x=data.time_utc[valid],y=q[valid]/q0,name=name,legendgroup=name,showlegend=pos==0,line=dict(color=color,width=1.5)),row=pos//2+1,col=pos%2+1)
  fig.add_vline(x=target.timestamp()*1000,line_color='purple',line_dash='dot',row=pos//2+1,col=pos%2+1)
  output.to_parquet(OUT/f'{ex}_refit_waves.parquet',index=False)
  pd.DataFrame(records).to_csv(OUT/'other_case_metrics.csv',index=False)
  pd.DataFrame(trials).to_csv(OUT/'other_case_search_trials.csv',index=False)
 fig.update_layout(title='Contrasting cases: β=1 versus β=0.25, n fitted to timing separately<br><sup>Exploratory equilibrium-start runs; height not used for fitting. 55 has hourly crest observations.</sup>',height=1100,template='plotly_white',legend=dict(orientation='h',y=1.08))
 fig.write_html(OUT/'other_cases_beta_comparison.html',include_plotlyjs=True)

def height_review():
 review=[]
 for ex in sorted(set(COMP.example)-{'11'}):
  _,data,meta,folder=inputs(ex);target=TARGETS[ex]
  valid=data.downstream_cms.notna();q0=meta['initial_cms'];down0=float(data.downstream_cms[valid].iloc[0]);upmax=float(data.upstream_cms[valid].max());downmax=float(data.downstream_cms[valid].max())
  raw=pd.read_parquet(folder/'observed_downstream.parquet');ts=pd.to_datetime(raw.time,utc=True)
  selected=pd.Timestamp(target['target_utc']);crest=raw[(ts>=pd.Timestamp(target['crest_start_utc']))&(ts<=pd.Timestamp(target['crest_end_utc']))]
  qt=float(np.interp(selected.value,ts.astype('int64').to_numpy(),raw.discharge_cms))
  notes=['Draft only: no height bounds approved. Observed crest min/max describe the chosen feature, not measurement uncertainty.']
  if downmax>upmax:notes.append('Downstream maximum exceeds upstream maximum in the common observation window: check lateral inputs, gauging comparability and full antecedent range before interpreting attenuation.')
  if target['selection_rule']!='retain':notes.append('Feature-specific height definition required; do not automatically replace selected crest/plateau by a global maximum.')
  if ex=='55':notes.append('Hourly source sampling at crest; one-minute interpolation does not establish minute-scale observational precision.')
  if ex in ['21','22']:notes.append('Source clock ambiguity unresolved.')
  review.append(dict(example=ex,time_target_utc=target['target_utc'],feature_rule=target['selection_rule'],feature_start_utc=target['crest_start_utc'],feature_end_utc=target['crest_end_utc'],initial_forcing_cms=q0,initial_downstream_mean_cms=down0,upstream_observed_window_peak_cms=upmax,downstream_observed_window_peak_cms=downmax,downstream_point_reconstruction_at_target_cms=qt,selected_raw_crest_min_cms=None if crest.empty else float(crest.discharge_cms.min()),selected_raw_crest_max_cms=None if crest.empty else float(crest.discharge_cms.max()),downstream_over_upstream_peak=downmax/upmax,downstream_over_upstream_excursion=(downmax-down0)/(upmax-q0) if upmax>q0 else None,proposed_target_basis='downstream observed selected feature; subject to comparability review',accepted_height_low_cms=None,accepted_height_high_cms=None,height_weight=None,status='needs_manual_review',notes=notes))
 (OUT/'height_target_review.json').write_text(json.dumps(dict(status='DRAFT: evidence populated; no invented uncertainty bands',policy='Downstream observation is the target when its boundary conditions are represented. Upstream is the forcing/reference, not an assumed downstream peak. Consider baseline-relative excursion only under an explicitly justified baseline treatment. Do not rescale observed amplitude to force agreement.',records=review),indent=2))
 frame=pd.DataFrame(review)
 cols=['example','feature_rule','upstream_observed_window_peak_cms','downstream_observed_window_peak_cms','downstream_over_upstream_peak','downstream_over_upstream_excursion','selected_raw_crest_min_cms','selected_raw_crest_max_cms','status']
 page='<html><meta charset="utf-8"><title>Draft height targets</title><style>body{font:15px Arial;margin:30px}td,th{padding:8px}table{border-collapse:collapse}</style><h1>Draft height-target review</h1><p>Observed evidence only. No accepted height ranges or weights yet. Crest ranges are not uncertainty bounds. Edit the companion JSON after reviewing gauge comparability and the selected feature.</p>'+frame[cols].round(3).to_html(index=False)+'<h2>Review notes</h2>'
 import html
 for rec in review:page+='<p><b>'+rec['example']+'</b>: '+html.escape(' '.join(rec['notes']))+'</p>'
 (OUT/'height_target_review.html').write_text(page+'</html>')
 print(frame[cols].round(3).to_string(index=False),flush=True)
if __name__=='__main__':
 antecedent();height_review();other_cases()
