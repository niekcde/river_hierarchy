"""Regenerate transparent partial/final tables and standalone Plotly broad-study plots."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import broad_study as b
O=b.OUT
KEYS=['example','state','n','length','cfl','valid','cells','seconds','timing_error_minutes','feature_kind','feature_height_cms','height_distance_cms','raw_downstream_rmse_cms','raw_downstream_nse','peak_cms','maximum_depth_m','maximum_face_froude','minimum_face_discharge_cms','relative_mass_residual','maximum_last_48h_error_cms','error','folder']
METRICS=['peak_seconds','centroid_seconds','rising50_seconds','falling50_seconds']
def read(p):return json.loads(p.read_text())
def row(r):return {k:r.get(k) for k in KEYS}
def getfull(ex,state,length):
 f=b.frame(ex,state);n=float(f.n_manning.iloc[0]);found=[]
 for cfl in [.35,.175,.0875]:
  p=O/ex/f'{state}_dx{length:g}_n{n:.10f}_full_cfl{cfl:g}'/'result.json'
  if p.exists():
   r=b.refresh_feature(read(p));found.append(r)
   if r['valid']:return r
 return found[-1] if found else None

def main():
 selection=read(O/'selection.json');baselines=[];calibrations=[];transfers=[];grids=[];effects=[];tails=[]
 fig=go.Figure();groups=[];titles=[]
 for ex in b.CASES:
  start=len(fig.data);basepath=O/ex/'baseline.json';calpath=O/ex/'calibration.json';trpath=O/ex/'transfer.json';plot_status=[]
  base=b.refresh_feature(read(basepath)) if basepath.exists() else None
  plot_status.append('MC-n original: '+('accepted' if base and base['valid'] else 'rejected' if base else 'pending'))
  cal=read(calpath) if calpath.exists() else None
  tr=read(trpath) if trpath.exists() else None
  f=b.forcing(ex);obs=pd.read_parquet(O/'inputs'/ex/'observed_downstream.parquet')
  origin=f.interval_start_utc.iloc[0];event=f[f.phase.eq('input_event')]
  def add(w,label,color,dash='solid',observed=False):
   tc='time' if observed else 'time_utc';qc='discharge_cms' if observed else 'q_outlet_cms'
   # Plot all observed values; retain extrema in each 10-minute model block.
   if not observed:
    indices={0,len(w)-1}
    y=w[qc].to_numpy()
    for lo in range(0,len(w),10):
     chunk=y[lo:lo+10];indices.update([lo+int(np.argmin(chunk)),lo+int(np.argmax(chunk))])
    w=w.iloc[sorted(indices)]
   fig.add_trace(go.Scatter(x=w[tc],y=w[qc],name=label,mode='lines',line=dict(color=color,dash=dash,width=1.6),visible=(ex=='02'),connectgaps=False))
  add(obs,'Observed downstream','#111111',observed=True)
  offset=b.HEIGHTS[ex].get('downstream_display_offset_cms')
  if offset is not None and offset!=0:
   shifted=obs.copy();shifted['discharge_cms']+=offset
   add(shifted,'Observed: frozen prior display offset','#444444','dot',observed=True)
  add(pd.read_parquet(O/'inputs'/ex/'mc_base.parquet'),'Legacy MC reference (Oct 13)','#777777','dash')
  mcwave=pd.read_parquet(O/'inputs'/ex/'mc_base.parquet');mct=(mcwave.time_utc-origin).dt.total_seconds().to_numpy()
  mcbase=np.interp(mct,[(event.interval_start_utc.iloc[0]-origin).total_seconds(),(event.interval_end_utc.iloc[-1]-origin).total_seconds()],[event.forcing_cms.iloc[0],event.forcing_cms.iloc[-1]])
  mcmetrics=b.v.metrics(mct,mcwave.q_outlet_cms.to_numpy(),mcbase)
  if base:
   baselines.append(row(base))
   if base['valid']:add(pd.read_parquet(Path(base['folder'])/'wave.parquet'),'Local inertia: MC roughness','#f0a000','dot')
  final=None
  if cal:
   final=b.refresh_feature(cal['final']) if 'final' in cal else None
   cr=dict(example=ex,status=cal['status'],mc_n=cal.get('mc_n'),selected_n=cal.get('selected_n'),search_trials=len(cal.get('trials',[])))
   cr['search_method']=cal.get('search_method','adaptive_local_grid' if ex in ['22','37','41','46','55'] else 'fixed_coarse_scan')
   physical=b.frame(ex,'S000_base')
   cr['mc_stored_reference_depth_max_m']=float(physical.depth_m.max())
   if final:
    cr.update({k:final.get(k) for k in KEYS if k not in ['example','state']})
    cr['soft_height_target_met']=final.get('height_distance_cms')==0
    if final['valid']:
     cr['local_minus_mc_peak_height_percent']=100*(final['peak_cms']/mcmetrics['peak_cms']-1)
     for metric in METRICS:
      cr['local_minus_mc_'+metric.replace('_seconds','_minutes')]=(final[metric]-mcmetrics[metric])/60 if final.get(metric) is not None and mcmetrics.get(metric) is not None else None
   cr['height_target_low_cms']=b.HEIGHTS[ex].get('low_cms',b.HEIGHTS[ex].get('height_target_nominal_cms'))
   cr['height_target_high_cms']=b.HEIGHTS[ex].get('high_cms',b.HEIGHTS[ex].get('height_target_nominal_cms'))
   calibrations.append(cr)
   if final and final['valid']:add(pd.read_parquet(Path(final['folder'])/'wave.parquet'),'Local inertia: selected roughness','#006bb6')
  colors=['#d62728','#178844']
  for j,state in enumerate(selection['children'][ex]):
   add(pd.read_parquet(O/'inputs'/ex/(state+'_mc.parquet')),f'Legacy MC collapse {state}',colors[j],'dash')
   child=next((r for r in (tr or {}).get('runs',[]) if r['state']==state),None)
   plot_status.append(state+': '+('accepted' if child and child['valid'] else 'rejected' if child else 'pending'))
   if child:
    child=b.refresh_feature(child);transfers.append(row(child)|dict(parent_calibration_status=tr.get('parent_calibration_status')))
    if child['valid']:
     add(pd.read_parquet(Path(child['folder'])/'wave.parquet'),f'Local inertia collapse {state}',colors[j])
     if final and final['valid']:
      er=dict(example=ex,state=state,n=child['n'],parent_calibration_status=cal['status'],peak_change_cms=child['peak_cms']-final['peak_cms'])
      for key in METRICS:er[key.replace('_seconds','_change_minutes')]=(child[key]-final[key])/60 if child.get(key) is not None and final.get(key) is not None else None
      for label,st in [('base','mc_base'),('child',state+'_mc')]:
       w=pd.read_parquet(O/'inputs'/ex/(st+'.parquet'));t=(w.time_utc-origin).dt.total_seconds().to_numpy()
       baseline=np.interp(t,[(event.interval_start_utc.iloc[0]-origin).total_seconds(),(event.interval_end_utc.iloc[-1]-origin).total_seconds()],[event.forcing_cms.iloc[0],event.forcing_cms.iloc[-1]])
       m=b.v.metrics(t,w.q_outlet_cms.to_numpy(),baseline)
       if label=='base':mb=m
       else:
        for key in METRICS:er['mc_'+key.replace('_seconds','_change_minutes')]=(m[key]-mb[key])/60 if m.get(key) is not None and mb.get(key) is not None else None
      effects.append(er)
  for rr in [base,final,*((tr or {}).get('runs',[]))]:
   if rr and rr['valid']:
    threshold=.001*float(f.forcing_cms.max()-f.forcing_cms.min())
    tails.append(dict(example=ex,state=rr['state'],n=rr['n'],folder=rr['folder'],last48h_error_cms=rr['maximum_last_48h_error_cms'],threshold_cms=threshold,settled=rr['maximum_last_48h_error_cms']<=threshold))
  if ex in b.SUBSET:
   for length in [200,100,50]:
    grbase=getfull(ex,'S000_base',length)
    for state in ['S000_base',*selection['children'][ex]]:
     r=getfull(ex,state,length)
     if not r:continue
     gr=row(r)
     if r['valid'] and grbase and grbase['valid']:
      for key in METRICS:gr[key.replace('_seconds','_change_minutes')]=(r[key]-grbase[key])/60 if r.get(key) is not None and grbase.get(key) is not None else None
     grids.append(gr)
  nmc=float(b.frame(ex,'S000_base').n_manning.iloc[0]);ntext=f' (legacy MC n={nmc:.5f}'
  ntext+=(f"; selected n={cal['selected_n']:.5f})" if cal and 'selected_n' in cal else ')')
  groups.append((start,len(fig.data)));titles.append(f'Example {ex}: '+(cal['status'].replace('_',' ') if cal else 'calibration pending')+ntext+'<br><sup>Local-inertial runs — '+'; '.join(plot_status)+'</sup>')
 for name,rows in [('baseline',baselines),('calibration',calibrations),('transfer',transfers),('grid',grids),('collapse_effects',effects),('tail_checks',tails)]:
  pd.DataFrame(rows).to_csv(O/(name+'_summary.csv'),index=False)
 buttons=[]
 for ex,(a,z),title in zip(b.CASES,groups,titles):
  buttons.append(dict(label=ex,method='update',args=[dict(visible=[a<=i<z for i in range(len(fig.data))]),dict(title=dict(text=title),**{'xaxis.autorange':True,'yaxis.autorange':True},shapes=[dict(type='line',name='Reviewed timing target',showlegend=True,x0=b.TARGETS[ex]['target_utc'],x1=b.TARGETS[ex]['target_utc'],y0=0,y1=1,yref='paper',line=dict(color='black',dash='dot'))])]))
 fig.update_layout(title=titles[0],template='plotly_white',height=720,xaxis_title='Time (UTC)',yaxis_title='Discharge (m³/s)',hovermode='x unified',legend=dict(orientation='h',y=-.2),margin=dict(b=175),updatemenus=[dict(buttons=buttons,x=1,y=1.16,xanchor='right',yanchor='top')],shapes=buttons[0]['args'][1]['shapes'])
 fig.write_html(O/'hydrographs_plotly.html',include_plotlyjs=True,full_html=True)
 status=dict(baselines_completed=len(baselines),calibrations_completed=len(calibrations),calibrated=sum(r['status']=='timing_calibrated' for r in calibrations),transfers_attempted=len(transfers),transfers_valid=sum(bool(r['valid']) for r in transfers),grid_cases_completed=len(grids),grid_cases_valid=sum(bool(r['valid']) for r in grids),unsettled_tails=[r for r in tails if not r['settled']],plot_traces=len(fig.data))
 b.save(O/'summary_status.json',status);print(json.dumps(status,indent=2))
if __name__=='__main__':main()
