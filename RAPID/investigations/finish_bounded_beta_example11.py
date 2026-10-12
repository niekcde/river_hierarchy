"""Broader timing scan and independent resolution controls for example 11."""
from pathlib import Path
import sys,json
import numpy as np,pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(Path(__file__).parent),str(ROOT/'RAPID/src')]
from calibrate_common_grid_height import feature
from rapid_tools import cunge
from rapid_tools.common_grid import route_interval_means
OUT=ROOT/'RAPID/outputs/bounded_beta_example11_20261009'

def main():
 data=pd.read_parquet(OUT/'inputs/11/60s.parquet');meta=next(r for r in json.loads((OUT/'inputs/11/60s.json').read_text()) if r['method']=='cunge')
 review=next(r for r in json.loads((OUT/'timing_targets.json').read_text())['records'] if r['example']=='11')
 table=pd.read_csv(OUT/'inputs/baseline/RAPID/outputs/cunge_calibration_reviewed_20261006/comparison.csv',dtype={'example':str});row=table[table.example.eq('11')&table.method.eq('cunge')].iloc[0]
 grid=pd.read_parquet(OUT/'inputs/baseline'/Path(row.output_dir).relative_to(ROOT)/'frozen_grid.parquet')
 valid=data.downstream_cms.notna().to_numpy();tt=data.time_utc.astype('int64').to_numpy()[valid]/1e9;goal=pd.Timestamp(review['target_utc']).timestamp();dt=meta['dt_s'];initial=meta['initial_cms']
 def evaluate(beta,n,step=60,dt=dt,mult=1):
  g=grid.copy();g['count']*=mult;g['rapid_effective_length_m']=g.link_length_m/g['count'];f=cunge.parameters(g,n,beta=beta)
  wave=data if step==60 else pd.read_parquet(OUT/'inputs/11/30s.parquet');ok=wave.downstream_cms.notna().to_numpy();t=wave.time_utc.astype('int64').to_numpy()[ok]/1e9
  q,a=route_interval_means(f,wave.forcing_cms.to_numpy(),round(step/dt),dt,initial_flow=initial);assert a['minimum_native_flow_cms']>=0
  tm,h,kind,lobes=feature(q[ok],t,'11',review)
  return dict(beta=float(beta),n_manning=float(n),timing_error_minutes=(tm-goal)/60,peak_cms=h,feature_utc=pd.Timestamp(tm,unit='s',tz='UTC').isoformat(),feature_kind=kind),q
 records=[];best=None;bestq=None
 for beta in np.geomspace(.5,2,9):
  for n in np.unique(np.r_[np.linspace(.01,.1,19),meta['n_manning']]):
   r,q=evaluate(beta,n);records.append(r)
   if best is None or abs(r['timing_error_minutes'])<abs(best['timing_error_minutes']):best,bestq=r,q
  print('11 dense beta',beta,'best error',best['timing_error_minutes'],flush=True)
 pd.DataFrame(records).to_csv(OUT/'dense_timing_scan.csv',index=False)
 checks=[]
 for name,step,clock,mult in [('30s',30,dt,1),('half_dt',60,dt/2,1),('half_dx_half_dt',60,dt/2,2)]:
  r,q=evaluate(best['beta'],best['n_manning'],step,clock,mult);checks.append(dict(control=name,**r))
 offset=initial-data.downstream_cms.dropna().iloc[0];obs=data.downstream_cms.to_numpy()[valid]+offset
 comparison=[]
 for name,q in [('Original Cunge',data.cunge_cms.to_numpy()),('Bounded tuned Cunge',bestq),('Previously calibrated fixed X',data.fixed_x_cms.to_numpy())]:
  tm,h,kind,_=feature(q[valid],tt,'11',review);comparison.append(dict(method=name,timing_error_minutes=(tm-goal)/60,peak_cms=h,shifted_rmse_cms=float(np.sqrt(np.mean((q[valid]-obs)**2)))))
 (OUT/'dense_scan_summary.json').write_text(json.dumps(dict(selected=best,trials=len(records),controls=checks,comparison=comparison,scope='Diagnostic finite beta/n scan; no proof of global optimality. No production reinstatement.'),indent=2))
 w=data.copy();w['bounded_scan_cunge_cms']=bestq;w.to_parquet(OUT/'diagnostic_wave.parquet',index=False)
 fig=make_subplots(rows=2,cols=1,subplot_titles=['Example 11: common initial plateau; downstream shifted only','Timing error across tested beta and Manning n'],vertical_spacing=.16)
 for name,q,color,dash in [('Input',data.forcing_cms,'#2580a5','dash'),('Shifted downstream',data.downstream_cms+offset,'#18212b','solid'),('Original Cunge',data.cunge_cms,'#9b7477','dash'),('Bounded tuned Cunge',bestq,'#ce3d45','solid'),('Fixed X comparator',data.fixed_x_cms,'#dc8a16','solid')]:
  fig.add_trace(go.Scatter(x=data.time_utc[valid],y=np.asarray(q)[valid]/initial,name=name,line=dict(color=color,dash=dash),customdata=np.asarray(q)[valid],hovertemplate='%{x}<br>Q / initial: %{y:.3f}<br>Plotted discharge: %{customdata:.2f} m³/s<extra>%{fullData.name}</extra>'),row=1,col=1)
 fig.add_shape(type='line',x0=review['target_utc'],x1=review['target_utc'],y0=0,y1=1,xref='x',yref='y domain',line=dict(color='purple',dash='dot'))
 fig.add_vrect(x0=review['crest_start_utc'],x1=review['crest_end_utc'],fillcolor='purple',opacity=.07,line_width=0,row=1,col=1)
 for beta in sorted(set(r['beta'] for r in records)):
  group=[r for r in records if r['beta']==beta]
  fig.add_trace(go.Scatter(x=[r['n_manning'] for r in group],y=[r['timing_error_minutes']/60 for r in group],name=f'β={beta:.3g}',mode='lines+markers'),row=2,col=1)
 fig.update_layout(height=1000,title=f"11 diagnostic: best tested β={best['beta']:.3g}, n={best['n_manning']:.5f}; crest {abs(best['timing_error_minutes'])/60:.2f} hours early",template='plotly_white')
 fig.update_yaxes(title_text='Q / original initial forcing',row=1,col=1);fig.update_yaxes(title_text='Timing error (hours)',row=2,col=1);fig.update_xaxes(title_text='Manning n',row=2,col=1)
 fig.write_html(OUT/'11_bounded_beta_diagnostic.html',include_plotlyjs=True)
 print(json.dumps(comparison),flush=True)
if __name__=='__main__':main()
