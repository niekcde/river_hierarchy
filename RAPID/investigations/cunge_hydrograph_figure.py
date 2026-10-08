"""Interactive 16-event comparison, using archived calibrated hydrographs."""
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'RAPID/outputs/cunge_calibration_alpha2'
STYLE={
 'upstream':dict(name='Upstream gauge',color='#808080',dash='dash'),
 'downstream':dict(name='Downstream gauge',color='#000000',dash='dash'),
 'fixed_x':dict(name='Fixed X model',color='#00BFD8',dash='solid'),
 'cunge':dict(name='Cunge model',color='#E31A1C',dash='solid'),
}

def build_figure(results=None,paths=None):
 if results is None:results=pd.read_csv(OUT/'comparison.csv',dtype={'example':str})
 if paths is None:paths={r.example:Path(r.output_dir) for r in results.itertuples()}
 examples=sorted(paths)
 if len(examples)!=16:raise ValueError('Expected the 16 Excel-selected calibration events.')
 fig=make_subplots(rows=4,cols=4,subplot_titles=[f'Network {ex}' for ex in examples],horizontal_spacing=.055,vertical_spacing=.075)
 for j,ex in enumerate(examples):
  row,col=divmod(j,4);row+=1;col+=1;p=Path(paths[ex])
  downstream=pd.read_parquet(p/'observed_downstream.parquet');times=pd.to_datetime(downstream.time,utc=True);start,end=times.min(),times.max()
  for kind,style in STYLE.items():
   if kind in ['upstream','downstream']:
    f=pd.read_parquet(p/f'observed_{kind}.parquet');t=pd.to_datetime(f.time,utc=True);q=f.discharge_cms
   else:
    f=pd.read_parquet(p/f'{kind}_best_wave.parquet');t=pd.to_datetime(f.time_utc,utc=True);q=f.q_outlet_cms
   mask=(t>=start)&(t<=end);tt=t[mask];qq=q[mask].to_numpy(float)
   if not len(qq) or not np.isfinite(qq).all():raise ValueError(f'Invalid series: {ex}/{kind}')
   yy=(qq-qq.min())/max(float(np.ptp(qq)),1e-12)
   xx=(tt-start).dt.total_seconds().to_numpy()/86400
   custom=np.column_stack([tt.dt.strftime('%Y-%m-%d %H:%M:%S UTC').to_numpy(),qq])
   fig.add_trace(go.Scatter(x=xx,y=yy,mode='lines',name=style['name'],legendgroup=kind,showlegend=j==0,
    line=dict(color=style['color'],dash=style['dash'],width=1.8),customdata=custom,
    meta=dict(example=ex,series=kind,window_start_utc=start.isoformat(),normalization='Independent min-max within displayed event window'),
    hovertemplate=f'<b>Network {ex} · {style["name"]}</b><br>%{{customdata[0]}}<br>Day: %{{x:.3f}}<br>Normalized Q: %{{y:.4f}}<br>Discharge: %{{customdata[1]:.2f}} m³/s<extra></extra>'),row=row,col=col)
  fig.update_xaxes(title_text='Days',title_standoff=7,range=[0,(end-start).total_seconds()/86400],nticks=5,showgrid=True,zeroline=False,row=row,col=col)
  fig.update_yaxes(title_text='Normalized Q' if col==1 else None,title_standoff=7,range=[-.04,1.06],dtick=.25,tickformat='.2f',showgrid=True,zeroline=False,row=row,col=col)
 fig.update_layout(template='plotly_white',width=1400,height=1550,font=dict(family='Arial, sans-serif',size=12),
  margin=dict(t=165,b=85,l=85,r=35),
  title=dict(text='<b>Calibrated hydrographs · 16 selected events</b><br><sup>Independent min–max normalization; original UTC timing retained</sup>',x=.055,y=.985,xanchor='left',yanchor='top',font=dict(size=22)),
  legend=dict(orientation='h',x=.5,y=1.065,xanchor='center',yanchor='bottom',font=dict(size=13),groupclick='togglegroup',traceorder='normal'),
  hovermode='closest',paper_bgcolor='white',plot_bgcolor='white')
 for annotation in fig.layout.annotations:annotation.font=dict(size=15,color='#222222')
 fig.add_annotation(x=.5,y=-.052,xref='paper',yref='paper',showarrow=False,font=dict(size=12,color='#555555'),
  text='Days are measured from each panel’s downstream observation-window start. No peak alignment or time stretching. Legend clicks toggle a series across all panels.')
 assert len(fig.data)==64 and sum(bool(t.showlegend) for t in fig.data)==4
 return fig

def save_figure(fig,destination=OUT):
 destination=Path(destination);destination.mkdir(exist_ok=True,parents=True)
 div=fig.to_html(full_html=False,include_plotlyjs=True,config={'displaylogo':False,'responsive':False,'toImageButtonOptions':{'format':'svg','filename':'calibration_hydrographs'}})
 html='<!doctype html><html><head><meta charset="utf-8"><title>Calibrated hydrographs — 16 events</title><style>body{margin:0;background:white;font-family:Arial,sans-serif}.figure{overflow-x:auto}p{margin:12px 50px 24px;max-width:1300px;color:#444;font-size:14px}</style></head><body><div class="figure">'+div+'</div><p>These are the 16 Excel-selected calibration events. Each curve is normalized separately; hover shows original discharge and UTC. Model curves use their separately calibrated Manning n. Gauge lines connect available samples and do not imply observations exist within gaps. Additional automatically screened events are not included.</p></body></html>'
 (destination/'calibration_hydrographs.html').write_text(html)
 fig.write_json(destination/'calibration_hydrographs.plotly.json')
 return destination/'calibration_hydrographs.html'

if __name__=='__main__':print(save_figure(build_figure()))
