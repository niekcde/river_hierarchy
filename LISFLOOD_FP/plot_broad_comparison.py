"""Compare selected collapse timing effects across the two routing models."""
import json
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import broad_study as b

def main():
 p=b.OUT/'collapse_effects_summary.csv'
 if not p.exists():return
 d=pd.read_csv(p,dtype={'example':str})
 if d.empty:return
 sel=json.loads((b.OUT/'selection.json').read_text())['children']
 d['collapse_size']=[sel[e].index(s) for e,s in zip(d.example,d.state)]
 d['calibrated']=d.parent_calibration_status.eq('timing_calibrated')
 fig=make_subplots(rows=2,cols=2,subplot_titles=['Peak timing','Centroid timing','Rising half-height crossing','Falling half-height crossing'])
 for i,metric in enumerate(['peak','centroid','rising50','falling50']):
  key=metric+'_change_minutes';mc='mc_'+key;valid=d[d[key].notna()&d[mc].notna()];values=np.r_[valid[key],valid[mc],0.];lo=float(values.min());hi=float(values.max());pad=max(1.,.08*(hi-lo));bounds=[lo-pad,hi+pad]
  for accepted in [True,False]:
   for size in [0,1]:
    x=valid[valid.calibrated.eq(accepted)&valid.collapse_size.eq(size)]
    if x.empty:continue
    group=f'{accepted}_{size}';name=('Timing calibrated' if accepted else 'Unresolved parent')+': '+('smaller collapse' if size==0 else 'larger collapse')
    fig.add_trace(go.Scatter(x=x[mc],y=x[key],mode='markers',name=name,legendgroup=group,showlegend=i==0,marker=dict(size=10,symbol='circle' if size==0 else 'diamond',color='#1676b5' if accepted else '#888888',line=dict(width=1,color='white')),customdata=x[['example','state','n']].to_numpy(),hovertemplate='Example %{customdata[0]}: %{customdata[1]}<br>Local n=%{customdata[2]:.5f}<br>MC effect=%{x:.3f} min<br>Local-inertial effect=%{y:.3f} min<extra></extra>'),row=i//2+1,col=i%2+1)
  fig.add_trace(go.Scatter(x=bounds,y=bounds,mode='lines',line=dict(color='#999999',dash='dot'),name='Equal effect',showlegend=False,hoverinfo='skip'),row=i//2+1,col=i%2+1)
  fig.update_xaxes(title_text='Legacy MC: collapse − original (min)',range=bounds,row=i//2+1,col=i%2+1)
  fig.update_yaxes(title_text='Local inertia: collapse − original (min)',range=bounds,row=i//2+1,col=i%2+1)
 fig.update_layout(title='Legacy MC (Oct 13) versus local-inertial routing',template='plotly_white',height=1050,legend=dict(orientation='h',y=-.15),margin=dict(b=160))
 fig.add_annotation(text='Historical MC reference; not the current predictor-training MC setup. Negative values mean earlier passage; dotted lines show equal effects.',x=0,y=-.09,xref='paper',yref='paper',showarrow=False)
 fig.write_html(b.OUT/'collapse_comparison_plotly.html',include_plotlyjs=True)
if __name__=='__main__':main()
