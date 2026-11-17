"""Standalone grid-sensitivity plot; rejected configurations remain missing."""
import json
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import broad_study as b

def main():
 g=pd.read_csv(b.OUT/'grid_summary.csv',dtype={'example':str});fig=make_subplots(rows=4,cols=2,subplot_titles=[f'Example {e}' for e in b.SUBSET]+[''])
 selection=json.loads((b.OUT/'selection.json').read_text())['children']
 names=['peak','centroid','rising50','falling50'];trace_metric=[]
 # Keep every subplot axis instantiated, including panels with rejected runs.
 # Plotly cannot relayout an absent axis when a dropdown changes the metric.
 for i in range(len(b.SUBSET)):
  fig.add_trace(go.Scatter(x=[None],y=[None],mode='markers',showlegend=False,hoverinfo='skip'),row=i//2+1,col=i%2+1)
  trace_metric.append(-1)
 for mi,metric in enumerate(names):
  for i,ex in enumerate(b.SUBSET):
   x=g[(g.example==ex)&(g.state!='S000_base')]
   for j,(state,df) in enumerate(x.groupby('state')):
    rank=selection[ex].index(state)
    df=df.set_index('length').reindex([200,100,50]);y=df[metric+'_change_minutes']
    fig.add_trace(go.Scatter(x=[200,100,50],y=y,mode='lines+markers',name=['Smaller selected collapse','Larger selected collapse'][rank],legendgroup=str(rank),line=dict(color=['#d62728','#178844'][rank]),showlegend=i==0,visible=mi==0,connectgaps=False,hovertemplate=f'{state}<br>max length %{{x}} m<br>change %{{y:.3f}} min<extra></extra>'),row=i//2+1,col=i%2+1);trace_metric.append(mi)
 fig.update_xaxes(title_text='Maximum cell length (m)',autorange='reversed',tickvals=[200,100,50]);fig.update_yaxes(title_text='Collapse − original (min)')
 titles={m:f'Grid sensitivity: {m} timing effect at fixed MC roughness' for m in names}
 fig.update_layout(title=titles['peak'],height=1200,template='plotly_white',margin=dict(t=140),legend=dict(orientation='h',x=0,y=1.045),updatemenus=[dict(buttons=[dict(label=m,method='update',args=[{'visible':[j in (-1,i) for j in trace_metric]},{'title':{'text':titles[m]},**{('yaxis'+(str(ax) if ax>1 else '')+'.autorange'):True for ax in range(1,8)}}]) for i,m in enumerate(names)],x=1,y=1.08)])
 fig.update_xaxes(visible=False,row=4,col=2);fig.update_yaxes(visible=False,row=4,col=2)
 for i,ex in enumerate(b.SUBSET):
  sub=g[g.example.eq(ex)&g.state.ne('S000_base')]
  if sub.empty or not sub['peak_change_minutes'].notna().any():
   axis=str(i+1) if i else ''
   fig.add_annotation(text='No accepted comparison available',x=.5,y=.5,xref=f'x{axis} domain',yref=f'y{axis} domain',showarrow=False,font=dict(color='#777777'))
 fig.add_annotation(text='Missing points: rejected or unfinished configurations. Grids retain physical reaches shorter than the maximum.',x=0,y=-.07,xref='paper',yref='paper',showarrow=False)
 fig.write_html(b.OUT/'grid_sensitivity_plotly.html',include_plotlyjs=True)
if __name__=='__main__':main()
