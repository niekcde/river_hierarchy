"""Plotly maps and hydrographs for the isolated pilot; no rerouting on import."""
import json
import numpy as np
import pandas as pd
import geopandas as gpd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pilot import OUT


def pair_map(example,state):
    selection=next(r for r in json.loads((OUT/'selection.json').read_text()) if r['example']==example and r['state']==state)
    blocks={b['state']:b for b in json.loads((OUT/'eligible_blocks.json').read_text())[example]}
    parent=gpd.read_file(OUT/'variants'/example/'parent.gpkg');child=gpd.read_file(OUT/'variants'/example/state/'links.gpkg')
    parent=parent[parent.id_link.isin(selection['parent_ids'])].copy()
    lineage=pd.read_csv(OUT/'variants'/example/state/'lineage.csv');ids=lineage[lineage.parent_link_id.isin(selection['parent_ids'])].child_link_id
    child=child[child.id_link.isin(ids)].copy();parent['role']='Connector';child['role']='Connector'
    for key,label in [('state_a','Unit A'),('state_b','Unit B')]:
        parent.loc[parent.id_link.isin(blocks[selection[key]]['parent_ids']),'role']=label
        child.loc[child.block_state.eq(selection[key]),'role']=label
    fig=make_subplots(rows=1,cols=2,subplot_titles=['Original closed units','Both units collapsed; connector retained'],horizontal_spacing=.08)
    colors={'Unit A':'#2878b5','Unit B':'#c85824','Connector':'#5e6472'}
    for col,frame in [(1,parent),(2,child)]:
        for role,color in colors.items():
            xs=[];ys=[];hover=[]
            for r in frame[frame.role.eq(role)].itertuples():
                coords=list(r.geometry.coords);xs.extend([x/1000 for x,y in coords]+[None]);ys.extend([y/1000 for x,y in coords]+[None]);hover.extend([f'{role}; link {r.id_link}<br>L={r.link_length_m:.1f} m; W={r.width:.1f} m']*len(coords)+[''])
            fig.add_trace(go.Scatter(x=xs,y=ys,text=hover,hovertemplate='%{text}<extra></extra>',mode='lines',name=role,legendgroup=role,showlegend=col==1,line=dict(color=color,width=3 if role!='Connector' else 2)),row=1,col=col)
    xmin,ymin,xmax,ymax=parent.total_bounds/1000;dx=max(xmax-xmin,.1);dy=max(ymax-ymin,.1)
    for col in [1,2]:
        fig.update_xaxes(title_text='Easting (km)',range=[xmin-.08*dx,xmax+.08*dx],row=1,col=col)
        fig.update_yaxes(title_text='Northing (km)',range=[ymin-.08*dy,ymax+.08*dy],scaleanchor='x' if col==1 else 'x2',scaleratio=1,row=1,col=col)
    fig.update_layout(template='plotly_white',height=650,title=f'{example}: {selection["state_a"]} + {selection["state_b"]}',legend=dict(orientation='h',y=1.05))
    return fig


def hydrograph(example,state='all_closed_units_collapsed',window='0.5%'):
    m=pd.read_parquet(OUT/'state_metrics.parquet');m=m[(m.example==example)&(m.window==window)].set_index('state')
    if state=='all_closed_units_collapsed':names=['S000_base','single_centerline',state]
    else:
        selection=next(r for r in json.loads((OUT/'selection.json').read_text()) if r['example']==example and r['state']==state)
        names=['S000_base','control_'+selection['state_a'],'control_'+selection['state_b'],state]
    fig=go.Figure();colors=['#3b4856','#dc8843','#358f78','#5b57a3']
    labels={'S000_base':'Original parent','single_centerline':'Legacy uniform centerline','all_closed_units_collapsed':'Segmented single path'}
    for i,name in enumerate(names):
        r=m.loc[name];f=pd.read_parquet(r.outlet);start=pd.Timestamp(r.rise_seconds,unit='s',tz='UTC');end=pd.Timestamp(r.fall_seconds,unit='s',tz='UTC');f=f[f.time_utc.between(start,end)]
        fig.add_trace(go.Scatter(x=f.time_utc,y=f.q_outlet_cms,name=labels.get(name,name),mode='lines',line=dict(color=colors[i],width=2)))
        fig.add_vline(x=float(r.centroid_seconds)*1000,line_color=colors[i],line_dash='dot',opacity=.55)
    fig.update_layout(template='plotly_white',height=500,title=f'{example}: {state}<br><sup>Shared {window} window; dotted lines are full excess-volume centroids; no alignment</sup>',xaxis_title='UTC',yaxis_title='Discharge (m³/s)',legend=dict(orientation='h',y=-.25))
    return fig


def export():
    folder=OUT/'figures';folder.mkdir(exist_ok=True);pairs=json.loads((OUT/'selection.json').read_text())
    for r in pairs:
        pair_map(r['example'],r['state']).write_html(folder/f'{r["example"]}_{r["state"]}_map.html',include_plotlyjs=True)
        hydrograph(r['example'],r['state']).write_html(folder/f'{r["example"]}_{r["state"]}_hydrograph.html',include_plotlyjs=True)
    for ex in ['03','22','55']:hydrograph(ex).write_html(folder/f'{ex}_segmented_vs_uniform.html',include_plotlyjs=True)
    print('Saved 6 interactive domain maps, 6 pair hydrographs and 3 whole-network comparisons.')
if __name__=='__main__':export()
