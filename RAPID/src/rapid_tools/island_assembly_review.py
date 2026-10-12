"""Matched-scale old/new review of the four corrected island assemblies."""
from pathlib import Path
import json,math
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.windows import from_bounds
from rasterio.enums import Resampling
from rasterio.features import geometry_mask
from affine import Affine
from shapely.geometry import box
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from rapid_tools import collapse_creation_review as review
from rapid_tools.anchor_repair_trial import node_mapping
ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'RAPID/outputs/island_assembly_fix_20261008'
ISLANDS=(5,55,75,138)


def case_data(island,buffer_m=150):
    if int(island) not in ISLANDS:raise ValueError(f'Choose an island from {ISLANDS}')
    if buffer_m<=0:raise ValueError('Buffer must be positive')
    state=f'island_41_{int(island):03d}';d=OUT/state
    co=json.loads((d/'actual_island_contract.json').read_text());result=json.loads((d/'review_result.json').read_text())
    if not result['geometry_accepted']:raise ValueError('This repair has not passed geometry checks')
    row=review.case_row('41',state);parent=review.vectors(row.base_links_path,'links');pn=review.vectors(row.base_nodes_path,'nodes')
    old=review.vectors(row.child_links_path,'links').copy();on=review.vectors(row.child_nodes_path,'nodes').copy();on['parent_identity']=on.id_node.map(node_mapping(pd.read_parquet(row.frame_path)))
    new=gpd.read_parquet(d/'fixed_links.parquet');nn=gpd.read_parquet(d/'fixed_nodes.parquet');islands=review.catalog('41')
    target=islands[islands.island_id.eq(state)].geometry.iloc[0]
    selected=parent[parent.id_link.isin(co['selected_parent_links'])].geometry.union_all();xmin,ymin,xmax,ymax=selected.union(target).bounds
    bounds=(xmin-buffer_m,ymin-buffer_m,xmax+buffer_m,ymax+buffer_m);region=box(*bounds)
    mask=Path(row.base_links_path).parents[2]/'variant/mask/41__S000_base_collapsed.tif'
    with rasterio.open(mask) as src:
        win=from_bounds(*bounds,src.transform).round_offsets().round_lengths();factor=max(1,math.ceil(max(win.width,win.height)/1000));shape=(math.ceil(win.height/factor),math.ceil(win.width/factor))
        base=src.read(1,window=win,out_shape=shape,resampling=Resampling.nearest);tr=src.window_transform(win)*Affine.scale(win.width/shape[1],win.height/shape[0])
    filled=geometry_mask([target],base.shape,tr,invert=True,all_touched=True)
    z=np.where(base>0,0,np.where(filled,1,2));xs=tr.c+(np.arange(z.shape[1])+.5)*tr.a;ys=tr.f+(np.arange(z.shape[0])+.5)*tr.e
    return dict(state=state,folder=d,contract=co,result=result,parent=parent,parent_nodes=pn,old=old,old_nodes=on,new=new,new_nodes=nn,islands=islands,target=target,bounds=bounds,region=region,z=z,xs=xs,ys=ys,transform=tr)


def plot_case(island,buffer_m=150,show_link_labels=False):
    data=case_data(island,buffer_m);co=data['contract'];region=data['region'];bounds=data['bounds'];parent_ids=set(data['parent_nodes'].id_node)
    fig=make_subplots(rows=1,cols=2,subplot_titles=['Old · current production geometry','New · corrected isolated geometry'],shared_xaxes='all',shared_yaxes='all',horizontal_spacing=.055)
    colors=[[0,'white'],[.24999,'white'],[.25,'#aaa'],[.74999,'#aaa'],[.75,'black'],[1,'black']]
    for col,stage,color in [(1,'old','#c45532'),(2,'new','#0087b5')]:
        fig.add_trace(go.Heatmap(x=data['xs'],y=data['ys'],z=data['z'],zmin=0,zmax=2,colorscale=colors,showscale=False,hoverinfo='skip'),1,col)
        frame=data[stage];nodes=data[stage+'_nodes'];mp=dict(zip(nodes.id_node,nodes.parent_identity))
        def label(raw):
            p=int(mp[raw]);return f'P{p}' if p in parent_ids else f'C{int(raw)}'
        first=True;labels=[]
        for r in frame[frame.intersects(region)].itertuples():
            clipped=r.geometry.intersection(region);parts=list(clipped.geoms) if hasattr(clipped,'geoms') else [clipped]
            for part in parts:
                if part.geom_type!='LineString':continue
                x,y=part.xy
                fig.add_trace(go.Scatter(x=list(x),y=list(y),mode='lines',line=dict(color=color,width=2),name='Old links' if col==1 else 'New links',legendgroup=stage,showlegend=first,hovertemplate=f'L{r.id_link}<br>{label(r.id_us_node)} → {label(r.id_ds_node)}<br>{r.geometry.length:.1f} m<extra></extra>'),1,col);first=False
            mid=clipped.representative_point()
            if not clipped.is_empty:labels.append((mid.x,mid.y,f'L{r.id_link}'))
        if labels:
            fig.add_trace(go.Scatter(x=[z[0] for z in labels],y=[z[1] for z in labels],text=[z[2] for z in labels],mode='text',textfont=dict(size=10,color=color),name='Link labels',legendgroup='link_labels',showlegend=col==1,visible=True if show_link_labels else 'legendonly',hoverinfo='skip'),1,col)
        visible=nodes[nodes.intersects(region)]
        fig.add_trace(go.Scatter(x=visible.geometry.x,y=visible.geometry.y,mode='markers',marker=dict(size=5,color=color),text=[f'C{int(r.id_node)} · {label(r.id_node)}' for r in visible.itertuples()],name='Child nodes',showlegend=False,hovertemplate='%{text}<extra></extra>'),1,col)
        ports=data['parent_nodes'][data['parent_nodes'].id_node.isin(co['boundary_nodes'])]
        fig.add_trace(go.Scatter(x=ports.geometry.x,y=ports.geometry.y,mode='markers+text',marker=dict(size=11,color='#9b0089',symbol='circle-open',line=dict(width=2)),text=[f'P{int(n)}' for n in ports.id_node],textposition=[('top left' if r.geometry.x < min((q.geometry for q in ports.itertuples() if q.id_node != r.id_node), key=lambda g: g.distance(r.geometry)).x else 'bottom right') for r in ports.itertuples()],textfont=dict(color='#9b0089',size=12),name='Required parent boundaries',legendgroup='boundaries',showlegend=col==1,hovertemplate='%{text}<extra></extra>'),1,col)
        for r in data['islands'][data['islands'].intersects(region)].itertuples():
            clipped=r.geometry.intersection(region)
            if clipped.area<20:continue
            pt=clipped.representative_point();target=r.island_id==data['state'];txt=f"Island {int(r.island_id.rsplit('_',1)[1])}"+(' · collapsed' if target else '')
            fig.add_annotation(x=pt.x,y=pt.y,text=txt,showarrow=False,font=dict(color='black' if target else 'white',size=11),bgcolor='#c3c3c3' if target else '#333',opacity=.93,row=1,col=col)
        fig.update_xaxes(range=[bounds[0],bounds[2]],title_text='Easting (m)',showgrid=False,row=1,col=col)
        fig.update_yaxes(range=[bounds[1],bounds[3]],title_text='Northing (m)' if col==1 else None,showgrid=False,row=1,col=col)
    # Give both panels identical physical scales without conflicting matches/scaleanchor constraints.
    fig.update_xaxes(matches=None, constrain='domain')
    fig.update_yaxes(matches=None, constrain='domain')
    for col, anchor in [(1, 'x'), (2, 'x2')]:
        fig.update_yaxes(scaleanchor=anchor, scaleratio=1, row=1, col=col)
    for color,name in [('#aaa','Collapsed island footprint'),('black','Remaining islands / land'),('white','Water')]:
        fig.add_trace(go.Scatter(x=[None],y=[None],mode='markers',marker=dict(color=color,size=12,line=dict(color='#555',width=1)),name=name),1,1)
    fig.update_layout(title=dict(text=f'Network 41 · Island {int(island)} · old / repaired<br><sup>{buffer_m:g} m context buffer · hover for link/node IDs · P = original parent; C = child-only node</sup>',x=.03),template='plotly_white',height=900,margin=dict(t=125,b=130,l=85,r=40),legend=dict(orientation='h',x=0,y=-.13,groupclick='togglegroup'),hovermode='closest',meta=dict(state=data['state'],geometry_accepted=True,buffer_m=buffer_m,production_changed=False))
    return fig


def static_preview(island,path,buffer_m=150):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    data=case_data(island,buffer_m);tr=data['transform'];z=data['z'];bounds=data['bounds'];region=data['region']
    extent=(tr.c,tr.c+z.shape[1]*tr.a,tr.f+z.shape[0]*tr.e,tr.f)
    fig,axs=plt.subplots(1,2,figsize=(15,9),sharex=True,sharey=True)
    for ax,stage,color,title in zip(axs,['old','new'],['#c45532','#0087b5'],['Old · current production','New · corrected trial']):
        ax.imshow(z,cmap=ListedColormap(['white','#aaa','black']),vmin=0,vmax=2,extent=extent,origin='upper')
        for r in data[stage][data[stage].intersects(region)].itertuples():
            g=r.geometry.intersection(region);parts=list(g.geoms) if hasattr(g,'geoms') else [g]
            for part in parts:
                if part.geom_type=='LineString':ax.plot(*part.xy,color=color,lw=1.4)
        for r in data['islands'][data['islands'].intersects(region)].itertuples():
            pt=r.geometry.intersection(region).representative_point();target=r.island_id==data['state'];ax.text(pt.x,pt.y,str(int(r.island_id.rsplit('_',1)[1])),ha='center',color='black' if target else 'white',fontsize=10,bbox=dict(facecolor='#ccc' if target else '#333',edgecolor='none',alpha=.85))
        for i,r in enumerate(data['parent_nodes'][data['parent_nodes'].id_node.isin(data['contract']['boundary_nodes'])].itertuples()):
            ax.scatter(r.geometry.x,r.geometry.y,color='#9b0089',s=18);ax.annotate(f'P{r.id_node}',(r.geometry.x,r.geometry.y),xytext=(-8,10) if r.geometry.x < min((q.geometry for q in data['parent_nodes'][data['parent_nodes'].id_node.isin(data['contract']['boundary_nodes'])].itertuples() if q.id_node != r.id_node), key=lambda g: g.distance(r.geometry)).x else (8,-15),ha='right' if r.geometry.x < min((q.geometry for q in data['parent_nodes'][data['parent_nodes'].id_node.isin(data['contract']['boundary_nodes'])].itertuples() if q.id_node != r.id_node), key=lambda g: g.distance(r.geometry)).x else 'left',textcoords='offset points',fontsize=9,color='#9b0089',bbox=dict(facecolor='white',alpha=.8,edgecolor='none',pad=1))
        ax.set(xlim=bounds[::2],ylim=bounds[1::2],title=title,xlabel='Easting (m)');ax.set_aspect('equal')
    axs[0].set_ylabel('Northing (m)');fig.suptitle(f'Network 41 · Island {island}\nGrey: collapsed island · Black: remaining land · White: water',fontsize=14);fig.tight_layout();fig.savefig(path,dpi=140);plt.close(fig)
