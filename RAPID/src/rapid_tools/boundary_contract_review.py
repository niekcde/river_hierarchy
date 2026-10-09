"""Parent/current/trial views with explicit ID namespaces and rectangular coverage."""
from __future__ import annotations
import math
import pandas as pd
from shapely.geometry import box
from shapely.ops import unary_union
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def identity_label(identity, parent_ids):
    if pd.isna(identity):return 'unmapped'
    return ('P' if int(identity) in parent_ids else 'H')+str(int(identity))


def correspondence_table(before_nodes,after_nodes,parent_nodes,required,*,extra_child_nodes=()):
    """Distinguish identity corrections, new graph cuts, and coordinate changes."""
    parent=parent_nodes.set_index('id_node').geometry;parent_ids=set(parent.index)
    before=before_nodes.set_index('id_node');rows=[]
    selected=after_nodes[after_nodes.parent_identity.isin(required)|after_nodes.id_node.isin(extra_child_nodes)]
    represented=set(selected.parent_identity)
    for r in selected.itertuples():
        raw=int(r.id_node);p=int(r.parent_identity);old=before.loc[raw] if raw in before.index else None
        rows.append(dict(child_node=f'C{raw}',before_identity=identity_label(old.parent_identity,parent_ids) if old is not None else 'node absent',
            trial_identity=identity_label(p,parent_ids),prescribed_boundary=p in required,
            original_parent_node=p if p in parent_ids else None,
            parent_displacement_m=float(r.geometry.distance(parent.loc[p])) if p in parent_ids else None,
            identity_changed=old is not None and int(old.parent_identity)!=p,
            graph_node_added=old is None,
            coordinate_shift_m=float(r.geometry.distance(old.geometry)) if old is not None else None))
    for p in sorted(set(required)-represented):
        rows.append(dict(child_node='unresolved',before_identity='',trial_identity=f'P{p}',prescribed_boundary=True,original_parent_node=p,
                         parent_displacement_m=None,identity_changed=False,graph_node_added=False,coordinate_shift_m=None))
    return pd.DataFrame(rows)


def plot_comparison(before,after,parent,nodes_before,nodes_after,parent_nodes,required,selected,title,*,
                    context_padding_m=None,focus_parent_node=None,focus_child_node=None,radius_m=80,
                    show_link_ids=True,show_context_node_labels=True):
    """Render all links intersecting the rectangular viewport, not a branch buffer.

    P IDs exist in the original parent; C IDs are raw child graph IDs; H IDs are
    new hydraulic identities. Context dots do not assert unit membership.
    """
    crs=parent.crs
    if any(f.crs!=crs for f in [before,after,nodes_before,nodes_after,parent_nodes]):raise ValueError('All layers must share the projected CRS')
    if focus_parent_node is not None and focus_child_node is not None:raise ValueError('Choose one focus node namespace')
    required=set(map(int,required));selected=set(map(int,selected));parent_ids=set(parent_nodes.id_node)
    if focus_parent_node is not None:
        focus=parent_nodes[parent_nodes.id_node.eq(int(focus_parent_node))]
    elif focus_child_node is not None:
        focus=nodes_after[nodes_after.id_node.eq(int(focus_child_node))]
    else:focus=None
    if focus is not None:
        if len(focus)!=1:raise ValueError('Focus node must exist uniquely in the chosen namespace')
        if radius_m<=0:raise ValueError('Radius must be positive')
        p=focus.geometry.iloc[0];bounds=(p.x-radius_m,p.y-radius_m,p.x+radius_m,p.y+radius_m)
    else:
        features=list(parent[parent.id_link.isin(selected)].geometry)+list(parent_nodes[parent_nodes.id_node.isin(required)].geometry)
        features+=list(nodes_after[nodes_after.parent_identity.isin(required)].geometry)
        if not features:raise ValueError('No selected domain or boundary geometry')
        xmin,ymin,xmax,ymax=unary_union(features).bounds
        pad=context_padding_m if context_padding_m is not None else max(150,.08*max(xmax-xmin,ymax-ymin))
        bounds=(xmin-pad,ymin-pad,xmax+pad,ymax+pad)
    region=box(*bounds)
    fig=make_subplots(rows=1,cols=3,horizontal_spacing=.045,
                      subplot_titles=['Original parent: selected links highlighted','Current anchored output','Trial repair'])
    shown=set();meta={'viewport_bounds':list(bounds),'selection_filter':'rectangular_viewport','visible_link_ids':{},'visible_child_nodes':{}}

    def add(trace,col,key):
        trace.legendgroup=key;trace.showlegend=key not in shown;shown.add(key)
        fig.add_trace(trace,row=1,col=col)

    def draw_links(frame,col,color,name,key,*,dash='solid',width=2,labels=False):
        visible=frame[frame.intersects(region)]
        for r in visible.itertuples():
            clipped=r.geometry.intersection(region)
            parts=list(clipped.geoms) if clipped.geom_type=='MultiLineString' else [clipped]
            for part in parts:
                if part.geom_type!='LineString':continue
                x,y=part.xy;hover=f'{name}<br>Link L{r.id_link}<br>Raw nodes {r.id_us_node} → {r.id_ds_node}'
                if hasattr(r,'u'):hover+=f'<br>Hydraulic identities {identity_label(r.u,parent_ids)} → {identity_label(r.v,parent_ids)}'
                if hasattr(r,'source_link'):hover+=f'<br>Source child link L{r.source_link}'
                add(go.Scatter(x=list(x),y=list(y),mode='lines',line=dict(color=color,width=width,dash=dash),name=name,
                               meta=dict(kind='link',id_link=int(r.id_link)),hovertemplate=hover+'<extra></extra>'),col,key)
            if labels and not clipped.is_empty and clipped.length>0:
                p=clipped.interpolate(.5,normalized=True)
                fig.add_trace(go.Scatter(x=[p.x],y=[p.y],mode='text',text=[f'L{r.id_link}'],textfont=dict(size=9,color=color),
                    showlegend=False,hoverinfo='skip',meta=dict(kind='link_label',id_link=int(r.id_link))),row=1,col=col)
        return visible.id_link.astype(int).tolist()

    outside=parent[~parent.id_link.isin(selected)];inside=parent[parent.id_link.isin(selected)]
    visible_parent=draw_links(outside,1,'#777','Parent exterior links','parent_exterior',labels=show_link_ids)
    visible_parent+=draw_links(inside,1,'#e47b12','Selected parent links','parent_selected',width=3,labels=show_link_ids)
    meta['visible_link_ids']['parent']=visible_parent
    pn=parent_nodes[parent_nodes.intersects(region)&~parent_nodes.id_node.isin(required)]
    add(go.Scatter(x=pn.geometry.x,y=pn.geometry.y,mode='markers+text' if show_context_node_labels else 'markers',
        text=[f'P{v}' for v in pn.id_node],textposition='top right',textfont=dict(size=10,color='#333'),
        marker=dict(size=6,color='#555'),name='Original parent nodes',hovertemplate='%{text}<extra></extra>'),1,'parent_nodes')
    ports=parent_nodes[parent_nodes.id_node.isin(required)&parent_nodes.intersects(region)]
    for col,child,nodes,stage in [(2,before,nodes_before,'current'),(3,after,nodes_after,'trial')]:
        meta['visible_link_ids'][stage]=draw_links(child,col,'#007d98' if col==2 else '#00865b',
            'Current child links' if col==2 else 'Trial child links',stage+'_links',width=3,labels=show_link_ids)
        # Draw dashed references above the child so coincident parent paths remain visible.
        draw_links(parent,col,'#8d6aa8','Parent reference (toggle)','reference',dash='dot',width=1.4)
        near=nodes[nodes.intersects(region)]
        meta['visible_child_nodes'][stage]=near.id_node.astype(int).tolist()
        for boundary in [False,True]:
            n=near[near.parent_identity.isin(required).eq(boundary)]
            labels=[f'C{r.id_node} → {identity_label(r.parent_identity,parent_ids)}' if boundary else f'C{r.id_node}' for r in n.itertuples()]
            hover=[f'C{r.id_node}<br>Identity: {identity_label(r.parent_identity,parent_ids)}<br>'+('Prescribed boundary' if boundary else 'Context node; not a prescribed boundary') for r in n.itertuples()]
            add(go.Scatter(x=n.geometry.x,y=n.geometry.y,mode='markers+text' if boundary or show_context_node_labels else 'markers',
                text=labels,textposition='bottom right',textfont=dict(size=10,color='#8a4900' if boundary else '#666'),
                marker=dict(size=10 if boundary else 5,symbol='diamond' if boundary else 'circle',color='#ef9b27' if boundary else '#999'),
                customdata=hover,hovertemplate='%{customdata}<extra></extra>',name='Child boundary nodes' if boundary else 'Child context nodes',
                meta=dict(kind='child_nodes',boundary=boundary)),col,'boundary_child' if boundary else 'context_child')
        for r in near[near.parent_identity.isin(required)].itertuples():
            p=parent_nodes[parent_nodes.id_node.eq(int(r.parent_identity))].geometry.iloc[0]
            d=p.distance(r.geometry)
            if d>1e-5:
                add(go.Scatter(x=[p.x,r.geometry.x],y=[p.y,r.geometry.y],mode='lines',line=dict(color='#bb258c',width=2,dash='dash'),
                    name='Parent–child displacement',hovertemplate=f'P{int(r.parent_identity)} ↔ C{r.id_node}<br>{d:.3f} m<extra></extra>'),col,'displacement')
        if col==3:
            added=near[~near.id_node.isin(nodes_before.id_node)]
            if len(added):add(go.Scatter(x=added.geometry.x,y=added.geometry.y,mode='markers',marker=dict(size=17,symbol='star-open',color='#00865b'),
                text=[f'New graph cut C{r.id_node} → {identity_label(r.parent_identity,parent_ids)}' for r in added.itertuples()],name='Inserted graph cut',hovertemplate='%{text}<extra></extra>'),col,'inserted_cut')
    for col in [1,2,3]:
        add(go.Scatter(x=ports.geometry.x,y=ports.geometry.y,mode='markers+text',text=[f'P{v}' for v in ports.id_node],
            textposition='top left',textfont=dict(size=11,color='#b32132'),marker=dict(size=12,symbol='x-open',color='#b32132'),
            name='Original prescribed boundary',hovertemplate='Original boundary %{text}<extra></extra>'),col,'parent_boundary')
        fig.update_xaxes(range=[bounds[0],bounds[2]],title_text='Easting (m)',matches='x' if col>1 else None,row=1,col=col)
        fig.update_yaxes(range=[bounds[1],bounds[3]],title_text='Northing (m)' if col==1 else None,
                         matches='y' if col>1 else None,scaleanchor='x' if col==1 else None,scaleratio=1,row=1,col=col)
    fig.update_layout(title=dict(text=title,x=.02),template='plotly_white',height=800,autosize=True,
                      margin=dict(t=100,b=180,l=65,r=25),legend=dict(orientation='h',x=0,y=-.16,font=dict(size=11)),hovermode='closest',meta=meta)
    fig.add_annotation(x=0,y=-.32,xref='paper',yref='paper',xanchor='left',showarrow=False,
        text='P = original parent ID · C = raw child ID · H = new hydraulic ID (not a parent node). Grey child dots are context, not unit membership.',font=dict(size=11,color='#555'))
    return fig
