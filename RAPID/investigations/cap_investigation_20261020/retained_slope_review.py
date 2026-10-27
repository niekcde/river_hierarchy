"""Read-only notebook figures for the saved all-network retained-slope audit."""
import json
import geopandas as gpd
import numpy as np
import pandas as pd
import plotly.graph_objects as go


def load_audit(root):
    folder=root/'RAPID/outputs/uncapped_event_window_20261020/retained_slope_audit_all_networks_20261027'
    manifest=json.loads((folder/'audit.json').read_text())
    assert manifest['complete']
    return dict(manifest=manifest,
        states=pd.read_csv(folder/'state_summary.csv',dtype={'example':str}),
        by_network=pd.read_csv(folder/'by_network.csv',dtype={'example':str}),
        by_type=pd.read_csv(folder/'by_type.csv'),
        links=pd.read_parquet(folder/'changed_retained_links.parquet'),
        inventory=pd.read_parquet(root/'RAPID/outputs/uncapped_event_window_20261020/validation_group_inventory_20261023/records.parquet'))


def review_case(root,audit,example,state):
    example=str(example).zfill(2)
    selection=audit['states'].query('example == @example and state == @state')
    if len(selection)!=1:raise ValueError(f'Unknown audit case: {example}/{state}; choose from the displayed case list.')
    record=audit['inventory'].query('example == @example and state == @state').iloc[0]
    links=audit['links'].query('example == @example and state == @state').copy()
    links['slope_change_pct']=100*(links.slope_child/links.slope_base-1)
    links=links.sort_values('slope_change_pct',key=abs,ascending=False)
    assert len(links)==int(selection.changed_retained_links.iloc[0])
    suffix='_direction130' if example=='15' else ''
    parent=root/f'network_variants/output/{example}_indep{suffix}__anchored/states/S000_base'
    manifest=json.loads((parent/'hydraulic_context/hydraulic_context_manifest.json').read_text())
    geo=gpd.read_file(manifest['input_paths']['directed_links'])[['id_link','geometry']]
    assert geo.id_link.is_unique
    if not set(links.parent_link).issubset(set(geo.id_link)):
        raise ValueError('Saved parent link IDs do not all match the current parent geometry.')
    if geo.crs is None or geo.crs.is_geographic:
        raise ValueError('Expected projected parent geometry for the metric map.')
    geo=geo.merge(links,left_on='id_link',right_on='parent_link',how='left',validate='one_to_one')
    footprint=set(record.parent_footprint_links)
    geo['layer']='Parent context (not classified as changed)'
    geo.loc[geo.id_link.isin(footprint),'layer']='Selected collapse footprint'
    changed=geo.slope_child.notna()
    geo.loc[changed & ~geo.id_link.isin(footprint),'layer']='Changed retained reach outside footprint'
    geo.loc[changed & geo.id_link.isin(footprint),'layer']='Changed retained reach inside footprint'
    fig=go.Figure()
    layers=[('Parent context (not classified as changed)','#c4c4c4',1),
            ('Selected collapse footprint','#0072b2',4),
            ('Changed retained reach outside footprint','#d62728',2),
            ('Changed retained reach inside footprint','#8e44ad',3)]
    for label,color,width in layers:
        x=[];y=[];text=[]
        for row in geo[geo.layer.eq(label)].itertuples():
            hover=f'Parent link {row.id_link}<br>{label}'
            if pd.notna(row.slope_child):
                hover+=f'<br>Child link {int(row.child_link)}<br>Parent slope {row.slope_base:.7g}<br>Child slope {row.slope_child:.7g}<br>Change {row.slope_change_pct:+.2f}%'
            parts=list(row.geometry.geoms) if row.geometry.geom_type=='MultiLineString' else [row.geometry]
            for part in parts:
                coords=np.asarray(part.coords)
                x.extend(coords[:,0]/1000);x.append(None)
                y.extend(coords[:,1]/1000);y.append(None)
                text.extend([hover]*(len(coords)+1))
        if x:fig.add_trace(go.Scatter(x=x,y=y,text=text,mode='lines',name=label,
            line=dict(color=color,width=width),hovertemplate='%{text}<extra></extra>'))
    fig.update_layout(title=f'{example} / {state}: saved BASED slope changes on parent geometry',
        height=720,template='plotly_white',xaxis_title='Projected easting (km)',yaxis_title='Projected northing (km)',
        legend=dict(orientation='h',y=-.15),margin=dict(b=100))
    fig.update_yaxes(scaleanchor='x',scaleratio=1)
    scatter=go.Figure()
    if len(links):
        for outside,label,color in [(True,'Outside selected footprint','#d62728'),(False,'Inside selected footprint','#8e44ad')]:
            g=links[links.outside_saved_footprint.eq(outside)]
            scatter.add_trace(go.Scatter(x=g.slope_base,y=g.slope_child,mode='markers',name=label,
                marker=dict(color=color,size=6),text=g.parent_link.astype(str),
                hovertemplate='Parent link %{text}<br>Parent %{x:.6g}<br>Child %{y:.6g}<extra></extra>'))
        lo=min(links.slope_base.min(),links.slope_child.min());hi=max(links.slope_base.max(),links.slope_child.max())
        scatter.add_trace(go.Scatter(x=[lo,hi],y=[lo,hi],mode='lines',name='Unchanged slope',line=dict(color='grey',dash='dash')))
    else:scatter.add_annotation(text='No retained-reach slope changes detected for this case.',showarrow=False)
    scatter.update_layout(title='Changed retained reaches: parent versus child slope',template='plotly_white',height=480,
        xaxis=dict(type='log',title='Parent slope (m/m)'),yaxis=dict(type='log',title='Child slope (m/m)'))
    return dict(summary=selection.T,map=fig,scatter=scatter,links=links)
