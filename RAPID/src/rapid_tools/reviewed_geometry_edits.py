"""Explicit reviewed graph edits for isolated trials, with lineage and guards."""
import json
import geopandas as gpd
import pandas as pd
from shapely.geometry import LineString,Point


def merge_degree_two(links,nodes,raw_node,keep_link):
    """Suppress a directed degree-two node without changing its routed path."""
    ins=links[links.id_ds_node.eq(raw_node)];outs=links[links.id_us_node.eq(raw_node)]
    if len(ins)!=1 or len(outs)!=1:raise ValueError('Only a one-in/one-out node can be suppressed')
    a,b=ins.iloc[0],outs.iloc[0]
    if keep_link not in [int(a.id_link),int(b.id_link)]:raise ValueError('Retained ID must belong to merged chain')
    if Point(a.geometry.coords[-1]).distance(Point(b.geometry.coords[0]))>1e-6:raise ValueError('Disconnected geometry')
    if int(a.id_us_node)==int(b.id_ds_node):raise ValueError('Cannot create self-loop')
    rec=b.to_dict();rec.update(id_link=keep_link,id_us_node=int(a.id_us_node),id_ds_node=int(b.id_ds_node),u=int(a.u),v=int(b.v),geometry=LineString(list(a.geometry.coords)+list(b.geometry.coords)[1:]))
    sources=[]
    for row in [a,b]:sources.extend(json.loads(row.source_links))
    rec['source_links']=json.dumps(sorted(set(sources)));rec['edit_kind']='merged_reviewed_internal_node'
    out=links[~links.id_link.isin([a.id_link,b.id_link])].copy()
    out=gpd.GeoDataFrame(pd.concat([out,gpd.GeoDataFrame([rec],crs=links.crs)],ignore_index=True),crs=links.crs)
    return out,nodes[~nodes.id_node.eq(raw_node)].copy()


def restore_short_parent_connector(links,nodes,parent,parent_nodes,raw_node,parent_node,connector_link,parent_link):
    """Move a reviewed junction and restore its connector to parent geometry."""
    out=links.copy();ns=nodes.copy();target=parent_nodes.set_index('id_node').geometry.loc[parent_node]
    source=parent.set_index('id_link').loc[parent_link]
    connector=out[out.id_link.eq(connector_link)].iloc[0]
    if int(connector.id_ds_node)!=raw_node or int(connector.u)!=int(source.id_us_node) or int(source.id_ds_node)!=parent_node:
        raise ValueError('Connector endpoints disagree with reviewed parent reach')
    if any(ns.parent_identity.eq(parent_node)&~ns.id_node.eq(raw_node)):raise ValueError('Parent identity collision')
    oldpt=ns.loc[ns.id_node.eq(raw_node),'geometry'].iloc[0]
    for i,row in out[out.id_us_node.eq(raw_node)|out.id_ds_node.eq(raw_node)].iterrows():
        coords=list(row.geometry.coords)
        if row.id_us_node==raw_node:coords[0]=(target.x,target.y);out.at[i,'u']=parent_node
        if row.id_ds_node==raw_node:coords[-1]=(target.x,target.y);out.at[i,'v']=parent_node
        out.at[i,'geometry']=LineString(coords);out.at[i,'edit_kind']='reviewed_junction_relocation'
    ci=out.index[out.id_link.eq(connector_link)][0];out.at[ci,'geometry']=source.geometry;out.at[ci,'edit_kind']='restored_parent_connector';out.at[ci,'restored_parent_link']=parent_link
    ns.loc[ns.id_node.eq(raw_node),'geometry']=target;ns.loc[ns.id_node.eq(raw_node),'parent_identity']=parent_node
    return out,ns,float(oldpt.distance(target))
