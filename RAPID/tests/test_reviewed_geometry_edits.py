import json
import geopandas as gpd
import pytest
from shapely.geometry import Point,LineString
from rapid_tools.reviewed_geometry_edits import merge_degree_two,restore_short_parent_connector

def fixture():
 nodes=gpd.GeoDataFrame(dict(id_node=[1,2,3],parent_identity=[11,12,13]),geometry=[Point(0,0),Point(1,0),Point(3,0)],crs=32631)
 links=gpd.GeoDataFrame([dict(id_link=7,id_us_node=1,id_ds_node=2,u=11,v=12,source_links='[7]',geometry=LineString([(0,0),(1,0)])),dict(id_link=8,id_us_node=2,id_ds_node=3,u=12,v=13,source_links='[8]',geometry=LineString([(1,0),(3,0)]))],crs=nodes.crs)
 return links,nodes

def test_merge_conserves_directed_path_and_records_both_sources():
 links,nodes=fixture();out,ns=merge_degree_two(links,nodes,2,8)
 assert len(out)==1 and set(ns.id_node)=={1,3}
 assert out.iloc[0].u==11 and out.iloc[0].v==13
 assert out.length.sum()==links.length.sum()
 assert json.loads(out.iloc[0].source_links)==[7,8]

def test_merge_rejects_branching_junction():
 links,nodes=fixture();links.loc[2]=links.iloc[1];links.loc[2,'id_link']=9
 with pytest.raises(ValueError,match='one-in/one-out'):merge_degree_two(links,nodes,2,8)

def test_restore_moves_both_incident_endpoints_and_restores_parent_geometry():
 links,nodes=fixture();parent=gpd.GeoDataFrame([dict(id_link=90,id_us_node=11,id_ds_node=14,geometry=LineString([(0,0),(.5,.4),(1,1)]))],crs=nodes.crs)
 pn=gpd.GeoDataFrame(dict(id_node=[14]),geometry=[Point(1,1)],crs=nodes.crs)
 out,ns,dist=restore_short_parent_connector(links,nodes,parent,pn,2,14,7,90)
 assert dist==1 and out.iloc[0].geometry.equals(parent.iloc[0].geometry)
 assert Point(out.iloc[1].geometry.coords[0]).equals(Point(1,1))
 assert out.iloc[0].v==14 and out.iloc[1].u==14
 assert ns.loc[ns.id_node.eq(2),'parent_identity'].iloc[0]==14


def test_paired_cuts_recover_both_ends_of_retained_exterior_reach():
    from network_variants.boundary_contract_repair import derive_contract,paired_exterior_cuts
    pn=gpd.GeoDataFrame(dict(id_node=[0,1,2,3]),geometry=[Point(x,0) for x in range(4)],crs=32631)
    parent=gpd.GeoDataFrame([dict(id_link=i,id_us_node=i,id_ds_node=i+1,geometry=LineString([(i,0),(i+1,0)])) for i in range(3)],crs=pn.crs)
    cn=gpd.GeoDataFrame(dict(id_node=[10,30]),geometry=[Point(0,0),Point(3,0)],crs=pn.crs)
    child=gpd.GeoDataFrame([dict(id_link=20,id_us_node=10,id_ds_node=30,geometry=LineString([(0,0),(3,0)]))],crs=pn.crs)
    co=derive_contract(parent,[0,2]);result=paired_exterior_cuts(parent,pn,child,cn,{10:0,30:3},co,geometry_tolerance_m=.1)
    assert len(result)==1 and set(result[0]['cuts'])=={1,2}
    assert result[0]['parent_exterior_link']==1
    # A duplicate candidate path is ambiguous and must not be accepted.
    child.loc[1]=child.iloc[0];child.loc[1,'id_link']=21
    assert paired_exterior_cuts(parent,pn,child,cn,{10:0,30:3},co,geometry_tolerance_m=.1)==[]
