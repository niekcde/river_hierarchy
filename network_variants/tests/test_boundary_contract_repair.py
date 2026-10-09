import geopandas as gpd
import pandas as pd
from shapely.geometry import LineString,Point
from network_variants.boundary_contract_repair import derive_contract,propose_repairs,validate_result,PathIndex
from RAPID.src.rapid_tools.anchor_repair_trial import repair


def fixture():
    points={0:(-1,0),1:(0,0),2:(2,0),3:(1,1),4:(3,0)}
    edges=[(0,1),(1,2),(1,3),(3,2),(2,4)]
    parent=gpd.GeoDataFrame([dict(id_link=i,id_us_node=u,id_ds_node=v,geometry=LineString([points[u],points[v]])) for i,(u,v) in enumerate(edges)],crs=32631)
    pn=gpd.GeoDataFrame([dict(id_node=p,geometry=Point(xy)) for p,xy in points.items()],crs=32631)
    child=gpd.GeoDataFrame([dict(id_link=10,id_us_node=10,id_ds_node=12,geometry=LineString([points[0],points[2]])),dict(id_link=14,id_us_node=12,id_ds_node=14,geometry=LineString([points[2],points[4]]))],crs=32631)
    cn=gpd.GeoDataFrame([dict(id_node=10+p,geometry=Point(points[p])) for p in [0,2,4]],crs=32631)
    mapping={10:0,12:2,14:4}
    return parent,pn,child,cn,mapping,derive_contract(parent,[1,2,3])


def test_contract_excludes_internal_compound_nodes_and_rejects_disagreement():
    p,pn,c,cn,m,contract=fixture()
    assert contract['boundary_nodes']==[1,2]
    assert contract['entry_nodes']==[1] and contract['exit_nodes']==[2]
    assert not derive_contract(p,[1,2,3],declared_boundaries=[1,3])['valid']
    assert not derive_contract(p,[1,2,3],target_ok=False)['valid']


def test_topological_exterior_witness_supports_cut():
    p,pn,c,cn,m,contract=fixture()
    proposal=propose_repairs(p,pn,c,cn,m,contract,pd.DataFrame(),geometry_tolerance_m=.1)
    assert set(proposal['cuts'])=={1}
    a,n,_=repair(c,cn,pn,m,[1,2],cuts=[1],cut_link_ids={1:10})
    validation=validate_result(p,c,a,dict(zip(n.id_node,n.parent_identity)),contract,geometry_tolerance_m=.1)
    assert validation['anchor_pass'] and validation['exterior_exact']
    assert validation['geometry_preserved'] and validation['topology_preserved']


def test_displaced_existing_node_is_reused_instead_of_extra_cut():
    p,pn,c,cn,m,contract=fixture()
    c.loc[c.id_link==10,'id_ds_node']=11
    c.loc[c.id_link==10,'geometry']=LineString([(-1,0),(.05,0)])
    c.loc[2]=[11,11,12,LineString([(.05,0),(2,0)])]
    cn.loc[3]=[11,Point(.05,0)];m[11]=5
    matches=pd.DataFrame(dict(child_id_node=[11],matched_parent_node_id=[1],match_within_tolerance=[True]))
    proposal=propose_repairs(p,pn,c,cn,m,contract,matches,geometry_tolerance_m=.1)
    assert proposal['matches']=={11:1} and not proposal['cuts']


def test_nearby_but_wrong_topology_is_not_accepted():
    p,pn,c,cn,m,contract=fixture()
    cn.loc[3]=[11,Point(.05,0)];m[11]=5
    matches=pd.DataFrame(dict(child_id_node=[11],matched_parent_node_id=[1],match_within_tolerance=[True]))
    proposal=propose_repairs(p,pn,c,cn,m,contract,matches,geometry_tolerance_m=.1)
    assert not proposal['matches'] and not proposal['cuts']
    assert proposal['unresolved']==[1]


def test_exterior_chain_allows_subdivision_but_not_branch_junction():
    p,pn,c,cn,m,contract=fixture()
    exterior=gpd.GeoDataFrame(dict(id_link=[10,11],id_us_node=[10,15],id_ds_node=[15,11],geometry=[LineString([(-1,0),(-.5,0)]),LineString([(-.5,0),(0,0)])]),crs=32631)
    index=PathIndex(exterior,{10:0,15:5,11:1})
    assert index.compare(p.iloc[0],1e-5)[0]['child_links']==[10,11]
    exterior.loc[2]=[12,15,16,LineString([(-.5,0),(-.5,1)])]
    assert not PathIndex(exterior,{10:0,15:5,11:1,16:6}).compare(p.iloc[0],1e-5)


def test_displacement_envelope_does_not_accept_large_offset_claim():
    p,pn,c,cn,m,contract=fixture()
    cn.loc[3]=[11,Point(1,0)];m[11]=5
    matches=pd.DataFrame(dict(child_id_node=[11],matched_parent_node_id=[1],match_within_tolerance=[True]))
    proposal=propose_repairs(p,pn,c,cn,m,contract,matches,geometry_tolerance_m=.1)
    assert not proposal['matches']


def test_wrong_target_blocks_all_mutations():
    p,pn,c,cn,m,contract=fixture();contract['valid']=False
    proposal=propose_repairs(p,pn,c,cn,m,contract,pd.DataFrame(),geometry_tolerance_m=.1)
    assert not proposal['matches'] and not proposal['cuts']
    assert proposal['blocked']=='target_or_boundary_contract_mismatch'


def test_scoped_cut_does_not_split_nearby_parallel_reach():
    p,pn,c,cn,m,contract=fixture()
    c.loc[2]=[15,10,12,c.iloc[0].geometry]
    after,nodes,actions=repair(c,cn,pn,m,[1,2],cuts=[1],cut_link_ids={1:10})
    assert after[after.source_link.eq(15)].shape[0]==1
    assert after[after.source_link.eq(10)].shape[0]==2


def test_two_parent_exterior_branches_cannot_reuse_one_child_reach():
    p,pn,c,cn,m,contract=fixture()
    p.loc[5]=[5,0,1,p.iloc[0].geometry]
    contract=derive_contract(p,[1,2,3])
    a,n,_=repair(c,cn,pn,m,[1,2],cuts=[1],cut_link_ids={1:10})
    validation=validate_result(p,c,a,dict(zip(n.id_node,n.parent_identity)),contract,geometry_tolerance_m=.1)
    assert validation['exterior_path_reuse']
    assert not validation['anchor_pass']


def test_local_incidence_is_not_enough_if_replacement_exits_at_wrong_boundary():
    from network_variants.boundary_contract_repair import replacement_connections
    parent=pd.DataFrame(dict(id_link=[0,1],id_us_node=[0,2],id_ds_node=[1,3]))
    child=pd.DataFrame(dict(id_link=[10,11],id_us_node=[10,12],id_ds_node=[13,11]))
    contract=dict(selected_parent_links=[0,1],entry_nodes=[0,2],exit_nodes=[1,3])
    evidence=replacement_connections(parent,child,{10:0,11:1,12:2,13:3},contract,set())
    assert len(evidence)==4 and all(r['expected']!=r['actual'] for r in evidence)


def test_multiport_reachability_change_is_separate_from_anchor_correctness():
    points={-1:(-1,0),0:(0,0),1:(2,0),4:(3,0),-2:(-1,1),2:(0,1),3:(2,1),5:(3,1)}
    edges=[(-1,0),(0,1),(1,4),(-2,2),(2,3),(3,5)]
    parent=gpd.GeoDataFrame([dict(id_link=i,id_us_node=u,id_ds_node=v,geometry=LineString([points[u],points[v]])) for i,(u,v) in enumerate(edges)],crs=32631)
    child=parent.copy()
    child.loc[1,['id_ds_node','geometry']]=[3,LineString([points[0],points[3]])]
    child.loc[4,['id_ds_node','geometry']]=[1,LineString([points[2],points[1]])]
    child['source_link']=child.id_link
    contract=derive_contract(parent,[1,4])
    validation=validate_result(parent,child,child,{p:p for p in points},contract,geometry_tolerance_m=.1)
    assert validation['anchor_pass'] and validation['exterior_exact']
    assert not validation['replacement_connections_preserved']
