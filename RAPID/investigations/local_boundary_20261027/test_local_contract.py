from pathlib import Path
import sys
sys.path[:0]=[str(Path(__file__).resolve().parents[3]),str(Path(__file__).resolve().parents[3]/'RAPID/src')]
import numpy as np
import pandas as pd
import pytest
from rapid_tools.cunge import freeze_grid,parameters
from local_contract import correct,ContractError


def fixture():
    p=pd.DataFrame(dict(id_link=[0,1,2,3],u=[0,1,2,3],v=[1,2,3,4],link_length_m=[100.]*4,width=[20.]*4,depth_m=[2.]*4,head_us=[4.,3.,2.,1.],head_ds=[3.,2.,1.,0.],slope=[.01]*4))
    p=parameters(freeze_grid(p,10),.035)
    c=p.iloc[[0,1,3]].copy();c.loc[c.id_link.eq(1),['v','link_length_m','head_ds']]=[3,200.,1.]
    return p,c,{0:0,3:3},[1,2]


def test_local_replacement_keeps_exterior_exact_and_conserves_flow():
    p,c,m,s=fixture();c.loc[c.id_link.eq(0),'width']=99
    f,qa=correct(p,c,m,s)
    assert qa['flow_residual']<1e-12
    assert f.set_index('id_link').loc[0,'width']==20
    assert (f.slope>0).all()
    assert f.set_index('id_link').loc[1,'slope']==.01


def test_fixed_fixed_edge_must_obey_floor():
    p,c,m,s=fixture()
    with pytest.raises(ContractError,match='slope floor infeasible'):correct(p,c,m,s,slope_floor=.02)


def test_unknown_internal_node_is_solved_without_releasing_exterior():
    p,c,m,s=fixture();c=p.copy();c.loc[c.id_link.eq(1),'v']=8;c.loc[c.id_link.eq(2),'u']=8
    f,qa=correct(p,c,m,s)
    assert qa['free_nodes']==1
    assert f.loc[f.id_link.eq(1),'head_ds'].iloc[0]==pytest.approx(2.)
    assert f.loc[f.id_link.eq(2),'head_us'].iloc[0]==pytest.approx(2.)


def test_infeasible_free_node_heads_rejected():
    p,c,m,s=fixture();c=p.copy();c.loc[c.id_link.eq(1),'v']=8;c.loc[c.id_link.eq(2),'u']=8
    with pytest.raises(ContractError,match='Infeasible fixed boundary heads'):correct(p,c,m,s,slope_floor=.02)


def test_flow_infeasibility_not_hidden_by_positive_heads():
    p,c,m,s=fixture()
    # Reverse replacement with positive drop but incompatible parent port fluxes.
    p.loc[p.u.eq(1),'head_us']=0.;p.loc[p.v.eq(1),'head_ds']=0.
    c.loc[c.id_link.eq(1),['u','v']]=[3,1]
    with pytest.raises(ContractError,match='Infeasible positive reference flow'):correct(p,c,m,s)


def test_port_path_reduction_preserves_boundary_geometry_and_flux():
    import geopandas as gpd
    from shapely.geometry import LineString
    from local_contract import port_path_replacement,exterior_map
    p,_,_,selected=fixture()
    geo=gpd.GeoDataFrame(p[['id_link','u','v']],geometry=[LineString([(100*u,0),(100*v,0)]) for u,v in zip(p.u,p.v)],crs='EPSG:32631')
    child,cg,lineage=port_path_replacement(p,geo,selected)
    assert len(child)==3 and lineage[0]['parent_path']==[1,2]
    mapping=exterior_map(p,child,geo,cg,selected)
    result,qa=correct(p,child,mapping,selected)
    assert qa['boundary_connectivity_changes']==[]
    assert qa['flow_residual']==0
    assert result.loc[~result.id_link.isin(mapping),'link_length_m'].iloc[0]==200
