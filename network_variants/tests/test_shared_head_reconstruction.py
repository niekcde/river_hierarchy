import numpy as np
import pandas as pd
import pytest
from scipy import sparse
from network_variants.shared_head_reconstruction import reconstruct_shared_heads, _constrained_least_squares


def fixture():
    base=pd.DataFrame(dict(id_us_node=[0,1,1,2,3,4],id_ds_node=[1,2,3,4,4,5],actual_length_m=[100.]*6,ownership_kind=['direct','simple_unit','simple_unit','simple_unit','simple_unit','direct'],ownership_domain_id=[np.nan,1,1,1,1,np.nan]))
    nodes=pd.DataFrame(dict(id_node=range(6),reference_section_dist_out_m=[400.,300.,200.,200.,100.,0.],head=[4.,3.,2.,2.,1.,0.]))
    links=base.rename(columns={'id_us_node':'u','id_ds_node':'v','actual_length_m':'link_length_m'}).copy()
    return base,nodes,links


def test_unchanged_heads_preserved_exactly():
    base,nodes,links=fixture();heads=nodes.set_index('id_node')['head'];out,qa=reconstruct_shared_heads(links,base,nodes,heads)
    assert qa['free_nodes']==0
    assert np.array_equal(out.head_us,links.u.map(heads))


def test_changed_branch_reconstructs_interior_and_keeps_outer_anchors():
    base,nodes,links=fixture();links.loc[links.u.eq(2),'link_length_m']=200.
    out,qa=reconstruct_shared_heads(links,base,nodes,nodes.set_index('id_node')['head'])
    assert qa['free_nodes']==2
    # Minimize (d1/L1 - Sref)^2 + (d2/L2 - Sref)^2 with d1+d2=2.
    assert out.loc[out.u.eq(2),'head_us'].iloc[0]==pytest.approx(2.2)
    assert out.loc[out.u.eq(0),'head_us'].iloc[0]==4.
    assert out.loc[out.v.eq(4),'head_ds'].eq(1.).all()
    assert (out.slope>0).all()


def test_constrained_solution_is_optimal_at_active_floor():
    C=sparse.csc_matrix([[10.],[-1.]])
    # Short-link preference competes with the second edge's active floor.
    x,qa=_constrained_least_squares(C,np.array([0.,.02]),.01,np.array([.005]))
    assert x[0]==pytest.approx(.01,abs=1e-7)
    assert qa['stationarity']<1e-9


def test_infeasible_fixed_heads_fail_explicitly():
    C=sparse.csc_matrix([[1.],[-1.]])
    with pytest.raises(ValueError,match='Infeasible'):
        _constrained_least_squares(C,np.array([0.,0.]),.01,np.array([0.]))


def test_coincident_child_ports_remain_distinct_graph_nodes():
    import geopandas as gpd
    from shapely.geometry import Point
    from network_variants.shared_head_reconstruction import match_child_nodes
    base=pd.DataFrame(dict(id_us_node=[0,1],id_ds_node=[1,2],actual_length_m=[100.,100.]))
    bn=gpd.GeoDataFrame({'id_node':[0,1,2]},geometry=[Point(0,0),Point(1,0),Point(2,0)])
    cn=gpd.GeoDataFrame({'id_node':[10,11,12,13]},geometry=[Point(0,0),Point(1,0),Point(1,0),Point(2,0)])
    child=pd.DataFrame(dict(id_us_node=[10,11,12],id_ds_node=[11,12,13],link_length_m=[100.,20.,80.]))
    mapping,unknown,qa=match_child_nodes(bn,cn,base,child)
    assert mapping[11]==1 and mapping[12] in unknown
    assert mapping[11]!=mapping[12]
    assert len(qa['coincident_ports'])==1


def test_ambiguous_coincident_ports_are_not_silently_merged():
    import geopandas as gpd
    from shapely.geometry import Point
    from network_variants.shared_head_reconstruction import match_child_nodes
    bn=gpd.GeoDataFrame({'id_node':[0,1]},geometry=[Point(0,0),Point(1,0)])
    cn=gpd.GeoDataFrame({'id_node':[10,11]},geometry=[Point(0,0),Point(0,0)])
    base=pd.DataFrame(dict(id_us_node=[0],id_ds_node=[1],actual_length_m=[100.]))
    child=pd.DataFrame(dict(id_us_node=[10],id_ds_node=[11],link_length_m=[20.]))
    with pytest.raises(ValueError,match='Ambiguous'):
        match_child_nodes(bn,cn,base,child)
