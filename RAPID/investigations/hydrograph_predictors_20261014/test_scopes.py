import numpy as np
import pandas as pd
import pytest
import scopes
import resistance


def test_reviewed_identity_must_match_routed_frame(tmp_path):
    child=pd.DataFrame(dict(id_us_node=[10,11],id_ds_node=[11,12],u=[0,1],v=[1,2]))
    pd.DataFrame(dict(id_node=[10,11,12],parent_identity=[0,1,2])).to_csv(tmp_path/'node_hydraulic_context.csv',index=False)
    assert scopes.reviewed_mapping(child,tmp_path,{0,1,2})=={0:10,1:11,2:12}
    child.loc[0,'u']=99
    with pytest.raises(ValueError,match='does not agree'):scopes.reviewed_mapping(child,tmp_path,{0,1,2})


def test_open_unit_is_distinct_from_closed_compound():
    parent=pd.DataFrame(dict(id_link=[1,2,3,4,5],id_us_node=[0,0,1,1,2],id_ds_node=[1,2,3,2,3],link_length_m=[1.]*5,width=[1.]*5))
    # Unit 0->2 excludes the branch 1->3, so its internal node exchanges with outside.
    unit=parent[parent.id_link.isin([1,2,4])]
    p,c,pc,cc=scopes.graph_pair(parent,parent,unit,{0:0,1:1,2:2,3:3})
    assert not pc and not cc
    assert scopes.graph_pair(parent,parent,parent,{0:0,1:1,2:2,3:3})[2:]==(True,True)


def test_equivalent_resistance_parallel_formula_and_scale(tmp_path,monkeypatch):
    monkeypatch.setattr(scopes,'OUT',tmp_path);monkeypatch.setattr(resistance,'OUT',tmp_path)
    (tmp_path/'graphs').mkdir();(tmp_path/'resistance_cache').mkdir()
    f=pd.DataFrame(dict(id_link=[1,2],id_us_node=[0,0],id_ds_node=[1,1],link_length_m=[4.,9.],width=[1.,1.]))
    key,_,_=scopes.save_graph(f);result=resistance.solve_graph(key,1.)
    # Q = 1/sqrt(K1)+1/sqrt(K2); K_eq=1/Q^2.
    assert result['success'];np.testing.assert_allclose(result['K'],(1/2+1/3)**-2,rtol=1e-8)
    f['width']*=2;key,_,_=scopes.save_graph(f);scaled=resistance.solve_graph(key,1.)
    np.testing.assert_allclose(scaled['K']/result['K'],2**-5,rtol=1e-8)


def test_flow_energy_matches_existing_head_solver_on_braided_core():
    from flow_resistance import solve_unit_flow
    from hierarchy_level_definition.metrics.equivalent_resistance import solve_bubble_equivalent_K
    edges=pd.DataFrame(dict(edge_id=range(5),u=[0,0,1,2,1],v=[1,2,3,3,2],K_e=[1.,3.,2.,4.,2.]))
    flow=solve_unit_flow(edges,0,3,tolerance=1e-10)
    head=solve_bubble_equivalent_K(None,edges,0,3,solver_backend='dense_legacy')
    assert flow['success'] and head['solver_success']
    assert flow['continuity']<1e-8 and flow['relative_gap']<1e-10
    np.testing.assert_allclose(flow['K_eq'],head['K_eq'],rtol=1e-7)
