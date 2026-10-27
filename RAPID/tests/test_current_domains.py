from pathlib import Path
import sys
import numpy as np
import pandas as pd
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from rapid_tools.current_domains import allocation_tables,validate_domain


def frame():
    return pd.DataFrame(dict(id_link=range(6),id_us_node=[0,0,1,2,2,3],
        id_ds_node=[1,1,2,3,3,4],width=[9.,1.,10.,9.,1.,10.],link_length_m=[1.,2.,1.,1.,2.,1.]))


def test_confluences_restore_probability_after_repeated_ninety_ten_splits():
    nodes,edges=allocation_tables(frame())
    r=nodes.set_index('node').visit_probability
    np.testing.assert_allclose(r.loc[[1,2,3,4]],1.)
    np.testing.assert_allclose(edges.set_index('id_link').loc[[2,5],'r_edge'],1.)
    assert np.isclose(edges[edges.u.eq(2)].r_edge.max(),.9)


def test_domain_rejects_changed_topology_and_unrepresented_side_branch():
    f=frame();part,bounds=validate_domain(f,f)
    assert bounds==(0,4) and len(part)==6
    old=f.copy();old.loc[0,'id_ds_node']=99
    with pytest.raises(ValueError,match='endpoints'):validate_domain(f,old)
    external=pd.DataFrame(dict(id_link=[9],id_us_node=[2],id_ds_node=[99],width=[1.],link_length_m=[1.]))
    with pytest.raises(ValueError,match='internal domain node'):
        validate_domain(pd.concat([f,external]),f)


def test_recorded_parent_child_anchor_mapping_allows_renumbering(tmp_path):
    from rapid_tools.current_domains import load_domain_pair
    base=frame();child=base.copy();child['id_us_node']+=100;child['id_ds_node']+=100
    root=tmp_path/'experiment';state=root/'states/S001';prep=state/'rapid/cunge/prep';prep.mkdir(parents=True)
    base_path=tmp_path/'parent.parquet';base.to_parquet(base_path);child_path=prep/'parameters.parquet';child.to_parquet(child_path)
    directory=state/'equivalent_resistance';directory.mkdir()
    pd.concat([base.assign(domain_role='parent'),child.assign(domain_role='child')]).to_csv(directory/'selected_hydraulic_domain_links.csv',index=False)
    cmp=pd.DataFrame([dict(parent_upstream_boundary_node=0,parent_downstream_boundary_node=4,
        child_upstream_boundary_node=100,child_downstream_boundary_node=104,
        child_upstream_anchor_match_method='stable_node_match',child_downstream_anchor_match_method='stable_node_match',
        hydraulic_domain_type='closed_unit',selected_unit_ids='1')])
    cmp.to_csv(directory/'bubble_equivalent_state_comparison.csv',index=False)
    rec=pd.Series(dict(registry_path=str(root/'rapid_run_registry.csv'),state_id='S001',
        parent_parameters_parquet=str(base_path),parameters_parquet=str(child_path)))
    _,_,meta=load_domain_pair(rec)
    assert meta['source']==0 and meta['child_source']==100
    cmp.loc[0,'child_upstream_boundary_node']=101
    cmp.to_csv(directory/'bubble_equivalent_state_comparison.csv',index=False)
    with pytest.raises(ValueError,match='Child anchors'):load_domain_pair(rec)
