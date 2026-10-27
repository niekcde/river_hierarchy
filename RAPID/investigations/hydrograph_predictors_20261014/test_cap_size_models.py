import numpy as np
import pandas as pd
import cap_size_models as m


def test_overlap_is_transitive_and_example_specific():
    d=pd.DataFrame(dict(example=['01','01','01','01','02']))
    g=m.overlap_components(d,{0:[1,2],1:[2,3],2:[3,4],3:[5],4:[1,2]})
    assert g[0]==g[1]==g[2]
    assert g[0]!=g[3] and g[0]!=g[4]


def test_known_network_splits_keep_overlaps_together():
    d=pd.DataFrame(dict(example=['01']*4+['02']*4,overlap_group=['a','a','b','c','d','d','e','f']))
    folds=m.outer_splits(d,'within');seen=[]
    for _,tr,te in folds:
        assert set(tr.overlap_group).isdisjoint(te.overlap_group)
        assert set(te.example)<=set(tr.example)
        seen.extend(te.index)
        for ti,vi in m.inner_splits(tr,'within'):
            assert set(tr.iloc[ti].overlap_group).isdisjoint(tr.iloc[vi].overlap_group)
    assert sorted(seen)==list(d.index)


def test_specifications_preserve_ablations_and_target_scope():
    p=m.panels()
    assert set(p['revised'])-set(p['no_resistance'])=={'B_K_log_a1'}
    assert set(p['revised'])-set(p['no_mean_width'])=={'log1p__B_ratio_width_mean'}
    assert 'B_exposed_length_change' not in p['revised']
    assert 'B_mean_length_fraction' not in p['revised']
    assert len(m.specifications())==56
    assert {s['panel'] for s in m.specifications() if s['target']=='cap70_hours'}=={'revised'}


def test_single_component_has_no_known_network_test_rows():
    d=pd.DataFrame(dict(example=['01','01','02','02'],overlap_group=['a','a','b','c']))
    f=m.assign_within_folds(d)
    assert (f.iloc[:2]==-1).all()
    assert (f.iloc[2:]>=0).all()
