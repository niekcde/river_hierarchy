import pandas as pd
import pytest
from network_variants.equivalent_transition_domain import complete_anchor_map_from_unchanged_lineage


def inputs():
    parent = pd.DataFrame(dict(id_link=[64], id_us_node=[45], id_ds_node=[42]))
    child = pd.DataFrame(dict(id_link=[629], id_us_node=[416], id_ds_node=[42]))
    lineage = pd.DataFrame(dict(id_link=[629], lineage_type=['unchanged_1to1'],
        dominant_parent_link_id=[64], matched_parent_us_node=[45], matched_parent_ds_node=[42]))
    return dict(parent_links=parent, child_links=child, lineage=lineage,
                parent_to_child_node={42:42,415:416}, anchors=[45])


def test_retained_link_recovers_anchor_and_ignores_new_canonical_ids():
    mapping, audit = complete_anchor_map_from_unchanged_lineage(**inputs())
    assert mapping == {42:42,45:416}
    assert audit[0]['evidence'] == [dict(parent_link=64,child_link=629)]


@pytest.mark.parametrize('change', ['opposite_missing','changed_link','wrong_endpoints'])
def test_unproven_anchor_is_not_recovered(change):
    args=inputs()
    if change=='opposite_missing': args['parent_to_child_node']={}
    elif change=='changed_link': args['lineage']['lineage_type']='collapsed'
    else: args['lineage']['matched_parent_ds_node']=99
    mapping,audit=complete_anchor_map_from_unchanged_lineage(**args)
    assert 45 not in mapping
    assert not audit


def test_ambiguous_lineage_fails():
    args=inputs()
    args['child_links']=pd.concat([args['child_links'],pd.DataFrame(dict(id_link=[630],id_us_node=[417],id_ds_node=[42]))])
    args['lineage']=pd.concat([args['lineage'],args['lineage'].assign(id_link=630)])
    with pytest.raises(ValueError,match='Ambiguous'):
        complete_anchor_map_from_unchanged_lineage(**args)
