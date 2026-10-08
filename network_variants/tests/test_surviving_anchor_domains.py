from dataclasses import replace
import pandas as pd
import pytest
from network_variants.equivalent_transition_domain import HydraulicDomain, enclose_domain_at_surviving_anchors


def inputs():
    parent = pd.DataFrame([(1,0,1), (2,1,2), (3,2,3), (4,2,3), (5,3,4), (6,1,4), (7,4,5)],
                          columns=["id_link", "id_us_node", "id_ds_node"])
    child = pd.DataFrame([(10,0,11), (20,11,14), (30,11,14), (40,14,15)], columns=parent.columns)
    original = HydraulicDomain("closed_unit", 2, 3, parent[parent.id_link.isin([3,4])], (9,), None, True, ())
    return dict(parent_domain=original, parent_links=parent, child_links=child,
                parent_to_child_node={0:0,1:11,4:14,5:15})


def test_removed_ports_use_smallest_shared_closed_enclosure():
    args = inputs()
    p, c, audit = enclose_domain_at_surviving_anchors(**args)
    assert (p.source_node, p.sink_node) == (1,4)
    assert (c.source_node, c.sink_node) == (11,14)
    assert set(p.links.id_link) == {2,3,4,5,6}
    assert set(c.links.id_link) == {20,30}
    assert audit["added_parent_link_ids"] == [2,5,6]
    assert p.selected_unit_ids == (9,)
    pd.testing.assert_frame_equal(args['parent_links'], inputs()['parent_links'])


def test_external_flow_exchange_is_not_accepted():
    args = inputs()
    args['parent_to_child_node'] = {1:11,4:14}
    args['child_links'] = pd.concat([args['child_links'], pd.DataFrame([(50,0,14)], columns=args['child_links'].columns)])
    with pytest.raises(ValueError, match="No local closed comparison"):
        enclose_domain_at_surviving_anchors(**args)


def test_whole_network_substitution_is_rejected():
    args = inputs()
    args['parent_to_child_node'] = {0:0,5:15}
    with pytest.raises(ValueError, match="whole-network substitution is disabled"):
        enclose_domain_at_surviving_anchors(**args)


def test_existing_boundaries_keep_original_domain():
    args=inputs()
    args['child_links']=args['parent_links'].copy()
    args['parent_to_child_node']={n:n for n in range(6)}
    p,c,audit=enclose_domain_at_surviving_anchors(**args)
    assert p is args['parent_domain']
    assert set(c.links.id_link)=={3,4}
    assert audit=={'expanded':False}
