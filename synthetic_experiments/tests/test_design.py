import networkx as nx
import pytest

from synthetic_experiments.design import (
    Design, FAMILIES, apply_heads_and_lengths, build_original, planned_states,
)


@pytest.mark.parametrize("family", FAMILIES)
def test_baseline_controls(family):
    net = build_original(family)
    assert sum(d["w"]*d["length_m"] for *_, d in net.G.edges(data=True)) == pytest.approx(500*20000)
    for path in nx.all_simple_edge_paths(net.G, (0., "main"), (20000., "main")):
        assert sum(net.G.edges[e]["length_m"] for e in path) == pytest.approx(20000)
        assert sum(net.G.edges[e]["slope"]*net.G.edges[e]["length_m"] for e in path) == pytest.approx(2)


def test_length_adjustment_preserves_path_heads():
    net = build_original("nested")
    edge = next(e for e in net.G.edges(keys=True) if net.G.edges[e]["kind"] == "loop")
    before = net.G.edges[edge]["slope"]
    apply_heads_and_lengths(net, 2, {edge: 1.5})
    assert net.G.edges[edge]["slope"] == pytest.approx(before/1.5)
    for path in nx.all_simple_edge_paths(net.G, (0., "main"), (20000., "main")):
        assert sum(net.G.edges[e]["slope"]*net.G.edges[e]["length_m"] for e in path) == pytest.approx(2)


def test_crossover_alternatives_are_siblings():
    states = planned_states("crossover", "case1")
    assert len(states) == 4
    assert all(s["parent_state_id"] == "original" for s in states[1:])
    assert states[-1]["selection_type"] == "compound"


def test_crossover_detector_finds_overlapping_not_nested_units():
    from hierarchy_level_definition.unit_detection.bifurcation_confluence_units import build_structural_units
    net = build_original("crossover")
    graph = nx.convert_node_labels_to_integers(net.G, label_attribute="synthetic_node")
    normalized = nx.MultiDiGraph()
    normalized.add_nodes_from(graph.nodes(data=True))
    for i, (u,v,d) in enumerate(graph.edges(data=True), 1):
        normalized.add_edge(u,v,key=i,id_link=i,length=d["length_m"])
    units = build_structural_units(normalized)
    assert len(units) == 2
    assert all(not u.parents and not u.children for u in units)
    assert len(units[0].edge_set & units[1].edge_set) == 1
    assert units[0].compound_bubble_id == units[1].compound_bubble_id


def test_invalid_fraction_rejected():
    with pytest.raises(ValueError):
        Design(outer_a_fraction=1)
