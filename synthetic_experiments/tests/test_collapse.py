import networkx as nx
import pytest
from synthetic_experiments.design import build_original
from synthetic_experiments.collapse import drawable_graph, collapse_selected
from hierarchy_level_definition.unit_detection.bifurcation_confluence_units import build_structural_units


def units(graph):
    node_ids = {n:i for i,n in enumerate(graph)}
    norm=nx.MultiDiGraph()
    lookup={}
    for i,(u,v,k,d) in enumerate(graph.edges(keys=True,data=True),1):
        e=(node_ids[u],node_ids[v],i)
        norm.add_edge(e[0],e[1],key=i,id_link=i,length=d['length_m'])
        lookup[e]=(u,v,k)
    return [(unit,{lookup[e] for e in unit.edge_set}) for unit in build_structural_units(norm)]


@pytest.mark.parametrize('family,selection,remaining,extent',[
    ('simple','full',0,None),('nested','inner',1,(5000,15000)),
    ('nested','full',0,None),('crossover','upstream',1,(8000,15000)),
    ('crossover','downstream',1,(5000,12000)),('crossover','full',0,None),
])
def test_realized_child_topology_and_controls(family,selection,remaining,extent):
    original=drawable_graph(build_original(family))
    detected=units(original)
    if selection=='full':
        selected=set().union(*(edges for _,edges in detected))
    elif selection=='inner':
        selected=min((edges for _,edges in detected),key=len)
    else:
        ordered=sorted((edges for _,edges in detected),key=lambda edges:min(original.nodes[e[0]]['x'] for e in edges))
        selected=ordered[0 if selection=='upstream' else 1]
    child=collapse_selected(original,selected)
    assert nx.is_directed_acyclic_graph(child)
    detected_child=units(child)
    assert len(detected_child)==remaining
    if extent:
        edges=detected_child[0][1]
        assert (min(child.nodes[e[0]]['x'] for e in edges),max(child.nodes[e[1]]['x'] for e in edges))==extent
    assert sum(d['w']*d['length_m'] for *_,d in child.edges(data=True))==pytest.approx(500*20000)
    for node in child:
        if child.in_degree(node) and child.out_degree(node):
            assert sum(d['w'] for *_,d in child.in_edges(node,data=True))==pytest.approx(sum(d['w'] for *_,d in child.out_edges(node,data=True)))
    for path in nx.all_simple_edge_paths(child,(0.,'main'),(20000.,'main')):
        assert sum(child.edges[e]['slope']*child.edges[e]['length_m'] for e in path)==pytest.approx(2)
    for u,v,k,d in child.edges(keys=True,data=True):
        assert d['plot_y'][0]==pytest.approx(child.nodes[u]['y'])
        assert d['plot_y'][-1]==pytest.approx(child.nodes[v]['y'])


@pytest.mark.parametrize('outer,inner,multiplier',[(.5,.5,1),(.2,.5,1),(.8,.2,1.5)])
def test_main_island_preserves_small_island_and_length(outer,inner,multiplier):
    from synthetic_experiments.design import Design
    from synthetic_experiments.collapse import nested_main_island_edges
    from synthetic_experiments.networks import validate_graph
    parent=drawable_graph(build_original('nested',Design(outer_a_fraction=outer,
        inner_fraction=inner,length_multiplier=multiplier)))
    child=collapse_selected(parent,nested_main_island_edges(parent))
    validate_graph(child)
    assert parent.number_of_edges()-parent.number_of_nodes()+1==2
    assert child.number_of_edges()-child.number_of_nodes()+1==1
    assert len(units(child))==1
    branch_edges=[d for u,v,d in child.edges(data=True)
        if child.nodes[u]['x']==8000 and child.nodes[v]['x']==12000]
    assert sorted(d['w'] for d in branch_edges)==pytest.approx(sorted([500*outer*inner,500*(1-outer*inner)]))
    survivor=next(d for d in branch_edges if d.get('kind')=='loop')
    assert survivor['length_m']==pytest.approx(4000*multiplier)
    assert survivor['slope']==pytest.approx(1e-4/multiplier)
    assert sorted(d['x'] for n,d in child.nodes(data=True) if child.in_degree(n)>1 or child.out_degree(n)>1)==[8000,12000]
    assert sum(d['w']*d['length_m'] for *_,d in child.edges(data=True))==pytest.approx(
        sum(d['w']*d['length_m'] for *_,d in parent.edges(data=True)))
