"""Construct comparisons and canonical hydraulic networks for reusable routing."""
import copy
import hashlib
import json

import networkx as nx
import numpy as np

from .collapse import collapse_selected, drawable_graph, nested_main_island_edges
from .design import Design, FAMILIES, build_original


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()[:24]


def from_parameters(parameters):
    p = dict(parameters)
    p['head_drop_m'] = p.pop('reference_slope') * p['total_length_m']
    return Design(**p)


def detect_units(graph):
    from hierarchy_level_definition.unit_detection.bifurcation_confluence_units import build_structural_units
    ids = {n:i for i,n in enumerate(graph)}
    norm = nx.MultiDiGraph()
    lookup = {}
    for i,(u,v,k,d) in enumerate(graph.edges(keys=True,data=True),1):
        e = (ids[u], ids[v], i)
        norm.add_edge(e[0],e[1],key=i,id_link=i,length=d['length_m'])
        lookup[e] = (u,v,k)
    return [(unit, {lookup[e] for e in unit.edge_set}) for unit in build_structural_units(norm)]


def make_states(parameters):
    design = from_parameters(parameters)
    states = {f'{f}/original':drawable_graph(build_original(f,design), detailed=False) for f in FAMILIES}
    pairs = []
    for family in ('simple','nested','crossover'):
        parent_id = f'{family}/original'
        parent = states[parent_id]
        units = detect_units(parent)
        all_edges = set().union(*(edges for _,edges in units))
        if family == 'simple':
            choices = [('full',all_edges,'single simple')]
        elif family == 'nested':
            choices = [('inner_only', min((e for _,e in units),key=len),'nested unit'),
                       ('main_island_only',nested_main_island_edges(parent),'Main island'),
                       ('full',all_edges,'compound unit compound')]
        else:
            ordered = sorted((e for _,e in units), key=lambda edges:min(parent.nodes[e[0]]['x'] for e in edges))
            choices = [('upstream_only',ordered[0],'part of compound unit'),
                       ('downstream_only',ordered[1],'part of compound unit'),
                       ('full',all_edges,'compound unit compound')]
        for label,selected,category in choices:
            child_id = f'{family}/{label}'
            states[child_id] = collapse_selected(parent, selected)
            nodes = {n for u,v,k in selected for n in (u,v)}
            pairs.append(dict(family=family, parent_state=parent_id, child_state=child_id,
                              selection_type='island' if label=='main_island_only' else 'topological',
                              selection_start_m=min(parent.nodes[n]['x'] for n in nodes),
                              selection_end_m=max(parent.nodes[n]['x'] for n in nodes),
                              plot_category=category))
    for graph in states.values():
        validate_graph(graph)
    return states, pairs


def validate_graph(graph):
    if not nx.is_directed_acyclic_graph(graph):
        raise ValueError('Synthetic network must be a DAG')
    sources = [n for n in graph if graph.in_degree(n)==0]
    sinks = [n for n in graph if graph.out_degree(n)==0]
    if len(sources)!=1 or len(sinks)!=1:
        raise ValueError('Expected exactly one inlet and outlet')
    for u,v,k,d in graph.edges(keys=True,data=True):
        if not all(np.isfinite(d[key]) and d[key]>0 for key in ('w','length_m','slope')):
            raise ValueError('Invalid edge hydraulics')
        if not np.isclose(d['slope']*d['length_m'],graph.nodes[u]['head_m']-graph.nodes[v]['head_m']):
            raise ValueError('Inconsistent head drop')
    for node in graph:
        if graph.in_degree(node) and graph.out_degree(node):
            incoming = sum(d['w'] for *_,d in graph.in_edges(node,data=True))
            outgoing = sum(d['w'] for *_,d in graph.out_edges(node,data=True))
            if not np.isclose(incoming,outgoing):
                raise ValueError('Width budget not conserved at junction')


def canonical_network(graph):
    """Remove hydraulically invisible degree-two breaks before shared segmentation.

    Equal width and slope series reaches can merge. Node/edge names, diagram y,
    and family labels do not affect routing and are excluded from the key.
    """
    g = copy.deepcopy(graph)
    changed = True
    while changed:
        changed=False
        for node in list(g):
            if g.in_degree(node)!=1 or g.out_degree(node)!=1:
                continue
            u,_,ka,a = next(iter(g.in_edges(node,keys=True,data=True)))
            _,v,kb,b = next(iter(g.out_edges(node,keys=True,data=True)))
            if not all(np.isclose(a[key],b[key],rtol=1e-12,atol=0) for key in ('w','slope')):
                continue
            data=dict(a, length_m=a['length_m']+b['length_m'])
            g.remove_node(node)
            g.add_edge(u,v,**data)
            changed=True
            break
    ordered=sorted(g,key=lambda n:g.nodes[n]['x'])
    xs=[g.nodes[n]['x'] for n in ordered]
    if len(set(xs))!=len(xs):
        raise ValueError('Canonical baseline adapter requires distinct junction x stations')
    ids={n:i for i,n in enumerate(ordered)}
    # Ignore arithmetic noise from repeated series sums (14 significant digits).
    number = lambda value: float(format(float(value), '.14g'))
    edges=sorted([ids[u],ids[v],number(d['length_m']),number(d['w']),number(d['slope'])]
                  for u,v,d in g.edges(data=True))
    return dict(nodes=[[number(g.nodes[n]['x']),number(g.nodes[n]['head_m'])] for n in ordered],edges=edges)
