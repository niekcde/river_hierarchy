"""Matched closed-domain predictors using the real-workflow resistance solver."""
import math
import networkx as nx
import pandas as pd

from hierarchy_level_definition.metrics.equivalent_resistance import compute_K_chezy, solve_bubble_equivalent_K
from network_variants.equivalent_transition_domain import extract_complete_directed_domain, audit_hydraulic_domain_closure
from .networks import digest

_CACHE={}


def tables(graph):
    nodes=sorted(graph,key=lambda n:graph.nodes[n]['x'])
    ids={n:i for i,n in enumerate(nodes)}
    rows=[dict(id_link=i,id_us_node=ids[u],id_ds_node=ids[v],length=d['length_m'],width=d['w'])
          for i,(u,v,k,d) in enumerate(graph.edges(keys=True,data=True),1)]
    return pd.DataFrame(rows), {graph.nodes[n]['x']:ids[n] for n in nodes}


def solve(domain,source,sink,settings):
    # Scale the resistance system to order one for numerical conditioning.
    frame=pd.DataFrame(dict(u=domain.id_us_node,v=domain.id_ds_node,
                            K=compute_K_chezy(domain.length.to_numpy(),domain.width.to_numpy(),settings.chezy_cz,settings.kb)))
    scale=float(frame.K.max())
    key=digest(dict(edges=frame.to_dict('records'),source=source,sink=sink))
    if key not in _CACHE:
        frame['K']=frame.K/scale
        result=solve_bubble_equivalent_K(None,frame,source,sink,smooth=False)
        if not result['solver_success'] or not math.isfinite(result['K_eq']):
            raise ValueError(f'Resistance solve failed: {result["solver_message"]}')
        _CACHE[key]=(float(result['K_eq'])*scale,float(result['residual_relative']))
    return _CACHE[key]


def predictor(parent,child,pair,settings):
    full,pnodes=tables(parent)
    after,cnodes=tables(child)
    a,b=pair['selection_start_m'],pair['selection_end_m']
    selected=extract_complete_directed_domain(full,pnodes[a],pnodes[b])
    closed=audit_hydraulic_domain_closure(selected,full,pnodes[a],pnodes[b])['is_closed']
    mode='closed_unit'
    if pair.get('selection_type') == 'island':
        # The enclosing outer unit is comparison context, not the island edit.
        # Both the removed main-island paths and surviving inner branch matter.
        L=max(d['x'] for _,d in parent.nodes(data=True))
        a,b=.25*L,.75*L
        mode='containing_unit_island_context'
    elif not closed:
        L=max(d['x'] for _,d in parent.nodes(data=True))
        a,b=.25*L,.75*L
        mode='containing_compound_context'
    elif pair['plot_category']=='compound unit compound':
        mode='closed_compound_unit'
    before=extract_complete_directed_domain(full,pnodes[a],pnodes[b])
    child_domain=extract_complete_directed_domain(after,cnodes[a],cnodes[b])
    for domain,whole,ids in ((before,full,pnodes),(child_domain,after,cnodes)):
        if not audit_hydraulic_domain_closure(domain,whole,ids[a],ids[b])['is_closed']:
            raise ValueError('Matched domain is not hydraulically closed')
    # Width-derived inlet exposure propagated through the original graph.
    node_by_x={d['x']:n for n,d in parent.nodes(data=True)}
    exposure={n:0. for n in parent}
    for n in nx.topological_sort(parent):
        if parent.in_degree(n)==0: exposure[n]=1.
        outgoing=list(parent.out_edges(n,keys=True,data=True))
        total=sum(d['w'] for *_,d in outgoing)
        for _,v,k,d in outgoing: exposure[v]+=exposure[n]*d['w']/total
    E=exposure[node_by_x[a]] if mode=='closed_unit' else 1.
    k0,res0=solve(before,pnodes[a],pnodes[b],settings)
    k1,res1=solve(child_domain,cnodes[a],cnodes[b],settings)
    ratio=k1/k0; fraction=float(before.length.sum()/full.length.sum())
    return dict(domain_mode=mode,domain_start_m=a,domain_end_m=b,K_before=k0,K_after=k1,
                Lambda_matched=ratio,length_fraction=fraction,width_fraction=E,
                resistance_predictor=fraction*E*(-math.log10(ratio)),
                length_exposure_lambda=fraction*E*ratio,
                resistance_residual_before=res0,resistance_residual_after=res1,
                parent_sum_wl=float((full.length*full.width).sum()),child_sum_wl=float((after.length*after.width).sum()),
                parent_total_channel_length_m=float(full.length.sum()),child_total_channel_length_m=float(after.length.sum()))
