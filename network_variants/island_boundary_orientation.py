"""Conservative direction fallback for island replacements with exterior bypasses."""
from __future__ import annotations
import networkx as nx
import numpy as np
from shapely.geometry import LineString


def orient_with_exterior_constraints(links, *, inserted_link_ids, source_nodes, sink_nodes, parent_node_order):
    """Freeze exterior directions and degree-one interface roles before new edges.

    Parent-order harmonic values only rank unconstrained choices. A choice may
    not close a directed loop through retained exterior branches. Acceptance
    additionally requires no new graph source/sink and all interface roles.
    """
    inserted=set(map(int,inserted_link_ids));sources=set(source_nodes);sinks=set(sink_nodes)
    result=links.copy();new=result[result.id_link.isin(inserted)];fixed=result[~result.id_link.isin(inserted)]
    graph=nx.DiGraph();graph.add_edges_from((int(r.id_us_node),int(r.id_ds_node)) for r in fixed.itertuples())
    if not nx.is_directed_acyclic_graph(graph):raise ValueError('Exterior graph is cyclic')
    adjacency={}
    for r in new.itertuples():
        for n in [int(r.id_us_node),int(r.id_ds_node)]:adjacency.setdefault(n,[]).append(r)
    orientations={};forced=[]
    for n in sorted(sources|sinks):
        if n in sources&sinks:raise ValueError('Shared-role interface needs explicit direction treatment')
        edges=adjacency.get(n,[])
        if len(edges)!=1:raise ValueError(f'Fallback requires one replacement edge at boundary {n}, got {len(edges)}')
        r=edges[0];other=int(r.id_ds_node if r.id_us_node==n else r.id_us_node);uv=(n,other) if n in sources else (other,n);lid=int(r.id_link)
        if lid in orientations and orientations[lid]!=uv:raise ValueError('Conflicting boundary direction constraints')
        orientations[lid]=uv;graph.add_edge(*uv);forced.append(lid)
    if not nx.is_directed_acyclic_graph(graph):raise ValueError('Boundary constraints conflict with retained exterior directions')
    # Harmonic interpolation of stable parent order on the inserted graph.
    known={n:float(parent_node_order[n]) for n in adjacency if n in parent_node_order}
    unknown=sorted(set(adjacency)-set(known));idx={n:i for i,n in enumerate(unknown)};a=np.zeros((len(idx),len(idx)));b=np.zeros(len(idx))
    for n,i in idx.items():
        for r in adjacency[n]:
            other=int(r.id_ds_node if r.id_us_node==n else r.id_us_node);w=1/max(float(r.geometry.length),1e-6);a[i,i]+=w
            if other in idx:a[i,idx[other]]-=w
            else:b[i]+=w*known[other]
    if unknown:known.update(zip(unknown,np.linalg.solve(a,b)))
    remaining=[r for r in new.itertuples() if int(r.id_link) not in orientations]
    remaining.sort(key=lambda r:(-abs(known[int(r.id_us_node)]-known[int(r.id_ds_node)]),int(r.id_link)))
    reversals=[]
    for r in remaining:
        u,v=int(r.id_us_node),int(r.id_ds_node)
        preferred=(u,v) if (known[u],u)<(known[v],v) else (v,u)
        x,y=preferred
        if y in graph and x in graph and nx.has_path(graph,y,x):
            x,y=y,x;reversals.append(int(r.id_link))
        if y in graph and x in graph and nx.has_path(graph,y,x):raise ValueError('Neither orientation is acyclic')
        orientations[int(r.id_link)]=(x,y);graph.add_edge(x,y)
    for i,r in result[result.id_link.isin(inserted)].iterrows():
        u,v=orientations[int(r.id_link)]
        if int(r.id_us_node)!=u:result.at[i,'geometry']=LineString(list(r.geometry.coords)[::-1])
        result.at[i,'id_us_node']=u;result.at[i,'id_ds_node']=v
        if 'id_nodes' in result:result.at[i,'id_nodes']=f'{u}, {v}'
    if not nx.is_directed_acyclic_graph(graph):raise ValueError('Final graph cyclic')
    return result,dict(method='fixed_exterior_and_boundary_precedence',forced_interface_link_ids=sorted(set(forced)),cycle_preventing_orientation_choices=reversals,requires_full_network_terminal_and_cycle_validation=True)
