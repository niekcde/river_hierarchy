"""Read-only slope feasibility on a declared replacement with fixed exterior heads.

No optimization objective, routing, width/depth allocation, or geometry edits.
For a DAG, inspect every fixed-to-fixed path through free internal nodes. The
longest weighted path gives the most restrictive lower-drop constraint.
"""
import networkx as nx
import numpy as np
import pandas as pd


def head_series(frame):
    h=pd.concat([frame[['u','head_us']].rename(columns={'u':'node','head_us':'head'}),
                 frame[['v','head_ds']].rename(columns={'v':'node','head_ds':'head'})])
    if (h.groupby('node')['head'].agg(lambda x:x.max()-x.min())>1e-9).any():
        raise ValueError('Reference shared-node heads disagree')
    return h.drop_duplicates('node').set_index('node')['head']


def path_constraints(edges, fixed_heads, weights):
    """Return all consecutive-anchor longest paths, including parallel reaches."""
    graph=nx.DiGraph();graph.add_edges_from(zip(edges.u,edges.v))
    if not nx.is_directed_acyclic_graph(graph):raise ValueError('Replacement graph has a cycle')
    fixed=set(fixed_heads.index)&set(graph);order=list(nx.topological_sort(graph))
    outgoing={v:[] for v in graph}
    for r,w in zip(edges.itertuples(),weights):outgoing[r.u].append((r.v,float(w),int(r.id_link)))
    # Each free vertex needs anchor control upstream and downstream.
    reached=set(fixed)
    for u in order:
        if u in reached:reached.update(graph.successors(u))
    backed=set(fixed)
    for u in reversed(order):
        if any(v in backed for v in graph.successors(u)):backed.add(u)
    if set(graph)-reached or set(graph)-backed:raise ValueError('Unanchored internal source/sink')
    rows=[]
    for start in sorted(fixed):
        distance={start:0.};paths={start:[]}
        for u in order:
            if u not in distance or (u!=start and u in fixed):continue
            for v,w,lid in outgoing[u]:
                value=distance[u]+w
                if value>distance.get(v,-np.inf):distance[v]=value;paths[v]=paths[u]+[lid]
        for end in sorted(fixed&set(distance)-{start}):
            drop=float(fixed_heads.loc[start]-fixed_heads.loc[end]);length=distance[end]
            rows.append(dict(upstream=int(start),downstream=int(end),head_upstream=float(fixed_heads.loc[start]),head_downstream=float(fixed_heads.loc[end]),available_drop_m=drop,path_weight=length,path_links=paths[end],ratio=drop/length if length>0 else None))
    return rows


def check_slopes(replacement, reference_heads, fixed_nodes, positions, *, floor_fraction=.01):
    fixed=set(fixed_nodes);nodes=set(replacement.u)|set(replacement.v);free=nodes-fixed
    h=reference_heads.loc[sorted(nodes&fixed)]
    positive=path_constraints(replacement,h,replacement.link_length_m.to_numpy())
    bad=[p for p in positive if p['available_drop_m']<=1e-10]
    # Match the current method's reference slope for each free-node component:
    # fixed boundary head span / reference-section distance span.
    graph=nx.DiGraph();graph.add_edges_from(zip(replacement.u,replacement.v))
    floor_records=[];invalid=[];weights={}
    for number,component in enumerate(nx.connected_components(graph.subgraph(free).to_undirected())):
        patch=replacement[replacement.u.isin(component)|replacement.v.isin(component)]
        boundary=(set(patch.u)|set(patch.v))-set(component)
        span=float(h.loc[list(boundary)].max()-h.loc[list(boundary)].min())
        extent=float(positions.loc[list(boundary)].max()-positions.loc[list(boundary)].min())
        if span<=0 or extent<=0:
            invalid.append(dict(component=number,head_span_m=span,reference_extent_m=extent));continue
        reference=span/extent
        paths=path_constraints(patch,h.loc[list(boundary)],patch.link_length_m.to_numpy()*reference)
        for p in paths:p.update(component=number,reference_slope=reference,required_drop_m=floor_fraction*p['path_weight'])
        floor_records.extend(paths)
        weights.update(dict(zip(patch.id_link,patch.link_length_m*reference*floor_fraction)))
    floor_bad=[p for p in floor_records if p['available_drop_m']<p['required_drop_m']-1e-9]
    # Production checks positive slopes, but not its relative floor, on fixed-fixed edges.
    status='nonpositive_fixed_head_path' if bad else 'invalid_floor_reference' if invalid else 'relative_floor_infeasible' if floor_bad else 'feasible'
    critical=min(positive,key=lambda p:p['ratio']) if positive else None
    critical_floor=min(floor_records,key=lambda p:p['ratio']) if floor_records else None
    return dict(status=status,free_nodes=len(free),boundary_nodes=len(nodes&fixed),positive_slopes_feasible=not bad,
                current_relative_floor_feasible=not bad and not invalid and not floor_bad,
                maximum_uniform_positive_slope=min(p['ratio'] for p in positive) if positive else None,
                maximum_floor_fraction=min(p['ratio'] for p in floor_records) if floor_records else None,
                critical_positive_path=critical,critical_floor_path=critical_floor,nonpositive_paths=bad,
                floor_conflicts=floor_bad,invalid_floor_components=invalid,
                fixed_fixed_replacement_links=int((replacement.u.isin(fixed)&replacement.v.isin(fixed)).sum()))
