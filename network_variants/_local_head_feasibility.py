"""Read-only slope feasibility on a declared replacement with fixed exterior heads.

No optimization objective, routing, width/depth allocation, or geometry edits.
For a DAG, inspect every fixed-to-fixed path through free internal nodes. The
longest weighted path gives the most restrictive lower-drop constraint.
"""
import networkx as nx
import numpy as np
import pandas as pd


def head_series(frame):
    h = pd.concat(
        [
            frame[["u", "head_us"]].rename(columns={"u": "node", "head_us": "head"}),
            frame[["v", "head_ds"]].rename(columns={"v": "node", "head_ds": "head"}),
        ]
    )
    if (h.groupby("node")["head"].agg(lambda x: x.max() - x.min()) > 1e-9).any():
        raise ValueError("Reference shared-node heads disagree")
    return h.drop_duplicates("node").set_index("node")["head"]


def path_constraints(edges, fixed_heads, weights):
    """Return all consecutive-anchor longest paths, including parallel reaches."""
    graph = nx.DiGraph()
    graph.add_edges_from(zip(edges.u, edges.v))
    if not nx.is_directed_acyclic_graph(graph):
        raise ValueError("Replacement graph has a cycle")
    fixed = set(fixed_heads.index) & set(graph)
    order = list(nx.topological_sort(graph))
    outgoing = {v: [] for v in graph}
    for r, w in zip(edges.itertuples(), weights):
        outgoing[r.u].append((r.v, float(w), int(r.id_link)))
    # Each free vertex needs anchor control upstream and downstream.
    reached = set(fixed)
    for u in order:
        if u in reached:
            reached.update(graph.successors(u))
    backed = set(fixed)
    for u in reversed(order):
        if any(v in backed for v in graph.successors(u)):
            backed.add(u)
    if set(graph) - reached or set(graph) - backed:
        raise ValueError("Unanchored internal source/sink")
    rows = []
    for start in sorted(fixed):
        distance = {start: 0.0}
        paths = {start: []}
        for u in order:
            if u not in distance or (u != start and u in fixed):
                continue
            for v, w, lid in outgoing[u]:
                value = distance[u] + w
                if value > distance.get(v, -np.inf):
                    distance[v] = value
                    paths[v] = paths[u] + [lid]
        for end in sorted(fixed & set(distance) - {start}):
            drop = float(fixed_heads.loc[start] - fixed_heads.loc[end])
            length = distance[end]
            rows.append(
                dict(
                    upstream=int(start),
                    downstream=int(end),
                    head_upstream=float(fixed_heads.loc[start]),
                    head_downstream=float(fixed_heads.loc[end]),
                    available_drop_m=drop,
                    path_weight=length,
                    path_links=paths[end],
                    ratio=drop / length if length > 0 else None,
                )
            )
    return rows
