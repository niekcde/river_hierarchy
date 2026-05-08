"""Synthetic-network adapters for the shared RAPID layer."""

from __future__ import annotations

import networkx as nx


def _tag_to_y(tag: str) -> float:
    if tag == "A":
        return 1.0
    if tag == "B":
        return -1.0
    return 0.0


def _linestring_wkt(x1, y1, x2, y2) -> str:
    return f"LINESTRING ({x1:.6f} {y1:.6f}, {x2:.6f} {y2:.6f})"


def build_single_edge_graph(length_m: float, width_m: float) -> nx.MultiDiGraph:
    """Build a simple 2-node, 1-edge control graph for baseline runs."""
    G = nx.MultiDiGraph()
    u = 0
    v = 1
    x0 = 0.0
    x1 = float(length_m)
    y0 = 0.0
    G.add_node(u, x=x0, y=y0, node_type="source")
    G.add_node(v, x=x1, y=y0, node_type="outlet")
    geom = _linestring_wkt(x0, y0, x1, y0)
    G.add_edge(
        u,
        v,
        key="1",
        reach_id=1,
        width=float(width_m),
        length=float(length_m),
        geometry=geom,
    )
    return G


def rivernetwork_to_rapid_graph(net) -> nx.MultiDiGraph:
    """Convert a `RiverNetworkNX`-style network into a RAPID-ready graph."""
    G_src = net.G
    G = nx.MultiDiGraph()

    for n, data in G_src.nodes(data=True):
        x = float(data.get("x", 0.0))
        tag = data.get("tag", "main")
        y = _tag_to_y(tag)
        G.add_node(n, x=x, y=y, node_type="internal")

    edges = []
    for u, v, k, data in G_src.edges(keys=True, data=True):
        xu = float(G_src.nodes[u]["x"])
        xv = float(G_src.nodes[v]["x"])
        if xv <= xu:
            continue
        kind = data.get("kind", "")
        branch = data.get("branch", "")
        edges.append((xu, xv, branch, kind, str(k), u, v, data))
    edges.sort()

    for idx, (_, _, _, _, _, u, v, data) in enumerate(edges, start=1):
        xu = float(G_src.nodes[u]["x"])
        xv = float(G_src.nodes[v]["x"])
        yu = float(G.nodes[u]["y"])
        yv = float(G.nodes[v]["y"])
        length = float(xv - xu)
        w = float(data.get("w", 0.0))
        geom = _linestring_wkt(xu, yu, xv, yv)
        G.add_edge(
            u,
            v,
            key=str(idx),
            reach_id=int(idx),
            width=w,
            length=length,
            geometry=geom,
            kind=data.get("kind", ""),
            branch=data.get("branch", ""),
            from_branch=data.get("from_branch", ""),
            to_branch=data.get("to_branch", ""),
            curve=data.get("curve", 0),
        )

    for n in G.nodes:
        if G.in_degree(n) == 0:
            G.nodes[n]["node_type"] = "source"

    return G


__all__ = ["build_single_edge_graph", "rivernetwork_to_rapid_graph"]
