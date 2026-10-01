"""Analytical baseline collapse for synthetic examples (not raster reconstruction).

Selected edges are replaced by one x-monotone corridor. External attachments
retain their x station; widths sum across each selected cross-section. This
preserves baseline sum(WL), boundary heads, and surviving connections.
"""
import copy

import networkx as nx
import numpy as np

from .plotting import plot_network


def nested_main_island_edges(graph):
    """Boundary of the main island, excluding the surviving upper inner branch.

    This analytical face selection is specific to our nested geometry. It is
    not the outer topological unit, which also contains the small island.
    """
    loops = [(u, v, k, d) for u, v, k, d in graph.edges(keys=True, data=True)
             if d.get('kind') == 'loop']
    if (len(loops) != 2 or len({(u, v) for u, v, _, _ in loops}) != 1
            or sorted(d.get('curve', 0) for *_, d in loops) != [-1, 1]):
        raise ValueError('Main-island selection requires the original nested geometry')
    return {(u, v, k) for u, v, k, d in graph.edges(keys=True, data=True)
            if d.get('kind') == 'corridor'
            or (d.get('kind') == 'loop' and d.get('curve', 0) < 0)}


def drawable_graph(net, *, detailed=True):
    graph = copy.deepcopy(net.G)
    if not detailed:
        # Batch calculations need endpoint geometry only; avoid constructing Plotly figures.
        for node,d in graph.nodes(data=True):
            sign = {"A": 1, "B": -1}.get(d.get("tag"), 0)
            d["y"] = sign*(net.p.Y0+net.p.amp_corr)*np.sin(np.pi*(d["x"]-net.p.xs)/(net.p.xe-net.p.xs))
        for u,v,k,d in graph.edges(keys=True,data=True):
            d.update(plot_x=[graph.nodes[u]["x"],graph.nodes[v]["x"]],
                     plot_y=[graph.nodes[u]["y"],graph.nodes[v]["y"]],color="#64748b")
        return graph
    fig = plot_network(net)
    for node, x, y in zip(graph.nodes, fig.data[-1].x, fig.data[-1].y):
        graph.nodes[node].update(x=float(x), y=float(y))
    for trace, edge in zip(fig.data[:-1], graph.edges(keys=True)):
        graph.edges[edge].update(plot_x=list(trace.x), plot_y=list(trace.y), color=trace.line.color)
        u,v,_ = edge
        # Sampled legacy curves can stop just short of a split station.
        for index,node in ((0,u),(-1,v)):
            graph.edges[edge]["plot_x"][index] = graph.nodes[node]["x"]
            graph.edges[edge]["plot_y"][index] = graph.nodes[node]["y"]
    return graph


def collapse_selected(graph, selected):
    selected = set(selected)
    if not selected or not selected <= set(graph.edges(keys=True)):
        raise ValueError("Select existing edges")
    nodes = {n for u,v,k in selected for n in (u,v)}
    ordered = sorted(nodes, key=lambda n: graph.nodes[n]["x"])
    start, end = ordered[0], ordered[-1]
    x0, x1 = (graph.nodes[n]["x"] for n in (start,end))
    y0, y1 = (graph.nodes[n]["y"] for n in (start,end))
    h0, h1 = (graph.nodes[n]["head_m"] for n in (start,end))
    stations = sorted({graph.nodes[n]["x"] for n in nodes})
    child = nx.MultiDiGraph()
    mapping = {n: (graph.nodes[n]["x"], "collapsed") for n in nodes}
    for n,d in graph.nodes(data=True):
        if n not in nodes:
            child.add_node(n, **copy.deepcopy(d))
    for x in stations:
        t = (x-x0)/(x1-x0)
        child.add_node((x,"collapsed"), x=x, y=y0+t*(y1-y0),
                       head_m=h0+t*(h1-h0), tag="collapsed")
    for u,v,k,d in graph.edges(keys=True, data=True):
        if (u,v,k) in selected:
            continue
        a,b = mapping.get(u,u), mapping.get(v,v)
        data = copy.deepcopy(d)
        xx = np.asarray(data["plot_x"])
        t = (xx-graph.nodes[u]["x"])/(graph.nodes[v]["x"]-graph.nodes[u]["x"])
        dy0 = child.nodes[a]["y"]-graph.nodes[u]["y"]
        dy1 = child.nodes[b]["y"]-graph.nodes[v]["y"]
        data["plot_y"] = (np.asarray(data["plot_y"])+(1-t)*dy0+t*dy1).tolist()
        child.add_edge(a,b,key=k,**data)
    for i,(a,b) in enumerate(zip(stations[:-1],stations[1:])):
        mid = (a+b)/2
        width = sum(graph.edges[e]["w"] for e in selected
                    if graph.nodes[e[0]]["x"] < mid < graph.nodes[e[1]]["x"])
        u,v = (a,"collapsed"),(b,"collapsed")
        drop = child.nodes[u]["head_m"]-child.nodes[v]["head_m"]
        child.add_edge(u,v,key=f"collapsed_{i}", w=width, length_m=b-a,
                       length_base_m=b-a, length_factor=1., head_drop_m=drop,
                       slope=drop/(b-a), kind="collapsed", color="#7c3aed",
                       plot_x=[a,b], plot_y=[child.nodes[u]["y"],child.nodes[v]["y"]])
    return child
