from __future__ import annotations
import argparse
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
import random

import geopandas as gpd
import networkx as nx
from shapely.geometry import LineString, Point
import plotly.graph_objects as go
import numpy as np
from collections import defaultdict, deque
import math


# ============================================================
# Data structures
# ============================================================

@dataclass
class Node:
    id: int
    x: float
    y: float = 0.0
    node_type: str = "internal"  # source, bifurcation, confluence, outlet, internal
    upstream_edges: List[int] = field(default_factory=list)
    downstream_edges: List[int] = field(default_factory=list)


@dataclass
class Edge:
    id: int
    upstream_node: int
    downstream_node: int
    x_start: float
    x_end: float
    centerline_length: float = 0.0
    width: float = 1.0
    geometry: Optional[List[Tuple[float, float]]] = None

    # reach indices along the 1D domain
    reach_start: int = 0
    reach_end: int = 0

    # how many times this edge's lineage has been split (nested)
    split_depth: int = 0


# ============================================================
# Deterministic River Network with Nested Diamonds (reach-based)
# ============================================================
class RiverNetwork:
    def __init__(self, domain_length: float, root_width: float, min_ratio: float,
                 min_width: float = 20.0,
                 min_width_difference: float = 10.0):
        """
        Added parameters:
        - min_width: smallest allowable width for any channel
        - min_width_difference: minimum absolute difference between bifurcating widths
        """
        self.domain_length = float(domain_length)
        self.root_width = float(root_width)
        self.min_ratio = float(min_ratio)

        self.min_width = float(min_width)
        self.min_width_difference = float(min_width_difference)

        # Reach length and count
        self.reach_length = self.root_width * self.min_ratio
        self.n_total_reaches = max(int(self.domain_length // self.reach_length), 1)
        self.effective_length = self.n_total_reaches * self.reach_length

        self.nodes: Dict[int, Node] = {}
        self.edges: Dict[int, Edge] = {}

        self._next_node = 1
        self._next_edge = 1
        self.trunk_edge_id: Optional[int] = None

        self._initialize_trunk()

        # Assign the starting width to the trunk
        self.edges[self.trunk_edge_id].width = self.root_width

    # ============================================================
    # edge and node helpers
    # ============================================================
    def _new_node(self, x: float) -> int:
        """
        Create a node at location x.
        No deduplication by x; each call creates a new node.
        """
        nid = self._next_node
        self._next_node += 1
        self.nodes[nid] = Node(id=nid, x=x)
        return nid

    def _new_edge(self, up: int, dn: int, x0: float, x1: float, split_depth: int = 0,
                  width:float = 1) -> int:
        """
        Create an edge between existing nodes with given x-coordinates.

        reach_start/reach_end are inferred from x0, x1 and reach_length
        (but may be overridden afterward where needed).
        """
        eid = self._next_edge
        self._next_edge += 1

        # infer reach indices
        rs = int(round(x0 / self.reach_length))
        re = int(round(x1 / self.reach_length))

        e = Edge(
            id=eid,
            upstream_node=up,
            downstream_node=dn,
            x_start=x0,
            x_end=x1,
            centerline_length=(x1 - x0),
            width=width,
            reach_start=rs,
            reach_end=re,
            split_depth=split_depth,
        )

        self.edges[eid] = e
        self.nodes[up].downstream_edges.append(eid)
        self.nodes[dn].upstream_edges.append(eid)
        return eid

    # ============================================================
    # 0. Initialize trunk
    # ============================================================
    def _initialize_trunk(self):
        """
        Initialize a single trunk from x=0 to x=effective_length, spanning all reaches.
        """
        n0 = self._new_node(0.0)
        n1 = self._new_node(self.effective_length)

        self.nodes[n0].node_type = "source"
        self.nodes[n1].node_type = "outlet"

        eid = self._new_edge(n0, n1, 0.0, self.effective_length, split_depth=0)
        # trunk spans all reaches [0, n_total_reaches]
        self.edges[eid].reach_start = 0
        self.edges[eid].reach_end = self.n_total_reaches

        self.trunk_edge_id = eid
    
        # ============================================================
    
        # ============================================================
    
    # ============================================================
    # Subdivide edges into 1-reach segments
    # ============================================================
    def subdivide_edges_into_reaches(self):
        """
        After all bifurcations are created, subdivide each edge that spans
        multiple reaches into a chain of edges, each exactly 1 reach long.

        This preserves topology (same upstream/downstream nodes at ends),
        but introduces internal nodes along the edge.
        """
        original_edges = list(self.edges.items())

        for eid, e in original_edges:
            rs, re = e.reach_start, e.reach_end
            nR = re - rs
            if nR <= 1:
                continue  # already 1 reach

            up_node = e.upstream_node
            down_node = e.downstream_node

            # Remove original edge from adjacency
            if eid in self.nodes[up_node].downstream_edges:
                self.nodes[up_node].downstream_edges.remove(eid)
            if eid in self.nodes[down_node].upstream_edges:
                self.nodes[down_node].upstream_edges.remove(eid)

            if eid in self.edges:
                del self.edges[eid]


            # Build chain of 1-reach edges
            prev_node = up_node
            for i in range(rs, re):
                if i == re - 1:
                    next_node = down_node
                else:
                    x_mid = (i + 1) * self.reach_length
                    next_node = self._new_node(x_mid)

                x0 = i * self.reach_length
                x1 = (i + 1) * self.reach_length

                seg_id = self._new_edge(prev_node, next_node, x0, x1, split_depth=e.split_depth,
                                        width = e.width)
                seg_edge = self.edges[seg_id]
                seg_edge.reach_start = i
                seg_edge.reach_end = i + 1

                prev_node = next_node
    
    
    # ============================================================
    # Geometry: diamond-like layout
    # ============================================================

    def generate_centerlines(self, n_midpoints: int = 7):
        """
        Generate simple diamond-like centerline geometries for all edges.
        We use a topological x-position propagation and simple y-offsets
        for branching and parallels.
        """
        # print("Generate centerlines")

        # Reset y
        for n in self.nodes.values():
            n.y = 0.0

        # --- Build outgoing, incoming, parallel edge registries --- #
        outgoing = defaultdict(list)
        incoming = defaultdict(list)
        out_by_node = defaultdict(list)
        parallel = defaultdict(list)

        for eid, e in self.edges.items():
            u = e.upstream_node
            v = e.downstream_node
            L = e.centerline_length

            outgoing[u].append((v, eid, L))
            incoming[v].append(u)
            out_by_node[u].append(eid)
            parallel[(u, v)].append(eid)

        # ============================================================
        # STEP 1 — Compute X positions (topological propagation)
        # ============================================================

        x = defaultdict(float)
        visited = set()

        def assign_x(node):
            if node in visited:
                return
            visited.add(node)
            base_x = x[node]
            for v, eid, length in outgoing[node]:
                x[v] = max(x[v], base_x + length)
                assign_x(v)

        if outgoing:
            root = min(outgoing.keys(), key=lambda nid: self.nodes[nid].x)
            assign_x(root)

        # ============================================================
        # STEP 2 — Find branching nodes
        # ============================================================

        branch_nodes = {u for u, outs in outgoing.items() if len(outs) > 1}

        # ============================================================
        # STEP 3 — Propagate branching upstream (BFS)
        # ============================================================

        branch_reachable = defaultdict(bool)
        queue = deque(branch_nodes)

        while queue:
            node = queue.popleft()
            if branch_reachable[node]:
                continue
            branch_reachable[node] = True
            for parent in incoming[node]:
                queue.append(parent)

        # ============================================================
        # STEP 4 — Any edge whose DOWNSTREAM node leads to a branch must offset
        # ============================================================

        offset_groups = defaultdict(list)
        for eid, e in self.edges.items():
            if branch_reachable[e.downstream_node]:
                offset_groups[e.upstream_node].append(eid)

        # ============================================================
        # STEP 5 — Assign offsets
        # ============================================================

        def spaced_offsets(n, base):
            if n == 1:
                return [0]
            k = n // 2
            return [(i - k) * base * 2 for i in range(n)]

        y_offset: Dict[int, float] = {}

        for u, eids in offset_groups.items():
            offs = spaced_offsets(len(eids), 20.0)
            for eid, off in zip(eids, offs):
                y_offset[eid] = off

        for (u, v), eids in parallel.items():
            offs = spaced_offsets(len(eids), 5.0)
            for eid, off in zip(eids, offs):
                y_offset[eid] = y_offset.get(eid, 0.0) + off

        # ============================================================
        # STEP 6 — Create diamond-shaped geometry
        # ============================================================

        coords: Dict[int, List[Tuple[float, float]]] = {}

        for eid, e in self.edges.items():
            u = e.upstream_node
            v = e.downstream_node
            L = e.centerline_length

            x1 = x[u]
            x2 = x[v]

            off = y_offset.get(eid, 0.0)
            eps = L * 0.01 if L > 0 else 0.01
            xm = (x1 + x2) / 2.0

            coords[eid] = [
                (x1, 0.0),
                (x1 + eps, off),
                (xm, off),
                (x2, 0.0),
            ]

            self.edges[eid].geometry = coords[eid]
    
    
    # ============================================================
    # Node classification
    # ============================================================
    def classify_nodes(self):
        for nid, n in self.nodes.items():
            indeg = len(n.upstream_edges)
            outdeg = len(n.downstream_edges)
            if indeg == 0 and outdeg > 0:
                n.node_type = "source"
            elif outdeg == 0:
                n.node_type = "outlet"
            elif indeg == 1 and outdeg == 2:
                n.node_type = "bifurcation"
            elif indeg == 2 and outdeg == 1:
                n.node_type = "confluence"
            else:
                n.node_type = "internal"

    # ============================================================
    # Reach-based helpers
    # ============================================================

    def _edge_num_reaches(self, e: Edge) -> int:
        """
        Number of reaches spanned by this edge.
        """
        return max(e.reach_end - e.reach_start, 0)
    

    def _can_split_edge_by_reaches(self, e: Edge, rL: int) -> bool:
        """
        Check if edge e can be split such that:
            [up ... B]  (rL reaches)
            [B ... C]   (>= 1 reach)
            [C ... dn]  (rL reaches)

        i.e. total reaches >= 2*rL + 1.
        """
        return self._edge_num_reaches(e) >= 2 * rL + 1
    # ============================================================
    # 1. Width splitting with constraints
    # ============================================================
    # def random_width_split(self, W: float, rng: random.Random) -> Tuple[float, float]:
    #     """
    #     Produces w1, w2 satisfying:
    #     - w1 + w2 = W
    #     - w1 >= min_width
    #     - w2 >= min_width
    #     - |w1 – w2| >= min_width_difference
    #     """

    #     Wmin = self.min_width
    #     Dmin = self.min_width_difference

    #     # Check feasibility: total width must allow both constraints
    #     if W < 2*Wmin:
    #         # impossible case
    #         return Wmin, Wmin

    #     # feasible range for w1:
    #     # w1 ∈ [Wmin, W - Wmin]
    #     low = Wmin
    #     high = W - Wmin

    #     # but must satisfy |w1 – (W - w1)| >= Dmin
    #     # i.e. |2*w1 - W| >= Dmin   → two feasible zones

    #     # zone A: w1 <= (W - Dmin) / 2
    #     zoneA_max = (W - Dmin) / 2

    #     # zone B: w1 >= (W + Dmin) / 2
    #     zoneB_min = (W + Dmin) / 2

    #     feasible_intervals = []

    #     # Add interval A if it intersects [low, high]
    #     if low <= zoneA_max:
    #         feasible_intervals.append((low, min(zoneA_max, high)))

    #     # Add interval B if it intersects [low, high]
    #     if zoneB_min <= high:
    #         feasible_intervals.append((max(zoneB_min, low), high))

    #     if not feasible_intervals:
    #         # No feasible random split — force a deterministic wide split
    #         w1 = low
    #         w2 = W - w1
    #         return w1, w2

    #     # Choose one interval uniformly
    #     interval = rng.choice(feasible_intervals)
    #     w1 = interval[0] + rng.random() * (interval[1] - interval[0])
    #     w2 = W - w1

    #     return w1, w2
    def random_width_split(self, W: float, rng: random.Random) -> Tuple[float, float]:
        """
        Randomly split width W into (w1, w2) such that:

        - w1 and w2 are multiples of width_step (default = 10)
        - w1 >= min_width
        - w2 >= min_width
        - w1 + w2 = W
        """

        Wmin = self.min_width
        step = self.min_width_difference  # repurposed as step size, e.g. 10

        # If parent width is too small, clamp
        if W < 2 * Wmin:
            return Wmin, Wmin

        # Generate all feasible w1 values
        valid_w1 = []
        w1 = Wmin

        while w1 <= W - Wmin:
            if (w1 % step == 0) and ((W - w1) % step == 0):
                valid_w1.append(w1)
            w1 += step

        if not valid_w1:
            # Fallback — assign minimal widths
            return Wmin, W - Wmin

        # Choose a random valid split
        w1 = rng.choice(valid_w1)
        w2 = W - w1

        return w1, w2


    # ============================================================
    # 2. Modified bifurcation: assign widths immediately
    # ============================================================
    def _insert_first_diamond(self, rL: int, rng: random.Random) -> bool:
        """
        First split of the trunk edge. Includes width assignment.
        """
        if self.trunk_edge_id not in self.edges:
            return False

        e = self.edges[self.trunk_edge_id]
        parent_width = e.width

        # --- geometric checks identical to original ---
        nR = self._edge_num_reaches(e)
        if nR < 2 * rL + 1:
            return False

        B_idx = e.reach_start + rL
        C_idx = e.reach_end - rL

        xb = B_idx * self.reach_length
        xc = C_idx * self.reach_length

        up = self.nodes[e.upstream_node]
        dn = self.nodes[e.downstream_node]

        # Remove original edge
        up.downstream_edges.remove(e.id)
        dn.upstream_edges.remove(e.id)
        del self.edges[e.id]

        # New nodes
        B = self._new_node(xb)
        C = self._new_node(xc)
        self.nodes[B].node_type = "bifurcation"
        self.nodes[C].node_type = "confluence"

        base_depth = e.split_depth

        # Upstream trunk piece
        up_id = self._new_edge(up.id, B, e.x_start, xb, split_depth=base_depth)
        self.edges[up_id].width = parent_width

        # Downstream trunk piece
        dn_id = self._new_edge(C, dn.id, xc, e.x_end, split_depth=base_depth)
        # (downstream width = total width)
        self.edges[dn_id].width = parent_width

        # --- stochastic width assignment ---
        w1, w2 = self.random_width_split(parent_width, rng)

        # First parallel branch
        m1 = self._new_edge(B, C, xb, xc, split_depth=base_depth + 1)
        self.edges[m1].width = w1

        # Second parallel branch
        m2 = self._new_edge(B, C, xb, xc, split_depth=base_depth + 1)
        self.edges[m2].width = w2

        self.trunk_edge_id = None
        return True


    def _split_edge(self, host_edge_id: int, rL: int, rng: random.Random):
        """
        Same as above, but inside existing diamonds.
        """
        if host_edge_id not in self.edges:
            return None

        e = self.edges[host_edge_id]
        Wparent = e.width

        # geometric feasibility
        if not self._can_split_edge_by_reaches(e, rL):
            return None

        B_idx = e.reach_start + rL
        C_idx = e.reach_end - rL

        xb = B_idx * self.reach_length
        xc = C_idx * self.reach_length

        up = self.nodes[e.upstream_node]
        dn = self.nodes[e.downstream_node]

        # Remove host
        up.downstream_edges.remove(e.id)
        dn.upstream_edges.remove(e.id)
        del self.edges[e.id]

        # New nodes
        B = self._new_node(xb)
        C = self._new_node(xc)
        self.nodes[B].node_type = "bifurcation"
        self.nodes[C].node_type = "confluence"

        base_depth = e.split_depth

        # Upstream piece inherits parent width
        e_up = self._new_edge(up.id, B, e.x_start, xb, split_depth=base_depth)
        self.edges[e_up].width = Wparent

        # Downstream piece inherits parent width
        e_dn = self._new_edge(C, dn.id, xc, e.x_end, split_depth=base_depth)
        self.edges[e_dn].width = Wparent

        # --- stochastic widths ---
        w1, w2 = self.random_width_split(Wparent, rng)

        # Two parallel edges
        m1 = self._new_edge(B, C, xb, xc, split_depth=base_depth + 1)
        self.edges[m1].width = w1

        m2 = self._new_edge(B, C, xb, xc, split_depth=base_depth + 1)
        self.edges[m2].width = w2

        return m1, m2


    # ============================================================
    # 3. Modified selection of candidate edges
    # ============================================================
    def generate_bifurcations(self, levels: int, reaches_before_break: int, rng: random.Random) -> int:
        """
        Adds width-based restrictions.
        """
        placed = 0
        rL = reaches_before_break

        # ------------------ Level 0 ------------------
        if not self._insert_first_diamond(rL, rng):
            return 0
        placed += 1

        # ------------------ Nested levels ------------------
        for level in range(1, levels):
            candidates = []

            # group edges by parallel pairs
            groups = defaultdict(list)
            for eid, e in self.edges.items():
                key = (e.upstream_node, e.downstream_node)
                groups[key].append(eid)

            for (u, v), eids in groups.items():
                rep = self.edges[eids[0]]

                # must be bifurcation→confluence
                if not (self.nodes[u].node_type == "bifurcation" and
                        self.nodes[v].node_type == "confluence"):
                    continue

                # must satisfy reach rule
                if not self._can_split_edge_by_reaches(rep, rL):
                    continue
                
                # must satisfy width rule
                if rep.width <= self.min_width:
                    continue
                
                # candidate scoring: (depth, -reach_span, eid)
                candidates.append((rep.split_depth,
                                   -(rep.reach_end - rep.reach_start),
                                   rep.id))

            if not candidates:
                raise "Error: No candidates available"

            candidates.sort()
            _, _, eid = candidates[0]
            print()
            result = self._split_edge(eid, rL, rng)
            if result:
                placed += 1

        return placed


    # ============================================================
    # 4. assign_widths disabled
    # ============================================================

    def assign_widths(self, root_width: float):
        """Width assignment now handled during bifurcation."""
        pass


# ============================================================
# GIS helpers
# ============================================================

def edges_to_gdf(net: RiverNetwork, crs=None):
    rec = []
    for e in net.edges.values():
        if e.geometry:
            print(e.width)
            rec.append(
                {
                    "edge_id": e.id,
                    "upstream": e.upstream_node,
                    "downstream": e.downstream_node,
                    "width": e.width,
                    "length": e.centerline_length,
                    "reach_start": e.reach_start,
                    "reach_end": e.reach_end,
                    "n_reaches": e.reach_end - e.reach_start,
                    "split_depth": e.split_depth,
                    "geometry": LineString(e.geometry),
                }
            )
    return gpd.GeoDataFrame(rec, geometry="geometry", crs=crs)


def nodes_to_gdf(net: RiverNetwork, crs=None):
    rec = []
    for n in net.nodes.values():
        rec.append(
            {
                "node_id": n.id,
                "x": n.x,
                "y": n.y,
                "node_type": n.node_type,
                "geometry": Point(n.x, n.y),
            }
        )
    return gpd.GeoDataFrame(rec, geometry="geometry", crs=crs)

# ============================================================
# Plotly
# ============================================================

def plot_network_plotly(net: RiverNetwork, title: str = "", show: bool = True, html: Optional[str] = None):
    fig = go.Figure()
    for e in net.edges.values():
        if not e.geometry:
            continue
        xs = [p[0] for p in e.geometry]
        ys = [p[1] for p in e.geometry]
        fig.add_trace(
            go.Scatter(
                x=xs,
                y=ys,
                mode="lines",
                line=dict(width=max(1, e.width / 10)),
                hoverinfo="text",
                text=(
                    f"Edge {e.id}"
                    f"<br>w={e.width:.2f}"
                    f"<br>R={e.reach_end - e.reach_start}"
                    f"<br>reach[{e.reach_start},{e.reach_end})"
                    f"<br>depth={e.split_depth}"
                ),
            )
        )
    fig.update_layout(
        title=title,
        yaxis=dict(scaleanchor="x", scaleratio=1),
        showlegend=False,
        template="plotly_white",
    )
    if html:
        fig.write_html(html)
    if show:
        fig.show()


# ============================================================
# CLI
# ============================================================

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--domain-length", type=float, default=1000.0)
    p.add_argument("--n-bifurcations", type=int, default=4)
    p.add_argument(
        "--reaches-per-break",
        type=int,
        default=1,
        help="Number of reaches before each break (rL).",
    )
    p.add_argument(
        "--min-length-ratio",
        type=float,
        default=5.0,
        help="Defines reach_length = root_width * min_length_ratio (only for info).",
    )
    p.add_argument("--root-width", type=float, default=100.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--gpkg", type=str, default="river_network.gpkg")
    p.add_argument("--geojson", type=str, default="river_network.geojson")
    p.add_argument("--no-plot", action="store_true")
    p.add_argument("--html-plot", type=str, default=None)
    p.add_argument("--crs", type=str, default="EPSG:3857")
    return p


def main():
    args = parse_args().parse_args()
    rng = random.Random(args.seed)

    # Build network with reach-based discretization
    net = RiverNetwork(
        domain_length=args.domain_length,
        root_width=args.root_width,
        min_ratio=args.min_length_ratio,
    )

    placed = net.generate_bifurcations(
        levels=args.n_bifurcations,
        reaches_before_break=args.reaches_per_break,
        rng=rng,
    )

    # Subdivide all edges into 1-reach segments
    net.subdivide_edges_into_reaches()

    # net.assign_widths(args.root_width)
    net.classify_nodes()
    net.generate_centerlines(n_midpoints=7)

    edges = edges_to_gdf(net, crs=args.crs)
    nodes = nodes_to_gdf(net, crs=args.crs)

    if args.gpkg:
        edges.to_file(args.gpkg, driver="GPKG", layer="edges")
        nodes.to_file(args.gpkg, driver="GPKG", layer="nodes")

    if args.geojson:
        edges.to_file(args.geojson, driver="GeoJSON")

    if not args.no_plot:
        plot_network_plotly(
            net,
            title=(
                f"Nested Diamond Network (placed {placed}) - "
                f"{net.n_total_reaches} reaches, rL={args.reaches_per_break}"
            ),
            show=True,
            html=args.html_plot,
        )


if __name__ == "__main__":
    main()
