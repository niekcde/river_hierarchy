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
    node_type: str = "internal"  # source, bifurcation, confluence, outlet
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


# ============================================================
# Deterministic River Network with Nested Diamonds
# ============================================================

class RiverNetwork:
    def __init__(self, domain_length: float):
        self.domain_length = float(domain_length)

        self.nodes: Dict[int, Node] = {}
        self.edges: Dict[int, Edge] = {}

        self._next_node = 1
        self._next_edge = 1

        self._initialize_trunk()

    # ------------------------ ID helpers -----------------------

    def _new_node(self, x: float) -> int:
        """
        Create a node unless one already exists at this x.
        """
        # for nid, n in self.nodes.items():
        #     if abs(n.x - x) < 1e-12:
        #         return nid  # reuse existing node

        nid = self._next_node
        self._next_node += 1
        self.nodes[nid] = Node(id=nid, x=x)
        return nid

    def _new_edge(self, up: int, dn: int, x0: float, x1: float) -> int:
        eid = self._next_edge
        self._next_edge += 1
        e = Edge(
            id=eid,
            upstream_node=up,
            downstream_node=dn,
            x_start=x0,
            x_end=x1,
            centerline_length=(x1 - x0),
            width=1.0,
        )
        self.edges[eid] = e
        self.nodes[up].downstream_edges.append(eid)
        self.nodes[dn].upstream_edges.append(eid)
        return eid

    # ------------------------ trunk ----------------------------

    def _initialize_trunk(self):
        n0 = self._new_node(0.0)
        n1 = self._new_node(self.domain_length)
        self.nodes[n0].node_type = "source"
        self.nodes[n1].node_type = "outlet"
        self._new_edge(n0, n1, 0.0, self.domain_length)

    # ============================================================
    # Core Bifurcation Logic
    # ============================================================

    def _can_split(self, e: Edge, frac_up: float, frac_down: float, min_ratio: float) -> bool:
        """
        Check if edge e can be split at fraction f considering
        the width-dependent minimal length constraint:
             (segment_length) >= width * min_ratio
        """
        L = e.x_end - e.x_start
        xb = e.x_start + frac_up * L
        xc = e.x_end - frac_down * L

        segA = xb - e.x_start
        segB = xc - xb
        segC = e.x_end - xc

        minL = e.width * min_ratio
        return (segA >= minL) and (segB >= minL) and (segC >= minL)

    def _split_edge(self, host_edge_id: int, 
                    frac_up: float, frac_down: float) -> Optional[Tuple[int, int]]:
        """
        Replace the host edge by:
            up -> B
            two parallel edges B -> C
            C -> down
        Returns (mid1, mid2) or None.
        """
        if host_edge_id not in self.edges:
            return None

        e = self.edges[host_edge_id]
        up = self.nodes[e.upstream_node]
        dn = self.nodes[e.downstream_node]

        L = e.x_end - e.x_start
        xb = e.x_start + frac_up * L
        xc = e.x_end   - frac_down * L

        # remove host edge
        if host_edge_id in up.downstream_edges:
            up.downstream_edges.remove(host_edge_id)
        if host_edge_id in dn.upstream_edges:
            dn.upstream_edges.remove(host_edge_id)
        del self.edges[host_edge_id]

        # new nodes (reuse if already present)
        B = self._new_node(xb)
        C = self._new_node(xc)
        self.nodes[B].node_type = "bifurcation"
        self.nodes[C].node_type = "confluence"

        # upstream and downstream edges
        self._new_edge(up.id, B, e.x_start, xb)
        self._new_edge(C, dn.id, xc, e.x_end)

        # two symmetric edges B -> C
        mid1 = self._new_edge(B, C, xb, xc)
        mid2 = self._new_edge(B, C, xb, xc)

        return mid1, mid2

    def generate_bifurcations(
        self,
        levels: int,
        frac_up: float,
        frac_down: float,
        min_ratio: float,
        rng: Optional[random.Random] = None,
        ) -> int:
            """
            Generate nested diamonds:

            - Level 0: split the trunk into the first diamond.
            - Level 1+: each level adds *one* smaller diamond inside an existing
            diamond, by picking one (up, down) pair that has ≥2 parallel edges.

            The chosen pairs are the *innermost* ones (smallest span in x),
            so diamonds are strictly nested.
            """
            placed = 0
            if levels <= 0:
                return 0

            # --- Level 0: trunk → first diamond ---
            if not self._insert_first_diamond(frac_up, frac_down, min_ratio):
                return 0
            placed += 1

            # --- Levels 1+ : nested inside existing diamonds ---
            for level in range(1, levels):
                # Group edges by (upstream_node, downstream_node)
                groups: Dict[Tuple[int, int], List[int]] = {}

                # Candidate diamond pairs: at least 2 parallel edges
                candidates: List[Tuple[int, int]] = []
                candidate_edge: List[int] = [] 
                for eid, e in self.edges.items():
                    key = (e.upstream_node, e.downstream_node)
                    node_type_up = self.nodes[e.upstream_node].node_type
                    node_type_dn = self.nodes[e.downstream_node].node_type
                    if (node_type_up == 'bifurcation') & (node_type_dn == 'confluence') &\
                        (key not in candidates):
                        candidates.append(key)
                        candidate_edge.append(eid)
                    groups.setdefault(key, []).append(eid)


                if not candidates:
                    break

                # Among these, keep only those with enough room to split again
                pairs_with_len: List[Tuple[Tuple[int, int], float]] = []
                candidate_edge_len = []
                count = 0
                for up, dn in candidates:
                    x0 = self.nodes[up].x
                    x1 = self.nodes[dn].x
                    L = x1 - x0
                    if self._can_split_interval(x0, x1, 1.0, min_ratio, frac_up, frac_down):
                        pairs_with_len.append(((up, dn), L))
                        candidate_edge_len.append(candidate_edge[count])
                    count +=1

                if not pairs_with_len:
                    break
                eid = candidate_edge_len[np.argmax([l for (u,v), l in pairs_with_len])]

                # Pick the *innermost* diamond (shortest x-span) to nest inside
                pairs_with_len.sort(key=lambda t: t[1])  # ascending by length
                (up, dn), _ = pairs_with_len[-1]

                # (If you really want some randomness/alternation, you could swap
                #  to rng.choice(pairs_with_len), but "smallest first" gives clean nesting.)
                if not self._split_edge(eid, frac_up, frac_down):
                # if not self._insert_nested_diamond_on_pair(up, dn, frac, min_ratio):
                    break
                for eid, e in self.edges.items():
                    key = (e.upstream_node, e.downstream_node)
                    node_type_up = self.nodes[e.upstream_node].node_type
                    node_type_dn = self.nodes[e.downstream_node].node_type

                placed += 1

            return placed


    # ============================================================
    # Width assignment
    # ============================================================

    def assign_widths(self, root_width: float):
        """
        Deterministic width propagation.
        """
        for e in self.edges.values():
            e.width = 1e-12

        G = nx.DiGraph()
        for eid, e in self.edges.items():
            G.add_edge(e.upstream_node, e.downstream_node, eid=eid)

        try:
            topo = list(nx.topological_sort(G))
        except:
            topo = sorted(self.nodes.keys(), key=lambda nid: self.nodes[nid].x)

        for nid in topo:
            n = self.nodes[nid]
            outs = n.downstream_edges
            if not outs:
                continue

            if n.upstream_edges:
                incoming_width = sum(self.edges[eid].width for eid in n.upstream_edges)
            else:
                incoming_width = root_width

            if len(outs) == 1:
                self.edges[outs[0]].width = incoming_width
            else:
                share = incoming_width / len(outs)
                for eid in outs:
                    self.edges[eid].width = share

    # ============================================================
    # New helpers for nested diamonds
    # ============================================================
    def _can_split_interval(self, x0: float, x1: float, width: float, min_ratio: float, 
                            frac_up: float, frac_down: float) -> bool:
        """
        Like _can_split, but works on an arbitrary [x0, x1] interval instead of an Edge.
        Ensures the three segments created by the split are long enough.
        """
        L = x1 - x0
        xb = x0 + frac_up * L
        xc = x1 - frac_down * L

        segA = xb - x0
        segB = xc - xb
        segC = x1 - xc

        minL = width * min_ratio
        return (segA >= minL) and (segB >= minL) and (segC >= minL)

    def _insert_first_diamond(self, frac_up: float, frac_down: float, min_ratio: float) -> bool:
        """
        Replace the initial trunk (single edge 0→L) by:
            up → B
            two edges B → C
            C → down
        """
        if not self.edges:
            return False

        # There should be exactly one trunk edge at initialization
        trunk_id = next(iter(self.edges))
        e = self.edges[trunk_id]
        x0, x1 = e.x_start, e.x_end

        if not self._can_split_interval(x0, x1, e.width, min_ratio, frac_up, frac_down):
            return False

        up = self.nodes[e.upstream_node]
        dn = self.nodes[e.downstream_node]

        L = x1 - x0
        xb = x0 + frac_up * L
        xc = x1 - frac_down * L

        # Remove the trunk edge
        if trunk_id in up.downstream_edges:
            up.downstream_edges.remove(trunk_id)
        if trunk_id in dn.upstream_edges:
            dn.upstream_edges.remove(trunk_id)
        del self.edges[trunk_id]

        # New nodes B, C
        B = self._new_node(xb)
        C = self._new_node(xc)
        self.nodes[B].node_type = "bifurcation"
        self.nodes[C].node_type = "confluence"

        # Trunk pieces
        self._new_edge(up.id, B, x0, xb)
        self._new_edge(C, dn.id, xc, x1)

        # First diamond: two parallel B → C edges
        self._new_edge(B, C, xb, xc)
        self._new_edge(B, C, xb, xc)

        return True

    def _insert_nested_diamond_on_pair(
        self,
        up_node: int,
        dn_node: int,
        frac: float,
        min_ratio: float,
    ) -> bool:
        """
        Given an existing diamond between (up_node, dn_node) (i.e., ≥2 edges),
        add a *smaller* diamond inside it:

            up → B
            two edges B → C
            C → down

        The existing edges up_node → dn_node are kept, so both the outer
        and inner diamonds coexist.
        """
        x0 = self.nodes[up_node].x
        x1 = self.nodes[dn_node].x

        # Width doesn't matter here; use 1.0 to scale min_ratio
        if not self._can_split_interval(x0, x1, 1.0, min_ratio, frac):
            return False

        L = x1 - x0
        xb = x0 + (frac * L)
        xc = x1 - (frac * L)

        B = self._new_node(xb)
        C = self._new_node(xc)
        self.nodes[B].node_type = "bifurcation"
        self.nodes[C].node_type = "confluence"

        # Connect new diamond into the existing diamond's "channel"
        self._new_edge(up_node, B, x0, xb)
        self._new_edge(C, dn_node, xc, x1)

        # Inner diamond: two parallel B → C edges
        self._new_edge(B, C, xb, xc)
        self._new_edge(B, C, xb, xc)

        return True



    # ============================================================
    # Geometry: perfect diamonds
    # ============================================================

    def generate_centerlines(self, n_midpoints: int = 7):
        """
        Draw perfect diamonds for every B→C pair having two edges.
        """
        # if source y == 0 if bifurcation y end point of edge
        # if edge start is bif start no change end changes
            # if node end is confluence then half way change
            # if end bif straight line
        # print('Generate centerlines')
        x0, x1 = 0, 0
        import matplotlib.pyplot as plt


        # topo      = sorted(self.nodes.keys(), key=lambda nid: self.nodes[nid].x)
        # topo_edge = sorted(self.edges.keys(), key=lambda nid: self.edges[nid].x_start)
        # for nid in topo:
        #     e = self.nodes[nid]
        #     if e.node_type == 'bifurcation':
        #         angle = 45
        #         for eid in e.downstream_edges:
        #             self.edges[eid].angle = angle
        #             angle *= -1
        #     else:
        #         if e.node_type != 'outlet':
        #             for eid in e.downstream_edges:
        #                 self.edges[eid].angle = 0
        
        # for eid in topo_edge:
        #     e = self.edges[eid]
        #     u = e.upstream_node
        #     v = e.downstream_node

        #     L   = e.centerline_length
        #     upx = self.nodes[u].x
        #     dnx = self.nodes[v].x
        #     uy  = self.nodes[u].y
        #     # print('edge node locations',eid, u,v,upx, dnx)
        #     utype = self.nodes[u].node_type
        #     vtype = self.nodes[v].node_type

        #     # inherit default incoming angle
        #     incoming_angle = getattr(self.nodes[u], "incoming_angle", 0)

        #     angle = 45
        #     straight = True
        #     # print(u,v,eid, e.angle)
        #     # ---------------------------
        #     # ANGLE RULES
        #     # ---------------------------
        #     # if vtype == 'bifurcation'
        #     if utype == 'source':
        #         angle = 0
        #         self.edges[eid].level = 0
        #         self.nodes[u].level   = 0
        #         self.nodes[v].level   = 0

        #     elif utype == 'bifurcation' and vtype in ('bifurcation', 'confluence'):
        #         idx = self.nodes[u].downstream_edges.index(eid)

        #         if idx == 1:
        #             angle = -angle
        #         else:
        #             angle = +angle

        #         if vtype == 'confluence':
        #             straight = False  # curved into confluence
        #             self.nodes[v].level   = self.nodes[u].level 
        #             self.edges[eid].level = self.nodes[u].level + 1
        #             # print('Previous angle', self.edges[self.nodes[u].upstream_edges[0]].angle, e.angle)
        #         else:
        #             self.nodes[v].level   = self.nodes[u].level + 1
        #             self.edges[eid].level = self.nodes[u].level + 1

        #     elif utype == 'confluence' and vtype == 'confluence':
        #         # Return the branch toward trunk → opposite sign
        #         self.nodes[v].level   = self.nodes[u].level - 1
        #         self.edges[eid].level = self.nodes[u].level - 1

        #         angle = -incoming_angle
        #         straight = True

        #     elif utype == 'confluence' and vtype == 'outlet':
        #         self.nodes[v].level   = 0
        #         self.edges[eid].level = 0

        #         angle = 0
        #         straight = True

        #     # ---------------------------
        #     # STORE INCOMING ANGLE
        #     # ---------------------------
        #     self.nodes[v].incoming_angle = angle
        #     self.edges[eid].angle = angle
            
        #     # ---------------------------
        #     # GEOMETRY CALCULATION
        #     # ---------------------------
        #     dx = dnx - upx
        #     dy = math.sqrt(max(L*L - dx*dx, 0.0))
        #     # dy, L*L, dx*dx, L*(np.sqrt(2)/2), L * math.tan(angle)
        #     # print(eid, self.edges[eid].level, straight)

        #     if straight:
        #         self.nodes[v].y = uy - dy

        #     else:
        #         # add midpoint bend
        #         h = 0.25 * L
        #         midx = (upx + dnx)/2
        #         midy = uy - dy/2

        #         if angle < 0:
        #             midy -= h
        #         else:
        #             midy += h

        #         e.midpoint = (midx, midy)
        #         self.nodes[v].y = uy - dy
            # print(eid, u,v, angle)
            # try:
            #     print(e.midpoint, self.nodes[u].x, self.nodes[u].y, self.nodes[v].x, self.nodes[v].y)
            #     plt.plot([self.nodes[u].x,e.midpoint[0], self.nodes[v].x], [self.nodes[u].y,e.midpoint[1], 
            #                                                                 self.nodes[v].y], 
            #                                                                 marker = 'o', label = eid)
            # except:
            #     # plt.plot([self.nodes[u].x, self.nodes[v].x], [self.nodes[u].y, self.nodes[v].y], 
            #              marker = 'o', label = eid)


            #     print(self.nodes[u].x, self.nodes[u].y, self.nodes[v].x, self.nodes[v].y)
            # print()
        # plt.legend()
        # plt.show()
        
        for n in self.nodes.values():
            n.y = 0.0

        # group parallel edges
        groups: Dict[Tuple[int, int], List[int]] = {}
        for eid, e in self.edges.items():
            key = (e.upstream_node, e.downstream_node)
            groups.setdefault(key, []).append(eid)



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
            incoming[v].append(u)    # <-- CRITICAL LINE FOR UPSTREAM PROPAGATION
            out_by_node[u].append(eid)
            parallel[(u, v)].append(eid)

        # ============================================================
        # STEP 1 — Compute X positions
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
            assign_x(min(outgoing.keys()))

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

            # Move UPSTREAM
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

        y_offset = {}

        for u, eids in offset_groups.items():
            offs = spaced_offsets(len(eids), 20)
            for eid, off in zip(eids, offs):
                y_offset[eid] = off

        for (u, v), eids in parallel.items():
            offs = spaced_offsets(len(eids), 5)
            for eid, off in zip(eids, offs):
                y_offset[eid] = y_offset.get(eid, 0) + off

        # ============================================================
        # STEP 6 — Create diamond-shaped geometry
        # ============================================================

        coords = {}

        for eid, e in self.edges.items():
            u = e.upstream_node
            v = e.downstream_node
            L = e.centerline_length

            x1 = x[u]
            x2 = x[v]

            off = y_offset.get(eid, 0)
            eps = L * 0.01
            xm = (x1 + x2) / 2

            coords[eid] = [
                (x1, 0),
                (x1 + eps, off),
                (xm, off),
                (x2, 0)
            ]

            self.edges[eid].geometry = coords[eid]

        # DEBUG
        # for eid in sorted(coords):
            # print(eid, coords[eid])


        # def arch(x0, x1, amp, sign):
        #     coords = []
        #     for k in range(n_midpoints + 2):
        #         t = k / (n_midpoints + 1)
        #         x = x0 + t * (x1 - x0)
        #         y = sign * amp * 4 * t * (1 - t)
        #         coords.append((x, y))
        #     return coords

        # for (up, dn), eids in groups.items():
        #     x0 = self.nodes[up].x
        #     x1 = self.nodes[dn].x

        #     # amplitude = 0.5 * incoming width (rule A)
        #     inflow = sum(self.edges[eid].width for eid in self.nodes[up].upstream_edges)
        #     A = 0.5 * inflow if inflow > 0 else 1.0

        #     if len(eids) == 1:
        #         eid = eids[0]
        #         self.edges[eid].geometry = [(x0, 0.0), (x1, 0.0)]
        #     else:
        #         # exactly 2 edges due to symmetric construction
        #         self.edges[eids[0]].geometry = arch(x0, x1, A, +1.0)
        #         self.edges[eids[1]].geometry = arch(x0, x1, A, -1.0)

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
# GIS
# ============================================================
def edges_to_gdf(net, crs=None):
    rec = []
    for e in net.edges.values():
        # print(e.id,e.width,e.centerline_length, e.upstream_node,e.downstream_node)
        if e.geometry:
            rec.append({
                "edge_id": e.id,
                "upstream": e.upstream_node,
                "downstream": e.downstream_node,
                "width": e.width,
                "length": e.centerline_length,
                "geometry": LineString(e.geometry)
            })
    return gpd.GeoDataFrame(rec, geometry="geometry", crs=crs)


def nodes_to_gdf(net, crs=None):
    rec = []
    for n in net.nodes.values():
        rec.append({
            "node_id": n.id,
            "x": n.x,
            "y": n.y,
            "node_type": n.node_type,
            "geometry": Point(n.x, n.y)
        })
    return gpd.GeoDataFrame(rec, geometry="geometry", crs=crs)


# ============================================================
# Plotly
# ============================================================
def plot_network_plotly(net, title="", show=True, html=None):
    fig = go.Figure()
    for e in net.edges.values():
        if not e.geometry:
            continue
        xs = [p[0] for p in e.geometry]
        ys = [p[1] for p in e.geometry]
        fig.add_trace(go.Scatter(
            x=xs, y=ys, mode="lines",
            line=dict(width=max(1, e.width / 10)),
            hoverinfo="text",
            text=f"Edge {e.id}<br>w={e.width:.2f}"
        ))
    fig.update_layout(
        title=title,
        yaxis=dict(scaleanchor="x", scaleratio=1),
        showlegend=False,
        template="plotly_white"
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
    p.add_argument("--domain-length", type=float, default=1000)
    p.add_argument("--n-bifurcations", type=int, default=4)
    p.add_argument("--bifurcation-frac_up", type=float, default=0.1)
    p.add_argument("--bifurcation-frac_down", type=float, default=0.1)
    p.add_argument("--min-length-ratio", type=float, default=5.0)
    p.add_argument("--root-width", type=float, default=100)
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

    net = RiverNetwork(args.domain_length)

    placed = net.generate_bifurcations(
        levels=args.n_bifurcations,
        frac_up  =args.bifurcation_frac_up,
        frac_down=args.bifurcation_frac_down,
        min_ratio=args.min_length_ratio,
        rng=rng,
    )

    net.assign_widths(args.root_width)
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
            title=f"Nested Diamond Network (placed {placed})",
            show=True,
            html=args.html_plot,
        )


if __name__ == "__main__":
    main()
