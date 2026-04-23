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
    def __init__(self, domain_length: float, root_width: float, min_ratio: float):
        """
        domain_length : total length of the trunk (m)
        root_width    : characteristic root width (m)
        min_ratio     : length/width ratio used to define reach length

        reach_length = root_width * min_ratio
        n_total_reaches = floor(domain_length / reach_length)
        """
        self.domain_length = float(domain_length)
        self.root_width = float(root_width)
        self.min_ratio = float(min_ratio)

        # Reach length and count
        self.reach_length = self.root_width * self.min_ratio
        self.n_total_reaches = max(int(self.domain_length // self.reach_length), 1)

        # Effective length actually used (integer number of reaches)
        self.effective_length = self.n_total_reaches * self.reach_length

        self.nodes: Dict[int, Node] = {}
        self.edges: Dict[int, Edge] = {}

        self._next_node = 1
        self._next_edge = 1

        # trunk edge id (single trunk before bifurcations)
        self.trunk_edge_id: Optional[int] = None

        self._initialize_trunk()

    # ------------------------ ID helpers -----------------------

    def _new_node(self, x: float) -> int:
        """
        Create a node at location x.
        No deduplication by x; each call creates a new node.
        """
        nid = self._next_node
        self._next_node += 1
        self.nodes[nid] = Node(id=nid, x=x)
        return nid

    def _new_edge(self, up: int, dn: int, x0: float, x1: float, split_depth: int = 0) -> int:
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
            width=1.0,
            reach_start=rs,
            reach_end=re,
            split_depth=split_depth,
        )

        self.edges[eid] = e
        self.nodes[up].downstream_edges.append(eid)
        self.nodes[dn].upstream_edges.append(eid)
        return eid

    # ------------------------ trunk ----------------------------

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
    # Core Bifurcation Logic (reach-based)
    # ============================================================

    def _insert_first_diamond(self, rL: int) -> bool:
        """
        Replace the initial trunk edge by:
            up → B
            two edges B → C
            C → down

        Split is done in reach space:
            B at (reach_start + rL)
            C at (reach_end   - rL)
        """
        if self.trunk_edge_id is None or self.trunk_edge_id not in self.edges:
            return False

        trunk_id = self.trunk_edge_id
        e = self.edges[trunk_id]
        up = self.nodes[e.upstream_node]
        dn = self.nodes[e.downstream_node]

        nR = self._edge_num_reaches(e)
        if nR < 2 * rL + 1:
            return False

        B_idx = e.reach_start + rL
        C_idx = e.reach_end - rL

        xb = B_idx * self.reach_length
        xc = C_idx * self.reach_length
        x0 = e.x_start
        x1 = e.x_end

        # Remove old trunk edge
        if trunk_id in up.downstream_edges:
            up.downstream_edges.remove(trunk_id)
        if trunk_id in dn.upstream_edges:
            dn.upstream_edges.remove(trunk_id)
        del self.edges[trunk_id]

        # New nodes at reach boundaries
        B = self._new_node(xb)
        C = self._new_node(xc)
        self.nodes[B].node_type = "bifurcation"
        self.nodes[C].node_type = "confluence"

        base_depth = e.split_depth  # should be 0 for the initial trunk

        # Upstream trunk piece: up → B (same depth)
        e_up_id = self._new_edge(up.id, B, x0, xb, split_depth=base_depth)
        self.edges[e_up_id].reach_start = e.reach_start
        self.edges[e_up_id].reach_end = B_idx

        # Downstream trunk piece: C → dn (same depth)
        e_dn_id = self._new_edge(C, dn.id, xc, x1, split_depth=base_depth)
        self.edges[e_dn_id].reach_start = C_idx
        self.edges[e_dn_id].reach_end = e.reach_end

        # First diamond: two parallel B → C edges with depth+1
        mid1 = self._new_edge(B, C, xb, xc, split_depth=base_depth + 1)
        self.edges[mid1].reach_start = B_idx
        self.edges[mid1].reach_end = C_idx

        mid2 = self._new_edge(B, C, xb, xc, split_depth=base_depth + 1)
        self.edges[mid2].reach_start = B_idx
        self.edges[mid2].reach_end = C_idx

        # after this, trunk_edge_id is no longer meaningful
        self.trunk_edge_id = None
        return True

    def _split_edge(self, host_edge_id: int, rL: int) -> Optional[Tuple[int, int]]:
        """
        Given an existing edge from up→dn, replace it by:
            up → B
            two edges B → C
            C → down

        Splitting is done at reach indices:
            B at (reach_start + rL)
            C at (reach_end   - rL)
        Nested inside an existing diamond edge.
        """
        if host_edge_id not in self.edges:
            return None

        e = self.edges[host_edge_id]
        up = self.nodes[e.upstream_node]
        dn = self.nodes[e.downstream_node]

        nR = self._edge_num_reaches(e)
        if nR < 2 * rL + 1:
            return None

        B_idx = e.reach_start + rL
        C_idx = e.reach_end - rL

        xb = B_idx * self.reach_length
        xc = C_idx * self.reach_length
        x0 = e.x_start
        x1 = e.x_end

        # Remove the host edge
        if host_edge_id in up.downstream_edges:
            up.downstream_edges.remove(host_edge_id)
        if host_edge_id in dn.upstream_edges:
            dn.upstream_edges.remove(host_edge_id)
        del self.edges[host_edge_id]

        # New nodes at B, C
        B = self._new_node(xb)
        C = self._new_node(xc)
        self.nodes[B].node_type = "bifurcation"
        self.nodes[C].node_type = "confluence"

        base_depth = e.split_depth

        # Upstream piece up → B (same depth)
        e_up_id = self._new_edge(up.id, B, x0, xb, split_depth=base_depth)
        self.edges[e_up_id].reach_start = e.reach_start
        self.edges[e_up_id].reach_end = B_idx

        # Downstream piece C → dn (same depth)
        e_dn_id = self._new_edge(C, dn.id, xc, x1, split_depth=base_depth)
        self.edges[e_dn_id].reach_start = C_idx
        self.edges[e_dn_id].reach_end = e.reach_end

        # Two parallel B → C edges with depth+1
        mid1 = self._new_edge(B, C, xb, xc, split_depth=base_depth + 1)
        self.edges[mid1].reach_start = B_idx
        self.edges[mid1].reach_end = C_idx

        mid2 = self._new_edge(B, C, xb, xc, split_depth=base_depth + 1)
        self.edges[mid2].reach_start = B_idx
        self.edges[mid2].reach_end = C_idx

        return mid1, mid2

    def generate_bifurcations(
        self,
        levels: int,
        reaches_before_break: int,
        rng: Optional[random.Random] = None,
    ) -> int:
        """
        Generate nested diamonds using reach-based splits, but *balance*
        nesting depth across branches:

        - Level 0: split the trunk into the first diamond.
        - Level 1+: each level adds *one* smaller diamond by picking a
          (bifurcation, confluence) pair with ≥ 2 parallel edges that:
            * can be split (≥ 2*rL + 1 reaches), and
            * has the smallest current split_depth (shallower first),
              breaking ties by choosing the longest reach span.

        This avoids always drilling into the same deepest branch.
        """
        placed = 0
        if levels <= 0:
            return 0

        rL = reaches_before_break

        # --- Level 0: trunk → first diamond ---
        if not self._insert_first_diamond(rL):
            return 0
        placed += 1

        # --- Levels 1+ : nested diamonds ---
        for level in range(1, levels):
            # print(f'Level: {level}')
            # Group edges by (upstream_node, downstream_node)
            groups: Dict[Tuple[int, int], List[int]] = defaultdict(list)
            for eid, e in self.edges.items():

                key = (e.upstream_node, e.downstream_node)
                # print(key, eid)
                groups[key].append(eid)
            # print()
            candidates: List[Tuple[int, int, int]] = []
            # entries: (split_depth, -num_reaches, representative_eid)
            for (up, dn), eids in groups.items():
                # if len(eids) < 2:
                #     continue  # need at least 2 parallels for a diamond

                node_type_up = self.nodes[up].node_type
                node_type_dn = self.nodes[dn].node_type
                # print(level, up, dn, eids,node_type_up, node_type_dn)
                if not (node_type_up == "bifurcation" and node_type_dn == "confluence"):
                    continue

                rep_eid = eids[0]
                rep_edge = self.edges[rep_eid]

                if not self._can_split_edge_by_reaches(rep_edge, rL):
                    raise RuntimeError(f"Edge not allowed to be split at level: {level}")


                depth = rep_edge.split_depth
                nR = self._edge_num_reaches(rep_edge)
                # print(rep_eid ,depth, nR)
                # We want minimal depth first, then longest span
                candidates.append((depth, -nR, rep_eid))

            if not candidates:
                raise RuntimeError(f"No candidate to be split")

            candidates.sort()
            _, _, eid_to_split = candidates[0]

            if not self._split_edge(eid_to_split, rL):
                break

            placed += 1

        return placed

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

                seg_id = self._new_edge(prev_node, next_node, x0, x1, split_depth=e.split_depth)
                seg_edge = self.edges[seg_id]
                seg_edge.reach_start = i
                seg_edge.reach_end = i + 1

                prev_node = next_node

    # ============================================================
    # Width assignment
    # ============================================================

    def assign_widths(self, root_width: float):
        """
        Deterministic width propagation from source(s) downstream.
        """
        for e in self.edges.values():
            e.width = 1e-12

        G = nx.DiGraph()
        for eid, e in self.edges.items():
            G.add_edge(e.upstream_node, e.downstream_node, eid=eid)

        try:
            topo = list(nx.topological_sort(G))
        except Exception:
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
# GIS helpers
# ============================================================

def edges_to_gdf(net: RiverNetwork, crs=None):
    rec = []
    for e in net.edges.values():
        if e.geometry:
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
            title=(
                f"Nested Diamond Network (placed {placed}) - "
                f"{net.n_total_reaches} reaches, rL={args.reaches_per_break}"
            ),
            show=True,
            html=args.html_plot,
        )


if __name__ == "__main__":
    main()
