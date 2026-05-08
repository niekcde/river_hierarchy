from __future__ import annotations
import argparse
import random
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Callable, Optional

import numpy as np
import geopandas as gpd
import networkx as nx
from shapely.geometry import LineString, Point
import plotly.graph_objects as go


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
    sinuosity: float = 1.0
    centerline_length: float = 0.0
    width: float = 1.0          # attribute width (constrained by length/ratio)
    geometry: Optional[List[Tuple[float, float]]] = None  # list[(x, y)]


# ============================================================
# Synthetic River Network Generator
# ============================================================

class RiverNetwork:

    def __init__(self, domain_length: float, domain_width: float, rng=None):
        self.domain_length = float(domain_length)
        self.domain_width = float(domain_width)
        # internal safety buffer to avoid zero-length connectors
        self._min_dx = 0.001 * self.domain_length

        self.nodes: Dict[int, Node] = {}
        self.edges: Dict[int, Edge] = {}

        self._next_node = 0
        self._next_edge = 0

        self.rng = rng or random.Random()
        self._initialize_trunk()

    # ----------------------------------------------------------
    # ID creation
    # ----------------------------------------------------------

    def _new_node(self):
        i = self._next_node
        self._next_node += 1
        return i

    def _new_edge(self):
        i = self._next_edge
        self._next_edge += 1
        return i

    # ----------------------------------------------------------
    # Initial trunk
    # ----------------------------------------------------------

    def _initialize_trunk(self):
        nid0 = self._new_node()
        nid1 = self._new_node()

        n0 = Node(id=nid0, x=0.0, node_type="source")
        n1 = Node(id=nid1, x=self.domain_length, node_type="outlet")
        self.nodes[nid0] = n0
        self.nodes[nid1] = n1

        eid = self._new_edge()
        e = Edge(
            id=eid,
            upstream_node=nid0,
            downstream_node=nid1,
            x_start=0.0,
            x_end=self.domain_length
        )
        self.edges[eid] = e

        n0.downstream_edges.append(eid)
        n1.upstream_edges.append(eid)

    # ============================================================
    # Bifurcations
    # ============================================================

    def _min_dx_for_edge(self, e: Edge) -> float:
        """
        Local minimum branch length tied to the host edge, not the whole domain.
        """
        L = e.x_end - e.x_start
        return max(0.05 * L, self._min_dx)

    def _pick_global_x(self) -> float:
        """
        Draw a uniform x location across the domain where a bifurcation might land.
        """
        return self.rng.uniform(self._min_dx, self.domain_length - self._min_dx)

    def generate_bifurcations(self, n):

        added = 0
        attempts = 0
        max_attempts = max(5000 * n, n + 2)
        while added < n and attempts < max_attempts:
            before = len(self.edges)
            self._add_one_bifurcation()
            if len(self.edges) > before:
                added += 1
            attempts += 1
        return added == n

    # ---------------- robust split -----------------------------

    def _split_edge(self, edge_id, x_split, node_type):
        """
        Split an edge at x_split, creating a NEW node at that x.
        """
        e = self.edges[edge_id]
        lo = e.x_start + self._min_dx
        hi = e.x_end - self._min_dx
        if not (lo < x_split < hi):
            # too close to endpoints; skip split
            return None, None, None

        up = self.nodes[e.upstream_node]
        dn = self.nodes[e.downstream_node]

        # remove old from node connectivity
        if edge_id in up.downstream_edges:
            up.downstream_edges.remove(edge_id)
        if edge_id in dn.upstream_edges:
            dn.upstream_edges.remove(edge_id)

        del self.edges[edge_id]

        new_nid = self._new_node()
        new_node = Node(id=new_nid, x=x_split, node_type=node_type)
        self.nodes[new_nid] = new_node

        e1_id = self._new_edge()
        e2_id = self._new_edge()

        e1 = Edge(
            id=e1_id,
            upstream_node=up.id,
            downstream_node=new_nid,
            x_start=e.x_start,
            x_end=x_split,
            sinuosity=e.sinuosity,
            width=e.width
        )
        e2 = Edge(
            id=e2_id,
            upstream_node=new_nid,
            downstream_node=dn.id,
            x_start=x_split,
            x_end=e.x_end,
            sinuosity=e.sinuosity,
            width=e.width
        )

        self.edges[e1_id] = e1
        self.edges[e2_id] = e2

        up.downstream_edges.append(e1_id)
        new_node.upstream_edges.append(e1_id)

        new_node.downstream_edges.append(e2_id)
        dn.upstream_edges.append(e2_id)

        return new_nid, e1_id, e2_id

    def _split_edge_at_existing_node(self, edge_id: int, node_id: int):
        """
        Split edge_id at the x-position of an EXISTING node node_id.

        Used when a centerline passes through a node that it is
        not yet connected to – we promote that crossing to a junction.
        """
        e = self.edges.get(edge_id)
        if e is None:
            return

        node = self.nodes[node_id]
        x_split = node.x

        lo = e.x_start + self._min_dx
        hi = e.x_end - self._min_dx
        if not (lo < x_split < hi):
            return

        up = self.nodes[e.upstream_node]
        dn = self.nodes[e.downstream_node]

        # remove old edge from node connectivity
        if edge_id in up.downstream_edges:
            up.downstream_edges.remove(edge_id)
        if edge_id in dn.upstream_edges:
            dn.upstream_edges.remove(edge_id)

        del self.edges[edge_id]

        # create two new edges sharing the existing node
        e1_id = self._new_edge()
        e2_id = self._new_edge()

        e1 = Edge(
            id=e1_id,
            upstream_node=up.id,
            downstream_node=node_id,
            x_start=e.x_start,
            x_end=x_split,
            sinuosity=e.sinuosity,
            width=e.width
        )
        e2 = Edge(
            id=e2_id,
            upstream_node=node_id,
            downstream_node=dn.id,
            x_start=x_split,
            x_end=e.x_end,
            sinuosity=e.sinuosity,
            width=e.width
        )

        self.edges[e1_id] = e1
        self.edges[e2_id] = e2

        up.downstream_edges.append(e1_id)
        node.upstream_edges.append(e1_id)

        node.downstream_edges.append(e2_id)
        dn.upstream_edges.append(e2_id)

    def _add_one_bifurcation(self):
        """
        Place a bifurcation by picking a global x-location, finding which edge
        spans it, and enforcing a local min_dx based on that edge's length.
        """
        local_tries = 0
        while local_tries < 500:
            local_tries += 1

            x_b = self._pick_global_x()
            candidates = []
            for e in self.edges.values():
                min_dx_local = self._min_dx_for_edge(e)
                L = e.x_end - e.x_start

                # Edge must be long enough to host a split with the local rule
                if L <= 2 * min_dx_local:
                    continue

                # x_b must land far enough from endpoints to respect local min_dx
                if not (e.x_start + min_dx_local < x_b < e.x_end - min_dx_local):
                    continue

                # need room downstream of the bifurcation for a confluence node
                if (e.x_end - x_b) <= (min_dx_local + self._min_dx):
                    continue

                candidates.append((e, min_dx_local))

            if not candidates:
                continue

            host, min_dx_local = self.rng.choice(candidates)

            # choose a confluence position downstream with at least local min_dx branch length
            x_c_min = x_b + min_dx_local
            x_c_max = host.x_end - self._min_dx
            if x_c_min >= x_c_max:
                continue
            x_c = self.rng.uniform(x_c_min, x_c_max)

            # split at bifurcation
            nid_b, e1, e2 = self._split_edge(host.id, x_b, "bifurcation")
            if nid_b is None:
                continue

            # confluence split on e2
            ed = self.edges.get(e2)
            if ed is None:
                continue
            if not (ed.x_start < x_c < ed.x_end):
                continue

            nid_c, _, _ = self._split_edge(e2, x_c, "confluence")
            if nid_c is None:
                continue

            # create side branch
            e_new = self._new_edge()
            edge_new = Edge(
                id=e_new,
                upstream_node=nid_b,
                downstream_node=nid_c,
                x_start=x_b,
                x_end=x_c
            )
            self.edges[e_new] = edge_new

            self.nodes[nid_b].downstream_edges.append(e_new)
            self.nodes[nid_c].upstream_edges.append(e_new)
            return

    # ============================================================
    # Topology Validation & Repair
    # ============================================================

    def validate_and_repair_topology(self):
        """
        Ensure the network is a valid directed tree (no dangling middle nodes).
        """
        min_fix_dx = self._min_dx

        # Fix edge references & x-ordering
        for eid, e in list(self.edges.items()):
            up = self.nodes[e.upstream_node]
            dn = self.nodes[e.downstream_node]

            if eid not in up.downstream_edges:
                up.downstream_edges.append(eid)
            if eid not in dn.upstream_edges:
                dn.upstream_edges.append(eid)

            if e.x_end <= e.x_start:
                e.x_end = e.x_start + min_fix_dx
            e.centerline_length = e.sinuosity * (e.x_end - e.x_start)

        # Fix degree defects
        for nid, n in self.nodes.items():
            indeg = len(n.upstream_edges)
            outdeg = len(n.downstream_edges)

            # no outgoing but not outlet
            if outdeg == 0 and n.node_type != "outlet":
                downstream = [m for m in self.nodes.values() if m.x > n.x]
                if downstream:
                    target = min(downstream, key=lambda m: m.x)
                    if (target.x - n.x) <= min_fix_dx:
                        n.node_type = "outlet"
                        continue
                    eid = self._new_edge()
                    efix = Edge(
                        id=eid,
                        upstream_node=nid,
                        downstream_node=target.id,
                        x_start=n.x,
                        x_end=max(target.x, n.x + min_fix_dx)
                    )
                    self.edges[eid] = efix
                    n.downstream_edges.append(eid)
                    target.upstream_edges.append(eid)
                else:
                    n.node_type = "outlet"

            # no incoming but not source
            if indeg == 0 and n.node_type != "source":
                upstream = [m for m in self.nodes.values() if m.x < n.x]
                if upstream:
                    target = max(upstream, key=lambda m: m.x)
                    if (n.x - target.x) <= min_fix_dx:
                        n.node_type = "source"
                        continue
                    eid = self._new_edge()
                    efix = Edge(
                        id=eid,
                        upstream_node=target.id,
                        downstream_node=nid,
                        x_start=target.x,
                        x_end=max(n.x, target.x + min_fix_dx)
                    )
                    self.edges[eid] = efix
                    target.downstream_edges.append(eid)
                    n.upstream_edges.append(eid)
                else:
                    n.node_type = "source"

        # Update node types
        for n in self.nodes.values():
            indeg = len(n.upstream_edges)
            outdeg = len(n.downstream_edges)

            if indeg == 0 and outdeg > 0:
                n.node_type = "source"
            elif outdeg == 0:
                n.node_type = "outlet"
            elif indeg >= 2 and outdeg >= 1:
                n.node_type = "confluence"
            elif indeg == 1 and outdeg == 2:
                n.node_type = "bifurcation"
            else:
                n.node_type = "internal"

    # ============================================================
    # Attribute assignment
    # ============================================================

    def assign_sinuosity(self, sampl: Optional[Callable[[Edge], float]] = None):
        if sampl is None:
            def sampl(e: Edge) -> float:
                return 1.0 + 0.5 * self.rng.random()

        for e in self.edges.values():
            S = max(1.0, float(sampl(e)))
            e.sinuosity = S
            e.centerline_length = S * (e.x_end - e.x_start)

    def assign_width(self, sampl: Optional[Callable[[Edge], float]] = None):
        """
        Assign raw widths. We keep only a tiny positive floor here; all
        geomorphic constraints are handled in enforce_min_length_vs_width().
        """
        if sampl is None:
            def sampl(e: Edge) -> float:
                return 20.0 * (0.5 + self.rng.random())

        for e in self.edges.values():
            w = float(sampl(e))
            e.width = max(1e-6, w)

    def assign_width_with_total(self,
                                total_width: Optional[float],
                                split_mode: str = "equal",
                                min_pct: float = 0.2,
                                min_length_ratio: float = 5.0):
        """
        Assign widths so that, at each bifurcation, the outgoing widths
        split from the parent while keeping the cross-section total roughly
        constant across x.

        - split_mode "equal": divide evenly across downstream edges.
        - split_mode "random": for two-way splits, draw a random fraction
          between [min_pct, 1-min_pct] for branch A, remainder to branch B.
          For higher-order splits, fall back to equal shares.

        If total_width is None or <= 0, fallback to the legacy random widths.
        """
        if total_width is None or total_width <= 0:
            self.assign_width()
            return True

        # initialize all widths to tiny floor
        for e in self.edges.values():
            e.width = 1e-6

        # process nodes in topological order (upstream -> downstream)
        topo = nx.DiGraph()
        topo.add_nodes_from(self.nodes.keys())
        for e in self.edges.values():
            topo.add_edge(e.upstream_node, e.downstream_node)
        try:
            node_ids = list(nx.topological_sort(topo))
        except nx.NetworkXUnfeasible:
            node_ids = [n.id for n in sorted(self.nodes.values(), key=lambda n: n.x)]
        node_order = [self.nodes[i] for i in node_ids]

        feasible = True

        for n in node_order:
            outs = n.downstream_edges
            if not outs:
                continue

            # incoming width = sum of upstream edge widths; seed with total_width at sources
            if n.upstream_edges:
                width_in = sum(self.edges[eid].width for eid in n.upstream_edges)
            else:
                width_in = total_width if total_width and total_width > 0 else 0.0

            if width_in <= 0:
                width_in = 1e-6

            # initial split proposal
            if split_mode == "equal" or len(outs) == 1:
                proposal = {eid: width_in / len(outs) for eid in outs}

            elif split_mode == "random" and len(outs) == 2:
                lo = max(0.0, min_pct)
                hi = min(1.0, 1.0 - min_pct)
                if hi <= lo:
                    lo = hi = 0.5
                frac = self.rng.uniform(lo, hi)
                wA = width_in * frac
                wB = width_in - wA
                proposal = {outs[0]: wA, outs[1]: wB}

            else:
                # fallback: equal split for multi-way random
                proposal = {eid: width_in / len(outs) for eid in outs}

            # apply length-based caps, then redistribute remaining if possible
            caps = {}
            for eid in outs:
                e = self.edges[eid]
                dx = max(self._min_dx, e.x_end - e.x_start)
                caps[eid] = dx / min_length_ratio

            widths = {eid: min(proposal[eid], caps[eid]) for eid in outs}
            total_assigned = sum(widths.values())

            if total_assigned + 1e-9 < width_in:
                # distribute remaining to unsaturated edges proportionally to spare capacity
                spare = {eid: max(0.0, caps[eid] - widths[eid]) for eid in outs}
                total_spare = sum(spare.values())
                if total_spare > 0:
                    for eid in outs:
                        add = spare[eid] / total_spare * (width_in - total_assigned)
                        widths[eid] += add
                total_assigned = sum(widths.values())
                if total_assigned + 1e-6 < width_in:
                    feasible = False  # could not meet target due to caps

            # if still over target (due to caps), scale down uniformly
            if total_assigned > 0 and total_assigned != width_in:
                f = width_in / total_assigned
                for eid in outs:
                    widths[eid] *= f

            for eid, w in widths.items():
                self.edges[eid].width = max(1e-6, w)

        return feasible

    def enforce_min_length_vs_width(self, ratio: float):
        """
        Strictly enforce: width <= length / ratio
        """
        for e in self.edges.values():
            dx = e.x_end - e.x_start
            if dx <= 0:
                e.width = 1e-6
                continue
            max_w = dx / ratio
            if e.width > max_w:
                e.width = max_w
            if e.width <= 0:
                e.width = 1e-6

    def rescale_lengths_to_total(self, target_total_length: Optional[float]):
        """
        Uniformly rescale sinuosity (and resulting centerline lengths)
        to hit a target total network length. Does not alter x positions.
        """
        if target_total_length is None or target_total_length <= 0:
            return

        total_len = sum(e.centerline_length for e in self.edges.values())
        if total_len <= 0:
            return

        factor = target_total_length / total_len
        for e in self.edges.values():
            e.sinuosity *= factor
            dx = e.x_end - e.x_start
            e.centerline_length = e.sinuosity * dx

    # ============================================================
    # Layout (with total-width constraint)
    # ============================================================

    def build_x_grid(self, N=300):
        base = np.linspace(0.0, self.domain_length, N)
        node_x = {n.x for n in self.nodes.values()}
        return np.unique(np.concatenate([base, list(node_x)]))

    def active_edges_at(self, x_grid):
        out = []
        for x in x_grid:
            active = [eid for eid, e in self.edges.items()
                      if e.x_start <= x <= e.x_end]
            out.append(active)
        return out

    def compute_global_edge_order(self):
        es = sorted(self.edges.values(),
                    key=lambda e: (self.nodes[e.upstream_node].x, e.id))
        return [e.id for e in es]

    def assign_lateral_slots(self,
                             x_grid,
                             gap_min=5.0,
                             total_width: Optional[float] = None,
                             min_length_ratio: float = 5.0):
        """
        Hybrid spacing with a cross-section total-width constraint:

        - Baseline width = e.width (already <= length / min_length_ratio)
        - At each x:
            * Let base_w[eid] = e.width for active edges
            * If total_width is set:
                - If sum(base_w) > total_width:
                    scale down uniformly
                - If sum(base_w) < total_width:
                    scale up but never above dx/ratio
        - We use these 'geom widths' only for plotting & slots.
        """
        active_per_x = self.active_edges_at(x_grid)
        layout: Dict[int, List[Tuple[float, float, float]]] = {eid: [] for eid in self.edges}

        order = self.compute_global_edge_order()
        lane = {eid: i for i, eid in enumerate(order)}

        # parent-child relationships for small visual offsets
        parent = {}
        for n in self.nodes.values():
            for e_up in n.upstream_edges:
                for e_down in n.downstream_edges:
                    if e_up != e_down:
                        parent[e_down] = e_up

        for x, act in zip(x_grid, active_per_x):
            if not act:
                continue

            act = sorted(act, key=lambda e: lane[e])
            base_w = {eid: self.edges[eid].width for eid in act}
            sum_base = sum(base_w.values())

            # default: use base widths as geometry widths
            geom_w = dict(base_w)

            if total_width is not None and total_width > 0 and sum_base > 0:
                # Each edge has a hard max based on length & ratio
                w_cap = {}
                for eid in act:
                    e = self.edges[eid]
                    dx = e.x_end - e.x_start
                    w_cap[eid] = dx / min_length_ratio

                # Case 1: sum_base >= target -> simple uniform downscale
                if sum_base >= total_width:
                    f = total_width / sum_base
                    for eid in act:
                        geom_w[eid] = base_w[eid] * f
                else:
                    # Case 2: sum_base < target -> scale UP with caps
                    sum_caps = sum(w_cap[eid] for eid in act)
                    if sum_caps <= total_width:
                        geom_w = w_cap
                    else:
                        low, high = 1.0, max(
                            (w_cap[eid] / base_w[eid]) if base_w[eid] > 0 else 1.0
                            for eid in act
                        )
                        for _ in range(40):
                            mid = 0.5 * (low + high)
                            test_sum = 0.0
                            for eid in act:
                                val = base_w[eid] * mid
                                if val > w_cap[eid]:
                                    val = w_cap[eid]
                                test_sum += val
                            if test_sum > total_width:
                                high = mid
                            else:
                                low = mid
                        f = low
                        for eid in act:
                            val = base_w[eid] * f
                            if val > w_cap[eid]:
                                val = w_cap[eid]
                            geom_w[eid] = val

            # Now we have geom_w[eid] for visual slots
            ycur = 0.0
            for eid in act:
                w = geom_w[eid]
                e = self.edges[eid]

                # small hierarchical offset for children
                offset = 0.0
                if eid in parent:
                    offset = 0.2 * self.edges[parent[eid]].width

                yc = ycur + 0.5 * w + offset
                layout[eid].append((x, yc, w))
                ycur += w + gap_min

        return layout

    # Snap nodes to y-positions in layout
    def align_nodes_in_layout(self, layout, tol=1e-6):
        for n in self.nodes.values():
            x0 = n.x
            hits = []
            for eid, samples in layout.items():
                for i, (x, y, w) in enumerate(samples):
                    if abs(x - x0) <= tol:
                        hits.append((eid, i, x, y, w))
            if not hits:
                continue
            avg_y = sum(h[3] for h in hits) / len(hits)
            n.y = avg_y
            for eid, i, x, y, w in hits:
                layout[eid][i] = (x0, avg_y, w)
        return layout

    @staticmethod
    def _smooth_1d(vals, window=5):
        if window <= 1 or len(vals) <= window:
            return vals[:]
        pad = [vals[0]] * (window // 2) + vals + [vals[-1]] * (window // 2)
        out = []
        for i in range(len(vals)):
            seg = pad[i:i+window]
            out.append(sum(seg) / len(seg))
        return out

    def generate_centerlines(self, layout, smooth=True):
        """
        Build geometry that starts/ends at node coords, with internal points
        from layout. Ensures edges connect perfectly at nodes.
        """
        for eid, samples in layout.items():
            e = self.edges[eid]
            up = self.nodes[e.upstream_node]
            dn = self.nodes[e.downstream_node]

            s_sorted = sorted(samples, key=lambda t: t[0])
            xs = [s[0] for s in s_sorted]
            ys = [s[1] for s in s_sorted]

            if len(xs) == 1:
                xs = [xs[0] - 1e-6, xs[0] + 1e-6]
                ys = [ys[0], ys[0]]

            if smooth and len(xs) > 3:
                ys = self._smooth_1d(ys, 5)

            coords = [(up.x, up.y)]
            for x, y in zip(xs, ys):
                if up.x < x < dn.x:
                    coords.append((x, y))
            coords.append((dn.x, dn.y))

            if len(coords) < 2:
                coords.append((dn.x + 1e-6, dn.y))

            e.geometry = coords

    # ============================================================
    # Soft intersection resolution (geometry only)
    # ============================================================

    def resolve_intersections_soft(self,
                                   clearance_factor: float = 0.4,
                                   max_iter: int = 8):
        """
        Reduce unrealistic crossings by gently pushing overlapping
        reaches apart in the lateral (y) direction.

        - Uses attribute widths to define a pairwise clearance:
              min_clearance = clearance_factor * (w_i + w_j)
        - Never moves nodes (endpoints) directly.
        - Only adjusts midpoints of each edge geometry.
        - Does NOT change topology, x_start/x_end, or widths.
        """
        # Precompute global lane order for consistent directions
        order = self.compute_global_edge_order()
        lane = {eid: i for i, eid in enumerate(order)}

        for _ in range(max_iter):
            # Build LineStrings for current geometry
            lines = {}
            for eid, e in self.edges.items():
                if e.geometry and len(e.geometry) >= 2:
                    lines[eid] = LineString(e.geometry)

            shifts_y = {eid: 0.0 for eid in lines.keys()}
            any_close = False

            eids = list(lines.keys())
            n = len(eids)
            for i in range(n):
                for j in range(i + 1, n):
                    id1 = eids[i]
                    id2 = eids[j]
                    line1 = lines[id1]
                    line2 = lines[id2]

                    # quick check: if bounding boxes don't overlap in x, skip
                    if line1.bounds[2] < line2.bounds[0] or line2.bounds[2] < line1.bounds[0]:
                        continue

                    dist = line1.distance(line2)
                    w1 = self.edges[id1].width
                    w2 = self.edges[id2].width
                    clearance = clearance_factor * (w1 + w2)

                    if dist < clearance:
                        any_close = True
                        delta = 0.5 * (clearance - dist)

                        # Decide direction based on lane ordering
                        if lane[id1] <= lane[id2]:
                            shifts_y[id1] -= delta
                            shifts_y[id2] += delta
                        else:
                            shifts_y[id1] += delta
                            shifts_y[id2] -= delta

            if not any_close:
                break  # stable

            # Apply tapered vertical shifts (nodes/endpoints ~ fixed)
            for eid, dy in shifts_y.items():
                if abs(dy) < 1e-9:
                    continue
                e = self.edges[eid]
                coords = e.geometry
                if not coords or len(coords) < 3:
                    continue  # nothing to bend

                new_coords = []
                npts = len(coords)
                for k, (x, y) in enumerate(coords):
                    # keep endpoints anchored (weight 0)
                    t = k / (npts - 1)
                    # piecewise-linear weight, max in middle
                    w = min(t, 1.0 - t) * 2.0  # 0 at ends, 1 in center
                    new_coords.append((x, y + dy * w))
                e.geometry = new_coords

    # ============================================================
    # Hard fix: edges crossing existing nodes
    # ============================================================

    def fix_edges_crossing_nodes(self, tol=1e-6):
        """
        Detect edges whose centerline passes exactly through a node
        that they are not connected to, and split those edges at the
        existing node to create a proper junction.

        This solves cases like:
        - edge 10 going straight through node 12.
        """
        # Build LineStrings for distance checks
        lines = {}
        for eid, e in self.edges.items():
            if e.geometry and len(e.geometry) >= 2:
                lines[eid] = LineString(e.geometry)

        # For each node and each unrelated edge, check for crossing
        for nid, node in self.nodes.items():
            p = Point(node.x, node.y)
            for eid, line in list(lines.items()):
                e = self.edges.get(eid)
                if e is None:
                    continue

                # skip edges that already end at this node
                if e.upstream_node == nid or e.downstream_node == nid:
                    continue

                # check if node is interior in x-range
                if not (e.x_start < node.x < e.x_end):
                    continue

                # if distance is effectively zero, treat as crossing
                if line.distance(p) < tol:
                    # split this edge at existing node
                    self._split_edge_at_existing_node(eid, nid)

        # After we changed topology, ensure consistency again
        self.validate_and_repair_topology()

    # ============================================================
    # end class
    # ============================================================


# ============================================================
# GIS Export
# ============================================================

def edges_to_gdf(net, crs=None):
    rec = []
    for e in net.edges.values():
        if e.geometry and len(e.geometry) >= 2:
            rec.append({
                "edge_id": e.id,
                "upstream": e.upstream_node,
                "downstream": e.downstream_node,
                "width_attr": e.width,
                "sinuosity": e.sinuosity,
                "length": e.centerline_length,
                "geometry": LineString(e.geometry),
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
            "geometry": Point(n.x, n.y),
        })
    return gpd.GeoDataFrame(rec, geometry="geometry", crs=crs)


# ============================================================
# Plotly
# ============================================================

def plot_network_plotly(net, title="", show=True, output_html=None):
    fig = go.Figure()
    for e in net.edges.values():
        if not e.geometry:
            continue
        xs = [p[0] for p in e.geometry]
        ys = [p[1] for p in e.geometry]
        fig.add_trace(go.Scatter(
            x=xs,
            y=ys,
            mode="lines",
            line=dict(width=max(1, e.width / 10)),
            hoverinfo="text",
            text=f"Edge {e.id}<br>W_attr={e.width:.2f}<br>S={e.sinuosity:.2f}"
        ))
    fig.update_layout(
        title=title,
        yaxis=dict(scaleanchor="x", scaleratio=1),
        template="plotly_white",
        showlegend=False
    )
    if output_html:
        fig.write_html(output_html)
    if show:
        fig.show()


# ============================================================
# CLI
# ============================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="Synthetic directed river network with width–length and total-width constraints."
    )
    p.add_argument("--domain-length", type=float, default=1000)
    p.add_argument("--domain-width", type=float, default=200)
    p.add_argument("--n-bifurcations", type=int, default=3)
    p.add_argument("--x-points", type=int, default=400)
    p.add_argument("--gap-min", type=float, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--min-length-ratio", type=float, default=5.0,
                  help="Minimum reach length / width ratio (L/W).")
    p.add_argument("--total-width", type=float, default=None,
                  help="Target sum of channel widths per cross-section (optional).")
    p.add_argument("--width-split-mode", type=str, choices=["equal", "random"],
                  default="equal", help="How to split total width at bifurcations.")
    p.add_argument("--width-split-min-pct", type=float, default=0.2,
                  help="Minimum fraction for a branch when split mode is random.")
    p.add_argument("--length-mode", type=str, choices=["domain", "total"],
                  default="domain", help="Domain: keep as-is. Total: rescale sinuosity to hit target total length.")
    p.add_argument("--target-total-length", type=float, default=None,
                  help="Target total network length when length-mode=total. Defaults to domain length if not set.")
    p.add_argument("--max-attempts", type=int, default=30,
                  help="Maximum attempts to place bifurcations while satisfying width/length constraints.")
    p.add_argument("--gpkg", type=str, default="river_network.gpkg")
    p.add_argument("--geojson", type=str, default="river_network.geojson")
    p.add_argument("--crs", type=str, default="EPSG:3857")
    p.add_argument("--no-plot", action="store_true")
    p.add_argument("--html-plot", type=str, default=None)
    return p.parse_args()


# ============================================================
# Main
# ============================================================

def main():
    args = parse_args()
    max_attempts = max(1, args.max_attempts)
    rng_base_seed = args.seed
    success = False
    best_net = None

    for attempt in range(max_attempts):
        rng = random.Random(rng_base_seed + attempt)

        # 1. topology
        net = RiverNetwork(args.domain_length, args.domain_width, rng)
        bif_ok = net.generate_bifurcations(args.n_bifurcations)
        net.validate_and_repair_topology()

        # verify actual bifurcation count after repairs
        actual_bif = sum(1 for n in net.nodes.values() if len(n.downstream_edges) > 1)
        if actual_bif < args.n_bifurcations:
            bif_ok = False

        # 2. attributes
        net.assign_sinuosity()
        feasible = net.assign_width_with_total(
            args.total_width,
            split_mode=args.width_split_mode,
            min_pct=args.width_split_min_pct,
            min_length_ratio=args.min_length_ratio
        )
        # Optionally rescale to hit a target total length by adjusting sinuosity
        target_len = args.target_total_length
        if args.length_mode == "total" and target_len is None:
            target_len = args.domain_length
        if args.length_mode == "total":
            net.rescale_lengths_to_total(target_len)
        net.enforce_min_length_vs_width(args.min_length_ratio)

        if feasible:
            best_net = net
            success = True
            if bif_ok:
                break
        else:
            best_net = net  # keep last attempt even if infeasible

        if not bif_ok:
            # not enough bifurcations placed; try another layout
            success = False

    if not success:
        raise RuntimeError(f"Failed to satisfy total width and bifurcation count within constraints after {max_attempts} attempts.")
    net = best_net

    # 3. layout & geometry (with total-width constraint)
    xg = net.build_x_grid(args.x_points)
    layout = net.assign_lateral_slots(
        xg,
        gap_min=args.gap_min,
        total_width=args.total_width,
        min_length_ratio=args.min_length_ratio
    )
    layout = net.align_nodes_in_layout(layout)
    net.generate_centerlines(layout)

    # 3b. hard fix: if any edge passes through an existing node, split it
    net.fix_edges_crossing_nodes(tol=1e-6)

    # 3c. after topology changes, we need to rebuild layout & geometry
    xg = net.build_x_grid(args.x_points)
    layout = net.assign_lateral_slots(
        xg,
        gap_min=args.gap_min,
        total_width=args.total_width,
        min_length_ratio=args.min_length_ratio
    )
    layout = net.align_nodes_in_layout(layout)
    net.generate_centerlines(layout)

    # 3d. soft intersection resolution for edge-edge overlaps
    net.resolve_intersections_soft(clearance_factor=0.4, max_iter=8)

    # 4. export
    edges_gdf = edges_to_gdf(net, crs=args.crs)
    nodes_gdf = nodes_to_gdf(net, crs=args.crs)

    if args.gpkg:
        edges_gdf.to_file(args.gpkg, driver="GPKG", layer="edges")
        nodes_gdf.to_file(args.gpkg, driver="GPKG", layer="nodes")
        # print(f"Saved {args.gpkg} (edges + nodes).")

    if args.geojson:
        edges_gdf.to_file(args.geojson, driver="GeoJSON")
        # print(f"Saved {args.geojson} (edges only).")

    # 5. plot
    if not args.no_plot:
        plot_network_plotly(
            net,
            title=f"Synth Network (bif={args.n_bifurcations}, L/W>={args.min_length_ratio})",
            output_html=args.html_plot,
            show=True
        )

    # print(f"Done: {len(net.nodes)} nodes, {len(net.edges)} edges")


if __name__ == "__main__":
    main()
