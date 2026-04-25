"""
Synthetic admissible river networks (NetworkX MultiDiGraph)

What this file provides (all-in-one):
- Discrete enumeration of all possible networks under your rules:
  * domain length L
  * initial corridor region [xs, xe]
  * total width W_total
  * number of breaks max_breaks
  * min_width + width_step constraints
  * break positions on x-grid (jump)
  * loop = TWO parallel loop edges, and (by default) it REPLACES the corridor segment xb..xr
  * cross = ONE straight edge A->B or B->A
- Width conservation enforced automatically via recompute_widths()
- K-stats admissibility computed on all edges with positive length and positive width
- Path counting that works with same-x connector edges (special ordering at xs/xe)
- Plotly interactive plotting with rich hover for every edge (including sampled crosses)
- Saving:
  * summary.parquet or summary.csv
  * networks.jsonl.gz (recipes; compact and reconstructable)
  * optional graph gpickles (heavy)

Notes about node naming (your requirement):
- At initialization, backbone only has 4 nodes:
    (0,"main"), (xs,"main"), (xe,"main"), (L,"main")
- instantiate_corridor(WA,WB) creates:
    (xs,"A"), (xe,"A"), (xs,"B"), (xe,"B")
  and later splits create only A/B nodes, never new "main" nodes.
"""

import copy
import gzip
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import networkx as nx
import plotly.graph_objects as go

from datetime import datetime as dt
import sys
# ============================================================
# Helpers: filesystem + json
# ============================================================

def _ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def _json_default(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, tuple):
        return list(obj)
    raise TypeError(f"Unserializable type: {type(obj)}")


def _edge_uid(u, v, key) -> str:
    return f"{u}->{v}::{key}"



# ============================================================
# Entropic braiding index
# ============================================================
def x_midpoints(G, x_attr="x"):
    xs = sorted({G.nodes[n][x_attr] for n in G.nodes})
    mids = [(xs[i] + xs[i+1]) / 2 for i in range(len(xs) - 1)]
    return xs, mids

def edges_spanning_x(G, x0):
    spanning = []
    for u, v,k, data in G.edges(keys = True,data=True):
        xu = u[0]
        xv = v[0]
        if min(xu, xv) < x0 < max(xu, xv):
            spanning.append((u, v, data))
    return spanning

def ebi(w):
    x = w/w.sum()
    H = -np.sum(x*np.log2(x))
    return 2**H

def metrics_by_midpoint(G, x_attr="x", width_attr="width"):
    xs, mids = x_midpoints(G, x_attr=x_attr)
    e =  []
    for x0 in mids[1:-1]:
        spanning = edges_spanning_x(G, x0)
        ws = np.array([float(data[width_attr]) for _, _, data in spanning])
        e.append(ebi(ws))
    return np.mean(e), np.max(e)



# ============================================================
# K utilities
# ============================================================

def compute_k(L, W, kb=20, S=1e-3, n=0.35):
    return (3/5) * n * (L / math.sqrt(S)) * (kb**(2/3)) / (W**(2/3))

def k_stats_from_graph(G: nx.MultiDiGraph, x_stability=0.3):
    """
    Computes k on ALL edges that have:
      - positive width
      - positive length (x[v]-x[u] > 0)
    This automatically excludes same-x connector edges.
    """
    ks = []
    for u, v, k, data in G.edges(keys=True, data=True):
        w = float(data.get("w", 0.0))
        if w <= 0:
            continue
        xu = float(G.nodes[u]["x"])
        xv = float(G.nodes[v]["x"])
        L = xv - xu
        if L <= 0:
            continue
        ks.append(compute_k(L, w))

    if not ks:
        return None

    ks = np.array(ks, dtype=float)
    kmin, kmax = ks.min(), ks.max()
    ratio = (kmax / kmin) if kmin > 0 else np.inf

    ebi_mean, ebi_max = metrics_by_midpoint(G, width_attr='w')
    return {
        "k_sum": float(ks.sum()),
        "k_min": float(kmin),
        "k_max": float(kmax),
        "k_mean": float(ks.mean()),
        "k_ratio": float(ratio),
        "ebi_mean": ebi_mean,
        "ebi_max": ebi_max,
        "admissible": bool(ratio <= (1 - x_stability) / x_stability),
    }

def admissable(ratio, x_stability = 0.1):
    return (ratio <= (1 - x_stability) / x_stability)

# ============================================================
# Discrete utilities
# ============================================================

def _grid_values(x0: float, x1: float, step: float) -> np.ndarray:
    return np.arange(x0, x1, step, dtype=float)


def _iter_width_splits_two(W_in: float, min_width: float, step: float):
    """
    All (W1, W2) with:
      W1 + W2 = W_in
      W1, W2 >= min_width
      both multiples of step
    """
    W_in = float(W_in)
    min_width = float(min_width)
    step = float(step)

    start = math.ceil(min_width / step) * step
    end = math.floor((W_in - min_width) / step) * step
    if end < start:
        return

    n_steps = int(round((end - start) / step)) + 1
    for i in range(n_steps):
        W1 = start + i * step
        W2 = W_in - W1
        W1 = round(W1, 10)
        W2 = round(W2, 10)
        if W2 < min_width - 1e-9:
            continue
        if abs((W2 / step) - round(W2 / step)) > 1e-9:
            continue
        yield (W1, W2)


def _half_sine(x, x0, x1):
    t = (x - x0) / (x1 - x0)
    return np.sin(np.pi * t)

# ============================================================
# Overlap check
# ============================================================

def _disjoint(a: Tuple[float, float], b: Tuple[float, float]) -> bool:
    # allow disjoint or full nesting (either direction)
    a0, a1 = a; b0, b1 = b
    disjoint = (a1 <= b0 + 1e-12) or (b1 <= a0 + 1e-12)
    return disjoint

def _cross_loop_intersect(a: float, b: Tuple[float,float]) -> bool:
    b0, b1 = b
    return (b0 <= a <= b1)

def _crosses_intersect(a: Tuple[float,float], b: Tuple[float,float]) -> bool:
    # a=(xb1,xr1), b=(xb2,xr2)
    xb1, xr1 = a
    xb2, xr2 = b
    # strict intersection if ordering reversed
    return np.sign(xb1 - xb2) != np.sign(xr1 - xr2)

# ============================================================
# Parameters
# ============================================================

@dataclass(frozen=True)
class Params:
    L: float
    W_total: float
    xs: float
    xe: float
    jump: float
    max_breaks: int

    min_width: float = 20.0
    width_step: float = 10.0
    x_stability: float = 0.3

    # plotting geometry
    Y0: float = 1.0
    amp_corr: float = 1.5
    amp_loop: float = 0.7


# ============================================================
# Core model
# ============================================================

class RiverNetworkNX:
    """
    Nodes: (x, tag) where tag ∈ {"main","A","B"}.
    Backbone init: ONLY 4 nodes (0,xs,xe,L) on "main" and ONLY 2 main edges.
    instantiate_corridor(WA,WB): creates A/B endpoints at xs/xe + connectors + initial corridor edges.

    Breaks:
    - Loop (A->A or B->B): creates TWO loop edges nb->nr; by default replaces corridor segment nb..nr.
    - Cross (A->B or B->A): creates ONE cross edge nb->nr.

    Width conservation:
    - recompute_widths() sweeps A and B nodes by x, sending flow into:
        * outgoing corridor (if exists)
        * outgoing connector at xe (if exists)
        * loop/cross edges (break outflow)
    """

    def __init__(self, params: Params):
        self.p = params
        self.G = nx.MultiDiGraph()
        self.breaks: List[dict] = []
        self.WA0: Optional[float] = None
        self.WB0: Optional[float] = None
        self._init_backbone()

    # ----------------------------
    # Nodes / backbone
    # ----------------------------

    def _node(self, x: float, tag: str):
        x = round(float(x), 6)
        n = (x, tag)
        if n not in self.G:
            self.G.add_node(n, x=x, tag=tag)
        return n

    def _init_backbone(self):
        """
        Only 4 nodes and main edges.
        """
        p = self.p
        self.G.clear()
        self.breaks.clear()

        n0 = self._node(0.0, "main")
        ns = self._node(p.xs, "main")
        ne = self._node(p.xe, "main")
        nL = self._node(p.L, "main")

        self.G.add_edge(n0, ns, key="main_up", kind="main", branch="main", w=p.W_total, curve=0)
        self.G.add_edge(ne, nL, key="main_dn", kind="main", branch="main", w=p.W_total, curve=0)

    # ----------------------------
    # Corridor instantiation (renamed)
    # ----------------------------

    def instantiate_corridor(self, WA: float, WB: float):
        """
        Create the A/B corridor region between xs and xe, including:
          - A/B endpoint nodes at xs, xe
          - connectors main<->A/B at xs/xe
          - initial corridor edges xsA->xeA and xsB->xeB

        After calling this, all later created nodes via splitting are (x,"A") or (x,"B").
        """
        p = self.p

        # ensure main nodes exist
        ns = self._node(p.xs, "main")
        ne = self._node(p.xe, "main")

        # create branch endpoint nodes
        # xsA = self._node(p.xs, "A")
        # xeA = self._node(p.xe, "A")
        # xsB = self._node(p.xs, "B")
        # xeB = self._node(p.xe, "B")

        WA = float(WA)
        WB = float(WB)

        if abs((WA + WB) - p.W_total) > 1e-6:
            # optional: you can relax this if you want, but it’s consistent with conservation
            raise ValueError("WA + WB must equal W_total")
        self.WA0 = WA
        self.WB0 = WB
        # connectors
        # if not self.G.has_edge(ns, xsA, "conn_A_in"):
        #     self.G.add_edge(ns, xsA, key="conn_A_in", kind="connector", branch="A", w=WA, curve=0)
        # else:
        #     self.G.edges[ns, xsA, "conn_A_in"]["w"] = WA

        # if not self.G.has_edge(ns, xsB, "conn_B_in"):
        #     self.G.add_edge(ns, xsB, key="conn_B_in", kind="connector", branch="B", w=WB, curve=0)
        # else:
        #     self.G.edges[ns, xsB, "conn_B_in"]["w"] = WB

        # if not self.G.has_edge(xeA, ne, "conn_A_out"):
        #     self.G.add_edge(xeA, ne, key="conn_A_out", kind="connector", branch="A", w=WA, curve=0)
        # else:
        #     self.G.edges[xeA, ne, "conn_A_out"]["w"] = WA

        # if not self.G.has_edge(xeB, ne, "conn_B_out"):
        #     self.G.add_edge(xeB, ne, key="conn_B_out", kind="connector", branch="B", w=WB, curve=0)
        # else:
        #     self.G.edges[xeB, ne, "conn_B_out"]["w"] = WB

        # initial corridor edges
        if not self.G.has_edge(ns, ne, "A0"):
            self.G.add_edge(ns, ne, key="A0", kind="corridor", branch="A", w=WA, curve=+1)
        else:
            self.G.edges[ns, ne, "A0"]["w"] = WA

        if not self.G.has_edge(ns, ne, "B0"):
            self.G.add_edge(ns, ne, key="B0", kind="corridor", branch="B", w=WB, curve=-1)
        else:
            self.G.edges[ns, ne, "B0"]["w"] = WB

        self.recompute_widths()

    # ----------------------------
    # Corridor splitting
    # ----------------------------

    def _corridor_edges(self, branch: str):
        for u, v, k, d in self.G.edges(keys=True, data=True):
            if d.get("kind") == "corridor" and d.get("branch") == branch:
                yield (u, v, k, d)

    def _find_corridor_edge_covering(self, branch: str, x: float):
        x = float(x)
        for u, v, k, d in self._corridor_edges(branch):
            xu = self.G.nodes[u]["x"]
            xv = self.G.nodes[v]["x"]
            if xu < x < xv:
                return (u, v, k)
        return None

    def split_corridor_at(self, branch: str, x: float):
        """
        Split the corridor edge on 'branch' that covers x.

        The corridor edge endpoints may be:
        - main nodes (xs/xe)
        - or interior branch nodes (x,'A') / (x,'B')

        This creates an interior branch node (x, branch) and replaces u->v with u->n and n->v.
        """
        x = round(float(x), 6)
        found = self._find_corridor_edge_covering(branch, x)
        if found is None:
            raise ValueError(f"No corridor edge on branch {branch} covers x={x}")
        
        u, v, key = found
        xu = self.G.nodes[u]["x"]
        xv = self.G.nodes[v]["x"]
        if not (xu < x < xv):
            raise ValueError("Invalid split location")

        n = self._node(x, branch)
        data = self.G.edges[u, v, key].copy()
        w = float(data["w"])

        self.G.remove_edge(u, v, key)

        kL = f"{key}_L@{x}"
        kR = f"{key}_R@{x}"
        self.G.add_edge(u, n, key=kL, **{**data, "w": w})
        self.G.add_edge(n, v, key=kR, **{**data, "w": w})
        return n

    def break_is_legal(self, kind: str, bf: str, bt: str, xb: float, xr: float) -> bool:
        xb = float(xb); xr = float(xr)
        newI = (xb, xr)

        eps = 1e-12

        for b in self.breaks:
            old_kind = b["kind"]
            old_f = b["from_branch"]
            old_t = b["to_branch"]


            set1, set2 = {(xb,bf), (xr,bt)}, {(float(b["xb"]),old_f), (float(b["xr"]),old_t)}
            # Check that none from list1 are in list2
            if len(set1 | set2) < 4:
                return False

            sorted_s1 = sorted(set1, key=lambda x: x[1])
            sorted_s2 = sorted(set2, key=lambda x: x[1])
            newI, oldI = [x[0] for x in sorted_s1], [x[0] for x in sorted_s2]
            # 1) cross-cross cannot geometrically intersect
            if kind == "cross" and old_kind == "cross":
                if _crosses_intersect(newI, oldI):
                    return False

            # unordered points for the loop-loop and the cross loop interactions
            newI = [xb, xr]
            oldI = [float(b['xb']), float(b['xr'])]
            # 2) loop-loop same branch must be disjoint
            if kind == "loop" and old_kind == "loop" and bf == old_f:
                if not _disjoint(newI, oldI):
                    return False

            # 3) cross vs loop: allowed to overlap in x-span,
            # but forbidden if the cross endpoint on the loop's branch lies inside the loop interval.
            if kind == "cross" and old_kind == "loop":
                loop_br = old_f
                if (bf == loop_br) and _cross_loop_intersect(xb, oldI):
                    return False
                if (bt == loop_br) and _cross_loop_intersect(xr, oldI):
                    return False

            if kind == "loop" and old_kind == "cross":
                loop_br = bf
                # old cross endpoint on its from-branch is oldI[0] (xb), on its to-branch is oldI[1] (xr)
                if (loop_br == old_f) and _cross_loop_intersect(oldI[0], newI):
                    return False
                if (loop_br == old_t) and _cross_loop_intersect(oldI[1], newI):
                    return False

        return True



    # ----------------------------
    # Loop corridor removal (replace corridor segment)
    # ----------------------------

    def _remove_corridor_chain(self, branch: str, nb, nr):
        """
        Remove the corridor path from nb to nr on the given branch.
        """
        corridor_sub = nx.DiGraph()
        for u, v, k, d in self._corridor_edges(branch):
            corridor_sub.add_edge(u, v, key=k)

        if not nx.has_path(corridor_sub, nb, nr):
            raise ValueError(f"No corridor path exists from {nb} to {nr} on branch {branch}")

        path = nx.shortest_path(corridor_sub, nb, nr)

        for i in range(len(path) - 1):
            u = path[i]
            v = path[i + 1]
            for k, d in list(self.G.get_edge_data(u, v, default={}).items()):
                if d.get("kind") == "corridor" and d.get("branch") == branch:
                    self.G.remove_edge(u, v, k)

        # prune isolated internal nodes
        for mid in path[1:-1]:
            if self.G.in_degree(mid) == 0 and self.G.out_degree(mid) == 0:
                self.G.remove_node(mid)

    # ----------------------------
    # Width queries
    # ----------------------------

    def corridor_incoming_width_at(self, branch: str, node: tuple) -> float:
        """
        node must be (x, branch). For xs, returns WA0/WB0.
        Else, returns width of the incoming corridor edge into node.
        """
        x = float(self.G.nodes[node]["x"])
        if abs(x - self.p.xs) < 1e-9:
            return float(self.WA0 if branch == "A" else self.WB0)

        for u, v, k, d in self.G.in_edges(node, keys=True, data=True):
            if d.get("kind") == "corridor" and d.get("branch") == branch:
                return float(d["w"])
        raise RuntimeError(f"No incoming corridor edge for {branch} into {node}")

    # ----------------------------
    # Width recomputation (conservation)
    # ----------------------------

    def recompute_widths(self):
        """
        Enforce conservation on corridor branches A and B.
        Outflow from a branch node u can go to:
          - loop/cross edges (breaks)
          - corridor continuation (if exists)
          - connector to (xe,"main") at xe (if exists)
        """
        p = self.p
        if self.WA0 is None or self.WB0 is None:
            raise RuntimeError("Call instantiate_corridor(WA,WB) first")

        # xsA = self._node(p.xs, "A")
        # xeA = self._node(p.xe, "A")
        # xsB = self._node(p.xs, "B")
        # xeB = self._node(p.xe, "B")
        ne = self._node(p.xe, "main")
        ns = self._node(p.xs, "main")

        corridor_nodes = set([ns, ne])
        # corridor_nodes = set([xsA, xeA, xsB, xeB])
        for br in ("A", "B"):
            for u, v, k, d in self._corridor_edges(br):
                corridor_nodes.add(u)
                corridor_nodes.add(v)

        nodes_sorted = sorted(corridor_nodes, key=lambda n: self.G.nodes[n]["x"])

        inflow = {n: 0.0 for n in nodes_sorted}
        # Inject initial branch inflows at xs main
        inflow[ns] = float(self.WA0 + self.WB0)  # total, mostly for checking
        # print('inflow', inflow, float(self.WA0), float(self.WB0))
        # inflow[xsA] = float(self.WA0)
        # inflow[xsB] = float(self.WB0)

        # def outgoing_corridor(u):
        #     # br = u[1]
        #     outs = []
        #     for _, v, k, d in self.G.out_edges(u, keys=True, data=True):
        #         if d.get("kind") == "corridor" and d.get("branch") == br:
        #             outs.append((v, k))
        #     if len(outs) > 1:
        #         raise RuntimeError(f"Multiple outgoing corridor edges from {u}")
        #     return outs[0] if outs else None
        def outgoing_corridor(u, branch: str):
            outs = []
            for _, v, k, d in self.G.out_edges(u, keys=True, data=True):
                if d.get("kind") == "corridor" and d.get("branch") == branch:
                    outs.append((v, k))
            if len(outs) > 1:
                raise RuntimeError(f"Multiple outgoing corridor edges from {u} for branch {branch}")
            return outs[0] if outs else None

        def outgoing_connector_to_main(u):
            outs = []
            for _, v, k, d in self.G.out_edges(u, keys=True, data=True):
                if d.get("kind") == "connector" and v == ne:
                    outs.append((v, k))
            if len(outs) > 1:
                raise RuntimeError(f"Multiple outgoing connectors from {u} to main")
            return outs[0] if outs else None

        # for u in nodes_sorted:
        #     if u[1] not in ("A", "B"):
        #         continue

        #     Win = inflow[u]

        #     # break outflows
        #     break_out = 0.0
        #     for _, v, k, d in self.G.out_edges(u, keys=True, data=True):
        #         if d.get("kind") in ("loop", "cross"):
        #             w = float(d["w"])
        #             break_out += w
        #             if v in inflow:
        #                 inflow[v] += w

        #     remaining = Win - break_out
        #     if remaining < -1e-6:
        #         raise ValueError(f"Negative remaining width at {u}: Win={Win}, break_out={break_out}")

        #     out_corr = outgoing_corridor(u)
        #     out_conn = outgoing_connector_to_main(u)

        #     if out_corr is None and out_conn is None:
        #         # internal dead-end: all must leave via breaks
        #         if abs(remaining) > 1e-6:
        #             raise ValueError(
        #                 f"Width conservation violated at {u}: Win={Win}, break_out={break_out}, remaining={remaining}"
        #             )
        #         continue

        #     if out_corr is not None:
        #         v_corr, kcorr = out_corr
        #         Wcorr = remaining
        #         if Wcorr < p.min_width - 1e-9:
        #             raise ValueError(f"Corridor width below min at {u}: {Wcorr}")
        #         self.G.edges[u, v_corr, kcorr]["w"] = float(Wcorr)
        #         if v_corr in inflow:
        #             inflow[v_corr] += Wcorr
        #         remaining = 0.0

        #     if out_corr is None and out_conn is not None:
        #         v_conn, kconn = out_conn
        #         Wconn = remaining
        #         if Wconn < p.min_width - 1e-9:
        #             raise ValueError(f"Connector width below min at {u}: {Wconn}")
        #         self.G.edges[u, v_conn, kconn]["w"] = float(Wconn)
        #         remaining = 0.0

        #     if abs(remaining) > 1e-6:
        #         raise ValueError(f"Unassigned outflow at {u}: remaining={remaining}")

        # # keep incoming connectors at xs consistent
        # if self.G.has_edge(ns, xsA, "conn_A_in"):
        #     self.G.edges[ns, xsA, "conn_A_in"]["w"] = float(self.WA0)
        # if self.G.has_edge(ns, xsB, "conn_B_in"):
        #     self.G.edges[ns, xsB, "conn_B_in"]["w"] = float(self.WB0)
            # 1) Force the first corridor widths at xs main to be WA0/WB0
        for br, W0 in (("A", self.WA0), ("B", self.WB0)):
            out = outgoing_corridor(ns, br)
            if out is None:
                raise RuntimeError(f"Missing initial corridor edge for branch {br} from xs main")
            v, kcorr = out
            self.G.edges[ns, v, kcorr]["w"] = float(W0)
            if v in inflow:
                inflow[v] += float(W0)
            # if v is ne, inflow will accumulate there too

        # 2) Sweep all other nodes (excluding ns because we already handled its corridor emission)
        for u in nodes_sorted:
            if u == ns:
                continue

            tag = u[1]  # "main" or "A"/"B"
            Win = inflow[u]

            # Break outflows (loop/cross) exist only from branch nodes typically,
            # but we process them regardless.
            break_out = 0.0
            for _, v, k, d in self.G.out_edges(u, keys=True, data=True):
                if d.get("kind") in ("loop", "cross"):
                    w = float(d["w"])
                    break_out += w
                    if v in inflow:
                        inflow[v] += w

            remaining = Win - break_out
            if remaining < -1e-6:
                raise ValueError(f"Negative remaining width at {u}: Win={Win}, break_out={break_out}")

            # If this is a branch node, it may have corridor continuation on that branch
            if tag in ("A", "B"):
                out = outgoing_corridor(u, tag)
                if out is None:
                    # no corridor continuation: must fully leave via breaks
                    if abs(remaining) > 1e-6:
                        raise ValueError(
                            f"Width conservation violated at {u}: Win={Win}, break_out={break_out}, remaining={remaining}"
                        )
                    continue

                v, kcorr = out
                Wcorr = remaining
                if Wcorr < p.min_width - 1e-9:
                    raise ValueError(f"Corridor width below min at {u}: {Wcorr}")

                self.G.edges[u, v, kcorr]["w"] = float(Wcorr)
                if v in inflow:
                    inflow[v] += Wcorr
                continue

            # If tag == "main", the only main node we expect here is ne (xe main),
            # and it should have no corridor continuation. We just accumulate inflow there.
            # (main_dn is fixed width; optionally you can check consistency below.)

        # 3) Optional: enforce that total arriving at xe main equals W_total
        if abs(inflow[ne] - p.W_total) > 1e-6:
            raise ValueError(f"Total width arriving at xe main is {inflow[ne]}, expected {p.W_total}")
    # ----------------------------
    # Break operations
    # ----------------------------

    def add_loop(self, branch: str, xb: float, xr: float, W1: float, W2: float, replace_corridor: bool = True):
        branch = str(branch)
        xb = float(xb)
        xr = float(xr)
        W1 = float(W1)
        W2 = float(W2)
        if branch not in ("A", "B") or xr <= xb:
            raise ValueError

        nb = self.split_corridor_at(branch, xb)
        nr = self.split_corridor_at(branch, xr)

        if replace_corridor:
            self._remove_corridor_chain(branch, nb, nr)

        self.G.add_edge(nb, nr, key=f"loop+@{branch}{xb}->{xr}", kind="loop",
                        from_branch=branch, to_branch=branch, w=W1, curve=+1)
        self.G.add_edge(nb, nr, key=f"loop-@{branch}{xb}->{xr}", kind="loop",
                        from_branch=branch, to_branch=branch, w=W2, curve=-1)

        self.breaks.append(dict(kind="loop", from_branch=branch, to_branch=branch,
                                xb=xb, xr=xr, w1=W1, w2=W2, replace_corridor=bool(replace_corridor)))

        self.recompute_widths()

    def add_cross(self, bf: str, bt: str, xb: float, xr: float, W_cross: float):
        bf = str(bf)
        bt = str(bt)
        xb = float(xb)
        xr = float(xr)
        W_cross = float(W_cross)
        if bf not in ("A", "B") or bt not in ("A", "B") or bf == bt or xr <= xb:
            raise ValueError

        nb = self.split_corridor_at(bf, xb)
        nr = self.split_corridor_at(bt, xr)

        self.G.add_edge(nb, nr, key=f"cross@{bf}{xb}->{bt}{xr}", kind="cross",
                        from_branch=bf, to_branch=bt, w=W_cross, curve=0)

        self.breaks.append(dict(kind="cross", from_branch=bf, to_branch=bt,
                                xb=xb, xr=xr, w_cross=W_cross))

        self.recompute_widths()

    # ----------------------------
    # Path counting (handles same-x connectors at xs/xe)
    # ----------------------------

    def _node_phase(self, n) -> int:
        """
        Defines a topological-like order even when edges have same x (connectors at xs/xe).
        At xs: main must come before A/B (main->A/B)
        At xe: A/B must come before main (A/B->main)
        Else: doesn't matter much.
        """
        x = float(self.G.nodes[n]["x"])
        tag = n[1]
        if abs(x - self.p.xs) < 1e-9:
            return 0 if tag == "main" else 1
        if abs(x - self.p.xe) < 1e-9:
            return 0 if tag in ("A", "B") else 1
        return 0

    def count_paths(self) -> int:
        """
        Count directed paths from (0,'main') to (L,'main') in MultiDiGraph.
        Parallel edges count as distinct.
        """
        s = (round(0.0, 6), "main")
        t = (round(self.p.L, 6), "main")
        if s not in self.G or t not in self.G:
            raise RuntimeError("Missing source/sink main nodes")

        nodes = sorted(self.G.nodes, key=lambda n: (self.G.nodes[n]["x"], self._node_phase(n), n[1]))
        paths = {n: 0 for n in nodes}
        paths[s] = 1

        # assume no cycles (your x-based construction ensures this)
        for u in nodes:
            pu = paths[u]
            if pu == 0:
                continue
            for _, v, k, d in self.G.out_edges(u, keys=True, data=True):
                paths[v] += pu

        return int(paths[t])

    # ----------------------------
    # Plotly interactive plotting
    # ----------------------------

    def _precompute_corridor_centerlines(self, N=600):
        p = self.p
        xg = np.linspace(p.xs, p.xe, N)
        t = (xg - p.xs) / (p.xe - p.xs)
        bump = np.sin(np.pi * t)  # 0..1..0
        yA = (+p.Y0) * bump + (+p.amp_corr) * bump
        yB = (-p.Y0) * bump + (-p.amp_corr) * bump
        return xg, yA, yB

    def plotly_figure(self, N=600, cross_points=60):
        p = self.p
        fig = go.Figure()
        xg, yA, yB = self._precompute_corridor_centerlines(N=N)

        def y_on(tag, x):
            if tag == "A":
                return float(np.interp(x, xg, yA))
            if tag == "B":
                return float(np.interp(x, xg, yB))
            return 0.0

        def lw_from_w(w):
            return max(1.0, float(w) / 20.0)

        def add_trace(x, y, meta, lw):
            fig.add_trace(go.Scatter(
                x=x, y=y, mode="lines",
                line=dict(width=lw),
                customdata=[meta] * len(x),
                hovertemplate=(
                    "edge_id: %{customdata[0]}<br>"
                    "kind: %{customdata[1]}<br>"
                    "width: %{customdata[2]:.2f}<br>"
                    "length: %{customdata[3]:.2f}<br>"
                    "u: %{customdata[4]}<br>"
                    "v: %{customdata[5]}<extra></extra>"
                ),
                showlegend=False
            ))

        for u, v, k, d in self.G.edges(keys=True, data=True):
            x0 = float(self.G.nodes[u]["x"])
            x1 = float(self.G.nodes[v]["x"])
            kind = d.get("kind", "unknown")
            w = float(d.get("w", 1.0))
            length = x1 - x0
            edge_id = d.get("edge_id", _edge_uid(u, v, k))
            meta = [edge_id, kind, w, length, str(u), str(v)]
            lw = lw_from_w(w)

            if kind in ("main", "connector"):
                xx = np.linspace(x0, x1, 100)
                yy = np.linspace(0  , 0, 100)
                add_trace(xx, yy, meta, lw)

            elif kind == "corridor":
                br = d.get("branch")
                mask = (xg >= x0 - 1e-9) & (xg <= x1 + 1e-9)
                xx = xg[mask]
                yy = (yA if br == "A" else yB)[mask]
                if len(xx) < 2:
                    xx = np.array([x0, x1])
                    yy = np.array([y_on(br, x0), y_on(br, x1)])
                add_trace(xx, yy, meta, lw)

            elif kind == "cross":
                # straight, sampled so hover works along the line
                y0 = y_on(u[1], x0)
                y1 = y_on(v[1], x1)
                xx = np.linspace(x0, x1, cross_points)
                yy = np.linspace(y0, y1, cross_points)
                add_trace(xx, yy, meta, lw)

            elif kind == "loop":
                br = d.get("from_branch", u[1])
                xx = np.linspace(x0, x1, 140)
                base = np.interp(xx, xg, yA if br == "A" else yB)
                sign = +1 if int(d.get("curve", +1)) >= 0 else -1
                yy = base + sign * p.amp_loop * _half_sine(xx, x0, x1)
                add_trace(xx, yy, meta, lw)

            else:
                add_trace([x0, x1], [0.0, 0.0], meta, lw)

        fig.update_layout(
            template="plotly_white",
            xaxis_title="x",
            yaxis_title="y",
            height=480,
            margin=dict(l=20, r=20, t=20, b=20),
        )
        return fig

    def show_plotly(self, renderer="browser", N=600, cross_points=60):
        fig = self.plotly_figure(N=N, cross_points=cross_points)
        fig.show(renderer=renderer)
        return fig

    # ----------------------------
    # Saving: recipes
    # ----------------------------

    def to_recipe(self) -> dict:
        if self.WA0 is None or self.WB0 is None:
            raise RuntimeError("Corridor not instantiated")
        meta = dict(
            L=self.p.L, W_total=self.p.W_total,
            xs=self.p.xs, xe=self.p.xe,
            jump=self.p.jump,
            min_width=self.p.min_width,
            width_step=self.p.width_step,
            x_stability=self.p.x_stability,
            max_breaks=self.p.max_breaks,
            Y0=self.p.Y0, amp_corr=self.p.amp_corr, amp_loop=self.p.amp_loop,
        )
        return {
            "meta": meta,
            "initial_split": {"WA": float(self.WA0), "WB": float(self.WB0)},
            "breaks": copy.deepcopy(self.breaks),
        }

    @staticmethod
    def from_recipe(recipe: dict):
        meta = recipe["meta"]
        p = Params(
            L=meta["L"],
            W_total=meta["W_total"],
            xs=meta["xs"],
            xe=meta["xe"],
            jump=meta["jump"],
            max_breaks=meta.get("max_breaks", len(recipe.get("breaks", []))),
            min_width=meta["min_width"],
            width_step=meta["width_step"],
            x_stability=meta["x_stability"],
            Y0=meta.get("Y0", 1.0),
            amp_corr=meta.get("amp_corr", 1.5),
            amp_loop=meta.get("amp_loop", 0.7),
        )
        net = RiverNetworkNX(p)
        WA = recipe["initial_split"]["WA"]
        WB = recipe["initial_split"]["WB"]
        net.instantiate_corridor(WA, WB)

        for b in recipe.get("breaks", []):
            if b["kind"] == "loop":
                net.add_loop(
                    branch=b["from_branch"],
                    xb=b["xb"],
                    xr=b["xr"],
                    W1=b["w1"],
                    W2=b["w2"],
                    replace_corridor=b.get("replace_corridor", True),
                )
            elif b["kind"] == "cross":
                net.add_cross(
                    bf=b["from_branch"],
                    bt=b["to_branch"],
                    xb=b["xb"],
                    xr=b["xr"],
                    W_cross=b["w_cross"],
                )
            else:
                raise ValueError(f"Unknown break kind: {b['kind']}")
        return net


# ============================================================
# Check for duplicate networks
# ============================================================
def canonical_signature(net: RiverNetworkNX):
    """
    Canonical A<->B-invariant signature.
    Uses actual graph edges (corridor + loop/cross widths and x-positions).
    """

    G = net.G
    def extract(swap: bool):
        # map branch labels under swap
        def bmap(br):
            if not swap:
                return br
            return {"A": "B", "B": "A"}.get(br, br)

        # corridor segments: collect per mapped branch
        corr_A = []
        corr_B = []
        loops = []
        crosses = []

        # corridor
        for u, v, k, d in G.edges(keys=True, data=True):
            kind = d.get("kind")
            if kind == "corridor":
                br = bmap(d.get("branch"))
                xu = float(G.nodes[u]["x"])
                xv = float(G.nodes[v]["x"])
                w = round(float(d.get("w", 0.0)), 10)
                seg = (round(xu, 6), round(xv, 6), w)
                if br == "A":
                    corr_A.append(seg)
                elif br == "B":
                    corr_B.append(seg)

        corr_A.sort()
        corr_B.sort()

        # loops: group by (u,v) because there are two parallel edges
        # We'll detect both loop edges and record sorted widths.
        loop_groups = {}
        for u, v, k, d in G.edges(keys=True, data=True):
            if d.get("kind") == "loop":
                xu = round(float(G.nodes[u]["x"]), 6)
                xv = round(float(G.nodes[v]["x"]), 6)
                br = bmap(d.get("from_branch", u[1] if isinstance(u, tuple) else "A"))
                w = round(float(d.get("w", 0.0)), 10)
                key2 = (br, xu, xv)  # loop span on branch
                loop_groups.setdefault(key2, []).append(w)

        for (br, xb, xr), ws in loop_groups.items():
            ws = sorted(ws)
            # should be exactly 2, but keep robust:
            loops.append((br, xb, xr, tuple(ws)))

        loops.sort()

        # crosses: single edge; direction flips under swap
        for u, v, k, d in G.edges(keys=True, data=True):
            if d.get("kind") == "cross":
                xb = round(float(G.nodes[u]["x"]), 6)
                xr = round(float(G.nodes[v]["x"]), 6)
                w = round(float(d.get("w", 0.0)), 10)
                fb = bmap(d.get("from_branch", "A"))
                tb = bmap(d.get("to_branch", "B"))
                crosses.append((fb, tb, xb, xr, w))

        crosses.sort()

        # initial widths as unordered pair (already symmetric)
        init_pair = tuple(sorted([round(float(net.WA0), 10), round(float(net.WB0), 10)]))

        return (init_pair, tuple(corr_A), tuple(corr_B), tuple(loops), tuple(crosses))

    sig0 = extract(False)
    sig1 = extract(True)
    return min(sig0, sig1)

# ============================================================
# Enumeration
# ============================================================

def enumerate_admissible_networks_nx(p: Params, dtStart):
    nets: List[RiverNetworkNX] = []
    rows: List[dict] = []
    
    tested = 0
    tested_by_breaks = {d: 0 for d in range(p.max_breaks + 1)}
    admissible_by_breaks = {d: 0 for d in range(p.max_breaks + 1)}
    seen = set()
    xs_grid = _grid_values(p.xs + p.jump, p.xe, p.jump)

    for WA in np.arange(p.min_width, p.W_total, p.width_step, dtype=float):
        WB = p.W_total - WA
        if WB < p.min_width - 1e-9:
            continue

        base = RiverNetworkNX(p)

        # Your rule: start with only 4 main nodes, then instantiate corridor (first split region)
        base.instantiate_corridor(WA, WB)
        
        def dfs(net: RiverNetworkNX):
            nonlocal tested, nets, rows, dtStart
            # Avoid double networks (A-B copies)
            sig = canonical_signature(net)
            if sig in seen:
                return
            seen.add(sig)
            
            # Keep track of the number of tested networks
            tested += 1
            depth = len(net.breaks)
            tested_by_breaks[depth] += 1
            
            if (tested % 5000) == 0:
                print(f"Number of tested networks: {tested}, time elapsed: {dt.now() - dtStart}")
                dtStart = dt.now()

                print("tested", tested,
                    "seen", len(seen),
                    "nets", len(nets),
                    "rows", len(rows))
                print("sizes (bytes):",
                    "seen set", sys.getsizeof(seen),
                    "nets list", sys.getsizeof(nets),
                    "rows list", sys.getsizeof(rows))


            ks = k_stats_from_graph(net.G, p.x_stability)
            if ks is None or not ks["admissible"]:
                return
            admissible_by_breaks[depth] += 1

            network_id = len(nets)
            try:
                n_paths = net.count_paths()
            except Exception:
                return

            rows.append({
                "network_id": network_id,
                "n_breaks": len(net.breaks),
                "n_paths": n_paths,
                "WA0": float(net.WA0),
                "WB0": float(net.WB0),
                **ks
            })
            nets.append(net)
            

            if len(net.breaks) >= p.max_breaks:
                return

            for bf in ("A", "B"):
                for xb in xs_grid:
                    xr_candidates = _grid_values(xb + p.jump, p.xe + 1e-9, p.jump)
                    for xr in xr_candidates:
                        if xr <= xb or xr > p.xe + 1e-9:
                            continue

                        # temp split to read W_in at xb
                        tmp = copy.deepcopy(net)
                        try:
                            nb_tmp = tmp.split_corridor_at(bf, xb)
                            tmp.recompute_widths()
                            W_in = tmp.corridor_incoming_width_at(bf, nb_tmp)
                        except Exception:
                            continue

                        # CROSS: pick W_cross from discrete splits of incoming
                        bt = "B" if bf == "A" else "A"
                        if not net.break_is_legal(kind="cross",
                            bf=bf,bt=bt,xb=xb,xr=xr):
                            continue   # prune EARLY
                        for W_rem, W_cross in _iter_width_splits_two(W_in, p.min_width, p.width_step):
                            new = copy.deepcopy(net)
                            try:
                                new.add_cross(bf=bf, bt=bt, xb=xb, xr=xr, W_cross=W_cross)
                            except Exception:
                                continue
                            dfs(new)

                        # LOOP: since loop replaces corridor segment, W1+W2 should equal W_in (no remaining corridor)
                        # With min/step rules: enumerate (W1,W2) splits of W_in directly.
                        if not net.break_is_legal(kind="loop",
                            bf=bf,bt=bf,xb=xb,xr=xr):
                            continue   # prune EARLY
                        for W1, W2 in _iter_width_splits_two(W_in, p.min_width, p.width_step):
                            new = copy.deepcopy(net)
                            try:
                                new.add_loop(branch=bf, xb=xb, xr=xr, W1=W1, W2=W2, replace_corridor=True)
                            except Exception:
                                continue
                            dfs(new)

        dfs(base)

    df = pd.DataFrame(rows)
    if not df.empty:
        df["n_tested_total"]       = tested
        df["tested_n_breaks"] = df["n_breaks"].map(tested_by_breaks)
        df["admissible_n_breaks"] = df["n_breaks"].map(admissible_by_breaks)

        
        df["admissable_0.1"] = admissable(df['k_ratio'], 0.1)
        df["admissable_0.2"] = admissable(df['k_ratio'], 0.2)
        df["admissable_0.3"] = admissable(df['k_ratio'], 0.3)
    return nets, df, tested


# ============================================================
# Saving outputs
# ============================================================

def save_run_outputs(
    out_dir: str | Path,
    params: Params,
    summary_df: pd.DataFrame,
    networks: List[RiverNetworkNX],
    *,
    summary_format: str = "parquet",      # "parquet" or "csv"
    save_graph_pickles: bool = False,     # optional heavy output
    pickle_stride: int = 1,
):
    out = _ensure_dir(out_dir)

    meta_path = out / "run_meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(params.__dict__, f, indent=2, default=_json_default)

    if summary_format.lower() == "parquet":
        summary_path = out / "summary.parquet"
        summary_df.to_parquet(summary_path, index=False)
    elif summary_format.lower() == "csv":
        summary_path = out / "summary.csv"
        summary_df.to_csv(summary_path, index=False)
    else:
        raise ValueError("summary_format must be 'parquet' or 'csv'")

    recipes_path = out / "networks.jsonl.gz"
    with gzip.open(recipes_path, "wt", encoding="utf-8") as gz:
        for net in networks:
            gz.write(json.dumps(net.to_recipe(), default=_json_default) + "\n")

    written = {"meta": str(meta_path), "summary": str(summary_path), "recipes": str(recipes_path)}

    if save_graph_pickles:
        gp_dir = _ensure_dir(out / "graphs_gpickle")
        for i, net in enumerate(networks):
            if i % max(1, int(pickle_stride)) != 0:
                continue
            nx.write_gpickle(net.G, gp_dir / f"G_{i:06d}.gpickle")
        written["gpickle_dir"] = str(gp_dir)

    return written


# ============================================================
# Load a specific network by id from networks.jsonl.gz
# ============================================================

def load_network_by_id(recipes_gz_path: str | Path, network_id: int) -> RiverNetworkNX:
    with gzip.open(recipes_gz_path, "rt", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i == network_id:
                recipe = json.loads(line)
                return RiverNetworkNX.from_recipe(recipe)
    raise ValueError(f"Network {network_id} not found in {recipes_gz_path}")


# # ============================================================
# # Example run
# # ============================================================

# if __name__ == "__main__":
#     p = Params(
#         L=10000,
#         W_total=120,
#         xs=3000,
#         xe=8000,
#         jump=500,
#         max_breaks=2,
#         min_width=20,
#         width_step=20,
#         x_stability=0.000000001,
#     )

#     # sanity: backbone has only 4 nodes
#     tmp0 = RiverNetworkNX(p)
#     assert len(tmp0.G.nodes) == 4, f"Backbone nodes != 4, got {len(tmp0.G.nodes)}: {list(tmp0.G.nodes)}"

#     dtStart = dt.now()
#     nets, summary, n_tested = enumerate_admissible_networks_nx(p, dtStart)
#     print("networks tested:", n_tested)
#     print("admissible networks found:", len(nets))
#     print(summary.sort_values(['n_breaks', 'n_paths']).head(10))

#     paths = save_run_outputs(
#         out_dir=f"/Volumes/PhD/river_hierarchy/output/synthetic_network/synthetic_run_{p.max_breaks}",
#         # out_dir = "/Users/6256481/Desktop/PhD_icloud/USA_UNC/work/river_hierarchy/synthetic/output/new/",
#         params=p,
#         summary_df=summary,
#         networks=nets,
#         summary_format="parquet",
#         save_graph_pickles=False,
#     )
#     print(paths)

    # if nets:
    #     nets[-1].show_plotly(renderer="browser")


# ============================================================
# DROP-IN: streamed enumeration + chunked summary (bounded RAM)
# ============================================================
# What this does:
# - Keeps `seen` in memory (as you want).
# - DOES NOT keep `nets` in memory (optional small sample supported).
# - Writes ONE master recipes file: networks.jsonl.gz (batched writes).
# - Writes summary in chunks to summary_parts/, then merges to ONE final summary.parquet or summary.csv.
# - Returns (networks, df, tested) like before:
#     * networks will be [] unless keep_nets_in_memory=True (or keep a small sample).
#     * df is loaded from the final merged summary file (so no huge RAM during run).
#
# Requirements:
# - uses pandas + pyarrow (for parquet). If pyarrow isn't available, use summary_format="csv".
#
# Usage (example):
#   nets, summary, n_tested = enumerate_admissible_networks_nx_streamed(
#       p, dtStart, out_dir=".../output/run_001",
#       summary_format="parquet",
#       rows_chunk=50000,
#       recipe_chunk=5000,
#       keep_nets_in_memory=False,
#   )

import os
import shutil
from pathlib import Path

def _merge_summary_parts(parts_dir: Path, out_dir: Path, summary_format: str = "parquet") -> pd.DataFrame:
    """
    Merge chunk files from parts_dir into a single summary file in out_dir.
    Returns the merged DataFrame (loaded into memory at the end).
    If you expect the final summary to be huge, you can modify this to avoid loading everything at once.
    """
    parts = sorted(parts_dir.glob(f"summary_part_*.{ 'parquet' if summary_format=='parquet' else 'csv' }"))
    if not parts:
        # empty result
        df = pd.DataFrame()
        # still write an empty summary file for consistency
        if summary_format == "parquet":
            df.to_parquet(out_dir / "summary.parquet", index=False)
        else:
            df.to_csv(out_dir / "summary.csv", index=False)
        return df

    if summary_format == "parquet":
        # pandas concat of parts (OK if merged size is reasonable)
        dfs = [pd.read_parquet(p) for p in parts]
        df = pd.concat(dfs, ignore_index=True)
        df.to_parquet(out_dir / "summary.parquet", index=False)
    else:
        dfs = [pd.read_csv(p) for p in parts]
        df = pd.concat(dfs, ignore_index=True)
        df.to_csv(out_dir / "summary.csv", index=False)

    return df


def enumerate_admissible_networks_nx_streamed(
    p: Params,
    dtStart,
    *,
    out_dir: str | Path,
    summary_format: str = "parquet",   # "parquet" or "csv"
    rows_chunk: int = 50000,           # how many summary rows to buffer before dumping a part
    recipe_chunk: int = 5000,          # how many recipes to buffer before writing to gzip
    keep_nets_in_memory: bool = False, # set True only for small runs
    keep_nets_sample: int = 0,         # if >0, keep up to this many nets (first N) even if keep_nets_in_memory=False
    dedup_hash: bool = False,          # optional: store hashes instead of full signature (saves a lot of RAM for huge runs)
):
    """
    Drop-in alternative to enumerate_admissible_networks_nx(), but disk-streaming.

    Outputs written to out_dir:
      - run_meta.json  (you can still write this outside via save_run_outputs, but here we create dirs)
      - networks.jsonl.gz  (master recipes file)
      - summary_parts/summary_part_00000.parquet (or .csv)
      - summary.parquet (or summary.csv) merged at end

    Returns:
      nets_in_mem (maybe empty), merged_summary_df, tested
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)


    # write run metadata once (same as save_run_outputs)
    meta_path = out_dir / "run_meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(p.__dict__, f, indent=2, default=_json_default)



    parts_dir = out_dir / "summary_parts"
    if parts_dir.exists():
        shutil.rmtree(parts_dir)
    parts_dir.mkdir(parents=True, exist_ok=True)

    # master recipes file
    recipes_path = out_dir / "networks.jsonl.gz"
    if recipes_path.exists():
        recipes_path.unlink()

    # in-memory collections (bounded)
    nets_in_mem: List[RiverNetworkNX] = []
    rows_buf: List[dict] = []
    recipe_buf: List[str] = []

    tested = 0
    tested_by_breaks = {d: 0 for d in range(p.max_breaks + 1)}
    admissible_by_breaks = {d: 0 for d in range(p.max_breaks + 1)}

    # dedup
    seen = set()

    def _sig_token(net: RiverNetworkNX):
        sig = canonical_signature(net)
        if not dedup_hash:
            return sig
        # compact stable hash
        # (blake2b is fast; 16-byte digest is plenty for practical collision avoidance)
        import hashlib, json as _json
        b = _json.dumps(sig, separators=(",", ":"), default=_json_default).encode("utf-8")
        return hashlib.blake2b(b, digest_size=16).digest()

    # x-grid
    xs_grid = _grid_values(p.xs + p.jump, p.xe, p.jump)

    # write summary parts counter
    part_idx = 0

    # ID counter that matches line-number in networks.jsonl.gz
    network_id = 0

    # open gzip once
    with gzip.open(recipes_path, "wt", encoding="utf-8") as gz:

        # iterate initial splits
        for WA in np.arange(p.min_width, p.W_total, p.width_step, dtype=float):
            WB = p.W_total - WA
            if WB < p.min_width - 1e-9:
                continue

            base = RiverNetworkNX(p)
            base.instantiate_corridor(WA, WB)

            def dump_rows_part():
                nonlocal part_idx, rows_buf
                if not rows_buf:
                    return
                df_part = pd.DataFrame(rows_buf)
                suffix = "parquet" if summary_format.lower() == "parquet" else "csv"
                part_path = parts_dir / f"summary_part_{part_idx:05d}.{suffix}"
                if summary_format.lower() == "parquet":
                    df_part.to_parquet(part_path, index=False)
                else:
                    df_part.to_csv(part_path, index=False)
                rows_buf = []
                part_idx += 1

            def dump_recipe_batch():
                nonlocal recipe_buf
                if not recipe_buf:
                    return
                gz.write("\n".join(recipe_buf) + "\n")
                recipe_buf = []
                # no flush needed every time; OS+gzip buffering is fine.
                # But you can flush occasionally if you want crash-safety:
                # gz.flush()

            def dfs(net: RiverNetworkNX):
                nonlocal tested, dtStart, network_id, nets_in_mem, rows_buf, recipe_buf

                tok = _sig_token(net)
                if tok in seen:
                    return
                seen.add(tok)

                tested += 1
                depth = len(net.breaks)
                tested_by_breaks[depth] += 1

                if (tested % 5000) == 0:
                    print(f"Number of tested networks: {tested}, time elapsed: {dt.now() - dtStart}")
                    dtStart = dt.now()

                ks = k_stats_from_graph(net.G, p.x_stability)
                if ks is None or not ks["admissible"]:
                    return

                admissible_by_breaks[depth] += 1

                # count paths
                try:
                    n_paths = net.count_paths()
                except Exception:
                    return

                # buffer summary row
                row = {
                    "network_id": network_id,
                    "n_breaks": depth,
                    "n_paths": n_paths,
                    "WA0": float(net.WA0),
                    "WB0": float(net.WB0),
                    **ks
                }
                rows_buf.append(row)

                # buffer recipe line (matches network_id order)
                recipe_buf.append(json.dumps(net.to_recipe(), default=_json_default))

                # optionally keep small sample of graphs
                if keep_nets_in_memory or (keep_nets_sample > 0 and len(nets_in_mem) < keep_nets_sample):
                    nets_in_mem.append(net)

                network_id += 1

                # dump buffers if large
                if len(recipe_buf) >= int(recipe_chunk):
                    dump_recipe_batch()

                if len(rows_buf) >= int(rows_chunk):
                    dump_rows_part()

                # stop recursion if depth limit reached
                if depth >= p.max_breaks:
                    return

                # expand breaks
                for bf in ("A", "B"):
                    for xb in xs_grid:
                        xr_candidates = _grid_values(xb + p.jump, p.xe + 1e-9, p.jump)
                        for xr in xr_candidates:
                            if xr <= xb or xr > p.xe + 1e-9:
                                continue

                            # temp split to read W_in at xb
                            tmp = copy.deepcopy(net)
                            try:
                                nb_tmp = tmp.split_corridor_at(bf, xb)
                                tmp.recompute_widths()
                                W_in = tmp.corridor_incoming_width_at(bf, nb_tmp)
                            except Exception:
                                continue

                            bt = "B" if bf == "A" else "A"

                            # CROSS
                            if net.break_is_legal(kind="cross", bf=bf, bt=bt, xb=xb, xr=xr):
                                for W_rem, W_cross in _iter_width_splits_two(W_in, p.min_width, p.width_step):
                                    new = copy.deepcopy(net)
                                    try:
                                        new.add_cross(bf=bf, bt=bt, xb=xb, xr=xr, W_cross=W_cross)
                                    except Exception:
                                        continue
                                    dfs(new)

                            # LOOP
                            if net.break_is_legal(kind="loop", bf=bf, bt=bf, xb=xb, xr=xr):
                                for W1, W2 in _iter_width_splits_two(W_in, p.min_width, p.width_step):
                                    new = copy.deepcopy(net)
                                    try:
                                        new.add_loop(branch=bf, xb=xb, xr=xr, W1=W1, W2=W2, replace_corridor=True)
                                    except Exception:
                                        continue
                                    dfs(new)

            dfs(base)

        # final dumps
        if recipe_buf:
            gz.write("\n".join(recipe_buf) + "\n")
            recipe_buf = []

    # dump remaining summary rows
    if rows_buf:
        df_part = pd.DataFrame(rows_buf)
        suffix = "parquet" if summary_format.lower() == "parquet" else "csv"
        part_path = parts_dir / f"summary_part_{part_idx:05d}.{suffix}"
        if summary_format.lower() == "parquet":
            df_part.to_parquet(part_path, index=False)
        else:
            df_part.to_csv(part_path, index=False)
        rows_buf = []
        part_idx += 1

    # merge parts into final summary
    merged = _merge_summary_parts(parts_dir, out_dir, summary_format.lower())

    # add run-level columns and admissable thresholds (same as your original),
    # then overwrite final summary file
    if not merged.empty:
        merged["n_tested_total"] = tested
        merged["tested_n_breaks"] = merged["n_breaks"].map(tested_by_breaks)
        merged["admissible_n_breaks"] = merged["n_breaks"].map(admissible_by_breaks)

        merged["admissable_0.1"] = admissable(merged["k_ratio"], 0.1)
        merged["admissable_0.2"] = admissable(merged["k_ratio"], 0.2)
        merged["admissable_0.3"] = admissable(merged["k_ratio"], 0.3)

        if summary_format.lower() == "parquet":
            merged.to_parquet(out_dir / "summary.parquet", index=False)
        else:
            merged.to_csv(out_dir / "summary.csv", index=False)

    # clean up temp parts (so you end with just the final 3-ish files)
    # If you want to keep parts for debugging, comment this out.
    shutil.rmtree(parts_dir, ignore_errors=True)

    # return drop-in outputs
    return nets_in_mem, merged, tested

if __name__ == "__main__":
    p = Params(
        L=10000,
        W_total=120,
        xs=3000,
        xe=8000,
        jump=500,
        max_breaks=2,
        min_width=20,
        width_step=20,
        x_stability=0.000000001,
    )

    dtStart = dt.now()

    nets, summary, n_tested = enumerate_admissible_networks_nx_streamed(
        p, dtStart,
        out_dir=f"/Volumes/PhD/river_hierarchy/output/synthetic_network/synthetic_run_{p.max_breaks}",
        summary_format="parquet",   # or "csv" if parquet deps are annoying
        rows_chunk=50000,
        recipe_chunk=5000,
        keep_nets_in_memory=False,  # keep RAM small
        keep_nets_sample=0,
        dedup_hash=False,           # set True if seen gets too big later
    )

    print("networks tested:", n_tested)
    print("admissible networks found:", 0 if summary.empty else len(summary))
    print(summary.sort_values(['n_breaks', 'n_paths']).head(10))

    # you still get the same recipes file name:
    #   out_dir/networks.jsonl.gz
    # and the same summary name:
    #   out_dir/summary.parquet
