"""Core synthetic network classes and canonical recipe signatures."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import List, Optional

import networkx as nx
import numpy as np
import plotly.graph_objects as go

from .helpers import _cross_loop_intersect, _crosses_intersect, _disjoint, _edge_uid


def _half_sine(x, x0, x1):
    t = (x - x0) / (x1 - x0)
    return np.sin(np.pi * t)


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

    Y0: float = 1.0
    amp_corr: float = 1.5
    amp_loop: float = 0.7


class RiverNetworkNX:
    """
    Directed synthetic river network with a main stem and two corridor branches.

    The class starts from a 4-node backbone on the main channel. Once the
    corridor is instantiated, breaks can be added as same-branch loops or
    cross-branch shortcuts while width conservation is maintained by
    `recompute_widths()`.
    """

    def __init__(self, params: Params):
        self.p = params
        self.G = nx.MultiDiGraph()
        self.breaks: List[dict] = []
        self.WA0: Optional[float] = None
        self.WB0: Optional[float] = None
        self._init_backbone()

    def _node(self, x: float, tag: str):
        x = round(float(x), 6)
        n = (x, tag)
        if n not in self.G:
            self.G.add_node(n, x=x, tag=tag)
        return n

    def _init_backbone(self):
        p = self.p
        self.G.clear()
        self.breaks.clear()

        n0 = self._node(0.0, "main")
        ns = self._node(p.xs, "main")
        ne = self._node(p.xe, "main")
        nL = self._node(p.L, "main")

        self.G.add_edge(n0, ns, key="main_up", kind="main", branch="main", w=p.W_total, curve=0)
        self.G.add_edge(ne, nL, key="main_dn", kind="main", branch="main", w=p.W_total, curve=0)

    def instantiate_corridor(self, WA: float, WB: float):
        p = self.p
        ns = self._node(p.xs, "main")
        ne = self._node(p.xe, "main")

        WA = float(WA)
        WB = float(WB)
        if abs((WA + WB) - p.W_total) > 1e-6:
            raise ValueError("WA + WB must equal W_total")

        self.WA0 = WA
        self.WB0 = WB

        if not self.G.has_edge(ns, ne, "A0"):
            self.G.add_edge(ns, ne, key="A0", kind="corridor", branch="A", w=WA, curve=+1)
        else:
            self.G.edges[ns, ne, "A0"]["w"] = WA

        if not self.G.has_edge(ns, ne, "B0"):
            self.G.add_edge(ns, ne, key="B0", kind="corridor", branch="B", w=WB, curve=-1)
        else:
            self.G.edges[ns, ne, "B0"]["w"] = WB

        self.recompute_widths()

    def _corridor_edges(self, branch: str):
        for u, v, k, d in self.G.edges(keys=True, data=True):
            if d.get("kind") == "corridor" and d.get("branch") == branch:
                yield (u, v, k, d)

    def _find_corridor_edge_covering(self, branch: str, x: float):
        x = float(x)
        for u, v, k, _d in self._corridor_edges(branch):
            xu = self.G.nodes[u]["x"]
            xv = self.G.nodes[v]["x"]
            if xu < x < xv:
                return (u, v, k)
        return None

    def split_corridor_at(self, branch: str, x: float):
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
        xb = float(xb)
        xr = float(xr)

        for b in self.breaks:
            old_kind = b["kind"]
            old_f = b["from_branch"]
            old_t = b["to_branch"]

            set1 = {(xb, bf), (xr, bt)}
            set2 = {(float(b["xb"]), old_f), (float(b["xr"]), old_t)}
            if len(set1 | set2) < 4:
                return False

            sorted_s1 = sorted(set1, key=lambda x: x[1])
            sorted_s2 = sorted(set2, key=lambda x: x[1])
            newI, oldI = [x[0] for x in sorted_s1], [x[0] for x in sorted_s2]
            if kind == "cross" and old_kind == "cross":
                if _crosses_intersect(newI, oldI):
                    return False

            newI = [xb, xr]
            oldI = [float(b["xb"]), float(b["xr"])]
            if kind == "loop" and old_kind == "loop" and bf == old_f:
                if not _disjoint(newI, oldI):
                    return False

            if kind == "cross" and old_kind == "loop":
                loop_br = old_f
                if (bf == loop_br) and _cross_loop_intersect(xb, oldI):
                    return False
                if (bt == loop_br) and _cross_loop_intersect(xr, oldI):
                    return False

            if kind == "loop" and old_kind == "cross":
                loop_br = bf
                if (loop_br == old_f) and _cross_loop_intersect(oldI[0], newI):
                    return False
                if (loop_br == old_t) and _cross_loop_intersect(oldI[1], newI):
                    return False

        return True

    def _remove_corridor_chain(self, branch: str, nb, nr):
        corridor_sub = nx.DiGraph()
        for u, v, k, _d in self._corridor_edges(branch):
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

        for mid in path[1:-1]:
            if self.G.in_degree(mid) == 0 and self.G.out_degree(mid) == 0:
                self.G.remove_node(mid)

    def corridor_incoming_width_at(self, branch: str, node: tuple) -> float:
        x = float(self.G.nodes[node]["x"])
        if abs(x - self.p.xs) < 1e-9:
            return float(self.WA0 if branch == "A" else self.WB0)

        for u, v, k, d in self.G.in_edges(node, keys=True, data=True):
            if d.get("kind") == "corridor" and d.get("branch") == branch:
                return float(d["w"])
        raise RuntimeError(f"No incoming corridor edge for {branch} into {node}")

    def recompute_widths(self):
        p = self.p
        if self.WA0 is None or self.WB0 is None:
            raise RuntimeError("Call instantiate_corridor(WA,WB) first")

        ne = self._node(p.xe, "main")
        ns = self._node(p.xs, "main")

        corridor_nodes = {ns, ne}
        for br in ("A", "B"):
            for u, v, k, d in self._corridor_edges(br):
                corridor_nodes.add(u)
                corridor_nodes.add(v)

        nodes_sorted = sorted(corridor_nodes, key=lambda n: self.G.nodes[n]["x"])
        inflow = {n: 0.0 for n in nodes_sorted}
        inflow[ns] = float(self.WA0 + self.WB0)

        def outgoing_corridor(u, branch: str):
            outs = []
            for _, v, k, d in self.G.out_edges(u, keys=True, data=True):
                if d.get("kind") == "corridor" and d.get("branch") == branch:
                    outs.append((v, k))
            if len(outs) > 1:
                raise RuntimeError(f"Multiple outgoing corridor edges from {u} for branch {branch}")
            return outs[0] if outs else None

        for br, W0 in (("A", self.WA0), ("B", self.WB0)):
            out = outgoing_corridor(ns, br)
            if out is None:
                raise RuntimeError(f"Missing initial corridor edge for branch {br} from xs main")
            v, kcorr = out
            self.G.edges[ns, v, kcorr]["w"] = float(W0)
            if v in inflow:
                inflow[v] += float(W0)

        for u in nodes_sorted:
            if u == ns:
                continue

            tag = u[1]
            Win = inflow[u]

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

            if tag in ("A", "B"):
                out = outgoing_corridor(u, tag)
                if out is None:
                    if abs(remaining) > 1e-6:
                        raise ValueError(
                            f"Width conservation violated at {u}: Win={Win}, "
                            f"break_out={break_out}, remaining={remaining}"
                        )
                    continue

                v, kcorr = out
                Wcorr = remaining
                if Wcorr < p.min_width - 1e-9:
                    raise ValueError(f"Corridor width below min at {u}: {Wcorr}")

                self.G.edges[u, v, kcorr]["w"] = float(Wcorr)
                if v in inflow:
                    inflow[v] += Wcorr

        if abs(inflow[ne] - p.W_total) > 1e-6:
            raise ValueError(f"Total width arriving at xe main is {inflow[ne]}, expected {p.W_total}")

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

        self.G.add_edge(
            nb,
            nr,
            key=f"loop+@{branch}{xb}->{xr}",
            kind="loop",
            from_branch=branch,
            to_branch=branch,
            w=W1,
            curve=+1,
        )
        self.G.add_edge(
            nb,
            nr,
            key=f"loop-@{branch}{xb}->{xr}",
            kind="loop",
            from_branch=branch,
            to_branch=branch,
            w=W2,
            curve=-1,
        )

        self.breaks.append(
            dict(
                kind="loop",
                from_branch=branch,
                to_branch=branch,
                xb=xb,
                xr=xr,
                w1=W1,
                w2=W2,
                replace_corridor=bool(replace_corridor),
            )
        )
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

        self.G.add_edge(
            nb,
            nr,
            key=f"cross@{bf}{xb}->{bt}{xr}",
            kind="cross",
            from_branch=bf,
            to_branch=bt,
            w=W_cross,
            curve=0,
        )

        self.breaks.append(dict(kind="cross", from_branch=bf, to_branch=bt, xb=xb, xr=xr, w_cross=W_cross))
        self.recompute_widths()

    def _node_phase(self, n) -> int:
        x = float(self.G.nodes[n]["x"])
        tag = n[1]
        if abs(x - self.p.xs) < 1e-9:
            return 0 if tag == "main" else 1
        if abs(x - self.p.xe) < 1e-9:
            return 0 if tag in ("A", "B") else 1
        return 0

    def count_paths(self) -> int:
        s = (round(0.0, 6), "main")
        t = (round(self.p.L, 6), "main")
        if s not in self.G or t not in self.G:
            raise RuntimeError("Missing source/sink main nodes")

        nodes = sorted(self.G.nodes, key=lambda n: (self.G.nodes[n]["x"], self._node_phase(n), n[1]))
        paths = {n: 0 for n in nodes}
        paths[s] = 1

        for u in nodes:
            pu = paths[u]
            if pu == 0:
                continue
            for _, v, k, d in self.G.out_edges(u, keys=True, data=True):
                paths[v] += pu

        return int(paths[t])

    def _precompute_corridor_centerlines(self, N=600):
        p = self.p
        xg = np.linspace(p.xs, p.xe, N)
        t = (xg - p.xs) / (p.xe - p.xs)
        bump = np.sin(np.pi * t)
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
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=y,
                    mode="lines",
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
                    showlegend=False,
                )
            )

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
                yy = np.linspace(0, 0, 100)
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

    def to_recipe(self) -> dict:
        if self.WA0 is None or self.WB0 is None:
            raise RuntimeError("Corridor not instantiated")

        meta = dict(
            L=self.p.L,
            W_total=self.p.W_total,
            xs=self.p.xs,
            xe=self.p.xe,
            jump=self.p.jump,
            min_width=self.p.min_width,
            width_step=self.p.width_step,
            x_stability=self.p.x_stability,
            max_breaks=self.p.max_breaks,
            Y0=self.p.Y0,
            amp_corr=self.p.amp_corr,
            amp_loop=self.p.amp_loop,
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
        net.instantiate_corridor(recipe["initial_split"]["WA"], recipe["initial_split"]["WB"])

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


def canonical_signature(net: RiverNetworkNX):
    """Return an A/B-invariant signature for realized networks."""

    G = net.G

    def extract(swap: bool):
        def bmap(br):
            if not swap:
                return br
            return {"A": "B", "B": "A"}.get(br, br)

        corr_A = []
        corr_B = []
        loops = []
        crosses = []

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

        loop_groups = {}
        for u, v, k, d in G.edges(keys=True, data=True):
            if d.get("kind") == "loop":
                xu = round(float(G.nodes[u]["x"]), 6)
                xv = round(float(G.nodes[v]["x"]), 6)
                br = bmap(d.get("from_branch", u[1] if isinstance(u, tuple) else "A"))
                w = round(float(d.get("w", 0.0)), 10)
                key2 = (br, xu, xv)
                loop_groups.setdefault(key2, []).append(w)

        for (br, xb, xr), ws in loop_groups.items():
            loops.append((br, xb, xr, tuple(sorted(ws))))
        loops.sort()

        for u, v, k, d in G.edges(keys=True, data=True):
            if d.get("kind") == "cross":
                xb = round(float(G.nodes[u]["x"]), 6)
                xr = round(float(G.nodes[v]["x"]), 6)
                w = round(float(d.get("w", 0.0)), 10)
                fb = bmap(d.get("from_branch", "A"))
                tb = bmap(d.get("to_branch", "B"))
                crosses.append((fb, tb, xb, xr, w))

        crosses.sort()
        init_pair = tuple(sorted([round(float(net.WA0), 10), round(float(net.WB0), 10)]))
        return (init_pair, tuple(corr_A), tuple(corr_B), tuple(loops), tuple(crosses))

    return min(extract(False), extract(True))


__all__ = ["Params", "RiverNetworkNX", "canonical_signature"]
