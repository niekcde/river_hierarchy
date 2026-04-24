import copy
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# K utilities
# ============================================================

def compute_k(L, W, kb=20, S=1e-3, n=0.35):
    return (3/5) * n * (L / math.sqrt(S)) * (kb**(2/3)) / (W**(2/3))


def k_stats(edges, x=0.3):
    ks = np.array([compute_k(e.length, e.width) for e in edges])
    if len(ks) == 0:
        return None
    kmin, kmax = ks.min(), ks.max()
    return {
        "k_sum": ks.sum(),
        "k_min": kmin,
        "k_max": kmax,
        "k_ratio": kmax / kmin,
        "admissible": (kmax / kmin) <= (1-x)/x
    }


# ============================================================
# Core graph objects
# ============================================================

class Node:
    def __init__(self, x):
        self.x = float(x)
        self.in_edges = []
        self.out_edges = []

    def __hash__(self):
        return hash(round(self.x, 6))

    def __eq__(self, other):
        return isinstance(other, Node) and abs(self.x - other.x) < 1e-6


class Edge:
    def __init__(self, start, end, width, branch,
                 corridor=True, is_cross=False, name=None,
                 y0=0.0, y1=0.0):
        self.start = start
        self.end = end
        self.width = float(width)
        self.branch = branch          # "A", "B", "main", or None
        self.corridor = corridor      # ONLY true for A/B/main lanes
        self.is_cross = is_cross
        self.name = name

        self.x0 = start.x
        self.x1 = end.x
        self.y0 = y0
        self.y1 = y1

        start.out_edges.append(self)
        end.in_edges.append(self)

    @property
    def length(self):
        return self.end.x - self.start.x


# ============================================================
# Network
# ============================================================
class Network:
    def __init__(self, L, W_total, jump,
                 min_width=10, width_step=10,
                 Y0=1.0, dy=0.4):
        self.L = L
        self.W_total = W_total
        self.jump = jump
        self.min_width = min_width
        self.width_step = width_step
        self.Y0 = Y0
        self.dy = dy

        self.nodes = {}
        self.edges = []

        self.parallel_start = None
        self.parallel_end = None

        self.bifurcations = []

        self._A_loop_level = 0
        self._B_loop_level = 0

    # --------------------------------------------------------

    def _node(self, x):
        k = round(x, 6)
        if k not in self.nodes:
            self.nodes[k] = Node(x)
        return self.nodes[k]

    def _add_edge(self, *args, **kwargs):
        e = Edge(*args, **kwargs)
        self.edges.append(e)
        return e

    # --------------------------------------------------------

    def setup_initial(self):
        self.nodes.clear()
        self.edges.clear()
        self.bifurcations.clear()

        n0 = self._node(0.0)
        nL = self._node(self.L)
        self._add_edge(n0, nL, self.W_total,
                       branch="main", corridor=True,
                       name="main")

    # --------------------------------------------------------

    def apply_first_bifurcation(self, xs, xe, WA, WB):
        self.parallel_start = xs
        self.parallel_end = xe

        main = self.edges[0]
        self.edges.clear()
        main.start.out_edges.clear()
        main.end.in_edges.clear()

        n0 = self._node(0)
        n1 = self._node(xs)
        n2 = self._node(xe)
        nL = self._node(self.L)

        self._add_edge(n0, n1, self.W_total, "main", True, name="main_up")
        self._add_edge(n2, nL, self.W_total, "main", True, name="main_dn")

        self._add_edge(n1, n2, WA, "A", True, name="A",
                       y0=self.Y0, y1=self.Y0)
        self._add_edge(n1, n2, WB, "B", True, name="B",
                       y0=-self.Y0, y1=-self.Y0)

    # --------------------------------------------------------

    def _find_corridor_edge(self, branch, x):
        for e in self.edges:
            if e.branch == branch and e.corridor:
                if e.start.x < x < e.end.x:
                    return e
        return None

    # --------------------------------------------------------

    def _split_edge(self, e, x):
        if not (e.start.x < x < e.end.x):
            raise RuntimeError("Invalid split")

        self.edges.remove(e)
        e.start.out_edges.remove(e)
        e.end.in_edges.remove(e)

        n = self._node(x)

        l = self._add_edge(
            e.start, n, e.width,
            e.branch, e.corridor, e.is_cross,
            name=(e.name or "")+"_L",
            y0=e.y0, y1=e.y0)

        r = self._add_edge(
            n, e.end, e.width,
            e.branch, e.corridor, e.is_cross,
            name=(e.name or "")+"_R",
            y0=e.y1, y1=e.y1)

        return l, r, n

    # --------------------------------------------------------

    def _corridor_degree(self, node):
        ins = sum(1 for e in node.in_edges if e.corridor)
        outs = sum(1 for e in node.out_edges if e.corridor)
        return ins, outs

    # --------------------------------------------------------

    def add_bifurcation(self, bf, bt, xb, xr, WC):
        if xr <= xb:
            raise ValueError

        ef = self._find_corridor_edge(bf, xb)
        et = self._find_corridor_edge(bt, xr)

        if ef is None or et is None:
            raise ValueError("Must bifurcate on corridor edges only")

        # split safely
        if ef is et:
            _, _, nr = self._split_edge(et, xr)
            ef = self._find_corridor_edge(bf, xb)
            _, _, nb = self._split_edge(ef, xb)
        else:
            _, _, nb = self._split_edge(ef, xb)
            _, _, nr = self._split_edge(et, xr)

        # corridor simplicity rule (THE KEY FIX)
        if self._corridor_degree(nb) != (1,1):
            raise ValueError("Break in branching region")
        if self._corridor_degree(nr) != (1,1):
            raise ValueError("Return in branching region")

        # width update
        down = [e for e in nb.out_edges if e.corridor and e.branch == bf][0]
        if down.width - WC < self.min_width:
            raise ValueError
        down.width -= WC

        # loop or crossover edge (non-corridor!)
        if bf == bt:
            level = self._A_loop_level if bf == "A" else self._B_loop_level
            y = (self.Y0 + (level+1)*self.dy) * (1 if bf=="A" else -1)
            self._add_edge(nb, nr, WC, None, False,
                           name=f"{bf}_loop",
                           y0=y, y1=y)
            if bf == "A": self._A_loop_level += 1
            else: self._B_loop_level += 1
        else:
            y0 = self.Y0 if bf=="A" else -self.Y0
            y1 = self.Y0 if bt=="A" else -self.Y0
            self._add_edge(nb, nr, WC, None, False,
                           is_cross=True,
                           name=f"{bf}->{bt}",
                           y0=y0, y1=y1)

        self.bifurcations.append((bf,bt,xb,xr,WC))

    # --------------------------------------------------------

    def count_paths(self):
        start = self._node(0)
        end = self._node(self.L)
        memo = {}

        def dfs(n):
            if n == end:
                return 1
            if n in memo:
                return memo[n]
            s = sum(dfs(e.end) for e in n.out_edges)
            memo[n] = s
            return s

        return dfs(start)

    # --------------------------------------------------------

    def plot(self):
        plt.figure(figsize=(10,4))
        for e in self.edges:
            c = "black"
            if e.branch == "A": c="tab:blue"
            if e.branch == "B": c="tab:orange"
            if e.is_cross: c="tab:red"
            if e.branch is None and not e.is_cross: c="tab:green"
            plt.plot([e.x0,e.x1],[e.y0,e.y1],c,linewidth=max(1,e.width/20))
        # plt.axhline(0,color="gray",lw=0.5)
        plt.show()


# ============================================================
# Enumeration
# ============================================================

def enumerate_admissible_networks(
    L, W_total, jump,
    first_split_start, first_split_end,
    max_bifurcations,
    x_stability=0.3,
    min_width=10,
    width_step=10
):
    nets = []
    rows = []
    tested = 0

    base = Network(L,W_total,jump,min_width,width_step)
    base.setup_initial()

    for WA in np.arange(min_width, W_total, width_step):
        WB = W_total - WA
        if WB < min_width: continue

        net0 = copy.deepcopy(base)
        net0.apply_first_bifurcation(first_split_start, first_split_end, WA, WB)

        def dfs(net):
            nonlocal tested
            tested += 1

            ks = k_stats(net.edges, x_stability)
            if ks is None or not ks["admissible"]:
                return

            
            network_id = len(nets)
            rows.append({
                "n_bifurcations": len(net.bifurcations),
                "n_paths": net.count_paths(),
                "network_id": network_id,
                **ks
            })
            nets.append(net)
            if len(net.bifurcations) == max_bifurcations:
                return

            xs = np.arange(net.parallel_start+jump,
                           net.parallel_end,
                           jump)

            for bf in ("A","B"):
                for bt in ("A","B"):
                    for xb in xs:
                        for xr in xs:
                            if xr <= xb: continue
                            WC = min_width
                            while WC <= WA - min_width:
                                new = copy.deepcopy(net)
                                try:
                                    new.add_bifurcation(bf,bt,xb,xr,WC)
                                except Exception:
                                    WC += width_step
                                    continue
                                dfs(new)
                                WC += width_step

        dfs(net0)

    df = pd.DataFrame(rows)
    df["n_tested_total"] = tested
    return nets, df, tested


if __name__ == "__main__":
    # from river_network import Network, enumerate_admissible_networks

    L           = 10000
    W_total     = 100
    jump        = 500
    first_start = 3000
    first_end   = 7000
    max_bif     = 2
    x           = 0.1
    width_step  = 10
    min_width   = 10

    nets, summary, n_tested = enumerate_admissible_networks(
        L=L,
        W_total=W_total,
        jump=jump,
        first_split_start=first_start,
        first_split_end=first_end,
        max_bifurcations=max_bif,
        x_stability=x,
        width_step=width_step,
        min_width=min_width
    )
    print(f"networks tested: {n_tested}")
    print("Networks found:", len(nets))
    print(summary.head(20))
    print(summary[-10:])

    # sanity check
    if not summary.empty and "n_paths" in summary.columns:
        print("Max paths:", summary["n_paths"].max())
        print("Max bifurcations:", summary["n_bifurcations"].max())
    else:
        print("No admissible networks found.")


    directory = '/Users/6256481/Desktop/PhD_icloud/USA_UNC/work/river_hierarchy/synthetic/output/'
    # print(type(nets))
    # print(type(summary))
    # pd.DataFrame(nets))
    print(len(nets), len(summary))

    summary.to_csv(directory + f'synthetic_{L}_{W_total}_{jump}_{first_start}_{first_end}_{x}_{width_step}_{min_width}.csv')
    # nets[-1].plot()




    payload = {
        "meta": dict(L=L, W_total=W_total, jump=jump,
                    first_start=first_start, first_end=first_end,
                    max_bif=max_bif, x=x, width_step=width_step, min_width=min_width),
        "summary": summary,
        "networks": nets,
    }

    import pickle
    with open(directory + "synthetic_run.pkl", "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)