import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import copy


# ============================================================
# K utilities (your RAPID-style k computation and admissibility)
# ============================================================

def compute_k(L: float, W: float, kb: int = 20, S: float = 1e-3, n: float = 0.35) -> float:
    """
    Compute k value for RAPID discharge model
    - L: reach length (m)
    - W: reach width (m)
    - kb: width to depth scaling
    - S: slope (m/m)
    - n: Manning's n
    """
    kb = 20
    S = 1e-3
    n = 0.35
    return (3 / 5) * n * ((L / (S ** 0.5)) * ((kb ** (2 / 3)) / (W ** (2 / 3))))


def compute_k_array(edges):
    """
    edges: iterable of Edge objects
    returns: (k_values np.array, k_sum float)
    """
    k_vals = np.array([compute_k(e.length, e.width) for e in edges])
    return k_vals, float(k_vals.sum())


def is_admissible(edges, x: float = 0.3) -> bool:
    """
    True if k-ratio <= allowed ratio for stability.
    """
    if not edges:
        return False
    k_vals, _ = compute_k_array(edges)
    kmin = k_vals.min()
    kmax = k_vals.max()
    k_ratio = kmax / kmin
    allowed_ratio = (1 - x) / x
    return k_ratio <= allowed_ratio


# ============================================================
# Core graph classes: Node, Edge, Network
# ============================================================

class Node:
    def __init__(self, x: float, y: float, name: str = None):
        self.x = x
        self.y = y
        self.name = name
        self.in_edges = []
        self.out_edges = []

    def __repr__(self):
        return f"Node({self.name}, x={self.x}, y={self.y})"


class Edge:
    def __init__(self, start: Node, end: Node, width: float,
                 name: str = None, branch: str = None, is_cross: bool = False):
        """
        branch: 'main', 'A', 'B', or None/cross
        """
        self.start = start
        self.end = end
        self.width = width
        self.name = name
        self.branch = branch
        self.is_cross = is_cross

        start.out_edges.append(self)
        end.in_edges.append(self)

    @property
    def length(self) -> float:
        return abs(self.end.x - self.start.x)

    def __repr__(self):
        return f"Edge({self.name}, {self.start.name}->{self.end.name}, W={self.width:.2f}, L={self.length:.2f})"


class Network:
    """
    Network with:
    - domain length L
    - initial width W_total
    - grid spacing jump (min distance between allowed breakpoints)
    - visualization params Y0, dy

    Workflow:
    - create Network(L, W_total, jump)
    - call setup_initial_channel()
    - call apply_first_bifurcation()
    - call add_bifurcation(...) for more splits
    - call plot() to visualize
    - call to_dataframe() for analysis
    """

    def __init__(self, L: float, W_total: float,
                 jump: float, Y0: float = 1.0, dy: float = 0.5,
                 min_width_value: float = 10.0, min_width_step: float = 10.0):
        self.L = L
        self.W_total = W_total
        self.jump = jump
        self.Y0 = Y0
        self.dy = dy
        self.min_width_value = min_width_value
        self.min_width_step = min_width_step

        self.nodes = []
        self.edges = []

        # Track A/B main branches as ordered lists of edges (by x)
        self.branch_A_edges = []
        self.branch_B_edges = []

        # History of bifurcations for monotonicity and visualization
        # each item: (branch_from, branch_to, x_break, x_return, WC)
        self.bifurcation_history = []

        # Forbidden intervals for A and B (for same-branch splits A->A, B->B)
        # list of (x_break, x_return)
        self.forbidden_intervals_A = []
        self.forbidden_intervals_B = []

        # Counters for vertical offset stacking
        self.A_loop_count = 0
        self.B_loop_count = 0

        # track last break and return for monotonic downstream rule
        self.last_break_x = 0.0
        self.last_return_x = 0.0

        # Convenience: upstream and downstream nodes
        self.upstream_node = None
        self.downstream_node = None

    # -------------------
    # Graph helpers
    # -------------------
    def add_node(self, x: float, y: float, name: str = None) -> Node:
        node = Node(x, y, name=name)
        self.nodes.append(node)
        return node

    def add_edge(self, start: Node, end: Node, width: float,
                 name: str = None, branch: str = None, is_cross: bool = False) -> Edge:
        edge = Edge(start, end, width, name=name, branch=branch, is_cross=is_cross)
        self.edges.append(edge)
        return edge

    # -------------------
    # Initial channel and first bifurcation
    # -------------------
    def setup_initial_channel(self):
        """
        Create a single main-channel edge from x=0 to x=L at y=0.
        """
        self.nodes.clear()
        self.edges.clear()
        self.branch_A_edges.clear()
        self.branch_B_edges.clear()
        self.bifurcation_history.clear()
        self.forbidden_intervals_A.clear()
        self.forbidden_intervals_B.clear()
        self.A_loop_count = 0
        self.B_loop_count = 0
        self.last_break_x = 0.0
        self.last_return_x = 0.0

        n_up = self.add_node(0.0, 0.0, name="N0")
        n_down = self.add_node(self.L, 0.0, name=f"N{int(self.L)}")
        self.upstream_node = n_up
        self.downstream_node = n_down

        self.add_edge(n_up, n_down, self.W_total, name="E_main", branch="main")

    def _width_splits_primary(self, mode="computed"):
        """
        Primary W_total -> WA + WB splits.
        Returns list of (WA, WB).
        """
        W = self.W_total
        mval = self.min_width_value
        mstep = self.min_width_step

        vals = []
        for WA in np.arange(mval, W - mval + 1e-9, mstep):
            WB = W - WA
            if WB < mval:
                continue
            vals.append((WA, WB))

        if mode == "computed":
            return vals
        elif mode == "equal":
            if W % 2 == 0:
                WA = WB = W / 2
                if WA >= mval and WB >= mval:
                    return [(WA, WB)]
            return []
        else:
            raise ValueError(f"Unknown mode '{mode}' for primary splits")

    def apply_first_bifurcation(self, x_split_start: float, x_split_end: float,
                                WA: float = None, WB: float = None,
                                width_mode="computed"):
        """
        First bifurcation: split initial main edge between x_split_start and x_split_end
        into A and B branches at y = +Y0 and -Y0.

        If WA and WB are None, use the first admissible combination from width_mode="computed".
        """
        if not self.edges:
            raise RuntimeError("Call setup_initial_channel() first.")

        main_edge = self.edges[0]
        if main_edge.branch != "main":
            raise RuntimeError("Expected the only edge to be main channel.")

        # Check grid alignment
        for x in [x_split_start, x_split_end]:
            if abs((x / self.jump) - round(x / self.jump)) > 1e-6:
                raise ValueError("Split positions must align with the jump grid.")

        if not (0 < x_split_start < x_split_end < self.L):
            raise ValueError("Split locations must be within (0, L).")

        # Pick widths if not provided
        if WA is None or WB is None:
            splits = self._width_splits_primary(mode=width_mode)
            if not splits:
                raise RuntimeError("No valid primary width splits found.")
            WA, WB = splits[0]

        # Remove main edge
        self.edges.remove(main_edge)
        main_edge.start.out_edges.remove(main_edge)
        main_edge.end.in_edges.remove(main_edge)

        # Create nodes on main channel at split points
        n0 = self.upstream_node
        n3 = self.downstream_node

        n1 = self.add_node(x_split_start, 0.0, name=f"N{int(x_split_start)}")
        n2 = self.add_node(x_split_end, 0.0, name=f"N{int(x_split_end)}")

        # Main single-channel segments before and after parallel region
        E_up = self.add_edge(n0, n1, self.W_total, name="E_up", branch="main")
        E_down = self.add_edge(n2, n3, self.W_total, name="E_down", branch="main")

        # Parallel A/B nodes at same x but different y
        A1 = self.add_node(x_split_start, +self.Y0, name=f"A{int(x_split_start)}")
        A2 = self.add_node(x_split_end, +self.Y0, name=f"A{int(x_split_end)}")

        B1 = self.add_node(x_split_start, -self.Y0, name=f"B{int(x_split_start)}")
        B2 = self.add_node(x_split_end, -self.Y0, name=f"B{int(x_split_end)}")

        # Connect split point on main to start of A/B, and end of A/B to main
        # (conceptually junctions). Here we create A/B edges only;
        # we treat vertical transitions as conceptual, not explicit edges.
        E_A = self.add_edge(A1, A2, WA, name="E_A_main", branch="A")
        E_B = self.add_edge(B1, B2, WB, name="E_B_main", branch="B")

        self.branch_A_edges = [E_A]
        self.branch_B_edges = [E_B]

        # Track monotonic references
        # self.last_break_x = x_split_start
        # self.last_return_x = x_split_end

        # Record history
        self.bifurcation_history.append(("main", "AB", x_split_start, x_split_end, None))

    # -------------------
    # Bifurcation helpers
    # -------------------
    def _find_edge_on_branch(self, branch: str, x: float) -> Edge:
        """
        Find the edge on branch A or B that contains position x (start.x < x < end.x).
        """
        if branch == "A":
            edges = self.branch_A_edges
        elif branch == "B":
            edges = self.branch_B_edges
        else:
            raise ValueError("branch must be 'A' or 'B'.")

        for e in edges:
            if e.start.x < x < e.end.x:
                return e
        return None

    def _split_edge_at(self, edge: Edge, x: float):
        """
        Split a given edge at position x (between start.x and end.x).
        Returns (edge_left, edge_right, new_node).
        """
        if not (edge.start.x < x < edge.end.x):
            raise ValueError("Split position x must lie strictly inside the edge.")

        # Remove old edge
        self.edges.remove(edge)
        edge.start.out_edges.remove(edge)
        edge.end.in_edges.remove(edge)

        # Create new node at x
        new_y = edge.start.y  # same vertical track
        new_node = self.add_node(x, new_y, name=f"N{int(x)}_{edge.branch or 'X'}")

        # Create new edges
        e_left = self.add_edge(edge.start, new_node, edge.width,
                               name=(edge.name or "") + "_L", branch=edge.branch,
                               is_cross=edge.is_cross)
        e_right = self.add_edge(new_node, edge.end, edge.width,
                                name=(edge.name or "") + "_R", branch=edge.branch,
                                is_cross=edge.is_cross)
        return e_left, e_right, new_node

    def _update_branch_edge_list(self, branch: str):
        """
        Keep branch_A_edges / branch_B_edges sorted by start.x.
        """
        if branch == "A":
            self.branch_A_edges = sorted(
                [e for e in self.edges if e.branch == "A"],
                key=lambda e: e.start.x
            )
        elif branch == "B":
            self.branch_B_edges = sorted(
                [e for e in self.edges if e.branch == "B"],
                key=lambda e: e.start.x
            )

    def _check_forbidden_interval(self, branch: str, x_break: float) -> bool:
        """
        True if x_break lies IN any forbidden interval for that branch.
        """
        if branch == "A":
            intervals = self.forbidden_intervals_A
        else:
            intervals = self.forbidden_intervals_B

        for xb, xr in intervals:
            if xb < x_break < xr:
                return True
        return False

    def _secondary_width_splits(self, W_parent: float):
        """
        Possible secondary splits W_parent -> W_parent_down + WC
        where WC is the diverted width.

        We require:
        - W_parent >= 2 * min_width_value
        - WC in [min_width_value, W_parent - min_width_value] with step
        """
        mval = self.min_width_value
        mstep = self.min_width_step
        if W_parent < 2 * mval:
            return []

        vals = []
        for WC in np.arange(mval, W_parent - mval + 1e-9, mstep):
            W_down = W_parent - WC
            if W_down < mval:
                continue
            vals.append((WC, W_down))
        return vals

    # -------------------
    # Add bifurcation (A→A, B→B, A→B, B→A)
    # -------------------
    def add_bifurcation(self,
                        branch_from: str,
                        branch_to: str,
                        x_break: float,
                        x_return: float,
                        WC: float):
        """
        Add a bifurcation:
        - branch_from, branch_to in {'A','B'}
        - x_break, x_return on the common x-axis (0..L)
        - WC: diverted width

        Rules enforced:
        - x_break, x_return on grid (multiples of jump)
        - x_return > x_break
        - monotonic downstream relative to previous bifurcations
        - forbidden-interval rule for same-branch loops
        - width constraints
        - no splitting on cross edges
        """
        if branch_from not in ("A", "B") or branch_to not in ("A", "B"):
            raise ValueError("branch_from and branch_to must be 'A' or 'B'.")

        # grid alignment
        for x in [x_break, x_return]:
            if abs((x / self.jump) - round(x / self.jump)) > 1e-6:
                raise ValueError("Break/return positions must align with the jump grid.")

        if not (0 < x_break < x_return < self.L):
            raise ValueError("Break/return must satisfy 0 < x_break < x_return < L.")

        # monotonic downstream constraint
        if x_break <= self.last_break_x or x_return <= self.last_return_x:
            raise ValueError("New bifurcation must be downstream of previous ones.")

        # forbidden interval rule for same-branch loops
        if branch_from == branch_to:
            if self._check_forbidden_interval(branch_from, x_break):
                raise ValueError(f"x_break={x_break} lies in a forbidden interval on branch {branch_from}.")

        # find edges containing break and return
        edge_from = self._find_edge_on_branch(branch_from, x_break)
        if edge_from is None:
            raise RuntimeError(f"No edge on branch {branch_from} covering x_break={x_break}.")

        edge_to = self._find_edge_on_branch(branch_to, x_return)
        if edge_to is None:
            raise RuntimeError(f"No edge on branch {branch_to} covering x_return={x_return}.")

        # width constraints on edge_from for WC
        splits = self._secondary_width_splits(edge_from.width)
        if not splits:
            raise RuntimeError(f"Edge on {branch_from} has width {edge_from.width} too small to split.")
        possible_WCs = [wc for wc, wdown in splits]
        if WC not in possible_WCs:
            raise ValueError(
                f"WC={WC} not allowed; must be in {possible_WCs} given parent width {edge_from.width}."
            )

        # --- Split edge_from at x_break and (if same branch) at x_return ---
        # For cross-branch case, we split edge_from only at x_break,
        # and edge_to only at x_return.
        # For same-branch loops, we split the same branch at both x_break and x_return.

        # 1) Split edge_from at x_break
        left_from, right_from, node_break = self._split_edge_at(edge_from, x_break)

        # 2) Decrease width downstream on branch_from by WC (on the right_from edge)
        right_from.width = edge_from.width - WC

        # Update branch list
        self._update_branch_edge_list(branch_from)

        # 3) Split edge_to at x_return
        # If same branch, we must locate the correct edge again (right_from may cover x_return or not)
        edge_to = self._find_edge_on_branch(branch_to, x_return)
        if edge_to is None:
            raise RuntimeError(f"No edge on branch {branch_to} covering x_return={x_return} after first split.")

        left_to, right_to, node_return = self._split_edge_at(edge_to, x_return)

        # 4) Increase width downstream on branch_to by WC (on right_to)
        right_to.width = edge_to.width + WC if branch_from != branch_to else edge_to.width

        # For same-branch loops (A→A or B→B):
        # we want: upstream width = original; middle segment width reduced; downstream width back to original.
        # We already reduced right_from.width = W_parent - WC;
        # now we need to ensure that after return, width is restored to original on right_to.
        if branch_from == branch_to:
            # For same-branch: restore W_parent on segment after return
            # The parent width is stored in edge_from.width (before split) and edge_to is the segment containing return.
            # We assume left_to is the middle segment and right_to is the downstream segment.
            # left_to already has original width (edge_to.width before change),
            # right_to should be the restored width = edge_from.width.
            right_to.width = edge_from.width

        # Update branch lists
        self._update_branch_edge_list(branch_to)

        # 5) Create crossover / loop edge
        if branch_from == "A":
            y_start = node_break.y
        else:
            y_start = node_break.y

        if branch_to == "A":
            y_end = node_return.y
        else:
            y_end = node_return.y

        # For loops A→A or B→B, we give a unique y-offset to the loop.
        # We do this by creating a new pair of "loop nodes" at slightly offset y.
        if branch_from == branch_to:
            if branch_from == "A":
                self.A_loop_count += 1
                loop_y = self.Y0 + self.A_loop_count * self.dy
            else:
                self.B_loop_count += 1
                loop_y = -self.Y0 - self.B_loop_count * self.dy

            # loop start and end nodes at offset y
            loop_start = self.add_node(node_break.x, loop_y,
                                       name=f"{branch_from}_loop_start_{int(node_break.x)}")
            loop_end = self.add_node(node_return.x, loop_y,
                                     name=f"{branch_to}_loop_end_{int(node_return.x)}")

            # connect main branch node_break to loop_start vertically (conceptual, not as edges),
            # we just create the loop edge between loop_start and loop_end
            E_loop = self.add_edge(loop_start, loop_end, WC,
                                   name=f"{branch_from}_loop_{len(self.bifurcation_history)}",
                                   branch=None, is_cross=False)

            # Note: we don't alter A/B main nodes at j_break/j_return y positions;
            # the loop is visually offset.

        else:
            # Cross-branch: straight line from node_break to node_return
            E_cross = self.add_edge(node_break, node_return, WC,
                                    name=f"{branch_from}_to_{branch_to}_{len(self.bifurcation_history)}",
                                    branch=None, is_cross=True)

        # If same-branch loop, add forbidden interval
        if branch_from == branch_to:
            if branch_from == "A":
                self.forbidden_intervals_A.append((x_break, x_return))
            else:
                self.forbidden_intervals_B.append((x_break, x_return))

        # Update last break/return
        self.last_break_x = x_break
        self.last_return_x = x_return

        # Record history
        self.bifurcation_history.append((branch_from, branch_to, x_break, x_return, WC))

    # -------------------
    # Analysis and export
    # -------------------
    def admissible(self, x: float = 0.3) -> bool:
        return is_admissible(self.edges, x=x)

    def compute_k_stats(self, x: float = 0.3):
        k_vals, k_sum = compute_k_array(self.edges)
        return {
            "k_values": k_vals,
            "k_sum": k_sum,
            "k_min": float(k_vals.min()),
            "k_max": float(k_vals.max()),
            "k_ratio": float(k_vals.max() / k_vals.min()),
            "admissible": bool(is_admissible(self.edges, x=x)),
        }

    def to_dataframe(self) -> pd.DataFrame:
        """
        Flatten edges into a DataFrame with:
        - edge_name, branch, is_cross
        - x_start, x_end, y_start, y_end
        - length, width, k_value
        """
        records = []
        for e in self.edges:
            k_val = compute_k(e.length, e.width)
            records.append({
                "edge_name": e.name,
                "branch": e.branch,
                "is_cross": e.is_cross,
                "x_start": e.start.x,
                "x_end": e.end.x,
                "y_start": e.start.y,
                "y_end": e.end.y,
                "length": e.length,
                "width": e.width,
                "k_value": k_val,
            })
        return pd.DataFrame.from_records(records)

    # -------------------
    # Path counting (source -> sink)
    # -------------------
    def count_paths(self) -> int:
        """
        Count the number of simple directed paths from upstream_node to downstream_node.
        Since the network is a DAG (x always increases), we can do DFS with memoization.
        """
        if self.upstream_node is None or self.downstream_node is None:
            raise RuntimeError("Network must have upstream_node and downstream_node set.")

        memo = {}

        def dfs(node):
            if node is self.downstream_node:
                return 1
            if node in memo:
                return memo[node]

            total = 0
            for e in node.out_edges:
                total += dfs(e.end)

            memo[node] = total
            return total

        return dfs(self.upstream_node)

    # -------------------
    # K-distribution plotting (for a single network)
    # -------------------
    def plot_k_histogram(self, bins: int = 20, figsize=(6, 4), show=True):
        """
        Plot a histogram of k-values across all edges in this network.
        """
        if not self.edges:
            print("No edges to plot.")
            return

        k_vals, _ = compute_k_array(self.edges)

        plt.figure(figsize=figsize)
        plt.hist(k_vals, bins=bins, edgecolor="black", alpha=0.75)
        plt.xlabel("k-value")
        plt.ylabel("Frequency")
        plt.title("Distribution of k-values in this network")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        if show:
            plt.show()

    # -------------------
    # Path counting (source -> sink)
    # -------------------
    def count_paths(self) -> int:
        """
        Count the number of simple directed paths from upstream_node to downstream_node.
        Since the network is a DAG (x always increases), we can do DFS with memoization.
        """
        if self.upstream_node is None or self.downstream_node is None:
            raise RuntimeError("Network must have upstream_node and downstream_node set.")

        memo = {}

        def dfs(node):
            if node is self.downstream_node:
                return 1
            if node in memo:
                return memo[node]

            total = 0
            for e in node.out_edges:
                total += dfs(e.end)

            memo[node] = total
            return total

        return dfs(self.upstream_node)

    # -------------------
    # K-distribution plotting (for a single network)
    # -------------------
    def plot_k_histogram(self, bins: int = 20, figsize=(6, 4), show=True):
        """
        Plot a histogram of k-values across all edges in this network.
        """
        if not self.edges:
            print("No edges to plot.")
            return

        k_vals, _ = compute_k_array(self.edges)

        plt.figure(figsize=figsize)
        plt.hist(k_vals, bins=bins, edgecolor="black", alpha=0.75)
        plt.xlabel("k-value")
        plt.ylabel("Frequency")
        plt.title("Distribution of k-values in this network")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        if show:
            plt.show()




    # -------------------
    # Visualization
    # -------------------
    def plot(self, scale_width: float = 20.0, figsize=(10, 4), show=True):
        """
        Plot the network using matplotlib.

        scale_width: divide width by this to get reasonable line widths.
        """
        plt.figure(figsize=figsize)

        for e in self.edges:
            xs = [e.start.x, e.end.x]
            ys = [e.start.y, e.end.y]

            if e.branch == "A":
                color = "tab:blue"
                zorder = 2
            elif e.branch == "B":
                color = "tab:orange"
                zorder = 2
            elif e.is_cross:
                color = "tab:red"
                zorder = 3
            else:
                color = "gray"
                zorder = 1

            lw = max(0.5, e.width / scale_width)
            plt.plot(xs, ys, color=color, linewidth=lw, alpha=0.8, zorder=zorder)

        plt.axhline(0, color="black", linewidth=0.5, alpha=0.3)
        plt.xlabel("x (m)")
        plt.ylabel("y (lane)")
        plt.title("River Network")
        plt.grid(alpha=0.2)
        plt.tight_layout()
        if show:
            plt.show()


def enumerate_admissible_networks(
    L: float,
    W_total: float,
    jump: float,
    first_split_start: float,
    first_split_end: float,
    max_bifurcations: int,
    x_stability: float = 0.3,
    primary_width_mode: str = "computed",
    require_first_admissible: bool = True,
):
    """
    Enumerate all admissible networks up to `max_bifurcations` total bifurcations
    (including the initial A/B split).

    Strategy:
    - Fix the first bifurcation x-positions (first_split_start, first_split_end)
    - Enumerate possible (WA, WB) for the first split
    - For each such base network, recursively add up to (max_bifurcations - 1)
      further bifurcations using DFS + pruning.
    - Keep only networks that are k-admissible at every stage.

    Returns:
        networks: list of Network objects (each distinct)
        summary_df: pandas DataFrame with per-network stats
    """
    # 1) Build a dummy network to get primary width splits
    tmp_net = Network(L=L, W_total=W_total, jump=jump)
    tmp_net.setup_initial_channel()
    primary_splits = tmp_net._width_splits_primary(mode=primary_width_mode)

    all_networks = []
    n_tested = 0  # count how many candidate networks we actually evaluated

    # Helper: gather summary stats for final DF
    summary_records = []

    def collect_stats(net: Network):
        stats = net.compute_k_stats(x=x_stability)
        n_paths = net.count_paths()
        n_bif = len(net.bifurcation_history) - 1  # minus initial ('main','AB',...) record
        rec = {
            "n_bifurcations": n_bif,
            "k_sum": stats["k_sum"],
            "k_min": stats["k_min"],
            "k_max": stats["k_max"],
            "k_ratio": stats["k_ratio"],
            "admissible": stats["admissible"],
            "n_paths": n_paths,
        }
        return rec

    # Depth-first search over bifurcation sequences
    def dfs(current_net: Network, current_depth: int):
        nonlocal n_tested, all_networks, summary_records

        # At every node we consider the current network as a valid candidate
        # (as long as it's admissible).
        n_tested += 1
        if current_net.admissible(x=x_stability):
            stats_rec = collect_stats(current_net)
            all_networks.append(current_net)
            summary_records.append(stats_rec)
        else:
            # prune: don't go deeper
            return

        # If we've reached the max number of bifurcations: stop exploring deeper
        # Here "total bifurcations" includes the first A/B split, which is already in history.
        total_bif = len(current_net.bifurcation_history) - 1
        if total_bif >= max_bifurcations:
            return

        # Try adding one more bifurcation in all possible ways
        # branch_from / branch_to in {'A', 'B'}
        branches = ["A", "B"]
        grid_points = np.arange(jump, L, jump)

        for branch_from in branches:
            for branch_to in branches:
                # Loop over possible x_break, x_return on the grid
                for x_break in grid_points:
                    # skip if not downstream of 0; add_bifurcation will enforce other constraints
                    if x_break <= 0 or x_break >= L:
                        continue
                    for x_return in grid_points:
                        if x_return <= x_break or x_return >= L:
                            continue

                        # To propose WC, we need the width of the edge on branch_from at x_break
                        edge_from = current_net._find_edge_on_branch(branch_from, x_break)
                        if edge_from is None:
                            continue
                        sec_splits = current_net._secondary_width_splits(edge_from.width)
                        if not sec_splits:
                            continue

                        for WC, _Wdown in sec_splits:
                            # Try to apply bifurcation on a deep copy
                            new_net = copy.deepcopy(current_net)
                            try:
                                new_net.add_bifurcation(
                                    branch_from=branch_from,
                                    branch_to=branch_to,
                                    x_break=float(x_break),
                                    x_return=float(x_return),
                                    WC=float(WC),
                                )
                            except (ValueError, RuntimeError):
                                # invalid geometry/width/forbidden interval/monotonicity, skip
                                continue

                            # Early pruning on K
                            if not new_net.admissible(x=x_stability):
                                continue

                            # Recurse deeper
                            dfs(new_net, current_depth + 1)

    # 2) Enumerate primary (WA, WB) splits + first bifurcation
    for (WA, WB) in primary_splits:
        base_net = Network(L=L, W_total=W_total, jump=jump,
                           Y0=tmp_net.Y0, dy=tmp_net.dy,
                           min_width_value=tmp_net.min_width_value,
                           min_width_step=tmp_net.min_width_step)
        base_net.setup_initial_channel()
        base_net.apply_first_bifurcation(first_split_start, first_split_end, WA=WA, WB=WB)

        if require_first_admissible and not base_net.admissible(x=x_stability):
            continue

        # Start DFS from this base network
        dfs(base_net, current_depth=1)

    # Build summary DataFrame
    if summary_records:
        summary_df = pd.DataFrame(summary_records)
        summary_df["n_tested_total"] = n_tested
    else:
        summary_df = pd.DataFrame(columns=[
            "n_bifurcations", "k_sum", "k_min", "k_max", "k_ratio",
            "admissible", "n_paths", "n_tested_total"
        ])

    return all_networks, summary_df




# from network_module import Network  # or just paste the code above in the same file

# Parameters
L       = 10000
W_total = 200
jump    = 500

# net = Network(L=L, W_total=W_total, jump=jump,
#               Y0=1.0, dy=0.5,
#               min_width_value=10.0, min_width_step=10.0)

# # 1) Initial straight channel
# net.setup_initial_channel()



# # 2) First bifurcation between 2000 and 8000, with WA=40, WB=60
# net.apply_first_bifurcation(x_split_start=1000, x_split_end=9000, WA=100, WB=100)

# # 4) Add an A->A loop downstream
# net.add_bifurcation(branch_from="B", branch_to="B",
#                     x_break=2000, x_return=4000, WC=50)

# # 3) Add a crossover A->B
# net.add_bifurcation(branch_from="A", branch_to="B",
#                     x_break=3000, x_return=6000, WC=50)

# # 4) Add an A->A loop downstream
# net.add_bifurcation(branch_from="B", branch_to="B",
#                     x_break=7000, x_return=8000, WC=50)


# print("Number of paths:", net.count_paths())
# stats = net.compute_k_stats(x=0.3)
# print(stats)

# net.plot_k_histogram()



# # 6) Turn to DataFrame
# df = net.to_dataframe()
# print(df)

# # 7) Plot
# net.plot()


networks, summary = enumerate_admissible_networks(
    L=L,
    W_total=W_total,
    jump=jump,
    first_split_start=2000,
    first_split_end=8000,
    max_bifurcations=1,    # includes the first A/B split
    x_stability=0.1,
    primary_width_mode="computed",
    require_first_admissible=True,
)

print("Number of admissible networks found:", len(networks))
print(summary.head())
print("Total tested configurations:", summary["n_tested_total"].iloc[0] if not summary.empty else 0)

# Inspect one network
if networks:
    net0 = networks[0]
    print("Paths in first network:", net0.count_paths())
    net0.plot()
    net0.plot_k_histogram()