from __future__ import annotations

import argparse
from collections import defaultdict, deque
from pathlib import Path
import tkinter as tk
from tkinter import messagebox

import geopandas as gpd
import networkx as nx
import pandas as pd
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
from shapely.geometry import LineString, MultiLineString, Point


ROLE_UNKNOWN = "unknown"
ROLE_INTERNAL = "internal"
ROLE_INLET = "inlet"
ROLE_OUTLET = "outlet"

ROLE_COLORS = {
    ROLE_UNKNOWN: "white",
    ROLE_INTERNAL: "0.15",
    ROLE_INLET: "#2ca25f",
    ROLE_OUTLET: "#3182bd",
}

ZOOM_IN_MULTIPLIER = 0.8
ZOOM_OUT_MULTIPLIER = 1.25
MIN_PAD_FACTOR = 0.1
MAX_PAD_FACTOR = 100.0
DEFAULT_LINK_CSV_NAME = "manual_link_directions.csv"
DEFAULT_NODE_CSV_NAME = "manual_node_roles.csv"
DEFAULT_LINKS_GPKG_NAME = "directed_links.gpkg"
DEFAULT_NODES_GPKG_NAME = "reviewed_nodes.gpkg"


def parse_node_pair(value: object) -> tuple[int, int]:
    if value is None or pd.isna(value):
        raise ValueError("Encountered missing id_nodes value.")
    parts = [part.strip() for part in str(value).replace("[", "").replace("]", "").split(",") if part.strip()]
    if len(parts) != 2:
        raise ValueError(f"Expected exactly two nodes in id_nodes, got {value!r}.")
    return int(parts[0]), int(parts[1])


def as_bool(value: object) -> bool:
    if value is None or pd.isna(value):
        return False
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "t", "yes", "y"}
    return bool(value)


def source_role(is_inlet: object, is_outlet: object) -> str:
    inlet = as_bool(is_inlet)
    outlet = as_bool(is_outlet)
    if inlet and outlet:
        return "both"
    if inlet:
        return ROLE_INLET
    if outlet:
        return ROLE_OUTLET
    return ROLE_INTERNAL


def manual_role_to_flags(role: str) -> tuple[bool, bool]:
    return role == ROLE_INLET, role == ROLE_OUTLET


def node_role_label(role: str) -> str:
    return role if role else ROLE_UNKNOWN


def normalize_manual_role(value: object) -> str:
    if value is None or pd.isna(value):
        return ROLE_UNKNOWN
    role = str(value).strip().lower()
    if role in {ROLE_UNKNOWN, ROLE_INTERNAL, ROLE_INLET, ROLE_OUTLET}:
        return role
    return ROLE_UNKNOWN


def geometry_endpoints(geometry) -> tuple[Point, Point]:
    if geometry is None or geometry.is_empty:
        raise ValueError("Cannot determine endpoints for empty geometry.")
    if isinstance(geometry, LineString):
        coords = list(geometry.coords)
        return Point(coords[0]), Point(coords[-1])
    if isinstance(geometry, MultiLineString):
        parts = list(geometry.geoms)
        first_coords = list(parts[0].coords)
        last_coords = list(parts[-1].coords)
        return Point(first_coords[0]), Point(last_coords[-1])
    if hasattr(geometry, "boundary") and geometry.boundary is not None:
        boundary = geometry.boundary
        if hasattr(boundary, "geoms") and len(boundary.geoms) >= 2:
            return boundary.geoms[0], boundary.geoms[-1]
    raise TypeError(f"Unsupported linear geometry type: {geometry.geom_type}")


def geometry_node_order(
    geometry,
    node_a: int,
    node_b: int,
    node_geom_a,
    node_geom_b,
) -> tuple[int, int]:
    start_point, end_point = geometry_endpoints(geometry)
    score_ab = start_point.distance(node_geom_a) + end_point.distance(node_geom_b)
    score_ba = start_point.distance(node_geom_b) + end_point.distance(node_geom_a)
    if score_ab <= score_ba:
        return node_a, node_b
    return node_b, node_a


def reverse_linear_geometry(geometry):
    if geometry is None or geometry.is_empty:
        return geometry
    if isinstance(geometry, LineString):
        return LineString(list(geometry.coords)[::-1])
    if isinstance(geometry, MultiLineString):
        reversed_parts = [LineString(list(part.coords)[::-1]) for part in reversed(list(geometry.geoms))]
        return MultiLineString(reversed_parts)
    if hasattr(geometry, "reverse"):
        return geometry.reverse()
    raise TypeError(f"Unsupported linear geometry type: {geometry.geom_type}")


def infer_network_name(path: str | Path) -> str:
    stem = Path(path).stem
    for suffix in ("_links", "_nodes"):
        if stem.endswith(suffix):
            return stem[: -len(suffix)]
    return stem


def default_output_dir_for_links(links_path: str | Path) -> Path:
    return Path(__file__).resolve().parent / "outputs" / infer_network_name(links_path)


def resolve_output_paths(
    *,
    links_path: str | Path,
    output_dir: str | Path | None,
    out_link_csv: str | Path | None,
    out_node_csv: str | Path | None,
    out_links_gpkg: str | Path | None,
    out_nodes_gpkg: str | Path | None,
) -> dict[str, Path]:
    base_dir = Path(output_dir) if output_dir is not None else default_output_dir_for_links(links_path)
    return {
        "output_dir": base_dir,
        "out_link_csv": Path(out_link_csv) if out_link_csv is not None else base_dir / DEFAULT_LINK_CSV_NAME,
        "out_node_csv": Path(out_node_csv) if out_node_csv is not None else base_dir / DEFAULT_NODE_CSV_NAME,
        "out_links_gpkg": Path(out_links_gpkg) if out_links_gpkg is not None else base_dir / DEFAULT_LINKS_GPKG_NAME,
        "out_nodes_gpkg": Path(out_nodes_gpkg) if out_nodes_gpkg is not None else base_dir / DEFAULT_NODES_GPKG_NAME,
    }


def build_undirected_graph(link_nodes: dict[int, tuple[int, int]]) -> nx.MultiGraph:
    graph = nx.MultiGraph()
    for link_id, (node_a, node_b) in sorted(link_nodes.items()):
        graph.add_edge(node_a, node_b, key=link_id, id_link=link_id)
    return graph


def build_link_review_order(link_nodes: dict[int, tuple[int, int]]) -> list[int]:
    graph = build_undirected_graph(link_nodes)
    if graph.number_of_edges() == 0:
        return []

    order: list[int] = []
    seen_links: set[int] = set()
    visited_nodes: set[int] = set()

    seed_nodes = sorted(node for node in graph.nodes if graph.degree(node) <= 1)
    if not seed_nodes:
        seed_nodes = sorted(graph.nodes)

    node_queue: deque[int] = deque(seed_nodes)
    while node_queue:
        start_node = node_queue.popleft()
        if start_node in visited_nodes:
            continue

        bfs_queue: deque[int] = deque([start_node])
        visited_nodes.add(start_node)
        while bfs_queue:
            node = bfs_queue.popleft()
            incident_edges = sorted(graph.edges(node, keys=True), key=lambda edge: (edge[2], min(edge[0], edge[1]), max(edge[0], edge[1])))
            for node_u, node_v, link_id in incident_edges:
                if link_id not in seen_links:
                    order.append(int(link_id))
                    seen_links.add(int(link_id))
                neighbor = node_v if node == node_u else node_u
                if neighbor not in visited_nodes:
                    visited_nodes.add(neighbor)
                    bfs_queue.append(neighbor)

        leftover_nodes = sorted(node for node in graph.nodes if node not in visited_nodes)
        node_queue.extend(leftover_nodes)

    for link_id in sorted(link_nodes):
        if link_id not in seen_links:
            order.append(link_id)
    return order


class ManualDirectionReviewGui:
    def __init__(
        self,
        root: tk.Tk,
        *,
        links_path: str | Path,
        nodes_path: str | Path,
        out_link_csv: str | Path,
        out_node_csv: str | Path,
        out_links_gpkg: str | Path,
        out_nodes_gpkg: str | Path,
        detail_pad_factor: float,
        context_pad_factor: float,
    ) -> None:
        self.root = root
        self.root.title("River network manual direction review")

        self.links_path = Path(links_path)
        self.nodes_path = Path(nodes_path)
        self.out_link_csv = Path(out_link_csv)
        self.out_node_csv = Path(out_node_csv)
        self.out_links_gpkg = Path(out_links_gpkg)
        self.out_nodes_gpkg = Path(out_nodes_gpkg)
        self.detail_pad_factor = detail_pad_factor
        self.context_pad_factor = context_pad_factor

        self.links = gpd.read_file(self.links_path)
        self.nodes = gpd.read_file(self.nodes_path)
        self.link_geom_col = self.links.geometry.name
        self.node_geom_col = self.nodes.geometry.name

        self.links["id_link"] = self.links["id_link"].astype(int)
        self.nodes["id_node"] = self.nodes["id_node"].astype(int)
        self.links["_node_pair"] = self.links["id_nodes"].apply(parse_node_pair)

        self.link_index = dict(zip(self.links["id_link"], self.links.index))
        self.node_index = dict(zip(self.nodes["id_node"], self.nodes.index))
        self.link_nodes = {
            int(link_id): (int(node_pair[0]), int(node_pair[1]))
            for link_id, node_pair in zip(self.links["id_link"], self.links["_node_pair"])
        }
        self.node_geom = dict(zip(self.nodes["id_node"], self.nodes.geometry))
        self.node_source_role = {
            int(row.id_node): source_role(
                getattr(row, "is_inlet", False),
                getattr(row, "is_outlet", False),
            )
            for row in self.nodes.itertuples(index=False)
        }

        self.node_to_links: dict[int, list[int]] = defaultdict(list)
        for link_id, (node_a, node_b) in self.link_nodes.items():
            self.node_to_links[node_a].append(link_id)
            self.node_to_links[node_b].append(link_id)
        for link_ids in self.node_to_links.values():
            link_ids.sort()

        self.undirected_graph = build_undirected_graph(self.link_nodes)
        self.review_order = build_link_review_order(self.link_nodes)
        self.boundary_candidate_nodes = sorted(node for node in self.undirected_graph.nodes if self.undirected_graph.degree(node) <= 1)

        self.node_component = {}
        self.component_links: dict[int, list[int]] = defaultdict(list)
        for component_id, component_nodes in enumerate(nx.connected_components(self.undirected_graph), start=1):
            for node_id in component_nodes:
                self.node_component[int(node_id)] = component_id
            for link_id, (node_a, node_b) in self.link_nodes.items():
                if node_a in component_nodes:
                    self.component_links[component_id].append(link_id)
            self.component_links[component_id].sort()

        self.geometry_node_orders = {
            link_id: geometry_node_order(
                self.links.at[self.link_index[link_id], self.link_geom_col],
                node_a,
                node_b,
                self.node_geom[node_a],
                self.node_geom[node_b],
            )
            for link_id, (node_a, node_b) in self.link_nodes.items()
        }

        self.assigned_upstream: dict[int, int] = {}
        self.node_roles = {node_id: ROLE_INTERNAL for node_id in self.node_index}
        self.link_history: list[tuple[int, int | None]] = []
        self.current_pos: int | None = 0 if self.review_order else None

        self._load_existing_reviews()
        if self.current_pos is not None:
            next_unresolved = self._find_next_unresolved(start_pos=0)
            if next_unresolved is not None:
                self.current_pos = next_unresolved

        self._build_ui()
        self._bind_shortcuts()
        self._draw()

    def _load_existing_reviews(self) -> None:
        if self.out_link_csv.exists():
            existing_links = pd.read_csv(self.out_link_csv)
            if {"link_id", "usnode"}.issubset(existing_links.columns):
                for row in existing_links.itertuples(index=False):
                    link_id = int(row.link_id)
                    upstream_node = int(row.usnode)
                    if link_id in self.link_index:
                        self.assigned_upstream[link_id] = upstream_node

        if self.out_node_csv.exists():
            existing_nodes = pd.read_csv(self.out_node_csv)
            if {"node_id", "manual_role"}.issubset(existing_nodes.columns):
                for row in existing_nodes.itertuples(index=False):
                    node_id = int(row.node_id)
                    if node_id in self.node_roles:
                        self.node_roles[node_id] = normalize_manual_role(row.manual_role)

    def _build_ui(self) -> None:
        controls = tk.Frame(self.root)
        controls.pack(side=tk.TOP, fill=tk.X, padx=6, pady=6)

        self.info_var = tk.StringVar()
        info_label = tk.Label(
            controls,
            textvariable=self.info_var,
            anchor="w",
            justify="left",
            wraplength=1100,
        )
        info_label.pack(side=tk.TOP, fill=tk.X, padx=4, pady=(0, 6))

        button_row = tk.Frame(controls)
        button_row.pack(side=tk.TOP, fill=tk.X)
        tk.Button(button_row, text="A upstream [A]", command=lambda: self._assign_current_link("A")).pack(side=tk.LEFT, padx=2)
        tk.Button(button_row, text="B upstream [B]", command=lambda: self._assign_current_link("B")).pack(side=tk.LEFT, padx=2)
        tk.Button(button_row, text="Prev", command=lambda: self._move_relative(-1)).pack(side=tk.LEFT, padx=2)
        tk.Button(button_row, text="Next", command=lambda: self._move_relative(1)).pack(side=tk.LEFT, padx=2)
        tk.Button(button_row, text="Skip unresolved [Space]", command=self._skip_to_next_unresolved).pack(side=tk.LEFT, padx=2)
        tk.Button(button_row, text="Undo last [U]", command=self._undo_last_link_assignment).pack(side=tk.LEFT, padx=2)
        tk.Button(button_row, text="Save [W]", command=self._save).pack(side=tk.LEFT, padx=2)

        zoom_row = tk.Frame(controls)
        zoom_row.pack(side=tk.TOP, fill=tk.X, pady=(6, 0))
        tk.Label(zoom_row, text="Detail zoom:").pack(side=tk.LEFT, padx=(0, 4))
        tk.Button(zoom_row, text="In [Z]", command=lambda: self._adjust_zoom("detail", ZOOM_IN_MULTIPLIER)).pack(side=tk.LEFT, padx=2)
        tk.Button(zoom_row, text="Out [X]", command=lambda: self._adjust_zoom("detail", ZOOM_OUT_MULTIPLIER)).pack(side=tk.LEFT, padx=2)
        tk.Label(zoom_row, text="Context zoom:").pack(side=tk.LEFT, padx=(12, 4))
        tk.Button(zoom_row, text="In [C]", command=lambda: self._adjust_zoom("context", ZOOM_IN_MULTIPLIER)).pack(side=tk.LEFT, padx=2)
        tk.Button(zoom_row, text="Out [V]", command=lambda: self._adjust_zoom("context", ZOOM_OUT_MULTIPLIER)).pack(side=tk.LEFT, padx=2)

        role_row_a = tk.Frame(controls)
        role_row_a.pack(side=tk.TOP, fill=tk.X, pady=(6, 0))
        tk.Label(role_row_a, text="Node A role:").pack(side=tk.LEFT, padx=(0, 6))
        tk.Button(role_row_a, text="Inlet", command=lambda: self._set_endpoint_role("A", ROLE_INLET)).pack(side=tk.LEFT, padx=2)
        tk.Button(role_row_a, text="Outlet", command=lambda: self._set_endpoint_role("A", ROLE_OUTLET)).pack(side=tk.LEFT, padx=2)
        tk.Button(role_row_a, text="Internal", command=lambda: self._set_endpoint_role("A", ROLE_INTERNAL)).pack(side=tk.LEFT, padx=2)
        tk.Button(role_row_a, text="Clear", command=lambda: self._set_endpoint_role("A", ROLE_UNKNOWN)).pack(side=tk.LEFT, padx=2)

        role_row_b = tk.Frame(controls)
        role_row_b.pack(side=tk.TOP, fill=tk.X, pady=(4, 0))
        tk.Label(role_row_b, text="Node B role:").pack(side=tk.LEFT, padx=(0, 6))
        tk.Button(role_row_b, text="Inlet", command=lambda: self._set_endpoint_role("B", ROLE_INLET)).pack(side=tk.LEFT, padx=2)
        tk.Button(role_row_b, text="Outlet", command=lambda: self._set_endpoint_role("B", ROLE_OUTLET)).pack(side=tk.LEFT, padx=2)
        tk.Button(role_row_b, text="Internal", command=lambda: self._set_endpoint_role("B", ROLE_INTERNAL)).pack(side=tk.LEFT, padx=2)
        tk.Button(role_row_b, text="Clear", command=lambda: self._set_endpoint_role("B", ROLE_UNKNOWN)).pack(side=tk.LEFT, padx=2)

        self.figure = Figure(figsize=(14, 7), dpi=100)
        self.ax_detail = self.figure.add_subplot(121)
        self.ax_context = self.figure.add_subplot(122)
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.root)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        toolbar = NavigationToolbar2Tk(self.canvas, self.root)
        toolbar.update()
        self.canvas.mpl_connect("button_press_event", self._on_click)

    def _bind_shortcuts(self) -> None:
        self.root.bind("<Key>", self._on_key_press)

    def _current_link_id(self) -> int | None:
        if self.current_pos is None or not self.review_order:
            return None
        return self.review_order[self.current_pos]

    def _current_nodes(self) -> tuple[int, int] | None:
        current_link = self._current_link_id()
        if current_link is None:
            return None
        return self.link_nodes[current_link]

    def _find_next_unresolved(self, *, start_pos: int) -> int | None:
        if not self.review_order:
            return None
        n_links = len(self.review_order)
        for offset in range(n_links):
            position = (start_pos + offset) % n_links
            link_id = self.review_order[position]
            if link_id not in self.assigned_upstream:
                return position
        return None

    def _move_relative(self, step: int) -> None:
        if self.current_pos is None or not self.review_order:
            return
        self.current_pos = max(0, min(len(self.review_order) - 1, self.current_pos + step))
        self._draw()

    def _skip_to_next_unresolved(self) -> None:
        if self.current_pos is None:
            return
        next_pos = self._find_next_unresolved(start_pos=(self.current_pos + 1) % len(self.review_order))
        if next_pos is not None:
            self.current_pos = next_pos
        self._draw()

    def _undo_last_link_assignment(self) -> None:
        if not self.link_history:
            return
        link_id, previous_upstream = self.link_history.pop()
        if previous_upstream is None:
            self.assigned_upstream.pop(link_id, None)
        else:
            self.assigned_upstream[link_id] = previous_upstream
        self.current_pos = self.review_order.index(link_id)
        self._draw()

    def _assign_current_link(self, which: str) -> None:
        current_link = self._current_link_id()
        nodes = self._current_nodes()
        if current_link is None or nodes is None:
            return

        node_a, node_b = nodes
        upstream_node = node_a if which == "A" else node_b
        previous_upstream = self.assigned_upstream.get(current_link)
        self.link_history.append((current_link, previous_upstream))
        self.assigned_upstream[current_link] = upstream_node

        next_pos = self._find_next_unresolved(start_pos=(self.current_pos + 1) % len(self.review_order))
        if next_pos is not None:
            self.current_pos = next_pos
        self._draw()

    def _set_endpoint_role(self, which: str, role: str) -> None:
        nodes = self._current_nodes()
        if nodes is None:
            return
        node_id = nodes[0] if which == "A" else nodes[1]
        self.node_roles[node_id] = role
        self._draw()

    def _on_key_press(self, event) -> None:
        key = (event.keysym or "").lower()
        if key == "a":
            self._assign_current_link("A")
        elif key == "b":
            self._assign_current_link("B")
        elif key == "space":
            self._skip_to_next_unresolved()
        elif key == "u":
            self._undo_last_link_assignment()
        elif key == "w":
            self._save()
        elif key == "left":
            self._move_relative(-1)
        elif key == "right":
            self._move_relative(1)
        elif key == "z":
            self._adjust_zoom("detail", ZOOM_IN_MULTIPLIER)
        elif key == "x":
            self._adjust_zoom("detail", ZOOM_OUT_MULTIPLIER)
        elif key == "c":
            self._adjust_zoom("context", ZOOM_IN_MULTIPLIER)
        elif key == "v":
            self._adjust_zoom("context", ZOOM_OUT_MULTIPLIER)

    def _on_click(self, event) -> None:
        current_link = self._current_link_id()
        nodes = self._current_nodes()
        if current_link is None or nodes is None:
            return
        if event.inaxes not in {self.ax_detail, self.ax_context} or event.xdata is None or event.ydata is None:
            return

        node_a, node_b = nodes
        point_a = self.node_geom[node_a]
        point_b = self.node_geom[node_b]
        dist_a = (event.xdata - point_a.x) ** 2 + (event.ydata - point_a.y) ** 2
        dist_b = (event.xdata - point_b.x) ** 2 + (event.ydata - point_b.y) ** 2
        self._assign_current_link("A" if dist_a <= dist_b else "B")

    def _role_nodes_in_component(self, component_id: int) -> list[int]:
        return sorted(
            node_id
            for node_id, role in self.node_roles.items()
            if role != ROLE_UNKNOWN and self.node_component.get(node_id) == component_id
        )

    def _set_limits(self, ax, bounds_source: gpd.GeoDataFrame, pad_factor: float) -> None:
        minx, miny, maxx, maxy = bounds_source.total_bounds
        dx = max(maxx - minx, 1.0)
        dy = max(maxy - miny, 1.0)
        pad = max(dx, dy) * pad_factor
        ax.set_xlim(minx - pad, maxx + pad)
        ax.set_ylim(miny - pad, maxy + pad)
        ax.set_aspect("equal", adjustable="box")

    def _adjust_zoom(self, which: str, multiplier: float) -> None:
        if which == "detail":
            self.detail_pad_factor = min(MAX_PAD_FACTOR, max(MIN_PAD_FACTOR, self.detail_pad_factor * multiplier))
        elif which == "context":
            self.context_pad_factor = min(MAX_PAD_FACTOR, max(MIN_PAD_FACTOR, self.context_pad_factor * multiplier))
        else:
            raise ValueError(f"Unknown zoom target: {which}")
        self._draw()

    def _label_position(
        self,
        point: Point,
        other_point: Point,
        *,
        label_offset: float,
        direction_sign: float,
    ) -> tuple[float, float]:
        delta_x = other_point.x - point.x
        delta_y = other_point.y - point.y
        length = (delta_x**2 + delta_y**2) ** 0.5
        if length <= 1e-9:
            return point.x + direction_sign * label_offset, point.y + label_offset
        perp_x = -delta_y / length
        perp_y = delta_x / length
        return point.x + direction_sign * perp_x * label_offset, point.y + direction_sign * perp_y * label_offset

    def _plot_endpoint_markers(
        self,
        ax,
        node_a: int,
        node_b: int,
        *,
        label_offset: float,
    ) -> None:
        point_a = self.node_geom[node_a]
        point_b = self.node_geom[node_b]
        for label, node_id in (("A", node_a), ("B", node_b)):
            point = self.node_geom[node_id]
            role = self.node_roles[node_id]
            ax.scatter(
                [point.x],
                [point.y],
                s=100,
                facecolor=ROLE_COLORS[role],
                edgecolor="black",
                zorder=6,
            )
            label_x, label_y = self._label_position(
                point,
                point_b if node_id == node_a else point_a,
                label_offset=label_offset,
                direction_sign=1.0 if node_id == node_a else -1.0,
            )
            ax.text(
                label_x,
                label_y,
                label,
                fontsize=9,
                weight="bold",
                ha="center",
                va="center",
                bbox={"facecolor": "white", "edgecolor": "black", "boxstyle": "round,pad=0.2", "alpha": 0.85},
                zorder=7,
            )

    def _plot_role_markers(self, ax, node_ids: list[int]) -> None:
        for node_id in node_ids:
            point = self.node_geom[node_id]
            role = self.node_roles[node_id]
            ax.scatter(
                [point.x],
                [point.y],
                s=50,
                facecolor=ROLE_COLORS[role],
                edgecolor="black",
                linewidth=0.7,
                zorder=5,
            )

    def _draw_panel(
        self,
        ax,
        *,
        component_links: list[int],
        current_link: int,
        neighbor_links: set[int],
        node_a: int,
        node_b: int,
        geometry_start: int,
        geometry_end: int,
        title: str,
        pad_factor: float,
    ) -> None:
        ax.clear()

        component_frame = self.links[self.links["id_link"].isin(component_links)]
        bounds_source = component_frame[component_frame["id_link"].isin(neighbor_links | {current_link})]
        if bounds_source.empty:
            bounds_source = component_frame
        minx, miny, maxx, maxy = bounds_source.total_bounds
        span = max(maxx - minx, maxy - miny, 1.0)
        label_offset = span * 0.04

        reviewed_links = set(self.assigned_upstream)
        reviewed_frame = component_frame[component_frame["id_link"].isin(reviewed_links)]
        neighbor_frame = component_frame[component_frame["id_link"].isin(neighbor_links)]
        current_frame = component_frame[component_frame["id_link"] == current_link]

        component_frame.plot(ax=ax, color="0.88", linewidth=1)
        if not reviewed_frame.empty:
            reviewed_frame.plot(ax=ax, color="0.55", linewidth=1.5)
        if not neighbor_frame.empty:
            neighbor_frame.plot(ax=ax, color="#f28e2b", linewidth=2)
        current_frame.plot(ax=ax, color="#d62728", linewidth=3)

        component_id = self.node_component[node_a]
        candidate_nodes = [node_id for node_id in self.boundary_candidate_nodes if self.node_component.get(node_id) == component_id]
        if candidate_nodes:
            candidate_frame = self.nodes[self.nodes["id_node"].isin(candidate_nodes)]
            candidate_frame.plot(ax=ax, color="white", edgecolor="0.3", markersize=25, linewidth=0.7)

        manual_role_nodes = self._role_nodes_in_component(component_id)
        self._plot_role_markers(ax, manual_role_nodes)
        self._plot_endpoint_markers(ax, node_a, node_b, label_offset=label_offset)

        self._set_limits(ax, bounds_source, pad_factor)
        ax.set_title(f"{title} (pad={pad_factor:.2f})")

    def _draw(self) -> None:
        self.ax_detail.clear()
        self.ax_context.clear()

        current_link = self._current_link_id()
        if current_link is None:
            self.info_var.set(
                f"No links to review. Assigned {len(self.assigned_upstream)}/{len(self.review_order)} links."
            )
            self.ax_detail.set_title("No active link")
            self.ax_context.set_title("No active link")
            self.canvas.draw_idle()
            return

        node_a, node_b = self.link_nodes[current_link]
        geometry_start, geometry_end = self.geometry_node_orders[current_link]
        component_id = self.node_component[node_a]
        component_links = self.component_links[component_id]
        neighbor_links = set(self.node_to_links[node_a]) | set(self.node_to_links[node_b])
        neighbor_links.discard(current_link)

        self._draw_panel(
            self.ax_detail,
            component_links=component_links,
            current_link=current_link,
            neighbor_links=neighbor_links,
            node_a=node_a,
            node_b=node_b,
            geometry_start=geometry_start,
            geometry_end=geometry_end,
            title="Detail view",
            pad_factor=self.detail_pad_factor,
        )
        self._draw_panel(
            self.ax_context,
            component_links=component_links,
            current_link=current_link,
            neighbor_links=neighbor_links,
            node_a=node_a,
            node_b=node_b,
            geometry_start=geometry_start,
            geometry_end=geometry_end,
            title="Zoomed-out context",
            pad_factor=self.context_pad_factor,
        )

        assigned_upstream = self.assigned_upstream.get(current_link)
        manual_direction = "unassigned"
        reverse_note = ""
        if assigned_upstream is not None:
            downstream_node = node_b if assigned_upstream == node_a else node_a
            manual_direction = f"{assigned_upstream} -> {downstream_node}"
            reverse_note = " | geometry will reverse on save" if assigned_upstream != geometry_start else " | geometry order already matches"

        info_lines = [
            (
                f"Link {current_link} | order {self.current_pos + 1}/{len(self.review_order)} | "
                f"assigned {len(self.assigned_upstream)}/{len(self.review_order)} links | "
                f"boundary candidates {len(self.boundary_candidate_nodes)} | default node role: {ROLE_INTERNAL}"
            ),
            (
                f"File id_nodes: {node_a} -> {node_b} | geometry endpoints: {geometry_start} -> {geometry_end} | "
                f"manual: {manual_direction}{reverse_note}"
            ),
            (
                f"A=node {node_a} | manual role={node_role_label(self.node_roles[node_a])} | "
                f"source role={self.node_source_role[node_a]}"
            ),
            (
                f"B=node {node_b} | manual role={node_role_label(self.node_roles[node_b])} | "
                f"source role={self.node_source_role[node_b]}"
            ),
            (
                f"Zoom: detail pad={self.detail_pad_factor:.2f}, context pad={self.context_pad_factor:.2f} "
                f"| keys: Z/X detail in/out, C/V context in/out"
            ),
        ]
        self.info_var.set("\n".join(info_lines))
        self.canvas.draw_idle()

    def _build_link_output(self) -> tuple[pd.DataFrame, gpd.GeoDataFrame]:
        records = []
        links_out = self.links.drop(columns=["_node_pair"]).copy()

        links_out["manual_direction_reviewed"] = False
        links_out["manual_us_node"] = pd.NA
        links_out["manual_ds_node"] = pd.NA
        links_out["geometry_reversed_manual"] = False

        for link_id in self.review_order:
            row_index = self.link_index[link_id]
            node_a, node_b = self.link_nodes[link_id]
            upstream_node = int(self.assigned_upstream[link_id])
            downstream_node = node_b if upstream_node == node_a else node_a
            geometry_start, geometry_end = self.geometry_node_orders[link_id]
            geometry_reversed = upstream_node != geometry_start
            upstream_is_inlet = self.node_roles[upstream_node] == ROLE_INLET
            downstream_is_outlet = self.node_roles[downstream_node] == ROLE_OUTLET

            links_out.at[row_index, "manual_direction_reviewed"] = True
            links_out.at[row_index, "manual_us_node"] = upstream_node
            links_out.at[row_index, "manual_ds_node"] = downstream_node
            links_out.at[row_index, "id_us_node"] = upstream_node
            links_out.at[row_index, "id_ds_node"] = downstream_node
            links_out.at[row_index, "id_nodes"] = f"{upstream_node}, {downstream_node}"
            links_out.at[row_index, "geometry_reversed_manual"] = geometry_reversed
            if "is_inlet" in links_out.columns:
                links_out.at[row_index, "is_inlet"] = upstream_is_inlet
            if "is_outlet" in links_out.columns:
                links_out.at[row_index, "is_outlet"] = downstream_is_outlet
            if geometry_reversed:
                links_out.at[row_index, self.link_geom_col] = reverse_linear_geometry(links_out.at[row_index, self.link_geom_col])

            records.append(
                {
                    "link_id": link_id,
                    "usnode": upstream_node,
                    "dsnode": downstream_node,
                    "source_node_a": node_a,
                    "source_node_b": node_b,
                    "source_geometry_start_node": geometry_start,
                    "source_geometry_end_node": geometry_end,
                    "geometry_reversed": geometry_reversed,
                }
            )

        return pd.DataFrame.from_records(records), gpd.GeoDataFrame(links_out, geometry=self.link_geom_col, crs=self.links.crs)

    def _build_node_output(self) -> tuple[pd.DataFrame, gpd.GeoDataFrame]:
        nodes_out = self.nodes.copy()

        if "is_inlet" in nodes_out.columns:
            nodes_out["source_is_inlet"] = nodes_out["is_inlet"]
        else:
            nodes_out["source_is_inlet"] = False
        if "is_outlet" in nodes_out.columns:
            nodes_out["source_is_outlet"] = nodes_out["is_outlet"]
        else:
            nodes_out["source_is_outlet"] = False

        manual_roles = [self.node_roles[int(node_id)] for node_id in nodes_out["id_node"]]
        nodes_out["manual_role"] = manual_roles
        nodes_out["node_role_reviewed"] = nodes_out["manual_role"] != ROLE_UNKNOWN
        nodes_out["is_inlet"] = nodes_out["manual_role"] == ROLE_INLET
        nodes_out["is_outlet"] = nodes_out["manual_role"] == ROLE_OUTLET

        records = []
        for row in nodes_out.itertuples(index=False):
            node_id = int(row.id_node)
            manual_role = self.node_roles[node_id]
            inlet, outlet = manual_role_to_flags(manual_role)
            records.append(
                {
                    "node_id": node_id,
                    "manual_role": manual_role,
                    "node_role_reviewed": manual_role != ROLE_UNKNOWN,
                    "is_inlet": inlet,
                    "is_outlet": outlet,
                    "source_role": self.node_source_role[node_id],
                }
            )

        return pd.DataFrame.from_records(records), gpd.GeoDataFrame(nodes_out, geometry=self.node_geom_col, crs=self.nodes.crs)

    def _save(self) -> None:
        if len(self.assigned_upstream) != len(self.review_order):
            messagebox.showwarning(
                "Incomplete link review",
                f"Assigned {len(self.assigned_upstream)} of {len(self.review_order)} links. "
                "Finish link directions before saving a directed network.",
            )
            return

        link_frame, links_out = self._build_link_output()
        node_frame, nodes_out = self._build_node_output()

        self.out_link_csv.parent.mkdir(parents=True, exist_ok=True)
        self.out_node_csv.parent.mkdir(parents=True, exist_ok=True)
        self.out_links_gpkg.parent.mkdir(parents=True, exist_ok=True)
        self.out_nodes_gpkg.parent.mkdir(parents=True, exist_ok=True)

        link_frame.to_csv(self.out_link_csv, index=False)
        node_frame.to_csv(self.out_node_csv, index=False)
        links_out.to_file(self.out_links_gpkg, driver="GPKG")
        nodes_out.to_file(self.out_nodes_gpkg, driver="GPKG")

        messagebox.showinfo(
            "Saved review outputs",
            (
                f"Wrote {self.out_link_csv}\n"
                f"Wrote {self.out_node_csv}\n"
                f"Wrote {self.out_links_gpkg}\n"
                f"Wrote {self.out_nodes_gpkg}"
            ),
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Manual review GUI for RivGraph links/nodes. Assign upstream endpoints per link, "
            "edit node inlet/outlet roles, and export corrected GeoPackages with reversed "
            "LineString coordinates when needed."
        )
    )
    parser.add_argument("links_gpkg", help="Path to the links GeoPackage.")
    parser.add_argument("nodes_gpkg", help="Path to the nodes GeoPackage.")
    parser.add_argument(
        "--output-dir",
        default=None,
        help=(
            "Directory for review outputs. Defaults to "
            "`hierarchy_level_definition/manual_review/outputs/<network_name>/`."
        ),
    )
    parser.add_argument("--out-link-csv", default=None, help=f"Override the reviewed link CSV path. Default: <output-dir>/{DEFAULT_LINK_CSV_NAME}.")
    parser.add_argument("--out-node-csv", default=None, help=f"Override the reviewed node CSV path. Default: <output-dir>/{DEFAULT_NODE_CSV_NAME}.")
    parser.add_argument("--out-links-gpkg", default=None, help=f"Override the reviewed links GeoPackage path. Default: <output-dir>/{DEFAULT_LINKS_GPKG_NAME}.")
    parser.add_argument("--out-nodes-gpkg", default=None, help=f"Override the reviewed nodes GeoPackage path. Default: <output-dir>/{DEFAULT_NODES_GPKG_NAME}.")
    parser.add_argument(
        "--detail-pad-factor",
        type=float,
        default=1.2,
        help="Padding multiplier for the left detail view around the current link and neighbors.",
    )
    parser.add_argument(
        "--context-pad-factor",
        type=float,
        default=4.0,
        help="Padding multiplier for the right zoomed-out context view.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output_paths = resolve_output_paths(
        links_path=args.links_gpkg,
        output_dir=args.output_dir,
        out_link_csv=args.out_link_csv,
        out_node_csv=args.out_node_csv,
        out_links_gpkg=args.out_links_gpkg,
        out_nodes_gpkg=args.out_nodes_gpkg,
    )

    root = tk.Tk()
    ManualDirectionReviewGui(
        root,
        links_path=args.links_gpkg,
        nodes_path=args.nodes_gpkg,
        out_link_csv=output_paths["out_link_csv"],
        out_node_csv=output_paths["out_node_csv"],
        out_links_gpkg=output_paths["out_links_gpkg"],
        out_nodes_gpkg=output_paths["out_nodes_gpkg"],
        detail_pad_factor=args.detail_pad_factor,
        context_pad_factor=args.context_pad_factor,
    )
    root.mainloop()


if __name__ == "__main__":
    main()
