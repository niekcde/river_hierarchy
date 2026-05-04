from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import geopandas as gpd
import networkx as nx
import pandas as pd


@dataclass
class UnitPath:
    path_id: int
    node_path: list[int]
    edge_path: list[tuple[int, int, int]]
    id_links: list[int]
    total_length: float


@dataclass
class StructuralUnit:
    unit_id: int
    bifurcation: int
    confluence: int
    n_paths: int
    unit_class: str
    min_path_length: float
    max_path_length: float
    paths: list[UnitPath] = field(default_factory=list)
    node_set: set[int] = field(default_factory=set)
    edge_set: set[tuple[int, int, int]] = field(default_factory=set)
    internal_bifurcations: list[int] = field(default_factory=list)
    internal_confluences: list[int] = field(default_factory=list)
    children: list[int] = field(default_factory=list)
    parents: list[int] = field(default_factory=list)
    path_cutoff_used: int | None = None
    path_enumeration_truncated: bool = False
    primary_parent_id: int | None = None
    root_unit_id: int | None = None
    depth_from_root: int | None = None
    collapse_level: int | None = None
    n_descendants: int = 0
    is_compound: bool = False
    compound_unit_id: int | None = None
    compound_bubble_id: int | None = None
    in_compound_bubble: bool = False
    compound_bubble_role: str = "standalone"

    def to_record(self) -> dict[str, Any]:
        return {
            "unit_id": self.unit_id,
            "bifurcation": self.bifurcation,
            "confluence": self.confluence,
            "n_paths": self.n_paths,
            "class": self.unit_class,
            "min_path_length": self.min_path_length,
            "max_path_length": self.max_path_length,
            "unit_node_ids": ",".join(str(node_id) for node_id in sorted(self.node_set)),
            "unit_node_count": len(self.node_set),
            "internal_bifurcations": self.internal_bifurcations,
            "internal_confluences": self.internal_confluences,
            "children": self.children,
            "parents": self.parents,
            "primary_parent_id": self.primary_parent_id,
            "root_unit_id": self.root_unit_id,
            "depth_from_root": self.depth_from_root,
            "collapse_level": self.collapse_level,
            "n_children": len(self.children),
            "n_descendants": self.n_descendants,
            "is_compound": self.is_compound,
            "path_cutoff_used": self.path_cutoff_used,
            "path_enumeration_truncated": self.path_enumeration_truncated,
            "compound_unit_id": self.compound_unit_id,
            "compound_bubble_id": self.compound_bubble_id,
            "in_compound_bubble": self.in_compound_bubble,
            "compound_bubble_role": self.compound_bubble_role,
        }

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "unit_id": self.unit_id,
            "bifurcation": self.bifurcation,
            "confluence": self.confluence,
            "n_paths": self.n_paths,
            "class": self.unit_class,
            "min_path_length": self.min_path_length,
            "max_path_length": self.max_path_length,
            "node_ids": sorted(self.node_set),
            "internal_bifurcations": self.internal_bifurcations,
            "internal_confluences": self.internal_confluences,
            "children": self.children,
            "parents": self.parents,
            "primary_parent_id": self.primary_parent_id,
            "root_unit_id": self.root_unit_id,
            "depth_from_root": self.depth_from_root,
            "collapse_level": self.collapse_level,
            "n_children": len(self.children),
            "n_descendants": self.n_descendants,
            "is_compound": self.is_compound,
            "path_cutoff_used": self.path_cutoff_used,
            "path_enumeration_truncated": self.path_enumeration_truncated,
            "compound_unit_id": self.compound_unit_id,
            "compound_bubble_id": self.compound_bubble_id,
            "in_compound_bubble": self.in_compound_bubble,
            "compound_bubble_role": self.compound_bubble_role,
            "paths": [
                {
                    "path_id": path.path_id,
                    "node_path": path.node_path,
                    "edge_path": [list(edge) for edge in path.edge_path],
                    "id_links": path.id_links,
                    "total_length": path.total_length,
                }
                for path in self.paths
            ],
        }


def parse_id_nodes(value: Any) -> tuple[int, int]:
    if pd.isna(value):
        raise ValueError("Encountered missing id_nodes value.")
    parts = [part.strip() for part in str(value).split(",")]
    if len(parts) != 2:
        raise ValueError(f"Expected 'u, v' id_nodes format, got {value!r}.")
    return int(parts[0]), int(parts[1])


def edge_length_from_attrs(attrs: dict[str, Any]) -> float:
    for key in ("len_adj", "len"):
        value = attrs.get(key)
        if value is not None and not pd.isna(value):
            return float(value)
    geometry = attrs.get("geometry")
    if geometry is not None:
        return float(geometry.length)
    return 0.0


def load_network(
    links_path: str | Path,
    nodes_path: str | Path,
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    links = gpd.read_file(links_path)
    nodes = gpd.read_file(nodes_path)
    return links, nodes


def build_graph(
    links: gpd.GeoDataFrame,
    nodes: gpd.GeoDataFrame,
) -> nx.MultiDiGraph:
    graph = nx.MultiDiGraph()

    for row in nodes.itertuples(index=False):
        attrs = row._asdict()
        node_id = int(attrs["id_node"])
        graph.add_node(node_id, **attrs)

    for row in links.itertuples(index=False):
        attrs = row._asdict()
        u, v = parse_id_nodes(attrs["id_nodes"])
        edge_key = int(attrs["id_link"])
        graph.add_edge(u, v, key=edge_key, **attrs)

        if u not in graph:
            graph.add_node(u, id_node=u)
        if v not in graph:
            graph.add_node(v, id_node=v)

    return graph


def unique_successors(graph: nx.MultiDiGraph, node: int) -> list[int]:
    return sorted(set(graph.successors(node)))


def unique_predecessors(graph: nx.MultiDiGraph, node: int) -> list[int]:
    return sorted(set(graph.predecessors(node)))


def find_bifurcations(graph: nx.MultiDiGraph) -> list[int]:
    return sorted(node for node in graph.nodes if graph.out_degree(node) > 1)


def find_confluences(graph: nx.MultiDiGraph) -> list[int]:
    return sorted(node for node in graph.nodes if graph.in_degree(node) > 1)


def outgoing_branch_edges(graph: nx.MultiDiGraph, node: int) -> list[tuple[int, int, int]]:
    return sorted((int(u), int(v), int(key)) for u, v, key in graph.out_edges(node, keys=True))


def downstream_distances_without_bifurcation(
    graph: nx.MultiDiGraph,
    bifurcation: int,
    branch_edge: tuple[int, int, int],
) -> dict[int, int]:
    _, branch_start, _ = branch_edge
    branch_graph = graph.copy()
    if bifurcation in branch_graph:
        branch_graph.remove_node(bifurcation)
    return nx.single_source_shortest_path_length(branch_graph, branch_start)


def _candidate_confluence_score(
    branch_distances: dict[tuple[int, int, int], dict[int, int]],
    candidate: int,
) -> tuple[int, int, int]:
    distances = [distances_by_branch[candidate] for distances_by_branch in branch_distances.values()]
    return max(distances), sum(distances), candidate


def first_rejoin_confluence(
    graph: nx.MultiDiGraph,
    bifurcation: int,
    confluences: set[int] | list[int],
) -> int | None:
    """
    Return the nearest common downstream confluence for all immediate branches.

    The graph can contain directed loops. To keep a branch from "rejoining" by
    simply cycling back through the bifurcation node, each branch is traced in a
    copy of the graph with the bifurcation removed. "First" is then approximated
    as the common confluence that minimizes the branch-wise shortest-hop score.
    """
    branch_edges = outgoing_branch_edges(graph, bifurcation)
    if len(branch_edges) < 2:
        return None

    confluence_set = set(confluences)
    branch_distances = {
        branch_edge: downstream_distances_without_bifurcation(graph, bifurcation, branch_edge)
        for branch_edge in branch_edges
    }

    common_reachable = set.intersection(*(set(distances) for distances in branch_distances.values()))
    candidates = sorted(common_reachable & confluence_set)
    if not candidates:
        return None

    return min(candidates, key=lambda candidate: _candidate_confluence_score(branch_distances, candidate))


def normalize_edge_step(step: tuple[Any, ...]) -> tuple[int, int, int | None]:
    if len(step) == 3:
        u, v, key = step
        return int(u), int(v), int(key)
    if len(step) == 2:
        u, v = step
        return int(u), int(v), None
    raise ValueError(f"Unexpected edge step format: {step!r}")


def _resolve_edge_key(
    graph: nx.MultiDiGraph,
    u: int,
    v: int,
    key: int | None,
) -> int:
    if key is not None:
        return key
    edge_keys = list(graph[u][v].keys())
    if len(edge_keys) != 1:
        raise ValueError(f"Edge ({u}, {v}) is ambiguous in MultiDiGraph without an explicit key.")
    return int(edge_keys[0])


def unit_candidate_subgraph(
    graph: nx.MultiDiGraph,
    bifurcation: int,
    confluence: int,
) -> nx.MultiDiGraph:
    downstream_nodes = nx.descendants(graph, bifurcation) | {bifurcation}
    upstream_nodes = nx.ancestors(graph, confluence) | {confluence}
    candidate_nodes = downstream_nodes & upstream_nodes
    return graph.subgraph(candidate_nodes).copy()


def extract_simple_paths(
    graph: nx.MultiDiGraph,
    bifurcation: int,
    confluence: int,
    *,
    max_path_cutoff: int = 100,
    max_paths: int = 5000,
) -> tuple[list[UnitPath], int, bool]:
    subgraph = unit_candidate_subgraph(graph, bifurcation, confluence)

    shortest_hops = nx.shortest_path_length(subgraph, bifurcation, confluence)
    max_simple_cutoff = max(1, subgraph.number_of_nodes() - 1)
    cutoff = min(max_simple_cutoff, max(max_path_cutoff, shortest_hops))

    paths: list[UnitPath] = []
    truncated = False
    path_iter = nx.all_simple_edge_paths(subgraph, bifurcation, confluence, cutoff=cutoff)
    for path_id, edge_path in enumerate(path_iter, start=1):
        if path_id > max_paths:
            truncated = True
            break

        normalized_edges: list[tuple[int, int, int]] = []
        id_links: list[int] = []
        total_length = 0.0
        node_path: list[int] = []

        for index, step in enumerate(edge_path):
            u, v, raw_key = normalize_edge_step(step)
            key = _resolve_edge_key(subgraph, u, v, raw_key)
            attrs = subgraph[u][v][key]
            normalized_edges.append((u, v, key))
            id_links.append(int(attrs["id_link"]))
            total_length += edge_length_from_attrs(attrs)
            if index == 0:
                node_path.append(u)
            node_path.append(v)

        paths.append(
            UnitPath(
                path_id=path_id,
                node_path=node_path,
                edge_path=normalized_edges,
                id_links=id_links,
                total_length=total_length,
            )
        )

    return paths, cutoff, truncated


def classify_unit(
    n_paths: int,
    internal_bifurcations: list[int],
    internal_confluences: list[int],
) -> str:
    if internal_bifurcations or internal_confluences:
        return "compound_or_nested_complex"
    if n_paths == 2:
        return "simple_bifurcation_confluence_pair"
    if n_paths > 2:
        return "multi_thread_pair"
    return "compound_or_nested_complex"


def build_structural_units(
    graph: nx.MultiDiGraph,
    *,
    max_path_cutoff: int = 100,
    max_paths: int = 5000,
    debug: bool = False,
) -> list[StructuralUnit]:
    bifurcations = find_bifurcations(graph)
    confluences = set(find_confluences(graph))

    units_by_pair: dict[tuple[int, int], StructuralUnit] = {}
    next_unit_id = 1

    for bifurcation in bifurcations:
        confluence = first_rejoin_confluence(graph, bifurcation, confluences)
        if confluence is None:
            if debug:
                print(f"[debug] bifurcation {bifurcation}: no common downstream confluence found")
            continue

        pair = (bifurcation, confluence)
        if pair in units_by_pair:
            continue

        try:
            paths, cutoff_used, truncated = extract_simple_paths(
                graph,
                bifurcation,
                confluence,
                max_path_cutoff=max_path_cutoff,
                max_paths=max_paths,
            )
        except nx.NetworkXNoPath:
            if debug:
                print(f"[debug] bifurcation {bifurcation}: no path to confluence {confluence}")
            continue

        if len(paths) < 2:
            if debug:
                print(
                    f"[debug] bifurcation {bifurcation}: confluence {confluence} "
                    f"yielded only {len(paths)} simple path(s); skipped"
                )
            continue

        node_set = set()
        edge_set = set()
        for path in paths:
            node_set.update(path.node_path)
            edge_set.update(path.edge_path)

        internal_nodes = sorted(node for node in node_set if node not in {bifurcation, confluence})
        internal_bifurcations = [node for node in internal_nodes if graph.out_degree(node) > 1]
        internal_confluences = [node for node in internal_nodes if graph.in_degree(node) > 1]
        path_lengths = [path.total_length for path in paths]

        unit = StructuralUnit(
            unit_id=next_unit_id,
            bifurcation=bifurcation,
            confluence=confluence,
            n_paths=len(paths),
            unit_class=classify_unit(len(paths), internal_bifurcations, internal_confluences),
            min_path_length=min(path_lengths),
            max_path_length=max(path_lengths),
            paths=paths,
            node_set=node_set,
            edge_set=edge_set,
            internal_bifurcations=internal_bifurcations,
            internal_confluences=internal_confluences,
            path_cutoff_used=cutoff_used,
            path_enumeration_truncated=truncated,
        )
        units_by_pair[pair] = unit
        next_unit_id += 1

        if debug:
            print(
                f"[debug] unit {unit.unit_id}: bif={bifurcation}, conf={confluence}, "
                f"paths={unit.n_paths}, class={unit.unit_class}"
            )

    units = list(units_by_pair.values())
    assign_hierarchy(units)
    annotate_unit_context(units, graph=graph)
    return sorted(units, key=lambda unit: unit.unit_id)


def unit_contains(parent: StructuralUnit, child: StructuralUnit) -> bool:
    if parent.unit_id == child.unit_id:
        return False
    if not child.edge_set.issubset(parent.edge_set):
        return False
    if child.edge_set == parent.edge_set:
        return False
    if child.bifurcation not in parent.node_set:
        return False
    if child.confluence not in parent.node_set:
        return False
    return True


def assign_hierarchy(units: list[StructuralUnit]) -> None:
    containing_pairs = {
        (parent.unit_id, child.unit_id)
        for parent in units
        for child in units
        if unit_contains(parent, child)
    }

    direct_pairs: set[tuple[int, int]] = set()
    for parent_id, child_id in containing_pairs:
        is_transitive = any(
            (parent_id, middle.unit_id) in containing_pairs and (middle.unit_id, child_id) in containing_pairs
            for middle in units
            if middle.unit_id not in {parent_id, child_id}
        )
        if not is_transitive:
            direct_pairs.add((parent_id, child_id))

    units_by_id = {unit.unit_id: unit for unit in units}
    for unit in units:
        unit.children = []
        unit.parents = []

    for parent_id, child_id in sorted(direct_pairs):
        units_by_id[parent_id].children.append(child_id)
        units_by_id[child_id].parents.append(parent_id)

    for unit in units:
        unit.children.sort()
        unit.parents.sort()


def _count_descendants(children_map: dict[int, list[int]], unit_id: int) -> int:
    children = children_map.get(unit_id, [])
    return len(children) + sum(_count_descendants(children_map, child_id) for child_id in children)


def _collapse_level(children_map: dict[int, list[int]], unit_id: int) -> int:
    children = children_map.get(unit_id, [])
    if not children:
        return 0
    return 1 + max(_collapse_level(children_map, child_id) for child_id in children)


def _node_order_index(graph: nx.MultiDiGraph | None) -> dict[int, int]:
    if graph is None or graph.number_of_nodes() == 0:
        return {}
    try:
        ordered_nodes = list(nx.topological_sort(graph))
    except nx.NetworkXUnfeasible:
        source_nodes = sorted(node for node in graph.nodes if graph.in_degree(node) == 0 and graph.out_degree(node) > 0)
        if source_nodes:
            distances = nx.multi_source_dijkstra_path_length(graph, source_nodes, weight=None)
            ordered_nodes = sorted(
                graph.nodes,
                key=lambda node: (
                    distances.get(node, float("inf")),
                    int(node),
                ),
            )
        else:
            ordered_nodes = sorted(int(node) for node in graph.nodes)
    return {int(node): index for index, node in enumerate(ordered_nodes)}


def _unit_context_available(units: list[StructuralUnit]) -> bool:
    if not units:
        return True
    return all(unit.root_unit_id is not None and unit.compound_bubble_id is not None for unit in units)


def annotate_unit_context(
    units: list[StructuralUnit],
    graph: nx.MultiDiGraph | None = None,
) -> None:
    if graph is None and _unit_context_available(units):
        return

    units_by_id = {unit.unit_id: unit for unit in units}
    primary_parent_by_child: dict[int, int] = {}
    node_order = _node_order_index(graph)

    for unit in units:
        unit.primary_parent_id = None
        unit.root_unit_id = None
        unit.depth_from_root = None
        unit.collapse_level = None
        unit.n_descendants = 0
        unit.is_compound = False
        unit.compound_unit_id = None
        unit.compound_bubble_id = None
        unit.in_compound_bubble = False
        unit.compound_bubble_role = "standalone"

    for unit in units:
        if not unit.parents:
            continue
        primary_parent_by_child[unit.unit_id] = min(
            unit.parents,
            key=lambda parent_id: (
                len(units_by_id[parent_id].edge_set),
                len(units_by_id[parent_id].node_set),
                parent_id,
            ),
        )

    primary_children: dict[int, list[int]] = {unit.unit_id: [] for unit in units}
    for child_id, parent_id in primary_parent_by_child.items():
        primary_children[parent_id].append(child_id)
    for child_ids in primary_children.values():
        child_ids.sort()

    def walk(unit_id: int, *, root_unit_id: int, depth_from_root: int) -> None:
        unit = units_by_id[unit_id]
        unit.primary_parent_id = primary_parent_by_child.get(unit_id)
        unit.root_unit_id = root_unit_id
        unit.depth_from_root = depth_from_root
        unit.collapse_level = _collapse_level(primary_children, unit_id)
        unit.n_descendants = _count_descendants(primary_children, unit_id)
        unit.is_compound = len(unit.children) > 0
        root_is_compound = len(primary_children[root_unit_id]) > 0
        unit.compound_unit_id = root_unit_id if root_is_compound else None

        for child_id in primary_children[unit_id]:
            walk(
                child_id,
                root_unit_id=root_unit_id,
                depth_from_root=depth_from_root + 1,
            )

    root_ids = sorted(unit.unit_id for unit in units if unit.unit_id not in primary_parent_by_child)
    for root_id in root_ids:
        walk(root_id, root_unit_id=root_id, depth_from_root=0)

    bubble_graph = nx.Graph()
    for unit in units:
        bubble_graph.add_node(unit.unit_id)

    sorted_units = sorted(units, key=lambda unit: unit.unit_id)
    for index, left_unit in enumerate(sorted_units):
        for right_unit in sorted_units[index + 1 :]:
            if left_unit.edge_set & right_unit.edge_set:
                bubble_graph.add_edge(left_unit.unit_id, right_unit.unit_id)

    def bubble_root_key(unit_id: int) -> tuple[int, int, int, int, int]:
        unit = units_by_id[unit_id]
        return (
            node_order.get(unit.bifurcation, int(unit.bifurcation)),
            node_order.get(unit.confluence, int(unit.confluence)),
            -len(unit.edge_set),
            -len(unit.node_set),
            unit.unit_id,
        )

    components = [sorted(component) for component in nx.connected_components(bubble_graph)]
    bubble_specs = []
    for component in components:
        bubble_root_unit_id = min(component, key=bubble_root_key)
        bubble_specs.append((component, bubble_root_unit_id))

    bubble_specs.sort(key=lambda item: bubble_root_key(item[1]))

    for bubble_id, (component, bubble_root_unit_id) in enumerate(bubble_specs, start=1):
        component_units = [units_by_id[unit_id] for unit_id in component]
        bubble_size = len(component_units)
        for unit in component_units:
            unit.compound_bubble_id = bubble_id
            unit.in_compound_bubble = bubble_size > 1
            if bubble_size == 1:
                unit.compound_bubble_role = "standalone"
            elif unit.unit_id == bubble_root_unit_id:
                unit.compound_bubble_role = "bubble_root"
            else:
                unit.compound_bubble_role = "bubble_member"


def build_unit_context_frame(
    units: list[StructuralUnit],
    graph: nx.MultiDiGraph | None = None,
) -> pd.DataFrame:
    annotate_unit_context(units, graph=graph)
    records = [
        {
            "unit_id": unit.unit_id,
            "primary_parent_id": unit.primary_parent_id,
            "root_unit_id": unit.root_unit_id,
            "depth_from_root": unit.depth_from_root,
            # This is a hierarchy scale label only. It does not imply recursive geometric collapse.
            "collapse_level": unit.collapse_level,
            "n_children": len(unit.children),
            "n_descendants": unit.n_descendants,
            "is_compound": unit.is_compound,
            "compound_unit_id": unit.compound_unit_id,
            "compound_bubble_id": unit.compound_bubble_id,
            "in_compound_bubble": unit.in_compound_bubble,
            "compound_bubble_role": unit.compound_bubble_role,
            "unit_node_ids": ",".join(str(node_id) for node_id in sorted(unit.node_set)),
            "unit_node_count": len(unit.node_set),
        }
        for unit in units
    ]
    metadata = pd.DataFrame.from_records(records)
    if metadata.empty:
        return pd.DataFrame(
            columns=[
                "unit_id",
                "primary_parent_id",
                "root_unit_id",
                "depth_from_root",
                "collapse_level",
                "n_children",
                "n_descendants",
                "is_compound",
                "compound_unit_id",
                "compound_bubble_id",
                "in_compound_bubble",
                "compound_bubble_role",
                "unit_node_ids",
                "unit_node_count",
            ]
        )
    metadata = metadata.sort_values("unit_id").reset_index(drop=True)
    for column in (
        "primary_parent_id",
        "root_unit_id",
        "depth_from_root",
        "collapse_level",
        "n_children",
        "n_descendants",
        "compound_unit_id",
        "compound_bubble_id",
        "unit_node_count",
    ):
        metadata[column] = metadata[column].astype("Int64")
    metadata["is_compound"] = metadata["is_compound"].astype(bool)
    metadata["in_compound_bubble"] = metadata["in_compound_bubble"].astype(bool)
    return metadata


def build_summary_frame(
    units: list[StructuralUnit],
    graph: nx.MultiDiGraph | None = None,
) -> pd.DataFrame:
    annotate_unit_context(units, graph=graph)
    records = [unit.to_record() for unit in units]
    if not records:
        return pd.DataFrame(
            columns=[
                "unit_id",
                "bifurcation",
                "confluence",
                "n_paths",
                "class",
                "min_path_length",
                "max_path_length",
                "unit_node_ids",
                "unit_node_count",
                "internal_bifurcations",
                "internal_confluences",
                "children",
                "parents",
                "primary_parent_id",
                "root_unit_id",
                "depth_from_root",
                "collapse_level",
                "n_children",
                "n_descendants",
                "is_compound",
                "path_cutoff_used",
                "path_enumeration_truncated",
                "compound_unit_id",
                "compound_bubble_id",
                "in_compound_bubble",
                "compound_bubble_role",
            ]
        )
    summary = pd.DataFrame.from_records(records).sort_values(["bifurcation", "confluence", "unit_id"]).reset_index(drop=True)
    for column in (
        "primary_parent_id",
        "root_unit_id",
        "depth_from_root",
        "collapse_level",
        "n_children",
        "n_descendants",
        "compound_unit_id",
        "compound_bubble_id",
        "unit_node_count",
    ):
        summary[column] = summary[column].astype("Int64")
    summary["is_compound"] = summary["is_compound"].astype(bool)
    summary["in_compound_bubble"] = summary["in_compound_bubble"].astype(bool)
    return summary


def build_hierarchy_forest(
    units: list[StructuralUnit],
    graph: nx.MultiDiGraph | None = None,
) -> list[dict[str, Any]]:
    annotate_unit_context(units, graph=graph)
    units_by_id = {unit.unit_id: unit for unit in units}
    primary_children: dict[int, list[int]] = {unit.unit_id: [] for unit in units}
    for unit in units:
        if unit.primary_parent_id is None:
            continue
        primary_children[unit.primary_parent_id].append(unit.unit_id)
    for child_ids in primary_children.values():
        child_ids.sort()

    def recurse(unit_id: int) -> dict[str, Any]:
        unit = units_by_id[unit_id]
        return {
            "unit_id": unit.unit_id,
            "bifurcation": unit.bifurcation,
            "confluence": unit.confluence,
            "n_paths": unit.n_paths,
            "class": unit.unit_class,
            "node_ids": sorted(unit.node_set),
            "compound_bubble_id": unit.compound_bubble_id,
            "compound_bubble_role": unit.compound_bubble_role,
            "children": [recurse(child_id) for child_id in primary_children[unit_id]],
        }

    root_ids = sorted(unit.unit_id for unit in units if unit.primary_parent_id is None)
    return [recurse(unit_id) for unit_id in root_ids]


def analyze_network(
    links_path: str | Path,
    nodes_path: str | Path,
    *,
    max_path_cutoff: int = 100,
    max_paths: int = 5000,
    debug: bool = False,
) -> tuple[pd.DataFrame, list[StructuralUnit], list[dict[str, Any]]]:
    links, nodes = load_network(links_path, nodes_path)
    graph = build_graph(links, nodes)
    units = build_structural_units(
        graph,
        max_path_cutoff=max_path_cutoff,
        max_paths=max_paths,
        debug=debug,
    )
    summary = build_summary_frame(units, graph=graph)
    hierarchy = build_hierarchy_forest(units, graph=graph)
    return summary, units, hierarchy


def write_outputs(
    output_dir: str | Path,
    summary: pd.DataFrame,
    units: list[StructuralUnit],
    hierarchy: list[dict[str, Any]],
) -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    summary.to_csv(output_path / "unit_summary.csv", index=False)

    units_payload = [unit.to_json_dict() for unit in units]
    with (output_path / "units.json").open("w", encoding="utf-8") as handle:
        json.dump(units_payload, handle, indent=2)

    with (output_path / "hierarchy.json").open("w", encoding="utf-8") as handle:
        json.dump(hierarchy, handle, indent=2)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Identify bifurcation-confluence structural units from links and nodes GeoPackages "
            "and build parent-child relationships for nested units."
        )
    )
    parser.add_argument("--links", required=True, help="Path to the links GeoPackage.")
    parser.add_argument("--nodes", required=True, help="Path to the nodes GeoPackage.")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Optional directory for unit_summary.csv, units.json, and hierarchy.json.",
    )
    parser.add_argument(
        "--max-path-cutoff",
        type=int,
        default=100,
        help="Maximum edge-count cutoff for simple path enumeration.",
    )
    parser.add_argument(
        "--max-paths",
        type=int,
        default=5000,
        help="Maximum number of simple paths to enumerate per unit before truncating.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print per-bifurcation debug information during extraction.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    summary, units, hierarchy = analyze_network(
        args.links,
        args.nodes,
        max_path_cutoff=args.max_path_cutoff,
        max_paths=args.max_paths,
        debug=args.debug,
    )

    print(summary.to_string(index=False))
    print()
    print(json.dumps(hierarchy, indent=2))

    if args.output_dir:
        write_outputs(args.output_dir, summary, units, hierarchy)
        print()
        print(f"Wrote outputs to {Path(args.output_dir).resolve()}")


if __name__ == "__main__":
    main()
