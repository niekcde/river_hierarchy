from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import geopandas as gpd
import networkx as nx
import pandas as pd


def infer_network_name(path: str | Path) -> str:
    stem = Path(path).stem
    for suffix in ("_links", "_nodes"):
        if stem.endswith(suffix):
            return stem[: -len(suffix)]
    return stem


def parse_id_nodes(value: Any) -> tuple[int, int]:
    if value is None or pd.isna(value):
        raise ValueError("Encountered missing id_nodes value.")
    parts = [part.strip() for part in str(value).replace("[", "").replace("]", "").split(",") if part.strip()]
    if len(parts) != 2:
        raise ValueError(f"Expected exactly two node ids in id_nodes, got {value!r}.")
    return int(parts[0]), int(parts[1])


def as_bool(value: Any) -> bool:
    if value is None or pd.isna(value):
        return False
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "t", "yes", "y"}
    return bool(value)


def load_network(
    links_path: str | Path,
    nodes_path: str | Path,
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    links = gpd.read_file(links_path)
    nodes = gpd.read_file(nodes_path)
    return links, nodes


def edge_endpoints_from_row(row: pd.Series, *, prefer_explicit_direction: bool = True) -> tuple[int, int]:
    if prefer_explicit_direction and {"id_us_node", "id_ds_node"}.issubset(row.index):
        us = row.get("id_us_node")
        ds = row.get("id_ds_node")
        if us is not None and ds is not None and not pd.isna(us) and not pd.isna(ds):
            return int(us), int(ds)
    return parse_id_nodes(row["id_nodes"])


def build_directed_graph(
    links: gpd.GeoDataFrame,
    nodes: gpd.GeoDataFrame,
    *,
    prefer_explicit_direction: bool = True,
) -> nx.MultiDiGraph:
    graph = nx.MultiDiGraph()

    for row in nodes.itertuples(index=False):
        attrs = row._asdict()
        node_id = int(attrs["id_node"])
        graph.add_node(node_id, **attrs)

    for _, row in links.iterrows():
        attrs = row.to_dict()
        link_id = int(attrs["id_link"])
        upstream, downstream = edge_endpoints_from_row(row, prefer_explicit_direction=prefer_explicit_direction)
        graph.add_edge(upstream, downstream, key=link_id, **attrs)

        if upstream not in graph:
            graph.add_node(upstream, id_node=upstream)
        if downstream not in graph:
            graph.add_node(downstream, id_node=downstream)

    return graph


def build_simple_graph(graph: nx.MultiDiGraph) -> nx.DiGraph:
    simple_graph = nx.DiGraph()
    simple_graph.add_nodes_from(graph.nodes(data=True))
    for upstream, downstream, attrs in graph.edges(data=True):
        if simple_graph.has_edge(upstream, downstream):
            simple_graph[upstream][downstream]["edge_count"] += 1
            if "id_link" in attrs:
                simple_graph[upstream][downstream].setdefault("id_links", []).append(int(attrs["id_link"]))
        else:
            simple_graph.add_edge(upstream, downstream, edge_count=1, id_links=[int(attrs["id_link"])])
    return simple_graph


def build_degree_frame(graph: nx.MultiDiGraph) -> pd.DataFrame:
    simple_graph = build_simple_graph(graph)
    records = []
    for node_id, attrs in graph.nodes(data=True):
        in_degree = int(graph.in_degree(node_id))
        out_degree = int(graph.out_degree(node_id))
        records.append(
            {
                "node_id": int(node_id),
                "in_degree": in_degree,
                "out_degree": out_degree,
                "total_degree": in_degree + out_degree,
                "n_unique_predecessors": len(set(graph.predecessors(node_id))),
                "n_unique_successors": len(set(graph.successors(node_id))),
                "simple_in_degree": int(simple_graph.in_degree(node_id)),
                "simple_out_degree": int(simple_graph.out_degree(node_id)),
                "is_source": in_degree == 0 and out_degree > 0,
                "is_sink": out_degree == 0 and in_degree > 0,
                "is_isolated": in_degree == 0 and out_degree == 0,
                "flag_is_inlet": as_bool(attrs.get("is_inlet")),
                "flag_is_outlet": as_bool(attrs.get("is_outlet")),
            }
        )
    frame = pd.DataFrame.from_records(records).sort_values("node_id").reset_index(drop=True)
    return frame


@dataclass
class ValidationReport:
    n_nodes: int
    n_edges: int
    n_weak_components: int
    weak_components: list[list[int]] = field(default_factory=list)
    source_nodes: list[int] = field(default_factory=list)
    sink_nodes: list[int] = field(default_factory=list)
    isolated_nodes: list[int] = field(default_factory=list)
    flagged_inlets: list[int] = field(default_factory=list)
    flagged_outlets: list[int] = field(default_factory=list)
    issues: list[str] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        return not self.issues

    def to_dict(self) -> dict[str, Any]:
        return {
            "is_valid": self.is_valid,
            "n_nodes": self.n_nodes,
            "n_edges": self.n_edges,
            "n_weak_components": self.n_weak_components,
            "weak_components": self.weak_components,
            "source_nodes": self.source_nodes,
            "sink_nodes": self.sink_nodes,
            "isolated_nodes": self.isolated_nodes,
            "flagged_inlets": self.flagged_inlets,
            "flagged_outlets": self.flagged_outlets,
            "issues": self.issues,
        }


def validate_single_inlet_single_outlet(
    graph: nx.MultiDiGraph,
    *,
    require_one_component: bool = True,
    require_flag_match: bool = True,
) -> ValidationReport:
    degree_frame = build_degree_frame(graph)
    weak_components = [sorted(int(node) for node in component) for component in nx.weakly_connected_components(graph)]
    source_nodes = sorted(int(node) for node in degree_frame.loc[degree_frame["is_source"], "node_id"])
    sink_nodes = sorted(int(node) for node in degree_frame.loc[degree_frame["is_sink"], "node_id"])
    isolated_nodes = sorted(int(node) for node in degree_frame.loc[degree_frame["is_isolated"], "node_id"])
    flagged_inlets = sorted(int(node) for node in degree_frame.loc[degree_frame["flag_is_inlet"], "node_id"])
    flagged_outlets = sorted(int(node) for node in degree_frame.loc[degree_frame["flag_is_outlet"], "node_id"])

    issues: list[str] = []
    if require_one_component and len(weak_components) != 1:
        issues.append(f"Expected one weakly connected component, found {len(weak_components)}.")
    if isolated_nodes:
        issues.append(f"Found isolated node(s) with zero in-degree and zero out-degree: {isolated_nodes}.")
    if len(source_nodes) != 1:
        issues.append(
            "Expected exactly one inlet/source node with in_degree == 0 and out_degree > 0, "
            f"found {len(source_nodes)}: {source_nodes}."
        )
    if len(sink_nodes) != 1:
        issues.append(
            "Expected exactly one outlet/sink node with out_degree == 0 and in_degree > 0, "
            f"found {len(sink_nodes)}: {sink_nodes}."
        )

    if require_flag_match:
        if len(flagged_inlets) != 1:
            issues.append(f"Expected exactly one node flagged as inlet, found {len(flagged_inlets)}: {flagged_inlets}.")
        if len(flagged_outlets) != 1:
            issues.append(f"Expected exactly one node flagged as outlet, found {len(flagged_outlets)}: {flagged_outlets}.")
        if len(flagged_inlets) == 1 and len(source_nodes) == 1 and flagged_inlets[0] != source_nodes[0]:
            issues.append(
                f"Flagged inlet node {flagged_inlets[0]} does not match the topological source node {source_nodes[0]}."
            )
        if len(flagged_outlets) == 1 and len(sink_nodes) == 1 and flagged_outlets[0] != sink_nodes[0]:
            issues.append(
                f"Flagged outlet node {flagged_outlets[0]} does not match the topological sink node {sink_nodes[0]}."
            )

    return ValidationReport(
        n_nodes=graph.number_of_nodes(),
        n_edges=graph.number_of_edges(),
        n_weak_components=len(weak_components),
        weak_components=weak_components,
        source_nodes=source_nodes,
        sink_nodes=sink_nodes,
        isolated_nodes=isolated_nodes,
        flagged_inlets=flagged_inlets,
        flagged_outlets=flagged_outlets,
        issues=issues,
    )


def analyze_reviewed_network(
    links_path: str | Path,
    nodes_path: str | Path,
    *,
    prefer_explicit_direction: bool = True,
) -> tuple[nx.MultiDiGraph, pd.DataFrame, ValidationReport]:
    links, nodes = load_network(links_path, nodes_path)
    graph = build_directed_graph(links, nodes, prefer_explicit_direction=prefer_explicit_direction)
    degree_frame = build_degree_frame(graph)
    report = validate_single_inlet_single_outlet(graph)
    return graph, degree_frame, report


def default_output_dir_for_links(links_path: str | Path) -> Path:
    return Path(__file__).resolve().parent / "outputs" / infer_network_name(links_path)


def write_outputs(
    output_dir: str | Path,
    degree_frame: pd.DataFrame,
    report: ValidationReport,
) -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    degree_frame.to_csv(output_path / "node_degree_summary.csv", index=False)
    with (output_path / "validation_report.json").open("w", encoding="utf-8") as handle:
        json.dump(report.to_dict(), handle, indent=2)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build a directed graph from reviewed links/nodes files and validate that the "
            "network has exactly one inlet/source, exactly one outlet/sink, and no extra local minima/maxima."
        )
    )
    parser.add_argument("links_gpkg", help="Reviewed links GeoPackage.")
    parser.add_argument("nodes_gpkg", help="Reviewed nodes GeoPackage.")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Optional directory for node_degree_summary.csv and validation_report.json.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    _, degree_frame, report = analyze_reviewed_network(args.links_gpkg, args.nodes_gpkg)

    print(degree_frame.to_string(index=False))
    print()
    print(json.dumps(report.to_dict(), indent=2))

    if args.output_dir is not None:
        output_dir = args.output_dir
    else:
        output_dir = default_output_dir_for_links(args.links_gpkg)
    write_outputs(output_dir, degree_frame, report)
    print()
    print(f"Wrote outputs to {Path(output_dir).resolve()}")


if __name__ == "__main__":
    main()
