from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd

from .forcing import ForcingConfig, infer_forcing_dt_seconds, load_forcing_table, write_inflow_netcdf
from .k_values import KValueConfig, compute_k_values, compute_routing_dt_seconds
from .registry import RapidStateContext, iter_preparable_states, load_state_registry
from .slope import SlopeConfig, compute_link_slopes


@dataclass(frozen=True, slots=True)
class RapidPrepConfig:
    width_field: str = "wid_adj_wet"
    x_value: float = 0.1
    kb_value: float = 20.0
    n_manning: float = 0.35
    min_width: float = 1.0
    min_slope: float = 1e-6
    preferred_length_field: str = "len"
    include_base_state: bool = True
    strict_sword: bool = True


def create_conn_file(graph: nx.MultiDiGraph, output_dir: Path) -> Path:
    edge_list = list(graph.edges(data=True))
    node_to_edges_dn: dict[object, list[int]] = {}
    node_to_edges_up: dict[object, list[int]] = {}
    for u, v, data in edge_list:
        eid = int(data["reach_id"])
        node_to_edges_dn.setdefault(v, []).append(eid)
        node_to_edges_up.setdefault(u, []).append(eid)

    records: list[dict[str, object]] = []
    for u, v, data in edge_list:
        eid = int(data["reach_id"])
        downstream_edges = node_to_edges_dn.get(v, [])
        upstream_edges = node_to_edges_up.get(u, [])
        row: dict[str, object] = {
            "reach_id": eid,
            "n_rch_dn": len(downstream_edges),
            "n_rch_up": len(upstream_edges),
        }
        for i, e_dn in enumerate(downstream_edges, start=1):
            row[f"rch_id_dn_{i}"] = e_dn
        for i, e_up in enumerate(upstream_edges, start=1):
            row[f"rch_id_up_{i}"] = e_up
        records.append(row)

    frame = pd.DataFrame(records)
    if frame.empty:
        path = output_dir / "conn.csv"
        path.write_text("")
        return path
    dn_cols = sorted([c for c in frame.columns if c.startswith("rch_id_dn_")], key=lambda c: int(c.split("_")[3]))
    up_cols = sorted([c for c in frame.columns if c.startswith("rch_id_up_")], key=lambda c: int(c.split("_")[3]))
    frame = frame[["reach_id", "n_rch_dn"] + dn_cols + ["n_rch_up"] + up_cols].fillna(0)
    path = output_dir / "conn.csv"
    frame.to_csv(path, header=False, index=False)
    return path


def create_riv_file(graph: nx.MultiDiGraph, output_dir: Path) -> tuple[Path, list[int]]:
    node_order = list(nx.topological_sort(nx.DiGraph(graph)))
    reach_ids: list[int] = []
    for u in node_order:
        for v in graph.successors(u):
            for key in graph[u][v]:
                reach_ids.append(int(graph[u][v][key]["reach_id"]))

    path = output_dir / "riv.csv"
    pd.DataFrame({"reach_id": reach_ids}).to_csv(path, header=False, index=False)
    return path, reach_ids


def write_reach_ratios_csv(ratio_dict: dict[int, list[list[float]]], output_dir: Path) -> Path:
    rat_path = output_dir / "rat.csv"
    max_cols = max((len(values) for values in ratio_dict.values()), default=0)
    with rat_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        if max_cols == 0:
            return rat_path
        for reach_id, pairs in ratio_dict.items():
            row: list[object] = [reach_id]
            for down_id, ratio in pairs:
                row.append([down_id, float(ratio)])
            while len(row) < max_cols + 1:
                row.append([0, 0])
            writer.writerow(row)
    return rat_path


def compute_reach_ratios(graph: nx.MultiDiGraph, output_dir: Path, *, use_widths: bool = True) -> Path:
    reach_to_uv: dict[int, tuple[object, object]] = {}
    width_of: dict[int, float] = {}
    for u, v, data in graph.edges(data=True):
        rid = int(data["reach_id"])
        reach_to_uv[rid] = (u, v)
        width_of[rid] = float(data.get("width", 1.0))

    reach_graph = nx.DiGraph()
    reach_graph.add_nodes_from(reach_to_uv.keys())
    for rid, (_u, v) in reach_to_uv.items():
        for _, _v2, downstream_data in graph.out_edges(v, data=True):
            reach_graph.add_edge(rid, int(downstream_data["reach_id"]))
    order = list(nx.topological_sort(reach_graph))

    max_down = max((len(list(reach_graph.successors(r))) for r in order), default=1)
    ratios: dict[int, list[list[float]]] = {}
    for rid in order:
        downstream = list(reach_graph.successors(rid))
        if not downstream:
            pairs = [[0, 0]]
        elif len(downstream) == 1:
            pairs = [[downstream[0], 1]]
        elif not use_widths:
            pairs = [[downstream_id, 1] for downstream_id in downstream]
        else:
            ds_widths = [width_of[downstream_id] for downstream_id in downstream]
            total = float(sum(ds_widths))
            if total <= 0:
                values = [1.0 / len(downstream)] * len(downstream)
            else:
                values = [width / total for width in ds_widths]
            sum_values = float(sum(values))
            values = [value / sum_values for value in values]
            pairs = [[downstream_id, round(value, 6)] for downstream_id, value in zip(downstream, values)]

        while len(pairs) < max_down:
            pairs.append([0, 0])
        ratios[rid] = pairs

    rat_path = write_reach_ratios_csv(ratios, output_dir)
    rat_srt_path = output_dir / "rat_srt.csv"
    if not ratios:
        rat_srt_path.write_text("")
        return rat_srt_path

    conn_frame = pd.read_csv(output_dir / "conn.csv", header=None)
    ratio_frame = pd.read_csv(rat_path, header=None)
    order = conn_frame[0].drop_duplicates()
    ratio_sorted = ratio_frame.set_index(0).loc[order].reset_index()
    ratio_sorted.to_csv(rat_srt_path, header=False, index=False)
    return rat_srt_path


def write_routing_parameter_files(prepared_links: pd.DataFrame, output_dir: Path) -> tuple[Path, Path, Path]:
    kfc_path = output_dir / "kfc.csv"
    xfc_path = output_dir / "xfc.csv"
    coords_path = output_dir / "coords.csv"

    prepared_links[["rapid_k"]].to_csv(kfc_path, header=False, index=False)
    prepared_links[["rapid_x"]].to_csv(xfc_path, header=False, index=False)
    prepared_links[["reach_id", "centroid_x", "centroid_y"]].to_csv(coords_path, header=False, index=False)
    return kfc_path, xfc_path, coords_path


def build_rapid_graph(
    prepared_links: pd.DataFrame,
    nodes: gpd.GeoDataFrame,
) -> nx.MultiDiGraph:
    graph = nx.MultiDiGraph()
    node_lookup = nodes.set_index(pd.to_numeric(nodes["id_node"], errors="coerce").astype("Int64"))
    for node_id, row in node_lookup.iterrows():
        point = row.geometry
        graph.add_node(
            int(node_id),
            x=float(point.x),
            y=float(point.y),
            node_type="source" if bool(row.get("is_inlet", False)) else "internal",
        )

    outlet_nodes = set(pd.to_numeric(nodes.loc[nodes["is_outlet"].fillna(False), "id_node"], errors="coerce").dropna().astype(int).tolist())
    for row in prepared_links.itertuples(index=False):
        us = int(row.id_us_node)
        ds = int(row.id_ds_node)
        if ds in outlet_nodes:
            graph.nodes[ds]["node_type"] = "outlet"
        graph.add_edge(
            us,
            ds,
            key=str(int(row.reach_id)),
            reach_id=int(row.reach_id),
            width=float(row.rapid_width_m),
            length=float(row.link_length_m),
            geometry=row.geometry.wkt,
            slope_local=float(row.slope_used),
        )
    return graph


def prepare_state(
    context: RapidStateContext,
    *,
    forcing_path: str | Path | None = None,
    forcing_config: ForcingConfig | None = None,
    prep_config: RapidPrepConfig | None = None,
) -> dict[str, object]:
    prep_config = prep_config or RapidPrepConfig()
    forcing_config = forcing_config or ForcingConfig()
    context.rapid_prep_dir.mkdir(parents=True, exist_ok=True)

    links = gpd.read_file(context.link_widths_path)
    nodes = gpd.read_file(context.directed_nodes_path)

    if prep_config.strict_sword:
        missing = [column for column in ("sword_wse", "sword_wse_field", "sword_wse_fill_method") if column not in nodes.columns]
        if missing:
            raise ValueError(
                f"State {context.state_id} is missing required SWORD node columns for strict RAPID prep: {missing}"
            )

    slope_frame = compute_link_slopes(
        links,
        nodes,
        config=SlopeConfig(
            min_slope=prep_config.min_slope,
            preferred_length_field=prep_config.preferred_length_field,
        ),
    )
    prepared_links = links.merge(slope_frame, on=["id_link", "id_us_node", "id_ds_node"], how="left")
    prepared_links["reach_id"] = pd.to_numeric(prepared_links["id_link"], errors="coerce").astype(int)
    prepared_links["centroid_x"] = prepared_links.geometry.centroid.x.astype(float)
    prepared_links["centroid_y"] = prepared_links.geometry.centroid.y.astype(float)
    prepared_links = compute_k_values(
        prepared_links,
        config=KValueConfig(
            width_field=prep_config.width_field,
            x_value=prep_config.x_value,
            kb_value=prep_config.kb_value,
            n_manning=prep_config.n_manning,
            min_width=prep_config.min_width,
        ),
    )

    graph = build_rapid_graph(prepared_links, nodes)
    conn_path = create_conn_file(graph, context.rapid_prep_dir)
    riv_path, reach_order = create_riv_file(graph, context.rapid_prep_dir)
    rat_srt_path = compute_reach_ratios(graph, context.rapid_prep_dir, use_widths=True)
    kfc_path, xfc_path, coords_path = write_routing_parameter_files(prepared_links, context.rapid_prep_dir)

    forcing_table_path = None
    inflow_path = None
    forcing_dt_seconds = None
    routing_dt_seconds = None
    if forcing_path is not None:
        forcing = load_forcing_table(forcing_path, config=forcing_config)
        forcing_dt_seconds = infer_forcing_dt_seconds(forcing)
        routing_dt_seconds = compute_routing_dt_seconds(
            prepared_links["rapid_k"],
            x_value=prep_config.x_value,
            forcing_dt_seconds=forcing_dt_seconds,
        )
        forcing_table_path = context.rapid_prep_dir / "forcing_normalized.csv"
        forcing.to_csv(forcing_table_path, index=False)
        inflow_path = write_inflow_netcdf(prepared_links, forcing, context.rapid_prep_dir / "inflow.nc")

    rapid_links_path = context.rapid_prep_dir / "rapid_link_attributes.csv"
    prepared_links.drop(columns=prepared_links.geometry.name).to_csv(rapid_links_path, index=False)
    rapid_nodes_path = context.rapid_prep_dir / "rapid_node_attributes.csv"
    nodes.drop(columns=nodes.geometry.name).to_csv(rapid_nodes_path, index=False)

    manifest = {
        "state_id": context.state_id,
        "state_role": context.state_role,
        "prep_config": asdict(prep_config),
        "forcing_config": asdict(forcing_config) if forcing_path is not None else None,
        "paths": {
            "directed_links": str(context.directed_links_path),
            "directed_nodes": str(context.directed_nodes_path),
            "link_widths": str(context.link_widths_path),
            "conn_csv": str(conn_path),
            "riv_csv": str(riv_path),
            "rat_srt_csv": str(rat_srt_path),
            "kfc_csv": str(kfc_path),
            "xfc_csv": str(xfc_path),
            "coords_csv": str(coords_path),
            "rapid_link_attributes_csv": str(rapid_links_path),
            "rapid_node_attributes_csv": str(rapid_nodes_path),
            "forcing_normalized_csv": str(forcing_table_path) if forcing_table_path is not None else None,
            "inflow_nc": str(inflow_path) if inflow_path is not None else None,
        },
        "counts": {
            "n_links": int(len(prepared_links)),
            "n_nodes": int(len(nodes)),
            "n_inlet_links": int(prepared_links["is_inlet"].fillna(False).sum()) if "is_inlet" in prepared_links.columns else 0,
            "n_slope_adjusted": int(prepared_links["slope_adjusted"].fillna(False).sum()),
            "n_width_adjusted": int(prepared_links["rapid_width_adjusted"].fillna(False).sum()),
        },
        "routing": {
            "reach_order": reach_order,
            "forcing_dt_seconds": forcing_dt_seconds,
            "routing_dt_seconds": routing_dt_seconds,
        },
    }
    manifest_path = context.rapid_prep_dir / "rapid_prep_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))

    return {
        "state_id": context.state_id,
        "state_role": context.state_role,
        "rapid_prep_dir": str(context.rapid_prep_dir),
        "rapid_prep_manifest": str(manifest_path),
        "conn_csv": str(conn_path),
        "riv_csv": str(riv_path),
        "rat_srt_csv": str(rat_srt_path),
        "kfc_csv": str(kfc_path),
        "xfc_csv": str(xfc_path),
        "coords_csv": str(coords_path),
        "rapid_link_attributes_csv": str(rapid_links_path),
        "rapid_node_attributes_csv": str(rapid_nodes_path),
        "forcing_normalized_csv": str(forcing_table_path) if forcing_table_path is not None else "",
        "inflow_nc": str(inflow_path) if inflow_path is not None else "",
        "forcing_dt_seconds": forcing_dt_seconds if forcing_dt_seconds is not None else "",
        "routing_dt_seconds": routing_dt_seconds if routing_dt_seconds is not None else "",
        "status": "prepared",
    }


def prepare_experiment(
    experiment_dir: str | Path,
    *,
    forcing_path: str | Path | None = None,
    forcing_config: ForcingConfig | None = None,
    prep_config: RapidPrepConfig | None = None,
) -> pd.DataFrame:
    experiment_path = Path(experiment_dir).expanduser().resolve()
    registry = load_state_registry(experiment_path)
    prep_rows: list[dict[str, object]] = []

    for context in iter_preparable_states(
        registry,
        include_base_state=(prep_config.include_base_state if prep_config is not None else True),
    ):
        try:
            result = prepare_state(
                context,
                forcing_path=forcing_path,
                forcing_config=forcing_config,
                prep_config=prep_config,
            )
        except Exception as exc:  # pragma: no cover - exercised through registry rows
            result = {
                "state_id": context.state_id,
                "state_role": context.state_role,
                "rapid_prep_dir": str(context.rapid_prep_dir),
                "rapid_prep_manifest": "",
                "status": "failed",
                "error": str(exc),
            }
        prep_rows.append(result)

    prep_registry = pd.DataFrame(prep_rows)
    prep_registry_path = experiment_path / "rapid_prep_registry.csv"
    prep_registry.to_csv(prep_registry_path, index=False)
    manifest_path = experiment_path / "rapid_prep_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "experiment_dir": str(experiment_path),
                "forcing_path": str(Path(forcing_path).expanduser().resolve()) if forcing_path is not None else None,
                "states_total": int(len(prep_registry)),
                "states_prepared": int(prep_registry["status"].eq("prepared").sum()) if not prep_registry.empty else 0,
                "states_failed": int(prep_registry["status"].eq("failed").sum()) if not prep_registry.empty else 0,
            },
            indent=2,
        )
    )
    return prep_registry
