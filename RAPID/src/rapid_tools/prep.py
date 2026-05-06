from __future__ import annotations

import csv
import json
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path

import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd
from shapely.geometry import LineString
from shapely.ops import substring

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
    use_celerity_capping: bool = False
    min_celerity_mps: float = 0.28
    max_celerity_mps: float = 1.524
    target_subreach_length_m: float | None = None
    min_slope: float = 1e-6
    preferred_length_field: str = "len"
    include_base_state: bool = True
    strict_sword: bool = True


def _compute_subreach_count(link_length_m: float, target_length_m: float | None) -> int:
    if target_length_m is None:
        return 1
    if target_length_m <= 0:
        raise ValueError("target_subreach_length_m must be positive when provided.")
    return max(1, int(round(float(link_length_m) / float(target_length_m))))


def _extract_subreach_geometry(geometry, start_distance: float, end_distance: float):
    if start_distance <= 0.0 and end_distance >= float(geometry.length):
        return geometry
    try:
        segment = substring(geometry, start_distance, end_distance)
    except Exception:
        start_point = geometry.interpolate(start_distance)
        end_point = geometry.interpolate(end_distance)
        return LineString([start_point, end_point])
    if segment.geom_type == "LineString":
        return segment
    start_point = geometry.interpolate(start_distance)
    end_point = geometry.interpolate(end_distance)
    return LineString([start_point, end_point])


def split_links_into_subreaches(
    prepared_links: gpd.GeoDataFrame,
    nodes: gpd.GeoDataFrame,
    *,
    target_length_m: float | None,
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    rapid_nodes = nodes.copy()
    rapid_nodes["id_node"] = pd.to_numeric(rapid_nodes["id_node"], errors="coerce").astype(int)
    rapid_nodes["rapid_node_source"] = "original"
    rapid_nodes["parent_node_id"] = rapid_nodes["id_node"].astype(int)
    rapid_nodes["rapid_node_split_from_link_id"] = pd.Series([pd.NA] * len(rapid_nodes), dtype="Int64")
    rapid_nodes["rapid_node_subreach_boundary_index"] = pd.Series([pd.NA] * len(rapid_nodes), dtype="Int64")

    if target_length_m is None:
        rapid_links = prepared_links.copy()
        rapid_links["reach_id"] = pd.to_numeric(rapid_links["id_link"], errors="coerce").astype(int)
        rapid_links["parent_link_id"] = pd.to_numeric(rapid_links["id_link"], errors="coerce").astype(int)
        rapid_links["parent_link_length_m"] = rapid_links["link_length_m"].astype(float)
        rapid_links["subreach_index"] = 1
        rapid_links["subreach_count"] = 1
        rapid_links["subreach_length_fraction"] = 1.0
        rapid_links["rapid_link_split"] = False
        return rapid_links, rapid_nodes

    link_sort_order = pd.to_numeric(prepared_links["id_link"], errors="coerce")
    rapid_links_records: list[pd.Series] = []
    virtual_node_records: list[dict[str, object]] = []
    next_virtual_node_id = int(rapid_nodes["id_node"].max()) + 1 if not rapid_nodes.empty else 1
    next_reach_id = int(pd.to_numeric(prepared_links["id_link"], errors="coerce").max()) + 1 if not prepared_links.empty else 1

    for _, row in prepared_links.assign(_rapid_sort_id=link_sort_order).sort_values("_rapid_sort_id").drop(columns="_rapid_sort_id").iterrows():
        parent_link_id = int(row["id_link"])
        link_length_m = float(row["link_length_m"])
        subreach_count = _compute_subreach_count(link_length_m, target_length_m)
        node_sequence = [int(row["id_us_node"])]
        geometry = row.geometry
        geom_length = float(geometry.length) if geometry is not None else 0.0

        for boundary_index in range(1, subreach_count):
            distance = geom_length * (boundary_index / subreach_count)
            point = geometry.interpolate(distance)
            virtual_node_id = next_virtual_node_id
            next_virtual_node_id += 1
            node_sequence.append(virtual_node_id)

            virtual_record = {column: pd.NA for column in rapid_nodes.columns if column != rapid_nodes.geometry.name}
            virtual_record["id_node"] = virtual_node_id
            if "is_inlet" in rapid_nodes.columns:
                virtual_record["is_inlet"] = False
            if "is_outlet" in rapid_nodes.columns:
                virtual_record["is_outlet"] = False
            virtual_record["rapid_node_source"] = "subreach_virtual"
            virtual_record["parent_node_id"] = pd.NA
            virtual_record["rapid_node_split_from_link_id"] = parent_link_id
            virtual_record["rapid_node_subreach_boundary_index"] = boundary_index
            virtual_record[rapid_nodes.geometry.name] = point
            virtual_node_records.append(virtual_record)

        node_sequence.append(int(row["id_ds_node"]))

        if subreach_count == 1:
            reach_ids = [parent_link_id]
        else:
            reach_ids = list(range(next_reach_id, next_reach_id + subreach_count))
            next_reach_id += subreach_count

        for subreach_index in range(subreach_count):
            child = row.copy()
            child["reach_id"] = int(reach_ids[subreach_index])
            child["parent_link_id"] = parent_link_id
            child["parent_link_length_m"] = link_length_m
            child["subreach_index"] = subreach_index + 1
            child["subreach_count"] = subreach_count
            child["subreach_length_fraction"] = 1.0 / subreach_count
            child["rapid_link_split"] = bool(subreach_count > 1)
            child["id_us_node"] = int(node_sequence[subreach_index])
            child["id_ds_node"] = int(node_sequence[subreach_index + 1])
            child["link_length_m"] = link_length_m / subreach_count
            if "is_inlet" in child.index:
                child["is_inlet"] = bool(row.get("is_inlet", False)) and subreach_index == 0
            if "is_outlet" in child.index:
                child["is_outlet"] = bool(row.get("is_outlet", False)) and subreach_index == (subreach_count - 1)
            start_distance = geom_length * (subreach_index / subreach_count)
            end_distance = geom_length * ((subreach_index + 1) / subreach_count)
            child.geometry = _extract_subreach_geometry(geometry, start_distance, end_distance)
            rapid_links_records.append(child)

    if virtual_node_records:
        virtual_nodes = gpd.GeoDataFrame(virtual_node_records, geometry=rapid_nodes.geometry.name, crs=rapid_nodes.crs)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            rapid_nodes = pd.concat([rapid_nodes, virtual_nodes], ignore_index=True)
        rapid_nodes = gpd.GeoDataFrame(rapid_nodes, geometry=nodes.geometry.name, crs=nodes.crs)

    rapid_links = gpd.GeoDataFrame(rapid_links_records, geometry=prepared_links.geometry.name, crs=prepared_links.crs)
    return rapid_links, rapid_nodes


def create_conn_file(graph: nx.MultiDiGraph, output_dir: Path) -> tuple[Path, list[int]]:
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
        return path, []
    dn_cols = sorted([c for c in frame.columns if c.startswith("rch_id_dn_")], key=lambda c: int(c.split("_")[3]))
    up_cols = sorted([c for c in frame.columns if c.startswith("rch_id_up_")], key=lambda c: int(c.split("_")[3]))
    frame = frame[["reach_id", "n_rch_dn"] + dn_cols + ["n_rch_up"] + up_cols].fillna(0)
    path = output_dir / "conn.csv"
    frame.to_csv(path, header=False, index=False)
    return path, frame["reach_id"].astype(int).tolist()


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


def order_prepared_links_for_total_reach_order(
    prepared_links: pd.DataFrame,
    reach_order: list[int],
) -> pd.DataFrame:
    if not reach_order:
        return prepared_links.iloc[0:0].copy()
    by_reach = prepared_links.copy()
    by_reach["reach_id"] = pd.to_numeric(by_reach["reach_id"], errors="coerce").astype(int)
    by_reach = by_reach.set_index("reach_id", drop=False)
    missing = [reach_id for reach_id in reach_order if reach_id not in by_reach.index]
    if missing:
        raise ValueError(
            "Could not align RAPID prepared links to the connectivity order. "
            f"Missing reach IDs: {missing[:10]}"
        )
    return by_reach.loc[reach_order].reset_index(drop=True)


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
    prepared_links["link_length_m"] = pd.to_numeric(prepared_links["link_length_m"], errors="coerce").astype(float)

    rapid_links, rapid_nodes = split_links_into_subreaches(
        prepared_links,
        nodes,
        target_length_m=prep_config.target_subreach_length_m,
    )
    rapid_links["centroid_x"] = rapid_links.geometry.centroid.x.astype(float)
    rapid_links["centroid_y"] = rapid_links.geometry.centroid.y.astype(float)
    rapid_links = compute_k_values(
        rapid_links,
        config=KValueConfig(
            width_field=prep_config.width_field,
            x_value=prep_config.x_value,
            kb_value=prep_config.kb_value,
            n_manning=prep_config.n_manning,
            min_width=prep_config.min_width,
            use_celerity_capping=prep_config.use_celerity_capping,
            min_celerity_mps=prep_config.min_celerity_mps,
            max_celerity_mps=prep_config.max_celerity_mps,
        ),
    )

    graph = build_rapid_graph(rapid_links, rapid_nodes)
    conn_path, total_reach_order = create_conn_file(graph, context.rapid_prep_dir)
    rapid_links_total_order = order_prepared_links_for_total_reach_order(rapid_links, total_reach_order)
    riv_path, reach_order = create_riv_file(graph, context.rapid_prep_dir)
    rat_srt_path = compute_reach_ratios(graph, context.rapid_prep_dir, use_widths=True)
    kfc_path, xfc_path, coords_path = write_routing_parameter_files(rapid_links_total_order, context.rapid_prep_dir)

    rapid_links_path = context.rapid_prep_dir / "rapid_link_attributes.csv"
    rapid_links.drop(columns=rapid_links.geometry.name).to_csv(rapid_links_path, index=False)
    rapid_nodes_path = context.rapid_prep_dir / "rapid_node_attributes.csv"
    rapid_nodes.drop(columns=rapid_nodes.geometry.name).to_csv(rapid_nodes_path, index=False)

    forcing_table_path = None
    inflow_path = None
    forcing_dt_seconds = None
    routing_dt_seconds = None
    if forcing_path is not None:
        forcing = load_forcing_table(forcing_path, config=forcing_config)
        forcing_dt_seconds = infer_forcing_dt_seconds(forcing)
        routing_dt_seconds = compute_routing_dt_seconds(
            rapid_links["rapid_k"],
            x_value=prep_config.x_value,
            forcing_dt_seconds=forcing_dt_seconds,
        )
        forcing_table_path = context.rapid_prep_dir / "forcing_normalized.csv"
        forcing.to_csv(forcing_table_path, index=False)
        inflow_path = write_inflow_netcdf(rapid_links_total_order, forcing, context.rapid_prep_dir / "inflow.nc")

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
            "n_source_links": int(len(prepared_links)),
            "n_links": int(len(rapid_links)),
            "n_nodes": int(len(rapid_nodes)),
            "n_virtual_nodes": int(rapid_nodes["rapid_node_source"].eq("subreach_virtual").sum()) if "rapid_node_source" in rapid_nodes.columns else 0,
            "n_split_parent_links": int(rapid_links.loc[rapid_links["rapid_link_split"], "parent_link_id"].nunique()) if "rapid_link_split" in rapid_links.columns else 0,
            "n_inlet_links": int(rapid_links["is_inlet"].fillna(False).sum()) if "is_inlet" in rapid_links.columns else 0,
            "n_slope_adjusted": int(rapid_links["slope_adjusted"].fillna(False).sum()),
            "n_width_adjusted": int(rapid_links["rapid_width_adjusted"].fillna(False).sum()),
        },
        "routing": {
            "reach_order": reach_order,
            "forcing_dt_seconds": forcing_dt_seconds,
            "routing_dt_seconds": routing_dt_seconds,
        },
        "diagnostics": {
            "link_multiplier": float(len(rapid_links) / len(prepared_links)) if len(prepared_links) else float("nan"),
            "n_celerity_capped": int(rapid_links["rapid_celerity_capped"].fillna(False).sum()) if "rapid_celerity_capped" in rapid_links.columns else 0,
            "pct_celerity_capped": float(rapid_links["rapid_celerity_capped"].fillna(False).astype(bool).mean()) if "rapid_celerity_capped" in rapid_links.columns and len(rapid_links) else float("nan"),
            "min_link_length_m": float(rapid_links["link_length_m"].min()) if len(rapid_links) else float("nan"),
            "max_link_length_m": float(rapid_links["link_length_m"].max()) if len(rapid_links) else float("nan"),
            "rapid_k_min": float(rapid_links["rapid_k"].min()) if "rapid_k" in rapid_links.columns and len(rapid_links) else float("nan"),
            "rapid_k_max": float(rapid_links["rapid_k"].max()) if "rapid_k" in rapid_links.columns and len(rapid_links) else float("nan"),
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
        "n_source_links": int(len(prepared_links)),
        "n_links": int(len(rapid_links)),
        "link_multiplier": float(len(rapid_links) / len(prepared_links)) if len(prepared_links) else float("nan"),
        "n_split_parent_links": int(rapid_links.loc[rapid_links["rapid_link_split"], "parent_link_id"].nunique()) if "rapid_link_split" in rapid_links.columns else 0,
        "pct_split_parent_links": (
            float(rapid_links.loc[rapid_links["rapid_link_split"], "parent_link_id"].nunique() / len(prepared_links))
            if "rapid_link_split" in rapid_links.columns and len(prepared_links)
            else 0.0
        ),
        "n_celerity_capped": int(rapid_links["rapid_celerity_capped"].fillna(False).sum()) if "rapid_celerity_capped" in rapid_links.columns else 0,
        "pct_celerity_capped": float(rapid_links["rapid_celerity_capped"].fillna(False).astype(bool).mean()) if "rapid_celerity_capped" in rapid_links.columns and len(rapid_links) else float("nan"),
        "min_link_length_m": float(rapid_links["link_length_m"].min()) if len(rapid_links) else float("nan"),
        "max_link_length_m": float(rapid_links["link_length_m"].max()) if len(rapid_links) else float("nan"),
        "rapid_k_min": float(rapid_links["rapid_k"].min()) if "rapid_k" in rapid_links.columns and len(rapid_links) else float("nan"),
        "rapid_k_max": float(rapid_links["rapid_k"].max()) if "rapid_k" in rapid_links.columns and len(rapid_links) else float("nan"),
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
