"""Shared RAPID input preparation helpers for multiple workflows."""

from __future__ import annotations

import csv
import json
import warnings
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd
import shapely
import shapely.wkt
from scipy.io import netcdf_file
from shapely import wkt
from shapely.geometry import LineString
from shapely.ops import substring

from .forcing import ForcingConfig, infer_forcing_dt_seconds, prepare_forcing_table, write_inflow_netcdf
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


def _normalize_dir_path(raw: str | Path) -> Path:
    return Path(raw).expanduser().resolve()


def _coerce_output_dir(
    output_dir: str | Path | None = None,
    *,
    directory: str | Path | None = None,
) -> tuple[Path, bool]:
    if output_dir is None and directory is None:
        raise TypeError("Either output_dir or directory must be provided.")
    if output_dir is not None and directory is not None:
        raise TypeError("Pass either output_dir or directory, not both.")
    legacy_call = directory is not None and output_dir is None
    raw = directory if legacy_call else output_dir
    path = _normalize_dir_path(raw)
    path.mkdir(parents=True, exist_ok=True)
    return path, legacy_call


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


def create_conn_file(
    graph: nx.MultiDiGraph,
    output_dir: str | Path | None = None,
    *,
    directory: str | Path | None = None,
) -> tuple[Path, list[int]] | None:
    path, legacy_call = _coerce_output_dir(output_dir, directory=directory)
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

    conn_path = path / "conn.csv"
    frame = pd.DataFrame(records)
    if frame.empty:
        conn_path.write_text("")
        return None if legacy_call else (conn_path, [])

    dn_cols = sorted([c for c in frame.columns if c.startswith("rch_id_dn_")], key=lambda c: int(c.split("_")[3]))
    up_cols = sorted([c for c in frame.columns if c.startswith("rch_id_up_")], key=lambda c: int(c.split("_")[3]))
    frame = frame[["reach_id", "n_rch_dn"] + dn_cols + ["n_rch_up"] + up_cols].fillna(0)
    frame.to_csv(conn_path, header=False, index=False)

    if legacy_call:
        return None
    return conn_path, frame["reach_id"].astype(int).tolist()


def create_riv_file(
    graph: nx.MultiDiGraph,
    output_dir: str | Path | None = None,
    *,
    directory: str | Path | None = None,
) -> tuple[Path, list[int]] | list[int]:
    path, legacy_call = _coerce_output_dir(output_dir, directory=directory)
    node_order = list(nx.topological_sort(nx.DiGraph(graph)))
    reach_ids: list[int] = []
    for u in node_order:
        for v in graph.successors(u):
            for key in graph[u][v]:
                reach_ids.append(int(graph[u][v][key]["reach_id"]))

    riv_path = path / "riv.csv"
    pd.DataFrame({"reach_id": reach_ids}).to_csv(riv_path, header=False, index=False)
    if legacy_call:
        return reach_ids
    return riv_path, reach_ids


def write_reach_ratios_csv(
    ratio_dict: dict[int, list[list[float]]],
    output_dir: str | Path | None = None,
    *,
    directory: str | Path | None = None,
) -> Path:
    path, _legacy_call = _coerce_output_dir(output_dir, directory=directory)
    rat_path = path / "rat.csv"
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


def compute_reach_ratios(
    graph: nx.MultiDiGraph,
    output_dir: str | Path | None = None,
    *,
    use_widths: bool = True,
    directory: str | Path | None = None,
) -> Path | dict[int, list[list[float]]]:
    path, legacy_call = _coerce_output_dir(output_dir, directory=directory)
    reach_to_uv: dict[int, tuple[object, object]] = {}
    width_of: dict[int, float] = {}
    for u, v, data in graph.edges(data=True):
        rid = int(data["reach_id"])
        reach_to_uv[rid] = (u, v)
        width_of[rid] = float(data.get("width", 1.0))

    outlet_sentinel = -1
    while outlet_sentinel in reach_to_uv:
        outlet_sentinel -= 1

    reach_graph = nx.DiGraph()
    reach_graph.add_nodes_from(reach_to_uv.keys())
    for rid, (_u, v) in reach_to_uv.items():
        for _, _v2, downstream_data in graph.out_edges(v, data=True):
            reach_graph.add_edge(rid, int(downstream_data["reach_id"]))
        if reach_graph.out_degree(rid) == 0:
            reach_graph.add_edge(rid, outlet_sentinel)
    if outlet_sentinel in reach_graph.nodes:
        reach_graph.nodes[outlet_sentinel]["outlet"] = True

    if len(reach_to_uv) <= 1:
        order = list(reach_to_uv.keys()) + ([outlet_sentinel] if outlet_sentinel in reach_graph.nodes else [])
    else:
        try:
            order = list(nx.topological_sort(reach_graph))
        except nx.NetworkXUnfeasible as exc:
            raise ValueError("Graph contains cycles; reach connectivity invalid") from exc
    order = [rid for rid in order if rid != outlet_sentinel] + (
        [outlet_sentinel] if outlet_sentinel in reach_graph.nodes else []
    )

    max_down = max((len(list(reach_graph.successors(rid))) for rid in order if rid != outlet_sentinel), default=1)
    ratios: dict[int, list[list[float]]] = {}
    for rid in order:
        if rid == outlet_sentinel:
            continue
        downstream = list(reach_graph.successors(rid))
        if outlet_sentinel in downstream and len(downstream) > 1:
            downstream.remove(outlet_sentinel)

        if not use_widths:
            if downstream == [outlet_sentinel]:
                pairs = [[0, 0]]
            elif len(downstream) == 1:
                pairs = [[downstream[0], 1]]
            else:
                pairs = [[downstream_id, 1] for downstream_id in downstream]
        else:
            if downstream == [outlet_sentinel]:
                pairs = [[0, 0]]
            elif len(downstream) == 1:
                pairs = [[downstream[0], 1]]
            else:
                ds_widths = [width_of[downstream_id] for downstream_id in downstream]
                total = float(sum(ds_widths))
                if total <= 0:
                    values = [1.0 / len(downstream)] * len(downstream)
                else:
                    values = [width / total for width in ds_widths]
                sum_values = float(sum(values))
                if sum_values > 0:
                    values = [value / sum_values for value in values]
                pairs = [[downstream_id, round(value, 6)] for downstream_id, value in zip(downstream, values)]

        while len(pairs) < max_down:
            pairs.append([0, 0])
        ratios[rid] = pairs

    rat_path = path / "rat.csv"
    rat_srt_path = path / "rat_srt.csv"
    if not ratios:
        rat_path.write_text("")
        rat_srt_path.write_text("")
        return ratios if legacy_call else rat_srt_path

    write_reach_ratios_csv(ratios, path)
    try:
        conn_frame = pd.read_csv(path / "conn.csv", header=None)
        ratio_frame = pd.read_csv(rat_path, header=None)
    except pd.errors.EmptyDataError:
        rat_srt_path.write_text("")
        return ratios if legacy_call else rat_srt_path

    if conn_frame.empty or ratio_frame.empty:
        rat_srt_path.write_text("")
        return ratios if legacy_call else rat_srt_path

    order_frame = conn_frame[0].drop_duplicates()
    ratio_sorted = ratio_frame.set_index(0).loc[order_frame].reset_index()
    ratio_sorted.to_csv(rat_srt_path, header=False, index=False)

    if legacy_call:
        return ratios
    return rat_srt_path


def compute_area_csv(
    graph: nx.MultiDiGraph,
    a_width_area: float = 3.0,
    b_width_area: float = 0.40,
    output_dir: str | Path | None = None,
    *,
    directory: str | Path | None = None,
):
    """Compute drainage area from width and write RAPID catchment TSV."""

    path, _legacy_call = _coerce_output_dir(output_dir, directory=directory)

    def get_area(data):
        reach_id = data.get("reach_id")
        width_m = data.get("width")
        geom = wkt.loads(data.get("geometry"))
        centroid = geom.centroid
        area_km2 = (width_m / a_width_area) ** (1.0 / b_width_area)
        return reach_id, area_km2, centroid.x, centroid.y

    catchment_path = path / "rapid_catchment.csv"
    with catchment_path.open("w", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        if isinstance(graph, nx.MultiDiGraph):
            for u, v, k, data in graph.edges(data=True, keys=True):
                reach_id, area_km2, x, y = get_area(data)
                graph[u][v][k]["area_km2"] = area_km2
                writer.writerow([reach_id, area_km2, x, y])
        elif isinstance(graph, nx.DiGraph):
            for u, v, data in graph.edges(data=True):
                reach_id, area_km2, x, y = get_area(data)
                graph[u][v]["area_km2"] = area_km2
                writer.writerow([reach_id, area_km2, x, y])
    return graph


def create_runoff(
    graph: nx.MultiDiGraph,
    output_dir: str | Path | None = None,
    dt_seconds: int = 10800,
    runOffC: list[float] | np.ndarray | None = None,
    return_times: bool = False,
    *,
    directory: str | Path | None = None,
):
    """Create RAPID inflow NetCDF from per-source runoff forcing."""

    path, _legacy_call = _coerce_output_dir(output_dir, directory=directory)
    output_netcdf = path / "inflow.nc"
    if output_netcdf.exists():
        output_netcdf.unlink()

    field_reach_id = "reach_id"
    field_length = "length"
    field_area = "area_km2"
    field_geom = "geometry"

    if runOffC is None:
        runOffC = [0.2]

    reach_ids = []
    lengths_m = []
    areas_m2 = []
    lons = []
    lats = []
    use_keys = graph.is_multigraph()
    edge_iter = graph.edges(keys=True, data=True) if use_keys else graph.edges(data=True)
    source_mask = []
    for edge in edge_iter:
        if use_keys:
            u, v, _k, data = edge
        else:
            u, v, data = edge
        source_mask.append(graph.nodes[u]["node_type"] == "source")
        rid = int(data[field_reach_id])
        length = float(data[field_length])
        area = float(data[field_area])
        geom = shapely.wkt.loads(data[field_geom])
        centroid = geom.centroid
        reach_ids.append(rid)
        lengths_m.append(length)
        areas_m2.append(area * 1_000_000)
        lons.append(centroid.x)
        lats.append(centroid.y)

    source_mask = np.array(source_mask)
    reach_ids = np.array(reach_ids)
    lengths_m = np.array(lengths_m)
    areas_m2 = np.array(areas_m2)
    n_reaches = len(reach_ids)
    n_timesteps = len(runOffC)

    m3_riv = np.zeros((n_timesteps, n_reaches), dtype=np.float32)
    for timestep_index in range(n_timesteps):
        runoff_mm = runOffC[timestep_index] * dt_seconds * lengths_m
        volume_m3 = runoff_mm * 0.001 * areas_m2
        volume_m3[~source_mask] = 0.0
        m3_riv[timestep_index, :] = volume_m3

    with netcdf_file(output_netcdf, "w") as ds:
        ds.title = b"Synthetic runoff for RAPID routing"
        ds.institution = b"River Hierarchy Synthetic runs"
        ds.createDimension("time", n_timesteps)
        ds.createDimension("rivid", n_reaches)
        ds.createDimension("nv", 2)

        m3_var = ds.createVariable("m3_riv", "f4", ("time", "rivid"))
        rivid_var = ds.createVariable("rivid", "i8", ("rivid",))
        time_var = ds.createVariable("time", "i8", ("time",))
        time_bnds_var = ds.createVariable("time_bnds", "i8", ("time", "nv"))
        lon_var = ds.createVariable("lon", "f8", ("rivid",))
        lat_var = ds.createVariable("lat", "f8", ("rivid",))
        ds.createVariable("crs", "i4", ())

        ds.Conventions = b"CF-1.6"
        ds.featureType = b"timeSeries"
        ds.history = f"Created on {datetime.now(UTC).isoformat()}".encode("utf-8")

        rivid_var[:] = reach_ids
        lon_var[:] = lons
        lat_var[:] = lats
        m3_var[:, :] = m3_riv

        base_time = datetime(2015, 1, 1)
        times = [
            int((base_time + timedelta(seconds=dt_seconds * timestep_index) - base_time).total_seconds())
            for timestep_index in range(n_timesteps)
        ]
        time_var[:] = times
        time_var.units = b"seconds since 2015-01-01 00:00:00 +00:00"
        time_var.calendar = b"gregorian"
        time_var.bounds = b"time_bnds"
        for timestep_index in range(n_timesteps):
            time_bnds_var[timestep_index, 0] = times[timestep_index]
            time_bnds_var[timestep_index, 1] = times[timestep_index] + dt_seconds
    if return_times:
        return np.array(times, dtype=np.int64)
    return None


def create_routing_parameters(
    graph: nx.MultiDiGraph,
    output_dir: str | Path | None = None,
    field_reach_id: str = "reach_id",
    field_length: str = "length",
    field_geom: str = "geometry",
    xfc_value: float = 0.1,
    kb: float = 20.0,
    S_global: float = 1e-3,
    n_manning: float = 0.35,
    *,
    directory: str | Path | None = None,
) -> dict[int, float]:
    """Create `kfc.csv`, `xfc.csv`, and `coords.csv` for legacy RAPID runs."""

    path, _legacy_call = _coerce_output_dir(output_dir, directory=directory)
    use_keys = graph.is_multigraph()
    edge_iter = graph.edges(keys=True, data=True) if use_keys else graph.edges(data=True)
    reach_ids = []
    coords_x = []
    coords_y = []
    kfac = []
    xfac = []
    k_by_rid: dict[int, float] = {}

    for edge in edge_iter:
        if use_keys:
            _u, _v, _k, data = edge
        else:
            _u, _v, data = edge
        rid = int(data[field_reach_id])
        length_m = float(data[field_length])
        geom = shapely.wkt.loads(data[field_geom])
        centroid = geom.centroid
        slope_local = float(data.get("slope_local", S_global))
        k_value = (3 / 5) * n_manning * (
            (length_m / (slope_local**0.5)) * ((kb ** (2 / 3)) / (data["width"] ** (2 / 3)))
        )
        reach_ids.append(rid)
        coords_x.append(centroid.x)
        coords_y.append(centroid.y)
        kfac.append(round(k_value, 4))
        k_by_rid[rid] = float(k_value)
        xfac.append(xfc_value)

    with (path / "kfc.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        for value in kfac:
            writer.writerow([value])

    with (path / "xfc.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        for value in xfac:
            writer.writerow([value])

    with (path / "coords.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        for rid, x_coord, y_coord in zip(reach_ids, coords_x, coords_y):
            writer.writerow([rid, x_coord, y_coord])

    return k_by_rid


def compute_dt_from_K(Kdir: str | Path, x: float, return_midpoint: bool = True):
    """Compute admissible Muskingum routing dt from `kfc.csv`."""

    path = _normalize_dir_path(Kdir)
    k_values = np.atleast_1d(np.loadtxt(path / "kfc.csv", dtype=float))
    k_min = k_values.min()
    k_max = k_values.max()
    ratio = k_max / k_min
    allowed_ratio = (1 - x) / x
    eps = 0.001

    assert ratio <= (allowed_ratio + eps), (
        f"No stable dt exists! Kmax/Kmin = {ratio:.3f} but allowed max = "
        f"{allowed_ratio:.3f} for x={x}.\nIncrease x or reduce K variability."
    )

    dt_min = np.max(2 * k_values * x)
    dt_max = np.min(2 * k_values * (1 - x))
    assert dt_min <= (dt_max + eps), f"Stability range collapsed: dt_min={dt_min}, dt_max={dt_max}"
    if return_midpoint:
        return np.round(0.5 * (dt_min + dt_max), -1)
    return dt_min, dt_max


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
    forcing_metadata: dict[str, object] | None = None
    if forcing_path is not None:
        forcing, forcing_metadata = prepare_forcing_table(forcing_path, config=forcing_config)
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
        "forcing_metadata": forcing_metadata,
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
        "forcing_station_key": forcing_metadata.get("selected_station_key", "") if forcing_metadata is not None else "",
        "forcing_cache_csv": forcing_metadata.get("forcing_cache_csv", "") if forcing_metadata is not None else "",
        "forcing_loaded_from_cache": bool(forcing_metadata.get("forcing_loaded_from_cache", False)) if forcing_metadata is not None else False,
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
        except Exception as exc:  # pragma: no cover
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
                "forcing_config": asdict(forcing_config) if forcing_config is not None else None,
                "states_total": int(len(prep_registry)),
                "states_prepared": int(prep_registry["status"].eq("prepared").sum()) if not prep_registry.empty else 0,
                "states_failed": int(prep_registry["status"].eq("failed").sum()) if not prep_registry.empty else 0,
            },
            indent=2,
        )
    )
    return prep_registry


__all__ = [
    "RapidPrepConfig",
    "build_rapid_graph",
    "compute_area_csv",
    "compute_dt_from_K",
    "compute_reach_ratios",
    "create_conn_file",
    "create_riv_file",
    "create_routing_parameters",
    "create_runoff",
    "order_prepared_links_for_total_reach_order",
    "prepare_experiment",
    "prepare_state",
    "split_links_into_subreaches",
    "write_reach_ratios_csv",
    "write_routing_parameter_files",
]
