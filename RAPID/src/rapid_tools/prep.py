"""Shared RAPID input preparation helpers extracted from the legacy runners."""

from __future__ import annotations

import csv
import os
from datetime import UTC, datetime, timedelta
from pathlib import Path

import netCDF4
import networkx as nx
import numpy as np
import pandas as pd
import shapely
import shapely.wkt
from shapely import wkt


def create_conn_file(G, directory):
    """Write RAPID connectivity CSV for a reach graph."""
    edge_list = list(G.edges(data=True))
    node_to_edges_dn = {}
    node_to_edges_up = {}
    for u, v, data in edge_list:
        eid = data["reach_id"]
        node_to_edges_dn.setdefault(u, []).append(eid)
        node_to_edges_up.setdefault(v, []).append(eid)

    records = []
    for u, v, data in edge_list:
        eid = data["reach_id"]
        downstream_edges = node_to_edges_dn.get(v, [])
        upstream_edges = node_to_edges_up.get(u, [])

        row = {
            "reach_id": eid,
            "n_rch_dn": len(downstream_edges),
            "n_rch_up": len(upstream_edges),
        }
        for i, e_dn in enumerate(downstream_edges, start=1):
            row[f"rch_id_dn_{i}"] = e_dn
        for i, e_up in enumerate(upstream_edges, start=1):
            row[f"rch_id_up_{i}"] = e_up
        records.append(row)

    df_connections = pd.DataFrame(records)
    dn_cols = sorted(
        [c for c in df_connections.columns if c.startswith("rch_id_dn_")],
        key=lambda x: int(x.split("_")[3]),
    )
    up_cols = sorted(
        [c for c in df_connections.columns if c.startswith("rch_id_up_")],
        key=lambda x: int(x.split("_")[3]),
    )
    df_connections = df_connections[
        ["reach_id", "n_rch_dn"] + dn_cols + ["n_rch_up"] + up_cols
    ]
    df_connections = df_connections.fillna(0)
    df_connections.to_csv(directory + "conn.csv", header=False, index=False)


def create_riv_file(G, directory):
    """Write RAPID basin ordering CSV and return reach ordering."""
    node_order = list(nx.topological_sort(G))
    sorted_reach_ids = []
    for u in node_order:
        for v in G.successors(u):
            for k in G[u][v]:
                sorted_reach_ids.append(G[u][v][k]["reach_id"])

    pd.DataFrame({"reach_id": sorted_reach_ids}).to_csv(
        directory + "riv.csv", header=False, index=False
    )
    return sorted_reach_ids


def write_reach_ratios_csv(ratio_dict, output_csv):
    """Write RAPID downstream ratio CSV in the preserved legacy format."""
    rat_path = output_csv + "rat.csv"
    max_cols = max((len(v) for v in ratio_dict.values()), default=0)
    with open(rat_path, "w", newline="") as f:
        writer = csv.writer(f)
        if max_cols == 0:
            return
        for rid, pairs in ratio_dict.items():
            row = [rid]
            for down_id, ratio in pairs:
                row.append([down_id, float(ratio)])
            while len(row) < max_cols + 1:
                row.append([0, 0])
            writer.writerow(row)


def compute_reach_ratios(G, use_widths=True, directory=""):
    """Compute downstream reach ratios for a RAPID-ready graph."""
    reach_to_uv = {}
    width_of = {}
    for u, v, data in G.edges(data=True):
        rid = data.get("reach_id")
        if rid is None:
            raise ValueError("All edges must have reach_id")
        reach_to_uv[rid] = (u, v)
        width_of[rid] = data.get("width", 1.0)

    RG = nx.DiGraph()
    RG.add_nodes_from(reach_to_uv.keys())
    for rid, (u, v) in reach_to_uv.items():
        for _, _v2, data2 in G.out_edges(v, data=True):
            RG.add_edge(rid, data2["reach_id"])
        if RG.out_degree(rid) == 0:
            RG.add_edge(rid, 0)
    if 0 in RG.nodes:
        RG.nodes[0]["outlet"] = True

    if len(reach_to_uv) <= 1:
        order = list(reach_to_uv.keys()) + ([0] if 0 in RG.nodes else [])
    else:
        try:
            order = list(nx.topological_sort(RG))
        except nx.NetworkXUnfeasible as exc:
            raise ValueError("Graph contains cycles; reach connectivity invalid") from exc
    order = [r for r in order if r != 0] + ([0] if 0 in RG.nodes else [])

    max_down = max((len(list(RG.successors(r))) for r in order if r != 0), default=1)
    results = {}
    for rid in order:
        if rid == 0:
            continue
        downstream = list(RG.successors(rid))
        if 0 in downstream and len(downstream) > 1:
            downstream.remove(0)

        if not use_widths:
            if downstream == [0]:
                pairs = [[0, 0]]
            elif len(downstream) == 1:
                pairs = [[downstream[0], 1]]
            else:
                pairs = [[d, 1] for d in downstream]
        else:
            if downstream == [0]:
                pairs = [[0, 0]]
            elif len(downstream) == 1:
                pairs = [[downstream[0], 1]]
            else:
                ds_widths = [width_of[d] for d in downstream]
                total = sum(ds_widths)
                if total <= 0:
                    ratios = [1.0 / len(downstream)] * len(downstream)
                else:
                    ratios = [w / total for w in ds_widths]
                sum_rat = sum(ratios)
                if sum_rat > 0:
                    ratios = [r / sum_rat for r in ratios]
                pairs = [[d, round(r, 6)] for d, r in zip(downstream, ratios)]

        while len(pairs) < max_down:
            pairs.append([0, 0])
        results[rid] = pairs

    if not results:
        open(directory + "rat.csv", "w").close()
        open(directory + "rat_srt.csv", "w").close()
        return results

    write_reach_ratios_csv(results, directory)

    def save_sorted_ratios(directory):
        rat_path = directory + "rat.csv"
        rat_srt_path = directory + "rat_srt.csv"
        if not os.path.exists(rat_path):
            open(rat_srt_path, "w").close()
            return
        try:
            con = pd.read_csv(directory + "conn.csv", header=None)
            rat = pd.read_csv(rat_path, header=None)
        except pd.errors.EmptyDataError:
            open(rat_srt_path, "w").close()
            return
        if con.empty or rat.empty:
            open(rat_srt_path, "w").close()
            return
        order = con[0].drop_duplicates()
        rat_srt = rat.set_index(0).loc[order].reset_index()
        rat_srt.to_csv(rat_srt_path, header=None, index=None)

    save_sorted_ratios(directory)
    return results


def compute_area_csv(G, a_width_area: float = 3.0, b_width_area: float = 0.40, directory=""):
    """Compute drainage area from width and write RAPID catchment TSV."""

    def get_area(data):
        reach_id = data.get("reach_id")
        width_m = data.get("width")
        geom = wkt.loads(data.get("geometry"))
        centroid = geom.centroid
        area_km2 = (width_m / a_width_area) ** (1.0 / b_width_area)
        return reach_id, area_km2, centroid.x, centroid.y

    file = directory + "rapid_catchment.csv"
    with open(file, "w", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        if isinstance(G, nx.MultiDiGraph):
            for u, v, k, data in G.edges(data=True, keys=True):
                reach_id, area_km2, x, y = get_area(data)
                G[u][v][k]["area_km2"] = area_km2
                writer.writerow([reach_id, area_km2, x, y])
        elif isinstance(G, nx.DiGraph):
            for u, v, data in G.edges(data=True):
                reach_id, area_km2, x, y = get_area(data)
                G[u][v]["area_km2"] = area_km2
                writer.writerow([reach_id, area_km2, x, y])
    return G


def create_runoff(G, directory, dt_seconds=10800, runOffC=[0.2], return_times=False):
    """Create RAPID inflow NetCDF from per-source runoff forcing."""
    output_netcdf = directory + "inflow.nc"
    if os.path.exists(output_netcdf):
        os.remove(output_netcdf)

    field_reach_id = "reach_id"
    field_length = "length"
    field_area = "area_km2"
    field_geom = "geometry"

    reach_ids = []
    lengths_m = []
    areas_m2 = []
    lons = []
    lats = []
    use_keys = G.is_multigraph()
    edge_iter = G.edges(keys=True, data=True) if use_keys else G.edges(data=True)
    source_mask = []
    for edge in edge_iter:
        if use_keys:
            u, v, _k, data = edge
        else:
            u, v, data = edge
        source_mask.append(G.nodes[u]["node_type"] == "source")
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
    for t in range(n_timesteps):
        runoff_mm = runOffC[t] * dt_seconds * lengths_m
        volume_m3 = runoff_mm * 0.001 * areas_m2
        volume_m3[~source_mask] = 0.0
        m3_riv[t, :] = volume_m3

    ds = netCDF4.Dataset(output_netcdf, "w", format="NETCDF4")
    ds.title = "Synthetic runoff for RAPID routing"
    ds.institution = "River Hierarchy Synthetic runs"
    ds.createDimension("time", n_timesteps)
    ds.createDimension("rivid", n_reaches)
    ds.createDimension("nv", 2)

    m3_var = ds.createVariable("m3_riv", "f4", ("time", "rivid"))
    rivid_var = ds.createVariable("rivid", "i8", ("rivid",))
    time_var = ds.createVariable("time", "i8", ("time",))
    time_bnds_var = ds.createVariable("time_bnds", "i8", ("time", "nv"))
    lon_var = ds.createVariable("lon", "f8", ("rivid",))
    lat_var = ds.createVariable("lat", "f8", ("rivid",))
    ds.createVariable("crs", "i4")

    ds.Conventions = "CF-1.6"
    ds.featureType = "timeSeries"
    ds.history = f"Created on {datetime.now(UTC).isoformat()}"

    rivid_var[:] = reach_ids
    lon_var[:] = lons
    lat_var[:] = lats
    m3_var[:, :] = m3_riv

    base_time = datetime(2015, 1, 1)
    times = [
        int((base_time + timedelta(seconds=dt_seconds * t) - base_time).total_seconds())
        for t in range(n_timesteps)
    ]
    time_var[:] = times
    time_var.units = "seconds since 2015-01-01 00:00:00 +00:00"
    time_var.calendar = "gregorian"
    time_var.bounds = "time_bnds"
    for t in range(n_timesteps):
        time_bnds_var[t, 0] = times[t]
        time_bnds_var[t, 1] = times[t] + dt_seconds

    ds.close()
    if return_times:
        return np.array(times, dtype=np.int64)


def create_routing_parameters(
    G,
    directory,
    field_reach_id="reach_id",
    field_length="length",
    field_geom="geometry",
    xfc_value=0.1,
    kb: float = 20.0,
    S_global: float = 1e-3,
    n_manning: float = 0.35,
):
    """Create `kfc.csv`, `xfc.csv`, and `coords.csv` for RAPID."""
    use_keys = G.is_multigraph()
    edge_iter = G.edges(keys=True, data=True) if use_keys else G.edges(data=True)
    reach_ids = []
    coords_x = []
    coords_y = []
    kfac = []
    xfac = []
    k_by_rid = {}

    for edge in edge_iter:
        if use_keys:
            _u, _v, _k, data = edge
        else:
            _u, _v, data = edge
        rid = int(data[field_reach_id])
        length_m = float(data[field_length])
        geom = shapely.wkt.loads(data[field_geom])
        centroid = geom.centroid
        S_local = float(data.get("slope_local", S_global))
        k_value = (3 / 5) * n_manning * (
            (length_m / (S_local**0.5)) * ((kb ** (2 / 3)) / (data["width"] ** (2 / 3)))
        )
        reach_ids.append(rid)
        coords_x.append(centroid.x)
        coords_y.append(centroid.y)
        kfac.append(round(k_value, 4))
        k_by_rid[rid] = float(k_value)
        xfac.append(xfc_value)

    with open(directory + "kfc.csv", "w") as f:
        writer = csv.writer(f)
        for k in kfac:
            writer.writerow([k])

    with open(directory + "xfc.csv", "w") as f:
        writer = csv.writer(f)
        for x in xfac:
            writer.writerow([x])

    with open(directory + "coords.csv", "w") as f:
        writer = csv.writer(f)
        for rid, x, y in zip(reach_ids, coords_x, coords_y):
            writer.writerow([rid, x, y])

    return k_by_rid


def compute_dt_from_K(Kdir, x, return_midpoint=True):
    """Compute admissible Muskingum routing dt from `kfc.csv`."""
    K = np.loadtxt(Kdir + "kfc.csv", dtype=float)
    Kmin = K.min()
    Kmax = K.max()
    ratio = Kmax / Kmin
    allowed_ratio = (1 - x) / x
    eps = 0.001

    assert ratio <= (allowed_ratio + eps), (
        f"No stable dt exists! Kmax/Kmin = {ratio:.3f} but allowed max = "
        f"{allowed_ratio:.3f} for x={x}.\nIncrease x or reduce K variability."
    )

    dt_min = np.max(2 * K * x)
    dt_max = np.min(2 * K * (1 - x))
    assert dt_min <= (dt_max + eps), (
        f"Stability range collapsed: dt_min={dt_min}, dt_max={dt_max}"
    )
    if return_midpoint:
        return np.round(0.5 * (dt_min + dt_max), -1)
    return dt_min, dt_max


__all__ = [
    "compute_area_csv",
    "compute_dt_from_K",
    "compute_reach_ratios",
    "create_conn_file",
    "create_riv_file",
    "create_routing_parameters",
    "create_runoff",
    "write_reach_ratios_csv",
]
