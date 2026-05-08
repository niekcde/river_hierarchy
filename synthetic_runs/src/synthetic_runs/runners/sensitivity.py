"""Sensitivity-grid workflow extracted from the legacy top-level runner."""

from __future__ import annotations

import gzip
import json
import shutil
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

from synthetic_runs.core import load_network_by_id

from .controls import load_single_edge_control, resolve_single_edge_control
from .shared import (
    _find_outlet_reach_id,
    _find_source_reach_id,
    _load_rapid_tools,
    build_first_part,
    compute_q_weighted_metrics,
)

try:
    from tqdm import tqdm
except ImportError:  # Fallback: no progress bar if tqdm isn't available.
    def tqdm(iterable, **_kwargs):
        return iterable


def load_recipe_by_id(recipes_gz_path: str | Path, network_id: int) -> dict:
    recipes_gz_path = Path(recipes_gz_path)
    with gzip.open(recipes_gz_path, "rt", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i == network_id:
                return json.loads(line)
    raise ValueError(f"Network {network_id} not found in {recipes_gz_path}")


def infer_network_type(recipe: dict) -> str:
    breaks = recipe.get("breaks", [])
    if not breaks:
        return "no_break"
    kinds = {b.get("kind") for b in breaks}
    if "loop" in kinds:
        return "loop"
    if "cross" in kinds:
        return "cross"
    return "unknown"


def _collect_branch_edges(G: nx.MultiDiGraph) -> dict[str, list[tuple]]:
    edges = {"A": [], "B": []}
    for u, v, k, data in G.edges(keys=True, data=True):
        if data.get("kind") == "corridor" and data.get("branch") in ("A", "B"):
            edges[data.get("branch")].append((u, v, k))
    return edges


def _collect_loop_edges(G: nx.MultiDiGraph) -> dict[str, list[tuple]]:
    edges = {"w1": [], "w2": []}
    for u, v, k, data in G.edges(keys=True, data=True):
        if data.get("kind") == "loop":
            curve = float(data.get("curve", 0))
            key = "w1" if curve >= 0 else "w2"
            edges[key].append((u, v, k))
    return edges


def _collect_cross_edges(G: nx.MultiDiGraph) -> tuple[list[tuple], list[tuple]]:
    cross_edges = []
    corridor_dn = []
    for u, v, k, data in G.edges(keys=True, data=True):
        if data.get("kind") != "cross":
            continue
        cross_edges.append((u, v, k))
        to_branch = data.get("to_branch")
        for _, v2, k2, d2 in G.out_edges(v, keys=True, data=True):
            if d2.get("kind") == "corridor" and d2.get("branch") == to_branch:
                corridor_dn.append((v, v2, k2))
                break
    return cross_edges, corridor_dn


def _apply_local_mods(
    G: nx.MultiDiGraph,
    slope_edges: set[tuple],
    sinuosity_edges: set[tuple],
    S_global: float,
    S_local: float,
    sinuosity: float,
):
    for u, v, k, data in G.edges(keys=True, data=True):
        key = (u, v, k)
        base_len = float(data.get("length_base", data.get("length", 0.0)))
        data["length_base"] = base_len
        slope = float(S_global)
        length_factor = 1.0
        if key in slope_edges:
            slope = float(S_local)
            if slope > 0:
                length_factor *= float(S_global) / slope
        if key in sinuosity_edges:
            length_factor *= float(sinuosity)
        data["slope_global"] = float(S_global)
        data["slope_local"] = float(slope)
        data["sinuosity_applied"] = float(sinuosity) if key in sinuosity_edges else 1.0
        data["length_factor"] = float(length_factor)
        data["length"] = float(base_len) * float(length_factor)


def _config_signature(cfg: dict, sym_branch: bool, sym_loop: bool) -> tuple:
    eps = 1e-12
    slope_effective = abs(cfg["S_local"] - cfg["S_global"]) > eps
    sinuosity_effective = abs(cfg["sinuosity"] - 1.0) > eps

    slope_target = cfg.get("slope_target")
    sinuosity_target = cfg.get("sinuosity_target")

    if cfg["network_type"] == "no_break" and sym_branch:
        if slope_target in ("A", "B"):
            slope_target = "sym"
        if sinuosity_target in ("A", "B"):
            sinuosity_target = "sym"
    if cfg["network_type"] == "loop" and sym_loop:
        if slope_target in ("w1", "w2"):
            slope_target = "sym"
        if sinuosity_target in ("w1", "w2"):
            sinuosity_target = "sym"

    if not slope_effective:
        slope_target = "none"
    if not sinuosity_effective:
        sinuosity_target = "none"

    return (
        cfg["network_id"],
        cfg["network_type"],
        cfg["kb"],
        cfg["S_global"],
        cfg["forcing_hours"],
        cfg.get("fall_hours", cfg["forcing_hours"]),
        cfg["peak"],
        cfg.get("baseflow"),
        cfg["S_local"] if slope_effective else None,
        cfg["sinuosity"] if sinuosity_effective else None,
        slope_target,
        sinuosity_target,
    )


def run_sensitivity_grid(
    *,
    recipes_path: str | Path,
    out_dir: str | Path,
    network_ids: list[int],
    kb_values: list[float],
    slope_values: list[float],
    sinuosity_values: list[float],
    forcing_hours_values: list[float],
    fall_hours_values: list[float] | None = None,
    peak_values: list[float],
    baseflow_values: list[float],
    x: float = 0.1,
    single_edge_control_path: str | Path | None = None,
    keep_intermediate: bool = False,
    write_netcdf: bool = False,
    max_paths: int = 100,
):
    """
    Sensitivity grid runner with per-network targeting logic, automatic dedup,
    and an explicit single-edge control artifact.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    rapid = _load_rapid_tools()
    recipes_path = Path(recipes_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if fall_hours_values is None:
        fall_hours_values = forcing_hours_values

    recipe_for: dict[int, dict] = {}
    for nid in network_ids:
        if nid < 0:
            continue
        recipe_for[nid] = load_recipe_by_id(recipes_path, nid)

    if recipe_for:
        meta0 = next(iter(recipe_for.values())).get("meta", {})
        single_edge_spec, single_edge_control_used = load_single_edge_control(single_edge_control_path)
        single_edge_control = resolve_single_edge_control(single_edge_spec, meta0)
    else:
        raise ValueError("At least one structural recipe is required to resolve the single-edge control.")

    all_configs = []
    seen = set()

    def add_cfg(cfg: dict, sym_branch: bool, sym_loop: bool):
        sig = _config_signature(cfg, sym_branch=sym_branch, sym_loop=sym_loop)
        if sig in seen:
            return
        seen.add(sig)
        all_configs.append(cfg)

    for nid in network_ids:
        if nid < 0:
            continue
        recipe = recipe_for[nid]
        network_type = infer_network_type(recipe)

        WA = float(recipe.get("initial_split", {}).get("WA", 0.0))
        WB = float(recipe.get("initial_split", {}).get("WB", 0.0))
        sym_branch = abs(WA - WB) < 1e-9

        sym_loop = False
        if network_type == "loop":
            for b in recipe.get("breaks", []):
                if b.get("kind") == "loop":
                    w1 = float(b.get("w1", 0.0))
                    w2 = float(b.get("w2", 0.0))
                    sym_loop = abs(w1 - w2) < 1e-9
                    break

        for kb in kb_values:
            for S_global in slope_values:
                for forcing_hours in forcing_hours_values:
                    for fall_hours in fall_hours_values:
                        for peak in peak_values:
                            for baseflow in baseflow_values:
                                if network_type == "no_break":
                                    for S_local in slope_values:
                                        for sinuosity in sinuosity_values:
                                            for target in ["A", "B"]:
                                                cfg = dict(
                                                    network_id=nid,
                                                    network_type=network_type,
                                                    kb=float(kb),
                                                    S_global=float(S_global),
                                                    S_local=float(S_local),
                                                    sinuosity=float(sinuosity),
                                                    forcing_hours=float(forcing_hours),
                                                    fall_hours=float(fall_hours),
                                                    peak=float(peak),
                                                    baseflow=float(baseflow),
                                                    slope_target=target,
                                                    sinuosity_target=target,
                                                )
                                                add_cfg(cfg, sym_branch=sym_branch, sym_loop=sym_loop)
                                elif network_type == "loop":
                                    for S_local in slope_values:
                                        for sinuosity in sinuosity_values:
                                            for target in ["w1", "w2"]:
                                                cfg = dict(
                                                    network_id=nid,
                                                    network_type=network_type,
                                                    kb=float(kb),
                                                    S_global=float(S_global),
                                                    S_local=float(S_local),
                                                    sinuosity=float(sinuosity),
                                                    forcing_hours=float(forcing_hours),
                                                    fall_hours=float(fall_hours),
                                                    peak=float(peak),
                                                    baseflow=float(baseflow),
                                                    slope_target=target,
                                                    sinuosity_target=target,
                                                )
                                                add_cfg(cfg, sym_branch=sym_branch, sym_loop=sym_loop)
                                elif network_type == "cross":
                                    for S_local in slope_values:
                                        for sinuosity in sinuosity_values:
                                            for slope_target in ["cross", "corridor_dn"]:
                                                for sinuosity_target in ["cross", "corridor_dn"]:
                                                    cfg = dict(
                                                        network_id=nid,
                                                        network_type=network_type,
                                                        kb=float(kb),
                                                        S_global=float(S_global),
                                                        S_local=float(S_local),
                                                        sinuosity=float(sinuosity),
                                                        forcing_hours=float(forcing_hours),
                                                        fall_hours=float(fall_hours),
                                                        peak=float(peak),
                                                        baseflow=float(baseflow),
                                                        slope_target=slope_target,
                                                        sinuosity_target=sinuosity_target,
                                                    )
                                                    add_cfg(cfg, sym_branch=sym_branch, sym_loop=sym_loop)
                                else:
                                    cfg = dict(
                                        network_id=nid,
                                        network_type=network_type,
                                        kb=float(kb),
                                        S_global=float(S_global),
                                        S_local=float(S_global),
                                        sinuosity=1.0,
                                        forcing_hours=float(forcing_hours),
                                        fall_hours=float(fall_hours),
                                        peak=float(peak),
                                        baseflow=float(baseflow),
                                        slope_target=None,
                                        sinuosity_target=None,
                                    )
                                    add_cfg(cfg, sym_branch=sym_branch, sym_loop=sym_loop)

    single_edge_id = int(single_edge_control["network_id"])
    if any(int(nid) == single_edge_id for nid in network_ids):
        raise ValueError(
            f"Single-edge control network_id={single_edge_id} collides with structural sensitivity network_ids."
        )
    for kb in kb_values:
        for S_global in slope_values:
            for forcing_hours in forcing_hours_values:
                for fall_hours in fall_hours_values:
                    for peak in peak_values:
                        for baseflow in baseflow_values:
                            cfg = dict(
                                network_id=single_edge_id,
                                network_type=str(single_edge_control["network_type"]),
                                kb=float(kb),
                                S_global=float(S_global),
                                S_local=float(S_global),
                                sinuosity=1.0,
                                forcing_hours=float(forcing_hours),
                                fall_hours=float(fall_hours),
                                peak=float(peak),
                                baseflow=float(baseflow),
                                slope_target=None,
                                sinuosity_target=None,
                            )
                            add_cfg(cfg, sym_branch=False, sym_loop=False)

    for grid_id, cfg in enumerate(all_configs, start=1):
        cfg["grid_id"] = grid_id
        cfg["single_edge_control_path"] = str(single_edge_control_used)

    if not all_configs:
        raise ValueError("No grid configurations generated after dedup.")

    grid_manifest = out_dir / "grid_manifest.csv"
    pd.DataFrame(all_configs).to_csv(grid_manifest, index=False)
    print(f"Wrote grid manifest: {grid_manifest}")
    with open(out_dir / "run_meta_sensitivity.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "recipes_path": str(recipes_path),
                "network_ids": [int(i) for i in network_ids],
                "kb_values": [float(v) for v in kb_values],
                "slope_values": [float(v) for v in slope_values],
                "sinuosity_values": [float(v) for v in sinuosity_values],
                "forcing_hours_values": [float(v) for v in forcing_hours_values],
                "fall_hours_values": [float(v) for v in fall_hours_values],
                "peak_values": [float(v) for v in peak_values],
                "baseflow_values": [float(v) for v in baseflow_values],
                "x": float(x),
                "max_paths": int(max_paths),
                "single_edge_control_path": str(single_edge_control_used),
                "single_edge_control": single_edge_control,
            },
            f,
            indent=2,
        )

    q_schema = pa.schema(
        [
            ("grid_id", pa.int64()),
            ("network_id", pa.int64()),
            ("geometry_id", pa.int64()),
            ("sample_type", pa.string()),
            ("network_type", pa.string()),
            ("time", pa.int64()),
            ("Q", pa.float64()),
            ("kb", pa.float64()),
            ("S_global", pa.float64()),
            ("S_local", pa.float64()),
            ("sinuosity", pa.float64()),
            ("forcing_hours", pa.float64()),
            ("fall_hours", pa.float64()),
            ("peak", pa.float64()),
            ("baseflow", pa.float64()),
            ("slope_target", pa.string()),
            ("sinuosity_target", pa.string()),
        ]
    )
    q_parquet = out_dir / "q_outlet.parquet"
    q_writer = pq.ParquetWriter(q_parquet, q_schema)

    edge_schema = pa.schema(
        [
            ("grid_id", pa.int64()),
            ("network_id", pa.int64()),
            ("geometry_id", pa.int64()),
            ("sample_type", pa.string()),
            ("network_type", pa.string()),
            ("reach_id", pa.int64()),
            ("width", pa.float64()),
            ("length", pa.float64()),
            ("length_base", pa.float64()),
            ("slope_local", pa.float64()),
            ("slope_global", pa.float64()),
            ("sinuosity_applied", pa.float64()),
            ("k_value", pa.float64()),
            ("v_k", pa.float64()),
            ("q_tc_mean", pa.float64()),
            ("v_QA", pa.float64()),
            ("kb", pa.float64()),
            ("S_global", pa.float64()),
            ("S_local", pa.float64()),
            ("sinuosity", pa.float64()),
            ("forcing_hours", pa.float64()),
            ("fall_hours", pa.float64()),
            ("peak", pa.float64()),
            ("baseflow", pa.float64()),
            ("slope_target", pa.string()),
            ("sinuosity_target", pa.string()),
        ]
    )
    edge_parquet = out_dir / "edge_velocity_tc.parquet"
    edge_writer = pq.ParquetWriter(edge_parquet, edge_schema)

    kq_rows = []
    peak_rows = []
    error_rows = []

    for cfg in tqdm(all_configs, desc="Running grid"):
        nid = int(cfg["network_id"])
        network_type = cfg["network_type"]
        kb = float(cfg["kb"])
        S_global = float(cfg["S_global"])
        S_local = float(cfg["S_local"])
        sinuosity = float(cfg["sinuosity"])
        forcing_hours = float(cfg["forcing_hours"])
        fall_hours = float(cfg.get("fall_hours", forcing_hours))
        peak_val = float(cfg["peak"])
        baseflow_val = float(cfg["baseflow"])

        if nid == single_edge_id:
            G = rapid["build_single_edge_graph"](
                single_edge_control["length_m"],
                single_edge_control["width_m"],
            )
            geometry_id = int(single_edge_control["geometry_id"])
            sample_type = str(single_edge_control["sample_type"])
        else:
            recipe = recipe_for[nid]
            net = load_network_by_id(recipes_path, nid)
            G = rapid["rivernetwork_to_rapid_graph"](net)
            geometry_id = int(recipe.get("geometry_id", -1))
            sample_type = str(recipe.get("sample_mode", "custom"))

        work_dir = out_dir / f"grid_{cfg['grid_id']:06d}_net_{nid:06d}"
        work_dir.mkdir(parents=True, exist_ok=True)

        slope_edges = set()
        sinuosity_edges = set()
        if network_type == "no_break":
            branch_edges = _collect_branch_edges(G)
            slope_edges = set(branch_edges.get(cfg.get("slope_target"), []))
            sinuosity_edges = set(branch_edges.get(cfg.get("sinuosity_target"), []))
        elif network_type == "loop":
            loop_edges = _collect_loop_edges(G)
            slope_edges = set(loop_edges.get(cfg.get("slope_target"), []))
            sinuosity_edges = set(loop_edges.get(cfg.get("sinuosity_target"), []))
        elif network_type == "cross":
            cross_edges, corridor_dn = _collect_cross_edges(G)
            if cfg.get("slope_target") == "cross":
                slope_edges = set(cross_edges)
            elif cfg.get("slope_target") == "corridor_dn":
                slope_edges = set(corridor_dn)
            if cfg.get("sinuosity_target") == "cross":
                sinuosity_edges = set(cross_edges)
            elif cfg.get("sinuosity_target") == "corridor_dn":
                sinuosity_edges = set(corridor_dn)

        if cfg.get("slope_target") and not slope_edges and network_type != "single_edge":
            error_rows.append({**cfg, "stage": "target", "error": "slope target edges not found"})
            if not keep_intermediate and work_dir.exists():
                shutil.rmtree(work_dir)
            continue
        if cfg.get("sinuosity_target") and not sinuosity_edges and network_type != "single_edge":
            error_rows.append({**cfg, "stage": "target", "error": "sinuosity target edges not found"})
            if not keep_intermediate and work_dir.exists():
                shutil.rmtree(work_dir)
            continue

        _apply_local_mods(
            G,
            slope_edges=slope_edges,
            sinuosity_edges=sinuosity_edges,
            S_global=S_global,
            S_local=S_local,
            sinuosity=sinuosity,
        )

        try:
            rapid["create_conn_file"](G, directory=str(work_dir) + "/")
            riv_order = rapid["create_riv_file"](G, directory=str(work_dir) + "/")
            rapid["compute_reach_ratios"](G, use_widths=True, directory=str(work_dir) + "/")
            rapid["compute_area_csv"](G, directory=str(work_dir) + "/")
            k_by_rid = rapid["create_routing_parameters"](
                G,
                directory=str(work_dir) + "/",
                xfc_value=x,
                kb=kb,
                S_global=S_global,
            )
            dt = rapid["compute_dt_from_K"](str(work_dir) + "/", x, True)
            runOffC, _durations, bounds = build_first_part(
                dt,
                rise_hours=forcing_hours,
                fall_hours=fall_hours,
                peak=peak_val,
                baseflow=baseflow_val,
            )
            times = np.asarray(
                rapid["create_runoff"](
                    G,
                    directory=str(work_dir) + "/",
                    dt_seconds=dt,
                    runOffC=runOffC,
                    return_times=True,
                )
            )
        except Exception as exc:
            error_rows.append({**cfg, "stage": "prep", "error": str(exc)})
            if not keep_intermediate and work_dir.exists():
                shutil.rmtree(work_dir)
            continue

        outlet_reach_id = _find_outlet_reach_id(G)
        outlet_idx = None
        if outlet_reach_id is not None and outlet_reach_id in riv_order:
            outlet_idx = riv_order.index(outlet_reach_id)

        out_path = work_dir / f"Qout_{nid:06d}_g{cfg['grid_id']:06d}.nc"
        if out_path.exists():
            out_path.unlink()

        try:
            _qout_path, qout = rapid["run_rapid"](
                str(work_dir),
                ROUTING_TIMESTEP_SECONDS=dt,
                runType="sampled",
                seed=int(nid),
                output_path=str(out_path),
                return_qout=True,
            )
        except Exception as exc:
            error_rows.append({**cfg, "stage": "run_rapid", "error": str(exc)})
            if not keep_intermediate and work_dir.exists():
                shutil.rmtree(work_dir)
            continue

        if outlet_idx is not None:
            q = qout[:, outlet_idx].astype(float)
            table = pa.table(
                {
                    "grid_id": pa.array([cfg["grid_id"]] * len(times)),
                    "network_id": pa.array([nid] * len(times)),
                    "geometry_id": pa.array([geometry_id] * len(times)),
                    "sample_type": pa.array([sample_type] * len(times)),
                    "network_type": pa.array([network_type] * len(times)),
                    "time": pa.array(times),
                    "Q": pa.array(q),
                    "kb": pa.array([kb] * len(times)),
                    "S_global": pa.array([S_global] * len(times)),
                    "S_local": pa.array([S_local] * len(times)),
                    "sinuosity": pa.array([sinuosity] * len(times)),
                    "forcing_hours": pa.array([forcing_hours] * len(times)),
                    "fall_hours": pa.array([fall_hours] * len(times)),
                    "peak": pa.array([peak_val] * len(times)),
                    "baseflow": pa.array([baseflow_val] * len(times)),
                    "slope_target": pa.array([str(cfg.get("slope_target"))] * len(times)),
                    "sinuosity_target": pa.array([str(cfg.get("sinuosity_target"))] * len(times)),
                }
            )
            q_writer.write_table(table)
        else:
            error_rows.append({**cfg, "stage": "outlet", "error": "outlet reach not found"})

        peak_row = {
            "grid_id": cfg["grid_id"],
            "network_id": nid,
            "geometry_id": geometry_id,
            "sample_type": sample_type,
            "network_type": network_type,
            "kb": kb,
            "S_global": S_global,
            "S_local": S_local,
            "sinuosity": sinuosity,
            "forcing_hours": forcing_hours,
            "fall_hours": fall_hours,
            "peak": peak_val,
            "baseflow": baseflow_val,
            "slope_target": cfg.get("slope_target"),
            "sinuosity_target": cfg.get("sinuosity_target"),
        }
        source_reach_id, source_err = _find_source_reach_id(G)
        if source_reach_id is None:
            peak_row["error"] = source_err or "source reach not found"
        elif source_reach_id not in riv_order:
            peak_row["error"] = "source reach not in riv_order"
        else:
            q_src = qout[:, riv_order.index(source_reach_id)].astype(float)
            if q_src.size == 0 or np.all(np.isnan(q_src)):
                peak_row["error"] = "qout empty or all NaN"
            else:
                peak_idx = int(np.nanargmax(q_src))
                peak_row["reach_id"] = int(source_reach_id)
                peak_row["peak_q"] = float(q_src[peak_idx])
                peak_row["peak_time"] = int(times[peak_idx])
        peak_rows.append(peak_row)

        kq_row = {
            "grid_id": cfg["grid_id"],
            "network_id": nid,
            "geometry_id": geometry_id,
            "sample_type": sample_type,
            "network_type": network_type,
            "kb": kb,
            "S_global": S_global,
            "S_local": S_local,
            "sinuosity": sinuosity,
            "forcing_hours": forcing_hours,
            "fall_hours": fall_hours,
            "peak": peak_val,
            "baseflow": baseflow_val,
            "slope_target": cfg.get("slope_target"),
            "sinuosity_target": cfg.get("sinuosity_target"),
        }
        kq_metrics, kq_info = compute_q_weighted_metrics(
            G,
            qout,
            riv_order,
            dt_seconds=dt,
            kb=kb,
            S_global=S_global,
            max_paths=max_paths,
            network_id=nid,
        )
        if kq_metrics is None:
            kq_row["error"] = kq_info.get("error", "unknown")
        else:
            kq_row.update(kq_metrics)
            kq_row.update(kq_info)
        kq_rows.append(kq_row)

        t_c_start, t_c_end = bounds["C"]
        mask = (times >= t_c_start) & (times < t_c_end)
        if np.any(mask):
            q_tc_mean = np.nanmean(qout[mask, :], axis=0)
        else:
            q_tc_mean = np.full(qout.shape[1], np.nan)

        rid_to_idx = {rid: i for i, rid in enumerate(riv_order)}
        edge_rows = []
        for u, v, k, data in G.edges(keys=True, data=True):
            rid = int(data.get("reach_id"))
            idx = rid_to_idx.get(rid, None)
            q_mean = float(q_tc_mean[idx]) if idx is not None else np.nan
            length = float(data.get("length", np.nan))
            length_base = float(data.get("length_base", length))
            width = float(data.get("width", np.nan))
            slope_local = float(data.get("slope_local", S_global))
            slope_global = float(data.get("slope_global", S_global))
            sinu_applied = float(data.get("sinuosity_applied", 1.0))
            k_val = float(k_by_rid.get(rid, np.nan))
            v_k = (length / k_val) if k_val > 0 else np.nan
            area = (width * width) / kb if kb > 0 else np.nan
            v_QA = q_mean / area if area and area > 0 else np.nan

            edge_rows.append(
                {
                    "grid_id": cfg["grid_id"],
                    "network_id": nid,
                    "geometry_id": geometry_id,
                    "sample_type": sample_type,
                    "network_type": network_type,
                    "reach_id": rid,
                    "width": width,
                    "length": length,
                    "length_base": length_base,
                    "slope_local": slope_local,
                    "slope_global": slope_global,
                    "sinuosity_applied": sinu_applied,
                    "k_value": k_val,
                    "v_k": v_k,
                    "q_tc_mean": q_mean,
                    "v_QA": v_QA,
                    "kb": kb,
                    "S_global": S_global,
                    "S_local": S_local,
                    "sinuosity": sinuosity,
                    "forcing_hours": forcing_hours,
                    "fall_hours": fall_hours,
                    "peak": peak_val,
                    "baseflow": baseflow_val,
                    "slope_target": str(cfg.get("slope_target")),
                    "sinuosity_target": str(cfg.get("sinuosity_target")),
                }
            )

        if edge_rows:
            edge_writer.write_table(pa.Table.from_pandas(pd.DataFrame(edge_rows), preserve_index=False))

        if not write_netcdf and out_path.exists():
            out_path.unlink()
        if not keep_intermediate and work_dir.exists():
            if write_netcdf and out_path.exists():
                shutil.move(str(out_path), str(out_dir / out_path.name))
            shutil.rmtree(work_dir)

    q_writer.close()
    edge_writer.close()

    if kq_rows:
        kq_path = out_dir / "k_q_metrics.csv"
        pd.DataFrame(kq_rows).to_csv(kq_path, index=False)
        print(f"Wrote {len(kq_rows)} rows to {kq_path}")
    if peak_rows:
        peak_path = out_dir / "peak_q_metrics.csv"
        pd.DataFrame(peak_rows).to_csv(peak_path, index=False)
        print(f"Wrote {len(peak_rows)} rows to {peak_path}")
    err_path = out_dir / "run_errors.csv"
    if error_rows:
        pd.DataFrame(error_rows).to_csv(err_path, index=False)
        print(f"Wrote {len(error_rows)} errors to {err_path}")
    else:
        base_cols = list(all_configs[0].keys()) + ["stage", "error"] if all_configs else ["stage", "error"]
        pd.DataFrame(columns=base_cols).to_csv(err_path, index=False)
        print(f"Wrote empty error file to {err_path}")

    return all_configs


__all__ = [
    "_apply_local_mods",
    "_collect_branch_edges",
    "_collect_cross_edges",
    "_collect_loop_edges",
    "_config_signature",
    "infer_network_type",
    "load_recipe_by_id",
    "run_sensitivity_grid",
]
