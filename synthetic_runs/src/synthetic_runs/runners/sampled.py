"""Sampled synthetic-run workflow extracted from the legacy top-level runner."""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from synthetic_runs.core import load_network_by_id

from .controls import load_single_edge_control, resolve_single_edge_control
from .shared import (
    _find_outlet_reach_id,
    _find_source_reach_id,
    _load_rapid_tools,
    _load_run_meta,
    _load_summary_table,
    _pick_recipes_path,
    build_first_part,
    compute_q_weighted_metrics,
    maximin_sample_ids,
)

try:
    from tqdm import tqdm
except ImportError:  # Fallback: no progress bar if tqdm isn't available.
    def tqdm(iterable, **_kwargs):
        return iterable


def run_sampled_realizations(
    sampled_dir: str | Path,
    out_dir: str | Path,
    n_benchmarks: int,
    n_non_benchmarks: int,
    seed: int = 123,
    use_admissible: bool = True,
    output_mode: str = "parquet_outlet",
    keep_intermediate: bool = False,
    write_netcdf: bool = False,
    x=0.1,
    maximin_log_cols: tuple = ("k_mean",),
    maximin_start: str = "extreme",
    max_paths: int = 100,
    single_edge_control_path: str | Path | None = None,
):
    """
    Sample admissible networks from summary using maximin selection on
    (k_mean, ebi_mean). Benchmarks and non-benchmarks are sampled
    independently, then combined with the explicit single-edge control artifact.
    """
    rapid = _load_rapid_tools()
    sampled_dir = Path(sampled_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print("print sampled dir:", sampled_dir, "\n\n")
    run_meta = _load_run_meta(sampled_dir)
    params = run_meta.get("params", {})
    single_edge_spec, single_edge_control_used = load_single_edge_control(single_edge_control_path)
    single_edge_control = resolve_single_edge_control(single_edge_spec, params)
    summary = _load_summary_table(sampled_dir)
    summary["admissible"] = summary["k_ratio"] <= ((1 - x) / x)
    if use_admissible and "admissible" in summary.columns:
        summary = summary[summary["admissible"] == True]
    required_cols = {"network_id", "k_mean", "ebi_mean", "k_ratio", "is_benchmark", "geometry_id"}
    missing = required_cols.difference(summary.columns)
    if missing:
        raise ValueError(f"summary missing required columns: {sorted(missing)}")
    benches = summary[summary["is_benchmark"] == True]
    non_benches = summary[summary["is_benchmark"] == False]
    bench_ids = maximin_sample_ids(
        benches,
        n_benchmarks,
        seed=seed,
        log_cols=maximin_log_cols,
        start=maximin_start,
    )
    non_bench_ids = maximin_sample_ids(
        non_benches,
        n_non_benchmarks,
        seed=seed,
        log_cols=maximin_log_cols,
        start=maximin_start,
    )
    if n_benchmarks > 0 and not bench_ids:
        raise ValueError("No benchmark network_ids selected; check summary filters.")
    if n_non_benchmarks > 0 and not non_bench_ids:
        raise ValueError("No non-benchmark network_ids selected; check summary filters.")

    network_ids = []
    sample_types = {}
    geom_for = summary.set_index("network_id")["geometry_id"].to_dict()
    for bid in bench_ids:
        network_ids.append(int(bid))
        sample_types[int(bid)] = "benchmark"
    for nid in non_bench_ids:
        network_ids.append(int(nid))
        sample_types[int(nid)] = "non_benchmark"

    single_edge_id = int(single_edge_control["network_id"])
    if single_edge_id in network_ids:
        raise ValueError(
            f"Single-edge control network_id={single_edge_id} collides with sampled selection."
        )
    if single_edge_id not in network_ids:
        network_ids.append(single_edge_id)
    sample_types[single_edge_id] = str(single_edge_control["sample_type"])
    geom_for[single_edge_id] = int(single_edge_control["geometry_id"])
    if not network_ids:
        raise ValueError("No network_ids selected after maximin sampling.")

    routing_meta = {
        "sampled_dir": str(sampled_dir),
        "recipes_path": str(_pick_recipes_path(sampled_dir)),
        "seed": int(seed),
        "use_admissible": bool(use_admissible),
        "output_mode": output_mode,
        "x": float(x),
        "max_paths": int(max_paths),
        "selected_benchmark_ids": [int(i) for i in bench_ids],
        "selected_non_benchmark_ids": [int(i) for i in non_bench_ids],
        "selected_network_ids": [int(i) for i in network_ids],
        "single_edge_control_path": str(single_edge_control_used),
        "single_edge_control": single_edge_control,
    }
    with open(out_dir / "run_meta_routing.json", "w", encoding="utf-8") as f:
        json.dump(routing_meta, f, indent=2)

    recipes_path = Path(routing_meta["recipes_path"])
    parquet_writer = None
    if output_mode == "parquet_outlet":
        import pyarrow as pa
        import pyarrow.parquet as pq

        schema = pa.schema(
            [
                ("network_id", pa.int64()),
                ("geometry_id", pa.int64()),
                ("sample_type", pa.string()),
                ("time", pa.int64()),
                ("Q", pa.float64()),
            ]
        )
        parquet_path = out_dir / "q_outlet.parquet"
        parquet_writer = pq.ParquetWriter(parquet_path, schema)

    results = []
    kq_metrics_rows = []
    peak_q_rows = []
    for network_id in tqdm(network_ids, total=len(network_ids), desc="Running networks"):
        if int(network_id) == single_edge_id:
            G = rapid["build_single_edge_graph"](
                single_edge_control["length_m"],
                single_edge_control["width_m"],
            )
        else:
            net = load_network_by_id(recipes_path, int(network_id))
            G = rapid["rivernetwork_to_rapid_graph"](net)

        work_dir = out_dir / f"network_{int(network_id):06d}"
        work_dir.mkdir(parents=True, exist_ok=True)
        rapid["create_conn_file"](G, directory=str(work_dir) + "/")
        riv_order = rapid["create_riv_file"](G, directory=str(work_dir) + "/")
        rapid["compute_reach_ratios"](G, use_widths=True, directory=str(work_dir) + "/")
        rapid["compute_area_csv"](G, directory=str(work_dir) + "/")
        X = 0.1
        rapid["create_routing_parameters"](G, directory=str(work_dir) + "/", xfc_value=X)
        dt = rapid["compute_dt_from_K"](str(work_dir) + "/", X, True)
        runOffC, _durations, _bounds = build_first_part(dt)
        times = rapid["create_runoff"](
            G,
            directory=str(work_dir) + "/",
            dt_seconds=dt,
            runOffC=runOffC,
            return_times=True,
        )
        outlet_reach_id = _find_outlet_reach_id(G)
        outlet_idx = None
        if outlet_reach_id is not None and outlet_reach_id in riv_order:
            outlet_idx = riv_order.index(outlet_reach_id)

        out_path = work_dir / f"Qout_{int(network_id):06d}.nc"
        if out_path.exists():
            out_path.unlink()
        if not os.access(work_dir, os.W_OK):
            raise PermissionError(f"Output directory not writable: {work_dir}")
        qout_path, qout = rapid["run_rapid"](
            str(work_dir),
            ROUTING_TIMESTEP_SECONDS=dt,
            runType="sampled",
            seed=int(network_id),
            output_path=str(out_path),
            return_qout=True,
        )

        peak_row = {
            "network_id": int(network_id),
            "geometry_id": int(geom_for.get(int(network_id), -1)),
            "sample_type": sample_types.get(int(network_id), "unknown"),
        }
        source_reach_id, source_err = _find_source_reach_id(G)
        if source_reach_id is None:
            peak_row["error"] = source_err or "source reach not found"
        elif source_reach_id not in riv_order:
            peak_row["error"] = "source reach not in riv_order"
        else:
            q = qout[:, riv_order.index(source_reach_id)].astype(float)
            if q.size == 0 or np.all(np.isnan(q)):
                peak_row["error"] = "qout empty or all NaN"
            else:
                peak_idx = int(np.nanargmax(q))
                peak_row["reach_id"] = int(source_reach_id)
                peak_row["peak_q"] = float(q[peak_idx])
                peak_row["peak_time"] = int(times[peak_idx])
        peak_q_rows.append(peak_row)

        kq_row = {
            "network_id": int(network_id),
            "geometry_id": int(geom_for.get(int(network_id), -1)),
            "sample_type": sample_types.get(int(network_id), "unknown"),
        }
        kq_metrics, kq_info = compute_q_weighted_metrics(
            G,
            qout,
            riv_order,
            dt_seconds=dt,
            max_paths=max_paths,
            network_id=int(network_id),
        )
        if kq_metrics is None:
            kq_row["error"] = kq_info.get("error", "unknown")
        else:
            kq_row.update(kq_metrics)
            kq_row.update(kq_info)
        kq_metrics_rows.append(kq_row)

        if output_mode == "parquet_outlet":
            import pyarrow as pa

            if outlet_idx is None:
                raise ValueError(f"Outlet reach not found for network_id={network_id}")
            q = qout[:, outlet_idx].astype(float)
            table = pa.table(
                {
                    "network_id": pa.array([int(network_id)] * len(times)),
                    "geometry_id": pa.array([int(geom_for.get(int(network_id), -1))] * len(times)),
                    "sample_type": pa.array([sample_types.get(int(network_id), "unknown")] * len(times)),
                    "time": pa.array(times),
                    "Q": pa.array(q),
                }
            )
            parquet_writer.write_table(table)

        results.append(
            {
                "network_id": int(network_id),
                "dt": float(dt),
                "qout_path": str(qout_path),
            }
        )
        if not write_netcdf and out_path.exists():
            out_path.unlink()
        if not keep_intermediate:
            if write_netcdf and out_path.exists():
                shutil.move(str(out_path), str(out_dir / out_path.name))
            shutil.rmtree(work_dir)

    if parquet_writer is not None:
        parquet_writer.close()
    if kq_metrics_rows:
        kq_path = out_dir / "k_q_metrics.csv"
        pd.DataFrame(kq_metrics_rows).to_csv(kq_path, index=False)
        print(f"Wrote {len(kq_metrics_rows)} rows to {kq_path}")
    if peak_q_rows:
        peak_path = out_dir / "peak_q_metrics.csv"
        pd.DataFrame(peak_q_rows).to_csv(peak_path, index=False)
        print(f"Wrote {len(peak_q_rows)} rows to {peak_path}")
    return results


__all__ = ["run_sampled_realizations"]
