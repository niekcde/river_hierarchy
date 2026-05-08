"""Config-driven entry points for the regular synthetic workflow."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from synthetic_runs.analysis.k_metrics import compute_metrics
from synthetic_runs.enumerate import (
    enumerate_geometric_recipes_streamed,
    sample_realized_networks_from_geometry,
)
from synthetic_runs.runners import run_sampled_realizations
from synthetic_runs.runners.shared import _pick_recipes_path

from .config import (
    json_dumps_pretty,
    load_json_config,
    params_from_config,
    require_keys,
    resolve_path,
)


def run_geometry_from_config(config_path: str | Path) -> dict:
    config, config_path = load_json_config(config_path)
    require_keys(config, ["out_dir"], label="geometry config")
    params = params_from_config(config)
    out_dir = resolve_path(config_path, config["out_dir"])
    result = enumerate_geometric_recipes_streamed(
        params,
        out_dir=out_dir,
        min_span=config.get("min_span"),
        dedup_sets=bool(config.get("dedup_sets", True)),
        store_set_hash=bool(config.get("store_set_hash", True)),
    )
    return result


def run_sampling_from_config(config_path: str | Path) -> dict:
    config, config_path = load_json_config(config_path)
    require_keys(config, ["out_dir"], label="sampling config")
    geometry_dir = resolve_path(config_path, config.get("geometry_dir"))
    params_json = resolve_path(config_path, config.get("params_json"))
    geometry_recipes_gz = resolve_path(config_path, config.get("geometry_recipes_gz"))
    if geometry_dir is not None:
        if params_json is None:
            params_json = geometry_dir / "run_meta_geometry.json"
        if geometry_recipes_gz is None:
            geometry_recipes_gz = geometry_dir / "geometry_recipes.jsonl.gz"
    if params_json is None or geometry_recipes_gz is None:
        raise ValueError(
            "Sampling config must provide either geometry_dir or both params_json and geometry_recipes_gz."
        )
    out_dir = resolve_path(config_path, config["out_dir"])
    return sample_realized_networks_from_geometry(
        params_json=params_json,
        geometry_recipes_gz=geometry_recipes_gz,
        out_dir=out_dir,
        n_samples=int(config.get("n_samples", 5)),
        ratios=config.get("ratios"),
        seed=int(config.get("seed", 123)),
        max_attempts_per_sample=int(config.get("max_attempts_per_sample", 500)),
        filter_k_admissible=bool(config.get("filter_k_admissible", True)),
        summary_format=str(config.get("summary_format", "parquet")),
        write_edges=bool(config.get("write_edges", True)),
        rows_chunk=int(config.get("rows_chunk", 50000)),
        W_total=config.get("W_total"),
        min_width=config.get("min_width"),
    )


def run_routing_from_config(config_path: str | Path) -> list[dict]:
    config, config_path = load_json_config(config_path)
    require_keys(config, ["sampled_dir", "out_dir", "n_benchmarks", "n_non_benchmarks"], label="routing config")
    sampled_dir = resolve_path(config_path, config["sampled_dir"])
    out_dir = resolve_path(config_path, config["out_dir"])
    single_edge_control_path = resolve_path(config_path, config.get("single_edge_control_path"))
    return run_sampled_realizations(
        sampled_dir=sampled_dir,
        out_dir=out_dir,
        n_benchmarks=int(config["n_benchmarks"]),
        n_non_benchmarks=int(config["n_non_benchmarks"]),
        seed=int(config.get("seed", 123)),
        use_admissible=bool(config.get("use_admissible", True)),
        output_mode=str(config.get("output_mode", "parquet_outlet")),
        keep_intermediate=bool(config.get("keep_intermediate", False)),
        write_netcdf=bool(config.get("write_netcdf", False)),
        x=float(config.get("x", 0.1)),
        maximin_log_cols=tuple(config.get("maximin_log_cols", ["k_mean"])),
        maximin_start=str(config.get("maximin_start", "extreme")),
        max_paths=int(config.get("max_paths", 100)),
        single_edge_control_path=single_edge_control_path,
    )


def run_k_metrics_from_config(config_path: str | Path) -> dict:
    config, config_path = load_json_config(config_path)
    require_keys(config, ["sampled_dir", "q_outlet", "out_csv"], label="k-metrics config")
    sampled_dir = resolve_path(config_path, config["sampled_dir"])
    q_outlet_path = resolve_path(config_path, config["q_outlet"])
    out_csv = resolve_path(config_path, config["out_csv"])
    recipes_path = _pick_recipes_path(sampled_dir)
    df_ids = pd.read_parquet(q_outlet_path, columns=["network_id"])
    network_ids_all = df_ids["network_id"].dropna().astype(int).unique().tolist()
    if not network_ids_all:
        raise ValueError(f"No network_ids found in {q_outlet_path}")
    network_ids = [nid for nid in network_ids_all if nid >= 0]
    excluded_control_ids = [nid for nid in network_ids_all if nid < 0]
    frames = []
    if network_ids:
        frames.append(compute_metrics(recipes_path, network_ids, max_paths=int(config.get("max_paths", 100))))
    if excluded_control_ids:
        frames.append(
            pd.DataFrame(
                {
                    "network_id": excluded_control_ids,
                    "error": ["control network excluded from K metrics"] * len(excluded_control_ids),
                }
            )
        )
    df_out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=["network_id", "error"])
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(out_csv, index=False)
    return {
        "sampled_dir": str(sampled_dir),
        "q_outlet": str(q_outlet_path),
        "recipes_path": str(recipes_path),
        "out_csv": str(out_csv),
        "n_rows": int(len(df_out)),
        "n_networks": int(len(network_ids_all)),
        "excluded_control_ids": excluded_control_ids,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run config-driven steps for the regular synthetic workflow.")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ["build-geometry", "sample-widths", "route", "k-metrics"]:
        cmd = sub.add_parser(name)
        cmd.add_argument("--config", required=True, help="Path to a JSON config file.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.command == "build-geometry":
        result = run_geometry_from_config(args.config)
    elif args.command == "sample-widths":
        result = run_sampling_from_config(args.config)
    elif args.command == "route":
        result = run_routing_from_config(args.config)
    elif args.command == "k-metrics":
        result = run_k_metrics_from_config(args.config)
    else:
        raise ValueError(f"Unsupported command: {args.command}")
    print(json_dumps_pretty(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
