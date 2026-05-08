"""Config-driven entry points for the sensitivity workflow."""

from __future__ import annotations

import argparse
from pathlib import Path

from synthetic_runs.runners.sensitivity import run_sensitivity_grid
from synthetic_runs.runners.sensitivity_recipes import (
    build_default_sensitivity_recipes,
    validate_recipes,
    write_recipes,
)

from .config import json_dumps_pretty, load_json_config, require_keys, resolve_path


def build_recipes_from_config(config_path: str | Path) -> dict:
    config, config_path = load_json_config(config_path)
    require_keys(config, ["out_path"], label="sensitivity recipe config")
    out_path = resolve_path(config_path, config["out_path"])
    recipes = build_default_sensitivity_recipes()
    write_recipes(recipes, out_path)
    result = {"out_path": str(out_path), "n_recipes": int(len(recipes))}
    validate_against = resolve_path(config_path, config.get("validate_against"))
    if validate_against is not None:
        result["validated"] = bool(validate_recipes(recipes, validate_against))
        result["validate_against"] = str(validate_against)
        if not result["validated"]:
            raise ValueError(f"Recipe validation failed against {validate_against}")
    return result


def run_grid_from_config(config_path: str | Path):
    config, config_path = load_json_config(config_path)
    required = [
        "recipes_path",
        "out_dir",
        "network_ids",
        "kb_values",
        "slope_values",
        "sinuosity_values",
        "forcing_hours_values",
        "peak_values",
        "baseflow_values",
    ]
    require_keys(config, required, label="sensitivity grid config")
    recipes_path = resolve_path(config_path, config["recipes_path"])
    out_dir = resolve_path(config_path, config["out_dir"])
    single_edge_control_path = resolve_path(config_path, config.get("single_edge_control_path"))
    return run_sensitivity_grid(
        recipes_path=recipes_path,
        out_dir=out_dir,
        network_ids=[int(v) for v in config["network_ids"]],
        kb_values=[float(v) for v in config["kb_values"]],
        slope_values=[float(v) for v in config["slope_values"]],
        sinuosity_values=[float(v) for v in config["sinuosity_values"]],
        forcing_hours_values=[float(v) for v in config["forcing_hours_values"]],
        fall_hours_values=None
        if config.get("fall_hours_values") is None
        else [float(v) for v in config["fall_hours_values"]],
        peak_values=[float(v) for v in config["peak_values"]],
        baseflow_values=[float(v) for v in config["baseflow_values"]],
        x=float(config.get("x", 0.1)),
        single_edge_control_path=single_edge_control_path,
        keep_intermediate=bool(config.get("keep_intermediate", False)),
        write_netcdf=bool(config.get("write_netcdf", False)),
        max_paths=int(config.get("max_paths", 100)),
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run config-driven steps for the sensitivity workflow.")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ["build-recipes", "run-grid"]:
        cmd = sub.add_parser(name)
        cmd.add_argument("--config", required=True, help="Path to a JSON config file.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.command == "build-recipes":
        result = build_recipes_from_config(args.config)
    elif args.command == "run-grid":
        result = run_grid_from_config(args.config)
    else:
        raise ValueError(f"Unsupported command: {args.command}")
    print(json_dumps_pretty(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
