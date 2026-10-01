"""Rebuild one island experiment from a corrected independent experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from network_variants.island_experiment import run_island_experiment


def _one_path(directory: Path, pattern: str, description: str) -> Path:
    paths = sorted(path for path in directory.glob(pattern) if not path.name.startswith("._"))
    if len(paths) != 1:
        raise FileNotFoundError(f"Expected one {description} matching {pattern!r} in {directory}, found {len(paths)}.")
    return paths[0]


def build_plan(
    historical_island_dir: str | Path,
    corrected_independent_dir: str | Path,
    *,
    output_dir: str | Path,
    corrected_exit_sides: str,
) -> dict[str, object]:
    historical = Path(historical_island_dir).expanduser().resolve()
    corrected = Path(corrected_independent_dir).expanduser().resolve()
    output = Path(output_dir).expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Corrected island output is non-empty: {output}")
    old_manifest = json.loads((historical / "experiment_manifest.json").read_text(encoding="utf-8"))
    base_manifest = json.loads((corrected / "experiment_manifest.json").read_text(encoding="utf-8"))
    example_id = str(old_manifest["example_id"]).zfill(2)
    base_variant = corrected / "states" / "S000_base" / "variant"
    setup = corrected / "island_only_setup_8connected"
    raw_links = Path(str(base_manifest["root_paths"]["reviewed_links"])).expanduser().resolve()
    raw_run = raw_links.parent.parent
    options = dict(base_manifest.get("options", {}).get("variant", {}))
    return {
        "historical_island_dir": str(historical),
        "corrected_independent_dir": str(corrected),
        "output_dir": str(output),
        "example_id": example_id,
        "exit_sides": str(corrected_exit_sides).upper(),
        "reviewed_setup_dir": str(setup),
        "cleaned_mask": str(_one_path(base_variant / "mask", "*_collapsed.tif", "base collapsed mask")),
        "reviewed_links": str(_one_path(base_variant / "widths", "links_with_width_families.gpkg", "base width links")),
        "reviewed_nodes": str(_one_path(base_variant / "directed", "*_directed_nodes.gpkg", "base directed nodes")),
        "boundary_component_labels": str(_one_path(corrected / "boundary_components", "*land_water_labels*.tif", "boundary labels")),
        "local_patch_reference_centerline": str(_one_path(raw_run / "rivgraph", "*_centerline.gpkg", "raw reference centerline")),
        "execution_mode": str(old_manifest.get("execution_mode", "local_patch")),
        "compute_equivalent_resistance": bool(old_manifest.get("equivalent_resistance_registry")),
        "sword_node_source": options.get("sword_node_source_path"),
        "sword_wse_field": options.get("sword_wse_field"),
        "sword_match_tolerance": options.get("sword_match_tolerance"),
        "sword_example_station_source": options.get("sword_example_station_source_path"),
        "sword_station_match_source": options.get("sword_station_match_source_path"),
        "sword_reach_buffer_steps": int(options.get("sword_reach_buffer_steps", 2)),
    }


def execute(plan: dict[str, object]) -> dict[str, object]:
    result = run_island_experiment(
        reviewed_setup_dir=plan["reviewed_setup_dir"],
        base_experiment_dir=plan["corrected_independent_dir"],
        output_dir=plan["output_dir"],
        base_mask_path=plan["cleaned_mask"],
        parent_links_path=plan["reviewed_links"],
        parent_nodes_path=plan["reviewed_nodes"],
        boundary_component_labels_path=plan["boundary_component_labels"],
        wet_reference_mask_path=plan["cleaned_mask"],
        exit_sides=plan["exit_sides"],
        example_id=plan["example_id"],
        execution_mode=plan["execution_mode"],
        compute_equivalent_resistance=bool(plan["compute_equivalent_resistance"]),
        local_patch_reference_centerline_path=plan["local_patch_reference_centerline"],
        sword_node_source_path=plan["sword_node_source"],
        sword_wse_field=plan["sword_wse_field"],
        sword_match_tolerance=plan["sword_match_tolerance"],
        sword_example_station_source_path=plan["sword_example_station_source"],
        sword_station_match_source_path=plan["sword_station_match_source"],
        sword_reach_buffer_steps=int(plan["sword_reach_buffer_steps"]),
    )
    return {**plan, "completed": True, **{key: str(value) for key, value in result.items()}}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("historical_island_dir")
    parser.add_argument("corrected_independent_dir")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--corrected-exit-sides", required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    plan = build_plan(
        args.historical_island_dir,
        args.corrected_independent_dir,
        output_dir=args.output_dir,
        corrected_exit_sides=args.corrected_exit_sides,
    )
    print(json.dumps(execute(plan) if args.execute else {"execute_requested": False, **plan}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
