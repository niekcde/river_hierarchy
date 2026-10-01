"""Rebuild one direction-corrected experiment into a versioned physical root.

The source experiment is read only.  Its historical configuration is replayed
against the now-corrected canonical raw RivGraph inputs, except that derived
reference-section and solver-cache paths are rebuilt under the new output.
This is the required precursor to canonical alias promotion.
"""

from __future__ import annotations

import argparse
from dataclasses import fields
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
RAPID_SRC = ROOT / "RAPID" / "src"
if str(RAPID_SRC) not in sys.path:
    sys.path.insert(0, str(RAPID_SRC))

from hierarchy_level_definition.metrics import BubbleEquivalentResistanceConfig
from rapid_tools.forcing import ForcingConfig
from network_variants.collapse_experiment import run_collapse_experiment


SCHEMA_VERSION = "direction-corrected-experiment-rebuild-v1"


def _filtered_dataclass_payload(payload: dict[str, object], cls: type, *, drop: set[str] = set()) -> dict[str, object]:
    allowed = {field.name for field in fields(cls)}.difference(drop)
    return {key: value for key, value in payload.items() if key in allowed}


def build_rebuild_plan(
    source_experiment: str | Path,
    *,
    output_dir: str | Path,
    corrected_exit_sides: str,
) -> dict[str, object]:
    source = Path(source_experiment).expanduser().resolve()
    manifest_path = source / "experiment_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Source experiment manifest does not exist: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    target = Path(output_dir).expanduser().resolve()
    if target.exists() and any(target.iterdir()):
        raise FileExistsError(f"Corrected target root is non-empty: {target}")
    roots = dict(manifest.get("root_paths") or {})
    required_roots = ("cleaned_mask", "reviewed_links", "reviewed_nodes")
    missing = [key for key in required_roots if not roots.get(key)]
    if missing:
        raise ValueError(f"Source manifest lacks root paths: {missing}")
    absent = [str(Path(str(roots[key]))) for key in required_roots if not Path(str(roots[key])).exists()]
    if absent:
        raise FileNotFoundError("Corrected raw inputs are absent: " + ", ".join(absent))
    raw_run = Path(str(roots["reviewed_links"])).resolve().parent.parent
    continuity = raw_run / "width_review" / "direction_only_continuity.json"
    if not continuity.exists():
        raise FileNotFoundError(f"Corrected raw input lacks direction continuity evidence: {continuity}")
    options = dict(manifest.get("options") or {})
    variant = dict(options.get("variant") or {})
    hierarchy = dict(options.get("hierarchy") or {})
    execution = dict(manifest.get("execution_backends") or variant.get("execution_backends") or {})
    modes = execution.get("requested_modes") or ["full_raster"]
    return {
        "schema_version": SCHEMA_VERSION,
        "source_experiment": str(source),
        "source_experiment_manifest": str(manifest_path),
        "source_experiment_id": manifest.get("experiment_id"),
        "source_exit_sides": manifest.get("exit_sides"),
        "target_experiment": str(target),
        "target_experiment_id": f"{manifest.get('experiment_id')}__direction_corrected",
        "mode": manifest.get("mode"),
        "example_id": str(manifest.get("example_id")),
        "corrected_exit_sides": str(corrected_exit_sides).upper(),
        "raw_inputs": {key: str(Path(str(roots[key])).resolve()) for key in required_roots},
        "direction_continuity_record": str(continuity),
        "execution_modes": list(modes),
        "variant_options": variant,
        "hierarchy_options": hierarchy,
        "does_not": [
            "modify the source experiment",
            "modify a canonical experiment root",
            "create canonical aliases",
        ],
    }


def execute_rebuild(plan: dict[str, object]) -> dict[str, object]:
    variant = dict(plan["variant_options"])
    hierarchy = dict(plan["hierarchy_options"])
    equivalent_payload = dict(hierarchy.get("equivalent_resistance_config") or {})
    equivalent_config = None
    forcing_config = None
    if equivalent_payload:
        # These fields are output-root specific and therefore cannot be copied
        # from the historical source. The runner rebuilds them under target.
        config_payload = _filtered_dataclass_payload(
            equivalent_payload,
            BubbleEquivalentResistanceConfig,
            drop={"reference_kb_summary", "solver_cache_dir"},
        )
        equivalent_config = BubbleEquivalentResistanceConfig(**config_payload)
        forcing_payload = dict(hierarchy.get("equivalent_forcing_config") or {})
        if forcing_payload:
            forcing_config = ForcingConfig(
                **_filtered_dataclass_payload(forcing_payload, ForcingConfig)
            )

    outputs = run_collapse_experiment(
        str(plan["mode"]),
        cleaned_mask_path=plan["raw_inputs"]["cleaned_mask"],
        reviewed_links_path=plan["raw_inputs"]["reviewed_links"],
        reviewed_nodes_path=plan["raw_inputs"]["reviewed_nodes"],
        exit_sides=str(plan["corrected_exit_sides"]),
        experiment_id=str(plan["target_experiment_id"]),
        output_dir=str(plan["target_experiment"]),
        collapse_corridor_mode=str(variant.get("collapse_corridor_mode", "anchored")),
        max_path_cutoff=int(hierarchy.get("max_path_cutoff", 100)),
        max_paths=int(hierarchy.get("max_paths", 5000)),
        pixel_width_fields=hierarchy.get("pixel_width_fields"),
        pixel_width_percentiles=hierarchy.get("pixel_width_percentiles"),
        use_pixel_widths_for_extremes=bool(hierarchy.get("use_pixel_widths_for_extremes", True)),
        debug_hierarchy=bool(hierarchy.get("debug_hierarchy", False)),
        preferred_width_field=str(variant.get("preferred_width_field", "wid_adj_routing")),
        footprint_buffer_scale=float(variant.get("footprint_buffer_scale", 0.5)),
        all_touched=bool(variant.get("all_touched", True)),
        allow_noop=bool(variant.get("allow_noop", False)),
        single_thread=bool(variant.get("single_thread", False)),
        export_sword=bool(variant.get("export_sword", True)),
        transect_scale=float(variant.get("transect_scale", 1.5)),
        min_transect_pixels=float(variant.get("min_transect_pixels", 5.0)),
        bank_candidate_neighbors_max=int(variant.get("bank_candidate_neighbors_max", 2048)),
        match_tolerance=variant.get("match_tolerance"),
        sword_node_source_path=variant.get("sword_node_source_path"),
        sword_wse_field=variant.get("sword_wse_field"),
        sword_match_tolerance=variant.get("sword_match_tolerance"),
        sword_example_station_source_path=variant.get("sword_example_station_source_path"),
        sword_station_match_source_path=variant.get("sword_station_match_source_path"),
        sword_reach_buffer_steps=int(variant.get("sword_reach_buffer_steps", 2)),
        verbose_rivgraph=bool(variant.get("verbose_rivgraph", False)),
        defer_child_hierarchy=bool(hierarchy.get("defer_child_hierarchy", False)),
        equivalent_resistance_config=equivalent_config,
        equivalent_forcing_path=hierarchy.get("equivalent_forcing_path"),
        equivalent_forcing_config=forcing_config,
        equivalent_kb_model_path=hierarchy.get("equivalent_kb_model_path"),
        equivalent_kb_width_sample_field=str(hierarchy.get("equivalent_kb_width_sample_field", "wid_adj_routing")),
        equivalent_kb_width_percentile=hierarchy.get("equivalent_kb_width_percentile"),
        execution_mode=tuple(plan["execution_modes"]),
        artifact_profile="analysis",
    )
    audit_path = Path(str(plan["target_experiment"])) / "direction_corrected_rebuild.json"
    payload = {
        **plan,
        "completed": True,
        "n_complete_states": int(outputs.manifest.get("n_complete_states", 0)),
        "n_failed_states": int(outputs.manifest.get("n_failed_states", 0)),
        "output_manifest": str(Path(str(plan["target_experiment"])) / "experiment_manifest.json"),
    }
    audit_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_experiment")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--corrected-exit-sides", required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    plan = build_rebuild_plan(
        args.source_experiment,
        output_dir=args.output_dir,
        corrected_exit_sides=args.corrected_exit_sides,
    )
    result = execute_rebuild(plan) if args.execute else {"execute_requested": False, **plan}
    print(json.dumps(result, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
