from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from network_variants.island_experiment import run_island_experiment


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run one reviewed island-only experiment. The output contains all conceptual "
            "islands but computes state products only for non-equivalent island collapses."
        )
    )
    parser.add_argument("--reviewed-setup-dir", required=True, help="Reviewed, approved island candidate setup directory.")
    parser.add_argument("--base-state-source", required=True, help="Completed independent experiment used for aliases and parent matching.")
    parser.add_argument("--output-dir", required=True, help="New standalone <example>_island__anchored experiment directory.")
    parser.add_argument(
        "--adopt-metadata-root",
        action="store_true",
        help="Upgrade only an existing island-experiment-root-v1 metadata directory; refuse roots containing setup or state products.",
    )
    parser.add_argument(
        "--resume-existing-experiment",
        action="store_true",
        help="Resume an interrupted island experiment from its frozen setup and observation manifest; never rebuild or overwrite those inputs.",
    )
    parser.add_argument(
        "--rebuild-observation-manifest",
        action="store_true",
        help=(
            "With --resume-existing-experiment, rebuild the small observation crosswalk "
            "from the frozen reviewed setup after a documented manual candidate adjustment."
        ),
    )
    parser.add_argument("--cleaned-mask", required=True)
    parser.add_argument("--reviewed-links", required=True)
    parser.add_argument("--reviewed-nodes", required=True)
    parser.add_argument("--boundary-component-labels", required=True)
    parser.add_argument("--wet-reference-mask", required=True)
    parser.add_argument("--exit-sides", required=True)
    parser.add_argument("--example-id", required=True)
    parser.add_argument("--single-thread", action="store_true")
    parser.add_argument(
        "--execution-mode",
        choices=("full_raster", "local_patch"),
        default="local_patch",
        help=(
            "Primary backend (default: local_patch). Local execution runs directly from "
            "the independent base; failures automatically use the parent experiment's "
            "fallback backend (full-raster or panelized)."
        ),
    )
    parser.add_argument(
        "--compute-equivalent-resistance",
        action="store_true",
        help="Compute anchored base-versus-island equivalent resistance for genuine island states.",
    )
    parser.add_argument("--local-patch-reference-centerline")
    parser.add_argument("--sword-node-source")
    parser.add_argument("--sword-wse-field")
    parser.add_argument("--sword-match-tolerance", type=float)
    parser.add_argument("--sword-example-stations-source")
    parser.add_argument("--sword-station-match-source")
    parser.add_argument("--sword-reach-buffer-steps", type=int, default=2)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = run_island_experiment(
        reviewed_setup_dir=args.reviewed_setup_dir,
        base_experiment_dir=args.base_state_source,
        output_dir=args.output_dir,
        base_mask_path=args.cleaned_mask,
        parent_links_path=args.reviewed_links,
        parent_nodes_path=args.reviewed_nodes,
        boundary_component_labels_path=args.boundary_component_labels,
        wet_reference_mask_path=args.wet_reference_mask,
        exit_sides=args.exit_sides,
        example_id=args.example_id,
        single_thread=args.single_thread,
        execution_mode=args.execution_mode,
        compute_equivalent_resistance=args.compute_equivalent_resistance,
        local_patch_reference_centerline_path=args.local_patch_reference_centerline,
        sword_node_source_path=args.sword_node_source,
        sword_wse_field=args.sword_wse_field,
        sword_match_tolerance=args.sword_match_tolerance,
        sword_example_station_source_path=args.sword_example_stations_source,
        sword_station_match_source_path=args.sword_station_match_source,
        sword_reach_buffer_steps=args.sword_reach_buffer_steps,
        adopt_metadata_root=args.adopt_metadata_root,
        resume_existing_experiment=args.resume_existing_experiment,
        rebuild_observation_manifest=args.rebuild_observation_manifest,
    )
    print(f"Island experiment: {result['experiment_dir']}")
    print(f"Observation manifest: {result['observation_manifest']}")
    print(f"Analysis manifest: {result['analysis_manifest']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
