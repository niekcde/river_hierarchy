from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from network_variants.island_setup import create_reviewed_island_setup


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Create the audited 8-connected island candidate setup for one completed "
            "independent experiment. This does not create island network variants."
        )
    )
    parser.add_argument("experiment_dir", help="Completed <example>_indep__anchored experiment directory.")
    parser.add_argument("--setup-dir", help="Destination; defaults to <experiment>/island_only_setup_8connected.")
    parser.add_argument("--tiny-island-pixels", type=int, default=9)
    parser.add_argument("--nearby-link-buffer-m", type=float, default=500.0)
    parser.add_argument("--dry-run", action="store_true", help="Print the audit without creating the setup directory.")
    parser.add_argument("--print-candidates", action="store_true", help="Print every candidate row after the summary.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = create_reviewed_island_setup(
        args.experiment_dir,
        setup_dir=args.setup_dir,
        tiny_island_pixels=args.tiny_island_pixels,
        nearby_link_buffer_m=args.nearby_link_buffer_m,
        dry_run=args.dry_run,
    )
    equivalence: pd.DataFrame | None = None
    if result.equivalence_registry_path is not None and result.equivalence_registry_path.exists():
        equivalence = pd.read_csv(result.equivalence_registry_path, low_memory=False)
    print(json.dumps(result.summary, indent=2, sort_keys=True))
    if result.candidate_registry.empty:
        print("\nNo enclosed island candidates were found.")
    else:
        audit = result.candidate_registry.copy()
        if equivalence is not None:
            audit = audit.merge(
                equivalence[
                    [
                        "island_id", "mask_equal", "anchor_equal", "bank_labels_equal",
                        "equivalence_status",
                    ]
                ],
                on="island_id",
                how="left",
                validate="1:1",
            )
            audit["execution_plan"] = audit["equivalence_status"].map(
                {
                    "alias_existing_unit_state": "reuse_existing_independent_state",
                    "requires_full_island_state": "materialize_island_state",
                }
            ).fillna("not_approved_or_not_checked")
        columns = [
            "island_id", "area_km2", "n_pixels", "current_unit_ids",
            "proposed_anchor_unit_id", "selection_status", "n_nearby_links",
        ]
        if equivalence is not None:
            columns.extend(
                ["mask_equal", "anchor_equal", "bank_labels_equal", "execution_plan"]
            )
        display = audit[columns]
        print("\nCandidate audit:")
        print(display.to_string(index=False) if args.print_candidates else display.head(25).to_string(index=False))
        if not args.print_candidates and len(display) > 25:
            print(f"... {len(display) - 25} more candidates; use --print-candidates to show all.")
    if equivalence is not None:
        reruns = equivalence.loc[
            equivalence["equivalence_status"].ne("alias_existing_unit_state")
        ].copy()
        print("\nIsland execution preflight:")
        print(
            f"{len(equivalence) - len(reruns)} aliases; {len(reruns)} island states to materialize."
        )
        if reruns.empty:
            print("All approved islands are exact aliases of existing independent states.")
        else:
            columns = [
                "island_id", "source_unit_id", "mask_equal", "anchor_equal",
                "bank_labels_equal", "equivalence_status",
            ]
            print(reruns[columns].to_string(index=False))
    if args.dry_run:
        print(f"\nDry run only; no files were written to: {result.setup_dir}")
    else:
        print(f"\nIsland setup: {result.setup_dir}")
        print(f"Registry: {result.candidate_registry_path}")
        print(f"Geometries: {result.candidate_geometries_path}")
        print(f"Manifest: {result.manifest_path}")
        print(f"Equivalence registry: {result.equivalence_registry_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
