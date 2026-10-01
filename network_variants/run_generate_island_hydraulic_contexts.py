from __future__ import annotations

import argparse
import sys
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from network_variants.island_experiment import generate_island_hydraulic_contexts
from network_variants.hydraulic_context import SlopeConfig


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Generate hydraulic contexts for a strict island analysis manifest. "
            "Independent aliases are referenced and topology-review exclusions are skipped."
        )
    )
    parser.add_argument("analysis_manifest", help="island_analysis_manifest_strict.csv")
    parser.add_argument("--base-link-context", required=True)
    parser.add_argument("--base-node-context", required=True)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Inspect the strict manifest and report the contexts that would be refreshed without writing files.",
    )
    parser.add_argument(
        "--refresh-existing",
        action="store_true",
        help="Replace existing genuine-island context products; aliases remain references to the refreshed independent context.",
    )
    parser.add_argument("--slope-profile-mode", choices=("network_dist", "reference_section"), default="reference_section")
    parser.add_argument("--reference-section-summary-path")
    parser.add_argument("--min-slope", type=float, default=1e-6)
    parser.add_argument("--max-slope-for-k", type=float, default=0.009)
    parser.add_argument("--section-slope-ratio-min", type=float)
    parser.add_argument("--section-slope-ratio-max", type=float, default=5.0)
    args = parser.parse_args(argv)
    manifest = Path(args.analysis_manifest).resolve()
    if args.dry_run:
        analysis = pd.read_csv(manifest, low_memory=False)
        strict = analysis.get("strict_analysis_status", pd.Series("", index=analysis.index)).astype(str)
        backend = analysis.get("analysis_backend", pd.Series("", index=analysis.index)).astype(str)
        excluded = strict.eq("exclude_pruning_or_mask_topology_exception")
        aliases = backend.eq("independent_unit_alias")
        print(f"Observations: {len(analysis)}")
        print(f"Genuine contexts to refresh: {int((~excluded & ~aliases).sum())}")
        print(f"Alias references retained: {int((~excluded & aliases).sum())}")
        print(f"Topology-review exclusions retained: {int(excluded.sum())}")
        print("Dry run only. Re-run with --refresh-existing to write refreshed genuine contexts.")
        return 0
    backup_root = None
    if args.refresh_existing:
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        backup_root = manifest.parent / "hydraulic_context_refresh_backups" / f"island_{stamp}"
    registry = generate_island_hydraulic_contexts(
        args.analysis_manifest,
        base_link_context_path=args.base_link_context,
        base_node_context_path=args.base_node_context,
        slope_config=SlopeConfig(
            min_slope=args.min_slope,
            slope_profile_mode=args.slope_profile_mode,
            reference_section_summary_path=args.reference_section_summary_path,
            max_slope_for_k=args.max_slope_for_k,
            section_slope_ratio_min=args.section_slope_ratio_min,
            section_slope_ratio_max=args.section_slope_ratio_max,
        ),
        refresh_existing=args.refresh_existing,
        backup_existing_dir=backup_root,
    )
    print(f"Generated island contexts: {int(registry['hydraulic_context_action'].eq('generated_island_context').sum())}")
    print(f"Refreshed island contexts: {int(registry['hydraulic_context_action'].eq('refreshed_island_context').sum())}")
    print(f"Alias references: {int(registry['hydraulic_context_action'].eq('reference_existing_independent_state').sum())}")
    print(f"Topology-review exclusions: {int(registry['hydraulic_context_action'].eq('excluded_topology_review').sum())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
