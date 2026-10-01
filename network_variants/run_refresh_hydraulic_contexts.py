from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from network_variants.hydraulic_context import SlopeConfig
from network_variants.hydraulic_context_refresh import refresh_experiment_hydraulic_contexts


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Refresh only derived hydraulic-context products for every complete "
            "state in an existing experiment. Masks, RivGraph, widths, and topology are untouched."
        )
    )
    parser.add_argument("experiment_dir")
    parser.add_argument("--execute", action="store_true", help="Perform the refresh. Omit for a validated dry run.")
    parser.add_argument("--no-backup-existing", action="store_true", help="Do not retain the prior three context artifacts per state before replacement.")
    parser.add_argument(
        "--exclude-base-state",
        action="store_true",
        help="Refresh only child states, preserving an already-promoted base hydraulic context.",
    )
    parser.add_argument(
        "--state-id",
        action="append",
        default=None,
        help="Refresh only this eligible state ID; repeat for a bounded audited batch.",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=1,
        help="Maximum independent states to refresh concurrently within each parent-depth wave (default: 1).",
    )
    parser.add_argument("--slope-profile-mode", choices=("network_dist", "reference_section"), default="reference_section")
    parser.add_argument("--reference-section-summary-path")
    parser.add_argument("--min-slope", type=float, default=1e-6)
    parser.add_argument("--max-slope-for-k", type=float, default=0.009)
    parser.add_argument("--section-slope-ratio-min", type=float)
    parser.add_argument("--section-slope-ratio-max", type=float, default=5.0)
    args = parser.parse_args(argv)

    experiment = Path(args.experiment_dir).expanduser().resolve()
    reference_summary = args.reference_section_summary_path
    if reference_summary is None:
        candidate = experiment / "reference_section" / "reference_section_summary.json"
        if candidate.exists():
            reference_summary = str(candidate)
    report = refresh_experiment_hydraulic_contexts(
        experiment,
        slope_config=SlopeConfig(
            min_slope=args.min_slope,
            slope_profile_mode=args.slope_profile_mode,
            reference_section_summary_path=reference_summary,
            max_slope_for_k=args.max_slope_for_k,
            section_slope_ratio_min=args.section_slope_ratio_min,
            section_slope_ratio_max=args.section_slope_ratio_max,
        ),
        execute=args.execute,
        backup_existing=not args.no_backup_existing,
        include_base_state=not args.exclude_base_state,
        state_ids=set(args.state_id) if args.state_id else None,
        max_workers=args.max_workers,
    )
    print(f"States: {len(report)}")
    print(f"Planned: {int(report['status'].eq('planned').sum())}")
    print(f"Refreshed: {int(report['status'].eq('refreshed').sum())}")
    print(f"Failed: {int(report['status'].eq('failed').sum())}")
    if not args.execute:
        print("Dry run only. Re-run with --execute after reviewing the plan.")
    return 1 if bool(report["status"].eq("failed").any()) else 0


if __name__ == "__main__":
    raise SystemExit(main())
