#!/usr/bin/env python3
"""Dry-run or apply validated unit-repair promotion."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

if __package__ is None or __package__ == "":
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

from network_variants.repair_promotion import apply_promotion_plan, build_promotion_plan


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Promote validated local-patch states from a filtered repair experiment "
            "into their original complete experiment. Defaults to a read-only dry-run."
        )
    )
    parser.add_argument("repair_experiment")
    parser.add_argument("target_experiment")
    parser.add_argument("--unit-id", action="append", type=int, default=None)
    parser.add_argument("--apply", action="store_true", help="Apply the validated plan.")
    parser.add_argument(
        "--allow-successful-fallback",
        action="store_true",
        help=(
            "Explicitly permit completed full-raster or panelized fallback states. "
            "Fallback audit, direction, readiness, hierarchy, and equivalent-transition "
            "validation remain mandatory."
        ),
    )
    parser.add_argument("--backup-root", default=None)
    parser.add_argument(
        "--skip-equivalent-refresh", action="store_true",
        help="Do not regenerate experiment-level equivalent-resistance aggregate tables.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    plan = build_promotion_plan(
        args.repair_experiment,
        args.target_experiment,
        unit_ids=args.unit_id,
        allow_successful_fallback=args.allow_successful_fallback,
    )
    print(json.dumps(plan.to_dict(), indent=2, sort_keys=True))
    if not args.apply:
        print("Dry run only. Re-run with --apply to promote these states.")
        return 0
    backup = apply_promotion_plan(
        plan,
        backup_root=args.backup_root,
        refresh_equivalent_resistance=not args.skip_equivalent_refresh,
    )
    print(f"Promotion complete. Recoverable backup: {backup}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
