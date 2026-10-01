"""Physically promote rebuilt direction-corrected experiments to canonical roots."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from network_variants.experiment_physical_promotion import (
    apply_experiment_physical_promotion,
    build_experiment_physical_promotion_plan,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Archive old physical roots and promote corrected rebuilds under canonical names."
    )
    parser.add_argument("output_root")
    parser.add_argument("example_ids", nargs="+", type=int)
    parser.add_argument("--correction-label", default="directionfix_20260923")
    parser.add_argument("--archive-root-name", default="_archived_pre_directionfix")
    parser.add_argument("--execute", action="store_true", help="Perform the physical promotion; dry-run otherwise.")
    args = parser.parse_args(argv)
    plan = build_experiment_physical_promotion_plan(
        args.output_root,
        example_ids=args.example_ids,
        correction_label=args.correction_label,
        archive_root_name=args.archive_root_name,
    )
    result = apply_experiment_physical_promotion(plan) if args.execute else plan
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
