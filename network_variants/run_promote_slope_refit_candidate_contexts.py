from __future__ import annotations

"""Dry-run or atomically promote the validated base slope-refit contexts."""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from network_variants.slope_refit_promotion import (
    apply_slope_refit_promotion_plan,
    build_slope_refit_promotion_plan,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Promote validated base slope-refit contexts with recoverable backups.")
    parser.add_argument("candidate_root", type=Path)
    parser.add_argument("validation_dir", type=Path)
    parser.add_argument("--execute", action="store_true", help="Perform promotion; without this flag only print the validated plan.")
    args = parser.parse_args(argv)
    plan = build_slope_refit_promotion_plan(args.candidate_root, args.validation_dir)
    print(json.dumps({
        "schema_version": plan["schema_version"],
        "candidate_root": plan["candidate_root"],
        "validation_csv": plan["validation_csv"],
        "n_examples": plan["n_examples"],
        "examples": [item["example_id"] for item in plan["items"]],
        "action": "execute" if args.execute else "dry_run",
    }, indent=2))
    if not args.execute:
        return 0
    record = apply_slope_refit_promotion_plan(plan)
    print(f"Promotion complete. Immutable record: {record}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
