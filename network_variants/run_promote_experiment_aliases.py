"""Create canonical aliases for a corrected versioned experiment root."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from network_variants.experiment_alias_promotion import (
    apply_experiment_alias_promotion,
    build_experiment_alias_promotion_plan,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Archive superseded canonical experiment roots and create canonical "
            "symlink aliases to corrected versioned roots. Dry-run by default."
        )
    )
    parser.add_argument("output_root")
    parser.add_argument("example_id", type=int)
    parser.add_argument("--correction-label", default="direction130")
    parser.add_argument("--archive-root-name", default="_archived_pre_correction")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    plan = build_experiment_alias_promotion_plan(
        args.output_root,
        example_id=args.example_id,
        correction_label=args.correction_label,
        archive_root_name=args.archive_root_name,
    )
    result = apply_experiment_alias_promotion(plan) if args.execute else plan
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
