from __future__ import annotations

"""Adopt validated independent-base slope contexts for equivalent compound bases."""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from network_variants.slope_refit_promotion import (
    apply_slope_refit_promotion_plan,
    build_equivalent_base_adoption_plan,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Adopt validated refit base contexts for direction-identical compound experiments.")
    parser.add_argument("candidate_root", type=Path)
    parser.add_argument("validation_dir", type=Path)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    plan = build_equivalent_base_adoption_plan(args.candidate_root, args.validation_dir)
    print(json.dumps({
        "promotion_kind": plan["promotion_kind"],
        "n_examples": plan["n_examples"],
        "examples": [item["example_id"] for item in plan["items"]],
        "action": "execute" if args.execute else "dry_run",
    }, indent=2))
    if args.execute:
        print(f"Equivalent-base adoption complete. Immutable record: {apply_slope_refit_promotion_plan(plan)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
