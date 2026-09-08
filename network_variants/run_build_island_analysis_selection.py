from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from network_variants.island_analysis_selection import build_strict_island_analysis_manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Create non-destructive strict island-analysis manifests from reviewed topology decisions."
    )
    parser.add_argument("experiment_dirs", nargs="+", help="Island experiment directories.")
    parser.add_argument(
        "--review-registry",
        required=True,
        help="CSV with reviewed island topology exclusions and explanatory comments.",
    )
    args = parser.parse_args(argv)
    for experiment_dir in args.experiment_dirs:
        result = build_strict_island_analysis_manifest(
            experiment_dir,
            review_registry_path=args.review_registry,
        )
        print(
            f"{Path(experiment_dir).resolve()}: "
            f"included={int(result['strict_analysis_status'].eq('include_primary_one_cycle_analysis').sum())}; "
            f"excluded={int(result['strict_analysis_status'].eq('exclude_pruning_or_mask_topology_exception').sum())}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
