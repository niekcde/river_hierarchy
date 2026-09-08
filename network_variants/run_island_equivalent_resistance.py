from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from network_variants.island_equivalent_resistance import run_island_equivalent_resistance


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compute equivalent resistance only for genuine island states.")
    parser.add_argument("analysis_manifest")
    parser.add_argument("base_experiment_dir")
    parser.add_argument("--output-dir")
    args = parser.parse_args(argv)
    registry = run_island_equivalent_resistance(
        args.analysis_manifest,
        base_experiment_dir=args.base_experiment_dir,
        output_dir=args.output_dir,
    )
    print(f"Computed island states: {int(registry['equivalent_resistance_status'].eq('computed_island_state').sum())}")
    print(f"Alias references: {int(registry['equivalent_resistance_status'].eq('referenced_existing_independent_state').sum())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
