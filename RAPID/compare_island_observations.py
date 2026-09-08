from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
if str(ROOT.parent) not in sys.path:
    sys.path.insert(0, str(ROOT.parent))

from rapid_tools.island_observations import compare_island_to_independent_hydrographs


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compare island routing outputs with their independent-unit counterparts.")
    parser.add_argument("analysis_manifest")
    parser.add_argument("base_experiment_dir")
    args = parser.parse_args(argv)
    result = compare_island_to_independent_hydrographs(
        args.analysis_manifest, base_experiment_dir=args.base_experiment_dir
    )
    print(f"Compared: {int(result['comparison_status'].eq('compared_island_vs_independent_unit').sum())}")
    print(f"Aliases: {int(result['comparison_status'].eq('identity_reference_existing_independent_state').sum())}")
    print(f"Failed: {int(result['comparison_status'].eq('comparison_failed').sum())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
