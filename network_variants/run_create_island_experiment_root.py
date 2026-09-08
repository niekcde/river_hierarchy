from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from network_variants.island_experiment import create_island_experiment_root


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Create a compact standalone island experiment root without copying state products."
    )
    parser.add_argument("analysis_manifest")
    parser.add_argument("base_experiment_dir")
    parser.add_argument("output_dir")
    args = parser.parse_args(argv)
    result = create_island_experiment_root(
        args.analysis_manifest,
        base_experiment_dir=args.base_experiment_dir,
        output_dir=args.output_dir,
    )
    print(f"Island experiment root: {result['experiment_dir']}")
    print(f"Observation registry: {result['observation_registry']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
