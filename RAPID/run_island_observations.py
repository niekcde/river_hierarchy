from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
if str(ROOT.parent) not in sys.path:
    sys.path.insert(0, str(ROOT.parent))

from rapid_tools.engine import RapidResultStorageConfig
from rapid_tools.island_observations import run_prepared_island_observations


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run prepared genuine island RAPID states.")
    parser.add_argument("prep_registry")
    parser.add_argument(
        "--compact-qout",
        action="store_true",
        help="Gzip Qout after successful hydrograph export, retaining the compressed .nc.gz result.",
    )
    parser.add_argument(
        "--write-mapping-parquet",
        action="store_true",
        help="Write a compact Parquet reach/subreach mapping in each completed island run directory.",
    )
    parser.add_argument(
        "--drop-prepared-inputs",
        action="store_true",
        help=(
            "Remove a regenerable island RAPID-preparation directory only after its final "
            "routing products publish successfully. Failed preparation and routing diagnostics remain."
        ),
    )
    args = parser.parse_args(argv)
    registry = run_prepared_island_observations(
        args.prep_registry,
        storage_config=RapidResultStorageConfig(
            compact_qout=args.compact_qout,
            retain_prepared_inputs=not args.drop_prepared_inputs,
            write_mapping_parquet=args.write_mapping_parquet,
        ),
    )
    print(f"Ran: {int(registry['status'].eq('ran').sum())}")
    print(f"Aliases: {int(registry['status'].eq('referenced_existing_independent_state').sum())}")
    print(f"Excluded topology-review observations: {int(registry['status'].eq('excluded_topology_review').sum())}")
    print(f"Failed: {int(registry['status'].eq('failed').sum())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
