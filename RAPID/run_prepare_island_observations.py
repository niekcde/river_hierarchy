from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
if str(ROOT.parent) not in sys.path:
    sys.path.insert(0, str(ROOT.parent))

from rapid_tools.island_observations import prepare_island_observations


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Prepare RAPID only for genuine island states; retain independent aliases as references."
    )
    parser.add_argument("analysis_manifest")
    parser.add_argument("hydraulic_context_registry")
    parser.add_argument("base_experiment_dir")
    parser.add_argument("--work-root", help="Optional disposable directory for regenerable island RAPID prep files.")
    parser.add_argument(
        "--island-id",
        action="append",
        dest="island_ids",
        help=(
            "Prepare only this genuine island state. May be repeated; existing "
            "alias and non-target registry records are preserved for a safe retry."
        ),
    )
    args = parser.parse_args(argv)
    registry = prepare_island_observations(
        args.analysis_manifest,
        hydraulic_context_registry_path=args.hydraulic_context_registry,
        base_experiment_dir=args.base_experiment_dir,
        work_root=args.work_root,
        island_ids=args.island_ids,
    )
    print(f"Prepared: {int(registry['status'].eq('prepared').sum())}")
    print(f"Aliases: {int(registry['status'].eq('referenced_existing_independent_state').sum())}")
    print(
        "Excluded topology-review observations: "
        f"{int(registry['status'].eq('excluded_topology_review').sum())}"
    )
    print(f"Failed: {int(registry['status'].eq('failed').sum())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
