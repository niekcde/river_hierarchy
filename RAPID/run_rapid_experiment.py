from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from rapid_tools.engine import run_prepared_experiment


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the RAPID Python routing engine for every prepared state in an experiment."
    )
    parser.add_argument("experiment_dir", help="Path to the network_variants experiment directory.")
    parser.add_argument(
        "--include-failed-prep",
        action="store_true",
        help="Attempt to run states even if the prep registry did not mark them as prepared.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    run_registry = run_prepared_experiment(
        args.experiment_dir,
        only_prepared=not args.include_failed_prep,
    )
    print(f"Wrote RAPID run outputs to {Path(args.experiment_dir).expanduser().resolve()}")
    print(f"Ran states: {int(run_registry['status'].eq('ran').sum()) if not run_registry.empty else 0}")
    print(f"Failed states: {int(run_registry['status'].eq('failed').sum()) if not run_registry.empty else 0}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
