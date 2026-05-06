from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from rapid_tools.engine import run_prepared_experiment
from rapid_tools.hydrograph import HydrographMetricConfig


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
    parser.add_argument(
        "--event-start-time",
        help=(
            "Optional ISO-8601 UTC timestamp used as the hydrograph event start. "
            "If omitted, RAPID chooses the lowest inflow discharge before the inflow peak."
        ),
    )
    parser.add_argument(
        "--event-start-window-hours",
        type=float,
        help=(
            "Optional window, in hours from the start of the forcing series, used when "
            "automatically selecting the event start from the minimum inflow discharge."
        ),
    )
    parser.add_argument(
        "--event-start-buffer-hours",
        type=float,
        help=(
            "Optional symmetric search buffer, in hours, applied around "
            "--event-start-time when selecting the minimum inflow discharge."
        ),
    )
    parser.add_argument(
        "--event-end-time",
        help=(
            "Optional ISO-8601 UTC timestamp used as a reference for the event end. "
            "If provided, fall metrics use the outlet discharge at the detected end time "
            "as the recession baseline."
        ),
    )
    parser.add_argument(
        "--event-end-buffer-hours",
        type=float,
        help=(
            "Optional symmetric search buffer, in hours, applied around "
            "--event-end-time when selecting the minimum inflow discharge."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    run_registry = run_prepared_experiment(
        args.experiment_dir,
        only_prepared=not args.include_failed_prep,
        hydrograph_config=HydrographMetricConfig(
            event_start_time=args.event_start_time,
            event_start_window_hours=args.event_start_window_hours,
            event_start_buffer_hours=args.event_start_buffer_hours,
            event_end_time=args.event_end_time,
            event_end_buffer_hours=args.event_end_buffer_hours,
        ),
    )
    print(f"Wrote RAPID run outputs to {Path(args.experiment_dir).expanduser().resolve()}")
    print(f"Ran states: {int(run_registry['status'].eq('ran').sum()) if not run_registry.empty else 0}")
    print(f"Failed states: {int(run_registry['status'].eq('failed').sum()) if not run_registry.empty else 0}")
    if "hydrograph_status" in run_registry.columns:
        print(
            "Hydrograph metrics computed: "
            f"{int(run_registry['hydrograph_status'].eq('computed').sum()) if not run_registry.empty else 0}"
        )
        print(
            "Hydrograph metric failures: "
            f"{int(run_registry['hydrograph_status'].eq('failed').sum()) if not run_registry.empty else 0}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
