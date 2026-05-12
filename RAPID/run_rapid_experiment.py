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
        "--event-start-mode",
        choices=("auto_local_min", "manual"),
        default="auto_local_min",
        help=(
            "How to define the event start. The default uses the nearest pre-peak "
            "local minimum on the forcing hydrograph; manual mode uses --event-start-time."
        ),
    )
    parser.add_argument(
        "--event-end-mode",
        choices=("auto_local_min", "manual", "series_end"),
        default="auto_local_min",
        help=(
            "How to define the event end. The default uses the first post-peak local "
            "minimum on the outlet hydrograph with a series-end fallback."
        ),
    )
    parser.add_argument(
        "--event-start-time",
        help=(
            "Optional ISO-8601 UTC timestamp used as a manual event-start override "
            "on the forcing hydrograph."
        ),
    )
    parser.add_argument(
        "--event-start-window-hours",
        type=float,
        help=(
            "Legacy alias for --max-start-search-window-hours."
        ),
    )
    parser.add_argument(
        "--max-start-search-window-hours",
        type=float,
        help=(
            "Optional window, in hours from the start of the forcing series, used to "
            "limit the automatic search for the pre-peak event-start minimum."
        ),
    )
    parser.add_argument(
        "--event-start-buffer-hours",
        type=float,
        help=(
            "Optional symmetric search buffer, in hours, applied around "
            "--event-start-time when selecting the minimum forcing discharge."
        ),
    )
    parser.add_argument(
        "--event-end-time",
        help=(
            "Optional ISO-8601 UTC timestamp used as a manual event-end override."
        ),
    )
    parser.add_argument(
        "--event-end-buffer-hours",
        type=float,
        help=(
            "Optional symmetric search buffer, in hours, applied around "
            "--event-end-time when selecting the minimum discharge in the manual search window."
        ),
    )
    parser.add_argument(
        "--event-smoothing-window-steps",
        type=int,
        default=1,
        help="Centered rolling-mean window used before local peak/minimum detection.",
    )
    parser.add_argument(
        "--event-min-peak-prominence-cms",
        type=float,
        default=0.0,
        help="Minimum peak prominence used when identifying forcing and outlet event peaks.",
    )
    parser.add_argument(
        "--event-min-trough-prominence-cms",
        type=float,
        default=0.0,
        help="Minimum trough prominence used when identifying event start and end minima.",
    )
    parser.add_argument(
        "--event-min-separation-steps",
        type=int,
        default=1,
        help="Minimum sample separation between candidate peaks/minima during event detection.",
    )
    parser.add_argument(
        "--event-end-fallback-mode",
        choices=("series_end", "error"),
        default="series_end",
        help="Fallback when no post-peak outlet minimum is found in automatic end detection.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    run_registry = run_prepared_experiment(
        args.experiment_dir,
        only_prepared=not args.include_failed_prep,
        hydrograph_config=HydrographMetricConfig(
            start_mode=args.event_start_mode,
            end_mode=args.event_end_mode,
            manual_start_time=args.event_start_time,
            max_start_search_window_hours=(
                args.max_start_search_window_hours
                if args.max_start_search_window_hours is not None
                else args.event_start_window_hours
            ),
            manual_start_buffer_hours=args.event_start_buffer_hours,
            manual_end_time=args.event_end_time,
            manual_end_buffer_hours=args.event_end_buffer_hours,
            smoothing_window_steps=args.event_smoothing_window_steps,
            min_peak_prominence_cms=args.event_min_peak_prominence_cms,
            min_trough_prominence_cms=args.event_min_trough_prominence_cms,
            min_separation_steps=args.event_min_separation_steps,
            end_fallback_mode=args.event_end_fallback_mode,
            # Preserve legacy fields in the manifest for backtracking older CLI use.
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
