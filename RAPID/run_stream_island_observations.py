"""Storage-bounded RAPID execution for island-only observations."""

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
from rapid_tools.island_observations import stream_island_observations


def _parse_timestep_overrides(values: list[str] | None) -> dict[str, int]:
    overrides: dict[str, int] = {}
    for value in values or []:
        island_id, separator, seconds_text = str(value).partition("=")
        island_id = island_id.strip()
        if not separator or not island_id:
            raise ValueError(
                "Use --island-forcing-routing-dt-seconds as island_id=SECONDS, "
                f"got {value!r}."
            )
        try:
            seconds = int(seconds_text.strip())
        except ValueError as exc:
            raise ValueError(
                "Island forcing/routing timestep seconds must be an integer, "
                f"got {value!r}."
            ) from exc
        if seconds <= 0:
            raise ValueError(f"Island forcing/routing timestep must be positive, got {value!r}.")
        if island_id in overrides and overrides[island_id] != seconds:
            raise ValueError(f"Conflicting timestep overrides were supplied for {island_id!r}.")
        overrides[island_id] = seconds
    return overrides


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare, route, compact, and clean genuine island RAPID observations one at a time. "
            "This bounds workspace use to one state."
        )
    )
    parser.add_argument("analysis_manifest")
    parser.add_argument("hydraulic_context_registry")
    parser.add_argument("base_experiment_dir")
    parser.add_argument("--work-root", required=True, help="Disposable RAPID preparation workspace.")
    parser.add_argument(
        "--island-id",
        action="append",
        dest="island_ids",
        help="Stream only this genuine island state. May be repeated; default is all genuine states.",
    )
    parser.add_argument(
        "--island-forcing-routing-dt-seconds",
        action="append",
        dest="island_timestep_overrides",
        metavar="ISLAND_ID=SECONDS",
        help="Documented per-island forcing/routing clock override; may be repeated.",
    )
    parser.add_argument("--compact-qout", action="store_true")
    parser.add_argument("--write-mapping-parquet", action="store_true")
    parser.add_argument("--drop-prepared-inputs", action="store_true")
    args = parser.parse_args(argv)
    storage = RapidResultStorageConfig(
        compact_qout=bool(args.compact_qout),
        retain_prepared_inputs=not bool(args.drop_prepared_inputs),
        write_mapping_parquet=bool(args.write_mapping_parquet),
    )
    prep, run, comparison = stream_island_observations(
        args.analysis_manifest,
        hydraulic_context_registry_path=args.hydraulic_context_registry,
        base_experiment_dir=args.base_experiment_dir,
        work_root=args.work_root,
        island_ids=args.island_ids,
        island_forcing_routing_dt_seconds=_parse_timestep_overrides(args.island_timestep_overrides),
        storage_config=storage,
    )
    print(f"Prepared: {int(prep['status'].eq('prepared').sum())}")
    print(f"Prep failed: {int(prep['status'].eq('failed').sum())}")
    print(f"Routed: {int(run['status'].eq('ran').sum())}")
    print(f"Route failed: {int(run['status'].eq('failed').sum())}")
    print(f"Compared: {int(comparison['comparison_status'].eq('compared_island_vs_independent_unit').sum())}")
    print(f"Comparison failed: {int(comparison['comparison_status'].eq('comparison_failed').sum())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
