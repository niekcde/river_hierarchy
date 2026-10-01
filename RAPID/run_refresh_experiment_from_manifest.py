from __future__ import annotations

"""Reprepare and reroute an experiment using its recorded RAPID configuration.

This is intentionally for an upstream-context refresh: it reads the previous
prep manifest before replacing it, reproduces the exact forcing/K/timestep
configuration, then prepares fresh inputs.  Fresh prep prevents the engine
from recovering a stale compact result.
"""

import argparse
import json
import sys
from dataclasses import fields
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
REPO_ROOT = ROOT.parent
for directory in (SRC, REPO_ROOT):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from rapid_tools.engine import RapidResultStorageConfig, run_prepared_experiment
from rapid_tools.forcing import ForcingConfig
from rapid_tools.hydrograph import HydrographMetricConfig
from rapid_tools.prep import RapidPrepConfig, prepare_experiment


def _dataclass_kwargs(cls: type, values: dict[str, object]) -> dict[str, object]:
    valid = {field.name for field in fields(cls)}
    return {key: value for key, value in values.items() if key in valid}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Refresh RAPID prep/run using the stored experiment configuration.")
    parser.add_argument("experiment_dir", type=Path)
    parser.add_argument("--run", action="store_true", help="Route immediately after preparation.")
    parser.add_argument("--compact-qout", action="store_true")
    parser.add_argument("--write-mapping-parquet", action="store_true")
    parser.add_argument("--drop-prepared-inputs", action="store_true")
    args = parser.parse_args(argv)

    experiment = args.experiment_dir.expanduser().resolve()
    manifest_path = experiment / "rapid_prep_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    forcing_path = Path(str(manifest["forcing_path"])).expanduser()
    if not forcing_path.exists():
        raise FileNotFoundError(f"Recorded forcing path is unavailable: {forcing_path}")
    forcing = ForcingConfig(**_dataclass_kwargs(ForcingConfig, dict(manifest["forcing_config"])))
    prep = RapidPrepConfig(**_dataclass_kwargs(RapidPrepConfig, dict(manifest["prep_config"])))
    registry = prepare_experiment(
        experiment,
        forcing_path=forcing_path,
        forcing_config=forcing,
        prep_config=prep,
        work_root=manifest.get("work_root") or None,
    )
    failed = int(registry["status"].eq("failed").sum()) if not registry.empty else 0
    print(f"Prepared states: {int(registry['status'].eq('prepared').sum()) if not registry.empty else 0}")
    print(f"Failed preparation states: {failed}")
    if failed or not args.run:
        return 1 if failed else 0
    runs = run_prepared_experiment(
        experiment,
        storage_config=RapidResultStorageConfig(
            compact_qout=args.compact_qout,
            retain_prepared_inputs=not args.drop_prepared_inputs,
            write_mapping_parquet=args.write_mapping_parquet,
        ),
        hydrograph_config=HydrographMetricConfig(),
    )
    run_failed = int(runs["status"].eq("failed").sum()) if not runs.empty else 0
    print(f"Ran states: {int(runs['status'].eq('ran').sum()) if not runs.empty else 0}")
    print(f"Failed routed states: {run_failed}")
    return 1 if run_failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
