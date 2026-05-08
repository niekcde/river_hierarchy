# RAPID

This subproject holds the shared RAPID routing layer for this repository. It
now serves two workflows:

- `synthetic_runs/`, which still uses the legacy-style RAPID helper API for
  synthetic network experiments
- `network_variants/`, which prepares and runs RAPID per state for directed,
  width-enriched experiment outputs

## Scope

- shared RAPID routing engine
- shared RAPID CSV and NetCDF input preparation
- synthetic-network adapters under `src/rapid_tools/adapters/`
- experiment/state preparation for `network_variants`
- experiment/state routing plus outlet hydrograph summaries

## Main Entry Points

- `src/rapid_tools/prep.py`
  Contains the state-based preparation workflow
  `prepare_state(...)` and `prepare_experiment(...)`

- `src/rapid_tools/engine.py`
  Contains the state-based routing workflow
  `run_prepared_state(...)` and `run_prepared_experiment(...)`

- `src/rapid_tools/adapters/synthetic.py`
  Contains synthetic-network adapters used by `synthetic_runs`

## Compatibility Rule

The shared RAPID package preserves the legacy synthetic helper surface so the
current `synthetic_runs` code can keep calling:

- `create_conn_file(...)`
- `create_riv_file(...)`
- `compute_reach_ratios(...)`
- `compute_area_csv(...)`
- `create_runoff(...)`
- `create_routing_parameters(...)`
- `compute_dt_from_K(...)`
- `run_rapid(...)`

At the same time, the newer `network_variants` workflow uses:

- `RapidPrepConfig`
- `prepare_experiment(...)`
- `run_prepared_experiment(...)`

## Network Variants Workflow

The state-based prep expects a `network_variants` experiment directory with:

- `state_registry.csv`
- per-state directed nodes
- per-state width-enriched links

Prep writes RAPID inputs under `states/<state_id>/rapid/prep/` and can:

- compute link slope from matched SWORD WSE
- compute RAPID `K` and `X`
- optionally cap implied celerity
- optionally split long links into RAPID-only subreaches
- normalize forcing and write `inflow.nc`

Routing then writes per-state RAPID outputs under `states/<state_id>/rapid/run/`
and can export outlet hydrograph summaries.

## Migration Note

Legacy RAPID copies under `synthetic_runs/legacy/` remain useful for
backtracking, but the shared package in `RAPID/src/rapid_tools/` is the active
implementation that both workflows should converge on.
