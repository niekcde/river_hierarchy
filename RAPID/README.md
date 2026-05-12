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

## Hydrograph Event Detection

The outlet hydrograph metrics now default to an automated event-boundary rule
that is intended to reduce per-run manual timestamp tuning:

- event start is detected once from the forcing hydrograph
- the forcing peak is identified first
- the event start is then taken as the nearest credible local minimum before
  that peak
- event end is detected separately for each routed network from the outlet
  hydrograph
- the outlet peak after the global start is identified first
- the event end is then taken as the first credible local minimum after that
  outlet peak
- if no post-peak outlet minimum is found, the series end is used as an
  explicit fallback and the event is flagged as censored

Manual `event_start_time` and `event_end_time` style overrides are still
supported for backtracking older runs, but they are now treated as override
modes rather than the default workflow.

The main CLI entry point is:

```bash
python RAPID/run_rapid_experiment.py /path/to/experiment \
  --event-start-mode auto_local_min \
  --event-end-mode auto_local_min \
  --event-smoothing-window-steps 3 \
  --event-min-trough-prominence-cms 0.0 \
  --event-end-fallback-mode series_end
```

Useful knobs:

- `--event-smoothing-window-steps`
  Smooths the forcing and outlet series before local peak/minimum detection.
- `--event-min-peak-prominence-cms`
  Rejects weak candidate peaks when identifying the forcing and outlet event
  peaks.
- `--event-min-trough-prominence-cms`
  Rejects weak candidate minima when identifying the start and end boundaries.
- `--event-min-separation-steps`
  Prevents clusters of adjacent samples from being treated as separate events.
- `--max-start-search-window-hours`
  Optionally constrains the automatic start search to the early part of the
  forcing record.

Per-run provenance is written into `hydrograph_metrics.csv`,
`hydrograph_metrics.json`, and the RAPID run manifest through fields such as:

- `event_definition_version`
- `event_start_source`
- `event_end_source`
- `event_end_censored`
- `metric_config`

## Event Definition References

These references are the basis for the local-minimum and explicit-boundary
approach used here:

- Sloto, R.A., and Crouse, M.Y. (1996). `HYSEP: A Computer Program for
  Streamflow Hydrograph Separation and Analysis`.
  The USGS local-minimum method is the clearest classical reference for using
  minima-based hydrograph boundaries.
  https://pubs.usgs.gov/publication/wri964040

- Oppel, H., and Mewes, B. (2020). `On the Automation of Flood Event Separation
  From Continuous Time Series`.
  Useful for the broader point that event boundaries are subjective and should
  be made explicit, reproducible, and auditable.
  https://www.frontiersin.org/journals/water/articles/10.3389/frwa.2020.00018/full

- Giani, G., Tarasova, L., Woods, R.A., and Rico-Ramirez, M. (2022).
  `An Objective Time-Series-Analysis Method for Rainfall-Runoff Event
  Identification`.
  Useful for parameter sensitivity and reproducibility when automating event
  extraction from continuous records.
  https://doi.org/10.1029/2021WR031283

- Tang, W., and Carey, S.K. (2017). `HydRun: A MATLAB toolbox for
  rainfall-runoff analysis`.
  Useful for practical automated event delineation and recession metrics with
  configurable thresholds.
  https://doi.org/10.1002/hyp.11185

## Migration Note

Legacy RAPID copies under `synthetic_runs/legacy/` remain useful for
backtracking, but the shared package in `RAPID/src/rapid_tools/` is the active
implementation that both workflows should converge on.
