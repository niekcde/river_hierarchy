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
- optionally guard local link slopes against section-scale outliers
- compute RAPID `K` and `X`
- optionally derive one example-level `kb` from a cached reference section
- optionally apply a K-only effective-length floor for very short routing links
- optionally cap implied celerity
- optionally split long links into RAPID-only subreaches
- normalize forcing and write `inflow.nc`

### Forcing Normalization In Prep

The forcing normalization step belongs to `run_prepare_experiment.py`, not to
`run_rapid_experiment.py`.

Prep can now:

- select one explicit `station_key` from a multi-station forcing source
- truncate the record to an inclusive UTC start/end window
- resample the selected hydrograph to a fixed interval in minutes
- interpolate missing interior bins using time interpolation
- optionally reuse a shared normalized-forcing cache across repeated prep runs

The canonical forcing actually used by RAPID is still written per state to:

- `states/<state_id>/rapid/prep/forcing_normalized.csv`

If a shared cache directory is provided, prep also writes a reusable normalized
forcing CSV plus sidecar metadata JSON keyed by:

- source file stem
- `station_key`
- forcing start/end window
- normalized timestep

Example:

```bash
python RAPID/run_prepare_experiment.py /path/to/experiment \
  --forcing-path SWORD_gauge_match/outputs/subdaily_values/BR/subdaily_timeseries.parquet \
  --forcing-station-key BR:3652880 \
  --station-key-column station_key \
  --time-column time \
  --discharge-column discharge \
  --forcing-start-time 2023-02-12T00:00:00Z \
  --forcing-end-time 2023-02-23T00:00:00Z \
  --forcing-resample-minutes 15 \
  --forcing-output-cache-dir SWORD_gauge_match/outputs/rapid_forcing_cache \
  --min-effective-length-for-k-m 100 \
  --max-slope-for-k 0.005 \
  --section-slope-ratio-max 5
```

After prep, routing and hydrograph metrics use the prepared forcing directly:

```bash
python RAPID/run_rapid_experiment.py /path/to/experiment
```

Important behavior:

- if the forcing source contains multiple station keys, `--forcing-station-key`
  is required
- if the source contains exactly one station key, prep can infer it
- forcing truncation uses inclusive bounds
- resampling changes the forcing timestep seen by RAPID, so prep will still
  validate that a stable routing timestep exists for the chosen normalized dt
- resampling densifies the series; it does not create new hydrologic
  information beyond the source observations

### K-Length And Slope Guard Options

Two optional prep controls are now available for unstable or noisy hydraulic
parameterization:

- `--min-effective-length-for-k-m`
  Applies a lower bound to the length term used inside the RAPID `K`
  calculation only. This does not change geometry, topology, or exported link
  length fields; it only regularizes `K` for very short routing elements.

- `--max-slope-for-k`, `--section-slope-ratio-min`,
  `--section-slope-ratio-max`
  Keep local slopes as the primary source for RAPID `K`, but flag obviously
  implausible local values relative to the section-scale WSE trend. Flagged
  links first try to borrow the nearest valid neighboring slope; if that fails,
  prep can fall back to the section reference slope.

These controls are intended as numerical and hydraulic safeguards, not as a
replacement for the underlying network geometry.

### Example-Level Reference `kb`

Prep can now optionally derive one shared `kb` per example using:

- one cached example-level `reference_section/` under the experiment root
- a single-thread reference network generated by collapsing all hierarchy units
- the `p90` of the raw `width_wet` transect samples along that reference
  section
- the section-scale slope derived from the most downstream and most upstream
  matched SWORD WSE anchors on that reference corridor
- the peak of the normalized forcing hydrograph as the reference discharge
- the BASED depth estimator to predict reference depth from `Q`, `W`, and `S`

The resulting scalar `kb = W_ref / D_ref` is then reused for every state in
that example.

This is an example-level parameterization choice. It is intended to preserve
comparability across variants, not to create a different `kb` for each state.

Relevant prep flags:

- `--kb-mode based_reference_section`
- `--kb-model-path /path/to/based_model_v2.ubj`
- `--kb-width-sample-field width_wet`
- `--kb-width-percentile 90`

The reference-section build writes cached artifacts under:

- `network_variants/output/<example>/reference_section/`

including:

- `reference_section_summary.json`
- `reference_kb_summary.json`
- `variant/` outputs for the single-thread reference network

The selected `kb` is exported per link in
`states/<state_id>/rapid/prep/rapid_link_attributes.csv` via:

- `rapid_kb_value`
- `rapid_kb_source_method`
- `rapid_kb_reference_discharge_cms`
- `rapid_kb_reference_width_m`
- `rapid_kb_reference_slope`
- `rapid_kb_reference_depth_m`

Example:

```bash
python RAPID/run_prepare_experiment.py /path/to/experiment \
  --forcing-path SWORD_gauge_match/outputs/subdaily_values/BR/subdaily_timeseries.parquet \
  --forcing-station-key BR:3652880 \
  --forcing-start-time 2023-02-13T00:00:00Z \
  --forcing-end-time 2023-02-22T00:00:00Z \
  --forcing-resample-minutes 18 \
  --kb-mode based_reference_section \
  --kb-model-path RAPID/assets/based_model_v2.ubj \
  --min-effective-length-for-k-m 200 \
  --max-slope-for-k 0.005 \
  --section-slope-ratio-max 5 \
  --use-celerity-capping \
  --min-celerity-mps 0.28 \
  --max-celerity-mps 1.524 \
  --target-subreach-length-m 250
```

Routing then writes per-state RAPID outputs under `states/<state_id>/rapid/run/`
and can export outlet hydrograph summaries.

### Interpretation In This Project

For the hierarchy-driven `network_variants` workflow, RAPID is used here as a
standardized comparative routing layer across topology variants, not as a
per-state calibrated hydraulic truth model.

That choice follows from the structure of this repository:

- `hierarchy_level_definition/` defines topology-aware collapse candidates
- `network_variants/` regenerates alternative directed networks from those
  collapse choices
- `RAPID/` propagates the same forcing through those alternative networks so
  routed response can be compared on a common basis

The primary question in this workflow is therefore:

- how routed hydrograph response changes when network topology changes under a
  shared forcing and shared routing configuration

and not:

- whether each individual state is a fully calibrated hydraulic model

Accordingly, RAPID prep for a given example is intended to use one shared
configuration across all states in that example:

- one forcing series and forcing timestep
- one Muskingum `x`
- one Manning `n`
- one `kb` method/value
- one celerity-cap range
- one effective-length floor
- one slope-guard rule

This shared configuration is part of the experiment definition. It should not
be retuned state by state, because per-state retuning would weaken the
comparability of the routing response metrics across variants.

In this framing, outputs such as peak timing, peak attenuation, and recession
behavior should be interpreted primarily as:

- relative differences between topology variants

rather than:

- absolute hydraulic truth for every individual regenerated edge

### Heterogeneous-Graph Caveat

This comparative use of RAPID is important because the regenerated graphs can
be both fine-scale and strongly heterogeneous. In such networks, a simple
reach-routing formulation with one `K` and one `X` per edge can become
sensitive to short links and strong width/slope contrasts.

The current prep controls, including:

- celerity caps
- effective-length floors
- guarded local slopes
- optional RAPID-only subreach splitting

should therefore be understood as explicit numerical and hydraulic
regularization used to preserve stable and comparable routing behavior across
variants.

These controls do not make the workflow a replacement for a fully calibrated
hydraulic model. They make it a more consistent comparative experiment.

The example-level reference `kb` should be interpreted in the same way: it
ties routing regularization to one shared geometry/forcing reference per
example, but it does not eliminate the limitations of simplified reach routing
on highly heterogeneous graphs.

### Alternative Routing Models

If the project goal changes from relative variant comparison to higher-fidelity
absolute hydraulic realism, other routing formulations may be better suited to
strongly heterogeneous fine-scale graphs, including:

- diffusive-wave routing
- local-inertial routing
- full 1D dynamic-wave routing

Those models can represent heterogeneous reach physics more directly, but they
also require substantially richer geometry, roughness, boundary-condition, and
calibration support than the current workflow assumes. For the present project,
RAPID is retained because it provides a simpler, auditable, example-level
comparative routing layer across many topology variants.

The current reference-section `kb` implementation uses the BASED estimator
(`based_model_v2.ubj`) from:

- Gearon, J. (2024). Boost-Assisted Stream Estimator for Depth (BASED)
  [Computer software]. https://github.com/JakeGearon/based-api
- Gearon, J.H. et al. (2024). Rules of river avulsion change downstream.
  Nature 634, 91-95. https://doi.org/10.1038/s41586-024-07964-2

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
