# RAPID

This subproject holds the shared RAPID routing layer for this repository. It
now serves two workflows:

- `synthetic_runs/`, which still uses the legacy-style RAPID helper API for
  synthetic network experiments
- `network_variants/`, which prepares and runs RAPID per state for directed,
  width-enriched experiment outputs

See [current response metric definitions](CURRENT_RESPONSE_METRICS.md) for the
three evaluation metrics and the distinction from historical spread analyses.

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
- `rapid_tools.analysis`

## Network Variants Workflow

The state-based prep expects a `network_variants` experiment directory with:

- `state_registry.csv`
- per-state directed nodes
- per-state width-enriched links
- per-state `hydraulic_context/` artifacts when available

Prep writes RAPID inputs under `states/<state_id>/rapid/prep/` and can:

- prefer precomputed state-level hydraulic context from `network_variants`
- fall back to internal slope derivation from matched SWORD WSE when that
  hydraulic context is missing
- optionally guard local link slopes against section-scale outliers
- compute RAPID `K` and `X`
- optionally derive one example-level `kb` from a cached reference section
- optionally apply a K-only effective-length floor for very short routing links
- optionally cap implied celerity
- optionally split long links into RAPID-only subreaches
- optionally derive one width per RAPID subreach from `link_width_samples.csv`
- normalize forcing and write `inflow.nc`
- export geometry-preserving RAPID reach and node GeoPackages for downstream
  visualization

### Hydraulic Context Ownership

Slope/profile derivation is now treated primarily as a `network_variants`
responsibility rather than a RAPID-only preprocessing step.

The target workflow is:

1. `network_variants` writes `states/<state_id>/hydraulic_context/`
2. RAPID prep reads `link_hydraulic_context.csv` and
   `node_hydraulic_context.csv`
3. RAPID prep uses those precomputed state-level slope/profile fields when
   building routing parameters

The old RAPID-internal slope path still exists as a transitional fallback for
incomplete older experiment directories, but it is not the intended long-term
owner of that logic. The desired end state is:

- all slope/profile construction happens in `network_variants`
- RAPID prep performs no reference-section fitting and no slope derivation
- RAPID prep only consumes cached hydraulic-context outputs
- this keeps hydraulic assumptions with the state geometry that created them
- this also keeps RAPID prep fast

The node-side hydraulic-context manifest also keeps three head concepts
separate:

- observed or matched node WSE:
  `sword_wse` / raw node WSE context
- derived profile WSE for slope:
  `rapid_wse_for_slope`
- normalized equivalent-resistance solver heads:
  not written in RAPID prep

Practical data-source rule:

- final per-link width for routing/K calculations should come from
  `variant/widths/link_width_families.csv` or the equivalent
  `links_with_width_families.gpkg`
- final per-link slope should come from
  `hydraulic_context/link_hydraulic_context.csv`
- `variant/widths/link_width_samples.csv` remains the raw sample/QC record and
  should only be used for debugging or for workflows that intentionally
  summarize raw samples, such as subreach-varying width

### Forcing Normalization In Prep

The forcing normalization step belongs to `run_prepare_experiment.py`, not to
`run_rapid_experiment.py`.

Prep can now:

- select one explicit `station_key` from a multi-station forcing source
- truncate the record to an inclusive UTC start/end window
- resample the selected hydrograph to a fixed interval in seconds, or by a
  fractional-minute value that resolves to whole seconds
- interpolate missing interior bins using time interpolation
- optionally prepend a constant spin-up at the first normalized discharge value
- optionally derive that spin-up automatically from the longest downstream
  summed-`K` path in the prepared RAPID graph
- by default append a constant tail with the same duration as the spin-up using
  the final normalized discharge value
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
  --forcing-resample-seconds 900 \
  --auto-spinup-from-k \
  --spinup-k-multiplier 1.1 \
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
- prefer `--forcing-resample-seconds` when the stable RAPID routing timestep is
  not a whole number of minutes; `--forcing-resample-minutes` still works for
  integer or fractional minute values that resolve to whole seconds
- resampling changes the forcing timestep seen by RAPID, so prep will still
  validate that a stable routing timestep exists for the chosen normalized dt
- resampling densifies the series; it does not create new hydrologic
  information beyond the source observations
- when `--preferred-routing-dt-seconds` is supplied, prep first builds the K
  values and Muskingum stability interval for every state. When the matching
  sibling experiment exists, the prepass automatically covers both the
  independent-unit and compound-bubble experiments. It then selects one stable
  multiple-of-five timestep for all states when the intervals overlap. If they
  do not overlap, it projects one common central target into each interval so
  that the state-specific values—and therefore the two experiment modes—remain
  as close as stability permits
- in that coordinated mode, each state's forcing interval is automatically set
  equal to its selected routing interval. This provides exact clock alignment;
  `--forcing-resample-seconds` remains recorded as the requested input setting
  but is not imposed on an incompatible state
- without `--preferred-routing-dt-seconds`, the legacy explicit-forcing and
  statewise divisor-selection behavior remains available
- manual `--spinup-hours` still works, but `--auto-spinup-from-k` can instead
  resolve spin-up as `spinup_k_multiplier * longest_downstream_sum(K)`
- use `--disable-end-padding-with-spinup` to turn off the constant tail after
  the event window if you need the older forcing behavior

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

- `--use-stability-effective-length-cap`
  First limits excessive `K` values on reaches at or near the configured
  minimum celerity. If that pass still leaves `dt_min > dt_max`, prep applies
  the same mathematically derived cap to the remaining high-`K` tail, only as
  far as needed to open the Muskingum stability interval and never below the
  configured effective-length floor. This adaptive fallback is recorded per
  subreach with `rapid_stability_length_cap_method=adaptive_high_k_tail` and
  summarized separately in the prep manifest. Geometry, exported physical
  length, width, slope, and celerity are not changed; only the effective length
  used to calculate routing `K` is regularized.

These controls are intended as numerical and hydraulic safeguards, not as a
replacement for the underlying network geometry.

### Subreach-Varying Width

When `--target-subreach-length-m` is used, RAPID can now optionally assign a
different width to each derived subreach instead of copying one parent-link
width across every segment.

This uses the existing per-link transect samples already written by
`network_variants` in:

- `variant/widths/link_width_samples.csv`

Relevant flags:

- `--vary-width-by-subreach`
- `--subreach-width-sample-field width_wet`

Current behavior:

- slope still stays constant within a split parent link
- width is summarized independently for each subreach from the requested raw
  width sample field
- the current summary statistic is the median of positive samples that fall on
  that subreach
- subreach-varying width is used for RAPID `K`
- bifurcation split ratios still use a stable parent-link width so the first
  split segment does not over-allocate a small branch just because the branch
  mouth is locally wide
- if a subreach has no usable samples, prep falls back to the parent-link
  width field

This is mainly intended to preserve along-link width variation when a long
collapsed link is segmented for RAPID routing.

Important distinction:

- final parent-link width inheritance in `network_variants` affects
  `link_width_families.csv` / `links_with_width_families.gpkg`
- subreach-varying width still summarizes raw
  `variant/widths/link_width_samples.csv` values intentionally, because that
  workflow is preserving along-link sample variation rather than only consuming
  the final parent-link family summary

### Example-Level Reference `kb`

The upstream collapse/equivalent-resistance workflow derives one shared `kb`
per example using:

- one cached example-level `reference_section/` under the experiment root
- a single-thread reference network generated by collapsing all hierarchy units
- the effective reference width `wid_adj_routing`, preferring the
  final reference-link summary in `link_width_families.csv` when available
- the configured percentile of raw width samples only as a fallback when the
  selected width field is not present in the family summary
- the section-scale slope derived from the most downstream and most upstream
  matched SWORD WSE anchors on that reference corridor
- the peak of the normalized forcing hydrograph as the reference discharge
- the BASED depth estimator to predict reference depth from `Q`, `W`, and `S`

The resulting scalar `kb = W_ref / D_ref` is stored in
`reference_section/reference_kb_summary.json`. RAPID prep reads that artifact
and reuses it for every state. RAPID does not run BASED again. This prevents a
later forcing-resampling choice from silently changing `kb`.

If `--kb-mode based_reference_section` is requested and the stored summary is
missing, prep fails with an instruction to refresh the upstream
equivalent-resistance products. The old model/width CLI options remain accepted
for command compatibility and provenance, but they do not trigger a new BASED
calculation inside RAPID prep.

This is an example-level parameterization choice. It is intended to preserve
comparability across variants, not to create a different `kb` for each state.

Relevant prep flags:

- `--kb-mode based_reference_section`
- `--kb-model-path /path/to/based_model_v2.ubj`
- `--kb-width-sample-field wid_adj_routing`

`wid_adj_routing` is read directly from the reference link's width-family
summary, so `--kb-width-percentile` should be omitted and is recorded as null.
The percentile option is required only when the requested field exists solely
as raw transect samples; there is no implicit P90 fallback.

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
  --forcing-resample-seconds 1080 \
  --auto-spinup-from-k \
  --spinup-k-multiplier 1.1 \
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

Current defaults:

- RAPID prep prefers `wid_adj_routing` as the parent-link width field for `K`
  preparation
- `kb-mode=based_reference_section` also defaults to
  `--kb-width-sample-field wid_adj_routing`

Routing then writes per-state RAPID outputs under `states/<state_id>/rapid/run/`
and can export outlet hydrograph summaries.

#### Planned Refactor: Move All Slope Construction To `network_variants`

The current `--slope-profile-mode reference_section` behavior in RAPID prep is
only a transitional bridge. The intended end state is that slope/profile
construction is done entirely during variant creation and cached with the state.

Implementation checklist:

1. Example-level reference profile in `network_variants`
   Build the smooth example-level `H_ref(s)` once from trusted SWORD anchors on
   the cached `reference_section/`.

2. Experiment-level cache artifacts
   Save the fitted `H_ref(s)` profile, anchor metadata, and reference-section
   stationing once at the experiment root so they can be reused by every state.

3. Base-state hydraulic context in `network_variants`
   Write the base-state node/profile heads and final link slopes into
   `states/S000_base/hydraulic_context/`.

4. Variant-state inheritance in `network_variants`
   For each derived state, inherit unchanged node stations and slope/profile
   values through `matched_parent_node_id` and link lineage where possible.

5. Local recomputation only on affected corridors
   Reproject only truly new child nodes and recompute only the corridor between
   the relevant anchor nodes instead of rebuilding the entire slope field for
   every state.

6. Compound-bubble enforcement stays in `network_variants`
   Apply the exact bubble boundary-drop enforcement during state hydraulic
   context generation, only on bubbles intersecting the affected corridor.

7. Final slope outputs written per state
   Write final `rapid_wse_for_slope`, `slope_used`, source-method fields, and
   diagnostics into each state's `hydraulic_context/` bundle before RAPID prep.

8. RAPID prep becomes a pure consumer
   `run_prepare_experiment.py` should read precomputed hydraulic context only.
   It should not fit `reference_section` profiles, derive slopes, or own any
   slope-related scientific choices for `network_variants` experiments.

9. Compatibility path becomes explicitly legacy-only
   Any RAPID-internal slope derivation should remain only as a fallback for old
   experiment folders that predate `hydraulic_context/`, not as the normal path
   for current workflows.

Expected result:

- slope assumptions live with the generated state geometry
- RAPID prep becomes much faster
- RAPID prep only handles routing parameter assembly, forcing normalization, and
  input writing

### Analysis Utilities

The reusable analysis helpers live in:

- `src/rapid_tools/analysis.py`

They are intended to be imported from the notebook instead of rewriting
analysis logic inline. The current utilities cover:

- per-experiment merged analysis tables from `rapid_run_registry.csv`,
  `rapid_prep_registry.csv`, `state_registry.csv`, `transition_registry.csv`,
  and base `collapse_ranking.csv`
- combined unit-plus-compound-bubble comparison tables
- outlet hydrograph translation/shape comparison against the base state
- Plotly summary figures for effect-signature and hydrograph overlays
- reach-level wave products for animation

Prep now also writes:

- `states/<state_id>/rapid/prep/rapid_reach_attributes.gpkg`
- `states/<state_id>/rapid/prep/rapid_node_attributes.gpkg`

Those geometry-preserving files are used by the wave-visualization helpers.

### Plotting Utilities

The notebook-facing plotting helpers live in:

- `src/rapid_tools/plotting.py`

They support multi-example timing plots built from a single list of example
IDs:

- `build_multi_example_timing_frame([...])`
- `build_geometry_context_frame(...)`
- `make_timing_scatter(...)`

The timing frame now merges outlet-shift metrics with lag-relative analysis
metrics, including:

- `delta_peak_time_hours`
- `delta_lag_to_inflow_peak_hours`
- `hydrograph_lag_vs_base_hours`
- `lag_to_inflow_peak_relative_to_base`
- `lag_to_inflow_peak_pct_shift_vs_base`
- `unit_to_bubble_lag_to_inflow_peak_relative`
- `unit_to_bubble_lag_to_inflow_peak_pct_shift`

The geometry context helper also exposes cleaner x-axis aliases:

- `length_fraction`
- `child_length_fraction`
- `width_fraction`
- `log_lambda`
- `Lambda_matched`
- `lambda_matched`
- `k_ratio_to_matched_base`

Its `plot_category` is a hydraulic/run-scope label, with mutually exclusive
definitions:

- `compound indep`: an independent run whose selected unit contains nested
  structure, but is not being represented by an enclosing compound domain;
- `compound unit compound`: a dedicated compound-bubble run;
- `single simple`: a closed simple unit outside a compound, on the main
  single-centerline branch;
- `part of compound unit`: a non-closed compound member evaluated in its
  complete enclosing compound domain; that member can itself be simple or
  compound;
- `nested unit`: a hydraulically closed unit inside a compound, off the main
  single-centerline branch.

The width/exposure metric is a bounded upstream-share metric on the base
directed graph rather than a direct width ratio. Plotting code:

- finds the first clean upstream mandatory edge using a dominator-based helper
- propagates width-based split shares downstream through the base DAG
- sums the shares that first enter the matched target corridor

The main plotting predictor uses the exposure factor in
`length_fraction * (-log10(Lambda_matched)) * width_fraction`. The factor is
aligned to the domain used to calculate `Lambda_matched`:

- a `closed_compound_unit` or `containing_compound_context` is the complete
  hydraulically closed compound between its outer terminals, so its exposure
  is `1.0`; its matched Lambda already describes that entire compound;
- a `closed_unit` nested inside a larger compound retains its routed
  base-network width share, because its Lambda describes only that subdomain.

For debugging and notebook diagnostics the geometry frame keeps:

- `width_fraction_of_total_raw`
- `width_fraction_of_total`
- `exposure_fraction_source`
- `upstream_reference_link_id`

`width_fraction_of_total_raw` is the uncapped routed-share diagnostic.
`width_fraction_of_total` is the capped, domain-aligned plotting exposure and
its source is recorded in `exposure_fraction_source`. The raw share should
already be bounded in `[0, 1]` apart from floating-point noise.

Notebook code can discover the currently supported plotting metric names
through:

- `supported_timing_x_metrics()`
- `supported_timing_y_metrics()`

Typical notebook imports:

```python
from rapid_tools.analysis import (
    build_combined_analysis_frame,
    build_parent_link_difference_timeseries,
    build_reach_timing_comparison,
    plot_corridor_difference_heatmap,
    plot_effect_signature,
    plot_hydrograph_overlay,
    plot_reach_timing_map,
    write_wave_comparison_mp4,
    write_wave_gif,
    write_wave_mp4,
)
```

`write_wave_mp4(...)` uses `ffmpeg` for video encoding. If `ffmpeg` is not on
your `PATH`, pass `ffmpeg_path=...` explicitly or install it in the active
environment before exporting MP4 files. For non-base states, `write_wave_mp4`
now defaults to a synchronized four-panel presentation layout:

- base state
- target variant
- signed difference (`variant - base`)
- zoomed difference panel

Use `compare_to_reference=False` if you want the old single-state MP4 instead,
or call `write_wave_comparison_mp4(...)` directly when you want to choose a
non-default reference state or zoom window.

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
physical configuration across all states in that example:

- one source forcing series; coordinated prep uses one forcing/routing timestep
  when all stability intervals overlap and the closest stable state-specific
  timesteps otherwise
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

## Event forcing limitations and discharge-shape audit — 5 October 2026

Reference **HYDROGRAPH-MEAN-SPREAD-2026-10-05**, added during interpretation of the all-pair predictor results and example 37's competing peaks. The current experiments route an upstream input hydrograph without observed lateral-inflow or tributary hydrographs. Internal branching/reconnection is represented, but added catchment inflow is not. Missing inputs can contribute to downstream discharge, shape and timing disagreement; they are not demonstrated to explain all disagreement. Gauge uncertainty, hydraulic assumptions, storage and event-window selection also matter.

The [mean-discharge and standard-deviation audit](validation/hydrograph_mean_spread_20261005/README.md) compares arithmetic upstream/downstream gauge means within shared observed windows, with sample counts, cadence/gap diagnostics and time-weighted means as a sensitivity check. Mean discharge differences are descriptive evidence of a discharge mismatch, not direct estimates of tributary flow or causal attribution. No independent-sample significance test is applied to autocorrelated hydrograph observations.

Discharge standard deviation is a candidate whole-hydrograph variability/smoothing descriptor. It is distinct from temporal spread, event duration and maximum peak attenuation. Its window dependence must be checked before adopting it as a response metric. Existing response definitions and routing remain unchanged.

The [standard-sampling TV audit](validation/standard_total_variation_20261005/README.md) extends the roughness diagnostic to all 2,214 pairs at 1-, 2- and 4-hour time-averaged sampling, with the user-proposed event windows recorded separately from unchanged production forcing.

## Buffer tests completed — 5 October 2026

[EVENT-BUFFERS-2026-10-05](validation/event_buffer_tests_20261005/README.md) audits all 2,230 existing routed states and tests cropped-event forcing on 39 selected states across all 16 examples. Full-waveform 0/6/24-hour spin-up comparisons agree within numerical precision under the current steady-flow initialization. A 0.1%-of-parent-amplitude settling criterion sustained for 24 hours plus a further 24-hour tail gives an estimated 83% reduction in aggregate simulated duration and 77% in segment-routing substeps. These are projections, not measured batch wall-time savings. Tested children can settle later than the parent, so every future child requires the final-settlement check. Production remains unchanged; the new windows/buffers are not yet activated.

Exact event and buffer timestamps are now documented in the review workbook's **Routing windows** sheet and the [routing schedule README](reviews/routing_windows/README.md), including the precise Excel cell locations. This records the tested proposal without activating new production forcing.

## Travel-time calibration notebook

[rapid_travel_time_calibration.ipynb](notebooks/rapid_travel_time_calibration.ipynb)
fits one Manning roughness value per original network to the selected event's
observed first-maximum travel time. It reads the reviewed Excel event windows,
keeps BASED depth, width–depth exponent, slopes, widths and Muskingum X fixed,
and routes only `S000_base`. A coarse n grid and local refinement retain
sampling ties, peak-interpretation notes and search-boundary warnings.

The workbook's shortened tail is checked and extended for each trial. A common
safe clock covers each search bracket; sparse-recurrence comparisons and native
segment flow checks accompany the trials. Cached candidates, hydrographs and
audits are isolated in `RAPID/outputs/travel_time_calibration/`. Production inputs
and the review workbook are unchanged. Separate-event validation and variant
reruns are outside this notebook's calibration scope.

Sections 9–12 refine the candidates within diagnostic bounds of 0.010–0.100
using bracketed lag residuals, stopping at the sampled-output timing scale.
They retain a separate 0.025–0.060 comparison band based on common natural-channel
examples. Published alluvial-bed guidance also permits lower roughness under
specific conditions; neither band establishes physical acceptability at a site.
The notebook links HEC-RAS, FHWA and USGS guidance and marks site roughness as
unverified. Refined outputs are isolated in
`RAPID/outputs/travel_time_calibration_refined/`, with timestep checks and the
forcing-peak offset reported separately from the fitted propagation lag.

## Constant-reference Muskingum–Cunge candidate

`rapid_tools.cunge` prepares the reviewed anchored hydraulic link frames with
`alpha=2`, exact physical lengths, link-specific reference discharge, and
`K=dx/c`, `X=(1-Q/(B*S*c*dx))/2`. It freezes the numerical grid across the
Manning-n search; each trial updates celerity, K and X together. Manning n is
the only calibrated parameter. Negative X is allowed under the explicit
`signed_inflow` policy, with native discharge and undershoot audits.

The engine's opt-in `upstream_two_time` boundary includes both inlet time
levels. Existing prepared states retain their legacy boundary by default.
The new numerical prep writes runnable connectivity, K/X, forcing NetCDF,
link mapping and manifests directly from the audited hydraulic frames. The
mapping preserves physical anchor nodes; source GIS geometry is retained in
its original state directory rather than rebuilt.

The parent-only assessment is reproducible with:

```sh
python RAPID/investigations/cunge_calibration.py --workers 2
python RAPID/investigations/cunge_validation.py select
python RAPID/investigations/cunge_validation.py independent --workers 2
python RAPID/investigations/cunge_validation.py prepared --workers 2
python RAPID/investigations/cunge_report.py
```

Use an environment with NumPy, pandas, SciPy, NetworkX, PyArrow, openpyxl and
the existing RAPID analysis dependencies. Results and new prep bundles live
under `RAPID/outputs/cunge_calibration_alpha2/`; `REPORT.txt` distinguishes
calibration, numerical checks and automatically screened independent events.
The fixed-X comparison uses the same new grid and corrected inlet, so it is
a controlled comparison of X formulations rather than a reproduction of the
historical 200 m production setup. Existing results and production defaults
are retained. Child runs are outside this assessment.

### Reviewed Cunge batch storage (2026-10-09)

The `cunge_full_hierarchy_reviewed.py --master-run prep/run` CLI publishes
completed products beside their geometry under
`<example>_<family>__anchored/states/<state>/rapid/cunge/{prep,run}/`.
One canonical state owns each unique hydraulic result; aliases have
`rapid/cunge/reference.json`, and their registry rows point to the canonical
product. Shared forcing is stored at `<example>_indep__anchored/rapid/cunge/`.
The active `rapid_prep_registry.csv` and `rapid_run_registry.csv` identify Cunge
and the compact outlet format explicitly. Outlet CSVs are supplied for the
existing hydrograph loader; reach-time NetCDF products are not generated.

Before active registries are replaced, their original versions and manifests
are copied into each experiment's `rapid/legacy_constant_x/`. Existing legacy
state `rapid/prep/` and `rapid/run/` products stay in place, preserving archived
registry paths. No legacy discharge files are deleted. Batch folders retain
checksummed computation outputs and recovery receipts; state publication is
recorded in `reports/state_publication_{prep,run}.json`.

For already-completed batches, `--master-run publish --manifest <manifest>`
performs this placement without rerouting. Publishing a different run over an
existing Cunge product is refused. Smoke tests with `--max-states` do not publish
partial preparation into active experiment registries.

### Batches including fully collapsed reference sections

Use `RAPID/run_cunge_batch.py prep --examples ...` followed by the same entry
point with `run --examples ... --workers 2` for future batches. This wrapper
runs the existing network-family workflow and one fully collapsed Cunge case
per selected example. It does not modify earlier/running batch manifests or
their routing implementation. The lower-level `--master-run` entry point is
network-family-only; use the new wrapper for inclusive future batches.

For previously completed or running network examples, add `--centerline-only`
to both commands. This prepares/routes only their reference sections and does
not repeat the network families. Reference products live at
`<example>_indep__anchored/reference_section/rapid/cunge/{prep,run}/`, with a
reference-section run registry. Existing legacy outputs are retained.

The collapsed geometry uses its measured routing width and the reference depth
verified against the current base's anchored depth relationship. Its slope is
the current base inlet/outlet head drop divided by collapsed geometry length;
the old reference-section slope summary is not reused. Manning n, beta,
reference discharge, subdivision policy and initial flow are inherited from
the frozen calibration. Native time steps respect the same stability rule and
are no coarser than an available published Cunge network clock; forcing and
outlet means use the common 60-second grid. Reference cases are comparison
runs, not separately recalibrated fits. Different native steps and trailing
settling windows are recorded, while event scoring retains the observation
window. All 15 reference sections passed input preparation; isolated routing
checks cover 03, 46 and 54.

### Accepted-threshold hydrograph response notebook

Use [hydrograph_response.ipynb](notebooks/hydrograph_response.ipynb) to inspect the
accepted horizontal thresholds and six event-excess response measures. All plots
use Plotly; the threshold explorer includes the base, single centerline, optional
partial-collapse child, and both gauges. Run with the `river-hierarchy-rivgraph`
Python environment. The notebook replaces the dated hydrograph response worked
examples notebook.

Calculated tables and input provenance are in `outputs/hydrograph_response/`.
The current catalogue includes 2,187 comparisons against 15 bases. Set
`RECALCULATE = True` in the notebook to rebuild metrics from saved hydrographs;
this does not rerun routing. Implementation: `src/rapid_tools/hydrograph_response.py`.
Older investigation scripts that append to the removed dated notebook are
historical workflows, not the generator for this notebook.

### Collapse interactions: individual impacts are not additive (2026-10-20)

**Do not estimate a combined collapse response by summing individual-island
or member-unit responses.** Each individual state measures an edit against its
parent network. The same edit can have a different effect after other islands
have been removed. A compound collapse must therefore be evaluated as its own
geometry and routed state.

#### Measured example: example 15

The section from nodes 412/411 to 291/293 contains 61 islands. The accepted
intermediate selection retains islands **155, 157, 160, 170, 188, 194, 196,
199, 207 and 210**, removing the other 51. Island 150 at the downstream
boundary is included; its collapse requires adjacent geometry adjustments
beyond that boundary.

The following are whole-network outlet **excess-volume centroid shifts**,
using the finalized shared **0.5% analysis window**. Negative means earlier
(child minus base). Joint responses were obtained from actual combined RAPID
runs, not sums or RF predictions.

| Experiment or diagnostic | Centroid shift (minutes) |
|---|---:|
| Collapse all selected 51 islands together | -26.89157 |
| Arithmetic sum of those 51 individual-versus-base shifts (diagnostic only) | -12.64319 |
| Collapse only islands 202, 204, 180 and 203 together | -3.65687 |
| Collapse the other 47; retain those four as well | -22.15817 |
| Full section collapse: all 61 islands | -86.31588 |

The four tested islands are the strongest standalone advances among the 51;
no individual advance exceeds 1.36 minutes. Keeping these four reduces the
joint advance by only **4.73340 minutes**, leaving **82.4%** of the original
advance. This supports cumulative effects and interactions among many edits,
rather than dominance by those four candidates. It is not a complete
per-island conditional attribution.

For disjoint edit sets A and B, define non-additivity relative to the same
base as `I(A,B) = shift(A∪B) - shift(A) - shift(B)`. For the four-versus-47
partition, `I = -1.07653 minutes`. Across all 51 individual comparisons, the
joint shift differs from their arithmetic sum by **-14.24838 minutes**.
These are different diagnostics; neither assigns a unique contribution to
each island.

Context can also change the sign: collapsing island 155 alone delays the
centroid by 2.13 minutes, whereas adding its collapse to the other 51 edits
increases the advance by 2.61 minutes (the previous nine-island-retained
version was -29.50324 minutes).

#### Implications for compound and individual-state comparisons

- The existing compound/member/island states provide a place to investigate
  this effect systematically, but the example above is **not** a completed
  interaction audit across all compounds or networks.
- Compare against the same parent, forcing, calibration, outlet and analysis
  window. Confirm that a proposed compound corresponds to the union of the
  individual edits. Matching an island to an anchor unit does not establish
  equivalent geometry: a unit collapse can remove several islands, nested
  units overlap, and aliases can duplicate the same physical experiment.
- Combined edits can change centerlines, widths, reconstructed heads and
  flow partitioning together. The measured non-additivity belongs to this
  complete geometry-and-routing workflow; these tests do not isolate its
  geometric and hydraulic mechanisms.
- Do not sum RF predictions for individual islands or domains as a validated
  joint prediction. Compute descriptors for the actual combined child in a
  consistent comparison domain and identify applications outside the validated
  training setup as exploratory.

Evidence and reproducibility:

- [Accepted intermediate and full-collapse results](outputs/example15_keep_ten_20261020/comparison.csv),
  with intermediate [routing results](outputs/example15_keep_ten_20261020/middle/results.json)
  and [island-preservation audit](outputs/example15_keep_ten_20261020/island_preservation_audit.json).
- [Interaction calculations](outputs/example15_attribution_20261020/attribution.json),
  [individual-island input table](outputs/example15_attribution_20261020/removed_island_individual_impacts.csv),
  and [diagnostic plot](outputs/example15_attribution_20261020/attribution.png).
- New diagnostic runs and their build/routing scripts:
  [retain the four candidates](outputs/example15_attribution_20261020/retain_top_four/middle/results.json)
  and [collapse only those four](outputs/example15_attribution_20261020/collapse_top_four_only/middle/results.json).
  Baseline reproduction, cycle counts, island-mask preservation and halved
  time-step checks accompany these runs.

#### Example 15: physical check of the depth-informed prediction (2026-10-26)

For the same ten-island-retained geometry under the uncapped recalibrated
setup, the outlet centroid shift is **-27.61636 minutes**; the geometry RF
predicts -37.87563 and the depth-informed RF -52.64480. Controlled diagnostic
refits reproduce both saved models. Changing only forest settings gives
-36.76452; adding only `delta_G_inverse_depth` to the geometry inputs with
the original settings gives -52.73298. This identifies the new feature as
the main local model-behavior driver, not a causal hydraulic attribution.

Channel terms and internal hydrograph tracing establish:

- Main compound 3 (nodes 403–280) supplies **31.71218 minutes** of the
  **33.96472-minute** reduction in flow-weighted mean transport time.
  Compound 2 supplies 2.25254 minutes; retained island 210 and the exterior
  network have unchanged transport terms. An exact symmetric product
  decomposition attributes 5.49605 minutes of compound 3's reduction to
  shorter mean route length and 26.21612 to lower mean time per unit length.
  The latter combines depth, slope and changed allocation; those causes
  have not been independently separated.
- `Gh = sum(r L / (h sqrt(s)))`, whereas the actual uncapped Cunge mean
  transport time is `(3 n / 5) sum(r L / (h^(2/3) sqrt(s)))`. No celerity
  caps are active in these frames. The depth parameter is still derived
  from width, not independently observed bathymetry. In compound 3,
  length/allocation-weighted mean depth increases 31.0%; Gh falls 32.5%,
  but actual mean transport time falls 24.7%. Thus the inverse-depth
  exponent gives a stronger relative sensitivity than the routing law
  for these edits. This does not quantify a unique share of the RF error.
- The event centroid is already **27.96104 minutes earlier at node 280**,
  versus **27.61636 minutes earlier at the final outlet**. Large downstream
  cancellation is not supported. The mean transport reduction itself
  persists downstream; the event centroid is a different statistic.
- Translating the base hydrograph earlier by the calculated 33.96472-minute
  mean transport reduction, then recalculating excess-volume centroid in
  the same fixed shared window, gives **-27.95306 minutes**, within 0.33670
  minutes of the actual routed result. This diagnostic illustrates why a
  transport-time change is not automatically the same event-centroid change
  for this waveform, baseline and window; it is not a new physical model.

The evidence supports excess sensitivity of the proxy and an imperfect learned
mapping from transport changes to this event response, rather than strong
downstream attenuation. It does not establish that replacing the exponent
will improve held-network prediction; that would require separate validation.
Boundary traces reproduce the saved whole-network outlet curves to 1e-8 cms.

See [controlled model comparisons](outputs/example15_uncapped_check_20261026/diagnosis/comparison.csv),
[physical findings](outputs/example15_uncapped_check_20261026/physical_diagnosis/findings.json),
[domain decomposition](outputs/example15_uncapped_check_20261026/physical_diagnosis/mechanism_summary.csv),
[boundary changes](outputs/example15_uncapped_check_20261026/physical_diagnosis/boundary_changes.csv),
and [diagnostic figure](outputs/example15_uncapped_check_20261026/physical_diagnosis/physical_diagnosis.png).
Channel-level terms, boundary hydrographs and reproduction scripts are stored
in that same `physical_diagnosis` directory. Production models are unchanged.

#### Example 15: depth-assumption sensitivity with fixed calibration (2026-10-26)

The base, ten-island-retained middle and full-section-collapse geometries were
routed under five one-at-a-time assumptions:
`h = scale * 2.7970428 m * (W / 878.7956754 m)^alpha`.
Reference is `scale=1, alpha=0.65`; scale was varied to 0.75/1.25 and alpha
to 0.45/0.85, separately. These are illustrative sensitivity brackets, **not
measured depth ranges or uncertainty intervals**. Geometry, widths, slopes,
flow allocation, forcing, Manning n=0.014309465 and beta=0.777460987 were
fixed. Numerical subdivision and clock retained stability; no recalibration
or RF refitting was performed.

| Depth assumption | Middle centroid shift (min) | Full centroid shift (min) | Middle / full magnitude |
|---|---:|---:|---:|
| Reference | -28.31 | -99.36 | 28.5% |
| Depth scale 0.75 | -33.89 | -118.58 | 28.6% |
| Depth scale 1.25 | -24.49 | -85.66 | 28.6% |
| Exponent 0.45 | -18.36 | -66.92 | 27.4% |
| Exponent 0.85 | -41.53 | -138.14 | 30.1% |

Negative means earlier. These comparisons use one shared 0.5% window covering
all 15 scenario/network combinations. Its end extends 72.77 hours beyond the
previous window; this changes the reference scores above. In the original
frozen window, the same reference curves reproduce **-27.61636 / -93.17878
minutes** exactly. Both scoring versions are saved; do not mix them.
All runs pass the 96-hour settling guard, use 336-hour tails, and halving the
3-second step changes response shifts by less than 0.004 seconds.

Within this limited test, retaining ten islands while removing 51 preserves
the contrast: the middle has only 27–30% of full-collapse centroid impact.
Absolute minutes remain depth-assumption-sensitive. This supports a relative
claim for this example, not a claim of negligible impact, an uncertainty bound,
robustness to simultaneous parameter changes, or validation against real depths.

Evidence: [scenario results](outputs/example15_depth_sensitivity_20261026/responses.csv),
[contrast summary](outputs/example15_depth_sensitivity_20261026/contrast_summary.csv),
[figure](outputs/example15_depth_sensitivity_20261026/depth_sensitivity.png),
and [settings, provenance and checks](outputs/example15_depth_sensitivity_20261026/manifest.json).

#### Depth sensitivity versus prediction error: comparison cases (2026-10-26)

The tuned ten-input model's saved nested held-network scores are pooled
equal-network R²=0.81338, regular R²=0.77045 and segmented R²=0.57199.
Pooled performance must not be described as segmented R²≈0.8.

Whole-network segmented collapses of examples 21 and 54 are accurately
predicted (4.01% and 4.64% relative error), while example 15 has 72.76%
error. These are whole-network states, **not the selected example15 local
section or its middle version**. Actual/predicted advances are respectively
106.37/110.64, 116.12/110.73 and 286.09/494.26 minutes.

Matched new exponent sensitivity runs (0.45/0.65/0.85, other settings fixed)
show all three cases are sensitive. Relative to each case's 0.65 reference,
the centroid advance changes by -25.7%/+28.6% for 21, -21.1%/+23.2% for 54,
and -28.5%/+34.7% for 15. Thus 15 is more sensitive in this comparison, but
the difference is moderate in relative terms; sensitivity alone does not
explain its much poorer prediction. Sensitivity percentages use the shared
expanded window within each network; validation errors use their original
published windows. Original reference curves were reproduced, and half-step
checks differed by less than 0.014 seconds.

The width/depth contrast helps explain sensitivity. Whole-network
flow-and-length-weighted widths change from 406→880 m in 15, versus 632→743 m
in 21 and 195→247 m in 54. Corresponding weighted estimated depths increase
70.2%, 11.7% and 17.3%. In 15, much of the original routing is through channels
well below its 879 m reference width, whereas the collapsed network is much
closer to that anchor. Increasing the width-depth exponent therefore slows
the parent much more than the child. For uncapped transport the exact
identity is `d ln(T)/d alpha = -(2/3) E_T[ln(W/Wref)]`, with travel-time weights.
At alpha 0.65 the parent elasticities are 0.758 for 15,0.217 for 21 and0.212 for 54.

Do not overextend the earlier inverse-depth-exponent explanation into a unique
explanation of RF error: a parent-scaled Gh proxy also overstates mean
transport changes in the accurately predicted peers. Calibration and event
context differ as well (Manning n: 0.01431/0.02434/0.03002 for 15/21/54), and
the RF does not explicitly receive n. These are explanatory leads, not a
causal decomposition or evidence to add a calibrated parameter as a predictor.
Three deliberately selected cases do not establish a population-wide
relationship between sensitivity and prediction error.

Evidence: [peer summary](outputs/example15_depth_peer_comparison_20261026/peer_summary.csv),
[all 15 segmented validation results](outputs/example15_depth_peer_comparison_20261026/all_segmented_validation.csv),
[comparison figure](outputs/example15_depth_peer_comparison_20261026/peer_comparison.png),
and [methods and limitations](outputs/example15_depth_peer_comparison_20261026/manifest.json).

### Physical context of remaining errors and closed-unit augmentation (2026-10-20)

The aim is useful prediction with understood limitations, not eliminating every
residual. This audit keeps the nine-input model and routed results unchanged.
It uses the `same_grid` **held-out-example** predictions, not fitted training
predictions. Negative centroid shifts are advances.

| Segmented case | Modelled centroid shift (h) | Predicted (h) | Main evidence |
|---|---:|---:|---|
| 46 | -23.09 | -11.59 | Edited parent size 435.61 km exceeds the held-out training maximum 290.21 km; its response also exceeds the training range. |
| 55 | -17.28 | -7.95 | Only one training case has an equally large advance; relatively slow hydraulic context compounds sparse support. |
| 05 | -6.53 | -2.14 | Large enough responses exist in training, but its parent routing is much slower than several geometrically nearby cases. |
| 22 | -8.87 | -14.10 | Large edited size does not imply equally large response: its routing is comparatively fast. |

**New physical check.** Across the 15 segmented examples, integrate only edited
links, retaining their network visit fractions. The diagnostic travel time is
`T = sum(r_e L_e / c_e)`; this is not the finite-window excess-volume centroid.
A counterfactual using the same length/slope but fixed `n = 0.03` and `h = 3 m`
separates geometric length/slope from hydraulic scale. Actual parent travel time
divided by this standardized diagnostic is 1.79 for 05, 1.53 for 55, 0.40 for 22,
and 0.50 for 41. The absolute reference constants are arbitrary; their common
use makes these relative contrasts informative. This is an algebraic diagnostic,
not a new RAPID run or a controlled causal attribution of RF error.

The parameter tables also confirm `h_e = a W_e^0.65` within each parent, with
variation in `a` at floating-point precision within an example. Consequently,
uncapped Manning wave routing admits the decomposition

`G = sum[r_e (L_e / 1 km) (W_e / 100 m)^(-13/30) / sqrt(s_e)]`,

`T = A_example G`,

where `G` is dimensionless and geometric, while `A_example` incorporates
roughness, the depth-width scale and unit conversions. Using the parent-derived
`A_example`, this reconstructs all 15 segmented parent-child travel-time changes
to within `1.1e-14 h`. It does **not** reconstruct the centroid response exactly.
The coefficient is 2.29 times larger for 55 than 22: the same change in this
geometric quantity would produce substantially different travel-time changes.

This identifies two distinct limitations: separate aggregate length/width/slope
inputs do not explicitly describe their link-by-link association, and geometry
alone does not specify each example's calibrated hydraulic scale. A bounded
follow-up is to test the geometric `G` change or absolute parent mean width,
one at a time, against the frozen baseline. This is a proposal, not a demonstrated
prediction improvement. Keep `n`, modelled depth, celerity and `A_example` as
explanatory diagnostics, not hydrography-only predictors. More same-network states
cannot remove all between-network hydraulic differences.

The partial-collapse audit also remains relevant: 22 CB9 and CB11 account for
about 44% of the network-weighted partial squared error; their fast routing
contrasts with the slower large-impact cases reaching similar RF leaves. In
41 unit126, a slightly shorter domain nevertheless becomes slower (effective
celerity about 1.88 to 1.44 m/s), consistent with its positive centroid shift.
These are better reasons for a targeted context check than indiscriminate
addition of predictors. They do not prove that any proposed geometric proxy
will generalize to a held-out network.

#### What the interaction evidence means for augmentation

The example15 island interaction runs above use **the historical 5 m/s cap and
old calibration** (`n=0.014309`, `beta=0.777461`). Their evidence of non-additivity
is valid for that setup. Do not mix those labels directly into the current
uncapped benchmark; reproduce selected comparisons under the current setup.

For the six existing **closed-block** pair pilots, compare actual joint runs
with their matching constructed single-control runs, using the same baseline
and window. At 0.5%, `I(A,B) = shift(A+B)-shift(A)-shift(B)` ranges from
**-0.12272 to +0.00839 minutes** (-7.36 to +0.50 seconds). The largest discrepancy
is the 22 compound+compound pair, whose joint shift is -119.47 minutes.
The discrepancies remain small at 1% and 2%. These historical capped pilots
are almost additive for centroid timing, unlike the island-edit example.
This does not establish additivity for other responses or new geometries.
Separate closed blocks preserve connectors and boundary conditions by design;
member/island edits can also alter shared geometry. That is a plausible reason
for the contrast, not an isolated mechanism demonstrated by this audit.

#### Concrete augmentation investigation

1. **Keep the geometry restrictions.** Use contiguous, disjoint, fully closed
   simple/full-compound blocks. Keep connector reaches in routing but exclude
   them from edited-domain predictors. Use the restored boundary anchors from
   the segmented representation. Validate combined closure, aliases, unchanged
   reaches, allocation and numerical stability before routing. Do not create
   overlapping compound members or duplicate the full segmented state.
2. **Test size coverage, not arbitrary sample inflation.** The saved adjacency
   inventory gives 420 pairs, 405 triples and 10,805 contiguous combinations
   in total (including full-state duplicates, excluded from proposals).
   A geometry-selected pair/triple pilot contains 43 proposed new states over
   all 15 examples. However, the largest triples in 05/55/46 are only
   14.30/29.87/37.19 km, versus full edited sizes 73.25/198.51/435.61 km.
   A separate selection therefore proposes 41 longer chains nearest 25%, 50%
   and 75% of each example's edited length; actual coverage is recorded rather
   than assumed. These are an inventory and staged proposals, not routed jobs.
3. **Stage the work.** First reproduce a small matched A/B/A+B closed-block
   interaction check with current parameters, alongside selected example15
   interaction comparisons. Then prioritize the approximately 50%-size chains
   across examples to bridge partial and full collapses. Add the 25%/75%
   selections only if that first augmentation shows useful coverage or benefit.
   Large chains can contain dozens of blocks (55's half-size chain has 20;
   46's has 40), so pairs alone cannot answer the size question.
4. **Route genuine combined states.** Use the current per-example calibration,
   forcing, baseline and shared 0.5% window, retaining 1%/2% sensitivity. Check
   tail coverage and any required shared-window extension before response
   comparison. Calculate all six responses from actual joint hydrographs;
   never sum labels or RF predictions. For interaction attribution, both
   component controls must use exactly the geometry employed in the joint run.
5. **Test augmentation separately from new predictors.** First keep the current
   nine predictors and settings fixed. In each held-out-example fold, add only
   training-network combinations; hold out every state of the test example.
   Compare on the identical original and segmented test rows, reporting the
   augmented states separately. Preserve equal network weight and check how
   extra rows redistribute weight between original, combined and segmented
   training cohorts. Report R², floored relative error and worst-case changes;
   do not claim success from training error or improvement of one extreme case.
6. **Bound the claim.** These states provide no new independent rivers. They
   can fill within-range gaps but cannot guarantee RF extrapolation to example46
   when all of 46 is excluded. No remaining training network has its edited
   length. Accept residual variation if the remaining geometric information
   does not reliably explain it; avoid extending the search just to fit outliers.

Reproducible audit:
[script](investigations/cap_investigation_20261020/physical_context_and_combinations.py),
[physical diagnostics](outputs/uncapped_event_window_20261020/physical_context_and_combinations/segmented_physical_context.csv),
[pair interactions at all three windows](outputs/uncapped_event_window_20261020/physical_context_and_combinations/existing_pair_interactions.csv),
[pair/triple proposals](outputs/uncapped_event_window_20261020/physical_context_and_combinations/proposed_pilot.csv),
[longer-chain proposals](outputs/uncapped_event_window_20261020/physical_context_and_combinations/proposed_scale_extension.csv),
[complete adjacency inventory](outputs/uncapped_event_window_20261020/physical_context_and_combinations/combination_inventory.csv),
[source hashes and limitations](outputs/uncapped_event_window_20261020/physical_context_and_combinations/manifest.json).
The audit verifies shared comparison windows and unchanged source hashes; no
production outputs, calibration, response definitions or fitted models changed.

### Two half-size chain information tests (2026-10-20)

Completed the proposed bounded check on the geometry-selected approximately
half-size chains in **22 (5 closed blocks, 144.07 km)** and **55 (20 closed
blocks, 99.30 km)**. Each uses actual current parent/segmented parameters and
restored section endpoints. The experiments include all individual component
controls, the complete joint collapse and a partition into two contiguous
subgroups. In total, 35 routing runs include two reproduced parents and two
joint runs at half time step. Unchanged reaches and their allocations were
verified; reversing composition order produces identical parameters.

The current nine-input forest was reconstructed separately with **all of the
corresponding example excluded**, using its existing selected settings. Saved
held-example predictions were reproduced to floating-point precision. No new
states entered training, no tuning was performed, and no aggregate performance
score was recalculated on an enlarged test set.

All values below are **centroid shifts in hours**, child minus the common parent:

| Example | Joint RAPID | Sum of individual RAPID | Joint RF | Sum of individual RF | Sum of two subgroup RF |
|---|---:|---:|---:|---:|---:|
| 22 | -4.14293 | -4.13117 | -7.72374 | -9.19209 | -8.55593 |
| 55 | -8.52810 | -8.48069 | -5.19964 | -4.77083 | -3.08773 |

At 0.5%, joint-minus-summed individual RAPID centroid shifts are **-0.706 min
(0.284% of the joint shift magnitude)** in 22 and **-2.845 min (0.556%)** in 55.
Against the two subgroup responses, the discrepancies are -0.287 and -1.423 min.
Thus both chains are nearly additive for this response at these scales.
The joint half-time-step centroid differences are about **-0.81 and -0.83 s**;
this check concerns joint-run convergence, not a full convergence bound on
all summed controls. At 1%/2%, the individual-sum discrepancies are approximately
-0.75/-0.72 min in 22 and -3.89/-5.48 min in 55. No shared analysis window or
baseline needed changing, and both parent outlet hydrographs reproduced exactly.

**The RF does not share this near-additivity.** Its joint-minus-summed individual
predictions differ by +88.10 min in 22 and -25.73 min in 55. Predicting two
subgroups instead of all individual blocks gives yet another sum. These are
aggregation inconsistencies of this fitted predictor, not evidence of physical
interactions. The joint prediction remains substantially wrong in both cases:
about 3.58 h too much advance in 22 and 3.33 h too little in 55. Independent
component predictions are also biased in the same direction; summing them does
not solve these context differences.

**Response-specific qualification:** peak-time shifts are not uniformly additive.
In 22 the joint peak shift is -6.2333 h versus -5.7667 h summed over components,
a **28-minute** difference; in 55 the difference is 2 minutes. Changes in global
peak timing need not obey the approximate centroid additivity. All six responses
are saved. Arithmetic sums of percentage or shape changes are only diagnostics,
not assumed conservation laws; no RF magnitude/shape predictions were made here.

**Decision:** these two chains add little independent centroid interaction
information. Do not launch a large combinatorial expansion or treat their count
as more independent evidence. They are useful diagnostics of scale/aggregation
behaviour and could be training augmentation, but that claim needs a separate
controlled comparison on the **identical original and segmented held-example
test cases**. Hold all states of each test example out and control training
weights. Near-additive augmentation does not automatically inflate such an
independent test score, but scoring many correlated augmented states together
can change the apparent result and must not be presented as more independent
validation. More closed chains also do not reproduce the shared-geometry
interactions seen in the example15 island/member construction.

Evidence:
[reproducible preparation/routing/analysis script](investigations/cap_investigation_20261020/combination_information_pilot.py),
[summary](outputs/uncapped_event_window_20261020/combination_information_pilot/summary.csv),
[all response additivity and numerical checks](outputs/uncapped_event_window_20261020/combination_information_pilot/routing_additivity.csv),
[component identities](outputs/uncapped_event_window_20261020/combination_information_pilot/variants.json),
[held-example predictions](outputs/uncapped_event_window_20261020/combination_information_pilot/predictions.parquet),
[window checks](outputs/uncapped_event_window_20261020/combination_information_pilot/window_audit.csv),
[parent reproduction](outputs/uncapped_event_window_20261020/combination_information_pilot/parent_reproduction.csv),
[model reproduction](outputs/uncapped_event_window_20261020/combination_information_pilot/model_reproduction.csv),
[completion](outputs/uncapped_event_window_20261020/combination_information_pilot/completion.json).

### Controlled 41-chain augmentation test (2026-10-20)

**Completed; retain the unaugmented baseline.** The 41 preselected contiguous
closed-unit combinations were routed, covering 14 networks. Example37 has only
three eligible outer blocks, so there is no distinct longer-chain candidate
under this selection rule; its full segmented state was not duplicated.
Targets are approximately 25/50/75% of edited parent length, with actual fractions
recorded. Example02 has two distinct selections after deduplication. The two
previous half-size pilot chains are included in the 41, not extra training rows.

Construction reuses current restored segmented sections and current per-example
calibration, preserving connectors and other unchanged reaches exactly. All 41
new states settled; all 14 parent reruns reproduced their existing hydrographs
exactly. Common 0.5%, 1% and 2% response windows and baselines required **no
changes**. There are zero unmatched records, and all 2,185 existing evaluation
rows retain exactly the same targets. The two earlier pilot responses also
reproduce. All six responses are saved, but the forest still trains only on
full excess-volume centroid shift.

Three treatments were evaluated using the same nine predictors, seed and
previously selected `same_grid` settings **within each held-example fold**:

- Baseline: existing 2,170 partial + 15 segmented observations only.
- Aug10: added combinations share 10% of each training network's total weight;
  previous observations share 90%, preserving their relative weights.
- Aug25: corresponding split 25%/75%.

Each training network keeps equal total weight. Example37 retains 100% on its
existing rows because it has no augmentation. All observations and combinations
of a held-out example are excluded from training. No hyperparameter tuning,
feature changes or production model replacement occurred. These are 45 fits
(15 folds x 3 treatments), with baseline predictions reproduced to floating-point
precision. The principal evaluation uses only the identical original/segmented
rows; combined-state predictions are stored separately, not pooled into scores.

| Training | Partial R² | Partial floored relative error | Segmented R² | Segmented floored relative error |
|---|---:|---:|---:|---:|
| Baseline | 0.6386 | 51.68% | 0.5100 | 36.07% |
| 10% combination weight | 0.6330 | 51.50% | 0.4273 | 38.64% |
| 25% combination weight | 0.6453 | 51.59% | 0.4063 | 40.02% |

Relative errors use the existing one-minute denominator floor and equal network
weight within each cohort. They are not unbuffered conventional MAPE. The mean
of the two cohort relative errors is 43.88% baseline, 45.07% Aug10 and 45.80%
Aug25. Partial MAE is 2.554, 2.607 and 2.623 min respectively; segmented MAE is
169.60, 171.92 and 170.79 min. Thus the tiny partial R² gain at 25% is not a
consistent improvement across metrics or cohorts.

There are meaningful improvements for some held-out full collapses:

| Example | Actual centroid shift (h) | Baseline prediction | Aug10 | Aug25 |
|---|---:|---:|---:|---:|
| 05 | -6.535 | -2.145 | -3.781 | -3.925 |
| 13 | -6.979 | -5.069 | -6.417 | -6.264 |
| 15 | -4.768 | -6.381 | -4.866 | -4.989 |
| 22 | -8.867 | -14.098 | -14.090 | -13.819 |
| 46 | -23.094 | -11.593 | -10.468 | -9.996 |
| 55 | -17.279 | -7.951 | -6.342 | -6.101 |

Aug10 improves segmented absolute error in 7/15 networks; Aug25 in 8/15, but
losses outweigh gains in the aggregate measures. Notable additional regressions
are 21 (baseline -2.109 h, Aug25 -3.538 h, actual -1.773 h) and 54 (baseline
-2.149 h, Aug25 -3.259 h, actual -1.935 h). The segmented degradation persists
when 46 is excluded from scoring: R² is 0.5078 baseline versus 0.4402/0.4399,
and relative error is 35.09% versus 37.50%/38.82%. It is not solely an effect
of the largest outlier.

**Interpretation:** additional combinations can change what the forest learns
about context and scale; the gains for 05/13/15 demonstrate a useful effect in
some held-out cases. However, this specific geometry-selected augmentation and
weighting does not improve the overall existing benchmark. It does not establish
that all possible augmentation is ineffective. More intermediate observations
and their increased weight can change tree partitions and predictions; the
mechanism behind each regression has not been isolated. Do not tune additional
weights or select chains retrospectively just to repair these test cases.
The current unaugmented model remains the reference, and all new runs remain
available as research data. With only 15 independent networks, these are
exploratory paired results, not independent external confirmation.

Evidence:
[script](investigations/cap_investigation_20261020/chain_augmentation.py),
[design and hashes](outputs/uncapped_event_window_20261020/chain_augmentation/design.json),
[selected components](outputs/uncapped_event_window_20261020/chain_augmentation/variants.json),
[scores](outputs/uncapped_event_window_20261020/chain_augmentation/scores.csv),
[network-level changes](outputs/uncapped_event_window_20261020/chain_augmentation/network_changes.csv),
[segmented predictions](outputs/uncapped_event_window_20261020/chain_augmentation/segmented_predictions.csv),
[training-weight audit](outputs/uncapped_event_window_20261020/chain_augmentation/weight_audit.csv),
[response/join checks](outputs/uncapped_event_window_20261020/chain_augmentation/response_check.json),
[previous pilot reproduction](outputs/uncapped_event_window_20261020/chain_augmentation/previous_pilot_reproduction.csv),
[completion](outputs/uncapped_event_window_20261020/chain_augmentation/completion.json).
Reproduce/resume with `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python RAPID/investigations/cap_investigation_20261020/chain_augmentation.py --run`
in the `river-hierarchy-rivgraph` environment. Existing completed routing and
model folds are verified and reused. `--run --models-only` skips routing and
response calculation after those phases have completed.

### Equal-case chain augmentation control (2026-10-20)

Completed the missing simple-appending control using the same 41 routed chains.
The baseline `fit` function was applied directly to concatenated training rows:
**every case within a network gets equal weight, and every training network
retains equal total weight**. No augmentation weight is reserved. All states
of the held-out example remain excluded. Same nine predictors, seed and frozen
per-fold `same_grid` settings; 15 new fits, no retuning or new routing. Evaluation
uses exactly the same 2,185 original/segmented rows and unchanged response values.
Historical baseline/10%/25% predictions were reused with source hashes and exact
test-key/target checks, rather than refitted unnecessarily.

The natural combined share is `m/(N+m)`, where N is the existing case count and
m the appended chain count. It is 0.73% in 22, 1.32% in 46 and 1.39% in 55.
It is not uniformly below 10%: example56 has 22 existing cases and 3 chains,
so its combined share is 12%. Example37 has no appended chains. The audit checks
equal per-case weight within each network, equal network totals, and exclusion
of the held-out network from every training fold.

| Training | Partial R² | Partial relative error | Segmented R² | Segmented relative error | Segmented MAE (min) |
|---|---:|---:|---:|---:|---:|
| Baseline | 0.6386 | 51.68% | 0.5100 | 36.07% | 169.60 |
| Equal-case append | 0.5995 | 52.08% | 0.4861 | 37.19% | 163.86 |
| 10% combination weight | 0.6330 | 51.50% | 0.4273 | 38.64% | 171.92 |
| 25% combination weight | 0.6453 | 51.59% | 0.4063 | 40.02% | 170.79 |

Relative errors retain the one-minute denominator floor and equal-network
weighting within each cohort. The equal-cohort relative objective is **44.63%**
for equal-case append, versus **43.88%** baseline, 45.07% Aug10 and 45.80% Aug25.
Partial MAE increases from 2.554 to 2.617 min. Full-collapse MAE improves by
5.74 min, but its R² and relative error worsen. Thus equal-case append reduces
some drawbacks of strongly weighted augmentation but is not an overall win.

| Full-collapse example | Actual shift (h) | Baseline prediction | Equal-case append |
|---|---:|---:|---:|
| 05 | -6.535 | -2.145 | -3.794 |
| 13 | -6.979 | -5.069 | -6.988 |
| 15 | -4.768 | -6.381 | -4.694 |
| 21 | -1.773 | -2.109 | -3.388 |
| 22 | -8.867 | -14.098 | -14.806 |
| 37 | -7.764 | -5.446 | -4.848 |
| 46 | -23.094 | -11.593 | -11.774 |
| 54 | -1.935 | -2.149 | -3.179 |
| 55 | -17.279 | -7.951 | -6.864 |

Seven of 15 networks improve in segmented absolute/relative error, and six
improve in partial mean absolute/relative error. Excluding46 from scoring does
not reverse the conclusion: segmented R² is 0.4499 versus 0.5078 baseline, with
relative error 36.34% versus 35.09%. This is a mixed redistribution of errors,
not just deterioration of the single largest case.

**Decision:** retain the unaugmented baseline. The earlier high assigned weights
were not the sole issue; ordinary appending also fails to improve the primary
relative objective in this experiment. This does not prove combinations are
inherently unhelpful, or that a different model could not exploit them. These
are one-seed, fixed-setting exploratory comparisons on the same 15 networks,
not significance tests. There is no current evidence to justify simply creating
many more combinations of the same type. Preserve the new data for subsequent
mechanism/coverage analysis instead of expanding or tuning until these existing
test cases improve. No production model was changed.

Evidence:
[script](investigations/cap_investigation_20261020/chain_equal_case.py),
[design](outputs/uncapped_event_window_20261020/chain_equal_case/design.json),
[all four treatments](outputs/uncapped_event_window_20261020/chain_equal_case/scores.csv),
[network changes](outputs/uncapped_event_window_20261020/chain_equal_case/network_changes.csv),
[segmented predictions](outputs/uncapped_event_window_20261020/chain_equal_case/segmented_predictions.csv),
[weight audit](outputs/uncapped_event_window_20261020/chain_equal_case/weight_audit.csv),
[completion](outputs/uncapped_event_window_20261020/chain_equal_case/completion.json).
Reproduce/resume in `river-hierarchy-rivgraph` with `OMP_NUM_THREADS=1
OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python
RAPID/investigations/cap_investigation_20261020/chain_equal_case.py`.

### 80-chain expansion and nested forest retuning (2026-10-20)

**Completed.** 80 distinct combined collapses cover all 15 networks: the previous 41 plus 39 selected by geometric size and constituent-set coverage, without response/error selection. Eight networks contribute six cases, six contribute five, and 37 contributes its two eligible closed-block pairs. All comparisons use current parent/segmented parameters and exclude unchanged connectors from edited-domain predictors. The added 39 are a mix of small and large combinations, not 39 new half-size cases: 46 and 55 still reach at most 75% of their full edited length, while 15/22/37 reach approximately 97–98%. See the actual coverage table before attributing any result to improved large-case coverage.

All 80 routed states settled. Parent reruns reproduce their saved hydrographs; the 0.5%, 1% and 2% shared windows/baselines required no changes. There are zero unmatched response joins. The 2,170 regular/partial comparisons and 15 segmented single-centerline comparisons retain exactly their previous targets. Regular comparisons include the existing compound-collapse cases; segmented means the separate full-network segmented representation.

**Training and validation.** Equal case weight within each network, equal total network weight; no reserved 10%/25% share. All states of the test network, including its new combinations, are held out together. The same nine predictors and centroid target are retained. Three grouped inner folds evaluate six predefined forest configurations for each of 15 outer folds. The tuning objective remains the mean of regular and segmented network-balanced relative errors with a one-minute denominator floor. New combinations are evaluated separately and do not change that objective. This preserves the earlier objective rather than rewarding the model for fitting a new test mix.

The original four-setting selection is retained as a control; the six-setting selection adds feature sampling and a more regularized squared-error forest. The frozen-setting control reuses the previously selected parameters within each held-example fold. Historical four/six-grid baselines are reproduced on existing cases, and their models also predict the same 80 held-out combinations without training on any combinations. Each individual grid setting additionally has a complete 15-network held-out report; those outer scores are descriptive and do not select the nested winner or final artifact.

There are **407 distinct completed fits**: 288 inner fits (270 outer-fold tuning + 18 all-data tuning), 90 augmented outer fits, 27 distinct baseline outer fits and two final research artifacts. Checkpointed resumes and execution-only thread scheduling preserve seeds, data, settings and scientific fingerprints. Interrupted work is not counted as an additional completed fit.

#### Held-out results for the three groups

Cells below are **R² / floored relative error (%)**. All predictions are from centroid-trained models. Peak comparisons are available in the score file as secondary diagnostics; no new peak/magnitude/shape models were trained.

| Training/selection | Regular units (2,170) | Segmented centerlines (15) | New combinations (80) | Existing-cohort relative objective |
|---|---:|---:|---:|---:|
| Current baseline: four-setting selection | 0.639 / 51.68% | 0.510 / 36.07% | 0.678 / 43.29% | 43.88% |
| Historical baseline: six-setting selection | 0.621 / 52.66% | 0.503 / 41.78% | 0.684 / 44.85% | 47.22% |
| 80 added, old fold settings | 0.584 / 51.72% | 0.473 / 34.83% | 0.642 / 37.96% | 43.28% |
| 80 added, retuned four-setting selection | 0.571 / 53.34% | 0.417 / 38.49% | 0.640 / 37.76% | 45.92% |
| 80 added, retuned six-setting selection | 0.574 / 52.88% | 0.415 / 38.55% | 0.642 / 37.71% | 45.72% |

Compared with the current baseline, six-setting retuning increases regular MAE from 2.554 to 2.586 min and segmented MAE from 169.60 to 190.90 min. Combination MAE improves from 65.81 to 60.44 min and relative error improves in 12/15 networks, but combination R² declines from 0.678 to 0.642. Thus improved typical/relative combination performance does not mean improved squared-error performance.

**Important distinction:** fixed-setting augmentation slightly improves the existing-cohort relative objective (43.88% to 43.28%), whereas retuning produces 45.72%. Neither augmented selection protocol improves the current baseline across all key metrics. The six-setting augmented procedure does improve relative error over the older six-setting unaugmented procedure (47.22% to 45.72%), but remains worse than the accepted four-setting baseline (43.88%). This avoids comparing different tuning protocols as if they were identical.

#### Full predefined setting sweep

All settings use 500 trees, bootstrap=True and max_samples=None. The following are individual-setting outer held-out R² / relative-error results, **not nested-selected model scores** and not an independent post-selection confirmation. Parameter selection still uses only inner folds.

| Setting | Loss; depth; leaf; split; feature fraction; ccp | Regular | Segmented | Combinations |
|---|---|---:|---:|---:|
| frozen | absolute_error; 16; 2; 20; 1.0; 0.001 | 0.604 / 63.78% | 0.285 / 40.95% | 0.708 / 42.50% |
| split2 | absolute_error; 16; 2; 2; 1.0; 0.001 | 0.563 / 63.66% | 0.465 / 36.47% | 0.654 / 44.07% |
| fine_absolute | absolute_error; 16; 1; 2; 1.0; 0.0 | 0.580 / 51.05% | 0.457 / 36.47% | 0.635 / 38.04% |
| fine_squared | squared_error; 16; 2; 2; 1.0; 0.0 | 0.572 / 51.72% | 0.454 / 35.10% | 0.638 / 38.68% |
| absolute_feature_sampling | absolute_error; None; 2; 2; 0.75; 0.0 | 0.599 / 51.23% | 0.446 / 36.84% | 0.659 / 41.45% |
| moderate_squared | squared_error; 10; 5; 10; 0.75; 0.0 | 0.628 / 53.12% | 0.473 / 38.22% | 0.680 / 43.65% |

The full inner sweep (all 90 outer-fold candidate evaluations), its mean scores, the 15-fold selection frequencies and the final six-candidate all-data selection are saved separately. The grid is deliberately the prior six-setting grid, not an exhaustive search across all possible RF hyperparameters.

#### Pooled results and interpretation

Pooling regular, segmented and combined test cases with equal total network weight gives R² **0.788 baseline / 0.807 fixed-setting augmentation / 0.792 six-setting retuning**, with relative errors 50.67% / 50.23% / 51.51%. On existing cases only the corresponding R² values are 0.781 / 0.793 / 0.766. These pooled statistics have a different response variance and case composition from each cohort; the approximately 0.79 pooled R² does not establish good performance on the 15 segmented collapses. The separate groups and the existing equal-cohort objective remain the primary assessment.

Retuning loss is concentrated in a few choices. For example41, actual segmented shift is -13.24 h: fixed-setting augmentation predicts -12.55 h, but retuning selects a setting predicting -8.38 h. In 13, fixed-setting augmentation predicts -6.84 h versus actual -6.98 h; retuning predicts -8.49 h. For 46, fixed and retuned predictions are both -11.55 h versus actual -23.09 h. For 55, the baseline/fixed/retuned predictions are -7.95/-6.91/-6.52 h versus actual -17.28 h. Excluding46 from scoring still leaves segmented R² 0.508 baseline, 0.442 fixed and 0.333 retuned; the loss is not solely the largest outlier. These are descriptive diagnostics, not a reason to tune on the held-out errors.

**Final all-data research artifact.** Both grid controls select `fine_squared`: 500 trees, squared_error, max_depth=16, min_samples_leaf=2, min_samples_split=2, max_features=1.0, bootstrap=True, ccp_alpha=0. The grouped all-data tuning score is 44.77%, close to 44.87% for moderate_squared and 44.91% for fine_absolute. That selection score is not independent test performance. Nested scores describe the selection procedure across folds; the individual fine_squared outer row is a setting comparison, not an independent validation of a model selected using all data. Two separately labelled research artifacts are saved and pass reload-prediction checks; neither replaces the current model.

**Decision:** keep the accepted baseline as the primary model. The 80-case data set gives useful additional context for predicting combinations and a small relative-objective gain with old settings, so it is not uniformly harmful or information-free. Retuning this grid does not deliver a robust improvement on regular and segmented targets. With only 15 independent networks, correlated augmentations, one seed and already-reused examples, the results do not support claiming statistical significance or automatically generating still more combinations.

Evidence and reproduction:

- [Scientific script](investigations/cap_investigation_20261020/chain80_tuning.py) and [parallel execution launcher](investigations/cap_investigation_20261020/run_chain80_parallel.py).
- [Selection](outputs/uncapped_event_window_20261020/chain80_tuning/selection80.csv), [component identities](outputs/uncapped_event_window_20261020/chain80_tuning/variants.json), [coverage and weights](outputs/uncapped_event_window_20261020/chain80_tuning/coverage_and_weight.csv).
- [All cohort/subset scores](outputs/uncapped_event_window_20261020/chain80_tuning/scores.csv), [full held-out setting sweep](outputs/uncapped_event_window_20261020/chain80_tuning/full_sweep_heldout_scores.csv), [full inner tuning sweep](outputs/uncapped_event_window_20261020/chain80_tuning/full_inner_sweep.csv), [selection frequencies](outputs/uncapped_event_window_20261020/chain80_tuning/selection_frequency.csv).
- [Segmented predictions](outputs/uncapped_event_window_20261020/chain80_tuning/segmented_predictions.csv), [network changes](outputs/uncapped_event_window_20261020/chain80_tuning/network_changes.csv), [all-data selection](outputs/uncapped_event_window_20261020/chain80_tuning/final_selection.json), [research model metadata](outputs/uncapped_event_window_20261020/chain80_tuning/research_model_expanded_grid.json).
- [Design/source hashes](outputs/uncapped_event_window_20261020/chain80_tuning/design.json), [fold audit](outputs/uncapped_event_window_20261020/chain80_tuning/fold_audit.csv), [response checks](outputs/uncapped_event_window_20261020/chain80_tuning/response_check.json), [fit counts](outputs/uncapped_event_window_20261020/chain80_tuning/fit_counts.json), [completion](outputs/uncapped_event_window_20261020/chain80_tuning/completion.json).

Run/resume routing and tuning with `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python RAPID/investigations/cap_investigation_20261020/chain80_tuning.py --run` in `river-hierarchy-rivgraph`. After routing/responses exist, the execution-only parallel launcher is `python RAPID/investigations/cap_investigation_20261020/run_chain80_parallel.py` with the same thread environment limits. It reuses the scientific checkpoints.

### Unrestricted combination feasibility inventory (2026-10-20)

This follows the mixed 80-case experiment. The question is whether additional, especially within-compound, collapse configurations are available—not whether the previous experiment proved augmentation cannot help. **Inventory only:** no new child geometries, calibration, routing, targets, model fits or changes to the accepted model. There is no 200-candidate limit.

The inventory reads the active parent graphs, enumerates all reachable bifurcation–confluence envelopes, and requires complete single-inlet/single-outlet closure. Every accepted membership is independently checked with `validate_domain`. A primitive closed region ends at the first closed confluence downstream of its inlet; larger serial envelopes are listed separately, avoiding counting them as additional independent building blocks. All **588 distinct saved closed U/C parent memberships** are rediscovered and their lengths/widths agree with active parameters. Three synthetic tests check serial envelopes, parallel closed subregions and an apparent bifurcation–confluence unit with a leaking internal branch.

| Candidate category | Count | Interpretation |
| --- | ---: | --- |
| Existing adjacent outer-block chains | 10,790 | Previous inventory of 10,805 minus 15 duplicates of full segmented collapse; includes previously tried chains |
| Closed parent bifurcation–confluence envelopes | 11,445 | Includes serial envelopes and nested regions; not 11,445 mutually compatible edits |
| Primitive closed regions | 616 | 435 outer blocks plus 181 regions inside outer compounds |
| Internal regions with a saved one-link child domain | 153 of 181 | Reuse candidates; geometric anchors and current hydraulics still require validation |
| Non-overlapping primitive pairs inside the same compound | 866 | Parent closure and disjointness checked; joint child not constructed |
| Strictly adjacent internal pairs | 21 | Meet the original unbranched-connector adjacency rule |
| Additional adjacent internal triples | 3 | Thus 24 internal adjacent chains in total |
| Nonadjacent internal pairs | 845 | 604 ordered with intervening branching; 241 on different branches |

**The distinction about adjacency matters.** More than 200 within-compound pair candidates exist if simultaneous disjoint edits separated within a compound are allowed. The original adjacent-only rule yields just 24 internal chains. The nonadjacent pairs are an inventory of a possible expanded construction, not automatic approval to implement it. Each constituent is itself closed; intervening reaches/branches would remain present. Open member regions are not treated as independently replaceable blocks: there are 1,415 saved U records whose parent domain is open. Combining overlapping/open members requires a separately defined joint geometry, even when their containing compound is closed.

| Network | Closed internal regions | All compatible internal pairs | Strict adjacent pairs |
| --- | ---: | ---: | ---: |
| 02 | 4 | 1 | 1 |
| 03 | 5 | 1 | 1 |
| 05 | 1 | 0 | 0 |
| 13 | 8 | 6 | 1 |
| 15 | 23 | 106 | 2 |
| 21 | 1 | 0 | 0 |
| 22 | 48 | 401 | 9 |
| 37 | 27 | 325 | 1 |
| 41 | 19 | 9 | 0 |
| 45 | 7 | 4 | 1 |
| 46 | 16 | 5 | 3 |
| 48 | 4 | 5 | 2 |
| 54 | 3 | 0 | 0 |
| 55 | 12 | 3 | 0 |
| 56 | 3 | 0 | 0 |

The pair counts are concentrated: 832/866 (96.1%) occur in 15, 22 and 37. Generating all pairs would therefore mostly add these contexts, with only five internal pairs for 46 and three for 55. Of all 866 pairs, 614 have saved one-link replacements for both constituents; 252 require at least one newly constructed replacement. These are parent-topology feasibility counts, not guaranteed routable geometries or distinct nine-predictor vectors.

Higher-order combinations are plentiful without explicitly materializing every subset. For example, `37 / CB001_bubble_3` contains 26 mutually disjoint leaf regions, giving `2^26 - 26 - 1 = 67,108,837` subsets of size at least two. This is a **combinatorial lower bound within the specified closed-block construction**, includes nonadjacent edits, and says nothing about geometric validation or information gain. The inventory writes the leaf identities and subset counts per compound rather than millions of near-duplicate rows. It does not claim to enumerate every nested antichain or every possible joint edit of overlapping members.

**Predictor coverage:** 2,185 existing rows have 2,185 distinct nine-input vectors, but distinctness does not establish coverage. Parent dominance and slope context each have only 450 distinct values. The previous 80 augmentations stay within every individual predictor's observed range, yet introduce new combinations of those values. Using global standard-deviation scaling solely for this descriptive inventory, the median nearest-existing-vector distance is 1.178 for the 80 augmentations, versus 0.167 for existing rows' nearest other row. 59/80 exceed the existing nearest-neighbour distance's 90th percentile (0.852). This supports the user's concern that marginal ranges and row counts do not establish saturation; it is not a proof that any particular added case will improve prediction. These distances are not tuning criteria or performance estimates.

The newly inventoried combinations do **not** yet have nine-input vectors: child geometry is needed for ratios and changes. The inventory's `edited_parent_km` is explicitly the sum of active **network** visit fractions times selected reach lengths, divided by 1 km; for an internal region it is not conditional domain mean length and must not be substituted directly for a model predictor. In a subsequent pilot, the predictor-domain convention for simultaneous internal edits must be fixed consistently, including treatment of unchanged connectors, before computing candidate novelty.

**Next step, not executed here:** select a small geometry-diverse internal pilot, validate replacement anchors, joint topology, unchanged external geometry and allocation, then calculate the existing nine predictors before routing. Compare joint runs with constituent controls under the same baseline/window. This separates added predictor coverage from response interactions before deciding the size or composition of another augmentation run.

Evidence:

- [Inventory script](investigations/cap_investigation_20261020/combination_feasibility_inventory.py), [topology tests](investigations/cap_investigation_20261020/test_combination_feasibility_inventory.py).
- [Network counts](outputs/uncapped_event_window_20261020/combination_feasibility_inventory/network_summary.csv), [closed regions and constituent identities](outputs/uncapped_event_window_20261020/combination_feasibility_inventory/closed_regions.parquet).
- [All 866 internal pairs](outputs/uncapped_event_window_20261020/combination_feasibility_inventory/within_compound_pairs.csv), [24 strictly adjacent chains](outputs/uncapped_event_window_20261020/combination_feasibility_inventory/within_compound_adjacent_chains.csv), [higher-order lower bounds](outputs/uncapped_event_window_20261020/combination_feasibility_inventory/higher_order_lower_bounds.csv).
- [Saved-domain reproduction](outputs/uncapped_event_window_20261020/combination_feasibility_inventory/saved_domain_reproduction.csv), [coverage summary](outputs/uncapped_event_window_20261020/combination_feasibility_inventory/coverage_summary.json), [predictor ranges](outputs/uncapped_event_window_20261020/combination_feasibility_inventory/predictor_ranges.csv), [source hashes and scope](outputs/uncapped_event_window_20261020/combination_feasibility_inventory/manifest.json).

Reproduce with `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python RAPID/investigations/cap_investigation_20261020/combination_feasibility_inventory.py` in `river-hierarchy-rivgraph`.


### Slope-ratio residual check (2026-10-23)

**Question:** does child/parent slope change explain the large errors of the accepted nine-input forest? The historical `closed_unit_extension_20261019/slope_attribution.py` already calculated an effective-slope ratio for the 15 segmented states, with historical response Spearman correlation 0.268, but did not fit a predictor model. The current nine-input panel includes parent inverse-square-root slope context only. No slope-ratio model fit was found in the inspected investigation scripts.

This check uses the frozen `nine_input_tuning` **same_grid held-network predictions**, current uncapped full-centroid targets and no augmentation. It introduces no routing or fits. **Completed after the external drive was reconnected: all 2,170 regular cases and 15 segmented states across all 15 networks are now covered from active parameters.** All 2,185 prediction/target joins match; zero source records are unavailable. Parent slope contexts and both sides' domain lengths reproduce the existing model inputs. The first, drive-disconnected screen covered only 1,142 regular cases; the numbers below supersede that subset screen.

Definitions, using each side's own allocation and lengths:

- Mean slope: `mean_s = sum(r_e L_e s_e) / sum(r_e L_e)`; ratio `mean_s_child / mean_s_parent`.
- Slope slowness context: `A_s = sum(r_e L_e / sqrt(s_e)) / sum(r_e L_e)`; ratio `R_A = A_s_child / A_s_parent`. Larger `R_A` indicates a slope contribution toward slower routing, with other hydraulic factors fixed.
- Effective slope: `s_eff = A_s^(-2)`; its ratio is `R_A^(-2)`, so it is a monotonic re-expression, not an additional independent candidate.
- Supplementary distribution diagnostic: `J_s = R_A / sqrt(mu_L_child / mu_L_parent)`. This removes the uniform slope-rescaling effect associated with length change; it is exploratory, not a newly accepted predictor.

**Domain correction to the conversational shorthand:** regular-case parent slope context is calculated over the existing B comparison domain (C for compound-member edits, otherwise the relevant closed domain), not universally over only edited reaches. Segmented cases use edited reaches and exclude unchanged connectors. Both sides of these ratio checks follow those same conventions. Edited parent size remains a separately defined input.

**Mean-slope ratio duplicates length ratio.** Since `s_e = (z_u-z_v)/L_e`, conservative route weights telescope the weighted head drops. For a closed domain, `mean_s = boundary_head_drop / mu_L`. With the same boundary drop in parent and child, `mean_s_child / mean_s_parent = mu_L_parent / mu_L_child`. The analogous conserved weighted-drop sum applies to the selected segmented blocks. In all 2,185 checked comparisons, the maximum error in `(mean-slope ratio)*(mean-length ratio) - 1` is **1.33e-15**. Thus this particular mean-slope ratio cannot supply an independent physical variable.

| Association (Spearman, equal network-total weighting within cohort) | Regular cases, 2,170 / 15 networks | Segmented, 15 / 15 networks |
| --- | ---: | ---: |
| Slowness ratio vs signed prediction error | -0.019 | +0.543 |
| Slowness ratio vs absolute prediction error | -0.212 | +0.332 |
| Slowness ratio vs existing mean-length ratio | +0.925 | +0.986 |
| Distribution diagnostic `J_s` vs signed error | -0.033 | +0.282 |

Error means **prediction minus observed centroid shift**: positive error on an advance means too little advance predicted. The segmented signed-error association persists at roughly +0.46 to +0.71 when individually excluding 22, 37, 46, 54 or 55, but is strongly confounded with the existing length ratio. In regular cases, removing network fixed effects and the length-ratio rank trend leaves a partial rank correlation of only **-0.039** with signed error and **+0.015** with absolute error for `R_A`. These are descriptive controls, not cross-validated residual correction or proof that nonlinear interactions cannot help. No row-independent significance claim is made.

| Selected case | Slowness ratio `R_A` | Mean-length ratio | Signed centroid error, hours |
| --- | ---: | ---: | ---: |
| 22 segmented | 0.9494 | 0.9062 | -5.231 |
| 46 segmented | 0.9569 | 0.9190 | +11.500 |
| 55 segmented | 0.9705 | 0.9461 | +9.328 |
| 46 `S142_unit_37` | 1.3322 | 1.0165 | -0.791 |
| 22 `island_22_018` | 1.2981 | 1.1268 | -1.296 |
| 37 `CB001_bubble_3` | 0.8852 | 0.8256 | +2.379 |
| 55 `CB001_bubble_38` | 0.9699 | 0.9454 | +0.898 |
| 37 `island_37_209` | 1.0130 | 1.0102 | -0.633 |

The similar full-collapse slope ratios do not clearly distinguish the opposing large errors of 22 versus 46/55. Some regular cases are physically interesting: 46 `S142_unit_37` has a 33.2% increase in slope slowness context for just 1.65% extra mean length; its actual centroid delay is 49.8 minutes, versus 2.4 minutes predicted. For 22 `island_22_018`, the slowness ratio increases 29.8%, compared with a 12.7% length increase. These merit mechanism checks but do not establish a general performance gain. The completed check of 37 and 55 shows no simple missing slope correction: 37 CB3 has a slowness ratio of 0.8852, already largely associated with length shortening (length ratio 0.8256; `J_s` 0.9743). For 55 CB38, `J_s` is 0.9975, so the slope-context ratio is almost exactly the uniform-rescaling expectation from length. Island 37_209 has only a 1.30% slowness increase versus a 1.02% length increase, despite a 38-minute underpredicted delay. Within-network signed-error Spearman correlations vary in sign: 22 +0.355, 37 -0.482, 41 +0.441 and 55 -0.264. Thus pooled near-zero correlation does not rule out network-specific interactions, but provides no consistent cross-network correction.

**Conclusion:** do not add the mean-slope ratio—it is redundant under the present definitions. The slowness/effective-slope ratio is a legitimate candidate for a controlled future test, especially for particular delays, but this screen does not identify it as the missing explanation for the largest segmented errors. No accepted variables or model settings changed.

Evidence: [script](investigations/cap_investigation_20261020/slope_ratio_audit.py), [associations](outputs/uncapped_event_window_20261020/slope_ratio_audit_20261023/associations.csv), [controlled rank checks](outputs/uncapped_event_window_20261020/slope_ratio_audit_20261023/controlled_rank_associations.csv), [largest errors in the complete cohorts](outputs/uncapped_event_window_20261020/slope_ratio_audit_20261023/largest_errors.csv), [all segmented cases](outputs/uncapped_event_window_20261020/slope_ratio_audit_20261023/segmented.csv), [unavailable regular sources](outputs/uncapped_event_window_20261020/slope_ratio_audit_20261023/unavailable_sources.csv), [provenance and numerical checks](outputs/uncapped_event_window_20261020/slope_ratio_audit_20261023/manifest.json).

### Outlier explanation groups and priority register (2026-10-23)

This consolidates evidence for the **accepted nine-input, same-grid nested held-network forest**, trained on full excess-volume centroid shift. No augmentation results are substituted and no new model fits or routing runs were performed. A new reproducible register selects the union of the eight largest network-weighted squared-error contributors and eight largest one-minute-floored relative-error contributors **within each cohort**: **27 priority cases** (16 regular and 11 segmented). Selection is for diagnosis, not exclusion or training adjustment.

Error concentration differs sharply by metric. In the segmented cohort, 46, 55 and 22 alone contribute **86.25%** of weighted squared error; the top five contribute 94.91%. In the regular cohort, 22 CB9 and CB11 contribute **44.25%**; the top five contribute 55.95%, and 48 cases are needed for 80%. Regular relative error is dispersed: the top five contribute only 6.12%, and **807 cases** are needed for 80%. Fixing a handful of extreme shifts will therefore affect R² much more than the average relative-error score.

| Explanation group | Current evidence | Remaining discriminating check |
| --- | --- | --- |
| Insufficient comparable training examples | **Demonstrated range limitation:** 46 segmented has size 435.6 km beyond the held-fold maximum 290.2 km, length change beyond training range, and a -23.09 h target below training minimum -17.28 h. A bounded forest prediction necessarily misses by at least 5.81 h, versus the actual 11.50 h miss. 55 segmented has only one other network with an equally extreme advance. 37 CB3/segmented have out-of-range width ratios; 48 segmented has an out-of-range parent slope context. | Distinguish absent magnitude support from absent joint hydraulic context using neighbors and controlled additions from other training networks. Being inside every marginal range does not establish joint support. |
| Statistical model dependence | **Demonstrated sensitivity:** 22 island018 changes from -0.292 h to +0.800 h when bootstrapping is disabled, compared with actual +1.004 h. For 37 CB3, cohort-balanced training changes -5.083 to -6.573 h, closer to actual -7.461 h. Neither adjustment generally fixes all other outliers; these are targeted diagnostic comparisons, not validated replacements. | A small predefined comparison on the same grouped folds; do not select whichever model best fits each outlier. |
| Missing geometric information or unrepresented hydraulic context | **Observed physical differences, causal explanation of RF residual still provisional:** 22 is faster than similarly large slower examples, while 55 and 05 include slow routing contexts. Parent slope and width ratios do not specify absolute width or the reachwise association of length, width and slope. 41 unit126 slows from 1.88 to 1.44 m/s despite shortening; 46 unit37 has a large slope-slowness change missed by a length-only interpretation. Across networks, calibrated roughness/depth scale creates differences not uniquely inferable from current geometry. | Compare a predefined geometric joint width–slope–length quantity or absolute parent width with existing neighbors. Keep calibrated n, modeled depth and celerity as diagnostics, not added hydrography-only predictors. Slope-ratio screening found no general independent correction. |
| RAPID K/X, discretization or routing artifacts | **New checks found no basic implementation/stability failure** in the 27 priority parent–child comparisons: K equals effective subsection length/celerity, X matches the active Cunge/beta expression, all K are positive and finite, and the previous-outflow coefficient satisfies the signed-inflow timestep bound. No celerity clipping or effective-length flooring is active in these checked graphs. Saved final routing receipts are complete, settled, and report nonnegative native flows. | These checks do not establish timestep/grid convergence, physical realism of reconstructed slopes/depths, or insensitivity to beta, finite windows and segmentation. A controlled dt/2 and grid-refinement check on representative cases is needed before attributing a residual to a numerical artifact. |
| Complex topology or geometry changes hidden by aggregates | **Plausible mechanism, not a generic diagnosis:** route moments, width moments and entropy do not uniquely describe where slow/fast branches occur, split–rejoin ordering or the overlap between changed and high-flow routes. 37 island209 has known geometry concerns from prior inspection; historical island combinations show interactions, but those historical capped results cannot establish the mechanism of current uncapped residuals. | Trace the actual parent–child edited links, retained boundary heads and flow redistribution; quantify interactions under the current settings with constituent controls. A complex network is not, by itself, evidence of a construction error. |
| Response/window and near-zero effects (additional group) | **Demonstrated scoring effect:** 54 unit52 changes only +0.00065 h (~0.04 min) but is predicted +0.22044 h (~13.23 min); it is the leading regular relative-error contributor. 41 islands073/132 have actual shifts around 1.2 min but predictions around 14–16 min. These are real prediction errors with small denominators, not simply timing-resolution noise. Example54's equilibrium transition makes the finite-window centroid a different physical situation from an isolated pulse. | Inspect shared-window sensitivity and hydrograph shape; revisit the buffered unchanged strategy separately. Crest switching affects peak-time diagnostics, not automatically the centroid target used here. |

The distinction between an unusual X and a demonstrated artifact is important. The smallest audited X is approximately **-32,581** at 46 link237, a 42.0 m connector divided into eight subsections (K ≈9.41 s each). It is identical in the parent and segmented child. Negative X is permitted by the implemented signed-inflow routing policy; its magnitude alone is not proof of invalid routing or an explanation for the difference. Its strong diffusion and interaction with a finite analysis window remain possible sensitivity questions, not established causes. The minimum previous-outflow coefficient across the screened graphs is +0.024; there were no bound violations.

**Practical investigation order:** (1) 46/55 segmented for support limits versus hydraulic context; (2) 22 CB9/CB11 for the largest regular squared-error contribution and context mismatch; (3) 41 unit126 and 46 unit37 for missing delay mechanisms; (4) 37 CB3/island geometry and the already demonstrated model sensitivity of 22 island018; (5) a separate near-zero/relative-error cohort. Run routing sensitivity on representatives spanning these groups rather than labeling every residual a RAPID issue. For remaining priority cases with no direct mechanism test, the register records the error/support/routing evidence but does not invent a cause. Multiple groups can apply to the same case.

Reproduce with `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python RAPID/investigations/cap_investigation_20261020/outlier_diagnosis.py`.

Evidence: [script](investigations/cap_investigation_20261020/outlier_diagnosis.py), [27-case priority register](outputs/uncapped_event_window_20261020/outlier_diagnosis_20261023/priority_cases.csv), [error concentration](outputs/uncapped_event_window_20261020/outlier_diagnosis_20261023/error_concentration.csv), [K/X and timestep checks](outputs/uncapped_event_window_20261020/outlier_diagnosis_20261023/routing_parameters.csv), [saved routing receipts](outputs/uncapped_event_window_20261020/outlier_diagnosis_20261023/routing_receipts.csv), [provenance](outputs/uncapped_event_window_20261020/outlier_diagnosis_20261023/manifest.json). Previous evidence: [targeted model checks](outputs/uncapped_event_window_20261020/nine_error_audit/targeted_training_checks.csv), [partial-domain physics](outputs/uncapped_event_window_20261020/nine_error_audit/partial_domain_physics.csv), [joint geometry and hydraulic context](outputs/uncapped_event_window_20261020/physical_context_and_combinations/segmented_physical_context.csv).

### Dominant outliers: exclusion scores and influential-neighbor context (2026-10-23)

Using unchanged held-network predictions, removing 22 `CB001_bubble_9` and `CB002_bubble_11` from the **evaluation only** raises regular-case R² from **0.6386 to 0.7874** (2,170 to 2,168 cases). MAE changes only from **2.554 to 2.471 minutes**, and one-minute-floored relative error from **51.681% to 51.654%**. Removing CB9 alone gives R² 0.7405; removing CB11 alone gives 0.6809. Thus the squared-error problem is concentrated, while typical absolute and relative errors are largely unchanged.

For segmented cases, removing 22, 46 and 55 together gives **R² 0.7567 versus 0.5100**, MAE **81.71 versus 169.60 minutes**, and relative error **31.53% versus 36.07%** (12 versus 15 networks). These are post-hoc sensitivity scores, not replacement performance claims. Equal network weights and the target mean/variance are recomputed for each retained subset; no model is retrained. Removing 46 alone gives **R² 0.5078**, despite substantially lowering squared error, because that extreme target also contributes heavily to the denominator of R². An outlier's SSE share is not its direct contribution to R².

A fresh physical-context comparison examines the five focal cases and each case's three most influential saved bootstrap-leaf neighbors. All identities are explicitly normalized/joined, the descriptor domains match the current predictor conventions, and active geometric/hydraulic parameters are used. Bootstrap leaf weights describe which training states support the prediction; they are not exact causal importance or, for median trees, an exact prediction decomposition.

**22 CB9:** the three leading supports are 37 CB3 (26.8% leaf weight), 37 segmented (21.2%) and 13 segmented (20.6%). Their actual shifts are -7.46, -7.76 and -6.98 h, whereas CB9 is -2.29 h and predicted -7.32 h. Against 37 CB3, its parent mean length is nearly identical (91.68 versus 89.97 km), but its parent length-weighted mean width is **2.40 times larger**. Parent effective celerity is about 5.05 versus 1.61 m/s. The current width ratio/CV/dominance features do not explicitly encode absolute width. Absolute parent width was screened previously in the earlier seven-input feature panels; this neighbor evidence is a reason to revisit its incremental value alongside the current nine inputs, not to claim it has never been considered or will necessarily improve the model.

**22 CB11:** leading supports are segmented 15 (29.0%), 45 (26.6%) and 37 (10.4%). Its parent mean width is 2.19, 1.72 and 2.79 times those neighbors' widths. The forest borrows larger advances from these states (-4.77, -8.75, -7.76 h), whereas CB11 advances by only -2.84 h. Its current prediction is -6.36 h. This repeats the geometric-context mismatch rather than identifying a one-case numerical anomaly.

**22 segmented:** leading supports are 41 (36.4%), 55 (30.0%) and 46 (10.2%). Against 41, parent width is similar (0.94 times) and size is slightly larger (1.10 times), so width alone cannot fix every 22 comparison. A joint reachwise width/slope/length quantity and the network hydraulic scale both distinguish their routing changes.

**46 segmented:** leading supports are 41 (41.0%), 22 (16.6%) and 55 (16.4%). It is 1.65 times as long as 41 and has 1.67 times its geometric travel-change magnitude; its hydraulic time scale is also 1.27 times larger. Those factors multiply to a **2.11 times larger routing-time change**. Thus its large response has a coherent physical explanation as well as the already demonstrated forest/training-range limitation. This does not make a geometric forest capable of extrapolating beyond its training response range.

**55 segmented:** leading supports are 13 (25.6%), 45 (20.7%) and 05 (13.6%). Compared with 13, it is 1.57 times as long and only 0.52 times as wide on average. Its geometric travel-change magnitude is 1.55 times larger, and its hydraulic time scale is 1.26 times larger: together approximately **1.95 times the routing-time change**. Yet its forest prediction (-7.95 h) remains close to the smaller shifts of 13 and 45 (-6.98 and -8.75 h), rather than its actual -17.28 h.

The decomposition used here is the previously motivated `T = A_example G`, where `G = sum[r_e (L_e/1km) (W_e/100m)^(-13/30) / sqrt(s_e)]` and the active within-network depth law is `h_e = a_example W_e^0.65`. The parent-derived A reconstructs each checked parent–child routing-time difference to numerical precision. **A includes calibrated roughness and modeled depth scale; it is explanatory only, not an eligible hydrography-only predictor.** G is a geometric combination, while neither T nor A·deltaG equals the finite-window centroid response by definition.

For CB9 versus 37 CB3, the geometric change ratio is 0.799, and the hydraulic-scale ratio is 0.369: the routing-time-change ratio is therefore 0.295. For 22 segmented versus 55 segmented, the geometric change is actually 1.41 times larger, but the hydraulic scale is only 0.437 times as large, yielding a smaller routing-time change (ratio 0.617). This establishes why geometric size alone can point in the wrong direction. Some additional geometry may help; some residual differences may remain because the calibrated hydraulic scale is outside the permitted predictor set.

**Next discriminating tests:** evaluate absolute parent width and the joint geometric G change as separate, predefined additions to the frozen nine-input baseline under the same held-network validation; do not add A, n or depth. Separately check timestep/grid and analysis-window sensitivity for representative 22 and 46 cases before attributing residuals to routing artifacts. None of these new prediction or routing sensitivity tests was run in this diagnostic. Keep all five focal cases in the main evaluation population.

Evidence: [script](investigations/cap_investigation_20261020/dominant_outlier_context.py), [all exclusion scores](outputs/uncapped_event_window_20261020/dominant_outlier_context_20261023/exclusion_scores.csv), [matched physical context](outputs/uncapped_event_window_20261020/dominant_outlier_context_20261023/matched_physical_context.csv), [15 influential-neighbor contrasts](outputs/uncapped_event_window_20261020/dominant_outlier_context_20261023/neighbor_contrasts.csv), [provenance and interpretation](outputs/uncapped_event_window_20261020/dominant_outlier_context_20261023/manifest.json).

### Fixed-settings comparison: absolute width, joint geometry and resistance (2026-10-23)

**Completed 45 forest fits:** add one predefined feature at a time to the accepted nine-input model, with all 2,170 regular and 15 segmented cases, no augmentations, and the same 15 whole-network holdouts. Each fold reuses its baseline same-grid training-only-selected hyperparameters, seed, equal-network training weights and preprocessing. New panels were **not retuned**. Cached baseline predictions are retained unchanged. There are no response-join failures, missing added inputs, new RAPID runs or changes to the accepted model.

The additions are:

1. **Dimensionless absolute parent width:** `S_W = mean_width_parent / (100 m)`, with the existing length-weighted width definition. This preserves absolute size; 100 m is a fixed reference unit, not a physical threshold, cap or network-relative normalization. A different positive fixed reference simply rescales the variable; it does not add information. Regular cases use the B comparison domain, segmented cases the edited footprint excluding connectors.
2. **Joint geometric change:** `deltaG = G_child - G_parent`, where `G = sum[r_e (L_e/1km) (W_e/100m)^(-13/30) / sqrt(s_e)]`. Use each side's own allocation, same B/edited-footprint conventions as above. Every term is dimensionless. The exponent -13/30 follows the present assumed depth-width exponent 0.65 and Manning travel-time scaling: -(2/3)*0.65. Thus this is motivated by the present model physics, not a universal geometry law. It includes no modeled depth, calibrated roughness or hydrograph-derived input. Wider/steeper reaches contribute less; longer/high-visit reaches contribute more, combined reach by reach rather than as separate averages.
3. **Legacy geometric resistance change:** `-log10(K_child/K_parent)`, with `K_e = L_e/W_e^5` (the existing alpha=1 law). Regular cases retain the established outer B-domain resistance. Segmented resistance uses the **full network including unchanged connectors**, matching the outer-domain convention; it is not the edited-footprint resistance. The 15 segmented parent unit-flow solves are reused only after checking cached graph hashes, success, and exact agreement of active geometry. Their new child chains have exact serial resistance sums. Existing regular resistance ratios are independently reconstructed from their checked parent/child solver records. This compares the agreed feature definitions, not alternative resistance domains or alpha laws. Resistance K here is distinct from RAPID's routing travel-time K.

| Panel | Regular R² | Regular relative error | Segmented R² | Segmented relative error | Equal-cohort relative objective |
| --- | ---: | ---: | ---: | ---: | ---: |
| Accepted nine-input baseline | 0.6386 | 51.68% | 0.5100 | 36.07% | 43.88% |
| + absolute parent width | 0.6432 | 48.05% | 0.4729 | 38.70% | 43.37% |
| + joint geometric change | **0.7186** | **39.93%** | **0.5533** | **33.62%** | **36.77%** |
| + resistance | 0.6473 | 50.98% | 0.4904 | 38.14% | 44.56% |

Relative errors use the established one-minute denominator floor and equal network weighting within each cohort; the objective is the equal mean of the two cohort scores. Regular MAE changes from 2.554 min to 2.311 (width), **2.152 (joint)** and 2.490 (resistance). Segmented MAE changes from 169.60 min to 176.82, **160.98** and 174.63 respectively. Do not conflate the many regular cases with 15 independent segmented examples by relying on a pooled score.

The joint feature improves network-average regular relative error in **13/15 networks** and segmented absolute/relative error in **12/15**. It is not solely an outlier correction. Regular example54 worsens from 46.87% to 78.55%, and 56 from 25.12% to 27.17%. Excluding 54 for scoring, the joint panel still improves regular R² from 0.6369 to 0.7205 and relative error from 52.02% to 37.17%; segmented R² improves from 0.4840 to 0.5296 and relative error from 37.86% to 35.31%. Example54 remains in the fitted data and main evaluation; its deterioration must be investigated rather than hidden by exclusions.

Specific held-network predictions (hours; same full-centroid target):

| Case | Observed | Baseline | + joint geometry |
| --- | ---: | ---: | ---: |
| 22 CB9 | -2.286 | -7.321 | -6.934 |
| 22 CB11 | -2.836 | -6.365 | -6.260 |
| 22 island018 | +1.004 | -0.292 | +0.297 |
| 41 unit126 | +0.295 | -0.716 | +0.248 |
| 46 unit37 | +0.830 | +0.040 | +0.028 |
| 46 segmented | -23.094 | -11.593 | -12.007 |
| 55 segmented | -17.279 | -7.951 | -8.444 |
| 22 segmented | -8.867 | -14.098 | -14.314 |

The 41 unit126 delay is substantially better represented, and 22 island018 obtains the correct sign. The most extreme 46/55 advances remain strongly underestimated, and 22 segmented becomes slightly worse. Therefore the joint feature is a promising broad addition, not a resolution of the extrapolation/hydraulic-context problem.

**Resistance versus width:** their R² values are close, with resistance slightly better in both cohorts, but absolute width has better regular relative error and a slightly better equal-cohort objective. Neither gives a convincing improvement across both cohorts; resistance's objective is worse than the baseline. This does not justify restoring resistance automatically. Keep it as an alternative candidate if its interpretation is preferred, but the strongest current evidence favors testing/retuning the ten-input joint-geometry panel. Adding resistance on top of that panel, replacing any existing variable, alternative resistance laws/domains and seed sensitivity have **not** been tested here. No statistical significance or new external validation is claimed from these exploratory reused networks.

Reproduce/resume: `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python RAPID/investigations/cap_investigation_20261020/width_joint_resistance_check.py --run --jobs 4`.

Evidence: [script](investigations/cap_investigation_20261020/width_joint_resistance_check.py), [scores](outputs/uncapped_event_window_20261020/width_joint_resistance_20261023/scores.csv), [objective](outputs/uncapped_event_window_20261020/width_joint_resistance_20261023/objective.csv), [priority predictions](outputs/uncapped_event_window_20261020/width_joint_resistance_20261023/priority_predictions.csv), [all segmented predictions](outputs/uncapped_event_window_20261020/width_joint_resistance_20261023/segmented_predictions.csv), [network relative errors](outputs/uncapped_event_window_20261020/width_joint_resistance_20261023/network_relative_errors.csv), [segmented resistance audit](outputs/uncapped_event_window_20261020/width_joint_resistance_20261023/segmented_resistance_audit.csv), [preparation and source hashes](outputs/uncapped_event_window_20261020/width_joint_resistance_20261023/preparation.json), [design](outputs/uncapped_event_window_20261020/width_joint_resistance_20261023/design.json), [completion](outputs/uncapped_event_window_20261020/width_joint_resistance_20261023/completion.json).

### Requested six panels: depth-free length–slope change and explicit-depth resistance (2026-10-23)

The user rejected embedding the RAPID depth–width exponent in the primary joint predictor and requested six specific comparisons. **Delta G is always added, never substituted.** Only the mean-width ratio is replaced by resistance in panels 3 and 5; width CV and parent dominance stay in every panel. After clarification, all four resistance panels use the explicit-derived-depth version of the existing wide-Chezy resistance law. These definitions differ from the preceding three-addition test, so its scores must not be reused.

Definitions:

- New `G_s = sum[r_e (L_e/1km) / sqrt(s_e)]` and `delta_G_s = G_s_child - G_s_parent`. No depth-derived width exponent or explicit depth is used. Allocation probabilities remain the existing width-based probabilities. Equivalently `G_s = (mu_L/1km) * mean(1/sqrt(s))` with visit-length weights. The child slope context provides information beyond the parent slope input; this is not the preceding `G` containing `W^(-13/30)`.
- Explicit-depth edge resistance is `K_e = L_e/(W_e^2 h_e^3)` up to a common Chezy coefficient, with `R_h = -log10(K_eq_child/K_eq_parent)`. This is network hydraulic resistance, not RAPID travel-time K. Regular cases use the existing outer B domain; segmented cases use full-network resistance including unchanged connectors. G_s uses the existing B domain for regular cases and edited footprint excluding connectors for segmented cases.
- **Derived-depth limitation:** active reach depths obey `h_e = a_example W_e^0.65`, verified on all prepared domains. Thus explicit-depth resistance is exactly `a_example^-3` times the cached unit-flow solve with edge coefficients `L_e/W_e^3.95`. Parent and child a values agree to 1.56e-15 relative precision, so a cancels from the resistance ratio. The numerical explicit-depth ratio and normalized geometric ratio agree within 6.94e-16. Reusing those audited alpha=.65 solves is an algebraically exact calculation, not a substitution of the previous alpha=1 resistance. The segmented child resistance is independently calculated as the exact serial sum of `L/(W^2 h^3)`. This does **not** make derived depth an independent measurement or remove its within-network width-scaling assumption.

**Completed 75 fits**, five alternatives times 15 whole-network holdouts, plus unchanged cached baseline predictions. All 2,185 records are present, without imputations or dropped cases. Fixed baseline fold-selected settings, seed, equal-network training weights and training-only preprocessing are reused; no retuning, augmentation, RAPID reruns or production-model replacement. Primary target remains full excess-volume centroid shift in the agreed 0.5% window.

| Requested panel | Inputs | Regular R² | Regular relative error | Segmented R² | Segmented relative error | Equal-cohort objective |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1. Current model | 9 | **0.6386** | 51.68% | **0.5100** | **36.07%** | **43.88%** |
| 2. Current + delta G_s | 10 | 0.6130 | 50.92% | 0.4978 | 38.88% | 44.90% |
| 3. Replace mean-width ratio with R_h | 9 | 0.6242 | 51.14% | 0.4746 | 39.20% | 45.17% |
| 4. Current + R_h | 10 | 0.6463 | 50.97% | 0.4958 | 38.39% | 44.68% |
| 5. Panel 3 + delta G_s | 10 | 0.5997 | 50.54% | 0.4691 | 40.59% | 45.57% |
| 6. Panel 4 + delta G_s | 11 | 0.6062 | 50.54% | 0.4924 | 39.39% | 44.96% |

Relative errors retain the one-minute denominator floor and equal total network weight within each cohort. The objective is the equal average of the two cohort relative errors. Regular MAEs for panels 1–6 are 2.554, 2.636, 2.653, 2.512, 2.738 and 2.605 minutes; segmented MAEs are 169.60, 178.62, 177.96, 175.37, 183.91 and 178.18 minutes. The baseline is not the best regular R² (panel 4 is slightly higher), but remains best on the predefined equal-cohort objective and on segmented R²/relative error. The difference is not explained solely by example54: after excluding 54 for scoring, every alternative still has worse segmented R² and relative error than the baseline. These are reused exploratory folds, not evidence of statistical significance or of optimally tuned alternative models.

The extreme residuals remain. Across panels 2–6, 22 CB9 predictions range from -7.06 to -7.33 h against -2.29 observed, and CB11 remains around -6.24 to -6.44 h against -2.84 observed. Segmented 46 remains around -10.83 to -11.48 h against -23.09 observed; 55 remains around -7.96 to -8.11 h against -17.28. The earlier width-weighted G corrected 41 unit126's sign, but this new length–slope-only feature does not: all five alternatives still predict an advance where the observed centroid delay is +0.295 h. Removing the width factor from that joint construction loses its prior improvement; this is not proof that the particular former exponent is uniquely necessary or universally appropriate.

**Where to add depth separately:** prepare an optional predictor `S_h = mean_depth_parent/(1m)`, using `mean_depth = sum(r_e L_e h_e)/sum(r_e L_e)` in the same domain convention as G_s. This retains absolute depth scale, which R_h's parent/child ratio cancels. A child/parent mean-depth ratio is also prepared but is strongly correlated with the existing mean-width ratio (Spearman 0.925), so absolute parent depth is the more informative first *candidate*, not an established performance improvement. All 2,185 explicit-depth descriptors are available, parent means span about 0.77–7.76 m, and their inherited width^.65 dependence is documented. **Neither direct depth predictor was fitted as an additional input in these six panels.** A future optional depth-informed model can add S_h to a predefined geometry panel, without placing any depth exponent in G_s. Keep that result distinct from the geometry-only model.

**Decision supported by this test:** retain the current nine-input model as the main result. Adding R_h makes a small regular-case improvement but not an overall one; replacing the mean-width ratio with R_h does not help under the frozen settings. No alternative was promoted. The earlier width-weighted deltaG remains a diagnostic model-dependent result, not the proposed primary predictor.

Reproduce/resume with `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python RAPID/investigations/cap_investigation_20261020/six_panel_geometry_depth_resistance.py --run --jobs 4`. Optional depth descriptors are prepared by `prepare_depth_context.py` in the same investigation directory; it performs no fits.

Evidence: [six-panel script](investigations/cap_investigation_20261020/six_panel_geometry_depth_resistance.py), [six-panel scores](outputs/uncapped_event_window_20261020/six_panel_geometry_depth_resistance_20261023/scores.csv), [objective](outputs/uncapped_event_window_20261020/six_panel_geometry_depth_resistance_20261023/objective.csv), [network scores](outputs/uncapped_event_window_20261020/six_panel_geometry_depth_resistance_20261023/network_scores.csv), [priority predictions](outputs/uncapped_event_window_20261020/six_panel_geometry_depth_resistance_20261023/priority_predictions.csv), [explicit-depth resistance audit](outputs/uncapped_event_window_20261020/six_panel_geometry_depth_resistance_20261023/explicit_depth_resistance_audit.csv), [design](outputs/uncapped_event_window_20261020/six_panel_geometry_depth_resistance_20261023/design.json), [completion](outputs/uncapped_event_window_20261020/six_panel_geometry_depth_resistance_20261023/completion.json), [optional depth preparation](investigations/cap_investigation_20261020/prepare_depth_context.py), [depth provenance](outputs/uncapped_event_window_20261020/depth_context_20261023/manifest.json).

### Add parent depth to all six panels (2026-10-23)

Completed the requested paired check: add `parent_depth_nd` (S_h) to **each** of the six preceding panels. S_h = sum(r_e L_e h_e)/sum(r_e L_e)/(1m), using B comparison domains for regular cases and edited footprints excluding connectors for segmented cases. Delta G_s remains an addition, never a replacement; only panels 3/5 replace mean-width ratio with explicit-depth resistance. No depth ratio was added. The depths are derived from h=a_example W^0.65, not independent reach measurements. Absolute depth scale cancels in the resistance ratio but is retained by S_h.

All 90 fits completed: six panels × 15 whole-example holdouts, 2,185 matched cases per panel (2,170 regular and 15 segmented), zero unmatched records. Settings, seed, weights, target, training-only preprocessing and folds remain as in the preceding experiment; no retuning. Production results and the accepted model were not changed.

| Panel, each with added S_h | Regular R² | Regular relative error | Segmented R² | Segmented relative error | Equal-cohort objective, before → after depth |
|---|---:|---:|---:|---:|---:|
| 1 Current | 0.6173 | 47.62% | 0.4853 | 39.93% | 43.88% → 43.77% |
| 2 Current + Delta G_s | 0.5936 | 48.67% | 0.4782 | 40.81% | 44.90% → 44.74% |
| 3 Replace mean-width ratio with R_h | 0.5983 | 46.44% | 0.4657 | 41.65% | 45.17% → 44.04% |
| 4 Current + R_h | 0.6109 | 47.27% | 0.4751 | 41.04% | 44.68% → 44.16% |
| 5 Width replacement + Delta G_s | 0.5584 | 47.86% | 0.4664 | 42.15% | 45.57% → 45.00% |
| 6 Current + R_h + Delta G_s | 0.5960 | 48.96% | 0.4658 | 41.37% | 44.96% → 45.16% |

Relative errors retain equal network weights and the one-minute denominator floor. Adding depth improves regular relative error in every panel, but worsens regular R², segmented R² and segmented relative error in every panel relative to its matched depth-free version. The best combined score is panel 1 + depth, but its improvement over the accepted baseline is only 0.104 percentage points: regular relative error 51.68% → 47.62%, segmented 36.07% → 39.93%. Its MAEs change from 2.554 → 2.330 min (regular) and 169.60 → 177.54 min (segmented). Regular relative error improves in 9/15 networks; segmented improves in only 4/15.

The tradeoff persists without example 54: panel 1 regular R² 0.6369 → 0.6156, relative error 52.02% → 47.45%; segmented R² 0.4840 → 0.4582, relative error 37.86% → 41.50%. Large segmented residuals are not repaired: predictions for 22 move -14.10 → -14.30 h (actual -8.87 h), 46 -11.59 → -11.34 h (actual -23.09 h), and 55 -7.95 → -7.86 h (actual -17.28 h).

**Interpretation:** absolute parent depth alone does not recover the earlier width-weighted joint predictor's gain (that diagnostic achieved regular/segmented relative errors 39.93%/33.62%, objective 36.77%). Removing its width^-13/30 factor had removed a joint spatial interaction, not an independently measured depth input. These experiments support investigating that joint construction; they do not establish that its improvement was mostly caused by depth, nor identify a unique physical mechanism. Depth-informed panels retain model-derived assumptions, and these exploratory reused folds do not provide an independent final test. Retain the accepted nine-input model for now.

Reproduce/resume: `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python RAPID/investigations/cap_investigation_20261020/six_panel_parent_depth_check.py --run --jobs 4`.

Evidence: [script](investigations/cap_investigation_20261020/six_panel_parent_depth_check.py), [scores](outputs/uncapped_event_window_20261020/six_panel_parent_depth_20261023/scores.csv), [paired changes](outputs/uncapped_event_window_20261020/six_panel_parent_depth_20261023/paired_scores.csv), [network scores](outputs/uncapped_event_window_20261020/six_panel_parent_depth_20261023/network_scores.csv), [predictions](outputs/uncapped_event_window_20261020/six_panel_parent_depth_20261023/predictions.parquet), [design](outputs/uncapped_event_window_20261020/six_panel_parent_depth_20261023/design.json), [completion](outputs/uncapped_event_window_20261020/six_panel_parent_depth_20261023/completion.json).

### Explicit inverse-depth joint predictor (2026-10-23)

At the user's request, test a new joint geometry descriptor with no Manning-derived 2/3 depth exponent:

`G_h^D = sum_e r_e L_e / (h_e sqrt(s_e)); Delta G_h^D = G_h,child^D - G_h,parent^D`.

Length and depth are both in metres, so G_h is dimensionless. Depth enters inversely, and slope retains the agreed inverse-square-root relationship. This is a geometric hypothesis, not a calculated routing time. Stored depths remain derived from the reference depth and width^0.65; the feature code consumes depth directly, so representative observed reach depths could replace those estimates without changing the expression. It does not use roughness, celerity, RAPID K/X, or hydrograph-derived features. The exponent 0.65 remains in the current depth *estimation*, not in the predictor's explicit depth relationship.

Domains and probabilities are unchanged: regular B comparison domains (containing compound where applicable), segmented edited footprints excluding unchanged connectors. Regular probabilities are computed from each state's graph; segmented contributions use the existing reference flow fractions. Parent/child allocation-weighted lengths reproduce the existing domain mean lengths. The calculated mean depths also match the previously audited depth descriptors to relative tolerance 1e-12. All 2,185 comparisons join one-to-one with zero unmatched records.

Add exactly Delta G_h to the accepted nine-input forest (ten inputs total). Completed 15 whole-example holdout fits, reusing the same fold-selected settings, seed, equal-network training weights and training-only preprocessing. Target remains full excess-volume centroid shift with the agreed 0.5% shared window. No tuning, augmentation, production routing change or replacement of the accepted model. Prior predictions are reused for paired comparisons, with exact target and cohort alignment checks.

| Model | Regular R² | Regular relative error | Segmented R² | Segmented relative error | Equal-cohort objective |
|---|---:|---:|---:|---:|---:|
| Accepted nine inputs | 0.6386 | 51.68% | 0.5100 | 36.07% | 43.88% |
| + earlier width-weighted Delta G | 0.7186 | 39.93% | 0.5533 | 33.62% | 36.77% |
| + length–slope Delta G_s and separate mean parent depth | 0.5936 | 48.67% | 0.4782 | 40.81% | 44.74% |
| **+ inverse-depth Delta G_h** | **0.7624** | **34.48%** | **0.6148** | **33.01%** | **33.75%** |

Relative error uses the one-minute denominator floor, equal network weights within each cohort and equal cohort weights for the objective. Regular MAE falls 2.554 → 1.975 min; segmented MAE 169.60 → 156.26 min. Excluding example54 for scoring preserves the gain: regular R² 0.6369 → 0.7600 and relative error 52.02% → 35.65%; segmented R² 0.4840 → 0.5944 and relative error 37.86% → 35.03%.

Improvements are not universal: regular network-level relative error improves in 10/15 networks, segmented in 12/15. Regular 03,05,13,15,46 worsen; segmented 02,15,41 worsen. Example15 segmented relative error rises 33.83% → 74.19%, and regular 47.56% → 67.89%. This should remain visible when considering adoption.

Selected predictions (hours; actual, baseline, new): 22 CB9 (-2.286, -7.321, -5.000); 22 CB11 (-2.836, -6.365, -4.481); 22 segmented (-8.867, -14.098, -12.960); 46 segmented (-23.094, -11.593, -13.071); 55 segmented (-17.279, -7.951, -9.327). The large residuals shrink but persist. 41 unit126's sign is corrected: actual +0.295 h, baseline -0.716 h, new +0.226 h.

**Interpretation:** this is the strongest of these fixed-settings joint-feature checks on the predefined objective. Explicit inverse depth in a reach-level interaction performs substantially better than the separate mean-depth input. It supports retaining this candidate for further assessment; it does not prove a unique physical mechanism or independent observed-depth effect. Compared with the earlier width-based G, both the depth relationship and the example-specific absolute depth scale change, so their contributions are not isolated. Repeated exploratory use of these folds is not an independent final test; the production model remains unchanged.

Reproduce/resume: `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python RAPID/investigations/cap_investigation_20261020/inverse_depth_joint_check.py --run --jobs 4`.

Evidence: [script](investigations/cap_investigation_20261020/inverse_depth_joint_check.py), [scores](outputs/uncapped_event_window_20261020/inverse_depth_joint_20261023/scores.csv), [network scores](outputs/uncapped_event_window_20261020/inverse_depth_joint_20261023/network_scores.csv), [priority predictions](outputs/uncapped_event_window_20261020/inverse_depth_joint_20261023/priority_predictions.csv), [preparation](outputs/uncapped_event_window_20261020/inverse_depth_joint_20261023/preparation.json), [design](outputs/uncapped_event_window_20261020/inverse_depth_joint_20261023/design.json), [completion](outputs/uncapped_event_window_20261020/inverse_depth_joint_20261023/completion.json).

### Fresh six-panel resistance × inverse-depth joint comparison (2026-10-23)

Completed 90 fresh fits (six configurations × 15 held examples) to compare the accepted current model, explicit-depth resistance R_h, and the new inverse-depth Delta G_h. All configurations use the same 2,185 comparisons (2,170 regular, 15 segmented), centroid-shift target, 0.5% shared-window responses, fold-selected forest settings, seed, equal-network training weights and training-only preprocessing. No retuning. The four previously tested configurations reproduce their saved predictions within rtol=atol=1e-12. Input joins and targets match exactly; zero unmatched cases.

Definitions are unchanged: Delta G_h = child-parent sum r_e L_e/(h_e sqrt(s_e)), with lengths/depths in metres. R_h=-log10(K_child/K_parent), with edge K=L/(W^2 h^3), the existing explicit-depth equivalent resistance, not RAPID travel-time K. The removal of the Manning 2/3 exponent applies to G_h; resistance retains its previously specified hydraulic law. Both use derived depths currently. G_h uses regular B domains and segmented edited footprints excluding connectors; resistance uses regular B domains and full segmented networks including connectors. This comparison does not change those domains. Only the mean-width-ratio input is replaced in panels 3/5; width CV and parent dominance remain. No separate mean-depth predictor is added.

| Panel | Inputs | Regular R² | Regular relative error | Segmented R² | Segmented relative error | Equal-cohort objective |
|---|---:|---:|---:|---:|---:|---:|
| 1 Current | 9 | 0.6386 | 51.68% | 0.5100 | 36.07% | 43.877% |
| 2 Current + Delta G_h | 10 | 0.7624 | 34.48% | 0.6148 | 33.01% | 33.747% |
| 3 Replace mean-width ratio with R_h | 9 | 0.6242 | 51.14% | 0.4746 | 39.20% | 45.170% |
| 4 Current + R_h | 10 | 0.6463 | 50.97% | 0.4958 | 38.39% | 44.680% |
| 5 Width replacement + Delta G_h | 10 | 0.7687 | 34.24% | 0.5961 | 33.11% | 33.671% |
| 6 Current + R_h + Delta G_h | 11 | 0.7746 | 34.25% | 0.5943 | 33.24% | 33.746% |

Relative error retains the one-minute denominator floor and equal network weights; the objective averages the two cohort relative errors equally. Regular MAEs for panels 1–6: 2.554, 1.975, 2.653, 2.512, 1.964, 1.943 min. Segmented MAEs: 169.60, 156.26, 177.96, 175.37, 157.96, 158.35 min.

**Interpretation:** Delta G_h accounts for the main predictive gain. With it included, resistance gives a modest additional regular-case improvement, offset by weaker segmented performance. Panel 5 has the numerically lowest objective, only 0.0754 percentage points below panel 2; panel 6 is effectively tied with panel 2 (difference 0.0006 points). These tiny objective differences do not establish a meaningful superiority. Panel 2 has the best segmented R² and relative error; panel 6 has the best regular R² and MAE. Excluding 54 retains this near-tie: objectives for panels 2/5/6 are 35.340%, 35.290%, 35.302%.

Relative to panel 2, panel 5 improves network-level relative error in 8/15 regular and 9/15 segmented cases; panel 6 improves 9/15 regular and 5/15 segmented. Example15 remains problematic: segmented relative error is 74.19%, 72.76%, 70.60% for panels 2/5/6, respectively. Resistance does not resolve that weakness.

Retain current + Delta G_h as the simpler leading candidate in terms of required calculations. Width replacement + Delta G_h is a viable alternative if resistance is scientifically preferred, with the same ten-input count, but the experiment does not justify claiming a decisive performance advantage. No accepted/production model is replaced; these are fixed-settings exploratory comparisons on reused folds, not an independent final test.

Reproduce/resume: `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python RAPID/investigations/cap_investigation_20261020/resistance_inverse_depth_panels.py --run --jobs 4`.

Evidence: [script](investigations/cap_investigation_20261020/resistance_inverse_depth_panels.py), [scores](outputs/uncapped_event_window_20261020/resistance_inverse_depth_panels_20261023/scores.csv), [objectives](outputs/uncapped_event_window_20261020/resistance_inverse_depth_panels_20261023/objective.csv), [network scores](outputs/uncapped_event_window_20261020/resistance_inverse_depth_panels_20261023/network_scores.csv), [priority predictions](outputs/uncapped_event_window_20261020/resistance_inverse_depth_panels_20261023/priority_predictions.csv), [design](outputs/uncapped_event_window_20261020/resistance_inverse_depth_panels_20261023/design.json), [completion](outputs/uncapped_event_window_20261020/resistance_inverse_depth_panels_20261023/completion.json).

### Ten-input model: pooled performance and outlier exclusions (2026-10-23)

Audit panel 5 of the fresh six-panel comparison: replace mean-width ratio with explicit-depth resistance and add inverse-depth Delta G_h, ten predictors. All values below use the saved whole-example-held-out centroid predictions, not training predictions. These forests **were trained with the ten inputs**, one fresh forest per held example; their hyperparameters were inherited from the earlier nine-input fold-specific tuning. They were not retuned for these ten inputs, and this experiment did not create a final all-data model artifact. No new fits or production changes in this audit.

| Combined scoring convention, all 2,185 cases | R² | MAE (min) | Relative error |
|---|---:|---:|---:|
| Equal total weight per network across its states | 0.8252 | 3.183 | 34.23% |
| Every comparison equally weighted | 0.8168 | 3.108 | 37.18% |
| Equal cohorts, then equal networks within cohort | 0.7545 | 79.961 | 33.67% |

The last relative-error score is the existing equal-cohort objective. Pooled R² uses a global weighted target mean, not an average of cohort R² values. The low pooled MAE in the first two rows must not be mistaken for typical segmented performance: there are 2,170 regular versus only 15 segmented comparisons. Separate cohort metrics remain regular R² 0.7687, MAE 1.964 min, relative error 34.24%; segmented R² 0.5961, MAE 157.96 min, relative error 33.11%. Pooling can increase R² because between-cohort target variation contributes to the denominator. Excluding example54 alone gives pooled equal-network R² 0.8241 and equal-cohort R² 0.7502.

Largest contributors to squared error **within each cohort**, using equal-network weights (hours, negative = advance):

| Cohort | Example/state | Actual | Predicted | Absolute error | Share of cohort weighted SSE |
|---|---|---:|---:|---:|---:|
| Regular | 15 CB001_bubble_4 | -2.266 | -5.478 | 3.211 | 29.38% |
| Regular | 22 CB001_bubble_9 | -2.286 | -5.074 | 2.789 | 14.23% |
| Regular | 22 CB002_bubble_11 | -2.836 | -4.601 | 1.764 | 5.69% |
| Segmented | 46 | -23.094 | -12.513 | 10.580 | 47.50% |
| Segmented | 55 | -17.279 | -9.346 | 7.933 | 26.70% |
| Segmented | 22 | -8.867 | -13.035 | 4.168 | 7.37% |

The top three account for 49.30% of regular weighted SSE and 81.57% of segmented SSE. Next segmented case is 05, actual -6.535 h versus predicted -2.441 h, absolute error 4.094 h. Under pooled equal-network scoring, 05 ranks ahead of 55 because it belongs to a much smaller network sample: the top three pooled weighted-SSE contributors are 46 segmented (33.71%), 05 segmented (21.75%), and 55 segmented (19.93%). Outlier ranking therefore explicitly depends on the population/weighting, not just absolute error.

Post-hoc exclusion sensitivity, fixed predictions with weights/target mean/variance recomputed (no retraining):

| Population | Removed | Retained n | R² | Relative error |
|---|---|---:|---:|---:|
| Regular | None | 2170 | 0.7687 | 34.24% |
| Regular | 15 CB4 | 2169 | 0.8314 | 34.22% |
| Regular | 15 CB4 + 22 CB9 | 2168 | 0.8625 | 34.20% |
| Regular | All three listed regular cases | 2167 | 0.8721 | 34.19% |
| Segmented | None | 15 | 0.5961 | 33.11% |
| Segmented | 46 + 55 | 13 | 0.6551 | 31.14% |
| Segmented | 46 + 55 + 22 | 12 | 0.7312 | 29.82% |
| All, equal-network weights | All six listed cohort outliers | 2179 | 0.8719 | 34.19% |
| All, equal-cohort/network weights | All six listed cohort outliers | 2179 | 0.8546 | 32.01% |

Removing the three pooled-ranked outliers instead (46/05/55 segmented) gives equal-network pooled R² 0.9080, but this is a different exclusion set. Detailed top1/2/3/5/10 results and exact excluded IDs are saved. Exclusion scores are diagnostics of sensitivity, not replacement headline performance; removing extreme targets changes variance as well as error. The modest change in regular relative error despite a substantial R² increase shows that a few large misses dominate squared error without explaining most relative error.

Evidence: [audit script](investigations/cap_investigation_20261020/ten_input_outlier_audit.py), [scores and exclusions](outputs/uncapped_event_window_20261020/ten_input_outlier_audit_20261023/scores_and_exclusions.csv), [ranked predictions](outputs/uncapped_event_window_20261020/ten_input_outlier_audit_20261023/ranked_predictions.parquet), [cohort outliers](outputs/uncapped_event_window_20261020/ten_input_outlier_audit_20261023/largest_cohort_sse.csv), [relative-error contributors](outputs/uncapped_event_window_20261020/ten_input_outlier_audit_20261023/largest_relative_contributions.csv), [excluded IDs](outputs/uncapped_event_window_20261020/ten_input_outlier_audit_20261023/excluded_records.csv), [manifest](outputs/uncapped_event_window_20261020/ten_input_outlier_audit_20261023/manifest.json).

### Nested tuning and final artifact for the ten-input model (2026-10-23)

Completed the requested tuning for the exact ten predictors from panel 5 (replace mean-width ratio with explicit-depth resistance; add inverse-depth Delta G_h). There are 2,185 comparisons, 15 networks, no unmatched joins. This is a six-configuration search using the same existing candidate grid, not a broad new hyperparameter sweep. Total 304 fits: 270 inner fits (15 outer networks × six settings × three grouped folds), 15 outer fits, 18 final-selection fits, and one all-data model. Input features, target (full centroid shift), shared 0.5% window, network-balanced training weights and preprocessing conventions are fixed.

Each outer test network is excluded from all inner training/validation; GroupKFold(3) splits only the other 14 networks. Candidate selection minimizes the equal average of regular and segmented relative errors, with equal network weights within cohorts and the one-minute denominator floor. Preprocessing and target standardization are fitted only on the relevant training partition. The six candidates vary loss criterion, split/leaf constraints, depth, feature sampling and pruning. Outer choices: fine_absolute seven folds, fine_squared six, absolute_feature_sampling one, moderate_squared one. Baseline predictions are the prior ten-input model using settings inherited from nine-input tuning, aligned exactly to the new held-out records.

| Evaluation | Prior fixed-settings R² | Nested-tuned R² | Prior relative error | Nested-tuned relative error |
|---|---:|---:|---:|---:|
| Regular (2,170) | 0.7687 | 0.7705 | 34.24% | 32.69% |
| Segmented (15) | 0.5961 | 0.5720 | 33.11% | 34.40% |
| Combined, equal cohorts then equal networks | 0.7545 | 0.7399 | 33.67% | 33.55% |

Regular MAE improves 1.964 → 1.900 min; segmented MAE worsens 157.96 → 166.36 min. The tuning objective improves only 0.125 percentage points, while combined R² falls. This is a small tradeoff, not convincing broad improvement; tuning is not guaranteed to improve each held-out score. Without example54, the same pattern holds: combined relative error 35.29% → 35.16%, regular R² 0.7663 → 0.7681, segmented R² 0.5746 → 0.5492.

Notable tradeoffs: regular relative error improves in 13 (41.85% → 26.03%) and 41 (32.34% → 25.99%), but their segmented errors worsen (13: 14.90% → 29.29%; 41: 20.73% → 32.91%). Segmented 37 improves 15.80% → 9.34%. Example15 retains the same selected settings and errors, including CB4 and its segmented collapse. These differences explain why the relative-error objective changes little despite varying individual scores.

**Final saved candidate:** after the outer evaluation, three-fold grouped validation on all 15 networks selects `fine_squared`. Train a new forest on all 2,185 comparisons with these settings: 500 trees, squared_error criterion, max_depth=16, min_samples_leaf=2, min_samples_split=2, max_features=1.0, bootstrap=True, max_samples=None, ccp_alpha=0, random_state=20261014. This is one all-data model, separate from the 15 models used to estimate performance. Its selection score (32.736% relative error) is not an independent test score. Report the nested held-out scores above.

The artifact contains training-fitted preprocessing, target centering/scaling, estimator, ten-column schema, source fingerprint, software versions and training examples. Serialization/reload predictions agree to rtol=atol=1e-12; predictions from only the ten feature columns also match. The final schema is mean travelled-length ratio (log1p), edited parent-domain size, route-length SD ratio (log1p), explicit-depth resistance log ratio, width-CV ratio (log1p), route-entropy ratio (log1p), parent mean dominance, parent slope-slowness, length change in km, and inverse-depth Delta G_h. Output is signed centroid shift in hours.

The tuned model is saved as a separate candidate, not a production replacement. The earlier fixed-settings evaluation remains available. These networks have been reused in exploratory feature selection; nested hyperparameter tuning prevents outer-test leakage within this search but does not make prior feature exploration an independent external test. Only 15 independent segmented cases are available.

Reproduce/resume: `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python RAPID/investigations/cap_investigation_20261020/tune_ten_inputs.py --run --jobs 4`.

Evidence: [script](investigations/cap_investigation_20261020/tune_ten_inputs.py), [final model](outputs/uncapped_event_window_20261020/ten_input_tuning_20261023/final_model.joblib), [settings and schema](outputs/uncapped_event_window_20261020/ten_input_tuning_20261023/final_model.json), [held-out scores](outputs/uncapped_event_window_20261020/ten_input_tuning_20261023/scores.csv), [network scores](outputs/uncapped_event_window_20261020/ten_input_tuning_20261023/network_scores.csv), [outer selections](outputs/uncapped_event_window_20261020/ten_input_tuning_20261023/selections.json), [design](outputs/uncapped_event_window_20261020/ten_input_tuning_20261023/design.json), [completion](outputs/uncapped_event_window_20261020/ten_input_tuning_20261023/completion.json).

### Current two-model comparison and tuned outlier register (2026-10-23)

This is the consolidated reference for the two current model artifacts, before changing validation or revisiting predictors. The historical experiments above remain an audit trail. **Model A** is the accepted geometry-only nine-input `same_grid` forest; **Model B** is the newly nested-tuned ten-input depth-informed forest. Model B replaces mean-width ratio with equivalent resistance and adds the inverse-depth joint descriptor. Do not confuse Model B's scores with its earlier fixed-settings precursor, which had segmented R² 0.596 rather than the tuned 0.572. Both artifacts exist; no production model was replaced.

#### Response, population and predictor contract

Both predict **full excess-volume centroid shift**, `centroid_shift_hours`, signed child minus parent; negative means an advance. These are not peak-time scores. Use current uncapped outputs and the finalized shared 0.5% analysis window with the agreed fixed baseline/common boundaries. The primary sample is 2,185 comparisons from 15 networks: 2,170 regular collapse comparisons (including full compound collapses) and 15 segmented full single-centerline collapses. The combination-augmentation cases are not part of these fits. Example54 retains its finite-window response definition, including its equilibrium transition; its double crest does not trigger a selective response redefinition.

Let p/c denote parent/child, D the descriptor comparison domain, E the edited parent footprint, e a reach, r_e its traversal probability, L length, W width, h depth and s slope. Ratios below are child/parent. Regular D follows the saved B domain convention (containing compound where applicable); segmented D excludes unchanged connectors. Edited size S_L uses E rather than assuming every predictor uses the identical footprint.

| Predictor symbol/description | Input definition | Model A | Model B | Saved column |
|---|---|:---:|:---:|---|
| Mean travelled-length ratio | ln(1 + mu_L,c^D / mu_L,p^D) | Yes | Yes | `log1p__B_ratio_route_length_mean` |
| S_L, edited parent size | sum over E of r_e L_e / 1 km | Yes | Yes | `domain_size` |
| Route-length SD ratio | ln(1 + sigma_L,c^D / sigma_L,p^D) | Yes | Yes | `log1p__B_ratio_route_length_sd` |
| Mean-width ratio | ln(1 + Wbar_c^D / Wbar_p^D) | Yes | No | `log1p__B_ratio_width_mean` |
| R_h, equivalent-resistance change | -log10(K_eq,c / K_eq,p) | No | Yes | `resistance_explicit_depth` |
| Width-CV ratio | ln(1 + CV_W,c^D / CV_W,p^D) | Yes | Yes | `log1p__B_ratio_width_cv` |
| Route-entropy ratio | ln(1 + H_c^D / H_p^D) | Yes | Yes | `log1p__B_ratio_route_entropy` |
| Parent mean dominance | dbar_p^D | Yes | Yes | `B_parent_dominance_mean` |
| Parent slope context | sum(r_e L_e / sqrt(s_e)) / sum(r_e L_e), parent D | Yes | Yes | `parent_slope_slow` |
| Mean-length change | (mu_L,c^D - mu_L,p^D) / 1 km | Yes | Yes | `length_delta_km` |
| Delta G_h, inverse-depth joint change | G_h,c^D - G_h,p^D | No | Yes | `delta_G_inverse_depth` |

The exact finite input values and existing zero-denominator conventions are preserved from the saved input tables; this documentation does not redefine ratio handling. Model A uses descriptors obtainable from hydrography, including calculated probabilities and topology measures, rather than direct measurements alone. Model B adds derived-depth information through two combined descriptors:

\[
G_h^D=\sum_{e\in D}\frac{r_e L_e}{h_e\sqrt{s_e}},
\qquad \Delta G_h^D=G_{h,c}^D-G_{h,p}^D.
\]

L and h are in metres, so G_h is dimensionless. It assumes inverse depth and inverse square-root slope; there is **no Manning 2/3 depth exponent**. Equivalent resistance retains its existing edge law K_e proportional to L_e/(W_e^2 h_e^3), solved over network connections; it is not RAPID travel-time K. Resistance uses regular B domains but full segmented networks **including connectors**, unlike G_h. This scope distinction is intentional in the completed comparison and remains a possible future sensitivity check.

Stored depths currently follow the reference-depth/width relationship h=a_example W^0.65. Suitable observed reach depths at comparable discharge conditions could replace the depth estimates in either descriptor. A hypothetical child still needs a defensible child-depth estimate. Neither model takes roughness, celerity, output hydrographs, RAPID K/X, or example identity as input predictors. The estimated depths and the chosen resistance law remain assumptions, not independently verified measurements or a causal explanation.

#### Validation and four headline scores

Outer validation holds out **one entire network at a time** (15 folds). All regular and segmented states of that network stay outside fitting and tuning. Three inner GroupKFold folds select settings on the other 14 networks. Training gives each network equal total weight across its states; preprocessing and target scaling are learned only from training rows. Scoring within each cohort also gives networks equal total weight. Current folds are not stratified by centroid response.

| Model | Evaluation cohort | n | R² | MAE (minutes) | Adjusted relative error |
|---|---|---:|---:|---:|---:|
| A: geometry-only, 9 inputs | Regular collapses | 2,170 | **0.6386** | **2.554** | **51.68%** |
| A: geometry-only, 9 inputs | Segmented full single centerlines | 15 | **0.5100** | **169.60** | **36.07%** |
| B: depth-informed, 10 inputs | Regular collapses | 2,170 | **0.7705** | **1.900** | **32.69%** |
| B: depth-informed, 10 inputs | Segmented full single centerlines | 15 | **0.5720** | **166.36** | **34.40%** |

Adjusted relative error (sometimes called adjusted MAPE in this investigation) is

\[
100\frac{\sum_i w_i\,|\widehat{\Delta t_i}-\Delta t_i|/
\max(|\Delta t_i|,1\text{ minute})}{\sum_i w_i}.
\]

This is not ordinary MAPE. Small shifts remain included; the one-minute floor stabilizes their denominators. It is an average error score, not an error interval. R² and MAE use the same within-cohort network weighting; R² is computed against that population's weighted target mean.

The equal-cohort relative-error objective is **43.88% for A** and **33.55% for B** (average the two cohort error scores). Retain the four rows above as headline performance rather than mixing differently weighted pooled R² values. The largest improvement is in regular collapses. Model B has only a modest segmented improvement over A; its own tuning slightly worsened segmented performance compared with the ten-input fixed-settings precursor. Outlier-exclusion scores are diagnostics, not headline replacements.

Model A's accepted `same_grid` selection uses four candidate configurations; Model B's tuning uses six. Thus this final-artifact comparison is not a perfectly matched tuning-budget ablation; the preceding fixed-settings six-panel comparison documents the predictor changes separately. Hyperparameter tuning is nested, but predictor choices arose from repeated exploratory use of these networks. This is not independent external validation, and only 15 independent segmented cases are available.

#### Final saved random-forest settings

| Setting | Model A | Model B |
|---|---|---|
| `n_estimators` | 500 | 500 |
| `criterion` | `absolute_error` | `squared_error` |
| `max_depth` | 16 | 16 |
| `min_samples_leaf` | 1 | 2 |
| `min_samples_split` | 2 | 2 |
| `max_features` | 1.0 (all inputs available) | 1.0 (all inputs available) |
| `bootstrap` | True | True |
| `max_samples` | None | None |
| `ccp_alpha` | 0.0 | 0.0 |
| `random_state` | 20261014 | 20261014 |
| Final training population | All 2,185 comparisons | All 2,185 comparisons |

These settings belong to the final **all-data** artifacts. Each outer fold used its own training-only selected settings; the table is not retrospectively imposed on the held-out predictions. Settings were selected by grouped validation on all networks before the final all-data fit. The all-data selection scores and training predictions are not test performance. Artifacts include fitted preprocessing/target transforms, not just a bare forest; output is centroid shift in hours. Artifact hashes and predictor schemas were checked for this comparison; Model B serialization and ten-column-only predictions were verified to rtol=atol=1e-12.

- **Model A:** [artifact](outputs/uncapped_event_window_20261020/nine_input_tuning/research_model_same_grid.joblib), [settings/schema](outputs/uncapped_event_window_20261020/nine_input_tuning/research_model_same_grid.json), [outer predictions](outputs/uncapped_event_window_20261020/nine_input_tuning/predictions.parquet) (select `variant=same_grid`).
- **Model B:** [artifact](outputs/uncapped_event_window_20261020/ten_input_tuning_20261023/final_model.joblib), [settings/schema](outputs/uncapped_event_window_20261020/ten_input_tuning_20261023/final_model.json), [outer predictions](outputs/uncapped_event_window_20261020/ten_input_tuning_20261023/predictions.parquet) (select `variant=nested_tuned`).

#### Current largest errors in Model B

Rank by contribution to within-cohort, network-weighted squared error. Values are hours; negative means advance.

| Cohort | Case | Actual | Predicted | Absolute error | Cohort SSE share |
|---|---|---:|---:|---:|---:|
| Regular | 15 CB001_bubble_4 | -2.266 | -5.478 | 3.211 | 29.60% |
| Regular | 22 CB001_bubble_9 | -2.286 | -5.074 | 2.789 | 14.33% |
| Regular | 22 CB002_bubble_11 | -2.836 | -4.601 | 1.764 | 5.74% |
| Segmented | 46 | -23.094 | -12.467 | 10.627 | 45.22% |
| Segmented | 55 | -17.279 | -9.346 | 7.933 | 25.20% |
| Segmented | 41 | -13.244 | -8.885 | 4.359 | 7.61% |
| Segmented | 22 | -8.867 | -13.035 | 4.168 | 6.96% |
| Segmented | 05 | -6.535 | -2.503 | 4.032 | 6.51% |

The top three regular cases contribute 49.67% of regular weighted SSE; segmented46/55 together contribute 70.42%. Relative-error contributors can be different; their separate ranking is saved. Earlier outlier tables for the fixed-settings model must not be substituted for this tuned-model register.

#### Follow-up agenda — proposed, not executed

Preserve these models, inputs and held-out predictions as the reference before further work.

1. **Clarify the prediction task before changing validation.** Whole-network holdout tests transfer to unseen networks. A separate within-network experiment would ask whether untested collapses can be predicted when other states of that network are already represented. Report these as different tasks, not interchangeable estimates of the same performance.
2. **Consider balanced grouped validation for transfer.** Define bins of signed centroid shift (near-zero, advances/delays and magnitude classes) and predefined repeated fold assignments. Approximately balance response distributions while keeping whole networks together; retain nested tuning. Group integrity takes priority over balance. Do not choose seeds/bins because they give favorable scores. A unique extreme network cannot be spread across folds, so balancing does not repair missing independent hydraulic context. No centroid-stratified validation has been run for the current models.
3. **If testing within-network prediction, first define related-collapse groups.** Shared edited reaches, nested/containing domains and combination constituent units can create dependence. Keep related cases together or exclude overlapping training cases for a held-out group. Audit the resulting group sizes: giant groups may make this experiment impractical, and shared forcing/calibration remains part of its within-network scope.
4. **Revisit combination augmentation as a paired experiment if appropriate.** Use identical fixed test cases/folds with and without eligible added combinations. Exclude combinations overlapping held-out collapse groups; all states of an outer held-out network remain excluded for the transfer test. Additional combinations add context, not independent rivers. Keep their evaluation separate and do not create new cases or change weights merely to inflate scores. The previous 80-case experiment did not establish robust improvement.
5. **Revisit additional metrics after the validation question is fixed.** Use the current residual patterns to formulate a small, physically motivated set of candidates and inspect redundancy with the existing ten inputs. Keep both saved models as controls; evaluate additions/replacements with unchanged test folds and training-only tuning. Do not use child hydrograph diagnostics or fitted RAPID response quantities as input predictors. Depth provenance, resistance/G_h domain differences, extrapolation and routing artifacts remain alternative explanations, not reasons to assume that every residual needs another metric.

No new validation split, model fit, metric or combination run was performed for this documentation consolidation. The older `prediction_audit_20261006` used structure-only folds with response-bin diagnostics; that is not evidence that the current models were response-stratified. General grouped-stratification reference: [scikit-learn StratifiedGroupKFold](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedGroupKFold.html).

Evidence: [comparison script](investigations/cap_investigation_20261020/current_model_comparison.py), [four scores](outputs/uncapped_event_window_20261020/current_model_comparison_20261023/four_scores.csv), [final settings](outputs/uncapped_event_window_20261020/current_model_comparison_20261023/settings.json), [updated outliers](outputs/uncapped_event_window_20261020/current_model_comparison_20261023/largest_tuned_outliers.csv), [relative contributors](outputs/uncapped_event_window_20261020/current_model_comparison_20261023/largest_relative_contributors.csv), [manifest](outputs/uncapped_event_window_20261020/current_model_comparison_20261023/manifest.json).

### Validation dependence-group inventory (2026-10-23)

Completed the requested inventory before designing alternative folds. No RF fits, new predictions, routing runs or production changes. The population is exactly the current 2,185 comparisons: 2,170 regular and 15 segmented. All regular records match saved unit metadata, active jobs and validated parent descriptor geometry; no unmatched records. Current centroid responses come only from the accepted model input table, not older response values in the geometry metadata.

**Grouping construction.** Build unit-family components separately within each network from explicit `in_compound_bubble=True` memberships and all saved parent/child nesting relationships. A non-null bubble ID alone does not establish compound membership. Every comparison receives the families of its selected units, including the source/anchor unit for island edits. Connect comparisons sharing a family, then transitively merge any groups sharing parent reaches from the unit-link memberships. Merely sharing a boundary node does not merge groups. Also inspect overlap of parent B descriptor domains as a more conservative sensitivity. Full compound collapses and their member/nested/island cases remain together. Aliases identify canonical records; they are not expanded into additional observations.

For islands, source-unit membership is a conservative footprint, not a reconstructed pixel-level edit mask. Group labels therefore protect the saved anchor relationship. These are non-overlapping *dependence groups*, not claims of statistical independence: forcing, calibration and network context remain shared within each network.

**Main result: 435 regular-case groups.** Family-only, family-plus-shared-reaches and family-plus-reaches-plus-descriptor-context all produce the same 435 groups, with identical case partitions. There are 177 groups containing a full compound collapse and 258 standalone simple groups. Thus explicit compound/nesting families already capture the inspected reach/context overlaps. Regular edit counts are 177 full compounds, 1,467 compound members, 268 island cases and 258 simple cases. The mutually exclusive display hierarchy labels nested compound states under their compound type: 1,102 records across 81 groups carry the separate nested-structure flag; zero standalone Nested-category rows does not mean nesting is absent.

| Network | Regular comparisons | Groups | Largest group | Share of network |
|---|---:|---:|---:|---:|
| 02 | 35 | 6 | 17 | 48.6% |
| 03 | 40 | 15 | 10 | 25.0% |
| 05 | 51 | 34 | 4 | 7.8% |
| 13 | 66 | 30 | 16 | 24.2% |
| 15 | 262 | 11 | 158 | 60.3% |
| 21 | 30 | 19 | 4 | 13.3% |
| 22 | 408 | 14 | 208 | 51.0% |
| 37 | 395 | 3 | 383 | 97.0% |
| 41 | 181 | 37 | 23 | 12.7% |
| 45 | 93 | 34 | 7 | 7.5% |
| 46 | 223 | 107 | 9 | 4.0% |
| 48 | 72 | 25 | 12 | 16.7% |
| 54 | 81 | 35 | 8 | 9.9% |
| 55 | 212 | 56 | 25 | 11.8% |
| 56 | 21 | 9 | 5 | 23.8% |

Five folds are structurally feasible for regular cases: 435 groups are available globally, and 14/15 networks have at least five groups. This does **not** imply balanced folds or independent observations. Network37 has only three groups: 383, 11 and 1 records. Its CB001_bubble_3 family holds 97% of its regular cases and 237 near-zero responses. A nominal equal-size regular fold contains 434 records, so that single group already fills 88% of a fold. Networks15 and22 also have large families (158 and208 records). Network46, by contrast, has107 groups, largest9 records; within-network grouping could expose much more same-network context there than whole-network holdout, without splitting a compound family.

**Response/type balancing limits.** Descriptive thresholds were specified before counting, not optimized for model scores: near-zero abs(centroid shift)<=1 minute; signed advance/delay magnitude bands (1,5], (5,60], and >60 minutes. Regular support is:

| Response band | Comparisons | Distinct dependence groups | Networks |
|---|---:|---:|---:|
| Near-zero, <=1 min | 846 | 85 | 13 |
| Advance, 1–5 min | 655 | 257 | 15 |
| Advance, 5–60 min | 524 | 295 | 15 |
| Advance, >60 min | 15 | 13 | 6 |
| Delay, 1–5 min | 92 | 21 | 8 |
| Delay, 5–60 min | 37 | 9 | 5 |
| Delay, >60 min | 1 | 1 | 1 |

Groups can contain multiple bands/types, so these group counts are not additive. The sole delay over one hour is 22/island_22_018 and cannot be represented in every fold. Crossed response×type strata are even sparser (e.g. two simple delays and two compound-member advances over one hour). Fold balance should therefore be approximate, prioritize keeping groups intact and consider coarse response bands plus separate type margins rather than requiring every fine intersection in every fold. Do not oversample or break families merely to achieve a balanced validation table. No fold assignments have yet been selected.

**Segmented cases join whole networks.** Adding each segmented case's *edited* parent footprint, even excluding unchanged connectors, merges all regular groups of its network. The result is exactly15 groups, one per network. Including the full-network resistance context produces the same result. Segmented cases therefore cannot be included as ordinary independent training rows in a mixed-within-network experiment while claiming no overlap with held-out unit families.

**Recommended interpretation of the three planned strategies:**

1. Current whole-network holdout remains the transfer baseline, including both cohorts.
2. Approximate response/type-balanced whole-network five-fold validation can include both cohorts, with nested tuning and predefined repetitions. Group37's size and the unique large delay limit balance.
3. **Updated user decision: mixed-network validation includes all 2,185 comparisons.** Keep the 435 regular compound-family/reach groups intact, and assign each of the 15 segmented full-network collapses its own singleton group (450 groups total). This explicitly permits a full-collapse case and its related regular groups to appear on opposite sides of a split. It tests prediction among related states of represented networks, not complete geometric independence. This supersedes the initial regular-only recommendation; there is no need to remove segmented cases or create regular-only controls on account of that earlier recommendation. Report how much same-network context remains in training for each held-out group, particularly37.

For later augmentation, assign cases by constituent/edited footprint. Added combinations overlapping a held-out family must be excluded from training. A combination connecting several otherwise separate groups can join them transitively, so consider fold-specific eligibility/purging instead of silently treating it as an independent new row. No augmentation or validation benchmark was started by this inventory.

**Checks:** active parent topology/length/width agree with saved descriptor geometry; selected-unit footprints lie inside their B domains; group assignments preserve every inspected family and reach relationship; all input hashes remained unchanged. Four focused tests passed for transitive family/overlap grouping, segmented bridging/context sensitivity, record-order invariance and response-bin boundaries.

Evidence: [inventory script](investigations/cap_investigation_20261020/validation_group_inventory.py), [tests](investigations/cap_investigation_20261020/test_validation_group_inventory.py), [network summary](outputs/uncapped_event_window_20261020/validation_group_inventory_20261023/network_summary.csv), [case assignments](outputs/uncapped_event_window_20261020/validation_group_inventory_20261023/group_assignments.csv), [group sizes/mixes](outputs/uncapped_event_window_20261020/validation_group_inventory_20261023/group_summary.csv), [response×type support](outputs/uncapped_event_window_20261020/validation_group_inventory_20261023/stratum_support.csv), [network mixes](outputs/uncapped_event_window_20261020/validation_group_inventory_20261023/network_mix.csv), [record metadata](outputs/uncapped_event_window_20261020/validation_group_inventory_20261023/records.parquet), [relationship edges](outputs/uncapped_event_window_20261020/validation_group_inventory_20261023/relation_edges.parquet), [manifest](outputs/uncapped_event_window_20261020/validation_group_inventory_20261023/manifest.json).

Reproduce: `python RAPID/investigations/cap_investigation_20261020/validation_group_inventory.py`; tests: `python -m unittest discover -s RAPID/investigations/cap_investigation_20261020 -p test_validation_group_inventory.py`.

### Accepted full-collapse grouping exception for strategy 3 (2026-10-23)

The user confirmed that each segmented full-network collapse should be treated as its own group, rather than merged with its constituent compound/unit groups. Saved a new grouping artifact while preserving the original strict-overlap inventory: **2,185 comparisons, 450 groups = 435 unchanged regular groups + 15 full-collapse singletons**. Each full-collapse case is one ordinary comparison; this grouping choice does not give it special weight or require a separate model. These are 15 individual singleton groups, not one combined group containing all full-collapse cases.

Strategies1/2 still keep entire networks together, including full-collapse states. Strategy3 intentionally allows a segmented full-collapse record to be in training while related regular cases are held out, and vice versa. Thus the strict-overlap inventory's conclusion remains geometrically true but is not the adopted strategy3 constraint. Compound/member/nested/island relationships among regular cases remain intact. Separate type-level reporting can still diagnose errors without separating training. The completed alternative-validation benchmark below now reports the folds and results from this grouping.

Checks verify unique matching of all 2,185 records, singleton membership for all15 segmented cases, disjoint singleton/regular group IDs, one network per group, and exact preservation of all435 regular groups. Inputs and source hashes remained unchanged.

Evidence: [preparation script](investigations/cap_investigation_20261020/prepare_validation_groups.py), [accepted assignments](outputs/uncapped_event_window_20261020/validation_groups_full_collapse_singletons_20261023/group_assignments.csv), [network counts](outputs/uncapped_event_window_20261020/validation_groups_full_collapse_singletons_20261023/network_summary.csv), [response/type support](outputs/uncapped_event_window_20261020/validation_groups_full_collapse_singletons_20261023/stratum_support.csv), [manifest](outputs/uncapped_event_window_20261020/validation_groups_full_collapse_singletons_20261023/manifest.json).

### Completed alternative validation benchmark (2026-10-23)

Completed both requested alternative validation strategies for the two fixed predictor sets. **1,900 forest fits**, 100 outer model/fold evaluations, five distinct repetitions of five outer folds per strategy, three inner folds per outer training set, and six candidate settings for each model. All 2,185 comparisons appear exactly once in testing per repetition/model/strategy (43,700 saved predictions in total). No augmentation, new predictors, changed targets or production model replacement. The existing saved all-data forests remain unchanged.

**Strategies and interpretation.**

- `network_balanced`: hold whole networks together, including their full-collapse states. Each outer test fold has three networks and each training set has12. Three inner folds hold out four of those12 networks at a time. This tests transfer to unseen networks.
- `compound_mixed`: use the agreed450 groups (435 regular compound-family/reach groups plus15 full-collapse singletons). Keep each regular family intact; permit the explicit full-collapse/component overlap exception. Every outer test case has other cases from the same network in training. This tests untested related states in represented networks, not independent transfer.
- Both train with the previous equal-network weights, training-only preprocessing and target standardization. Both tune on the equal-cohort average of network-balanced relative errors, with a one-minute denominator floor. Full-collapse cases remain ordinary training rows under those existing weights; the scoring/tuning objective is unchanged.

**Fold construction.** Predefined seeds20261024,20262033,20263042,20264051,20265060. Balance sample count and separate response/type margins using32 deterministic seeded greedy restarts per assignment, choosing only by distribution balance, never model performance. The response classes use the inventory's1/5/60-minute bands, merging the single >60-minute delay with other >5-minute delays for assignment; detailed bins remain in the diagnostics. Group counts have a small additional balancing term. All repetitions have distinct partitions. Each test fold contains three segmented cases, and each inner validation fold four. Test sizes range282–542 for whole-network folds and408–514 for mixed folds. Balancing is approximate: the large37 family remains indivisible, and broad >60-minute bins do not distinguish all large full-collapse magnitudes.

**How scores are aggregated.** Assemble one complete out-of-fold prediction vector per repetition, compute R²/MAE/adjusted relative error using the established weights, then report the mean/min/max/std across repetitions. Do not average predictions before scoring; doing that would evaluate an ensemble. Repeat spread describes split sensitivity, not a confidence interval. All scores below predict centroid shift, not peak timing. Regular cases include full compound collapses; the full-collapse column refers to the15 segmented full single centerlines.

| Model | Validation | Regular R² | Regular MAE (min) | Regular adjusted error | Full-collapse R² | Full-collapse MAE (min) | Full-collapse adjusted error |
|---|---|---:|---:|---:|---:|---:|---:|
| A: geometry, 9 | Existing LOEO | 0.639 | 2.55 | 51.68% | 0.510 | 169.60 | 36.07% |
| A: geometry, 9 | Balanced network five-fold | 0.639 | 2.64 | 52.94% | 0.514 | 170.86 | 39.22% |
| A: geometry, 9 | Mixed network, family-grouped five-fold | 0.676 | 2.34 | 46.38% | 0.491 | 169.66 | 40.74% |
| B: depth, 10 | Existing LOEO | 0.770 | 1.90 | 32.69% | 0.572 | 166.36 | 34.40% |
| B: depth, 10 | Balanced network five-fold | 0.801 | 1.89 | 33.59% | 0.568 | 166.12 | 36.72% |
| B: depth, 10 | Mixed network, family-grouped five-fold | 0.842 | 1.61 | 28.17% | 0.525 | 163.60 | 35.08% |

New-strategy rows are five-repetition means. Existing LOEO rows are the retained single leave-one-network-out evaluation, with different training sets (14 rather than12 entire networks in each transfer fit).

**Matched tuning-budget control.** The accepted geometry baseline used the four-candidate `same_grid` procedure, whereas both new strategies use six candidates. Its already available six-candidate LOEO control gives regular R²0.621, MAE2.639min, relative error52.66%; segmented R²0.503, MAE177.81min, relative error41.78%. That control is saved alongside the main comparison so changes cannot all be attributed to the validation split. The depth-informed LOEO baseline already used the same six candidates. Predictor selection remains exploratory on reused networks; this benchmark is not new external validation.

**Main findings.** Balanced whole-network five-fold validation broadly confirms the earlier transfer picture. For Model B, regular R² increases to0.801, but regular MAE is nearly unchanged (1.900→1.892min), adjusted error increases32.69%→33.59%, and full-collapse performance shows no consistent gain. It is a sensitivity estimate under different training/test mixtures, not a remedy for the residuals.

Mixed-network grouping improves regular predictions: Model B mean R²0.842, MAE1.610min and adjusted error28.17%, versus0.770/1.900min/32.69% under LOEO. Its adjusted regular error improves on average in all15 networks, most in46 (82.61%→61.81%),48 (25.82%→13.81%) and55 (23.37%→14.03%). Improvements are smaller in37 (28.21%→27.38%), whose huge family still leaves limited same-network regular context when held out. Model A also improves regular adjusted error51.68%→46.38%.

Full-collapse results do not show the same reliable improvement. In the mixed strategy, Model B mean full-collapse R² is0.525 (LOEO0.572), adjusted error35.08% (LOEO34.40%), while MAE is slightly lower163.60min (LOEO166.36). Full-collapse41 improves in mean absolute error261.55→115.11min, but46 worsens637.61→713.14min and55 worsens476.00→515.40min;15 remains large at214.25min versus208.16. Thus having other units of the same river in training does not resolve the largest-collapse/extrapolation issue.

**Repeat sensitivity.** Model B network-balanced regular R² ranges0.765–0.845 and full-collapse R²0.501–0.610. Mixed-network regular R² ranges0.801–0.862, adjusted error27.75–28.42%; full-collapse R² ranges0.404–0.592, adjusted error32.97–39.25%. The full-collapse sample has only15 cases and remains sensitive to which large cases are held out together. As a concrete aggregation warning, mixed-network37 full-collapse mean prediction nearly equals the target (-7.760 versus-7.764h), but its mean absolute error across repetitions is44.36min: opposite-signed errors cancel when predictions are averaged. The primary scores correctly do not use that ensemble-like averaging.

The equal-cohort adjusted-error objectives are A43.88%/46.08%/43.56% and B33.55%/35.15%/31.62% for existing LOEO/network-balanced/mixed, respectively. These summarize different generalization tasks and should not be used to pick a validation strategy simply because it yields the lowest number. Excluding54 for scoring preserves the main Model B contrast: network-balanced regular R²0.800/error34.04%, mixed regular R²0.841/error29.15%; full-collapse R²0.546/0.500 and errors37.99%/36.24%, respectively.

**Implication for further work:** retain whole-network transfer as the primary unseen-network claim, and keep the mixed-family benchmark as a separately labelled known-network result. It provides a useful baseline for a later paired augmentation experiment, with identical folds/test cases and permitted combinations only from training information. It does not yet demonstrate that new combinations or additional metrics help. No new augmentation or metric investigation was started here.

**Verification:** three focused splitter tests passed; all50 outer and150 inner partitions were checked for prohibited group overlap and full-collapse quotas. All predictions are finite, every expected comparison appears exactly once per repetition/model/strategy, all source fingerprints remain unchanged, and completion reports zero unmatched records. Saved scores include all-data and without54 sensitivity, collapse-type and network breakdowns, fold mixes, settings selections and case-level mean errors. Final-model artifacts are unchanged.

Reproduce/resume: `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python RAPID/investigations/cap_investigation_20261020/alternative_validation.py --run --workers 4`. Four independent workers each fit forests with n_jobs=1; checkpoints permit resuming completed candidates/folds.

Evidence: [benchmark script](investigations/cap_investigation_20261020/alternative_validation.py), [splitter tests](investigations/cap_investigation_20261020/test_alternative_validation.py), [mean/range summary](outputs/uncapped_event_window_20261020/alternative_validation_20261023/summary.csv), [per-repetition scores](outputs/uncapped_event_window_20261020/alternative_validation_20261023/repeat_scores.csv), [existing LOEO scores](outputs/uncapped_event_window_20261020/alternative_validation_20261023/baseline_scores.csv), [six-candidate geometry control](outputs/uncapped_event_window_20261020/alternative_validation_20261023/matched_grid_baseline.csv), [collapse-type scores](outputs/uncapped_event_window_20261020/alternative_validation_20261023/type_scores.csv), [network scores](outputs/uncapped_event_window_20261020/alternative_validation_20261023/network_scores.csv), [depth-model network comparison](outputs/uncapped_event_window_20261020/alternative_validation_20261023/depth_network_comparison.csv), [case errors](outputs/uncapped_event_window_20261020/alternative_validation_20261023/case_error_summary.csv), [depth-model case comparison](outputs/uncapped_event_window_20261020/alternative_validation_20261023/depth_case_comparison.csv), [fold assignments](outputs/uncapped_event_window_20261020/alternative_validation_20261023/outer_assignments.csv), [fold audit](outputs/uncapped_event_window_20261020/alternative_validation_20261023/fold_audit.csv), [fold response/type mixes](outputs/uncapped_event_window_20261020/alternative_validation_20261023/fold_mixes.csv), [design](outputs/uncapped_event_window_20261020/alternative_validation_20261023/design.json), [completion](outputs/uncapped_event_window_20261020/alternative_validation_20261023/completion.json).

### Held-out SHAP comparison across validation strategies (2026-10-24)

The follow-up explains both current predictor sets under the existing LOEO, balanced whole-network and mixed compound-family validation strategies. It refits the **130 saved selected outer forests**, without repeating hyperparameter search or changing the production models. Each refit must reproduce its saved held-out predictions before explanations are accepted. All held-out rows are explained, including all five repetitions of the new strategies; the full response remains excess-volume centroid shift.

The implementation uses SHAP 0.48.0 exact `TreeExplainer` with explicit `tree_path_dependent` semantics and the fitted forest's training cover as its reference. No held-out observations are used to construct that reference. Both target standardization and its offset are reversed: the reference and SHAP values are reported in **minutes**, and their sum is checked against the predicted centroid shift. Negative contributions move the prediction earlier relative to its fitted reference; positive contributions move it later. Positive SHAP does not necessarily mean a positive predicted shift. Predictors omitted by training-only preprocessing receive zero contribution and are recorded in the fold metadata.

Mean absolute SHAP is computed separately for regular units and segmented full collapses, with equal total weight per network within each cohort. Contributions are made absolute **before** averaging; then complete-repetition estimates are averaged. Reported repeat ranges are split sensitivity, not confidence intervals. Importance shares are shares of mean absolute SHAP, **not explained variance or percentages of R²**. Dependence plots use the predefined first repetition for readability and retain individual state identities. Scatter density is not network-balanced; importance bars are.

These are explanations of fitted predictions, not causal effects, significance tests, or decompositions of prediction error. Correlated inputs can share or exchange credit. Different training folds have different reference predictions, so changes combine training-context and RF-setting effects. The accepted geometry LOEO model used four tuning candidates, whereas the two new strategies used six; the depth-informed comparisons all used six. The mixed strategy retains the previously authorized full-collapse singleton exception and estimates known-network related-state prediction.

Method reference: [SHAP TreeExplainer documentation](https://shap.readthedocs.io/en/latest/generated/shap.TreeExplainer.html). Script: [validation_shap.py](investigations/cap_investigation_20261020/validation_shap.py). Output directory: `outputs/uncapped_event_window_20261020/validation_shap_20261024/`. The run is checkpointed; `completion.json` is written only after all 130 folds and 48,070 held-out explanations have passed checks and reports are generated.

Run with the existing model environment plus `shap==0.48.0`: `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python RAPID/investigations/cap_investigation_20261020/validation_shap.py --run --workers 4`. For this execution SHAP was installed into an isolated temporary virtual environment using the existing environment's scientific libraries; the project environment was not modified.

**Completed results.** All 130 selected forests and 48,070 held-out explanations completed. Maximum deviation from saved predictions was 3.56×10⁻¹⁵ hours; maximum SHAP reconstruction error was 3.30×10⁻¹¹ minutes. No predictor was dropped in these fits. All input/source hashes were rechecked. An export-only correction converted mixed string/integer fold identifiers to strings; fits and explanations were retained, with the provenance migration recorded in `report_serialization_fix.json`.

For regular units, the depth-informed model has the following mean absolute SHAP shares (equal network weighting):

| Predictor | LOEO | Balanced networks | Mixed families |
|---|---:|---:|---:|
| ΔGh | 69.1% | 64.8% | 68.5% |
| SL (edited parent size) | 10.8% | 13.7% | 11.4% |
| ΔμL / km | 10.5% | 11.6% | 10.6% |
| −log10(Keq,c / Keq,p) | 3.7% | 3.6% | 3.5% |
| Parent slope factor | 1.9% | 2.0% | 1.9% |
| ln(1 + μL,c / μL,p) | 1.3% | 1.4% | 1.4% |
| ln(1 + Hc / Hp) | 0.9% | 1.2% | 0.9% |
| d̄p | 0.8% | 0.8% | 0.9% |
| ln(1 + CVW,c / CVW,p) | 0.5% | 0.5% | 0.6% |
| ln(1 + σL,c / σL,p) | 0.4% | 0.3% | 0.3% |

The central predictor pattern is stable: ΔG_h contributes 64.8–69.1%, edited parent size 10.8–13.7%, and length change 10.5–11.6%. Resistance contributes 3.5–3.7%. Geometry-only regular predictions rely chiefly on parent size (42.5–47.0%) and length change (26.8–29.1%), with width ratio (8.1–10.2%) and parent slope factor (9.1–9.4%) next. Thus the mixed-validation score gain is not accompanied by a wholesale switch in the dominant predictors. This descriptive comparison does not isolate the cause of that gain.

Segmented full collapses are even more concentrated: geometry-only parent size plus length change account for approximately 94–95%; depth-informed ΔG_h plus those two length terms account for approximately 97.2–98.0%. Near-zero entropy/SD attribution in this cohort is consistent with its restricted predictor range and does not establish that those variables are useless for regular units.

Direction is also stable. For regular depth-informed cases, the network-weighted rank association between ΔG_h and its SHAP contribution is +0.93 to +0.95: more negative ΔG_h generally contributes toward an earlier centroid. Parent size has an association of −0.54 to −0.56, and the resistance change −0.59 to −0.62. These describe associations in held-out model explanations, not physical derivatives. Dependencies remain nonlinear, and the fold reference varies. Small global SHAP shares alone do not justify deleting predictors; removal must be tested through held-out prediction performance.

Inspect the [interactive importance and dependence report](outputs/uncapped_event_window_20261020/validation_shap_20261024/report.html), [compact comparison figure](outputs/uncapped_event_window_20261020/validation_shap_20261024/importance_comparison.png), [importance summary](outputs/uncapped_event_window_20261020/validation_shap_20261024/importance_summary.csv), [per-repetition importance](outputs/uncapped_event_window_20261020/validation_shap_20261024/importance_by_repeat.csv), [held-out contributions and input values](outputs/uncapped_event_window_20261020/validation_shap_20261024/values.parquet), and [manifest](outputs/uncapped_event_window_20261020/validation_shap_20261024/manifest.json). Production models and response tables remain unchanged. No augmentation was performed.

### Separate peak-target forest at frozen depth-informed settings (2026-10-26)

A separate forest was fitted to `peak_shift_hours`, keeping the ten depth-informed predictors and final centroid RF settings: 500 trees, squared-error criterion, maximum depth16, leaf size2, minimum split2, all features, bootstrap, no pruning, seed20261014. The original centroid artifact was not changed. This experiment fits15 whole-network holdouts plus a separate all-data peak model. There is no new hyperparameter search. Preprocessing, target mean/SD and equal-network training weights are learned within each training fold. Peak timing retains the global-maximum definition in the shared0.5% window, including54; no crest matching was introduced.

All scores below evaluate **peak-time shift**. The proxy column means the existing nested centroid predictions evaluated against peak timing, not against centroid timing. Scores give equal total weight to each network within each cohort. Adjusted error uses the existing one-minute denominator floor.

| Prediction | Cohort | Peak R² | Peak MAE (min) | Adjusted peak error |
|---|---|---:|---:|---:|
| Existing nested centroid prediction as proxy | Regular (2170) | 0.328 | 4.323 | 54.31% |
| Separate peak-target forest, fixed settings | Regular (2170) | 0.256 | 4.681 | 67.75% |
| Existing nested centroid prediction as proxy | Segmented full collapses (15) | 0.193 | 252.19 | 42.73% |
| Separate peak-target forest, fixed settings | Segmented full collapses (15) | 0.328 | 231.18 | 54.66% |

Excluding54 **from scoring only**, regular peak R² is0.350 (proxy0.626), MAE4.371min (proxy4.039), adjusted error65.11% (proxy54.80%). For the14 segmented cases, peak R² is0.519 (proxy0.388), MAE200.50min (proxy222.25), adjusted error52.53% (proxy39.65%). Other folds may still contain54 in training; this is not an entirely54-excluded training experiment. The pattern is mixed: dedicated peak training improves absolute performance on the large full collapses, but worsens regular performance and relative error. It does not justify replacing the centroid model or conclude that peak-specific tuning cannot help.

The largest regular weighted squared-error contribution is54/CB004_bubble_16 (RAPID−598min, predicted−60.90min), followed by22/CB001_bubble_9 (−184 versus−582.09min) and22/CB002_bubble_11 (−235 versus−606.69min). Largest segmented errors include54 (−782 versus−121.28min),05 (−767 versus−183.46min),46 (−1319 versus−836.69min), and55 (−1030 versus−669.05min). These are diagnostic values, not reasons to redefine or selectively exclude targets.

**Interpretation limits:** the peak run uses frozen settings selected through earlier centroid investigations on these same networks; it is a fixed-setting held-out diagnostic, not a newly nested peak-tuning estimate or independent external validation. Its comparison with the nested centroid proxy combines target and fold-setting differences. A historical `centroid_proxy_fixed_settings` comparison is also exported; those historical fixed settings came from the earlier panel investigation and must not be described as necessarily identical to the final settings used here. No multi-output forest was fitted.

Verification:2185 unique held-out peak predictions,15 non-overlapping network holdouts, finite predictor/target/prediction checks, no unmatched joins, source hashes unchanged, and all-data model serialization and predictor-only input checks passed. The output artifact identifies its target explicitly as `peak_shift_hours`. The all-data fit is for application; evaluation plots must use the held-out predictions.

Files: [script](investigations/cap_investigation_20261020/peak_target_check.py), [held-out peak predictions](outputs/uncapped_event_window_20261020/peak_target_check_20261026/predictions.parquet), [comparison scores](outputs/uncapped_event_window_20261020/peak_target_check_20261026/scores.csv), [network scores](outputs/uncapped_event_window_20261020/peak_target_check_20261026/network_scores.csv), [regular outliers](outputs/uncapped_event_window_20261020/peak_target_check_20261026/largest_errors_original.csv), [segmented outliers](outputs/uncapped_event_window_20261020/peak_target_check_20261026/largest_errors_segmented.csv), [final peak forest](outputs/uncapped_event_window_20261020/peak_target_check_20261026/final_peak_model.joblib), [model metadata](outputs/uncapped_event_window_20261020/peak_target_check_20261026/final_peak_model.json), [design](outputs/uncapped_event_window_20261020/peak_target_check_20261026/design.json), [completion](outputs/uncapped_event_window_20261020/peak_target_check_20261026/completion.json).

Reproduce/resume: `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python RAPID/investigations/cap_investigation_20261020/peak_target_check.py --run --jobs 4`.

### Frozen held-network failure audit (2026-10-26)

This investigation freezes the two models identified by
[`current_model_comparison_20261023/settings.json`](outputs/uncapped_event_window_20261020/current_model_comparison_20261023/settings.json).
**No fitting, tuning, replacement, routing rerun, target alteration or predictor
alteration was performed.** Earlier explanations in this README were treated as
hypotheses. The evidence below was rebuilt from saved predictions, fold
checkpoints, response tables, hydrographs and routing parameters before drawing
mechanistic conclusions.

#### Freeze and independent provenance checks

The [audit manifest](outputs/uncapped_event_window_20261020/frozen_failure_audit_20261026/manifest.json)
records full SHA-256 hashes for settings, both artifacts, both prediction files
and every consumed source. Artifact hashes are
`9834f719ac7dff7aee000f45be27261385d23a085c12692a9c06c615fd0c93f6`
for geometry_9 and
`896420f1c2c4894ec8088dd2fd327035099b14baf332bfa1d1d6c3c4261675fc`
for depth_informed_10. Settings hash:
`5bea17a4913ab6270daaee3359feae2790facbec49138938f8f3500f8f784d44`.
The audit refuses subsequent runs if a protected file changes.

Only `variant=same_grid` from `nine_input_tuning/predictions.parquet` and
`variant=nested_tuned` from `ten_input_tuning_20261023/predictions.parquet` were
used. These are **saved nested held-network procedure predictions**, not
predictions from the all-data artifacts: 7/15 geometry folds and 9/15 depth folds
have settings different from their final artifact. Every saved prediction was
matched exactly to its selected outer checkpoint. All 30 outer selections have
14 training networks; each recorded inner train/validation partition is disjoint
and excludes the outer held network. All 21 source-hash entries in the two tuning
designs and all eight entries in the previous comparison manifest still match.
This checks the recorded provenance; without refitting it is not an independent
reproduction of the forest computations.

Both files contain the same 2,185 unique `(example,state)` targets: 2,170 regular
collapses and 15 `all_closed_units_collapsed` segmented cases. Regular compound
collapses stay regular; legacy uniform full collapses and augmentation pairs are
excluded. The target is full excess-volume centroid shift in hours, **child minus
parent**. Negative is an advance. It is neither peak shift nor an observed gauge
response to an actual collapse.

The response source was independently assembled from the uncapped review's seven
replaced networks and the extension's eight retained networks. All 2,185 targets
match exactly. For the 22 distinct investigated failures/successes, the audit
also verified saved outlet hashes and recomputed parent/child centroids at all
three saved windows (66 comparisons). It used three-point Gaussian integration
of the piecewise-linear excess discharge, independently of the production
moment routine; target differences are below 1e-8 hours. Parent and child use
identical baseline and boundaries at each window. This verifies arithmetic and
file lineage, **not the physical validity of the baseline, finite window,
hydraulic assumptions or underlying observations**. The 0.5% target remains the
primary target throughout.

Recomputed cohort scores agree with the current comparison:

| Model | Cohort | R² | MAE, minutes | One-minute-floored relative error |
|---|---|---:|---:|---:|
| Geometry 9 | Regular | 0.6386 | 2.554 | 51.68% |
| Depth-informed 10 | Regular | 0.7705 | 1.900 | 32.69% |
| Geometry 9 | Fully segmented | 0.5100 | 169.604 | 36.07% |
| Depth-informed 10 | Fully segmented | 0.5720 | 166.364 | 34.40% |

#### Explicit selection and successful comparisons

For each model and cohort independently, let `e = prediction - target` and
`w = 1 / number_of_cases_in_that_network_and_cohort`. Three rankings are saved:
absolute error `|e|`; squared-error contribution `w e² / sum(w e²)`; and
relative-error contribution `w |e|/max(|target|,1/60)` normalized over the cohort.
The top five of each are retained. Detailed support/physical investigation uses
the union of the top three absolute errors, top three squared-error contributions
and top two relative contributions. Thus cases were selected by error, not by
agreement with an earlier explanation. No outliers are removed from scores.

A successful comparison has `|e| <= max(1/60 hour, 0.2 |target|)`. Controls come
from another network, the same cohort and target sign, preferring target
magnitudes within a factor of two. Among those, choose the nearest in that
model's predictor space using RMS distance scaled by the failure fold's
training-only feature IQR (SD, then 1, if zero). If no magnitude match exists,
record that explicitly. This is retrospective outcome matching, not a prospective
success classifier. For tiny responses the one-minute tolerance can admit a
predicted sign error. The distance is a descriptive metric, not forest leaf
membership, and row-based IQRs reflect the numerous small regular collapses.

**Regular large errors.** Values below are hours; positive error underpredicts
an advance and negative error overpredicts it.

| Network/state | Target | Geometry prediction (error) | Depth prediction (error) |
|---|---:|---:|---:|
| 22 / CB001_bubble_9 | -2.286 | -7.321 (-5.035) | -5.074 (-2.789) |
| 22 / CB002_bubble_11 | -2.836 | -6.365 (-3.529) | -4.601 (-1.764) |
| 37 / CB001_bubble_3 | -7.461 | -5.083 (+2.379) | -6.944 (+0.518) |
| 15 / CB001_bubble_4 | -2.266 | -1.885 (+0.381) | -5.478 (-3.211) |

Geometry's first three cases contribute 51.09% of regular weighted squared
error. Depth's cases 15/CB001, 22/CB001 and 22/CB002 contribute 49.67%.
The two network-22 rows are related evidence from **one** held network, not two
independent replications.

For geometry, 15/CB001 is a successful same-sign/magnitude comparator to both
network-22 failures (distances 11.02 and 10.85). For depth, 46/CB001_bubble_90 is
the selected successful comparator to all three large failures: target -1.869,
prediction -2.148, error -0.279 hours; distances are 7.58, 13.13 and 14.04.
These are the nearest eligible controls, **not close feature twins**. Network
37's geometry failure has no other regular training case within the factor-two
target band; its fallback success has only 30% of its target magnitude. The
same case succeeds under depth, while 15/CB001 changes in the opposite direction.
This disproves a blanket claim that the depth panel fixes all large regular
misses. Because the procedures also differ in tuning, it does not isolate a
causal effect of either added/replaced descriptor.

**Fully segmented large errors.** Every row represents the full segmented
collapse in that network.

| Network | Target | Geometry prediction (error) | Depth prediction (error) |
|---|---:|---:|---:|
| 46 | -23.094 | -11.593 (+11.500) | -12.467 (+10.627) |
| 55 | -17.279 | -7.951 (+9.328) | -9.346 (+7.933) |
| 22 | -8.867 | -14.098 (-5.231) | -13.035 (-4.168) |
| 41 | -13.244 | -13.666 (-0.422) | -8.885 (+4.359) |

Networks 46/55 account for 76.68% of geometry and 70.42% of depth segmented
weighted squared error. Geometry's successful network-41 comparator is useful:
it predicts a substantial advance accurately, but its distances to 46 and 55
are 40.66 and 54.08. Depth's successful network-45 comparator to 55 has target
-8.754 and error +1.634 hours, at distance 30.98. Depth has **no successful
factor-two-magnitude comparator for 46**; the fallback network 37 has only 34%
of its target magnitude. Network 41 itself changes from success to failure.
There are only four segmented successes under either model, with different
identities. These comparisons show limited support, not a controlled causal
contrast between otherwise equivalent networks.

**Relative failures are a different problem.** Geometry's leading regular
relative contributor is 54/S036_unit_52: target +0.000652 hours, error +0.219789
hours (13.19 minutes). Its successful comparator 22/S169_unit_151 has target
+0.000429 and error -0.013350 hours, distance 0.65. Depth's leading contributors
are 02/S013_unit_20 (target +0.044851, error -0.153962 hours) and
02/S012_unit_19 (target -0.022436, error -0.065451). The latter has a substantially
closer successful comparator, 21/S003_unit_4: target -0.032936, error +0.001837,
distance 0.41. Small target magnitudes magnify these misses; the minute floor
bounds that amplification but does not eliminate errors of 4–13 minutes.
The top depth relative contributor has the wrong predicted response sign.
These cases are not interchangeable with the multi-hour failures above.

The relative ranking also surfaces segmented 05 (errors +4.390/+4.032 hours),
15 (depth error -3.469, 72.76% relative error) and 48 (geometry error -0.658,
63.09%). Network 05 underpredicts while 15 overpredicts, again ruling out a
universal correction of one sign. Their matched successes and support checks
are included in the same saved tables.

#### Training support and information actually available

**Demonstrated support limitations.** Holding out network 46 leaves a most
negative training target of -17.279 hours. A forest averaging training-derived
leaf means/medians cannot reach its -23.094-hour target: at least 5.815 hours of
absolute error is unavoidable for this estimator class in that fold. This is
less than the actual 10.627–11.500-hour miss, so the range bound is only part of
the explanation. Its edited size and length change also exceed training ranges;
its depth-panel inverse-depth change is outside range as well.

Network 55 has **no individual feature or target outside training range**, yet
only four other segmented networks have same-sign targets within a factor of
two, and its nearest cross-network predictor distances are 13.62/24.98. Thus
marginal in-range status does not establish joint support. Regular 15/CB001 and
22/CB002 are likewise inside all marginal feature ranges. Their comparable
regular target bands contain only eight/seven cases from four other networks.
For 22/CB001, geometry is in range but depth resistance is outside range;
37/CB001 exceeds geometry's mean-width-ratio range. These are heterogeneous
support failures, not one common extrapolation flag.

Each outer training set has just 14 segmented rows. With the recorded
network-equal, within-network-case-equal training weights, those rows receive
only **1.28–1.58% of total training weight**, although the inner selection
objective gives the segmented cohort 50% of its weight. This arithmetic is
verified from the training populations. It is a plausible contributor to weak
segmented prediction, not proof that reweighting would improve it. Similarly,
nearest training examples for several large regular collapses are segmented
cases; distance alone does not establish that those cases occupy the fitted
forest's decisive leaves. No held-fold forests were refitted to make that claim.

**Demonstrated information gaps, with causal limits.** The inspected routing
parameters retain `h = a_network W^0.65` to numerical precision, with the same
anchor on parent and child. These are derived depths, not independent reach
measurements. The resistance preparation uses a ratio of equivalent
`L/(W² h³)` resistances; a common depth anchor cancels. The inverse-depth feature
retains depth scale, but uses `h^-1`, whereas saved raw celerity independently
verifies `(5/3) h^(2/3) sqrt(s)/n`. Neither model directly includes roughness,
forcing shape, baseline/end-point conditions, or a complete routing description.
Regular inverse-depth summaries use B domains, segmented summaries use edited
footprints, and segmented resistance includes connectors; this scope distinction
is established by the preparation code, not newly resolved by this audit.

Across the 22 inspected cases, roughness varies between networks and the
parent/child depth anchors agree. **No inspected parameter table has an active
celerity clip.** This weighs against active clipping as the immediate explanation
of these particular residuals; it does not validate the remaining hydraulics.
The absence of roughness/event information and the depth-law assumptions are
plausible mechanisms only: there is no matched intervention here isolating them.

An independently calculated full-network transport proxy,
`sum(reference_flow_fraction * length / saved_celerity)`, reinforces why it
cannot simply substitute for the finite-window target. Its child-parent change
is -35.766 hours for segmented 46 versus centroid -23.094; for 55 the pair is
-19.563 versus -17.279; for 41, -16.919 versus -13.244. Differences occur in
successful cases too. For regular 02/S013, the proxy predicts an advance
(-0.324 hours) while the saved finite-window centroid is a delay (+0.044851).
This demonstrates an operator mismatch, not its cause or a better predictor.
Finite-window volume changes, boundary conditions, propagation and numerical
representation remain competing contributors. No child hydrograph diagnostic
was promoted into a model input.

**Window sensitivity is insufficient to erase the main misses.** Across the
saved 0.5/1/2% windows, regular 15/CB001 changes by 0.051 hours and network-22's
two large regular cases by 0.011/0.014 hours. Segmented 55 changes by 0.122 hours
and 41 by 0.223. Segmented 46 is more sensitive (2.771 hours), but even its
least-negative target is -20.322 versus depth prediction -12.467: a 7.856-hour
miss remains with the frozen prediction. This is target sensitivity, not new
validation at a different target, and does not test arbitrary baseline changes
or an infinite-window definition.

#### Supported next questions; models remain frozen

The repeated large segmented misses in independent networks 46 and 55, with
additional depth-panel failure in 41, justify **one limited future development
hypothesis**: the combination of sparse large-collapse support, low segmented
training weight and bounded forest predictions limits transfer to large full
collapses. Before changing models, assess distance robustness to network-balanced
scaling and inspect saved fold models if available; a new fit would need to be
labelled explicitly. New independent networks spanning large segmented changes
would provide stronger evidence than more related states from the same network.

If a later development experiment is undertaken, predeclare a cohort-weight
sensitivity against both frozen controls with the same whole-network holdouts,
report all four cohort scores and individual 46/55/41/22 outcomes, and keep
hyperparameter selection inside training folds. Success must improve multiple
held networks without hiding regular degradation. Reweighting cannot overcome
46's training-target bound, so it must not be sold as a complete remedy. These
are proposals only; no such experiment was run here.

There is not yet multi-case controlled evidence to recommend adding roughness,
changing depth exponents, replacing resistance scopes, switching target, or
replacing the model family. The regular errors, near-zero sign errors and
finite-window/proxy discrepancy need separate tests. Which omitted information
explains them, whether the current hydraulic inputs are physically accurate,
and whether any improvement transfers beyond these 15 previously explored
networks remain unresolved.

Reproduce the read-only audit with a pandas/pyarrow environment:
`python RAPID/investigations/cap_investigation_20261020/frozen_failure_audit.py`.
Evidence: [script](investigations/cap_investigation_20261020/frozen_failure_audit.py),
[all errors](outputs/uncapped_event_window_20261020/frozen_failure_audit_20261026/all_errors.parquet),
[rankings](outputs/uncapped_event_window_20261020/frozen_failure_audit_20261026/rankings.csv),
[case support](outputs/uncapped_event_window_20261020/frozen_failure_audit_20261026/case_support.csv),
[matched successes](outputs/uncapped_event_window_20261020/frozen_failure_audit_20261026/matched_successes.csv),
[nearest distinct networks](outputs/uncapped_event_window_20261020/frozen_failure_audit_20261026/nearest_networks.csv),
[feature ranges](outputs/uncapped_event_window_20261020/frozen_failure_audit_20261026/feature_ranges.csv),
[fold/weight audit](outputs/uncapped_event_window_20261020/frozen_failure_audit_20261026/fold_audit.csv),
[historical hashes](outputs/uncapped_event_window_20261020/frozen_failure_audit_20261026/historical_source_checks.csv),
[raw hydrograph checks](outputs/uncapped_event_window_20261020/frozen_failure_audit_20261026/raw_wave_checks.csv),
[window sensitivity](outputs/uncapped_event_window_20261020/frozen_failure_audit_20261026/window_sensitivity.csv),
[physical context](outputs/uncapped_event_window_20261020/frozen_failure_audit_20261026/physical_context.csv),
and [scores](outputs/uncapped_event_window_20261020/frozen_failure_audit_20261026/scores.csv).

### Physical follow-up to the frozen failure audit (2026-10-26)

This follow-up addresses the physical mechanisms behind the audited failures,
rather than repeating their support statistics. Both models and their saved
held-network predictions remain frozen. It uses 23 cases: the previous 22
failures/comparators plus segmented network 13, the nearest training-network
case to segmented 55 in both predictor spaces. No model fits or routing reruns
were made. Results concern the **saved constant-reference routing system**;
its calibrated roughness, derived depths and geometric assumptions are not
independent observations of real collapse responses.

#### What was calculated independently

With no active celerity clipping in these cases, the routing law gives

\[
T=\sum_e r_e L_e/c_e
 =0.6n\sum_e r_e L_e h_e^{-2/3}s_e^{-1/2}.
\]

Here `r` is the actual routed flow fraction. A separate traversal of the network
reproduces this mean from each Cunge cell's mean delay `K`, its cell count and
branch probabilities. Thus T is the **mean of the saved linear routing transfer
operator**, not merely an empirical travel-time proxy. It is not automatically
the centroid of the fixed-window excess-discharge response.

For the descriptor domain, write this mean as
`0.6 n Lbar Hbar Sbar C`, with visit-length weights `r L`,
`Hbar=mean(h^(-2/3))`, `Sbar=mean(s^(-1/2))`, and
`C=mean(h^(-2/3)s^(-1/2))/(Hbar Sbar)`.
Averaging all 24 orders of replacing parent factors by child factors provides
an exact, symmetric decomposition into length, depth, slope and their spatial
covariance. These are **algebraic contributions**, not separately feasible
channel interventions: collapsing a channel couples its width, depth, slope,
length and flow distribution. The depth contribution therefore also reflects
the collapse rule that increases derived depth as width increases.

A second calculation translates the saved parent hydrograph by `Delta T`,
without changing its shape, then measures its centroid using the original
common baseline and window. This gives the exact accounting identity

`actual centroid change = Delta T + fixed-window translation correction + residual shape effect`.

No translated waveform required extrapolation beyond saved time coverage. The
residual shape term includes all differences between the actual child and a
purely translated parent; it is not automatically attributable to diffusion
alone. The translation experiment isolates the response operator from those
shape differences without rerouting or fitting anything.

| Case | Mean transport change | Window correction | Residual shape effect | Actual centroid change | Depth-model prediction |
|---|---:|---:|---:|---:|---:|
| 15 / CB001_bubble_4 | -3.120 | +0.547 | +0.307 | -2.266 | -5.478 |
| 22 / CB001_bubble_9 | -3.056 | +0.774 | -0.003 | -2.286 | -5.074 |
| 22 / CB002_bubble_11 | -3.783 | +0.956 | -0.009 | -2.836 | -4.601 |
| 37 / CB001_bubble_3, successful depth comparator | -10.365 | +2.881 | +0.022 | -7.461 | -6.944 |
| 46 / CB001_bubble_90, successful depth comparator | -2.972 | +1.147 | -0.044 | -1.869 | -2.148 |
| 55 / fully segmented | -19.563 | +2.386 | -0.102 | -17.279 | -9.346 |
| 41 / fully segmented | -16.919 | +3.809 | -0.135 | -13.244 | -8.885 |
| 45 / fully segmented, successful depth comparator | -9.661 | +0.922 | -0.015 | -8.754 | -7.120 |
| 13 / fully segmented, nearest predictor-space network to 55 | -10.051 | +2.917 | +0.154 | -6.979 | -9.024 |
| 46 / fully segmented | -35.766 | +13.519 | -0.846 | -23.094 | -12.467 |

All values are hours. The [waveform figure](outputs/uncapped_event_window_20261020/frozen_physical_followup_20261026/translation_controls.png)
compares the parent, actual child and translated parent for six principal
failure/control cases. The close actual-child/translated-parent agreement for
several cases explains why complex waveform deformation is not needed to
account for their centroid changes.

#### Regular network 15: a large inverse-depth change does not imply a large clock change

For 15/CB001, the descriptor-domain effective depth rises from 1.283 to 3.274 m,
mean travelled length falls from 42.56 to 35.34 km, and effective slope rises
from 0.000317 to 0.000465. The resulting transport reduction decomposes into
**-1.892 h depth, -0.584 h length, -0.604 h slope and -0.040 h covariance**.
It is a real acceleration, but only 3.120 hours before response-window effects.

Why does the depth model forecast an advance of 5.478 hours? Its inverse-depth
change is -1.475 million, but the physically correct conversion is only
**2.116 transport hours per million units**. In successful regular 37/CB001,
the conversion is **4.234 hours per million**, twice as large. Network 15's
calibrated n is 0.01431 versus 0.02943 in network 37; because n multiplies T,
this accounts for almost all of the factor-two conversion difference, with the
remaining difference arising from the depth exponent/spatial weighting.
Their centroid/transport ratios are almost equal, 0.726 and 0.720. Thus the
contrast is primarily a hydraulic-clock difference, not a mysterious difference
in how their event centroids are measured.

The inverse-depth feature uses `h^-1`, whereas transport uses `h^(-2/3)` and n.
More precisely, on each side,
`T = 0.6 n G_h <h^(1/3)>_(rL/(h sqrt(s)))`.
Taking child minus parent does not reduce this to a universal constant times
`Delta G_h`. The model receives neither n nor these separate side-specific
correction factors. Other predictors can correlate with them, but cannot make
the physical conversion universal.

This is also visible in **already saved** held-network SHAP values: the
inverse-depth feature contributes -4.679 hours of the -5.478-hour prediction.
For the segmented collapse in the same network it contributes -6.453 hours to
a -8.238-hour prediction, against actual -4.768. The same physical-scale
problem therefore appears in both collapse types in network 15; these two rows
are not independent networks. SHAP explains the fitted prediction allocation,
not an intervention on a physical variable. It does not prove that adding n
would fix either error.

#### Regular network 22: size is being used as a misleading time scale

For 22/CB001, effective depth increases from 3.823 to 11.529 m. Its transport
change is **-2.390 h depth, -0.389 h length, -0.301 h slope, +0.023 h covariance**.
For 22/CB002, the corresponding contributions are **-3.073, -0.433, -0.273 and
-0.005 hours**. Deep, fast child channels account for most of the advance.

Compare 22/CB001 with successful 46/CB001_bubble_90. Their inverse-depth changes
are similar (-0.914 versus -0.861 million), as are their actual transport changes
(-3.056 versus -2.972 hours). But their edited sizes are **91.68 versus 21.22 km**.
The long network-22 domain is relatively deep and steep; the short network-46
domain is shallower and flatter. Their parent domain transit times consequently
are only **5.043 versus 4.640 hours**, despite the fourfold difference in length.
Length is not a comparable hydraulic clock across these cases.

The saved SHAP decomposition makes this mismatch concrete: inverse-depth
contributions are nearly identical (-1.884 versus -1.831 hours), while the size
contribution is **-2.117 hours for the failure versus +0.052 for the success**.
Length-change contributions add -0.904 versus -0.206. The model predicts a much
larger advance for network 22 although the physically calculated transport
changes are almost equal. For 22/CB002, size likewise contributes -2.181 hours.
These attributions support a learned size/time association that does not
transfer to the deep, fast network-22 domains; they are not proof of a causal
size coefficient because correlated feature effects are allocated by SHAP.

The remaining distinction between transport and response is unusually clean.
Translating the parent by -3.056 hours gives a fixed-window centroid change of
-2.282 hours, within **0.003 hours** of the actual -2.286. CB002 closes within
0.009 hours. The parent begins the analysis window about **328 m³/s above its
fixed baseline** and ends about 25 m³/s above it. These nonzero boundary flows
mean that translating the hydrograph does not translate its finite-window
centroid by the same amount. This accounts for the roughly 25% attenuation;
it is not evidence that the target was calculated incorrectly.

For a parent excess hydrograph e over [a,b], volume V and centroid mu, the local
sensitivity of the measured centroid to a translation is exactly
`1 - (mu-a)e(a)/V - (b-mu)e(b)/V`.
For network 22 it is 0.7446: the start-boundary term removes 0.2136 and the end
term 0.0417. This independently explains the approximately 0.75 retention seen
in both large regular cases. None of these event/boundary quantities is an
input to the forests.

#### Segmented network 55: the model compresses a physically much larger response

Network 55's effective edited-domain depth increases from **1.395 to 2.362 m**;
length falls from 198.51 to 187.82 km and effective slope rises from 0.000176 to
0.000187. Its parent edited-domain transit time is **55.011 hours**, falling to
35.448 hours. The 19.563-hour acceleration decomposes into **15.590 h depth,
2.485 h length, 1.347 h slope and 0.141 h covariance**. Only 2.284 hours are lost
between this transport advance and the actual 17.279-hour centroid advance.
There is no large opposing waveform effect concealing a smaller physical change.

The most revealing contrast is with its nearest predictor-space network, 13.
The model predicts nearly the same advance for them: **9.346 versus 9.024 hours**.
But the actual advances are **17.279 versus 6.979 hours**. This factor 2.476 in
actual response separates into three measured factors:

- inverse-depth-change magnitude: **1.544 times larger** in 55;
- conversion from that descriptor to transport hours: **1.261 times larger**;
- fraction of transport advance retained by the finite-window centroid:
  **1.272 times larger** (0.883 versus 0.694).

Their product reconstructs the observed response ratio. Network 55's n is
0.02748 versus 0.01967 in 13, contributing to the clock difference. Network 13
also ends well above baseline (about 63 m³/s at the analysis boundary), whereas
55 ends near baseline (about 1.67 m³/s), contributing to stronger attenuation
in 13. The exact finite-window translations reproduce that contrast; it is not
inferred from residual correlation.

Saved SHAP shows that the model increases the inverse-depth contribution from
-6.170 hours in 13 to -7.436 in 55, but reduces the size contribution from -2.027
to -0.919. Its total prediction scarcely changes. The forest therefore misses
the combined scale increase, even though `Delta G_h` already signals part of it.
This is both a limited-support/model-response issue and a physically demonstrable
failure of a single descriptor-to-hours conversion. The successful segmented
45 comparator has a much smaller physical change (-9.661 h), correctly yielding
a smaller centroid advance (-8.754 h). A generic claim that all segmented
responses are intrinsically unpredictable is inconsistent with that success.

#### Segmented network 41: similar inverse-depth change, different physical clock

Networks 41 and 22 have very similar inverse-depth changes: -3.662 versus
-3.530 million, and similar edited sizes: 264.65 versus 290.21 km. Yet 41 has a
**16.919-hour** transport advance versus 22's **12.066**, and an actual centroid
advance of **13.244 versus 8.867 hours**. Their descriptor-to-transport conversion
is 4.620 versus 3.418 hours per million. Network 41's n is about 29% greater;
the depth correction also differs. Its physical decomposition is **-11.540 h
depth, -3.361 h length, -1.946 h slope and -0.072 h covariance**.

The depth model reverses the ordering, predicting 8.885 hours for 41 and 13.035
for 22. Saved SHAP attributes -5.363/-1.838 hours to inverse-depth/size in 41,
versus -6.612/-4.518 in 22. This is a second independent network contrast showing
that the learned relationship among inverse-depth change, size and travel time
does not preserve the actual physical clock. Different held-fold settings also
contribute to the learned mapping; the analysis does not isolate feature design
from estimator/tuning effects. Geometry's accurate 41 prediction (-13.666)
shows that a good result is possible, but does not establish that its mechanism
is more faithful.

#### Connector resistance and diffusion: measured, but not universal explanations

A specific scope effect is now quantified. For segmented cases, retained serial
connectors are unchanged and carry all the flow. Subtracting their resistance
from both parent and child equivalent resistance gives an edited-only contrast.
Network 41's full-network resistance log-ratio is **0.106**, but edited-only it
is **0.632**: unchanged connectors supply 71.8% of parent and 91.6% of child
resistance. Network 46 changes from **0.194 to 0.492**, and 48 from **0.085 to
0.716**. Thus connector dilution is real and occurs in multiple networks.

However, saved SHAP gives resistance very little direct contribution in these
segmented predictions; inverse depth, size and length change dominate. It would
be incorrect to promote connector dilution to the established primary cause
of their model errors. Further, in **all 23 inspected cases**, the change in
full-network mean transport time equals the change inside the descriptor domain
to numerical precision. Unchanged outside links are not secretly generating
the missing mean-time changes. Resistance-scope dilution and missing outside
travel-time change are different hypotheses; the latter is rejected here.

Diffusion is also distinguishable from path-length spread. The network traversal
separates transfer variance into variance of path means and accumulated Cunge
cell variance `count K²(1-2X)`. For large segmented examples the latter dominates:
55 has parent diffusion/path variances **194.64/3.09 h²**; 41 has **626.99/7.69 h²**.
Collapse removes path mixing, but reduces much more diffusion variance. Even so,
their actual-minus-translated centroid residuals are only -0.102/-0.135 hours.
Changed spreading is visible without being the principal cause of their large
centroid-model residuals.

A counterexample is **02/S013_unit_20**, a near-zero sign failure. Its physical
transport change is -0.3244 h; the window correction is +0.1346 h; actual shape
change contributes **+0.2346 h**, reversing the final response to **+0.04485 h**.
Parent transfer SD falls from 26.91 to 23.80 h. Diffusion variance falls by
157.44 h², versus only 0.45 h² of path-mean variance. Its parent analysis window
ends **888.64 m³/s above baseline**; the end-boundary term alone removes 41.17%
of the local translation sensitivity. This is a concrete case where changed
spreading and a large retained endpoint can overwhelm a small travel-time
advance. The model predicts -0.1091 h, retaining the wrong sign. Diffusion is a
strong physical candidate for the shape change, but separating diffusion from
all other transfer-shape changes would require a controlled routing experiment;
none is claimed here. The successful small-response comparator 21/S003 has a
shape residual of only 0.000071 h. Near-zero responses need a different diagnosis
from the large translation-dominated failures.

#### What the new evidence supports

The clearest repeated limitations are now specific: **the physical clock varies
with calibrated n and depth weighting; geometric size can misrepresent that
clock; and the finite-window target retains a network/event-dependent fraction
of the transport shift**. Networks 15/37, 55/13 and 41/22 provide independent
contrasts for the conversion problem. Networks 22, 13, 41, 46 and 55 demonstrate
boundary attenuation directly through saved-wave translation. These are more
specific than generic missing-information claims. They still do not demonstrate
that a particular retrained model would improve held-network performance.

A justified next experiment would separate those two mappings: test a
physically appropriate, independently available hydraulic-time descriptor and
an event/boundary correction against the frozen controls, with whole-network
validation and training-only selection. Calibrated n must not be silently
presented as a deployable geometric predictor; first establish what roughness
information is available at prediction time. Likewise, child-outlet information
used here for diagnosis must never be used as a predictor. A separate controlled
routing experiment is warranted for the near-zero sign cases. No model change,
new predictor fit or routing experiment has been performed in this follow-up.

Evidence: [read-only analysis script](investigations/cap_investigation_20261020/frozen_physical_followup.py),
[case decompositions and boundary terms](outputs/uncapped_event_window_20261020/frozen_physical_followup_20261026/decomposition.csv),
[hydraulic scales on each side](outputs/uncapped_event_window_20261020/frozen_physical_followup_20261026/hydraulic_terms.csv),
[dominant links](outputs/uncapped_event_window_20261020/frozen_physical_followup_20261026/dominant_links.csv),
[saved held-network SHAP cases](outputs/uncapped_event_window_20261020/frozen_physical_followup_20261026/saved_shap_cases.csv),
[waveform comparisons](outputs/uncapped_event_window_20261020/frozen_physical_followup_20261026/translation_controls.png),
and [source hashes/method](outputs/uncapped_event_window_20261020/frozen_physical_followup_20261026/manifest.json).
All 4,370 reused LOEO SHAP predictions match the frozen saved predictions, and
SHAP sums reconstruct them; no alternative-validation explanations were mixed
in. All protected source/model hashes remain unchanged.

### Combined-state expansion with compound-group validation (2026-10-26)

The new isolated experiment reuses the previous 80 combinations and expands to
240 distinct combinations (160 newly constructed and routed). The 435 existing
family/reach groups stay intact. Each fully segmented case and each combination
is its own validation entity, giving 690 groups in the expanded population.
Example/network identity does not define the holdouts. Overlapping constituent
states may be in training, as explicitly requested for this validation scheme.

Five repeated five-fold partitions compare no combinations, +80 and +240 on the
same test cases. Both predictor panels retain their original training-only
nested-selected settings for each outer fold; all treatments are freshly fit,
without retuning. Equal-network training and reporting weights are retained.

Mean one-minute-floored relative error across repetitions:

| Panel / test population | No combinations | +80 | +240 |
|---|---:|---:|---:|
| Geometry, regular units/compounds | 46.46% | 46.49% | 46.47% |
| Geometry, fully segmented | 40.78% | 32.72% | 24.68% |
| Geometry, all 240 combinations | 41.08% | 34.91% | 30.85% |
| Depth informed, regular units/compounds | 28.18% | 28.23% | 28.47% |
| Depth informed, fully segmented | 35.32% | 25.50% | 19.39% |
| Depth informed, all 240 combinations | 28.53% | 26.21% | 25.37% |

Both expansions improve the equal-cohort original+segmented objective in all five
repetitions. The benefit concentrates in fully segmented and combined states;
regular-state relative error is essentially unchanged or slightly worse.
These are known-network related-state results, not evidence of transfer to new
rivers. The additional states do not increase the number of independent networks.

All 160 new runs settled. Harmonized windows changed 31 existing centroid targets
by at most 0.0882 seconds and six prior combination targets by at most 0.0800
seconds; every treatment uses those same revised targets. The 150 experimental
fits produced 72,750 held-out predictions. Eleven relevant tests pass, all 80
previous predictor vectors are preserved, and both depth descriptors reproduce
the 15 segmented reference cases. Previous experiments and production models
are unchanged.

See the [full report and reproduction commands](outputs/uncapped_event_window_20261020/combination_grouped_validation_20261026/REPORT.md),
[comparison plot](outputs/uncapped_event_window_20261020/combination_grouped_validation_20261026/comparison.png),
[expansion implementation](investigations/cap_investigation_20261020/combination_expansion.py),
[grouped validation](investigations/cap_investigation_20261020/combination_grouped_validation.py),
and [integrity audit](investigations/cap_investigation_20261020/audit_combination_expansion.py).
