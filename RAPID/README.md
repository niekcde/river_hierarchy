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
