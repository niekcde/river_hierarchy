# RAPID

This subproject prepares and runs RAPID routing for the network states produced
by `network_variants`.

## Scope

- read a `network_variants` experiment registry
- prepare per-state RAPID inputs
- compute link slope from matched SWORD WSE and variant geometry length
- interpolate fallback SWORD WSE values along the ordered example corridor
  using `sword_dist_out` when enough true requested-field anchors exist
- compute RAPID K/X values per link
- normalize inlet forcing from CSV or parquet
- run the shared Python RAPID routing engine for prepared states

## Inputs

The prep workflow expects one `network_variants` experiment directory with:

- `state_registry.csv`
- per-state directed links and directed nodes
- per-state width-enriched links at
  `states/<state_id>/variant/widths/links_with_width_families.gpkg`

The directed nodes should already carry SWORD fields from
`network_variants/sword_matching.py`, especially:

- `sword_wse`
- `sword_wse_field`
- `sword_wse_fill_method`

## Current assumptions

- link slope uses upstream and downstream SWORD WSE
- slope denominator is the variant link geometry length
- forcing is currently a single discharge time series applied to inlet links
- if multiple inlet links exist, discharge is distributed by the chosen RAPID
  width field

## K Parameterization and Optional Celerity Capping

The default RAPID prep in this repository computes Muskingum `K` from the
variant link length `L`, repaired slope `S`, and chosen width `W`:

\[
K \propto \frac{L}{\sqrt{S}\,W^{2/3}}
\]

This keeps slope and width in the routing parameterization, but it also means
that very small link slopes can produce unrealistically slow implied wave
celerities and therefore very large `K` values. On graph-derived multichannel
networks like the ones generated here, that effect is amplified because:

- slopes are inferred from matched SWORD WSE values, not surveyed channel
  profiles
- widths come from planform extraction, not bankfull hydraulic geometry
- collapsed variants can contain links that are geometrically valid but still
  hydraulically noisy at local scale

To make that regularization explicit, RAPID prep exports the implied hydraulic
celerity

\[
c_{\mathrm{raw}} = \frac{L}{K}
\]

and offers an optional celerity cap before converting back to `K`:

\[
K_{\mathrm{used}} = \frac{L}{c_{\mathrm{used}}}
\]

When `--use-celerity-capping` is enabled, the code clips the implied celerity
to a literature-informed interval and records both the raw and capped values in
`rapid_link_attributes.csv`.

### Why the cap is on celerity, not directly on K

This is intentionally phrased as a celerity constraint rather than a direct `K`
clip because Muskingum `K` is a travel time. The physically interpretable
quantity underneath `K` is wave travel speed.

So the implemented assumption is:

- keep the hydraulic dependence on slope and width
- but prevent graph-derived local artifacts from implying wave speeds outside a
  plausible routing range for basin-scale RAPID runs

This is a regularization choice for reduced vector river networks. It is not a
claim that every capped link has a directly observed local celerity.

### Literature basis for the default bounds

The default cap values are:

- minimum celerity: `0.28 m/s`
- maximum celerity: `1.524 m/s` (`5 ft/s`)

These defaults are meant as an initial literature-informed range:

- Collins et al. (2025) use RAPID with a reference celerity of `0.28 m/s` and
  global `x = 0.3` for multichannel routing experiments, then perturb
  Muskingum `k` through `λ_k` to alter wave speeds in side channels.
  Source: https://doi.org/10.1029/2025WR041417
  Mirror: https://www.researchgate.net/publication/397018908_River_Network_Routing_and_Discharge_Partitioning_on_a_Multichannel_River_Network

- The HEC-HMS Muskingum-Cunge guidance states that, for many applications, a
  celerity of `5 ft/s` is an adequate practical value when the celerity index
  method is used.
  Source: https://www.hec.usace.army.mil/confluence/hmsdocs/hmsguides/applying-reach-routing-methods-within-hec-hms/applying-the-muskingum-cunge-routing-method

- HEC-HMS also expresses Muskingum travel time as `K = L / c`, which is the
  basis for using celerity as the regularized quantity rather than clipping `K`
  directly.
  Source: https://www.hec.usace.army.mil/confluence/hmsdocs/hmstrm/channel-flow/muskingum-model

These values should be treated as starting defaults, not universal constants.
If basin-specific calibration or regional hydraulic knowledge exists, the cap
range should be updated accordingly.

## Optional RAPID-Only Subreach Splitting

Even with celerity capping, Muskingum stability can still fail when the network
contains a very wide spread in link lengths. In that case, the longest links
control `dt_min` and the shortest links control `dt_max`.

To address that discretization problem without changing the source
`network_variants` graph, RAPID prep can split long links into virtual
subreaches:

- this happens only inside `RAPID` preparation
- the source network on disk is left unchanged
- each subreach inherits the parent link's width, slope, and celerity inputs
- only the routing length is divided among the child reaches

The current rule is:

\[
n_{\mathrm{sub}} = \max\left(1,\mathrm{round}\left(\frac{L}{L_{\mathrm{target}}}\right)\right)
\]

so, for example:

- `1100 m` with `L_target = 500 m` becomes `2 x 550 m`
- `1450 m` with `L_target = 500 m` becomes `3 x 483.3 m`

This is intentionally a target-length rule, not a hard maximum-length rule.

### Literature basis for subreaches

HEC-HMS Muskingum guidance explicitly includes the number of subreaches as a
reach-routing control and states that, for an idealized channel, the travel
time through a subreach should be approximately equal to the simulation time
step. It also notes that, for natural channels, the number of subreaches can be
treated as a calibration parameter and used to introduce numerical attenuation.

Sources:

- Muskingum model technical reference:
  https://www.hec.usace.army.mil/confluence/hmsdocs/hmstrm/channel-flow/muskingum-model
- Applying the Muskingum routing method:
  https://www.hec.usace.army.mil/confluence/hmsdocs/hmsguides/applying-reach-routing-methods-within-hec-hms/applying-the-muskingum-routing-method

## Prepare an Experiment

```bash
/opt/anaconda3/envs/river-hierarchy-rivgraph/bin/python \
  RAPID/run_prepare_experiment.py \
  network_variants/output/sarl03_indep_v2 \
  --forcing-path /path/to/discharge.csv \
  --time-column time \
  --discharge-column discharge \
  --width-field wid_adj_wet \
  --use-celerity-capping \
  --min-celerity-mps 0.28 \
  --max-celerity-mps 1.524 \
  --target-subreach-length-m 500
```

This writes per-state prep outputs under:

- `states/<state_id>/rapid/prep/`

and experiment-level summaries:

- `rapid_prep_registry.csv`
- `rapid_prep_manifest.json`

The prep registry now also records state-level diagnostics that make the impact
of celerity capping and RAPID-only subreach splitting explicit:

- `n_source_links`
- `n_links`
- `link_multiplier`
- `n_split_parent_links`
- `pct_split_parent_links`
- `n_celerity_capped`
- `pct_celerity_capped`
- `min_link_length_m`
- `max_link_length_m`
- `rapid_k_min`
- `rapid_k_max`

Per-state prep outputs include:

- `rapid_link_attributes.csv`
- `rapid_node_attributes.csv`
- `conn.csv`
- `riv.csv`
- `rat.csv`
- `rat_srt.csv`
- `kfc.csv`
- `xfc.csv`
- `coords.csv`
- `forcing_normalized.csv` when forcing is supplied
- `inflow.nc` when forcing is supplied
- `rapid_prep_manifest.json`

`rapid_link_attributes.csv` now also records the celerity diagnostics used in
the `K` calculation, including:

- `rapid_celerity_mps_raw`
- `rapid_celerity_mps`
- `rapid_celerity_capped`
- `rapid_k_source_method`
- `rapid_celerity_cap_enabled`
- `rapid_celerity_cap_min_mps`
- `rapid_celerity_cap_max_mps`

When subreach splitting is enabled, `rapid_link_attributes.csv` also records:

- `parent_link_id`
- `parent_link_length_m`
- `subreach_index`
- `subreach_count`
- `subreach_length_fraction`
- `rapid_link_split`

and `rapid_node_attributes.csv` records the RAPID-only virtual split nodes with:

- `rapid_node_source`
- `rapid_node_split_from_link_id`
- `rapid_node_subreach_boundary_index`

## Run RAPID

```bash
/opt/anaconda3/envs/river-hierarchy-rivgraph/bin/python \
  RAPID/run_rapid_experiment.py \
  network_variants/output/sarl03_indep_v2 \
  --event-start-time 2023-02-12T06:00:00Z \
  --event-start-buffer-hours 6 \
  --event-end-time 2023-02-22T12:00:00Z \
  --event-end-buffer-hours 12
```

This writes per-state run outputs under:

- `states/<state_id>/rapid/run/`

and experiment-level summaries:

- `rapid_run_registry.csv`
- `rapid_run_manifest.json`

Per-state RAPID run outputs now also include:

- `Qout_rapid_framework.nc`
- `outlet_hydrograph.csv`
- `hydrograph_metrics.csv`
- `hydrograph_metrics.json`

`rapid_run_registry.csv` includes the hydrograph metrics directly so the
experiment-level table can be used for quick comparisons across states.

### Hydrograph Metric Definitions

The run step computes outlet-hydrograph metrics from the summed discharge across
all outlet reaches in the prepared RAPID state. The event start is defined from
the normalized inflow series used during prep:

- if `--event-start-time` is provided, the first inflow timestep at or after
  that UTC timestamp is used
- if `--event-start-time` and `--event-start-buffer-hours` are both provided,
  RAPID searches symmetrically around that timestamp and uses the minimum inflow
  discharge found within the window
- otherwise, RAPID chooses the minimum inflow discharge before the inflow peak
- if `--event-start-window-hours` is provided, that automatic minimum search is
  restricted to the first `N` hours of the inflow series

An optional end reference can also be supplied:

- if `--event-end-time` is provided, the first inflow timestep at or after that
  UTC timestamp is used as the event end reference
- if `--event-end-time` and `--event-end-buffer-hours` are both provided,
  RAPID searches symmetrically around that timestamp and uses the minimum inflow
  discharge found within the window

When an end reference is supplied, the outlet discharge at that detected event
end time becomes the recession baseline for:

- `fall_time_seconds`
- `fall_time_50_seconds`
- `fall_time_10_seconds`
- `e_folding_time_seconds`

If no end reference is supplied, those recession metrics continue to use the
event-start outlet discharge as the baseline.

The following metrics are exported:

- `event_start_time_*`
  baseline time used for the event definition
- `event_end_time_*`
  optional end-of-event reference time used for recession-baseline selection
- `event_duration_seconds`
  `event_end_time - event_start_time` when an end reference is supplied
- `peak_time_*`
  time of the outlet peak
- `peak_discharge_cms`
  outlet peak magnitude
- `peak_excess_cms`
  outlet peak above the outlet discharge at the event start
- `peak_excess_to_end_baseline_cms`
  outlet peak above the recession baseline defined by the event end when one is
  supplied
- `time_to_peak_seconds`
  `peak_time - event_start_time`
- `fall_time_seconds`
  time from the peak until the outlet hydrograph first returns to the outlet
  discharge at the event start
- `fall_time_50_seconds`
  time from the peak until the outlet discharge falls to 50% of the peak excess
  above baseline
- `fall_time_10_seconds`
  time from the peak until the outlet discharge falls to 10% of the peak excess
  above baseline
- `e_folding_time_seconds`
  time from the peak until the outlet discharge falls to
  `baseline + peak_excess / e`
- `lag_to_inflow_peak_seconds`
  lag between the inflow peak and outlet peak
- `peak_attenuation_ratio`
  `outlet_peak / inflow_peak`
- `outlet_volume_m3`
  total outlet routed volume over the simulated run
- `outlet_excess_volume_m3`
  integrated outlet volume above the event-start outlet discharge
- `rise_rate_cms_per_hour`
  `peak_excess / time_to_peak`

These definitions are intentionally consistent with the current single-event
workflow: one forcing hydrograph, one event start, and one outlet hydrograph per
state.

For interactive post-run analysis, use:

- `RAPID/notebooks/rapid_hydrograph_analysis.ipynb`

## Notes

- The current engine uses the shared Python RAPID implementation that was
  previously kept inside `synthetic_runs`.
- In this environment the code uses `scipy.io.netcdf_file` rather than
  `netCDF4`, so the prep and run outputs are written as NetCDF3-compatible
  files.
