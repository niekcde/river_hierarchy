# RAPID

This subproject prepares and runs RAPID routing for the network states produced
by `network_variants`.

## Scope

- read a `network_variants` experiment registry
- prepare per-state RAPID inputs
- compute link slope from matched SWORD WSE and variant geometry length
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

## Prepare an Experiment

```bash
/opt/anaconda3/envs/river-hierarchy-rivgraph/bin/python \
  RAPID/run_prepare_experiment.py \
  network_variants/output/sarl03_indep_v2 \
  --forcing-path /path/to/discharge.csv \
  --time-column time \
  --discharge-column discharge \
  --width-field wid_adj_wet
```

This writes per-state prep outputs under:

- `states/<state_id>/rapid/prep/`

and experiment-level summaries:

- `rapid_prep_registry.csv`
- `rapid_prep_manifest.json`

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

## Run RAPID

```bash
/opt/anaconda3/envs/river-hierarchy-rivgraph/bin/python \
  RAPID/run_rapid_experiment.py \
  network_variants/output/sarl03_indep_v2
```

This writes per-state run outputs under:

- `states/<state_id>/rapid/run/`

and experiment-level summaries:

- `rapid_run_registry.csv`
- `rapid_run_manifest.json`

## Notes

- The current engine uses the shared Python RAPID implementation that was
  previously kept inside `synthetic_runs`.
- In this environment the code uses `scipy.io.netcdf_file` rather than
  `netCDF4`, so the prep and run outputs are written as NetCDF3-compatible
  files.
