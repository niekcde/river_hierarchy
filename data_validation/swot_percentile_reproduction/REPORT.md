# SWORD percentile reproduction check

Six reaches selected from previously validated examples: low, middle, and high inherited slope within Asia and South America, requiring populated SWORD percentiles. This is a diagnostic sample, not a globally representative validation.

Sources are local SWORD v17c Parquet files and local SWOT RiverSP D node/reach files. The local SWORD files were not verified byte-for-byte against the latest Zenodo release. All six reaches have swot_obs_source=node.

164 node granules were identified using the previous exhaustive archive scan, yielding 9,526 raw node rows and 177 reach-product counterpart rows. Node row counts were checked against that cache. Reach testing is limited to these counterpart granules, not an independent full scan of the reach archive.

Screening: wse_sm_u <= 1, dark_frac <= 0.5, 10000 <= abs(xtrk_dist) <= 60000, wse_sm > 0. Width additionally excludes nonfinite and nonpositive values. The reach analogue substitutes wse/wse_u. Percentiles use pandas linear interpolation.

No tested reach/method combination reproduced all six percentiles within 1e-6 m. Tested pooled nodes, per-cycle/pass node means and medians, deduplicated pooled nodes, valid-only nodes, reach-product observations with analogous screening, reach screening without cross-track restriction, and valid-only reach observations. Additional tests aggregate node temporal percentiles using spatial means/medians, from both local SWOT and SWORD node summaries. These also do not establish an exact reproduction.

SWORD node counts often substantially exceed the number of local filtered visits; local data availability, product selection, and screening therefore cannot be assumed to match SWORD. Count differences alone do not identify the cause. Local valid timestamps extend from July 2023 to November 2025.

## Median comparison using pooled nodes and existing screening

| Reach | SWORD WSE p50 (m) | Local WSE p50 (m) | SWORD width p50 (m) | Local width p50 (m) |
|---|---:|---:|---:|---:|
| 44230000251 | 22.574 | 25.266 | 2235.603 | 2340.904 |
| 44279300041 | 155.999 | 160.101 | 634.642 | 653.019 |
| 44279300141 | 178.549 | 181.817 | 394.965 | 563.063 |
| 61660400011 | 44.820 | 44.049 | 555.145 | 489.323 |
| 62252500211 | 6.321 | 11.332 | 762.469 | 767.574 |
| 62495100041 | 222.010 | 221.920 | 453.692 | 404.762 |

Full p10/p50/p90 comparisons: comparison.csv and node_summary_comparison.csv. Coverage diagnostics: coverage_comparison.csv. Node-level agreement: node_percentile_comparison.csv.

Reproduce by running check.py, then check_node_summaries.py with the river-hierarchy-rivgraph Python environment. Source data are read only.

Conclusion: computations are feasible, but these data and screening do not reproduce SWORD. An exact-match claim would require the SWORD processing code and matching input data/version/time window.
