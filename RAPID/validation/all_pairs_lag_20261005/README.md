# All parent–child pairs — corrected analysis, 5 October 2026

Reference **ALL-PAIRS-LAG-2026-10-05**. The earlier 1,652-row analysis was incorrectly described as the full population: it retained an inherited first/last compound-group filter. The user explicitly requested every distinct pair. That filter is now removed from active plotting and prediction. This report supersedes the filtered-population timing results.

**2,214 canonical parent–child pairs, 16 network examples, zero edge-compound exclusions, zero path-count exclusions, zero missing-feature exclusions.** Duplicate representations remain collapsed to one comparison; aliases are retained in the original identity audit. All lag responses are valid. Exact comparison identities are checked against the complete canonical inventory, not just a row count.

## Coverage

| example_id | previous_filtered | all_pairs | restored |
| --- | --- | --- | --- |
| 02 | 29 | 35 | 6 |
| 03 | 34 | 40 | 6 |
| 05 | 45 | 51 | 6 |
| 11 | 3 | 26 | 23 |
| 13 | 47 | 66 | 19 |
| 15 | 252 | 262 | 10 |
| 21 | 24 | 30 | 6 |
| 22 | 396 | 408 | 12 |
| 37 | 1 | 396 | 395 |
| 41 | 181 | 198 | 17 |
| 45 | 84 | 93 | 9 |
| 46 | 208 | 223 | 15 |
| 48 | 58 | 72 | 14 |
| 54 | 75 | 81 | 6 |
| 55 | 200 | 212 | 12 |
| 56 | 15 | 21 | 6 |

Example 37 was especially affected: 395 pairs were removed with a first/last compound group, leaving only one. The old filter removed member units as well as whole-compound collapses; it was not a scientific reason to discard their model results.

Scalable path descriptors retain the same equations, including shared-edge path counting. The 1,128 old enumerated records still agree within rtol=1e-8/atol=1e-10. Single-path invariants and quadrature convergence are checked. No geometry imputation or path sampling is used. The largest path counts now exceed 1e25, which reinforces the need to interpret path-based width/dominance cautiously rather than as counts of independent channels.

## Real timing search

The same 61 candidate model/feature combinations are compared on every pair. Outer validation holds out an entire network example; four grouped inner folds select by equal-example MAE. Scalers are fitted within training folds. The logarithmic surrounding-length candidates use log10(1+L/metre), which can retain zero-length context if a domain touches an external boundary. Other positive geometric variables use log10. This does not change any raw descriptor or drop boundary cases.

| response | method | n | r2 | mae | skill_vs_zero |
| --- | --- | --- | --- | --- | --- |
| interpolated_lag_shift_pct | components/boosted_7 | 2214 | 0.0126 | 0.387 | -0.0692 |
| interpolated_lag_shift_pct | nested_selected | 2214 | 0.2119 | 0.269 | 0.2568 |
| interpolated_lag_shift_pct | omega/boosted_7 | 2214 | 0.1019 | 0.3235 | 0.1063 |
| interpolated_lag_shift_pct | selected_six/boosted_7 | 2214 | -0.1667 | 0.4888 | -0.3505 |
| interpolated_peak_shift_hours | components/boosted_7 | 2214 | 0.2585 | 0.2005 | 0.0511 |
| interpolated_peak_shift_hours | nested_selected | 2214 | 0.5721 | 0.1176 | 0.4433 |
| interpolated_peak_shift_hours | omega/boosted_7 | 2214 | 0.1995 | 0.2069 | 0.0205 |
| interpolated_peak_shift_hours | selected_six/boosted_7 | 2214 | 0.3979 | 0.1646 | 0.221 |

| response | method | equal_example_mae | median_example_r2 | examples_beating_zero |
| --- | --- | --- | --- | --- |
| interpolated_lag_shift_pct | components/boosted_7 | 0.3644 | -0.0879 | 11 |
| interpolated_lag_shift_pct | nested_selected | 0.2662 | 0.5968 | 14 |
| interpolated_lag_shift_pct | omega/boosted_7 | 0.3854 | -0.2519 | 12 |
| interpolated_lag_shift_pct | selected_six/boosted_7 | 0.4287 | -0.4353 | 9 |
| interpolated_peak_shift_hours | components/boosted_7 | 0.2527 | -0.1122 | 8 |
| interpolated_peak_shift_hours | nested_selected | 0.1032 | 0.5857 | 14 |
| interpolated_peak_shift_hours | omega/boosted_7 | 0.2524 | 0.0488 | 6 |
| interpolated_peak_shift_hours | selected_six/boosted_7 | 0.1343 | 0.3373 | 10 |

Pooled R² and MAE answer different questions; model selection optimizes inner equal-example MAE, not pooled outer R². Fixed-candidate tables are descriptive benchmarks; inspecting them to choose the highest outer R² would introduce selection optimism. These network examples were previously explored and may belong to related basins. This is grouped exploratory validation, not untouched basin-independent confirmation.

## Synthetic selection and transfer

Selection uses only original synthetic development geometries. The original holdout and the novel embedding extension are excluded from training/model choice; 63 original-grid extension anchors are excluded from extension scoring. Transfer now evaluates **all 2,214 real pairs**, not the old filtered subset. The same guarded peak-timing definitions are used. These previously examined test populations remain exploratory.

| response | dataset | n | r2 | mae | skill_vs_zero |
| --- | --- | --- | --- | --- | --- |
| interpolated_lag_shift_pct | synthetic_holdout | 3362 | 0.9966 | 0.2056 | 0.9858 |
| interpolated_lag_shift_pct | novel_embedding_extension | 917 | -0.9292 | 10.9372 | -1.5162 |
| interpolated_lag_shift_pct | full_real_transfer | 2214 | -25.6148 | 6.9724 | -18.2631 |
| interpolated_peak_shift_hours | synthetic_holdout | 3362 | 0.9824 | 0.0612 | 0.943 |
| interpolated_peak_shift_hours | novel_embedding_extension | 917 | 0.583 | 0.5625 | 0.5576 |
| interpolated_peak_shift_hours | full_real_transfer | 2214 | 0.1031 | 0.352 | -0.6664 |

Shape predictor search remains paused. Hydraulic-domain results are unchanged by this population correction; no routing was rerun. The earlier 1,652/1,128-row reports remain historical, explicitly filtered analyses. Current notebook plots and the active timing-search sections use all 2,214 pairs.

Reproduce: `full_population_features.py`, `full_population_lag_search.py`, `full_population_synthetic_transfer.py`, `report_all_pairs_lag.py`. `coverage.csv` lists every restored count. `features.parquet`, fold predictions, `scores.csv`, `selections.csv`, `search_plan.json`, geometry parity and provenance are retained here. The input-signature guard prevents resuming cached folds against different features/model definitions.
