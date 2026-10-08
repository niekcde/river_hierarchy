# Full-population lag predictor search — 5 October 2026

Reference **FULL-POPULATION-LAG-2026-10-05**. Supersedes the path-limited timing benchmarks for population coverage. Scope is percentage lag and absolute timing only. No RAPID rerouting or response-definition change.

## Complete coverage and unchanged metrics

All **1,652 plotted comparisons across 16 network examples** now have the required descriptors. All 524 formerly excluded cases are recovered. The existing edge-compound filter remains: full population means the complete agreed plotted population, not every raw selection/alias in the repository.

The path algorithm retains the original definitions. It counts suffix paths by dynamic programming, calculates uniform-path length mean/variance, and evaluates E[L/R] and E[L²/R], where R=Σ(l/w), using 1/R=∫exp(−tR)dt. Thus total equivalent width=P E[L/R] and equivalent length=E[L²/R]/E[L/R]. Maximum harmonic path width is obtained by fractional optimization on the DAG. Checked quadrature replaces enumeration, not the metric. There is no path sampling, truncation by count or missing-value filling. Single-path dominance/association/CV invariants are calculated analytically, and association roundoff below 1e-12 is set to zero to avoid artificial near-zero splits in tree models.

All 1,128 previously enumerated records agree at rtol=1e-8/atol=1e-10. Maximum observed quadrature refinement difference is about 1e-13. Synthetic tests check random DAGs against direct enumeration and a billion-path network. Actual maximum counts are approximately 2.28e11 parent paths and 1.04e13 child paths.

**Interpretation remains important:** shared-edge path sums still count shared edges on multiple paths. Path-derived equivalent width, elongation and dominance can become extreme in large compound domains. Eliminating the computational cutoff does not turn these into independent-channel measurements. Log-transformed candidates address scale statistically, not this physical interpretation.

## Expanded real-case validation

61 candidates combine 11 feature sets with Ridge regularization, quadratic Ridge, two boosted-tree complexities and Extra Trees. Candidate sets include omega, separate components, size/slope, selected six, changes/context, all descriptors, logarithmic transforms and the existing single-product axes. `search_plan.json` is the complete specification.

Outer validation holds out one **network example** at a time. Four grouped inner folds in the remaining examples select by equal-example MAE. No outer response enters selection. Example grouping does not certify basin independence: examples may be geographically related. These rivers have been examined previously, so this remains exploratory validation, not an untouched external test.

| response | method | n | r2 | mae | skill_vs_zero |
| --- | --- | --- | --- | --- | --- |
| interpolated_lag_shift_pct | components/boosted_7 | 1652 | 0.4284 | 0.2117 | 0.2926 |
| interpolated_lag_shift_pct | nested_selected | 1652 | 0.2208 | 0.2317 | 0.2259 |
| interpolated_lag_shift_pct | omega/boosted_7 | 1652 | 0.3343 | 0.2124 | 0.2903 |
| interpolated_lag_shift_pct | selected_six/boosted_7 | 1652 | 0.3169 | 0.2633 | 0.1202 |
| interpolated_peak_shift_hours | components/boosted_7 | 1652 | 0.3067 | 0.2028 | 0.126 |
| interpolated_peak_shift_hours | nested_selected | 1652 | 0.4723 | 0.1347 | 0.4192 |
| interpolated_peak_shift_hours | omega/boosted_7 | 1652 | 0.1631 | 0.2231 | 0.0384 |
| interpolated_peak_shift_hours | selected_six/boosted_7 | 1652 | 0.5541 | 0.147 | 0.3665 |

The nested search yields R² **0.221 for percentage lag** and **0.472 for absolute timing**. Its MAEs are 0.232 percentage points and 0.135 h (8.1 minutes). Those improve on no change by 22.6% and 41.9%. For percentage lag, the simple separate-component benchmark remains stronger (R² .428, MAE .212). For absolute timing, the fixed selected-six boosted model has higher R² (.554) but slightly worse MAE (.147 h). Inner equal-example MAE selection need not optimize pooled outer R²; report both, rather than choosing whichever score looks best.

The largest held-out errors are complete-compound collapses in examples 15 and 22, recovered by removing the old path cutoff. For example, a −22.31% response in example 15 is predicted as −3.31%; a −8.57 h response in example 22 is predicted as −1.95 h. These are retained, not removed to improve scores. `largest_errors.csv` identifies the cases.

The earlier .68 percentage-lag score belonged to the restricted cohort. The full-population search does not justify carrying that headline forward. Fixed-candidate rankings in `scores.csv` are descriptive candidate comparisons; selecting a winner from them after inspecting outer performance incurs selection optimism. The nested result is the assessment of this run's selection process, not necessarily the strongest single fixed model.

| response | method | equal_example_mae | median_example_r2 | examples_beating_zero |
| --- | --- | --- | --- | --- |
| interpolated_lag_shift_pct | components/boosted_7 | 0.2818 | 0.341 | 16 |
| interpolated_lag_shift_pct | nested_selected | 0.2854 | 0.2753 | 15 |
| interpolated_lag_shift_pct | omega/boosted_7 | 0.2694 | 0.4116 | 14 |
| interpolated_lag_shift_pct | selected_six/boosted_7 | 0.3393 | 0.2701 | 14 |
| interpolated_peak_shift_hours | components/boosted_7 | 0.2227 | 0.0118 | 8 |
| interpolated_peak_shift_hours | nested_selected | 0.1029 | 0.3776 | 13 |
| interpolated_peak_shift_hours | omega/boosted_7 | 0.224 | -0.1213 | 8 |
| interpolated_peak_shift_hours | selected_six/boosted_7 | 0.1242 | 0.4457 | 13 |

## Synthetic selection and transfer

The same 61 candidates are selected using four geometry-grouped folds **only within the original synthetic development set** (9,851 unique comparisons). Models are then frozen and evaluated on the original 3,362-row holdout, the existing 917 novel extension comparisons (63 original anchors excluded), and all 1,652 real comparisons. Extension outlet peaks were reread and interpolated with the same guards; per-pair forcing clocks agree. Extension and real outcomes never select the synthetic model. These are previously explored transfer populations, so do not label them untouched confirmatory tests.

| response | dataset | n | r2 | mae | skill_vs_zero |
| --- | --- | --- | --- | --- | --- |
| interpolated_lag_shift_pct | synthetic_holdout | 3362 | 0.9966 | 0.2056 | 0.9858 |
| interpolated_lag_shift_pct | novel_embedding_extension | 917 | -0.9292 | 10.9372 | -1.5162 |
| interpolated_lag_shift_pct | full_real_transfer | 1652 | -145.9832 | 7.6603 | -24.5939 |
| interpolated_peak_shift_hours | synthetic_holdout | 3362 | 0.9824 | 0.0612 | 0.943 |
| interpolated_peak_shift_hours | novel_embedding_extension | 917 | 0.583 | 0.5625 | 0.5576 |
| interpolated_peak_shift_hours | full_real_transfer | 1652 | 0.1325 | 0.3523 | -0.5184 |

The synthetic-selected percentage model is Extra Trees with selected-six/change/context features; its original holdout R² is .9966, but extension R² is −.929 and real transfer R² −145.98. Strong interpolation within the original design still does not establish transfer.

The synthetic-selected absolute model uses boosted trees on logged geometry. Its R² is .9824 on the original holdout and .583 on the novel extension. Real transfer R² is only .133, and MAE .352 h is worse than predicting zero change (.232 h). Positive R² alone therefore does not establish useful real transfer.

## Current conclusion and next decision

Keep omega/separate components as readable percentage-lag benchmarks. Absolute timing benefits from dimensional geometry and is more promising on the wider synthetic extension, but synthetic-trained absolute predictions still do not beat no change in real cases. The larger full-population search is now complete; it did not reveal a universal predictor or a consistent benefit of more model complexity.

Next inspect the full-population residuals and category/example dependence before choosing a compact final predictor. The physical applicability issue in the separate hydraulic-domain notebook is unchanged by statistical model performance. Shape model search remains paused by user decision.

Reproduce `full_population_features.py`, `full_population_lag_search.py`, `full_population_synthetic_transfer.py`, then this report script. Outputs, fold choices and candidate scores are retained here. Model fits use one CPU thread; no hydraulic settings, simulation inputs or legacy results were overwritten.
