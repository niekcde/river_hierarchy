**Population correction — ALL-PAIRS-LAG-2026-10-05:** the 1,652-pair analysis still had an inherited edge-compound filter and was incorrectly called the full population. The active analysis now covers all **2,214 distinct pairs** with no such exclusions. [Current report](../../RAPID/validation/all_pairs_lag_20261005/README.md). All preceding filtered-population conclusions below are historical.

**Superseding update — FULL-POPULATION-LAG-2026-10-05:** the feature-coverage restriction is resolved and both expanded timing searches are complete. [Current results and full report](../../RAPID/validation/full_population_lag_20261005/README.md). The planning checkpoint below is retained as dated history; its pending-search statements are no longer current.

# Lag-focused analysis status — 5 October 2026

Reference **LAG-FOCUS-2026-10-05**. User decision: focus subsequent predictor work on percentage lag and absolute timing. Retain peak excess/event duration as documented secondary findings; do not continue their model search now.

## Established

- Production real runs and matched resistance calculations are complete; current analysis uses deduplicated comparisons and refined outlet-peak timing.
- Full plotted population: 1,652 comparisons / 16 rivers. Scalar omega/components are available for all.
- Detailed path-feature population: 1,128 comparisons. The 10,000-path cutoff excludes 215 cases from example 15 and 309 from example 22. These are computational exclusions, not invalid model runs.
- Fixed boosted omega percentage-lag R²: .334 full population; .682 restricted population. Do not present .682 as full-population performance.
- Restricted nested model/feature selection: percentage-lag R² .622; absolute-timing R² .544. The broader 12-candidate search did not establish improvement on the earlier simpler timing models. Absolute timing selected the six-feature boosted model in all 16 outer folds.
- Original synthetic holdout prediction is strong; frozen synthetic-to-real transfer is weak. Geometry coverage differs substantially. The completed wider embedding extension has not yet been included in the matched timing-transfer benchmark.
- Real shape changes are mostly tiny; selected shape models do not beat a no-change baseline. Preserve this result without treating it as universal absence of shape effects.

## Unresolved before the next larger search

1. Remove the path-feature coverage restriction without silently redefining metrics, using an exact/scalably aggregated method where possible and checking parity on the currently enumerated domains. Do not simply raise the cap and launch exponential enumeration, or fill missing values as though they were measured geometry.
2. Until expanded features cover all cases, full-population scalar searches are valid; richer subset results remain supplementary. The full plotted population is still subject to the explicitly retained edge-compound display filter.
3. Broaden feature/model candidates only for percentage and absolute lag. Keep model selection inside river-grouped validation; report pooled and equal-river error, simple/no-change baselines and feature coverage.
4. For synthetics retain grouped geometry validation, the balanced factorial sensitivity separately, and the wider embedding extension as an explicit transfer test. Use the same response definitions across synthetic/real cases. Do not claim cross-setting R² is a matched measure of difficulty.
5. Inspect high-residual cases and assess practical error; choose a compact interpretable predictor only after the full-population comparison.

The new full-population expanded-feature search is **not complete**. Existing R² values above come from the specified earlier analyses. No routing changes or new hydraulic calibration are implied by this scope change.

## Hydraulic interpretation

### Roughness-only comparison · 5 October 2026

The legacy original synthetic manifest confirms n=.35; current production uses .035. At fixed depth/slope, raw Manning velocity scales as 1/n, so T⁺ scales as 1/n and F² as 1/n². Applying n=.35 counterfactually to the **current real base networks**, with the same broad-rise durations, gives **2.79%** passing channel length; current n=.035 gives **17.57%**. Thus the roughness change alone moves more current real links into the provisional screen. This is not a matched historical rerun: old slopes/depths, event definitions and cohorts also differed. Clipped routing celerity is not used as water velocity. Allocated-Q coordinates do not depend directly on n at fixed Q/width/depth/slope.

For the current **original synthetic grid**, Manning-reference passing length is **0%**, and allocated baseflow passing length is **0%**. The allocated peak-Q scenario reaches **0.643%**, at fixed reference depth. The current figure excludes the wider embedding extension. These results limit physical interpretation of the synthetic routing experiments; passing/failing this provisional screen is not a full wave-model classification. Do not tune roughness or event period merely to place points inside it. Source: `roughness_only_counterfactual.csv` and the selected synthetic hydraulic summary.


This is the main physical limitation, separate from statistical prediction. Synthetic experiments currently support controlled sensitivity of the selected routing model; physical transfer to the real river domain is not yet demonstrated. Establish defensible reference flow/event period and assess a small regime-matched sensitivity design before broad claims. A longer forcing rise changes the event being tested and should be reported as a distinct sensitivity, not used retroactively to relabel existing runs.

## Current notebooks

- `RAPID/notebooks/rapid_hydrograph_analysis.ipynb`: deduplicated real results, absolute/percentage timing, predictor benchmarks and metric inventory.
- `synthetic_experiments/notebooks/synthetic_results_synthesis.ipynb`: synthetic experimental sensitivity.
- `synthetic_experiments/notebooks/synthetic_predictor_validation.ipynb`: synthetic predictor development and extension investigations.
- `synthetic_experiments/notebooks/synthetic_hydraulic_applicability.ipynb`: current real/synthetic overlay and explicitly dated historical diagnostics.
