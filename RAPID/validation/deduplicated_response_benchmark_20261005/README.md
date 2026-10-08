# Deduplicated response benchmark — 5 October 2026

Reference: **DEDUPLICATED-RESPONSES-2026-10-05**.
Written after the decision to retain distinct island operations separately, remove overlapping representations, and compare refined timing and shape responses before choosing final predictors. These are exploratory model-response results, not validation against observed downstream hydrographs.

## Population and identity

A comparison is identified by river, routed-child fingerprint, matched parent edge set and child edge set. Topological representations take priority over island aliases. Identical scatter coordinates alone are never grounds for removal. Duplicate numerical predictors/responses must agree. `selection_aliases.csv` retains the original selection mapping.

With the existing first/last compound exclusion, 2,623 plotted selections become **1,652 distinct comparisons**, including **204 island-only collapses**. “Island-only” describes a distinct operation, not a new structural unit type. Their comparison domains are 173 containing-compound contexts, 17 closed compound units, 11 closed units and 3 enclosing surviving-anchor domains. These domain labels alone do not establish that every operation is a proper subset of a nested or compound unit.

The path-descriptor benchmark has **1,128 comparisons across 16 rivers**. Another 524 lack eligible cached descriptors; path enumeration requires a valid single-source/sink domain and no more than 10,000 paths. They remain in the full scalar benchmark. Scores from these two populations must not be compared as though only predictors changed.

## Responses and validation

The same rows are used for all four responses within each population: interpolated percentage lag change; absolute child-minus-parent peak timing in hours; event duration change; peak excess change. Existing accepted shape definitions are retained. Synthetic outlet timing now uses the same guarded three-point peak interpolation as real cases, with run-specific spin-up removed and input clocks checked per pair. All 9,519 synthetic runs passed the interpolation guard; no pair had mismatched input clocks. This resolves a measurement-method mismatch, not all physical/model differences.

Real validation holds out entire rivers. Fixed Ridge and shallow boosted-tree models are reused without hyperparameter tuning. MAE skill is 1 minus model MAE divided by held-out training-mean baseline MAE. Positive skill improves upon that baseline; it is not a percentage of explained variance. Candidate ranking remains exploratory: there is no final independent test set after choosing among these candidates.

## Findings

Common 1,128-comparison cohort, boosted models, pooled out-of-river R²:

| Predictor set | Percentage lag | Absolute hours | Duration | Peak excess |
|---|---:|---:|---:|---:|
| Omega | 0.682 | 0.108 | -0.019 | -0.010 |
| Resistance, length fraction, exposure separately | 0.635 | 0.151 | -0.078 | -0.060 |
| Selected six | 0.360 | 0.544 | -0.038 | -0.073 |
| Selected six + changes + surrounding lengths | 0.650 | 0.537 | 0.070 | 0.062 |

The selected six are resistance log, exposure, parent equivalent length, parent dominant width fraction, signed length–width association and domain reference slope. Changes comprise log child/parent equivalent-length and elongation ratios, path-length CV difference and width-dominance difference. Context comprises upstream/downstream shortest physical lengths. Path descriptors share edges and are not independent physical channel counts; compound width-dominance changes remain exploratory.

- **Omega remains a useful percentage-lag axis.** The linear model also reaches R² 0.662 on the common cohort. The boosted model beats the baseline in 15/16 rivers. On the full 1,652-comparison population, however, boosted omega R² is only **0.334**, and separate components reach **0.428**. The restricted geometry cohort therefore substantially changes apparent performance.
- **Absolute timing benefits from dimensional geometry and slope.** On the common cohort omega MAE skill is 0.135, versus 0.480 for the selected six. The latter beats the baseline in 15/16 rivers, with median within-river R² 0.528. Its linear R² is only 0.035: the result depends strongly on model form, and does not establish a simple universal formula or independence from surrounding routing.
- **Shape prediction remains weak.** Adding changes/context gives slightly positive duration/excess R², but pooled MAE skill is negative (-0.124/-0.053). This is not convincing general predictive improvement. Even a small positive MAE skill for simpler linear models accompanies negative R².
- **Residual structure is suggestive, not a settled explanation.** Median within-river signed-residual correlations for duration/excess are approximately -0.27/+0.30 with width dominance and +0.31/-0.33 with length–width association. Relations vary across rivers. Example 46 supplies 9/12 largest absolute-timing residuals, 8/12 duration residuals and 9/12 peak-excess residuals of the omega model. Percentage-lag outliers are distributed across examples 03, 15, 22, 54 and 56.

## Synthetic transfer scope

The frozen-transfer test uses the **original full-grid development/geometry-holdout split**, not the wider embedding extension. After comparison deduplication it contains 13,213 unique valid comparisons. Selected-six synthetic holdout R² is 0.949 for percentage lag, 0.924 for hours, 0.774 for duration and 0.814 for excess. Transfer to the real cohort fails for that model (all four R² negative). A simpler component model reaches only 0.151 for absolute hours.

The domains are markedly different: original synthetic development length fractions are 0.235–0.722, while real fractions are 0.0004–0.1885. All real comparisons fall below this synthetic training range; about 94–95% also fall outside its upstream/downstream distance ranges. This is a specific limitation of the original-grid training population, not a claim that the already-completed embedding extension lacks smaller fractions. That extension remains a separate external stress test and was not used to select or refit these predictors.

Synthetic holdout success therefore cannot be presented as successful real-network validation. Small real shape changes also make transfer R² sensitive to incorrect predicted amplitudes. Forcing, allocation, routing, geometry range and surrounding network scale remain relevant. Results are conditional on those choices, not evidence of a universal physical law.

## Next scientific decisions

1. Retain percentage lag with omega for interpretation, and absolute timing as a complementary prediction target; neither is established as universally preferable.
2. Inspect the exported high-residual domains, particularly example 46, and numerical resolution of small shape changes before interpreting them physically.
3. Use the existing wider synthetic embedding extension to assess overlap with real geometry and test frozen candidates separately. Do not merge extension points into balanced factorial variance contributions.
4. Only then freeze a parsimonious predictor set and define an independent validation design. No new hydraulic calibration or routing run was undertaken here.

## Reproduction and files

Run `RAPID/investigations/deduplicated_response_benchmark.py` in the river-hierarchy environment, then execute `RAPID/notebooks/rapid_hydrograph_analysis.ipynb`. `provenance.json` records sources, features and populations; `scores.csv` includes pooled, river and category scores; `predictions.parquet` contains held-out predictions. Residual tables, synthetic transfer scores and feature ranges accompany the three static figures.

Validation: 20 targeted tests passed; the complete analysis notebook executed successfully, including its assertion that notebook and benchmark comparison identities agree. Static benchmark figures were visually inspected.

### Which additions help in synthetics and real rivers? · 5 October 2026

Fixed boosted models, identical rows within each dataset, with no additional model tuning. This compares **predictive value of adding a group to omega**, not independent causal importance. Synthetic geometry holdout and real leave-river-out are different generalization tasks; their R² values are not directly comparable measures of difficulty.

For percentage lag, width dominance improves synthetic R² from .721 to .903, size/proportions to .873 and length–width association to .861. In real cases those additions give .680, .676 and .687 versus .682 for omega alone. For absolute timing, size helps in both (.203→.762 synthetic, .108→.368 real), and slope helps in both (.203→.314 synthetic, .108→.409 real). Real shape-response predictions remain weak despite useful synthetic shape predictions.

The controlled grid, different parameter ranges, stronger responses and simpler structures are plausible contributors. This comparison does not isolate their individual causes. A broader model search may help, but use nested river-grouped validation (inner model selection, outer assessment), retain simple baselines, and report equal-river scores. Choosing the best model on the current test scores would make the reported score optimistic.

Reproduce the additional table with `RAPID/investigations/matched_synthetic_predictor_additions.py`; results are in `synthetic_predictor_additions.csv`.
