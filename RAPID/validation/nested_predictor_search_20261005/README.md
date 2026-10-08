## Bounded nested predictor search · 5 October 2026

Reference **NESTED-PREDICTOR-SEARCH-2026-10-05**. This extends predictor investigation after establishing that real shape changes are small. No routing or response definition changed.

All candidates use the same 1,128 eligible comparisons from 16 rivers. The 524 path-rich cases excluded from this cohort remain outside these results. Each outer fold holds out one complete river. Four grouped inner folds within the other rivers select among 12 candidates: four feature sets (omega; separate resistance/length fraction/exposure; selected six; selected six plus changes/context) crossed with linear Ridge, regularized quadratic Ridge and shallow histogram boosting. Scaling is fitted inside each training fold. Selection minimizes the mean of per-river MAEs, rather than pooling observations across unequally sized rivers. The outer river never enters this selection.

| Response | Outer pooled R² | Model MAE | No-change MAE | Rivers beating no-change |
|---|---:|---:|---:|---:|
| Percentage lag (percentage points) | 0.622 | 0.15372 | 0.29948 | 16/16 |
| Absolute timing (hours) | 0.544 | 0.13496 | 0.26578 | 15/16 |
| Event duration (percentage points) | -0.009 | 0.00684 | 0.00524 | 2/16 |
| Peak excess (percentage points) | -0.128 | 0.02088 | 0.01434 | 5/16 |

The timing models approximately halve pooled MAE relative to predicting no change. Absolute timing consistently selects the boosted selected-six model in all 16 outer folds; its MAE is about 8.1 minutes. Percentage lag selects quadratic omega in 10 folds and the extended boosted model in six. The nested search's R² of .622 does not improve on the prior fixed omega benchmark (.682); selecting by inner equal-river MAE need not maximize pooled outer R². This search does not justify additional complexity for the readable lag axis.

Both shape models lose to the no-change baseline, despite their very small errors in percentage-point units. Event duration selects a simple linear model in every fold. The result supports reporting effect-size distributions, resolution checks and the larger-response tail before seeking a more elaborate universal shape predictor. It does not prove exact zero physical change.

**Interpretation of small shape responses:** fixed routing coefficients produce a linear response to the imposed inlet hydrograph. Broad input features and common surrounding routing can make local changes appear mainly as small timing shifts. The real cases use saved upstream event hydrographs, which can contain multiple peaks, not necessarily an idealized single pulse. There is one imposed upstream input; unmodelled tributary event inputs and stage-dependent floodplain processes are absent. These are plausible mechanisms, not a quantified attribution of the small response. Event duration also does not measure every possible internal reshaping. A focused input-duration sensitivity with hydraulics held fixed could distinguish forcing dependence from a universally weak geometry effect.

**Validation limits:** candidate specifications reflect prior exploration of these same rivers. Nested selection prevents direct outer-fold tuning in this run, but does not make these rivers an untouched external validation set. Scores are exploratory, and comparison rows within a river are correlated. No broad hyperparameter optimization or synthetic model refit was performed.

Reproduce `RAPID/investigations/nested_predictor_search.py`. `provenance.json` fixes candidates and selection rules; `inner_scores.csv` retains every candidate/fold-selection score; `selections.csv` records each outer choice; `predictions.parquet` contains held-out predictions; `scores.csv` includes pooled/per-river results and zero/training-mean/training-median baselines. Fold assertions verify disjoint training/validation rivers.
