# Quadratic ridge: remaining decisions and five-input definitions

Reference **MODEL-FINALIZATION-2026-11-04**. Written on 4 November 2026 after synthetic compact-selection revisions and matched real-network RF/ridge comparisons. Purpose: record the remaining work before choosing one main predictive model. This is a plan, not a record of completed finalization.

## Remaining steps

1. Resolve the fifth input with one matched quadratic-ridge comparison: the existing mean-width-ratio/complexity-change five with Manning ΔGₘ versus the same first four with planform-area-proxy change ΔA. The latter is the user's preferred simpler alternative. Do not reopen the broad candidate search. This ridge comparison has not yet been run; existing no-G results below are RF results.
2. Freeze the selected five inputs, their definitions/scopes/transforms, response definition and degree-two expansion. Explicitly distinguish predictor selection, model-family choice and penalty selection.
3. Prespecify a small ridge-penalty sensitivity grid around the current per-row penalty 1e-4. Fit scaling and select penalties only within grouped inner training folds; retain outer folds for assessment. Do not select a penalty from the displayed outer scores. No penalty grid has been executed as part of this note.
4. Retain compound-family grouped, leave-one-network-out and network-grouped fivefold assessment. Report ordinary parent–child cases (2,170) separately from the 15 full-network controls. Preserve RF as a fixed-settings benchmark; report R² and MAE with per-river errors and repeat variability.
5. Inspect largest errors, coefficient/prediction sensitivity and out-of-range behavior of quadratic terms. In particular, check the full-network controls, unusual geometry ranges and whether apparent improvement depends on a few rivers. Do not interpret ridge coefficients as independent causal effects.
6. If the checks support it, freeze one main equation, preprocessing, penalty rule and permitted input scope; retain sensitivity results and RF comparison as supporting results. Document that the candidate pool and real validation data were repeatedly inspected: this is exploratory assessment, not fully independent synthetic-only feature discovery.

## What leaving out G currently means

The tested five-input no-G panel replaces ΔGₘ by ΔA; simply removing G would leave four predictors. The first four inputs remain the successful mean-width-ratio/complexity-change panel, not the later absolute-parent-width/bifurcation-count selection.

Existing RF centroid R² on ordinary real cases:

| Validation | Five with G | Five with ΔA |
|---|---:|---:|
| Compound-family grouped | 0.7573 | 0.6892 |
| Leave one network out | 0.7062 | 0.5844 |
| Network-grouped fivefold | 0.6983 | 0.5414 |

These differences are not negligible. The preference for a simpler no-G expression is valid as an interpretability tradeoff, but equivalent performance has not been demonstrated. The existing real quadratic-ridge comparison used G and cannot establish no-G ridge performance.

Evidence: `RAPID/outputs/mean_width_compact_20261103/summary.csv` and `RAPID/outputs/mean_width_ridge_vs_rf_20261103/summary.csv`.

## Definitions for the five-input no-G panel

Let s be p (parent) or c (child); e is a reach; L_e is reach length in metres, B_e width in metres, and S_e positive slope in m/m. D_s denotes the selected parent/child domain. Let f_e be the geometry-derived fraction of domain-inlet discharge traversing reach e. At each bifurcation, outgoing fractions are proportional to width; fractions add at confluences. Thus f_e is also the probability of traversing e under width-based path allocation, not a fitted or simulated discharge.

For ordinary comparisons, use matched closed domains at common boundary anchors. For full-network controls, length/width/area sums use selected edited reaches, excluding unchanged connector rewards; preserve connected graph context for allocation and route counting. Topological complexity is computed on the connected reference graph, as in the existing preparation code. Do not sum independent region route counts as a replacement.

### 1. Change in allocated mean route length

\[
\mu_{L,s}=\sum_{e\in D_s}f_{e,s}L_{e,s},\qquad
x_1=\mu_{L,c}-\mu_{L,p}.
\]

Units: metres. Equivalent to the width-allocation-probability-weighted mean complete-route length difference. Code: `B_delta_route_length_mean`.

### 2. Parent inverse-square-root slope descriptor

\[
x_2=\frac{\sum_{e\in D_p}f_{e,p}L_{e,p}S_{e,p}^{-1/2}}
{\sum_{e\in D_p}f_{e,p}L_{e,p}}.
\]

Code: `parent_slope_slow`. This is an allocated-length-weighted mean of inverse square-root slope, not arithmetic mean slope. Larger values indicate flatter slope conditions. Slope is dimensionless under m/m.

### 3. Transformed mean-width ratio

\[
\overline B_s=\frac{\sum_{e\in D_s}L_{e,s}B_{e,s}}{\sum_{e\in D_s}L_{e,s}},\qquad
x_3=\ln\left(1+\frac{\overline B_c}{\overline B_p}\right).
\]

Code: `log1p__B_ratio_width_mean`. Each physical reach enters once, weighted by length. The model uses log1p of the ratio, not its change or percentage. Unchanged mean width gives ln(2), not zero; training standardization handles centering.

### 4. Change in topological complexity

\[
C_s=\ln(1+N_{\mathrm{routes},s})
+\ln(1+N_{\mathrm{bif,int},s})
+\ln(1+N_{\mathrm{conf,int},s}),\qquad x_4=C_c-C_p.
\]

N_routes counts complete directed inlet-to-outlet routes. N_bif,int counts bifurcation nodes excluding the inlet; N_conf,int counts confluence nodes excluding the outlet. A bifurcation is out-degree >1 and a confluence is in-degree >1, using directed multigraph edges. Counts are not weighted by flow. Code: `path_delta_topologic_complexity_score`. Natural logarithms throughout. This is a constructed topology summary, not entropy.

### 5. Change in channel planform-area proxy

\[
A_s=\sum_{e\in D_s}L_{e,s}B_{e,s},\qquad x_5=A_c-A_p.
\]

Units: square metres. Code: `B_delta_area_proxy`. This is summed reach length × width, not storage volume or necessarily a unioned map polygon area. No depth or roughness enters it. It replaces G in the existing no-G five-input model.

Code references: `RAPID/investigations/synthetic_selected_sets_20261102/prepare.py`, `RAPID/investigations/matched_centroid_sets_20261103/compare.py`, `RAPID/investigations/closed_unit_extension_20261019/edited_features.py`, and `RAPID/investigations/mean_width_compact_20261103/run.py`.

## Quadratic ridge and linear regression

A quadratic response surface remains linear in its fitted coefficients. For example β₀+β₁x+β₂x² is curved in x but linear in β. Our five inputs give five main effects, five squares and ten pair products: twenty terms plus an intercept. Ridge adds an L2 penalty on the standardized-term coefficients to weighted least-squares fitting; it leaves the intercept unpenalized. It stabilizes fitting but does not remove predictor redundancy or establish causality.

Read: [scikit-learn on polynomial regression as a linear model](https://scikit-learn.org/stable/modules/linear_model.html#polynomial-regression-extending-linear-models-with-basis-functions) and [ridge regression](https://scikit-learn.org/stable/modules/linear_model.html#ridge-regression-and-classification).


## Status update — 4 November 2026, entropy replacement test

The previously pending no-G complexity ridge baseline has now been fitted using the identical real folds, engineered inputs and penalty as the G comparison. It is substantially weaker for held-out networks and full-network controls; equivalent ridge performance without G is not supported. Replacing complexity change with entropy improves average ordinary performance but does not resolve negative network-grouped ridge R². [Detailed results](../../RAPID/outputs/entropy_replacement_20261104/report.md). The final predictor choice remains open; bounded training-fold penalty sensitivity and residual/extrapolation checks remain pending. Entropy+G has not been tested.


## Status update — 4 November 2026, entropy + G tested

The five-input entropy+G candidate has now been tested at the unchanged ridge penalty. Ordinary R² improves to 0.7732/0.6243/0.5651, but full-network control R² falls to 0.5260/0.5072/0.4346 compared with complexity+G ridge 0.8374/0.8571/0.7915. [Matched results](../../RAPID/outputs/entropy_g_ridge_20261104/report.md). The candidate is promising for ordinary cases but not uniformly superior. No model is frozen by this test; bounded inner-fold penalty sensitivity and residual/extrapolation diagnostics remain pending.


## Status update — 4 November 2026, six-input candidate

User explicitly allowed six predictors and requested retaining both entropy and complexity. This candidate improves ordinary R² to 0.8157/0.6878/0.6737 and full-network R² to 0.7019/0.7451/0.7665 versus the entropy-only topology panel. It is a stronger compromise, although complexity-only ridge retains higher full-network R² and RF retains lower ordinary MAE. [Matched results](../../RAPID/outputs/entropy_complexity_g_ridge_20261104/report.md). The earlier five-input preference is superseded by authorization to consider six; broad candidate search remains closed. Six-input ridge is a candidate for the bounded inner-fold penalty and residual/extrapolation checks, which remain pending. No automatic final model freeze or size-based causal claim follows.


## Decision update — six carried forward

User accepted six-input quadratic ridge as the working model structure. [Recorded decision and rationale](2026-11-04_six_input_ridge_decision.md). This supersedes the open five-versus-six structural decision, but not the pending inner-fold penalty and residual checks. KGE is now available as a supplementary diagnostic on held-out centroid-shift predictions; signed-mean and weighting qualifications are documented. No final coefficient/penalty freeze has occurred.


## Status update — exact six-input area alternative tested

The user-requested G→area replacement in the six-input model has now been tested. All headline metrics and separate-population diagnostics favor retaining G at the matched fixed penalty. [Results](../../RAPID/outputs/six_area_vs_g_20261104/report.md). This closes the immediate area alternative check; bounded inner-fold penalty sensitivity and residual/extrapolation diagnostics remain pending.


## 6 November 2026 — consolidated settings and retained secondary model

Reference: **GEOMETRY-PREDICTOR-SPEC-2026-11-06**. At the user's request, the accepted settings, equations, numerical values, selection rationale and outstanding checks are now consolidated in [Current geometry-based prediction models](../../RAPID/CURRENT_GEOMETRY_PREDICTION_MODELS.md), with a [machine-readable manifest](../../RAPID/config/geometry_prediction_models.json).

Carry forward the six-input quadratic ridge with ΔGₘ as primary. Retain the matched **six-input area-change replacement** as the secondary model without ΔGₘ; this is not the five-input model obtained by simply dropping G. Both use the existing completed evaluations and are selectable in `hydrography_rf_prediction_vs_modelled.ipynb` via `RIDGE_MODEL`. No new fits or coefficient changes were made for this documentation update.

The fixed ridge penalty is inherited, not demonstrated optimal. Grouped penalty sensitivity, residual/extrapolation review and the final full-data coefficient artifact remain pending. Retain the weaker secondary results when completing those checks.
