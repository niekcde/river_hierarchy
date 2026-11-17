# Decision: carry the six-input quadratic ridge forward

Reference **SIX-INPUT-RIDGE-DECISION-2026-11-04**. User explicitly accepted the six-input model on 4 November 2026 and requested recording the reasoning and adding a KGE assessment.

## Accepted model structure

Carry forward quadratic ridge with these six geometric inputs:

1. Change in allocated mean route length.
2. Parent allocated-length-weighted inverse-square-root slope descriptor.
3. Log1p(child/parent length-weighted mean width).
4. Change in topological complexity (route count and internal junction counts).
5. Geometry-only Manning accumulation change ΔGₘ.
6. Change in width-allocation-based route entropy.

Use all degree-two terms: six main effects, six squares, fifteen interactions, and an unpenalized intercept. No depth, Manning n or simulated-discharge predictor is required. G's physical derivation remains conditional on Manning hydraulics and width-based allocation.

This accepts the input set and model family for the remaining finalization checks. It does not declare the current penalty optimally chosen, freeze final fitted coefficients, or claim independent final validation. Do not reopen the broad candidate search by default.

## Why this choice

- The original summed equivalent-path-width descriptor failed mathematically as a representation of complex overlapping routes; replacing it removed that problem. Mean-width ratio counts each reach once and has a clear interpretation.
- Synthetic selection identified useful geometric information but did not uniquely select the representation that generalized best to complex real networks. The final choice therefore combines synthetic evidence, mathematical scrutiny and exploratory real-network assessment; it is not purely automatic synthetic-only discovery.
- Entropy change captures allocation imbalance/diversity, whereas complexity change captures structural branching. Keeping both is supported by the matched prediction tests, not by a claim that their individual effects are independent or causal.
- The six-input ridge improves ordinary R² over both five-input ridge models in every matched repeat, while recovering much of the full-network control performance lost when complexity was replaced by entropy.
- It provides an explicit equation and a stronger compromise across collapse populations. It is not uniformly best: complexity-only ridge retains higher full-network R²; the previous RF has lower ordinary MAE. Six inputs expand to 27 fitted terms, so avoid calling it a six-coefficient model.
- Ordinary versus full-network results do not isolate a size mechanism. Topology, edit scope and response magnitude also differ; only 15 full-network controls are available.

## Evidence at the accepted fixed penalty

| Validation | Ordinary R² | Ordinary MAE (min) | Full-network R² | Full-network MAE (min) |
|---|---:|---:|---:|---:|
| Compound-family grouped | 0.8157 | 0.7687 | 0.7019 | 25.1054 |
| Leave one network out | 0.6878 | 0.8770 | 0.7451 | 25.8120 |
| Network-grouped fivefold | 0.6737 | 0.8746 | 0.7665 | 25.9777 |

Ordinary n=2,170; full-network n=15. Same held-out partitions, real responses and weighting as preceding comparisons. Full model comparison: [six-input results](../../RAPID/outputs/entropy_complexity_g_ridge_20261104/report.md).

## Remaining work

1. Prespecify and run bounded ridge-penalty sensitivity around the existing per-row value 1e-4. Select penalty only within grouped inner training folds, with training-only preprocessing.
2. Inspect residuals, unusual geometry ranges, extreme predictions, coefficient/prediction sensitivity and repeat variability. Retain all cases unless an independently established data error requires correction.
3. Retain all three validation strategies, separate ordinary and full-network populations, and keep the fixed RF as a benchmark.
4. If checks support the model, freeze preprocessing, penalty-selection rule, fitted equation and scope. Report the repeated real-data inspection and limits on generalization.

The previously documented five-input preference is superseded by this explicit acceptance of six.

## KGE: supplementary, not a new fitting objective

KGE-2009 was computed on saved out-of-fold centroid-shift predictions, not on discharge time series and not against gauge observations:

\[
\mathrm{KGE}=1-\sqrt{(r-1)^2+(a-1)^2+(b-1)^2},\quad
a=\sigma_{\widehat y}/\sigma_y,\quad b=\mu_{\widehat y}/\mu_y.
\]

The reference y is the routed parent–child centroid shift; the prediction is the geometric regression estimate. Primary metrics use equal-river weighted means, standard deviations and correlation, matching existing R²/MAE weighting. This is a **sample-weighted extension** of the usual KGE expression, distinct from assigning unequal weights to its three components. Components retain equal importance. Compute each score on complete out-of-fold predictions for a repeat, then average repeat scores. Equal-pair conventional scores are retained as sensitivity; do not average per-river KGE.

| Validation | Ordinary KGE, equal river | Ordinary KGE, equal pair | Full-network KGE |
|---|---:|---:|---:|
| Compound-family grouped | 0.8976 | 0.8528 | 0.6941 |
| Leave one network out | 0.8142 | 0.6610 | 0.7425 |
| Network-grouped fivefold | 0.8033 | 0.6822 | 0.7484 |

The weighting difference matters: large examples have many more pairs. Full-network equal-pair and equal-river scores coincide because each river contributes one control.

KGE adds a useful decomposition into correlation, variability ratio and mean ratio, but signed timing changes may cancel to a near-zero mean. The denominator in b then becomes unstable or undefined. The ordinary equal-river reference mean is −1.2234 minutes and SD 3.7466 minutes; 770 targets are positive and 1,400 negative. Example-level means are particularly small for 22 (+0.1791 min) and 37 (−0.2778 min), so per-river KGE needs caution. No absolute-value conversion, offset or logarithm was applied to responses.

Keep R² and MAE primary; report KGE with its components and signed mean bias. KGE does not measure explained variance and cannot inherit R²/NSE thresholds. A high predictor KGE is not proof of accurate RAPID hydrographs against gauges. The current R² implementation is 1 minus weighted squared-error ratio to the test mean, not Pearson correlation squared.

Read [Knoben, Freer & Woods (2019), KGE equations and interpretation](https://hess.copernicus.org/articles/23/4323/2019/). The signed/near-zero warning follows directly from the mean-ratio denominator. Reproducible calculation and components: [KGE report](../../RAPID/outputs/six_input_kge_20261104/report.md), `RAPID/investigations/six_input_kge_20261104/run.py`.


## Reporting preference update — 4 November 2026

User chooses the combined 2,185-pair, equal-pair-weighted table as the working headline for now: R² 0.8666/0.8571/0.8627, KGE 0.8366/0.8741/0.8595, MAE 0.9397/1.0839/1.0782 minutes across the three strategies. This includes G and both entropy and complexity. Training remains equal-river weighted; only reporting weights differ. Separate-population and balanced scores remain diagnostics. The exact six-input panel replacing G with area has NOT been tested. In the earlier matched five-input entropy panels, combined equal-pair R² with G versus area was 0.7927 vs 0.5578, 0.7557 vs 0.6414, and 0.7190 vs -2.1041. These five-input results cannot be substituted for an unperformed six-input area comparison.


## Exact six-input area replacement — 4 November 2026

User requested the matched replacement of G with area. Completed 65 fits with all other inputs and settings unchanged. Combined equal-pair R² falls from 0.8666/0.8571/0.8627 to 0.4178/0.6117/-1.0093; KGE falls from 0.8366/0.8741/0.8595 to 0.6614/0.7555/0.2003; MAE increases from 0.9397/1.0839/1.0782 to 1.2267/1.3811/1.6351 minutes. Separate ordinary and full-network diagnostics also worsen. [Matched test and rationale](../../RAPID/investigations/six_area_vs_g_20261104/README.md). This supports retaining G under the accepted specification. G has hydraulically informed assumptions but no depth, Manning n or simulated-discharge input; it is not leakage simply because its exponents are physically derived. Scope/assumption dependence and fixed-penalty limitations remain explicit.


## 6 November 2026 — consolidated settings and retained secondary model

Reference: **GEOMETRY-PREDICTOR-SPEC-2026-11-06**. At the user's request, the accepted settings, equations, numerical values, selection rationale and outstanding checks are now consolidated in [Current geometry-based prediction models](../../RAPID/CURRENT_GEOMETRY_PREDICTION_MODELS.md), with a [machine-readable manifest](../../RAPID/config/geometry_prediction_models.json).

Carry forward the six-input quadratic ridge with ΔGₘ as primary. Retain the matched **six-input area-change replacement** as the secondary model without ΔGₘ; this is not the five-input model obtained by simply dropping G. Both use the existing completed evaluations and are selectable in `hydrography_rf_prediction_vs_modelled.ipynb` via `RIDGE_MODEL`. No new fits or coefficient changes were made for this documentation update.

The fixed ridge penalty is inherited, not demonstrated optimal. Grouped penalty sensitivity, residual/extrapolation review and the final full-data coefficient artifact remain pending. Retain the weaker secondary results when completing those checks.
