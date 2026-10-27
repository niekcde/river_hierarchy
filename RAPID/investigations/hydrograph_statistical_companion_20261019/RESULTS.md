# Statistical companion: first-stage findings

Run: 19 October 2026. Research intent and amendment rules: [DESIGN.md](DESIGN.md). These are exploratory conditional associations in simulated hydrograph responses, not causal effects or independent validation.

## Predictive performance

Identical held-network folds; unseen network intercepts are zero. All metrics use equal network weights. MAE is minutes; relative error uses a one-minute denominator floor.

| model | r2 | mae_minutes | relative_error_pct |
| --- | --- | --- | --- |
| frozen_forest | 0.615 | 2.822 | 63.737 |
| mixed | 0.070 | 6.580 | 224.695 |
| wls | 0.257 | 5.009 | 144.486 |

Both linear companions lose substantial predictive skill. Equal-network WLS performs better than the mixed model here, but neither approaches the forest. Fitting squared error rather than relative loss is also relevant to the high floored relative errors. These results do not demonstrate that geometric information is insufficient.

## Cohort and dependence

All 2,170 input/prediction keys and response values match; all seven transformations reproduce exactly. The finalized 0.5% response table is verified separately. Whole-network states remain excluded; response_coverage.csv records the 15 whole-network cases and two other omissions without claiming their exclusion provenance has been resolved. Network sizes range from 21 to 408.

Each network shares one S000_base response parent. Geometric parent signatures also recur: network 37 has 395 rows but only three signatures/overlap groups; network 22 has 408 rows but 14. A network intercept does not capture every dependence induced by overlapping edits. Bootstrap sampling retains each whole network, including all these dependencies; it assumes networks are the independent sampling units.

Equal-network pooled entropy VIF is 12.29, versus 9.77 after within-network centering. Pooled mean-width and width-CV VIFs are 7.80 and 6.95. The scaled design condition number is 8.60 (within-network: 7.48). This is full rank, but individual conditional coefficients are difficult to isolate. These coefficients combine within- and between-network information; they are not pure within-network effects.

## Conditional contrasts and uncertainty

The table reports equal-network marginal Q25→Q75 contrasts, holding the other six predictors constant. Intervals are pointwise 95% whole-network percentile-bootstrap intervals; they are not seven-test simultaneous intervals or significance tests. All 2,000 draws succeeded for both models. Only 15 original groups are available; percentile tail coverage and post-selection bias remain unresolved. LONO sign agreement describes stability, not a probability of a physical effect.

| model | term | iqr_effect_minutes | iqr_low | iqr_high | lono_same_sign |
| --- | --- | --- | --- | --- | --- |
| mixed | Mean length ratio | 3.504 | -1.151 | 8.367 | 1.000 |
| mixed | Parent length (km) | -0.662 | -9.097 | -0.199 | 1.000 |
| mixed | Length SD ratio | 2.959 | 1.191 | 6.489 | 1.000 |
| mixed | Mean width ratio | -36.241 | -65.431 | -18.083 | 1.000 |
| mixed | Width CV ratio | -12.534 | -20.955 | -7.936 | 1.000 |
| mixed | Route entropy ratio | -24.691 | -52.330 | -2.654 | 1.000 |
| mixed | Parent split dominance | 1.391 | 0.096 | 3.256 | 1.000 |
| wls | Mean length ratio | 0.265 | -2.411 | 4.005 | 0.800 |
| wls | Parent length (km) | -0.337 | -9.688 | -0.100 | 1.000 |
| wls | Length SD ratio | 0.087 | -1.624 | 2.741 | 0.533 |
| wls | Mean width ratio | -21.940 | -34.238 | -13.075 | 1.000 |
| wls | Width CV ratio | -7.899 | -12.182 | -4.944 | 1.000 |
| wls | Route entropy ratio | -7.374 | -20.697 | 4.320 | 1.000 |
| wls | Parent split dominance | 0.759 | -0.337 | 2.083 | 1.000 |

Mean width, width CV and entropy have negative conditional coefficients in both fits; parent split dominance is positive. This describes later/earlier simulated centroid timing conditional on correlated covariates, not what independently changing channel geometry would cause. Mean-length direction is much less stable in WLS, and length-SD association is sensitive to the model/weighting choice. The mixed-model mean-length bootstrap interval spans zero despite all 15 single-network deletions retaining its sign: deleting one network and resampling networks measure different aspects of stability.

Some IQR contrasts are large compared with prediction errors because they hold strongly correlated variables fixed in combinations that may be poorly supported. contrast_support.csv records descriptive joint-distance diagnostics. The reference contrasts in effects_uncertainty.csv (ratio 0.5→1, parent length +1 km, dominance +0.1) are algebraic benchmarks, not guaranteed feasible interventions.

## Explicit exclusions and refits

training_scenario=all with a reduced evaluation is evaluation-only exclusion. A reduced training_scenario means the entire LONO procedure and full coefficient fit were rerun without those observations. Saved forest predictions are only rescored; no forest refits are implied.

| model | training_scenario | evaluation | r2 | mae_minutes | relative_error_pct |
| --- | --- | --- | --- | --- | --- |
| frozen_forest | all | all | 0.615 | 2.822 | 63.737 |
| frozen_forest | all | without54 | 0.622 | 2.840 | 60.043 |
| frozen_forest | all | without_extreme | 0.713 | 2.761 | 63.733 |
| frozen_forest | all | without_both | 0.725 | 2.775 | 60.039 |
| mixed | all | all | 0.070 | 6.580 | 224.695 |
| mixed | all | without54 | 0.136 | 6.505 | 213.595 |
| mixed | all | without_extreme | -0.009 | 6.518 | 224.711 |
| mixed | all | without_both | 0.078 | 6.439 | 213.613 |
| mixed | without54 | without54 | 0.088 | 6.744 | 225.808 |
| mixed | without_both | without_both | 0.175 | 5.864 | 195.747 |
| mixed | without_extreme | without_extreme | 0.156 | 5.720 | 195.147 |
| wls | all | all | 0.257 | 5.009 | 144.486 |
| wls | all | without54 | 0.275 | 4.977 | 133.249 |
| wls | all | without_extreme | 0.268 | 4.944 | 144.494 |
| wls | all | without_both | 0.293 | 4.907 | 133.258 |
| wls | without54 | without54 | 0.266 | 5.126 | 140.531 |
| wls | without_both | without_both | 0.289 | 4.837 | 129.257 |
| wls | without_extreme | without_extreme | 0.280 | 4.743 | 134.744 |

The extreme 37 case matters for fitting as well as scoring: removing it only from mixed-model evaluation gives R² −0.009, whereas refitting without it gives R² 0.156 on that same reduced cohort. Excluding 54 does not rescue either linear model. Refit coefficient changes are retained in refit_coefficient_sensitivity.csv; they must be considered alongside bootstrap and LONO stability.

## Adequacy and next decision

residual_patterns.png shows descriptive binned held-network errors versus predictions and each input. Systematic patterns and the performance gap warrant considering a restrained nonlinear companion, but do not uniquely diagnose nonlinearity: correlated predictors, heterogeneous network relationships, influence and extrapolation may also contribute. No GAM was fitted in this stage. A later GAM requires a documented extension with low-complexity smooths and training-only selection; these same data cannot become an untouched test.

Peak-transfer metrics are in scores.csv, explicitly using centroid predictions; no peak model was trained. The separate whole-network transfer check remains separate.

## Verification and limitations

Independent statsmodels REML checks on the full cohort and two network-deletion cohorts agree in fixed-effect predictions within 3.3e-9 hours. WLS agrees with statsmodels, and population predictions are invariant to test outcomes and group labels. statsmodels emits a small-variance boundary warning on these fits; optimized variance estimates are positive, and variance diagnostics are saved rather than suppressed. Custom-fit boundary counts are in bootstrap_diagnostics.csv. Source hashes include the frozen serialized model and are verified unchanged at run completion.

No naive independent-row p-values are reported. Model-based prediction intervals, simultaneous inference and causal claims are outside this stage.
