# Statistical companion: durable research contract

Status: implementation started 19 October 2026. This document records intent; RESULTS.md records findings. Changes to implementation must preserve this contract or explicitly record a design amendment in CHANGELOG.md. This is a companion to the frozen forest, not its replacement.

## Question and estimands

Describe conditional direction, magnitude and uncertainty of associations between seven geometric descriptors and simulated full excess-volume centroid shift (child minus parent). These are not causal physical effects or estimates of uncertainty in real-world hydrology. Prior exploration informed model selection, so inference and performance comparisons are exploratory.

Primary mixed model: y_ij = beta_0 + x_ij beta + b_j + epsilon_ij, with independent network intercepts b_j ~ N(0,tau²), fitted by REML. Its working residual model assumes equal residual variance and conditional independence. Shared parents, overlapping edits and repeated domains may violate that working covariance; uncertainty therefore resamples entire networks. Random intercepts do not implement equal-network weighting and assume independence between network intercepts and predictors. Within/between-network diagnostics will flag interpretive limitations; coefficients are not automatically within-network effects.

Sensitivity: linear regression with each network receiving equal total squared-error weight. This estimates a different weighted association. Neither method fits relative-error loss.

## Frozen data contract

Exactly 2,170 unique (example,state) comparisons in 15 networks: 02,03,05,13,15,21,22,37,41,45,46,48,54,55,56. Read input from hydrograph_relative_sweep_20261019/input.parquet and seven_inputs predictions from hydrograph_no_exposure_20261019/predictions.parquet; feature order comes from final_model.json. Preserve shared 0.5% windows, existing outer domains, and all seven transformations. No exposure, resistance, response-derived predictors, or whole-network single-centerline states. Do not regenerate response tables. Preserve source hashes.

Response parent is S000_base within each network. Geometric outer domains can recur across child edits; an overlap-group label is a diagnostic, not another independent sampling unit. Audit repeated geometric parent signatures without claiming they uniquely identify physical domains.

## Evaluation contract

Leave one entire network out. Fit preprocessing only on training rows. Predict unseen networks using fixed effects, with random intercept zero and no use of held-out responses. Pool held-out predictions with equal network weights; report weighted R², MAE in minutes, and 100*mean_network(mean_rows(abs(error)/max(abs(response),1 minute))). Report per-network scores too. Compare against saved frozen predictions; no new untouched test is claimed. Peak evaluation transfers centroid predictions only.

Sensitivity scenarios: exclude 54, exclude 37/CB001_bubble_3, and exclude both. Evaluation-only exclusions filter unchanged baseline-cohort predictions. Refit exclusions remove those observations before full fitting and every held-network fold; weights are recomputed. Do not label saved forest exclusions as forest refits.

## Uncertainty and effect sizes

Use 2,000 seeded pairs-bootstrap draws of whole networks, keeping all rows of each sampled network and assigning a fresh group ID to each draw. Recompute regression weights per sampled group. Report percentile 95% intervals as pointwise exploratory uncertainty, not simultaneous coverage or significance tests. With only 15 observed networks, resampling cannot establish reliable tail calibration or remove post-selection bias. Report failures, boundary fits, coefficient sign stability, and leave-one-network-out ranges. No p-values in this stage. If formal tests are later introduced, document a validated small-group procedure and adjust the seven tests (e.g. Holm); do not retrofit ordinary row-level tests.

Report coefficients in minutes per transformed unit and contrasts from equal-network weighted predictor Q25 to Q75, keeping others fixed. These are algebraic conditional contrasts and may have poor joint support under collinearity. Also report ratio 0.5 to 1, parent length +1 km, and dominance +0.1 as explicit reference contrasts, with range/support caveats. Positive means later centroid (or less advance). Hold contrast endpoints fixed across bootstrap fits to isolate coefficient uncertainty.

## Adequacy and extensions

Inspect held-network residuals versus fitted values and predictors, fold influence, weighted and within-network correlations, VIF and condition numbers. Do not drop collinear baseline predictors silently. A GAM is a separately documented extension only if linear inadequacy warrants it: same inputs, low-rank penalized smooths, training-only smoothing selection, no broad interaction search. First stage does not fit a GAM.

## Implementation and reproducibility

All new source stays in this investigation directory; outputs in RAPID/outputs/hydrograph_statistical_companion_20261019; dedicated notebook only. Never overwrite shared notebooks or frozen artifacts. The run records source hashes, versions, seed, diagnostics and checks. Adjust implementation freely for bugs/numerics, but record substantive changes and rerun affected outputs. Output directory is owned exclusively by this investigation.

The scalar random-intercept REML implementation profiles sigma² and beta with V = sigma²(I + lambda ZZ'), explicitly comparing lambda=0. It uses group sufficient statistics, a broad log-lambda grid and bounded local optimization, and is checked against statsmodels MixedLM independently. This avoids changing the project environment and makes whole-network bootstrap practical. Reference: https://www.statsmodels.org/stable/_modules/statsmodels/regression/mixed_linear_model.html
