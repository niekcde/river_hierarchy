# Individual predictor screen — 2 November 2026

Reference: SYN-INDIVIDUAL-PREDICTOR-SCREEN-2026-11-02. All 100 working affine classes evaluated; no candidates removed.

## Design and interpretation

Five deterministic shuffled case-group folds (seed 20261102) hold all seven collapse entries of a parameter combination together. The same folds apply across all predictors, responses and depth approaches. This evaluates interpolation to held-out combinations within the existing grid, not extrapolation to new control ranges, unseen collapse types or real networks. Controls irrelevant to a treatment can yield repeated geometries across different cases; folds do not claim unique-geometry holdout. The synthetic data have already informed prior decisions; these are comparative internal validation scores, not untouched external validation.

Each predictor alone fits a cubic spline with at most six training-quantile knot locations (duplicate knots removed), linear extrapolation, and ridge alpha 1e-6. Training-only centring/scaling and knot fitting prevent fold leakage. Hyperparameters are fixed across candidates; there is no search. Constant predictors return the training response mean. One multi-output fit shares only the predictor basis, with independent coefficients for the six method/response targets. No topology indicator or second predictor is supplied.

Pooled models fit all collapse types together; within-type models are fitted separately. These answer different questions. The main ranking is equal-weight mean out-of-fold R² across response-resolved types. Constant predictors remain in that average as baseline predictions; correlations are undefined for them and their available-type counts are reported. Pooled R² can reflect between-type structure and must not substitute for within-type sensitivity.

The existing operational resolution screen is retained: full-range centroid change >4 s, peak change >30 s. These are screening conventions, not validated numerical error bounds. Unresolved strata still have MAE reported, but R² is not ranked. Responses are not rounded. Predictor roundoff handling follows the earlier audit. Signed correlations are retained in detailed tables. Negative R² is not clipped in tables or averages. No p-values or independent-replication confidence intervals are claimed.

Five undecided affine classes use temporary labels listed in screened_candidates.csv. This does not settle real-network representatives. Domain size is converted from km to m in the analysis; other source features are preserved.

Cross-method figures and common_support_methods.csv use the same six resolved treatments for all three approaches. Method-specific rankings below retain all resolved treatments: six for Manning, seven for BASED. Do not attribute differences between those unequal-support averages solely to hydraulics.

## Leading standalone candidates

### variable_manning: centroid_shift_minutes

| Candidate | Mean type R² | Worst type R² | Pooled R² | Resolved types |
|---|---:|---:|---:|---:|
| `delta_G_manning` | 1.000 | 1.000 | 1.000 | 6 |
| `B_delta_route_length_mean` | 0.823 | 0.805 | 0.855 | 6 |
| `B_delta_area_proxy` | 0.822 | 0.810 | 0.848 | 6 |
| `path_delta_equivalent_length` | 0.818 | 0.805 | 0.854 | 6 |
| `B_parent_route_length_mean` | 0.816 | 0.776 | 0.776 | 6 |
| `path_parent_branching_density_by_length` | 0.665 | -0.001 | 0.309 | 6 |
| `path_delta_branching_density_by_length` | 0.649 | -0.001 | 0.077 | 6 |
| `B_parent_route_length_sd` | 0.647 | 0.643 | 0.618 | 6 |
| `B_delta_route_length_sd` | 0.647 | 0.643 | 0.668 | 6 |
| `B_parent_branch_length_variance_sum` | 0.643 | 0.640 | 0.628 | 6 |

### variable_manning: peak_shift_minutes

| Candidate | Mean type R² | Worst type R² | Pooled R² | Resolved types |
|---|---:|---:|---:|---:|
| `B_delta_area_proxy` | 0.727 | 0.719 | 0.767 | 6 |
| `B_parent_route_length_mean` | 0.628 | 0.599 | 0.637 | 6 |
| `B_delta_route_length_mean` | 0.623 | 0.613 | 0.680 | 6 |
| `path_delta_equivalent_length` | 0.622 | 0.613 | 0.679 | 6 |
| `delta_G_manning` | 0.565 | 0.554 | 0.630 | 6 |
| `path_parent_branching_density_by_length` | 0.513 | -0.001 | 0.229 | 6 |
| `parent_slope_slow` | 0.503 | 0.456 | 0.447 | 6 |
| `path_delta_branching_density_by_length` | 0.503 | -0.001 | 0.062 | 6 |
| `B_parent_route_length_sd` | 0.492 | 0.490 | 0.497 | 6 |
| `B_delta_route_length_sd` | 0.492 | 0.490 | 0.532 | 6 |

### fixed_manning: centroid_shift_minutes

| Candidate | Mean type R² | Worst type R² | Pooled R² | Resolved types |
|---|---:|---:|---:|---:|
| `delta_G_manning` | 1.000 | 1.000 | 1.000 | 6 |
| `B_delta_route_length_mean` | 0.831 | 0.814 | 0.863 | 6 |
| `B_delta_area_proxy` | 0.825 | 0.814 | 0.850 | 6 |
| `path_delta_equivalent_length` | 0.825 | 0.814 | 0.861 | 6 |
| `B_parent_route_length_mean` | 0.824 | 0.783 | 0.782 | 6 |
| `path_parent_branching_density_by_length` | 0.671 | -0.001 | 0.314 | 6 |
| `path_delta_branching_density_by_length` | 0.654 | -0.001 | 0.077 | 6 |
| `B_parent_route_length_sd` | 0.651 | 0.647 | 0.621 | 6 |
| `B_delta_route_length_sd` | 0.651 | 0.647 | 0.672 | 6 |
| `B_parent_branch_length_variance_sum` | 0.647 | 0.643 | 0.630 | 6 |

### fixed_manning: peak_shift_minutes

| Candidate | Mean type R² | Worst type R² | Pooled R² | Resolved types |
|---|---:|---:|---:|---:|
| `B_delta_area_proxy` | 0.707 | 0.699 | 0.752 | 6 |
| `B_parent_route_length_mean` | 0.619 | 0.586 | 0.629 | 6 |
| `B_delta_route_length_mean` | 0.619 | 0.606 | 0.679 | 6 |
| `path_delta_equivalent_length` | 0.615 | 0.606 | 0.678 | 6 |
| `parent_slope_slow` | 0.545 | 0.500 | 0.485 | 6 |
| `delta_G_manning` | 0.527 | 0.513 | 0.597 | 6 |
| `path_parent_branching_density_by_length` | 0.504 | -0.001 | 0.235 | 6 |
| `path_delta_branching_density_by_length` | 0.494 | -0.001 | 0.062 | 6 |
| `B_parent_route_length_sd` | 0.482 | 0.480 | 0.487 | 6 |
| `B_delta_route_length_sd` | 0.482 | 0.480 | 0.524 | 6 |

### fixed_based: centroid_shift_minutes

| Candidate | Mean type R² | Worst type R² | Pooled R² | Resolved types |
|---|---:|---:|---:|---:|
| `B_parent_route_length_mean` | 0.708 | 0.637 | 0.631 | 7 |
| `B_parent_bifurcation_density` | 0.689 | 0.626 | 0.557 | 7 |
| `B_delta_total_length` | 0.681 | 0.627 | 0.619 | 7 |
| `B_delta_bifurcation_density` | 0.679 | 0.626 | 0.045 | 7 |
| `domain_size` | 0.666 | 0.626 | 0.656 | 7 |
| `path_parent_branching_density_by_length` | 0.605 | -0.001 | 0.363 | 7 |
| `B_parent_route_length_sd` | 0.545 | 0.264 | 0.546 | 7 |
| `B_parent_branch_length_variance_sum` | 0.545 | 0.265 | 0.549 | 7 |
| `B_child_total_length` | 0.534 | 0.405 | 0.300 | 7 |
| `path_delta_branching_density_by_length` | 0.523 | -0.001 | 0.148 | 7 |

### fixed_based: peak_shift_minutes

| Candidate | Mean type R² | Worst type R² | Pooled R² | Resolved types |
|---|---:|---:|---:|---:|
| `path_parent_head_drop` | 0.398 | 0.362 | 0.325 | 7 |
| `parent_slope_slow` | 0.346 | 0.298 | 0.312 | 7 |
| `B_parent_area_proxy` | 0.326 | 0.281 | 0.319 | 7 |
| `path_parent_slope_mean` | 0.326 | 0.293 | 0.284 | 7 |
| `path_parent_slope_min` | 0.316 | 0.286 | 0.247 | 7 |
| `path_delta_slope_mean` | 0.261 | 0.202 | 0.139 | 7 |
| `path_delta_slope_min` | 0.236 | -0.000 | 0.167 | 7 |
| `B_delta_area_proxy` | 0.232 | -0.000 | 0.286 | 7 |
| `domain_size` | 0.179 | 0.114 | 0.219 | 7 |
| `B_parent_route_length_mean` | 0.179 | 0.114 | 0.220 | 7 |

## Limitations and next step

This fixed univariate model measures standalone usefulness, not optimal univariate predictability or added information in combinations. Poor scores do not establish that a descriptor is wrong or useless; interactions can make weak marginal variables useful. Fold variation is a partition sensitivity description, not an iid confidence interval. Next compare related candidates and documented alternatives as additions to common baselines using the same grouped splits, with subsequent selection confined to training folds. Keep real-network checks separate.

Outputs: candidate_summary.csv and per-response ranking tables; individual_scores.csv for all strata; fold_scores.csv; fit_diagnostics.csv; case_folds.csv; screened_candidates.csv; two top-20 figures.
