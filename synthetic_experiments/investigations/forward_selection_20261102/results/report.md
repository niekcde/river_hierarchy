# Forward selection up to ten geometric predictors

Reference: SYN-FORWARD-SELECTION-2026-11-02. Status: completed internal synthetic model-selection experiment; not a final real-network predictor selection.

## Population and coverage

The accepted standalone cutoff leaves 68 candidates from the 100-class pool. It eliminates all six pure topology candidates. Those six are restored as explicit coverage reserves for BOTH searches, giving the same 74-candidate pool. Without Manning ΔG there are 73. The former 100-class inventory remains archived, not deleted. candidate_pool.csv records each variable, concept, coverage category and restoration status.

Coverage requires at least one explicit length, width, slope and topology representative. Combined variables do not satisfy multiple categories. In particular ΔG, area, elongation, width entropy, head loss and branching density are not allowed to stand in for all four principles. Pure graph-architecture descriptors qualify as topology; actual reach-slope descriptors qualify as slope. Distance distribution descriptors qualify as length even when allocation weights also use width.

## Fixed model and search

The model is quadratic ridge regression: intercept, main effects, squared terms and all pair products among the selected variables. Ten predictors therefore imply 65 expanded non-intercept terms, not ten coefficients. This explicitly permits width–length and other pair interactions. Expanded columns are centred/scaled using training moments; penalty is 1e-4 times training row count. Numeric global scaling is purely for conditioning and cancels under training-moment scaling. No model-family or regularisation search is performed; this is not RF optimisation.

Unconstrained greedy forward selection evaluates every remaining addition up to ten inputs. The coverage version chooses one missing principle at each of its first four steps, then performs up to two same-principle coordinate-swap passes on that seed before continuing unrestricted additions. This avoids forcing missing principles into the final slots, but remains a heuristic rather than an exhaustive constrained optimum. The smallest path model within 0.005 absolute INNER R² of the best path score is selected; this prespecified parsimony tolerance is a practical choice, not a significance threshold. Coverage models require at least four predictors.

There are three outer case-group folds and three inner folds. All seven collapse entries of each complete parameter combination stay together; folds are shared across hydraulic methods, responses and search variants. Selection and size choice use inner predictions only. The selected model is refit on outer training data and evaluated on the untouched outer fold for that search. The objective is pooled R² over all seven types, rather than the previous screen’s average of separately fitted within-type models. Outer per-type metrics are also exported, and unresolved near-null strata retain MAE but do not receive R².

Both searches run with and without ΔG for each of the three depth approaches and both responses. A fixed ΔG-alone quadratic baseline is evaluated on the same outer splits. Sufficient-statistic caching computes exact ridge solutions for this fixed basis; it is not training-data subsampling. Numerical SSE was verified against direct predictions, and the solution against scikit-learn Ridge in every outer split.

## Important interpretation limits

These are nested subset-selection scores CONDITIONAL on an already inspected/frozen candidate inventory. The earlier 0.05 filter and similarity decisions used the full synthetic population; they were not rebuilt inside these outer folds. Therefore this does not constitute unbiased validation of the entire discovery process. The synthetic design has been repeatedly inspected. Cases with irrelevant control differences can also repeat geometry across folds. Scores describe interpolation within this factorial design, not unseen network types, new control ranges, or observed rivers. Independent real-network evaluation remains necessary.

Pure topology variables can act as proxies for the seven synthetic treatment geometries. Their selection does not establish that the same topology mechanism transfers to more complex real cases. Coverage is an imposed interpretability constraint, not evidence that all four groups are independently necessary. Alternative greedy paths and model families may give different selections.

## Results

### variable_manning

| Search | Response | Outer R² | MAE (min) | Selected inputs across folds |
|---|---|---:|---:|---|
| ΔG alone | Centroid | 0.999805 | 0.157 | 1 |
| ΔG alone | Peak | 0.627900 | 5.043 | 1 |
| Unconstrained · ΔG allowed | Centroid | 0.999805 | 0.157 | 1 |
| Unconstrained · ΔG allowed | Peak | 0.986415 | 1.013 | 6 |
| Four groups · ΔG allowed | Centroid | 0.999900 | 0.150 | 5 |
| Four groups · ΔG allowed | Peak | 0.985650 | 1.044 | 7–8 |
| Unconstrained · without ΔG | Centroid | 0.998638 | 0.569 | 3 |
| Unconstrained · without ΔG | Peak | 0.985646 | 0.965 | 6 |
| Four groups · without ΔG | Centroid | 0.999101 | 0.478 | 5 |
| Four groups · without ΔG | Peak | 0.986858 | 0.954 | 8 |

### fixed_manning

| Search | Response | Outer R² | MAE (min) | Selected inputs across folds |
|---|---|---:|---:|---|
| ΔG alone | Centroid | 1.000000 | 0.004 | 1 |
| ΔG alone | Peak | 0.594655 | 5.127 | 1 |
| Unconstrained · ΔG allowed | Centroid | 1.000000 | 0.004 | 1 |
| Unconstrained · ΔG allowed | Peak | 0.988619 | 0.876 | 6 |
| Four groups · ΔG allowed | Centroid | 0.995297 | 0.905 | 4 |
| Four groups · ΔG allowed | Peak | 0.988906 | 0.879 | 8 |
| Unconstrained · without ΔG | Centroid | 0.999053 | 0.469 | 3 |
| Unconstrained · without ΔG | Peak | 0.987287 | 0.955 | 6–7 |
| Four groups · without ΔG | Centroid | 0.995297 | 0.905 | 4 |
| Four groups · without ΔG | Peak | 0.988267 | 0.916 | 8 |

### fixed_based

| Search | Response | Outer R² | MAE (min) | Selected inputs across folds |
|---|---|---:|---:|---|
| ΔG alone | Centroid | 0.561083 | 18.690 | 1 |
| ΔG alone | Peak | 0.159399 | 23.430 | 1 |
| Unconstrained · ΔG allowed | Centroid | 0.992505 | 2.230 | 7 |
| Unconstrained · ΔG allowed | Peak | 0.909426 | 7.994 | 10 |
| Four groups · ΔG allowed | Centroid | 0.992535 | 2.218 | 8 |
| Four groups · ΔG allowed | Peak | 0.899389 | 8.160 | 10 |
| Unconstrained · without ΔG | Centroid | 0.991208 | 2.408 | 7 |
| Unconstrained · without ΔG | Peak | 0.904571 | 8.276 | 10 |
| Four groups · without ΔG | Centroid | 0.991517 | 2.337 | 7–8 |
| Four groups · without ΔG | Peak | 0.895986 | 8.533 | 10 |

## Selected inputs: variable Manning

Counts below are selection frequencies out of three outer training folds. They are not feature importance or confidence intervals. No single final full-data model has been fitted or frozen.

### Unconstrained · ΔG allowed — centroid

- `delta_G_manning` — combined; 3/3 folds.

### Unconstrained · ΔG allowed — peak

- `B_delta_area_proxy` — combined; 3/3 folds.
- `B_ratio_route_length_mean` — length; 3/3 folds.
- `delta_G_manning` — combined; 3/3 folds.
- `parent_slope_slow` — slope; 3/3 folds.
- `path_delta_slope_min` — slope; 3/3 folds.
- `path_parent_slope_mean` — slope; 3/3 folds.

### Four groups · ΔG allowed — centroid

- `B_delta_route_length_mean` — length; 3/3 folds.
- `delta_G_manning` — combined; 3/3 folds.
- `parent_slope_slow` — slope; 3/3 folds.
- `path_delta_topologic_complexity_score` — topology; 3/3 folds.
- `path_parent_equivalent_width` — width; 3/3 folds.

### Four groups · ΔG allowed — peak

- `B_child_width_p90` — width; 3/3 folds.
- `B_delta_route_length_mean` — length; 3/3 folds.
- `B_parent_bifurcations` — topology; 3/3 folds.
- `delta_G_manning` — combined; 3/3 folds.
- `parent_slope_slow` — slope; 3/3 folds.
- `path_parent_slope_mean` — slope; 3/3 folds.
- `B_delta_route_length_cv` — length; 2/3 folds.
- `B_delta_area_proxy` — combined; 1/3 folds.
- `B_ratio_route_length_mean` — length; 1/3 folds.

### Unconstrained · without ΔG — centroid

- `B_delta_area_proxy` — combined; 3/3 folds.
- `B_delta_route_length_mean` — length; 3/3 folds.
- `parent_slope_slow` — slope; 3/3 folds.

### Unconstrained · without ΔG — peak

- `B_delta_area_proxy` — combined; 3/3 folds.
- `B_delta_route_length_mean` — length; 3/3 folds.
- `B_ratio_route_length_mean` — length; 3/3 folds.
- `parent_slope_slow` — slope; 3/3 folds.
- `path_delta_slope_mean` — slope; 3/3 folds.
- `path_parent_slope_mean` — slope; 3/3 folds.

### Four groups · without ΔG — centroid

- `B_delta_area_proxy` — combined; 3/3 folds.
- `B_delta_route_length_mean` — length; 3/3 folds.
- `parent_slope_slow` — slope; 3/3 folds.
- `path_delta_topologic_complexity_score` — topology; 3/3 folds.
- `path_parent_equivalent_width` — width; 3/3 folds.

### Four groups · without ΔG — peak

- `B_child_width_p90` — width; 3/3 folds.
- `B_delta_area_proxy` — combined; 3/3 folds.
- `B_delta_route_length_mean` — length; 3/3 folds.
- `B_parent_bifurcations` — topology; 3/3 folds.
- `B_ratio_route_length_mean` — length; 3/3 folds.
- `parent_slope_slow` — slope; 3/3 folds.
- `path_delta_slope_mean` — slope; 3/3 folds.
- `path_parent_slope_mean` — slope; 3/3 folds.

## Main findings

- Under variable Manning, ΔG alone explains centroid shift almost exactly. Without ΔG, a stable three-input set (mean route-length change, area change, and inverse-root slope) also performs strongly. Its pair products are part of the model; three inputs do not mean a simple three-term linear law.
- Peak shift requires a richer combination. Six unconstrained inputs reach high internal held-out scores; the four-principle constraint typically needs more inputs for similar performance.
- The forced-coverage search achieves similar scores in many settings, but that does not demonstrate unique added information from every category. For fixed Manning centroid, the parsimony tolerance accepts a four-input model without ΔG even when ΔG is allowed; the resulting score is slightly below the almost-exact ΔG baseline.
- BASED peak shift is harder and reaches the ten-input cap. It is not justified to claim that ten is an optimal universal size.
- No final model or universal physical law is selected here. Next review the stable small combinations, test transfer and coverage on real networks, and compare model families if the goal is predictive optimisation.

## Outputs

nested_comparison.png; candidate_pool.csv; case_folds.csv; search_paths.csv; candidate_trials.csv (forward-addition trials; seed-swap trials are evaluated but not individually exported); selected_variables_by_fold.csv; selection_frequency.csv; model_sizes.csv; outer_fold_scores.csv; outer_summary.csv; outer_predictions.parquet; selected_fold_models.json; numerical_checks.csv; provenance.json. The model JSON records exact coefficients and numeric scales for the selected outer-fold models.
