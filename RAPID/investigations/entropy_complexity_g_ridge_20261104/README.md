# Six-input quadratic ridge: retain entropy and complexity with G

Reference **ENTROPY-COMPLEXITY-G-RIDGE-2026-11-04**. User requested combining the complementary topology/allocation descriptors rather than replacing one with the other.

Inputs: allocated mean route-length change, parent slope descriptor, log1p(child/parent length-weighted mean width), topological complexity change, geometry-only Manning accumulation change ΔGₘ, and allocated route-entropy change. No depth, Manning n or simulated discharge input. G's derivation remains conditional on Manning hydraulics.

Same real data, response, folds, weighting and fixed ridge penalty as the five-input tests. Degree-two expansion now has six main effects, six squares and fifteen pair products: **27 terms plus intercept**. Thus adding entropy adds its main effect, square and five interactions. The result tests this enlarged quadratic panel, not an isolated linear entropy term.

Fit 65 real training/test partitions spanning compound-family grouped, leave-one-network-out, and network-grouped fivefold validation. Retain all 2,185 comparisons; report 2,170 ordinary and 15 full-network controls separately. Weighted training-only standardization; alpha = 1e-4 times normalized training-weight sum; unpenalized intercept; no tuning. Numerical normal-equation checks passed for every fit, and all baseline row keys and targets match exactly. Source hashes verified. Completed in 1.8 seconds.

```sh
/opt/anaconda3/envs/river-hierarchy-rivgraph/bin/python -u RAPID/investigations/entropy_complexity_g_ridge_20261104/run.py
```

Outputs: `RAPID/outputs/entropy_complexity_g_ridge_20261104/`, including `summary.csv`, `report.md`, `paired_changes.csv`, `per_river_errors.csv`, `largest_errors.csv`, `numerical_checks.csv` and `completion.json`.

## Findings

Ordinary centroid-shift R²:

| Validation | Six-input ridge | Entropy + G ridge (five) | Complexity + G ridge (five) | Complexity + G RF (five) |
|---|---:|---:|---:|---:|
| Compound-family grouped | 0.8157 | 0.7732 | 0.7299 | 0.7573 |
| Leave one network out | 0.6878 | 0.6243 | 0.5464 | 0.7062 |
| Network-grouped fivefold | 0.6737 | 0.5651 | 0.4960 | 0.6983 |

Six-input ridge improves ordinary R² over both five-input ridge models in every matched repeat. Ordinary MAE is 0.7687/0.8770/0.8746 minutes, improved over both ridge baselines but higher than the RF benchmark (0.5661/0.6608/0.6694). Its higher compound-family R² does not imply lower average absolute error than RF.

Full-network control R²:

| Validation | Six-input ridge | Entropy + G ridge (five) | Complexity + G ridge (five) |
|---|---:|---:|---:|
| Compound-family grouped | 0.7019 | 0.5260 | 0.8374 |
| Leave one network out | 0.7451 | 0.5072 | 0.8571 |
| Network-grouped fivefold | 0.7665 | 0.4346 | 0.7915 |

Full-network MAE is 25.11/25.81/25.98 minutes, substantially improved over the entropy-only topology panel. Complexity-only ridge still has the higher full-network R²; six-input ridge has lower full-network MAE for network-grouped fivefold (25.98 versus 29.50 minutes) despite lower R².

Interpretation: the six-input ridge is a stronger compromise across populations and a reasonable main-model candidate for the planned bounded penalty and residual checks. It is not uniformly best. The observed ordinary/full-network contrast does not establish size as the causal driver: these populations also differ in topology, edit scope and timing-change magnitude. Only 15 full-network controls exist. No size-stratified mechanism test or six-input RF has been performed here.

The validation data have been repeatedly inspected; describe this as exploratory model development. Do not freeze the model or claim independent final validation before the documented checks.
