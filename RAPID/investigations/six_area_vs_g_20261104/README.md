# Area change versus G in the accepted six-input quadratic ridge

Reference **SIX-AREA-VS-G-2026-11-04**. User requested the exact six-input comparison before retaining G. Only `delta_G_manning` is replaced by `B_delta_area_proxy`; allocated mean route-length change, parent slope descriptor, log1p mean-width ratio, complexity change and entropy change remain fixed.

Same 65 real validation tasks, all 2,185 comparisons, targets, training-only standardization, equal-river training weights and fixed ridge penalty 1e-4 per normalized training-weight sum. Both panels have 27 degree-two terms plus intercept. New area fits pass independent normal-equation checks; unchanged G predictions reproduce the accepted R²/KGE/MAE scores. No feature or penalty search and no routing reruns. Completed in 1.0 seconds.

Primary reporting follows the user's choice: combined units plus full-network controls, equal weight per comparison. Training remains equal-river weighted. Metrics are calculated from complete held-out predictions per repeat and averaged across repeats. Separate populations and balanced weights remain available as diagnostics. KGE uses the original SD-ratio and mean-ratio expression, with signed responses unchanged.

```sh
/opt/anaconda3/envs/river-hierarchy-rivgraph/bin/python -u RAPID/investigations/six_area_vs_g_20261104/run.py
```

Outputs: `RAPID/outputs/six_area_vs_g_20261104/`, including `report.md`, `summary.csv`, `paired_changes.csv`, `numerical_checks.csv`, `largest_errors.csv` and `completion.json`.

## Findings

| Validation | G R² | Area R² | G KGE | Area KGE | G MAE (min) | Area MAE (min) |
|---|---:|---:|---:|---:|---:|---:|
| Compound-family grouped | 0.8666 | 0.4178 | 0.8366 | 0.6614 | 0.9397 | 1.2267 |
| Leave one network out | 0.8571 | 0.6117 | 0.8741 | 0.7555 | 1.0839 | 1.3811 |
| Network-grouped fivefold | 0.8627 | -1.0093 | 0.8595 | 0.2003 | 1.0782 | 1.6351 |

Area performs worse on all three headline metrics under all strategies. Area network-grouped R² ranges from -3.264 to 0.599 across repeats, indicating large errors in some partitions. Separate equal-river ordinary R² drops from 0.8157/0.6878/0.6737 to 0.6835/0.5051/0.0885. Full-network control R² drops from 0.7019/0.7451/0.7665 to -0.4394/0.1892/-3.9326. Thus the decline is not exclusively an artifact of combining populations, although large full-collapse errors strongly affect pooled squared-error metrics.

## Reasons to consider area, and reasons not to overstate them

Area is a simple sum of reach length × width and has a direct geometric interpretation. G instead combines length, width, allocated fractions and slope with Manning-derived exponents. Its interpretation depends on hydraulic assumptions and width-based allocation; dependence on slope can convey slope-estimation uncertainty. Its overlap with other predictors complicates component-wise causal interpretation. A useful predictor of outputs from related routing assumptions is not by itself proof of a general physical law.

These are interpretability and generalization qualifications, not a prohibition. G uses no depth, Manning n or simulated-discharge input. Provided it is computed only from geometry, slope and the prescribed allocation rule, it is not target leakage merely because its derivation is hydraulically informed. Adding area instead does not eliminate all assumptions from the model: the other predictors still use slope and allocation.

This comparison preserves existing descriptor scopes: area uses the affected reach footprint; G includes whole-network allocation changes. Consequently, it compares the current candidate definitions and cannot attribute the performance gap to any one exponent, scope choice or physical mechanism. The fixed penalty was not tuned separately for area, so this result establishes a clear advantage for G in the current matched specification, not that no area-based model could ever work.

Recommendation: retain G in the already accepted six-input model and proceed with the planned bounded penalty and residual checks. This is exploratory evidence from repeatedly inspected validation data, not independent final confirmation.
