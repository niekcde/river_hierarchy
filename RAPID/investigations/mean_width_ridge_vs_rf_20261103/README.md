# Same five predictors: quadratic ridge versus RF on real cases

Reference **MEAN-WIDTH-RIDGE-VS-RF-2026-11-03**. User requested model-family comparison for the successful compact panel after replacing summed equivalent path width with mean-width ratio.

The five engineered inputs are change in allocated mean route length, parent slope descriptor, `log1p(child/parent length-weighted mean width)`, change in topologic complexity score, and whole-network Manning geometric accumulation change ΔGₘ. Both estimators receive exactly these columns.

Quadratic ridge follows the original synthetic model family and penalty: five main effects, five squares, ten pair products, unpenalized intercept; expanded terms standardized using training-only weighted moments; alpha = 1e-4 times normalized training weight sum. Thus five descriptors yield twenty penalized terms. The original synthetic pool used raw width ratio; here log1p is retained for both estimators to match the existing successful RF panel. No real-data penalty search is performed. Equal river training weights match RF; independent normal-equation checks validate every ridge fold.

Use the same 65 real validation tasks, all 2,185 comparisons and targets as the completed mean-width RF run. RF predictions are reused after hash verification. Each ridge is trained on its real training fold: this is not synthetic-to-real coefficient transfer. Ordinary and full-network cohorts are scored separately, with existing weighted scoring.

```sh
/opt/anaconda3/envs/river-hierarchy-rivgraph/bin/python -u RAPID/investigations/mean_width_ridge_vs_rf_20261103/run.py
```

Completed results in `RAPID/outputs/mean_width_ridge_vs_rf_20261103/`, including `report.md`, `summary.csv`, `repeat_scores.csv`, `per_river_errors.csv`, `numerical_checks.csv` and `completion.json`.

## Findings

| Validation | Ordinary ridge R² | Ordinary RF R² | Full-network ridge R² | Full-network RF R² |
|---|---:|---:|---:|---:|
| Compound-family grouped | 0.7299 | 0.7573 | 0.8374 | 0.3630 |
| Leave one network out | 0.5464 | 0.7062 | 0.8571 | 0.4142 |
| Network-grouped fivefold | 0.4960 | 0.6983 | 0.7915 | 0.4175 |

Ordinary MAE is 0.938/1.057/1.076 minutes for ridge versus 0.566/0.661/0.669 for RF. Full-network MAE is 23.16/23.45/29.50 minutes for ridge versus 46.68/45.93/45.43 for RF. There are only 15 full-network controls; this strong difference warrants reporting but is not an independently confirmed general advantage.

RF is stronger for ordinary comparisons, especially entire-network holdouts. Ridge is much stronger for the large full-network controls. The pooled cohort-balanced R² therefore favors ridge, but must not replace the separate results. This comparison does not identify why the estimators differ: extrapolation versus RF averaging is a possible mechanism requiring direct diagnostics. It is not a test of an optimized ridge penalty or of a simple five-coefficient linear regression.

All results remain exploratory on repeatedly inspected real validation data. The width replacement was motivated by real-network failures, so do not describe this as fully independent synthetic-only discovery.
