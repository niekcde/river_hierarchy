# Five-input quadratic ridge with entropy and Manning geometry G

Reference **ENTROPY-G-RIDGE-2026-11-04**. User-authorized test of the proposed five-predictor main-model candidate.

Inputs: allocated mean route-length change, parent slope descriptor, log1p(child/parent length-weighted mean width), allocated route-entropy change, and whole-network geometry-only Manning descriptor ΔGₘ. No depth, roughness or simulated-discharge inputs. The G derivation retains Manning assumptions.

Same quadratic ridge specification as previous tests: five main effects, five squares, ten pair products and an unpenalized intercept. Training-only weighted standardization; alpha = 1e-4 times normalized training-weight sum. Same equal-river weights, all 2,185 real comparisons and 65 saved fits over three validation strategies. No hyperparameter tuning or routing reruns. All 65 numerical solutions checked against independent normal equations.

Comparators reused from completed predictions:

- Complexity + G ridge: changes only the topology descriptor.
- Entropy + area ridge: changes only the fifth predictor.
- Complexity + G RF: benchmark with a different topology descriptor and estimator, not entropy + G RF.

Exact row keys and target values match each baseline for every validation repeat. Source hashes verified at completion. Completed in 1.7 seconds.

```sh
/opt/anaconda3/envs/river-hierarchy-rivgraph/bin/python -u RAPID/investigations/entropy_g_ridge_20261104/run.py
```

Results in `RAPID/outputs/entropy_g_ridge_20261104/`: `report.md`, `summary.csv`, `paired_changes.csv`, `per_river_errors.csv`, `numerical_checks.csv` and `completion.json`.

## Findings

Ordinary comparisons, n=2,170, centroid-shift R²:

| Validation | Entropy + G ridge | Complexity + G ridge | Entropy + area ridge | Complexity + G RF |
|---|---:|---:|---:|---:|
| Compound-family grouped | 0.7732 | 0.7299 | 0.7260 | 0.7573 |
| Leave one network out | 0.6243 | 0.5464 | 0.4670 | 0.7062 |
| Network-grouped fivefold | 0.5651 | 0.4960 | -0.2220 | 0.6983 |

The new ridge improves ordinary R² versus complexity + G in every matched repeat. Ordinary MAE is 0.8000/0.9239/0.9613 minutes; the existing complexity + G RF has lower MAE (0.5661/0.6608/0.6694), including the strategy where its R² is lower. The no-G ridge's negative network-grouped scores are not present in this G panel: repeat R² range 0.4225–0.7051, which still shows substantial split sensitivity.

Full-network controls, n=15, have R² 0.5260/0.5072/0.4346 and MAE 40.36/43.22/45.60 minutes. This is substantially worse than complexity + G ridge R² 0.8374/0.8571/0.7915 (MAE 23.16/23.45/29.50). Thus entropy replaces rather than universally improves the topology representation: it improves ordinary prediction while losing much of the strong full-network ridge result.

This is a plausible compact candidate, not a uniformly best final model. Penalty sensitivity and residual/extrapolation checks remain pending. Results use repeatedly inspected real validation data, not an independent final validation. No entropy + G RF has been fitted here.
