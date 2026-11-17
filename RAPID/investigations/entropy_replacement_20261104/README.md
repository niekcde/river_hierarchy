# Replace topological complexity change by allocated route-entropy change

Reference **ENTROPY-REPLACEMENT-2026-11-04**. User requested this replacement after reviewing the five-input set without Manning G. Scope is that no-G panel; G is not silently reintroduced.

Keep allocated mean route-length change, parent slope descriptor, log1p(child/parent length-weighted mean width), and planform-area-proxy change fixed. Replace only `path_delta_topologic_complexity_score` with the existing `delta_entropy` column. Entropy uses width-based allocation and natural logarithms; for a closed domain, H = −sum_r π_r ln(π_r), with ΔH = H_child − H_parent. Existing edited-footprint allocation conventions are preserved for full-network controls.

Fit both quadratic ridge and RF on identical real-network folds. RF parameters, preprocessing, seed and equal-river training weights are unchanged. Ridge uses five main effects, five squares and ten interactions, training-only weighted standardization, an unpenalized intercept and fixed penalty 1e-4 times normalized training-weight sum. Independent normal-equation checks validate each ridge fit.

Three validation strategies: compound-family grouped, leave-one-network-out, and network-grouped fivefold. All 2,185 comparisons retained; ordinary n=2,170 and full-network controls n=15 scored separately. Reuse completed complexity RF predictions. New fits: 65 complexity ridge + 65 entropy ridge + 65 entropy RF. Ridge fits are inexpensive; only 65 new forests are needed. Two workers, no routing runs or feature search.

```sh
/opt/anaconda3/envs/river-hierarchy-rivgraph/bin/python -u RAPID/investigations/entropy_replacement_20261104/run.py
```

Results: `RAPID/outputs/entropy_replacement_20261104/`, including frozen `design.json`, `summary.csv`, `paired_changes.csv`, `per_river_errors.csv`, `largest_errors.csv`, and `report.md`. Completion is established by `completion.json`. No results are independently uninspected: this is an exploratory predictor-replacement comparison, not proof of a general physical relationship.

## Completed findings

All 195 fits completed in 110.7 seconds: 130 ridge fits and 65 new RF fits; completed complexity RF results reused. Input snapshots and all 65 partitions were verified identical across panels. Each ridge passed an independent normal-equation prediction check; all source hashes verified at completion.

Ordinary real cases (2,170), centroid-shift R²:

| Validation | Complexity ridge | Entropy ridge | Complexity RF | Entropy RF |
|---|---:|---:|---:|---:|
| Compound-family grouped | 0.7004 | 0.7260 | 0.6892 | 0.7243 |
| Leave one network out | 0.2866 | 0.4670 | 0.5844 | 0.6153 |
| Network-grouped fivefold | -0.3906 | -0.2220 | 0.5414 | 0.5758 |

Entropy improves the mean ordinary R² and MAE for both estimators under all three strategies. RF MAE changes only slightly (0.6556→0.6508, 0.7259→0.7217, 0.7530→0.7462 minutes), so avoid describing this as a large general improvement. Ridge MAE improves from 0.9295/1.0830/1.1336 to 0.8220/0.9755/1.0601 minutes, but large errors remain. Ordinary network-grouped entropy-ridge R² varies from -1.253 to 0.339 across repeats: its negative average is not a transcription error.

The 15 full-network controls show no uniformly favorable entropy effect. Entropy ridge R² is -0.0885/0.2844/-6.7171 versus complexity ridge 0.2509/-3.0413/-11.2521. Entropy RF is 0.3465/0.3863/0.3899 versus complexity RF 0.3486/0.3818/0.3815. Large full-network ridge errors include example 22, which can be predicted with timing changes of either sign under different network-grouped repeats. See largest_errors.csv; these are not grounds for removing cases.

Interpretation: entropy is a promising replacement for ordinary predictions, but does not make the fixed-penalty no-G quadratic ridge robust to held-out-network variation. The now-available no-G ridge baseline is substantially weaker than the previously tested complexity+G ridge on held-out networks/full-network controls. Do not claim that omitting G preserves ridge performance. Penalty sensitivity and extrapolation diagnostics remain pending; entropy+G has not been tested here.
