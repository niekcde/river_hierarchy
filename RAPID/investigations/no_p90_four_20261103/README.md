# Real-network RF evaluation of the exact no-P90 four-input set

Reference **NO-P90-FOUR-2026-11-03**. Evaluate the four inputs selected in every fixed-Manning centroid coverage fold after excluding P90:

- `B_delta_route_length_mean`: change in allocated mean route length.
- `parent_slope_slow`: parent slope descriptor.
- `B_parent_width_mean`: absolute length-weighted arithmetic mean parent width, metres.
- `B_parent_bifurcations`: parent bifurcation count.

There is no ΔG or area input. Do not interpret differences from previous five-input models as an isolated width test: topology representation and input count also differ.

Keep the existing 500-tree RF, preprocessing, random seed, targets, training weights and all 2,185 real comparisons. Use exactly the saved compound-family grouped, leave-one-network-out and network-grouped fivefold splits. Report 2,170 ordinary cases separately from 15 full-network controls. Perform 65 new fits with two workers and reuse completed baseline predictions. No tuning, feature search or routing reruns.

Parent mean width and bifurcation count are computed from the existing closed parent domains for ordinary cases, and from selected collapsed-block parent reaches for full-network controls. The latter exclude unchanged connectors. Verify ordinary bifurcation counts against the historical geometry table and ensure edited masks do not cut through any split. Retain a row-level geometry audit and source hashes.

```sh
/opt/anaconda3/envs/river-hierarchy-rivgraph/bin/python -u RAPID/investigations/no_p90_four_20261103/run.py
```

Outputs are in `RAPID/outputs/no_p90_four_20261103/`: `summary.csv`, `report.md`, `per_river_errors.csv`, `geometry_audit.csv`, frozen `design.json`, and resumable checkpoints. Only `completion.json` establishes completion.

This remains exploratory assessment on repeatedly inspected real validation data, not an independent final confirmation of predictor selection.

## Completed result

Completed all 65 fits in 92.6 seconds; all saved train/test partitions match the previous comparison exactly. Source hashes verified at completion.

Ordinary real comparisons (2,170):

| Validation | New four R² | New four MAE (min) | Old nine R² | Mean-width ratio five + G R² |
|---|---:|---:|---:|---:|
| Compound-family grouped | 0.5361 | 0.8383 | 0.6436 | 0.7573 |
| Leave one network out | 0.5593 | 0.7583 | 0.5753 | 0.7062 |
| Network-grouped fivefold | 0.5459 | 0.8768 | 0.5844 | 0.6983 |

The 15 full-network controls have R² 0.2827/0.3394/0.3619 and MAE 46.79/46.26/44.31 minutes in the same order. They are not combined with ordinary scores in the main table.

The exact synthetic-selected four-input panel is weaker on real cases than the previous five-with-G panels. This cannot be attributed to excluding P90 alone: the panel also omits G/area and changes topology from a complexity difference to parent bifurcation count. The synthetic 0.005 parsimony threshold accepts a four-input approximation; its near-perfect synthetic score does not establish real-network sufficiency. A separate four-plus-G test could isolate the G omission, but is not part of this run.
