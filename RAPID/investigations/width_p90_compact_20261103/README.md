# Child width P90 versus mean-width ratio

Reference **WIDTH-P90-COMPACT-2026-11-03**. Targeted follow-up to the matched synthetic ridge reselection. The aim is to isolate the width representation, not to repeat feature selection.

Both panels contain five inputs:

1. Change in allocated mean route length.
2. Parent slope descriptor.
3. Either `log1p(B_ratio_width_mean)` or `B_child_width_p90`.
4. Change in topologic complexity score.
5. Either Manning geometric accumulation change ΔGₘ or the area-proxy change.

Child P90 is the length-weighted width quantile in metres: the smallest width at which cumulative reach length, sorted by width, reaches 90% of selected-domain length. Mean width is the length-weighted arithmetic mean, compared as child/parent and transformed with log1p. Neither descriptor enumerates complete routes.

Use all 45,927 fixed-depth Manning synthetic comparisons and all 2,185 real comparisons. Preserve the previous 500-tree RF, preprocessing, seed, response, training weights and splits. Three real validation strategies: compound-family grouped, leave-one-network-out, and network-grouped fivefold. Report 2,170 ordinary comparisons separately from the 15 full-network controls. Reuse verified completed mean-width predictions; perform only 136 new P90 fits, with two workers.

Ordinary P90 values come from the earlier matched closed-domain geometry table, with shared descriptors and response checked against the current rows. For full-network controls, recompute P90 on collapsed-block child reaches, excluding unchanged connectors, to match the scope of the existing mean-width input. This scope correction is recorded in `full_network_scope_audit.csv`.

Run:

```sh
/opt/anaconda3/envs/river-hierarchy-rivgraph/bin/python -u RAPID/investigations/width_p90_compact_20261103/run.py
```

Outputs: `RAPID/outputs/width_p90_compact_20261103/`. `completion.json` establishes completion. `report.md`, `summary.csv` and `comparison.png` show the main results; `paired_changes.csv` preserves same-repeat differences; `per_river_errors.csv` checks concentration of error in particular rivers. Source and prepared-input hashes are frozen in `design.json`; completed fits are resumable.

This is an exploratory, post-hoc comparison using previously inspected real validation data. It does not independently validate the original feature-discovery process. It does not test the full four-input ridge-selected panel: all four non-width inputs remain fixed deliberately.

## Completed findings

All 136 fits completed in 259 seconds (4.3 minutes). Same train/test partitions verified for every paired fit; source hashes checked at completion.

For the 2,170 ordinary comparisons, mean-width → P90 R²:

| Validation | With ΔGₘ | Without ΔGₘ (area instead) |
|---|---:|---:|
| Compound-family grouped | 0.7573 → 0.7638 | 0.6892 → 0.6916 |
| Leave one network out | 0.7062 → 0.7146 | 0.5844 → 0.5762 |
| Network-grouped fivefold | 0.6983 → 0.7065 | 0.5414 → 0.5388 |

P90 plus G has lower average absolute error in all three strategies, but improves individual-river MAE for only 8/15, 8/15 and 7/15 rivers respectively. Network-grouped repeated-fold R² differences range from −0.0125 to +0.0226. Thus its modest average advantage is not a uniform or established statistically significant improvement.

For the 15 full-network controls, P90 plus G is slightly worse: R² 0.3452/0.3911/0.3996 versus mean-width 0.3630/0.4142/0.4175. Synthetic R² is effectively tied (about 0.99999 with G, 0.99998 without). The synthetic-selected width representation is competitive on ordinary real cases; the exact width representation is much less consequential than eliminating the problematic summed equivalent-width descriptor. This test preserves the five-input panels and does not establish that the full four-input ridge-selected set has the same performance.
