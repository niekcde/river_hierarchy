# Add ΔGₘ to the exact no-P90 four-input set

Reference **NO-P90-FOUR-PLUS-G-2026-11-03**. User-authorized incremental test after the exact four-input synthetic selection gave real ordinary R² of 0.536/0.559/0.546.

Inputs, in fixed order:

1. Change in allocated mean route length.
2. Parent slope descriptor.
3. Absolute length-weighted mean parent width (metres).
4. Parent bifurcation count.
5. **Added:** whole-network Manning geometric accumulation change ΔGₘ.

Only the fifth input is added. The prepared real input file is copied byte-for-byte from the completed four-input experiment. All 2,185 comparisons, responses, geometry scopes, feature order of the original four, fixed 500-tree RF, preprocessing, seed, weights and validation partitions remain unchanged. No hyperparameter search, synthetic selection or routing reruns. Reuse completed baseline predictions; perform 65 new fits with two workers.

Validation: compound-family grouped, leave-one-network-out and network-grouped fivefold. Report the 2,170 ordinary comparisons and 15 full-network controls separately. `paired_changes.csv` records repeat-matched differences against the exact four-input model. Differences from other five-input panels also involve width/topology representations and are not isolated G effects.

```sh
/opt/anaconda3/envs/river-hierarchy-rivgraph/bin/python -u RAPID/investigations/no_p90_four_plus_g_20261103/run.py
```

Outputs: `RAPID/outputs/no_p90_four_plus_g_20261103/`, including `summary.csv`, `report.md`, `paired_changes.csv`, `per_river_errors.csv`, frozen `design.json` and resumable checkpoints. `completion.json` establishes completion.

This is exploratory incremental predictive value on repeatedly inspected validation data, not independent final validation or causal attribution to G.

## Completed findings

All 65 fits completed in 89.9 seconds. Input file is byte-identical to the four-input run; all 65 saved partitions match exactly. Source hashes verified at completion.

| Ordinary real validation | Four R² | Four + G R² | Four MAE (min) | Four + G MAE (min) |
|---|---:|---:|---:|---:|
| Compound-family grouped | 0.5361 | 0.6034 | 0.8383 | 0.6610 |
| Leave one network out | 0.5593 | 0.5884 | 0.7583 | 0.6650 |
| Network-grouped fivefold | 0.5459 | 0.5635 | 0.8768 | 0.7794 |

The addition improves average R² and MAE in all three strategies, but does not recover the previous mean-width-ratio/complexity-change/G panel (R² 0.7573/0.7062/0.6983). The remaining two predictor differences are absolute parent mean width versus child/parent mean-width ratio, and parent bifurcation count versus change in topologic complexity score. This experiment cannot assign the residual gap to either one. The real data remain repeatedly inspected; these are exploratory model comparisons.

Full-network controls improve slightly: R² 0.2936/0.3406/0.3672 versus 0.2827/0.3394/0.3619. Their MAE remains about 43–46 minutes, unlike sub-minute ordinary MAE; keep these populations separate.
