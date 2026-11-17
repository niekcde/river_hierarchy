# RF compact reselection after the equivalent-width audit

Reference **SYN-RF-RESELECTION-2026-11-03**. Authorized after the mean-width replacement improved real-network centroid prediction. This is a new exploratory selection experiment, not independent confirmation of that replacement.

Question: does a defensible width variable emerge from synthetic selection without forcing mean width into the model?

## Frozen protocol

- Primary population: all 45,927 fixed-Manning-depth synthetic parent–child comparisons, 6,561 parameter combinations, seven states each. Primary response: centroid shift. No routing reruns, variable-depth response pooling or real-network response data.
- Restart from the existing 100 fully defined working candidate classes. Restore weak standalone predictors, including allocated entropy and dominance. Exclude `path_parent_equivalent_width`, `path_delta_equivalent_width`, `path_parent_elongation` and `path_delta_elongation` because they contain the problematic sum over overlapping complete-path widths. This leaves 96 candidates, or 95 when Manning geometric accumulation ΔGₘ is unavailable. Normalized path-width descriptors are not automatically excluded; their generalizability still requires checking.
- Mean-width ratio competes using the existing `log1p(child/parent)` representation. Domain size uses metres. No mean-width preference in selection or tie-breaking.
- Reuse the fixed RF: 500 trees, depth 16, leaf size 2, all features at each split, bootstrap, seed 20261014. Reuse the existing training-only preprocessing and equal case weights. No hyperparameter tuning or row subsampling.
- Reuse the previous three outer case-group folds and three inner case-group folds. All seven related states remain together. These are synthetic parameter-group splits, not the three different real-network validation strategies.
- Require explicit length, width, slope and topology representation. Combined descriptors do not replace that coverage. Greedy additions, two same-category coordinate-swap passes at size four, then additions up to ten. Select the smallest tested subset within 0.005 inner out-of-fold R² of the best tested path. This is not an exhaustive subset search.
- Exact early termination is allowed only if the first valid four-variable subset already reaches R² ≥ 0.995 after the swap passes. It is then within 0.005 of the mathematical upper bound of one: additional variables cannot change the selected minimum size. The longer path is not computed or claimed.
- “With ΔGₘ” means eligible, not compulsory. The two variants can legitimately return the same model.
- Outer folds are evaluated only after selecting their subset on inner folds. Save selected variables per fold, outer predictions and pooled out-of-fold scores. Selection frequency concerns stability across these three folds, not a probabilistic significance test.

## Execution and outputs

```sh
/opt/anaconda3/envs/river-hierarchy-rivgraph/bin/python synthetic_experiments/investigations/rf_reselection_20261103/run.py --run
```

Two single-threaded workers. Each subset's three inner fits are checkpointed together; identical subsets are reused across steps and the with/without-G variants. Interrupted execution can resume with the same command. The process lock prevents duplicate launches. Inputs and script are hashed before checkpoint reuse. Data are read from the existing local analysis snapshot, so this search does not need the external drive to remain mounted.

`results/design.json` freezes the protocol and sources; `candidate_ledger.csv` lists all 100 entries and exclusions. `progress.json` records progress, `path_*.json` records completed selection steps, and `selection_*.json` records completed selections. Final results are `summary.csv`, `selected_variables_by_fold.csv`, `selection_frequency.csv` and `outer_predictions.parquet`. Only `completion.json` with `complete: true` establishes completion.

## Interpretation limits

The candidate pool and prior grouping were informed by inspection of the synthetic population; the exclusion was motivated by a real-network failure. Nested scores therefore assess this frozen pool, not the independence of the entire discovery process. High synthetic R² alone does not establish performance on complex real networks. A different selected width descriptor is a valid outcome. Failure to select mean width does not invalidate its separate real-network test. Peak shift remains a secondary follow-up; no peak-selection results are implied by this centroid run.
