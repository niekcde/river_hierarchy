# Matched centroid predictor test — 3 November 2026

Reference: **MATCHED-CENTROID-SETS-2026-11-03**.

Requested to distinguish a predictor-set change from the change between the
existing notebook's compound-family validation and the recent whole-river
holdout test. This is a real-data refit, not synthetic-to-real model transfer.

Compare `geometry_normalized_9` from
`RAPID/notebooks/hydrography_rf_prediction_vs_modelled.ipynb` with the two
synthetic-selected five-input centroid sets. Keep all 2,185 comparisons, the
historical 25 outer partitions and their inner splits, six RF configurations,
weights, preprocessing, target scaling, tuning objective and scoring.

The previous nine-input predictions are reused only after an exact first-fold
refit check. New results are isolated from production. Report ordinary and
full-network cohorts separately; pooled R² can obscure ordinary-case failure.

```bash
/opt/anaconda3/envs/river-hierarchy-rivgraph/bin/python \
  RAPID/investigations/matched_centroid_sets_20261103/compare.py --run --workers 2
/opt/anaconda3/envs/river-hierarchy-rivgraph/bin/python \
  RAPID/investigations/matched_centroid_sets_20261103/report.py
```

Raw outputs: `RAPID/outputs/matched_centroid_sets_20261103/`. Inspect `progress.json`
for actual completion. `report.py` requires a completion receipt. The run is
checkpointed and the same command resumes completed inner/outer fits.

The raw-width check found that standardization followed by RF float32 conversion
collapses hundreds of ordinary-domain equivalent widths to one or two distinct
values. The primary comparison therefore uses `log1p(equivalent width)` before
the same training-only scaler. This is an order-preserving numerical correction,
chosen using predictors alone, not outcome performance. The raw-width first
five-fold repeat is retained as a diagnostic; it is not the full headline test.

Run/resume the corrected comparison and automatically produce its report:

```bash
/opt/anaconda3/envs/river-hierarchy-rivgraph/bin/python \
  RAPID/investigations/matched_centroid_sets_20261103/safe_width.py
```

Corrected outputs: `RAPID/outputs/matched_centroid_sets_20261103_logwidth/`.

## All three validation approaches

The subsequent request extends the matched comparison to whole-river holdout
and network-grouped five-fold validation, alongside compound-family grouping.
Run/resume:

```bash
/opt/anaconda3/envs/river-hierarchy-rivgraph/bin/python \
  RAPID/investigations/matched_centroid_sets_20261103/all_strategies.py
```

This waits for the existing compound run, reuses its predictions and the saved
normalized-nine baseline, then computes 120 missing outer jobs on two workers.
The baseline is **geometry_normalized_9** in every strategy; the earlier
**geometry_9** results are not silently substituted for it. Identical historical
outer/inner splits, the same six-candidate tuning grid and the same populations
are used. No additional candidate features are introduced.

The live comparison report is
`RAPID/outputs/matched_centroid_all_strategies_20261103/report.md`.
It only compares repetitions completed for **all three predictor sets**, and
reports full-compound collapses separately from compound members and the other
edit types. Grouping by compound families is not the same as evaluating only
full-compound collapses. The report explicitly marks incomplete comparisons;
`completion.json` is the final completion record.

Full-network feature alignment: unchanged connectors are excluded from edge
length/area/width rewards, consistent with the historical benchmark. Connectivity
and topology remain defined on the complete graph. Whole-network ΔGₘ remains
whole-network by its frozen definition. Ordinary domains are unchanged.

This test estimates within-river, held-out-family performance. It cannot erase
or replace the previous whole-river holdout findings. The inherited partitions
and feature selection are exploratory; repeated folds are not independent
replications. Full-network singleton controls may share constituent units with
training cases.
