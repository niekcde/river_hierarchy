# Mean-width replacement — 3 November 2026

User-authorized follow-up to the example-37 diagnostic. Replace only the
equivalent path-width predictor in each compact five-input panel by the
previous nine-input panel's mean-width-ratio representation:

\[
\bar B=\frac{\sum_e L_e B_e}{\sum_e L_e},\qquad
x_B=\log\left(1+\frac{\bar B_c}{\bar B_p}\right).
\]

Each reach enters the selected domain's moments once. Both tested panels retain
five inputs: mean route-length change, parent slope descriptor, this mean-width
ratio, topology-complexity change, and either Manning geometry ΔG or area change.

Use the existing fixed-depth Manning centroid targets, 500-tree squared-error
RF, seed, training-only preprocessing and group weights. The exact previous
outer splits are reused: three synthetic case-group folds, 15 real leave-river-
out folds, 25 network-grouped folds and 25 compound-family-grouped folds.
This gives 136 new fits on two workers. The previously saved compact and
nine-input predictions are verified and reused as references.

All 45,927 synthetic comparisons and 2,185 real comparisons remain included.
The original nine's synthetic score is not shown on full support because its
width-CV ratio is undefined in part of that population; use revised nine for
the full synthetic comparison and the earlier common-support audit for old nine.

```bash
MPLCONFIGDIR=/tmp/mean_width_mpl \
  /opt/anaconda3/envs/river-hierarchy-rivgraph/bin/python \
  RAPID/investigations/mean_width_compact_20261103/run.py
```

Outputs: `RAPID/outputs/mean_width_compact_20261103/`. The report and comparison
figure are generated automatically after fitting. Check `completion.json` for
completion. The same command resumes hashed completed checkpoints.

This is an exploratory post-hoc replacement prompted by example-37 errors.
Evaluate all three validation strategies and per-river error contributions
before making a recommendation. Reused holdouts do not provide independent
confirmation of the feature-discovery procedure. No routing or tuning search
is included. Synthetic full-pulse and real finite-window centroid definitions
remain distinct, as documented in the preceding comparison.
