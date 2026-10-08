# Refreshed real geometry benchmark — 5 October 2026

Reference: REAL-GEOMETRY-2026-10-05. Requested to repeat the synthetic geometric
checks on the completed real-network runs before interpreting transfer.

## Review entry points

- `RAPID/notebooks/rapid_hydrograph_review.ipynb`: all 16 observed gauge overlays,
  optional model visibility, normalized/absolute discharge choice, and selected
  collapse/domain inspection. The refresh used a 2020 origin for 13 cases;
  actual UTC is restored only after matching the original forcing arrays.
  The other three retain recorded gauge-event UTC. No peak alignment is applied.
- `RAPID/notebooks/rapid_hydrograph_analysis.ipynb`: final sections contain
  predictor scores, correlations, per-river scores, synthetic transfer and the
  full-cohort scalar/component control.

## Responses and cohorts

The adapter now exposes interpolated peak-time differences as a separate
sensitivity, preserving the old sampled timing columns. Three-point local
quadratic interpolation refuses boundary/tied/nonconcave/out-of-bin estimates.
All 3,517 nontechnical comparison rows passed these guards and no longer have
exactly zero interpolated timing changes. This is model interpolation, not
proof that all changes are physically measurable or numerically resolved.
The targeted native-output/half-clock verification is documented in the sibling
`peak_resolution_20261005` directory. Peak excess was corrected to subtract the
same parent max(initial, final) plateau from parent and child maxima.

There are **2,214 unique non-base geometry/domain comparisons** after alias
deduplication, covering all 16 rivers. Scalar resistance, length fraction,
exposure and omega can be tested on all of them. The exact same path-based
geometry definitions used in synthetics require path enumeration. A hard
10,000-path bound excludes 883 unique comparisons (1,424 original rows) in
15/22/37, leaving **1,331 unique comparisons** for the extended feature sets.
Exclusions are explicit in excluded_geometry.csv. This is not a random subset,
and its scores must not be generalized to the full population. Width dominance
and elongation based on enumerated paths also reuse shared links; they are
algebraic graph descriptors, not independent physical-channel counts.

## Fixed evaluation

No real-data feature selection, hyperparameter tuning or per-river calibration.
Fixed standardized Ridge and shallow boosted regressors, matching the synthetic
estimator settings. All real fits hold out entire rivers. Scores include pooled,
per-river and category R² and MAE; the training-mean baseline is separate. Real
responses are simulated collapse effects, not observed effects of river removal.
Synthetic transfer fits the unchanged original development rows and evaluates
real cases without refitting. Synthetic scores use the original holdout; pooled
scores here differ from the older treatment-averaged summaries. Interpolated
real-lag transfer is explicitly a sensitivity to sampled synthetic-lag training.

## Initial findings

For the **full cohort**, boosted omega gives leave-river-out R² of .113 for
sampled lag and .102 for interpolated lag; duration is -.030 and peak excess
.002. The separated components give .013 for interpolated lag, .021 duration
and .093 excess. Thus the current scalar relationship is not a strong universal
predictor across all real domains.

For the **path-enumerable common cohort**, omega's interpolated-lag R² is .645.
The selected six plus changes/surroundings gives .663, but duration/excess only
.067/.065. The selected six alone gives .352 for interpolated lag. This cohort
contrast is a finding, not a reason to drop the difficult domains.

Frozen synthetic-to-real models perform poorly (negative R²); selected-six
original-grid pooled holdout R² is .964/.766/.774, but real transfer is strongly
negative. The real domain/network length fraction is below the original
synthetic training range in 98.8% of the common cohort. Path dominance and
elongation also show substantial distribution differences. Feature-range
mismatch is demonstrated; its causal share in transfer failure is not yet
isolated. Different event forcing and real network complexity remain relevant.

## Interpretation and next decision

A suitable claim is an uncalibrated, controlled study of routing responses to
geometric simplification under a common hydraulic policy. Absolute agreement
is improved, but not established as uniformly acceptable. The agreement and
parameter/forcing sensitivity of paired responses must be shown before claiming
real-world magnitudes. Agreement of a learned model with RAPID output does not
independently validate RAPID physics or the unobserved collapsed river.

Next examine high-leverage rivers/domains, compare the wider synthetic embedding
extension, audit path descriptors in compound networks, and assess forcing and
hydraulic robustness. Keep the original evaluation fixed; any revised predictor
is a new exploratory candidate requiring a new validation design.

Reproduce in the repository environment:

```
python RAPID/investigations/refreshed_predictor_validation.py
python RAPID/investigations/refreshed_scalar_validation.py
```

`--reuse-analysis` is an explicit diagnostic restart option for an already
refreshed analysis.parquet; the default rebuilds from production. Production
routing files were not changed. Both notebooks executed successfully, and 13
focused helper/regression tests passed.
