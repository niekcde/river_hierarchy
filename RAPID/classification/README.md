# RAPID collapse-impact classifier

`train_rapid_impact_classifier.py` is an additive, end-to-end analysis for
classifying the expected timing impact of a candidate river-network collapse
from its geometric / equivalent-resistance change.  It reads the existing
real-network experiment outputs but never changes them.

Run it from the repository root with the environment that contains the RAPID
analysis dependencies (for example, the local `UNC` conda environment):

```bash
PYTHONPATH=RAPID/src /opt/anaconda3/envs/UNC/bin/python \
  RAPID/classification/train_rapid_impact_classifier.py \
  --output-root /Volumes/PhD/river_hierarchy/network_variants/output
```

To select a different real-network cohort, add `--example-ids` followed by
space-separated IDs. The script requires at least five usable example IDs
after the existing boundary/completeness/edge filters:

```bash
PYTHONPATH=RAPID/src /opt/anaconda3/envs/UNC/bin/python \
  RAPID/classification/train_rapid_impact_classifier.py \
  --output-root /Volumes/PhD/river_hierarchy/network_variants/output \
  --example-ids 02 03 05 11 21
```

Outputs are written to a new UTC-stamped directory below
`RAPID/classification/results/` by default. This directory is intentionally
separate from the read-only experiment outputs; use `--results-dir` to choose
another new, versioned location.

The default target is the absolute value of
`lag_to_inflow_peak_pct_shift_vs_base`.  The ordered classes are
`no_impact` (<0.5%), `intermediate_impact` (0.5% to <1%), and `high_impact`
(>=1%).  These thresholds are explicit named configuration values in the
script and a 0.25%/1% sensitivity audit is included in every report.

## Continuous RAPID-response prediction

`train_rapid_response_regressor.py` predicts the signed continuous value of
`lag_to_inflow_peak_pct_shift_vs_base` rather than a class. It uses the same
real-network filters and leave-one-example-out validation as the classifier:

```bash
PYTHONPATH=RAPID/src /opt/anaconda3/envs/UNC/bin/python \
  RAPID/classification/train_rapid_response_regressor.py \
  --output-root /Volumes/PhD/river_hierarchy/network_variants/output \
  --example-ids 02 03 05 11 21
```

The two companion notebooks are
`RAPID/notebooks/rapid_response_prediction_investigation.ipynb` and
`RAPID/notebooks/rapid_impact_classifier_investigation.ipynb`. Each creates
its own timestamped result directory and then visualizes grouped-CV results.

## Response-predictor feature/model grid

`search_rapid_response_predictors.py` evaluates every one-, two-, and
three-variable subset (plus the full set) of the current five resistance /
footprint candidates against ridge, elastic-net, shallow-tree, random-forest,
extra-trees, and histogram-gradient-boosting settings. All combinations use
the same complete-case, leave-one-example-out cohort:

```bash
/opt/anaconda3/envs/UNC/bin/python \
  RAPID/classification/search_rapid_response_predictors.py \
  --output-root /Volumes/PhD/river_hierarchy/network_variants/output \
  --example-ids 02 03 05 11 13 15 21 56
```

Treat the resulting ranking as a screening result: selecting the lowest score
from a large grid is optimistic. Confirm a chosen configuration with nested
grouped CV or new examples before freezing it.

## Geometry-only clustering

`train_rapid_geometry_clustering.py` compares k-means, Gaussian-mixture,
agglomerative, spectral, DBSCAN, and OPTICS clustering on geometry/resistance
features only. It reports RAPID timing response only after clusters have been
formed. The companion notebook is
`RAPID/notebooks/rapid_geometry_clustering_investigation.ipynb`.
For the review-oriented version that does not choose a winner automatically,
use `RAPID/notebooks/rapid_geometry_clustering_review.ipynb`.
