# Expanded RAPID response-prediction grid

The grid-screening script is `search_rapid_response_predictors.py`; the
companion execution notebook is
`RAPID/notebooks/rapid_response_prediction_expanded_grid.ipynb`.

## Targets

The script fits separate models for two responses. It does not combine them.

- `lag_to_inflow_peak_pct_shift_vs_base = 100 * (lag_candidate / lag_base - 1)`,
  where `lag` is outlet-peak time minus inflow-peak time.
- `peak_discharge_pct_shift_vs_base = 100 * (Qpeak_candidate / Qpeak_base - 1)`.

The base value is `S000_base` for the same example and collapse mode.

## Predictors

The candidate set has five resistance/footprint metrics and seven geometry/
topology metrics. All one-, two-, and three-predictor subsets are evaluated,
as well as the full 12-predictor set.

Topology values use the base-state hierarchy. Independent collapse rows use
the matching unit directly. Compound bubble rows aggregate member-unit values
with `equivalent_width` weights:

`bubble metric = sum(member metric * member equivalent_width) / sum(member equivalent_width)`.

This preserves the contribution of larger effective flow paths. The exact
method per experiment is recorded in `topology_metric_provenance.csv` for
each run.

## Evaluation and interpretation

Each candidate is evaluated with leave-one-example-out cross-validation. The
screen ranking uses MAE in percentage points, then fewer predictors and RMSE
as tie breakers. It is a model-selection screen: use unseen examples or
nested grouped CV before reporting its selected score as final performance.
