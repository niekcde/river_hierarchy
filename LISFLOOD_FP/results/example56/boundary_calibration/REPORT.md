# Example 56: outlet scenarios, roughness calibration and three collapses

Completed four parent-only calibrations and twelve collapse runs. Widths and
proxy-bed slopes are fixed. Roughness is re-estimated for each outlet scenario
on the original network only, then transferred without refitting to its children.
This is the custom diffusive-wave prototype, not the LISFLOOD-FP executable.

## Calibration results

| Scenario | Fitted n | Peak error (min) | Peak (m³/s) | RMSE (m³/s) | NSE |
| --- | --- | --- | --- | --- | --- |
| A_normal | 0.024854 | -0.16 | 1851.46 | 70.38 | 0.9558 |
| B_half_slope | 0.023940 | +0.01 | 1851.32 | 70.13 | 0.9561 |
| C_double_slope | 0.025352 | +0.06 | 1851.46 | 70.51 | 0.9557 |
| D_extend10km | 0.024814 | -0.10 | 1851.24 | 70.33 | 0.9559 |


The objective follows the earlier reviewed peak approach: target
**27 June 2025, 14:40 UTC**, while retaining the prior soft peak-height interval
**1,800–1,900 m³/s**. Five roughness probes span n=0.012–0.060, followed by a
bracketed timing-root search. Full-resolution verification requires peak error
within five minutes. Peak times use a local quadratic interpolation of 5-minute
output; sub-minute values are numerical estimates, not observational precision.

RMSE, bias and NSE are diagnostics over **24 June 06:09:30–13 July 00:00:30 UTC**,
at common 5-minute timestamps. They were not the optimization objective.
Downstream gaps over 15 minutes are excluded from these diagnostics; upstream
gaps are linearly interpolated (maximum 65 minutes).
No downstream discharge is imposed as a boundary and no observed stage is used.

The raw downstream peak is **1934.85 m³/s**, versus
**1862.30 m³/s** upstream. The soft height interval is
the prior reviewed calibration target, not the raw observed peak. A timing fit
therefore must not be described as reproducing the raw peak amplitude. Missing
lateral inflow, gauge differences and model/input assumptions remain possible
explanations; this experiment does not identify which is responsible.

Among these timing-calibrated candidates, B_half_slope has the lowest in-sample
whole-window RMSE (70.13 m³/s). This is not independent validation
or proof that its boundary is physically correct. Fitting n separately can
compensate for outlet assumptions, so assess the full range and collapse stability.

**Recommended working choice:** Retain A (normal flow) as the simplest baseline, with B/C/D as sensitivity cases. The tested longer extension changes peak timing by less than one minute at the original port. The whole-window RMSE spread
across these timing-calibrated scenarios is only **0.382 m³/s**. This event
does not strongly distinguish them. No separate event has been held out for validation.

![Hydrographs](calibrated_hydrographs.png)

## Collapse responses

| Scenario | Collapse | Peak shift (min) | Centroid shift (min) | Peak change (m³/s) |
| --- | --- | --- | --- | --- |
| A_normal | S014_unit_10 | -20.36 | -23.27 | +0.257 |
| A_normal | S008_unit_6 | -3.13 | -3.63 | +0.037 |
| A_normal | S012_unit_12 | -11.30 | -13.76 | +0.163 |
| B_half_slope | S014_unit_10 | -19.80 | -22.68 | +0.238 |
| B_half_slope | S008_unit_6 | -3.06 | -3.54 | +0.031 |
| B_half_slope | S012_unit_12 | -11.05 | -13.49 | +0.154 |
| C_double_slope | S014_unit_10 | -20.68 | -23.59 | +0.268 |
| C_double_slope | S008_unit_6 | -3.18 | -3.68 | +0.039 |
| C_double_slope | S012_unit_12 | -11.46 | -13.92 | +0.169 |
| D_extend10km | S014_unit_10 | -20.28 | -23.24 | +0.259 |
| D_extend10km | S008_unit_6 | -3.11 | -3.63 | +0.039 |
| D_extend10km | S012_unit_12 | -11.25 | -13.75 | +0.167 |


Negative timing changes mean the collapse arrives earlier. The centroid uses
positive excess above a baseline joining first and last upstream observations,
held constant during the 96-hour tail, integrated over the complete simulation.
It is a descriptive pilot metric and not automatically the repository's official
response definition. No child is fitted to the downstream observations.
The roughly three-minute S008 peak shift is below the five-minute output interval:
it relies on peak interpolation and should be treated as a small model-derived
effect, not an observationally resolved timing difference.

![Collapse sensitivity](collapse_sensitivity.png)

Three preselected saved collapses span different sizes. The geometry audit is:

| State | Original links | Child links | Exactly unchanged links |
| --- | --- | --- | --- |
| S014_unit_10 | 46 | 42 | 41 |
| S008_unit_6 | 46 | 45 | 44 |
| S012_unit_12 | 46 | 42 | 41 |


The same forcing, external ports, shared n and boundary setting are used for every
parent–child pair. These are historical local caches; the active master drive
was unavailable, so current production-geometry identity is not certified.

## What the boundary scenarios mean

- **A:** normal-flow outlet using terminal bed slope as friction slope.
- **B/C:** halve/double only the outlet friction slope. Internal bed slopes,
  channel widths and physical link lengths are unchanged.
- **D:** add an idealized 10 km continuation with terminal width and bed slope,
  then place normal flow at its far end. Discharge is evaluated at the original
  outlet cross-section, with half-cell storage correction. The extra routing
  delay to the far end is not counted as the observed-reach response.

Both the outlet rating and internal friction use the scenario's fitted n. For
the extension, its roughness is the same as the network. The extension is an
assumption/sensitivity experiment, not surveyed downstream geometry.

## Numerical and extension checks

Eight unit tests pass, including the unchanged-bed boundary-factor check and an
extension test that verifies flux/storage conservation at the original port.
Across the 16 final parent/child runs, maximum global mass residual is
0.000146836 m³; original-port residual is
0.000146836 m³. Independent input-volume
error is at most 1.83e-07
relative. Depths stay positive; maximum Froude diagnostic is
0.321. All final tails satisfy the node-flux settling
criterion. These checks verify numerical behavior, not physical correctness.

Without refitting roughness, selected A/D runs are refined from 200 m to 100 m.
For D, 10 km is also compared with 20 km at the same n, measuring at the original
port. Results are:

| Scenario | State | Check | Peak-time change (min) | Max trace change (m³/s) |
| --- | --- | --- | --- | --- |
| A_normal | base | 200m_to_100m | +0.027 | 0.0025 |
| A_normal | S014_unit_10 | 200m_to_100m | +0.006 | 0.0030 |
| A_normal | S014_minus_base | 200m_to_100m_effect | -0.021 | — |
| D_extend10km | base | 200m_to_100m | +0.018 | 0.0025 |
| D_extend10km | base | 10km_to_20km_no_refit | +0.221 | 0.0724 |
| D_extend10km | S014_unit_10 | 200m_to_100m | +0.004 | 0.0031 |
| D_extend10km | S014_unit_10 | 10km_to_20km_no_refit | +0.180 | 0.0707 |
| D_extend10km | S014_minus_base | 200m_to_100m_effect | -0.015 | — |
| D_extend10km | S014_minus_base | 10km_to_20km_effect | -0.041 | — |


The `S014_minus_base` rows measure change in the collapse effect itself. The
extension check covers the original and S014; it does not independently certify
extension-length insensitivity for every possible collapse. B/C and the two
additional collapses have not each received their own mesh-refinement run.

![Calibration curves](calibration_curves.png)

## Reproduce and inspect

Run `run_boundary_experiment.py --workers 3 --stage all`, followed by
`summarize_boundaries56.py`, in the existing scientific Python environment.
`protocol.json` records source hashes, assumptions, target and geometry audit.
Each trial stores a result JSON and hydrograph Parquet. Tables:
`calibration_summary.csv`, `collapse_comparison.csv`, `numerical_controls.csv`.
Earlier uncalibrated and same-roughness trials remain unchanged.
