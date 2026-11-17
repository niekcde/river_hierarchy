# Example 56: first custom 1D hydraulic attempt

The prototype ran successfully for the original network and saved collapse
**S014_unit_10**. The collapse advances the outlet hydrograph while changing its
peak height very little. These are results of the new custom hydraulic model,
not saved RAPID hydrographs.

![Hydrographs](hydrograph_comparison.png)

## Results (100 m maximum segment length)

| Quantity | Original | Collapse |
|---|---:|---:|
| Physical links | 46 | 42 |
| Numerical faces | 1595 | 1456 |
| Outlet peak, m³/s | 1845.664 | 1846.227 |
| Outlet peak, UTC | 2025-06-27 19:00:00+00:00 | 2025-06-27 18:35:00+00:00 |
| Lag behind inflow peak, hours | 20.500 | 20.083 |
| Minimum depth anywhere, m | 1.088 | 1.152 |
| Maximum depth anywhere, m | 3.601 | 3.602 |
| Maximum face Froude diagnostic | 0.226 | 0.226 |
| Initialization spin-up, days | 3 | 3 |

The upstream peak was 1862.302 m³/s at
2025-06-26 22:30:00+00:00. Relative to the original, the collapse peak occurs
**25 minutes earlier**, with
**+0.563 m³/s
(+0.0305%)** change in magnitude. The
maximum simultaneous difference is
**19.661 m³/s**.
Five-minute output sampling limits interpretation of peak timing below that scale.

The positive-excess centroid shifts by
**-29.363 minutes**
(negative means earlier). This descriptive centroid integrates positive discharge
above a straight baseline joining the first and last observed inflow values,
with the final baseline held through the tail. It is baseline dependent and is
not the established repository response metric without further harmonization.

## Behavior and interpretation

- The original routes water through parallel paths with different lengths and
  conveyances. Replacing the five-link section with its saved equivalent channel
  changes both conveyance and storage; the collapsed hydrograph leads on the rise
  and generally falls sooner. This interpretation is consistent with the results,
  but the experiment does not isolate length, width and storage effects separately.
- Flow splitting is dynamic. At node 20 the direct branch to node 17 carries
  **51.40–54.20%**
  of outgoing flow, compared with a constant width fraction of
  **50.55%**. The remaining flow follows the other branch.
  Fractions are calculated from simulated face flows, not imposed in routing.
- All sampled channels stay wet and all sampled face flows remain downstream.
  The code permits signed reverse flows, but this event did not produce them.
  The Froude diagnostic remains subcritical, consistent with this slow-flow pilot;
  it is not by itself proof of diffusive-wave validity.
- Depth evolves above the fixed proxy bed. No fixed BASED depth, RAPID celerity
  cap, K/X coefficient or prescribed branch fraction is used during routing.
- Both runs settle back to the final inflow of 686.624 m³/s
  during the 96-hour constant tail. Final inflow is lower than initial inflow,
  so the network releases stored water: outflow volume exceeding inflow volume
  is expected and is explicitly accounted for.

![Dynamic split](dynamic_split.png)

## Conservation and numerical verification

Six physical/numerical unit tests passed: uniform normal flow; internal lake at
rest; symmetric transient splitting and conservation; reversal and datum shift;
conflicting junction-bed rejection; and analytic Jacobian verification.

For the two accepted 100 m runs, the largest absolute storage/boundary-volume
residual is 0.000124186 m³,
or 4.68e-14
of total inflow volume. This checks the discretized mass identity, not all errors
in the continuous hydraulic solution. An independent exact integral of the
piecewise-linear input is 2650733091.654 m³; the maximum relative discrepancy
with the solver's integrated inflow is
1.08e-07.

Refining from 200 m to 100 m (same 300 s maximum internal step and tolerances)
changes either complete outlet trace by at most **0.00217 m³/s**.
The collapse-induced centroid shift changes by
**0.267 seconds**.
The sampled peak times are unchanged. The approximately 25-minute effect is
therefore resolved at these two mesh sizes. This is a first refinement check,
not a comprehensive validation across flows, boundaries and geometries.

## Inputs, scope and limitations

- Original: cached `S000_base`, 46 links. Collapse: cached `S014_unit_10`,
  42 links. Exactly 41 exterior links are unchanged. Five links in the section
  from node 20 to node 17 become one 10,243.126 m channel of width 1,100.125 m.
  The original five links total 24,045.729 m across parallel paths; that total
  is **not** a single upstream-to-downstream travel distance.
- Fixed bed = the cached reconstructed water-surface heads, as agreed. The
  original and collapsed section have identical endpoint elevations. Widths
  are treated as rectangular widths at all depths; no bank overtopping occurs.
- Shared Manning n = 0.035, deliberately uncalibrated. Both runs use the same
  inlet series and identical terminal normal-flow rating. No lateral input.
- Observed point inflow: 2025-06-21 00:00:00+00:00 to
  2025-07-17 23:55:00+00:00. Linear interpolation bridges
  2 gaps longer than 15 minutes; the largest gap
  is 65 minutes. The missing observations
  are not recovered by this interpolation.
- These historical local caches were available while the current master drive
  was unmounted. This is an example-56 pilot, not certification of the current
  production geometry release. Exact source hashes are in the run manifests.
- The cached case's reviewed observed peak lag is about 16.17 hours; this
  uncalibrated original run gives 20.50 hours.
  Thus it is about 4.33 hours slower
  by that peak-only comparison. This is a useful calibration warning, not a
  formal validation score, and the collapse comparison is not a fit to observations.
- This is a custom diffusive-wave graph implementation, not the official
  LISFLOOD-FP engine. It omits inertia, junction momentum/losses, floodplain
  storage, wetting/drying and uncertainty in the proxy bed. The small peak-height
  change should not be treated as physically established to three decimals.

## Files and next step

`dx100/hydrographs.csv` contains both hydrographs, input, storage and mass residuals.
`dx100/*_link_flows.parquet` preserves upstream/downstream link traces.
`dx100/*_nodes.csv` stores depth ranges; `*_mesh.csv` stores the discretization.
`verification.json` compares resolutions; each run has a complete `manifest.json`.
See ../../README.md for equations, assumptions, test and run commands.

The prototype demonstrates the requested coupling and collapse comparison.
Before expanding the population, check downstream-boundary influence and
roughness/bed sensitivity, then audit the active geometry when the drive is
available. Those tests would address physical uncertainty rather than numerical
mass balance alone.
