# Same-roughness trial

Both custom hydraulic states were rerun at the saved MC roughness **n = 0.020446875**.
Geometry, point-inflow series, 100 m maximum segment length, 300 s maximum internal
step, tolerances, normal-flow outlet and tail duration are unchanged from the
n = 0.035 hydraulic trial. Initialization is recomputed at the new roughness.

![Comparison](roughness_comparison.png)

| Quantity | Original | Collapse S014_unit_10 |
|---|---:|---:|
| Peak discharge, m³/s | 1853.292 | 1853.455 |
| Peak time, UTC | 2025-06-27 12:40:00+00:00 | 2025-06-27 12:20:00+00:00 |
| Lag behind raw inflow peak, hours | 14.167 | 13.833 |
| Minimum depth, m | 0.775 | 0.832 |
| Maximum depth, m | 2.611 | 2.611 |
| Maximum Froude diagnostic | 0.378 | 0.378 |

The collapse shifts the sampled peak by -20 minutes
and changes peak height by +0.163 m³/s.
Changing roughness alone shifts the original hydraulic peak by
-380 minutes relative to n = 0.035.
Negative shifts mean earlier. Peak sampling resolution is five minutes.

Both runs pass the mass-balance, independent input-volume and tail-settling checks
in `roughness_comparison.json`. The lower roughness trial has not received its own
additional spatial refinement study; the previous 100 m/200 m check was at n = 0.035.

The saved MC original peaks at 2025-06-27 14:39:30 UTC, and its collapse at
14:03:30 UTC (36 minutes earlier). The plot is a contextual comparison: MC uses a
different forcing window/preprocessing, fixed reference depths and fixed branch
fractions. Matching n does not make those assumptions equivalent, and the MC-calibrated
n is not automatically calibrated for the custom hydraulic model.

Full traces and provenance: `dx100_mc_roughness/hydrographs.csv` and `manifest.json`.
