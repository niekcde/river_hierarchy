# Peak-window audit — 5 October 2026

Reference: RAPID-PEAK-WINDOWS-2026-10-05. Requested after cases 03/05 showed
that first-maximum gauge timing can misrepresent agreement. This first audit
covers all 16 cases; it is not a causal diagnosis of additional downstream
inflow. No forcing, routing output, primary metric or predictor was changed.

`summary.csv` reports exact-equal-maximum spans, disconnected maximum blocks,
first and span-midpoint timing comparisons and normalization sensitivities.
`near_peak_windows.csv` retains separate connected blocks above 100%, 99.5%
and 99% of absolute peak discharge. These are descriptive diagnostic thresholds,
not confidence intervals or replacements for the 1% event-duration definition.
`equal_maxima.csv` records each maximum sample; every case has its own PNG.
Actual UTC comes from the verified original forcing chronology.

Case 03: the continuous downstream maximum plateau is 17 February 17:45–23:00.
Its midpoint is 20:22:30; the sampled model peak at 22:40 is 2h17m30s later.
Observed midpoint lag is 12h07m30s versus model lag 14h27m45s: +19.3% rather
than +52.2% under first-maximum selection. The model peak is inside the maximum
plateau. The plateau midpoint is a reproducible summary, not a measured unique
peak time. This does not validate the full hydrograph shape or volume.

Case 05: user proposed 7 January 19:00–8 January 09:00 as an earlier shoulder.
Observed Q spans 1089.7–1107.0 m3/s there, a range equal to 1.50% of the event
maximum. The model peak is inside that window, 2h35m50s before its midpoint.
This is explicitly exploratory and does not replace the later recorded maximum.
Neither lateral input nor the identity of a propagating feature can be proven
from these two hydrographs alone. Defining a window merely to improve model
agreement would be circular; apply an observation-based feature rule and retain
both the first-feature and full-event maximum comparisons.

For fixed model outputs, changing only the selected OBSERVED gauge peak does
not change parent-child absolute timing, model percentage lag, duration or peak
excess. It changes baseline validation and gauge-scaled speed estimates.
For fixed parent-child peaks, changing the selected FORCING peak ti rescales
r=(tc-tp)/(tp-ti) but leaves tc-tp unchanged. Moving from the earliest to latest
equal forcing maximum increases |r| by 4.19% in 05, 20% in 11, 18.01% in 41,
and 53.85% in 48. These are relative changes in metric magnitude, not percentage
points. They are interval bounds, not probabilities; distinct maxima need not
constitute one physically flat plateau. With case 05's particular 16:08:20
choice rather than 15:49:10, |r| increases by only about 0.58%.

At fixed r and gauge distance L, gauge-scaled speed difference is
Delta c = (L/Tobs) * [1/(1+r)-1]. Thus observed-lag ambiguity rescales its
magnitude. The model-only network speed metric does not use observed Tobs,
but does depend on the chosen model input peak through model travel lags.

Smoothing the actual forcing would change the model experiment and require new
routing to quantify its effects. A smoothing sensitivity for observed feature
identification can be examined without changing existing simulations, provided
it is labelled and does not replace the forcing that was actually routed.

Next review per-case plateaus, gaps and multi-peak shapes, then choose a common
peak-window/feature rule independent of model fit. Retain exact maxima and
ambiguity bounds alongside representative times. Inspect whole-event rise,
recession, amplitude and volume before claiming acceptable baseline realism.
