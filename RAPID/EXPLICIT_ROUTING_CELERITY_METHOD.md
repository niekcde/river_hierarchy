> Historical alternative, superseded on 3 October 2026 by the [fixed-K production policy](HYDRAULIC_PRODUCTION_POLICY.md). Retained to document the tested option; do not enable it with signed-current-inflow routing.

# Explicit routing celerity regularization

Decision reference: EXPLICIT-ROUTING-CELERITY-2026-10-02.

Added after the all-16 hydraulic stability investigation. Local roughness
assignments optimized solely for stability were rejected as a physical
interpretation. This option instead labels a numerical routing correction
explicitly and preserves the hydraulic estimate. It is opt-in, not a new
default, and has not been applied to published/production runs.

## Rule

For a shared routing timestep dt, effective subreach length L and fixed X,
the nonnegative-coefficient interval is 2 X L/dt <= c <= 2(1-X)L/dt.
With the selected 1% interior margin, use

```
lower = max(common_min_celerity, 1.01 * 2 * X * L / dt)
upper = min(common_max_celerity, 0.99 * 2 * (1-X) * L / dt)
c_routing = clip(c_common_capped, lower, upper)
K_routing = L / c_routing
```

An empty interval raises an error; the common caps are never silently widened.
Depth, width, slope, n, X and effective length are unchanged. This can change
the hydrograph substantially; it is not evidence of a physically correct
celerity. A lower routing floor can be just as consequential as an upper cap.

## Interface and audit

`rapid_tools.celerity_regularization.regularize_routing_celerity` accepts a
prepared link frame and an explicit shared integer clock. It preserves
`rapid_celerity_mps_raw` and creates pre-regularization celerity/K, routing
celerity, correction factor, direction/reason, clock, margin and interval
columns. It updates `rapid_k` and the celerity used by routing only.

For state preparation, use `RapidPrepConfig` with
`use_routing_celerity_regularization=True`, ordinary celerity caps enabled,
and effective-length correction disabled. Pass the reviewed common clock as
`prepare_state(..., planned_routing_dt_seconds=dt)` for parent and child.
The normal forcing-divisibility check still applies. Saved link attributes
contain the audit fields; prep summaries count numerical corrections separately
from ordinary celerity caps. The existing n/Kb configuration is not changed
by enabling this option.

Automatic experiment clock planning is intentionally not enabled for this
option. It fails early with instructions to use the explicit shared-clock
interface. The case checks below show why automatically promoting this
safeguard across an experiment would be premature.

Do not use independently selected clocks for parent and child. Unchanged links
with identical hydraulic inputs and a common clock receive identical numerical
corrections. This is necessary but does not guarantee an unbiased collapse
response: changed topology and replacement links can interact with the limits.

## Verification and present limitation

Seven focused tests cover preservation of hydraulic inputs, the interior
interval, shared-link consistency, invalid/cap-incompatible clocks and repeat
application. Existing preparation tests also pass. A separate split/rejoin
impulse check matches the frequency-domain sensitivity runner to the existing
RAPID recurrence to machine precision.

The real-network pilot routes cases 11 and 46 with selected simple-unit
children, original and bounded-adjusted heads, and two clocks dividing each
saved forcing interval. See
`assets/bounded_robust_slopes_20261002/common_caps/explicit_regularization_check/README.md`.
The largest corrections remain materially clock-dependent. Do not treat the
small corrected-link fraction as validation of their timing predictions.

## Subsequent investigation: signed coefficients

Reference SIGNED-MUSKINGUM-2026-10-02 tests a different candidate: keep the
hydraulic celerities fixed, retain the upper timestep bound, and allow negative
current-inflow coefficients. This distinguishes positivity from boundedness.
See `assets/bounded_robust_slopes_20261002/common_caps/signed_coefficient_check/README.md`.
That test does not enable signed coefficients in production preparation or
change the opt-in safeguard documented above. Historical "stability failures"
in the earlier audits should be read as coefficient-positivity/clock failures,
not demonstrated divergence of the unmodified hydraulic recurrence.

## Expanded validation — 2026-10-03

Reference: FULL-SIGNED-ROUTING-2026-10-02 (started October 2, completed October 3).
See [the full report](assets/bounded_robust_slopes_20261002/full_signed_validation/README.md)
and the final dated section of `notebooks/slope_smoothing_floor_comparison.ipynb`.

The fixed-hydraulic screen covers 2,185 processable unique states and 2,169
base–child comparisons at two clocks. All 16 base networks also received native
segment audits; 13 use saved actual-event forcing and three use a standard pulse.
The tests support the numerical candidate, but do not approve a production rerun:
76 conceptual entries lack usable refreshed inputs, some processable child-head
allocations create extreme positive slopes, and saved production settings still
differ from n=.035 / alpha=.65 / common .05–5 m/s caps. A few capped child links
are long enough that their cap has a material timing effect. Fix/verify the full
child hydraulic reconstruction before interpreting those results physically.

The candidate configuration and complete coverage/exclusion tables are saved
with the report. No production hydraulic settings or results were changed.

## Child reconstruction resolved — 2026-10-03

The child-head issues in the preceding expanded audit have been resolved for all
3,551 included entries. The five remaining exclusions were intentional topology
exclusions, not missing inputs. See [the method](CHILD_HEAD_RECONSTRUCTION_METHOD.md)
and its dated results. The case-22 long-link cap sensitivity and production
configuration migration remain separate from this resolved reconstruction issue.
