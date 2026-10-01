# Current RAPID response metrics

Decision recorded 29 September 2026. These are the three responses to evaluate
for parent–child comparisons in both synthetic and real-network analyses.
They are analysis definitions, not a claim that all historical notebooks have
already been migrated. Synthetic full-grid results using these definitions are
in `synthetic_experiments/outputs/full_grid_island/current_analysis`.

| Response | Definition | Meaning of a positive change |
| --- | --- | --- |
| Peak-lag change (%) | `100 × (lag_child / lag_parent − 1)`, with lag = outlet peak time minus the common inlet peak time | Child's dominant peak arrives later relative to the input peak |
| Event-duration change (%) | `100 × (duration_child / duration_parent − 1)` | Child event lasts longer |
| Peak-excess change (%) | `100 × [(Qpeak_child − Qref) / (Qpeak_parent − Qref) − 1]` | Child's peak is higher above the shared reference |

These are signed relative changes, not absolute values. Values are not generally
bounded by +100%. Changing a percentage response between two experimental
settings gives a difference in **percentage points**.

## Event duration: 1% departure to final settlement

Use the known stable initial and final plateaus, Q0 and Qf. After spin-up,
find the parent's event peak Qp. The rise tolerance is `0.01 × (Qp − Q0)`;
the recession tolerance is `0.01 × (Qp − Qf)`. Both amplitudes must be positive.
Apply exactly these same absolute tolerances to parent and child.

- Start: first departure outside the tolerance around Q0.
- End: final return within the tolerance around Qf, after the dominant peak.
  Every subsequent saved value must remain within tolerance. Require at least
  one hour of saved settled tail; retain longer tails where available.
- Duration: end minus start, measured separately at each network's outlet.
- Interpolate boundary crossings between discharge samples. Synthetic outputs
  are interval averages, placed at interval midpoints for this extraction.
- Repeat at 0.5% and 2% as sensitivity checks. A late secondary peak outside the
  tolerance extends the event even after an intervening quiet period.

This is not width at half maximum, cumulative-volume t01–t99, or time from inlet
onset to outlet settlement. It does not require an assumed background curve
connecting unequal plateaus. A pure translation leaves duration unchanged.
A small peak below tolerance is deliberately excluded. Separate peak counts or
hydrographs are still needed to distinguish widening from multiple peaks.

For real cases, exclude numerical startup and use a common forcing event with
adequate complete parent/child records. Establish both plateau levels from the
steady forcing/output sections. Do not infer settlement from a truncated record.
The new duration metric has not yet been evaluated in the real-case pilot.

## Peak excess and lag conventions

For equal synthetic plateaus, Qref is the prescribed 100 m³/s background.
For unequal real plateaus, retain the real pilot's explicit convention
`Qref = max(Q0, Qf)` for peak excess, shared by the pair. This is a scalar
reference for peak height, not a background curve or flood-volume separation.
Changing this convention rescales the percentage response and must be disclosed.

Use the same inlet peak reference for parent and child. Require positive parent
lag and positive parent peak excess; otherwise flag the percentage as undefined.
Peak lag can switch discontinuously when a different peak becomes dominant.
Report output timing resolution and retain hydrographs to interpret such cases.

## Supporting diagnostics and historical names

Keep temporal spread (event-volume temporal standard deviation), peak count,
cumulative-volume duration, volume balance, sampling resolution, celerity caps
and effective-length floors as diagnostics. Temporal spread is no longer one of
the three primary responses in the current synthesis.

Older `event_duration_change_pct` columns in the factor/geometry analyses refer
to **cumulative-volume t01–t99**. They must not be silently mixed with the current
plateau duration. The current dataset preserves that old value as
`volume_quantile_duration_change_pct` and uses `event_duration_change_pct` for
the new 1% plateau definition. Original files and executed notebooks remain
historical records.

## Omega

Use **omega** for `Lf × E × R`, where `R = −log10(Lambda_K)` and
`Lambda_K = equivalent hydraulic resistance_child / resistance_parent`.
Lf is parent selected-domain channel length divided by total parent channel
length; E is the matched-domain exposure. Omega is not the raw resistance ratio,
not the Muskingum X parameter, and not the routing travel-time K.

Keep R, Lf and E available separately for prediction. Their product is an
interpretable scalar projection; it need not retain all information in its
components or predict all three responses equally well.
