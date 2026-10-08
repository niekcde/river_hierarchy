# Event buffers: tests and conditional runtime projection

Reference **EVENT-BUFFERS-2026-10-05**. Requested after selecting shorter observed event windows and proposing a short steady spin-up plus settling-based spin-down. Production forcing, routing outputs and analysis metrics are unchanged. These are isolated tests at the existing hydraulic parameters, segmentation and native routing clocks.

## Tested scope

- Audited saved outlet settlement for **2,230 distinct routed states** (16 parents plus 2,214 children).
- Tested the newly cropped forcing on **39 states across all 16 examples**: every parent, largest cumulative travel-time candidate, latest-settling child under old forcing, and the three example-37 shape/peak cases where not already selected. `selection.csv` records the choices.
- All states use declared initial equilibrium (`steady_first_inflow`). Compare no initial buffer with 6 and 24 hours at the new event's first discharge. Buffers are rounded up to whole forcing intervals, keeping event samples and phase fixed.
- Full waveform comparison, including the long diagnostic tail, gives maximum difference **1.45519e-11 m³/s**, or **7.08993e-15 of waveform range**. Maximum interpolated peak-time difference: **0 seconds**. This supports a 24-hour plateau under equilibrium initialization; it does not validate short warm-up from zero flow.
- The production engine bypasses exactly constant initial forcing while initialized at equilibrium. Thus retaining the old long initial plateau with the same NEW event/initial discharge is equivalent by that declared shortcut. The old event itself can have different antecedent conditions and is a separate sensitivity question.

## Reproduction and numerical checks

The efficient spatial recurrence calls the same linear Muskingum coefficients/held-forcing recurrence as the production sparse engine. Match the six-decimal internal bifurcation allocations and float32 connectivity representation used in preparation; keep inlet allocation unchanged. The optional serialized-allocation switch is in an investigation helper only, defaults off for earlier clock investigations, and changes no production code.

Each selected state is checked against 32 new-event forcing intervals evaluated with production sparse operations: maximum outlet difference **1.09139e-11 m³/s**. A separate first-128-active-interval check against saved production output has maximum difference **0.000226322 m³/s** and passes a 1-ppm-of-forcing-range tolerance. The spin-up comparison then covers complete newly routed waveforms, not just these parity probes. No flow clipping or timestep changes are used.

## Final settling rule

Hold the input at the new event's final discharge. Use a tolerance of **0.1% of the new parent peak excess above max(initial, final discharge)**. From the last outlet exceedance after forcing-event end, require **24 continuous hours** within tolerance, followed by **another 24 hours**. The diagnostic runs continue substantially longer to check that this is not a temporary crossing. Use the latest required end among the selected parent/children, rounded up to a forcing interval.

All tested peaks remain within the proposed shortened output; maximum subsequent omitted-tail deviation is **4.62355e-06 of parent amplitude**, below the 0.001 threshold. The settlement helper was checked against a temporary plateau followed by a later excursion and a censored tail.

**9 examples** have a tested child settling later than the parent; maximum additional time is **5.62 hours**. Therefore do not use the parent cutoff without a child check. These tests certify the 39 chosen states, not all newly forced children. A future full rerun must verify every state at the common cutoff and extend that example's shared tail if needed. All old saved states settled, but that alone does not certify the changed forcing.

## Event-cropping sensitivity

Compared with the existing saved responses, the maximum absolute change in the available selected parent–child interpolated timing differences is **0.00316318 seconds** (23 child comparisons). This is encouraging, but is not a complete shape, volume or all-state invariance claim. Early/late excluded features and event-duration definitions can change. The source event windows are recorded separately, with July 5 confirmed for example 41.

## Estimated lengths and routing work

Proposed total = 24-hour initial plateau (rounded up) + new observed event + latest tested settlement tail including both 24-hour periods. Dates/end inclusivity are inherited from the user-reviewed proposals. Tiny duration differences reflect forcing-grid rounding.

| example | total_states | old_total_days | event_days | new_common_tail_days | proposed_total_days | duration_reduction_pct | dynamic_work_reduction_pct |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 02 | 36 | 177.03 | 14.99 | 2.67 | 18.66 | 89.46 | 89.49 |
| 03 | 41 | 30.96 | 8.98 | 2.66 | 12.65 | 59.13 | 41.61 |
| 05 | 52 | 94.34 | 14.99 | 4.46 | 20.47 | 78.3 | 68.41 |
| 11 | 27 | 91.92 | 26.99 | 3.06 | 31.07 | 66.2 | 63.31 |
| 13 | 67 | 54.05 | 7.98 | 6.9 | 15.88 | 70.63 | 59.13 |
| 15 | 263 | 70.85 | 15.98 | 4.01 | 21.0 | 70.35 | 53.5 |
| 21 | 31 | 46.7 | 11.11 | 3.24 | 15.35 | 67.12 | 50.33 |
| 22 | 409 | 116.94 | 8.99 | 5.79 | 15.79 | 86.49 | 76.5 |
| 37 | 397 | 54.48 | 6.99 | 3.14 | 11.14 | 79.55 | 70.79 |
| 41 | 199 | 216.8 | 24.99 | 6.92 | 32.92 | 84.82 | 74.64 |
| 45 | 94 | 42.68 | 17.99 | 4.99 | 23.98 | 43.82 | 26.65 |
| 46 | 224 | 374.46 | 18.99 | 16.6 | 36.61 | 90.22 | 82.12 |
| 48 | 73 | 48.0 | 17.99 | 2.45 | 21.44 | 55.34 | 46.15 |
| 54 | 82 | 41.88 | 11.99 | 3.08 | 16.08 | 61.61 | 46.03 |
| 55 | 213 | 146.59 | 18.98 | 6.03 | 26.04 | 82.23 | 71.31 |
| 56 | 22 | 70.35 | 26.99 | 3.18 | 31.16 | 55.7 | 42.71 |

Weighted by the number of distinct states, total simulated state-days fall by **83.1%**. Weighting active forcing steps by routing substeps and state segment counts gives **77.4% less routing work**, approximately **4.43× fewer segment-substeps**. Full-network output-value count falls by **86.1%**; this is not a measured compressed-file-size reduction.

The routing-work projection excludes the already skipped initial plateau. It assumes the tested common tails are sufficient for the remaining states. It is not a full-batch wall-clock measurement or an ETA: preparation, equivalent resistance, sparse-solver sparsity, serialization, compression and I/O do not scale identically. Parent sparse seconds-per-interval are exported as local diagnostics, not used to promise an absolute completion time. No equivalent-resistance recalculation is needed merely to shorten forcing at unchanged geometry/hydraulics.

## Files and next implementation

Run `event_buffer_tests.py`, `full_event_spinup_check.py`, then `report_event_buffer_tests.py` under `RAPID/investigations`. Saved parquet trajectories are isolated test results. Exported selections, all-state old-settlement audit, new-event tests, full-waveform spin-up differences and per-example runtime projections accompany this report.

Recommended next implementation: retain equilibrium initialization; use a 24-hour initial plateau; rebuild forcing from observed event samples before padding once; use the tested tail as an initial plan and verify/extend for every child. Keep the existing long runs as references until the complete rerun's metrics are checked. This report documents the tests; it does not switch active production to them.
