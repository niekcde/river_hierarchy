# Routing stages and exact UTC schedule

Reference **ROUTING-SCHEDULE-2026-10-05**. Recorded at the user's request after the spin-up and settling tests. These are the exact timestamps of the tested proposal, reconstructed on the existing forcing sample grid. They have **not** replaced production forcing. The current production start/end are shown separately.

## Where to find the values in Excel

Workbook: `RAPID/reviews/hydrograph_match_review.xlsx`, worksheet **Routing windows**. The root-level workbook with the same basename is an older empty workbook and is not the updated review file.

| Information | Cells |
| --- | --- |
| Example IDs in each table | A7:A22, A28:A43, A49:A64 |
| User-requested event start / exclusive end | B7:C22 |
| First / last retained event samples on the existing grid | D7:E22 |
| Initial / final plateau discharge (m³/s) | F7:G22 |
| Tested simulation start (initial plateau begins) | B28:B43 |
| Initial plateau ends / final plateau begins | C28:D43 |
| Outlet remains within the settling tolerance from | E28:E43 |
| 24-hour sustained settlement confirmed | F28:F43 |
| Tested simulation end, including extra 24-hour buffer | G28:G43 |
| Current production simulation start / end | B49:C64 |
| Current / tested duration in days and reduction | D49:F64 |
| Number of tested / total routed states | G49:G64 |
| Protocol, status and source explanation | A68:G82 |

Dates are UTC, displayed to the second. “End exclusive” means the new named end dates include the whole calendar day, ending at midnight the next day. Boundaries marked current keep their existing timestamp. The first/last retained samples can differ slightly from those requested bounds because the tested event preserves the existing forcing grid. Do not confuse the last retained event sample with the exclusive selection boundary.

## Routing stages

1. Initialize every link at equilibrium with the first retained event discharge (`steady_first_inflow`).
2. Hold that discharge for at least 24 hours, rounded up to complete forcing intervals. This is the initial plateau, from simulation start to the first event sample.
3. Apply the retained observed event samples at the unchanged forcing cadence and routing timestep.
4. Hold the last retained event discharge constant. Allow the parent and tested children to reach their final plateau. The tolerance is 0.1% of parent peak excess above max(initial, final discharge). Use the latest final exceedance among tested states, not a temporary early crossing.
5. Require 24 continuous hours within tolerance, then another 24-hour buffer. Round the final simulation endpoint up to a forcing sample. Parent and children use one shared endpoint.
6. When implementing the full batch, verify every child's final settlement and extend that example's shared tail if necessary. The schedule currently covers selected-state tests, not a certification of every newly forced child.

The source tests audited all 2,230 existing routed states and rerouted 39 selected states. Full 0/6/24-hour spin-up comparisons agree within numerical precision. Estimated batch reductions are 83% in simulated state-days and 77% in segment-routing substeps, not a measured whole-pipeline speedup. See [test report](../../validation/event_buffer_tests_20261005/README.md).

## Exact tested simulation boundaries

| example | tested_simulation_start_utc | first_retained_event_sample_utc | last_retained_event_sample_utc | tested_simulation_end_utc |
| --- | --- | --- | --- | --- |
| 02 | 2025-07-28 00:06:00+00:00 | 2025-07-29 00:09:00+00:00 | 2025-08-12 23:52:20+00:00 | 2025-08-15 16:00:20+00:00 |
| 03 | 2023-02-12 00:16:00+00:00 | 2023-02-13 00:22:15+00:00 | 2023-02-21 23:49:30+00:00 | 2023-02-24 15:58:45+00:00 |
| 05 | 2023-12-29 23:46:40+00:00 | 2023-12-31 00:03:20+00:00 | 2024-01-14 23:45:00+00:00 | 2024-01-19 11:05:00+00:00 |
| 11 | 2024-05-02 23:57:30+00:00 | 2024-05-04 00:06:30+00:00 | 2024-05-30 23:56:30+00:00 | 2024-06-03 01:32:30+00:00 |
| 13 | 2024-09-30 00:18:00+00:00 | 2024-10-01 00:18:00+00:00 | 2024-10-08 23:42:00+00:00 | 2024-10-15 21:18:00+00:00 |
| 15 | 2026-03-30 00:12:00+00:00 | 2026-03-31 00:19:50+00:00 | 2026-04-15 23:46:30+00:00 | 2026-04-20 00:17:50+00:00 |
| 21 | 2025-07-25 21:04:00+00:00 | 2025-07-26 21:05:00+00:00 | 2025-08-06 23:50:00+00:00 | 2025-08-10 05:34:00+00:00 |
| 22 | 2025-08-14 00:00:50+00:00 | 2025-08-15 00:09:10+00:00 | 2025-08-23 23:52:30+00:00 | 2025-08-29 19:03:20+00:00 |
| 37 | 2014-06-23 23:59:15+00:00 | 2014-06-25 00:06:20+00:00 | 2014-07-01 23:53:00+00:00 | 2014-07-05 03:23:00+00:00 |
| 41 | 2024-07-04 00:00:10+00:00 | 2024-07-05 00:04:40+00:00 | 2024-07-29 23:52:20+00:00 | 2024-08-05 22:01:10+00:00 |
| 45 | 2024-06-12 00:00:00+00:00 | 2024-06-13 00:00:00+00:00 | 2024-06-30 23:42:00+00:00 | 2024-07-05 23:24:00+00:00 |
| 46 | 2024-08-23 00:00:00+00:00 | 2024-08-24 00:06:40+00:00 | 2024-09-11 23:48:40+00:00 | 2024-09-28 14:32:55+00:00 |
| 48 | 2025-05-22 00:00:00+00:00 | 2025-05-23 00:00:00+00:00 | 2025-06-09 23:42:00+00:00 | 2025-06-12 10:30:00+00:00 |
| 54 | 2023-06-17 00:01:30+00:00 | 2023-06-18 00:10:30+00:00 | 2023-06-29 23:52:30+00:00 | 2023-07-03 01:54:00+00:00 |
| 55 | 2025-04-05 23:41:20+00:00 | 2025-04-07 00:04:20+00:00 | 2025-04-25 23:39:20+00:00 | 2025-05-02 00:40:20+00:00 |
| 56 | 2025-06-20 00:00:00+00:00 | 2025-06-21 00:00:00+00:00 | 2025-07-17 23:42:00+00:00 | 2025-07-21 03:54:00+00:00 |

## Reproducibility

Machine-readable full schedule: `RAPID/validation/event_buffer_tests_20261005/routing_schedule_utc.csv`. Requested windows: `RAPID/reviews/proposed_event_windows_20261005.csv`. Durations/settlement: the test report's `duration_runtime_projection.csv` and `selected_state_tests.csv`. Reconstruct with `RAPID/investigations/document_routing_windows.py`.

This report and the new worksheet preserve previous review values and notes. Changing workbook dates alone does not regenerate routing or recertify the tested tail. No active input files or model results were changed by documenting this schedule.
