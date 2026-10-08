# Standard-sampling total variation and proposed event windows

Reference **STANDARD-TV-2026-10-05**. Requested after the example-37 native-sampling TV diagnostic. All 2,214 canonical pairs, all 16 network examples, no population exclusions. This is a descriptive sensitivity analysis, not an adopted primary response or a routing rerun.

## Sampling and comparison

Main diagnostic: 1-hour discharge averages on complete UTC clock-hour bins. Sensitivity: 2-hour and 4-hour averages with a shared UTC origin. Integrate the piecewise-linear saved hydrograph exactly within each bin, then divide by bin duration. This is time averaging before computing variation, rather than taking every nth sample or treating interpolation as extra information. All native maximum steps are below 1 hour (9.17–25.67 minutes). Parent/child time arrays and shared inputs are verified. Analytic checks cover linear means, integral conservation and irregular sample times.

TV = sum(abs(Qbar[i+1]-Qbar[i])); response = 100*(TV_child/TV_parent-1). Also report TV divided by each waveform's own binned peak excess above its FULL-RUN endpoint reference (max(initial,final)). This fixed endpoint-reference rule avoids redefining baseline simply because the diagnostic window was cropped. It differs from the previous cropped-window endpoint normalization; do not mix those normalized scores. The primary production peak-excess metric remains unchanged.

Analyze both full saved runs and the user-proposed target windows. Complete bins only: sub-hour partial bins at window ends are omitted. Cropping existing output is not rerouting a cropped forcing, and shifts across window boundaries can change TV. Time averaging also changes TV and peak amplitude; shared clock bins do not eliminate sensitivity to waveform phase. Very small changes must not automatically be interpreted as robust physical smoothing effects. Raw TV has discharge units; normalized TV removes amplitude scale but is not a diffusion coefficient.

## Results

| example | state_id | tv_change_pct | normalized_tv_change_pct |
| --- | --- | --- | --- |
| 37 | CB001_bubble_3 | 10.55 | 7.676 |
| 02 | S013_unit_20 | 2.368 | 2.132 |
| 46 | CB009_bubble_74 | 2.0 | 1.342 |
| 22 | CB002_bubble_11 | 1.356 | 0.386 |
| 15 | CB001_bubble_4 | 1.092 | 0.249 |
| 54 | S065_unit_9 | 1.019 | 0.878 |

Table: strongest hourly raw-TV increase per example, top six examples. This is explicitly selected to illustrate responses, not a representative sample. Across all 2,214 pairs, **12** have >1% increase and **1** have >5% increase. Both increases and decreases occur. Most changes are small; example 37 is exceptional in this inventory. The figure compares the same states at all three intervals.

![Sampling sensitivity](variation_sampling_comparison.png)

For example 37's compound, full-event TV increases by 10.55%, 6.70% and 4.71% at 1, 2 and 4 hours respectively. Hourly own-amplitude-normalized change is +7.68%. In the proposed June 25–July 2 window, hourly TV increases by +7.23%, normalized +4.44%. This supports a timescale-dependent roughness difference, not a universal attenuation percentage. Results do not establish significance or treat pairs as independent river samples.

## Proposed dates (not applied to forcing)

User dates interpreted as UTC, years taken from the existing event. New named end dates include that entire date (exclusive midnight next day). “Current/good” keeps the exact configured boundary, not merely the observed coverage. New boundaries are proposals for target events, not final production warm-up/recession limits.

| example | proposed_start_utc | proposed_end_exclusive_utc | parent_peak_in_proposal |
| --- | --- | --- | --- |
| 02 | 2025-07-29 00:00:00+00:00 | 2025-08-13 00:00:00+00:00 | True |
| 03 | 2023-02-13 00:01:00+00:00 | 2023-02-22 00:00:00+00:00 | True |
| 05 | 2023-12-31 00:00:00+00:00 | 2024-01-15 00:00:00+00:00 | True |
| 11 | 2024-05-04 00:00:00+00:00 | 2024-05-31 00:00:00+00:00 | True |
| 13 | 2024-10-01 00:01:00+00:00 | 2024-10-09 00:00:00+00:00 | True |
| 15 | 2026-03-31 00:01:00+00:00 | 2026-04-16 00:00:00+00:00 | True |
| 21 | 2025-07-26 21:00:00+00:00 | 2025-08-07 00:00:00+00:00 | True |
| 22 | 2025-08-15 00:01:00+00:00 | 2025-08-24 00:00:00+00:00 | True |
| 37 | 2014-06-25 00:00:00+00:00 | 2014-07-02 00:00:00+00:00 | True |
| 41 | 2024-07-05 00:00:00+00:00 | 2024-07-30 00:00:00+00:00 | True |
| 45 | 2024-06-13 00:00:00+00:00 | 2024-07-01 00:00:00+00:00 | True |
| 46 | 2024-08-24 00:00:00+00:00 | 2024-09-12 00:00:00+00:00 | False |
| 48 | 2025-05-23 00:00:00+00:00 | 2025-06-10 00:00:00+00:00 | True |
| 54 | 2023-06-18 00:01:00+00:00 | 2023-06-30 00:00:00+00:00 | True |
| 55 | 2025-04-07 00:00:00+00:00 | 2025-04-26 00:00:00+00:00 | True |
| 56 | 2025-06-21 00:00:00+00:00 | 2025-07-18 00:00:00+00:00 | True |

- Example 02: July 29–August 12 is assigned to the 2025 event. Existing configured end was August 10, 2026; this discrepancy is recorded, not silently rewritten upstream.
- Example 41: July 5 confirmed by the user, correcting the initial June 5 proposal. Proposed-window diagnostics were regenerated.
- Example 46: current sampled model parent peak is September 12, 2024 at 00:24:50 UTC, about 25 minutes beyond the proposed inclusive September 11 end. Keep that end as an observed target-event proposal; output analysis/routing needs extra trailing support. Child peak coverage is exported per pair.
- All proposals lie within saved MODEL output, but saved padding is not evidence of actual gauge coverage. Observational coverage and initial/final settlement must be checked before changing forcing.

Reproduce: `RAPID/investigations/standard_total_variation.py`. Exports: `comparison_total_variation.csv` (13,284 rows), `network_summary.csv`, `proposed_event_windows.csv`. Dates also recorded in the review folder. No Excel workbook, gauge-quality settings, forcing, routing or existing primary metrics were changed.

## Event bounds versus steady buffers — user clarification, 5 October 2026

The supplied dates bound the observed forcing event, not the full simulation. Before the event, hold Q at its first value; after it, hold Q at its final value (not necessarily equal to the initial value). Use shared forcing/buffers for all states of an example. Initial/final plateaus already exist: saved example 37 forcing has about 19.75 days at each end, example 41 about 90.9 days, and example 46 about 175.23 days. These are measured constant stretches, not a newly selected requirement. The production refresh reuses saved forcing where available; its fallback builder has a separate conservative padding rule. Do not confuse generic configuration defaults with provenance of these completed runs.

Rebuild future forcing from the target observed event and add buffers once; do not crop a padded forcing and then stack padding onto it. Verify adequate initial settlement and final return for parent and children, and test shorter buffers against existing long runs before adopting them. Beginning at the first-event steady discharge replaces actual antecedent history by an explicit steady-state assumption. Output metrics must extend past the forcing-event end until the routed response settles. Production forcing, buffers and routing have not been changed.
