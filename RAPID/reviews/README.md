# Hydrograph matching review

**Reference: HYDROGRAPH-MATCH-2026-10-05.** Created after the visual review of all 16 gauge events showed that first-maximum times, plateau spans and potential tributary-driven features needed separate treatment.

`hydrograph_match_review.xlsx` is the case-review record. Yellow cells preserve the user's observations as editable notes. Measured comparisons, interpretation and follow-up checks occupy separate columns. The workbook includes the original/global maxima, connected near-peak windows, three event-start definitions, SWORD feeder evidence, distributed routing-time diagnostics and hydrograph figures. No primary response metric or routed result was replaced.

The initial root workbook was empty on disk and had an Excel lock file. It was left untouched to avoid losing an open session's unsaved changes. Use the populated workbook here for subsequent reviews.

## Evidence and reproducibility

- Production: `shared-head-anchored-depth-signed-inflow-20261003`.
- Generate numerical evidence with `RAPID/investigations/hydrograph_match_review.py`.
- Assemble interpretations with `RAPID/investigations/prepare_hydrograph_review_workbook.py`.
- Evidence: `RAPID/validation/hydrograph_match_review_20261005/`; earlier peak-window figures: `RAPID/validation/peak_windows_20261005/`.
- The workbook builder is `RAPID/investigations/build_hydrograph_review_workbook.mjs`. Run a copy in a temporary directory with `node_modules` linked to the installed `@oai/artifact-tool` runtime; set `RIVER_HIERARCHY_ROOT` to the repository root. It rebuilds the workbook snapshot. **Do not rebuild over subsequently edited reviewer notes without preserving them first.**
- Network selection and actual parent/child geometry plots remain in `RAPID/notebooks/rapid_hydrograph_review.ipynb`. `UNIT_SELECTION=None` selects all eligible units/states with an explicit six-pair default display limit.

## Interpretation limits

The 95% windows mean 95% of maximum discharge, not the 95th percentile or confidence bounds. Broader windows can contain substantially delayed peaks. Tributary drainage areas support potential lateral input but do not supply tributary event hydrographs. The example 05 shoulder and example 11 bump are retained exploratory features, not replacements selected to maximize agreement. Example 48's four daily features are provisionally paired by chronology; their matching uncertainty remains explicit.

Case 46 remains materially late: about 204 hours versus the first observed maximum. Its approximately 339-hour analytical routing residence is distributed across a roughly 775-km corridor. Effective-length segmentation inflation contributes only about 3.3 hours. These diagnostics identify a distributed travel-time problem; they do not establish that any specific slope or depth is wrong.

Changing observed references leaves model-only parent–child differences unchanged but changes gauge-normalized comparisons. Freeze any adopted matching convention before updating those comparisons.

## Proposed event-window review — 5 October 2026 (not implemented)

Reference **HYDROGRAPH-MEAN-SPREAD-2026-10-05**. See the [gauge-mean and example-37 spread audit](../validation/hydrograph_mean_spread_20261005/README.md). Proposed future Excel fields: target event/peak IDs; proposed forcing start/end; pre-event warm-up; post-event recession/settlement buffer; observation cadence/gaps; justification; review status; and full-versus-cropped routing comparison. No workbook has been edited for this proposal.

Shorter forcing windows could reduce routing cost and isolate the intended event. Choose windows from the observed event, not predictor agreement. Preserve enough antecedent forcing to reproduce initial storage and enough trailing forcing to capture delayed peaks and final settlement, especially example 46. Parent and children must use the same forcing and analysis window. Validate against the existing full runs before replacing production inputs. Merely cropping plotted output does not save routing cost; it also changes mean/SD statistics. Do not silently clip delayed model peaks to the observed gauge window.

## User-supplied target windows — 5 October 2026

The 16 proposals are recorded in [proposed_event_windows_20261005.csv](proposed_event_windows_20261005.csv). Newly named end dates include the entire date; unchanged boundaries preserve their configured timestamp. These are target windows, not yet production forcing limits. Example 41 is confirmed as July 5; example 46 requires trailing output beyond September 11. See the [standard-sampling TV audit](../validation/standard_total_variation_20261005/README.md). The Excel workbook remains unchanged.

The user confirmed these are event boundaries. Initial/final constant-discharge buffers are additional and already exist in production; buffer reduction requires a settlement/convergence check, separately from cropping the observed event.

## Buffer tests completed — 5 October 2026

[EVENT-BUFFERS-2026-10-05](../validation/event_buffer_tests_20261005/README.md) audits all 2,230 existing routed states and tests cropped-event forcing on 39 selected states across all 16 examples. Full-waveform 0/6/24-hour spin-up comparisons agree within numerical precision under the current steady-flow initialization. A 0.1%-of-parent-amplitude settling criterion sustained for 24 hours plus a further 24-hour tail gives an estimated 83% reduction in aggregate simulated duration and 77% in segment-routing substeps. These are projections, not measured batch wall-time savings. Tested children can settle later than the parent, so every future child requires the final-settlement check. Production remains unchanged; the new windows/buffers are not yet activated.

## Exact routing schedule added to Excel — 5 October 2026

The populated `RAPID/reviews/hydrograph_match_review.xlsx` now contains **Routing windows**. Requested event dates: **B7:C22**; retained sample dates: **D7:E22**; plateau discharge: **F7:G22**. The tested simulation stages occupy **B28:G43**, with simulation start in **B28:B43** and final simulation end in **G28:G43**. Current production dates and duration comparison: **B49:G64**. All times are UTC to the second. These are tested proposals, not activated production inputs. Original review values and formatting were preserved.

See [Routing stages and exact UTC schedule](routing_windows/README.md) for the complete report, date conventions, protocol and cell map. This worksheet addition supersedes the earlier note that the workbook had not yet been updated.

## Calibration target proposals — 6 October 2026 (not implemented)

The user’s case-by-case crest/plateau timing proposals and the proposed common review protocol are recorded in [calibration_target_proposals_20261006.txt](calibration_target_proposals_20261006.txt). Current targets, calibrations and runs remain unchanged by explicit user instruction. The preliminary fixed-X/Cunge comparison is in `RAPID/outputs/cunge_fixed_comparison_20261006/REPORT.txt`.

## Formal calibration target specification — 6 October 2026 (not applied)

The observation-based proposals now have a [formal specification](calibration_targets_20261006/SPECIFICATION.txt), [machine-readable UTC targets](calibration_targets_20261006/targets.json), and an [interactive observation review](calibration_targets_20261006/review.html). All sixteen nominal UTC targets are specified, including explicitly proposed interpretations for 37 and 45. The user selected event 48’s middle equal maximum at **2025-06-02 23:45 UTC**, 90 minutes after the previous target; the earlier broad-crest alternative is not adopted. Crest intervals are representativeness intervals, not confidence intervals. Case 11 remains diagnostic and unresolved.

The future calibration objective is downstream peak UTC minus downstream target UTC. Prepared-inlet and observed-upstream peak timestamps are recorded separately because the current lag-based objective can introduce an offset when they differ. The table has not been connected to calibration or routing. Generate these isolated review artifacts with `RAPID/investigations/formalize_calibration_targets.py`; do not overwrite later reviewer edits without preserving them.

## Reviewed-target base calibration completed — 6 October 2026

The target specification above has now been applied in a separate [reviewed UTC calibration](../outputs/cunge_calibration_reviewed_20261006/REPORT.txt), preserving the historical outputs and the original target-file hash. See the [updated interactive hydrographs](../outputs/cunge_calibration_reviewed_20261006/calibration_hydrographs.html) and [hierarchy reuse/rerun plan](../outputs/cunge_calibration_reviewed_20261006/hierarchy_rerun_plan.csv). The earlier “not applied” statements describe the specification at creation; no production hierarchy inputs have yet been switched.

Both methods were calibrated on all 16 base states. Thirteen Cunge cases meet the original observation-sampling tolerance; 55 is limited by the reporting clock and passes that tolerance under diagnostic peak interpolation. Case 11 remains unresolved. Case 54 fits the observed crest span but cannot match its nominal midpoint with the current global-maximum observable: two competing crests exchange dominance. Its full-hierarchy run is held for target review. All 16 Cunge base prep bundles passed full engine verification; 14 relevant tests passed.

For Cunge hierarchies: reuse 02; rerun 03, 05, 21, 37, 45, 48, 55; first runs for 13, 15, 22, 41, 46, 56; retain 11 as diagnostic and hold 54. The hierarchy runner still defaults to the previous calibration; update its calibration source and the previously reviewed shared-reference-flow acceptance policy before launching. Existing hierarchy outputs remain intact.

## Subsequent target correction for 45

The user corrected 45 to **20 June 2024, 20:20 UTC**, with crest interval **20:00–20:40 UTC**, and accepted all other targets. Both hydrograph figures and the live target table are updated. The completed calibration snapshot remains unchanged for reproducibility; 45 needs a target-specific refit before using its parameters in the next batch. See `cunge_calibration_reviewed_20261006/target_revision_status.json` in outputs.

## Case 45 refit and sequential batch ready

The corrected 45 target was refitted for both methods; both produce a sampled downstream peak at **2024-06-20 20:24 UTC**, four minutes after the target, and full Cunge engine verification passed. The first reviewed snapshot is retained under `outputs/cunge_calibration_reviewed_20261006/before_case45_correction`. All 14 requested input preflights (excluding 02 and 11) passed; evidence is in `outputs/cunge_reviewed_batch_preflight/summary.json`. Network 54 is included at the user’s request with the crest-span-only caveat retained.

Use `RAPID/investigations/cunge_full_hierarchy_reviewed.py` with the reviewed comparison table, `--overwrite --resume`, and the external output root. It accepts only previously reviewed, hash-matched Q/X-redistribution transitions, verifies flow conservation, and keeps the remaining numerical checks. Each network is staged and validated before replacement. Old summaries and outlet hydrographs are archived; large old prep/run arrays are removed only after a successful replacement. `--retire-legacy-root RAPID/outputs/cunge_full_hierarchy_pilot` also retires the replaced local 03/05 arrays; 02 remains untouched. No full hierarchy jobs were started during preparation of this command.
