# Synthetic results: current analysis structure

**Real-case comparison, 5 October 2026:** [Deduplicated four-response benchmark](../RAPID/validation/deduplicated_response_benchmark_20261005/README.md). Distinct island operations are retained; matched original-grid transfer remains weak. This test does not include the wider embedding extension in training.

**Review index, 5 October 2026:** [Completed investigations, findings and remaining real-case checks](notes/2026-10-05_review_inventory.md). Includes the distinction between strong original-grid interpolation and poor wider-extension transfer.

**Update, 30 September 2026:** [Selected hydraulic rerun](notes/2026-09-30_selected_hydraulic_rerun.md) refreshes the full grid, extension and current notebooks at alpha=.65 / n=.035. Real-case validation remains deferred.

**Earlier decision record, 29 September 2026:** [SYN-2026-09-29 checkpoint](notes/2026-09-29_synthetic_checkpoint.md). Sensitivity/predictor exploration is paused pending additional real-case comparison; hydraulic applicability is the active workstream.

Start at [synthetic_results_synthesis.ipynb](notebooks/synthetic_results_synthesis.ipynb).
The metric contract is [CURRENT_RESPONSE_METRICS.md](../RAPID/CURRENT_RESPONSE_METRICS.md).
The current output directory is `outputs/hydraulic_alpha065_20260930/full_grid_island/current_analysis`, resolved through `configs/active_analysis.json`.

## 1. What changes the responses?

Use all 2,187 configurations and seven collapse treatments with three responses:
peak lag, 1% plateau event duration, and peak excess. Show:

1. Main and total variance shares: nonnegative 0–100%.
2. Direction: signed matched high-minus-low response changes in percentage
   points, averaging over identical settings of the other six factors.
3. All 21 unique factor pairs, after subtracting their main effects.
4. A few conditional response curves selected by interaction strength.

Do not attach signs to variance shares. Do not clip positive peak changes above
100%. Treat this as a deterministic decomposition of this discrete grid, not
uncertainty estimates or population-wide river sensitivity.

## 2. What geometry predicts the responses?

This is a separate result, eventually shown alongside a separate real-network
validation. Fit one model across collapse treatments, without treatment labels
as inputs; report R² separately for each treatment and response. Compare:

- omega;
- its three separate components;
- omega plus domain size/proportions and width dominance;
- the extended geometry/slope set;
- the full set with one component removed at a time and refitted.

Compare fixed regularized linear, additive spline, and shallow boosted models.
Use identical geometry-group splits. Reusing the earlier holdout makes this an
exploratory comparison, not a new untouched validation. Leave-collapse-type-out
is an additional synthetic transfer check. Neither is independent real-river
validation. Negative R² values must remain visible.

Component-removal losses measure added predictive information conditional on
other variables and the model. Correlated geometry can substitute for removed
features; these are not unique physical causal contributions. Main factorial
effects, model ablation and predictor residuals answer different questions.

Historical predictor studies already included boosted trees, scalar spline/Ridge
product fits, and a linear Ridge real-case pilot. The new benchmark compares
three model families systematically on the same current response definitions.
Do not compare earlier family-specific scores directly with these pooled fits.

## 3. Why does omega leave vertical bands?

The grid contains discrete omega values, with many configurations sharing each
value. Diagnose within each treatment before mixing treatments. First vary one
factor while fixing the other six: which factors move lag without moving omega?
Then ask which geometry-only descriptors encode the missing scale or arrangement.

Report the exact-within-tolerance omega-band variance and the controlled lag
ranges. Repeat the band diagnostic excluding celerity-capped cases, without
pretending that this filtered sample is still the balanced factorial design.
Do not use an in-sample residual decomposition as a prediction score.

## Avoid learning RAPID-specific behaviour as a universal law

Distinguish three levels of evidence:

1. **Information loss in omega:** demonstrate with controlled cases at constant
   omega. This is a property of this predictor and experimental design.
2. **Mechanism within this routing model:** investigate absolute length, width,
   head gradient, flow allocation, and pulse duration. Geometry-only predictors
   may use geometry and slope; timing, cap flags, model X and forcing duration
   remain diagnostic controls rather than predictor inputs.
3. **Numerical/model robustness:** check spatial refinement and admissible
   timesteps separately from changes to Muskingum X and forcing duration.
   Convergence alone does not establish hydrodynamic validity. A relationship
   depending on X is conditional on the routing model; one depending on pulse
   duration is conditional on forcing. Both must be stated explicitly.

The small audit in `omega_numerical_check.py` varies width, slope or network
extent in nested/full at fixed fractions/length multiplier. It uses 200/100/50 m
segmentation with matching length floors, a larger stable timestep, X=0.2, and
a doubled input pulse. These reruns are not added to the balanced grid.

## What remains before a final result

- The development-only compact search now supplies a three-axis benchmark and
  a development-selected near-best candidate within the original pool (six inputs after the hydraulic revision). Parent-to-child
  change descriptors improve the larger model; next reduce that added group
  and test transfer before treating any candidate as final.
- Follow up the cases that fail refinement, and distinguish residual geometric
  effects from discretization/length-floor effects before physical attribution.
- Freeze metrics and predictors, then validate on additional real networks,
  including island cases, with the new duration definition.
- Extend the synthetic range toward small domain/network length fractions;
  the earlier real pilot motivated the separate embedding extension. Its full 140 configurations / 980 comparisons are now rerun at the selected settings; real overlap must be reassessed after updated real slopes.
- Only then decide which compact figures and claims belong in the final paper.

Historical notebooks remain evidence for earlier definitions. They are not
additional competing starting points. The synthesis is the current entry point.

## Completed compact-selection extension

The predictor notebook now includes single-variable held-out strength,
within-treatment development correlations, a bounded small-subset search and
separate tests of child-geometry changes and outside lengths. The removal-loss
plot remains a conditional-information diagnostic, not the selection method.
The original nine-input reference remains visible. See
[the full equations](GEOMETRIC_PREDICTORS.md) for scope and algebraic overlap.

## Integrated width/length and embedding extension

The existing results synthesis (section 5) contains the extension setup, omega–lag range comparison, three response plots and numerical flags. The existing predictor validation contains unchanged original-development fits evaluated on novel extension comparisons, with anchors excluded and no extension fitting. The full 140-configuration extension is complete under the selected hydraulic setup and remains separate from the original balanced factorial analysis. The 63 original-grid anchors are verified and excluded from scores. Extension outcomes do not select predictors.


## Historical hydraulic parameterization: width–depth scaling diagnostic

The hydraulic applicability notebook now tests BASED-anchored local kb scaling
at alpha=1/.8/.71/.6/.5 with unchanged 200m segmentation and n=.035. Stronger
moderation resolves the synthetic extension's coefficient conflicts, but changes
all three response metrics and does not eliminate branch discharge-consistency
mismatches. Two of four stored real base networks remain infeasible at alpha=.5
under the nominal audit settings. No exponent had been adopted at that diagnostic stage; the later selection below supersedes this status. See checkpoint
**HYD-2026-09-29-WIDTH-DEPTH** before freezing a hydraulic baseline or rerunning
main synthetic and real analyses.

The alpha=.6 extension passes every planned synthetic configuration. At .71,
only A=.05/m=3 fails (four extents); optional exclusions are documented but not
applied. Real examples21/55 retain low-slope controller conflicts at .6. Inspect
those raw-WSE slopes and physical depth evidence before choosing an exponent or
restricting the experiment domain. Detailed counts distinguish configurations,
unique network states and parent/child clocks in the hydraulic notebook.


## Decision · 30 September 2026: alpha=0.65 for synthetic experiments

User selected alpha=.65; this supersedes the earlier open .71/.6 candidate
choice for synthetics. See `configs/selected_hydraulic_reference.json` and
checkpoint **HYD-2026-09-30-SELECTED-ALPHA065**. The focused diagnostic includes
the selection. A separate versioned rerun now covers the complete main grid and extension, and the current notebooks have been refreshed. Original outputs are preserved. The real-case
choice explicitly waits for new runs using updated slopes, so old stored-base
controller findings are historical context only. Retain the depth/allocation
consistency limitation when interpreting the selected parameterization.


## Timing-definition comparison · 30 September 2026

See [SYN-2026-09-30-TIMING-OPTIONS](notes/2026-09-30_timing_response_options.md). The existing notebooks now compare network-relative timing, absolute timing, apparent speed and parent-Leq normalization. Parent-Leq normalization is promising for local-impact comparison but does not eliminate peak reshaping effects. Centerline-reference formulas are synthetic sensitivities, not observed gauge calibration. Existing primary metrics and factorial responses remain unchanged until the timing interpretation is chosen; then rerun compact selection allowing change descriptors to replace original inputs.
