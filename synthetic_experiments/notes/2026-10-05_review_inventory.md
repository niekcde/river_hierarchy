# Synthetic investigations: review inventory

Reference: **SYN-REAL-2026-10-05-REVIEW**. Written 5 October 2026 after the
16-case real hydraulic refresh, to distinguish completed experiments from
outstanding transfer and response-resolution checks. This is an index of
existing evidence, not a new grid run or predictor selection.

Start with [results synthesis](../notebooks/synthetic_results_synthesis.ipynb)
and [predictor validation](../notebooks/synthetic_predictor_validation.ipynb).
The selected results are identified by `configs/active_analysis.json`, pointing
to `outputs/hydraulic_alpha065_20260930`. The old default grid JSON still
contains historical hydraulic parameters; do not use it alone to identify
the settings of the selected results.

## What we investigated

| Investigation | Scope and purpose | Where to review |
| --- | --- | --- |
| Geometry and collapse definitions | Single centerline, simple split, nested split and crossover; simple/full, nested/full, nested/inner, nested/main-island, crossover/full and the two individual crossover collapses: seven comparisons per configuration | [Geometry examples](../notebooks/synthetic_experiment_examples.ipynb) |
| Seven-factor sensitivity | 2,187 configurations; total width 250/500/1000 m, reference slope 5e-5/1e-4/5e-4, outer A share .5/.8/.9, inner share .2/.5/.8, selected-branch multiplier 1/1.25/1.5, network extent 10/20/40 km, internal location upstream/central/downstream | [Synthesis](../notebooks/synthetic_results_synthesis.ipynb), [factor interactions](../notebooks/synthetic_factor_interactions.ipynb) |
| Main effects and interactions | Three responses; nonnegative main/total variance shares, signed contrasts for direction, all 21 factor pairs, conditional curves | Same two notebooks; variance contributions are not signed percentages |
| Uneven widths, long side channels and shortening | Focused A shares .5/.2/.1/.05 and multipliers 1/1.25/1.5/2/3; earlier 5x exploratory argument. Shortening preserves topology and reduces a 15 km branch to 12.5 and 10 km at fixed 10 km endpoint separation. No negative physical lengths | [Focused experiment script](../investigations/side_channel_width_length.py); detailed hydrographs in factor interactions. Historical hydraulics must be distinguished from selected reruns |
| Widening versus secondary peaks | Separate arrivals can widen the flood or produce another peak, depending on width allocation, path times and input pulse. Peak lag alone misses these changes | Factor interactions; verify the selected-hydraulics figures before carrying forward old dramatic examples |
| Event-shape definitions | Replaced primary temporal spread with 1% plateau departure-to-final-settlement duration; parent-derived thresholds shared with child; peak excess measures height above a shared plateau reference; 0.5%/2% sensitivity | [Event-duration verification](../notebooks/synthetic_event_duration_verification.ipynb), [metric contract](../../RAPID/CURRENT_RESPONSE_METRICS.md) |
| Geometric predictors | Resistance, length fraction and exposure separately and as omega; parent/domain scale and proportions, width dominance, signed length–width association, slope; child-change descriptors and surrounding lengths | [Predictor validation](../notebooks/synthetic_predictor_validation.ipynb), [equations](../GEOMETRIC_PREDICTORS.md) |
| Compact model selection | Single-variable prediction, within-treatment correlations, bounded subset search, conditional removal tests; regularized linear, spline and shallow boosted alternatives; grouped splits and collapse-type transfer | Predictor validation. Correlation, removal loss and factorial effects answer different questions |
| Wider embedding extension | 140 configurations / 980 comparisons: A shares .05/.1/.2/.5/.8/.9/.95; multipliers 1/1.25/1.5/2/3; extents 20/100/400/800 km with fixed 10 km unit span and equal approach/exit additions | Synthesis extension panels and predictor extension validation; [configuration](../configs/embedded_extension.json) |
| Timing normalizations | Percentage lag, absolute timing, network-speed change, timing per parent equivalent length, reference-scaled variants, equivalent-speed index | Synthesis timing section; [timing decision note](2026-09-30_timing_response_options.md) |
| Omega bands and residuals | Identify configurations that change response at fixed omega; distinguish information missing from a scalar predictor from numerical effects | Synthesis and numerical audit |
| Numerical robustness | Seven nested/full geometries at 200/100/50 m, separate timestep, X and pulse-duration tests; 42 selected pair reruns | [Selected rerun note](2026-09-30_selected_hydraulic_rerun.md). Not full-grid convergence; refinement also changes effective-length floor |
| Hydraulic applicability | BASED anchors, Manning-implied discharge/roughness, n sensitivity, depth–width exponent and stability boundary, celerity caps, provisional wave-regime diagnostics | [Hydraulic applicability](../notebooks/synthetic_hydraulic_applicability.ipynb); [dated checkpoint](2026-09-29_synthetic_checkpoint.md) |

## Findings to retain, with their limits

The selected full grid has **15,309 comparisons and 9,519 distinct routing jobs**;
the extension has **980 comparisons and 981 jobs**. Both use BASED anchoring,
alpha=.65, n=.035, X=.1 and 200 m target/effective floor. The synthetic upper
celerity cap was 2.524 m/s, whereas the real production cap is now 5 m/s.
No selected synthetic job activated its cap, so that cap difference alone
would not change those saved runs. The clock policies still need an explicit
comparability check; real production now permits signed current-inflow weights.

Changing from the legacy hydraulic setup substantially reduced shape effects.
Selected main-grid median responses were -14.286% lag, -1.056% duration and
+1.329% peak excess. These are joint setup changes, not an isolated alpha effect.

The selected six-variable boosted model had treatment-averaged reused-holdout
R² of **.909 / .713 / .790** for lag / duration / peak excess. Adding change and
surrounding descriptors gave **.941 / .789 / .839**. These are exploratory
within-grid interpolation results, not evidence of a universal predictor.

**The wider extension exposed poor transfer:** the frozen selected shared model
averaged R² **-2.454 / -.138 / -.088** across extension treatments; omega alone
averaged .114 for lag. There were 63 original-grid anchors, excluded from the
917 novel-comparison scores. Do not describe the synthetic evidence only by its
high within-grid R², or retune on these evaluation rows and call them untouched.

Embedding diluted percentage lag strongly. Absolute timing was less dependent
on embedding but was not invariant: among 172 well-resolved geometries, median
relative drift was 23.6%, versus 96.4% for percentage lag. The invariant event
centroid in this linear experiment is a diagnostic, not an adopted replacement
for peak timing. Parent-Leq normalization does not remove peak reshaping.

At 200 versus 50 m, the small numerical audit found maximum changes of 1.31,
.54 and .71 percentage points in the three responses. It does not establish
that extremely small real-network responses are above numerical uncertainty.

## Why real results were needed, and what is now possible

We waited for updated real slopes, consistent local depths/resistance, completed
island runs and the new event-duration response. We can now compare geometry
and response ranges, inspect real omega residuals, and test frozen synthetic
predictors on real geometries. That tests transfer of routing-model responses;
observations validate only the available uncollapsed baseline.

Before final comparisons: resolve sampled zero peak shifts; correct the real
adapter's peak-excess reference to the agreed shared max(initial,final)
plateau; assess observed timing and shape; audit numerical/forcing sensitivity;
then compare frozen synthetic-to-real transfer with separate leave-river-out
real-data fits. Keep identical geometries/aliases grouped and keep zero and
unresolved responses visible. Choose main-text metrics for their scientific
meaning and resolution, not whichever gives the highest R².
