# Synthetic restart protocol — 26 October 2026

Reference: **SYN-CUNGE-2026-10-26**. Written after the user identified
`RAPID/outputs/uncapped_event_window_20261020/current_model_comparison_20261023/settings.json`
as the current predictor specification and authorized the next synthetic update.

## Status and purpose

This is the restart protocol and numerical pilot, not a completed new factorial
experiment. September routing and its active selector remain unchanged. The
synthetics will (1) isolate geometry effects on centroid timing, (2) explain the
current predictors, and (3) test their limits under hydraulic/forcing changes.
Current real results comprise 2,170 regular and 15 segmented full-collapse
comparisons across 15 networks. The response is signed child-minus-parent full
excess-volume centroid shift, not peak lag. Current predictor settings define
nine-input geometry and ten-input depth-informed random forests. Later validation
and SHAP work did not replace these artifacts.

## Fixed reference for the pilot

- Existing simple, nested, crossover and centerline originals; seven collapse
  comparisons, including nested main-island-only collapse.
- BASED depth from total width, reference slope and Qref=1000 m3/s, shared across
  states in a configuration. Local depth exponent 0.65 remains prescribed.
- Constant-reference Cunge through `rapid_tools.cunge`; n=0.025, beta=1 are
  **provisional controlled reference values**, not calibrated synthetic parameters.
- Exact physical lengths, alpha=2 numerical subdivision, frozen grid designed
  across n=0.01–0.10, followed by fourfold subdivision selected after the
  initial pilot failed its refinement screen. This alpha is distinct from the depth exponent.
- Lower celerity bound 0.05 m/s, no active upper bound. A finite implementation
  ceiling exceeds every raw speed throughout the grid's n range and is checked.
- Width-proportional conservative flow allocation recomputed per state. Geometry
  unchanged does not imply unchanged Qref if its hydraulic role changes.
- Steady initial flow, two-time inlet, signed-inflow coefficient policy with
  native nonnegative-flow audits. No lateral inflow.
- Shared native clock within each configuration, at most 15 seconds, and exact
  60-second interval-mean forcing/output. Pulse: 100 to 1000 m3/s in 12 hours,
  falling to 100 in 24 hours; 24-hour initial plateau.
- Shared tails extend until every state is within 0.1% of inlet event amplitude
  for at least 48 trailing hours **and** full-tail excess-volume error is below
  1e-4. The amplitude-only stopping rule was insufficient in the narrow-branch pilot.

Pilot configurations: baseline; outer A=0.05 with multiplier 3; outer A=0.95
with multiplier 3. Length perturbations retain the generator's family-specific
targets (simple A, nested first inner branch, crossover edge).

## Response convention

Use the shared `rapid_tools.hydrograph_response.distribution` implementation:
centroid = integral(t * excess)/integral(excess). Time units in the output response
are hours. Negative shift means an earlier child centroid.

The synthetic baseline is the known constant 100 m3/s. For each original and all
its children, interpolate first/last crossings of 0.5% of the original outlet
excess amplitude, then use the earliest start/latest end as common boundaries.
This is an explicit synthetic counterpart to reviewed real-event shared windows;
it does not claim to reproduce their manual boundary selection. Repeat at 1%
and 2%. Retain endpoint excess; never force endpoints to zero or silently clip
negative excess. A common baseline correction of at most 1e-8 m3/s is allowed
solely for floating-point roundoff and recorded in every response row; larger
corrections fail. Save duration80 and excess-volume changes as diagnostics.

Native negative flows, negative excess in the analysis window, or censored
crossings fail explicitly. Report any numerical pulse undershoot separately.
The pilot compares half timesteps and doubled spatial subdivision. Its provisional
screen is <1 minute difference in centroid-shift responses, <0.1% inlet-amplitude
wave difference and <1e-4 full-tail relative volume error. These are numerical
screens, not measured uncertainty or proof of field applicability.

## Full experiment after pilot review

1. Reuse the original seven-factor 3^7 grid (2,187 configurations, seven
   comparisons each). Recompute unique job identities under new routing.
2. Keep the 140-configuration uneven-width/embedding extension separate from
   balanced factorial variance decomposition. Retain fixed-unit embeddings.
3. Specify a physically interpretable shortening extension separately: an edge
   shorter than endpoint distance is not a literal channel geometry.
4. Selected hydraulic/forcing sensitivity: n=0.015/0.025/0.045/0.065,
   beta=0.5/1/2 and half/original/double pulse durations. These are proposed
   sensitivity levels, not an instruction to multiply the entire grid.
5. Add representative multiple-peak forcing only as a separate robustness test.

## Predictor bridge and validation

Before prediction, reproduce the current feature definitions and parent/child
B-domain scopes exactly. Read the frozen settings file for column names and
artifact locations; do not replace missing inputs by similarly named September
variables. Verify depth provenance and the explicit-depth resistance/Delta Gh
formulas. A matching number of columns alone is insufficient.

Apply the frozen real models to the new synthetics first. Report predictor-range
coverage/extrapolation by family and geometry. Any later model development must
retain grouped synthetic holdouts and whole-real-network holdout; known-network
mixed-family validation answers a different question. Do not combine training
and test geometries through aliases or related collapse states.

The predictor bridge, new full-grid run, forcing/hydraulic sensitivities and
updated full-result notebooks remain pending. Old R2 values are historical.
The current pilot does not load or refit the forest artifacts.

## Reproduce and inspect

From the repository root:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
/opt/anaconda3/envs/river-hierarchy-rivgraph/bin/python \
  synthetic_experiments/investigations/cunge_pilot.py
```

Configuration: `configs/cunge_pilot_20261026.json`.
Outputs: `outputs/cunge_pilot_20261026_refined/` (or `--output <new-directory>`).
Inspect `completion.json`, `numerical_checks.csv`, `responses.csv`, and each
configuration's `audit.json` / `hydrographs.parquet`. Source hashes reject reuse
of a directory after tracked inputs change. The pilot is deliberately small;
repeating unchanged inputs reruns the pilot rather than implying batch resume.

Promote a new active analysis version only after full routing, response/feature
parity and numerical checks are complete. Refresh existing synthesis, interaction,
predictor-validation and applicability notebooks then; do not relabel old results.

## Completed pilot evidence

The refined pilot completed all three configurations, 11 state references each,
and seven comparisons each (21 comparisons; 63 response rows across three windows).
All used a 15 s native timestep and 60 s reporting. No active selector or real
model artifact was changed. These state references are not deduplicated job counts.

The initial alpha=2 subdivision alone failed the numerical screen: doubling
subdivision changed the dominant-long centroid response by 2.68 minutes and
waveforms by up to 0.553% of inlet amplitude. Fourfold subdivision of that initial
grid reduced the next doubling difference to **0.176 minutes (10.53 seconds)**,
and **0.0332%** of inlet amplitude. Half-timestep differences in centroid shift
were below 1e-6 minutes. This supports the refined pilot, not automatic convergence
of every future grid configuration.

Amplitude-only settlement left 0.53% of excess volume unaccounted for in a long
narrow branch. Requiring volume balance as well extended the shared narrow-long
tail from 96 to **768 hours**; the baseline needs 96 hours and dominant-long 192.
The maximum remaining full-tail excess-volume error is **0.00970%**, below the
0.01% pilot limit. These long tails are a routing/reference-hydraulic diagnostic;
we have not tuned n, beta or geometry to remove them. Pilot runtime does not yet
establish full-grid cost.

The 0.5% centroid shifts include small delays for some narrow-long collapses,
while baseline and dominant-long comparisons advance. These are provisional
finite-window pilot responses. Interpret alongside the 1%/2% window sensitivities
and full hydrographs; they are not a new real-network predictor result.

Shared Cunge and response regression tests: **11 passed**. Additional sparse-engine
parity and baseline 1-vs-24-hour spin-up results are saved in
`outputs/cunge_pilot_20261026_refined/additional_verification.json`.

Earlier failed pilot outputs remain separately dated diagnostic directories;
`cunge_pilot_20261026_refined` is the completed screened pilot. The remaining next
step is the exact current-domain/feature adapter and frozen-model transfer check,
followed by configuration-wide numerical preflight before the factorial launch.

The zero-plateau probe changes the initialized discharge because its first
interval is already on the rising limb (100.625 rather than 100 m3/s). It differs
by 0.596 m3/s and is not a matched initialization test. The 1-hour versus 24-hour
plateau check retains identical initial discharge. Keep the explicit plateau.

## Predictor bridge and grid preflight completed — 26 October 2026

This update supersedes the earlier pending-adapter paragraph above. See
[the bridge report](outputs/current_predictor_bridge_20261026/REPORT.md) and
[the frozen prediction figure](outputs/current_predictor_bridge_20261026/frozen_transfer.png).

- Recomputed all required model inputs for 18 real reference cases across all
  15 networks, including three containing-compound checks. Matched saved features;
  four additional semantic tests passed. Current forests were loaded unchanged.
- Applied both frozen forests to 14 eligible synthetic pairs each. The seven
  equal-length baseline pairs have an undefined route-spread ratio (0/0), so they
  are explicitly retained as prediction abstentions rather than imputed.
- Pilot MAE: geometry model 37.36 minutes; depth-informed model 44.23 minutes.
  These stress-case errors are not representative validation scores. Forcing,
  hydraulic settings and joint predictor coverage differ from the real cohort.
- All 2,187 original factorial configurations passed input/clock preflight, with
  15-second clocks and 7,236 provisional unique parameter/clock jobs. Final shared
  windows must enter deduplication before a production plan can be frozen.
- Four selected grid extremes (28 additional comparisons) passed routing,
  settlement, volume and refinement screens. Maximum refinement change was 0.094
  minutes. Two needed 384-hour tails. This is spot-check evidence, not full-grid
  hydrograph verification. The 140-configuration embedding extension has not been
  preflighted in this step.

At least 5,589 of the 15,309 grid comparisons have an undefined spread or width-CV
ratio for the unchanged frozen predictors. These are valid controlled geometries,
not invalid routing cases. Preserve them for sensitivity analysis. Direct transfer
uses the explicit eligible subset; revising descriptors or ratio conventions
requires a separate version and revalidation on real data. Do not alter geometries
or add tiny denominators simply to obtain finite predictions.

The full grid has not been routed. Conditional in-memory single-core throughput
estimates range from 1.2–1.5 minutes at a common 96-hour tail to 12.1–15.3 minutes
at a common 1,536-hour tail. They exclude preparation, file I/O, repeated tail
extension, feature generation and verification; they are not end-to-end schedules.
Outlet/time storage estimates range from approximately 1.0 to 10.3 GiB for those
scenarios. See `outputs/cunge_grid_preflight_20261026/conditional_cost.csv`.

Reproduce the bridge, preflight and report from the repository root:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /opt/anaconda3/envs/river-hierarchy-rivgraph/bin/python synthetic_experiments/investigations/current_predictor_bridge.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /opt/anaconda3/envs/river-hierarchy-rivgraph/bin/python synthetic_experiments/investigations/cunge_grid_preflight.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /opt/anaconda3/envs/river-hierarchy-rivgraph/bin/python synthetic_experiments/investigations/report_cunge_bridge.py
```

The stress selection/configuration and source hashes are retained at
`outputs/cunge_grid_preflight_20261026/stress_config.json` and
`stress_runs/manifest.json`. It uses the same pilot runner with that configuration;
no distinct numerical implementation was introduced.

## Notebook review added — 26 October 2026

The existing `notebooks/synthetic_experiment_examples.ipynb` now ends with the
core-grid/settings table and a read-only representative-case review. It retains
all previous geometry panels, including the combined seven-case panel. Its 20 km
reference length is now explicit rather than inherited from a changed generator
default. Historical fixed-X text is clearly distinguished from the current pilot.

Controls: `REVIEW_CASES`, `REVIEW_COMPARISONS` (or None), `MAX_REVIEW_PAIRS`, and
`SHOW_FULL_TAIL`. The default view is simple/full for the three saved pilot cases.
The 5/95, 95/5 and 3× length stress cases are labelled outside the core grid;
showing their existing outputs does not expand the agreed experiment. The
notebook includes geometry, hydrographs, signed centroid and sampled peak shifts,
parent/child descriptor changes, and optional frozen-prediction diagnostics.

Peak shift is the difference between first sampled maxima inside the same shared
window used for the centroid. Tied maxima retain their last occurrence for
inspection. All comparisons use 60-second interval-mean output; no interpolated
peak or new response normalization is introduced. Full saved tails can be shown
without changing the analysis window.

The notebook was executed in-process with IPython and static image formatting
because the environment disallowed Jupyter kernel sockets. All code cells
completed and static outputs were saved. All 21 pilot pair centroids were checked
against the saved responses, all selectable figures rendered, and all 42 frozen
prediction rows joined. No routing or fitting ran; no active results selector or
factor levels changed.


## Agreed next comparison — 27 October 2026

User approved a 2:1 peak/start forcing ratio and three hydraulic approaches after the depth-reference review. Prospective settings are in configs/depth_comparison_20261027.json: retain peak 1000 m3/s and use start/end 500 m3/s, preserving pulse durations. The completed 26 October pilot and its configuration remain historical 10:1 results; do not relabel them as 2:1 or overwrite their outputs. The comparison config is a design record, not a completed production runner.

Compare fixed BASED at peak, fixed Manning at midpoint flow, and a validated mass-conservative variable-Manning method. The naive three-point screen is diagnostic only because it failed excess-volume conservation. Real-case peak-based BASED preparation remains unchanged and is recorded as a method choice.

For this comparison, widths, lengths, slopes, topology, n and beta remain fixed within each run; outgoing allocation fractions remain proportional to widths. Local discharge varies over time and allocated flow amounts follow those fixed fractions. This is not dynamic hydraulic flow partitioning. Exposure/allocation descriptors share this fixed-fraction logic, but exposure itself is not a timestep solver. Under variable Manning, local h, c, D, K and X change; channel geometry remains fixed.

Equal-length/equal-slope wide-rectangular branches with width-proportional flow have equal Q/B, hence equal Manning depths, celerities and diffusivities. If their widths sum to the collapsed width and collapsed length and slope are unchanged, their combined response equals the single-channel response in the ideal equations; numerical discretization may introduce small differences. This invariance is a useful check, not a universal claim about real bifurcations. The model omits junction losses, nonrectangular cross-sections and transverse/backwater dynamics. Shortening the collapsed path or changing its endpoint-consistent slope breaks this equivalence.

## Integrated hydraulic comparison and corridor widening — 27 October 2026

This implementation supersedes the earlier statement that a conservative variable-depth solver is still missing. The exploratory `variable_depth_screen` remains a historical, rejected nonconservative diagnostic. It is not used by the integrated runner.

### Geometry and invariants

`Design.branch_width_factor` defaults to 1 for backward compatibility. It scales all original channels within the outer 25–75% test corridor once. It does not compound at nested bifurcations. The 0–25% approach and 75–100% exit retain `total_width_m` in every original and child state. Thus 500 m inlet, two 300 m branches, and 500 m outlet are supported. The area-preserving analytical collapse gives a 600 m single channel within the test corridor, with 500 m approach and exit. It does not forcibly narrow that child to 500 m.

Geometry validation checks the declared expansion and contraction at the outer anchors, unchanged width budgets at internal junctions, inlet/outlet widths, and endpoint head losses. Each of the seven parent-child selections still preserves baseline sum(WL). The uniform original centerline remains a control, not an alias for the expanded full-collapse state when the factor differs from 1.

### Three routing methods

- `fixed_based`: peak-based BASED reference depth, width^.65 branch scaling, peak-based reference diffusion.
- `fixed_manning`: normal depth and Cunge coefficients frozen at midpoint discharge (750 m3/s for the agreed 500–1000 pulse), allocated by local widths.
- `variable_manning`: normal depth from current local discharge, using the conservative storage-consistent MCT formulation (Todini 2007 and corrigendum). Width, slope, length, n and diffusion multiplier beta stay fixed; local discharge, depth, velocity/celerity and diffusion vary. Fractions remain fixed width-based allocations.

The new comparison uses no celerity clipping, to avoid breaking the normal-flow h/Q/c relationship. This is explicit in its config and does not change the real or historical synthetic production settings. n=.025 and diffusion multiplier beta=1 are the current comparison reference, not new real-case calibration choices.

MCT uses wide rectangular normal flow. With q=(I+O)/2, h=(nq/(B sqrt(S)))^.6, v=q/(Bh), c=5v/3 and D=beta*q/(2BS), the cell storage is

    V(I,O) = dx * B * h(q) + D(q)/c(q)^2 * (O-I).

At every native step solve

    V_new - V_old = dt/2 * (I_old + I_new - O_old - O_new).

The scalar positive root uses bracketed Newton iterations with an analytic derivative; failure is rejected, never repaired by clipping or rescaling discharge. This is the storage form of corrected MCT, not the naive changing-coefficient recurrence. At steady flow storage is B*h*dx; its small-amplitude linearization recovers ordinary fixed Cunge. The paper's correction ratio c/v is distinct from our user-set diffusion multiplier beta.

References: https://doi.org/10.5194/hess-11-1645-2007 and https://doi.org/10.5194/hess-11-1783-2007 (read together). This is numerical/formulation validation, not a field or full Saint-Venant validation.

### Execution and results

`run_depth_comparison.py` and package modules `hydraulics.py`/`depth_pipeline.py` provide the supported comparison route. `run_experiment.py` remains the legacy fixed-X workflow. The new runner supports plans, resume with source/input checks, grid expansion and per-case/method filters. The C++ kernel compiles automatically to the temporary cache using c++/clang++/g++; no compiled artifact is committed. NumPy, SciPy, pandas, networkx, XGBoost and the existing project environment are used.

All methods use the same explicit 200 m target segmentation, exact reach lengths and 15 s native sampling by default. Metrics are native-endpoint peak shifts and full-excess-volume centroid shifts; saved visual hydrographs are trapezoidal 60 s interval means. Full centroids require both volume and first-moment convergence. The runner doubles common group tails until event-volume error <1e-6, centroid convergence <1 second, final excess <1e-7 of forcing amplitude, and storage-aware mass residual <1e-7 of event volume. Fixed centroids are checked against sum(allocation*L/c); variable centroids against the previous longer tail. Results that fail are not marked complete.

Completed integration checks: `outputs/depth_methods_20261027` contains 9 groups/63 comparisons (equal outer split, all three width factors and methods); `outputs/depth_methods_stress_20261027` contains 3 groups/21 comparisons (narrow-long, conservative variable method, all width factors). Maximum event-volume discrepancy in these completed groups is 4.6e-9 fraction, approximately 0.00000046%. Independent narrow-long simple/nested/island checks halving dt or dx moved centroids by <.005 seconds and native peaks by <=7.5 seconds. These checks cover selected configurations, not the entire factorial domain.

Updated, executed review sections are in geometry examples, results synthesis, predictor validation, factor interactions and routing results notebooks. Earlier sections remain historical and are labelled accordingly. No new predictor fitting or hydraulic-applicability claim follows automatically from these comparisons.

The factorial plan now has eight geometric factors, including width factor [1,1.1,1.2], and three separate method strata. It contains 6561 geometries, 19683 groups and 137781 comparison references before reuse. There are 28 two-factor geometric interactions per method, not 21; no factorial analysis can be regenerated until the new grid is routed. `outputs/depth_methods_factorial_plan_20261027` is a plan only. The legacy active-analysis selector is unchanged.

### Targeted fixed-versus-variable test completed (27 October 2026)

`outputs/fixed_variable_manning_20261027/REPORT.md` records 126 matched comparisons (84 core-range and 42 stress), all seven treatments and three width factors under a 2:1 pulse. Core median absolute differences are .8735 min for centroid shift and .25 min for peak shift; maxima 6.7937 and 4.5 min. Core centroid relative difference has median 6.55% for fixed effects >=1 min. Overall Spearman agreement is .996 for centroid and .9989 for peak; direction agrees for every pair using a +/-0.25 min zero band. Stress-case maxima reach 36.48 min centroid and 12.75 min peak, so agreement of ranking does not imply identical magnitudes. Both approaches produce a small positive peak shift in the low-slope simple/full case while the centroid advances. These are conditional, deterministic comparisons, not evidence of general validity or justification for an automatic method switch. New executed result sections are in results synthesis and factor interactions notebooks.
