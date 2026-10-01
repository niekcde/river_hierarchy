# Focused investigations

Run scripts from the repository root with the river-hierarchy-rivgraph Python environment.
These investigations write separate outputs and preserve the original factorial results. Current analysis scripts use `analysis_paths.py` and `configs/active_analysis.json`; `SYNTHETIC_RESULTS_ROOT` can explicitly select another version.

## Selected hydraulic rerun · 30 September 2026

`selected_hydraulic_workflow.py all --workers 4` plans, routes and analyses the
complete original grid and embedding extension with the selected BASED/local-kb
law (alpha=.65, n=.035). Restartable jobs and per-job reference settings are
stored in `outputs/hydraulic_alpha065_20260930`. The original geometry IDs,
validation groups and folds are reused; responses and resistance are recomputed.
Then run `current_synthetic_analysis.py`, `compact_predictor_selection.py`,
`omega_numerical_check.py`, and `embedded_predictor_validation.py` in that order.
`selected_hydraulic_report.py` checks volume/plateaus, unchanged geometry/splits,
35 matching focused comparisons, and writes previous/selected summaries.
`selected_hydraulic_applicability.py` writes the selected reference-flow audit;
it is separate from a transient-discharge or real-network validation.
See [the dated record](../notes/2026-09-30_selected_hydraulic_rerun.md).

The following original investigative runners keep their historical settings
unless explicitly parameterized; their old outputs are not relabeled.

## Width, length, and shortening

`side_channel_width_length.py` tests simple units with selected branch A shares
50%, 20%, 10%, and 5%, and physical length multipliers 1, 1.25, 1.5, 2, and 3.
All 20 parents and the shared collapsed centerline use a common stable clock,
forcing, and sufficient warm-up/recession window. Results are written to
`synthetic_experiments/outputs/side_channel_width_length`.

The shortening comparison reuses the 20/80 and 5/95 networks: the reference
branch A is 15 km, shortened to 12.5 and 10 km. Relative multipliers are 1,
0.8333, and 0.6667. Endpoint separation remains 10 km, branch B remains 10 km,
widths and heads remain fixed, and topology remains unchanged. Physical
multipliers therefore remain >=1; this is distinct from full collapse.

`comparison.csv` contains parent-to-centerline collapse metrics and parent
hydrograph shape metrics. `shortening_comparison.csv` compares same-topology
networks against their 15 km branch reference; resistance is the equivalent
resistance of the split unit. `shortening_resolution_check.csv` repeats these
comparisons at 100 m instead of 200 m segmentation.

Peak screening requires prominence >=1% of the input pulse's excess amplitude;
it is an exploratory threshold, not a physical classification. The t10–t90
width is based on cumulative positive event volume. The after-65-hour fraction
uses interval start times and is a fixed-window diagnostic, not a universal tail
metric. Hydrograph plots show discharge above the common 100 m³/s baseflow.

Inspect `capped_segments`: the narrowest lengthened branches reach the existing
minimum-celerity safeguard. Their length response must be interpreted with that
constraint. All routes enforce the existing stability, spin-up, recession, and
volume-conservation checks. Peak-time changes are timestep-resolved.

## Current three-metric synthesis

`current_synthetic_analysis.py` recomputes plateau event duration for all saved
full-grid pairs, the factorial decomposition, controlled omega residuals, and
a fixed three-model predictor benchmark. It writes `outputs/full_grid_island/
current_analysis` without altering routing or the older shape outputs.

`omega_numerical_check.py` adds the separate seven-geometry robustness audit
(42 pair reruns) to that directory. Run it after the main analysis script.

Read `notebooks/synthetic_results_synthesis.ipynb` first, then
`notebooks/synthetic_predictor_validation.ipynb` for predictive comparisons.
Shared metric definitions: `RAPID/CURRENT_RESPONSE_METRICS.md`.

## Standalone variables and compact selection

`compact_predictor_selection.py` extends the current predictor benchmark using
four development folds. It evaluates all 175 one/two/three-axis combinations,
forward addition and backward removal, then applies the selected sets to the
reused held-out geometry groups. It also evaluates parent-to-child geometric
changes and shortest upstream/downstream context lengths as separate blocks.
Outputs are in `outputs/full_grid_island/current_analysis/compact_predictors`.
The existing `synthetic_predictor_validation.ipynb` now presents standalone
strength, correlations and compact selections before removal diagnostics.
Equations and domain scope are in `GEOMETRIC_PREDICTORS.md`.

## Fixed-unit embedding extension

`embedded_width_length_extension.py --phase plan|pilot|full` validates and runs
the separate extension described in `EMBEDDED_EXTENSION.md`. The default pilot
has nine configurations / 63 comparisons; the full plan has 140 / 980. Existing
routing jobs within this extension are reused when their geometry, settings,
clock and implementation fingerprint match. The original factorial grid is
unchanged.

## BASED–Manning reference consistency

`based_manning_reference_audit.py` reads stored preparation references for examples
03, 21, 55 and 56, verifies BASED depth reproducibility, and compares the implied
Manning discharge with the actual peak-reference discharge. It also evaluates
BASED for the saved synthetic baseline peak reference, without changing routing.
Outputs and provenance are in `outputs/full_grid_island/applicability/based_reference_audit/`;
the existing hydraulic applicability notebook presents them. Width-refresh markers
are recorded explicitly; this does not certify current branch-product freshness.

## Focused Manning routing sensitivity

`manning_sensitivity.py` resumes exact cached runs for seven configurations and
all seven collapse treatments. Legacy and BASED-depth-only controls accompany
BASED kb with n=0.025, 0.035, 0.050 and reference-derived n. One kb/n pair is
shared across each configuration's parent and child states. Results live in
`outputs/manning_sensitivity/`; the existing hydraulic applicability notebook
contains the figures. Infeasible shared all-state clocks are recorded explicitly
in `infeasible.csv`, without modifying segmentation, X or celerity limits.

## Segmentation and timestep sensitivity

`segmentation_sensitivity.py` compares uniform 200/100/50 m segments, selective
slow-edge subdivision at those maximum lengths, and the actual real-workflow
stability-length helper with its 200 m floor. Geometry, n=0.035, BASED kb, forcing
and X remain fixed. Three configurations and all seven collapse treatments are
covered; infeasible clocks remain explicit. It uses an isolated preparation
binding with the existing routing solver and checks parity against the stock
adapter. No production code is modified. Outputs live in
`outputs/segmentation_sensitivity/`, with figures in the existing hydraulic
applicability notebook. Theoretical path moments and measured excess-flow moments
separate mean travel time, discretization spreading, and between-path arrivals.

`stability_edge_audit.py` traces each timestep bound to canonical edges in the
seven focused configurations at n=0.035 and BASED kb. It saves source edge labels,
width, depth, slope, length, allocation share, celerity clipping and controller
flags under `outputs/segmentation_sensitivity/edge_audit/`. It does not route or
change numerical settings; the hydraulic notebook presents the audit tables.


## BASED-anchored width–depth scaling

`width_depth_scaling.py` preflights the original factorial grid, planned embedded
extension and four stored real base networks, then routes seven focused synthetic
geometries at depth exponents 1, 0.8, 0.71, 0.65, 0.6 and 0.5. The BASED reference stays fixed;
local kb changes depth/celerity and is also used in the matched Chezy resistance
solver. Width allocation stays unchanged. Native and common-clock controls,
old-method parity, conservation and plateau checks are included. Results and
cached hydrographs are in `outputs/width_depth_scaling/`; read the appended section
in `notebooks/synthetic_hydraulic_applicability.ipynb`. This diagnostic does not
change production preparation or adopt an exponent. See the dated checkpoint
reference **HYD-2026-09-29-WIDTH-DEPTH** for findings and limitations.

`width_depth_failure_review.py` summarizes configuration, deduplicated-network
and parent/child-clock failures; identifies failed parameter combinations;
reports counterfactual exclusions without removing any cases; and exports real
edge-controller provenance and normalized geometry ranges. Run it after the
scaling diagnostic. Reference: **HYD-2026-09-29-ALPHA06-FAILURE-REVIEW**.

On 30 September, alpha=.65 was selected for synthetic experiments in
`configs/selected_hydraulic_reference.json`. `width_depth_scaling.py` consumes
that selection and includes it in the sensitivity set. The selected exponent
is not automatically applied to the existing original-grid runner or its saved
outputs. Real-case selection awaits new runs using updated slopes.


## Timing response alternatives

`timing_response_comparison.py` compares seven timing definitions on the saved selected grid and extension, with parent Leq as the unit normalizer. It writes an isolated `current_analysis/timing_response_comparison/` analysis, using the original development/holdout groups and no extension fitting. It includes controlled embedding, fixed predictor sets, a separate delta-width-dominance addition and a centroid diagnostic. The current synthesis and predictor notebooks present it. No observed gauges exist for synthetics; centerline-reference variants are sensitivity checks only. See [SYN-2026-09-30-TIMING-OPTIONS](../notes/2026-09-30_timing_response_options.md).
