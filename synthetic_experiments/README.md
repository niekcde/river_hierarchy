# Controlled synthetic experiments

New synthetic experiment implementation, separate from the preserved legacy
workflow in `../synthetic_runs/`. Copied generator/plotting provenance is in
`COPY_PROVENANCE.json`. Shared hierarchy detection and RAPID remain dependencies;
this package does not import the legacy synthetic package.

The current analysis version is `outputs/hydraulic_alpha065_20260930/`, selected
by [active_analysis.json](configs/active_analysis.json). Original outputs in
`outputs/full_grid_island`, `outputs/baseline_island`, `outputs/extremes_island`
and the original extension are preserved with their original settings.

Start with [RUNBOOK.md](RUNBOOK.md) for planning, baseline execution, validation,
and restartable full-grid commands.

## Selected synthetic hydraulic reference · 30 September 2026

The selected depth exponent is **alpha=0.65**, anchored to the existing BASED
reference section with local kb varying with width. The decision and controlled
settings are recorded in [selected_hydraulic_reference.json](configs/selected_hydraulic_reference.json).
The full 2,187-configuration grid and 140-configuration extension have now been
rerun in the selected version. Current sensitivity, predictor and routing
notebooks use these results; earlier outputs remain separate. **The real-case choice is deferred until
new runs using updated slopes are available.** Read the current decision in the
[hydraulic applicability notebook](notebooks/synthetic_hydraulic_applicability.ipynb).

## Current results and metrics

Start with [the results synthesis](notebooks/synthetic_results_synthesis.ipynb),
[analysis roadmap](ANALYSIS_ROADMAP.md), and the shared
[current RAPID metric definitions](../RAPID/CURRENT_RESPONSE_METRICS.md).
The [predictor notebook](notebooks/synthetic_predictor_validation.ipynb) now includes
standalone variables, correlations and development-only compact selection;
[full equations and domain scope](GEOMETRIC_PREDICTORS.md) are documented separately.
The current responses are lag, plateau-to-plateau event duration, and peak excess.
Earlier spread-based notebooks are historical supporting analyses.

## Experiment design

Baseline: 20 km total extent, 500 m inlet/combined branch width, reference slope
0.0001 (2 m total head drop), 80/20 outer split, 50/50 inner split, no extra
branch length, and internal junctions at 8 and 12 km.

The outer test section spans 25–75% of the domain; the rest is approach and exit.
Spatial buffers are separate from temporal initialization.

| Grid factor | Levels |
| --- | --- |
| Total width (m) | 250, 500, 1000 |
| Reference slope | 0.00005, 0.0001, 0.0005 |
| Branch A width fraction | 0.5, 0.8, 0.9 |
| Inner branch / crossover fraction of A | 0.2, 0.5, 0.8 |
| Selected branch length multiplier | 1, 1.25, 1.5 |
| Network extent (m) | 10000, 20000, 40000 |
| Internal location | upstream, central, downstream |

The full factorial has 2187 configurations and 15309 comparisons (seven per
configuration). The selected plan has 24,057 state references and 9,519 unique
routing jobs. The earlier hydraulic setup had 9,985 unique jobs; coordinated
clocks/windows affect deduplication as well as geometry. Routing job counts are computed by planning, not assumed equal
to the number of comparisons. At 20 km, internal locations are 7–11, 8–12, and
9–13 km; positions scale with domain extent. Outer location and internal span
are not independently varied in this grid.

## Heads and analytical collapse

Set `head_drop = reference_slope * total_length`, outlet head zero, and
`H(x) = head_drop * (1 - x / total_length)`. Each edge slope is its endpoint
head difference divided by channel length. Node heads are independent of
schematic y coordinates; drawn curvature does not add routing length.

Extra length targets branch A for simple, the first inner branch for nested,
and the crossover edge for crossover. The centerline has no extra-length target.
This is a family-specific perturbation, not the same spatial intervention in
all families. Widths are retained as length changes. Equal sum(WL) is a baseline
condition, not imposed after length adjustment.

Analytical collapse replaces selected edges by one x-monotone corridor, sums
selected widths by downstream station, preserves external attachment stations
and node heads, and uses downstream span as replacement length. It removes
extra length within the selected unit and retains extra length outside it.
This is a controlled graph construction, not the raster reconstruction workflow.

## State relationships

Each configuration has its own originals and derived children:

| Original | Children compared directly to that original |
| --- | --- |
| Simple | Full collapse |
| Nested | Inner-only collapse; main-island-only collapse; full outer collapse |
| Crossover | Upstream-only; downstream-only; full compound collapse |
| Centerline | Control, no collapse |

Nested units have structural parent/child relationships. The two crossover
units overlap at the cross edge and belong to one compound; neither contains
the other. Their three collapse alternatives are sibling states.

Sequential nested collapse remains a possible future comparison, not an extra
comparison in the current grid. Baseline child identities are verified: seven
unique hydraulic networks. With extra length, nested-inner collapse can differ
from the separately lengthened simple original and must have its own run.

### Main-island collapse in the nested network

`nested/main_island_only` removes the main island boundary paths while retaining
both the small island and its upper inner branch. This is a sibling of
`inner_only` and `full`, not a sequential collapse of either child. At baseline,
the remaining bifurcation/confluence lie at 8–12 km. For total width W, outer A
fraction a, and inner fraction f, its branch widths are `a*f*W` and
`(1-a*f)*W`; the stems carry W. Thus outer/inner splits 50/50 and 50/50 give
125/375 m, while outer 20/80 and inner 50/50 give 50/450 m at W=500 m.

The retained upper inner branch preserves its physical length and slope,
including the selected length multiplier. Its endpoint heads are unchanged.
The main-island treatment removes one independent cycle; full topological
collapse removes both. This is an analytical analogue of filling one dry-land
island, not a raster-regenerated geometry. Pair metadata identifies `island`
versus `topological` selection. Resistance uses the closed outer 5–15 km domain
as context (including the surviving branch), with exposure one.

## Routing and deduplication

Current cases use the shared approximately 200 m subreach splitter, `X=0.1`,
Manning `n=0.035`, celerity bounds 0.05–2.524 m/s, and a 200 m minimum
effective K length. BASED estimates reference depth at each configuration’s
total width, slope and peak reference discharge of 1000 m³/s. Parent and child
share that anchor. Locally, `D_i = D_ref (W_i/W_ref)^0.65` and
`kb_i = kb_ref (W_i/W_ref)^0.35`; width-based flow allocation is unchanged.
The earlier `kb=20`, `n=0.35` settings remain only in preserved legacy outputs. Slopes come only from prescribed heads; no automatic
slope repair is applied. No effective-length ceiling is applied. Unstable
comparison groups fail explicitly rather than changing scientific parameters.

Hydraulically identical degree-two series edges are merged before segmentation,
so arbitrary original break nodes cannot cause discretization differences in
otherwise identical networks. This preserves constant width/slope and total
physical length. Canonical numeric inputs use 14 significant digits to eliminate
floating-point summation noise. Diagram coordinates, node names, and family
labels do not enter hydraulic identities.

Common stable integer dt is selected across all states in a configuration.
The forcing window is shared too. Jobs include the canonical network, settings,
dt, window, and code fingerprint. Identical complete jobs run once. Different
coordinated timesteps/windows remain distinct jobs even for the same geometry.

The inlet pulse rises from 100 to 1000 m³/s over 12 hours and falls over 24 hours.
Exact interval-averaged forcing preserves pulse volume across dt choices. A
baseflow buffer on either side is at least 48 hours and at least three times
the longest path sum of routing K across that configuration. Pulse onset is
aligned to a timestep boundary. Runs check pre-event/final settling and excess
volume balance; inadequate windows fail rather than pass silently.

The adapter constructs width-proportional connectivity and calls the shared
RAPID matrix/step functions directly to retain only outlet results. A parity
test compares it against the shared file-based engine.

## Resistance and analysis

Use the shared Chezy/wide geometry resistance law (`Cz=22`) with the same
local `kb_i` depth law as the selected routing run and a normalized unit-head
solver. Within a matched reference, edge resistance scales as `L/W^3.95`.
This remains a Chezy geometry predictor, not a Manning-equivalent resistance. Resistance systems are scaled for conditioning, then restored
to physical K units. Routing K and hydraulic resistance K are separate quantities.

The selected domain is checked with the real-workflow closure audit. Open
individual crossover selections expand to the containing compound; original and
child domains must both be closed. Exposure is the propagated original width
share for a closed unit and one for full compound/context comparisons. Length
fraction sums original domain links once and divides by total original network
link length. The primary predictor is `f_L * E * -log10(K_after/K_before)`.

Analysis outputs signed lag change, absolute lag change, peak change, hydrograph
RMSE and normalized absolute difference, volume balance, numerical flags,
resistance diagnostics, total lengths, and sum(WL). Shared RAPID scatter helpers
produce matching signed/logarithmic resistance–timing figures. Synthetic speed
uses modeled base/child lags and fixed reference extent, not observed gauge lags.
Peak timing remains discrete; inspect numerical-validation results before
interpreting small differences.

## Notebooks

- [Factor effects and interactions](notebooks/synthetic_factor_interactions.ipynb):
  full-grid main and total sensitivities, all 21 pair interactions, matched
  factor changes, and comparison with the resistance predictor. Uses saved
  results; exports tables and static figures without rerunning routing.
- [Hydraulic applicability](notebooks/synthetic_hydraulic_applicability.ipynb):
  retains dated hydraulic investigations and appends the selected full-run
  quality and reference-flow diagnostics. Earlier real-case overlays are
  historical; updated-slope real validation is still deferred. The published
  wave-regime diagram is linked; its full zone boundaries are not implemented.
- [Routing results](notebooks/synthetic_routing_results.ipynb): executed extended full-grid
  resistance–timing plot, paired hydrographs, and numerical diagnostics. Change
  the active analysis version to inspect another saved hydraulic setup.
- [Geometry examples](notebooks/synthetic_experiment_examples.ipynb): four original
  setups and seven two-panel parent–child comparisons. Saved static plots, editable
  baseline parameters, and an optional Plotly view.
- The command-line `preview` module can also export original geometry HTML files;
  it does not execute routing.

## Fixed-unit embedding extension

The [width/length and network embedding extension](EMBEDDED_EXTENSION.md) adds
95/5 and 5/95 splits, multipliers through 3, and fixed 10 km units embedded in
20–800 km networks. All 140 configurations / 980 comparisons are now complete
in the selected version and feed the existing
[results synthesis](notebooks/synthetic_results_synthesis.ipynb) (section 5) and
[predictor validation](notebooks/synthetic_predictor_validation.ipynb) (extension section).

After routing new extension results, refresh the unchanged original-development predictor evaluations:

```bash
python synthetic_experiments/investigations/embedded_predictor_validation.py
```

The predictor notebook excludes original-grid anchors from extension scores and never fits to extension rows. The selected full extension has 981 unique routing jobs. Its 63 original-grid
anchor comparisons are checked and excluded from extension scores, leaving
917 novel comparisons. It remains separate from factorial variance analysis.

## Decision records

- [SYN-2026-09-30-TIMING-OPTIONS](notes/2026-09-30_timing_response_options.md): timing definitions, parent-Leq normalization, embedding and transfer checks; main-text metric choice remains provisional.

- [SYN-2026-09-30-SELECTED-RERUN](notes/2026-09-30_selected_hydraulic_rerun.md): selected setup, rerun coverage, matched changes, checks, and limits.

- [SYN-2026-09-29 checkpoint](notes/2026-09-29_synthetic_checkpoint.md): what is ready, why sensitivity/prediction is paused, outstanding checks, and the restart of hydraulic applicability.

## Interpretation checkpoint — 5 October 2026

Reference **HYDROGRAPH-MEAN-SPREAD-2026-10-05**. Synthetic forcing isolates topology/geometry effects under a shared upstream event. Real runs likewise omit observed tributary/lateral input hydrographs, so transfer and gauge agreement are conditional on that simplification. Missing lateral inputs can contribute to disagreement but do not explain it uniquely. See the [current descriptive audit](../RAPID/validation/hydrograph_mean_spread_20261005/README.md).

Discharge standard deviation is being considered as a smoothing descriptor, with an initial example-37 check. It is not yet an adopted response variable. Comparisons require common windows and a fixed forcing; different event lengths and padded plateaus can change SD independently of collapse. Proposed event shortening and Excel review fields remain unimplemented.
