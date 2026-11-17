# Completed three-method synthetic grid: first analysis update

**Reference: SYN-FULL-DEPTH-GRID-2026-10-28.** Written on 28 October 2026 after the
full factorial completed, to replace pilot-led notebook views with the complete
population and separate historical conclusions from the new hydraulic comparison.
No routing was rerun for this update.

## Population and location

6,561 geometries × three methods × seven comparisons = 137,781 response rows in
19,683 completed groups. The eight three-level controls are reference width,
reference slope, outer A fraction, inner/crossover fraction, selected-branch
length multiplier, total network extent, internal location and branch-width
factor. Methods are fixed BASED + width scaling, fixed midpoint Manning normal
depth, and conservative variable Manning/MCT. Shared forcing is 500–1000–500 m³/s;
n=0.025, BASED exponent=0.65, beta=1, target cells=200 m, native dt=15 s,
reported interval=60 s. The internal width factor preserves equal inlet/outlet
reference widths and is retained in the collapsed corridor.

Results are mounted at `/Volumes/SyntheticResults20261027`; all underlying data
remain on the PhD drive in the APFS container. Derived CSV/Parquet tables, PNGs
and source hashes are under `analysis/full_grid_20261028/`.

The main notebooks are now executed views of the completed grid:

- `synthetic_results_synthesis.ipynb`: population, definitions, ranges, response
  directions and matched hydraulic-method comparisons.
- `synthetic_factor_interactions.ipynb`: eight main/total effects, all 28 pairs,
  higher-order residual and matched signed contrasts for centroid and peak shift.
- `synthetic_predictor_validation.ipynb`: physical travel-time diagnostic,
  no-fit performance and residual controls; no new RF or model-selection claims.
- `synthetic_routing_results.ipynb`: selectable paired hydrographs and multiple
  collapsed states on one panel, including the uniform centerline control.

Prior notebooks and their outputs are retained under
`notebooks/legacy/pre_full_depth_grid_20261028/`. Legacy active-analysis selectors
were not repointed to an incompatible output format. Existing hydraulic-domain
and historical event-shape analyses are not revalidated by this update.

## Response definitions and numerical interpretation

Responses are child-minus-parent native outlet peak time and full excess-event
centroid time, displayed in minutes. The peak uses the first native maximum,
so peak switching and ties can affect its interpretation. The saved centroid
lag percentage uses parent centroid minus input centroid in its denominator;
it is not a percentage relative to the absolute clock time.

Direction summaries use ±15 s for peak and ±2 s for centroid as interpretation
bands. These do not quantify spatial or timestep discretization error. The
centroid band reflects the sum of the two saved 1 s tail-convergence tolerances.

A response span ≤4 s for centroid or ≤30 s for peak is too small here to support
normalized variance/R² interpretation. Raw values and absolute formula errors
are retained, but normalized effects and R² are marked undefined. This affects
the main-island treatment under both Manning methods. Its largest variable-depth
centroid shift is about 2.59 s: that deserves numerical refinement if interpreted,
not an unqualified physical claim of delay. The screening threshold is explicit
and is not a substitute for convergence/refinement studies.

## First findings

1. **Centroid and peak tell different stories.** Across the balanced population,
   BASED centroid shifts are earlier beyond the band in 100% of comparisons.
   Its peaks are later in about 23.22%. Under either Manning method, about 57.14%
   of centroid shifts are earlier and 42.85–42.86% are near zero. About 5.84% of
   peaks are later. A later dominant peak therefore does not establish a later
   full-event centroid.
2. **The hydraulic closure changes sensitivity.** Averaged across treatments,
   network extent contributes about 53% of BASED centroid variance as a main
   effect. Among the six resolved Manning treatments, branch-length multiplier
   contributes about 32–33% on average. These are equal-treatment summaries of
   uniform grid variance, not universal river-parameter importance rankings.
3. **A Manning travel-time expression explains centroids.** With fixed
   width-based allocation fractions f, define
   `G_M = sum_e L_e * B_e^(2/5) * f_e^(3/5) * S_e^(-3/10)`.
   Wide-rectangular fixed Manning hydraulics imply
   `T = (3/5) * n^(3/5) * Q_ref^(-2/5) * G_M`.
   The child-minus-parent difference uses Q_ref=750 m³/s. G_M uses geometry,
   slope and the prescribed allocation; conversion to time additionally uses
   roughness and reference discharge. Sums are over whole networks; common
   unchanged contributions cancel. This does not redefine unit length or omega.
4. **Do not mistake an identity for validation.** The maximum fixed-Manning
   centroid discrepancy is about 0.044 s. This checks the implementation and
   linear-network first moment; it is not independent predictive validation.
   Applying the same unfitted formula to variable Manning gives R²≈0.990–0.992
   across the six resolved treatments, but the largest absolute discrepancy is
   still about 35 min. High R² does not eliminate meaningful magnitude errors.
5. **That centroid formula is poor for peaks.** Fixed-Manning peak R² is negative
   across resolved treatments; variable-Manning values are only about 0.04–0.09.
   Peak location requires an explanation of dispersion/event-shape behavior as
   well as mean transport time. The formula is not a universal replacement for
   the existing resistance framework.

## What is and is not settled

The synthetic sensitivity analysis now uses the entire completed population,
separately by method and treatment. Variance contributions remain unsigned;
direction is shown in matched endpoint contrasts. Main effects, distinct pairs
and the higher-order remainder sum to 100% in resolved strata. Total sensitivities
overlap and are not additive. No rows were removed to improve normalized scores.

The physical diagnostic is the first explanatory step agreed before a new
predictive search. The old resistance/length-fraction/exposure benchmark has not
been refitted on this grid. Rebuild those and the agreed geometry descriptors
with matching domain semantics before candidate selection and held-out testing.
Keep all methods and related treatments of a geometry in the same validation
fold. Multicollinearity, RF/model-family search, event-shape variables and real
network transfer remain separate next steps.

All findings are conditional on this grid, forcing, allocation and hydraulic
closures. Deterministic agreement with these outputs is not a general physical law.

## Validation and reproduction

Two analytic tests check the known main/pair/higher-order variance partition,
matched contrast direction and handling of near-constant responses. The full
loader verifies completeness, factorial balance, uniqueness and finite timing
responses. Geometry-derived travel times agree with stored fixed-Manning edge
celerities; the centroid identity is checked against the saved convergence
threshold. All four notebooks execute without rerouting. Figures are static and
visible in notebook previews; PNGs are also exported to the analysis folder.

Build tables: `investigations/update_completed_depth_grid.py`.
Notebook source generator: `investigations/write_completed_grid_notebooks.py`.
The latter preserves the dated legacy copies, but regenerating a notebook clears
its displayed outputs until it is executed again.
