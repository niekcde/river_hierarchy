# Boundary-conditioned slope refit: current status

## Current monotone-slope method and candidate implementation

The notebook established the replacement reference longitudinal WSE profile.
Its immutable artifact writer and a non-destructive base candidate-context
runner are now implemented, but neither changes the established hydraulic
contexts, resistance, RAPID, hierarchy, or collapse outputs. Promotion remains
conditional on the all-base candidate audit. The method keeps the existing
realised directed network and frozen single centerline; it does not introduce a
second hierarchy graph.

### 1. Frozen corridor and reference head loss

1. Build the SWORD reach corridor between the realised directed-network inlet
   and outlet, retaining a short reach buffer at each end.
2. Project every accepted corridor SWORD node onto the frozen single centerline.
   The profile coordinate, `x`, is distance to outlet, so WSE should increase
   upstream.
3. Project the exact realised inlet and outlet nodes onto the same line.  Their
   static `wse_obs_p50` PAVA-interpolated heads define the whole-network,
   full-corridor reference slope:

   \[
   S_\mathrm{ref}=
   \frac{H_\mathrm{inlet}-H_\mathrm{outlet}}{L_\mathrm{single\ centerline}}.
   \]

   This deliberately does **not** use the first and last retained raw-pass
   observations: a raw-observation gap must not redefine the network head loss.
   These P50 heads are not imposed on the raw-pass fit. They provide its initial
   profile, set the scale of the small numerical gradient floor, and remain a
   formal audit reference. The multi-pass fit is allowed to estimate its own
   end-to-end head loss.

### 2. Stage-adjusted raw RiverSP profile

The fit uses every accepted raw RiverSP/SWOT node observation after the existing
quality filter and newest-product-per-acquisition rule.  A measurement is
represented as

\[
z_{ij}=g(x_i)+a_j+\epsilon_{ij},
\]

where `g(x)` is the common spatial WSE profile and `a_j` is one free stage
offset for raw acquisition `j`.  Thus multi-pass observations inform the
profile without interpreting changing river stage as longitudinal slope.

The unknown profile is represented on the **complete fixed SWORD corridor
grid**, plus the two exact realised endpoint coordinates.  Grid nodes with no
retained raw observations remain in the solve; the curvature penalty bridges
them instead of allowing a raw-data gap to become a false hydraulic plateau.

For a candidate smoothing weight `lambda`, the notebook solves the convex
objective

\[
\min_{g,a}\ \sum_{ij}w_{ij}[z_{ij}-g(x_i)-a_j]^2+
\lambda\,g^\mathsf{T}Kg,
\]

where `K` is the non-uniform-grid second-derivative (curvature) penalty.  It
alternates two convex blocks: a projected-gradient update for `g`, and the
closed-form weighted mean residual for each `a_j`. Offsets are then centred to
have mean zero, transferring their common shift into `g`; this identifies the
otherwise arbitrary vertical datum without changing fitted values or slopes.

### 3. Intrinsic numerical minimum-gradient constraint

The prototype enforces a small positive gradient directly in the optimisation:

\[
g(x_{i+1})-g(x_i)\ge S_\min(x_{i+1}-x_i),\qquad
S_\min=\eta S_\mathrm{ref},\quad\eta=0.01.
\]

It is imposed exactly by applying PAVA to
`q(x)=g(x)-S_min*x`, then adding `S_min*x` back.  This is not a post-fit
correction and it is not a universal physical river-slope claim.  `eta=0.01`
was chosen as a small numerical conditioning rule: in the pre-floor,
direction-corrected 16-example diagnostic, the first percentile of eligible
positive interval slopes was about `0.044 S_ref`; the floor is below that
observed lower tail.  If every interval binds, the constraint alone produces
an end-to-end drop exactly equal to 1% of the reference head loss; it does not
prescribe the actual profile head loss, which can be larger. The batch audit
reports the number and total length of floor-active intervals explicitly.

### 4. Smoothing selection and downstream use

- Test a broad 31-value `lambda` path, from `10^-1` through `10^8`, at about
  twofold spacing.
- Use its L-curve initially to bracket seven candidates: one weaker-than-knee
  value, the knee, and five increasingly smoother values. If the one-standard-
  error choice is still the smoothest tested candidate, extend only that
  example's CV bracket by three successively smoother values at a time, using
  cached earlier scores, until it is bracketed or the full 31-value path is
  exhausted.
- Select the final value by three-fold **whole-acquisition** cross-validation;
  held-out passes receive a newly estimated stage offset before their residual
  score is calculated.
- Use the one-standard-error rule: from the candidate profiles whose mean
  acquisition-level validation score is within one standard error of the
  minimum, retain the largest `lambda` (the smoothest equally supported fit).
- Compare the selected raw-pass profile against the old static P50/PAVA profile
  and the prior P50 natural-spline sensitivity line for all 16 examples.
- Construct the sparse curvature operator once per example and reuse it for
  every lambda trial and cross-validation fold. This is mathematically
  identical to the prior dense implementation but avoids repeated dense matrix
  construction and eigendecomposition.

The batch audit reports fitted versus static-P50 inlet/outlet heads, total head
loss, global slope, signed and relative head-loss differences, and the selected
one-SE smoothing value. It does not force agreement. A companion diagnostic
screens the largest local fitted-slope changes in **every** example; a selected
example can then be inspected against its stage-aligned individual acquisitions
and static P10--P90 envelope.
The batch summary records the number of adaptive CV-bracket extensions and
whether the complete lambda path was exhausted. A selected lambda at the
smoothest tested candidate is therefore unresolved only if the full path was
actually exhausted.

For a full-path edge result, the notebook now provides a separate incremental
upper-tail continuation cell. It retains the completed per-lambda CV summaries,
replays only the lean raw-pass setup for edge cases, and evaluates a configured
set of further factor-of-two lambda values per affected example. Re-running that cell advances
only examples still selected at the current upper edge; it does not repeat the
16-example baseline search.

A following convergence-audit cell compares the original one-SE profile with
the latest upper-tail profile on their identical fixed SWORD grids. It reports
RMS and maximum WSE/head changes, RMS and maximum local-slope changes scaled by
the reference slope, and transparent proposed numerical-equivalence checks. It
is diagnostic only: profile promotion remains an explicit review decision.

If accepted, the profile supplies heads only at external simple-unit and
assembled-compound-bubble boundaries. Internal compound-member nodes remain
solver nodes. Each simple unit or bubble then receives its two fixed boundary
heads and uses the positive, equal-per-link graph-potential solve with actual
`len`: it minimises the squared relative slope mismatch once per realised
link, rather than multiplying that mismatch by link length. This prevents
short links from becoming cheap locations for concentrated head loss. The
16-example v2 candidate batch removed all nine prior slopes above 0.01 m/m
and reduced the twelve slopes above 0.005 m/m to one pre-existing Example 48
link. A solved-domain link also has the physical lower bound
\(S_e\ge0.01S_\mathrm{ref}\), rather than only an almost-zero numerical
positivity constraint. The Bubble 13 demonstration in Example 41 passed the
topology check: all 50 links were monotone and all 356 directed paths had the
same boundary head loss.

## Connector decision (resolved; candidate implementation present)

Links with empty `unit_ids` are inter-domain connector sections. The initial
proposal to condition an apparent anomalous connector by solving a local
three-span `U -> A -> B -> D` system was tested and **rejected for production**.
It would add a second heuristic head-allocation layer without evidence that it
improves the data-supported reference profile.

### What the completed calibration showed

- The direction-corrected 16-example population contains 456 unassigned-link
  components: 424 are positive, context-complete internal simple connectors;
  the remaining 32 are exactly the two network-endpoint links in every example
  (16 inlet and 16 outlet links), not internal topology ambiguities.
- For the 424 internal connectors, the robust central log-ratio spread around
  the historical nearest-anchor screen corresponds to a symmetric factor of
  `2.11571`; the transparent rounded factor-five diagnostic covers `96.70%`.
- Fourteen links exceeded factor five under that historical screen. The screen
  target was based on the nearest accepted graph anchors and could therefore
  include a distant compound domain; it is retained for historical comparison
  only and is not a correction target.
- A read-only physical-support audit measured the frozen fitted profile over
  1 km, 2 km, and 5 km windows on both sides of all 14 links. Four links are
  already within factor five of their bounded physical support. The other ten
  have heterogeneous physical support, rather than an isolated mismatch. No
  link was an `isolated_mismatch_candidate`.
- Example 22/C001 (link 45, 10.84 km) is decisive: its connector and all six
  bounded support slopes equal `2.298136e-6 m/m`. Its former factor `182.9`
  arose solely because the graph-anchor screen extended into a large adjacent
  compound domain. It is a sustained low-gradient profile section, not a
  4.5 m head-loss error.

### Candidate and eventual production rule

1. Assign an internal unassigned link's two endpoint heads by interpolation
   from the frozen raw-pass monotone reference profile.
2. Compute its slope from those heads and its geometric preferred length
   (`len`, falling back only where the existing preferred-length rule requires
   it).
3. Retain the intrinsic `0.01 S_ref` profile gradient floor; do not add a
   separate connector slope floor or a three-span connector solver.
4. Exclude network-endpoint links from internal-connector screening. They have
   only one exterior support side and continue to use their realised endpoint
   boundary condition.
5. Retain the bounded 1/2/5 km physical-support audit as QA only. Its outcomes
   are `within_physical_factor_band`, `heterogeneous_physical_support`,
   `insufficient_profile_support`, or a future manually reviewed
   `isolated_mismatch_candidate`; none changes heads automatically.

This preserves the profile as the sole data-supported longitudinal head field.
Simple-unit and compound-domain solvers use its boundary heads, but no later
step overrides a genuine local plateau merely to conform to a distant slope.

## Current diagnostic state after direction correction

Examples 13, 45, and 48 had confirmed whole-network direction reversals. They
have now been corrected, rebuilt, and physically promoted under their
canonical experiment names:

- Example 13: `SW -> WS` (plus retained manual internal link corrections 94,
  97, and 146).
- Example 45: `NS -> SN`.
- Example 48: `WN -> NW`.

The refreshed `slope_investigation.ipynb` is now the authoritative test bed.
Its first executed Example 41 / Bubble 13 pass has 76 accepted external
anchors, identifies 38 unassigned components, and verifies the proposed
two-boundary graph-potential solve: all 356 Bubble 13 paths reproduce a
2.154203 m boundary loss with zero numerical residual.

The final direction-corrected 16-example connector calibration supersedes the
earlier figures above. It has 424 positive internal simple connectors and 32
network-endpoint links, with no internal ambiguous component. Its robust
three-sigma factor is `2.11571`; factor five is retained only as a transparent
QA flag and covers `96.70%` of the internal population. The bounded
physical-support review resolved all fourteen historical factor-five flags in
favour of retaining the profile-derived slopes; see **Connector decision**
above.

The immutable profile fitter is implemented in
`RAPID/src/rapid_tools/reference_profile.py`, its endpoint-bounded corridor
adapter in `RAPID/src/rapid_tools/reference_profile_inputs.py`, and the base
candidate context runner in `network_variants/reference_profile_candidate.py`.
The candidate runner is deliberately isolated from `hydraulic_context/` and
has passed a real Example 11 artifact-to-context smoke test. The remaining
all-base validation and promotion sequence is recorded in
`RAPID/SLOPE_REFIT_IMPLEMENTATION_PLAN.md`.

### Candidate base execution record (2026-09-30)

All 16 reviewed profiles were materialised with their accepted notebook lambda
in the isolated output root
`network_variants/output/_slope_refit_candidates/`. The candidate contexts do
not alter any established experiment product. Their validation passed for every
example: all 6,050 links use `len`, have positive slopes, and reconstruct from
their written node heads exactly; the 440 solved simple/compound domains have a
maximum path-head-loss residual of `5.68e-14 m`.

The initial current-versus-candidate impact screen is intentionally a review
gate. Across 5,315 links with a stored RAPID configuration it found median
candidate/current slope ratio `0.97775`, with 3,672 links changing by more than
25%. Candidate lower-cap celerity links decrease from 1,811 to 1,706 and four
links newly reach the upper cap. Equivalent resistance is analytically
unchanged in its current normalized geometry/width/roughness formulation:
`slope_used` is not an input. Examples 13, 45, and 48 have no retained
canonical RAPID-prep manifest after direction promotion, so their celerity
translation is explicitly unavailable—not estimated from unrelated settings.

## Example 48 isolated direction-rebuild pilot (2026-09-23)

The base-only RivGraph rebuild was run in temporary local storage from the frozen
`48__S000_base_collapsed.tif`; no existing experiment product was changed.

- The corrected `NW` run has 189 links and 127 nodes, exactly matching the
  original link count.
- Every one of the 189 link geometries and every undirected endpoint pair is
  identical to the original.  187 internal links have reversed directed
  endpoints, as expected for a whole-network reversal; the remaining two are
  the endpoint-related special cases.
- The corrected inlet is node 0 (`wse_obs_p50 = 1964.15127 m`) and the
  corrected outlet is node 126 (`1737.68016 m`): the network is now directed
  downhill.

### Width validation still required

The fresh RivGraph output contains the unreviewed scalar `wid`, `wid_adj`, and
`wid_med` values for 16 links.  This is expected: those are exactly the links
marked `bad` in the completed permanent base-width review.  The current base
pipeline retains those decisions as review provenance while measuring physical
two-bank families from the mask; it does **not** numerically scale the physical
families themselves.  This distinction matters and was tested explicitly.

The production-semantics width replay completed:

- 189 links and 12,899 two-bank samples were recomputed from the frozen mask.
- Physical wet widths agree closely despite the reversed sampling order:
  median relative difference `0.45%`, 95th percentile `3.41%`, and only 3
  links exceed `5%` (one reaches `10.35%`).  This is a small sampling-order
  sensitivity, not a new topology or manual-review effect.
- Visual QA of the three links above the `5%` threshold (34, 115, and 141)
  found identical centreline geometry and identical two-bank transect/bank-pair
  selection in the old and corrected runs.  The physical-width check therefore
  passes; the small differences are retained as numerical sampling sensitivity.
- Final routing widths are not expected to be invariant: the existing solver
  is directional.  Their median relative difference is `0.79%`, but 23 links
  exceed `5%`, 17 exceed `10%`, 14 exceed `25%`, and 8 exceed `50%`.
  The largest changes occur on links 131, 64, 27, and 36, where the physical
  wet width is unchanged but the corrected upstream/downstream context leads
  the routing solver to a different correction.

Therefore the direction correction can preserve masks, physical geometry, and
the completed manual-width review, but it must regenerate and retain audited
`wid_adj_routing`.  It would be methodologically wrong to copy the old routing
widths forward unchanged.

## Completed direction-rebuild pilot set

The same isolated, production-semantics validation was completed for all three
confirmed whole-network reversals.  In every case, the corrected raw network
has exactly the same undirected topology and link geometries as its current
base; only direction changes.

| Example | Correction | Links | Reversed internal endpoints | Physical wet-width result | Corrected WSE direction |
| --- | --- | ---: | ---: | --- | --- |
| 13 | `SW -> WS` | 175 | 172 | median 0.13%, P95 1.21%, maximum 2.81%; no link exceeds 5% | 76.59472 m inlet -> 41.80000 m outlet |
| 45 | `NS -> SN` | 223 | 221 | median 0.14%, P95 0.74%, maximum 2.94%; no link exceeds 5% | 238.67472 m inlet -> 213.61197 m outlet |
| 48 | `WN -> NW` | 189 | 187 | median 0.45%, P95 3.41%; three visually cleared links exceed 5% | 1964.15127 m inlet -> 1737.68016 m outlet |

Each routing-width solver completed successfully.  Its changed
`wid_adj_routing` values are expected consequences of the corrected
upstream/downstream continuity; they are retained as normal derived outputs,
not treated as a failure of the direction correction.

The three direction corrections may therefore be promoted as clean corrected
base rebuilds.  A promotion must rebuild all direction-dependent downstream
products, while retaining existing masks and completed manual width-review
decisions by their unchanged link identities.

## Raw-direction promotion record (2026-09-23)

The validated raw direction products have been published under the canonical
raw RivGraph run names, with the old raw directories retained where available:

| Example | Raw correction record | Historical raw directory |
| --- | --- | --- |
| 13 | `rivgraph_centerline/outputs/runs/13/direction_raw_promotion_20260923T155536Z.json` | An interrupted promotion left no complete raw directory to move. Its reviewed original-link snapshot and the complete pre-correction experiment roots remain retained; the short partial skeleton/log are retained under `direction_promotion_backups/20260923T155239Z/interrupted_partial_rivgraph/`. The final record marks this explicitly as `interrupted_recovery=true`. |
| 45 | `rivgraph_centerline/outputs/runs/45/direction_raw_promotion_20260923T155543Z.json` | `direction_promotion_backups/20260923T155543Z/rivgraph/` |
| 48 | `rivgraph_centerline/outputs/runs/48/direction_raw_promotion_20260923T155544Z.json` | `direction_promotion_backups/20260923T155544Z/rivgraph/` |

Every record includes the frozen-mask hash, old and corrected raw graph hashes
(where the source raw graph was present), the validated trial source hashes,
and the complete direction-only continuity audit. The associated
`width_review/direction_only_continuity.json` files prove the raw correction
preserves stable link/node IDs and geometry while reversing only directed
endpoints. The corrected experiments have subsequently been physically
promoted, not aliased. The nine former canonical roots for Examples 13, 45,
and 48 are retained in the dated archive under
`network_variants/output/_archived_pre_directionfix/`; the promoted canonical
manifests contain no remaining versioned `directionfix_20260923` references.
