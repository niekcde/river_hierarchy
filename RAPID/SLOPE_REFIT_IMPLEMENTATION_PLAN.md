# Production plan: boundary-conditioned hydraulic slope refit

## Status and scope

This is the implementation specification accepted from
`notebooks/slope_investigation.ipynb`. It replaces only the hydraulic slope
and head-assignment stage. It does **not** require regenerating masks, RivGraph
topology, directed geometries, bank widths, or collapse masks.

The source of record for detailed diagnostics is
`notebooks/slope_investigation_status.md`; this document fixes the decisions
that production code must follow.

## Frozen longitudinal-profile method

For each base example, construct a single longitudinal WSE profile on the
reviewed SWORD corridor and frozen single-centerline coordinate `x` (distance
to outlet):

\[
z_{ij}=g(x_i)+a_j+\epsilon_{ij}.
\]

`g(x)` is a common spatial WSE profile and `a_j` is a raw RiverSP/SWOT
acquisition-stage offset. Use the reviewed raw-observation quality filters,
complete fixed corridor grid, sparse curvature penalty, acquisition-wise
offsets, and whole-acquisition three-fold CV described in the notebook.

Select the smoothest lambda within one standard error of the best CV score.
For an edge selection, continue the upper lambda tail incrementally until the
profile has a reviewed asymptotic minimum-curvature selection or a finite
one-SE selection. The profile enforces:

\[
g(x_{i+1})-g(x_i)\ge0.01S_{ref}(x_{i+1}-x_i).
\]

The static SWORD P50 endpoint profile is initialisation and audit information;
it is not a hard end-head constraint on the raw-pass fit.

### Reviewed profile-level exception

Example 46's weakly supported 370--410 km interval is replaced by a single
continuous linear bridge between the existing fitted heads at 369,970.164 m
and 410,002.530 m. It is recorded as
`manual_profile_bridge_46_370_410` in the profile selection and audit JSON;
it is not a per-link slope override.

## Head and slope assignment rules

1. Project realised network nodes onto the frozen single centerline.
2. External simple-unit and assembled-compound boundaries obtain fixed heads
   by interpolating `g(x)`.
3. A simple unit and a base compound domain use a positive, actual-`len`,
   equal-per-link graph-potential solver with those fixed boundary heads.
   It minimises the sum of squared relative deviations of realised link slope
   from the local reference slope. Internal compound nodes remain solver
   unknowns. Geometric length remains in each link's slope calculation, but
   does not weight the objective; this prevents short links from becoming
   cheap locations for concentrated head loss. Every solved link is also
   constrained to \(S_e\ge0.01S_\mathrm{ref}\), matching the accepted
   profile-gradient floor and preventing an effectively flat numerical link.
4. A child collapse state uses the existing boundary-conditioned child solver,
   preserving its realised upstream/downstream boundary heads and exact path
   head loss.
5. An internal unassigned link (`unit_ids` empty) receives its endpoint heads
   directly from `g(x)` and its slope from their difference divided by its
   preferred geometric length. Do not condition these links with a separate
   local optimiser.
6. Network-endpoint links are not internal connectors and retain their realised
   endpoint boundary condition.

## Explicit connector decision

The old nearest-accepted-anchor connector screen is not a production target:
it can incorporate a distant compound reach. The completed 16-example test
found 424 positive internal connectors and 32 network-endpoint links. Fourteen
old factor-five flags were reviewed with bounded 1/2/5 km physical support;
four were within the local band and ten had heterogeneous physical support.
There were no isolated mismatch candidates. Example 22 link 45 demonstrated
that the old target could imply a false multi-metre correction across a genuine
10.84 km low-gradient profile section.

Therefore the production rule is profile-direct assignment plus the intrinsic
profile gradient floor only. Retain the bounded physical-support test as an
audit, never as an automatic head correction.

## Required durable artifacts per base example

- `reference_profile.parquet`: fixed profile grid, fitted WSE, profile slope,
  numerical-floor indicators, and corridor coordinate metadata.
- `reference_profile_observations.parquet`: accepted raw observations,
  acquisition IDs, weights, fitted offsets, residuals, and quality provenance.
- `reference_profile_selection.json`: lambda grid, CV scores, one-SE decision,
  any upper-tail continuation, final selected lambda, input hashes, and method
  version.
- `reference_profile_audit.json`: static-P50 versus fitted endpoint heads,
  fitted head loss, reference slope, floor activity, and profile-validation
  summaries.
- `connector_physical_support_audit.parquet`: endpoint exclusions and the
  read-only 1/2/5 km diagnostic for internal unassigned links.

## Implementation status and remaining sequence

The immutable profile fitter and artifact writer are now implemented in
`RAPID/src/rapid_tools/reference_profile.py`, with its source adapter in
`RAPID/src/rapid_tools/reference_profile_inputs.py`. Its operational entrypoint
is `RAPID/run_generate_reference_profile.py`.

`RAPID/run_materialize_slope_refit_candidates.py` is the controlled 16-example
batch entrypoint.  It contains the reviewed lambda registry, defaults to a
read-only preflight, and writes only to a new candidate root when `--execute`
is supplied. Each run produces a batch manifest that records the canonical
experiment root, frozen lambda, profile path, candidate-context path, and
completion/error state.

Where the notebook has already reviewed and frozen a final lambda, the CLI's
`--fixed-lambda` mode materialises that exact profile without repeating the
expensive CV selection. Its selection JSON explicitly records that CV was
reused from the accepted notebook decision. The normal no-flag CLI remains the
fully independent selection route for a future new example.

A non-mutating candidate base-context runner is also implemented in
`network_variants/reference_profile_candidate.py`:

```bash
python -B network_variants/run_generate_reference_profile_candidate_context.py \
  EXPERIMENT_ROOT REFERENCE_PROFILE_DIR CANDIDATE_OUTPUT_DIR
```

It consumes only frozen profile artifacts and base inputs. It writes separate
candidate link/node contexts, ownership, and exact path-head-loss audits. It
does not modify `hydraulic_context/`, equivalent-resistance, or RAPID outputs.
If a domain cannot be solved, its direct profile slopes are retained and the
audit fails closed rather than applying an arbitrary fallback.

The final read-only inspector is
`RAPID/run_inspect_slope_refit_candidates.py`.  It reports network-global,
domain-local, and immediate-topology contrasts separately.  A link exactly at
the explicit domain constraint \(0.01S_\mathrm{ref}\) is reported as
`domain_physical_floor_bound_active`, rather than being conflated with a
numerical near-zero slope or an unexplained local outlier.  It also verifies
that no solved link violates its recorded domain minimum slope.

Remaining work:

Completed 2026-09-30 in the isolated root
`network_variants/output/_slope_refit_candidates/`:

- all 16 reviewed profile artifacts and candidate base contexts;
- validation: all 16 passed; all 6,050 candidate links use `len`, are
  positive, reconstruct exactly from their assigned heads, and all 440 solved
  domains meet their prescribed path head loss (maximum residual
  `5.68e-14 m`);
- the initial slope/celerity screen. It reports a median candidate/current
  slope ratio of `0.97775`; 3,672 of 5,315 comparable links change by more
  than 25%. Candidate lower-celerity-cap links decrease from 1,811 to 1,706;
  four links newly reach the upper cap. Equivalent resistance remains exactly
  invariant under the current normalized geometry/width/roughness solver.

The impact screen is a review gate, not a promotion result. Three corrected
examples (13, 45, 48) have no retained canonical base RAPID-prep manifest, so
their slope comparison is complete but their stored-config celerity translation
is correctly reported as unavailable rather than guessed.

Completed 2026-10-01 in the isolated root
`network_variants/output/_slope_refit_candidates_equal_link_bridged_floor_v4/`:

- the equal-per-link solve now has the explicit physical lower bound
  \(S_e\ge0.01S_\mathrm{ref}\), with the reviewed Example 46 profile bridge
  included;
- all 16 examples, 5,315 candidate links, and solved-domain path constraints
  validated; the largest absolute path-head residual was
  `6.40e-14 m` and there were zero minimum-slope violations;
- the final audit found four intentional floor-bound links, all in Example 15
  bubble 4. Seven additional non-floor factor-five local contrasts remain in
  four complex bubbles (Example 15, 22, 37, and 54) and are retained as
  explicit review provenance rather than silently corrected. They preserve
  their exact boundary path head losses.

Remaining work:

1. Review the durable candidate validation and impact artifacts, including the
   largest changed links and the changed celerity-cap population.
2. Decide whether the base candidate contexts are promoted. If accepted,
   recompute equivalent
   resistance and RAPID inputs/runs; masks, topology, widths, and existing
   collapse geometry are reused.
3. Apply the same immutable base-profile artifacts to independent, compound,
   local-patch, panelized, and island child states. Re-run only their hydraulic
   context, resistance, and RAPID layers; do not rebuild raster/network outputs
unless a separate topology correction is required.

### Documented manual profile adjustments

The immutable profile CLI supports a narrowly scoped, fully recorded
profile-level intervention when a reviewed corridor interval is demonstrably
weakly supported. `--manual-linear-bridge ADJUSTMENT_ID START_M END_M` replaces
only the fitted-profile rows between its resolved fixed-grid endpoint stations
by a line joining the pre-existing fitted endpoint heads. It therefore remains
continuous, changes all later node/link interpolations in the interval
consistently, and never acts as an untracked per-link slope override. The
selection and audit JSON retain the adjustment ID, requested/resolved bounds,
endpoint heads, bridge slope, and number of replaced grid rows.

The first isolated application is `manual_profile_bridge_46_370_410`: the
Example 46 369.970--410.003 km low-support interval is bridged at
`6.713119e-05 m/m` using its existing fitted endpoint heads. Its candidate
trial is retained separately beneath
`_slope_refit_candidates/manual_profile_trials/46_bridge_370_410/`; no
established hydraulic context has been overwritten.

## Non-negotiable audits

- Every directed path inside a solved domain has exactly its prescribed
  upstream-to-downstream head loss, within numeric tolerance.
- Every output slope is positive and uses geometric preferred length.
- The profile's fitted global head loss is reported, not silently replaced by
  static P50 endpoints.
- A missing or ambiguous profile/node projection fails explicitly; it does not
  silently fall back to an unrelated nearest-link slope.
- Every output records method version and profile-artifact hash.
