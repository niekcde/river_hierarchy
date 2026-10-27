# Local hydraulic scope investigation

27 October 2026. Isolated network-22 pilot; no production edits, calibration,
all-network routing rerun, or RF training. The proposed invariants and solver failure
rules are in [BOUNDARY_CONTRACT.md](BOUNDARY_CONTRACT.md).

## Verified findings

All 2,206 files hashed in the original audit still match. Its 581,395 detailed
changed-link rows reconcile with every state summary. The reported totals agree:
2,185 comparisons across 15 networks; 1,633 change matched retained slopes;
1,631 extend outside the saved footprint; 1,591 have changes exceeding 1%.
By-type case/change counts are compound member 1,467/1,386; island 268/247;
simple 258/0; full compound 177/0; segmented full network 15/0.

The original matching routine uses mapped directed endpoints, length and width,
with stable IDs resolving parallel candidates. It does not compare polylines.
Thus its findings are valid for that matched population, not certification of every
exterior geometry or width. The pilot checks actual polylines separately.

The shared-head solver releases interior heads across the affected containing
compound, and minimizes relative-slope departures with a positive floor. There is
no original-head or individual-slope preservation term. `freeze_grid` independently
reallocates Q_ref over the whole child graph using width fractions. These mechanisms
explain why changing the depth formula revealed an existing scope problem.

Prior boundary geometry repairs were adopted: active release
`master_integrated_20261009`, regeneration complete, master products published.
However, sampled routing inputs still declare `existing_domain_policy`; geometry
adoption did not adopt the proposed strict exterior hydraulic contract.

## Concrete obstructions in the saved replacements

- Unit 30 has two changed exterior geometries: parent 208 → child 1010 and parent
  225 → child 1012. Directed mapped endpoints agree, but polylines differ by about
  7.011 m and lengths by ±2.904 m. An isolated restoration of these two lines and
  two incident replacement endpoints resolves that geometry failure.
- Its saved replacement still cannot preserve original gross boundary fluxes and
  exterior split coefficients. At boundary 125, the selected parent carries
  0.223885031 of network reference flow in and 0.123625312 out. The replacement has
  an incoming edge but no outgoing selected edge. Other mixed ports also lose an
  incidence. This is an explicit topology/flow infeasibility, not a solver problem.
  Preserving only the net boundary flux would conceal the changed exterior split.
- Island 41 has one changed exterior geometry (parent 148 → child 1008). Restoring
  it also leaves an infeasible boundary-flow contract for the saved topology.
- Island 18 fails exterior correspondence on 62 parent reaches under its conservative
  footprint. This includes parent 169, whose polyline is retained but direction is
  reversed. No local routing candidate was accepted for this case. Missing or
  subdivided correspondences are unresolved, not automatically proven destructive.

These failures concern the saved topologies under the stated contract. They do not
prove all strictly local replacements impossible. Explicit restoration actions and
missing-incidence witnesses are saved with the pilot results.

## Feasible isolated alternatives

The simple unit 8 replacement passes unchanged. For unit 30 and island 41, a separate
port-to-port path reduction of the selected parent subgraph preserves all boundary
ports and their gross flows. It reduces 33→19 and 12→8 selected reaches respectively.
All exterior geometry, heads, slopes, widths, depths, Q_ref, split fractions, K/X,
celerities and discretization are inherited exactly. Shared-head and full-network
reference conservation checks pass; maximum local flow residual is 2.78e-17.

This alternative changes internal representation: concatenated parent path geometry,
length-weighted width and depth, unchanged calibrated n and beta, locally recomputed
Cunge parameters. It is a feasibility demonstrator, **not the same intervention as
the saved raster collapse**. It neither fits nor forces the parent routing response.
Island 41 uses the existing conservative 12-link footprint; this does not resolve
whether a smaller physical island-only selection was intended.

| Case | Exterior reaches | Existing changed exterior slopes | Existing changed exterior Q_ref | Existing changed exterior widths/depths | Local changes in any of these |
|---|---:|---:|---:|---:|---:|
| Simple unit 8 | 991 | 0 | 0 | 0 | 0 |
| Compound member 30 | 960 | 481 | 325 | 8 | 0 |
| Island 41 | 981 | 496 | 4 | 3 | 0 |

These broader exterior counts include the explicitly restored geometry correspondences
and do not filter out changed widths. They therefore differ from the earlier
unit-30 retained-match counts (470 stable-ID / 473 width-and-length-matched slopes,
316 stable-ID Q_ref changes). The populations differ; the old audit is not contradicted.

## Routing response

The pilot uses saved BASED depth outside the selection, frozen network-22 n and beta,
and a common 2.7027027-second timestep. A 100→200→100 cms triangular pulse spans
12–36 hours. The extended tail reaches about 1,989 hours. No event-window metrics or
Manning calibration were rerun.

The exact centroid contrast of this fixed linear model is the difference in
`sum(reference_flow_fraction * count * rapid_k)`. This avoids finite-tail bias.

| Case | Existing centroid shift vs parent, min | Local alternative shift vs parent, min |
|---|---:|---:|
| Simple unit 8 | −2.251390 | −2.251390 |
| Compound member 30 | +17.796998 | −0.763711 |
| Island 41 | −0.201233 | −0.057726 |

An initial 663-hour pulse window underestimated the existing unit-30 contrast by
about 1.54 minutes despite a good timestep check. It is retained as preliminary
output. The extended pulse is the numerical validation of the exact contrasts.
The first-window timestep-halving contrast difference is at most 0.000143 seconds.
The source signed-coefficient policy is retained; native flows stay positive, with
small pulse-baseline undershoots exposed in the audit rather than clipped.
See `extended_pulse/responses.csv` for extended volume and centroid residuals.

## Implementation and limits

Six focused tests cover exact exterior inheritance, shared unknown heads, fixed-edge
and free-node slope infeasibility, incompatible reference flux, and port-path reduction.
Saved trial outputs were revalidated against exact exterior values and complete
reference conservation. Source/result manifests and CSVs are under
`RAPID/outputs/local_boundary_20261027_v2`; the earlier strict screening remains under
`RAPID/outputs/local_boundary_20261027`.

The prototype rejects exterior subdivisions and ambiguous identities. Its absolute
slope floor (1e-8), path averaging, and constrained internal allocation are explicit
experimental choices, not production defaults. It retains parallel paths and can
increase path counts on complex selections; it is not a general-purpose efficient
collapse algorithm. The initial width-filtered audit alone cannot establish exact
exterior geometry preservation.

Before a broader rerun, decide whether to adopt the strict port/split contract and
which local geometric/width/depth aggregation represents the intended experiment.
Then revise the collapse generator to preserve every required port, establish exact
island selections and correspondence, and validate a larger stratified sample.
Simply adding fixed heads to the current solver is insufficient. No shared-file
change is needed for this investigation; coordinate with the Manning work before
production integration. No existing results have been replaced.
