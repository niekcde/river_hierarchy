# Selected-collapse boundary contract — investigation, 27 October 2026

Status: isolated prototype; no production adoption or broader rerun proposed yet.
The comparison parent is S000_base. A later sequential-collapse workflow must explicitly
name its immediate parent instead. Files in this directory and the new local_boundary
output directories are the only files owned by this investigation.

## Scope and invariants

The selected parent reach set is declared before solving. The boundary includes every
node incident to both selected and exterior reaches, plus network terminals inside the
selection. A boundary can have both incoming and outgoing selected reaches even when
an earlier entry/exit label suggests otherwise. All such incidences matter.

| Quantity | Exterior contract | Replacement allowance |
|---|---|---|
| Geometry | Exact parent polylines, directed endpoints, length, connectivity; IDs may change only with proven injective correspondence | Change inside the declared section; meet original boundary positions exactly |
| Heads | Every exterior node head and every boundary head fixed | One shared unknown per internal node |
| Slopes | Parent slopes fixed and consistent with head drop / length | Positive slopes consistent with shared heads; explicit floor checked on every edge |
| Width, depth, roughness | Parent values fixed, including under a Manning-depth policy | Explicit declared local policy; n remains reference-calibrated and frozen |
| Reference flow | Every exterior Q_ref and flow fraction fixed | Nonnegative conservative allocation with original gross incoming AND outgoing flux at every port |
| Junction splits | Exterior reach split fractions fixed | Interior splits follow constrained reference flow, not a fresh global width allocation |
| Routing | Parent celerity, diffusivity, K, X, segment count, effective cell lengths and cap settings fixed | Recompute locally using the same n, beta and cap settings |
| Numerical clock | Identical clock across each comparison, with convergence check | Replacement may require a smaller common clock; this changes numerical recurrence coefficients for both compared states, not physical K/X |
| Forcing | Same inlet forcing and initial-equilibrium convention | No calibration to make responses match |

Routed time-varying discharge is an outcome, not an invariant. An upstream replacement
can legitimately change downstream timing, attenuation and transient storage on a
reach whose coefficients are unchanged. At a split, frozen fractions conserve the
instantaneous incoming signal. With the current fixed-reference model there is no
justification for recomputing exterior Q_ref, depths or K/X from that changed signal.
Reference allocation must also match the initialization equilibrium; simply copying
Q_ref while retaining incompatible child split fractions is not a correction.

## Failure rules

Missing/ambiguous exterior correspondence, changed exterior geometry/direction,
missing boundary ports, inconsistent shared heads, nonpositive imposed drops,
cycles, infeasible reference-flow demands, or failed solver residuals block the case.
Never expand selection, release a boundary head, clip a negative slope, or relax a
port flux implicitly. Subdivided exterior reaches are currently rejected: equivalent
continuous geometry alone does not prove identical discrete routing transfer.

For a prescribed slope floor, every directed path between fixed heads needs at least
the sum of edge floor-drops. The linear feasibility solve checks all such constraints
simultaneously. Positive heads/slopes do not establish flow feasibility: the separate
node/port linear program must pass. A net port flux alone is insufficient at mixed
entry/exit nodes if exterior split coefficients are to remain fixed.

The prototype uses an explicit absolute slope floor of 1e-8, not the production
relative-floor objective. This is a feasibility trial, not a proposed replacement
calibration. It fits saved child slopes only inside the selected section. Reference
flow minimizes absolute departure from the child allocation subject to exact port
constraints; parallel-path allocation can be nonunique and requires a documented
physical policy before production. No exterior head/slope fitting penalty is used:
exterior values are hard constraints.

## Existing workflow and previous repairs

`shared_head_reconstruction.py` frees the interiors of affected owned domains;
compound-member edits therefore release heads across their containing compound.
Its objective fits relative slopes with a floor. The initial parent heads are not
preservation targets. Optional `fixed_node_ids` exists, but is not the default policy.
`freeze_grid` propagates width-based branch fractions over the whole child graph;
its resulting Q_ref enters diffusivity, X and grid selection independently of depth.

The active ReviewedGeometry release is `master_integrated_20261009`, with hydraulic
regeneration complete and master products published. Earlier boundary geometry work
was adopted. That does not establish strict hydraulic locality: the sampled saved
parameter frames still declare `existing_domain_policy`. Geometry correspondence
accepted under an earlier tolerance also does not establish exact exterior equality.

## Alternative tested when saved topology is infeasible

The port-path prototype starts from the selected parent subgraph, enumerates paths
between all boundary ports, and replaces each path by one reach. It stops at mixed
ports and retains parallel paths separately. Reference path fractions follow the
parent flow splits. Path lengths and polylines follow parent paths; width and depth
are length-weighted means, an explicit experimental policy rather than measured
new-channel geometry. Shared exterior geometry and all its parameters are inherited.

This changes representation locally and can change routing response. It is NOT the
same intervention as the saved raster collapse, nor a geometry-equivalent repair of
that collapse. Path enumeration may grow combinatorially; the implementation has an
explicit 10,000-path limit. Repeated/overlapping path geometry must not be interpreted
as newly observed parallel river channels. A preferred physical aggregation policy
and intended island footprint remain decisions before broader use.

## Reproduction and coordination

Use `/opt/anaconda3/envs/river-hierarchy-rivgraph/bin/python`:

- `-m pytest -q RAPID/investigations/local_boundary_20261027/test_local_contract.py`
- `RAPID/investigations/local_boundary_20261027/run_pilot.py`
- `RAPID/investigations/local_boundary_20261027/review_results.py`

The pilot refuses to replace an existing output directory; choose a new OUT path for
a fresh run. The initial strict screening is preserved in `RAPID/outputs/local_boundary_20261027`;
the full pilot is in `RAPID/outputs/local_boundary_20261027_v2`. No shared modules,
notebooks, production routing inputs, model fits or existing results were edited.
The continuing Manning work should consume the eventual boundary-contract decision;
freezing depth alone cannot correct the scope. Coordinate before touching shared
reconstruction, `freeze_grid`, or Manning preparation code.
