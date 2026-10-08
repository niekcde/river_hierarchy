# Hydraulic production policy — 3 October 2026

Decision reference: `shared-head-anchored-depth-signed-inflow-20261003`.
This records the implementation following the shared-head reconstruction audit,
and supersedes the earlier stability-driven celerity regularization proposal.

## Frozen choices

| Component | Choice |
|---|---|
| Reference profile | Bounded robust fit; documented bridges and boundary heads retained |
| Changed unit interior | Joint shared-head reconstruction; unchanged exterior heads fixed |
| Depth | `D = D_ref (W/W_ref)^0.65`, using stored BASED reference anchors |
| Local width/depth ratio | `Kb = W/D`, identical in routing and resistance |
| Manning roughness | `n = 0.035` |
| Celerity | `(5/(3n)) D^(2/3) sqrt(S)`, capped at 0.05–5 m/s |
| Segmentation | 200 m target; existing rounding and 200 m effective-length floor |
| Muskingum X | 0.1 |
| Routing K | Effective segment length / capped celerity; fixed across clocks |
| Numerical policy | Signed current-inflow coefficient allowed; previous-outflow coefficient remains in `[0,1)` |
| Initial condition | Steady network discharge under the first forcing value |
| Equivalent resistance | Preserve Chezy law and coefficient 22; update local geometry and use fresh caches |

For each case the clock is shared across every included state and is an integer
divisor of the forcing interval, below `0.99 min[2K(1-X)]`. Neither K, slope, X,
roughness nor effective length is adjusted to achieve that clock. The signed
coefficient is the coefficient multiplying **current inflow**, not discharge.
Every native routing substep is checked for finite values and its minimum flow
is recorded. The batch rejects negative flow beyond a `1e-10` relative
floating-point tolerance. There is no flow clipping.

Legacy defaults remain available for historical reproduction. New work must
use `rapid_tools.hydraulic_policy.production_prep_config()`, rather than silently
loading an old preparation manifest. The engine also checks timestep overrides.

## Production verification and refresh

`run_hydraulic_policy_refresh.py` uses production preparation and routing APIs.
It checks segment counts and K against the audited frames and checks that
resistance and routing use identical depths. All 2,230 unique hydraulic states
passed the K/depth check. Inventory aliases retain all 3,551 included entries;
five pre-existing topology exclusions remain explicit.

The initial native checks reproduced the investigation's paired peak shifts:

| Case / child | Native minus investigation timing contrast |
|---|---:|
| 11 / S003_unit_18 | +3.36 s |
| 46 / S004_unit_24 | +1.26 s |
| 46 / S010_unit_105 | −0.48 s |

These small differences include the difference between sampled forcing-interval
outputs and the investigation's band-limited output grid. They do not establish
universal positivity or physical applicability for other forcing.

`refresh_selected_domain_resistance.py` refreshes affected-domain comparisons,
including island domains reconstructed from the existing source-unit selection
and the injective node mapping. It reports missing domains and failed solves;
it never substitutes whole-network resistance for a missing unit comparison.
Case 11's 42 entries were recomputed successfully. The all-case refresh is a
separate, long-running operation. Historical aggregate tables must not import
old selected-domain transition values under the new depth policy.

Outputs live in a fresh policy-version directory; historical experiment results
remain available until the refreshed products pass coverage checks. Do not mix
the two generations in analysis notebooks. Read `refresh_status.csv`,
`selected_domain_resistance/comparisons.csv`, and each policy manifest to establish
completion, rather than assuming a directory's existence means success.

Examples 13, 45 and 48 had no saved actual-event outlet run. Their new base
runs have now passed production preparation and routing. Their recorded
gauge event windows in `experiment_manifest.json` provide the forcing for new
runs, with shared constant padding. Earlier investigation coverage for these
three was a standard pulse and is not an actual-event validation.

The full sequential batch was launched on 3 October 2026 under
`/Volumes/PhD/river_hierarchy/network_variants/hydraulic_refresh/shared-head-anchored-depth-signed-inflow-20261003`.
`batch_status.json` records progress and `full_refresh.log` records output. A failed
preparation, flow check or unresolved resistance comparison stops the batch.
The regression suite passed 100 tests before launch; the refreshed full batch
is not yet a completed validation result.

Example commands, from the repository root:

```sh
python RAPID/run_full_hydraulic_refresh.py --output /path/to/fresh/policy/output
# Individual stages, when needed:
python RAPID/run_hydraulic_policy_refresh.py --route --compact --resume --output /path/to/fresh/policy/output
python RAPID/refresh_selected_domain_resistance.py 02 03 05 11 13 15 21 22 37 41 45 46 48 54 55 56 --output /path/to/fresh/policy/output
```

Compact runs preserve lossless compressed full reach discharge, full attributes
in Parquet, reach geometry, outlet hydrographs and numerical audits. Geometry
stores reach IDs and shapes; the analysis loader rejoins attributes from the
mapping. This avoids duplicating large source metadata on every subreach.
The batch checks available disk space before each state. Resume requires matching
code, policy, hydraulic frame and forcing, plus a completed discharge artifact.

## Repository cleanup

Removed 4,610 superseded generated files (about 2.16 GB): two abandoned initial
slope attempts and the superseded pre-reconstruction full-screen waveforms.
Current reconstructed contexts and waveforms remain on disk, ignored by Git.
Compact final audit evidence is retained in `validation/hydraulic_policy_20261003`.
Notebook source is retained; executed preview copies live in the ignored
`assets/notebook_previews_20261002` directory. Regression tests and reproducible
investigation scripts are retained. No commit or staging operation was performed.

See also [reference-profile methodology](ROBUST_REFERENCE_PROFILE_METHOD.md) and
[shared-head reconstruction](CHILD_HEAD_RECONSTRUCTION_METHOD.md).

## Case 15 boundary recovery — 3 October 2026

The initial batch stopped at `island_15_025`: parent boundary 45 was absent
from exact-coordinate matches. Its child counterpart is node 416, displaced
7.0317 m during reconstruction. Retained-link lineage maps parent link 64
(45→42) to child link 629 (416→42); the opposite endpoint 42 remains matched.
The recovered child comparison is closed and uses links 621–624, compared
with parent links 68–72. The fallback requires a unique unchanged one-to-one
link with consistent directed endpoints and a matched opposite end; ambiguous
evidence still fails. This changes comparison-domain matching only, not slope,
depth, celerity, routing K or hydrographs.

The corrected resistance ratio is approximately 0.241599. All 420 case 15
comparison entries now solve. All 263 completed routing identities were checked
and remain valid; no routing rerun is needed. The batch journal retains its
original failure and records the repair before continuation from case 21.

## Case 41 enclosing comparison domains — 4 October 2026

Ahead of the batch reaching case 41, preflight identified 17 empty saved
comparison memberships and three island comparisons whose original boundary
nodes no longer exist after collapse and serial merging. Unlike case 15,
unchanged-link lineage cannot recover these genuinely removed boundaries.

`repair_missing_comparison_domains.py` stages the smallest enclosing comparison
with surviving matched boundaries. Candidates follow upstream dominators and
downstream postdominators of the original parent boundaries. A candidate must
contain every original parent-domain link and be hydraulically closed in both
parent and child. The unique inclusion-smallest valid candidate is selected;
ambiguous alternatives and whole-network substitution are rejected. This does
not infer node identity from proximity or change hydraulic geometry or parameters.

Twenty memberships were installed: 13 independent, four compound, and three
island comparisons. Original empty files were backed up; each installed file
has a dated recovery audit recording original/new boundaries, added links,
and hydraulic-frame hashes. Historical numeric resistance results were left
untouched. All 291 case 41 comparisons passed the staged resistance preflight;
all 291 also passed using the installed source files, with identical resistance
ratios. The maximum relative solver residual was below 2.5e-10.
34 relevant regression tests passed. Evidence is retained in
`validation/hydraulic_policy_20261003/case41_domain_repair_20261004`.

These 20 resistance ratios describe an enclosing affected section, which can
be substantially larger than the originally selected unit. Subsequent geometry
and exposure predictors must use the recorded comparison domain consistently
and retain the nominal unit selection as separate provenance. This repair is
not evidence that the original missing-boundary unit comparison was valid.

Only domain membership inputs were promoted. The running batch reads these
files when it reaches case 41; restarting case 22 or rerunning completed routing
is unnecessary. Case 41 routing still needs to pass the normal batch checks.

## Fractional routing-clock investigation — 4 October 2026

The slow-run investigation found an arithmetic constraint in the production
clock selection: the routing clock must currently be an integer-second divisor
of the saved forcing interval. Case 37 has a 755 s forcing interval and a
99%-of-upper-bound limit of about 95.85 s, but integer divisibility selects 5 s.
Case 46 has a 1085 s interval and a corresponding limit of 71.28 s, selecting
35 s. These small integer divisors are not themselves hydraulic requirements.

The isolated investigation `investigations/check_fractional_routing_clocks.py`
tests case 37 at 5, 47.1875 and 94.375 s, and case 46 at 35, 33.90625 and
67.8125 s. All clocks divide their forcing interval into an integer number of
equal substeps. Hydraulic inputs, held forcing, full event/tail duration and
forcing-interval output averaging remain fixed. Only the exactly steady initial
plateau is skipped, as in production. Fractional clocks are passed directly to
the existing coefficient and sparse recurrence functions in an isolated runner;
the production preparation/execution wrappers still coerce clocks to integers
and have not been changed by this investigation.

The retained previous-outflow coefficient bound passes for all 397 case 37
and 224 case 46 unique geometries. Full native-recurrence hydrographs cover
four case 37 and five case 46 geometries at all three clocks. Selection includes
the base, minimum timestep-bound and longest-travel geometries, extrema of the
previous three-response screen, and compound/island coverage. Every segment
is checked at every substep for finite and nonnegative flow. The spatial
`lfilter` evaluation uses the same DAG recurrence; it is cross-checked against
the existing sparse solver, with additional fractional-clock split/rejoin tests.

At the largest proposed clock, maximum outlet differences are 0.08053% and
0.01162% of the input event discharge range for cases 37 and 46 respectively.
Maximum absolute peak shifts are 44.01 s and 17.27 s. The corresponding maximum
changes in parent–child timing differences are much smaller: 5.41 s and 0.157 s.
Maximum changes in event-duration contrasts are 0.000675 and 0.00000376
percentage points; in peak-excess contrasts, 0.005205 and 0.00001075 percentage
points. All tested full-event segment flows remain positive. These are observed
pilot differences, not proof for arbitrary forcing or every unsimulated child.

`investigations/benchmark_fractional_routing_clocks.py` verifies both base events
with the production sparse recurrence at their original and proposed clocks.
Compact evidence is retained in
`validation/hydraulic_policy_20261003/fractional_clocks_20261004`; detailed outlet
and segment-audit Parquet files remain in ignored `outputs/fractional_clock_check_20261004`.
Timing measurements cover routing, excluding preparation, export/compression,
resistance calculation and any future parallel-worker contention.
The full-event sparse base checks agree with the spatial recurrence to within
3.9e-13 of the inlet discharge range. Measured routing times were 77.23 versus
4.57 s for case 37 (16.9-fold speedup), and 114.28 versus 45.41 s for case 46
(2.52-fold). These are single measurements alongside the ongoing batch, not
guaranteed whole-batch speedup factors. Both fractional-clock regression tests
passed.

The pilot supports 94.375 s for case 37 and 67.8125 s for case 46 as candidates
for implementation with the existing full-run flow checks retained. It does not
change the running batch, scientific hydraulic choices, or completed outputs.

### Applied to the production batch at 11:27 CEST, 4 October 2026

After the pilot, the user authorized applying these clocks. Preparation and
execution now preserve explicit fractional timesteps, enforce the same
coefficient bounds and require whole substep counts within each forcing
interval. Integer clock selection remains unchanged. The refresh selects
94.375 s only for case 37 and 67.8125 s only for case 46, rechecking their
forcing divisibility and 99%-of-upper-bound safety margin. The cache identity
now includes the routing timestep explicitly.

Seventy preparation, hydraulic-policy and fractional-clock tests passed.
Both base states completed full production preparation, routing and lossless
compressed publication. End-to-end times were 18.23 s for case 37 and 124.76 s
for case 46. Their outlet waveforms agree with the pilot within 9.7e-7 of the
input discharge range; no negative native-step flows were found. These two
published states are reusable by the resumed batch.

The original process was interrupted and restarted from case 22 with one
worker and one BLAS/OpenMP thread. All 280 completed case 22 states were
verified against their exact prior implementation/configuration/frame/forcing
identity and unchanged 55 s clock. A narrowly scoped compatibility check
allows their reuse; it does not accept old-clock outputs for 37 or 46.
The interrupted unfinished state is rerun. Earlier completed cases are not
scheduled again. The old progress journal and implementation hashes remain
in the audit trail. `resume_launch_fractional_20261004.json` records the new
process and its process-bound idle-sleep inhibitor. This does not prevent
lid-closure sleep or loss of power/external storage.

The launch-time single-worker estimate is 14–20 hours remaining, conditional
on uninterrupted execution and successful checks. It uses actual end-to-end
base measurements, current case 22 pace, and estimated work for the other
cases; it is not a measured completion guarantee or a parallel-run benchmark.

This refresh covers the 16 real-network cases, 2,230 unique ready hydraulic
states with aliases retained across independent, compound and island
comparisons. It retains full-segment discharge at every forcing/output
interval in `Qout.nc.gz`, outlet hydrographs/metrics, reach geometry, Parquet
hydraulic attributes, hydraulic contexts, numerical audits, and recalculated
selected-domain resistance comparisons. Native substeps are audited but their
individual discharge arrays are not stored. Reproducible preparation files
are removed after publication. Synthetic grids, parameter ensembles and
notebook regeneration are outside this batch.

### Two-worker scheduling — 17:36 CEST, 4 October 2026

To target completion before 06:00 CEST on 5 October, the user authorized a
second worker. Case 37 had completed both routing and resistance. The serial
process was interrupted during case 41 and replaced with two disjoint queues:
worker A handles 41, 45, 48, 54, 55 and 56; worker B handles 46. Both reuse
matching completed results, including the case 46 base. Scientific settings,
output products and routing implementation identities remain unchanged.

`run_full_hydraulic_refresh.py --journal` permits separate progress files.
Worker A retains `batch_status.json`; worker B writes
`batch_status_worker_B.json`. Overall completion therefore requires checking
both journals. Logs are `worker_A_20261004.log` and `worker_B_20261004.log`;
the previous `full_refresh.log` is historical after the split.
`parallel_launch_20261004.json` records the assignments and process IDs.
The original journal was backed up before interruption. Each worker has its
own process-bound idle-sleep inhibitor and BLAS/OpenMP/MKL/Accelerate/NumExpr
thread counts set to one. Workers are not pinned to specific physical cores.
There is no overlap between their example/output directories; future launches
must preserve that exclusivity. This scheduling change does not reduce output
coverage or skip resistance validation.

### Completed-run analysis and connectivity correction — 5 October 2026

The hydrograph analysis notebook now reads the versioned refresh directly,
including island comparisons and the refreshed resistance comparison domains.
Historical experiment trees provide selection and observed-gauge metadata only.
There are 2,230 distinct routed geometries and 3,519 comparison rows; two
technical promotion aliases remain in coverage accounting but are excluded
from plots. The 20 expanded case 41 domains use their actual refreshed domain
geometry. Event duration uses fixed parent-derived 1% departure/return
tolerances, alongside peak excess and timing responses.

The completed coefficient audit found negative applied current-inflow weights
in 1,777 distinct states across 12 of 16 cases. Among the 5,315 original base
network links, 367 contain affected internal routing segments (4,164 segments).
Across all states there are 89,218 affected original-link occurrences and
968,893 affected segment occurrences; these repeat physical links across
experiments. External-source first segments are excluded because the held
forcing enters through the combined weight a+b. The minimum a is -0.0921642;
all previous-outflow weights remain in [0,1). Actual event audits at every
internal timestep report no negative discharge, with a minimum of 0.498786
m3/s after the correction below. This is event-specific evidence, not a
positivity guarantee for arbitrary forcing.

Notebook validation detected a separate connectivity bug in case 15 geometry
18cec08d236aa2f032da (S054_unit_1 and island_15_002): an unsplit internal link
with original ID 0 became routing reach ID 0, which also denotes no downstream
reach. Its outlet response was therefore zero. A scan of all 2,230 geometries
found no other collision. Preparation now assigns a unique positive routing
ID while retaining original link ID 0 in the mapping. That one geometry was
rerun without changing hydraulic parameters or resistance; its initial and
final outlet plateaus match the forcing within 1e-6 relative tolerance.
Original outputs and the full status table are preserved under case 15's
corrections_20261005 directory, with repair.json recording the correction.
The analysis loader now rejects flat outlet responses to varying forcing.
Nonnegative flow alone would not detect this connectivity error.

Reproduce the counts with investigations/audit_completed_coefficients.py;
the tables are in validation/hydraulic_policy_20261003/
completed_coefficients_20261005. The updated notebook can be restarted and
run against the mounted external production volume without rerouting.
