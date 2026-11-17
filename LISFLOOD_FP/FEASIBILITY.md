# LISFLOOD-FP feasibility for river-hierarchy

**Subsequent implementation:** the user clarified that the intended approach is
a custom hydraulic solver and authorized using WSE as the bed proxy. A first
custom diffusive-wave graph implementation and example-56 collapse comparison
are now available in [README.md](README.md). The assessment below concerns the
packaged LISFLOOD-FP options and records the earlier feasibility stage.

Assessment: 6 November 2026. This is a feasibility assessment, not a completed or validated hydraulic implementation.

**Verdict:** much of our preprocessing is reusable, and a channel-focused LISFLOOD-FP experiment is realistic. However, the standard classic 1D river solver is not a direct replacement for our split-and-rejoin network routing. The sub-grid channel solver is the more promising LISFLOOD-FP candidate, subject to a topology-preserving raster pilot. A strictly graph-based 1D implementation would need general junction coupling that the inspected classic solver does not supply.

## What was checked

- Current synthetic generator, canonical topology, fixed BASED, fixed Manning and variable Manning/MCT implementations.
- RAPID/Cunge topology, flow allocation, geometry preparation, forcing, and reviewed-geometry selection code.
- One accessible historical real hydraulic cache: example 22, `RAPID/outputs/cunge_calibration_alpha2/22/850604f0b090a210/frozen_grid.parquet`.
- The developers' official [LISFLOOD-FP 8.1 release](https://zenodo.org/records/6912932), downloaded and source-inspected. Archive MD5 matches the published value `77fa78a9c074d7baf9fbb2157fac4090`.
- Bristol's [classic manual](https://www.bristol.ac.uk/media-library/sites/geography/migrated/documents/lisflood-manual-v5.9.6.pdf) and the developers' [current solver overview](https://www.seamlesswave.com/LISFLOOD8.0.html). The latter also describes 8.2 developments; this assessment does not assume every version or research fork has identical capabilities.

The machine-readable observations and input hash are in [the audit](feasibility/audit_20261106.json). Neither a LISFLOOD-FP executable nor a hydraulic simulation was tested. Source files were inspected in temporary storage, not vendored into this repository.

The active real-geometry manifest points to `/Volumes/PhD/river_hierarchy/network_variants/output/reviewed_master_manifest.json`. That drive and `/Volumes/SyntheticResults20261027` were unavailable. The historical cache establishes the kind of data and topology we have; it does not certify the current production population.

## The solver distinction matters

| Approach | Suitability for this project |
|---|---|
| Classic 1D kinematic/diffusive `.river` solver | Suitable for a single-channel benchmark and tributary trees; not a direct general split-and-rejoin network backend. |
| Sub-grid channels with the local-inertial solver | Promising for a connected, raster-embedded channel network; needs grid, bed, cross-section and junction checks. It is not our existing vector graph running unchanged. |
| Fully resolved 2D shallow-water solver | A possible later benchmark, but a larger terrain/data/computational task. |
| New graph-based 1D hydraulic solver | Fits the topology naturally, but is a separate numerical-model development project, not an input-format conversion. |

The manual distinguishes the classic kinematic/diffusive channels from the sub-grid local-inertial formulation. The latter is embedded in a raster domain. The developers' overview also states that FV1/DG2 do not integrate with the sub-grid channel model; solver names and GPU options are not interchangeable.

### Direct source evidence for the classic 1D limitation

In the official 8.1 archive, `ch_flow.cpp:648–659` specifies upstream discharge from `QFIX`/`QVAR` and takes a tributary's downstream water level from its receiving segment. At `ch_flow.cpp:740–743`, the tributary outflow is passed into that receiving segment. `input.cpp:784–803` records those tributary connections; `input.cpp:471–482` parses `QOUT`/`TRIB`.

This is tributary-to-main-stem coupling. There is no corresponding general upstream junction solve that divides a shared incoming discharge between multiple outgoing channels according to their evolving water levels. Adding externally prescribed split hydrographs would preserve an allocation assumption, not implement that missing hydraulic coupling. Multiple river files represent separate river networks, not a solution to this issue.

Our fresh default synthetic construction contains one split/merge pair in the simple original and two in each nested/crossover original. The historical example-22 cache contains **993 links, 662 nodes, 331 split nodes and 331 merge nodes**. It is directed acyclic, but has 332 independent undirected cycles. Directed acyclicity is not sufficient for compatibility with a tributary-tree solver.

## Data we can reuse, and what must change

| Existing product | Reuse | Required adjustment |
|---|---|---|
| Reviewed directed links/nodes and state lineage | Topology and base/child comparisons | Load through the active geometry selection; preserve parallel branches and correspondence. |
| Physical link lengths and centerlines | Channel location and distance | Use physical lengths, not RAPID numerical length floors. Check rasterized path length and connectivity. |
| `wid_adj_routing`, width families and transect samples | Initial rectangular cross-section widths | State explicitly whether this represents wetted or bankfull width; test along-channel variation and junction transitions. |
| Reconstructed heads, SWORD WSE and `head_us/head_ds` | Reference water-surface constraints | These are not bed elevations. Establish vertical datum and a coherent reference hydraulic state. |
| BASED/width-scaled or Manning depth estimates | Bathymetric prior or initial-state estimate | Select one interpretation and document uncertainty. Normal depth changes with discharge and is not a time-varying bed. |
| Normalized observed forcing and synthetic pulses | Upstream forcing | Export a solver boundary series with explicit units, time origin and point/interval-mean semantics; verify integrated inflow volume. |
| Existing Manning parameters | Starting sensitivity values | Do not automatically transfer calibrated effective routing roughness or legacy defaults. |
| Hydrograph metrics and base/child pairing | Evaluation framework | Add stage, branch-flow, storage and boundary-flux diagnostics. |

The inspected cache has positive widths, depths, slopes and lengths. Width ranges approximately 8.95–2,919 m, depth 0.28–12.16 m, and link length 14–16,787 m. These are historical ranges, not verified production bounds. Their spread warns against choosing a single raster resolution without testing branch separation and short-link representation.

### Bed elevation is the main physical input gap

A possible initial estimate is `z_b(s) = H_ref(s) - d_ref(s)`, but only when the depth and water surface correspond to the same reference condition. The cached BASED estimate can be anchored to peak discharge while SWORD WSE represents a different observation/summary condition. Blind subtraction would give an unjustified bed profile.

At junctions, require a consistent reference water level and explicitly represent any bed steps; beds need not all have identical elevations. Smooth transitions where justified and audit resulting storage and conveyance. Do not simply integrate every existing slope independently: those slopes were constructed from water-surface heads. In the hydraulic run, water-surface gradients must evolve with the solution.

For an idealized synthetic pilot, we can prescribe a fixed bed and rectangular sections explicitly and solve the initial equilibrium. This would be a controlled hydraulic experiment, not surveyed reconstruction. Synthetic diagram coordinates are also not a physical map: the generator can lengthen a branch without changing its endpoint stations, so a raster embedding must preserve its intended hydraulic length separately.

### Additional inputs and choices

- Downstream boundary: measured stage where available, otherwise a documented rating/normal-flow assumption with sensitivity tests. A downstream discharge observation used for evaluation is not automatically a stage boundary.
- Initial condition and spin-up: establish hydraulic equilibrium and monitor storage/flow settling; the current summed-K spin-up rule is not a hydraulic convergence test.
- Bank/floodplain treatment: begin with a deliberately confined channel experiment if the aim is routing comparison. A water mask is not a terrain DEM. Represent artificial confinement explicitly and verify negligible overbank exchange; realistic inundation needs terrain and bank information.
- Sub-grid rasterization: check false contacts between nearby branches, missing diagonal connections, merged parallel channels, added junction storage, path length and channel volume. One raster cell cannot retain arbitrary separate overlapping channel states. Refine or reject a representation that changes the target network.

The manual specifies width, Manning roughness and bed elevation for classic channels, and raster inputs for sub-grid channels. It also distinguishes sub-grid bank information from the DEM elevation controlling overtopping. Exact input keywords and boundary units should be checked against the chosen pinned release during adapter implementation.

## What changes scientifically

`RAPID/src/rapid_tools/cunge.py::freeze_grid` assigns fixed local width fractions. The current synthetic `hydraulics.py::route` uses those fractions for all three depth methods. Even variable Manning/MCT therefore retains prescribed branch allocation.

A connected hydraulic model would allow allocation to respond to evolving water levels, bed geometry and resistance, subject to its equation approximations. Backwater and junction interaction could change timing and attenuation after a collapse. This is a useful extension of the research question, but it means a mismatch with RAPID is not automatically an implementation error.

Separate changes in routing equations from changes in allocation and bathymetry. Keep the bed/roughness assumptions, forcing, external boundaries and unaffected geometry consistent across paired states. Recompute the new model's responses before fitting any new geometry-response relationships; existing regression coefficients describe the existing routing experiments.

## Recommended implementation and acceptance sequence

1. **Pin the solver and test installation.** Use a documented CPU sub-grid-capable release, record source checksum/build flags, and reproduce its supplied benchmark. The official build documentation emphasizes Windows/Linux; a working macOS build has not been established here.
2. **Single rectangular channel.** Reuse a synthetic hydrograph, prescribe a fixed bed, and verify steady normal flow, event propagation, units, boundary integration, and storage balance. This is also a useful classic-1D control.
3. **One split-and-rejoin island.** Start symmetric, then vary one branch's width/length. Verify symmetry where expected, junction conservation and hydraulic redistribution. Confirm that all intended paths remain separate on the grid.
4. **One collapse pair.** Compare the original island and collapsed channel under shared forcing and external boundary conditions. Examine discharge, stage, branch allocation, volume, centroid and peak timing.
5. **One small real case.** Once the data drive is mounted, audit current geometry, reference WSE/depth compatibility, bank assumptions and downstream boundary. Treat inferred bathymetry as a sensitivity ensemble.
6. **Only then batch the hierarchy.** Create separate `lisflood_fp/prep` and `lisflood_fp/run` products with manifests recording geometry identity, bed scenario, solver version, grid, forcing and boundaries. Preserve current routing results as comparators.

Proposed pilot acceptance targets, to be frozen before comparisons: cumulative mass residual below 0.1% of event inflow volume after accounting for storage and every boundary/source term; refinement changing event centroid by less than 1% of travel time and peak magnitude by less than 1%; peak timing stable to the selected reporting resolution. Assess small parent–child effects against absolute numerical error, not only these percentages. Require no topology changes under accepted rasterization, no unexplained negative depths/nonfinite states, and demonstrably settled initialization. These are proposed project criteria, not guarantees supplied by LISFLOOD-FP.

**Recommended decision:** proceed with a small sub-grid feasibility pilot if LISFLOOD-FP is the intended platform. Do not commit the whole hierarchy to the classic 1D solver. If the conference referred to a modified 1D implementation with general junction coupling, inspect that exact implementation before choosing between the two routes.
