# Width Method Project Checklist

This is the project-wide implementation checklist for replacing the current
regenerated-link width workflow with the new two-bank, component-aware,
distance-transform-based method described in:

- [WIDTH_METHOD_OVERHAUL_SPEC.md](/Users/6256481/Code/river-hierarchy/network_variants/WIDTH_METHOD_OVERHAUL_SPEC.md)
- [WIDTH_METHOD_REPO_CHECKLIST.md](/Users/6256481/Code/river-hierarchy/network_variants/WIDTH_METHOD_REPO_CHECKLIST.md)

This checklist is intentionally **not** tied to one example. It is the migration
plan for the full `river-hierarchy` codebase.

## 1. Project Rules

These rules should hold throughout the migration.

- Keep one target width methodology across the project.
- Do not mix native RivGraph width for one quantity and custom transect width
  for another in the final design.
- Preserve stable downstream field names where possible:
  - `wid_adj_total`
  - `wid_adj_wet`
  - `wid_adj_dry`
  - `wid_med_total`
  - `wid_med_wet`
  - `wid_med_dry`
- Root reusable caches at the experiment level, not per state.
- Keep a temporary legacy path only as long as needed to migrate safely.
- Avoid building the new algorithm as conditionals inside the old
  `compute_width_families(...)` logic for long.

## 2. Deliverable Definition

The migration is complete only when:

- `run_collapse_experiment.py` uses the new width method automatically
- `generate_network_variant(...)` and `derive_anchored_variant_from_realized(...)`
  both use the new width method
- state width refresh tooling uses the new width method
- equivalent resistance reads width outputs produced by the new method without
  special handling
- RAPID prep continues to consume the same width-family fields without needing
  method-specific branching
- legacy `width_scope_mask` logic is removed or clearly deprecated

## 3. Workstream A: Core Data Model

### A1. Replace width-scope data model

Current model:

- `collapsed_mask`
- `width_scope_mask`

Target model:

- `collapsed_total_mask`
- `origin_mask`
- experiment-level boundary-component cache

Tasks:

- define canonical semantics for `origin_mask`
  - `0` = blocking land/background
  - `1` = original wet
  - `100+` = selected collapse-added pixels
- define active obstacle-set semantics
- define how selected collapse objects map to `100+` labels

### A2. Update common output structures

Files:

- [variant_generation.py](/Users/6256481/Code/river-hierarchy/network_variants/variant_generation.py)

Tasks:

- update `NetworkVariantOutputs`
- replace `width_scope_mask_path` with:
  - `origin_mask_path`
  - `rivgraph_distance_path`
  - optional cache references
- update manifests and `output_paths`

## 4. Workstream B: New Width Engine Module

### B1. Create a dedicated module

New file:

- `network_variants/width_two_bank.py`

This module should contain the full target method and should not depend on the
old `width_scope_mask` design.

### B2. Implement boundary preprocessing helpers

Tasks:

- extract land-water boundaries
- remove open-end border artifacts
- label connected obstacle boundary components
- build component metadata
- build spatial lookup

### B3. Implement state-level helpers

Tasks:

- build `origin_mask`
- define active obstacle set from the collapse selection
- resolve centerline sample inputs

### B4. Implement sample-level width logic

Tasks:

- global DT lookup from collapsed-state RivGraph distance raster
- candidate component search
- left/right side classification by local normal
- local controlling pair selection
- primary width:
  - `d_left + d_right`
- QC chord width:
  - `||p_right - p_left||`
- decomposition along `bank_left -> center -> bank_right`
- compute:
  - `width_total`
  - `width_wet`
  - `width_dry`
  - `n_channels`

### B5. Implement aggregation

Tasks:

- sample-to-link aggregation
- RivGraph-style end trimming
- quantiles / medians / means
- QC rollups

## 5. Workstream C: Experiment-Level Cache

### C1. Introduce cache location

Experiment runner should own:

- `width_method_cache/`

Suggested contents:

- boundary-component labels or boundary-pixel store
- component metadata
- optional persisted spatial index assets

### C2. Build once, reuse everywhere

Files:

- [collapse_experiment.py](/Users/6256481/Code/river-hierarchy/network_variants/collapse_experiment.py)

Tasks:

- create the cache once per experiment root
- pass cache references into all variant generation paths
- reuse cache for:
  - realized
  - anchored
  - refresh/patch workflows

## 6. Workstream D: Variant Pipeline Integration

### D1. `generate_network_variant(...)`

Tasks:

- stop depending on `width_scope_mask`
- write `origin_mask.tif`
- persist RivGraph `Idist`
- call the new width engine
- write new sample/link outputs

### D2. `derive_anchored_variant_from_realized(...)`

Tasks:

- use the same new width engine
- reuse experiment-level boundary cache
- rebuild state `origin_mask`
- keep anchored and realized outputs method-consistent

### D3. Refresh/patch path

Current tool:

- [run_width_scope_patch.py](/Users/6256481/Code/river-hierarchy/network_variants/run_width_scope_patch.py)

Tasks:

- replace or rename this tool for the new method
- make it rebuild width outputs from:
  - existing collapsed mask
  - existing directed links
  - existing RivGraph distance
  - reviewed parent inputs
- ensure it does not rerun RivGraph

## 7. Workstream E: Manifest, Files, and Artifacts

### E1. State outputs

Under `variant/mask/`:

- keep:
  - `*_collapsed.tif`
- add:
  - `origin_mask.tif`
- remove/deprecate:
  - `width_scope_mask.tif`

### E2. Width outputs

Under `variant/widths/`:

- keep:
  - `link_width_samples.csv`
  - `link_width_families.csv`
  - `links_with_width_families.gpkg`
- update sample schema to include:
  - left/right component IDs
  - left/right bank coordinates
  - `d_left`, `d_right`
  - `width_total`
  - `width_wet`
  - `width_dry`
  - `n_channels`
  - QC flags

### E3. Manifest fields

Add:

- `origin_mask`
- `rivgraph_distance`
- `width_method_version`
- `width_method_cache_dir`

Remove/de-emphasize:

- `width_scope_mask`
- `width_scope_patch`

## 8. Workstream F: Downstream Compatibility

### F1. Equivalent resistance

File:

- [equivalent_resistance_experiment.py](/Users/6256481/Code/river-hierarchy/network_variants/equivalent_resistance_experiment.py)

Tasks:

- confirm it still reads `links_with_width_families.gpkg`
- confirm the width fields it expects remain populated
- avoid adding method-specific branches if field names stay stable

### F2. RAPID prep

Tasks:

- preserve compatibility with:
  - `width_wet` sample usage
  - `wid_adj_wet` representative usage
- confirm new outputs remain valid for:
  - equivalent resistance
  - KB estimation
  - subreach width sampling

### F3. Notebook and diagnostic consumers

Tasks:

- update analysis notebooks only after width outputs stabilize
- document changed sample-level columns

## 9. Workstream G: Testing

### G1. Replace old width-scope tests

Current tests tied to the old method are legacy and should be retired after
migration.

### G2. Add project-level test coverage

Required categories:

- open-end filtering
- boundary-component labeling
- candidate component lookup
- left/right selection
- simple-channel width
- simple two-thread collapse
- compound/multi-thread collapse
- foreign `100+` exclusion
- QC triggers
- refresh/patch path without RivGraph rerun
- realized and anchored pipeline parity

### G3. Add regression fixtures

Create compact synthetic masks that isolate:

- base single-thread channel
- simple bifurcation-confluence with one removable island
- nested/compound geometry with multiple candidate obstacle components
- border/open-end example

These should be small enough for fast unit tests and not depend on one real
example directory.

## 10. Workstream H: Performance

### H1. Do not implement per-component full-scene EDT

This is explicitly rejected as the default strategy.

### H2. Adopt the selected performance approach

Chosen first implementation:

- one global DT per state from RivGraph `Idist`
- local candidate lookup via cached boundary-component spatial index
- nearest-point evaluation only for candidate components

### H3. Benchmark on project-scale runs

Measure:

- base-state preprocessing time
- per-state width recompute time
- refresh/patch time
- memory growth with many obstacle components

The performance target should be good enough for:

- independent-units experiments
- compound-bubbles experiments
- realized and anchored families

## 11. Workstream I: Migration Control

### I1. Introduce a clear migration boundary

Do one of the following:

- add a separate new entry point and switch callers over once ready
- or replace the old path in one controlled merge

Do **not** leave a long-lived half-old half-new hybrid inside the same function.

### I2. Preserve fallback only briefly

If a temporary legacy fallback is needed:

- label it explicitly as legacy
- isolate it in code
- remove it after validation

### I3. Version the method

Add explicit manifest field:

- `width_method_version`

So outputs can be traced unambiguously during migration.

## 12. Recommended Delivery Phases

### Phase 1. Infrastructure

Deliver:

- new width module skeleton
- updated manifests/output dataclass
- RivGraph distance persistence
- experiment-level cache location

### Phase 2. Boundary cache

Deliver:

- open-end filtering
- component labeling
- metadata and spatial index creation

### Phase 3. State masks

Deliver:

- `origin_mask`
- active obstacle-set construction
- selected-collapse `100+` labeling

### Phase 4. Sample-level engine

Deliver:

- local bank-pair selection
- two-bank decomposition
- sample QC

### Phase 5. Aggregation and outputs

Deliver:

- link family outputs with stable field names
- sample output schema
- GPKG merge outputs

### Phase 6. Pipeline cutover

Deliver:

- `generate_network_variant(...)` switched
- `derive_anchored_variant_from_realized(...)` switched
- refresh/patch tool switched

### Phase 7. Validation and cleanup

Deliver:

- tests migrated
- equivalent resistance and RAPID compatibility verified
- legacy `width_scope_mask` path removed or deprecated decisively

## 13. Exit Criteria

The project-wide migration is done when:

- no active pipeline depends on `width_scope_mask`
- all experiment modes use the same two-bank width method
- width outputs are stable enough for equivalent resistance and RAPID reruns
- the refresh/patch path rebuilds the new width outputs without rerunning
  collapse or RivGraph
- the old width method is removed or clearly archived as legacy
