# Width Method Repo Checklist

This checklist maps the target methodology in
[WIDTH_METHOD_OVERHAUL_SPEC.md](/Users/6256481/Code/river-hierarchy/network_variants/WIDTH_METHOD_OVERHAUL_SPEC.md)
onto the actual codebase.

The goal is not to prescribe exact code, but to specify:

- which files/functions must change
- which new files/helpers should be added
- which current artifacts become obsolete
- which tests need replacement

## 1. Recommended Implementation Shape

Do **not** try to implement the full overhaul directly inside the existing
`compute_width_families(...)` body.

Recommended structure:

1. keep orchestration in `variant_generation.py`
2. move the new width method into a dedicated helper module
3. keep a refresh/patch entry point for already-generated variants

Recommended new module:

- `network_variants/width_two_bank.py`

Reason:

- the new method is conceptually different from the current
  transect-geometry/`width_scope_mask` workflow
- isolating it avoids making `variant_generation.py` even larger and makes the
  old/new methods easier to compare during migration

## 2. New Module to Add

### File to add

- `network_variants/width_two_bank.py`

### Functions to add there

Suggested helpers:

- `extract_open_end_filtered_boundaries(...)`
- `label_obstacle_boundary_components(...)`
- `build_boundary_component_index(...)`
- `build_or_load_example_boundary_cache(...)`
- `build_state_origin_mask(...)`
- `build_state_active_obstacle_index(...)`
- `compute_global_dt_width_seed(...)`
- `select_candidate_boundary_components(...)`
- `select_local_controlling_bank_pair(...)`
- `decompose_two_bank_section(...)`
- `compute_two_bank_width_samples(...)`
- `aggregate_two_bank_width_families(...)`

This module should own:

- boundary-component preprocessing
- local bank-pair selection
- wet/dry/channel decomposition
- sample and link aggregation

## 3. Existing Orchestration to Update

### 3.1 `NetworkVariantOutputs`

File:

- [variant_generation.py](/Users/6256481/Code/river-hierarchy/network_variants/variant_generation.py:101)

Current issue:

- the dataclass still exposes `width_scope_mask_path`
- it does not expose the RivGraph distance raster
- it does not expose a state `origin_mask`

Required changes:

- deprecate or remove:
  - `width_scope_mask_path`
- add:
  - `origin_mask_path`
  - `rivgraph_distance_path`
  - optional `boundary_component_cache_dir` or similar

## 3.2 `_run_rivgraph(...)`

File:

- [variant_generation.py](/Users/6256481/Code/river-hierarchy/network_variants/variant_generation.py:3401)

Current issue:

- RivGraph already computes and returns `distance` (`Idist`), but the manifest
  does not currently persist it under `output_paths`

Required changes:

- persist the returned RivGraph distance raster path into:
  - manifest `output_paths`
  - `NetworkVariantOutputs`

Why:

- the overhaul uses one global DT per state as a baseline and search-radius
  seed
- there is no reason to recompute it if RivGraph already produced it

## 3.3 `_apply_collapse_components_to_mask(...)`

File:

- [variant_generation.py](/Users/6256481/Code/river-hierarchy/network_variants/variant_generation.py:1032)

Current issue:

- it currently builds:
  - `collapsed_mask`
  - `width_scope_mask`
- and labels `component_filled` domains for the current width method

Required changes:

- keep producing:
  - `collapsed_mask`
- stop making `width_scope_mask` the core width artifact
- instead produce:
  - `origin_mask`
    - `0` = blocking land/background
    - `1` = original wet
    - `100+` = selected collapse-added pixels only
- keep `component_add` semantics explicit

Important:

- do **not** label the whole `component_filled` domain for the new method
- only the collapse-added pixels belong in the `100+` mask

## 3.4 `_build_selection_masks(...)`

File:

- [variant_generation.py](/Users/6256481/Code/river-hierarchy/network_variants/variant_generation.py:1185)

Current issue:

- currently acts as the mask-construction bridge for:
  - `collapsed_mask`
  - `width_scope_mask`

Required changes:

- return:
  - `collapsed_mask`
  - `origin_mask`
  - collapse-component summary
  - edit geometries
- stop treating `width_scope_mask` as a required output

This is the main state-mask handoff that the new width method will consume.

## 3.5 `compute_width_families(...)`

File:

- [variant_generation.py](/Users/6256481/Code/river-hierarchy/network_variants/variant_generation.py:768)

Current issue:

- this is the current transect + local corridor + `width_scope_mask` method
- it is the core function being replaced conceptually

Required changes:

- do **not** incrementally patch this function into the new method
- either:
  - replace its internals completely, or preferably
  - leave it as a legacy implementation and add a new orchestration wrapper that
    calls the new two-bank module

Recommended path:

- add new orchestrator in `variant_generation.py`, for example:
  - `compute_width_families_two_bank(...)`
- have `generate_network_variant(...)` switch to that new entry point

Eventually obsolete helpers in the current method:

- `_local_width_corridor(...)`
- `_build_width_scope_thread_caps(...)`
- `_select_segment_containing_point(...)`
- `_cap_wet_thread_geometry(...)`

Those may remain temporarily during migration, but they are not part of the
target method.

## 3.6 `generate_network_variant(...)`

File:

- [variant_generation.py](/Users/6256481/Code/river-hierarchy/network_variants/variant_generation.py:3457)

Current issue:

- writes `width_scope_mask.tif`
- calls current `compute_width_families(...)`

Required changes:

1. Build or load example-level boundary cache.
2. Build/write state `origin_mask.tif`.
3. Persist RivGraph `distance` raster path.
4. Call the new two-bank width routine.
5. Write new sample/link outputs.
6. Update manifest fields and `output_paths`.

New example-level cache input to pass in or resolve:

- experiment-level cache directory, e.g.
  - `.../<experiment_root>/width_method_cache/`

Because this cache is shared across many states, `generate_network_variant(...)`
should not try to invent a per-state-only cache location by itself.

## 3.7 `derive_anchored_variant_from_realized(...)`

File:

- [variant_generation.py](/Users/6256481/Code/river-hierarchy/network_variants/variant_generation.py:3916)

Current issue:

- recomputes width families using the current method

Required changes:

- mirror the same width-overhaul path used by `generate_network_variant(...)`
- ensure anchored derivation reuses:
  - experiment-level boundary cache
  - collapsed-state RivGraph distance raster
  - state `origin_mask`

Important:

- anchored derivation must not silently fall back to the old width method after
  the overhaul

## 3.8 `refresh_variant_width_scope_and_families(...)`

File:

- [variant_generation.py](/Users/6256481/Code/river-hierarchy/network_variants/variant_generation.py:4243)

Current issue:

- this refresh path is entirely named and structured around `width_scope_mask`

Required changes:

- replace it with a refresh path for the new method, for example:
  - `refresh_variant_width_method_outputs(...)`

This refresh must be able to rebuild, without rerunning RivGraph:

- `origin_mask.tif`
- bank-component sample diagnostics
- link width families
- merged width-family GPKG

Optional backward compatibility:

- keep a wrapper with the old function name for one transition period
- mark it legacy in comments/help text

## 3.9 `run_width_scope_patch.py`

File:

- [run_width_scope_patch.py](/Users/6256481/Code/river-hierarchy/network_variants/run_width_scope_patch.py:1)

Current issue:

- name and help text are specific to the old `width_scope_mask` method

Required changes:

- either rename to something like:
  - `run_width_method_patch.py`
- or keep the filename temporarily but change behavior and help text

CLI should rebuild:

- state `origin_mask`
- state width outputs under the new method

## 3.10 `collapse_experiment.py`

File:

- [collapse_experiment.py](/Users/6256481/Code/river-hierarchy/network_variants/collapse_experiment.py:183)

Current issue:

- the experiment runner currently does not know about an example-level width
  cache

Required changes:

- add explicit experiment-level boundary-cache location
- pass that cache location into:
  - `generate_network_variant(...)`
  - `derive_anchored_variant_from_realized(...)`

This is the natural place to create:

- `.../<experiment_root>/width_method_cache/`

Suggested contents:

- boundary-component raster or pixel table
- component metadata
- boundary spatial index cache if persisted

## 4. Manifest and Output Path Changes

### Fields to remove or de-emphasize

- `width_scope_mask`
- `width_scope_patch`

### Fields to add

- `origin_mask`
- `rivgraph_distance`
- `width_method_version`
- `width_method_cache_dir`
- optional:
  - `boundary_component_catalog`
  - `boundary_component_metadata`
  - `bank_pair_samples`

Files:

- [variant_generation.py](/Users/6256481/Code/river-hierarchy/network_variants/variant_generation.py:3790)
- [variant_generation.py](/Users/6256481/Code/river-hierarchy/network_variants/variant_generation.py:4166)
- [variant_generation.py](/Users/6256481/Code/river-hierarchy/network_variants/variant_generation.py:4357)

## 5. Suggested New Artifacts

### Example-level cache

At experiment root:

- `width_method_cache/boundary_component_labels.tif`
- `width_method_cache/boundary_component_metadata.csv`
- optional cached pixel coordinate arrays / index blobs

### State-level mask outputs

Under `variant/mask/`:

- `*_collapsed.tif`
- `origin_mask.tif`

### State-level width diagnostics

Under `variant/widths/`:

- `link_width_samples.csv`
  - now sample-level two-bank outputs
- `link_width_families.csv`
- `links_with_width_families.gpkg`
- optional:
  - `bank_pair_samples.gpkg`
  - or `bank_pair_samples.csv`

## 6. Equivalent Resistance and RAPID Touch Points

### `equivalent_resistance_experiment.py`

File:

- [equivalent_resistance_experiment.py](/Users/6256481/Code/river-hierarchy/network_variants/equivalent_resistance_experiment.py:249)

Current issue:

- it already reads `links_with_width_families.gpkg` first, which is good

Required changes:

- likely none in the core loader if width-family field names stay stable
- but confirm the new method continues to populate:
  - `wid_adj_total`
  - `wid_adj_wet`
  - `wid_adj_dry`

### RAPID prep

No direct algorithm change should be required if field names consumed downstream
remain stable.

But after implementation, verify:

- `--kb-width-sample-field width_wet`
- `--subreach-width-sample-field width_wet`

still map to the intended regenerated outputs.

## 7. Tests to Replace and Add

### Existing tests that become legacy

File:

- [test_variant_generation.py](/Users/6256481/Code/river-hierarchy/network_variants/tests/test_variant_generation.py)

Legacy tests tied to the old method:

- `test_compute_width_families_respects_width_scope_mask_domains`
- `test_compute_width_families_caps_simple_component_to_two_wet_threads`
- `test_compute_width_families_uses_local_corridor_for_base_scope_label`

These should eventually be removed or rewritten.

### New tests to add

#### A. Boundary preprocessing

- open-end boundary removal leaves only valid obstacle components
- island and outer-bank components receive stable unique IDs

#### B. Candidate selection

- local candidate lookup excludes far irrelevant components
- local normal correctly splits candidates into opposite sides

#### C. Two-bank selection

- simple channel selects one left and one right bank
- same-side nearest components are rejected
- bends still choose opposite-side components

#### D. Decomposition

- `width_total = d_left + d_right`
- `width_wet` counts only label `1`
- `width_dry` counts only selected `100+`
- `n_channels` counts contiguous `1` runs

#### E. Collapse-specific cases

- simple two-thread collapse returns `n_channels = 2`
- compound/multi-thread synthetic case returns `n_channels > 2`
- foreign `100+` labels are ignored

#### F. Refresh path

- rebuilds `origin_mask`
- rebuilds sample/link width outputs
- does not rerun RivGraph

## 8. Recommended Implementation Order

### Phase 1: Data plumbing

Change:

- `NetworkVariantOutputs`
- manifest `output_paths`
- `_run_rivgraph(...)` output persistence

Goal:

- make the RivGraph distance raster and new mask artifacts visible everywhere

### Phase 2: Example-level cache

Add:

- `width_two_bank.py`
- boundary extraction / open-end filtering / component labeling
- experiment-level cache creation from `collapse_experiment.py`

### Phase 3: State masks

Replace:

- `width_scope_mask` construction

With:

- `origin_mask`
- active obstacle component bookkeeping

### Phase 4: New sample-level method

Implement:

- local candidate lookup
- side-aware pair selection
- two-bank decomposition

### Phase 5: Link aggregation + outputs

Implement:

- new sample CSV schema
- new link family aggregation
- merge onto `links_with_width_families.gpkg`

### Phase 6: Refresh/patch path

Replace:

- `refresh_variant_width_scope_and_families(...)`
- `run_width_scope_patch.py`

With new-method refresh logic.

### Phase 7: Test migration

Remove or rewrite old `width_scope_mask` tests and replace with the new
boundary/two-bank/decomposition suite.

## 9. Key Migration Warning

Do not try to keep both methods half-active inside the same `compute_width_families(...)`
branching logic for long.

That would create:

- hard-to-read code
- ambiguous outputs
- hard-to-trust regression behavior

Prefer:

- one explicit legacy method
- one explicit new method
- a short migration period
- then remove the legacy path

## 10. Minimal First Milestone

The smallest credible repo milestone is:

1. example-level boundary cache created
2. state `origin_mask.tif` written
3. one global RivGraph DT path persisted in manifests
4. new two-bank sample-level widths computed for one state
5. link-level `wid_adj_total`, `wid_adj_wet`, `wid_adj_dry`, `n_channels_*`
   written to the standard width outputs
6. equivalent resistance can read the new width files without modification

That milestone should be enough to validate the method on `56 / unit 3` before
rolling it out across the full experiment pipeline.
