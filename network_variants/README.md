# Network Variants

This module generates collapsed network variants from the hierarchy-unit
workflow outputs.

Current V1 scope:

- select target `unit_id`s or one `group_label`
- split the selection into connected collapse components
- build provisional local collapse footprints from reviewed link geometries
- fill enclosed dry holes within those footprints on the parent cleaned mask
- rerun RivGraph on the collapsed mask
- match regenerated nodes/links back to the reviewed parent graph
- assign regenerated-link flow direction from parent matching plus topology rules
- optionally match regenerated nodes to external SWORD nodes and propagate those
  node matches through sequential states
- compute regenerated-link width families:
  - `total` width on the collapsed mask
  - `wet` width on the reference wet mask
  - `dry` width as `total - wet`
- orchestrate full collapse experiments from one base state using:
  - independent base-unit variants
  - independent compound-bubble variants
  - adaptive sequential unit collapse
  - adaptive sequential group collapse
- write state-level hydraulic-context artifacts after each hierarchy state so
  slope and profile-WSE derivation live in `network_variants` rather than in
  RAPID prep
- optionally compute geometry-informed equivalent-resistance predictors for
  detected bubbles within each state hierarchy
- optionally run one experiment-level equivalent-resistance comparison workflow
  for `independent-units` and `compound-bubbles`

Still intentionally separate from this module:

- running RAPID prep and RAPID routing themselves

`network_variants` now writes state outputs in the layout consumed by the RAPID
subproject, but the actual prep/run steps remain separate entry points under
`RAPID/`.

## Collapse Corridor Modes

Collapsed variants can now be materialized in two scientifically distinct ways:

- `realized`
  Keep the full realized collapsed corridor produced by the regenerated child
  network, even if the effective child object expands beyond the nominal unit
  or bubble boundaries.
- `anchored`
  Split the realized collapsed corridor back at the nominal selection anchor
  nodes before writing the final directed child graph. This preserves a more
  local "replace only the selected interval" interpretation.

These modes are selected with:

- `--collapse-corridor-mode realized`
- `--collapse-corridor-mode anchored`
- `--collapse-corridor-mode both`

`both` runs two sibling experiments and writes separate output roots suffixed
with `__realized` and `__anchored`.

Interpretation:

- `realized` is the better description of what the regenerated child network
  actually became after collapse.
- `anchored` is the better description when you want to hold the replacement to
  the nominal upstream/downstream selection anchors and avoid changing a longer
  surrounding corridor more than necessary.

## Width Logic

The current implemented width method is `two_bank_dt`.

For each regenerated link, width families are measured from the regenerated
link geometry using one centerline-sample-based left/right bank search rather
than a legacy explicit transect-intersection workflow.

- `total width`
  Measured on the collapsed mask.
- `wet width`
  Measured by decomposing the selected bank-pair section against the state
  `origin_mask`.
- `dry width`
  `dry width = total width - wet width`

State mask semantics:

- `collapsed mask`
  Binary traversable mask for the regenerated state.
- `origin_mask.tif`
  - `0` = blocking land/background
  - `1` = original wet water
  - `101+` = collapse-added fill for the selected state object
- `bank_labels.tif`
  Example-level boundary-component labels rewritten for the current state so
  collapsed target-island labels are treated as water during bank
  identification.

Sample-level bank selection currently does all of the following:

- builds one global DT on the collapsed mask
- finds nearby boundary candidates from `bank_labels.tif`
- smooths the local tangent/normal used for left/right side classification
- adaptively expands the number of queried candidate boundary pixels up to
  `bank_candidate_neighbors_max`
- rejects candidate bank points whose center-to-bank segment re-intersects the
  centerline
- requires the final left/right pair to come from distinct boundary-component
  IDs

This method writes both raw sample diagnostics and final link summaries.

The module stores:

- sample-level arrays:
  - `wid_pix_total`
  - `wid_pix_wet`
  - `wid_pix_dry`
- aggregated link-level values:
  - `wid_adj_total`, `wid_med_total`
  - `wid_adj_wet`, `wid_med_wet`
  - `wid_adj_dry`, `wid_med_dry`
  - `p05/p50/p95` for all three width families
  - `wet_fraction_adj`, `dry_fraction_adj`
  - `n_wet_threads_mean`, `n_wet_threads_max`

The raw sample table also stores:

- center sample coordinates
- chosen left/right component IDs
- chosen left/right bank coordinates
- `width_total`, `width_wet`, `width_dry`
- `dt_width_global`
- `qc_relative_dt_diff`
- `qc_opposition_cos`
- `qc_width_chord_diff`
- `width_invalid`

Important distinction:

- `link_width_samples.csv`
  Raw per-sample measurement and QC record. This is the debug surface and is
  not rewritten when final family inheritance is applied.
- `link_width_families.csv`
  Final authoritative per-link width summary.
- `links_with_width_families.gpkg`
  Same final width summary attached to geometry.

If a link is an anchored segmented stub with poor sample-level opposition QC
and inflated width relative to a clean family representative, the final width
summary can inherit from that representative. This inheritance is applied only
to the final link-family outputs, not to the raw sample table.

Anchored derivation reuses complete width products for links whose geometry is
unchanged from the realized state. A newly numbered anchored stub may instead
reuse one unique exact geometry from the reviewed base state. Only unmatched
new or shortened segments are recalculated with link-local raster windows.
The variant manifest records this classification under `anchored_width_patch`
and its internal timings under `anchored_stage_timings`. Each completed child
state also writes `stage_timings.json` for variant generation, hierarchy, and
hydraulic-context timing. The downstream hierarchy, slope, hydraulic, and
equivalent-resistance workflows still run on the complete anchored graph.

Full-raster child generation uses the same inheritance contract before width
measurement. After RivGraph generation, direction assignment, and lineage
resolution, an `unchanged_1to1` link is accepted for inheritance only when its
geometry is topologically equal to one unique parent link. Its complete parent
width fields and sample rows are copied. Only links that fail that strict test
are passed to the raster-backed two-bank width calculation. The calculation
currently retains the full-raster context, so this changes execution cost but
not the bank-search or width semantics. The manifest records
`measurement_scope=changed_links_only_full_raster_context` together with the
measured and skipped link IDs and width-stage timings under
`width_product_inheritance`.

These final family-summary width fields now also matter downstream in RAPID
prep and equivalent-resistance workflows. The current default reference-width
choice for derived `kb` calculations is `wid_adj_routing`, and the reference-
section helper prefers the summary width already written to
`link_width_families.csv` over recomputing a percentile from raw sample widths
when that field is available. The percentile setting defaults to null and must
be supplied explicitly only for a field that exists solely in raw samples;
there is no implicit P90 fallback.

Audit columns for this logic include:

- `family_width_inheritance_candidate`
- `family_width_inheritance_applied`
- `family_representative_link_id`
- `family_width_inherited_from_link`
- `family_width_inheritance_reason`
- `family_stub_bad_opposition_fraction`
- `family_representative_bad_opposition_fraction`
- `family_stub_to_representative_wet_ratio`
- `family_stub_to_representative_total_ratio`

The `links_with_width_families.gpkg` output is the directed version of the
regenerated link layer. It now includes:

- `id_us_node`
- `id_ds_node`
- `matched_parent_link_ids`
- `primary_parent_link_ids`
- `touch_parent_link_ids`
- `dominant_parent_link_id`
- `lineage_type`
- `matched_parent_node_path`
- `direction_assignment_method`
- `geometry_reversed_to_match_flow`
- final width-family attributes
- width-family inheritance audit fields when inheritance is applied

The directed node layer can now also carry standardized SWORD node attributes:

- `sword_node_id`
- `sword_reach_id`
- `sword_region`
- `sword_dist_out`
- `sword_wse`
- `sword_wse_field`
- `sword_match_distance`
- `sword_match_method`
- `sword_match_from_parent`
- `sword_match_within_tolerance`
- `sword_source_file`

Important:

- `link_match.csv` is the raw candidate table
- `link_lineage.csv` is the resolved lineage table

The resolved lineage logic now works in two layers:

1. `link_match.csv`
   Raw geometric candidates with overlap scores and `core_overlap` /
   `touch_only` classification.
2. `link_lineage.csv`
   Resolved lineage from the matched upstream/downstream parent nodes.
   The workflow takes the directed parent-node path between those matched
   endpoints and keeps all parent links on that path, including parallel links
   between consecutive parent nodes.

The raw overlap scores are still retained, but they are supporting evidence
rather than the primary lineage selector. The trimming step still matters
because it stops simple endpoint touches from being mistaken for real interior
lineage. In practice this means:

- true interior overlap is kept as lineage
- endpoint-touch-only neighbors are retained only as `touch_parent_link_ids`
- parallel parent links on the resolved parent path are retained in
  `matched_parent_link_ids`, even if only one of them has the largest direct
  overlap score

After direction assignment, the regenerated link geometry is reversed when
needed so the first coordinate is always the upstream endpoint.

## Hydraulic Context

After each state graph is materialized and the hierarchy workflow is rerun,
`network_variants` now writes one state-level hydraulic-context bundle under:

- `states/<state_id>/hydraulic_context/`

Artifacts:

- `link_hydraulic_context.csv`
- `node_hydraulic_context.csv`
- `hydraulic_context_manifest.json`

This step owns the slope/profile derivation that used to live only in RAPID
prep. RAPID prep now prefers these precomputed artifacts and only falls back to
its internal slope routine when the state-level hydraulic context is missing.

Target ownership:

- all slope/profile construction for `network_variants` experiments should be
  completed here during variant creation
- RAPID prep should only consume `hydraulic_context/`
- the RAPID-side slope path is transitional compatibility for older experiment
  directories, not the intended normal workflow

Planned implementation checklist:

1. Build one smooth example-level `H_ref(s)` from trusted SWORD anchors on the
   cached `reference_section/`.
2. Save that fitted profile and its anchor metadata once at the experiment
   root.
3. Write full base-state hydraulic context in `states/S000_base/`.
4. For each derived state, inherit unchanged node/profile values from the
   parent through `matched_parent_node_id` and lineage where possible.
5. Recompute only the affected corridor and intersecting compound bubbles for
   each variant.
6. Enforce exact compound-bubble boundary drop here, before RAPID prep.
7. Write final `rapid_wse_for_slope`, `slope_used`, and diagnostics per state.
8. Leave RAPID prep as a consumer of cached hydraulic context only.

Three head concepts are intentionally kept separate:

- observed or matched node WSE:
  `sword_wse` / raw node WSE context
- derived profile WSE for slope:
  `rapid_wse_for_slope`
- normalized solver heads for equivalent-resistance solves:
  not stored in `hydraulic_context/`; written separately by the solver workflow

Derived states now inherit the final parent slope for direction-consistent
`unchanged_1to1` links with at least 99.9% dominant-parent overlap. Raw child
profile slopes remain available for provenance. Parent inheritance is recorded
with:

- `slope_parent_inheritance_applied`
- `slope_inherited_from_parent_link`
- `slope_parent_value`
- `slope_preinheritance_value`
- `slope_parent_inheritance_reason`

Collapsed local links are excluded from direct inheritance. If a valid child
direction maps to reversed parent endpoints and creates a nonpositive head
drop, the changed links and their one-hop context are solved together using
reliable parent-continuation slopes as targets.

Ordinary collapsed components use a path-relative head optimizer. Every
directed boundary-to-boundary path contributes its own target slope from the
base-derived boundary-head drop divided by the child path length. Link terms
are weighted by their share of path length, so a shared link participates in
every path that contains it rather than being forced toward one component-wide
mean slope. Boundary heads remain hard and every optimized link must satisfy
the configured positive minimum slope. The link context records target paths,
target slopes, optimized endpoint heads, and objective values before and after
the solve.

Base-component path-drop consistency is diagnostic and does not determine the
node-head solution. It is evaluated only between base nodes that match the
child component boundaries. For a directed acyclic component, path count and
the minimum, maximum, mean, and spread of path drops are calculated exactly by
DAG dynamic programming. Individual base paths are never materialized: their
number can grow exponentially in a nested compound bubble. This bounded
aggregation leaves the least-squares base heads, transferred child boundary
heads, and path-relative child slope optimizer unchanged. The manifest records
`summary_method=dag_dynamic_programming` with the aggregated diagnostics.

Compound-bubble slope QC is also bounded for derived states. A capped DAG path
count is evaluated before any explicit path sequence is created; bubbles above
`compound_bubble_max_paths_for_qc` therefore receive the same
`path_limit_exceeded` result without first searching for 501 complete paths.
When a parent hydraulic context is available, bubbles that do not touch a
changed link or changed endpoint are not recalculated. Their QC fields and
final slopes are inherited through the same strict, direction-consistent
`unchanged_1to1` contract. A bubble that touches changed topology is still
evaluated in full; unchanged links retain their parent final values afterward,
matching the pre-optimization result contract. The per-state manifest records
this under `compound_bubble_qc_scope`, and link-level provenance uses
`bubble_slope_qc_recalculated` and `bubble_slope_qc_inherited`.

The hydraulic context can also inherit slope from a family representative when
width inheritance was applied to an anchored segmented stub. This is recorded
with:

- `family_slope_inheritance_applied`
- `family_slope_inherited_from_link`
- `family_slope_inheritance_reason`

For independent-unit and compound-bubble runs,
`--defer-child-hierarchy` skips the complete hierarchy rebuild for derived
states. Variant and hydraulic artifacts are completed immediately; hierarchy,
equivalent-transition comparison, and other hierarchy-dependent exports are
marked deferred for later finalization. Sequential modes do not support this
option because their next selection depends on the preceding child hierarchy.

For downstream use:

- final width should come from:
  - `link_width_families.csv`
  - or `links_with_width_families.gpkg`
- final slope should come from:
  - `hydraulic_context/link_hydraulic_context.csv`
- raw sample QC and debugging should use:
  - `link_width_samples.csv`

## First-Step Collapse Rule

V1 does not draw arbitrary edit polygons manually.

Instead, for each connected collapse component it:

1. buffers the reviewed link geometries using representative width
2. unions those buffers into a local component footprint
3. rasterizes that footprint on the parent cleaned mask
4. fills enclosed dry holes within the footprint
5. writes the added dry polygons as collapse edits

So the first auto-collapse mode is:

- `fill enclosed dry area inside the selected local footprint`

This is intentionally conservative.

## Inputs

The main runner consumes:

- parent cleaned mask
- reviewed/directed links GeoPackage
- reviewed/directed nodes GeoPackage
- RivGraph `exit_sides`
- optional SWORD node source:
  - one geospatial file such as `gpkg` / `geojson`
  - one GeoParquet file
  - or a directory of SWORD parquet node tiles
- either:
  - explicit `unit_id`s
  - or one `group_label` from `selected_groups.csv` / `ordered_group_partitions.csv`

## Command Line

Example for `sarl_03`, using one selected group from the hierarchy workflow:

```bash
/opt/anaconda3/envs/river-hierarchy-rivgraph/bin/python \
  network_variants/run_variant_workflow.py \
  --workflow-output-dir hierarchy_level_definition/outputs/directed \
  --cleaned-mask rivgraph_centerline/outputs/smoke_tests/sarl_03/masks_cleaned/sarl_03_cleaned.tif \
  --reviewed-links hierarchy_level_definition/manual_review/outputs/sarl_03/directed_links.gpkg \
  --reviewed-nodes hierarchy_level_definition/manual_review/outputs/sarl_03/reviewed_nodes.gpkg \
  --exit-sides NS \
  --group-label G3_1 \
  --variant-id sarl_03__group_G3_1
```

Example with explicit unit IDs:

```bash
/opt/anaconda3/envs/river-hierarchy-rivgraph/bin/python \
  network_variants/run_variant_workflow.py \
  --cleaned-mask rivgraph_centerline/outputs/smoke_tests/sarl_03/masks_cleaned/sarl_03_cleaned.tif \
  --reviewed-links hierarchy_level_definition/manual_review/outputs/sarl_03/directed_links.gpkg \
  --reviewed-nodes hierarchy_level_definition/manual_review/outputs/sarl_03/reviewed_nodes.gpkg \
  --exit-sides NS \
  --unit-ids 17 18 \
  --variant-id sarl_03__units_17_18
```

Optional:

- `--example-id`
  Override the example identifier used in regenerated filenames. This is useful
  when chaining multiple collapse states manually, because it keeps file names
  short and stable instead of repeatedly embedding prior collapsed mask names.
- `--match-tolerance`
  Override the spatial tolerance used for parent-child node/link matching.
  By default the workflow uses `1.25` raster pixels from the cleaned mask
  transform.
- `--sword-node-source`
  Optional external SWORD node source used for node matching. When omitted,
  node-to-SWORD matching is skipped unless the parent directed nodes already
  carry propagated SWORD columns. When this points at a SWORD parquet
  directory, the workflow will also try to constrain candidate nodes to the
  current example's buffered SWORD reach corridor before running nearest-node
  matching.
- `--sword-wse-field`
  Optional WSE field name in the supplied SWORD node source. If the requested
  field is null for a matched node, the workflow first tries the nearest node
  on the same SWORD reach using that same requested field when `node_order` is
  available. If that still fails and the source also has `wse`, it falls back
  to same-node `wse` and records the fallback in the node output.
- `--sword-match-tolerance`
  Optional maximum distance for direct SWORD node matching.
- `--sword-example-stations-source`
  Optional example-station source used to derive example-specific SWORD reach
  corridors. Defaults to the repo
  `SWORD_gauge_match/outputs/hierarchy_examples_filtered_subdaily_manual_updates_final.gpkg`
  when present.
- `--sword-station-match-source`
  Optional station-to-SWORD match source used to derive example-specific SWORD
  reach corridors. Defaults to the repo
  `SWORD_gauge_match/outputs/selected_event_stations_same_main_path.gpkg`
  when present.
- `--sword-reach-buffer-steps`
  Number of reach steps to extend upstream and downstream beyond the matched
  example corridor endpoints. Default: `2`.

Additional node-matching outputs:

- `matching/node_sword_match.csv`
- SWORD columns written onto `directed/*_directed_nodes.gpkg`
- the variant manifest now records whether node matching used:
  - the full bbox candidate set
  - or an example-specific buffered SWORD reach corridor
- node outputs also carry:
  - `sword_wse`
  - `sword_wse_field`
  - `sword_wse_fallback_used`
  - `sword_wse_fill_method`
  - `sword_wse_source_node_id`

## Collapse Experiments

The experiment runner chains the hierarchy workflow and the variant-generation
workflow together so each collapse state is re-evaluated on its own directed
graph before the next decision is made.

Available modes:

- `independent-units`
  Collapse each base-state unit independently from the same base network.
- `compound-bubbles`
  Collapse each real multi-unit `compound_bubble_id` independently from the
  same base network. These states are written as a separate scenario family
  and are explicitly marked as compound-bubble states in the registry.
- `sequential-units`
  Collapse the currently highest-ranked unit, regenerate the network, rerun
  hierarchy, then repeat.
- `sequential-groups`
  Collapse the current first selected group, regenerate the network, rerun
  hierarchy, then repeat.

Important:

- sequential modes always rerun unit detection and collapse ranking on the
  regenerated directed graph
- group labels such as `G3_1` are local to the current state, not globally
  stable across all variants
- when SWORD node matching is enabled, propagated SWORD attributes are carried
  forward through the directed node layers of successive states
- every state now also writes `hydraulic_context/` after hierarchy outputs are
  available
- collapse experiments can be run in `realized`, `anchored`, or `both`
  corridor modes using `--collapse-corridor-mode`
- use `--base-only` to materialize just `states/S000_base/` when you want to
  inspect or manually review the directed base network before paying for the
  full family of derived states

### Routing Interpretation

When these collapse experiments are handed off to `RAPID/`, the routing stage
should be interpreted as an example-level comparative response model across
variants.

In practice that means:

- all states from the same example should use the same forcing series
- all states from the same example should use the same RAPID prep settings
- routing parameters should not be retuned per state just to make one variant
  pass numerically

The intent is to preserve a shared routing experiment while topology changes.
Under this interpretation, routed outputs are most defensible as:

- relative comparisons of peak timing, attenuation, and hydrograph shape
  between variants

rather than:

- fully calibrated absolute hydraulic truth for each regenerated edge

This matters because the regenerated graphs can contain short links and large
local width/slope contrasts. A simple edge-based reach-routing model can
become sensitive to those contrasts even after numerical regularization. That
sensitivity is treated here as part of the modeling limitation of the routing
layer, not as a reason to retune each state independently.

Example: independent base-unit variants

```bash
/opt/anaconda3/envs/river-hierarchy-rivgraph/bin/python \
  network_variants/run_collapse_experiment.py \
  independent-units \
  --cleaned-mask rivgraph_centerline/outputs/smoke_tests/sarl_03/masks_cleaned/sarl_03_cleaned.tif \
  --reviewed-links hierarchy_level_definition/manual_review/outputs/sarl_03/directed_links.gpkg \
  --reviewed-nodes hierarchy_level_definition/manual_review/outputs/sarl_03/reviewed_nodes.gpkg \
  --exit-sides NS \
  --experiment-id sarl_03_independent_units \
  --collapse-corridor-mode realized
```

Example: materialize only the base directed state for review

```bash
/opt/anaconda3/envs/river-hierarchy-rivgraph/bin/python \
  network_variants/run_collapse_experiment.py \
  independent-units \
  --cleaned-mask rivgraph_centerline/outputs/runs/02/masks_cleaned/02_cleaned.tif \
  --reviewed-links rivgraph_centerline/outputs/runs/02/rivgraph/02_links.gpkg \
  --reviewed-nodes rivgraph_centerline/outputs/runs/02/rivgraph/02_nodes.gpkg \
  --exit-sides SN \
  --experiment-id 02_indep__anchored \
  --output-dir network_variants/output/02_indep__anchored \
  --collapse-corridor-mode anchored \
  --base-only
```

This writes `states/S000_base/` including the base directed links/nodes and
hydraulic context, then stops before generating any unit or group child states.

### Bounded local-patch child execution

Completed anchored base states can generate independent-unit or
compound-bubble children with the bounded local raster workflow:

```bash
/opt/anaconda3/envs/river-hierarchy-rivgraph/bin/python -B \
  network_variants/run_collapse_experiment.py \
  independent-units \
  --cleaned-mask rivgraph_centerline/outputs/runs/56/masks_cleaned/56_cleaned.tif \
  --reviewed-links rivgraph_centerline/outputs/runs/56/rivgraph/56_links.gpkg \
  --reviewed-nodes rivgraph_centerline/outputs/runs/56/rivgraph/56_nodes.gpkg \
  --exit-sides WN \
  --experiment-id 56_indep_local_patch \
  --output-dir network_variants/output/56_indep_local_patch \
  --collapse-corridor-mode anchored \
  --execution-mode local_patch
```

This mode is deliberately separate from `full_raster` and `panelized`. It
recalculates RivGraph and raster-derived widths only inside a validated crop,
merges that result into the complete parent graph, and then uses the normal
hierarchy/hydraulic pipeline. Strict unchanged one-to-one links inherit their
parent products. In the standalone `local_patch` mode, the published collapsed
mask remains a genuine full-extent mask, written blockwise from the parent plus
the local collapse edit.

Safety is per child. If topology planning, crop validation, or conservative
interface merging is unsafe or ambiguous, that child automatically uses the
normal full-raster generator. The audit is written under
`states/<state>/variant/local_patch_attempt/fallback_to_full_raster.json` and
the fallback is recorded in `variant/summary/variant_manifest.json`.

At present this execution mode supports only `independent-units` and
`compound-bubbles`, and requires `--collapse-corridor-mode anchored`. Unlike
the historical anchored CLI behavior, it does not first generate all realized
children; only the requested anchored experiment is written (with the normal
`__anchored` output suffix).

### Panelized base with local-patch children

Large examples can combine the two execution modes in one command:

```bash
--execution-mode panelized local_patch
```

This combination has fixed semantics:

- `S000_base` is built with the established panelized workflow;
- every derived child first uses the validated local-patch workflow;
- an unsafe local child falls back to the panelized child workflow, never to a
  whole-raster child run;
- child collapsed, origin, and bank-label rasters are exposed as full-extent
  GDAL VRTs backed by the unchanged parent plus a small aligned patch.

The VRTs preserve the normal raster contract for downstream Rasterio/GDAL
readers without copying the unchanged full raster into every child directory.
When a collapsed channel absorbs a bank component, the bank-label VRT applies
that label remapping over the component's complete global extent. The child
manifest records `execution_backends`, `mask_publication`, the patch paths, and
any panelized fallback audit.

The hybrid mode remains anchored-only and supports `independent-units` and
`compound-bubbles`. A compound experiment can independently build its own
panelized base, or reuse the completed independent base with
`--base-state-source`; base-state compatibility and manifest path rewriting are
unchanged.

When `--equivalent-kb-mode based_reference_section` is enabled, a panelized
base also constructs the all-units reference section through the panelized
collapse and boundary-merge workflow. The reference section is one special
`realized` selection containing every detected base unit; it does not use the
local-patch backend because its footprint spans the network. The resulting
width, slope, forcing, and BASED calculations are unchanged. Until both
reference-section summaries have been written, the experiment manifest records
`stop_reason: reference_kb_prepass_pending` and does not constitute a reusable
hydraulic base.

### Reusing a completed base state

Independent-unit and compound-bubble experiments can either build their base
independently or start from a completed base
experiment with `--base-state-source`. Reuse is a complete base-state contract,
not just a copy of the directed vectors. When available, the new experiment
copies and reuses:

- the `S000_base` variant, hierarchy tables, and hydraulic context
- the experiment-level boundary-component artifacts
- the reference-section products and equivalent-resistance solve cache

The input mask, reviewed links, reviewed nodes, example ID, exit sides,
hierarchy settings, width settings, and equivalent-resistance configuration
recorded by the source manifest must match the requested run. A missing or
partial source is rejected before the destination experiment is reset; the
workflow never mixes copied and recalculated base products.

Copying is transactional. Internal paths in the copied variant, hierarchy,
hydraulic-context, boundary-component, reference-section, and solver-cache
JSON manifests are rewritten from the source experiment root to the
destination experiment root. External provenance paths (the original mask,
reviewed vectors, permanent width-review archive, and model bundle) remain
external references. The destination writes
`states/S000_base/base_state_reuse_manifest.json`, while the experiment-level
summary is recorded under `base_state_reuse` in `experiment_manifest.json`.

For example, a compound run can reuse an already completed independent base:

```bash
--base-state-source network_variants/output/15_indep__anchored
```

Omitting this option keeps the fully independent behavior and recalculates the
compound experiment's base. Supplying it avoids repeating width measurement,
hierarchy detection, SWORD matching, slopes, hydraulic context, reference
geometry, and the base resistance solve. Child states are still generated and
solved normally.

### Recommended Parent Review Workflow

For the current RivGraph-to-variant workflow, the reviewed parent RivGraph
network should be treated as the only manual source of truth for directions and
core widths.

Recommended order:

1. Run `rivgraph_centerline/run_workflow.py apply-edits`.
2. Run `rivgraph_centerline/run_workflow.py run-rivgraph`.
3. Inspect parent directions in the RivGraph run outputs.
4. If needed, apply direction fixes directly to the parent files with
   `hierarchy_level_definition/manual_review/apply_link_reversals.py`.
5. Materialize `S000_base` with `--base-only` when you want to inspect the
   directed base state before running the full variant family.
6. Run `network_variants/width_review_gui.py` on the base-state variant output
   to review short suspicious links in directed local context.
7. The width review GUI writes accepted core-width corrections back to the
   parent RivGraph links file, not to `S000_base` itself.
8. Run the full collapse experiment only after the parent directions and parent
   core widths are in the desired state.

Important interpretation:

- directions are manually corrected only in the parent RivGraph network
- `S000_base` is used as a review surface, not as a separate manual direction
  source of truth
- the width review GUI currently updates parent `wid`, `wid_adj`, and
  `wid_med` in the RivGraph links GeoPackage
- later per-state fields such as `wid_adj_wet` are recomputed inside the
  variant workflow, but they inherit from the corrected parent network and
  geometry context

This avoids maintaining separate manual edits in both the parent network and a
derived base-state product.

After model validation, review the union of held-out manual/model disagreements
and high-priority near-junction audit cases without changing the permanent
reviews or any width files:

```bash
MPLCONFIGDIR=/tmp/matplotlib-width-qc-disagreements \
/opt/anaconda3/envs/river-hierarchy-rivgraph/bin/python -m \
  network_variants.width_qc_disagreement_gui \
  --reviewer "<reviewer name>"
```

The adjudication is resumable and is saved separately under
`network_variants/output/width_qc_disagreement_review/` with hashes of the
training dataset, held-out predictions, and selected near-junction audit. By
default the GUI displays the exploratory 10%/250–1,000 m junction window;
generate it first with `python -m network_variants.width_junction_audit`.

After completing that review, compare hybrid whole-link/near-junction
continuity rules without changing labels, models, or network widths:

```bash
MPLCONFIGDIR=/tmp/matplotlib-width-junction-hybrid \
/opt/anaconda3/envs/river-hierarchy-rivgraph/bin/python -m \
  network_variants.width_junction_hybrid_audit
```

This tests endpoint substitution independently for every adjacent incoming
and outgoing reach at length/wet-width thresholds 3, 5, and 10, both with and
without a non-overlapping-window requirement. Results and exact input hashes
are written to `network_variants/output/width_qc_junction_hybrid_audit/`.

Fit the terminal adjudicated model after that audit with:

```bash
MPLCONFIGDIR=/tmp/matplotlib-width-qc-final \
/opt/anaconda3/envs/river-hierarchy-rivgraph/bin/python -m \
  network_variants.width_qc_final_model
```

This freezes the conservative `L/W >= 10` plus non-overlapping endpoint rule,
uses the completed adjudications once, excludes ambiguous cases, and writes
old/new held-out probability and decision comparisons under
`network_variants/output/width_qc_final_model/`. It does not change source
network widths. Numeric correction targets remain the responsibility of the
joint upstream-continuity solver so sibling outgoing corrections are not
calculated independently.

Evaluate that standalone joint solver with:

```bash
MPLCONFIGDIR=/tmp/matplotlib-width-routing-solver \
/opt/anaconda3/envs/river-hierarchy-rivgraph/bin/python -m \
  network_variants.width_routing_solver
```

The solver preserves measured `wid_adj_wet`, writes
`wid_adj_routing`, allocates multiple predicted-bad outgoing links jointly,
propagates correction scales downstream, and retains unsafe cases raw with an
explicit reason. Its outputs under
`network_variants/output/width_routing_solver/` remain evaluation artifacts.
The selected logistic model and adaptive 2:1 two-ended solver are now wired
into regular and panelized base materialization. Durable manual-review
archives contribute labels and provenance only; historical numeric review
targets are not reapplied. Each base writes a separate effective
`wid_adj_routing` plus a per-link audit and manifest under `widths/`.
The solver additionally writes a single combined
`width_routing_solver_rejected_corrections_v1.csv` for inspecting every
predicted correction that was retained raw, including its rejected candidate
width and complete junction accounting.

### Future Width-Review Automation

The current width workflow is intentionally manual:

- build a candidate set from short unit-involved links
- inspect the local directed context in the GUI
- label each candidate as `good` or `bad`
- write accepted parent-width corrections back to the RivGraph source links

If this review is repeated across all example networks, the saved review tables
can be combined into a labeled training dataset. A simple classifier could then
be trained to approximate the manual decision boundary.

Reasonable first models:

- logistic regression
- linear support vector machine

Reasonable first features:

- `qc_mean_opposition_cos_trimmed`
- `node_dev_max`
- `len_over_width`
- `us_dev`
- `ds_dev`
- `n_units`

The intended use of such a model would be triage rather than full automation:

- auto-pass obvious good links
- auto-flag obvious bad links
- keep borderline cases for manual GUI review

That future classifier is not yet part of the production workflow, but the
current review CSV outputs are designed to make that next step feasible.

Example: independent compound-bubble variants

```bash
/opt/anaconda3/envs/river-hierarchy-rivgraph/bin/python \
  network_variants/run_collapse_experiment.py \
  compound-bubbles \
  --cleaned-mask rivgraph_centerline/outputs/smoke_tests/sarl_03/masks_cleaned/sarl_03_cleaned.tif \
  --reviewed-links hierarchy_level_definition/manual_review/outputs/sarl_03/directed_links.gpkg \
  --reviewed-nodes hierarchy_level_definition/manual_review/outputs/sarl_03/reviewed_nodes.gpkg \
  --exit-sides NS \
  --experiment-id sarl_03_compound_bubbles \
  --collapse-corridor-mode realized
```

Example: adaptive sequential unit collapse

```bash
/opt/anaconda3/envs/river-hierarchy-rivgraph/bin/python \
  network_variants/run_collapse_experiment.py \
  sequential-units \
  --cleaned-mask rivgraph_centerline/outputs/smoke_tests/sarl_03/masks_cleaned/sarl_03_cleaned.tif \
  --reviewed-links hierarchy_level_definition/manual_review/outputs/sarl_03/directed_links.gpkg \
  --reviewed-nodes hierarchy_level_definition/manual_review/outputs/sarl_03/reviewed_nodes.gpkg \
  --exit-sides NS \
  --experiment-id sarl_03_sequential_units \
  --collapse-corridor-mode realized
```

Example: adaptive sequential group collapse

```bash
/opt/anaconda3/envs/river-hierarchy-rivgraph/bin/python \
  network_variants/run_collapse_experiment.py \
  sequential-groups \
  --cleaned-mask rivgraph_centerline/outputs/smoke_tests/sarl_03/masks_cleaned/sarl_03_cleaned.tif \
  --reviewed-links hierarchy_level_definition/manual_review/outputs/sarl_03/directed_links.gpkg \
  --reviewed-nodes hierarchy_level_definition/manual_review/outputs/sarl_03/reviewed_nodes.gpkg \
  --exit-sides NS \
  --experiment-id sarl_03_sequential_groups \
  --collapse-corridor-mode realized
```

Example: run both realized and anchored experiment families at once

```bash
/opt/anaconda3/envs/river-hierarchy-rivgraph/bin/python \
  network_variants/run_collapse_experiment.py \
  independent-units \
  --cleaned-mask rivgraph_centerline/outputs/smoke_tests/sarl_03/masks_cleaned/sarl_03_cleaned.tif \
  --reviewed-links hierarchy_level_definition/manual_review/outputs/sarl_03/directed_links.gpkg \
  --reviewed-nodes hierarchy_level_definition/manual_review/outputs/sarl_03/reviewed_nodes.gpkg \
  --exit-sides NS \
  --experiment-id sarl_03_independent_units \
  --output-dir network_variants/output/sarl_03_independent_units \
  --collapse-corridor-mode both
```

This writes:

- `network_variants/output/sarl_03_independent_units__realized/`
- `network_variants/output/sarl_03_independent_units__anchored/`

If the realized experiment already exists, build only its anchored sibling by
supplying the completed realized output root:

```bash
/opt/anaconda3/envs/river-hierarchy-rivgraph/bin/python -B \
  network_variants/run_collapse_experiment.py \
  compound-bubbles \
  --cleaned-mask rivgraph_centerline/outputs/runs/15/masks_cleaned/15_cleaned.tif \
  --reviewed-links rivgraph_centerline/outputs/runs/15/rivgraph/15_links.gpkg \
  --reviewed-nodes rivgraph_centerline/outputs/runs/15/rivgraph/15_nodes.gpkg \
  --exit-sides NS \
  --experiment-id 15_compound_bubbles \
  --output-dir network_variants/output/15_compound_bubbles \
  --collapse-corridor-mode anchored \
  --anchored-source-experiment network_variants/output/15_compound_bubbles__realized
```

This validates and reuses the realized experiment's `S000_base` and
`boundary_components`, then derives the anchored child states without rerunning
the realized experiment. Other scientific and hydraulic CLI settings should be
supplied as usual for the anchored run.

The experiment runner writes:

- `state_registry.csv`
- `transition_registry.csv`
- `equivalent_transition_registry.csv`
- `experiment_manifest.json`

and creates one `states/<state_id>/` folder per hierarchy state. Each state
contains:

- `hierarchy/`
  fresh unit metrics, collapse ranking, selected groups, and optional
  bubble-level equivalent-resistance outputs for that state
- `variant/`
  only for derived states; collapsed mask, regenerated RivGraph outputs,
  directed graph, matching, and width families
- `hydraulic_context/`
  state-level slope/profile artifacts derived from the directed graph plus node
  WSE context
- `equivalent_resistance/`
  state-to-parent equivalent-resistance outputs written by the collapse
  experiment workflow. The primary
  `bubble_equivalent_state_comparison.csv` evaluates the selected collapse in
  its smallest hydraulically closed domain: the selected unit itself when it
  is closed, otherwise its complete containing compound between stable outer
  anchors. `selected_hydraulic_domain_links.csv` records every parent and child
  link used by that solve. The previous independently re-detected bubble
  matches remain available as the diagnostic `redetected_bubble_matches.csv`.

Per-state hierarchy bubble solves derive their boundary nodes from the complete
assembled directed edge set, not from the nominal root unit alone. The solver
always fixes normalized source/sink heads to `1.0` and `0.0`. It first
initializes internal heads from available node WSE and retries failed solves
with every internal head initialized to `0.5`. Both attempts are recorded in
`bubble_equivalent_metrics.csv`; if neither converges, provisional equivalent
values and solved flows/heads are written as missing.

### Equivalent-resistance nonlinear solver

This section records both the hydraulic problem and the numerical algorithm so
that solver changes can be audited without reconstructing the reasoning from
code.

#### Hydraulic problem

For every two-terminal hydraulic domain, source and sink heads are normalized
to

```text
H_source = 1
H_sink   = 0
```

The internal node heads are unknown. For edge `e=(u,v)`, the signed unit-head
flow is

```text
delta_H_e = H_u - H_v
Q_e = sign(delta_H_e) * sqrt(abs(delta_H_e) / K_e)
```

The implementation uses a differentiable approximation near zero head
difference:

```text
Q_e = delta_H_e /
      (sqrt(K_e) * (delta_H_e^2 + epsilon^2)^(1/4))
```

At every internal node, signed incident flows must sum to zero. With internal
incidence matrix `B_i`, the continuity residual is

```text
r(H) = B_i Q(H)
```

When this residual is sufficiently small, upstream unit-head inflow `Q_unit`
defines

```text
K_eq = 1 / Q_unit^2
G_eq = 1 / K_eq
```

These are normalized equivalent-resistance heads and flows. They are not
absolute WSE values or routed RAPID flows.

#### Legacy nonlinear least squares

The original algorithm minimizes

```text
0.5 * ||r(H)||_2^2
```

It repeatedly linearizes the continuity residual around the current internal
heads and chooses a bounded trust-region head update. `dense_legacy` obtains
that update with dense SVD. `sparse_least_squares` uses the analytic sparse
Jacobian

```text
J = B_i diag(dQ/d(delta_H)) B_i^T
```

and solves the trust-region linear system with LSMR. `sparse` remains a CLI
alias for `sparse_least_squares` so older commands remain valid.

Direct least squares can be poorly conditioned in a large graph when many
initial edge head differences are close to zero. The square-root flow law then
has a very large local derivative. Squaring the continuity residual further
amplifies the conditioning problem, which can lead to many small steps and the
maximum function-evaluation limit.

#### Energy-sparse algorithm

`energy_sparse` first minimizes the convex hydraulic energy whose gradient is
exactly the continuity residual. For the smoothed flow law, edge energy is

```text
E_e(delta_H) = 2 / (3 * sqrt(K_e)) *
               ((delta_H^2 + epsilon^2)^(3/4) - epsilon^(3/2))
```

and total energy is the sum over all edges:

```text
E(H) = sum_e E_e(delta_H_e)
gradient(E) = r(H)
```

The algorithm is:

1. Fix source and sink heads to `1` and `0`.
2. Initialize internal heads from normalized available node WSE. Nodes without
   a usable WSE value use resistance-weighted graph distances to source and
   sink; the final fallback is `0.5`.
3. Minimize `E(H)` within `[0,1]` using bounded L-BFGS-B and its analytic
   gradient. L-BFGS-B stores only limited curvature history and therefore does
   not factor a dense Hessian.
4. Start sparse LSMR least squares from the energy solution. This is a
   continuity-polishing step, rather than asking least squares to discover the
   solution from a difficult initial state.
5. Compare the energy and polish residuals and retain the better solution.
6. Accept a solution only when its physical absolute or relative continuity
   residual passes the configured tolerance and upstream unit-head inflow is
   finite and positive. An optimizer success message alone is not sufficient.
7. If the complete solve fails, the existing outer fallback retries with all
   internal heads initialized to `0.5`. Failed values are written as missing;
   failed solves are never cached.

The energy formulation changes the numerical route, not the hydraulic model.
For a connected two-terminal domain with fixed boundary heads, it has the same
continuity solution as the original equations.

#### Available backends

- `dense_legacy`: original dense-SVD nonlinear least squares;
- `sparse` or `sparse_least_squares`: direct sparse LSMR nonlinear least
  squares, retained for regression and diagnosis;
- `energy_sparse`: convex L-BFGS-B energy solve followed by sparse LSMR
  polishing;
- `auto`: `dense_legacy` below the internal-node threshold and
  `energy_sparse` at or above it.

Select the backend with `--equivalent-solver-backend`. The `auto` threshold
defaults to 64 internal nodes and can be changed with
`--equivalent-sparse-min-internal-nodes`. `dense_legacy` remains the default
until the completed-run regression audit is accepted; production test runs
should currently request `--equivalent-solver-backend energy_sparse`
explicitly.

`bubble_equivalent_metrics.csv` records the requested and resolved backend,
algorithm version, selected solution stage, total evaluations, energy-stage
iterations/evaluations/residual, polish evaluations/residual, final residual,
optimizer messages, and cache provenance. Selected parent-child domain
comparisons retain the corresponding backend, selected stage, and stage
residual fields for both states.

#### Example 15 validation case

The direct sparse least-squares solver reached 4,000 evaluations for base
bubbles 3 and 4 and failed with relative residuals `0.0888` and `0.1116`.
Using `energy_sparse` on the production inputs produced:

| Bubble | Nodes | Edges | Energy residual | Final residual | K_eq |
|---|---:|---:|---:|---:|---:|
| 3 | 124 | 186 | `1.26e-5` | `3.08e-7` | `7.95024e-5` |
| 4 | 247 | 372 | `2.66e-6` | `3.88e-7` | `3.89581e-5` |

All 11 base bubbles then passed the continuity acceptance criterion.

#### Solve reuse and cache safety

Completed successful solves are stored content-addressably in
`equivalent_resistance/solve_cache`. The cache signature includes the complete
directed topology, source/sink boundaries, every edge resistance, normalized
head initialization, tolerances, smoothing settings, and backend. Exact
`unchanged_1to1` lineage aliases allow an unchanged child bubble to reuse the
base result even when child node or link IDs were renumbered; cached heads and
flows are remapped onto the child IDs before writing. Changed, split, merged,
or newly created links automatically miss the cache and are solved normally.
`solver_cache_hit`, `solver_cache_key`, and `solver_cache_path` provide an
audit trail in the bubble and selected-domain comparison outputs.

The cache schema and algorithm version are part of the signature. Introducing
the energy solver therefore invalidates incompatible older cache entries. A
cache hit is permitted only for an identical hydraulic problem and solver
configuration. A changed compound is solved completely, including its
unchanged one-to-one links, because its full-domain signature changes.

### Geometry-Informed Equivalent Resistance

Two related workflows now exist:

1. State-level bubble reduction inside the hierarchy workflow
   Enabled with `--compute-equivalent-resistance` on
   `run_collapse_experiment.py` or `hierarchy_level_definition/run_unit_workflow.py`.
   This writes, per state:
   - `bubble_equivalent_metrics.csv`
   - `bubble_edge_resistance.csv`
   - `bubble_node_heads.csv`

2. Experiment-level matched comparison workflow
   Run separately with:
   - `network_variants/run_equivalent_resistance_experiment.py`

   To regenerate equivalent resistance after a solver or hydraulic-domain
   implementation change, without rerunning masks, RivGraph, widths, slopes,
   or the other hierarchy metrics, use:

   ```bash
   /opt/anaconda3/envs/river-hierarchy-rivgraph/bin/python \
     network_variants/run_equivalent_resistance_experiment.py \
     network_variants/output/<experiment> \
     --recompute-state-hierarchy \
     --reuse-manifest-config \
     --equivalent-solver-backend energy_sparse
   ```

   This restores the resistance configuration from the collapse experiment
   manifest and, in this example, explicitly overrides only its solver backend.
   It atomically replaces each completed state's
   `bubble_equivalent_metrics.csv`, `bubble_edge_resistance.csv`, and
   `bubble_node_heads.csv`, updates the state registry counts, and then
   refreshes experiment-level comparisons. An old-versus-new audit is written
   to `equivalent_resistance/state_hierarchy_recompute_audit.csv`.

   The forcing normalization options used by this runner mirror RAPID prep and
   now accept either second-based resampling
   (`--equivalent-forcing-resample-seconds`) or minute-based values that
   resolve to whole seconds.

   This is currently intended for:
   - `independent-units`
   - `compound-bubbles`

   It writes one experiment-level folder:
   - `equivalent_resistance/`

   with:
   - `base_nominal_targets.csv`
   - `child_realized_objects.csv`
   - `matched_base_corridors.csv`
   - `equivalent_resistance_comparison.csv`
   - `equivalent_resistance_manifest.json`

The comparison workflow is important because the realized child corridor does
not always share the same outer boundaries as the nominal base unit or bubble.
The main physically consistent comparison field is therefore usually:

- `Lambda_matched = K_variant_realized / K_eq_matched_base`

rather than the nominal-boundary comparison.

## Output Layout

Default output root:

- `network_variants/outputs/<example_id>/<variant_id>/`

Inside it:

- `summary/`
  - `collapse_components.csv`
  - `variant_manifest.json`
- `matching/`
  - `node_match.csv`
  - `link_match.csv`
  - `link_lineage.csv`
- `mask/`
  - collapsed mask GeoTIFF
  - `collapse_edit_geometries.gpkg`
- `directed/`
  - directed regenerated links
  - directed regenerated nodes
  - `direction_validation_report.json`
- `rivgraph/`
  - regenerated links/nodes GeoPackages
  - regenerated centerline
  - skeleton geotiff
  - optional SWORD-style reaches/nodes
- `widths/`
  - `link_width_families.csv`
  - `link_width_samples.csv`
  - `links_with_width_families.gpkg`

Inside a collapse experiment root:

- `state_registry.csv`
- `transition_registry.csv`
- `equivalent_transition_registry.csv`
- `experiment_manifest.json`
- `states/<state_id>/variant/`
- `states/<state_id>/hierarchy/`
- `states/<state_id>/hydraulic_context/`
- optionally `states/<state_id>/equivalent_resistance/`
- optionally `<experiment>/equivalent_resistance/` when the standalone
  experiment-level comparison runner is used
