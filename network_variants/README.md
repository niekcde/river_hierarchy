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
  - adaptive sequential unit collapse
  - adaptive sequential group collapse

Not yet in scope:

- RAPID handoff packaging

## Width Logic

For each regenerated link, width families are measured from the same transects
on the regenerated link geometry.

- `total width`
  Measured on the collapsed mask.
- `wet width`
  Measured on the wet-reference mask, which defaults to the parent cleaned
  mask.
- `dry width`
  `dry width = total width - wet width`

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
  carry propagated SWORD columns.
- `--sword-wse-field`
  Optional WSE field name in the supplied SWORD node source.
- `--sword-match-tolerance`
  Optional maximum distance for direct SWORD node matching.

Additional node-matching outputs:

- `matching/node_sword_match.csv`
- SWORD columns written onto `directed/*_directed_nodes.gpkg`

## Collapse Experiments

The experiment runner chains the hierarchy workflow and the variant-generation
workflow together so each collapse state is re-evaluated on its own directed
graph before the next decision is made.

Available modes:

- `independent-units`
  Collapse each base-state unit independently from the same base network.
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

Example: independent base-unit variants

```bash
/opt/anaconda3/envs/river-hierarchy-rivgraph/bin/python \
  network_variants/run_collapse_experiment.py \
  independent-units \
  --cleaned-mask rivgraph_centerline/outputs/smoke_tests/sarl_03/masks_cleaned/sarl_03_cleaned.tif \
  --reviewed-links hierarchy_level_definition/manual_review/outputs/sarl_03/directed_links.gpkg \
  --reviewed-nodes hierarchy_level_definition/manual_review/outputs/sarl_03/reviewed_nodes.gpkg \
  --exit-sides NS \
  --experiment-id sarl_03_independent_units
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
  --experiment-id sarl_03_sequential_units
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
  --experiment-id sarl_03_sequential_groups
```

The experiment runner writes:

- `state_registry.csv`
- `transition_registry.csv`
- `experiment_manifest.json`

and creates one `states/<state_id>/` folder per hierarchy state. Each state
contains:

- `hierarchy/`
  fresh unit metrics, collapse ranking, and selected groups for that state
- `variant/`
  only for derived states; collapsed mask, regenerated RivGraph outputs,
  directed graph, matching, and width families

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
