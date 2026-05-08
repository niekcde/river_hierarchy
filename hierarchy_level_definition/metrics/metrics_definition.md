# Metrics Definition

This document describes the current metric definitions implemented in
`hierarchy_level_definition/metrics/unit_metrics.py`.

## Scope

The current metrics are:

- direct unit metrics
- computed from the unit's detected paths in the current directed graph
- hierarchy-aware through metadata fields such as `root_unit_id`, `depth_from_root`, and `collapse_level`

The current metrics are not:

- recursively collapsed geometry metrics
- parent metrics rebuilt from already-collapsed child units

So a compound unit currently receives direct metrics from its own detected paths, not a recursively rebuilt equivalent geometry from child units.

## Input Data

The public entry points are:

- `compute_unit_metrics(links_path, nodes_path, ...)`
- `compute_unit_metrics_from_units(links_gdf, units, ...)`

The file-based entry point expects reviewed GeoPackages, not raw RivGraph direction outputs.

### Links input

Expected / useful fields:

- `id_link`
- `id_nodes` in `"u, v"` format
- `geometry`
- `len_adj`
- `len`
- `wid_adj`
- `wid`
- `wid_med`

The graph is a directed `MultiDiGraph`, so parallel edges between the same two nodes are preserved and treated as separate links and potential separate branches.

### Nodes input

Expected / useful fields:

- `id_node`
- `geometry`

`is_inlet` and `is_outlet` may exist, but the metrics layer does not directly require them.

## Unit Structure

The metrics layer sits on top of the structural unit detector. A unit is a bifurcation-confluence structure with one or more simple directed paths between:

- a boundary `bifurcation`
- a selected downstream `confluence`

The structural labels are:

- `simple_bifurcation_confluence_pair`
- `multi_thread_pair`
- `compound_or_nested_complex`

The metrics layer preserves these and adds a second derived class:

- `unit_topodynamic_class`

## Width Concepts

There are two width concepts in the outputs.

### 1. Representative width

This is the width used for equivalent geometry and path-importance metrics.

Per link:

`W_i^{rep}` = first valid value from:

1. `wid_adj`
2. `wid`
3. `wid_med`

Only finite values greater than zero are valid.

Representative widths are used for:

- `path_width_eq`
- `equivalent_width`
- `equivalent_length` weighting
- entropy metrics
- effective-path metrics
- dominance metrics

### 2. Pixel/sample width

This is used for local extremes and percentile diagnostics when available.

Possible field names are configurable with `pixel_width_fields`. Current defaults are:

- `width_samples`
- `width_px`
- `pixel_widths`
- `wid_pix`
- `wid_samples`

Accepted formats:

- Python list or tuple
- NumPy array
- stringified list such as `"[12.4, 13.1, 15.0]"`
- comma-separated string such as `"12.4,13.1,15.0"`

Only finite values greater than zero are retained.

Pixel/sample widths are used for:

- `path_width_min`
- `path_width_max`
- `path_width_p05`
- `path_width_p50`
- `path_width_p95`

If no pixel/sample widths are available for a path, these diagnostics fall back to representative widths from the links along that path.

## Path Metrics

Each row in `path_metrics.csv` is one path inside one unit.

### Identity fields

- `unit_id`
- `path_id`
- `n_links`
- `id_links`

### Path length

For a path `p` with links `i`:

`L_path = sum(L_i)`

Length precedence per link:

1. `len_adj`
2. `len`
3. `geometry.length`

Output field:

- `path_length`

### Equivalent path width

The path equivalent width uses representative link widths:

`W_path = sum(L_i) / sum(L_i / W_i^{rep})`

This is a length-weighted harmonic mean width.

Output field:

- `path_width_eq`

If any required representative width is missing, non-finite, zero, or negative, `path_width_eq` becomes `NaN`.

### Extreme and percentile width diagnostics

If pixel/sample widths are available for the path, pool all valid samples from all links in the path:

- `path_width_min = min(S_path)`
- `path_width_max = max(S_path)`
- `path_width_p05 = P5(S_path)`
- `path_width_p50 = P50(S_path)`
- `path_width_p95 = P95(S_path)`

If pixel/sample widths are not available, compute the same diagnostics from the valid representative link widths along the path.

Output fields:

- `path_width_min`
- `path_width_max`
- `path_width_p05`
- `path_width_p50`
- `path_width_p95`

Interpretation:

- `path_width_min` and `path_width_max` are raw extremes
- `path_width_p05` is the preferred robust narrow-width diagnostic
- `path_width_p50` is a typical local width
- `path_width_p95` is the preferred robust wide-width diagnostic

### Path fractions within a unit

For valid paths only:

- `path_width_fraction = W_path / sum(W_path over valid paths)`
- `path_length_fraction = L_path / sum(L_path over valid paths)`

Output fields:

- `path_width_fraction`
- `path_length_fraction`

### Path ranks within a unit

Ranks are computed within each unit:

- `path_rank_by_width`: descending by `path_width_eq`, rank `1` is widest
- `path_rank_by_length`: descending by `path_length`, rank `1` is longest

Output fields:

- `path_rank_by_width`
- `path_rank_by_length`

## Unit Metrics

Each row in `unit_metrics.csv` is one structural unit.

### Structural identity

- `unit_id`
- `bifurcation`
- `confluence`
- `class`
- `n_paths`
- `n_valid_paths`
- `unit_node_ids`
- `unit_node_count`

`n_valid_paths` counts paths with:

- finite `path_width_eq > 0`
- finite `path_length > 0`

### Equivalent unit geometry

Using valid paths only:

`W_eq = sum(W_p)`

`L_eq = sum(W_p * L_p) / sum(W_p)`

`elongation = L_eq / W_eq`

Output fields:

- `equivalent_width`
- `equivalent_length`
- `elongation`

Interpretation:

- `equivalent_width` is the effective parallel width of the unit, obtained by summing the representative importance of all valid paths.
- `equivalent_length` is a width-weighted characteristic length of the unit, so wider paths influence it more strongly than narrow paths.
- `elongation` is a compact shape-style metric:
  - values near `1` suggest unit length and effective width are of similar magnitude
  - larger values suggest long, slender units
  - values less than `1` suggest relatively short and wide units

Important note:

- these are direct metrics of the currently detected unit paths
- they are not recursive collapsed-geometry metrics of a parent rebuilt from child units

### Path-length spread metrics

Computed on valid path lengths:

- `path_length_min`
- `path_length_max`
- `path_length_mean`
- `path_length_range = path_length_max - path_length_min`
- `path_length_range_norm = path_length_range / equivalent_length`
- `path_length_cv = std(path_length) / mean(path_length)`

Interpretation:

- `path_length_min` and `path_length_max` give the shortest and longest valid pathways through the unit.
- `path_length_range` measures absolute branch-length contrast.
- `path_length_range_norm` rescales that contrast by the unit's characteristic length, which makes it easier to compare small and large units.
- `path_length_cv` is the coefficient of variation of path lengths:
  - near `0` means the path lengths are similar
  - larger values mean greater spread in branch lengths

What these metrics are good for:

- distinguishing balanced versus asymmetric pathway lengths
- identifying long detour paths relative to short direct paths
- quantifying geometric asymmetry even when width balance is similar

### Path-width spread metrics

Computed on valid `path_width_eq` values:

- `path_width_eq_min`
- `path_width_eq_max`
- `path_width_eq_mean`
- `path_width_range = path_width_eq_max - path_width_eq_min`
- `path_width_range_norm = path_width_range / equivalent_width`
- `largest_path_width_fraction = max(W_p) / equivalent_width`
- `dominant_width_fraction = largest_path_width_fraction`

Interpretation:

- these metrics use `path_width_eq`, so they describe differences in representative path importance, not raw local pixel extremes.
- `path_width_range` is the absolute difference between the widest and narrowest valid path.
- `path_width_range_norm` rescales that difference by the total effective unit width.
- `largest_path_width_fraction` is often the most intuitive dominance metric:
  - close to `0.5` in a two-path unit suggests a balanced split
  - close to `1.0` suggests one path dominates the unit

Use this section for:

- representative branch dominance
- balance of pathway importance
- comparing how uneven the split is across different units

Do not confuse these with:

- `path_width_min`, `path_width_max`, `path_width_p05`, `path_width_p95`

Those path-level metrics describe local narrow/wide conditions along a path, while the width-spread metrics here describe differences between whole-path representative widths.

### Entropy-based path diversity metrics

For valid paths, define:

`p_i = W_i / sum(W_i)`

Then:

- `width_entropy = -sum(p_i * ln(p_i))`
- `width_evenness = width_entropy / ln(n_valid_paths)` for `n_valid_paths > 1`
- `effective_n_paths_width = exp(width_entropy)`
- `path_disparity_width = n_valid_paths / effective_n_paths_width`

Convention used here:

- if `n_valid_paths == 1`, then
  - `width_entropy = 0`
  - `width_evenness = 1`
  - `effective_n_paths_width = 1`

Interpretation:

- `effective_n_paths_width` is the number of equally important paths that would produce the same entropy
- `path_disparity_width` near `1` means path importance is evenly distributed
- larger `path_disparity_width` means many mapped paths exist but only a few are effectively important

Additional intuition:

- `width_entropy` grows as path importance becomes more evenly distributed.
- `width_evenness` is the normalized version and is usually easier to compare across units with different numbers of paths.
- `effective_n_paths_width` is often more intuitive than raw entropy because it is on a path-count scale.

Examples:

- widths `[1, 1, 1, 1]`
  - `width_evenness = 1`
  - `effective_n_paths_width = 4`
- widths `[0.9, 0.1]`
  - lower entropy
  - `effective_n_paths_width` much closer to `1` than to `2`

So this section answers:

- how many paths are effectively important?
- are multiple paths genuinely sharing the unit, or is one path doing most of the work geometrically?

### Two-path metrics

Only defined when `n_valid_paths == 2`:

- `width_ratio_2 = min(W_1, W_2) / max(W_1, W_2)`
- `smaller_width_fraction_2 = min(W_1, W_2) / (W_1 + W_2)`
- `dominant_width_fraction_2 = max(W_1, W_2) / (W_1 + W_2)`
- `length_ratio_2 = min(L_1, L_2) / max(L_1, L_2)`

For all other units these are `NaN`.

### Topologic complexity metrics

These come from the detected structure of the unit:

- `internal_bifurcation_count`
- `internal_confluence_count`
- `total_bifurcation_count = internal_bifurcation_count + 1`
- `total_confluence_count = internal_confluence_count + 1`
- `internal_branch_node_count = internal_bifurcation_count + internal_confluence_count`
- `branching_density_by_length = internal_branch_node_count / equivalent_length`
- `path_redundancy = max(n_valid_paths - 1, 0)`
- `compound_indicator = 1 if n_children > 0 else 0`

Simple first-score:

`topologic_complexity_score = ln(1 + n_valid_paths) + ln(1 + internal_bifurcation_count) + ln(1 + internal_confluence_count)`

#### What counts as "internal" here

Each unit has:

- one boundary bifurcation
- one boundary confluence

The topologic complexity metrics ask:

- what additional branching or merging happens inside the unit between those boundaries?

So:

- `internal_bifurcation_count`
  counts bifurcation nodes inside the unit, excluding the boundary bifurcation
- `internal_confluence_count`
  counts confluence nodes inside the unit, excluding the boundary confluence

These are local unit-structure metrics, not whole-network metrics.

#### Meaning of each field

- `internal_bifurcation_count`
  Measures how many additional branch-splitting events occur inside the unit.
- `internal_confluence_count`
  Measures how many additional rejoining events occur inside the unit.
- `total_bifurcation_count`
  Adds the boundary bifurcation back in, so it describes the full number of bifurcation nodes associated with the unit structure.
- `total_confluence_count`
  Adds the boundary confluence back in for the same reason.
- `internal_branch_node_count`
  Counts all internal topologic decision points, regardless of whether they are splits or rejoins.
- `branching_density_by_length`
  Normalizes internal branching complexity by unit scale:
  more internal branch nodes per unit length means denser internal structure.
- `path_redundancy`
  Measures how many additional valid paths exist beyond a single-path baseline.
  It is intentionally simple:
  - `0` for one valid path
  - `1` for two valid paths
  - `2` for three valid paths
  and so on.
- `compound_indicator`
  Marks whether the unit has child units in the nesting tree.
  This is about hierarchy membership, not just internal branch-node counts.

#### How to read `topologic_complexity_score`

The score combines three things:

1. number of valid paths
2. number of internal bifurcations
3. number of internal confluences

The log transform does two useful things:

- keeps the score from exploding for large values
- makes the jump from `0` to `1` more meaningful than the jump from `5` to `6`

So it is a compact "how structurally busy is this unit?" score.

Examples:

- simple two-path unit with no internal branching:
  - `n_valid_paths = 2`
  - `internal_bifurcation_count = 0`
  - `internal_confluence_count = 0`
  - `topologic_complexity_score = ln(3)`

- nested three-path unit with one internal bifurcation and one internal confluence:
  - `topologic_complexity_score = ln(4) + ln(2) + ln(2)`

So a higher score can come from:

- more valid paths
- more internal splits
- more internal rejoins
- or a combination of all three

#### What these metrics do and do not capture

They do capture:

- whether the unit is topologically simple or internally branched
- whether internal complexity is sparse or dense relative to length
- whether the unit contains nested structure

They do not directly capture:

- exact spatial placement of the internal nodes
- width balance between paths
- local bottlenecks
- whether a unit belongs to a larger compound bubble

Those are handled by other fields:

- width and entropy metrics for pathway importance
- `compound_bubble_id` and related fields for broader multi-channel context

### Dynamic-proxy complexity metrics

These are width-based pathway-importance proxies, not hydrodynamic discharge metrics.

Define:

`q_i = W_i / sum(W_i)`

Currently this is the same weighting used for the entropy metrics above.

Output fields:

- `dynamic_proxy_method = "path_equivalent_width_fraction"`
- `dynamic_proxy_weight_field = "path_width_eq"`
- `dynamic_proxy_entropy = -sum(q_i * ln(q_i))`
- `effective_n_paths_dyn_width = exp(dynamic_proxy_entropy)`
- `dominant_dyn_fraction_width = max(q_i)`
- `dynamic_proxy_complexity_score = width_evenness`

Interpretation:

- these metrics deliberately reuse width fractions as a first-order proxy for pathway importance
- they are useful when discharge is unavailable but you still want a "more than one important path?" style diagnostic

In the current implementation:

- `dynamic_proxy_entropy` is numerically the same as `width_entropy`
- `effective_n_paths_dyn_width` is numerically the same as `effective_n_paths_width`
- `dominant_dyn_fraction_width` is numerically the same as `dominant_width_fraction`
- `dynamic_proxy_complexity_score` is numerically the same as `width_evenness`

So at the moment this block is conceptually important rather than mathematically independent.

Why keep it anyway:

- it marks the intended place where later hydrodynamic or flux-based metrics can plug in
- it cleanly separates "representative geometry metrics" from "pathway importance proxy metrics"

### Metric-derived classification

The metrics layer adds:

- `unit_topodynamic_class`

This is a rule-based first classification that combines:

- number of valid paths
- width balance / dominance
- internal complexity
- compound status

Current labels include:

- `invalid_or_no_valid_paths`
- `single_valid_path`
- `balanced_simple_split`
- `dominant_simple_split`
- `balanced_multi_thread_unit`
- `dominant_multi_thread_unit`
- `compound_topologically_complex_dynamically_complex`
- `compound_topologically_complex_dynamically_simple`
- `topologically_complex_dynamically_simple_unit`
- `topologically_complex_dynamically_complex_unit`
- `intermediate_unit`

In this classification, "dynamic" means width-based dynamic proxy, not measured discharge.

Interpretation:

- this field is best treated as a compact label for exploratory grouping
- it is not a replacement for the underlying metrics
- when the label looks surprising, inspect:
  - `n_valid_paths`
  - `largest_path_width_fraction`
  - `width_evenness`
  - `internal_bifurcation_count`
  - `internal_confluence_count`
  - `topologic_complexity_score`

### Hierarchy metadata

These fields describe position in the detected nesting structure:

- `primary_parent_id`
- `root_unit_id`
- `depth_from_root`
- `collapse_level`
- `n_children`
- `n_descendants`
- `is_compound`
- `compound_unit_id`
- `compound_bubble_id`
- `in_compound_bubble`
- `compound_bubble_role`

Important note:

- `collapse_level` is currently a hierarchy scale label
- it does not mean that recursive geometric collapse has been performed

How to read these together:

- `primary_parent_id`
  is the chosen parent in the unit nesting tree when multiple containment relationships are possible.
- `root_unit_id`
  is the top-most enclosing unit in that nesting tree.
- `depth_from_root`
  is top-down depth:
  - `0` means tree root
  - larger numbers mean more deeply nested units
- `collapse_level`
  is bottom-up scale:
  - `0` means no children
  - larger numbers mean the unit sits above smaller nested units
- `n_children`
  is the number of direct child units
- `n_descendants`
  is the total number of child, grandchild, and deeper nested units

This means `depth_from_root` and `collapse_level` are not the same:

- `depth_from_root` says how far down you are from the outer unit
- `collapse_level` says how much smaller structure exists beneath the unit

The current distinction is:

- `is_compound`
  means the unit is locally compound in the nesting tree, i.e. it has child units.
- `compound_unit_id`
  is the outer enclosing unit ID from the unit hierarchy. In the current implementation this is the top-most enclosing outer unit in that nested unit tree. Standalone units have `NA`.
- `compound_bubble_id`
  is a separate bubble ID, independent from `unit_id`. It is assigned to every unit and represents the maximal multi-channel bubble component based on overlapping unit footprints.
- `in_compound_bubble`
  is `True` when the bubble contains more than one unit, and `False` when the unit is the only member of its bubble.
- `compound_bubble_role`
  is one of:
  - `standalone`
  - `bubble_root`
  - `bubble_member`

This means a unit can be:

- locally simple (`is_compound = False`)
- but still part of a larger compound bubble (`in_compound_bubble = True`)

or:

- locally complex
- but still not have an enclosing outer unit (`compound_unit_id = NA`)

if it is a complex standalone unit rather than a nested child.

This is the intended distinction for cases where a local bifurcation-confluence pair closes, but the broader river context is still multi-channel.

Practical reading:

- if `class` is complex but `compound_unit_id` is `NA`, the unit is locally complex but not nested inside an enclosing outer unit.
- if `compound_bubble_id` matches across several units, those units belong to one broader multi-channel bubble even if some of them are locally simple.
- `unit_node_ids` is the quickest link back to the GeoPackage for visual QA in QGIS.

## Hierarchy-Level Summary Metrics

`hierarchy_level_metrics.csv` contains one row per `collapse_level`.

Output fields:

- `collapse_level`
- `n_units`
- `n_compound_units`
- `n_leaf_units`
- `mean_equivalent_width`
- `median_equivalent_width`
- `sum_equivalent_width`
- `mean_equivalent_length`
- `median_equivalent_length`
- `mean_elongation`
- `median_elongation`
- `mean_n_paths`
- `mean_n_valid_paths`
- `mean_effective_n_paths_width`
- `mean_width_evenness`
- `mean_largest_path_width_fraction`
- `mean_topologic_complexity_score`
- `mean_dynamic_proxy_complexity_score`
- `width_weighted_mean_elongation`
- `width_weighted_mean_effective_n_paths_width`
- `width_weighted_mean_width_evenness`
- `width_weighted_mean_topologic_complexity_score`
- `width_weighted_mean_dynamic_proxy_complexity_score`

Width-weighted means use `equivalent_width` as the weight.

## Important Metrics

If you want a reduced set for interpretation, these are the main ones to focus on.

### Important path metrics

- `path_length`
  Path length from summed link lengths.
- `path_width_eq`
  Main representative path width using harmonic averaging of representative link widths.
- `path_width_p05`
  Preferred robust bottleneck / narrow-width diagnostic.
- `path_width_p50`
  Typical local width along the path.
- `path_width_p95`
  Preferred robust wide-width diagnostic.
- `path_width_fraction`
  Relative importance of the path within the unit.
- `path_rank_by_width`
  Quick indicator of which path dominates.

### Important unit metrics

- `equivalent_width`
  Total effective width of the unit.
- `equivalent_length`
  Width-weighted characteristic length of the unit.
- `elongation`
  Dimensionless length-to-width style metric.
- `n_valid_paths`
  Number of valid geometric pathways.
- `largest_path_width_fraction`
  Share of total effective width carried by the dominant path.
- `width_evenness`
  How evenly width importance is distributed across paths.
- `effective_n_paths_width`
  Effective number of important paths.
- `path_disparity_width`
  Difference between mapped path count and effective path count.
- `smaller_width_fraction_2`
  Best simple width-balance metric for two-path units.
- `length_ratio_2`
  Best simple path-length asymmetry metric for two-path units.
- `topologic_complexity_score`
  Compact topologic complexity indicator.
- `unit_topodynamic_class`
  Practical summary class combining path balance and structural complexity.
- `compound_bubble_id`
  The key field for grouping local units into one larger multi-channel bubble. This ID is separate from `unit_id`.
- `compound_bubble_role`
  Quick indicator of whether the unit is the bubble root or just a member.
- `compound_unit_id`
  The outer enclosing unit ID when the current unit is nested inside a larger unit hierarchy.
- `unit_node_ids`
  Direct link back to node IDs in the GeoPackage for QGIS cross-checking.

## Recommended Usage

Use these conventions when interpreting widths:

- Use `path_width_eq` and `equivalent_width` for representative geometry and path-importance interpretation.
- Use `path_width_p05` as the preferred robust minimum-width diagnostic.
- Use `path_width_p50` as a typical local-width diagnostic.
- Use `path_width_p95` as the preferred robust maximum-width diagnostic.
- Use `path_width_min` and `path_width_max` as raw diagnostic extremes, not the preferred robust summary values.

For simple two-path units, the most interpretable balance metrics are:

- `smaller_width_fraction_2`
- `width_ratio_2`
- `length_ratio_2`

For larger or compound units, the most informative summary metrics are usually:

- `equivalent_width`
- `equivalent_length`
- `elongation`
- `n_valid_paths`
- `largest_path_width_fraction`
- `width_evenness`
- `effective_n_paths_width`
- `topologic_complexity_score`
- `unit_topodynamic_class`

## Recommended Score Set

If the goal is to compare units rather than individual paths, the current recommended unit-based score set is:

- `topologic_complexity_score`
- `effective_n_paths_width`
- `n_valid_paths`
- `equivalent_length`
- `elongation`

This set is aimed at unit comparison and collapse ranking, not at reproducing every path-level detail.

In this framing:

- `topologic_complexity_score` captures structural richness
- `effective_n_paths_width` captures effective redundancy / width-partition complexity
- `n_valid_paths` preserves literal path multiplicity
- `equivalent_length` is the preferred direct persistence metric
- `elongation` is the preferred normalized persistence metric

Optional substitutions:

- replace `effective_n_paths_width` with `width_evenness` if a bounded `0-1` balance metric is preferred
- replace `effective_n_paths_width` with `dominant_width_fraction` if a more direct dominance metric is preferred
- add `equivalent_width` back into the score set when absolute breadth or total parallel conveyance scale is scientifically important

### Proposed ranking schema

For practical use, it helps to separate:

- variables that should be used for all units
- variables that become especially important for compound units
- variables that are scientifically useful but not yet implemented as output columns

#### Base ranking variables for all units

The current base ranking set is:

- `topologic_complexity_score`
- `effective_n_paths_width`
- `n_valid_paths`
- `equivalent_length`
- `elongation`

These represent:

1. structural richness inside the unit
2. effective pathway redundancy
3. literal multiplicity of valid paths
4. persistence of the split in absolute length units
5. persistence of the split relative to unit width

#### Additional ranking variables for compound units

For compound units, the current output fields that add useful context are:

- `n_descendants`
- `collapse_level`
- `compound_unit_id`
- `compound_bubble_id`

These describe where the unit sits in the larger hierarchy:

- `n_descendants` indicates how much nested substructure lies inside the unit
- `collapse_level` indicates hierarchy scale
- `compound_unit_id` identifies the enclosing outer unit in the nesting tree
- `compound_bubble_id` identifies the broader maximal multi-channel zone

#### Descendant-aware extensions that are not yet implemented

One important limitation of the current topologic metrics is that they capture the presence of nested structure, but not the geometric size or importance of that nested structure.

For future compound-unit ranking, useful descendant-aware additions would be:

- `max_child_equivalent_length`
- `sum_child_equivalent_length`
- `max_child_effective_n_paths_width`
- `sum_child_equivalent_width`

These are not current output columns. They are proposed future additions for ranking and collapse studies where the geometric importance of nested subunits matters.

### Why this reduced set is recommended

#### 1. Keep the analysis unit-based, not path-based

If the scientific question is:

- how complex is this bifurcation-confluence unit?
- how balanced is flow-path importance inside it?
- how large and elongated is it?

then the path table is mainly diagnostic support.

The unit metrics already aggregate the path information into:

- effective width partitioning
- effective path counts
- structural complexity
- overall unit scale and shape

So the path metrics are useful for QA and interpretation, but they do not need to be part of the main score set.

#### 2. Skip the two-path-only metrics for the main score set

The two-path metrics:

- `width_ratio_2`
- `smaller_width_fraction_2`
- `dominant_width_fraction_2`
- `length_ratio_2`

are highly interpretable, but they only apply to units with exactly two valid paths.

If the goal is one score family that works across:

- simple two-path units
- multi-thread units
- nested compound units

then the multipath metrics are preferable because they generalize the same ideas to any number of valid paths.

This is why the recommended score set uses:

- `effective_n_paths_width`
- `n_valid_paths`

instead of the 2-path-specific metrics.

#### 3. Do not keep all entropy-family metrics at once

These metrics are tightly related:

- `width_entropy`
- `width_evenness`
- `effective_n_paths_width`
- `path_disparity_width`

They all derive from the same width-share distribution across valid paths.

So they should usually be treated as alternative expressions of the same underlying concept:

- how evenly the unit's effective width is distributed among its paths

Recommended choice:

- `effective_n_paths_width`

because it is on an interpretable path-count scale and is the closest analog to entropic braiding logic.

#### 4. Keep literal multiplicity alongside effective multiplicity

`effective_n_paths_width` is intentionally not the same as `n_valid_paths`.

Two units can have:

- different `n_valid_paths`
- similar `effective_n_paths_width`

This happens when one unit has more mapped paths, but those extra paths contribute little representative width.

So:

- `n_valid_paths` captures literal path multiplicity
- `effective_n_paths_width` captures effective path multiplicity

Both are useful, and they should not be treated as interchangeable.

#### 5. Prefer persistence-oriented geometry over breadth-oriented geometry for collapse ranking

These three are not independent:

- `equivalent_width`
- `equivalent_length`
- `elongation = equivalent_length / equivalent_width`

But for collapse ranking, the main scientific question is usually not pure width or size. It is:

- how long the flow remains partitioned
- how strongly that partition persists relative to unit width

That makes:

- `equivalent_length` the preferred direct persistence metric
- `elongation` the preferred normalized persistence metric

while:

- `equivalent_width` remains a useful secondary breadth / magnitude descriptor

This is also why `elongation` should not be interpreted as pure size. It is better understood as:

- a unit aspect ratio
- a normalized split-persistence ratio

rather than an area-like measure.

#### 6. Keep one topologic metric, but recognize its limit

`topologic_complexity_score` is recommended because it compresses:

- number of valid paths
- internal bifurcations
- internal confluences

into one structural complexity variable.

That makes it a good companion to:

- `effective_n_paths_width`

which captures how that structure is effectively partitioned geometrically.

However, `topologic_complexity_score` does not encode the size of nested children. It tells you that nested structure exists, but not whether that nested subunit is short and minor or long and dominant. That is why compound-unit ranking often needs both:

- the topologic score
- separate descendant-aware context such as `n_descendants`, `collapse_level`, and future child-size summaries

#### 7. Do not force one scalar ranking too early

The collapse problem mixes several distinct dimensions:

- structural complexity
- effective pathway redundancy
- literal path multiplicity
- split persistence
- hierarchy context

It is usually better to rank or cluster units using a small feature vector than to collapse everything immediately into one scalar.

That is especially true for compound units, where a single scalar can hide the difference between:

- a locally simple unit inside a large compound bubble
- a genuinely nested outer unit with important child substructure

## Defensibility and Literature Position

This score choice is defensible if it is described precisely.

### What is defensible to claim

It is defensible to say that:

- the current unit metrics are inspired by entropy-based channel-network complexity ideas associated with Alejandro Tejedor and related river/delta network work
- `effective_n_paths_width` is the closest unit-scale analog in this framework to an entropic braiding-style metric
- the current implementation adapts that logic from cross-section/channel-count style formulations to bifurcation-confluence units and representative path widths

### What should not be claimed

It would not be accurate to say that:

- this framework is a direct implementation of Tejedor's eBI
- the current unit metrics are numerically equivalent to cross-section-based eBI values

The current implementation is:

- unit-based, not cross-section-based
- path-based, not direct channel-intercept based
- driven by representative path widths, not observed discharge partitioning

So the correct wording is:

- analogous to
- inspired by
- aligned with

rather than:

- identical to
- reproduces exactly

### Why the Tejedor connection helps

Alejandro Tejedor is a well-established researcher in entropy-based and graph-theoretic analysis of river and delta channel networks. That makes the use of entropy/effective-path style metrics scientifically legible and defensible, especially when your framework is presented as an adaptation rather than a copy.

The logic that works in your favor is:

1. the literature already supports entropy-based metrics as meaningful measures of network/channel complexity
2. your framework preserves that conceptual backbone
3. your adaptation is targeted to bifurcation-confluence units rather than cross-sections or entire delta graphs

So the defensibility comes from:

- conceptual continuity with published entropy-based river-network metrics
- clear documentation of what is preserved
- clear documentation of what is changed

### Practical wording recommendation

In methods text, the safest wording is something like:

"We characterize unit-scale pathway complexity using an entropy-based effective-path metric derived from representative path-width fractions. This metric is conceptually aligned with entropic braiding formulations, in the sense that it measures the effective number of important pathways, but it is adapted here to bifurcation-confluence units rather than cross-sections."

That wording is strong, accurate, and easy to defend.
