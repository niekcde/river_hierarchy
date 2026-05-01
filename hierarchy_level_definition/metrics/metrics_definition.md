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

### Path-length spread metrics

Computed on valid path lengths:

- `path_length_min`
- `path_length_max`
- `path_length_mean`
- `path_length_range = path_length_max - path_length_min`
- `path_length_range_norm = path_length_range / equivalent_length`
- `path_length_cv = std(path_length) / mean(path_length)`

### Path-width spread metrics

Computed on valid `path_width_eq` values:

- `path_width_eq_min`
- `path_width_eq_max`
- `path_width_eq_mean`
- `path_width_range = path_width_eq_max - path_width_eq_min`
- `path_width_range_norm = path_width_range / equivalent_width`
- `largest_path_width_fraction = max(W_p) / equivalent_width`
- `dominant_width_fraction = largest_path_width_fraction`

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

Important note:

- `collapse_level` is currently a hierarchy scale label
- it does not mean that recursive geometric collapse has been performed

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
