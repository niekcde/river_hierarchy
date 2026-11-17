# Predictor similarity screen — 2 November 2026

Reference: SYN-PREDICTOR-SIMILARITY-2026-11-02.

## Scope and method

All 134 empirical affine classes are included (161 complete nonconstant representations). No earlier shortlist, centroid/peak responses, feature importance or predictive score is used. The 40 partial candidates and four constants remain outside this correlation screen. Source geometry is counted once across hydraulic approaches.

Class-to-class similarity is the minimum over ALL constituent variable pairs, so no arbitrary class representative determines the result. The rank rule requires matching sets of treatments with variation, consistent Spearman sign, and the threshold in the pooled population AND every informative treatment. Complete linkage requires every pair inside a proposed cluster to satisfy the rule; chains of weakly related endpoints cannot merge. If both variables vary only between treatments, the evidence is pooled only and cannot establish within-treatment redundancy.

A second, stricter screen additionally requires absolute Pearson correlation above the same threshold, pooled and within informative treatments. It measures linear as well as rank similarity. Thresholds are sensitivity choices, not significance tests. These clusters indicate empirical similarity, not universal equivalence or demonstrated lack of incremental prediction value. No candidate is deleted.

## Counts

| Rule | Threshold | Groups | Potential reduction from 134 | Singleton groups |
|---|---:|---:|---:|---:|
| rank | 0.85 | 69 | 65 | 38 |
| rank | 0.90 | 79 | 55 | 51 |
| rank | 0.95 | 98 | 36 | 77 |
| rank | 0.99 | 114 | 20 | 99 |
| rank_and_linear | 0.85 | 75 | 59 | 47 |
| rank_and_linear | 0.90 | 91 | 43 | 69 |
| rank_and_linear | 0.95 | 106 | 28 | 89 |
| rank_and_linear | 0.99 | 123 | 11 | 113 |

## Detailed conservative starting point: rank and linear similarity ≥ 0.95

Every member is listed; choosing a retained representative remains a separate formula/coverage decision. Classes containing several names are existing affine alternatives, not newly selected variables.

### Similarity group 4: 2 affine classes

- A123: `path_parent_effective_n_paths_width`
- A133: `path_parent_width_entropy`

### Similarity group 6: 2 affine classes

- A049: `B_parent_dominance_mean`
- A051: `B_parent_effective_branches_mean`

### Similarity group 11: 2 affine classes

- A131: `path_parent_slope_mean`
- A132: `path_parent_slope_min`

### Similarity group 12: 2 affine classes

- A101: `path_child_slope_mean`
- A102: `path_child_slope_min`

### Similarity group 16: 2 affine classes

- A046: `B_parent_branch_length_contrast_mean`
- A058: `B_parent_route_length_sd`

### Similarity group 18: 2 affine classes

- A127: `path_parent_path_length_cv`
- A128: `path_parent_path_length_range_norm`

### Similarity group 25: 2 affine classes

- A002: `B_child_area_proxy`
- A043: `B_parent_area_proxy`

### Similarity group 26: 2 affine classes

- A094: `path_child_equivalent_width`
- A126: `path_parent_equivalent_width`

### Similarity group 36: 4 affine classes

- A001: `B_child_area_elongation`
- A042: `B_parent_area_elongation`
- A092: `path_child_elongation`
- A124: `path_parent_elongation`

### Similarity group 39: 4 affine classes

- A056: `B_parent_route_length_max`
- A057: `B_parent_route_length_mean`
- A066: `B_parent_total_length`
- A125: `path_parent_equivalent_length`

### Similarity group 40: 3 affine classes

- A008: `B_child_route_length_max`
- A009: `B_child_route_length_mean`
- A093: `path_child_equivalent_length`

### Similarity group 61: 2 affine classes

- A030: `B_delta_route_length_max`
- A032: `B_delta_route_length_sd`; `route_spread_change_m`

### Similarity group 66: 3 affine classes

- A033: `B_delta_sinuosity_proxy`
- A073: `B_ratio_area_elongation`; `B_ratio_area_proxy`; `B_ratio_route_length_mean`; `B_ratio_sinuosity_proxy`
- A087: `log1p__B_ratio_route_length_mean`

### Similarity group 81: 3 affine classes

- A076: `B_ratio_route_entropy`
- A078: `B_ratio_split_impact_sum`
- A086: `log1p__B_ratio_route_entropy`

### Similarity group 82: 5 affine classes

- A006: `B_child_route_entropy`
- A013: `B_child_split_impact_sum`
- A090: `path_child_dominant_width_fraction`
- A091: `path_child_effective_n_paths_width`
- A104: `path_child_width_entropy`

### Similarity group 85: 3 affine classes

- A096: `path_child_path_length_cv`
- A097: `path_child_path_length_range_norm`
- A099: `path_child_slope_cv`

### Similarity group 91: 2 affine classes

- A050: `B_parent_dominance_sd`
- A052: `B_parent_effective_branches_sd`

## Interpretation and next decisions

Important: parent and child area, equivalent width and elongation can be strongly correlated because between-case scale dominates their differences. This does not make the parent-child difference uninformative. Preserve explicit change candidates and do not interpret a similarity cluster as permission to discard a state without checking its comparison role. Likewise, high correlation between slope variability and route variability follows partly from conserved head loss. These are conditional redundancy findings, not universal identities.

Start review with the joint 0.95 clusters, compare the rank-only clusters and 0.90 sensitivity, and retain singletons. Choose canonical SI representations only after formula review. Correlated but physically different descriptors stay as documented alternatives until real-network checks or grouped predictive comparisons distinguish them. Topology-rich real networks can break equalities observed on this grid. This screen does not establish whether any candidate predicts the response.

The complete cluster membership for all thresholds, class memberships, pairwise scores and pooled-only matches are exported beside this report.
