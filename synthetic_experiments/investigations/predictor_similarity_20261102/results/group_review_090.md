# Review of the 22 predictor-similarity groups

Reference: SYN-SIMILARITY-GROUP-REVIEW-2026-11-02.

Population: 134 empirical affine classes; joint absolute Spearman/Pearson threshold 0.90, pooled and within informative treatments under the consistency rules in the screen. There are 69 singleton classes and 22 multi-class clusters containing 65 classes. One representative per cluster gives 91 slots, but formula review can justify more. Cluster IDs below are the actual IDs in cluster_membership.csv, not consecutive row numbers.

These are recommendations for discussion, not applied deletions or a newly selected model. They use definitions and predictor similarity only; no response performance is used. Canonical lengths are in metres. Existing affine aliases are displayed in full so that old empirical equivalences are not mistaken for verified identities.

## Equation key

- $\mu_L=\sum_r\pi_rL_r$; $\sigma_L^2=\sum_r\pi_r(L_r-\mu_L)^2$; allocated $CV_L=\sigma_L/\mu_L$.
- $A=\sum_e L_eB_e$; $\bar B=\sum_eL_eB_e/\sum_eL_e$; $\bar S=\sum_eL_eS_e/\sum_eL_e$. Reach CVs use the same length weights.
- $a_{vj}=B_{vj}/\sum_kB_{vk}$; $p_v$ is visit probability; $D_v=\max_j a_{vj}$; $N_v=1/\sum_ja_{vj}^2$; $h_v=-\sum_ja_{vj}\ln a_{vj}/\ln k_v$. Local means and SDs are weighted by $p_v$.
- $H=-\sum_r\pi_r\ln\pi_r$; $I=\sum_{v\in\mathrm{bif}}p_v(1-\sum_ja_{vj}^2)$.
- $W_r=L_r/\sum_{e\in r}(L_e/B_e)$; $q_r=W_r/\sum_sW_s$; $W_{eq}=\sum_rW_r$; $L_{eq}=\sum_rq_rL_r$; $H_q=-\sum_rq_r\ln q_r$; effective width-path count $=\exp(H_q)$. These weights differ from routed allocation $\pi_r$.
- Unweighted $CV_0=\sqrt{N^{-1}\sum_r(L_r-\bar L_0)^2}/\bar L_0$; $\bar L_0=N^{-1}\sum_rL_r$; normalised range $=(L_{max}-L_{min})/L_{eq}$.
- Local distance contrast $c_v^2=\sum_ja_{vj}(L_{vj}+\mu_{L,j}-\bar d_v)^2$; mean contrast $=\sum_vp_vc_v/\sum_vp_v$; variance sum $=\sum_vp_vc_v^2=\sigma_L^2$.
- Parent inverse-root slope $=\sum_ef_eL_eS_e^{-1/2}/\sum_ef_eL_e$.
- $G_M=\sum_eL_eB_e^{2/5}f_e^{3/5}S_e^{-3/10}$ over the whole network; candidate is $\Delta G_M$.
- Harmonic reach width $=\sum_eL_e/\sum_e(L_e/B_e)$; P10 is the length-weighted width quantile.
- $\Delta x=x_c-x_p$; ratio $=x_c/x_p$; $p,c$ denote parent and child.

## Group 1: Parent allocation diversity

2 affine classes. Minimum conservative Spearman: 0.9316; minimum absolute Pearson floor: 0.9767.

H_p versus I_p. Both accumulate exposure-weighted local splitting diversity.

- A054: `B_parent_route_entropy`
- A065: `B_parent_split_impact_sum`

**Recommendation:** Prefer parent route entropy H_p. Keep intensity as a documented alternative; Shannon and quadratic diversity are not universally equivalent.

## Group 3: Parent path-width diversity

2 affine classes. Minimum conservative Spearman: 1.0000; minimum absolute Pearson floor: 0.9937.

H_q versus exp(H_q): exact nonlinear re-expression, not affine equivalence.

- A123: `path_parent_effective_n_paths_width`
- A133: `path_parent_width_entropy`

**Recommendation:** Prefer H_q; effective path count can be derived for interpretation. This is the same information for a split-order tree model.

## Group 5: Parent local allocation asymmetry

3 affine classes. Minimum conservative Spearman: 0.9487; minimum absolute Pearson floor: 0.9680.

Visit-weighted means of max(a), 1/sum(a²), and local normalised Shannon entropy.

- A049: `B_parent_dominance_mean`
- A051: `B_parent_effective_branches_mean`
- A060: `B_parent_split_entropy_mean`

**Recommendation:** Provisionally prefer parent dominance for direct interpretation. Binary splits dominate this design; dominance does not describe the distribution among all minor branches at multiway splits. Recheck these alternatives in real networks.

## Group 9: Slope scale

6 affine classes. Minimum conservative Spearman: 0.9382; minimum absolute Pearson floor: 0.9171.

Parent and child mean/minimum slope, a parent-child maximum-slope affine class, and parent allocated-length-weighted S^(-1/2).

- A088: `parent_slope_slow`
- A100: `path_child_slope_max`; `path_parent_slope_max`
- A101: `path_child_slope_mean`
- A102: `path_child_slope_min`
- A131: `path_parent_slope_mean`
- A132: `path_parent_slope_min`

**Recommendation:** Do not reduce all six classes to one yet. Retain parent mean slope, parent minimum slope and inverse-root slope as distinct candidates pending real-network checks. Parent/child means can be represented by parent mean plus the existing mean-slope change. Maximum slope is a design-limited alternative. Common reference-slope variation drives much of the correlation; inverse-root slope emphasises flat reaches.

## Group 11: Parent distance heterogeneity

2 affine classes. Minimum conservative Spearman: 0.9921; minimum absolute Pearson floor: 0.9765.

Visit-weighted mean local branch contrast versus whole-route SD sigma_L,p.

- A046: `B_parent_branch_length_contrast_mean`
- A058: `B_parent_route_length_sd`

**Recommendation:** Prefer parent route SD. It is the aggregate route-distance heterogeneity. Local mean contrast is not the same as the sum of contrast variances; distributed branching can break the observed similarity.

## Group 13: Parent relative heterogeneity

3 affine classes. Minimum conservative Spearman: 0.9464; minimum absolute Pearson floor: 0.9325.

Unweighted route CV, route range/L_eq, and slope CV.

- A127: `path_parent_path_length_cv`
- A128: `path_parent_path_length_range_norm`
- A130: `path_parent_slope_cv`

**Recommendation:** Among the two unweighted distance measures, prefer CV as the fuller distribution summary, pending comparison with weighted CV. Retain slope CV separately until real-network checks: conserved head loss couples slopes and lengths here. Do not call the unweighted descriptor mathematically wrong.

## Group 19: Parent and child area

2 affine classes. Minimum conservative Spearman: 0.9672; minimum absolute Pearson floor: 0.9780.

A_p versus A_c, where A=sum(LB).

- A002: `B_child_area_proxy`
- A043: `B_parent_area_proxy`

**Recommendation:** Prefer parent area as scale only if the existing delta A is retained: A_c=A_p+delta A exactly. High parent-child correlation alone does not show the difference is irrelevant.

## Group 20: Width scale

3 affine classes. Minimum conservative Spearman: 0.9186; minimum absolute Pearson floor: 0.9438.

Parent length-weighted reach width versus parent and child sum of harmonic route widths.

- A069: `B_parent_width_mean`
- A094: `path_child_equivalent_width`
- A126: `path_parent_equivalent_width`

**Recommendation:** Retain parent mean reach width and parent equivalent path width as distinct candidates for now. The latter counts widths across enumerated routes and can repeat shared reaches. Child equivalent width can be reconstructed if its existing difference is retained.

## Group 29: Elongation

4 affine classes. Minimum conservative Spearman: 0.9672; minimum absolute Pearson floor: 0.9714.

Parent/child mu_L²/A and parent/child L_eq/W_eq.

- A001: `B_child_area_elongation`
- A042: `B_parent_area_elongation`
- A092: `path_child_elongation`
- A124: `path_parent_elongation`

**Recommendation:** Retain one parent representative of EACH definition pending comparison: area elongation and equivalent elongation. They are different shape definitions. Child forms can be reconstructed with their respective existing differences, if retained; high correlation does not choose a definition.

## Group 32: Absolute length scale

8 affine classes. Minimum conservative Spearman: 0.9378; minimum absolute Pearson floor: 0.9449.

Parent and child mean/max/equivalent lengths, parent total channel length, and the common parent-child shortest length.

- A008: `B_child_route_length_max`
- A009: `B_child_route_length_mean`
- A010: `B_child_route_length_min`; `B_parent_route_length_min`
- A056: `B_parent_route_length_max`
- A057: `B_parent_route_length_mean`
- A066: `B_parent_total_length`
- A093: `path_child_equivalent_length`
- A125: `path_parent_equivalent_length`

**Recommendation:** Use parent allocated mean length as the primary scale representative, with existing delta mean length retaining child information. Preserve total length, shortest/longest length and equivalent length as real-network challengers: they have different meanings despite common scale variation on this design. This is a provisional preference, not proof of interchangeability.

## Group 41: Change in allocation diversity

2 affine classes. Minimum conservative Spearman: 0.9316; minimum absolute Pearson floor: 0.9813.

delta H versus delta I.

- A028: `B_delta_route_entropy`; `route_entropy_change`
- A034: `B_delta_split_impact_sum`

**Recommendation:** Prefer route-entropy change for consistency with parent entropy; splitting intensity remains a nonlinear-diversity alternative.

## Group 51: Change in distance heterogeneity

2 affine classes. Minimum conservative Spearman: 0.9874; minimum absolute Pearson floor: 0.9631.

delta longest route length versus delta route SD.

- A030: `B_delta_route_length_max`
- A032: `B_delta_route_length_sd`; `route_spread_change_m`

**Recommendation:** Prefer delta route SD: it accounts for allocation, whereas the longest route can carry negligible flow. Recheck long, low-allocation routes in real networks before treating the longest-route change as redundant.

## Group 53: Distance change versus Manning combination

2 affine classes. Minimum conservative Spearman: 0.9643; minimum absolute Pearson floor: 0.9035.

delta mu_L versus delta G_M.

- A031: `B_delta_route_length_mean`; `length_delta_km`
- A084: `delta_G_manning`

**Recommendation:** Retain BOTH. G_M has explicit width/allocation/slope scaling and whole-network scope; delta mu_L describes the evaluated B domain. Correlation does not establish that the additional scaling is dispensable. This is a key comparison, not a safe deletion.

## Group 55: Relative distance change

3 affine classes. Minimum conservative Spearman: 1.0000; minimum absolute Pearson floor: 0.9930.

delta(mu_L/L_min), an affine class containing mean-length/area/elongation/sinuosity ratios, and log1p(mean-length ratio).

- A033: `B_delta_sinuosity_proxy`
- A073: `B_ratio_area_elongation`; `B_ratio_area_proxy`; `B_ratio_route_length_mean`; `B_ratio_sinuosity_proxy`
- A087: `log1p__B_ratio_route_length_mean`

**Recommendation:** Prefer raw mean-route-length ratio as canonical; log1p is an exact invertible transformation on this positive domain. Equalities with area and elongation ratios are design-specific alternatives requiring real-network checks.

## Group 56: Relative heterogeneity change

2 affine classes. Minimum conservative Spearman: 0.9113; minimum absolute Pearson floor: 0.9208.

delta allocated route CV versus child/parent longest-route length.

- A029: `B_delta_route_length_cv`; `route_spread_change_over_parent_mean`
- A077: `B_ratio_route_length_max`

**Recommendation:** Provisionally prefer delta allocated route CV for allocation-aware heterogeneity. The maximum-length ratio measures an extremal route change, not the same quantity; retain as a real-network challenger.

## Group 59: Change in relative heterogeneity

2 affine classes. Minimum conservative Spearman: 0.9792; minimum absolute Pearson floor: 0.9340.

delta unweighted path CV versus delta slope CV.

- A112: `path_delta_path_length_cv`
- A115: `path_delta_slope_cv`

**Recommendation:** Retain both for now. Conserved head loss couples these on this grid; heterogeneous real slopes can separate them. The unweighted measure should later be compared with its weighted counterpart.

## Group 68: Relative allocation diversity

3 affine classes. Minimum conservative Spearman: 0.9961; minimum absolute Pearson floor: 0.9885.

Child/parent entropy ratio, child/parent splitting-intensity ratio, log1p entropy ratio.

- A076: `B_ratio_route_entropy`
- A078: `B_ratio_split_impact_sum`
- A086: `log1p__B_ratio_route_entropy`

**Recommendation:** Prefer raw route-entropy ratio; log1p adds no independent information. The quadratic-diversity ratio remains a documented alternative, not a universal duplicate.

## Group 69: Child diversity

5 affine classes. Minimum conservative Spearman: 0.9649; minimum absolute Pearson floor: 0.9511.

Child allocated-route entropy, summed splitting intensity, and harmonic-path-width dominance/effective count/entropy.

- A006: `B_child_route_entropy`
- A013: `B_child_split_impact_sum`
- A090: `path_child_dominant_width_fraction`
- A091: `path_child_effective_n_paths_width`
- A104: `path_child_width_entropy`

**Recommendation:** Provisionally prefer child allocated-route entropy. Child H can also be reconstructed from parent H and delta H if both are retained. Recheck route-weight definitions and multiway children in real networks before discarding harmonic-width alternatives.

## Group 70: Child distance spread

2 affine classes. Minimum conservative Spearman: 1.0000; minimum absolute Pearson floor: 0.9333.

Child branch variance sum V versus child route SD sigma_L. For this single-source/sink DAG allocation, V=sigma_L².

- A005: `B_child_branch_length_variance_sum`
- A011: `B_child_route_length_sd`

**Recommendation:** Prefer route SD in metres. The variance sum is mathematically determined by SD; this is a strong consolidation with stated graph/allocation assumptions.

## Group 71: Child relative heterogeneity

3 affine classes. Minimum conservative Spearman: 0.9615; minimum absolute Pearson floor: 0.9986.

Child unweighted route CV, normalised range and slope CV.

- A096: `path_child_path_length_cv`
- A097: `path_child_path_length_range_norm`
- A099: `path_child_slope_cv`

**Recommendation:** Prefer route CV over range within the unweighted family, but retain slope CV separately until real-network checks. Same distinction as the parent group.

## Group 77: Heterogeneity of local allocation

2 affine classes. Minimum conservative Spearman: 1.0000; minimum absolute Pearson floor: 0.9915.

Visit-weighted SD of local dominance versus SD of local effective branch count.

- A050: `B_parent_dominance_sd`
- A052: `B_parent_effective_branches_sd`

**Recommendation:** Provisionally prefer dominance SD, consistent with mean dominance. Averages/SDs of nonlinear transforms are not equivalent in general; real multiway splits require a check.

## Group 84: Narrow-width emphasis

2 affine classes. Minimum conservative Spearman: 0.9250; minimum absolute Pearson floor: 0.9487.

Child harmonic reach width versus length-weighted 10th-percentile width.

- A016: `B_child_width_harmonic`
- A018: `B_child_width_p10`

**Recommendation:** Retain both for now: harmonic width uses the full distribution with emphasis on narrow reaches; P10 is a quantile. Similarity on a grid with few width classes need not survive real width distributions.

## Overall assessment

The strongest consolidations are invertible transformations (entropy/effective count, raw/log1p ratios), variance/SD identities, and explicit parent-plus-difference reconstruction when the difference is retained. Other groupings mostly identify plausible alternatives, not proven redundancy. In particular do not merge distance change with Manning geometry, mean slope with slow-reach slope emphasis, or route heterogeneity with slope heterogeneity solely because this grid correlates them. The 91 slots are a screening organisation, not a forced final predictor count.

Sources: predictor_similarity_20261102/results cluster and pair tables; RAPID/investigations/parent_child_predictor_exploration/features.py; synthetic_experiments/investigations/geometry_predictor_screen.py; synthetic_experiments/investigations/current_predictor_bridge.py; concept inventory dated 2026-11-02.
