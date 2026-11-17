# Geometry-only candidate reduction before RF modelling

**Reference:** SYN-CANDIDATE-REDUCTION-2026-10-30 / SYN-PRE-RF-SHORTLIST-2026-10-30.  
**Decision context:** requested after the broad 205-representation screen, to reduce the library using mathematical validity, redundancy, mechanisms and synthetic response associations **before** a further RF subset search.  
**Status:** completed descriptive investigation; exploratory candidate pool, not a selected or externally validated predictive model.

## 1. Main finding and recommendation

The synthetic results support a small set of distance-change and route-heterogeneity concepts, augmented by width and slope scale, allocation and topology context. A single correlation cutoff does not justify selecting ten inputs from 205. The audit supports a **19-candidate pool: 14 primary candidates, three alternatives to test as replacements, and two topology-context reserves**. This is not a recommendation to put all 19 in one RF.

The strongest Manning-centroid descriptor is Manning ΔG. Mean route-length change, equivalent-length change, channel-area change and route spread also associate strongly with centroid shift. For peak shift, channel-area change leads this group. However, descriptors that rank highly under Manning depth do not necessarily rank highly under BASED depth. Retaining absolute scale and allocation information is therefore warranted even when their marginal correlations under Manning depth are modest.

The final pool is **analyst-curated using explicit physical and mathematical considerations**, informed by descriptive statistics. The numerical analyses alone do not uniquely identify these 19, and no claim of optimality is made. Several alternatives remain deliberately available because apparent equivalence on this simple grid need not transfer to real compound networks.

## 2. Population, scope and response definitions

- 6,561 parameter combinations: eight controls at three levels each.
- Seven parent–child collapse types, giving 45,927 comparison entries per depth approach.
- Three depth approaches: fixed BASED, fixed Manning and variable Manning; 137,781 response rows in total.
- The 205 geometry representations are invariant across the three hydraulic approaches; this was checked numerically. Repeated geometry across methods was used once for predictor–predictor redundancy, not counted as independent observations.
- Main responses: child-minus-parent centroid shift and peak shift, in minutes, as stored by the completed experiment pipeline.
- Most descriptors refer to the closed outer B domain; affected-domain size uses the selected source-unit footprint. Manning ΔG explicitly uses the whole network. These scopes must not be silently interchanged when implementing the real-case feature table.
- The main response screen uses variable Manning; fixed Manning and fixed BASED are reported as separate sensitivity checks. Cross-method tables use the **same six collapse types**. The near-null main-island Manning case is retained in geometry/validity checks but not assigned a normalized response correlation.

The existing response-resolution screen excludes a type if its response range is at most 4 s for centroid or 30 s for peak. These thresholds are operational screening choices, **not demonstrated error bounds**. In the main-island variable-Manning case, centroid change ranges approximately from −0.000003 to +0.043230 min, and peak change from −0.25 to +0.25 min. The BASED main-island response is not negligible and remains included in its method-specific exports.

These are deterministic, related factorial comparisons, not independent river observations. Some treatment response series are identical (e.g. crossover peak-shift series), so equal weighting of types is a reporting convention, not six independent replications.

## 3. Mathematical coverage: what “undefined” meant here

| Audit category | Representations | Interpretation |
|---|---:|---|
| Complete and varying within at least one type | 139 | Eligible for within-type descriptive screening |
| Complete but varying only between types | 22 | Topology/context information; cannot be judged using within-type sensitivity |
| Partly defined | 40 | Not suitable as universal inputs under the existing definitions |
| Constant across the entire grid | 4 | No available evidence of predictive value on this design |
| **Total** | **205** | Representations, not independent mechanisms |

The library has 66 metric labels, including transformed and derived labels; these are not 66 independent concepts either.

Every missing value was assigned a mathematical reason in `undefined_reasons.csv`:

1. No bifurcation exists in the relevant child state, so a mean of bifurcation properties has no population to average.
2. A parent denominator is zero, so a child/parent ratio is undefined.
3. A ratio inherits a missing parent or child statistic.
4. Route-length skewness is undefined when route spread is zero.

A valid zero is different: zero route spread, zero width CV and zero route entropy have meaningful interpretations. No missing values were imputed as zero.

The four global invariants are `B_delta_route_length_min`, `B_ratio_route_length_min`, `path_delta_head_drop`, and `path_delta_slope_max`. Their invariance follows from this experiment design and is not a general statement about river collapses.

### Numerical quality issue uncovered

Some theoretically zero differences retained floating-point subtraction residuals. A tolerance based only on the small difference could misclassify these as genuine variation. The audit therefore first rounds each predictor using its **full-column** scale, then checks within-type variation. The tolerance is `1e-10 × max(maximum absolute value, 1)` in the stored units. This is numerical bookkeeping, not a physical minimum effect.

The audit also flagged finite stored ratios/skewness values that are undefined once a roundoff-only denominator is recognised as zero. Examples include route-length CV ratios and branch-variance ratios. All such entries are recorded in `finite_but_undefined_audit.csv`; they are missing in audit copies, while the source results are preserved. The 40 partial-representation classification remains unchanged. None of the 19 shortlisted representations is affected by these undefined-ratio exceptions.

Repeating the variation check with tolerance factors 1e-8, 1e-10 and 1e-12 changed **none** of the predictor/type varying-versus-constant flags. This supports the numerical classification for this dataset, not a universal tolerance choice.

## 4. Redundancy: what was tested and what it does not prove

### 4.1 Exact and empirical affine redundancy

Among the 161 complete, globally nonconstant representations, affine comparisons identified **27 redundant representations**, leaving **134 empirical affine equivalence classes**. The maximum absolute affine residual had to be at most 1e-8 of the target predictor's observed range.

Some equivalences are definition-level identities, such as route-length change in metres versus kilometres, and the duplicate implementations of route-spread or entropy differences. Others are empirical consequences of the synthetic design. The latter must not be treated as universal identities on real networks. `affine_equalities.csv` contains 71 pairwise equality edges; **71 is not the number of redundant variables**.

For example, branch-length variance and route-length SD contain the same ordering information through a square/square-root relation, but are not affine duplicates. Pearson correlations can differ despite perfect Spearman correlation. For a tree predictor, such monotonic re-expressions usually offer little new split-order information; this is a reason to prefer the clearer representation, not evidence that the underlying mechanism is unimportant.

### 4.2 Predictor–predictor correlations

Spearman and Pearson correlations were calculated both across all seven types and within each type. Spearman measures monotonic association; Pearson measures linear association. Neither is a physical sensitivity coefficient, and equal correlation with a response does not imply two predictors contain the same information.

For a conservative redundancy grouping, two complete predictors had to:

- have the same set of types in which they vary;
- have consistent correlation sign across informative types and the pooled population;
- satisfy a correlation threshold in the pooled population **and every informative type**.

Complete-linkage hierarchical clustering was applied to one minus this conservative absolute-Spearman similarity. This prevents a chain of pairwise links from grouping mutually weakly related endpoints. Correlation distance need not be a Euclidean metric; the clusters are descriptive groupings, not latent physical classes.

| Absolute-Spearman threshold | Groups among 161 complete, nonconstant representations |
|---|---:|
| 0.85 | 69 |
| 0.90 | 79 |
| 0.95 | 98 |
| 0.99 | 114 |

The thresholds are sensitivity choices, not significance levels. **Correlation clustering alone does not reduce the library to 10–20 defensible predictors.** Formula interpretation and intended scope are necessary additional choices.

### 4.3 Important examples

- **Manning ΔG versus mean route-length change:** pooled Spearman 0.989, Pearson 0.929, and minimum within-type absolute Spearman 0.964. They are strongly related but not interchangeable: Manning ΔG adds explicit width and slope scaling. Their mean total-variation distance between factorial variance signatures is 0.161 on a 0–1 scale. This signature comparison is descriptive, not a predictive gain.
- **Equivalent-length change versus mean route-length change:** pooled Spearman 0.975 and Pearson 0.999, but their weighting definitions differ. In the main-island case, equivalent-length change spans about −50.3 to +127.7 m while allocated mean route-length change is zero. Historical harmonic-path-width weighting is therefore not equivalent to discharge allocation. Equivalent-length change is retained as a replacement candidate, not automatically added beside mean-route change.
- **Parent route spread versus route-spread change:** absolute within-type Spearman is 1 in the six informative types, yet pooled Spearman is only −0.857. The relationship differs between collapse types. One global conversion does not follow from perfect within-type correlations.
- **Bifurcation density versus parent mean route length:** pooled Spearman is −0.968. Because bifurcation counts are fixed within a type, much of this signal is inverse length scaling. It should not be described as independent evidence for topology.
- **Parent dominance versus parent length–width association:** pooled Spearman is only 0.259. Allocation asymmetry and which paths carry width are distinct questions; one does not substitute for the other.

## 5. Response association and robustness

The primary summary is the equal-weight mean of absolute within-type correlations. Signed per-type values, minima/maxima and the number of informative types are exported. Absolute summaries indicate strength, not direction, and are not R². No p-values or iid confidence intervals are reported for the deterministic factorial population.

| Descriptor | Centroid: mean absolute Spearman | Peak: mean absolute Spearman |
|---|---:|---:|
| Manning ΔG | 0.981 | 0.699 |
| Mean route-length change | 0.956 | 0.744 |
| Equivalent-length change | 0.953 | 0.750 |
| Channel-area change | 0.950 | 0.769 |
| Route-spread change | 0.914 | 0.728 |
| Parent slope CV | 0.807 | 0.556 |
| Mean slope change | 0.634 | 0.721 |

These values use variable Manning and the six response-resolved types. The table deliberately does not imply that the seven entries are independent predictors.

### Spearman versus Pearson

For Manning ΔG and variable-Manning centroid shift, mean absolute Pearson is approximately **0.99983**, while Spearman is 0.98144. The difference is largely a tie issue: cases with theoretically zero change retain tiny response differences, which receive different ranks. Rounding centroid shifts to one second gives Spearman approximately 0.99997. The primary unrounded result remains reported. The one- and four-second tests diagnose ranking sensitivity; they do not establish measurement precision or justify replacing the original response.

By contrast, mean slope change has centroid Spearman about 0.634 but Pearson about 0.123, and peak Spearman about 0.721 but Pearson about 0.341. A strong rank association is not a claim of a strong linear scaling. Scatterplots and type-specific behaviour matter.

### Dependence on the hydraulic approach

The following comparison uses identical six-type support:

| Descriptor | Fixed BASED centroid | Fixed Manning centroid | Variable Manning centroid |
|---|---:|---:|---:|
| Manning ΔG | 0.630 | 0.981 | 0.981 |
| Mean route-length change | 0.597 | 0.957 | 0.956 |
| Channel-area change | 0.548 | 0.949 | 0.950 |
| Parent mean route length | 0.859 | 0.541 | 0.559 |
| Affected-domain size | 0.844 | 0.470 | 0.489 |

Thus “length-related geometry dominates” is supported, but **which length representation is useful depends on the width–depth closure**. The excellent Manning ΔG relationship is also a consistency check of Manning-based hydraulics. It is not independent validation of a universal river law.

### Sensitivity to control ranges and type composition

Each of the 24 factor/level combinations was omitted in turn and correlations recomputed within the retained grid. These are design-sensitivity ranges, not confidence intervals or held-out prediction results. For centroid shift, across the six informative types:

- Manning ΔG mean absolute Spearman: approximately 0.936–1.000.
- Mean route-length change: 0.917–0.969.
- Equivalent-length change: 0.906–0.965.
- Channel-area change: 0.894–0.962.
- Route-spread change: 0.775–0.930.

The strongest distance-related associations persist, but ranks and magnitudes depend on the grid ranges. Leaving one collapse type out was also exported. The topology reserves have no within-type correlation to summarise; they are not assigned a zero score.

## 6. The proposed small pool

**Primary pool (14):** Manning ΔG; mean route-length change; route-spread change; channel-area change; parent mean route length; affected-domain size; parent mean width; parent mean slope; mean slope change; parent width CV; parent dominance; route-entropy change; parent length–width association; parent route spread.

**Replacement candidates (3):** equivalent-length change; parent slope CV; log1p mean-route-length ratio.

**Context reserves (2):** parent log1p path count; child-minus-parent log1p path count. These are reserved for the real-case stage because fixed synthetic topologies do not establish their within-type explanatory power.

This pool deliberately retains some weak marginal correlates. Parent width, width CV, dominance and entropy are not selected because their standalone response correlations are high. They preserve distinct scale/allocation mechanisms and possible interactions. They must earn their place in subsequent predictive tests. Weak marginal association cannot rule out a conditional effect, but it does not prove one either.

The 19 entries are not mandatory components of one model. In particular, initially compare equivalent-length change **instead of** mean route-length change, and a relative-distance representation **instead of or alongside explicitly tested scale components**, rather than automatically including all of them.

Every one of the 205 candidates has a recorded disposition in `candidate_decisions.csv`: 14 primary, 3 alternatives, 2 context reserves, 40 conditional-only, 27 empirical affine aliases, 27 correlated alternatives, 4 design invariants, 4 unselected topology reserves and 84 deferred from the small pool. The final 84 are **not demonstrated to be useless**. They remain available if the small pool fails, and their exclusion is a practical, interpretable reduction rather than a statistical proof.

## 7. Equations for the retained concepts

Let \(p_e\) denote the fraction of unit inflow traversing edge \(e\), obtained from width-based splitting and accumulated at confluences; \(p_v\) is the visit probability of node \(v\). At a bifurcation, \(a_{vj}=B_j/\sum_k B_k\). All changes below are child minus parent.

For complete paths \(r\), let \(\pi_r\) be the product of branch fractions along the path, and \(L_r=\sum_{e\in r} L_e\). Over the matched B domain:

\[
\mu_L=\sum_r\pi_rL_r=\sum_ep_eL_e,\qquad
\sigma_L=\sqrt{\sum_r\pi_r(L_r-\mu_L)^2}.
\]

The pool contains \(\Delta\mu_L\), \(\Delta\sigma_L\), and parent \(\mu_L,\sigma_L\). The relative alternative is \(\log(1+\mu_{L,c}/\mu_{L,p})\), matching the existing implementation; it is not \(\log(\mu_{L,c}/\mu_{L,p})\).

\[
A=\sum_e L_eB_e,\quad
\bar B=\frac{\sum_eL_eB_e}{\sum_eL_e},\quad
CV_B=\frac{\sqrt{\sum_eL_e(B_e-\bar B)^2/\sum_eL_e}}{\bar B}.
\]

The area predictor is \(\Delta A\), not polygon-union area or a width-only quantity. Width context is parent \(\bar B,CV_B\).

\[
\bar S=\frac{\sum_eL_eS_e}{\sum_eL_e},\qquad
CV_S=\frac{\sqrt{\sum_eL_e(S_e-\bar S)^2/\sum_eL_e}}{\bar S}.
\]

The pool contains parent \(\bar S\), \(\Delta\bar S\), and alternative parent \(CV_S\).

\[
D=\frac{\sum_{v\in\mathrm{bif}}p_v\max_j a_{vj}}{\sum_{v\in\mathrm{bif}}p_v},\qquad
H=-\sum_r\pi_r\log\pi_r
  =-\sum_{v\in\mathrm{bif}}p_v\sum_j a_{vj}\log a_{vj}.
\]

Retained quantities are parent \(D\) and \(\Delta H\). Parent dominance requires a bifurcating parent; this holds in the screened population but must be rechecked for any expanded real population. Entropy is zero for a single route and its difference remains meaningful after full collapse.

For the historical path-weighted quantities:

\[
W_r=\frac{L_r}{\sum_{e\in r}L_e/B_e},\quad
L_{eq}=\frac{\sum_rW_rL_r}{\sum_rW_r},\quad
A_{LW}=\frac{L_{eq}}{N^{-1}\sum_rL_r}-1.
\]

The pool contains parent \(A_{LW}\) and alternative \(\Delta L_{eq}\). Positive \(A_{LW}\) means larger harmonic path widths favour longer paths. These path weights are **not** the discharge-allocation weights \(\pi_r\). Explicit path enumeration in the historical implementation may be expensive for complex real DAGs; implementation cost and equivalent scalable calculation must be checked before using these across all real cases.

Affected-domain size is \(\sum_{e\in U}p_e L_e/1000\) km, using the selected source-unit membership \(U\) and its B-domain allocation. It is not the sum of all network lengths, nor necessarily only the physically deleted links.

Topology reserves are \(\log(1+N_p)\) and \(\log(1+N_c)-\log(1+N_p)\), where \(N\) is the number of complete directed paths.

Finally, over the **whole network**:

\[
\Delta G_M=\left[\sum_e L_eB_e^{2/5}p_e^{3/5}S_e^{-3/10}\right]_c-
\left[\sum_e L_eB_e^{2/5}p_e^{3/5}S_e^{-3/10}\right]_p.
\]

This input uses neither depth, calibrated roughness nor simulated flow, although its physical interpretation is conditional on Manning hydraulics. Unchanged contributions cancel; changes in allocation on surviving edges remain included.

## 8. What can be said in a paper now

### Suggested methods text

> We screened 205 geometric descriptor representations on a balanced synthetic design comprising 6,561 parameter combinations and seven collapse treatments. Predictor validity and redundancy were assessed separately from response association. Undefined ratios and statistics without a valid underlying population were retained as missing rather than replaced by zero. Numerical ties were handled using a scale-relative roundoff tolerance, with sensitivity checks over three tolerance levels. We evaluated empirical affine equivalence and both pooled and treatment-specific Spearman and Pearson correlations. Conservative complete-linkage correlation groups were compared at four thresholds. Descriptor responses to the imposed controls were examined using the existing factorial variance decomposition. An interpretable candidate pool was then curated from this evidence for subsequent predictive testing; no RF performance was used to optimise that pool in this investigation.

### Suggested results text

> Of the 205 representations, 139 were complete and varied within at least one treatment, 22 varied only between treatments, 40 were partially defined and four were invariant across the design. Twenty-seven complete representations were empirically affine duplicates. Correlation-based grouping remained sensitive to the threshold and did not by itself yield a small unique predictor set. Under Manning-based depth, centroid changes were most strongly associated with Manning ΔG and changes in mean route length, equivalent length, channel area and route spread. Peak changes showed weaker associations, with channel-area change providing the strongest association among these candidates. Under BASED depth, parent-network and affected-domain scale were more informative than these change descriptors. These differences indicate that the choice of hydraulic closure conditions the descriptor–response relationships. We therefore retained a mechanism-based pool of 19 candidates, including alternatives and topology-context reserves, for subsequent grouped predictive evaluation.

### Claims not supported yet

- These 19 are the optimal predictors, or all are necessary.
- A weak marginal correlation proves a predictor adds no information in combination.
- Topology is unimportant: much of its variation is not independently manipulated here.
- High synthetic association establishes physical validity in observed rivers.
- Reported correlations are held-out predictive performance, causal coefficients, or independent replications across six river types.
- The strongest variables will transfer unchanged to all real-network domains.

The shortlist is informed by the existing synthetic responses and historical analysis. Later performance assessment must avoid treating repeatedly inspected data as untouched validation. Any further target-guided subset choice should occur inside training folds; real-network transfer needs a separate grouped evaluation with explicit domain and coverage checks.

## 9. Reproducibility and outputs

No routing, calibration, new hydraulic runs, RF fits or production-model changes were performed for this investigation.

Scripts:

- `synthetic_experiments/investigations/candidate_reduction_20261030/audit.py`
- `synthetic_experiments/investigations/candidate_reduction_20261030/shortlist.py`

Outputs on the PhD-backed mounted volume:

`/Volumes/SyntheticResults20261027/analysis/candidate_reduction_20261030/`

Key tables: `inventory.csv`, `undefined_reasons.csv`, `finite_but_undefined_audit.csv`, `affine_equalities.csv`, `predictor_pairs.csv`, `predictor_pairs_by_type.parquet`, `clusters.csv`, `response_associations.csv`, `shortlist.csv`, `candidate_decisions.csv`, `common_support_methods.csv`, `leave_level_out_summary.csv`, `leave_type_out.csv`, `response_tie_sensitivity.csv`, `predictor_tolerance_sensitivity.csv`, and `shortlist_pair_evidence.csv`.

Figures, each exported as PNG and PDF:

1. `01_inventory`: coverage and available variation.
2. `02_response_associations`: shortlisted correlations, Pearson comparison and between-type ranges.
3. `03_predictor_correlation`: signed predictor correlation matrix.
4. `04_mechanisms`: main-effect variance shares showing what controls each descriptor.
5. `05_pair_examples`: illustrative relationships that a single correlation can conceal.

The notebook's Section 10 provides the main readable view. Source hashes and software versions are recorded in `provenance.json` and `shortlist_provenance.json`. Future code changes should be accompanied by a rerun rather than bypassing provenance checks.

## 10. Method references

- [SciPy Spearman correlation documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.spearmanr.html): rank-based monotonic association and constant-input limitations.
- [SciPy Pearson correlation documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.pearsonr.html): linear association.
- [SciPy hierarchical linkage documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.linkage.html): complete-linkage definition.
- [scikit-learn example on correlated predictors](https://scikit-learn.org/stable/auto_examples/inspection/plot_permutation_importance_multicollinear.html): why correlated features complicate individual importance and why rank-correlation grouping can be useful. Our conservative across-type criterion and shortlist decisions are specific to this investigation, not prescribed by that example.
