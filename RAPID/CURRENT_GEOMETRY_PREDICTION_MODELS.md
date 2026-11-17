# Current geometry-based prediction models

**Reference: GEOMETRY-PREDICTOR-SPEC-2026-11-06.** Consolidated on 6 November 2026 at the user's request. Model comparisons were completed on 3–4 November; this document records the accepted working specification and retained secondary model. No new fits or coefficient changes were made for this record.

**Primary:** six-input quadratic ridge with entropy, complexity and ΔGₘ.

**Secondary:** matched six-input quadratic ridge with entropy, complexity and ΔArea instead of ΔGₘ. Keep this as an interpretable geometric sensitivity model, not an equally performing alternative.

Machine-readable settings: [geometry_prediction_models.json](config/geometry_prediction_models.json). These describe the completed evaluation and current plotting choices; they are not a deployed coefficient file or a new routing configuration.

## Inputs and why they are retained

| Order | Group | Primary column | Rationale |
|---|---|---|---|
| 1 | Length | `B_delta_route_length_mean` | Allocated mean route-length change represents the main synthetic timing control while accounting for which paths carry width-based flow allocation. |
| 2 | Slope | `parent_slope_slow` | Parent allocated-length-weighted mean of S⁻¹ᐟ² describes the slope setting; it is not arithmetic mean slope. |
| 3 | Width | `log1p__B_ratio_width_mean` | Child/parent length-weighted mean-width ratio; each physical reach counted once. Replaces problematic summed equivalent-path-width representation. Log1p is inherited from the matched RF panel and retained for consistent comparisons, not proven optimal for ridge. |
| 4 | Topology | `path_delta_topologic_complexity_score` | Change in log route count plus log internal bifurcation/confluence counts; retains information that was useful for full-network collapses. |
| 5 | Combined | `delta_G_manning` | Geometry-only Manning accumulation change. Better than the matched area replacement under all three reporting metrics and strategies at the current penalty. |
| 6 | Topology/allocation | `delta_entropy` | Change in width-allocation-based route entropy; adds allocation diversity/imbalance information beyond graph counts. Improved ordinary prediction when combined with complexity. |

The secondary model substitutes **`B_delta_area_proxy` at position 5**, with the other five columns and their order unchanged. Area is sum of reach length × width; it is not storage volume. It is simpler to interpret and less dependent on the particular Manning-derived exponent structure, but predicts substantially worse in the completed comparison.

For state s ∈ {p,c}, let L be reach length (m), W width (m), S positive slope (m/m), and f the geometry-derived allocated fraction. At a split, allocation is proportional to outgoing widths; fractions add at confluences. On the matched affected domain U:

\[
\mu_{L,s}=\sum_{e\in U_s} f_{e,s}L_{e,s},\quad x_1=\mu_{L,c}-\mu_{L,p},\qquad
x_2=\frac{\sum_{e\in U_p}f_{e,p}L_{e,p}S_{e,p}^{-1/2}}{\sum_{e\in U_p}f_{e,p}L_{e,p}}.
\]

\[
\bar W_s=\frac{\sum_{e\in U_s}L_{e,s}W_{e,s}}{\sum_{e\in U_s}L_{e,s}},\qquad
x_3=\ln(1+\bar W_c/\bar W_p).
\]

\[
T_s=\ln(1+N_{\mathrm{routes},s})+\ln(1+N_{\mathrm{bif,int},s})+\ln(1+N_{\mathrm{conf,int},s}),\quad x_4=T_c-T_p.
\]

\[
G_{M,s}=\sum_{e\in\Omega_s}L_{e,s}W_{e,s}^{2/5}f_{e,s}^{3/5}S_{e,s}^{-3/10},\quad x_5=G_{M,c}-G_{M,p};\qquad
x_{5,\mathrm{secondary}}=\sum_{e\in U_c}L_{e,c}W_{e,c}-\sum_{e\in U_p}L_{e,p}W_{e,p}.
\]

\[
H_s=-\sum_k\pi_{k,s}\ln\pi_{k,s},\quad x_6=H_c-H_p.
\]

π is the complete-route probability under width splitting. Internal bifurcations exclude the domain inlet and internal confluences exclude the outlet. All logs are natural. The input groups overlap mathematically and are not independent causal effects.

**Scope:** ordinary cases use matched closed parent/child domains. Full-network controls retain the implemented edited-reach convention for length, width, area and split-entropy rewards, while preserving graph connectivity for allocation and topology. G sums over the whole network Ω, allowing allocation changes on surviving reaches to contribute. Do not silently replace these domain definitions when rebuilding features. Detailed definitions and code references: [finalization note](../synthetic_experiments/notes/2026-11-04_quadratic_ridge_finalization_plan.md).

## Model and exact numerical settings

| Setting | Recorded value | Why / status |
|---|---|---|
| Response | Saved fixed-Manning-depth RAPID child-minus-parent centroid shift | Same response and completed population as matched comparisons; no event-window or routing changes. Negative = advance. |
| Target units | Hours for fitting; minutes for reporting | Simple linear unit conversion; targets are not logged, offset or converted to absolute values. |
| Model | Degree-two polynomial ridge | Explicit equation; permits curvature and pairwise interactions. Six descriptors yield 6 main effects + 6 squares + 15 products = **27 terms**, plus intercept. |
| Preprocessing | Expand raw engineered columns, then training-only weighted centering/scaling of expanded terms | Keeps penalty comparable across term scales. Do not standardize on held-out rows or move scaling before expansion without treating it as a model change. |
| Constant terms | Zero training-variance terms dropped | Avoid uninformative columns; retain the intended 27-term basis where varying. |
| Ridge strength | Per-row λ = **0.0001**; scikit-learn α = λ × sum(w) | Inherited from original synthetic quadratic-ridge procedure, held fixed for matched comparisons. **Not selected as the optimal real-network penalty.** |
| Training weights | Equal total weight per river; normalized to mean one | Prevent rivers with many child states from dominating the fitting objective. For N rows across R rivers with nᵣ rows each: wᵢ = N/(R nᵣ). Therefore sum(w) = N. |
| Objective | Weighted squared error + α‖β‖² | Intercept unpenalized; no positivity constraints, target clipping or post-fit calibration. |
| Solver | `Ridge(solver='svd', fit_intercept=True)` | Deterministic numerical solution. Each fold was checked against independently computed normal equations. No stochastic fitting seed is needed. |
| Hyperparameter search | None in completed results | Only predictor-panel changes were evaluated. Bounded grouped inner-fold penalty assessment remains pending. |
| RF | Historical fixed benchmark, not the chosen main family | RF retained lower ordinary MAE in the comparison; ridge offers an explicit model and better full-network performance. |

The six-input choice is a compromise: entropy improved ordinary predictions; retaining complexity recovered much of the full-network performance. It did not dominate every comparator on every metric. An ordinary/full-network difference does not establish a size-driven mechanism.

## Validation and reporting settings

- Same saved real splits for every model comparison: **compound-family grouped, 5 repeats × 5 folds; leave one network out, 15 folds; network-grouped fivefold, 5 repeats × 5 folds**. Total: 65 fits per panel. These test held-out families within represented rivers and held-out rivers, respectively.
- All **2,185 pairs** retained: 2,170 ordinary comparisons and 15 full-network controls. No extreme-error cases removed.
- **Current headline:** combined population, **equal weight per pair**. This is deliberately different from equal-river training weights. Calculate metrics on complete out-of-fold predictions for each repeat, then average repeat metrics; do not average predictions first.
- **Diagnostics:** ordinary/full-network scores separately; equal-river scores (and equal-cohort/equal-river weighting for their combination); per-river errors and repeat ranges.
- R² = 1 − SSE/SST, not Pearson r². MAE in minutes. Supplementary KGE-2009 uses correlation, SD ratio and mean ratio with equal component weights. Signed near-zero means can destabilize KGE; preserve components and bias diagnostics.
- Signed-log plotting is **display only**, sign(t) log₁₀(1+|t|) with t in minutes. Scores and fits use untransformed shifts.
- The figure shows all saved held-out repeat predictions: 10,925 points for five-repeat strategies, 2,185 for leave-one-network-out. Repeated points are not independent samples.

## Completed headline performance

Combined, equal-pair weighting; same fixed penalty for both models:

| Validation | Primary R² | Primary KGE | Primary MAE (min) | Secondary R² | Secondary KGE | Secondary MAE (min) |
|---|---:|---:|---:|---:|---:|---:|
| Compound-family grouped | 0.8666 | 0.8366 | 0.9397 | 0.4178 | 0.6614 | 1.2267 |
| Leave one network out | 0.8571 | 0.8741 | 1.0839 | 0.6117 | 0.7555 | 1.3811 |
| Network-grouped fivefold | 0.8627 | 0.8595 | 1.0782 | -1.0093 | 0.2003 | 1.6351 |

Keep the secondary result even though it is worse: it documents the sensitivity to using the hydraulically informed G descriptor. It was **not independently penalty-tuned**, so it is a matched fixed-specification comparison rather than the best possible area model.

## Scientific reasoning and limitations

No depth, roughness or simulated-discharge input enters either predictor set. G's exponents still derive from Manning hydraulics and the allocation rule; its success is conditional and is not evidence of a universal physical law. Geometry/slope uncertainty and information overlap remain relevant. Replacing G with area changes both the descriptor's structure and its established scope, so this is not an isolated test of one exponent.

The selection history combines synthetic screening, exclusion of problematic equivalent-width representations, interpretability choices (P90 removal), and real-network comparisons. Mean-width ratio was not uniquely rediscovered by synthetic selection. Real validation data have been inspected repeatedly; describe performance as exploratory assessment, not independent final confirmation.

## Where to find and reproduce the results

- [Primary investigation and CLI](investigations/entropy_complexity_g_ridge_20261104/README.md), [primary results](outputs/entropy_complexity_g_ridge_20261104/report.md).
- [Secondary matched investigation and CLI](investigations/six_area_vs_g_20261104/README.md), [both-model score table](outputs/six_area_vs_g_20261104/summary.csv).
- [KGE calculation](investigations/six_input_kge_20261104/README.md).
- [Prediction figure notebook](notebooks/hydrography_rf_prediction_vs_modelled.ipynb): `RIDGE_MODEL='primary_G'` or `'secondary_area'`. Default remains primary G, compound-family, all pairs, equal-pair scores.
- [Chronological decision record](../synthetic_experiments/notes/2026-11-02_predictor_selection_decision_log.md) and [6-input decision](../synthetic_experiments/notes/2026-11-04_six_input_ridge_decision.md).

## Still pending

1. Prespecify bounded penalty sensitivity and evaluate only within grouped inner training folds, preserving the recorded outer evaluations.
2. Review extreme errors, repeat variability and extrapolation; check prediction/coefficient sensitivity.
3. Freeze final preprocessing, penalty rule, coefficients and applicability scope only after those checks. Current records contain fold evaluations, not a deployed full-data coefficient artifact.

Retain the secondary model and its existing matched results throughout; do not silently replace or discard it during finalization.
