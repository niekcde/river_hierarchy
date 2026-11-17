# Controls and proposed predictors — 29 October 2026

Reference: **SYN-CONTROL-PREDICTORS-2026-10-29**.

Why: identify which experimental controls drive centroid/peak shifts, then determine whether the proposed predictors represent those same controls and interactions. This supports explanatory variable selection before model fitting.

The analysis covers all 6,561 geometries, three depth methods and seven treatments (137,781 rows). It uses the ten `depth_informed_10` predictors from `RAPID/outputs/uncapped_event_window_20261020/current_model_comparison_20261023/settings.json`, plus the Manning timing expression `delta_G_manning`. This is explicitly different from the historical ten-variable omega candidate set. No routing or prediction models are run.

## Where to read it

Existing `synthetic_predictor_validation.ipynb`, sections 6–7: select method and treatment; inspect correlation, magnitude, factorial signatures and a selectable pair of controls. The response-only factorial analysis remains in `synthetic_factor_interactions.ipynb`.

Exports: `/Volumes/SyntheticResults20261027/analysis/control_predictors_20261029/` on the PhD-drive results container. Includes complete features, matched-control statistics, main/pair factorial signatures, coverage, figures and source provenance. Rebuild with `synthetic_experiments/investigations/control_predictor_screen.py`.

## Interpretation

- Matched Spearman holds seven controls fixed, varies the eighth across three levels, then summarizes directions across contexts. Constant contexts are counted, not assigned correlation zero. Three-level ranks are coarse; P10–P90 and constant/valid counts matter.
- Magnitude is median absolute endpoint change divided by the predictor's SD over the complete method/treatment population (finite values where undefined). This is dimensionless sensitivity, not predictive accuracy. Absolute differences are taken before aggregation.
- Functional ANOVA decomposes each variable under uniform factorial weights. Main and pair terms are exact; higher-order terms are not assigned to pairs. Percentages describe each variable's own variance, not shared physical units. Similar signatures support mechanistic relevance, not predictive validation.
- Partial predictor coverage withholds full-grid ANOVA. Matched statistics use only contexts with all three levels finite. No undefined ratio is replaced with an arbitrary number.
- Depth-dependent features use BASED peak-reference depth, or Manning normal depth at Q=750. For variable-depth MCT the latter is a predefined reference snapshot, not an inferred dynamic depth. The remaining geometry comes from the same shared descriptor implementations and B-domain/source-unit conventions as the real-model bridge.
- Network extent scales the affected domain as well as the surrounding network. Its effect is not evidence of sensitivity to adding only unchanged downstream length.

## Initial findings

For variable-Manning nested/full, the centroid's main length-multiplier contribution is approximately 30.5%, versus 31.2% for Manning G. Length × extent contributes approximately 9.1% and 8.9%, respectively; inner allocation × length approximately 7.3% and 7.5%. These close signatures explain the earlier strong centroid agreement without relying on an RF fit.

Depth resistance and route-length ratio represent allocation/relative-length effects, but are invariant to uniform network extent and absolute width/slope scaling. They cannot individually encode all the dimensional timing changes. Domain size adds scale; the slope descriptor adds slope; these are complementary mechanisms.

Peak timing has a different signature: in nested/full the slope main effect is approximately 8.4% and slope × length 6.2%, compared with centroid 3.3% and 2.2%. This supports investigating the extra shape/timing mechanism, not immediately adding many predictors.

A substantive feature-definition issue: route-spread ratio is undefined for the 2,187 equal-length configurations in every treatment (one third of the grid). Width-CV ratio is also undefined for the equal-width simple split. Entropy/width-CV ratios can be constant for full collapse. These are valid structural limitations of those ratio predictors, not missing simulation results. Any later fitted-model benchmark must address their definition explicitly.

Results remain conditional on the grid ranges, fixed width allocation, forcing and hydraulic closure. No significance p-values or held-out predictive claims are made.

## Presentation and resistance connection (29 October follow-up)

Section 6 now adds an overall summary across treatments within each hydraulic method. Default: equal average of resolved within-treatment variance shares. Optional response-variance weighting emphasizes absolute within-treatment variation. Neither includes between-treatment mean differences as a control effect. Unresolved near-zero treatments are excluded and listed explicitly.

Section 7 defaults to inverse-depth G and displays matched response/predictor sweeps for length multiplier × inner allocation and length multiplier × extent. Other factors are fixed at the displayed middle grid levels. Full-grid diagnostic heatmaps remain optional. This is illustration, not selection of a universal formula or a new model fit.

The code's existing edge resistance is K_e=L_e/(B_e^2 h_e^3). Algebraically the existing inverse-depth descriptor can therefore be expressed exactly as

G_inverse_depth = sum_e f_e K_e^(1/3) L_e^(2/3) B_e^(2/3) S_e^(-1/2).

The corresponding child-minus-parent difference is unchanged by this rewriting. It is a resistance-informed path descriptor, not the network's equivalent resistance K_eq. Replacing edge K values by K_eq would lose allocation/path information and is not an identity. Absolute differences retain scale information that a resistance ratio alone cannot retain.

For comparison, the fixed-depth Manning travel-time expression is

T = (3n/5) sum_e f_e K_e^(2/9) L_e^(7/9) B_e^(4/9) S_e^(-1/2).

This follows from c=(5/(3n)) h^(2/3) sqrt(S), using the same K definition. It is a reference travel-time identity, not a newly calibrated resistance law. No alternative predictor or routing implementation has been substituted based on these algebraic identities.

### Section 7 coverage implementation

Section 7 now starts with full-grid matched coverage rather than selected background curves. It counts contexts where a three-level control sweep changes centroid timing by more than 4 seconds (30 seconds for peak), then separately reports predictor availability, numerical variation, and span relative to within-type SD. Thresholds are descriptive screens, not statistical significance or a convergence guarantee. Default predictors are depth-resistance change and inverse-depth G. Overall summaries give equal weight to response-resolved treatments; conditional columns disclose how many types have eligible contexts. Type-specific views remain selectable. The second table compares the strongest response pair interactions with predictor factorial shares and available-type counts. No predictive model is fitted and no new storage/resistance formula is implemented.

### Agreed matrix view (supersedes the Section 7 coverage view)

Following clarification, Section 7 now uses exactly the ten current predictors and three matrices: ordinary within-treatment Spearman association (8 × 10), main-effect predictor variance shares (8 × 10), and pair-interaction predictor variance shares (28 × 10). Manning G is excluded from these matrices. Pearson is optional; neither correlation measures absolute change magnitude. This differs from the earlier matched-sweep median correlation. Section 6 remains the response analysis. Overall mode averages defined per-treatment summaries equally, without pooling raw populations; available counts and individual correlations are optional diagnostics. Missing/constant predictors remain undefined. Correlations use pairwise finite values; factorial percentages still require the complete grid. Coverage-count figures are no longer the primary notebook analysis; the earlier helper remains as historical exploratory code.

### Spread and entropy differences tested alongside ratios

Section 7d adds three exploratory candidates without replacing any current model inputs: child-minus-parent route-length SD (metres), that SD difference divided by positive parent mean route length, and child-minus-parent route entropy. All 137,781 rows have finite new values. Existing ratio values were independently reproduced during extraction. Shared B-domain descriptors and width-based route probabilities are unchanged.

The same association, main-effect and all-28-pair matrices are used. Absolute spread change retains uniform network scale; its normalized version deliberately removes it. Entropy change distinguishes full collapses by how much route entropy is removed, whereas their entropy ratios all equal zero. Main-island spread differences remain zero to arithmetic precision; entropy can still change even when the centroid response is near zero. Therefore these definition improvements do not establish predictive utility or justify automatically changing the ten-variable real model.

Exports: `/Volumes/SyntheticResults20261027/analysis/spread_entropy_comparison_20261029/`. Rebuild with `investigations/spread_entropy_comparison.py`. Source hashes, candidate definitions and full-grid coverage are recorded there. No routing or model fits were run.

### Predictor synthesis figures

Section 7e adds two summary figures including centroid/peak, all ten existing predictors, and the three exploratory spread/entropy differences: main-control contributions and the strongest response pair-interaction contributions. The default uses a common population with both responses resolved (six variable-Manning treatments), averages within-treatment shares equally, and reports each column's available treatment count. This avoids treating variable/undefined predictor columns as zeros. The summaries are mechanistic comparisons, not predictor performance scores. PNG/PDF exports sit in the spread-entropy analysis directory. A compact core-predictor display is selectable in the plotting cell.

### Two-figure main story

The main presentation is now Section 7e: Figure 1 shows response contributions for all eight main controls and the four strongest response pair interactions; Figure 2 shows predictor variance contributions for exactly those same terms. The second figure includes the original ten plus three exploratory differences, with available-type counts. Both use equal weighting of the common response-resolved types within one hydraulic method (variable Manning by default). Earlier sections remain supporting detail. The narrative is length/scale/allocation-driven timing change, followed by which descriptors represent each mechanism; no fitted-model superiority is asserted. Export filenames are `synthetic_story_1_variable_manning` and `synthetic_story_2_variable_manning` (PNG/PDF) in the spread-entropy analysis directory.
