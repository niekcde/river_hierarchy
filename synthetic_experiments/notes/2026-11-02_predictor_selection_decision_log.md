# Staged predictor-selection decisions

Reference: SYN-SELECTION-DECISIONS-2026-11-02. Updated after user agreement with the review of 22 similarity groups. This is a working selection record, not a validated predictive model.

## Chronology and accounting

| Stage | Decision and evidence | Result |
|---|---|---:|
| Original inventory | 205 geometry-derived representations; later organised into 21 concepts. These are not 205 independent physical quantities. | 205 |
| Earlier shortlist, 30 October | An interpretability-based 19-candidate pool was proposed. The user subsequently requested reopening the full population because 84 candidates had been deferred without adequate exclusion evidence. The 19 remain a historical benchmark, not the current eligibility filter. | Superseded as working pool |
| Coverage restart, 2 November | No candidate is undefined everywhere. Exclude 40 partly defined representations from the universal pool; retain their definitions and limitations in the archive. This is a coverage decision, not proof of no predictive value. | 165 complete |
| Design constants | Separate four complete but globally invariant representations for real-network checks. They remain in the inventory. | 161 nonconstant + 4 pending |
| Affine screen | 71 pairwise links form 12 non-singleton classes containing 39 representations. Consolidation removes 27 slots. Distinguish identities from design-specific equalities. | 134 empirical classes |
| Similarity screen | Require absolute Spearman and Pearson ≥0.90 pooled and within informative treatments, under consistent-sign/activity rules and complete linkage. User accepted this threshold for review, not as a 10% prediction-error bound. | 69 singletons + 22 multi-class groups = 91 provisional slots |
| Accepted formula review | Retain meaningful distinctions in eight multi-class groups rather than force one representative per group. Keep 31 representatives across the 22 groups, plus the 69 singleton classes. | **100 working classes** |
| Standalone screen and accepted cutoff | Exclude from active search candidates with mean within-type standalone R² below 0.05 for both responses and all three depth approaches. This is a practical marginal-performance filter, not proof of no conditional information. | **68 survivors; 32 archived** |
| Four-principle coverage requirement | The cutoff excludes all six pure topology candidates. Restore these six as explicit coverage reserves for BOTH constrained and unconstrained comparisons. Other weak variables remain archived. | **74 active search candidates** |
| Nested forward searches up to ten | Fixed quadratic ridge with pair interactions; three outer by three inner case-group folds; with/without ΔG; constrained/unconstrained; six method/response targets. Select smallest path model within 0.005 inner R² of best. No final universal set frozen. | **1–10 selected inputs, depending on target and constraint** |

## Why the accepted review gives 100, not 91

| Saved group ID | Retained distinction | Slots | Extra above one |
|---|---|---:|---:|
| 9 | Parent mean slope, minimum slope, inverse-root slope summary | 3 | 2 |
| 13 | Parent unweighted route CV and slope CV | 2 | 1 |
| 20 | Parent mean reach width and equivalent path width | 2 | 1 |
| 29 | Parent area-based elongation and equivalent elongation | 2 | 1 |
| 53 | Mean-route-length change and Manning geometric change | 2 | 1 |
| 59 | Change in unweighted route CV and slope CV | 2 | 1 |
| 71 | Child unweighted route CV and slope CV | 2 | 1 |
| 84 | Child harmonic width and width P10 | 2 | 1 |
| Other 14 groups | One provisional representative each | 14 | 0 |
| **Total grouped representatives** | | **31** | **9** |

The 34 other affine classes remain documented alternatives, not erased features or demonstrated failures. The 100 count is at the empirical-class level; some affine representatives still require a real-network/formula check. A class can split again if its members diverge in real networks.

## Accepted decision principles

- SI units: canonical lengths in metres, areas in square metres, slopes in m/m. Source features are preserved; this record does not silently rescale model input files. Remaining unit-labelled aliases must be resolved when building the final feature table.
- Invertible transformations do not provide independent information: prefer raw ratios to log1p versions, entropy to its effective-count transformation, and route SD in metres to its variance representation where appropriate.
- Parent and child similarity does not imply that changes are unimportant. Parent plus an explicit difference can reconstruct the child; existing change candidates remain available. This dependency must be checked again if later selection removes a difference.
- Similarity induced by shared scale, conserved head loss, limited branch counts or simple width patterns is not a universal identity. Preserve distinct slope, shape, allocation and width concepts where the accepted review identifies this concern.
- No response scores were used to make this screen. No RF was fitted and no routing run was changed.

## Remaining checks

Real-case follow-up requested by the user: evaluate predictive value of the four coverage sets using real-only fits; explicitly omit synthetic-to-real coefficient transfer. The seven-input peak-with-G set (two of three synthetic folds) was frozen before looking at real scores. Completed on all 2,185 established comparisons with 15 whole-river holdouts; fixed quadratic regression plus a fixed RF comparator, no new feature or hyperparameter search. [Full real-only report](../../RAPID/outputs/synthetic_selected_sets_20261102/report.md). All new inputs are geometry-derived; path-width sums are evaluated without enumerating enormous route populations and checked against enumeration on 429 small domains. Ordinary versus full-network cohort results must remain separate. No universal final set is accepted: ordinary-case centroid RF scores are negative, river 37 dominates squared error, and equivalent path width causes catastrophic quadratic extrapolation for river 46. The four design constants and general affine-equivalence transfer checks remain pending.

Historical update after the individual screen: all 100 classes were evaluated with signed correlations and fixed univariate spline prediction using five case-group folds. [Results and limitations](../investigations/individual_predictor_screen_20261102/results/report.md). At that stage all 100 were retained. The subsequent user-accepted practical cutoff and coverage exception produced the 74-candidate search pool recorded above. Five unresolved affine classes received temporary screen labels only; final real-network representatives remain pending. Cross-method figures use common treatment support. This is internal grid interpolation, not independent real-network validation.

Latest update: [forward-search results](../investigations/forward_selection_20261102/results/report.md). The nested scores evaluate subset/size selection conditional on the previously inspected candidate inventory. The full-population screening and correlation decisions were not repeated inside outer folds, so these are not unbiased scores for the entire discovery workflow. Topology selection may encode synthetic treatment identity. All four categories were verified in every coverage model. Numerical sufficient-statistic fits were checked against direct predictions and scikit-learn Ridge. No routing runs or production RF changes occurred.

1. Resolve affine aliases using equations, graph assumptions and real-network variation. Use canonical SI representations.
2. Check the four design constants in real cases: shortest-route-length change and ratio, head-loss change, maximum-slope change.
3. Check whether synthetic similarity classes remain similar in complex real networks, particularly multiway splits and heterogeneous slopes/widths.
4. Evaluate the working candidates and documented challengers in subsequent grouped predictive comparisons. The 100 are not a claim of optimality or mandatory inputs to one model.

## Traceable records

### 3 November 2026 — test the mean-width-ratio replacement

Reference **MEAN-WIDTH-COMPACT-2026-11-03**. The user selected the previous
nine-input panel's mean-width ratio as the next replacement to test. Both
compact five-input panels replace only log1p(parent equivalent path width) with
log1p(child/parent length-weighted arithmetic mean reach width). This evaluates
the requested width descriptor, rather than the edited-size replacement from
the one-river diagnostic. Each selected reach contributes once to the mean.

The fixed RF, four other inputs, responses, weights and folds remain frozen.
Both panels are evaluated on the full synthetic population and all three real
validation schemes (136 fits). Earlier fixed-RF predictions are verified and
reused. The real-world failure informed this candidate, so the comparison is
exploratory rather than independent confirmation of a complete discovery
pipeline. No final set is accepted automatically from its scores.

[Protocol and reproducible command](../../RAPID/investigations/mean_width_compact_20261103/README.md).
Outputs and completion receipt: `RAPID/outputs/mean_width_compact_20261103/`.

Completed outcome (136 fits): ordinary real-comparison R² for the mean-width
five with ΔG is 0.757 (compound-family), 0.706 (leave-river-out), and 0.698
(network-grouped); original-nine values under the same fixed RF are 0.644,
0.575, and 0.584. The replacement five without ΔG scores 0.689, 0.584, 0.541.
Both retain synthetic R² above 0.9999. Held-out river-37 ordinary MAE falls from
36.53 to 0.52 min for the with-ΔG set; its share of ordinary squared error falls
to roughly 3–6% across schemes. This supports the diagnosed equivalent-width
failure and a compact working candidate, not automatic final acceptance.
The revised nine remain stronger for the 15 full-network controls (R² roughly
0.42–0.47 versus 0.36–0.42 for the replacement five with ΔG). Preserve this
tradeoff and the post-hoc nature of the replacement in reporting.

### 3 November 2026 — example-37 equivalent-width diagnosis

Reference **COMPACT-37-DIAGNOSTIC-2026-11-03**. Seven post-hoc held-river-37
refits investigated the compact-set failure without changing routing or targets.
The three baseline refits reproduced stored predictions exactly. Exact RF leaf
donor reconstruction, including bootstrap multiplicities and river weights,
showed full-network training controls creating a large negative contribution
for small edits inside a very large parent domain. In 383 states the parent
equivalent path-width sum is shared, about 2.68e27 m, while median edited length
is approximately 402 m. The width sum is a combinatorial route descriptor, not
a physical channel width or local edit size.

On the same 395 ordinary example-37 cases, five-with-G MAE was 36.53 min;
removing equivalent width gave 0.49 min, replacing it with edited-domain size
gave 0.46 min, and revised-nine MAE was 0.63 min. Adding edited size while
retaining width left MAE at 22.48 min. This revises the previous interpretation:
the compact-set failure does not by itself establish that many additional
complexity descriptors are required. The equivalent-width representation is a
demonstrated source of this fitted model's error. Entropy/dominance extensions
retained that representation and did not resolve it.

These are explicitly post-hoc, single-river diagnostic results, not a new
validated final model. Full-population evaluation of width removal/replacement
remains necessary. Keep the existing populations and full-network controls;
their exclusion was only a diagnostic and degraded large-collapse predictions.

[Detailed report and exact training-case contributions](../../RAPID/outputs/compact_37_diagnostic_20261103/report.md).

### 3 November 2026 — align the hydraulic approach and prediction model

Reference **STREAMLINED-PREDICTORS-2026-11-03**. User-approved reconciliation:
fixed-depth Manning is primary and variable-depth Manning is sensitivity; use
RF in both datasets; retain entropy and dominance as justified complementary
candidates despite their low standalone synthetic scores. The nine-input set
is a reference, not a target that the synthetic search must be forced to select.

Eight sets are frozen before new fits: original nine, revised nine (width-CV
difference replaces ratio), two existing five-input sets, their seven-input
extensions with entropy and dominance, and two revised-nine removal tests
(allocation pair or normalized route-spread change). Use one existing RF
configuration for all sets and both datasets; no new tuning search. Refit the
old nine under that same configuration so comparisons are matched. Prior tuned
scores are historical context rather than the numerical baseline for this test.

The original width-CV ratio is undefined for 2,187 of 45,927 synthetic rows.
Original and revised nine use matched defined support for their direct
comparison; revised nine and compact panels additionally use all synthetic
rows. Zero CV remains meaningful; undefined ratios are not filled with zero.
Thirty direct real-domain checks verified the reconstructed width-CV differences.

[Frozen protocol and commands](../../RAPID/investigations/streamlined_predictors_20261103/README.md).
Check `RAPID/outputs/streamlined_predictors_20261103/completion.json` for final
completion. This experiment follows the existing three-strategy run using the
same two-worker limit. No production models or routing results are changed.

### 3 November 2026 — matched real-only centroid comparison

Reference **MATCHED-CENTROID-SETS-2026-11-03**. The earlier real-only test used
whole-river holdout and a fixed RF; the existing nine-input notebook used tuned,
compound-family grouped validation within known rivers. Their scores therefore
do not isolate the effect of choosing different predictors. The requested
follow-up freezes the notebook's exact 25 outer partitions, inner partitions,
six RF configurations, weighting and objective, comparing its nine inputs with
the two five-input centroid sets. All 2,185 canonical comparisons remain;
ordinary and full-network controls are reported separately. No feature search
or synthetic-to-real coefficient transfer is introduced.

An input-only precision audit found that equivalent path width, which sums
across potentially enormous path populations, becomes numerically indistinct
among ordinary cases after the historical scaler and RF float32 conversion.
Use `log1p(width in metres)` before that scaler; this is the same geometric
quantity with an order-preserving numerical representation, not an additional
predictor. The raw-width first repeat is retained as a diagnostic. This
exception to preferring raw representations is numerical, not score-driven.
Unchanged connecting reaches are excluded from full-network edge rewards to
align the previous benchmark's descriptor scope.

[Protocol and reproducible commands](../../RAPID/investigations/matched_centroid_sets_20261103/README.md).
The authoritative completion status and scores are in
`RAPID/outputs/matched_centroid_sets_20261103_logwidth/`; check `completion.json`
before treating the five-repeat comparison as complete. Neither the earlier
whole-river result nor production models are replaced by this experiment.

- [Population restart](2026-11-02_candidate_population_restart.md).
- [Similarity screen and all thresholds](../investigations/predictor_similarity_20261102/results/report.md).
- [All 22 group reviews, equations and recommendations](../investigations/predictor_similarity_20261102/results/group_review_090.md).
- [Decision ledger for all 134 classes](../investigations/predictor_similarity_20261102/results/accepted_class_decisions.csv).
- [100 working classes](../investigations/predictor_similarity_20261102/results/working_100_classes.csv).
- Reproducible ledger builder: `investigations/predictor_similarity_20261102/record_decisions.py`.


## 2026-11-03 — RF reselection without summed equivalent path width

Reference: SYN-RF-RESELECTION-2026-11-03. User authorized a new synthetic compact search to test whether the mean-width alternative emerges automatically. Restore the full 100 complete working classes (no standalone-response cutoff), remove four equivalent-width/summed-width elongation representations, and search the remaining 96 candidates with ΔGₘ eligible or 95 without it. Fixed-depth Manning centroid response, matched fixed RF, nested case-group folds, four-principle coverage and the existing 0.005 compactness tolerance. Mean width is not compulsory. This is post-hoc exploratory redesign, not independent rediscovery. See [frozen protocol and completion files](../investigations/rf_reselection_20261103/README.md); do not infer completion from this decision entry.


## 2026-11-03 — Return to matched original selection procedure

User questioned the expanded RF runtime and chose the original procedure first. Stopped the superseded RF search, retaining its checkpoints; no RF selection is complete. Repeated the original 74-candidate quadratic-ridge procedure with only the four structurally problematic width/elongation entries removed (70 remain; 69 without G). All methods, responses, folds and search settings unchanged. Completed in 22.5 seconds; identical folds and numerical checks verified. Fixed-Manning centroid selects child-width P90 in all folds, not mean-width ratio; four-input coverage score 0.995316. See [matched rerun and results](../investigations/forward_selection_width_revision_20261103/README.md). Do not present this as automatic recovery of the mean-width RF model.


## 2026-11-03 — Isolated P90 versus mean-width comparison

Reference WIDTH-P90-COMPACT-2026-11-03. User authorized replacing only the width input in the two existing five-input fixed RF panels, preserving the other four predictors, folds and populations. Compare child length-weighted width P90 with child/parent length-weighted mean width, both with Manning G and with area-change alternative. No feature search or routing reruns. Reuse completed mean-width predictions and perform 136 new fits. Full-network P90 uses the edited child footprint, matching existing mean-width scope. See [protocol](../../RAPID/investigations/width_p90_compact_20261103/README.md); final completion and results are recorded in the output folder.

Completed WIDTH-P90-COMPACT-2026-11-03: all 136 fits in 259 seconds. Ordinary real R² with P90+G: 0.7638 compound-family, 0.7146 held-out-network, 0.7065 network-grouped; mean-width+G: 0.7573/0.7062/0.6983. Small average P90 advantage, not uniform by river/repeat. Without G results mixed; full-network controls slightly favor mean width. Synthetic performance essentially tied. Keep the interpretation as a matched width-representation sensitivity test, not validation of the complete new four-input ridge panel or a claim of statistical superiority. Full report: `RAPID/outputs/width_p90_compact_20261103/report.md`.


## 2026-11-03 — Exclude P90 for interpretability

Reference SYN-FORWARD-NO-P90-2026-11-03. User prefers to exclude width P90 despite competitive predictive results. Repeat the original ridge procedure after excluding every available P90 representation, not only the child value. Parent, child and delta P90 occur in the previous 70-candidate pool; ratio P90 was already ineligible. This leaves 67 candidates (66 without G). Previous equivalent-width/elongation exclusions retained. P10 remains eligible. No other selection changes, RF refits or routing runs. See [protocol and results](../investigations/forward_selection_no_p90_20261103/README.md). Interpretability-driven post-hoc sensitivity, not a claim that P90 carries no useful information.

Completed SYN-FORWARD-NO-P90-2026-11-03 in 19.8 seconds; folds/numerical checks passed. Fixed-Manning centroid coverage selects the same four inputs in every fold, with or without G eligible: delta allocated mean route length, parent slope descriptor, absolute parent length-weighted mean width, parent bifurcation count. R² 0.995180 versus 0.995316 with P90; MAE 0.994839 versus 0.923656 min. Parent mean width is NOT the previously RF-tested child/parent width ratio. No new real RF performance is claimed.


## 2026-11-03 — Real RF test of the exact no-P90 four

Reference NO-P90-FOUR-2026-11-03. User requested real RF performance of the newly selected four. Completed 65 fixed RF fits in 92.6 seconds with identical saved folds, all 2,185 rows, and recomputed parent mean width/bifurcation count on matched domains. Ordinary R² 0.5361 compound-family, 0.5593 leave-one-network-out, 0.5459 network-grouped; MAE 0.8383/0.7583/0.8768 min. Lower than five with mean-width ratio plus G (0.7573/0.7062/0.6983); full-network controls R² 0.2827/0.3394/0.3619. This changes more than width (four inputs, no G/area, different topology representation); do not attribute the decline to P90 exclusion alone. No additional predictor searches or four-plus-G fits performed. [Protocol/results](../../RAPID/investigations/no_p90_four_20261103/README.md).


## 2026-11-03 — Isolated addition of G to the no-P90 four

Reference NO-P90-FOUR-PLUS-G-2026-11-03. Completed 65 real RF fits in 89.9 seconds; identical input snapshot and partitions, only delta_G_manning added. Ordinary R² improves 0.5361→0.6034, 0.5593→0.5884, 0.5459→0.5635 across compound-family, leave-one-network-out and network-grouped validation. MAE improves to 0.6610/0.6650/0.7794 min. Still below previous mean-width-ratio/complexity-change/G panel R² 0.7573/0.7062/0.6983. Remaining differences are width and topology representations; no attribution between them established. Full-network controls remain much harder. [Protocol/results](../../RAPID/investigations/no_p90_four_plus_g_20261103/README.md).


## 2026-11-03 — Quadratic ridge versus RF for the successful mean-width/G five

Reference MEAN-WIDTH-RIDGE-VS-RF-2026-11-03. User requested comparison of the successful compact five after width replacement. Fit 65 quadratic ridges using the same five engineered RF inputs (including log1p mean-width ratio), real training/test folds and equal-river weights; retain original synthetic ridge penalty 1e-4 per training row, no tuning. Independent normal-equation checks passed in every fold. Ordinary ridge R² 0.7299/0.5464/0.4960 versus RF 0.7573/0.7062/0.6983. Conversely full-network control ridge R² 0.8374/0.8571/0.7915 versus RF 0.3630/0.4142/0.4175. Do not hide this cohort difference in pooled scores. Not synthetic-to-real coefficient transfer, not five-coefficient linear regression, and not independently uninspected validation. [Protocol/results](../../RAPID/investigations/mean_width_ridge_vs_rf_20261103/README.md).


## 2026-11-04 — Main-model finalization plan and no-G definitions

Reference MODEL-FINALIZATION-2026-11-04. Recorded outstanding steps, quadratic-ridge explanation and code-checked equations in [the finalization note](2026-11-04_quadratic_ridge_finalization_plan.md). User prefers considering the no-G five; this replaces G with area-proxy change, not simply removal. Existing RF ordinary R² decreases from 0.757/0.706/0.698 with G to 0.689/0.584/0.541 without, so performance equivalence is not established. Matched quadratic-ridge with-G versus area comparison remains pending, followed by predictor freeze, bounded inner-fold penalty sensitivity, all three validation strategies, separated cohorts and residual/extrapolation checks. No new fits or final model choice made in this documentation step.


## 2026-11-04 — Replace complexity change with entropy, no-G five

Reference ENTROPY-REPLACEMENT-2026-11-04. User authorized replacement; retain area as fifth input and compare both RF and quadratic ridge. Completed 195 fits in 110.7 seconds with identical rows, weights and partitions; 130 ridge numerical checks passed. Ordinary entropy ridge R² 0.7260/0.4670/-0.2220 versus complexity ridge 0.7004/0.2866/-0.3906; entropy RF 0.7243/0.6153/0.5758 versus complexity RF 0.6892/0.5844/0.5414. Entropy improves mean ordinary metrics, but no-G ridge remains unstable across network-held-out repeats, with large full-network errors. This also completes the previously missing no-G complexity ridge baseline: performance equivalence to the G version is not supported. No penalty tuning, new feature search or entropy+G run performed. [Protocol/results](../../RAPID/investigations/entropy_replacement_20261104/README.md).


## 2026-11-04 — Entropy + G quadratic ridge candidate

Reference ENTROPY-G-RIDGE-2026-11-04. User authorized the five-input candidate: mean route-length change, parent slope, log1p mean-width ratio, entropy change, G change. Completed all 65 fixed-penalty real ridge fits in 1.7 seconds, with independent normal-equation checks and exact repeated row/target alignment to baselines. Ordinary R² 0.7732/0.6243/0.5651 improves versus complexity+G ridge 0.7299/0.5464/0.4960 in every paired repeat. Full-network control R² drops to 0.5260/0.5072/0.4346 versus 0.8374/0.8571/0.7915. Ordinary MAE 0.8000/0.9239/0.9613 min remains higher than the existing complexity+G RF. This is a tradeoff, not a uniformly best model. Penalty tuning and residual checks remain pending; no entropy+G RF fitted. [Protocol/results](../../RAPID/investigations/entropy_g_ridge_20261104/README.md).


## 2026-11-04 — Six-input ridge retaining entropy and complexity

Reference ENTROPY-COMPLEXITY-G-RIDGE-2026-11-04. User requested retaining both topology/allocation descriptors with G. Completed 65 fixed-penalty real fits in 1.8 seconds; all numerical checks, row/target alignment and source hashes passed. Six inputs expand to 27 terms plus intercept. Ordinary R² 0.8157/0.6878/0.6737 improves over both five-input ridge panels in every matched repeat. Full-network control R² 0.7019/0.7451/0.7665 recovers much of the entropy-only loss but remains below complexity-only ridge 0.8374/0.8571/0.7915. Ordinary MAE 0.7687/0.8770/0.8746 remains higher than prior RF. Stronger compromise, not uniformly best or independently validated. Ordinary/full-network distinction is not proof of a size mechanism. [Protocol/results](../../RAPID/investigations/entropy_complexity_g_ridge_20261104/README.md).


## 2026-11-04 — User accepts six-input ridge for finalization

Reference SIX-INPUT-RIDGE-DECISION-2026-11-04. User explicitly requested carrying the six forward and recording the rationale. [Decision, reasoning, limitations and remaining steps](2026-11-04_six_input_ridge_decision.md). Inputs/model family are the working choice; penalty tuning and residual/extrapolation checks remain pending. Broader predictor search is not reopened. KGE added as supplementary scoring of saved held-out centroid shifts, not gauge-hydrograph validation: ordinary equal-river KGE 0.8976/0.8142/0.8033; full-network 0.6941/0.7425/0.7484. Equal-pair ordinary KGE 0.8528/0.6610/0.6822 demonstrates weighting sensitivity. Signed near-zero mean can destabilize KGE; keep components and R²/MAE primary. No models refitted for scoring.


## 2026-11-04 — Exact six-input area versus G test

Reference SIX-AREA-VS-G-2026-11-04. User authorized replacing only G with area and asked about reasons against G. All 65 fits completed and numerical/baseline metric checks passed. Headline combined equal-pair R²: G 0.8666/0.8571/0.8627; area 0.4178/0.6117/-1.0093. Area also loses KGE/MAE under every strategy. Existing acceptance of six with G remains supported; no further candidate search triggered. Reasons to consider alternatives are interpretability, hydraulic/allocation assumptions and slope uncertainty, not depth or Manning n as input variables. No claim that the current fixed-penalty comparison rules out every possible area model. [Full record](../../RAPID/investigations/six_area_vs_g_20261104/README.md).


## 6 November 2026 — consolidated settings and retained secondary model

Reference: **GEOMETRY-PREDICTOR-SPEC-2026-11-06**. At the user's request, the accepted settings, equations, numerical values, selection rationale and outstanding checks are now consolidated in [Current geometry-based prediction models](../../RAPID/CURRENT_GEOMETRY_PREDICTION_MODELS.md), with a [machine-readable manifest](../../RAPID/config/geometry_prediction_models.json).

Carry forward the six-input quadratic ridge with ΔGₘ as primary. Retain the matched **six-input area-change replacement** as the secondary model without ΔGₘ; this is not the five-input model obtained by simply dropping G. Both use the existing completed evaluations and are selectable in `hydrography_rf_prediction_vs_modelled.ipynb` via `RIDGE_MODEL`. No new fits or coefficient changes were made for this documentation update.

The fixed ridge penalty is inherited, not demonstrated optimal. Grouped penalty sensitivity, residual/extrapolation review and the final full-data coefficient artifact remain pending. Retain the weaker secondary results when completing those checks.
