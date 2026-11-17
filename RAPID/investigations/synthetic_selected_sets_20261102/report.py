from pathlib import Path
import pandas as pd
import json
ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'RAPID/outputs/synthetic_selected_sets_20261102'
s=pd.read_csv(OUT/'real_only_scores.csv')
design=json.loads((OUT/'evaluation_design.json').read_text())
panels=list(design['panels'])
lines=['# Predictive value of four synthetic-selected sets on real networks','',
'Reference: REAL-SELECTED-SETS-PREDICTION-2026-11-02. Real-only fitting: no transfer of synthetic coefficients, no new routing runs, no production model changes.','',
'## Main conclusion','',
'The sets have some predictive value, but synthetic performance does not establish robust real-network performance. The fixed-RF combined cohort-balanced centroid R² is about 0.62–0.63, yet ordinary local-collapse centroid R² is negative. The large full-network collapses and between-cohort contrast make the combined number misleading if presented alone. River 37 dominates ordinary-case centroid squared error. Peak prediction is modestly useful in ordinary cases and weak in full-network collapse controls. Manning ΔG gives a small centroid improvement in this fixed test; the peak sets differ in several inputs, so their comparison is not a pure ΔG ablation.','',
'## Data and frozen sets','',
'2,185 canonical comparisons across 15 rivers: 2,170 original unit/island/compound comparisons and 15 segmented all-closed-unit collapses. This is the established completed predictor population; no additional cases were excluded from it. Example 11 is not in that existing population. The current completed fixed-Manning midpoint run supplies both responses, in the existing 0.5% finite-event windows. Targets are RAPID child-minus-parent centroid and peak shifts in minutes, not errors against observed gauge peaks.','',
'Centroid sets each contain five inputs; peak with ΔG uses the seven-input set selected in two synthetic folds, fixed before real scores were inspected. Peak without ΔG uses the stable eight-input set.','']
for panel,cols in design['panels'].items():
    lines += [f'### {panel}', '']+['- `'+c+'`' for c in cols]+['']
lines += ['## Domain and feature checks','',
'Original cases use the current routed parent B domain and the complete child domain between the same anchors. The fifteen full-network segmented controls use full closed physical networks, including unchanged connectors; this follows the synthetic domain formulas and differs from historical edited-only descriptions for those fifteen cases. Whole-network ΔG_M is reused only after verifying that all 2,200 source parameter files match the hashes behind its calculation. Original-case mean-distance change and inverse-root slope match the existing real feature table numerically. Both response columns match the current fixed-Manning analysis.','',
'Equivalent path width is the literal sum W_eq=sum_r L_r/sum_(e in r)(L_e/B_e). It is not physical cross-sectional width. Explicit enumeration is infeasible for large domains: the largest parent route count is about 1.46e44. We evaluate the same sum via a uniform-route dynamic-programming Laplace integral. Quadrature converges to a relative tolerance of 1e-9 (adaptive fallback tolerance 1e-7); 429 small domains match direct enumeration to maximum relative error 2.28e-14. There are 450 unique parent domains. No route sampling, clipping or alternative width definition is used.','',
'The largest W_eq is approximately 4.76e46 m (full-network case); original domains reach 2.68e27 m. These are mathematical route-sum values, not plausible physical river widths. Shared reaches are counted through many complete routes. This is a substantial descriptor-transferability concern, not a failed unit conversion.','',
'## Evaluation','',
'Each set is fitted only on real cases with one entire river held out: 15 outer fits per set per model, all related cases of the held river excluded from training. No new feature selection or hyperparameter tuning is performed. Training assigns equal total weight to each river. All four sets use identical holdouts. Previously studied rivers and frozen exploratory feature choices mean this is internal grouped evaluation, not independent external validation.','',
'Two fixed model families are reported. Quadratic ridge uses the same main/squared/pair-interaction formulation and penalty rule as the synthetic search. A fixed RF comparator uses 200 trees, maximum depth 12, minimum leaf size 3, squared-error criterion, all input features and seed 20261102. RF inputs receive training-scale signed log1p compression, which preserves ordering while avoiding float32 overflow and loss of resolution from extreme route-sum values; the response is not transformed. These RF settings are not the historically tuned nine-input model settings, so this is not a paired superiority test against that model.','',
'The combined equal-cohort-network metric assigns half the total weight to each cohort, then equal weight to rivers within each cohort. It therefore gives 15 full-network controls half the total weight. Equal-network and unweighted all-population scores are exported too. Cohort-specific scores must accompany the combined metric.','',
'## Fixed RF results','',
'| Set | Combined cohort-balanced R² | Original R² | Original MAE, min | Full-network R² | Full-network MAE, min |',
'|---|---:|---:|---:|---:|---:|']
for panel in panels:
    z=s[(s.panel==panel)&(s.model=='RF_fixed')]
    combined=z[(z.cohort=='all')&(z.weighting=='equal_cohort_network')].iloc[0]
    original=z[(z.cohort=='original')&(z.weighting=='equal_network')].iloc[0]
    full=z[(z.cohort=='segmented')&(z.weighting=='equal_network')].iloc[0]
    lines.append(f'| {panel} | {combined.r2:.3f} | {original.r2:.3f} | {original.mae_minutes:.3f} | {full.r2:.3f} | {full.mae_minutes:.3f} |')
lines += ['', 'Negative R² means worse squared error than the corresponding evaluation-population mean reference, despite potentially modest typical absolute errors. Scores are not directly comparable to earlier compound-mixed validation, which predicts related states within known rivers.', '',
'## Quadratic-model failure','',
'The centroid quadratic models extrapolate catastrophically for held-out river 46, associated with its extreme equivalent-path-width value. Predictions for its full-network collapse are about 1.4e18 minutes, yielding combined R² around -1e31. Those are unusable model predictions, not physical timing results. They are retained in the output without clipping. The peak quadratic model with ΔG has combined R² about 0.313; without ΔG it is about -6.35. A strong synthetic quadratic fit therefore does not establish a suitable real-network model family.','',
'## Next interpretation','',
'Do not freeze these sets as final predictors based on the combined score. Inspect river-37 errors, separate ordinary and full-network collapse behaviour, and revisit equivalent path width before optimising any model. A controlled follow-up could compare a scalable width measure or a training-only transform, but neither is silently substituted here. RF retuning or a different validation population would be a new experiment.','',
'## Reproducibility','',
'Scripts: prepare.py, evaluate.py, report.py in RAPID/investigations/synthetic_selected_sets_20261102. Outputs include input.parquet, preparation.json with source hashes, geometry_validation.json, equivalent_width_audit.csv, evaluation_design.json, real_only_predictions.parquet, real_only_scores.csv, real_only_fold_scores.csv, held_river_feature_ranges.csv, largest_real_errors.csv, real_only_RF_predictions.png and evaluation_completion.json. No synthetic coefficients enter any fit.']
(OUT/'report.md').write_text('\n'.join(lines)+'\n')
assert len(s)==108
pred=pd.read_parquet(OUT/'real_only_predictions.parquet')
assert len(pred)==26220 and not pred.duplicated(['panel','model','example','state']).any()
assert pred.groupby(['panel','model']).example.nunique().eq(15).all()
(OUT/'validation.json').write_text(json.dumps(dict(cases_per_model=2185,panels=4,models_including_baseline=3,whole_rivers=15,
    predictions_unique=True,features_finite=True,synthetic_coefficients_used=False),indent=2))
print(OUT/'report.md')
