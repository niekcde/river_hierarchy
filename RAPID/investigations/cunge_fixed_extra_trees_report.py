"""Readable results of the approved three-step Extra Trees comparison."""
from pathlib import Path
import json
import pandas as pd,numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'RAPID/outputs/cunge_fixed_extra_trees_20261006'
s=pd.read_csv(OUT/'scores.csv',dtype={'scope':str});h=pd.read_csv(OUT/'scores_in_hours.csv',dtype={'scope':str});fold=pd.read_csv(OUT/'fold_audit.csv',dtype={'held_out':str});lags=pd.read_csv(OUT/'parent_lags_and_n.csv',dtype={'example':str});m=json.loads((OUT/'provenance.json').read_text());p=pd.read_parquet(OUT/'held_out_predictions.parquet')
absolute=s[(s.scope=='pooled')&(s.response=='shift_h')].copy();absolute['mae_minutes']=60*absolute.mae
relative=s[(s.scope=='pooled')&(s.response=='relative_shift_pct')].copy()
rescaled=h[(h.scope=='pooled')&(h.trained_response=='relative_shift_pct')].copy();rescaled['mae_minutes']=60*rescaled.mae
summary=absolute[['routing','features','r2','mae_minutes','skill_vs_zero']].rename(columns={'r2':'direct_hours_r2','mae_minutes':'direct_hours_mae_minutes','skill_vs_zero':'direct_hours_skill_vs_zero'}).merge(relative[['routing','features','r2','mae','skill_vs_zero']].rename(columns={'r2':'relative_r2','mae':'relative_mae_percentage_points','skill_vs_zero':'relative_skill_vs_zero'}),on=['routing','features'],validate='one_to_one').merge(rescaled[['routing','features','r2','mae_minutes','skill_vs_zero']].rename(columns={'r2':'relative_rescaled_hours_r2','mae_minutes':'relative_rescaled_hours_mae_minutes','skill_vs_zero':'relative_rescaled_hours_skill_vs_zero'}),on=['routing','features'],validate='one_to_one')
summary.to_csv(OUT/'comparison_summary.csv',index=False)
text=f'''EXTRA TREES, MANNING n AND TIMING SCALE — 6 October 2026

Main findings
The previous 15-feature Extra Trees model gives a more favourable Cunge result than the small six-feature boosted benchmark when the response is expressed relative to the parent lag. The geometry relationships have not disappeared. Direct prediction of absolute hours remains weak across the four networks. Adding supplied calibrated Manning n does not fix that weakness in this test.

For Cunge, geometry alone:
- Direct hours: R2 -0.122; MAE 13.58 min, worse than predicting zero shift (12.22 min).
- Relative change: R2 0.410; MAE 0.369 percentage points, 35.1% lower MAE than zero-change prediction. Per-network R2 is positive for all four networks, but network 02's MAE remains slightly worse than the zero baseline.
- Relative prediction converted back to hours using the known parent lag: R2 0.503; MAE 7.06 min, versus 13.58 min for direct-hour training. This reduces MAE by about 48% and beats zero-change MAE by about 42%.
With n supplied, direct-hour MAE worsens to 15.35 min. Relative MAE is virtually unchanged at 0.370 percentage points; after conversion to hours, MAE is 7.18 min. This is no consistent improvement from the extra feature.

Interpretation
These results support accounting for the parent network's timing scale. They do NOT show that Manning n alone caused the earlier prediction problem. The parent lag incorporates geometry, roughness, routing/storage behaviour and the forcing event; dividing by it removes more than n. Calibration values vary only between four networks (three in each training fold), so n is partly confounded with network identity. Cunge networks 02 and 21 have held-out n values outside the training range. An Extra Trees model has no imposed Manning scaling law and is poorly suited to establishing roughness extrapolation from so few groups. Failure of this n feature is not evidence that n is physically unimportant or can be recovered from geometry.

The most promising tested approach is to predict relative changes from geometry and recover hours from an available parent lag. The parent lag is obtained without using the held-out child's outcome, so this transformation does not leak the child target. It does require a routed/otherwise independently available parent lag. Because these are calibrated parent runs, this remains conditional prediction, not untouched ungauged or independent-event validation. It does not remove the need for base-network calibration or resolve case 11.

Scope and controls
Same frozen snapshot: 156 canonical physical parent-child pairs in 02,03,05,21; no duplicate aliases and no missing response or feature exclusions. All stored routing and calibrations are reused; no routing or calibration was performed this turn. Completed networks outside the snapshot were not added. The current calibration-target proposals remain unimplemented.

Four configurations:
fixed_archived: previous fixed X=0.1 workflow, n=0.035, previous windows/history and grid.
fixed_calibrated: current event/grid, X=0.1, its previously fitted n.
fixed_same_n: current event/grid, X=0.1, Cunge's n; all physical c/K inputs match Cunge.
cunge: current calibrated constant-reference Cunge.
Only fixed_same_n versus Cunge isolates X. The archived comparison includes other workflow changes described in the preceding comparison report. Old 2020 timestamps were an artificial time origin, not evidence of a different gauge event year.

Approved step 1: reproduce the prior strongest fixed configuration
15-feature log_geometry Extra Trees with the exact previous settings: 96 estimators, min_samples_leaf=5, max_features=1, random_state=20261005, n_jobs=1. Full parameter dictionary is in provenance.json.
Features:
{chr(10).join('- '+c for c in m['geometry_features'])}
No hyperparameter selection or new feature search. Same four leave-one-network-out folds for all methods and responses. Every test river's rows are absent from training. This fixed configuration was previously selected in the study; the four-network exercise is exploratory, not pristine external validation. Its scores are not directly comparable to the previous R2 around 0.57 obtained with 16 networks and 2214 pairs.

Approved step 2: add calibrated n
The sixteenth candidate feature is the supplied n of each routing configuration, constant across parent and children within a network. It is not inferred from geometry. For archived fixed-X, n is constant everywhere, so the feature is omitted to avoid a purely random-tree difference from an uninformative column; predictions with and without n are verified exactly equal.

Approved step 3: absolute and relative timing
Signed absolute target: child interpolated peak time minus parent interpolated peak time, in hours. Negative means an earlier child peak.
Relative target: 100 * absolute shift / parent lag.
Parent lag: interpolated parent outlet peak time minus first sampled inlet maximum, using the same convention as the archived analysis. Parent and child share the forcing. Three-point interpolation is reused from the established analysis and does not imply new observational precision.
For the current runs, denominators were calculated from saved control parent hydrographs. For archived results, the exact existing denominators were recovered from their validated absolute/relative responses and checked for consistency across all pairs in each river. Every denominator is positive.
A secondary check converts held-out relative predictions back to hours: predicted_hours = predicted_relative_pct * known_parent_lag_hours / 100. No observed child shift enters that conversion.

Parent lags and n
{lags.to_string(index=False)}

Pooled held-out comparison
All MAE-hours fields below are expressed in MINUTES. Relative MAE is in percentage points, not percent error relative to an individual near-zero shift.
{summary.to_string(index=False)}

Per-network Cunge results
{s[(s.routing=='cunge')&(s.scope!='pooled')].to_string(index=False)}

Reporting limits
Scores include zero-change and training-mean baselines, pooled R2/MAE, per-network scores and equal-network MAE. Do not compare an R2 for hours directly with an R2 for percentages: normalisation changes between-network weighting. The back-converted-hour comparison evaluates both approaches on the same absolute target and is the relevant direct comparison. Per-network R2 for relative and back-converted predictions is identical because each network has one positive fixed denominator.
No p-values or row-wise confidence intervals are presented: there are four independent networks, not 156 independent roughness settings. Apparent gains require checking on the remaining completed networks with the same frozen model and definitions. Case 02 remains a weaker MAE result even under relative training.

Checks
64 forests fitted (4 routing configurations x 2 targets x 2 feature sets x 4 folds). All inputs/targets finite, exact pair identity and train/test river separation verified, all source hashes unchanged, archived constant-n placebo exactly identical. Routing/calibration targets, workbook, production inputs and active jobs unchanged.

Artifacts
comparison_summary.csv: compact pooled table.
scores.csv and equal_network_scores.csv: original-unit scores and baselines.
scores_in_hours.csv: both training targets evaluated on signed absolute hours.
held_out_predictions.parquet: every held-out prediction and parent lag.
matched_features_responses.parquet: the exact modelling table.
fold_audit.csv: training rivers, test rivers, n ranges and fold scores.
parent_lags_and_n.csv and provenance.json: denominator and input audit.
comparison.html: interactive plots of the Cunge target-scale and n comparisons.
'''
(OUT/'REPORT.txt').write_text(text)
labels=[('shift_h','geometry_15','Direct hours · geometry'),('shift_h','geometry_15_plus_n','Direct hours · geometry + n'),('relative_shift_pct','geometry_15','Relative → hours · geometry'),('relative_shift_pct','geometry_15_plus_n','Relative → hours · geometry + n')]
fig=make_subplots(rows=2,cols=2,subplot_titles=[x[2] for x in labels],vertical_spacing=.16,horizontal_spacing=.10)
colors={'02':'#0072B2','03':'#D55E00','05':'#009E73','21':'#CC79A7'}
z=p[p.routing=='cunge'];bounds=[min(z.observed_shift_h.min(),z.predicted_shift_h.min())*60,max(z.observed_shift_h.max(),z.predicted_shift_h.max())*60];pad=.04*(bounds[1]-bounds[0]);bounds=[bounds[0]-pad,bounds[1]+pad]
for i,(response,features,title) in enumerate(labels):
 row,col=i//2+1,i%2+1;g=z[(z.response==response)&(z.features==features)]
 for ex,v in g.groupby('example_id'):
  fig.add_trace(go.Scatter(x=v.observed_shift_h*60,y=v.predicted_shift_h*60,mode='markers',name='Network '+ex,legendgroup=ex,showlegend=i==0,text=v.state_id,marker=dict(color=colors[ex],size=7,opacity=.7),hovertemplate='%{text}<br>Routed %{x:.3f} min<br>Predicted %{y:.3f} min<extra>'+ex+'</extra>'),row=row,col=col)
 fig.add_trace(go.Scatter(x=bounds,y=bounds,mode='lines',line=dict(color='grey',dash='dash'),showlegend=False,hoverinfo='skip'),row=row,col=col)
 fig.update_xaxes(title_text='Routed child − parent peak time (min)',range=bounds,row=row,col=col);fig.update_yaxes(title_text='Held-out prediction (min)',range=bounds,row=row,col=col)
fig.update_layout(template='plotly_white',width=1250,height=1000,margin=dict(t=110,b=115,l=90,r=45),title='Cunge · previous Extra Trees model · 156 pairs / 4 held-out networks',legend=dict(orientation='h',y=-.10,yanchor='top'))
html='<html><head><meta charset="utf-8"><title>Extra Trees: n and timing scale</title></head><body style="font-family:Arial"><p>Same axes in every panel. Negative means an earlier child peak. Bottom panels predict relative change, then use the known routed parent lag to recover minutes. No child outcome is used in that conversion. Four-network exploratory evaluation; calibrations unchanged.</p>'+fig.to_html(full_html=False,include_plotlyjs=True)+'</body></html>'
(OUT/'comparison.html').write_text(html)
print(summary.to_string(index=False))
if __name__=='__main__':print('Report and plot complete.')
