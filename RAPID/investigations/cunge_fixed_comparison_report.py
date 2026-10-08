"""Summarise frozen, matched four-network comparison."""
from pathlib import Path
import json
import numpy as np,pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'RAPID/outputs/cunge_fixed_comparison_20261006'
d=pd.read_parquet(OUT/'paired_results.parquet');scores=pd.read_csv(OUT/'prediction_scores.csv',dtype={'example':str});corr=pd.read_csv(OUT/'correlations.csv',dtype={'example':str});forcing=pd.read_csv(OUT/'forcing_comparison.csv',dtype={'example':str});audit=pd.read_csv(OUT/'routing_audit.csv');coverage=pd.read_csv(OUT/'response_coverage.csv');prov=json.loads((OUT/'provenance.json').read_text())
assert len(d)==156*4 and not d.duplicated(['example_id','fingerprint','method']).any()
labels={'fixed_archived':'Previous fixed X=0.1','fixed_calibrated':'Current-event fixed X, calibrated n','fixed_same_n':'Current-event fixed X, Cunge n','cunge':'Cunge, calibrated n'}
summary=[]
for (ex,method),g in d.groupby(['example_id','method']):
 y=g.shift_h;summary.append(dict(example=ex,method=method,n=len(g),mean_abs_shift_minutes=abs(y).mean()*60,median_abs_shift_minutes=abs(y).median()*60,max_abs_shift_minutes=abs(y).max()*60,min_signed_shift_minutes=y.min()*60,max_signed_shift_minutes=y.max()*60))
summary=pd.DataFrame(summary);summary.to_csv(OUT/'absolute_shift_summary.csv',index=False)
pivot=d.pivot(index=['example_id','fingerprint'],columns='method',values='shift_h');comparisons=[]
for method in ['fixed_archived','fixed_calibrated','fixed_same_n']:
 for ex,g in [('pooled',pivot)]+list(pivot.groupby(level='example_id')):
  a=g[method];b=g.cunge;steps=np.array([max(forcing.set_index('example').loc[e,'old_reporting_step_s'],forcing.set_index('example').loc[e,'new_reporting_step_s'])/3600 for e in g.index.get_level_values(0)])
  resolved=(abs(a)>steps)&(abs(b)>steps);reverse=a*b<0
  comparisons.append(dict(example=ex,versus=method,n=len(g),spearman=a.corr(b,method='spearman'),mean_abs_between_method_difference_minutes=abs(a-b).mean()*60,max_abs_between_method_difference_minutes=abs(a-b).max()*60,raw_sign_reversals=int(reverse.sum()),both_exceed_one_reporting_interval=int(resolved.sum()),resolved_sign_reversals=int((reverse&resolved).sum())))
comparisons=pd.DataFrame(comparisons);comparisons.to_csv(OUT/'between_method_comparison.csv',index=False)
cols=['example_id','state_id','fingerprint','plot_category','method','shift_h','sampled_shift_h','peak_excess_change_pct','event_duration_change_pct']
d[cols].to_csv(OUT/'matched_pair_responses.csv',index=False)
# Equal-network rather than pair-weighted MAE complements pooled R2.
macro=scores[scores.example.ne('pooled')].groupby(['method','response','feature_set','model']).agg(equal_network_mae=('mae','mean'),equal_network_zero_mae=('zero_mae','mean'),equal_network_mean_baseline_mae=('mean_baseline_mae','mean')).reset_index();macro['skill_vs_zero']=1-macro.equal_network_mae/macro.equal_network_zero_mae;macro.to_csv(OUT/'equal_network_prediction_scores.csv',index=False)
main=scores[(scores.example=='pooled')&(scores.response=='shift_h')&scores.feature_set.isin(['omega','components','selected_six'])][['method','feature_set','model','n','r2','mae','skill_vs_zero','mean_baseline_mae']].copy();main['mae_minutes']=main.pop('mae')*60;main['mean_baseline_mae_minutes']=main.pop('mean_baseline_mae')*60
ct=corr[(corr.example=='pooled')&(corr.response=='shift_h')].pivot(index='feature',columns='method',values='spearman')
text=f'''FIXED-X / CUNGE COMPARISON — SNAPSHOT 6 OCTOBER 2026

Scope
156 distinct physical parent-child pairs: 02 (35), 03 (40), 05 (51), 21 (30).
The set was frozen at investigation start. Running or subsequently completed networks are not silently incorporated. Matches use physical fingerprints against the canonical 2214-pair feature table; all 156 eligible pairs match, no alias duplication, no feature-based exclusions. All parents here are S000_base. No calibration targets, n calibrations, production inputs or running jobs were changed.

Calibration ideas are recorded, pending review, in RAPID/reviews/calibration_target_proposals_20261006.txt.

Configurations
- fixed_archived: previous full-network X=0.1 results, uniform n=0.035, previous forcing windows/initial histories and segmentation. Responses/features taken from the previous canonical analysis.
- fixed_calibrated: X=0.1, current forcing and Cunge numerical grid, using the already fitted fixed-X n from the 16-case calibration. No new calibration.
- fixed_same_n: identical to Cunge in geometry, Q allocation, celerity/K, grid, forcing and n; replace X with 0.1. This is the controlled X-only comparison, not a calibrated fixed-X model.
- cunge: existing calibrated Cunge settings rerouted at the common control timestep, crosschecked against saved full-engine outlet results.

The current-event controls use the existing independent DAG recurrence, not new all-reach sparse-engine runs. Every current-event method/state uses one common timestep per network satisfying the strictest upper bound and every native discharge is audited. All {len(audit)} routed configurations settle and are nonnegative (minimum {audit.minimum_native_flow_cms.min():.6g} m3/s). Cunge common-clock versus archived full-engine maximum outlet difference: {audit.max_clock_change_cms.max():.6g} m3/s; maximum saved peak-bin shift: {audit.saved_peak_shift_s.abs().max():.6g} s. Source completion hashes and source feature hash are recorded in provenance.json.

Forcing and calibration differences
{forcing.to_string(index=False)}

IMPORTANT: the old January 2020 timestamps are an artificial origin assigned by run_hydraulic_policy_refresh.py when reusing saved actual event data. They are not proof of a different observed event. The archived and current waveforms have identical peaks here. Their overlapping, peak-aligned sample arrays coincide for 03/05/21, while 02 has changed values within the compared overlap; windows, initial histories and tails differ. The peak-aligned diagnostic is for provenance only; responses are never time-aligned to improve model fit. Old-versus-new differences conflate n, windows/history and numerical grid changes with routing changes.

Response definitions
Primary timing = signed child-minus-parent peak time, in HOURS, estimated with the same local three-point interpolation as the old analysis. Negative means earlier child peak. "Absolute shifts" means hours rather than percentages; absolute magnitudes are separately summarised. Sampled peak-bin differences are retained as a resolution check. Interpolation is not additional observational precision.
Secondary responses = peak-excess percentage change and full event-duration percentage change. Peak excess uses the parent maximum of initial/final plateau as reference. Duration uses parent-derived 1% departure/return thresholds, includes secondary pulses and requires final settlement, matching the previous analysis. These are model response metrics, not gauge predictive skill.
{coverage.to_string(index=False)}

Absolute timing changes by network
{summary.to_string(index=False)}

Cunge versus each comparator, same physical pairs
{comparisons.to_string(index=False)}

A raw sign reversal close to zero is not robust evidence of changed direction. The conservative resolved-reversal count requires BOTH shifts to exceed a full reporting interval (the larger archived/current interval). It is a sensitivity screen, not a statistical confidence interval.

Predictive capability: common design
Identical geometry predictors and matched rows for every method. All 16 scalar predictors have pooled and within-network Spearman correlations in correlations.csv, including resistance, exposure, length fraction, Omega, shape, slope, context and geometry changes. Correlation is association, not predictive validation.
Fixed feature sets: Omega; resistance+length fraction+exposure; previous selected six; selected six+upstream/downstream lengths. Fixed models: scaled Ridge(alpha=10) and a small boosted tree model. No tuning or model search on this subset. Four leave-one-network-out folds train on three networks and test on the fourth; random row splitting is not used. All scaling is fitted on training rows only. Scores include R2, MAE, zero-change and training-mean baselines, plus equal-network MAE. Existing feature choices were developed in this study, so this is exploratory study-river reuse, not pristine external validation. The four rivers are the independent groups; 156 pairs are not 156 independent rivers.

Pooled held-out timing scores (MAE in minutes)
{main.to_string(index=False)}

Pooled signed-timing Spearman correlations
{ct.to_string()}

Do not declare one method more predictable by selecting its best-scoring feature/model after seeing these results. Compare the same pre-existing feature/model across methods. Use both zero-change skill and MAE: merely shrinking all shifts can make errors smaller without improving explanatory skill. Per-network results may disagree with pooled scores. A four-network comparison cannot establish a general predictor ranking or replace the existing 16-network evaluation. Peak-excess and duration results are exploratory secondary diagnostics and have their own held-out score rows.

Files
matched_pair_responses.csv — compact, unambiguous active responses for all four methods.
absolute_shift_summary.csv / between_method_comparison.csv — timing changes, agreement, direction checks.
correlations.csv — every predictor, response, method and network.
prediction_scores.csv / equal_network_prediction_scores.csv — held-out scores and baselines.
held_out_predictions.parquet — fold predictions, actual responses and training-only baseline.
forcing_comparison.csv / routing_audit.csv / provenance.json — controls and provenance.
comparison.html — interactive timing and held-out prediction plots.
Individual isolated control outlet arrays are stored by network/fingerprint; no Qout files were changed.
'''
findings='''EARLY FINDINGS (exploratory, same matched subset):
- Typical timing effects are not uniformly smaller under Cunge: mean absolute shifts increase in 02 and 05 and decrease in 03 and 21 versus the archived workflow.
- Pooled Cunge/archived timing rank correlation is 0.708; mean absolute disagreement is 6.43 min. Of 18 raw sign reversals, none passes the conservative both-shifts-above-one-reporting-interval screen.
- Using the unchanged selected-six boosted predictor, held-out timing R2/MAE are archived fixed-X 0.253/5.58 min, current calibrated fixed-X -0.044/8.81 min, same-n fixed-X 0.070/7.85 min, and Cunge -0.275/14.32 min. This feature/model is a pre-existing benchmark, not selected as a winner on these data. All other declared models are reported.
- Cunge's selected-six timing model is worse than predicting zero shift (12.22 min MAE), and only marginally better than the training-mean baseline (14.44 min). Current evidence does not establish useful cross-network Cunge timing prediction with these descriptors.
- Do not equate this with every predictor becoming less informative: pooled resistance correlations are -0.396 versus -0.400 for current calibrated fixed-X and Cunge; Omega -0.365 versus -0.369. Slope association weakens; elongation association strengthens slightly. Per-network correlations are retained and pooled associations can reflect between-network differences.
- Cunge peak-excess and duration effects are more sensitive in this cohort, but the selected-six held-out R2 remains negative for both secondary responses. Generality awaits more networks. Geometry-predictor performance is not a test of which routing method is physically more accurate.

'''
(OUT/'REPORT.txt').write_text(findings+text)
fig=make_subplots(rows=1,cols=3,subplot_titles=['Previous workflow vs Cunge','Current calibrated fixed-X vs Cunge','Same n: fixed-X vs Cunge'],horizontal_spacing=.075)
colors={'02':'#0072B2','03':'#D55E00','05':'#009E73','21':'#CC79A7'}
for col,method in enumerate(['fixed_archived','fixed_calibrated','fixed_same_n'],1):
 for ex,g in pivot.groupby(level='example_id'):
  names=d[(d.method=='cunge')&(d.example_id==ex)].set_index('fingerprint').loc[g.index.get_level_values(1),'state_id']
  fig.add_trace(go.Scatter(x=g[method],y=g.cunge,mode='markers',name='Network '+ex,legendgroup=ex,showlegend=col==1,marker=dict(color=colors[ex],size=7,opacity=.7),text=names,hovertemplate='%{text}<br>Fixed-X %{x:.5f} h<br>Cunge %{y:.5f} h<extra>'+ex+'</extra>'),row=1,col=col)
 vals=np.r_[pivot[method].to_numpy(),pivot.cunge.to_numpy()];lo,hi=vals.min(),vals.max();fig.add_trace(go.Scatter(x=[lo,hi],y=[lo,hi],mode='lines',line=dict(color='grey',dash='dash'),showlegend=False,hoverinfo='skip'),row=1,col=col)
 fig.update_xaxes(title_text='Fixed-X child − parent peak time (h)',zeroline=True,row=1,col=col);fig.update_yaxes(title_text='Cunge child − parent peak time (h)',zeroline=True,row=1,col=col)
fig.update_layout(template='plotly_white',height=600,width=1500,margin=dict(t=100,b=115),title='Matched timing shifts · 156 pairs / 4 networks · negative = earlier child',legend=dict(orientation='h',y=-.19,yanchor='top'))
preds=pd.read_parquet(OUT/'held_out_predictions.parquet');fig2=make_subplots(rows=2,cols=3,subplot_titles=[labels[m]+'<br>'+s for s in ['components','selected_six'] for m in ['fixed_archived','fixed_calibrated','cunge']],horizontal_spacing=.075,vertical_spacing=.18)
for row,feature in enumerate(['components','selected_six'],1):
 for col,method in enumerate(['fixed_archived','fixed_calibrated','cunge'],1):
  z=preds[(preds.response=='shift_h')&(preds.feature_set==feature)&(preds.method==method)&(preds.model=='boosted_7')]
  for ex,g in z.groupby('example_id'):fig2.add_trace(go.Scatter(x=g.observed,y=g.predicted,mode='markers',name='Network '+ex,legendgroup=ex,showlegend=row==1 and col==1,marker=dict(color=colors[ex],size=6,opacity=.7),text=g.state_id,hovertemplate='%{text}<br>Routed %{x:.5f} h<br>Predicted %{y:.5f} h<extra>'+ex+'</extra>'),row=row,col=col)
  lo=min(z.observed.min(),z.predicted.min());hi=max(z.observed.max(),z.predicted.max());fig2.add_trace(go.Scatter(x=[lo,hi],y=[lo,hi],line=dict(color='grey',dash='dash'),showlegend=False,hoverinfo='skip'),row=row,col=col)
  fig2.update_xaxes(title_text='Routed timing shift (h)',row=row,col=col);fig2.update_yaxes(title_text='Held-out predicted shift (h)',row=row,col=col)
fig2.update_layout(template='plotly_white',height=950,width=1500,margin=dict(t=100,b=115),title='Predictor comparison · fixed boosted model · one complete network held out',legend=dict(orientation='h',y=-.10,yanchor='top'))
html='<html><head><meta charset="utf-8"><title>Cunge / fixed-X comparison</title></head><body style="font-family:Arial"><p>Snapshot: 02,03,05,21. Calibrations unchanged. Old workflow differs in n, forcing history/windows and grid. Same-n control isolates X. Read REPORT.txt for definitions, caveats and all scores.</p>'+fig.to_html(full_html=False,include_plotlyjs=True)+fig2.to_html(full_html=False,include_plotlyjs=False)+'</body></html>'
(OUT/'comparison.html').write_text(html)
print(summary.to_string(index=False));print(main.to_string(index=False));print(comparisons[comparisons.example.eq('pooled')].to_string(index=False))
