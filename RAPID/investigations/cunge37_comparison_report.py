"""Integrated five-network comparison, preserving earlier four-network results."""
from pathlib import Path
import json,hashlib
import pandas as pd,numpy as np
from sklearn.metrics import r2_score,mean_absolute_error
from plotly.subplots import make_subplots
import plotly.graph_objects as go
ROOT=Path(__file__).resolve().parents[2];C=ROOT/'RAPID/outputs/cunge_fixed_comparison_plus37_20261006';OUT=ROOT/'RAPID/outputs/cunge_fixed_extra_trees_plus37_20261006';OLD=ROOT/'RAPID/outputs/cunge_fixed_extra_trees_20261006'
d=pd.read_parquet(C/'paired_results.parquet');p=pd.read_parquet(OUT/'held_out_predictions.parquet');s=pd.read_csv(OUT/'scores.csv',dtype={'scope':str});h=pd.read_csv(OUT/'scores_in_hours.csv',dtype={'scope':str});macro=pd.read_csv(OUT/'equal_network_scores.csv');audit=pd.read_csv(C/'routing_audit.csv',dtype={'example':str});domain=pd.read_csv(C/'37_domain_screen.csv');fold=pd.read_csv(OUT/'fold_audit.csv',dtype={'held_out':str});m=json.loads((OUT/'provenance.json').read_text())
assert len(d)==552*4 and len(p)==552*4*4 and len(fold)==80
summary=s[s.scope.eq('pooled')][['routing','response','features','r2','mae','skill_vs_zero']].copy();summary['display_mae']=np.where(summary.response.eq('shift_h'),summary.mae*60,summary.mae);summary['mae_units']=np.where(summary.response.eq('shift_h'),'minutes','percentage points');summary.to_csv(OUT/'pooled_summary.csv',index=False)
absolute=h[h.scope.eq('pooled')].copy();absolute['mae_minutes']=absolute.mae*60;absolute.to_csv(OUT/'absolute_prediction_summary.csv',index=False)
shifts=[];agreement=[]
for (ex,method),g in d.groupby(['example_id','method']):
 shifts.append(dict(example=ex,method=method,pairs=len(g),mean_abs_shift_minutes=abs(g.shift_h).mean()*60,median_abs_shift_minutes=abs(g.shift_h).median()*60,earliest_shift_h=g.shift_h.min(),latest_shift_h=g.shift_h.max()))
shift=pd.DataFrame(shifts);shift.to_csv(OUT/'timing_shift_summary.csv',index=False)
piv=d.pivot(index=['example_id','fingerprint'],columns='method',values='shift_h');forcing=pd.read_csv(C/'forcing_comparison.csv',dtype={'example':str}).set_index('example')
for ex,g in [('pooled',piv)]+list(piv.groupby(level='example_id')):
 dt=np.array([max(forcing.loc[e,'old_reporting_step_s'],forcing.loc[e,'new_reporting_step_s'])/3600 for e in g.index.get_level_values('example_id')])
 for method in ['fixed_archived','fixed_calibrated','fixed_same_n']:
  a=g[method];b=g.cunge;resolved=(abs(a)>dt)&(abs(b)>dt);rev=a*b<0
  agreement.append(dict(example=ex,versus=method,pairs=len(g),spearman=a.corr(b,method='spearman'),mae_between_shifts_minutes=abs(a-b).mean()*60,raw_sign_reversals=int(rev.sum()),both_above_one_interval=int(resolved.sum()),resolved_sign_reversals=int((rev&resolved).sum())))
agreement=pd.DataFrame(agreement);agreement.to_csv(OUT/'timing_agreement.csv',index=False)
# Separate the new river's test results from gains on the original four.
parts=[]
for (routing,response,features),g in p.groupby(['routing','response','features']):
 for label,z in [('original_four_test_pairs',g[g.example_id.ne('37')]),('new_37_test_pairs',g[g.example_id.eq('37')])]:
  y=z.observed_shift_h.to_numpy();yp=z.predicted_shift_h.to_numpy();parts.append(dict(routing=routing,trained_response=response,features=features,cohort=label,n=len(z),r2=r2_score(y,yp),mae_minutes=mean_absolute_error(y,yp)*60,zero_mae_minutes=abs(y).mean()*60))
parts=pd.DataFrame(parts);parts.to_csv(OUT/'test_cohort_scores_hours.csv',index=False)
previous=pd.read_csv(OLD/'scores_in_hours.csv',dtype={'scope':str});prior=previous[(previous.scope=='pooled')&(previous.routing=='cunge')][['trained_response','features','r2','mae']].copy();prior['mae_minutes']=prior.pop('mae')*60;prior=prior.rename(columns={'r2':'previous_four_network_r2','mae_minutes':'previous_four_network_mae_minutes'});change=parts[(parts.routing=='cunge')&(parts.cohort=='original_four_test_pairs')].merge(prior,on=['trained_response','features'],validate='one_to_one');change.to_csv(OUT/'original_four_change.csv',index=False)
f=pd.read_parquet(OUT/'matched_features_responses.parquet');corr=[]
for method,z in f.groupby('method'):
 for ex,g in [('pooled',z)]+list(z.groupby('example_id')):
  for c in m['geometry_features']:
   for target in ['shift_h','relative_shift_pct']:corr.append(dict(method=method,example=ex,feature=c,target=target,n=len(g),spearman=g[c].corr(g[target],method='spearman') if g[c].nunique()>1 else np.nan))
pd.DataFrame(corr).to_csv(OUT/'correlations.csv',index=False)
fa=audit[audit.example.eq('37')&audit.method.ne('cunge')];direct=fa[fa.native_reference_max_error_cms.notna()];assert fa.padding_error_cms.max()<1e-6 and fa.omitted_alias_error_bound_cms.max()<=1e-7 and direct.native_reference_max_error_cms.max()<1e-6
ranges=[]
for method,g in f.groupby('method'):
 train=g[g.example_id.ne('37')];test=g[g.example_id.eq('37')]
 for target in ['shift_h','relative_shift_pct']:
  lo,hi=train[target].min(),train[target].max()
  ranges.append(dict(method=method,target=target,training_min=lo,training_max=hi,test_min=test[target].min(),test_max=test[target].max(),test_rows_outside_training_response_range=int(((test[target]<lo)|(test[target]>hi)).sum())))
ranges=pd.DataFrame(ranges);ranges.to_csv(OUT/'37_response_range_audit.csv',index=False)
errors=[]
for (target,features),g in p[(p.routing=='cunge')&(p.example_id=='37')].groupby(['response','features']):
 e=(g.observed_shift_h-g.predicted_shift_h)**2;j=e.idxmax();v=g.loc[j]
 errors.append(dict(trained_response=target,features=features,largest_squared_error_state=v.state_id,largest_error_share_pct=100*e.max()/e.sum(),routed_shift_h=v.observed_shift_h,predicted_shift_h=v.predicted_shift_h))
errors=pd.DataFrame(errors);errors.to_csv(OUT/'37_error_concentration.csv',index=False)
text=f'''NETWORK 37 ADDED — FIVE-NETWORK COMPARISON, 6 October 2026

Interpretation
37 broadens the response range substantially. Most of its Cunge shifts are small (median absolute shift 0.84 minutes), but CB001_bubble_3 advances the peak by 7.46 hours. This state accounts for about 72% of held-out-37 squared error for the relative-response geometry model. Its response lies far outside the original four networks' training range; this limitation also affects the fixed-X controls.
Relative-response prediction remains preferable to direct-hour prediction for Cunge, but transfer is weak on 37: 4.67-minute MAE versus 4.88 minutes for predicting no change. Adding 37 to training also worsens the original four networks' relative-to-hours MAE from 7.06 to 8.84 minutes. Therefore the lower pooled five-network MAE is not evidence of improved generalization. Supplying n does not resolve this.
Same-n fixed-X versus Cunge shifts on 37 retain a Spearman correlation of 0.841. One reversal exceeds a reporting interval in both methods: S214_unit_265 changes from +0.633 hours under fixed-X to -0.363 hours under Cunge. This is a substantive response difference requiring hydrograph inspection before attributing a physical mechanism. The other raw reversals involve at least one smaller shift.
Resistance has the largest absolute pooled single-feature Spearman correlation with Cunge relative shift in this cohort (-0.454), but only -0.299 within 37. This is association, not a held-out single-predictor performance ranking, and the pooled ranking is heavily influenced by 37's row count.

Coverage and reproducibility
37 completion verified: 397 distinct hydraulic states, 396 distinct parent-child pairs. One existing topology exclusion (island_37_151) remains excluded, consistently with the previous canonical analysis. All 396 match the previous physical fingerprints. Combined cohort: 552 distinct pairs, 02/03/05/21/37. Network 37 supplies 71.7% of rows but is only one of five independent networks. Aliases are not additional observations. All comparisons here have S000_base as parent.

Existing four-network routed controls and results are retained unchanged and reused. Current n calibrations and calibration targets remain unchanged. Extra Trees uses exactly the previous 15 log-geometry features and settings, then the same features plus supplied n; 80 forests across 4 routing configurations, 2 targets, 2 feature sets and 5 held-out networks. No tuning or feature search. Each training fold excludes the complete test network. Fitting remains pair-weighted to preserve the previous estimator; equal-network evaluation is provided to expose imbalance.

Configurations and targets
fixed_archived: previous X=.1, n=.035, earlier windows/history and grid.
fixed_calibrated: current forcing/grid, X=.1, previous calibrated fixed-X n.
fixed_same_n: current forcing/grid, X=.1 and Cunge n; isolates X against Cunge.
cunge: completed full-engine Cunge outputs and inherited base n.
Absolute shift = child interpolated peak minus parent interpolated peak (hours).
Relative shift = 100 * absolute shift / parent interpolated peak lag to first sampled inlet peak.
Relative predictions are also converted back to hours using the known parent lag, never the held-out child result. This is conditional prediction with a known parent response, not an ungauged roughness model or independent-event validation.

Network 37 timing effects
{shift[shift.example.eq('37')].to_string(index=False)}

Network 37 agreement between routing configurations
{agreement[agreement.example.eq('37')].to_string(index=False)}
Sign reversals counted as resolved require both shifts to exceed one reporting interval, conservatively; this is a numerical sensitivity screen, not a statistical confidence interval.

Five-network pooled Extra Trees scores
{summary[['routing','response','features','r2','display_mae','mae_units','skill_vs_zero']].to_string(index=False)}

Both target formulations evaluated on absolute hours
{absolute[['routing','trained_response','features','r2','mae_minutes','skill_vs_zero']].to_string(index=False)}

Equal-network scores (hours for shift_h; percentage points for relative_shift_pct)
{macro.to_string(index=False)}

Network 37 held out: trained only on original four networks
{parts[parts.cohort.eq('new_37_test_pairs')].to_string(index=False)}

Network 37 extrapolation diagnostic (no rows removed or refitting)
{ranges.to_string(index=False)}
{errors.to_string(index=False)}
Extra Trees averages training responses and cannot extrapolate beyond their range. A large held-out response can dominate squared error; it remains in every primary score.

Original four test networks: effect of adding 37 to their training data
{change.to_string(index=False)}
Pooled five-network R2 can improve merely because 37 adds much larger response variance. Judge transfer using the held-out-37 and original-four tables as well as equal-network MAE. The historical R2 around .57 used all 16 networks and cannot be directly compared with this five-network snapshot.

Hydraulic domain check across all 397 states of 37
Same first-1%-departure-to-inlet-peak period proxy ({domain.rise_proxy_h.iloc[0]:.3f} h), reference geometry and raw Manning water velocity. Fractions are physical channel length within each state, not fractions of valid rivers.
Provisional Tplus>=80 / F2<=.9 passing length range: {domain.old_screen_length_pct.min():.3f}%–{domain.old_screen_length_pct.max():.3f}%.
Diffusion timescale T*S*sqrt(g/h)>=30 plus F2<=.9 passing length range: {domain.diffusion_and_subcritical_length_pct.min():.3f}%–{domain.diffusion_and_subcritical_length_pct.max():.3f}%.
With half the period proxy: {domain.half_period_diffusion_and_subcritical_length_pct.min():.3f}%–{domain.half_period_diffusion_and_subcritical_length_pct.max():.3f}%.
These remain applicability screens, not validation of flow/depth consistency, backwater assumptions or local transients. Repeated unchanged links across states are not independent evidence. See previous cunge_domain_quick_check_20261006 report for published criterion and limitations.

Numerical control checks
37's completed Cunge full-engine runs passed their existing native positivity, tail settlement and independent-cascade checks. The fixed-X controls use the frequency response of the SAME native recurrence, with exact input-hold/report-average alias weights. Omitted aliases are bounded by the monotone link-response magnitude envelope; long prehistory and trailing padding approximate steady initialization without circular wrap contamination. Every fixed control is independently checked with doubled padding.
Across {len(fa)} fixed-control states: maximum alias bound {fa.omitted_alias_error_bound_cms.max():.3g} m3/s; maximum padding sensitivity {fa.padding_error_cms.max():.3g} m3/s. Direct native recurrence comparisons available for {len(direct)} control states ({direct.fingerprint.nunique()} physical states): maximum error {direct.native_reference_max_error_cms.max():.3g} m3/s; maximum peak-bin difference {direct.native_reference_peak_bin_shift.abs().max():.0f} bins. All reported control outlets are finite, nonnegative and meet the existing settlement criterion. This establishes numerical outlet agreement; it does not replace all-reach native positivity audits for the remaining fixed-control states, which were not rerun. No production inputs or user running jobs changed.

Artifacts
pooled_summary.csv / absolute_prediction_summary.csv / equal_network_scores.csv: overall scores.
test_cohort_scores_hours.csv / original_four_change.csv: separate new-network transfer from changes in the original four.
timing_shift_summary.csv / timing_agreement.csv: matched physical response changes.
correlations.csv: all 15 individual predictor associations, pooled and per river.
held_out_predictions.parquet / fold_audit.csv / provenance.json: exact predictions and evaluation identities.
comparison.html: interactive response and prediction plots.
../cunge_fixed_comparison_plus37_20261006/37_domain_screen.csv: all-state hydraulic screens.
../cunge_fixed_comparison_plus37_20261006/routing_audit.csv: numerical control diagnostics.
'''
(OUT/'REPORT.txt').write_text(text)
colors={'02':'#0072B2','03':'#D55E00','05':'#009E73','21':'#CC79A7','37':'#333333'}
fig=make_subplots(rows=2,cols=2,subplot_titles=['Previous fixed-X versus Cunge · routed shifts','Same n: fixed-X versus Cunge · routed shifts','Cunge · direct-hour held-out prediction','Cunge · relative prediction converted to hours'],horizontal_spacing=.10,vertical_spacing=.17)
for col,method in enumerate(['fixed_archived','fixed_same_n'],1):
 for ex,g in piv.groupby(level='example_id'):
  fig.add_trace(go.Scatter(x=g[method],y=g.cunge,mode='markers',name='Network '+ex,legendgroup=ex,showlegend=col==1,marker=dict(color=colors[ex],size=6,opacity=.55),hovertemplate='Fixed-X %{x:.4f} h<br>Cunge %{y:.4f} h<extra>'+ex+'</extra>'),row=1,col=col)
 vals=np.r_[piv[method],piv.cunge];lo,hi=vals.min(),vals.max();fig.add_trace(go.Scatter(x=[lo,hi],y=[lo,hi],line=dict(color='grey',dash='dash'),showlegend=False,hoverinfo='skip'),row=1,col=col);fig.update_xaxes(title_text='Fixed-X child − parent peak (h)',row=1,col=col);fig.update_yaxes(title_text='Cunge child − parent peak (h)',row=1,col=col)
for col,response in enumerate(['shift_h','relative_shift_pct'],1):
 z=p[(p.routing=='cunge')&(p.features=='geometry_15')&(p.response==response)]
 for ex,g in z.groupby('example_id'):fig.add_trace(go.Scatter(x=g.observed_shift_h,y=g.predicted_shift_h,mode='markers',name='Network '+ex,legendgroup=ex,showlegend=False,text=g.state_id,marker=dict(color=colors[ex],size=6,opacity=.55),hovertemplate='%{text}<br>Routed %{x:.4f} h<br>Predicted %{y:.4f} h<extra>'+ex+'</extra>'),row=2,col=col)
 vals=np.r_[z.observed_shift_h,z.predicted_shift_h];lo,hi=vals.min(),vals.max();fig.add_trace(go.Scatter(x=[lo,hi],y=[lo,hi],line=dict(color='grey',dash='dash'),showlegend=False,hoverinfo='skip'),row=2,col=col);fig.update_xaxes(title_text='Routed child − parent peak (h)',row=2,col=col);fig.update_yaxes(title_text='Held-out prediction (h)',row=2,col=col)
fig.update_layout(template='plotly_white',width=1400,height=1050,margin=dict(t=110,b=115,l=90,r=45),title='Five-network snapshot · 552 distinct pairs · calibrated values unchanged',legend=dict(orientation='h',y=-.10,yanchor='top'))
(OUT/'comparison.html').write_text('<html><head><meta charset="utf-8"><title>Network 37 extension</title></head><body style="font-family:Arial"><p>37 contributes 396 of 552 pairs. Use legend clicks and zoom to inspect the smaller responses. Models hold out a complete network; no recalibration. Read REPORT.txt for equal-network scores and the numerical control checks.</p>'+fig.to_html(full_html=False,include_plotlyjs=True)+'</body></html>')
verification=dict(frequency_implementation_sha256=hashlib.sha256((ROOT/'RAPID/investigations/cunge_frequency_control.py').read_bytes()).hexdigest(),report_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),direct_native_control_states=len(direct),max_native_difference_cms=float(direct.native_reference_max_error_cms.max()),max_padding_difference_cms=float(fa.padding_error_cms.max()),max_alias_bound_cms=float(fa.omitted_alias_error_bound_cms.max()))
(OUT/'numerical_verification.json').write_text(json.dumps(verification,indent=2))
print(shift[shift.example.eq('37')].to_string(index=False));print(summary[summary.routing.eq('cunge')].to_string(index=False));print(absolute[(absolute.routing=='cunge')][['trained_response','features','r2','mae_minutes']].to_string(index=False));print(change.to_string(index=False));print(verification)
