"""Report the isolated case-11 two-zone calibration."""
from pathlib import Path
import json
import numpy as np,pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'RAPID/outputs/cunge_case11_two_zone';PREV=ROOT/'RAPID/outputs/cunge_calibration_alpha2'
r=pd.read_csv(OUT/'candidates.csv');trials=pd.read_csv(OUT/'trials.csv');valid=trials[trials.valid].copy()
old=pd.read_csv(PREV/'comparison.csv',dtype={'example':str});old=old[old.example.eq('11')];source=Path(old.iloc[0].output_dir)
g=pd.read_parquet(source/'observed_downstream.parquet');tt=pd.to_datetime(g.time,utc=True);start,end=tt.min(),tt.max()
fig=make_subplots(rows=3,cols=1,subplot_titles=['Two-zone search: colours show modelled peak lag (observed: 63.5 h)','Downstream shapes: independently normalized, unchanged UTC','Downstream discharge: original units'],vertical_spacing=.10)
fig.add_trace(go.Scatter(x=valid.n_rest,y=valid.n_flat,mode='markers',marker=dict(color=valid.lag_h,colorscale='Viridis',size=7,colorbar=dict(title='Lag (h)',len=.26,y=.88)),customdata=np.c_[valid.abs_error_h,valid.normalized_shape_rmse],hovertemplate='n rest: %{x:.5f}<br>n flat: %{y:.5f}<br>Peak lag: %{marker.color:.3f} h<br>Timing error: %{customdata[0]:.3f} h<br>Shape RMSE: %{customdata[1]:.4f}<extra></extra>',showlegend=False),row=1,col=1)
fig.add_trace(go.Scatter(x=[.01,.1],y=[.01,.1],mode='lines',line=dict(dash='dash',color='grey'),name='Uniform n'),row=1,col=1)
series=[('Observed',tt,g.discharge_cms,'black')]
for method,name,color in [('fixed_x','Fixed X calibrated','#5473a4'),('cunge','Uniform-n Cunge','#cf7443')]:
 w=pd.read_parquet(source/f'{method}_best_wave.parquet');series.append((name,pd.to_datetime(w.time_utc,utc=True),w.q_outlet_cms,color))
for label,name,color in [('best_timing','Two-zone: best timing','#9467bd'),('best_shape','Two-zone: best shape','#16836b')]:
 w=pd.read_parquet(OUT/label/'wave.parquet');series.append((name,pd.to_datetime(w.time_utc,utc=True),w.q_outlet_cms,color))
 row=r[r.candidate.eq(label)].iloc[0];fig.add_trace(go.Scatter(x=[row.n_rest],y=[row.n_flat],mode='markers',marker=dict(symbol='star',color=color,size=16,line=dict(color='black',width=1)),name=name,legendgroup=name,showlegend=False),row=1,col=1)
for name,times,q,color in series:
 mask=(times>=start)&(times<=end);q=np.asarray(q)[mask];x=times[mask].dt.strftime('%Y-%m-%dT%H:%M:%SZ')
 for row,y in [(2,(q-q.min())/np.ptp(q)),(3,q)]:fig.add_trace(go.Scatter(x=x,y=y,mode='lines',name=name,legendgroup=name,showlegend=row==2,line=dict(color=color,width=2)),row=row,col=1)
fig.update_xaxes(title_text='Manning n: remaining network',row=1,col=1);fig.update_yaxes(title_text='Manning n: long flat reach',row=1,col=1)
fig.update_xaxes(title_text='Time (UTC)',row=2,col=1);fig.update_xaxes(title_text='Time (UTC)',row=3,col=1)
fig.update_yaxes(title_text='Normalized discharge',row=2,col=1);fig.update_yaxes(title_text='Discharge (m³/s)',row=3,col=1)
fig.update_layout(template='plotly_white',height=1200,title='Case 11: final two-zone Manning-n diagnostic',legend=dict(orientation='h',y=-.07),margin=dict(t=100))
fig.write_html(OUT/'two_zone_results.html',include_plotlyjs=True)
cols=['candidate','n_flat','n_rest','lag_h','abs_error_h','normalized_shape_rmse','raw_rmse_cms','model_to_observed_peak_ratio','within_sampling','half_clock_lag_change_h','double_grid_lag_change_h','double_grid_peak_change_pct','engine_probe_max_error_cms']
# Continuous mode shapes at fixed n are a descriptive, in-sample diagnostic.
summary={'trials':len(trials),'valid_trials':len(valid),'successful_timing_trials':int(valid.within_sampling.sum()),'minimum_valid_lag_h':float(valid.lag_h.min()),'maximum_valid_lag_h':float(valid.lag_h.max()),'observed_lag_h':63.5,'search_bounds':[.01,.1]}
(OUT/'summary.json').write_text(json.dumps(summary,indent=2))
report='''CASE 11 TWO-ZONE MANNING TEST
Zone 1: physical link 54, length 34.912 km. Zone 2: all other links.
BASED depths/Kb relationship, Qref, width, slope, upstream forcing, original grid and clock retained.
Two independently varied n values, each in 0.01–0.10. These are diagnostic bounds, not a site-specific plausibility statement.
13x13 initial grid plus original uniform candidate; two 7x7 neighborhood refinements around best timing and best normalized shape; target-crossing refinement when present.
Primary objective: absolute peak-lag error; normalized shape error breaks equal-timing ties. Separate best-shape candidate is reported, not substituted for the primary objective.
Trial validity requires tail settlement, nonboundary peak and nonnegative native flows within numerical tolerance.
Half-timestep and doubled spatial-grid checks on each candidate; changing-inlet sparse-engine probes independently cross-check routing.
Original uniform-n result reproduced as a regression check. No production edits, child runs, other-network recalibrations or independent-event validation.

SEARCH SUMMARY
'''+json.dumps(summary,indent=2)+'\n\nCANDIDATES\n'+r[cols].to_string(index=False)+'\n\nPREVIOUS CALIBRATIONS\n'+old[['method','n_manning','lag_h','abs_error_h','normalized_shape_rmse','raw_rmse_cms','model_to_observed_peak_ratio']].to_string(index=False)+'\n'
report+='\nINTERPRETATION\n'+('At least one sampled candidate meets the timing criterion. This is an additional-parameter, single-event fit; assess shape, parameter plausibility and independent validation before adopting it.\n' if valid.within_sampling.any() else 'No sampled valid two-zone candidate meets the timing criterion. Spatially differentiating Manning n between this flat reach and the rest does not resolve the discrepancy in the tested bounds. This finite search is not proof that no possible spatial parameterization can fit.\n')
(OUT/'REPORT.txt').write_text(report)
print(report)
# Static companion for visual review and export.
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig2,axes=plt.subplots(1,2,figsize=(13,4.8),constrained_layout=True)
for name,times,q,color in series:
 mask=(times>=start)&(times<=end);qq=np.asarray(q)[mask];days=(times[mask]-start).dt.total_seconds()/86400
 axes[0].plot(days,(qq-qq.min())/np.ptp(qq),label=name,color=color,lw=1.5)
 axes[1].plot(days,qq,label=name,color=color,lw=1.5)
axes[0].set(title='Normalized shapes; no time alignment',xlabel='Days from event-window start',ylabel='Normalized discharge')
axes[1].set(title='Absolute discharge',xlabel='Days from event-window start',ylabel='Discharge (m³/s)')
axes[0].legend(fontsize=8);fig2.suptitle('Case 11: two-zone Manning-n test')
fig2.savefig(OUT/'two_zone_hydrographs.png',dpi=150);plt.close(fig2)
