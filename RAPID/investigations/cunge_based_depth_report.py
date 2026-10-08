"""Summarize the bounded BASED depth experiment without modifying source runs."""
from pathlib import Path
import json
import pandas as pd,numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'RAPID/outputs/cunge_based_depth_sensitivity'
r=pd.read_csv(OUT/'comparison.csv');old=pd.read_csv(ROOT/'RAPID/outputs/cunge_calibration_alpha2/comparison.csv',dtype={'example':str})
base=old[old.example.eq('11')&old.method.eq('cunge')].iloc[0];source=Path(base.output_dir)
g=pd.read_parquet(source/'observed_downstream.parquet');start=pd.to_datetime(g.time,utc=True).min();end=pd.to_datetime(g.time,utc=True).max()
fig=make_subplots(rows=2,cols=1,subplot_titles=['Peak-lag calibration at each BASED depth multiplier','Best calibrated downstream shapes (independently normalized; no time shifts)'],vertical_spacing=.17)
colors=['#9467bd','#cf7443','#16836b','#5473a4','#bc5588']
for row,color in zip(r.itertuples(),colors):
 p=Path(row.output_dir);t=pd.read_csv(p/'trials.csv').sort_values('n_manning');valid=t.valid.astype(bool)
 fig.add_trace(go.Scatter(x=t.loc[valid,'n_manning'],y=t.loc[valid,'lag_h'],mode='lines+markers',name=f'Depth ×{row.depth_factor:g}',legendgroup=str(row.depth_factor),line=dict(color=color),customdata=t.loc[valid,'normalized_shape_rmse'],hovertemplate='n=%{x:.5f}<br>Lag=%{y:.3f} h<br>Shape RMSE=%{customdata:.4f}<extra></extra>'),row=1,col=1)
 if (~valid).any():fig.add_trace(go.Scatter(x=t.loc[~valid,'n_manning'],y=t.loc[~valid,'lag_h'],mode='markers',marker=dict(color=color,symbol='x',size=8),name=f'Invalid ×{row.depth_factor:g}',legendgroup=str(row.depth_factor),showlegend=False,hovertemplate='Excluded: unsettled tail or other trial validity failure<br>n=%{x:.5f}<br>Lag=%{y:.3f} h<extra></extra>'),row=1,col=1)
 w=pd.read_parquet(p/'best_wave.parquet');tt=pd.to_datetime(w.time_utc,utc=True);mask=(tt>=start)&(tt<=end);q=w.loc[mask,'q_outlet_cms'].to_numpy();norm=(q-q.min())/np.ptp(q)
 fig.add_trace(go.Scatter(x=tt[mask].dt.strftime('%Y-%m-%dT%H:%M:%SZ'),y=norm,mode='lines',name=f'Depth ×{row.depth_factor:g}',line=dict(color=color),legendgroup=str(row.depth_factor),showlegend=False),row=2,col=1)
fig.add_hline(y=63.5,line_dash='dash',annotation_text='Observed peak lag: 63.5 h',row=1,col=1)
q=g.discharge_cms.to_numpy();fig.add_trace(go.Scatter(x=pd.to_datetime(g.time,utc=True).dt.strftime('%Y-%m-%dT%H:%M:%SZ'),y=(q-q.min())/np.ptp(q),name='Observed downstream',line=dict(color='black')),row=2,col=1)
fig.update_xaxes(title_text='Manning n',row=1,col=1);fig.update_yaxes(title_text='Peak lag (hours)',row=1,col=1)
fig.update_xaxes(title_text='Time (UTC)',row=2,col=1);fig.update_yaxes(title_text='Normalized discharge',row=2,col=1)
fig.update_layout(template='plotly_white',height=900,title='Case 11: bounded BASED depth / Kb sensitivity',margin=dict(t=100),legend=dict(orientation='h',y=-.12))
fig.write_html(OUT/'case_11_depth_sensitivity.html',include_plotlyjs=True)
cols=['depth_factor','kb_factor','link54_depth_m','n_manning','lag_h','abs_error_h','normalized_shape_rmse','raw_rmse_cms','model_to_observed_peak_ratio','within_sampling','half_clock_lag_change_h','double_grid_lag_change_h','valid_trials','trials']
audit=pd.read_csv(OUT/'all_network_parameter_equivalence.csv')
report='''BOUNDED BASED DEPTH / Kb TEST
Depth factors: 0.75, 1, 1.25, 1.5, 2. Factor 2 is an extended stress test.
These are explicit sensitivity bounds, not empirically established depth uncertainty.
Depth-width exponent 0.65 retained; Kb divided by the depth multiplier. Qref, widths, slopes and inlet fixed.
All five case-11 candidates use one common grid selected at maximum depth and minimum n, and one common routing clock.
Manning n searched from 0.01 to 0.10 using 31 uniform samples plus original-equivalent candidates, local refinement and sign-crossing refinement.
Invalid (including unsettled-tail) trials excluded. Best candidates checked with half timestep and doubled spatial subdivision.
No new preparation bundles, production-input edits, child runs or independent validation events.

IDENTIFIABILITY
c = (5/3) h^(2/3) sqrt(S) / n. Replacing h by f*h and n by f^(2/3)*n leaves c, capped c, K and X unchanged on a fixed grid.
Thus scalar depth calibration and n calibration cannot be separately identified by routing hydrographs under these assumptions.
All 16 networks x 5 depth multipliers passed the original-grid coefficient identity check (including algebraic n values outside the fitting interval).
This is a parameter-equivalence audit, not 80 new event recalibrations or new independent validation.

CASE 11 RESULTS
'''+r[cols].to_string(index=False)+'\n\nMaximum relative parameter differences in identity audit:\n'+audit[['rapid_celerity_mps','rapid_k','rapid_x']].max().to_string()+'\n\nHistorical unchanged-grid Cunge baseline:\n'+str({k:base[k] for k in ['n_manning','lag_h','normalized_shape_rmse','raw_rmse_cms']})+'\n'
report+='\nINTERPRETATION\nNo depth candidate achieves the observed 63.5 h peak lag. Uniform depth/Kb scaling cannot independently correct this response when n is refitted. Original-grid identity confirms that other networks can retain exactly their previous response by rescaling n; the 0.01–0.10 search bound can prevent that rescaling for large depth factors. These checks support retaining BASED and fixed-X as the practical empirical benchmark, not a general rejection of Muskingum–Cunge. Shape and absolute-flow limitations remain; no added calibration variable is supported by this test.\n'
(OUT/'REPORT.txt').write_text(report)
print(r[cols].to_string(index=False));print(OUT/'REPORT.txt')
