"""Summarize the completed case-03 full hierarchy pilot."""
from pathlib import Path
import json,argparse
import numpy as np,pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
ROOT=Path(__file__).resolve().parents[2]
parser=argparse.ArgumentParser();parser.add_argument('--example',default='03');parser.add_argument('--output-root',type=Path,default=ROOT/'RAPID/outputs/cunge_full_hierarchy_pilot');args=parser.parse_args()
example=args.example.zfill(2);OUT=args.output_root/example
m=json.loads((OUT/'experiment.json').read_text());done=json.loads((OUT/'completion.json').read_text());unique=pd.read_csv(OUT/'unique_run_registry.csv');states=pd.read_csv(OUT/'state_run_registry.csv');pairs=pd.read_csv(OUT/'parent_child_comparison.csv')
source=Path(m['source_calibration']);g=pd.read_parquet(source/'observed_downstream.parquet');times=pd.to_datetime(g.time,utc=True);start,end=times.min(),times.max()
fig=make_subplots(rows=2,cols=1,subplot_titles=['All distinct state outlet hydrographs (event window; no normalization)','Parent–child peak-lag comparison'],vertical_spacing=.15)
for i,r in enumerate(unique.itertuples()):
 w=pd.read_parquet(Path(r.output_dir)/'outlet_hydrograph.parquet');t=pd.to_datetime(w.time_utc,utc=True);mask=(t>=start)&(t<=end);base=r.fingerprint==m['base_fingerprint'];aliases=states[states.fingerprint.eq(r.fingerprint)]
 label='Cunge base network' if base else 'Cunge child states'
 names='; '.join(aliases.family+': '+aliases.state)
 fig.add_trace(go.Scatter(x=t[mask].dt.strftime('%Y-%m-%dT%H:%M:%SZ'),y=w.loc[mask,'q_outlet_cms'],mode='lines',name=label,legendgroup=label,showlegend=base or i==1,
  line=dict(color='#E31A1C' if base else '#00BFD8',width=3 if base else 1),opacity=1 if base else .4,
  hovertemplate=names+'<br>%{x}<br>Q: %{y:.2f} m³/s<extra></extra>'),row=1,col=1)
fig.add_trace(go.Scatter(x=times.dt.strftime('%Y-%m-%dT%H:%M:%SZ'),y=g.discharge_cms,mode='lines',name='Downstream gauge',line=dict(color='black',dash='dash',width=2)),row=1,col=1)
for family,color in [('indep','#16836b'),('compound_bubbles','#9467bd'),('island','#cf7443')]:
 p=pairs[pairs.family.eq(family)];fig.add_trace(go.Scatter(x=p.parent_lag_h,y=p.child_lag_h,mode='markers',name=family,marker=dict(color=color,size=8),customdata=np.c_[p.parent_state,p.child_state,p.delta_lag_h,p.delta_peak_pct],hovertemplate='%{customdata[0]} → %{customdata[1]}<br>Parent lag: %{x:.3f} h<br>Child lag: %{y:.3f} h<br>Δlag: %{customdata[2]:.3f} h<br>Δpeak: %{customdata[3]:.3f}%<extra></extra>'),row=2,col=1)
lo=min(pairs.parent_lag_h.min(),pairs.child_lag_h.min())-.5;hi=max(pairs.parent_lag_h.max(),pairs.child_lag_h.max())+.5
fig.add_trace(go.Scatter(x=[lo,hi],y=[lo,hi],mode='lines',line=dict(color='gray',dash='dash'),name='Equal peak lag',showlegend=False),row=2,col=1)
fig.update_xaxes(title_text='Time (UTC)',row=1,col=1);fig.update_yaxes(title_text='Discharge (m³/s)',row=1,col=1)
fig.update_xaxes(title_text='Parent peak lag (hours)',range=[lo,hi],row=2,col=1);fig.update_yaxes(title_text='Child peak lag (hours)',range=[lo,hi],row=2,col=1)
fig.update_layout(template='plotly_white',height=1050,width=1250,title=f'Network {example} · full Cunge hierarchy pilot<br><sup>Inherited parent Manning n; shared forcing, timestep and reference hydraulics</sup>',margin=dict(t=120,b=130),legend=dict(orientation='h',y=-.15))
fig.write_html(OUT/'hierarchy_results.html',include_plotlyjs=True)
family=states.groupby('family').agg(state_entries=('state','size'),unique_hydraulic_states=('fingerprint','nunique'));family.to_csv(OUT/'family_summary.csv')
report=f'''FULL CUNGE HIERARCHY PILOT — NETWORK {example}
Completed on 6 October 2026. This is an isolated experimental run, not production promotion.

Known topology-review exclusions: {m.get('excluded_state_entries',0)} (see excluded_states.csv).
Coverage: {len(states)} registered state entries, {len(unique)} unique hydraulic states, {len(pairs)} parent-child comparisons.
Identical hydraulic states are routed once and retained as aliases in state_run_registry.csv.
{family.to_string()}

Inherited Manning n: {m['n_manning']:.12f}; no child recalibration.
Alpha: 2. BASED depth relationship and anchored physical geometry unchanged.
Shared native timestep: {m['routing_dt_s']} seconds (base calibration used a different clock).
Common forcing/report interval: {m['forcing_step_s']} seconds.
Common horizon: {m['simulation_days']:.3f} days; all state tails settled with the shared forcing.
Unchanged exterior-link parameter comparisons: {m['shared_exterior_links_verified']}; Qref, c, K, X and subsection counts verified unchanged.

All states routed successfully: {done['all_routed']}.
Minimum native-substep discharge: {done['min_native_flow_cms']:.6f} m³/s.
Maximum full sparse-engine versus independent-cascade difference: {done['max_engine_error_cms']:.4g} m³/s.
Base peak lag: {done['base_lag_h']:.6f} h; change from archived calibrated lag: {done['base_lag_change_from_calibration_h']:.6f} h.
Across-state peak-lag range: {done['lag_min_h']:.6f}–{done['lag_max_h']:.6f} h.
Largest absolute parent-child peak-lag change: {done['parent_child_max_abs_lag_change_h']:.6f} h.
Peak lags are based on saved forcing-interval means; equal sampled peaks do not imply identical hydrographs.

Storage at completion: {done['actual_bytes']/1024**2:.1f} MiB before this small report/plot addition.
Full Qout.nc retained for every numerical reach at every reporting interval; all native substeps are audited but not stored.
Prep files, parameter maps, outlet CSV/Parquet hydrographs and source hashes retained.
Original input and production trees are unchanged. Outputs are local, not on the external drive.

INTERPRETATION
This pilot demonstrates full hierarchy execution, consistent parent-child inputs and numerical agreement. It is not an independent physical validation of children or a reversal of the decision to retain fixed X for production. Case 11's unresolved Cunge limitation remains. Numerical or hydraulic aliases are not independent replicates.

FILES
experiment.json — source provenance, inherited calibration and shared clock.
completion.json — numerical completion checks and storage usage.
state_run_registry.csv — every hierarchy state, its physical fingerprint and output directory.
unique_run_registry.csv — distinct routed states.
parent_child_comparison.csv — peak timing and magnitude contrasts.
exterior_invariance.csv — unchanged-link parameter audit.
states/<fingerprint>/prep — complete RAPID inputs.
states/<fingerprint>/run/Qout.nc — full reported all-reach output.
states/<fingerprint>/outlet_hydrograph.csv — dated inlet/outlet discharge series.
hierarchy_results.html — standalone interactive overview.
'''
(OUT/'REPORT.txt').write_text(report);print(report)
