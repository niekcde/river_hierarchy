"""Diagnose the global-peak discontinuity; no additional calibration or target change."""
from pathlib import Path
import sys,json
import pandas as pd,numpy as np
from scipy.signal import find_peaks
import plotly.graph_objects as go
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT),str(ROOT/'RAPID/src'),str(Path(__file__).parent)]
import cunge_calibration_reviewed as cal
from rapid_tools import cunge
r=pd.read_csv(cal.OUT/'comparison.csv',dtype={'example':str});r=r[(r.example=='54')&(r.method=='cunge')].iloc[0];p=Path(r.output_dir);m=json.loads((p/'case.json').read_text());tr=pd.read_csv(p/'cunge_trials.csv').sort_values('n_manning')
pairs=[(a,b) for a,b in zip(tr.iloc[:-1].itertuples(),tr.iloc[1:].itertuples()) if a.error_h<0<b.error_h]
a,b=min(pairs,key=lambda pair:pair[1].n_manning-pair[0].n_manning)
grid=pd.read_parquet(p/'frozen_grid.parquet');w=pd.read_parquet(p/'cunge_best_wave.parquet');times=pd.to_datetime(w.time_utc,utc=True);flow=w.q_inflow_cms.to_numpy();waves=[];records=[];fig=go.Figure()
for label,v in [('before_switch',a),('after_switch',b)]:
 f=cunge.parameters(grid,v.n_manning);q,audit=cunge.route(f,flow,m['nsub'],m['routing_dt_s']);waves.append(q);ids,_=find_peaks(q);ids=sorted(ids,key=lambda j:q[j],reverse=True)[:2]
 record=dict(side=label,n_manning=v.n_manning,peak_utc=times.iloc[q.argmax()].isoformat(),error_h=(times.iloc[q.argmax()]-pd.Timestamp(r.target_utc)).total_seconds()/3600,minimum_native_flow_cms=audit['minimum_native_flow_cms'],two_highest_local_peaks=[dict(time_utc=times.iloc[j].isoformat(),q_cms=float(q[j])) for j in ids]);records.append(record)
 fig.add_trace(go.Scatter(x=times,y=q,name=label+' n='+format(v.n_manning,'.12f')))
fig.add_vline(x=pd.Timestamp(r.target_utc).isoformat(),line=dict(color='orange',dash='dash'))
fig.update_layout(template='plotly_white',title='54: almost unchanged hydrograph, different global maximum',xaxis=dict(title='UTC',range=['2023-06-21','2023-06-24']),yaxis_title='Discharge (m³/s)',legend=dict(orientation='h',y=-.2))
fig.write_html(cal.OUT/'case54_peak_switch.html',include_plotlyjs=True)
result=dict(records=records,n_difference=b.n_manning-a.n_manning,global_peak_jump_h=(pd.Timestamp(records[1]['peak_utc'])-pd.Timestamp(records[0]['peak_utc'])).total_seconds()/3600,max_wave_difference_cms=float(np.max(abs(waves[1]-waves[0]))),wave_difference_pct_inlet_amplitude=float(100*np.max(abs(waves[1]-waves[0]))/np.ptp(flow)),interpretation='The argmax switches between two competing crests. The waveform remains nearly unchanged; this is not evidence of routing instability. Selected n near this switch is not a robust unique roughness estimate.')
(cal.OUT/'case54_peak_switch.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
