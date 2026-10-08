"""Native and half-step checks on all 15 completed hierarchy base states."""
from pathlib import Path
import sys,json
import pandas as pd,numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT),str(ROOT/'RAPID/src')]
from rapid_tools import cunge
OUT=ROOT/'RAPID/outputs/temporal_and_target_audit_20261008'
rows=[]
for ex in ['02','03','05','13','15','21','22','37','41','45','46','48','54','55','56']:
 root=ROOT/'RAPID/outputs/cunge_full_hierarchy_pilot/02' if ex=='02' else Path('/Volumes/PhD/river_hierarchy/cunge_full_hierarchy_runs')/ex;m=json.loads((root/'experiment.json').read_text());p=root/'states'/m['base_fingerprint'];f=pd.read_parquet(p/'parameters.parquet');h=pd.read_parquet(p/'outlet_hydrograph.parquet');flow=h.q_inflow_cms.to_numpy();dt=m['routing_dt_s'];ns=m['nsub'];q,a=cunge.route(f,flow,ns,dt,return_native=True);fine,af=cunge.route(f,flow,ns*2,dt/2,return_native=True);t=np.arange(1,len(q)+1)*dt;tf=np.arange(1,len(fine)+1)*dt/2;mean=q.reshape(-1,ns).mean(axis=1);assert np.max(abs(mean-h.q_outlet_cms))<1e-7*max(1,np.ptp(flow));assert min(a['minimum_native_flow_cms'],af['minimum_native_flow_cms'])>=-1e-9
 row=dict(example=ex,routing_dt_s=dt,report_step_s=m['forcing_step_s'],native_mean_match_cms=float(np.max(abs(mean-h.q_outlet_cms))),half_clock_wave_difference_pct_inlet=float(np.max(abs(q-fine[1::2]))/np.ptp(flow)*100),half_clock_global_peak_change_seconds=float(tf[fine.argmax()]-t[q.argmax()]),native_peak_utc=str(pd.Timestamp(h.time_utc.iloc[0])+pd.to_timedelta(t[q.argmax()],unit='s')),native_minus_saved_label_minutes=float((t[q.argmax()]-h.time_seconds.iloc[h.q_outlet_cms.argmax()])/60),minimum_native_flow_cms=a['minimum_native_flow_cms'],minimum_half_clock_flow_cms=af['minimum_native_flow_cms']);rows.append(row);pd.DataFrame(rows).to_csv(OUT/'production_base_clock_audit.csv',index=False);print(row,flush=True)
