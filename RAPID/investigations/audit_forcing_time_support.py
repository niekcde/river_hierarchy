"""Controlled timing sensitivity: unchanged hydraulics, explicit observation integration."""
from pathlib import Path
import sys,json
import numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT),str(ROOT/'RAPID/src')]
from rapid_tools import cunge
from rapid_tools.time_support import point_series_interval_means
OUT=ROOT/'RAPID/outputs/temporal_and_target_audit_20261008';records=[]
for r in pd.read_csv(ROOT/'RAPID/outputs/cunge_calibration_reviewed_20261006/comparison.csv',dtype={'example':str}).sort_values(['example','method']).itertuples():
 p=Path(r.output_dir);h=pd.read_parquet(p/f'{r.method}_best_wave.parquet');obs=pd.read_parquet(p/'observed_upstream.parquet');start=pd.Timestamp(h.time_utc.iloc[0]);ot=(pd.to_datetime(obs.time,utc=True)-start).dt.total_seconds().to_numpy();oq=obs.discharge_cms.to_numpy();f=cunge.parameters(pd.read_parquet(p/'frozen_grid.parquet'),r.n_manning,method=r.method);dt=float(r.dt_s);ns=int(r.nsub);step=float(r.forcing_step_s);target=pd.Timestamp(r.target_utc)
 for policy in ['time_weighted_existing_bins','time_weighted_native_bins']:
  bin_step=step if policy=='time_weighted_existing_bins' else dt;bins=len(h) if policy=='time_weighted_existing_bins' else len(h)*ns;edges=np.arange(bins+1)*bin_step;flow=point_series_interval_means(ot,oq,edges,allow_endpoint_hold=True);q,a=cunge.route(f,flow,ns if policy=='time_weighted_existing_bins' else 1,dt,return_native=True);peak=(q.argmax()+1)*dt
  records.append(dict(example=r.example,method=r.method,policy=policy,forcing_interval_s=bin_step,native_dt_s=dt,peak_utc=str(start+pd.to_timedelta(peak,unit='s')),error_min=(start+pd.to_timedelta(peak,unit='s')-target).total_seconds()/60,minimum_native_flow_cms=a['minimum_native_flow_cms'],initial_flow_cms=float(flow[0]),previous_initial_flow_cms=float(h.q_inflow_cms.iloc[0]),maximum_observation_gap_min=float(np.max(np.diff(ot))/60),observation_median_step_min=float(np.median(np.diff(ot))/60),observation_interpretation='point samples, linear between samples, constant beyond record',initialization='equilibrium at first new forcing-bin mean',hydraulics_changed=False))
 pd.DataFrame(records).to_csv(OUT/'forcing_time_support_sensitivity.csv',index=False);print(r.example,r.method,[(z['policy'],round(z['error_min'],3)) for z in records[-2:]],flush=True)
