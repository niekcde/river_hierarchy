"""Read-only replay of all reviewed base calibrations on explicit physical clocks."""
from pathlib import Path
import sys,json,time
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'RAPID/src')]
from rapid_tools import cunge
OUT=ROOT/'RAPID/outputs/temporal_and_target_audit_20261008'
CAL=ROOT/'RAPID/outputs/cunge_calibration_reviewed_20261006/comparison.csv'
def utc(start,seconds):return (start+pd.to_timedelta(float(seconds),unit='s')).isoformat()
def score(t,q,g,start):
 gt=(pd.to_datetime(g.time,utc=True)-start).dt.total_seconds().to_numpy();v=g.discharge_cms.to_numpy();valid=(gt>=t[0])&(gt<=t[-1]);pred=np.interp(gt[valid],t,q);v=v[valid]
 norm=lambda a:(a-a.min())/max(np.ptp(a),1e-12)
 return float(np.sqrt(np.mean((norm(pred)-norm(v))**2)))
def main():
 OUT.mkdir(exist_ok=True,parents=True);records=[];forcing=[]
 for r in pd.read_csv(CAL,dtype={'example':str}).sort_values(['example','method']).itertuples():
  folder=Path(r.output_dir);m=json.loads((folder/'case.json').read_text());h=pd.read_parquet(folder/f'{r.method}_best_wave.parquet');g=pd.read_parquet(folder/'observed_downstream.parquet');up=pd.read_parquet(folder/'observed_upstream.parquet');grid=pd.read_parquet(folder/'frozen_grid.parquet');f=cunge.parameters(grid,r.n_manning,method=r.method)
  start=pd.Timestamp(h.time_utc.iloc[0]);t=h.time_seconds.to_numpy();step=float(r.forcing_step_s);dt=float(r.dt_s);nsub=int(r.nsub);flow=h.q_inflow_cms.to_numpy();saved=h.q_outlet_cms.to_numpy();tic=time.monotonic()
  q,a=cunge.route(f,flow,nsub,dt,return_native=True);nt=(np.arange(len(q))+1)*dt;means=q.reshape(-1,nsub).mean(axis=1);match=float(abs(means-saved).max());assert match<1e-7*max(1,np.ptp(flow));assert a['minimum_native_flow_cms']>=-1e-9
  ip=int(q.argmax());sp=int(saved.argmax());target=pd.Timestamp(r.target_utc);gt=(pd.to_datetime(g.time,utc=True)-start).dt.total_seconds().to_numpy();mask=(gt>=0)&(gt<=nt[-1]);validgt=gt[mask];atg=np.interp(validgt,np.r_[0,nt],np.r_[flow[0],q]);gp=int(atg.argmax());mid=t+step/2
  record=dict(example=r.example,method=r.method,scope='calibration_base',forcing_step_s=step,native_dt_s=dt,nsub=nsub,target_utc=target.isoformat(),gauge_step_s=float(np.median(np.diff(gt))),gauge_max_gap_s=float(np.max(np.diff(gt))),old_peak_label_utc=utc(start,t[sp]),midpoint_mean_peak_utc=utc(start,mid[sp]),native_peak_utc=utc(start,nt[ip]),native_on_gauge_peak_utc=utc(start,validgt[gp]),old_error_min=(start+pd.to_timedelta(t[sp],unit='s')-target).total_seconds()/60,midpoint_mean_error_min=(start+pd.to_timedelta(mid[sp],unit='s')-target).total_seconds()/60,native_error_min=(start+pd.to_timedelta(nt[ip],unit='s')-target).total_seconds()/60,native_on_gauge_error_min=(start+pd.to_timedelta(validgt[gp],unit='s')-target).total_seconds()/60,native_minus_old_peak_min=(nt[ip]-t[sp])/60,midpoint_minus_native_peak_min=(mid[sp]-nt[ip])/60,peak_mean_reduction_pct=(q[ip]-saved[sp])/q[ip]*100,native_mean_match_cms=match,old_shape_rmse=score(t,saved,g,start),midpoint_shape_rmse=score(mid,saved,g,start),native_shape_rmse=score(np.r_[0,nt],np.r_[flow[0],q],g,start),tolerance_min=float(r.timing_resolution_h)*60,minimum_native_flow_cms=a['minimum_native_flow_cms'],runtime_s=time.monotonic()-tic)
  record.update(old_within_tolerance=abs(record['old_error_min'])<=record['tolerance_min'],native_within_tolerance=abs(record['native_error_min'])<=record['tolerance_min']);records.append(record)
  pd.DataFrame(dict(time_utc=start+pd.to_timedelta(validgt,unit='s'),observed_cms=g.discharge_cms.to_numpy()[mask],model_native_interpolated_cms=atg)).to_parquet(OUT/f'{r.example}_{r.method}_at_gauge_times.parquet',index=False)
  if r.method=='cunge':
   ut=(pd.to_datetime(up.time,utc=True)-start).dt.total_seconds().to_numpy();bins=np.floor(ut/step).astype(int);v=up.discharge_cms.to_numpy();b=pd.DataFrame({'bin':bins,'q':v}).groupby('bin').q.mean();b=b[(b.index>=0)&(b.index<len(flow))];complete=(b.index*step>=ut.min())&((b.index+1)*step<=ut.max());b=b[complete];err=abs(flow[b.index.to_numpy()]-b.to_numpy());forcing.append(dict(example=r.example,observed_upstream_step_s=float(np.median(np.diff(ut))),report_step_s=step,complete_nonempty_bins=len(b),unweighted_bin_mean_max_error_cms=float(err.max()),unweighted_bin_mean_rmse_cms=float(np.sqrt(np.mean(err**2))),error_pct_inlet_amplitude=float(err.max()/np.ptp(flow)*100),native_initial_time_utc=start.isoformat(),forcing_max_label_utc=utc(start,t[flow.argmax()]),forcing_max_midpoint_utc=utc(start,t[flow.argmax()]+step/2)))
  pd.DataFrame(records).to_csv(OUT/'all_calibration_clock_audit.csv',index=False);pd.DataFrame(forcing).to_csv(OUT/'forcing_bin_audit.csv',index=False);print(r.example,r.method,'native error min',round(record['native_error_min'],3),'match',match,flush=True)
if __name__=='__main__':main()
