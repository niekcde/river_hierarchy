"""Native base recurrence cross-check and selected paired output-grid refinement."""
from pathlib import Path
import sys,json
sys.path.insert(0,str(Path(__file__).resolve().parent))
from route_full_signed_validation import *

def base_check(ex):
 dest=OUT/ex;qa=json.loads((dest/'segment_validation.json').read_text());dt=qa['dt_s'];inv=pd.read_csv(dest/'inventory.csv');r=inv[(inv.family=='indep')&(inv.state=='S000_base')].iloc[0]
 native=(BASE/'common_caps/signed_coefficient_check'/f'{ex}_audit_outlet.parquet') if ex in ['11','46'] else dest/f'{ex}_audit_outlet.parquet'
 h=pd.read_parquet(native);f=pd.read_parquet(r.frame);rows=[]
 for step in [120.,60.]:
  t,inp,sp,nfft,q0,qe,_=grid_input(ex,r.max_travel_seconds,step);q=q0+irfft(sp*response(f,dt,nfft,step),nfft)[:len(t)]
  mask=(t>=h.time_seconds.min())&(t<=h.time_seconds.max());old=np.interp(t[mask],h.time_seconds,h.actual)
  peak=quadratic_peak(pd.DataFrame(dict(time_seconds=t,outlet_cms=q)),'outlet_cms');op=quadratic_peak(h,'actual')
  rows.append(dict(example=ex,dt_s=dt,reporting_step_s=step,max_wave_error_fraction=float(max(abs(q[mask]-old))/np.ptp(inp)),peak_error_s=peak-op))
 print('NATIVE CHECK',ex,rows,flush=True);return rows

def pair_check(ex):
 dest=OUT/ex;inv=pd.read_csv(dest/'inventory.csv');d=pd.read_csv(dest/'comparisons.csv');key=d.groupby('fingerprint').timing_change_minutes.agg(lambda a:a.max()-a.min()).idxmax();base=inv[(inv.family=='indep')&(inv.state=='S000_base')].iloc[0]
 rows=[];cases=[base,inv[inv.fingerprint==key].iloc[0]];travel=max(r.max_travel_seconds for r in cases)
 for step in [120.,30.]:
  t,inp,sp,nfft,q0,qe,_=grid_input(ex,travel,step)
  for dt in sorted(set(list(d.dt_s.unique())+[5])):
   metrics=[]
   for r in cases:
    f=pd.read_parquet(r.frame);q=q0+irfft(sp*response(f,dt,nfft,step),nfft)[:len(t)];metrics.append(summarize(t,q,inp,q0,qe))
   rows.append(dict(example=ex,fingerprint=key,step_s=step,dt_s=int(dt),timing_change_minutes=(metrics[1]['peak_time_s']-metrics[0]['peak_time_s'])/60))
 print('PAIR REFINEMENT',ex,rows,flush=True);return rows

if __name__=='__main__':
 from concurrent.futures import ProcessPoolExecutor
 with ProcessPoolExecutor(max_workers=2) as pool:
  rows=list(pool.map(base_check,EXAMPLES))
 pd.DataFrame([r for group in rows for r in group]).to_csv(OUT/'all_base_native_crosscheck.csv',index=False)
 rows=[]
 for ex in ['21','56']:rows+=pair_check(ex)
 pd.DataFrame(rows).to_csv(OUT/'selected_pair_refinement.csv',index=False)
