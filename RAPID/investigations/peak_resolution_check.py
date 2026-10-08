"""Targeted output-resolution versus routing-clock audit; production stays read-only."""
from pathlib import Path
import sys,json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from check_fractional_routing_clocks import cascade,peak,sparse_benchmark
ROOT=Path(__file__).resolve().parents[2]
SOURCE=(ROOT/'network_variants/output').resolve().parent/'hydraulic_refresh/shared-head-anchored-depth-signed-inflow-20261003'
OUT=ROOT/'RAPID/validation/peak_resolution_20261005'

def run():
 OUT.mkdir(parents=True,exist_ok=True)
 df=pd.read_parquet(ROOT/'RAPID/outputs/refreshed_analysis_20261005/analysis.parquet');df=df[~df.technical_alias]
 records=[];selections=[];fig,axes=plt.subplots(3,2,figsize=(12,10),constrained_layout=True)
 for ex,axs in zip(['03','15','46'],axes):
  inv=pd.read_csv(SOURCE/ex/'inventory.csv');inv=inv[inv.status.eq('ready')].drop_duplicates('fingerprint').set_index('fingerprint')
  base=inv[(inv.family=='indep')&(inv.state=='S000_base')].iloc[0];basefp=base.name
  g=df[df.example_id.eq(ex)].drop_duplicates('fingerprint');z=g[g.delta_peak_time_hours.eq(0)].sort_values('centroid50_shift_hours',key=abs)
  chosen=[(z.iloc[len(z)//2],'zero shift / median centroid difference'),(z.iloc[-1],'zero shift / largest centroid difference'),(g[g.delta_peak_time_hours.ne(0)].sort_values('delta_peak_time_hours',key=abs).iloc[0],'smallest nonzero control')]
  h=pd.read_csv(SOURCE/ex/basefp/'run/outlet_hydrograph.csv');flow=h.q_inflow_cms.to_numpy();q0=flow[0];start=int(np.flatnonzero(flow!=q0)[0]);forcing_dt=float(np.median(np.diff(h.time_seconds)));t0=h.time_seconds.iloc[start];dt=json.loads((SOURCE/ex/basefp/'completed.json').read_text())['routing_dt_s'];waves={}
  states=[(basefp,'S000_base','base')]+[(r.fingerprint,r.state_id,reason) for r,reason in chosen]
  for fp,state,reason in states:
   f=pd.read_parquet(inv.loc[fp,'frame']);saved=pd.read_csv(SOURCE/ex/fp/'run/outlet_hydrograph.csv').q_outlet_cms.to_numpy()[start:]
   selections.append(dict(example=ex,state=state,fingerprint=fp,reason=reason))
   for divisor in [1,2]:
    step=dt/divisor;nsub=round(forcing_dt/step)
    q,_=cascade(f,flow[start:],nsub,step,q0,return_native=True);t=t0+np.arange(1,len(q)+1)*step;averaged=q.reshape(-1,nsub).mean(axis=1)
    if divisor==1:np.testing.assert_allclose(averaged,saved,rtol=1e-6,atol=1e-5)
    if fp==basefp:
     sparse,_,_=sparse_benchmark(f,flow[start:start+8],nsub,step,q0);np.testing.assert_allclose(averaged[:8],sparse,rtol=1e-9,atol=1e-7)
    tm=peak(t,q);waves[fp,divisor]=(t,q,tm)
    basepeak=waves[basefp,divisor][2]
    records.append(dict(example=ex,state=state,fingerprint=fp,reason=reason,dt_s=step,divisor=divisor,native_peak_s=tm,native_sampled_peak_s=float(t[q.argmax()]),native_delta_peak_s=tm-basepeak,saved_delta_peak_s=0 if fp==basefp else float(g[g.fingerprint.eq(fp)].delta_peak_time_hours.iloc[0]*3600),coarse_parabolic_delta_s=peak(h.time_seconds.to_numpy()[start:],saved)-peak(h.time_seconds.to_numpy()[start:],h.q_outlet_cms.to_numpy()[start:]),coarse_match_max_cms=float(abs(averaged-saved).max()),native_min_cms=float(q.min())))
    pd.DataFrame(records).to_csv(OUT/'runs.csv',index=False);print(ex,state,divisor,'delta_s',tm-basepeak,flush=True)
   if fp!=basefp and reason.startswith('zero'):
    ax=axs[0 if 'median' in reason else 1];tp,qp,_=waves[basefp,2];tc,qc,_=waves[fp,2];p0=waves[basefp,2][2];mask=abs(tp-p0)<2*forcing_dt
    ax.plot((tp[mask]-p0)/60,qp[mask],label='Parent');ax.plot((tc[mask]-p0)/60,qc[mask],label='Child');ax.set(title=f'{ex}: {reason}',xlabel='Minutes from parent peak',ylabel='Discharge (m³/s)');ax.legend()
 pd.DataFrame(selections).to_csv(OUT/'selection.csv',index=False);fig.savefig(OUT/'peak_closeups.png',dpi=150)
 (OUT/'provenance.json').write_text(json.dumps(dict(source=str(SOURCE),scope='Three examples, two zero-shift children and one nonzero control each, shared bases. Same forcing held within original intervals. Native and half-native routing clocks; no geometry/hydraulic changes. Peaks use three-point quadratic interpolation; sampled native peaks also retained. Coarse means checked against production; initial probe cross-checked with sparse engine.',production_changed=False),indent=2))
if __name__=='__main__':run()
