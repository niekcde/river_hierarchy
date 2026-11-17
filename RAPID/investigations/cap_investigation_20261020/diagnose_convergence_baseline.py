"""Read-only rerouting of representative resolution checks; retain diagnostics locally."""
import run_three_runs as r
import numpy as np,pandas as pd,json
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
OUTPUT=Path('/Volumes/PhD/river_hierarchy/three_run_comparison_20261028')
DEST=r.SETUP/'convergence_baseline_diagnosis'

def one(args):
 run,ex,state=args;r.init_worker(OUTPUT,'diagnostic-only')
 folder=OUTPUT/run/'states'/ex/state
 f=pd.read_parquet(folder/'parameters.parquet');meta=json.loads((folder/'completion.json').read_text());force=pd.read_parquet(r.CTX['cases'][ex]['forcing'])
 nb=int(force.phase.eq('spinup').sum());initial=float(force.forcing_cms.iloc[0]);t=force.time_utc.astype('int64').to_numpy()/1e9;rows=[]
 native=pd.read_parquet(folder/'metrics.parquet').set_index('window')
 for label,sub in [('half_dt',1),('half_dx_half_dt',2)]:
  g=f.copy();g['count']*=sub;g['rapid_effective_length_m']=g.link_length_m/g['count'];g=r.hydraulic_parameters(g,run,meta['n'],meta['beta'])
  q,a=r.solve_wave(g,force.forcing_cms.to_numpy()[nb:],meta['native_dt_seconds']/2,initial);q=np.r_[np.full(nb,initial),q]
  for _,w in r.CTX['windows'][r.CTX['windows'].example.eq(ex)].iterrows():
   tt=np.r_[w.rise_seconds,t[(t>w.rise_seconds)&(t<w.fall_seconds)],w.fall_seconds];excess=np.interp(tt,t,q)-w.baseline_cms
   rec=dict(example=ex,state=state,control=label,window=w.window,baseline_cms=w.baseline_cms,minimum_excess_cms=float(excess.min()),negative_samples=int((excess<0).sum()),maximum_excess_cms=float(excess.max()),negative_area_m3=float(np.trapz(np.maximum(-excess,0),tt)),min_native_flow=a['minimum_native_flow_cms'])
   try:
    m=r.state_metrics(t,q,w,state,[])
    rec.update(centroid_change_seconds=m['centroid_seconds']-native.loc[w.window,'centroid_seconds'],peak_change_seconds=m['peak_seconds']-native.loc[w.window,'peak_seconds'],peak_relative_change=abs(m['peak_cms']/native.loc[w.window,'peak_cms']-1))
   except ValueError as e:rec['error']=str(e)
   rows.append(rec)
  # Store full refined series only when its frozen window has negative excess.
  if any(x.get('error') for x in rows if x['control']==label):
   dest=DEST/ex/state;dest.mkdir(parents=True,exist_ok=True);pd.DataFrame(dict(time_utc=force.time_utc,q_outlet_cms=q)).to_parquet(dest/(label+'.parquet'),index=False)
 return rows

def main():
 DEST.mkdir(exist_ok=True);run=r.RUNS[0];d=pd.read_csv(OUTPUT/run/'responses.csv',dtype={'example':str});tasks=set()
 for ex,g in d[d.window.eq('0.5%')].groupby('example'):
  for state in ['S000_base','all_closed_units_collapsed','single_centerline',g.loc[g.centroid_shift_hours.idxmin(),'state'],g.loc[g.centroid_shift_hours.idxmax(),'state']]:tasks.add((run,ex,'single_centerline' if state=='fully_collapsed' else state))
 rows=[]
 with ProcessPoolExecutor(max_workers=2) as pool:
  futures={pool.submit(one,t):t for t in sorted(tasks)}
  for i,f in enumerate(as_completed(futures),1):
   rows.extend(f.result());pd.DataFrame(rows).to_csv(DEST/'checks.csv',index=False)
   print(i,len(tasks),futures[f],flush=True)
 df=pd.DataFrame(rows);print(df[df.minimum_excess_cms.lt(0)].to_string(index=False))
if __name__=='__main__':main()
