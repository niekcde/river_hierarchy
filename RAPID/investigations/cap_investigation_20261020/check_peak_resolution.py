"""Targeted crest-resolution diagnosis. Saved production results stay read-only."""
from pathlib import Path
import json,math
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
import pandas as pd
from scipy.signal import find_peaks
import run_three_runs as r
from convergence_moments import metrics

OUTPUT=Path('/Volumes/PhD/river_hierarchy/three_run_comparison_20261028')
DEST=r.SETUP/'peak_resolution_diagnosis'
CASES={'03':['single_centerline'],'13':['single_centerline'],'41':['single_centerline'],
       '45':['single_centerline'],'54':['all_closed_units_collapsed','single_centerline']}


def crest(q,t,base):
 i=int(np.argmax(q));peak=float(q[i]);excess=peak-base;floor=peak-1e-4*excess
 left=i;right=i
 while left>0 and q[left-1]>=floor:left-=1
 while right<len(q)-1 and q[right+1]>=floor:right+=1
 fractional=float(t[i]);curvature=np.nan
 if 0<i<len(q)-1:
  curvature=float(q[i-1]-2*q[i]+q[i+1])
  if curvature<0:
   offset=.5*(q[i-1]-q[i+1])/curvature
   if abs(offset)<=1:fractional+=float(offset*(t[i+1]-t[i]))
 peaks,_=find_peaks(q,prominence=max(excess*.001,1e-9),distance=30)
 tops=sorted(q[peaks],reverse=True)
 return dict(peak_seconds=float(t[i]),peak_cms=peak,quadratic_peak_seconds=fractional,
   contiguous_crest_width_001pct_excess_min=float((t[right]-t[left])/60),
   one_minute_curvature_cms=curvature,
   second_lobe_gap_pct_excess=float((peak-tops[1])/excess*100) if len(tops)>1 else None)


def one(args):
 ex,state,sub=args;r.init_worker(OUTPUT,'diagnostic-only');folder=OUTPUT/r.RUNS[0]/'states'/ex/state
 f=pd.read_parquet(folder/'parameters.parquet');rec=json.loads((folder/'completion.json').read_text())
 force=pd.read_parquet(r.CTX['cases'][ex]['forcing']);windows=r.CTX['windows'][r.CTX['windows'].example.eq(ex)]
 # Route all antecedent input, but omit tail after the latest reviewed window.
 end=float(windows.fall_seconds.max())+120
 force=force[(force.time_utc.astype('int64')/1e9)<=end].copy()
 nb=int(force.phase.eq('spinup').sum());initial=float(force.forcing_cms.iloc[0]);t=force.time_utc.astype('int64').to_numpy()/1e9
 dest=DEST/ex/state;dest.mkdir(parents=True,exist_ok=True);wavepath=dest/f'resolution_{sub}.parquet'
 if wavepath.exists():wave=pd.read_parquet(wavepath);q=wave.q_outlet_cms.to_numpy();assert np.array_equal(wave.time_utc,force.time_utc)
 elif sub==1:
  saved=pd.read_parquet(folder/'outlet.parquet');q=saved.q_outlet_cms.to_numpy()[:len(force)]
  pd.DataFrame(dict(time_utc=force.time_utc,q_outlet_cms=q)).to_parquet(wavepath,index=False)
 else:
  f['count']*=sub;f['rapid_effective_length_m']=f.link_length_m/f['count'];f=r.hydraulic_parameters(f,r.RUNS[0],rec['n'],rec['beta'])
  bound=float((2*f.rapid_k*(1-f.rapid_x)).min());dt=30/math.ceil(30/min(rec['native_dt_seconds']/sub,.99*bound))
  q,a=r.solve_wave(f,force.forcing_cms.to_numpy()[nb:],dt,initial);q=np.r_[np.full(nb,initial),q]
  pd.DataFrame(dict(time_utc=force.time_utc,q_outlet_cms=q)).to_parquet(wavepath,index=False)
 w=windows[windows.window.eq('0.5%')].iloc[0];mask=(t>=w.rise_seconds)&(t<=w.fall_seconds)
 c=crest(q[mask],t[mask],w.baseline_cms)
 m=metrics(ex,state,force,q,windows);centroid=float(m.loc[m.window.eq('0.5%'),'centroid_seconds'].iloc[0])
 result=dict(example=ex,state=state,subdivision=sub,centroid_seconds=centroid,baseline_cms=w.baseline_cms,**c)
 (dest/f'resolution_{sub}.json').write_text(json.dumps(result,indent=2)+'\n')
 return result


def report():
 rows=[json.loads(p.read_text()) for p in DEST.glob('*/*/resolution_*.json')];d=pd.DataFrame(rows);d.to_csv(DEST/'crest_results.csv',index=False)
 comparisons=[]
 for ex,states in CASES.items():
  for state in states:
   g=d[(d.example==ex)&(d.state==state)].set_index('subdivision');p=d[(d.example==ex)&(d.state=='S000_base')].set_index('subdivision')
   for sub in sorted(set(g.index)&set(p.index)):
    comparisons.append(dict(example=ex,state=state,subdivision=sub,
     peak_contrast_min=(g.loc[sub,'peak_seconds']-p.loc[sub,'peak_seconds'])/60,
     quadratic_peak_contrast_min=(g.loc[sub,'quadratic_peak_seconds']-p.loc[sub,'quadratic_peak_seconds'])/60,
     centroid_contrast_min=(g.loc[sub,'centroid_seconds']-p.loc[sub,'centroid_seconds'])/60))
 contrasts=pd.DataFrame(comparisons);contrasts.to_csv(DEST/'contrasts.csv',index=False)
 detail=[]
 for (ex,state),g in d.groupby(['example','state']):
  for lo,hi in [(1,2),(2,4),(4,8)]:
   if not {lo,hi}<=set(g.subdivision):continue
   a=pd.read_parquet(DEST/ex/state/f'resolution_{lo}.parquet');b=pd.read_parquet(DEST/ex/state/f'resolution_{hi}.parquet')
   av=a.q_outlet_cms.to_numpy();bv=b.q_outlet_cms.to_numpy();t=a.time_utc.astype('int64').to_numpy()/1e9
   ga=g[g.subdivision.eq(lo)].iloc[0];gb=g[g.subdivision.eq(hi)].iloc[0];excess=gb.peak_cms-gb.baseline_cms
   detail.append(dict(example=ex,state=state,coarse=lo,fine=hi,peak_time_change_min=(gb.peak_seconds-ga.peak_seconds)/60,
      quadratic_peak_time_change_min=(gb.quadratic_peak_seconds-ga.quadratic_peak_seconds)/60,
      max_wave_difference_cms=float(abs(av-bv).max()),max_wave_difference_pct_excess=float(abs(av-bv).max()/excess*100),
      fine_deficit_at_coarse_peak_cms=float(gb.peak_cms-np.interp(ga.peak_seconds,t,bv)),
      fine_deficit_at_coarse_peak_pct_excess=float((gb.peak_cms-np.interp(ga.peak_seconds,t,bv))/excess*100)))
 pd.DataFrame(detail).to_csv(DEST/'wave_differences.csv',index=False)
 print(contrasts.to_string(index=False),flush=True)


def main():
 import argparse
 p=argparse.ArgumentParser();p.add_argument('--levels',nargs='+',type=int,default=[1,2,4]);args=p.parse_args()
 DEST.mkdir(exist_ok=True)
 tasks=[(ex,state,level) for ex,states in CASES.items() for state in ['S000_base',*states] for level in args.levels]
 with ProcessPoolExecutor(max_workers=2) as pool:
  fs={pool.submit(one,a):a for a in tasks}
  for i,f in enumerate(as_completed(fs),1):f.result();print(i,len(tasks),fs[f],flush=True)
 report()
if __name__=='__main__':main()
