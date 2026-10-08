"""Effect of removing only the common upper cap, same geometry and clocks."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from route_full_signed_validation import *

def run(ex):
 inv=pd.read_csv(OUT/ex/'inventory.csv');good=inv[inv.status=='ready'].drop_duplicates('fingerprint');base=inv[(inv.family=='indep')&(inv.state=='S000_base')].iloc[0]
 good=pd.concat([good[good.fingerprint==base.fingerprint],good[good.fingerprint!=base.fingerprint]])
 frames={r.fingerprint:pd.read_parquet(r.frame) for r in good.itertuples()}
 t,inp,sp,nfft,q0,qe,fdt=grid_input(ex,good.max_travel_seconds.max(),120.)
 upper=min(1.8*(f.rapid_effective_length_m/f.rapid_celerity_mps_raw.clip(lower=.05)).min() for f in frames.values())
 dt=max(i for i in range(1,fdt+1) if fdt%i==0 and i<.99*upper)
 rows=[];links=[]
 for policy in ['cap5','no_upper_cap']:
  cache={};bm=None
  for key,f0 in frames.items():
   f=f0.copy()
   if policy=='no_upper_cap':f['rapid_k']=f.rapid_effective_length_m/f.rapid_celerity_mps_raw.clip(lower=.05)
   q=q0+irfft(sp*response(f,dt,nfft,120.,cache),nfft)[:len(t)]
   m=summarize(t,q,inp,q0,qe)
   if key==base.fingerprint:
    bm=m;rise=.01*m['peak_excess'];fall=.01*(q.max()-qe);bd=duration(t,q,q0,qe,rise,fall)['duration_seconds']
    for r in f[f.rapid_celerity_mps_raw>5].itertuples():links.append(dict(example=ex,id_link=r.id_link,width=r.width,depth=r.depth_m,slope=r.slope,raw_celerity=r.rapid_celerity_mps_raw,length=r.link_length_m))
   d=duration(t,q,q0,qe,rise,fall)['duration_seconds']
   rows.append(dict(example=ex,fingerprint=key,policy=policy,dt=dt,base=key==base.fingerprint,peak_time_s=m['peak_time_s'],timing_change_minutes=(m['peak_time_s']-bm['peak_time_s'])/60,duration_change_pct=100*(d/bd-1),peak_excess_change_pct=100*(m['peak_excess']/bm['peak_excess']-1)))
  print('CAP CHECK',ex,policy,len(frames),flush=True)
 pd.DataFrame(rows).to_csv(OUT/ex/'cap_sensitivity.csv',index=False)
 pd.DataFrame(links).drop_duplicates().to_csv(OUT/ex/'base_upper_cap_links.csv',index=False)

if __name__=='__main__':
 from concurrent.futures import ProcessPoolExecutor
 with ProcessPoolExecutor(max_workers=2) as pool:list(pool.map(run,['22','46']))
