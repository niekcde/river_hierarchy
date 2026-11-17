"""Isolated spatial-grid tests; original hydraulics and production data untouched."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import time,json
import numpy as np,pandas as pd
import broad_study as b
from test_input_sensitivity48 import BASE,N,solve,summarize_initial
OUT=b.OUT/'investigation/segmentation48';OUT.mkdir(exist_ok=True)
def grid(f,variant):
 rows=[];virtual=int(min(f.u.min(),f.v.min(),0))-1
 for r in f.itertuples():
  count=int(np.ceil(r.link_length_m/50));lengths=np.full(count,r.link_length_m/count)
  if variant=='split79' and r.id_link==79:lengths=np.full(2,r.link_length_m/2)
  if variant=='shift161' and r.id_link==161:lengths=np.r_[lengths[0]/2,lengths[:-1],lengths[0]/2]
  if variant=='count161' and r.id_link==161:lengths=np.full(count+1,r.link_length_m/(count+1))
  edges=np.r_[0.,np.cumsum(lengths)];edges[-1]=r.link_length_m;heads=r.head_us+(r.head_ds-r.head_us)*edges/r.link_length_m;start=int(r.u)
  for j,length in enumerate(lengths):
   end=int(r.v) if j==len(lengths)-1 else virtual
   if j<len(lengths)-1:virtual-=1
   rows.append(dict(id_link=r.id_link,subsection=j+1,u=start,v=end,link_length_m=length,width=r.width,head_us=heads[j],head_ds=heads[j+1]));start=end
 cells=pd.DataFrame(rows)
 for r in f.itertuples():
  x=cells[cells.id_link==r.id_link];assert np.isclose(x.link_length_m.sum(),r.link_length_m,rtol=0,atol=1e-8);assert x.iloc[0].head_us==r.head_us and abs(x.iloc[-1].head_ds-r.head_ds)<1e-10;assert (x.width==r.width).all()
 assert cells.link_length_m.max()<=50
 return cells

def run(variant,cap):
 dest=OUT/f'{variant}_cap{cap:g}';dest.mkdir(exist_ok=True)
 if (dest/'result.json').exists():return json.loads((dest/'result.json').read_text())
 print('START',dest.name,flush=True);t0=time.monotonic();f=pd.read_parquet(BASE);cells=grid(f,variant);cells.to_parquet(dest/'cells.parquet',index=False);m=b.build(cells,N)
 force=b.forcing('48');origin=force.interval_start_utc.iloc[0];times=np.r_[(force.interval_start_utc-origin).dt.total_seconds(),(force.interval_end_utc.iloc[-1]-origin).total_seconds()];values=np.r_[force.forcing_cms,force.forcing_cms.iloc[-1]];Q=float(values[0]);h,q=b.initial_guess(f,cells,m,Q)
 # Equilibrate at normal CFL first: timestep cap does not change the steady equations.
 d=dict(m,h=h,q=q,times=np.array([0.,86400.]),forcing=np.array([Q,Q]),cfl=.35,max_dt=150.,sample_indices=np.array([0,1]));spins=[]
 for day in range(1,61):
  x=solve(d,dest,'spin');a=dict(zip(b.v.AUDIT,map(float,x['audit'])));spins.append(a);d.update(h=x['h'],q=x['q'])
  if a['final_node_imbalance_cms']/Q<1e-8:break
 else:raise RuntimeError('spin failed')
 initial=summarize_initial(cells,m,x['h'],x['q']);d.update(times=times,forcing=values,max_dt=cap,sample_indices=np.array([0,len(times)-1]));x=solve(d,dest,'event');a=dict(zip(b.v.AUDIT,map(float,x['audit'])));wave=force[['time_utc','phase']].copy();wave['q_outlet_cms']=np.diff(x['wave'][:,4])/np.diff(times);wave.to_parquet(dest/'wave.parquet',index=False);clip=np.flatnonzero(x['wave'][:,5]>1e-6)
 fixed=bool(a['steps']==(times[-1]-times[0])/cap and abs(a['min_step_s']-cap)<1e-8 and a['max_step_s']==cap)
 result=dict(variant=variant,max_dt=cap,cell_count=len(cells),minimum_length_m=float(cells.link_length_m.min()),maximum_length_m=float(cells.link_length_m.max()),n=N,cfl=.35,audit=a,first_clipping_sample_hours=None if not len(clip) else float(times[clip[0]]/3600),peak_outlet_cms=float(wave.q_outlet_cms.max()),verified_exact_constant_steps=fixed,spinup_days=day,spinup_audits=spins,initial_hydraulics=initial,seconds=time.monotonic()-t0,folder=str(dest))
 b.save(dest/'result.json',result);print('DONE',dest.name,'clip',a['clipping_volume_m3'],'depth',a['maximum_depth_m'],'fixed',fixed,flush=True);return result
if __name__=='__main__':
 jobs=[('baseline',150),('baseline',1),('split79',1),('split79',150),('shift161',1),('count161',1),('baseline',.5),('split79',.5),('shift161',.5),('count161',.5)]
 with ThreadPoolExecutor(max_workers=2) as pool:results=list(pool.map(lambda args:run(*args),jobs))
 b.save(OUT/'all_results.json',results)
