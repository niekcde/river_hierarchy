"""Separate conditional timing recalibration: n only, beta fixed at active selection."""
import ast
import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from audit import ROOT,OUT,EXT,read,save,sha,uncapped_grid,cunge,shared_clock,SOURCES
from rapid_tools.common_grid import route_interval_means
BUNDLE=ROOT/'RAPID/outputs/bounded_beta_calibration_20261009'
# Reuse the exact archived calibration crest operator (including example54), not response definitions.
source=(BUNDLE/'source_code/calibrate_common_grid_height.py').read_text();tree=ast.parse(source);fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='feature');exec(compile(ast.Module(body=[fn],type_ignores=[]),str(BUNDLE/'source_code/calibrate_common_grid_height.py'),'exec'))

def main():
 dest=OUT/'conditional_recalibration';dest.mkdir(exist_ok=True);plan=read(EXT/'routing_plan.json');protocol=read(BUNDLE/'protocol.json');src=ROOT/protocol['source_directory'];targets=read(BUNDLE/'timing_targets.json');heights=read(BUNDLE/'height_targets.json');rows=[]
 protected={str(p):sha(p) for p in BUNDLE.rglob('*') if p.is_file()};save(OUT/'protected_active_calibration_before.json',protected)
 for case in plan['cases']:
  ex=case['example'];savedpath=dest/f'{ex}.json'
  if savedpath.exists():rows.append(read(savedpath));continue
  f=read(next(j for j in case['jobs'] if j['state_id']=='S000_base')['parameters']);selection=read(BUNDLE/ex/'result.json');assert np.isclose(selection['n_manning'],f.n_manning.iloc[0]) and np.isclose(selection['beta'],f.diffusion_multiplier.iloc[0]);oldn=float(f.n_manning.iloc[0]);beta=float(f.diffusion_multiplier.iloc[0]);hascap=bool((f.rapid_celerity_mps_raw>f.celerity_max_mps).any())
  if not hascap:
   result=dict(example=ex,status='no_parent_cap_active_retained',old_n=oldn,new_n=oldn,beta=beta,refitted=False);save(savedpath,result);rows.append(result);continue
  grid=uncapped_grid(f);data=read(src/ex/'60s.parquet');meta=next(r for r in read(src/ex/'60s.json') if r['method']=='cunge');review=next(r for r in targets['records'] if r['example']==ex);height=next(r for r in heights['records'] if r['example']==ex);valid=data.downstream_cms.notna().to_numpy();times=data.time_utc.astype('int64').to_numpy()/1e9;goal=pd.Timestamp(review['target_utc']).timestamp();clock=selection.get('native_dt_s') or meta['dt_s'];dt=shared_clock([cunge.parameters(grid,n,beta=beta) for n in [.01,.1]],[30,60],clock);cache={};trials=[]
  def run(n):
   n=float(n)
   if n in cache:return cache[n]
   g=cunge.parameters(grid,n,beta=beta);q,a=route_interval_means(g,data.forcing_cms.to_numpy(),round(60/dt),dt,initial_flow=meta['initial_cms']);tm,h,kind,_=feature(q[valid],times[valid],ex,review)
   lo=height['low_cms'];hi=height['high_cms']
   if lo is None:lo=hi=height['height_target_nominal_cms']
   rec=dict(n=n,timing_error_minutes=(tm-goal)/60,peak_cms=h,height_distance_cms=max(lo-h,0.,h-hi),minimum_native_flow_cms=a['minimum_native_flow_cms'],feature_kind=kind,valid=bool(a['minimum_native_flow_cms']>=-1e-8));trials.append(rec);cache[n]=rec;return rec
  original=run(oldn);candidates=[original]
  if abs(original['timing_error_minutes'])>1 or not original['valid']:
   # Local bracket first, expand to original declared n bounds only if needed.
   ns=sorted(set([max(.01,oldn*.8),oldn,min(.1,oldn*1.25)]))
   for n in ns:candidates.append(run(n))
   brackets=[(a,b) for a,b in zip(ns[:-1],ns[1:]) if run(a)['valid'] and run(b)['valid'] and run(a)['timing_error_minutes']*run(b)['timing_error_minutes']<=0]
   if not brackets:
    ns=sorted(set([.01,.0175,.025,.04,.07,.1,oldn]))
    for n in ns:candidates.append(run(n))
    brackets=[(a,b) for a,b in zip(ns[:-1],ns[1:]) if run(a)['valid'] and run(b)['valid'] and run(a)['timing_error_minutes']*run(b)['timing_error_minutes']<=0]
   for a,b in brackets:
    for _ in range(18):
     ea,eb=run(a)['timing_error_minutes'],run(b)['timing_error_minutes'];fraction=.5 if ea==eb else float(np.clip(-ea/(eb-ea),.15,.85));n=a+fraction*(b-a);rec=run(n);candidates.append(rec)
     if not rec['valid']:break
     if abs(rec['timing_error_minutes'])<=.5:break
     if ea*rec['timing_error_minutes']<=0:b=n
     else:a=n
  feasible=[r for r in candidates if r['valid']];best=min(feasible,key=lambda r:(abs(r['timing_error_minutes']),abs(r['n']-oldn)))
  g=cunge.parameters(grid,best['n'],beta=beta);q,a=route_interval_means(g,data.forcing_cms.to_numpy(),round(60/dt),dt,initial_flow=meta['initial_cms']);pd.DataFrame(dict(time_utc=data.time_utc,diagnostic_outlet_cms=q)).to_parquet(dest/f'{ex}_wave.parquet',index=False)
  result=dict(example=ex,status='conditional_timing_refit',old_n=oldn,new_n=best['n'],beta=beta,refitted=True,old_calibration_timing_error_minutes=selection['timing_error_minutes'],old_calibration_height_distance_cms=selection['height_distance_cms'],uncapped_old_n=original,selected=best,dt_seconds=dt,cells=int(grid['count'].sum()),trials=trials,qualification='Beta fixed; not full joint n/beta optimization. Same archived calibration targets/operator. No production changes.')
  save(savedpath,result);rows.append(result);print(ex,'old n',oldn,'new n',best['n'],'timing',best['timing_error_minutes'],'height gap',best['height_distance_cms'],flush=True)
 save(dest/'results.json',rows);save(dest/'sources.json',SOURCES);print('Conditional calibration checks finished',flush=True)
if __name__=='__main__':main()
