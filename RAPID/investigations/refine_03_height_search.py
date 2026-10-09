"""Refine a beta interval adjacent to a discontinuous timing-fit branch for 03."""
from pathlib import Path
import sys,json,hashlib,shutil
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'RAPID/investigations'),str(ROOT),str(ROOT/'RAPID/src')]
from calibrate_common_grid_height import feature,SOURCE
from rapid_tools import cunge
from rapid_tools.common_grid import route_interval_means

def main():
 dest=Path(json.loads((ROOT/'RAPID/outputs/common_grid_height_calibration_20261008/latest.json').read_text())['directory']);out=dest/'03'
 old=json.loads((out/'result.json').read_text());review=next(r for r in json.loads((dest/'timing_targets.json').read_text())['records'] if r['example']=='03')
 comp=pd.read_csv(SOURCE/'baseline/RAPID/outputs/cunge_calibration_reviewed_20261006/comparison.csv',dtype={'example':str});row=comp[(comp.example=='03')&(comp.method=='cunge')].iloc[0]
 grid=pd.read_parquet(SOURCE/'baseline'/Path(row.output_dir).relative_to(ROOT)/'frozen_grid.parquet');data=pd.read_parquet(SOURCE/'03/60s.parquet')
 meta=next(r for r in json.loads((SOURCE/'03/60s.json').read_text()) if r['method']=='cunge');valid=data.downstream_cms.notna().to_numpy();tt=data.time_utc.astype('int64').to_numpy()[valid]/1e9;goal=pd.Timestamp(review['target_utc']).timestamp();target=old['height_low_cms'];dt=meta['dt_s'];q0=meta['initial_cms']
 cache={};trials=[];candidates=[]
 def evaluate(beta,n):
  key=(float(beta),float(n))
  if key in cache:return cache[key]
  f=cunge.parameters(grid,n,method='cunge');f['rapid_x']=.5-beta*(.5-f.rapid_x)
  q,a=route_interval_means(f,data.forcing_cms.to_numpy(),int(round(60/dt)),dt,initial_flow=q0);assert a['minimum_native_flow_cms']>=0
  tm,h,kind,lobes=feature(q[valid],tt,'03',review)
  r=dict(beta=float(beta),n_manning=float(n),timing_error_minutes=(tm-goal)/60,peak_cms=h,feature_kind=kind,feature_utc=pd.Timestamp(tm,unit='s',tz='UTC').isoformat(),feature_lobes_utc=[pd.Timestamp(x,unit='s',tz='UTC').isoformat() for x in lobes],minimum_native_flow_cms=a['minimum_native_flow_cms'])
  cache[key]=(r,q);trials.append(r);return r,q
 def fit(beta):
  ns=np.unique(np.r_[.01,.04,.055,.065,.07,.085,.1]);rows=[]
  for n in ns:rows.append(evaluate(beta,n)[0])
  for a,b in zip(ns[:-1],ns[1:]):
   a=float(a);b=float(b);ra=evaluate(beta,a)[0];rb=evaluate(beta,b)[0]
   if ra['timing_error_minutes']*rb['timing_error_minutes']>=0:continue
   for _ in range(20):
    frac=np.clip(-ra['timing_error_minutes']/(rb['timing_error_minutes']-ra['timing_error_minutes']),.15,.85);mid=float(a+(b-a)*frac);r,q=evaluate(beta,mid);rows.append(r)
    if abs(r['timing_error_minutes'])<=.5:break
    if ra['timing_error_minutes']*r['timing_error_minutes']<=0:b=mid;rb=r
    else:a=mid;ra=r
  best=min(rows,key=lambda r:abs(r['timing_error_minutes']));candidates.append(best)
  print('REFINE03',beta,best['n_manning'],best['timing_error_minutes'],best['peak_cms'],flush=True)
  return best
 # Explore the interval excluded by the coarse search's timing-feasibility filter.
 for beta in [.8,.85,.9,.95,.975,.9875]:fit(beta)
 ordered=sorted(candidates,key=lambda r:r['beta'])
 for left,right in zip(ordered[:-1],ordered[1:]):
  if (left['peak_cms']-target)*(right['peak_cms']-target)>=0:continue
  a=left['beta'];b=right['beta']
  for _ in range(8):
   eligible=[r for r in candidates if abs(r['timing_error_minutes'])<=1]
   if eligible and min(abs(r['peak_cms']-target)/target for r in eligible)<=.001:break
   mid=(a+b)/2;r=fit(mid)
   if (left['peak_cms']-target)*(r['peak_cms']-target)<=0:b=mid
   else:a=mid;left=r
 eligible=[r for r in candidates if abs(r['timing_error_minutes'])<=1]
 best=min(eligible,key=lambda r:abs(r['peak_cms']-target)) if eligible else None
 (out/'refinement_trials.json').write_text(json.dumps(trials,indent=2))
 (out/'refinement_candidates.json').write_text(json.dumps(candidates,indent=2))
 if best is None or abs(best['peak_cms']-target)>=old['height_distance_cms']:
  print('No better feasible point fit');return
 backup=out/'pre_refinement';backup.mkdir(exist_ok=True)
 for name in ['result.json','wave.parquet']:shutil.copy2(out/name,backup/name)
 r,q=evaluate(best['beta'],best['n_manning']);result=dict(old);result.update(r)
 distance=abs(r['peak_cms']-target);result.update(height_distance_cms=distance,height_distance_fraction=distance/target,height_signed_residual=(r['peak_cms']-target)/target,point_height_within_numerical_01pct=distance/target<=.001,timing_within_one_minute=abs(r['timing_error_minutes'])<=1,trials=old['trials']+len(trials))
 manual=next(r for r in json.loads((dest/'height_targets.json').read_text())['records'] if r['example']=='03');obs=data.downstream_cms.to_numpy();result['shifted_rmse_cms']=float(np.sqrt(np.mean((q[valid]-obs[valid]-manual['downstream_display_offset_cms'])**2)));result['raw_rmse_cms']=float(np.sqrt(np.mean((q[valid]-obs[valid])**2)))
 result['search_refinement']='Additional beta values near the coarse timing discontinuity; prior result archived in pre_refinement.'
 checks=[];half=None
 for name,mult in [('half_dt',1),('half_dx_half_dt',2)]:
  g=grid.copy();g['count']*=mult;g['rapid_effective_length_m']=g.link_length_m/g['count'];f=cunge.parameters(g,best['n_manning'],method='cunge');f['rapid_x']=.5-best['beta']*(.5-f.rapid_x)
  qc,a=route_interval_means(f,data.forcing_cms.to_numpy(),int(round(120/dt)),dt/2,initial_flow=q0);assert a['minimum_native_flow_cms']>=0
  ref=q if half is None else half;tm,h,k,_=feature(qc[valid],tt,'03',review);rt,_,_,_=feature(ref[valid],tt,'03',review)
  checks.append(dict(control=name,max_event_difference_cms=float(np.max(abs(qc[valid]-ref[valid]))),feature_shift_minutes=(tm-rt)/60,feature_kind=k))
  if half is None:half=qc
 result['numerical_controls']=checks
 wave=data.copy();wave['calibrated_cunge_cms']=q;wave.to_parquet(out/'wave.parquet',index=False);(out/'result.json').write_text(json.dumps(result,indent=2))
 shutil.copy2(__file__,dest/'refinement_03_source.py')
 (dest/'refinement_03_provenance.json').write_text(json.dumps(dict(script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),reason='Coarse beta=1 timing branch jumps across nominal target; refine neighboring beta interval instead of claiming target infeasible.',original_result=old,selected=result),indent=2))
 print('SELECTED03',result['beta'],result['n_manning'],result['timing_error_minutes'],result['height_distance_cms'],checks,flush=True)
if __name__=='__main__':main()
