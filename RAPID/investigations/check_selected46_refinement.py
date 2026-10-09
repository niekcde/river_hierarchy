"""Resolve selected 46 spatial sensitivity; retain beta and original geometry."""
from pathlib import Path
import sys,json,shutil,hashlib
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'RAPID/investigations'),str(ROOT),str(ROOT/'RAPID/src')]
from calibrate_common_grid_height import feature,SOURCE
from rapid_tools import cunge
from rapid_tools.common_grid import route_interval_means
from rapid_tools.time_support import point_series_interval_means

def main():
 dest=Path(json.loads((ROOT/'RAPID/outputs/common_grid_height_calibration_20261008/latest.json').read_text())['directory']);out=dest/'46';selected=json.loads((out/'result.json').read_text());review=next(r for r in json.loads((dest/'timing_targets.json').read_text())['records'] if r['example']=='46')
 comp=pd.read_csv(SOURCE/'baseline/RAPID/outputs/cunge_calibration_reviewed_20261006/comparison.csv',dtype={'example':str});row=comp[(comp.example=='46')&(comp.method=='cunge')].iloc[0];grid=pd.read_parquet(SOURCE/'baseline'/Path(row.output_dir).relative_to(ROOT)/'frozen_grid.parquet')
 data=pd.read_parquet(SOURCE/'46/60s.parquet');meta=next(r for r in json.loads((SOURCE/'46/60s.json').read_text()) if r['method']=='cunge');q0=meta['initial_cms'];valid=data.downstream_cms.notna().to_numpy();tt=data.time_utc.astype('int64').to_numpy()[valid]/1e9;goal=pd.Timestamp(review['target_utc']).timestamp();beta=selected['beta'];n=selected['n_manning'];records=[]
 def run(n,mult,dt=5.,flow=None,initial=None):
  g=grid.copy();g['count']*=mult;g['rapid_effective_length_m']=g.link_length_m/g['count'];f=cunge.parameters(g,n,method='cunge');f['rapid_x']=.5-beta*(.5-f.rapid_x)
  q,a=route_interval_means(f,data.forcing_cms.to_numpy() if flow is None else flow,int(round(60/dt)),dt,initial_flow=q0 if initial is None else initial);assert a['minimum_native_flow_cms']>=0
  return q
 previous=None;waves={}
 for mult in [1,2,4,8]:
  q=run(n,mult);waves[mult]=q;tm,h,kind,_=feature(q[valid],tt,'46',review)
  rec=dict(subdivision_multiplier=mult,n=n,dt_s=5.,peak_cms=h,timing_error_minutes=(tm-goal)/60,max_difference_previous_cms=None if previous is None else float(np.max(abs(q[valid]-previous[valid]))));records.append(rec);previous=q;print('GRID46',rec,flush=True)
 # Keep this as convergence evidence first. A new n on a different numerical grid
 # is only adopted if it also has a stable peak under another refinement.
 mult=8;trials=[];cache={}
 def ev(n):
  if n in cache:return cache[n]
  q=run(n,mult);tm,h,kind,lobes=feature(q[valid],tt,'46',review);e=(tm-goal)/60;cache[n]=(e,q,h,tm);trials.append(dict(n=n,error_minutes=e,peak_cms=h));return cache[n]
 a=n*.98;b=n*1.02;ea=ev(a)[0];eb=ev(b)[0]
 if ea*eb<0:
  for _ in range(12):
   mid=float(a+np.clip(-ea/(eb-ea),.1,.9)*(b-a));e,q,h,tm=ev(mid)
   if abs(e)<=.5:break
   if ea*e<=0:b=mid;eb=e
   else:a=mid;ea=e
 best=min(cache,key=lambda v:abs(cache[v][0]));e,q,h,tm=cache[best]
 half=run(best,8,2.5);ht,_,_,_=feature(half[valid],tt,'46',review)
 finer=run(best,16,2.5);ft,fh,_,_=feature(finer[valid],tt,'46',review)
 controls=[dict(control='16x versus 8x cells at equal dt2.5',max_event_difference_cms=float(np.max(abs(finer[valid]-half[valid]))),feature_shift_minutes=(ft-ht)/60)]
 controls.append(dict(control='half_dt on 8x cells',max_event_difference_cms=float(np.max(abs(half[valid]-q[valid]))),feature_shift_minutes=(ht-tm)/60))
 print('FINALGRID46',best,e,h,controls,flush=True)
 results=dict(original_selected=selected,grid_controls=records,n_refit_trials=trials,refined_candidate=dict(n_manning=best,beta=beta,subdivision_multiplier=8,dt_s=5.,timing_error_minutes=e,peak_cms=h,controls=controls))
 if abs(e)<=1 and all(abs(r['feature_shift_minutes'])<=1 for r in controls):
  backup=out/'pre_spatial_refinement';backup.mkdir(exist_ok=True)
  for name in ['result.json','wave.parquet']:shutil.copy2(out/name,backup/name)
  result=dict(selected);result.update(n_manning=best,peak_cms=h,feature_utc=pd.Timestamp(tm,unit='s',tz='UTC').isoformat(),feature_lobes_utc=[pd.Timestamp(tm,unit='s',tz='UTC').isoformat()]*2,timing_error_minutes=e,height_distance_cms=abs(h-selected['height_low_cms']),height_distance_fraction=abs(h-selected['height_low_cms'])/selected['height_low_cms'],height_signed_residual=(h-selected['height_low_cms'])/selected['height_low_cms'],point_height_within_numerical_01pct=abs(h-selected['height_low_cms'])/selected['height_low_cms']<=.001,subdivision_multiplier=8,native_dt_s=5.,numerical_controls=controls,spatial_refinement_note='Selected 8x numerical subdivision after original-grid peak moved 10min under doubling. Physical reach lengths and geometry unchanged. n retimed at fixed beta; 16x control recorded.')
  target=next(r for r in json.loads((dest/'height_targets.json').read_text())['records'] if r['example']=='46');obs=data.downstream_cms.to_numpy();result['shifted_rmse_cms']=float(np.sqrt(np.mean((q[valid]-obs[valid]-target['downstream_display_offset_cms'])**2)));result['raw_rmse_cms']=float(np.sqrt(np.mean((q[valid]-obs[valid])**2)))
  wave=data.copy();wave['calibrated_cunge_cms']=q;wave.to_parquet(out/'wave.parquet',index=False);(out/'result.json').write_text(json.dumps(result,indent=2));results['adopted_as_review_candidate']=True
 else:results['adopted_as_review_candidate']=False
 # Actual antecedent sensitivity at the refined candidate, without another refit.
 antecedent=[]
 for days in [7,30]:
  points=pd.read_parquet(SOURCE/'46'/f'antecedent_{days}d_points.parquet');origin=data.interval_start_utc.iloc[0]-pd.Timedelta(days=days);t=(pd.to_datetime(points.time,utc=True)-origin).dt.total_seconds().to_numpy();v=points.discharge.to_numpy();edges=np.arange(len(data)+days*1440+1)*60.;flow=point_series_interval_means(t,v,edges,allow_endpoint_hold=True);flow[days*1440:]=data.forcing_cms
  qa=run(best,8,flow=flow,initial=float(np.interp(0,t,v)))[days*1440:];at,ah,_,_=feature(qa[valid],tt,'46',review)
  antecedent.append(dict(days=days,timing_error_minutes=(at-goal)/60,peak_cms=ah,max_event_difference_from_equilibrium_cms=float(np.max(abs(qa[valid]-q[valid])))));print('ANTECEDENT46',antecedent[-1],flush=True)
  pd.DataFrame(dict(time_utc=data.time_utc,cunge_cms=qa)).to_parquet(out/f'selected_antecedent_{days}d.parquet',index=False)
 results['antecedent_controls']=antecedent
 (out/'spatial_refinement_and_antecedent.json').write_text(json.dumps(results,indent=2));shutil.copy2(__file__,dest/'refinement_46_source.py')
 (dest/'refinement_46_provenance.json').write_text(json.dumps(dict(script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),reason='Original selected beta produces a 10-minute peak shift under cell refinement; test and refine before reporting.'),indent=2))
if __name__=='__main__':main()
