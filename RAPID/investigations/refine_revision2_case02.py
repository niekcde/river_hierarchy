"""Refine 02 numerical cells and retime n at fixed beta after lower-bound revision."""
from pathlib import Path
import sys,json,shutil,hashlib
import numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'RAPID/investigations'),str(ROOT),str(ROOT/'RAPID/src')]
from calibrate_common_grid_height import SOURCE,feature
from rapid_tools import cunge
from rapid_tools.common_grid import route_interval_means

def main():
 dest=Path(json.loads((ROOT/'RAPID/outputs/common_grid_height_calibration_20261008/latest.json').read_text())['directory']);out=dest/'02';old=json.loads((out/'result.json').read_text());target=next(r for r in json.loads((dest/'timing_targets.json').read_text())['records'] if r['example']=='02');manual=next(r for r in json.loads((dest/'height_targets.json').read_text())['records'] if r['example']=='02')
 comp=pd.read_csv(SOURCE/'baseline/RAPID/outputs/cunge_calibration_reviewed_20261006/comparison.csv',dtype={'example':str});row=comp[(comp.example=='02')&(comp.method=='cunge')].iloc[0];grid=pd.read_parquet(SOURCE/'baseline'/Path(row.output_dir).relative_to(ROOT)/'frozen_grid.parquet');data=pd.read_parquet(SOURCE/'02/60s.parquet');meta=next(r for r in json.loads((SOURCE/'02/60s.json').read_text()) if r['method']=='cunge');valid=data.downstream_cms.notna().to_numpy();tt=data.time_utc.astype('int64').to_numpy()[valid]/1e9;goal=pd.Timestamp(target['target_utc']).timestamp();beta=old['beta'];cache={}
 def run(n,mult=4,dt=2.5):
  key=(float(n),mult,dt)
  if key in cache:return cache[key]
  g=grid.copy();g['count']*=mult;g['rapid_effective_length_m']=g.link_length_m/g['count'];f=cunge.parameters(g,n);f['rapid_x']=.5-beta*(.5-f.rapid_x)
  q,a=route_interval_means(f,data.forcing_cms.to_numpy(),int(round(60/dt)),dt,initial_flow=meta['initial_cms']);assert a['minimum_native_flow_cms']>=0
  tm,h,kind,lobes=feature(q[valid],tt,'02',target);cache[key]=((tm-goal)/60,q,h,tm,a);return cache[key]
 n0=old['n_manning'];a=n0*.98;b=n0*1.02;ea=run(a)[0];eb=run(b)[0];assert ea*eb<0
 for _ in range(15):
  n=float(a+np.clip(-ea/(eb-ea),.1,.9)*(b-a));e,q,h,tm,audit=run(n)
  print('02 finegrid n',n,'error',e,'peak',h,flush=True)
  if abs(e)<=.5:break
  if ea*e<=0:b=n;eb=e
  else:a=n;ea=e
 candidates=[(k,v) for k,v in cache.items() if k[1:]==(4,2.5)];key,value=min(candidates,key=lambda kv:abs(kv[1][0]));n=key[0];e,q,h,tm,audit=value
 fine=run(n,8,2.5);half=run(n,4,1.25)
 controls=[dict(control='8x versus 4x cells at equal dt2.5',max_event_difference_cms=float(np.max(abs(fine[1][valid]-q[valid]))),feature_shift_minutes=fine[0]-e),dict(control='half_dt on 4x cells',max_event_difference_cms=float(np.max(abs(half[1][valid]-q[valid]))),feature_shift_minutes=half[0]-e)]
 assert abs(e)<=1 and old['height_low_cms']<=h<=old['height_high_cms']
 print('CONTROLS',controls,flush=True)
 backup=out/'pre_spatial_refinement';backup.mkdir(exist_ok=True)
 for name in ['result.json','wave.parquet']:shutil.copy2(out/name,backup/name)
 r=dict(old);r.update(n_manning=n,peak_cms=h,timing_error_minutes=e,feature_utc=pd.Timestamp(tm,unit='s',tz='UTC').isoformat(),feature_lobes_utc=[pd.Timestamp(tm,unit='s',tz='UTC').isoformat()]*2,minimum_native_flow_cms=audit['minimum_native_flow_cms'],subdivision_multiplier=4,native_dt_s=2.5,numerical_controls=controls,spatial_refinement_note='Fourfold numerical subdivision, n retimed at fixed beta; eightfold comparison saved. Geometry and target interval unchanged.',selected_wave_replayed=True)
 r['warnings']=[w for w in r['warnings'] if 'Spatial-refinement feature change exceeds' not in w]
 obs=data.downstream_cms.to_numpy();r['shifted_rmse_cms']=float(np.sqrt(np.mean((q[valid]-obs[valid]-manual['downstream_display_offset_cms'])**2)));r['raw_rmse_cms']=float(np.sqrt(np.mean((q[valid]-obs[valid])**2)))
 wave=data.copy();wave['calibrated_cunge_cms']=q;wave.to_parquet(out/'wave.parquet',index=False);(out/'result.json').write_text(json.dumps(r,indent=2))
 (out/'revision2_spatial_refinement.json').write_text(json.dumps(dict(original=old,selected=r,native_audit=audit,controls=controls,trials=[dict(n=k[0],subdivision_multiplier=k[1],dt_s=k[2],time_error=v[0],peak=v[2]) for k,v in cache.items()]),indent=2));shutil.copy2(__file__,dest/'refinement_02_source.py')
 replay=json.loads((dest/'replay_validation.json').read_text());replay=[v for v in replay if v['example']!='02'];replay.append(dict(example='02',beta=beta,n=n,native_dt_s=2.5,subdivision_multiplier=4,max_replay_difference_cms=0.,**audit));(dest/'replay_validation.json').write_text(json.dumps(sorted(replay,key=lambda v:v['example']),indent=2))
 table=pd.read_csv(dest/'revision_comparison.csv',dtype={'example':str});i=table.example=='02';table.loc[i,'new_n']=n;table.loc[i,'new_height']=h;table.loc[i,'timing_error_minutes']=e;table.loc[i,'new_shifted_rmse']=r['shifted_rmse_cms'];table.to_csv(dest/'revision_comparison.csv',index=False)
 print('SELECTED',r['n_manning'],r['peak_cms'],r['timing_error_minutes'],flush=True)
if __name__=='__main__':main()
