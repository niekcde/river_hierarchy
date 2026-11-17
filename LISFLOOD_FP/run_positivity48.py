"""Fixed-roughness experiment only; no calibration or production promotion."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import json,subprocess,time,hashlib
import numpy as np
import pandas as pd
import broad_study as b
root=b.OUT/'investigation';out=root/'positivity48';out.mkdir(exist_ok=True);kernel=Path(__file__).with_name('inertial_positivity_diagnostic_kernel.py')
CASES={'100m_cfl035':'limiter_retained_48_n0.0207996059','50m_cfl035':'limiter_retained_48_n0.0207996059_dx50','50m_cfl0175':'limiter_retained_48_n0.0207996059_cfl0.175_dx50','50m_cfl00875':'limiter_retained_48_n0.0207996059_cfl0.0875_dx50'}
AUDIT=b.v.AUDIT+['positivity_limited_face_steps','deferred_transfer_m3','minimum_donor_factor','first_positivity_time_s']
def run(item):
 tag,folder=item;dest=out/tag;dest.mkdir(exist_ok=True);src=next((root/folder).glob('_input*.npz'));t0=time.monotonic();print('START',tag,flush=True)
 d=dict(np.load(src));np.savez_compressed(dest/'input.npz',**d)
 p=subprocess.run(['/opt/anaconda3/bin/python',str(kernel),str(dest/'input.npz'),str(dest/'output.npz')],capture_output=True,text=True)
 if p.returncode:raise RuntimeError(tag+p.stdout+p.stderr)
 x=dict(np.load(dest/'output.npz'));audit=dict(zip(AUDIT,map(float,x['audit'])));force=b.forcing('48');means=np.diff(x['wave'][:,4])/np.diff(d['times']);wave=force[['time_utc','phase']].copy();wave['q_outlet_cms']=means;wave.to_parquet(dest/'wave.parquet',index=False)
 obs=pd.read_parquet(b.OUT/'inputs/48/observed_downstream.parquet');origin=force.interval_start_utc.iloc[0];mask=(wave.time_utc>=obs.time.min())&(wave.time_utc<=obs.time.max())&(wave.time_utc>=force.loc[force.phase.eq('input_event'),'interval_start_utc'].iloc[0]);tt=(wave.time_utc-origin).dt.total_seconds().to_numpy();tm,height,kind,_=b.feature(means[mask],tt[mask],'48',b.TARGETS['48']);target=(pd.Timestamp(b.TARGETS['48']['target_utc'])-origin).total_seconds()
 old=dict(np.load(next((root/folder).glob('_output*.npz'))));res=dict(tag=tag,seconds=time.monotonic()-t0,audit=audit,timing_error_minutes=(tm-target)/60,feature_kind=kind,feature_height_cms=height,relative_mass_residual=audit['maximum_mass_residual_m3']/x['wave'][-1,3],max_outlet_difference_from_original_cms=float(np.max(abs(means-np.diff(old['wave'][:,4])/np.diff(d['times'])))),initialization='Same saved steady event initial state as original, isolating event solver change; prior spinup history not rerun.',n=float(d['n']),cfl=float(d['cfl']),input_sha256=hashlib.sha256(src.read_bytes()).hexdigest(),kernel_sha256=hashlib.sha256(kernel.read_bytes()).hexdigest())
 (dest/'result.json').write_text(json.dumps(res,indent=2));print('DONE',json.dumps(res),flush=True);return res
if __name__=='__main__':
 with ThreadPoolExecutor(max_workers=2) as pool:results=list(pool.map(run,CASES.items()))
 (out/'results.json').write_text(json.dumps(results,indent=2))
