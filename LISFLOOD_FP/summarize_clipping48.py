"""Independent vectorized check of first clipping; MATLAB execution separately required."""
from pathlib import Path
import json,hashlib
import numpy as np
import pandas as pd
import broad_study as b
r=b.OUT/'investigation';dest=r/'first_clipping48';p=dict(np.load(dest/'first_failure.npz'));src=r/'limiter_retained_48_n0.0207996059_dx50/_input_29515.npz';d=dict(np.load(src));N=len(d['w']);a=d['a'];bb=d['b'];h=p['pre_h'];q=p['pre_q'];dt=p['step'][1];g=9.80665
eta=np.r_[d['z'][:N]+h,float(d['stage'])];hf=np.maximum(np.maximum(eta[a],eta[bb])-np.maximum(d['z'][a],d['z'][bb]),1e-12)
raw=(q-g*dt*hf*(eta[bb]-eta[a])/d['dxf'])/(1+g*dt*d['n']**2*np.abs(q)/hf**(7/3));limit=hf*np.sqrt(g*hf);newq=np.clip(raw,-limit,limit);terminal=bb==N
newq[terminal]=np.maximum(h[a[terminal]],0)**(5/3)*np.sqrt(d['outlet_slope'])/d['n'];Q=newq*d['wf'];bal=np.bincount(bb,weights=Q,minlength=N+1)-np.bincount(a,weights=Q,minlength=N+1);bal[int(d['source'])]+=p['step'][2];trial=h+dt*bal[:N]/(d['w']*d['dx']);i=int(np.argmin(trial))
rows=[];f=pd.read_parquet(r/'current_mc/states/48/S000_base/parameters.parquet');cells=b.v.grid(f,'max50')
for folder in ['first_clipping48','first_clipping48_half','first_clipping48_quarter']:
 s=dict(np.load(r/folder/'first_failure.npz'));ii=int(np.argmin(s['trial']));rows.append(dict(folder=folder,cell=ii,reach=int(cells.iloc[ii].id_link),subsection=int(cells.iloc[ii].subsection),time_hours=s['step'][0]/3600,dt_s=s['step'][1],depth_before_m=s['pre_h'][ii],trial_depth_m=s['trial'][ii]))
result=dict(first_failures=rows,vectorized_q_max_abs_difference=float(np.max(abs(newq-p['new_q']))),vectorized_trial_max_abs_difference=float(np.max(abs(trial-p['trial']))),available_volume_m3=float(h[i]*d['w'][i]*d['dx'][i]),net_removed_m3=float(-dt*bal[i]),required_draining_dt_s=float(h[i]*d['w'][i]*d['dx'][i]/-bal[i]),matlab_status='Not executed: university licence server unreachable, MathWorks error -15.2',input_sha256=hashlib.sha256(src.read_bytes()).hexdigest(),matlab_source_sha256=hashlib.sha256(Path('/Users/6256481/Downloads/LISFLOOD_FP_routing_GPU_2.m').read_bytes()).hexdigest())
(dest/'diagnosis.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
