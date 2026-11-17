"""Input-only sensitivity: preserve original local-inertial equations and fixed n.
No production input mutations; each geometry is copied into its result folder.
"""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import json,subprocess,time,hashlib
import numpy as np,pandas as pd
import broad_study as b
ROOT=Path(__file__).resolve().parent;ROOTOUT=b.OUT/'investigation';OUT=ROOTOUT/'input_sensitivity48';OUT.mkdir(exist_ok=True);KERNEL=ROOT/'inertial_verification_kernel.py';N=.020799605931143154
BASE=ROOTOUT/'current_mc/states/48/S000_base/parameters.parquet';NODES=ROOTOUT/'geometry_context48/48_node_source_snapshot.parquet';WIDTHS=ROOTOUT/'geometry_context48/48_width_source_qc.parquet'
CASES={
 'baseline':(False,'original',False),
 'width36':(True,'original',False),
 'slope36_half':(False,'half_drop',False),
 'outlet_raw':(False,'original',True),
 'width36_slope36_half':(True,'half_drop',False),
 'width36_outlet_raw':(True,'original',True),
 'slope36_half_outlet_raw':(False,'half_drop',True),
 'all_half':(True,'half_drop',True),
 'heads36_raw':(False,'raw',False),
 'all_raw':(True,'raw',True),
}
def geometry(case):
 f=pd.read_parquet(BASE);old=f.copy();nodes=pd.read_parquet(NODES).set_index('id_node');width=pd.read_parquet(WIDTHS).set_index('id_link');restore,head,tail=CASES[case];changes={};row=f[f.id_link==36].iloc[0]
 if restore:f.loc[f.id_link==36,'width']=float(width.loc[36,'wid_adj_wet'])
 if head=='half_drop':
  mid=(row.head_us+row.head_ds)/2;half_drop=(row.head_us-row.head_ds)/2;changes[int(row.u)]=mid+half_drop/2;changes[int(row.v)]=mid-half_drop/2
 elif head=='raw':
  changes[int(row.u)]=float(nodes.loc[row.u,'node_wse_raw']);changes[int(row.v)]=float(nodes.loc[row.v,'node_wse_raw'])
 if tail:changes[126]=float(nodes.loc[126,'node_wse_raw'])
 for node,value in changes.items():f.loc[f.u==node,'head_us']=value;f.loc[f.v==node,'head_ds']=value
 f['slope']=(f.head_us-f.head_ds)/f.link_length_m
 assert (f.slope>0).all() and (f.width>0).all()
 heads=pd.concat([f[['u','head_us']].rename(columns={'u':'node','head_us':'head'}),f[['v','head_ds']].rename(columns={'v':'node','head_ds':'head'})]);assert (heads.groupby('node')['head'].max()-heads.groupby('node')['head'].min()).max()==0
 delta=f[['id_link','width','head_us','head_ds','slope']].copy();mask=np.any(f[['width','head_us','head_ds']].to_numpy()!=old[['width','head_us','head_ds']].to_numpy(),axis=1);delta=delta[mask]
 for c in ['width','head_us','head_ds','slope']:delta['original_'+c]=old.loc[mask,c].to_numpy()
 return f,dict(case=case,width36_restored=restore,reach36_head_variant=head,outlet_raw=tail,changed_nodes=changes,changed_reaches=delta.to_dict('records'),interpretation='Sensitivity only. Raw WSE values are uncertain, and half-drop is an artificial diagnostic; neither is asserted to be a corrected bed.')
def solve(d,dest,label):
 src=dest/(label+'_input.npz');op=dest/(label+'_output.npz');np.savez_compressed(src,**d);p=subprocess.run(['/opt/anaconda3/bin/python',str(KERNEL),str(src),str(op)],capture_output=True,text=True)
 if p.returncode:raise RuntimeError(p.stdout+p.stderr)
 return dict(np.load(op))
def summarize_initial(cells,m,h,q):
 count=len(h);eta=np.r_[m['z'][:count]+h,m['stage']];hf=np.maximum(np.maximum(eta[m['a']],eta[m['b']])-np.maximum(m['z'][m['a']],m['z'][m['b']]),1e-12);terminal=m['b']==count;hf[terminal]=h[m['a'][terminal]];fr=abs(q)/(hf*np.sqrt(9.80665*hf));rows=[]
 for reach in [36,38,45,65,161,165,188]:
  ids=np.where(cells.id_link.to_numpy()==reach)[0];faces=np.where(np.isin(m['a'],ids))[0];last=ids[-1];outfaces=np.where(m['a']==last)[0];rows.append(dict(reach=reach,mean_depth_m=float(np.average(h[ids],weights=m['dx'][ids])),max_outgoing_face_Fr=float(max(fr[faces])),outflow_cms=float(sum(q[outfaces]*m['wf'][outfaces]))))
 return rows
def run(case,length=50,cfl=.35):
 dest=OUT/f'{case}_dx{length:g}_cfl{cfl:g}';dest.mkdir(exist_ok=True)
 if (dest/'result.json').exists():return json.loads((dest/'result.json').read_text())
 started=time.monotonic();print('START',dest.name,flush=True);f,desc=geometry(case);f.to_parquet(dest/'physical_parameters.parquet',index=False);b.save(dest/'changes.json',desc);cells=b.v.grid(f,f'max{length:g}');m=b.build(cells,N)
 force=b.forcing('48');origin=force.interval_start_utc.iloc[0];times=np.r_[(force.interval_start_utc-origin).dt.total_seconds(),(force.interval_end_utc.iloc[-1]-origin).total_seconds()];values=np.r_[force.forcing_cms,force.forcing_cms.iloc[-1]];Q=float(values[0]);initial=b.initial_guess(f,cells,m,Q)
 d=dict(m,h=initial[0],q=initial[1],times=np.array([0.,86400.]),forcing=np.array([Q,Q]),cfl=cfl,max_dt=150.,sample_indices=np.array([0,1]));audits=[]
 for day in range(1,61):
  x=solve(d,dest,'spin');audit=dict(zip(b.v.AUDIT,map(float,x['audit'])));audits.append(audit);d.update(h=x['h'],q=x['q'])
  if audit['final_node_imbalance_cms']/Q<1e-8:break
 else:raise RuntimeError('No steady state '+dest.name)
 initialhyd=summarize_initial(cells,m,x['h'],x['q']);d.update(times=times,forcing=values,sample_indices=np.array([0,len(times)-1]));x=solve(d,dest,'event');audit=dict(zip(b.v.AUDIT,map(float,x['audit'])));means=np.diff(x['wave'][:,4])/np.diff(times);w=force[['time_utc','phase']].copy();w['q_outlet_cms']=means;w.to_parquet(dest/'wave.parquet',index=False)
 obs=pd.read_parquet(b.OUT/'inputs/48/observed_downstream.parquet');mask=(w.time_utc>=obs.time.min())&(w.time_utc<=obs.time.max())&(w.time_utc>=force.loc[force.phase.eq('input_event'),'interval_start_utc'].iloc[0]);tt=(w.time_utc-origin).dt.total_seconds().to_numpy();tm,height,kind,_=b.feature(means[mask],tt[mask],'48',b.TARGETS['48']);target=(pd.Timestamp(b.TARGETS['48']['target_utc'])-origin).total_seconds();clipidx=np.where(x['wave'][:,5]>1e-6)[0];rel=audit['maximum_mass_residual_m3']/x['wave'][-1,3]
 res=dict(case=case,length=length,cfl=cfl,n=N,seconds=time.monotonic()-started,spinup_days=day,spinup_audits=audits,audit=audit,initial_hydraulics=initialhyd,first_clipping_sample_hours=None if not len(clipidx) else float(times[clipidx[0]]/3600),feature_height_cms=height,feature_kind=kind,timing_error_minutes=(tm-target)/60,relative_mass_residual_excluding_clipped_water=rel,clip_fraction_of_inflow=audit['clipping_volume_m3']/x['wave'][-1,3],finite_and_no_clipping=bool(np.isfinite(means).all() and audit['clipping_volume_m3']<1e-6),outlet_slope_m_per_km=m['outlet_slope']*1000,folder=str(dest),kernel_sha256=b.sha(KERNEL),source_parameter_sha256=b.sha(BASE),scope='Input changes only, original supplied-type local-inertial update, original critical cap retained; no q-centred or positivity stabilization and no calibration. Clipped runs are invalid hydrographs.')
 b.save(dest/'result.json',res);print('DONE',case,length,cfl,'clip',audit['clipping_volume_m3'],'hmax',audit['maximum_depth_m'],'height',height,flush=True);return res
if __name__=='__main__':
 # Execute the full 2x2x2 width / half-drop / outlet factorial, plus raw-head controls.
 with ThreadPoolExecutor(max_workers=2) as pool:results=list(pool.map(lambda x:run(*x),[(case,length) for length in [100,50] for case in CASES]))
 b.save(OUT/'primary_results.json',results)
