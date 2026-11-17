"""Reproducible hydraulic verification and grid/boundary sensitivity study."""
import json,sys,subprocess,hashlib,os
from pathlib import Path
import numpy as np
import pandas as pd
from local_inertial_pilot56 import geometry,route
from match_mc_grid56 import expand,SAVED,OUT as PREVIOUS
from calibrate_boundaries56 import peak

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'LISFLOOD_FP/results/example56/normal_flow_verification'
OUT.mkdir(exist_ok=True,parents=True)
KERNEL=Path(__file__).with_name('inertial_verification_kernel.py')
AUDIT=['steps','limited_face_steps','clipping_volume_m3','maximum_mass_residual_m3','maximum_depth_m','minimum_depth_m','minimum_face_discharge_cms','min_step_s','max_step_s','final_node_imbalance_cms','maximum_face_froude']

def build(f,n=.025,normal=True,stage=0.,slope_factor=1.):
    m=geometry(f,stage,n);terminal=m['b']==len(f)
    ids=m['a'][terminal];assert len(ids)==1
    r=f.iloc[ids[0]];slope=(r.head_us-r.head_ds)/r.link_length_m
    assert slope>0 or not normal
    m['z'][-1]=r.head_ds;m['dxf'][terminal]=m['dx'][ids]/2
    return {k:m[k] for k in ['z','w','dx','a','b','wf','dxf','source','n']}|dict(outlet_slope=slope*slope_factor,stage=stage,normal=normal)

def solve(m,times,forcing,initial=None,cfl=.7,max_dt=300,samples=None):
    if samples is None:samples=np.array([0,len(times)-1])
    h,q=(np.zeros(len(m['w'])),np.zeros(len(m['a']))) if initial is None else initial
    inp=OUT/f'_input_{os.getpid()}.npz';op=OUT/f'_output_{os.getpid()}.npz'
    np.savez(inp,**m,times=np.asarray(times,float),forcing=np.asarray(forcing,float),h=h,q=q,cfl=cfl,max_dt=max_dt,sample_indices=np.asarray(samples,int))
    subprocess.run(['/opt/anaconda3/bin/python',str(KERNEL),str(inp),str(op)],check=True)
    with np.load(op) as data:r={k:data[k].copy() for k in data.files}
    r['audit']=dict(zip(AUDIT,map(float,r['audit'])))
    return r

def spin(m,Q):
    initial=None
    for day in range(1,61):
        r=solve(m,[0,86400],[Q,Q],initial)
        initial=(r['h'],r['q'])
        if r['audit']['final_node_imbalance_cms']/max(Q,1)<1e-8:return initial,day,r['audit']
    raise RuntimeError('Spinup failed')

def channel(count=10,length=2000,width=100,slope=.001):
    x=np.linspace(0,length,count+1)
    return pd.DataFrame(dict(id_link=np.arange(count),u=np.arange(count),v=np.arange(1,count+1),width=width,link_length_m=np.diff(x),head_us=3-slope*x[:-1],head_ds=3-slope*x[1:]))

def checks():
    records=[]
    def record(name,passed,**metrics):
        records.append(dict(test=name,passed=bool(passed),**metrics));print(records[-1],flush=True)
        (OUT/'hydraulic_checks.json').write_text(json.dumps(records,indent=2))
    # Independent analytic equilibrium: no spin-up used to manufacture solution.
    for count in [2,10,40]:
        f=channel(count);m=build(f);h=np.full(count,2.);unit=2**(5/3)*np.sqrt(.001)/.025
        r=solve(m,[0,3600],[unit*100]*2,(h,np.full(count,unit)))
        err=float(abs(r['h']-2).max());record('uniform_Manning_'+str(count),err<1e-9,depth_error_m=err,**r['audit'])
    f=channel(10);f['head_us']=np.sin(np.arange(10))*.2;f['head_ds']=np.sin(np.arange(1,11))*.2
    m=build(f,normal=False,stage=2.);h=2-m['z'][:-1]
    r=solve(m,[0,3600],[0,0],(h,np.zeros(10)))
    err=float(abs(r['h']-h).max());record('lake_at_rest',err<1e-10,depth_error_m=err,maximum_flow_cms=float(abs(r['q']*m['wf']).max()))
    # Same fixed-stage discrete kernel as legacy numpy implementation.
    f=channel(5);old=geometry(f,2.5,.025);m=build(f,normal=False,stage=2.5)
    m['z']=old['z'];m['dxf']=old['dxf']
    init=(np.full(5,1.),np.zeros(5));t=np.arange(0,3601,60.)
    r=solve(m,t,np.full(len(t),40.),init,max_dt=20)
    w,d,s,a=route(old,t,t,np.full(len(t),40.),(np.r_[init[0],old['ghost_depth']],init[1]),max_dt=20,forcing_kind='hold')
    err=float(abs(r['wave'][:,1]-w.outlet_cms).max());record('numba_vs_existing_numpy_kernel',err<1e-8,maximum_outlet_difference_cms=err)
    # Backwater: compare two imposed stages to a separately spun normal-flow case.
    f=channel(20,slope=.0002);normal=build(f);init,_,_=spin(normal,100.)
    hn=(100*.025/(100*np.sqrt(.0002)))**.6;stages=[f.head_ds.iloc[-1]+hn,f.head_ds.iloc[-1]+hn+.5]
    depths=[]
    for stage in stages:
        fixed=build(f,normal=False,stage=stage);ini,day,audit=spin(fixed,100.);depths.append(ini[0])
    change=depths[1]-depths[0]
    record('raised_tailwater_backwater',np.min(change)>0,minimum_depth_increase_m=float(change.min()),upstream_depth_increase_m=float(change[0]))
    # Symmetric split followed by confluence; equal branches must stay equal.
    f=pd.DataFrame(dict(id_link=range(4),u=[0,1,1,2],v=[1,2,2,3],width=[100.,50.,50.,100.],link_length_m=[200.]*4,head_us=[3.,2.8,2.8,2.6],head_ds=[2.8,2.6,2.6,2.4]))
    m=build(f);init,_,_=spin(m,100.)
    t=np.arange(0,14401,60.);forcing=100+100*np.exp(-((t-3600)/900)**2)
    r=solve(m,t,forcing,init,samples=np.arange(len(t)))
    err=float(abs(r['depth'][:,1]-r['depth'][:,2]).max())
    record('symmetric_split_confluence_pulse',err<1e-10 and r['audit']['maximum_mass_residual_m3']<1e-5 and r['audit']['clipping_volume_m3']<1e-6,branch_depth_difference_m=err,**r['audit'])
    # Width and slope transitions: consistency/conservation check, not analytic truth.
    f=channel(20);f.loc[5:14,'width']=60.;f.loc[10:,'head_us']-=.15;f.loc[10:,'head_ds']-=.15
    m=build(f);init,_,_=spin(m,100.);r=solve(m,t,forcing,init)
    record('width_bed_transition_pulse',np.isfinite(r['h']).all() and r['audit']['maximum_mass_residual_m3']<1e-5 and r['audit']['clipping_volume_m3']<1e-6,**r['audit'])
    assert all(r['passed'] for r in records),'Hydraulic verification failed; inspect results before proceeding'


def grid(f,kind):
    f=f.copy()
    if kind=='mc':return expand(f)
    if kind=='physical':return f
    if kind=='floor200':
        f['link_length_m']=np.maximum(f.link_length_m,200.)
        return f
    length=float(kind.removeprefix('max'))
    f['count']=np.maximum(1,np.ceil(f.link_length_m/length)).astype(int)
    f['rapid_effective_length_m']=f.link_length_m/f['count']
    return expand(f)

def metrics(t,q,baseline):
    tp,qp=peak(t,q);ex=np.maximum(q-baseline,0)
    # Crossing times use 50% of each hydrograph's own peak above its initial flow.
    target=q[0]+.5*(qp-q[0]);ip=int(np.argmax(q));rise=np.flatnonzero(q[:ip+1]>=target);fall=np.flatnonzero(q[ip:]<=target)
    def cross(j):return float(t[j-1]+(target-q[j-1])/(q[j]-q[j-1])*(t[j]-t[j-1]))
    return dict(peak_cms=qp,peak_seconds=tp,centroid_seconds=float(np.trapz(t*ex,t)/np.trapz(ex,t)),rising50_seconds=cross(int(rise[0])) if len(rise) and rise[0]>0 else None,falling50_seconds=cross(ip+int(fall[0])) if len(fall) else None)

def study():
    forcing=pd.read_parquet(SAVED/'forcing.parquet');origin=forcing.interval_start_utc.iloc[0]
    times=np.r_[(forcing.interval_start_utc-origin).dt.total_seconds(),(forcing.interval_end_utc.iloc[-1]-origin).total_seconds()]
    vals=np.r_[forcing.forcing_cms,forcing.forcing_cms.iloc[-1]];t=(forcing.time_utc-origin).dt.total_seconds().to_numpy()
    event=forcing[forcing.phase.eq('input_event')]
    baseline=np.interp(t,[(event.interval_start_utc.iloc[0]-origin).total_seconds(),(event.interval_end_utc.iloc[-1]-origin).total_seconds()],[vals[0],vals[-1]])
    snapshots=np.array([0,*[int(np.argmin(abs(times-(pd.Timestamp(date)-origin).total_seconds()))) for date in ['2025-06-26T00:00Z','2025-06-27T12:00Z','2025-06-29T00:00Z']],len(times)-1])
    configs=[(g,1.,False) for g in ['physical','mc','floor200','max1000','max500','max250','max125']]
    configs += [(g,1.,True) for g in ['mc','max250','max125']]
    configs += [('max250',s,False) for s in [.5,2.]]
    configs += [('max125',1.,'quarter'),('max62.5',1.,True),('max62.5',1.,'quarter'),('max250',0.,True)]
    results=[]
    for state in ['S000_base','S014_unit_10']:
        f=pd.read_parquet(PREVIOUS/(state+'_mc_parameters.parquet'));n=float(f.n_manning.iloc[0])
        for kind,slope_factor,half in configs:
            tag=f'{state}_{kind}_s{slope_factor:g}'+('_quarterdt' if half=='quarter' else '_halfdt' if half else '')
            record=OUT/(tag+'.json')
            if record.exists():results.append(json.loads(record.read_text()));continue
            cells=grid(f,kind)
            stage=0.
            if slope_factor==0:
                terminal=cells[~cells.v.isin(cells.u)].iloc[0]
                slope=(terminal.head_us-terminal.head_ds)/terminal.link_length_m
                stage=float(terminal.head_ds+(vals[0]*n/(terminal.width*np.sqrt(slope)))**.6)
            m=build(cells,n,slope_factor=slope_factor,normal=slope_factor!=0,stage=stage)
            cells.to_csv(OUT/(tag+'_cells.csv'),index=False)
            initial,days,spinaudit=spin(m,vals[0]);print(tag,'spinup',days,flush=True)
            scale=.25 if half=='quarter' else .5 if half else 1.
            r=solve(m,times,vals,initial,cfl=.7*scale,max_dt=300*scale,samples=snapshots)
            means=np.diff(r['wave'][:,4])/60
            wave=forcing[['time_utc','phase']].copy();wave['q_outlet_cms']=means;wave.to_parquet(OUT/(tag+'.parquet'),index=False)
            np.savez_compressed(OUT/(tag+'_diagnostics.npz'),times=times[snapshots],depth=r['depth'],flow=r['flow'],hmax=r['hmax'],hmin=r['hmin'],bed=m['z'],a=m['a'],b=m['b'],wf=m['wf'],dxf=m['dxf'],storage=r['wave'][:,2])
            result=dict(tag=tag,state=state,grid=kind,slope_factor=slope_factor,halfdt=half,cells=len(cells),minimum_length_m=float(cells.link_length_m.min()),maximum_length_m=float(cells.link_length_m.max()),total_length_m=float(cells.link_length_m.sum()),storage_area_m2=float((cells.width*cells.link_length_m).sum()),n=n,spinup_days=days,spinup_audit=spinaudit,**metrics(t,means,baseline),**r['audit'])
            result['relative_mass_residual']=result['maximum_mass_residual_m3']/r['wave'][-1,3]
            result['boundary']='normal' if slope_factor else 'fixed_stage_at_physical_face'
            result['fixed_stage_m']=stage if slope_factor==0 else None
            result['cfl']=.7*scale
            result['peak_utc']=str(origin+pd.to_timedelta(result['peak_seconds'],unit='s'))
            record.write_text(json.dumps(result,indent=2));results.append(result)
            print(tag,{k:result[k] for k in ['cells','peak_utc','peak_cms','maximum_depth_m','maximum_face_froude','clipping_volume_m3']},flush=True)
    (OUT/'manifest.json').write_text(json.dumps(dict(engine='Python CPU local-inertial translation; MATLAB verification pending',boundary='normal flow: wide rectangular Manning, terminal cell depth, terminal reach slope times sensitivity factor',floor='only physical reaches below 200m stretched; endpoint elevations held; storage and distances change',forcing_sha256=hashlib.sha256((SAVED/'forcing.parquet').read_bytes()).hexdigest(),kernel_sha256=hashlib.sha256(KERNEL.read_bytes()).hexdigest(),origin_utc=str(origin),runs=results),indent=2))

if __name__=='__main__':
    if '--study-only' not in sys.argv:checks()
    if '--checks-only' not in sys.argv:study()
