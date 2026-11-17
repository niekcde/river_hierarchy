"""CPU translation of supplied MATLAB local-inertial kernel; not MATLAB-verified.

Retains segment storage, min interface width, max-bed flow depth, momentum
memory, semi-implicit friction and critical-flow limiter. Corrects step clock,
uses an exact fixed-stage ghost, linear forcing and records boundary flux/mass.
"""
import json
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.io import savemat
from diffusive_network import normal_depth
from calibrate_boundaries56 import ROOT, CACHE, DATA, STATES, forcing_data, peak

OUT=ROOT/'LISFLOOD_FP/results/example56/local_inertial_pilot'
G=9.80665
EPS=np.finfo(float).eps


def geometry(frame,stage,n):
    f=frame.reset_index(drop=True)
    z=((f.head_us+f.head_ds)/2).to_numpy(float)
    w=f.width.to_numpy(float);dx=f.link_length_m.to_numpy(float)
    a=[];b=[]
    for i,r in enumerate(f.itertuples()):
        downstream=np.flatnonzero(f.u.to_numpy()==r.v)
        if len(downstream)>2:raise ValueError('More than two downstream branches')
        for j in downstream if len(downstream) else [len(f)]:a.append(i);b.append(j)
    a=np.array(a);b=np.array(b);ghost=len(f)
    zg=min(float(z.min()),stage);z=np.r_[z,zg]
    wf=np.minimum(w[a],np.where(b==ghost,w[a],w[np.minimum(b,ghost-1)]))
    dxf=.5*(dx[a]+np.where(b==ghost,dx[a],dx[np.minimum(b,ghost-1)]))
    source=np.flatnonzero(~f.u.isin(f.v).to_numpy())
    assert len(source)==1
    return dict(z=z,w=w,dx=dx,area=w*dx,a=a,b=b,wf=wf,dxf=dxf,
                ghost=ghost,source=int(source[0]),stage=stage,ghost_depth=stage-zg,n=n)


def route(m,times,forcing_t,forcing_q,initial=None,max_dt=300,cfl=.7,forcing_kind='linear'):
    if forcing_kind not in ['linear','hold']:raise ValueError('Unknown forcing interpolation')
    a,b=m['a'],m['b'];N=m['ghost'];area=m['area'];terminal=b==N
    h=np.zeros(N+1) if initial is None else initial[0].copy()
    q=np.zeros(len(a)) if initial is None else initial[1].copy()
    h[-1]=m['ghost_depth'];initial_storage=float(area@h[:-1])
    vin=vout=clip=0.;t=float(times[0]);rows=[];depths=[];steps=limited=0
    hmax=float(h[:-1].max());hmin=float(h[:-1].min());minq=np.inf;dtmin=np.inf;dtmax=0
    def save():
        storage=float(area@h[:-1]);qout=float((q*m['wf'])[terminal].sum())
        rows.append([t,qout,storage,vin,vout,clip,storage-initial_storage-vin+vout-clip])
        depths.append(h[:-1].copy())
    save()
    for stop in times[1:]:
        while t<stop-1e-8:
            k=int(np.searchsorted(forcing_t,t+1e-8,side='right'))
            knot=forcing_t[k] if k<len(forcing_t) else np.inf
            dt=min(max_dt,cfl*float(np.min(m['dx']/np.sqrt(G*np.maximum(h[:-1],EPS)))),stop-t,knot-t)
            assert dt>0
            eta=m['z']+h;hf=np.maximum(np.maximum(eta[a],eta[b])-np.maximum(m['z'][a],m['z'][b]),EPS)
            slope=(eta[b]-eta[a])/m['dxf']
            newq=(q-G*dt*hf*slope)/(1+G*dt*m['n']**2*abs(q)/hf**(7/3))
            limit=hf*np.sqrt(G*hf);limited+=int(np.count_nonzero(abs(newq)>limit))
            q=np.sign(newq)*np.minimum(abs(newq),limit);Q=q*m['wf']
            inlet=float(np.interp(t+dt/2,forcing_t,forcing_q)) if forcing_kind=='linear' else float(forcing_q[max(0,min(k-1,len(forcing_q)-1))])
            balance=np.bincount(b,weights=Q,minlength=N+1)-np.bincount(a,weights=Q,minlength=N+1)
            balance[m['source']]+=inlet
            trial=h[:-1]+balance[:-1]*dt/area
            correction=np.maximum(EPS-trial,0);clip+=float(area@correction)
            h[:-1]=trial+correction;h[-1]=m['ghost_depth']
            vin+=inlet*dt;vout+=float(Q[terminal].sum())*dt
            hmax=max(hmax,float(h[:-1].max()));hmin=min(hmin,float(h[:-1].min()))
            minq=min(minq,float(Q.min()));steps+=1;dtmin=min(dtmin,dt);dtmax=max(dtmax,dt)
            t+=dt
        t=float(stop);save()
    wave=pd.DataFrame(rows,columns=['time_seconds','outlet_cms','storage_m3','inflow_volume_m3','outflow_volume_m3','clipping_volume_m3','mass_residual_m3'])
    Q=q*m['wf'];bal=np.bincount(b,weights=Q,minlength=N+1)-np.bincount(a,weights=Q,minlength=N+1)
    bal[m['source']]+=float(np.interp(times[-1],forcing_t,forcing_q))
    audit=dict(steps=steps,limited_face_steps=limited,clipping_volume_m3=clip,
               maximum_mass_residual_m3=float(abs(wave.mass_residual_m3).max()),
               maximum_depth_m=hmax,minimum_depth_m=hmin,minimum_face_discharge_cms=minq,
               min_step_s=dtmin,max_step_s=dtmax,final_node_imbalance_cms=float(abs(bal[:-1]).max()))
    return wave,np.array(depths), (h,q),audit


def self_check():
    f=pd.DataFrame(dict(u=[0,1],v=[1,2],width=[100.,100.],link_length_m=[200.,200.],head_us=[1.,.8],head_ds=[.8,.6]))
    m=geometry(f,.7,.025);h=np.r_[np.ones(2),m['ghost_depth']]
    w,d,s,a=route(m,np.arange(0,3601,300.),np.array([0.,3600.]),np.array([20.,20.]),(h,np.zeros(2)),max_dt=20)
    assert np.all(np.isfinite(d)) and a['maximum_mass_residual_m3']<1e-5
    assert np.array_equal(w.time_seconds.to_numpy(),np.arange(0,3601,300.))
    # Independent steady wide-channel Manning solution must remain stationary.
    m=geometry(f,2.5,.025);unit_q=2.**(5/3)*np.sqrt(.001)/.025
    h=np.r_[np.full(2,2.),m['ghost_depth']]
    w,d,s,a=route(m,np.array([0.,3600.]),np.array([0.,3600.]),np.full(2,unit_q*100),
                  (h,np.full(2,unit_q)),max_dt=20)
    np.testing.assert_allclose(d,2.,atol=1e-10)
    np.testing.assert_allclose(w.outlet_cms,unit_q*100,atol=1e-8)


def main():
    OUT.mkdir(exist_ok=True);self_check()
    origin,ft,fq=forcing_data();n=.024854130456733084
    frame=pd.read_parquet(CACHE/(STATES['base']+'.parquet'))
    sink=next(iter(set(frame.v)-set(frame.u)));r=frame[frame.v==sink].iloc[0]
    stage=float(r.head_ds+normal_depth(float(fq[0]),r.width,(r.head_us-r.head_ds)/r.link_length_m,n))
    end=ft[-1]+96*3600;times=np.unique(np.r_[np.arange(0,end,300.),end])
    summaries=[]
    for state in ['base','S014_unit_10']:
        frame=pd.read_parquet(CACHE/(STATES[state]+'.parquet'));frame.to_csv(OUT/(state+'_geometry.csv'),index=False)
        m=geometry(frame,stage,n)
        matlab_m={k:(v.reshape(-1,1) if isinstance(v,np.ndarray) else v) for k,v in m.items()}
        for k in ['a','b','source']:matlab_m[k]=matlab_m[k]+1
        savemat(OUT/(state+'_matlab_input.mat'),dict(m=matlab_m,times=times[:,None],ft=ft[:,None],fq=fq[:,None]))
        for tag,dt,cfl in [('standard',300,.7),('refined',150,.35)]:
            cache=OUT/(state+'_'+tag+'.json')
            if cache.exists():summaries.append(json.loads(cache.read_text()));continue
            initial=None
            for day in range(1,61):
                sw,sd,initial,sa=route(m,np.array([0.,86400.]),np.array([0.,86400.]),np.array([fq[0],fq[0]]),initial,max_dt=dt,cfl=cfl)
                if sa['final_node_imbalance_cms']/fq[0]<1e-7:break
            else:raise RuntimeError('Spinup failed')
            print(state,tag,'spun up',day,flush=True)
            wave,depths,final,audit=route(m,times,ft,fq,initial,max_dt=dt,cfl=cfl)
            wave['time_utc']=origin+pd.to_timedelta(times,unit='s')
            wave.to_parquet(OUT/(state+'_'+tag+'.parquet'),index=False)
            np.savez_compressed(OUT/(state+'_'+tag+'_depths.npz'),depth_m=depths,time_seconds=times,bed_m=m['z'][:-1])
            tp,qp=peak(times,wave.outlet_cms.to_numpy())
            baseline=np.interp(times,[0,ft[-1]],[fq[0],fq[-1]]);excess=np.maximum(wave.outlet_cms.to_numpy()-baseline,0)
            centroid=float(np.trapz(times*excess,times)/np.trapz(excess,times))
            expected=float(np.trapz(fq,ft)+96*3600*fq[-1])
            assert abs(wave.inflow_volume_m3.iloc[-1]-expected)/expected<1e-10
            assert audit['maximum_mass_residual_m3']<1
            result=dict(state=state,tag=tag,n=n,cells=len(frame),fixed_stage_m=stage,spinup_days=day,
                        peak_cms=qp,peak_seconds=tp,peak_utc=str(origin+pd.to_timedelta(tp,unit='s')),
                        centroid_seconds=centroid,**audit)
            cache.write_text(json.dumps(result,indent=2));summaries.append(result);print(result,flush=True)
    source=Path('/Users/6256481/Downloads/LISFLOOD_FP_routing_GPU_2.m')
    metadata=dict(engine='Python translation of supplied MATLAB kernel; MATLAB execution blocked by license error -96',
                  source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),n=n,fixed_stage_m=stage,
                  origin_utc=str(origin),standard=dict(cfl=.7,max_dt_s=300),refined=dict(cfl=.35,max_dt_s=150),
                  input_sha256={str(CACHE/(STATES[s]+'.parquet')):hashlib.sha256((CACHE/(STATES[s]+'.parquet')).read_bytes()).hexdigest() for s in ['base','S014_unit_10']},
                  upstream_sha256=hashlib.sha256((DATA/'observed_upstream.parquet').read_bytes()).hexdigest(),
                  forcing='same raw upstream as diffusive; piecewise linear, midpoint integral with steps split at input knots',
                  bed='mean endpoint proxy elevations; one cell per physical reach',runs=summaries)
    (OUT/'manifest.json').write_text(json.dumps(metadata,indent=2))


if __name__=='__main__':main()
