"""Requested grid series, downstream extensions and MATLAB parity fixtures."""
import sys,json,time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.io import savemat
import verify_inertial56 as v
BASE=v.OUT
OUT=BASE.parent/'extended_convergence';OUT.mkdir(exist_ok=True)
v.OUT=OUT


def run(state,length,extension=0,cfl=.35):
    tag=f'{state}_max{length:g}_ext{extension:g}_cfl{cfl:g}'
    record=OUT/(tag+'.json')
    if record.exists():
        try:return json.loads(record.read_text())
        except json.JSONDecodeError:pass
    start=time.monotonic();f=pd.read_parquet(v.PREVIOUS/(state+'_mc_parameters.parquet'));n=float(f.n_manning.iloc[0])
    terminal=f[~f.v.isin(f.u)].iloc[0];outlet_node=int(terminal.v)
    if extension:
        extra=terminal.copy();extra['id_link']=int(f.id_link.max())+1;extra['u']=outlet_node;extra['v']=int(max(f.u.max(),f.v.max()))+1
        extra['link_length_m']=extension;extra['head_us']=terminal.head_ds;extra['head_ds']=terminal.head_ds-extension*(terminal.head_us-terminal.head_ds)/terminal.link_length_m
        f=pd.concat([f,pd.DataFrame([extra])],ignore_index=True)
    cells=v.grid(f,f'max{length:g}');m=v.build(cells,n)
    monitor=int(np.flatnonzero((cells.v.to_numpy()==outlet_node))[0]);faces=np.flatnonzero(m['a']==monitor);assert len(faces)==1
    m['monitor_face']=int(faces[0])
    forcing=pd.read_parquet(v.SAVED/'forcing.parquet');origin=forcing.interval_start_utc.iloc[0]
    times=np.r_[(forcing.interval_start_utc-origin).dt.total_seconds(),(forcing.interval_end_utc.iloc[-1]-origin).total_seconds()]
    vals=np.r_[forcing.forcing_cms,forcing.forcing_cms.iloc[-1]];t=(forcing.time_utc-origin).dt.total_seconds().to_numpy()
    event=forcing[forcing.phase.eq('input_event')];baseline=np.interp(t,[(event.interval_start_utc.iloc[0]-origin).total_seconds(),(event.interval_end_utc.iloc[-1]-origin).total_seconds()],[vals[0],vals[-1]])
    # Warm-start from nearest already verified steady state, retaining exact reach IDs.
    initial=None
    for prev in [125,150,100,50,25]:
        paths=list(OUT.glob(f'{state}_max{prev}_ext{extension:g}_cfl*/initial.npz'))
        if not paths:continue
        data=np.load(paths[0]);old=pd.read_csv(paths[0].parent/'cells.csv');h=np.zeros(len(cells));q=np.zeros(len(m['a']))
        for link,g in cells.groupby('id_link',sort=False):
            ids=g.index.to_numpy();og=old[old.id_link==link];oi=og.index.to_numpy()
            if not len(oi):break
            x=(np.arange(len(ids))+.5)/len(ids);ox=(np.arange(len(oi))+.5)/len(oi)
            h[ids]=np.interp(x,ox,data['h'][oi])
        else:
            # Per-cell throughflow, including branch-specific interfaces, from old steady flux.
            old_a=data['a'];old_b=data['b'];old_flow=data['q']*data['wf']
            oa=old.id_link.to_numpy()[old_a];ob=np.array([int(old.id_link.iloc[k]) if k<len(old) else -1 for k in old_b])
            pairs={(int(la),int(lb)):float(old_flow[(oa==la)&(ob==lb)].mean()) for la,lb in set(zip(oa,ob))}
            through={int(la):float(old_flow[oa==la].mean()) for la in np.unique(oa)}
            for j,(a,b) in enumerate(zip(m['a'],m['b'])):
                la=int(cells.id_link.iloc[a]);lb=int(cells.id_link.iloc[b]) if b<len(cells) else -1
                q[j]=pairs.get((la,lb),through[la])/m['wf'][j]
            initial=(h,q);break
    # Retain the converged steady initial state for an independent MATLAB run.
    for day in range(1,61):
        spun=v.solve(m,[0,86400],[vals[0],vals[0]],initial,cfl=cfl,max_dt=150)
        initial=(spun['h'],spun['q'])
        if spun['audit']['final_node_imbalance_cms']/vals[0]<1e-8:break
    else:raise RuntimeError('spinup failed')
    folder=OUT/tag;folder.mkdir(exist_ok=True);cells.to_csv(folder/'cells.csv',index=False)
    np.savez(folder/'initial.npz',h=initial[0],q=initial[1],a=m['a'],b=m['b'],wf=m['wf'])
    samples=np.array([0,int(np.argmin(abs(times-366660))),len(times)-1])
    payload=dict(**m,times=times,forcing=vals,h=initial[0],q=initial[1],cfl=cfl,max_dt=150.,sample_indices=samples)
    savemat(folder/'input.mat',payload,oned_as='column')
    print(tag,'cells',len(cells),'spinup days',day,'event starting',flush=True)
    r=v.solve(m,times,vals,initial,cfl=cfl,max_dt=150,samples=samples)
    means=np.diff(r['wave'][:,8])/60
    wave=forcing[['time_utc','phase']].copy();wave['q_outlet_cms']=means;wave['q_remote_boundary_cms']=np.diff(r['wave'][:,4])/60;wave.to_parquet(folder/'wave.parquet',index=False)
    np.savez_compressed(folder/'result.npz',**{k:val for k,val in r.items() if k!='audit'})
    result=dict(tag=tag,state=state,length=length,extension_m=extension,cfl=cfl,cells=len(cells),n=n,seconds=time.monotonic()-start,spinup_days=day,spinup_audit=spun['audit'],**v.metrics(t,means,baseline),**r['audit'])
    result['relative_mass_residual']=result['maximum_mass_residual_m3']/r['wave'][-1,3]
    result['valid']=bool(result['clipping_volume_m3']<1e-6 and result['limited_face_steps']==0 and result['relative_mass_residual']<1e-9)
    result['peak_utc']=str(origin+pd.to_timedelta(result['peak_seconds'],unit='s'));record.write_text(json.dumps(result,indent=2));print(result,flush=True)
    return result

if __name__=='__main__':
    if '--extension' in sys.argv:
        for length in [125,50]:
            for extension in [0,5000,10000,20000]:
                for state in ['S000_base','S014_unit_10']:run(state,length,extension)
    else:
        for length in [1000,500,250,200,150,125,100,50,25,10]:
            for state in ['S000_base','S014_unit_10']:
                r=run(state,length)
                if not r['valid']:run(state,length,cfl=.175)
        for length in [100,25,10]:
            for state in ['S000_base','S014_unit_10']:run(state,length,cfl=.175)
