"""Parent-only peak calibration per outlet scenario, then frozen-n collapses.

No fitting of observed downstream discharge as a boundary condition. Historical
local geometry and point-flow caches are used with recorded hashes.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from boundary_options import build_boundary,gauge_discharge

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'LISFLOOD_FP/results/example56/boundary_calibration'
CACHE=ROOT/'RAPID/assets/bounded_robust_slopes_20261002/child_head_reconstruction_20261003/56'
DATA=ROOT/'RAPID/outputs/master_workflow_verification_20261009/calibration/bounded_beta_20261009/56'
STATES={'base':'c2578749d698d194b631','S014_unit_10':'c3894ba4eb9cbdccfc79',
        'S008_unit_6':'3e3fc3e17f39c1215876','S012_unit_12':'6a8ae685eea6ebb582b6'}
SCENARIOS={'A_normal':(1.,0.),'B_half_slope':(.5,0.),'C_double_slope':(2.,0.),'D_extend10km':(1.,10000.)}
TARGET=pd.Timestamp('2025-06-27T14:40:00Z')
FIT_START=pd.Timestamp('2025-06-24T06:09:30Z')
FIT_END=pd.Timestamp('2025-07-13T00:00:30Z')


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def peak(t,q):
    i=int(np.argmax(q));tpeak=float(t[i]);qpeak=float(q[i])
    if 0<i<len(q)-1 and np.isclose(t[i]-t[i-1],t[i+1]-t[i]):
        den=q[i-1]-2*q[i]+q[i+1]
        if den<0:
            offset=.5*(q[i-1]-q[i+1])/den
            if abs(offset)<=1:
                tpeak+=offset*(t[i+1]-t[i]);qpeak-=.25*(q[i-1]-q[i+1])*offset
    return tpeak,qpeak


def forcing_data():
    f=pd.read_parquet(DATA/'observed_upstream.parquet').sort_values('time')
    if f.time.duplicated().any() or not np.isfinite(f.discharge_cms).all() or (f.discharge_cms<=0).any():
        raise ValueError('Invalid upstream series')
    origin=f.time.iloc[0];t=(f.time-origin).dt.total_seconds().to_numpy();q=f.discharge_cms.to_numpy()
    return origin,t,q


def simulate(scenario,state,n,*,calibration=False,dx=None,extension_override=None):
    sf,extension=SCENARIOS[scenario]
    if extension_override is not None:extension=extension_override
    spacing=dx or (400. if calibration else 200.)
    rtol=2e-6 if calibration else 2e-7
    phase='search' if calibration else 'runs'
    tag=f'{state}_n{n:.9f}_dx{spacing:g}_ext{extension:g}'
    folder=OUT/scenario/phase/tag
    folder.mkdir(parents=True,exist_ok=True)
    meta=folder/'result.json'
    if meta.exists():
        result=json.loads(meta.read_text())
        if result['code_hashes']!=code_hashes():raise ValueError('Stale cached solver result')
        return result
    start=time.perf_counter();origin,tin,qin=forcing_data()
    end=float((pd.Timestamp('2025-07-01T00:00:00Z')-origin).total_seconds()) if calibration else tin[-1]+96*3600
    times=np.unique(np.r_[np.arange(0,end,300.),end])
    inflow=lambda t:np.interp(t,tin,qin)
    frame=pd.read_parquet(CACHE/(STATES[state]+'.parquet'))
    m,info=build_boundary(frame,n,spacing,sf,extension)
    initial,spin=m.spinup(float(qin[0]))
    sol=m.integrate(initial,times,inflow,max_step=600.,rtol=rtol)
    h=sol.y[:-2]
    qg=np.zeros(len(times));minq=float('inf');maxfr=0.
    for i in range(len(times)):
        qg[i],faces=gauge_discharge(m,info,h[:,i])
        hf=.5*(h[m.a,i]+h[m.b,i])
        maxfr=max(maxfr,float(np.max(abs(faces)/(m.width*hf*np.sqrt(9.81*hf)))))
        minq=min(minq,float(faces.min()))
    storage=m.storage_area@h
    mass=storage-storage[0]-(sol.y[-2]-sol.y[-1])
    tf=np.r_[tin[tin<end],end];qf=np.interp(tf,tin,qin)
    input_exact=float(np.trapz(qf,tf))
    volume_error=float((sol.y[-2,-1]-input_exact)/input_exact)
    if abs(volume_error)>1e-5 or abs(mass).max()/input_exact>1e-6:raise ValueError('Mass/forcing integration failure')
    # Cumulative original-port flux independently includes storage in extension.
    downstream_delta=0 if info['downstream_area'] is None else info['downstream_area']@(h-initial[:,None])
    cumulative_gauge=sol.y[-1]+downstream_delta
    upstream_delta=storage-storage[0]-downstream_delta
    port_mass=upstream_delta-(sol.y[-2]-cumulative_gauge)
    utc=origin+pd.to_timedelta(times,unit='s')
    wave=pd.DataFrame(dict(time_seconds=times,time_utc=utc,inflow_cms=inflow(times),gauge_cms=qg,
        cumulative_gauge_m3=cumulative_gauge,domain_storage_m3=storage,mass_residual_m3=mass))
    window=(utc>=FIT_START)&(utc<=FIT_END)
    tp,qp=peak(times[window],qg[window]);target=float((TARGET-origin).total_seconds())
    ds=pd.read_parquet(DATA/'observed_downstream.parquet').sort_values('time')
    td=(ds.time-origin).dt.total_seconds().to_numpy();qd=ds.discharge_cms.to_numpy()
    # Exclude interpolation across downstream gaps >15 minutes from fit diagnostics.
    idx=np.searchsorted(td,times,side='left');exact=(idx<len(td))&(td[np.minimum(idx,len(td)-1)]==times)
    left=np.clip(idx-1,0,len(td)-1);right=np.clip(idx,0,len(td)-1)
    supported=(times>=td[0])&(times<=td[-1])&(exact|((td[right]-td[left])<=900))
    valid=window&supported;observed=np.interp(times,td,qd)
    errors=qg[valid]-observed[valid]
    baseline=np.interp(times,[0,tin[-1]],[qin[0],qin[-1]])
    excess=np.maximum(qg-baseline,0.)
    result=dict(scenario=scenario,state=state,n=n,dx=spacing,rtol=rtol,max_step=600,
        slope_factor=sf,extension_m=extension,original_port=info['original_outlet_node'],
        peak_seconds=tp,peak_cms=qp,peak_utc=str(origin+pd.to_timedelta(tp,unit='s')),
        timing_error_minutes=(tp-target)/60,soft_height_violation_cms=max(1800-qp,qp-1900,0),
        rmse_cms=float(np.sqrt(np.mean(errors**2))),bias_cms=float(errors.mean()),
        nse=float(1-np.sum(errors**2)/np.sum((observed[valid]-observed[valid].mean())**2)),
        n_comparison_samples=int(valid.sum()),comparison_window_complete=not calibration,
        centroid_seconds=float(np.trapz(times*excess,times)/np.trapz(excess,times)),
        spinup=spin,minimum_depth_m=float(h.min()),maximum_depth_m=float(h.max()),maximum_froude=maxfr,
        minimum_face_discharge_cms=minq,maximum_mass_residual_m3=float(abs(mass).max()),
        original_port_mass_residual_m3=float(abs(port_mass).max()),
        independent_inflow_volume_relative_error=volume_error,
        final_node_imbalance_fraction=float(np.max(abs(m.rhs(end,sol.y[:,-1],inflow)[:-2]*m.storage_area))/inflow(end)),
        original_links=len(frame),mesh_faces=len(m.faces),mesh_nodes=len(m.bed),elapsed_seconds=time.perf_counter()-start,
        calibration=calibration,wave=str((folder/'wave.parquet').relative_to(ROOT)),code_hashes=code_hashes())
    if not calibration and result['final_node_imbalance_fraction']>1e-6:raise ValueError('Unsettled tail')
    wave.to_parquet(folder/'wave.parquet',index=False)
    meta.write_text(json.dumps(result,indent=2)+'\n')
    print(f'{scenario} {state} n={n:.7f} dx={spacing:g}: peak error {result["timing_error_minutes"]:+.2f} min, peak {qp:.2f}, {result["elapsed_seconds"]:.1f}s',flush=True)
    return result


def code_hashes():
    return {p:sha(Path(__file__).with_name(p)) for p in ['diffusive_network.py','boundary_options.py','calibrate_boundaries56.py']}


def calibrate(scenario):
    trials=[]
    def evaluate(n):
        r=simulate(scenario,'base',float(n),calibration=True)
        trials.append(r)
        return r['timing_error_minutes']
    nodes=[.012,.020446875,.028,.04,.06]
    errors=[evaluate(n) for n in nodes]
    brackets=[(nodes[i],nodes[i+1]) for i in range(len(nodes)-1) if errors[i]*errors[i+1]<=0]
    if not brackets:raise ValueError('No timing bracket in predeclared roughness bounds')
    lo,hi=brackets[0]
    best=float(brentq(evaluate,lo,hi,xtol=5e-6))
    selected=simulate(scenario,'base',best,calibration=True)
    if selected['soft_height_violation_cms']>0:raise ValueError('Timing fit violates prior soft peak band')
    final=simulate(scenario,'base',best)
    # Coarse-search timing must survive fine run within the 5-minute observation step.
    if abs(final['timing_error_minutes'])>5:
        slope=(errors[nodes.index(hi)]-errors[nodes.index(lo)])/(hi-lo)
        best=float(best-final['timing_error_minutes']/slope)
        if not .012<=best<=.06:raise ValueError('Fine correction outside bounds')
        final=simulate(scenario,'base',best)
    if abs(final['timing_error_minutes'])>5:raise ValueError('Fine timing acceptance failed')
    selected=dict(scenario=scenario,n=best,objective='Reviewed peak timing; retain soft 1800–1900 m3/s band; RMSE/NSE diagnostics only',
        trial_count=len(trials),trials=trials,parent=final)
    (OUT/scenario/'calibration.json').write_text(json.dumps(selected,indent=2)+'\n')
    return selected


def protocol():
    files=[DATA/'observed_upstream.parquet',DATA/'observed_downstream.parquet']
    files +=[CACHE/(v+'.parquet') for v in STATES.values()]
    files +=[ROOT/'RAPID/outputs/bounded_beta_calibration_20261009/56/result.json']
    f=pd.read_parquet(files[0]);d=pd.read_parquet(files[1])
    base=pd.read_parquet(CACHE/(STATES['base']+'.parquet'))
    k=['u','v','link_length_m','width','head_us','head_ds'];pairs={}
    for state,fp in STATES.items():
        child=pd.read_parquet(CACHE/(fp+'.parquet'))
        common=base.merge(child,on=k)
        pairs[state]=dict(original_links=len(base),child_links=len(child),exactly_unchanged_links=len(common))
    return dict(scope='Historical cached example56; WSE as bed, measured widths and prescribed bed slopes fixed',
        calibration='Parent only per boundary scenario; one n applied unchanged to all children',
        target_utc=str(TARGET),soft_peak_band_cms=[1800,1900],n_bounds=[.012,.06],
        diagnostic_window=[str(FIT_START),str(FIT_END)],
        forcing='Identical raw point-flow series, linear interpolation, independent steady spinup and 96h constant tail',
        downstream_max_gap_seconds=float(d.time.diff().dt.total_seconds().max()),
        upstream_max_gap_seconds=float(f.time.diff().dt.total_seconds().max()),
        raw_observed_peak_cms=float(d.discharge_cms.max()),upstream_peak_cms=float(f.discharge_cms.max()),
        scenarios=SCENARIOS,states=STATES,pair_audits=pairs,
        extension='Idealized continuation of terminal width and bed slope; evaluate at original port with half-cell storage correction',
        code_hashes=code_hashes(),input_hashes={str(p.relative_to(ROOT)):sha(p) for p in files})


def main():
    p=argparse.ArgumentParser();p.add_argument('--workers',type=int,default=3);p.add_argument('--stage',choices=['all','calibrate','children','checks'],default='all');args=p.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    pth=OUT/'protocol.json';current=protocol()
    if pth.exists() and json.loads(pth.read_text())!=current:raise ValueError('Existing protocol differs')
    pth.write_text(json.dumps(current,indent=2)+'\n')
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        if args.stage in ['all','calibrate']:
            for job in as_completed([pool.submit(calibrate,s) for s in SCENARIOS]):
                r=job.result();print(f'SELECTED {r["scenario"]}: n={r["n"]:.8f}',flush=True)
        if args.stage in ['all','children']:
            jobs=[]
            for s in SCENARIOS:
                fit=json.loads((OUT/s/'calibration.json').read_text())
                jobs +=[pool.submit(simulate,s,state,fit['n']) for state in STATES if state!='base']
            for job in as_completed(jobs):job.result()
        if args.stage in ['all','checks']:
            # Extension-length check and mesh checks, without refitting n.
            jobs=[]
            for s in ['A_normal','D_extend10km']:
                fit=json.loads((OUT/s/'calibration.json').read_text())
                for state in ['base','S014_unit_10']:
                    jobs.append(pool.submit(simulate,s,state,fit['n'],dx=100.))
                    if s=='D_extend10km':jobs.append(pool.submit(simulate,s,state,fit['n'],extension_override=20000.))
            for job in as_completed(jobs):job.result()


if __name__=='__main__':main()
