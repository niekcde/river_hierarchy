"""Diagnostic: retain the supplied scheme's limiter instead of rejecting its use.
Does not promote capped runs into the strict broad-study accepted population.
"""
import json,time
from pathlib import Path
import numpy as np
import pandas as pd
import broad_study as b

def run(ex='48',n=None,cfl=.35,length=100):
    tag='limiter_retained_'+ex+('' if n is None else f'_n{n:.10f}')+('' if cfl==.35 else f'_cfl{cfl:g}')+('' if length==100 else f'_dx{length:g}')
    dest=b.OUT/'investigation'/tag;dest.mkdir(exist_ok=True)
    if (dest/'result.json').exists():return json.loads((dest/'result.json').read_text())
    start=time.monotonic();f=pd.read_parquet(b.OUT/'investigation/current_mc/states'/ex/'S000_base/parameters.parquet');n=float(f.n_manning.iloc[0]) if n is None else float(n);cells=b.v.grid(f,f'max{length:g}');m=b.build(cells,n);force=b.forcing(ex);Q=float(force.forcing_cms.iloc[0]);initial=b.initial_guess(f,cells,m,Q)
    candidates=[]
    for p in (b.OUT/ex).glob(f'S000_base_dx{length:g}_n*_full_cfl*/initial.npz'):
        try:nn=float(p.parent.name.split('_n')[1].split('_')[0]);candidates.append((abs(np.log(nn/n)),p))
        except ValueError:pass
    if candidates:
        with np.load(min(candidates)[1]) as d:initial=(d['h'].copy(),d['q'].copy())
    b.v.OUT=dest
    for day in range(1,61):
        spun=b.v.solve(m,[0,86400],[Q,Q],initial,cfl=cfl,max_dt=150);initial=(spun['h'],spun['q'])
        if spun['audit']['final_node_imbalance_cms']/max(Q,1)<1e-8:break
    else:raise RuntimeError('Steady initial state did not converge')
    np.savez(dest/'initial.npz',h=initial[0],q=initial[1]);origin=force.interval_start_utc.iloc[0];times=np.r_[(force.interval_start_utc-origin).dt.total_seconds(),(force.interval_end_utc.iloc[-1]-origin).total_seconds()];vals=np.r_[force.forcing_cms,force.forcing_cms.iloc[-1]]
    result=b.v.solve(m,times,vals,initial,cfl=cfl,max_dt=150,samples=np.array([0,len(times)-1]));means=np.diff(result['wave'][:,4])/60;w=force[['time_utc','phase']].copy();w['q_outlet_cms']=means;w.to_parquet(dest/'wave.parquet',index=False)
    obs=pd.read_parquet(b.OUT/'inputs'/ex/'observed_downstream.parquet');mask=(w.time_utc>=obs.time.min())&(w.time_utc<=obs.time.max())&(w.time_utc>=force.loc[force.phase.eq('input_event'),'interval_start_utc'].iloc[0]);t=(w.time_utc-origin).dt.total_seconds().to_numpy();tm,h,kind,_=b.feature(means[mask],t[mask],ex,b.TARGETS[ex]);target=(pd.Timestamp(b.TARGETS[ex]['target_utc'])-origin).total_seconds();a=result['audit'];rel=float(a['maximum_mass_residual_m3']/result['wave'][-1,3])
    r=dict(example=ex,n=n,cfl=cfl,length=length,timing_error_minutes=(tm-target)/60,feature_kind=kind,feature_height_cms=h,spinup_days=day,spinup_audit=spun['audit'],event_audit=a,relative_mass_residual=rel,average_limited_faces_per_step=a['limited_face_steps']/a['steps'],limited_face_step_fraction=a['limited_face_steps']/(a['steps']*len(m['a'])),finite_and_conservative=bool(np.isfinite(means).all() and a['clipping_volume_m3']<1e-6 and rel<1e-9),strict_no_limiter_accepted=a['limited_face_steps']==0,seconds=time.monotonic()-start,folder=str(dest),scope='Original network; supplied critical-flow limiter retained; abort-on-limiter disabled only in this separate diagnostic. No automatic promotion into production.')
    b.save(dest/'result.json',r);print(json.dumps(r,indent=2),flush=True);return r

if __name__=='__main__':run()
