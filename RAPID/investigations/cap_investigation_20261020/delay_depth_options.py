"""Three depth options on two complete parent/child networks; frozen roughness.

Wide rectangular normal-depth closure. Dynamic option follows iterative
four-point averaging of c and q from Ponce & Yevjevich (1978), not a HEC clone.
"""
import ctypes
import hashlib
import json
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd

from delay_side_project import ROOT, OUT, EXT, SOURCES, read
from delay_kx_controls import metric
from rapid_tools.cunge import topology
from rapid_tools.common_grid import route_interval_means, outlet_interval_means

DEST=OUT/'delay_depth_options_20261027'
CPP=Path(__file__).with_name('variable_manning.cpp')
LIB=Path('/private/tmp/river_variable_manning.dylib')


def load_engine():
    subprocess.run(['clang++','-O3','-std=c++17','-dynamiclib',str(CPP),'-o',str(LIB)],check=True)
    lib=ctypes.CDLL(str(LIB))
    fun=lib.route_cells
    ptr=np.ctypeslib.ndpointer(dtype=np.float64,ndim=1,flags='C_CONTIGUOUS')
    fun.argtypes=[ptr,ctypes.c_long,ctypes.c_int]+[ctypes.c_double]*6+[ctypes.c_int]+[ctypes.c_double]*2+[ptr]
    fun.restype=ctypes.c_int
    return fun


def dynamic_route(fun,f,flow,dt,initial,variable=True):
    ns=round(60/dt);assert abs(ns*dt-60)<1e-9
    order,edges,root,end=topology(f)
    signals={root:np.repeat(flow,ns)}
    audit=np.array([np.inf,0.,0.,1.,np.inf,0.,np.inf,0.])
    for u in order:
        sig=signals.pop(u)
        if u==end:
            q=outlet_interval_means(sig,ns,initial_flow=initial)
            inlet_volume=flow.sum()*60+.5*dt*(initial-flow[-1])
            residual=inlet_volume-q.sum()*60-audit[1]
            scale=np.sum(abs(flow-initial))*60
            return q,dict(min_flow=audit[0],storage_change_m3=audit[1],iterations_max=audit[2],previous_outflow_coefficient_min=audit[3],
                          depth_min=audit[4],depth_max=audit[5],celerity_min=audit[6],celerity_max=audit[7],
                          storage_balance_residual_m3=residual,storage_balance_residual_pct=100*residual/scale if scale else 0.,
                          final_outlet=sig[-1],final_inlet=flow[-1])
        for i in edges[u]:
            r=f.iloc[i];val=np.ascontiguousarray(sig*r.split_fraction)
            status=fun(val,len(val),int(r['count']),dt,r.rapid_effective_length_m,r.width,r.slope,r.n_manning,
                       initial*r.reference_flow_fraction,int(variable),r.rapid_k,r.rapid_x,audit)
            if status:raise RuntimeError(f'Variable routing failed: status {status}, link {r.id_link}')
            if audit[3]<-1e-12:raise RuntimeError('Previous-outflow coefficient stability bound exceeded')
            signals[r.v]=signals.get(r.v,0.)+val
    raise RuntimeError('No outlet')


def normal_parameters(f):
    f=f.copy();q=f.reference_discharge_cms
    f['depth_m']=(q*f.n_manning/(f.width*np.sqrt(f.slope)))**.6
    f['rapid_celerity_mps']=5/3*f.depth_m**(2/3)*np.sqrt(f.slope)/f.n_manning
    f['rapid_k']=f.rapid_effective_length_m/f.rapid_celerity_mps
    f['rapid_x']=.5*(1-q/(f.width*f.slope*f.rapid_celerity_mps*f.rapid_effective_length_m))
    np.testing.assert_allclose(f.width/f.n_manning*f.depth_m**(5/3)*np.sqrt(f.slope),q,rtol=1e-12)
    return f


def tests(fun):
    f=pd.DataFrame([dict(u=0,v=1,id_link=0,count=3,rapid_effective_length_m=1000.,width=100.,slope=.0001,
                        n_manning=.035,reference_discharge_cms=100.,reference_flow_fraction=1.,split_fraction=1.)])
    f=normal_parameters(f)
    flow=100+20*np.sin(np.linspace(0,4*np.pi,500))**2
    q,_=dynamic_route(fun,f,flow,15,100,False)
    expected,_=route_interval_means(f,flow,4,15,initial_flow=100)
    np.testing.assert_allclose(q,expected,rtol=1e-12,atol=1e-10)
    steady,a=dynamic_route(fun,f,np.full(500,100.),15,100)
    np.testing.assert_allclose(steady,100,rtol=0,atol=1e-10)
    # Complete pulse has identical initial/final storage; volume error is unambiguous.
    pulse=np.r_[100+50*np.sin(np.linspace(0,np.pi,12*60))**2,np.full(10*24*60,100.)]
    rows=[]
    for dt in [30.,15.]:
        out,a=dynamic_route(fun,f,pulse,dt,100)
        error=100*((out-100).sum()/(pulse-100).sum()-1)
        assert abs(out[-1]-100)<1e-6
        rows.append(dict(test='single_reach_closed_pulse',dt_s=dt,volume_error_pct=error,
                         meets_half_percent_volume_screen=abs(error)<.5,**a))
    assert abs(rows[0]['volume_error_pct']-rows[1]['volume_error_pct'])<.01
    return rows


def main():
    DEST.mkdir(exist_ok=True);fun=load_engine()
    checks=tests(fun);pd.DataFrame(checks).to_csv(DEST/'implementation_checks.csv',index=False)
    print('Implementation checks:',checks,flush=True)
    old,new=read(EXT/'state_metrics.parquet'),read(OUT/'review/uncapped_state_metrics.parquet')
    states=pd.concat([old[~old.example.isin(new.example)],new]);states=states[states.window.eq('0.5%')].set_index(['example','state'])
    frames=read(OUT/'delay_side_project_20261026/routing_links.parquet')
    rows,waves,hydraulics=[],[],[]
    for ex,state in [('02','S013_unit_20'),('22','S057_unit_5')]:
        pr=states.loc[ex,'S000_base'];forcing=read(Path(pr.outlet).parents[1]/'forcing.parquet')
        nb=int(forcing.phase.eq('spinup').sum());flow=forcing.forcing_cms.to_numpy()[nb:]
        initial=float(forcing.forcing_cms.iloc[0]);t=forcing.time_utc.astype('int64').to_numpy()/1e9
        a,b,base=pr.rise_seconds,pr.fall_seconds,pr.baseline_cms
        for option in ['current_fixed','manning_fixed','manning_variable']:
            for dt in [30.,15.]:
                reference={}
                for side in ['parent','child']:
                    f=frames[(frames.example==ex)&(frames.state==state)&(frames.side==side)].reset_index(drop=True).copy()
                    if option!='current_fixed':f=normal_parameters(f)
                    if dt==30:
                        hydraulics.append(f.assign(option=option))
                    begin=time.monotonic()
                    if option=='manning_variable':q,audit=dynamic_route(fun,f,flow,dt,initial)
                    else:
                        q,audit=route_interval_means(f,flow,round(60/dt),dt,initial_flow=initial)
                        assert audit['minimum_native_flow_cms']>=-1e-9
                    q=np.r_[np.full(nb,initial),q]
                    record=dict(example=ex,state=state,option=option,dt_s=dt,side=side,seconds=time.monotonic()-begin,**audit)
                    for window,lo,hi in [('event',a,b),('saved_support',t[0],t[-1])]:
                        values=metric(t,q,lo,hi,base)
                        if side=='parent':reference[window]=values
                        ref=reference[window]
                        rec=dict(**record,window=window,centroid_shift_min=(values['centroid_s']-ref['centroid_s'])/60,
                                 peak_shift_min=(values['peak_s']-ref['peak_s'])/60,volume_change_pct=100*(values['volume']/ref['volume']-1),
                                 centroid_hours_from_window_start=(values['centroid_s']-lo)/3600,peak_hours_from_window_start=(values['peak_s']-lo)/3600)
                        rows.append(rec)
                        if window=='event':print({k:rec[k] for k in ['example','option','dt_s','side','centroid_shift_min','peak_shift_min','seconds']},flush=True)
                    if dt==15:waves.append(pd.DataFrame(dict(example=ex,state=state,option=option,side=side,hours=(t-a)/3600,excess_q=q-base)))
                    pd.DataFrame(rows).to_csv(DEST/'results.csv',index=False)
        # Reproduce original frozen comparison with the original native clock.
        for side,name in [('parent','S000_base'),('child',state)]:
            f=frames[(frames.example==ex)&(frames.state==state)&(frames.side==side)].reset_index(drop=True)
            dt=2.5 if ex=='02' else 60/14
            q,_=route_interval_means(f,flow,round(60/dt),dt,initial_flow=initial)
            saved=read(states.loc[ex,name].outlet)
            np.testing.assert_allclose(np.r_[np.full(nb,initial),q],saved.q_outlet_cms,rtol=0,atol=1e-7)
    result=pd.DataFrame(rows)
    coarse=result[result.dt_s==30].merge(result[result.dt_s==15],on=['example','state','option','side','window'],suffixes=('_30','_15'),validate='one_to_one')
    coarse['centroid_clock_difference_s']=60*(coarse.centroid_shift_min_15-coarse.centroid_shift_min_30)
    coarse.to_csv(DEST/'clock_check.csv',index=False)
    pd.concat(waves).to_parquet(DEST/'hydrographs.parquet',index=False)
    pd.concat(hydraulics).to_parquet(DEST/'reference_hydraulics.parquet',index=False)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,3,figsize=(15,7),constrained_layout=True)
    for i,ex in enumerate(['02','22']):
        for j,opt in enumerate(['current_fixed','manning_fixed','manning_variable']):
            for w in waves:
                if w.example.iloc[0]==ex and w.option.iloc[0]==opt:
                    axes[i,j].plot(w.hours,w.excess_q,label=w.side.iloc[0],linewidth=1)
            axes[i,j].set(xlim=(0,100 if ex=='02' else 255),title=f'{ex}: {opt}',xlabel='Hours from original window start',ylabel='Excess Q (m³/s)')
            axes[i,j].legend();axes[i,j].grid(alpha=.2)
    fig.savefig(DEST/'comparison.png',dpi=140);plt.close(fig)
    for path in [Path(__file__),CPP,ROOT/'RAPID/src/rapid_tools/cunge.py',ROOT/'RAPID/src/rapid_tools/common_grid.py']:
        SOURCES[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
    for path,digest in SOURCES.items():assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==digest
    (DEST/'manifest.json').write_text(json.dumps(dict(sources=SOURCES,
      scope='Whole-network depth alternatives; original geometry, slopes, n, cells, branch fractions, forcing, windows and baseline held fixed. No recalibration or production changes.',
      dynamic_method='Iterative four-point arithmetic averages of local point celerities and unit discharges. Three-point predictor. Relative iteration tolerance 1e-10, at most 40 iterations. Wide rectangular Manning closure; no celerity clipping.',
      reference='https://apo.sdsu.edu/muskingum_cunge_method_with_variable_parameters.html',
      mass_audit='Dynamic storage balance uses trapezoidal physical cross-section storage at final state. This is an approximate network diagnostic, not an exact discrete conservation identity. Independent closed-pulse test reports unambiguous volume error.'),indent=2)+'\n')
    print(result[(result.dt_s==15)&(result.side=='child')&(result.window=='event')][['example','option','centroid_shift_min','peak_shift_min']].to_string(index=False),flush=True)
    print('Max clock difference s',coarse.centroid_clock_difference_s.abs().max(),flush=True)


if __name__=='__main__':main()
