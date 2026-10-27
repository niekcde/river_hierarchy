"""Current synthetic comparison: fixed BASED, fixed Manning, variable Manning/MCT.

Legacy routing.py and historical outputs are deliberately separate. Width is
constant within a reach. Flow splits use fixed local width fractions. Variable
routing solves trapezoidal continuity using corrected MCT storage; no clipping,
post-hoc volume correction, or frozen-state coefficient substitution is used.
"""
from dataclasses import dataclass, asdict
import ctypes
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

import numpy as np
import pandas as pd
from scipy.signal import lfilter

from rapid_tools import cunge
from rapid_tools.reference_section import predict_based_depth
from .networks import canonical_network

METHODS=('fixed_based','fixed_manning','variable_manning')
LABELS=dict(fixed_based='Fixed BASED + width scaling',fixed_manning='Fixed Manning (midpoint)',
            variable_manning='Variable Manning (conservative MCT)')

@dataclass(frozen=True)
class HydraulicSettings:
    method: str = 'fixed_based'
    n_manning: float = .025
    beta: float = 1.
    depth_alpha: float = .65
    baseflow_cms: float = 500.
    peak_cms: float = 1000.
    rise_hours: float = 12.
    fall_hours: float = 24.
    spinup_hours: float = 24.
    tail_hours: float = 192.
    segment_length_m: float = 200.
    dt_seconds: float = 15.
    report_seconds: float = 60.

    def __post_init__(self):
        if self.method not in METHODS: raise ValueError('Unknown hydraulic method')
        for key,value in asdict(self).items():
            if key!='method' and (not np.isfinite(value) or value<=0): raise ValueError(key+' must be positive')
        if self.peak_cms<=self.baseflow_cms: raise ValueError('Require peak > plateau')
        if not np.isclose(self.report_seconds/self.dt_seconds,round(self.report_seconds/self.dt_seconds)):
            raise ValueError('Reporting interval must be an integer number of timesteps')


def normal_depth(q,width,slope,n):
    return (n*np.asarray(q)/(np.asarray(width)*np.sqrt(slope)))**.6


def prepare(graph, parameters, settings):
    network=canonical_network(graph)
    f=pd.DataFrame(network['edges'],columns=['u','v','link_length_m','width','slope'])
    f['id_link']=np.arange(1,len(f)+1)
    ref=predict_based_depth(discharge_cms=settings.peak_cms,width_m=parameters['total_width_m'],
                            slope=parameters['reference_slope'])
    f['depth_m']=ref*(f.width/parameters['total_width_m'])**settings.depth_alpha
    # Reuse the established DAG allocation. No celerity clipping in this
    # consistency comparison: a cap would break normal-flow h/c compatibility.
    f=cunge.freeze_grid(f,settings.peak_cms,n_bounds=(settings.n_manning,settings.n_manning),
                        min_celerity=1e-12,max_celerity=1e12)
    f['count']=np.ceil(f.link_length_m/settings.segment_length_m).astype(int)
    f['rapid_effective_length_m']=f.link_length_m/f['count']
    if settings.method!='fixed_based':
        f['reference_discharge_cms']=.5*(settings.baseflow_cms+settings.peak_cms)*f.reference_flow_fraction
        f['depth_m']=normal_depth(f.reference_discharge_cms,f.width,f.slope,settings.n_manning)
    f['celerity_times_n']=(5/3)*f.depth_m**(2/3)*np.sqrt(f.slope)
    # Direct construction: spatial admissibility is checked through convergence
    # and native coefficient diagnostics, not freeze_grid's design-alpha bound.
    f['rapid_celerity_mps']=f.celerity_times_n/settings.n_manning
    f['rapid_k']=f.rapid_effective_length_m/f.rapid_celerity_mps
    f['rapid_x']=.5-settings.beta*f.reference_discharge_cms/(2*f.width*f.slope*f.rapid_celerity_mps*f.rapid_effective_length_m)
    f['hydraulic_diffusivity_m2_s']=settings.beta*f.reference_discharge_cms/(2*f.width*f.slope)
    f['method']=settings.method
    return f


_KERNEL=None

def _kernel():
    global _KERNEL
    if _KERNEL is None:
        source=Path(__file__).with_name('_mct.cpp')
        signature=hashlib.sha256(source.read_bytes()).hexdigest()[:20]
        suffix='.dylib' if sys.platform=='darwin' else '.so'
        target=Path(tempfile.gettempdir())/('synthetic_mct_'+signature+suffix)
        if not target.exists():
            compiler=shutil.which('c++') or shutil.which('clang++') or shutil.which('g++')
            if compiler is None: raise RuntimeError('Conservative MCT requires a C++ compiler (c++/clang++/g++)')
            with tempfile.TemporaryDirectory() as folder:
                built=Path(folder)/('kernel'+suffix)
                subprocess.run([compiler,'-O3','-std=c++11','-shared','-fPIC',str(source),'-o',str(built)],check=True,capture_output=True)
                shutil.copyfile(built,target)
        library=ctypes.CDLL(str(target)); fun=library.mct_cell
        vec=np.ctypeslib.ndpointer(dtype=np.float64,flags='C_CONTIGUOUS')
        fun.argtypes=[vec,vec,ctypes.c_long]+[ctypes.c_double]*6+[vec];fun.restype=ctypes.c_int
        _KERNEL=(library,fun)
    return _KERNEL[1]


def mct_cell(flow,width,slope,n,dx,dt,beta=1.):
    flow=np.ascontiguousarray(flow,dtype=float)
    if flow.ndim!=1 or len(flow)<2 or not np.isfinite(flow).all() or (flow<=0).any():
        raise ValueError('MCT requires finite positive inflow')
    if any(not np.isfinite(x) or x<=0 for x in (width,slope,n,dx,dt,beta)): raise ValueError('Invalid cell hydraulics')
    out=np.empty_like(flow);audit=np.zeros(5)
    status=_kernel()(flow,out,len(flow),width,slope,n,dx,dt,beta,audit)
    if status: raise ValueError(f'MCT nonlinear solve failed (code {status}); refine timestep/geometry')
    return out,dict(initial_storage_m3=audit[0],final_storage_m3=audit[1],integrated_in_minus_out_m3=audit[2],
                    maximum_cell_balance_residual_m3=audit[3],maximum_iterations=int(audit[4]))


def forcing(settings):
    start=settings.spinup_hours*3600; peak=start+settings.rise_hours*3600
    end=peak+settings.fall_hours*3600;total=end+settings.tail_hours*3600
    size=int(np.ceil(total/settings.report_seconds))*int(round(settings.report_seconds/settings.dt_seconds))
    t=np.arange(size+1)*settings.dt_seconds
    q=np.interp(t,[0,start,peak,end,t[-1]], [settings.baseflow_cms,settings.baseflow_cms,settings.peak_cms,settings.baseflow_cms,settings.baseflow_cms])
    return t,q


def route(frame, flow, settings):
    """Native endpoint traces, with exact trapezoidal mass bookkeeping."""
    order,edges,source,sink=cunge.topology(frame)
    signals={source:np.asarray(flow,float)};initial=final=balance=0.;minimum=float('inf');residual=0.;iterations=0
    for u in order:
        sig=signals.pop(u)
        if u==sink:
            integrated=float(np.trapz(flow-sig,dx=settings.dt_seconds))
            return sig,dict(initial_storage_m3=initial,final_storage_m3=final,
                network_mass_residual_m3=integrated-(final-initial),
                maximum_cell_balance_residual_m3=residual,minimum_native_flow_cms=minimum,
                maximum_iterations=iterations)
        for i in edges[u]:
            r=frame.iloc[i];val=np.ascontiguousarray(sig*r.split_fraction)
            for _ in range(int(r['count'])):
                if settings.method=='variable_manning':
                    val,a=mct_cell(val,r.width,r.slope,settings.n_manning,r.rapid_effective_length_m,settings.dt_seconds,settings.beta)
                    initial+=a['initial_storage_m3'];final+=a['final_storage_m3']
                    residual=max(residual,a['maximum_cell_balance_residual_m3']);iterations=max(iterations,a['maximum_iterations'])
                else:
                    k=r.rapid_k;x=r.rapid_x;dt=settings.dt_seconds;den=dt+2*k*(1-x)
                    aa=(dt-2*k*x)/den;bb=(dt+2*k*x)/den;dd=(2*k*(1-x)-dt)/den
                    if abs(dd)>=1: raise ValueError('Unstable fixed routing timestep; reduce dt')
                    inlet_first=val[0];inlet_last=val[-1]
                    val,_=lfilter([aa,bb],[1.,-dd],val,zi=[(bb+dd)*inlet_first])
                    initial+=k*inlet_first;final+=k*(x*inlet_last+(1-x)*val[-1])
                if not np.isfinite(val).all() or val.min()<0: raise ValueError('Invalid native routed discharge')
                minimum=min(minimum,float(val.min()))
            signals[r.v]=signals.get(r.v,0.)+val
    raise ValueError('No outlet')


def summarize(t,q,baseline):
    e=q-baseline;volume=float(np.trapz(e,t))
    if volume<=0: raise ValueError('No positive event volume')
    return dict(centroid_seconds=float(np.trapz(t*e,t)/volume),peak_seconds=float(t[np.argmax(q)]),
                peak_cms=float(q.max()),volume_m3=volume,minimum_excess_cms=float(e.min()),final_excess_cms=float(e[-1]))


def reported(t,q,settings):
    n=int(round(settings.report_seconds/settings.dt_seconds))
    return pd.DataFrame(dict(time_seconds=(t[::n][:-1]+t[::n][1:])/2,
        discharge_cms=((q[:-1]+q[1:])/2).reshape(-1,n).mean(axis=1)))
