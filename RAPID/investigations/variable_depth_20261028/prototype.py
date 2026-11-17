"""Isolated four-point iterative Manning-depth prototype; not production approved.

Original cells and width-based branch allocation are preserved. Coefficients use
means of the four stencil celerities and discharges, not c(mean discharge).
The predictor uses the three known stencil values, then corrects to 1e-10
relative discharge tolerance (40 iterations maximum). No clipping of depth/flow.
"""
import ctypes,hashlib,json,subprocess,sys
from pathlib import Path
import numpy as np
import pandas as pd
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'cap_investigation_20261020'))
import run_three_runs as r
from rapid_tools.cunge import topology
from rapid_tools.common_grid import outlet_interval_means
DEST=r.OUT/'variable_depth_preparation_20261028'

def engine(source_name="engine.cpp"):
 source=HERE/source_name;key=hashlib.sha256(source.read_bytes()).hexdigest()[:16]
 lib=Path('/private/tmp')/f'variable_depth_{key}.dylib'
 if not lib.exists():subprocess.run(['clang++','-O3','-std=c++17','-dynamiclib',str(source),'-o',str(lib)],check=True)
 fun=ctypes.CDLL(str(lib)).route_cells
 ptr=np.ctypeslib.ndpointer(dtype=np.float64,ndim=1,flags='C_CONTIGUOUS')
 fun.argtypes=[ptr,ctypes.c_long,ctypes.c_int]+[ctypes.c_double]*6+[ctypes.c_int]+[ctypes.c_double]*3+[ptr]
 fun.restype=ctypes.c_int
 return fun

def route(fun,frame,flow,dt,initial,beta=1.,variable=True):
 ns=round(60/dt)
 if not np.isclose(ns*dt,60) or dt<=0:raise ValueError('Clock must divide 60 seconds')
 flow=np.asarray(flow,float)
 if not np.isfinite(flow).all() or np.any(flow<=0) or initial<=0:raise ValueError('Prototype requires strictly positive flow')
 if not np.isfinite(beta) or beta<=0:raise ValueError('Invalid beta')
 order,edges,root,end=topology(frame);signals={root:np.repeat(flow,ns)};rows=[]
 for u in order:
  sig=signals.pop(u)
  if u==end:
   means=outlet_interval_means(sig,ns,initial_flow=initial)
   audit=pd.DataFrame(rows);storage=float(audit.storage_change_m3.sum())
   vin=flow.sum()*60+.5*dt*(initial-flow[-1]);residual=vin-means.sum()*60-storage
   scale=np.sum(abs(flow-initial))*60
   return means,dict(storage_balance_residual_m3=float(residual),residual_pct_excess_volume=float(100*residual/scale) if scale else None,
      inlet_reconstruction_difference_m3=float(vin-flow.sum()*60),physical_storage_change_m3=storage,
      final_outlet_cms=float(sig[-1]),iterations_max=float(audit.iterations_max.max()),depth_min_m=float(audit.depth_min_m.min()),
      depth_max_m=float(audit.depth_max_m.max()),minimum_flow_cms=float(audit.minimum_flow_cms.min())),audit
  for i in edges[u]:
   row=frame.iloc[i];val=np.ascontiguousarray(sig*row.split_fraction);a=np.array([np.inf,0.,0.,1.,np.inf,0.,np.inf,0.])
   status=fun(val,len(val),int(row['count']),dt,row.rapid_effective_length_m,row.width,row.slope,row.n_manning,
      initial*row.reference_flow_fraction,int(variable),beta,row.rapid_k,row.rapid_x,a)
   if status:raise RuntimeError(f'Iteration/flow failure status={status}, link={row.id_link}')
   if a[3]<-1e-12:raise RuntimeError(f'Previous-outflow timestep bound violated, link={row.id_link}')
   rows.append(dict(id_link=int(row.id_link),minimum_flow_cms=a[0],storage_change_m3=a[1],iterations_max=a[2],
      previous_outflow_coefficient_min=a[3],depth_min_m=a[4],depth_max_m=a[5],celerity_min_mps=a[6],celerity_max_mps=a[7],
      depth_width_ratio_max=a[5]/row.width))
   signals[row.v]=signals.get(row.v,0.)+val
 raise RuntimeError('No outlet')

def parameters(f,n,beta):return r.hydraulic_parameters(f,r.RUNS[2],n,beta)

def implementation_checks():
 fun=engine();DEST.mkdir(exist_ok=True)
 f=pd.DataFrame([dict(u=0,v=1,id_link=1,count=3,link_length_m=3000.,rapid_effective_length_m=1000.,width=100.,slope=.0001,
    reference_discharge_cms=100.,reference_flow_fraction=1.,split_fraction=1.)]);rows=[]
 for beta in [.5,1.,2.]:
  g=parameters(f,.035,beta)
  pulse=np.r_[100+50*np.sin(np.linspace(0,np.pi,720))**2,np.full(14400,100.)]
  fixed,_,_=route(fun,g,pulse,15,100,beta,False)
  expected,_=r.solve_wave(g,pulse,15,100)
  np.testing.assert_allclose(fixed,expected,rtol=1e-11,atol=1e-9)
  steady,_,_=route(fun,g,np.full(60,100.),15,100,beta)
  np.testing.assert_allclose(steady,100,rtol=0,atol=1e-9)
  for dt in [15.,7.5]:
   q,a,_=route(fun,g,pulse,dt,100,beta)
   assert abs(q[-1]-100)<1e-6
   rows.append(dict(beta=beta,dt=dt,closed_pulse_volume_error_pct=100*((q-100).sum()/(pulse-100).sum()-1),**a))
 pd.DataFrame(rows).to_csv(DEST/'implementation_checks.csv',index=False)
 print(pd.DataFrame(rows).to_string(index=False),flush=True)
 return rows

if __name__=='__main__':implementation_checks()
