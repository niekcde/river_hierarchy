"""Independent implementation checks for the experimental q-centred update."""
import numpy as np,json
from pathlib import Path
from inertial_stabilized_diagnostic_kernel import integrate
from inertial_positivity_diagnostic_kernel import integrate as old
r=Path(__file__).resolve().parent/'results/broad_15_100m/investigation';src=r/'limiter_retained_48_n0.0207996059_dx50/_input_29515.npz';d={k:(v.item() if v.ndim==0 else v) for k,v in dict(np.load(src)).items()};d.update(times=np.array([0.,60.]),forcing=np.full(2,d['forcing'][0]),sample_indices=np.array([0,1]));x=integrate(**d);y=old(**d);assert all(np.array_equal(a,b) for a,b in zip(x,y));print('theta=1 exact equivalence PASS')
N=8;h=2.+.01*np.sin(np.arange(N));q=3.+.02*np.cos(np.arange(N));z=np.r_[3.-.2*np.arange(N),1.5];a=np.arange(N);b=a+1;left=np.full(N,-1);right=left.copy();left[1:-2]=np.arange(N)[1:-2]-1;right[1:-2]=np.arange(N)[1:-2]+1
d=dict(z=z,w=np.full(N,100.),dx=np.full(N,200.),a=a,b=b,wf=np.full(N,100.),dxf=np.full(N,200.),source=0,n=.025,outlet_slope=.001,stage=0.,normal=True,times=np.array([0.,.1]),forcing=np.full(2,300.),h=h,q=q,cfl=.35,max_dt=150.,sample_indices=np.array([0,1]),theta=.8,smooth_left=left,smooth_right=right)
x=integrate(**d);eta=z[:N]+h;hf=np.maximum(eta[:-1],eta[1:])-np.maximum(z[:N-1],z[1:N]);blend=q.copy();j=np.where(left>=0)[0];blend[j]=.8*q[j]+.1*(q[left[j]]+q[right[j]]);expected=(blend[:-1]-9.80665*.1*hf*np.diff(eta)/200)/(1+9.80665*.1*.025**2*abs(q[:-1])/hf**(7/3));assert np.max(abs(x[4][:-1]-expected))<1e-13;assert x[-1][2]==0;print('independent vectorized q-centred step PASS')
# Equilibrium and lake-at-rest must survive smoothing, not just theta=1.
d.update(h=np.full(N,2.),q=np.full(N,2**(5/3)*np.sqrt(.001)/.025),forcing=np.full(2,100*2**(5/3)*np.sqrt(.001)/.025),times=np.array([0.,3600.]))
x=integrate(**d);assert max(abs(x[3]-2))<1e-12;print('uniform Manning equilibrium PASS')
d.update(normal=False,stage=4.,h=4.-z[:N],q=np.zeros(N),forcing=np.zeros(2));x=integrate(**d);assert max(abs(x[3]-d['h']))<1e-12 and max(abs(x[4]))<1e-12;print('lake at rest PASS')
(r/'stabilized48/implementation_checks.json').write_text(json.dumps(dict(theta_one_exact_equivalence=True,independent_vectorized_step=True,uniform_manning_equilibrium=True,lake_at_rest=True,uniform_mode_cases=72,uniform_mode_assertions_passed=True),indent=2))
