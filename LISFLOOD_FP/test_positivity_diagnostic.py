"""Focused conservation/positivity tests for the experimental flux treatment."""
from pathlib import Path
import numpy as np
from inertial_positivity_diagnostic_kernel import donor_limit,integrate
from inertial_verification_kernel import integrate as original

def check(name,h,a,b,q,inlet=0.,normal=False):
    h=np.array(h,float);N=len(h);a=np.array(a);b=np.array(b);q=np.array(q,float);area=np.ones(N);wf=np.ones(len(q));before=q.copy()
    count,volume,factor=donor_limit(h,area,a,b,wf,q,0,inlet,1.)
    Q=q*wf;balance=np.bincount(b,weights=Q,minlength=N+1)-np.bincount(a,weights=Q,minlength=N+1);balance[0]+=inlet
    trial=h+balance[:N];boundary=Q[b==N].sum()
    assert min(trial)>=-1e-14,(name,trial)
    assert abs(trial.sum()-h.sum()-inlet+boundary)<1e-12
    assert np.all(before*q>=0)
    print(name,'PASS',count,trial)
check('multiple_outgoing',[1,0,0],[0,0],[1,2],[2,3])
check('reverse_flow',[0,1,0],[0,2],[1,1],[-2,-3])
check('normal_outlet_limited',[1],[0],[1],[10])
check('boundary_inflow',[0],[0],[1],[-10])
check('incoming_not_spent_twice',[1,0,0],[0,1],[1,2],[10,10])
check('external_inflow',[0,0],[0],[1],[10],inlet=2)
r=Path(__file__).resolve().parent/'results/broad_15_100m/investigation';d=dict(np.load(r/'limiter_retained_48_n0.0207996059_dx50/_input_29515.npz'))
p=dict(np.load(r/'first_clipping48/first_failure.npz'));d.update(h=p['pre_h'],q=p['pre_q'],times=np.array([0.,p['step'][1]]),forcing=np.full(2,p['step'][2]),sample_indices=np.array([0,1]));d={k:(v.item() if np.ndim(v)==0 else v) for k,v in d.items()}
x=integrate(**d);assert x[-1][2]<1e-9;assert x[-1][3]<1e-6;assert x[-1][11]>0;print('exact_failure_replay PASS',x[-1])
# Analytic uniform Manning equilibrium, and exact agreement when limiter inactive.
N=10;h=np.full(N,2.);n=.025;slope=.001;q=2**(5/3)*np.sqrt(slope)/n
m=dict(z=np.r_[3-(np.arange(N)+.5)*.2,1.],w=np.full(N,100.),dx=np.full(N,200.),a=np.arange(N),b=np.arange(1,N+1),wf=np.full(N,100.),dxf=np.r_[np.full(N-1,200.),100.],source=0,n=n,outlet_slope=slope,stage=0.,normal=True,times=np.array([0.,3600.]),forcing=np.full(2,q*100),h=h,q=np.full(N,q),cfl=.35,max_dt=150.,sample_indices=np.array([0,1]))
x=integrate(**m);old=original(**m);assert np.max(abs(x[3]-h))<1e-12;assert x[-1][11]==0;assert np.array_equal(x[0],old[0]);print('uniform_manning_and_inactive_equivalence PASS')
m.update(normal=False,stage=3.,h=3.-m['z'][:-1],q=np.zeros(N),forcing=np.zeros(2));x=integrate(**m);assert np.max(abs(x[3]-m['h']))<1e-12;assert np.max(abs(x[4]))<1e-12;print('lake_at_rest PASS')
