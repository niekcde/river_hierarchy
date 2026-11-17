"""Independent steady backwater ODE and smooth-channel pulse refinement."""
import json
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
import verify_inertial56 as v

BASE=v.OUT
v.OUT=BASE/'synthetic_benchmarks';v.OUT.mkdir(exist_ok=True)

def main():
    rows=[];Q=100.;n=.025;B=100.;S=.0002;L=2000.
    hn=(Q*n/(B*np.sqrt(S)))**.6;hout=hn+.5
    # Steady local-inertial balance (no advective acceleration): dh/dx=S0-Sf.
    solution=solve_ivp(lambda x,h:S-n*n*(Q/B)**2/h**(10/3),(L,0),[hout],rtol=1e-11,atol=1e-12,dense_output=True)
    for count in [10,20,40,80]:
        f=v.channel(count,length=L,width=B,slope=S);stage=float(f.head_ds.iloc[-1]+hout)
        m=v.build(f,n,normal=False,stage=stage);initial,days,audit=v.spin(m,Q)
        exact=solution.sol((np.arange(count)+.5)*L/count)[0]
        err=float(abs(initial[0]-exact).max())
        rows.append(dict(test='steady_backwater_ODE',cells=count,maximum_depth_error_m=err,**audit))
    errors=[r['maximum_depth_error_m'] for r in rows]
    assert np.all(np.diff(errors)<0) and errors[-1]<.02,errors
    pulse=[];times=np.arange(0,18001,30.);forcing=100+100*np.exp(-((times-3600)/900)**2)
    for count,cfl in [(10,.35),(20,.35),(40,.35),(80,.35),(80,.175)]:
        f=v.channel(count);m=v.build(f);initial,days,audit=v.spin(m,100.)
        result=v.solve(m,times,forcing,initial,cfl=cfl,max_dt=30)
        q=np.diff(result['wave'][:,4])/30;t=(times[:-1]+times[1:])/2
        metric=v.metrics(t,q,np.full(len(q),100.));row=dict(test='uniform_channel_pulse',cells=count,cfl=cfl,**metric,**result['audit']);pulse.append(row)
        assert row['clipping_volume_m3']<1e-6 and row['limited_face_steps']==0 and row['maximum_mass_residual_m3']<1e-5
    assert abs(pulse[-1]['peak_seconds']-pulse[-2]['peak_seconds'])<1.
    pd.DataFrame(rows).to_csv(v.OUT/'steady_backwater.csv',index=False)
    pd.DataFrame(pulse).to_csv(v.OUT/'pulse_convergence.csv',index=False)
    result=dict(backwater_errors_m=errors,backwater_ODE_converges=True,pulse_time_step_peak_difference_s=pulse[-1]['peak_seconds']-pulse[-2]['peak_seconds'],pulse_spatial_peak_changes_s=np.diff([r['peak_seconds'] for r in pulse[:4]]).tolist(),passed=True)
    (BASE/'additional_benchmarks.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2),flush=True)

if __name__=='__main__':main()
