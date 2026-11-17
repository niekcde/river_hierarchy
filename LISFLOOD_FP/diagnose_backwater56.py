"""Inspect actual hydraulic gradients; do not infer backwater from outlet timing."""
import json
import numpy as np
import pandas as pd
from calibrate_boundaries56 import OUT, CACHE, STATES, SCENARIOS, forcing_data
from boundary_options import build_boundary


def main():
    dest=OUT/'depth_diagnostics';dest.mkdir(exist_ok=True)
    rows=[]
    for state,fp in STATES.items():
        frame=pd.read_parquet(CACHE/(fp+'.parquet'))
        row=dict(state=state,mc_cached_reference_min_m=float(frame.depth_m.min()),
                 mc_cached_reference_max_m=float(frame.depth_m.max()))
        for s in SCENARIOS:
            n=json.loads((OUT/s/'calibration.json').read_text())['n']
            records=[json.loads(p.read_text()) for p in (OUT/s/'runs').glob('*/result.json')]
            r=next(r for r in records if r['state']==state and r['dx']==200 and r['extension_m']==SCENARIOS[s][1] and abs(r['n']-n)<1e-9)
            row[s+'_max_m']=r['maximum_depth_m']
        rows.append(row)
    pd.DataFrame(rows).to_csv(dest/'depth_comparison.csv',index=False)
    print(pd.DataFrame(rows).to_string(index=False),flush=True)
    frame=pd.read_parquet(CACHE/(STATES['base']+'.parquet'))
    n=json.loads((OUT/'A_normal/calibration.json').read_text())['n']
    m,info=build_boundary(frame,n,200)
    origin,tin,qin=forcing_data();end=tin[-1]+96*3600
    times=np.unique(np.r_[np.arange(0,end,300.),end])
    inflow=lambda t:np.interp(t,tin,qin)
    initial,spin=m.spinup(float(qin[0]))
    sol=m.integrate(initial,times,inflow,max_step=600,rtol=2e-7)
    h=sol.y[:-2];bed_slope=(m.bed[m.a]-m.bed[m.b])/m.dx
    ratio=1+(h[m.a]-h[m.b])/(m.dx*bed_slope)[:,None]
    snapshots={}
    for label,date in [('steady_initial',str(origin)),('rising','2025-06-26T00:00Z'),('peak','2025-06-27T14:40Z'),('falling','2025-06-29T00:00Z')]:
        idx=int(np.argmin(abs(times-(pd.Timestamp(date)-origin).total_seconds())))
        r=ratio[:,idx]
        snapshots[label]=dict(time_utc=str(origin+pd.to_timedelta(times[idx],unit='s')),
            length_fraction_water_slope_below_90pct_bed=float(m.dx[r<.9].sum()/m.dx.sum()),
            minimum_water_to_bed_slope_ratio=float(r.min()),maximum_ratio=float(r.max()))
        f=m.faces.copy();f['bed_slope']=bed_slope;f['water_to_bed_slope_ratio']=r
        f['upstream_depth_m']=h[m.a,idx];f['downstream_depth_m']=h[m.b,idx]
        f.to_csv(dest/(label+'_profile.csv'),index=False)
    node,ti=np.unravel_index(np.argmax(h),h.shape)
    old=json.loads((OUT/'A_normal/runs/base_n0.024854130_dx200_ext0/result.json').read_text())
    assert abs(float(h.max())-old['maximum_depth_m'])<1e-6
    result=dict(snapshots=snapshots,maximum_depth_m=float(h.max()),maximum_depth_node=str(m.node_ids[node]),
                maximum_depth_utc=str(origin+pd.to_timedelta(times[ti],unit='s')),
                minimum_water_to_bed_slope_ratio=float(ratio.min()),
                reverse_gradient_count=int((ratio<0).sum()),spinup=spin)
    (dest/'backwater.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
