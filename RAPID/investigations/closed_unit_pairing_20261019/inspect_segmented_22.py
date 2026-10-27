"""Explain the 22 uniform-versus-segmented difference; isolated routing refinement."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from pilot import OUT,ROOT,sha,save,cunge,common_grid,shared_clock
from rapid_tools.hydrograph_response import distribution
from rapid_tools.event_window_batch import settling_hours,equilibrium_check

def main():
    dest=OUT/'segmented_22_audit';dest.mkdir(exist_ok=True)
    plan=json.loads((OUT/'routing_plan.json').read_text());case=next(c for c in plan['cases'] if c['example']=='22');jobs={j['state_id']:j for j in case['jobs']}
    names=['single_centerline','all_closed_units_collapsed'];frames={n:pd.read_parquet(jobs[n]['parameters']) for n in names};u=frames[names[0]].iloc[0];s=frames[names[1]].copy()
    for f in frames.values():
        assert f.n_manning.eq(u.n_manning).all() and f.diffusion_multiplier.eq(u.diffusion_multiplier).all()
        np.testing.assert_allclose(f.slope,(f.head_us-f.head_ds)/f.link_length_m,atol=1e-12)
    s['travel_hours']=s.link_length_m/s.rapid_celerity_mps/3600
    s['extra_hours_vs_uniform_speed']=s.link_length_m*(1/s.rapid_celerity_mps-1/u.rapid_celerity_mps)/3600
    s.to_csv(dest/'section_attribution.csv',index=False)
    controls=[]
    for depth_name,depth in [('uniform',u.depth_m),('local',s.depth_m)]:
        for slope_name,slope in [('uniform',u.slope),('local',s.slope)]:
            speed=np.clip((5/3)*depth**(2/3)*np.sqrt(slope)/u.n_manning,u.celerity_min_mps,u.celerity_max_mps)
            controls.append(dict(depth=depth_name,slope=slope_name,travel_hours=float((s.link_length_m/speed).sum()/3600)))
    pd.DataFrame(controls).to_csv(dest/'kinematic_controls.csv',index=False)
    receipt=json.loads((OUT/'runs/22/completion.json').read_text());folder=Path(receipt['final_directory']);wave=pd.read_parquet(folder/'forcing.parquet');nb=int(wave.phase.eq('spinup').sum());t=wave.time_utc.astype('int64').to_numpy()/1e9
    metrics=pd.read_parquet(OUT/'state_metrics.parquet');metrics=metrics[(metrics.example=='22')&(metrics.window=='0.5%')].set_index('state');ref=metrics.loc['S000_base'];a=int(np.searchsorted(t,ref.rise_seconds));b=int(np.searchsorted(t,ref.fall_seconds))
    results=[];curves=[]
    for name in names:
        f=frames[name]
        for variant,mult,dtmult in [('reproduce',1,1),('half_timestep',1,.5),('double_cells',2,1),('quadruple_cells',4,.5)]:
            g=f.copy();g['count']=g['count']*mult;g['rapid_effective_length_m']=g.link_length_m/g['count'];g=cunge.parameters(g,u.n_manning,beta=u.diffusion_multiplier)
            dt=shared_clock([g],[30,60],jobs[name]['native_dt_seconds']*dtmult);equilibrium_check(g,dt)
            q,audit=common_grid.route_interval_means(g,wave.forcing_cms.to_numpy()[nb:],round(60/dt),dt,initial_flow=case['initial_cms']);assert audit['minimum_native_flow_cms']>=0
            required,settled=settling_hours(wave.time_utc.iloc[nb:],q,pd.Timestamp(case['window']['end_utc'])+pd.Timedelta(seconds=30),case['final_cms'],case['tolerance_cms']);assert settled
            q=np.r_[np.full(nb,case['initial_cms']),q];old=pd.read_parquet(folder/name/'outlet.parquet').q_outlet_cms.to_numpy()
            if variant=='reproduce':np.testing.assert_allclose(q,old,rtol=1e-12,atol=1e-8)
            curves.append(q);print(name,variant,'minimum relative to original baseline',float(q[a:b+1].min()-ref.baseline_cms),flush=True)
            results.append(dict(state=name,variant=variant,cells=int(g['count'].sum()),dt=dt,max_discharge_difference_cms=float(abs(q-old).max()),minimum_native_flow_cms=audit['minimum_native_flow_cms'],settled=settled))
    baseline=min([float(ref.baseline_cms)]+[float(q[nb:].min()) for q in curves])
    parent=pd.read_parquet(folder/'S000_base/outlet.parquet').q_outlet_cms.to_numpy()
    parent_centroid=distribution(t[a:b+1],parent[a:b+1]-baseline)['centroid_seconds']
    for r,q in zip(results,curves):
        excess=q[a:b+1]-baseline;m=distribution(t[a:b+1],excess);peak=a+int(q[a:b+1].argmax())
        r.update(centroid_shift_hours=(m['centroid_seconds']-parent_centroid)/3600,peak_shift_hours=(t[peak]-ref.peak_seconds)/3600,shared_baseline_cms=baseline,original_baseline_cms=float(ref.baseline_cms))
    pd.DataFrame(results).to_csv(dest/'refinement.csv',index=False)
    files=[Path(__file__),Path(cunge.__file__),Path(common_grid.__file__),OUT/'routing_plan.json',OUT/'state_metrics.parquet',folder/'forcing.parquet',*[Path(jobs[n]['parameters']) for n in names]]
    save(dest/'manifest.json',dict(example='22',sources={str(p):sha(p) for p in files},interpretation='sum L/c diagnoses travel time, not the finite-window excess centroid; slope/depth controls change the kinematic formula only and are not physical reruns. Refinement reruns preserve physical geometry, celerity and diffusivity. Diagnostic metrics use the fixed pilot window with a shared minimum baseline across all refinements; parent centroid recomputed consistently. Historical pilot responses remain unchanged.',uniform_raw_celerity=float(u.rapid_celerity_mps_raw),uniform_capped_celerity=float(u.rapid_celerity_mps),uniform_travel_hours=float(u.link_length_m/u.rapid_celerity_mps/3600),segmented_travel_hours=float(s.travel_hours.sum())))
    print(pd.DataFrame(results).to_string(index=False))
if __name__=='__main__':main()
