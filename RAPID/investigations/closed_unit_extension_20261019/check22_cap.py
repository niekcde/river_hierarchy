"""Diagnostic cap sensitivity at fixed calibration; not replacement calibrated runs."""
import numpy as np
import pandas as pd
from geometry_audit import OUT,pilot
from rapid_tools.hydrograph_response import distribution
DEST=OUT/'segmented_diagnostics'
def main():
 case=next(c for c in pilot.read(OUT/'routing_plan.json')['cases'] if c['example']=='22');jobs={j['state_id']:j for j in case['jobs']};folder=pilot.Path(pilot.read(OUT/'runs/22/completion.json')['final_directory']);wave=pilot.read(folder/'forcing.parquet');nb=int(wave.phase.eq('spinup').sum());t=wave.time_utc.astype('int64').to_numpy()/1e9
 m=pilot.read(OUT/'state_metrics.parquet');r=m[(m.example=='22')&m.window.eq('0.5%')&m.state.eq('S000_base')].iloc[0];mask=(t>=r.rise_seconds)&(t<=r.fall_seconds)
 results=[];curves=[]
 for cap in [5.,10.,50.]:
  for state in ['S000_base','all_closed_units_collapsed']:
   f=pilot.read(jobs[state]['parameters']);g=f.copy();g['celerity_max_mps']=cap
   speed=np.clip(g.celerity_times_n/g.n_manning,g.celerity_min_mps,cap);ld=g.reference_discharge_cms/(g.width*g.slope*speed);g['count']=np.maximum(g['count'],np.ceil(g.link_length_m/(g.cunge_alpha*ld))).astype(int);g['rapid_effective_length_m']=g.link_length_m/g['count'];g=pilot.cunge.parameters(g,float(g.n_manning.iloc[0]),beta=float(g.diffusion_multiplier.iloc[0]))
   dt=pilot.shared_clock([g],[30,60],jobs[state]['native_dt_seconds']);pilot.equilibrium_check(g,dt)
   q,audit=pilot.common_grid.route_interval_means(g,wave.forcing_cms.to_numpy()[nb:],round(60/dt),dt,initial_flow=case['initial_cms']);q=np.r_[np.full(nb,case['initial_cms']),q];assert audit['minimum_native_flow_cms']>=0
   if cap==5:np.testing.assert_allclose(q,pilot.read(folder/state/'outlet.parquet').q_outlet_cms,rtol=1e-12,atol=1e-8)
   results.append(dict(cap_mps=cap,state=state,route_time_hours=float((g.reference_flow_fraction*g['count']*g.rapid_k).sum()/3600),peak_seconds=float(t[mask][np.argmax(q[mask])]),remaining_capped_links=int(g.rapid_celerity_capped.sum()),cells=int(g['count'].sum()),dt_seconds=dt,minimum_native_flow_cms=audit['minimum_native_flow_cms']));curves.append(q);print(cap,state,'done',flush=True)
 baseline=min(float(r.baseline_cms),min(q[mask].min() for q in curves))
 for row,q in zip(results,curves):row.update(centroid_seconds=distribution(t[mask],q[mask]-baseline)['centroid_seconds'],diagnostic_baseline_cms=baseline,published_baseline_cms=float(r.baseline_cms))
 d=pd.DataFrame(results)
 for cap in [5.,10.,50.]:
  a=d[(d.cap_mps==cap)&d.state.eq('S000_base')].iloc[0];sel=d.cap_mps.eq(cap);d.loc[sel,'centroid_shift_hours']=(d.loc[sel,'centroid_seconds']-a.centroid_seconds)/3600;d.loc[sel,'peak_shift_hours']=(d.loc[sel,'peak_seconds']-a.peak_seconds)/3600
 d.to_csv(DEST/'cap_sensitivity22.csv',index=False);pilot.save(DEST/'cap_control_sources.json',pilot.SOURCES);print(d.to_string(index=False))
if __name__=='__main__':main()
