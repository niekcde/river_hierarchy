"""Isolated pulse and parent/child discretization controls; no published targets changed."""
import numpy as np
import pandas as pd
from geometry_audit import OUT,pilot
from rapid_tools.hydrograph_response import distribution
DEST=OUT/'segmented_diagnostics'

def main():
 plan=pilot.read(OUT/'routing_plan.json');case=next(c for c in plan['cases'] if c['example']=='22');jobs={j['state_id']:j for j in case['jobs']};receipt=pilot.read(OUT/'runs/22/completion.json');folder=receipt['final_directory'];wave=pilot.read(pilot.Path(folder)/'forcing.parquet');nb=int(wave.phase.eq('spinup').sum());times=wave.time_utc.astype('int64').to_numpy()/1e9
 metrics=pilot.read(OUT/'state_metrics.parquet');r=metrics[(metrics.example=='22')&metrics.window.eq('0.5%')&metrics.state.eq('S000_base')].iloc[0];mask=(times>=r.rise_seconds)&(times<=r.fall_seconds);curves=[];rows=[];pulses=[]
 for state in ['S000_base','all_closed_units_collapsed']:
  f=pilot.read(jobs[state]['parameters']);native=jobs[state]['native_dt_seconds']
  for variant,mult,dt in [('current',1,native),('refined',2,native/2)]:
   g=f.copy();g['count']*=mult;g['rapid_effective_length_m']=g.link_length_m/g['count'];g=pilot.cunge.parameters(g,float(g.n_manning.iloc[0]),beta=float(g.diffusion_multiplier.iloc[0]));pilot.equilibrium_check(g,dt)
   q,audit=pilot.common_grid.route_interval_means(g,wave.forcing_cms.to_numpy()[nb:],round(60/dt),dt,initial_flow=case['initial_cms']);q=np.r_[np.full(nb,case['initial_cms']),q]
   old=pilot.read(pilot.Path(folder)/state/'outlet.parquet').q_outlet_cms.to_numpy()
   if variant=='current':np.testing.assert_allclose(q,old,rtol=1e-12,atol=1e-8)
   curves.append(q);rows.append(dict(state=state,variant=variant,dt_seconds=dt,cells=int(g['count'].sum()),maximum_discharge_difference_cms=float(abs(q-old).max()),minimum_native_flow_cms=audit['minimum_native_flow_cms']))
   print(state,variant,'routed',flush=True)
  # Positive-baseline one-hour pulse isolates the transfer response from observed event shape/window.
  dt=native;length=1800*60;forcing=np.full(length,case['initial_cms']);forcing[60:120]+=100.
  q,audit=pilot.common_grid.route_interval_means(f,forcing,round(60/dt),dt,initial_flow=case['initial_cms']);perturbation=q-case['initial_cms'];t=(np.arange(length)+.5)/60
  mean=float(np.sum(t*perturbation)/np.sum(perturbation));expected=float((f.reference_flow_fraction*f['count']*f.rapid_k).sum()/3600)
  pulses.append(dict(state=state,expected_mean_route_hours=expected,simulated_pulse_lag_hours=mean-1.5,mass_ratio=float(perturbation.sum()/(forcing-case['initial_cms']).sum()),minimum_perturbation_cms=float(perturbation.min()),final_perturbation_cms=float(perturbation[-1])))
 baseline=min(float(r.baseline_cms),min(float(q[mask].min()) for q in curves))
 for row,q in zip(rows,curves):
  row.update(shared_diagnostic_baseline_cms=baseline,published_baseline_cms=float(r.baseline_cms),centroid_seconds=distribution(times[mask],q[mask]-baseline)['centroid_seconds'],peak_seconds=float(times[mask][np.argmax(q[mask])]))
 result=pd.DataFrame(rows)
 for variant in ['current','refined']:
  a=result[(result.variant==variant)&result.state.eq('S000_base')].iloc[0];sel=result.variant.eq(variant);result.loc[sel,'centroid_shift_hours']=(result.loc[sel,'centroid_seconds']-a.centroid_seconds)/3600;result.loc[sel,'peak_shift_hours']=(result.loc[sel,'peak_seconds']-a.peak_seconds)/3600
 result.to_csv(DEST/'refinement22.csv',index=False);pd.DataFrame(pulses).to_csv(DEST/'pulse22.csv',index=False);pilot.save(DEST/'routing_controls_sources.json',pilot.SOURCES)
 print(result.to_string(index=False));print(pd.DataFrame(pulses).to_string(index=False))
if __name__=='__main__':main()
