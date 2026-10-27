"""Explain modest segmented impact before considering forest error."""
import numpy as np
import pandas as pd
from geometry_audit import OUT,pilot
from rapid_tools.hydrograph_response import distribution
DEST=OUT/'impact22'
def main():
 DEST.mkdir(exist_ok=True);case=next(c for c in pilot.read(OUT/'routing_plan.json')['cases'] if c['example']=='22');jobs={j['state_id']:j for j in case['jobs']};parent=pilot.read(jobs['S000_base']['parameters']);seg=pilot.read(jobs['all_closed_units_collapsed']['parameters']);uniform=pilot.read(jobs['single_centerline']['parameters']);blocks=pilot.read(pilot.OUT/'eligible_blocks.json')['22']
 line=pilot.read(OUT/'variants/22/all_closed_units_collapsed/lineage.csv');u=uniform.iloc[0];parts=[]
 for role,g in line.groupby('role'):
  f=seg[seg.id_link.isin(g.child_link_id)];L=(f.reference_flow_fraction*f.link_length_m).sum();T=(f.reference_flow_fraction*f.link_length_m/f.rapid_celerity_mps).sum()/3600
  parts.append(dict(region=role,length_km=L/1000,segmented_time_hours=T,time_at_uniform_speed_hours=L/u.rapid_celerity_mps/3600,extra_time_hours=T-L/u.rapid_celerity_mps/3600))
 pd.DataFrame(parts).to_csv(DEST/'uniform_to_segmented.csv',index=False)
 folder=pilot.Path(pilot.read(OUT/'runs/22/completion.json')['final_directory']);wave=pilot.read(folder/'forcing.parquet');nb=int(wave.phase.eq('spinup').sum());t=wave.time_utc.astype('int64').to_numpy()/1e9;metrics=pilot.read(OUT/'state_metrics.parquet');ref=metrics[(metrics.example=='22')&metrics.window.eq('0.5%')&metrics.state.eq('S000_base')].iloc[0];mask=(t>=ref.rise_seconds)&(t<=ref.fall_seconds)
 curves={s:pilot.read(folder/s/'outlet.parquet').q_outlet_cms.to_numpy() for s in ['S000_base','all_closed_units_collapsed']};audit=[]
 for b in blocks:
  state=b['state'];source_case=next(c for c in pilot.read(pilot.PLAN)['cases'] if c['example']=='22');source_job=next(j for j in source_case['jobs'] if j['state_id']==state);f=pilot.read(source_job['parameters']);section=f[f.id_link.eq(b['child_id'])];new,_=pilot.compose(parent,[b],{state:section});dt=pilot.shared_clock([new],[30,60],jobs['S000_base']['native_dt_seconds']);pilot.equilibrium_check(new,dt)
  q,qa=pilot.common_grid.route_interval_means(new,wave.forcing_cms.to_numpy()[nb:],round(60/dt),dt,initial_flow=case['initial_cms']);q=np.r_[np.full(nb,case['initial_cms']),q];assert qa['minimum_native_flow_cms']>=0;curves[state]=q
  audit.append(dict(state=state,dt_seconds=dt,minimum_native_flow_cms=qa['minimum_native_flow_cms'],outside_parameters_preserved=True));print(state,'controlled collapse routed',flush=True)
 baseline=min(float(ref.baseline_cms),min(q[mask].min() for q in curves.values()));rows=[];pc=distribution(t[mask],curves['S000_base'][mask]-baseline)['centroid_seconds'];pp=float(t[mask][np.argmax(curves['S000_base'][mask])])
 for state,q in curves.items():
  m=distribution(t[mask],q[mask]-baseline);rows.append(dict(state=state,centroid_shift_hours=(m['centroid_seconds']-pc)/3600,peak_shift_hours=(float(t[mask][np.argmax(q[mask])])-pp)/3600,baseline_cms=baseline,published_baseline_cms=float(ref.baseline_cms)))
 pd.DataFrame(rows).to_csv(DEST/'controlled_block_responses.csv',index=False);pd.DataFrame(audit).to_csv(DEST/'control_audit.csv',index=False);pd.DataFrame(dict(time_utc=wave.time_utc,**curves)).to_parquet(DEST/'controlled_hydrographs.parquet',index=False)
 data=pilot.read(OUT/'segmented_diagnostics/edited_domain_summary.csv');data['length_reduction_pct']=100*(1-data.child_length_km/data.parent_length_km);data['speed_increase_pct']=100*(data.child_effective_speed/data.parent_effective_speed-1);data['edited_time_reduction_pct']=100*(1-data.child_time_hours/data.parent_time_hours);data['centroid_to_route_reduction']=data.centroid_shift_hours/data.delta_hours;data.to_csv(DEST/'cross_network_changes.csv',index=False)
 pilot.save(DEST/'manifest.json',dict(sources=pilot.SOURCES,code_sha256=pilot.sha(__file__),purpose='Only compare parent, uniform and segmented collapse mechanisms; no new predictor or model fitting',controls='Each selected closed block collapsed alone, all outside parameters exactly preserved; same forcing and shared published window; common minimum baseline across controls',baseline_changed=bool(baseline!=ref.baseline_cms)))
 print(pd.DataFrame(rows).to_string(index=False));print(pd.DataFrame(parts).to_string(index=False))
if __name__=='__main__':main()
