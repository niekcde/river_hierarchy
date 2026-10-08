"""Reviewed UTC calibration audit and hierarchy reuse plan; no hierarchy writes."""
from pathlib import Path
import sys,json,hashlib
import numpy as np,pandas as pd
import plotly.graph_objects as go
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT),str(ROOT/'RAPID/src'),str(Path(__file__).parent)]
import cunge_calibration_reviewed as cal
from cunge_hydrograph_figure import build_figure,save_figure
from rapid_tools.refreshed_analysis import interpolated_peak
OUT=cal.OUT

def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
 new=pd.read_csv(OUT/'comparison.csv',dtype={'example':str});old=pd.read_csv(cal.PREVIOUS/'comparison.csv',dtype={'example':str})
 assert len(new)==32 and not new.duplicated(['example','method']).any()
 targets={r['example']:r for r in json.loads((OUT/'targets_snapshot.json').read_text())['records']}
 rows=[];actions=[];sources={};seed_count=0
 for r in new.itertuples():
  ex=r.example;p=Path(r.output_dir);o=old[(old.example==ex)&(old.method==r.method)].iloc[0];record=targets[ex]
  manifest=json.loads((p/'case.json').read_text());wave=pd.read_parquet(p/f'{r.method}_best_wave.parquet');times=pd.to_datetime(wave.time_utc,utc=True)
  peak=times.iloc[wave.q_outlet_cms.to_numpy().argmax()];err=(peak-pd.Timestamp(record['target_utc'])).total_seconds()/3600
  assert abs(err-r.error_h)<1e-9 and r.valid and r.tail_settled
  oldwave=pd.read_parquet(Path(o.output_dir)/f'{r.method}_best_wave.parquet');ot=pd.to_datetime(oldwave.time_utc,utc=True);oldpeak=ot.iloc[oldwave.q_outlet_cms.to_numpy().argmax()]
  old_utc_error=(oldpeak-pd.Timestamp(record['target_utc'])).total_seconds()/3600
  assert r.abs_error_h<=abs(old_utc_error)+1e-8,('New fit worse than existing feasible solution',ex,r.method)
  interpolated_seconds,interpolation_status=interpolated_peak(wave.time_seconds,wave.q_outlet_cms)
  interpolated_time=times.iloc[0]+pd.to_timedelta(interpolated_seconds,unit='s')
  interpolation_error=(interpolated_time-pd.Timestamp(record['target_utc'])).total_seconds()/60
  target_seconds=(pd.Timestamp(record['target_utc'])-times.iloc[0]).total_seconds()
  reporting_floor=abs(target_seconds-round(target_seconds/r.forcing_step_s)*r.forcing_step_s)/60
  resolution_limited=reporting_floor>r.timing_resolution_h*60+1e-8 and abs(err)*60<=reporting_floor+1e-8
  changed=round(r.n_manning,12)!=round(float(o.n_manning),12)
  rows.append(dict(example=ex,method=r.method,old_n=float(o.n_manning),new_n=r.n_manning,n_changed=changed,n_change_pct=100*(r.n_manning/o.n_manning-1),target_utc=record['target_utc'],observed_target_shift_minutes=record['shift_from_old_target_minutes'],prepared_upstream_offset_minutes=record['prepared_minus_observed_upstream_peak_minutes'],old_solution_error_to_new_utc_minutes=old_utc_error*60,new_utc_error_minutes=err*60,within_sampling=r.within_sampling,reporting_step_minutes=r.forcing_step_s/60,minimum_possible_sampled_error_minutes=reporting_floor,reporting_resolution_limited=resolution_limited,interpolated_error_minutes=interpolation_error,interpolation_status=interpolation_status,within_crest_interval=r.within_crest_interval,crest_interval_error_minutes=r.crest_interval_error_h*60,old_shape_rmse=float(o.normalized_shape_rmse),new_shape_rmse=r.normalized_shape_rmse,half_clock_peak_change_seconds=r.half_clock_lag_change_s,fine_grid_peak_change_seconds=r.fine_grid_lag_change_s,old_simulation_days=float(o.simulation_days),new_simulation_days=r.simulation_days,output_dir=str(p)))
  for source,expected in manifest['source_hashes'].items():
   actual=digest(source)
   if actual!=expected and Path(source)==cal.TARGETS_PATH:
    snapshot=OUT/'before_case45_correction/targets_snapshot.json'
    assert digest(snapshot)==expected
    prior=next(v for v in json.loads(snapshot.read_text())['records'] if v['example']==ex)
    assert all(prior[k]==record[k] for k in ['target_utc','crest_start_utc','crest_end_utc'])
    sources[str(snapshot)]=expected
   else:
    assert actual==expected,source
    sources[source]=actual
  if r.method!='cunge':continue
  seed_count+=len(json.loads((p/'reused_trial_sources.json').read_text()))
  existing=next((path for path in [ROOT/'RAPID/outputs/cunge_full_hierarchy_pilot'/ex,Path('/Volumes/PhD/river_hierarchy/cunge_full_hierarchy_runs')/ex] if (path/'completion.json').exists()),None)
  reuse=False;reason='No completed full hierarchy exists.'
  if existing:
   m=json.loads((existing/'experiment.json').read_text());done=json.loads((existing/'completion.json').read_text());assert done['all_routed']
   if round(m['n_manning'],12)!=round(r.n_manning,12):reason='Selected Manning n changed; regenerate coefficients and reroute all unique states.'
   else:
    for source,expected in m['source_frame_sha256'].items():assert digest(source)==expected,source
    pd.testing.assert_frame_equal(pd.read_parquet(Path(m['source_calibration'])/'frozen_grid.parquet'),pd.read_parquet(p/'frozen_grid.parquet'))
    f=pd.read_parquet(existing/'common_forcing.parquet');length=min(len(f),len(wave));np.testing.assert_array_equal(f.q_inflow_cms.iloc[:length],wave.q_inflow_cms.iloc[:length]);np.testing.assert_array_equal(pd.to_datetime(f.time_utc.iloc[:length],utc=True),times.iloc[:length])
    assert m['forcing_step_s']==r.forcing_step_s and m['reference_discharge_cms']==manifest['reference_discharge_cms']
    reuse=True;reason='Same n, frozen base grid, reference discharge, event/report clock, and unchanged hashes for every hierarchy source frame; existing full hierarchy already passed settlement and engine checks.'
   base=json.loads((existing/'states'/m['base_fingerprint']/'result.json').read_text());existing_err=(pd.Timestamp(base['peak_utc'])-pd.Timestamp(record['target_utc'])).total_seconds()/3600
  else:existing_err=None
  action='reuse' if reuse else ('rerun' if existing else 'first_run')
  if ex=='11':action='diagnostic_only';reason='Cunge timing remains unresolved. Excluded from satisfactorily calibrated production batch.'
  qualification='nominal_target_within_sampling' if r.within_sampling else ('reporting_resolution_limited' if resolution_limited else ('crest_span_only_global_peak_switch' if ex=='54' else 'unresolved'))
  if ex=='54':action='first_run_qualified';reason='User requested all networks except 11 and reusable 02. Include 54 with documented caveat: observed crest span matched, nominal midpoint not matched; global maximum switches between crests.'
  actions.append(dict(example=ex,action=action,calibration_qualification=qualification,old_n=float(o.n_manning),new_n=r.n_manning,n_changed=changed,existing_completed=existing is not None,existing_directory=str(existing) if existing else None,existing_base_utc_error_minutes=existing_err*60 if existing_err is not None else None,new_calibration_directory=str(p),reason=reason))
 comparison=pd.DataFrame(rows);comparison.to_csv(OUT/'old_new_calibration.csv',index=False);plan=pd.DataFrame(actions);plan.to_csv(OUT/'hierarchy_rerun_plan.csv',index=False)
 verification=pd.read_csv(OUT/'prepared_verification.csv',dtype={'example':str});assert len(verification)==16 and verification.example.nunique()==16
 assert (verification.minimum_substep_flow_cms>=-1e-9).all()
 checks=dict(all_32_selected_routes_numerically_valid=True,nominal_target_within_sampling_by_method={str(k):int(v) for k,v in new.groupby('method').within_sampling.sum().items()},all_32_utc_errors_recomputed=True,all_selected_solutions_no_worse_than_previous_under_new_objective=True,source_hashes_verified=len(sources),old_trial_records_available_for_rescoring=seed_count,full_cunge_engine_runs=16,full_cunge_max_engine_difference_cms=float(verification.max_engine_cascade_error_cms.max()),full_cunge_minimum_native_flow_cms=float(verification.minimum_substep_flow_cms.min()),fixed_and_cunge_max_probe_difference_cms=float(new.engine_probe_max_error_cms.max()),reused_hierarchies=plan[plan.action=='reuse'].example.tolist(),source_target_sha256=digest(cal.TARGETS_PATH),snapshot_target_sha256=digest(OUT/'targets_snapshot.json'))
 assert checks['source_target_sha256']==checks['snapshot_target_sha256'];(OUT/'completion_checks.json').write_text(json.dumps(checks,indent=2))
 fig=build_figure(new,{r.example:Path(r.output_dir) for r in new.itertuples()})
 for j,ex in enumerate(sorted(targets)):
  record=targets[ex];origin=pd.Timestamp(record['plot_origin_utc']);row,col=divmod(j,4);x=(pd.Timestamp(record['target_utc'])-origin).total_seconds()/86400
  a=(pd.Timestamp(record['crest_start_utc'])-origin).total_seconds()/86400;b=(pd.Timestamp(record['crest_end_utc'])-origin).total_seconds()/86400
  fig.add_vrect(x0=a,x1=b,fillcolor='orange',opacity=.12,line_width=0,row=row+1,col=col+1)
  fig.add_trace(go.Scatter(x=[x],y=[1.035],mode='markers',name='Reviewed target',legendgroup='target',showlegend=j==0,marker=dict(color='darkorange',symbol='diamond',size=8),hovertemplate=record['target_utc']+'<extra>Reviewed target; marker height schematic</extra>'),row=row+1,col=col+1)
 fig.update_layout(title_text='<b>Recalibrated hydrographs · reviewed UTC targets</b><br><sup>Orange diamonds: nominal targets; shading: crest intervals. Case 11 remains unresolved.</sup>')
 save_figure(fig,OUT)
 table=comparison[comparison.method=='cunge'][['example','old_n','new_n','new_utc_error_minutes','within_sampling','half_clock_peak_change_seconds','fine_grid_peak_change_seconds','reporting_resolution_limited','interpolated_error_minutes']]
 report='''REVIEWED UTC BASE CALIBRATION — 6 October 2026

Completed both fixed-X and Cunge fits for all 16 observation events. No child states calibrated or hierarchy jobs started; previous results retained. New target snapshot is archived here. Nominal targets follow the reviewed table, including June 2 23:45 UTC for 48 and the proposed interpretations for 37/45 authorized by the request to proceed.

Objective: model downstream first sampled maximum UTC minus reviewed downstream target UTC. Gauge-to-gauge lag remains diagnostic and is not subtracted from a prepared-inlet-relative model lag. Targets, forcing, reference Q, geometry, bounds (.01,.1), alpha=2 and celerity caps were not adjusted to improve predictive correlations. Same target used for both routing methods. Old unchanged hydraulic trials were verified and rescored; extra n values were evaluated where needed. Selected solutions were rerouted, checked for settlement/positivity, probed against the independent sparse engine, and assessed with half timestep and Cunge grid refinement. All 16 selected Cunge base prep bundles also completed full sparse-engine runs.

Search limitations: retained original finite coarse/adaptive search and deterministic tie-break toward n=.035; this is not a proof of a global optimum or a unique Manning n. Multiple n values can share one sampled peak time. The reported within_sampling tolerance is the existing sum of upstream/downstream median sampling intervals; it is a diagnostic, not a confidence interval. Crest-span acceptance is reported separately. Case 11 is still unresolved; do not present it as successfully calibrated. Case 54 is also outside midpoint sampling tolerance: its global maximum switches between crests; the selected peak is within the crest span but 250.5 minutes before the midpoint. The user requested it in the next full batch; it remains explicitly qualified as a crest-span fit, not a nominal-midpoint fit. Case 55 has a 25.667-minute reporting interval: the closest possible output timestamp is 11.667 minutes from the target, outside the 10-minute observation-based tolerance. Both methods attain that grid minimum; interpolation gives errors of about 6.39 minutes (fixed-X) and 9.02 minutes (Cunge). The primary sampled-peak criterion is retained; interpolation is a diagnostic, not a silent objective change. No independent-event validation was rerun in this target-only calibration.

Cunge results\n'''+table.to_string(index=False)+'\n\nHierarchy action plan\n'+plan[['example','action','old_n','new_n']].to_string(index=False)+'\n\nChecks\n'+json.dumps(checks,indent=2)+'''

Next step: wire this calibration table into the hierarchy runner and apply the previously investigated reference-flow acceptance policy before launching first/rerun jobs. Reuse means keep the existing verified hydraulic results while linking them to the revised calibration assessment; do not pretend they were generated with the new target metadata. Old bulky arrays have not been removed. Use cunge_full_hierarchy_reviewed.py for the corrected calibration and hash-matched reviewed Q/X redistribution; the historical pilot runner retains its historical default.

Artifacts: old_new_calibration.csv, hierarchy_rerun_plan.csv, prepared_verification.csv, completion_checks.json, comparison.csv, calibration_hydrographs.html, targets_snapshot.json and case-level trials/prep/waves.
'''
 switch=json.loads((OUT/'case54_peak_switch.json').read_text())
 report+='\nCase 54 targeted diagnostic: changing n by '+format(switch['n_difference'],'.3g')+' changes the global peak time by '+str(switch['global_peak_jump_h'])+' h, while maximum hydrograph difference is only '+format(switch['wave_difference_pct_inlet_amplitude'],'.6g')+'% of inlet amplitude. This is a competing-crest argmax discontinuity, not evidence of routing instability. See case54_peak_switch.html and case54_peak_switch.json.\n'
 (OUT/'REPORT.txt').write_text(report);print(table.to_string(index=False));print(plan[['example','action']].to_string(index=False));print(json.dumps(checks,indent=2))
if __name__=='__main__':main()
