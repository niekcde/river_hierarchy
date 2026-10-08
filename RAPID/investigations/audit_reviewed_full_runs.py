"""Read-only first audit of completed reviewed Cunge hierarchy runs."""
from pathlib import Path
import sys,json,hashlib,time
import numpy as np,pandas as pd
from scipy.io import netcdf_file
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT),str(ROOT/'RAPID/src'),str(Path(__file__).parent)]
from cunge_full_hierarchy_reviewed import settling
OUT=ROOT/'RAPID/outputs/cunge_full_run_audit_20261007';OUT.mkdir(exist_ok=True)
CAL=ROOT/'RAPID/outputs/cunge_calibration_reviewed_20261006/comparison.csv'
cal=pd.read_csv(CAL,dtype={'example':str});cal=cal[cal.method=='cunge'].set_index('example');chash=hashlib.sha256(CAL.read_bytes()).hexdigest()
examples=['02','03','05','13','15','21','22','37','41','45','46','48','54','55','56'];rows=[];states=[];pairs_out=[];failures=[];samples=[]
def require(condition,msg):
 if not condition:raise AssertionError(msg)
for ex in examples:
 try:
  root=ROOT/'RAPID/outputs/cunge_full_hierarchy_pilot'/ex if ex=='02' else Path('/Volumes/PhD/river_hierarchy/cunge_full_hierarchy_runs')/ex
  m=json.loads((root/'experiment.json').read_text());done=json.loads((root/'completion.json').read_text());u=pd.read_csv(root/'unique_run_registry.csv');reg=pd.read_csv(root/'state_run_registry.csv');pairs=pd.read_csv(root/'parent_child_comparison.csv');inv=pd.read_csv(m['source_inventory']);f=pd.read_parquet(root/'common_forcing.parquet');flow=f.q_inflow_cms.to_numpy();times=pd.to_datetime(f.time_utc,utc=True)
  require(done['all_routed'],'completion flag');require(round(m['n_manning'],12)==round(cal.loc[ex,'n_manning'],12),'calibrated n mismatch')
  if ex!='02':require(m['calibration_table_sha256']==chash,'calibration table hash mismatch');require(m['source_calibration']==cal.loc[ex,'output_dir'],'wrong calibration source')
  ready=inv[inv.status=='ready'];require(set(u.fingerprint)==set(ready.fingerprint),'missing/unexpected distinct states');require(not u.fingerprint.duplicated().any(),'duplicate state fingerprints');require(len(reg)==len(ready)==done['state_entries'],'alias registry count');require(len(u)==done['unique_states']==m['n_unique_states'],'unique count')
  require(set(zip(reg.family,reg.state,reg.fingerprint))==set(zip(ready.family,ready.state,ready.fingerprint)),'alias identities differ')
  expected_pairs=ready[ready.parent_state.notna()&ready.parent_state.ne(ready.state)]
  require(set(zip(pairs.family,pairs.parent_state,pairs.child_state))==set(zip(expected_pairs.family,expected_pairs.parent_state,expected_pairs.state)),'parent-child membership')
  require(len(pairs)==done['parent_child_pairs'],'pair count');require(np.isfinite(flow).all() and (flow>=0).all(),'invalid forcing')
  require(len(f)==m['report_steps'],'report length');np.testing.assert_allclose(np.diff(f.time_seconds),m['forcing_step_s'],rtol=0,atol=0)
  for path,h in m['source_frame_sha256'].items():require(hashlib.sha256(Path(path).read_bytes()).hexdigest()==h,'changed input frame '+path)
  exterior=pd.read_csv(root/'exterior_invariance.csv')
  require(exterior['accepted' if 'accepted' in exterior else 'unchanged_parameters'].all(),'unaccepted exterior change')
  threshold=1e-8*max(1,float(np.ptp(flow)));lookup=u.set_index('fingerprint');wave_cache={};max_difference=0
  # Inspect every state outlet and every full-output header/native audit.
  for index,r in enumerate(u.itertuples()):
   p=Path(r.output_dir);require(p.resolve()==(root/'states'/r.fingerprint).resolve(),'stale staging/output path')
   result=json.loads((p/'result.json').read_text());audit=json.loads((p/'run/Qout.audit.json').read_text());prep=json.loads((p/'prep/rapid_prep_manifest.json').read_text())
   require(r.status=='routed' and r.tail_settled and result['tail_settled'],'state status/tail flag')
   require(round(r.n_manning,12)==round(m['n_manning'],12),'child-specific n')
   require(np.isfinite(audit['minimum_substep_flow_cms']) and audit['minimum_substep_flow_cms']>=-1e-9,'negative native flow')
   require(audit['negative_reported_flow_count']==0 and not audit['flow_clipped'],'reported negatives/clipping')
   require(audit['boundary_mode']=='upstream_two_time' and audit['initial_flow_mode']=='steady_first_inflow','boundary/initial condition')
   require(audit['routing_dt_seconds']==m['routing_dt_s'],'timestep mismatch')
   require(np.isfinite(r.engine_cascade_max_error_cms) and r.engine_cascade_max_error_cms<=threshold,'engine mismatch')
   max_difference=max(max_difference,r.engine_cascade_max_error_cms)
   qpath=p/'run/Qout.nc';require(qpath.stat().st_size==r.qout_bytes,'Qout size mismatch')
   with netcdf_file(qpath,'r',mmap=True) as nc:
    require(nc.variables['Qout'].shape==(len(f),r.n_sections),'Qout dimensions');require(nc.variables['Qout'].typecode()=='d','Qout precision')
    require(nc.variables['time'].data[0]==0 and nc.variables['time'].data[-1]==f.time_seconds.iloc[-1],'Qout time endpoints')
   with netcdf_file(p/'prep/inflow.nc','r',mmap=True) as nc:require(nc.variables['m3_riv'].shape==(len(f),r.n_sections),'prep dimensions')
   w=pd.read_parquet(p/'outlet_hydrograph.parquet');q=w.q_outlet_cms.to_numpy();require(len(w)==len(f) and np.isfinite(q).all() and (q>=-1e-9).all(),'outlet invalid')
   np.testing.assert_array_equal(w.q_inflow_cms,flow);np.testing.assert_array_equal(pd.to_datetime(w.time_utc,utc=True),times)
   require(settling(flow,q,m['event_last_seconds'],m['forcing_step_s']),'outlet tail not settled')
   peak=int(q.argmax());require(0<peak<len(q)-1,'boundary peak');require(pd.Timestamp(r.peak_utc)==times.iloc[peak],'peak timestamp mismatch');require(np.isclose(r.peak_cms,q[peak],rtol=1e-12),'peak magnitude mismatch')
   lag=(peak-int(flow.argmax()))*m['forcing_step_s']/3600;require(abs(lag-r.lag_h)<1e-9,'lag mismatch')
   states.append(dict(example=ex,fingerprint=r.fingerprint,lag_h=lag,minimum_native_flow_cms=audit['minimum_substep_flow_cms'],engine_error_cms=r.engine_cascade_max_error_cms,tail_settled=True))
   if r.fingerprint==m['base_fingerprint']:wave_cache['base']=q
  for r in pairs.itertuples():
   require(r.parent_fingerprint in lookup.index and r.child_fingerprint in lookup.index,'unknown pair state')
   a=lookup.loc[r.parent_fingerprint];b=lookup.loc[r.child_fingerprint]
   require(abs(r.delta_lag_h-(b.lag_h-a.lag_h))<1e-9,'pair delta mismatch')
   require(np.isclose(r.delta_peak_pct,100*(b.peak_cms/a.peak_cms-1),rtol=1e-8,atol=1e-10),'pair peak delta mismatch')
  unique_pairs=pairs.drop_duplicates(['parent_fingerprint','child_fingerprint'])
  for r in unique_pairs.itertuples():pairs_out.append(dict(example=ex,parent_fingerprint=r.parent_fingerprint,child_fingerprint=r.child_fingerprint,child_state=r.child_state,delta_lag_h=r.delta_lag_h,delta_peak_pct=r.delta_peak_pct))
  # Independently read full-output outlet columns for the base and largest-shift child.
  selected={m['base_fingerprint'],unique_pairs.loc[unique_pairs.delta_lag_h.abs().idxmax(),'child_fingerprint']}
  for fp in selected:
   p=root/'states'/fp;prep=json.loads((p/'prep/rapid_prep_manifest.json').read_text());w=pd.read_parquet(p/'outlet_hydrograph.parquet')
   with netcdf_file(p/'run/Qout.nc','r',mmap=True) as nc:
    ids=np.array(nc.variables['rivid'].data,copy=True);indices=[int(np.flatnonzero(ids==rid)[0]) for rid in prep['outlet_reach_ids']]
    q=np.zeros(len(f))
    for ix in indices:q+=np.array(nc.variables['Qout'].data[:,ix],copy=True)
   diff=float(abs(q-w.q_outlet_cms.to_numpy()).max());require(diff<=1e-10*max(1,float(abs(q).max())),'Qout/outlet extraction mismatch');samples.append(dict(example=ex,fingerprint=fp,max_saved_outlet_difference_cms=diff))
  base=lookup.loc[m['base_fingerprint']];target=pd.Timestamp(cal.loc[ex,'target_utc']);base_error=(pd.Timestamp(base.peak_utc)-target).total_seconds()/60
  require(abs(base.lag_h-done['base_lag_h'])<1e-9,'base completion mismatch');require(abs(base.lag_h-cal.loc[ex,'lag_h'])<1e-9,'full-clock base peak changed from calibration')
  rows.append(dict(example=ex,status='pass',reused=ex=='02',unique_states=len(u),unique_pairs=len(unique_pairs),alias_pairs=len(pairs),excluded_states=int((inv.status=='excluded_topology_review').sum()),minimum_native_flow_cms=float(u.minimum_substep_flow_cms.min()),maximum_engine_difference_cms=max_difference,engine_threshold_cms=threshold,base_utc_error_minutes=base_error,max_advance_h=float(unique_pairs.delta_lag_h.min()),max_delay_h=float(unique_pairs.delta_lag_h.max()),n_large_shifts_over_5h=int((unique_pairs.delta_lag_h.abs()>5).sum()),actual_gib=done['actual_bytes']/1024**3,reviewed_redistribution_transitions=int(m.get('reviewed_flow_redistribution_transitions',0)),calibration_qualification=m.get('calibration_qualification','reused_same_parameters')))
  print('PASS',ex,len(u),'states',flush=True)
 except Exception as error:
  failures.append(dict(example=ex,error=repr(error)));print('FAIL',ex,repr(error),flush=True)
 pd.DataFrame(rows).to_csv(OUT/'network_audit.csv',index=False);(OUT/'failures.json').write_text(json.dumps(failures,indent=2))
pd.DataFrame(states).to_csv(OUT/'state_audit.csv',index=False);pd.DataFrame(pairs_out).to_csv(OUT/'unique_pair_summary.csv',index=False);pd.DataFrame(samples).to_csv(OUT/'sampled_Qout_outlet_checks.csv',index=False)
summary=dict(networks_checked=len(examples),networks_passed=len(rows),failures=failures,unique_states=sum(r['unique_states'] for r in rows),unique_parent_child_pairs=sum(r['unique_pairs'] for r in rows),exclusions=sum(r['excluded_states'] for r in rows),scope='All state registries, input frame hashes, output headers and file sizes, native audit records and complete saved outlet series checked. Full Qout outlet columns independently reread for base and largest-shift child per network; not a fresh full all-reach payload scan or reroute.',calibration_table_sha256=chash,script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
(OUT/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))
if failures:raise SystemExit(1)
