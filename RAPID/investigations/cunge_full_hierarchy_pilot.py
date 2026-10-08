"""Full Cunge hierarchy pilot with frozen parent calibration and shared clocks.

All source hierarchies remain read-only. Identical hydraulic states are routed
once and exposed through an alias registry. Full all-reach Qout is retained.
"""
from pathlib import Path
import sys,json,hashlib,time,shutil,argparse,subprocess
from concurrent.futures import ProcessPoolExecutor
import numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'RAPID/src')]
from rapid_tools import cunge
from rapid_tools.engine import run_prepared_state
OUT=ROOT/'RAPID/outputs/cunge_full_hierarchy_pilot/03'
ASSETS=ROOT/'RAPID/assets/bounded_robust_slopes_20261002/child_head_reconstruction_20261003/03'

def configure(example,output_root):
 global OUT,ASSETS
 OUT=Path(output_root)/example
 ASSETS=ROOT/'RAPID/assets/bounded_robust_slopes_20261002/child_head_reconstruction_20261003'/example


def settling(flow,q,event_last,step):
 t=np.arange(len(q))*step;amp=max(float(q.max()-max(flow[0],flow[-1])),1e-12)
 outside=np.flatnonzero((t>=event_last)&(np.abs(q-flow[-1])>.001*amp))
 j=int(outside[-1]+1) if len(outside) else int(round(event_last/step))
 return bool(j<len(t) and t[j]+2*86400<=t[-1])

def run_state(fp):
 m=json.loads((OUT/'experiment.json').read_text());dest=OUT/'states'/fp;frame=pd.read_parquet(dest/'parameters.parquet');wave=pd.read_parquet(OUT/'common_forcing.parquet');flow=wave.q_inflow_cms.to_numpy();step=m['forcing_step_s'];dt=m['routing_dt_s'];nsub=m['nsub']
 predicted,audit=cunge.route(frame,flow,nsub,dt)
 if not settling(flow,predicted,m['event_last_seconds'],step):raise ValueError('Unsettled state '+fp)
 if audit['minimum_native_flow_cms']< -1e-9:raise ValueError('Negative native flow '+fp)
 begin=time.monotonic();metadata=dict(m,state_fingerprint=fp)
 prep=dest/'prep';pm=cunge.write_prepared_state(frame,flow,step,dt,prep,metadata=metadata)
 pm['state_role']='base' if fp==m['base_fingerprint'] else 'child';(prep/'rapid_prep_manifest.json').write_text(json.dumps(pm,indent=2))
 (dest/'run').mkdir(exist_ok=True);target=dest/'run/Qout.nc'
 _,q=run_prepared_state(prep,output_path=target,return_qout=True)
 outlet=q[:,np.array(pm['outlet_reach_ids'])-1].sum(axis=1);difference=float(np.max(abs(outlet-predicted)))
 if difference>1e-8*max(1,np.ptp(flow)):raise ValueError('Sparse engine disagreement '+fp)
 native=json.loads(target.with_suffix('.audit.json').read_text())
 if native['minimum_substep_flow_cms']< -1e-9:raise ValueError('Sparse engine negative native flow '+fp)
 times=pd.to_datetime(wave.time_utc,utc=True);j=int(outlet.argmax());lag=(j-int(flow.argmax()))*step/3600
 result=dict(fingerprint=fp,status='routed',n_manning=m['n_manning'],n_links=len(frame),n_sections=int(frame['count'].sum()),
  lag_h=float(lag),peak_cms=float(outlet[j]),peak_utc=times.iloc[j].isoformat(),engine_cascade_max_error_cms=difference,
  minimum_substep_flow_cms=native['minimum_substep_flow_cms'],tail_settled=True,
  final_tail_fraction=float(abs(outlet[-1]-flow[-1])/max(np.ptp(flow),1e-12)),runtime_seconds=time.monotonic()-begin,
  qout_bytes=target.stat().st_size,output_dir=str(dest))
 output=wave.copy();output['q_outlet_cms']=outlet;output.to_parquet(dest/'outlet_hydrograph.parquet',index=False)
 output.to_csv(dest/'outlet_hydrograph.csv',index=False)
 (dest/'result.json').write_text(json.dumps(result,indent=2))
 return result

def parent_row(registry,family,state):
 rows=registry[registry.family.eq(family)&registry.state.eq(state)]
 if rows.empty and state=='S000_base':
  rows=registry[registry.state.eq(state)].drop_duplicates('fingerprint')
 if len(rows)!=1:raise ValueError(f'Parent state ambiguous or missing: {family}/{state}')
 return rows.iloc[0]

def main(example='03',workers=2,check_only=False,overwrite=False):
 if not check_only and (OUT/'completion.json').exists() and not overwrite:
  raise FileExistsError(f'{OUT} already completed; use a new --output-root or --overwrite.')
 OUT.mkdir(parents=True,exist_ok=True)
 all_inventory=pd.read_csv(ASSETS/'inventory.csv')
 excluded=all_inventory[all_inventory.status.eq('excluded_topology_review')]
 blocked=all_inventory[~all_inventory.status.isin(['ready','excluded_topology_review'])]
 if len(blocked):raise ValueError(f'{example}: {len(blocked)} states have unresolved input status')
 excluded.to_csv(OUT/'excluded_states.csv',index=False)
 inv=all_inventory[all_inventory.status.eq('ready')].copy()
 resultrows=pd.read_csv(ROOT/'RAPID/outputs/cunge_calibration_alpha2/comparison.csv',dtype={'example':str});r=resultrows[resultrows.example.eq(example)&resultrows.method.eq('cunge')].iloc[0]
 source=Path(r.output_dir);base_manifest=json.loads((source/'case.json').read_text());wave=pd.read_parquet(source/'cunge_best_wave.parquet')[['time_utc','time_seconds','q_inflow_cms']]
 event_last=(pd.Timestamp(base_manifest['last_sample'])-pd.Timestamp(base_manifest['first_sample'])).total_seconds();step=base_manifest['forcing_step_s'];n=float(r.n_manning)
 grids={};frames={};hashes={}
 for row in inv.drop_duplicates('fingerprint').itertuples():
  path=Path(row.frame);g=cunge.freeze_grid(pd.read_parquet(path),base_manifest['reference_discharge_cms']);grids[row.fingerprint]=g
  frames[row.fingerprint]=cunge.parameters(g,n);hashes[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
 dtmax=min(cunge.common_clock(g,step)[0] for g in grids.values());nsub=int(np.ceil(step/dtmax));dt=step/nsub
 base_fp=inv[inv.family.eq('indep')&inv.state.eq('S000_base')].iloc[0].fingerprint
 # Check the earlier requirement that untouched exterior links keep hydraulics.
 exterior=[]
 for row in inv.itertuples():
  if pd.isna(row.parent_state) or row.parent_state==row.state:continue
  par=parent_row(inv,row.family,row.parent_state)
  a=frames[par.fingerprint];b=frames[row.fingerprint]
  keys=['u','v','link_length_m','width','slope','depth_m'];z=a.merge(b,on=keys,suffixes=('_p','_c'))
  checks={}
  for col in ['reference_discharge_cms','rapid_celerity_mps','rapid_k','rapid_x','count']:
   left=z[col+'_p'].to_numpy();right=z[col+'_c'].to_numpy()
   checks[col+'_mismatches']=int((~np.isclose(left,right,rtol=1e-10,atol=1e-8)).sum())
   checks[col+'_max_abs_change']=float(np.max(abs(right-left),initial=0))
  exterior.append(dict(family=row.family,state=row.state,parent_state=row.parent_state,shared_unchanged_links=len(z),unchanged_parameters=not any(v for k,v in checks.items() if k.endswith('_mismatches')),**checks))
 pd.DataFrame(exterior).to_csv(OUT/'exterior_invariance.csv',index=False)
 failed=[v for v in exterior if not v['unchanged_parameters']]
 if failed:raise ValueError(f'{example}: {len(failed)} transitions change shared-link parameters; see {OUT}/exterior_invariance.csv. No routing started.')
 if check_only:
  estimate=len(wave)*sum(int(f['count'].sum()) for f in frames.values())*8
  print(json.dumps(dict(example=example,ready_state_entries=len(inv),excluded_state_entries=len(excluded),unique_states=len(frames),estimated_Qout_MiB=estimate/1024**2,routing_dt_s=dt,shared_links_verified=sum(x['shared_unchanged_links'] for x in exterior),status='preflight_passed')),flush=True)
  return
 flow=wave.q_inflow_cms.to_numpy();tail_start=int(round(event_last/step))+1
 for attempt in range(7):
  bad=[]
  for fp,f in frames.items():
   q,audit=cunge.route(f,flow,nsub,dt)
   if audit['minimum_native_flow_cms'] < -1e-9:raise ValueError('Negative preflight flow '+fp)
   if not settling(flow,q,event_last,step):bad.append(fp)
  print(json.dumps(dict(phase='tail_preflight',samples=len(flow),unsettled=len(bad))),flush=True)
  if not bad:break
  oldtail=len(flow)-tail_start;flow=np.r_[flow,np.full(max(oldtail,int(np.ceil(3*86400/step))),flow[-1])]
 else:raise ValueError('Shared tail failed to settle')
 nt=len(flow);nsections=sum(int(f['count'].sum()) for f in frames.values());estimate=nt*nsections*8
 free=shutil.disk_usage(OUT).free
 if free<4*estimate+10*1024**3:raise OSError('Insufficient local free-space reserve')
 common=pd.DataFrame(dict(time_utc=pd.Timestamp(base_manifest['first_sample'])+pd.to_timedelta(np.arange(nt)*step,unit='s'),time_seconds=np.arange(nt)*step,q_inflow_cms=flow));common.to_parquet(OUT/'common_forcing.parquet',index=False)
 metadata=dict(example=example,excluded_state_entries=len(excluded),method='constant_reference_cunge',alpha=2,n_manning=n,calibration='Inherited from parent only; children not calibrated',base_fingerprint=base_fp,
  n_state_entries=len(inv),n_unique_states=len(frames),forcing_step_s=step,routing_dt_s=dt,nsub=nsub,report_steps=nt,simulation_days=(nt-1)*step/86400,event_last_seconds=event_last,
  reference_discharge_cms=base_manifest['reference_discharge_cms'],source_calibration=str(source),source_inventory=str(ASSETS/'inventory.csv'),
  source_frame_sha256=hashes,estimated_Qout_bytes=estimate,free_bytes_before=free,
  shared_exterior_links_verified=sum(x['shared_unchanged_links'] for x in exterior),
  storage_scope='All numerical-reach forcing-interval means retained as Qout.nc. Native substeps audited, not stored. Input prep bundles retained.',
  source_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
 (OUT/'experiment.json').write_text(json.dumps(metadata,indent=2))
 for fp,f in frames.items():
  dest=OUT/'states'/fp;dest.mkdir(exist_ok=True,parents=True);f.to_parquet(dest/'parameters.parquet',index=False)
 rows=[]
 with ProcessPoolExecutor(max_workers=workers,initializer=configure,initargs=(example,str(OUT.parent))) as pool:
  for record in pool.map(run_state,frames):
   rows.append(record);pd.DataFrame(rows).to_csv(OUT/'unique_run_registry.csv',index=False)
   print(json.dumps(dict(completed=len(rows),total=len(frames),fingerprint=record['fingerprint'],lag_h=record['lag_h'],runtime_s=record['runtime_seconds'])),flush=True)
 unique=pd.DataFrame(rows);registry=inv.merge(unique,on='fingerprint',suffixes=('_input',''))
 registry.to_csv(OUT/'state_run_registry.csv',index=False)
 comparisons=[]
 for row in registry.itertuples():
  if pd.isna(row.parent_state) or row.parent_state==row.state:continue
  parent=parent_row(registry,row.family,row.parent_state)
  comparisons.append(dict(family=row.family,parent_state=row.parent_state,child_state=row.state,parent_fingerprint=parent.fingerprint,child_fingerprint=row.fingerprint,
   parent_lag_h=parent.lag_h,child_lag_h=row.lag_h,delta_lag_h=row.lag_h-parent.lag_h,
   parent_peak_cms=parent.peak_cms,child_peak_cms=row.peak_cms,delta_peak_pct=100*(row.peak_cms/parent.peak_cms-1)))
 pd.DataFrame(comparisons).to_csv(OUT/'parent_child_comparison.csv',index=False)
 actual_bytes=sum(f.stat().st_size for f in OUT.rglob('*') if f.is_file())
 baseline=unique[unique.fingerprint.eq(base_fp)].iloc[0]
 finish=dict(state_entries=len(registry),unique_states=len(unique),parent_child_pairs=len(comparisons),all_routed=bool(unique.status.eq('routed').all()),
  actual_bytes=actual_bytes,free_bytes_after=shutil.disk_usage(OUT).free,min_native_flow_cms=float(unique.minimum_substep_flow_cms.min()),max_engine_error_cms=float(unique.engine_cascade_max_error_cms.max()),
  lag_min_h=float(unique.lag_h.min()),lag_max_h=float(unique.lag_h.max()),base_lag_h=float(baseline.lag_h),base_lag_change_from_calibration_h=float(baseline.lag_h-r.lag_h),
  parent_child_max_abs_lag_change_h=float(max(abs(x['delta_lag_h']) for x in comparisons)))
 (OUT/'completion.json').write_text(json.dumps(finish,indent=2));print(json.dumps(finish),flush=True)

def cli():
 parser=argparse.ArgumentParser(description='Run full Cunge hierarchies sequentially with inherited parent calibration.')
 parser.add_argument('--examples',nargs='+',default=['03'],help='Example numbers, e.g. 2 5 13 15; processed in this order.')
 parser.add_argument('--workers',type=int,default=2,help='Concurrent states within each example; examples remain sequential.')
 parser.add_argument('--output-root',type=Path,default=ROOT/'RAPID/outputs/cunge_full_hierarchy_pilot')
 parser.add_argument('--check-only',action='store_true',help='Check inputs and unchanged exterior parameters; do not route.')
 parser.add_argument('--overwrite',action='store_true',help='Explicitly rerun a completed experiment at this destination.')
 args=parser.parse_args()
 if args.workers<1:parser.error('--workers must be positive')
 examples=[]
 for value in args.examples:
  if not value.isdigit():parser.error('Example IDs must be numeric')
  ex=value.zfill(2)
  if ex not in examples:examples.append(ex)
 failures=[]
 for ex in examples:
  configure(ex,args.output_root.resolve())
  print(f'Example {ex}: '+('preflight' if args.check_only else 'full hierarchy'),flush=True)
  try:
   main(ex,args.workers,args.check_only,args.overwrite)
  except (ValueError,AssertionError,FileNotFoundError) as error:
   if not args.check_only:raise
   failures.append(ex);print(str(error),file=sys.stderr,flush=True);continue
  if not args.check_only:
   subprocess.run([sys.executable,str(Path(__file__).with_name('cunge_full_hierarchy_report.py')),'--example',ex,'--output-root',str(args.output_root.resolve())],check=True)
 if failures:raise SystemExit('Preflight failed: '+', '.join(failures))

if __name__=='__main__':cli()
