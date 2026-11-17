"""Wait for study stages, synchronize transfers after frontier refinement, finalize artifacts."""
import json,time,subprocess,sys,shutil
from pathlib import Path
import broad_study as b

def ready():
 sel=json.loads((b.OUT/'selection.json').read_text())
 for ex in b.CASES:
  if not all((b.OUT/ex/name).exists() for name in ['baseline.json','calibration.json','boundary_refinement.json','transfer.json','common_cfl_check.json']):return False
 for ex in b.SUBSET:
  for state in ['S000_base',*sel['children'][ex]]:
   n=float(b.frame(ex,state).n_manning.iloc[0])
   for length in [200,100,50]:
    done=False
    for cfl in [.35,.175,.0875]:
     p=b.OUT/ex/f'{state}_dx{length:g}_n{n:.10f}_full_cfl{cfl:g}/result.json'
     if p.exists():
      r=json.loads(p.read_text());spin=r.get('spinup_audit') or {}
      steady_cap=('Critical-flow limiter active' in r.get('error','') and spin.get('limited_face_steps',0)>0 and spin.get('clipping_volume_m3',1)<1e-6)
      if r['valid'] or cfl==.0875 or steady_cap:done=True;break
    if not done:return False
 return True

if __name__=='__main__':
 start=time.monotonic()
 while not ready():
  if time.monotonic()-start>86400:raise RuntimeError('Stage wait exceeded one day')
  time.sleep(30)
 redo=[]
 for ex in b.CASES:
  c=json.loads((b.OUT/ex/'calibration.json').read_text());t=json.loads((b.OUT/ex/'transfer.json').read_text())
  if 'selected_n' in c and (t.get('n')!=c['selected_n'] or t.get('parent_calibration_status')!=c['status']):redo.append(ex)
 print('Synchronizing final selected roughness for',redo,flush=True);b.pool(b.transfer,redo,2)
 for script in ['summarize_broad.py','plot_broad_grid.py','forecast_broad_runtime.py','report_broad.py']:
  subprocess.run([sys.executable,str(b.ROOT/'LISFLOOD_FP'/script)],check=True)
 source=b.OUT/'source_snapshot';source.mkdir(exist_ok=True)
 for name in ['broad_study.py','calibrate_broad_adaptive.py','refine_broad_boundaries.py','inertial_verification_kernel.py','verify_inertial56.py','summarize_broad.py','plot_broad_grid.py','forecast_broad_runtime.py','report_broad.py','finish_broad.py','transfer_broad_watch.py','recover_broad_workers.py','check_transfer_cfl.py','check_calibrated_grids.py']:
  shutil.copy2(b.ROOT/'LISFLOOD_FP'/name,source/name)
 protocol=json.loads((b.OUT/'protocol.json').read_text());protocol.update(kernel_sha256=b.sha(b.v.KERNEL),steady_cap_retry_policy='Skip redundant smaller-CFL retries when the converged steady initial state already activates the cap; example 15 and 22 repeats demonstrated persistence. Other failure modes retain retries.',source_hashes={p.name:b.sha(p) for p in source.glob('*.py')},calibration_feature_window='Observed downstream coverage from input-event start, including delayed downstream crests after upstream event end',search_horizon='Later of input-event end and target + three days, capped at full forcing end',unresolved_frontier_refinement='Bisect accepted/rejected n frontiers near the best timing candidate, up to nine trials; refine any valid sign-changing timing bracket; retain original bounds',finalized_utc=str(__import__('pandas').Timestamp.now(tz='UTC')))
 b.save(b.OUT/'protocol.json',protocol);b.save(b.OUT/'stages_complete.json',dict(complete=True,transfer_resynchronizations=redo,seconds_waiting_and_finalizing=time.monotonic()-start))
 print('STUDY STAGES COMPLETE',flush=True)
