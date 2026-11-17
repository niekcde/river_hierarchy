"""Resolve a timing target near a rejected/accepted roughness frontier.
Coarse scans can miss a valid timing root between an invalid n and a valid n.
Only unresolved original fits are inspected; model equations/bounds stay fixed.
"""
import json,time
import numpy as np
import broad_study as b

def refine(ex):
 path=b.OUT/ex/'calibration.json';cal=json.loads(path.read_text());done=b.OUT/ex/'boundary_refinement.json'
 if done.exists():return cal
 if cal['status']=='timing_calibrated':
  b.save(done,dict(status='not_needed'));return cal
 records={round(r['n'],10):r for r in cal['trials']};good=[r for r in records.values() if r['valid']];far_frontier_stop=False
 if not good:
  b.save(done,dict(status='no_accepted_frontier'));return cal
 best=min(good,key=lambda r:abs(r['timing_error_minutes']));threshold=max(120,1.5*abs(best['timing_error_minutes']));new=[]
 def evaluate(n):
  k=round(float(n),10)
  if k not in records:
   records[k]=b.robust(ex,n=k,phase='search');new.append(k)
  return records[k]
 ns=sorted(records)
 for aa,bb in zip(ns[:-1],ns[1:]):
  ra,rb=records[aa],records[bb]
  if ra['valid']==rb['valid']:continue
  rv=ra if ra['valid'] else rb;ri=rb if ra['valid'] else ra
  if abs(rv['timing_error_minutes'])>threshold:continue
  nv,ni=rv['n'],ri['n']
  for iteration in range(9):
   if abs(nv-ni)<1e-5:break
   if ex in ['22','41','46'] and iteration>=4 and abs(rv['timing_error_minutes'])>60 and abs(nv-ni)/nv<.05:
    far_frontier_stop=True;break
   r=evaluate((nv+ni)/2)
   if not r['valid']:ni=r['n'];continue
   if abs(r['timing_error_minutes'])<=.5:break
   if rv['timing_error_minutes']*r['timing_error_minutes']<0:
    a,c=sorted([rv,r],key=lambda rr:rr['n'])
    for it in range(16):
     ea,ec=a['timing_error_minutes'],c['timing_error_minutes'];frac=float(np.clip(-ea/(ec-ea),.15,.85));rr=evaluate(a['n']+frac*(c['n']-a['n']))
     if not rr['valid'] or abs(rr['timing_error_minutes'])<=.5:break
     if ea*rr['timing_error_minutes']<=0:c=rr
     else:a=rr
    break
   rv=r;nv=r['n']
 good=[r for r in records.values() if r['valid']];feasible=[r for r in good if abs(r['timing_error_minutes'])<=1]
 n0=cal.get('mc_n',float(b.frame(ex,'S000_base').n_manning.iloc[0]))
 best=min(feasible,key=lambda r:(r['height_distance_cms'],abs(np.log(r['n']/n0)))) if feasible else min(good,key=lambda r:abs(r['timing_error_minutes']))
 final=b.robust(ex,n=best['n']);updated=dict(example=ex,status='timing_calibrated' if final['valid'] and abs(final['timing_error_minutes'])<=1 else 'timing_target_unresolved',mc_n=n0,selected_n=best['n'],final=final,trials=list(records.values()),boundary_refined=True)
 if new:
  b.save(b.OUT/ex/'calibration_before_boundary_refinement.json',cal);b.save(path,updated)
 b.save(done,dict(status=updated['status'],additional_n=new,old_n=cal.get('selected_n'),new_n=best['n'],far_frontier_stop=far_frontier_stop,stopping_rule='For expensive networks only: after at least four frontier bisections, stop an unresolved search if the remaining n interval is <5% and accepted timing still misses by >60 minutes. No global impossibility claim.'))
 return updated

if __name__=='__main__':
 pending=set(b.CASES);started=time.monotonic()
 while pending:
  for ex in sorted(pending.copy()):
   if (b.OUT/ex/'calibration.json').exists():
    r=refine(ex);pending.remove(ex);print('BOUNDARY',ex,r['status'],r.get('selected_n'),flush=True)
  if pending:
   if time.monotonic()-started>86400:raise RuntimeError('Wait exceeded one day')
   time.sleep(15)
