"""Small local roughness grid with bounded expansion for expensive networks.
This is a local timing calibration, not an exhaustive search for every peak switch.
Completed broad-scan results are retained and reused.
"""
import json
import numpy as np
import broad_study as b
LARGE={'22','37','41','46','55'}

def calibrate(ex):
 path=b.OUT/ex/'calibration.json'
 if path.exists():return json.loads(path.read_text())
 if ex not in LARGE:return b.calibrate(ex)
 n0=float(b.frame(ex,'S000_base').n_manning.iloc[0]);records={}
 bp=b.OUT/ex/'baseline.json'
 if bp.exists():
  baseline=b.refresh_feature(json.loads(bp.read_text()));records[round(n0,10)]=baseline
 else:
  candidate=b.OUT/ex/f'S000_base_dx100_n{n0:.10f}_full_cfl0.35/result.json'
  if candidate.exists():
   baseline=b.robust(ex,n=n0);records[round(n0,10)]=baseline
 # Previously completed trials remain part of the evidence.
 cached=set()
 for p in (b.OUT/ex).glob('S000_base_dx100_*search_cfl*/result.json'):
  r=json.loads(p.read_text());cached.add(round(r['n'],10))
 for n in sorted(cached):records[n]=b.robust(ex,n=n,phase='search')
 def evaluate(n):
  k=round(float(np.clip(n,.01,.1)),10)
  if k not in records:records[k]=b.robust(ex,n=k,phase='search')
  return records[k]
 base=evaluate(n0)
 if not base['valid'] or base['timing_error_minutes']<-.5:evaluate(1.25*n0)
 elif base['timing_error_minutes']>.5:evaluate(.75*n0)
 def brackets():
  ns=sorted(records);pairs=[]
  for a,c in zip(ns[:-1],ns[1:]):
   ra,rc=records[a],records[c]
   if ra['valid'] and rc['valid'] and ra['timing_error_minutes']*rc['timing_error_minutes']<0:pairs.append((a,c))
  return pairs
 for it in range(12):
  good=[r for r in records.values() if r['valid']]
  if brackets() or any(abs(r['timing_error_minutes'])<=.5 for r in good):break
  if not good:
   new=min(.1,max(records)*1.5)
  else:
   best=min(good,key=lambda r:abs(r['timing_error_minutes']))
   lower=best['timing_error_minutes']>0
   if lower:
    lo=min(r['n'] for r in good)
    # An invalid lower frontier is refined separately after this first fit.
    if any(n<lo and not r['valid'] for n,r in records.items()):break
    new=max(.01,min(records)/1.5)
   else:new=min(.1,max(records)*1.5)
  if round(new,10) in records:break
  evaluate(new)
 for a,c in brackets():
  ra,rc=records[a],records[c]
  for it in range(16):
   ea,ec=ra['timing_error_minutes'],rc['timing_error_minutes'];frac=float(np.clip(-ea/(ec-ea),.15,.85));r=evaluate(a+frac*(c-a))
   if not r['valid'] or abs(r['timing_error_minutes'])<=.5:break
   if ea*r['timing_error_minutes']<=0:c=r['n'];rc=r
   else:a=r['n'];ra=r
 good=[r for r in records.values() if r['valid']]
 if not good:
  result=dict(example=ex,status='no_valid_trial',search_method='adaptive_local_grid',trials=list(records.values()));b.save(path,result);return result
 feasible=[r for r in good if abs(r['timing_error_minutes'])<=1]
 best=min(feasible,key=lambda r:(r['height_distance_cms'],abs(np.log(r['n']/n0)))) if feasible else min(good,key=lambda r:abs(r['timing_error_minutes']))
 final=b.robust(ex,n=best['n']);result=dict(example=ex,status='timing_calibrated' if final['valid'] and abs(final['timing_error_minutes'])<=1 else 'timing_target_unresolved',search_method='adaptive_local_grid',mc_n=n0,selected_n=best['n'],final=final,trials=list(records.values()))
 b.save(path,result);return result

if __name__=='__main__':b.pool(calibrate,list(b.CASES),2)
