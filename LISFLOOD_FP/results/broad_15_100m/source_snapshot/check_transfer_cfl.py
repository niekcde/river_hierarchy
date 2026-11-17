"""Use a common accepted CFL for each fitted original/collapse comparison."""
import json,time
import broad_study as b
pending=set(b.CASES);started=time.monotonic()
while pending:
 for ex in sorted(pending.copy()):
  d=b.OUT/ex;names=['calibration.json','boundary_refinement.json','transfer.json']
  if not all((d/name).exists() for name in names):continue
  cp=d/'calibration.json';tp=d/'transfer.json';cal=json.loads(cp.read_text());tr=json.loads(tp.read_text());dest=d/'common_cfl_check.json'
  if 'selected_n' not in cal:
   b.save(dest,dict(status='no_selected_n'));pending.remove(ex);continue
  if tr.get('n')!=cal['selected_n']:continue
  if dest.exists():pending.remove(ex);continue
  original=cal['final'];runs=tr['runs'];accepted=[r for r in [original,*runs] if r['valid']]
  if not accepted:
   b.save(dest,dict(status='no_accepted_run'));pending.remove(ex);continue
  cfl=min(r['cfl'] for r in accepted)
  if all(r['cfl']==cfl for r in accepted):
   b.save(dest,dict(status='already_common',cfl=cfl));pending.remove(ex);continue
  checked=[]
  for r in [original,*runs]:
   checked.append(b.trial(ex,r['state'],n=cal['selected_n'],cfl=cfl) if r['valid'] and r['cfl']!=cfl else r)
  fine=checked[0];changes=[]
  for old,new in zip([original,*runs],checked):
   diff=dict(state=old['state'],old_cfl=old['cfl'],new_cfl=new['cfl'],valid=new['valid'])
   if old['valid'] and new['valid']:
    for key in ['peak_seconds','centroid_seconds','rising50_seconds','falling50_seconds']:
     if old.get(key) is not None and new.get(key) is not None:diff[key.replace('_seconds','_change_seconds')]=new[key]-old[key]
   changes.append(diff)
  b.save(d/'calibration_before_common_cfl.json',cal);b.save(d/'transfer_before_common_cfl.json',tr)
  cal['final']=fine;cal['common_cfl']=cfl
  if cal['status']=='timing_calibrated' and not(fine['valid'] and abs(fine['timing_error_minutes'])<=1):cal['status']='timing_target_unresolved_at_common_cfl'
  tr['runs']=checked[1:];tr['parent_calibration_status']=cal['status'];tr['common_cfl']=cfl
  b.save(cp,cal);b.save(tp,tr);b.save(dest,dict(status='checked',cfl=cfl,changes=changes,parent_status=cal['status']))
  print('COMMON CFL',ex,cfl,cal['status'],flush=True);pending.remove(ex)
 if pending:
  if time.monotonic()-started>86400:raise RuntimeError('Wait exceeded one day')
  time.sleep(15)
