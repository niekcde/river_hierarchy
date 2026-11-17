"""Transfer final original fits after frontier refinement, reusing completed trials."""
import json,time
import broad_study as b
pending=set(b.CASES);start=time.monotonic()
while pending:
 for ex in sorted(pending.copy()):
  if all((b.OUT/ex/name).exists() for name in ['calibration.json','boundary_refinement.json']):
   c=json.loads((b.OUT/ex/'calibration.json').read_text());p=b.OUT/ex/'transfer.json';old=json.loads(p.read_text()) if p.exists() else {}
   if 'selected_n' in c and (old.get('n')!=c['selected_n'] or old.get('parent_calibration_status')!=c['status']):
    r=b.transfer(ex);b.save(p,r)
   elif 'selected_n' not in c:b.save(p,dict(example=ex,status='no_n_available'))
   pending.remove(ex);print('TRANSFER COMPLETE',ex,c['status'],flush=True)
 if pending:
  if time.monotonic()-start>86400:raise RuntimeError('Wait exceeded one day')
  time.sleep(15)
