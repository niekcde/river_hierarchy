"""Post-fit refinement checks, including the reversed collapse response in 37."""
import json,time
import pandas as pd
import broad_study as b
selection=json.loads((b.OUT/'selection.json').read_text());rows=[]
for ex in ['02','03','15','56','37']:
 while not all((b.OUT/ex/p).exists() for p in ['calibration.json','boundary_refinement.json','transfer.json']):time.sleep(15)
 cal=json.loads((b.OUT/ex/'calibration.json').read_text());tr=json.loads((b.OUT/ex/'transfer.json').read_text())
 if cal['status']!='timing_calibrated':continue
 state=selection['children'][ex][-1];base100=cal['final'];child100=next(r for r in tr['runs'] if r['state']==state)
 base50=b.robust(ex,n=cal['selected_n'],length=50);child50=b.robust(ex,state,n=cal['selected_n'],length=50)
 r=dict(example=ex,state=state,n=cal['selected_n'],timing_error_100_minutes=base100.get('timing_error_minutes'),timing_error_50_minutes=base50.get('timing_error_minutes'),valid=all(x['valid'] for x in [base100,child100,base50,child50]),base50=base50,child50=child50)
 if r['valid']:
  for key in ['peak_seconds','centroid_seconds','rising50_seconds','falling50_seconds']:
   if all(x.get(key) is not None for x in [base100,child100,base50,child50]):
    r[key.replace('_seconds','_original_100_minus_50_minutes')]=(base100[key]-base50[key])/60
    r[key.replace('_seconds','_effect_100_minus_50_minutes')]=((child100[key]-base100[key])-(child50[key]-base50[key]))/60
 rows.append(r);b.save(b.OUT/'calibrated_grid_checks.json',rows);print('FITTED GRID',ex,r['valid'],flush=True)
pd.DataFrame([{k:v for k,v in r.items() if k not in ['base50','child50']} for r in rows]).to_csv(b.OUT/'calibrated_grid_checks.csv',index=False)
