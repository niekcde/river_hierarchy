"""Read-only model-run status (writes a small progress manifest)."""
import json
import pandas as pd
import broad_study as b

def read(p):return json.loads(p.read_text()) if p.exists() else None

def main():
 selection=read(b.OUT/'selection.json')['children'];inventory=pd.read_csv(b.OUT/'inventory.csv',dtype={'example':str}).set_index('example');rows=[]
 for ex in b.CASES:
  d=b.OUT/ex;base=read(d/'baseline.json');cal=read(d/'calibration.json');transfer=read(d/'transfer.json');grid_done=grid_valid=0
  if ex in b.SUBSET:
   n=float(inventory.loc[ex,'mc_n'])
   for state in ['S000_base',*selection[ex]]:
    for length in [200,100,50]:
     for cfl in [.35,.175,.0875]:
      r=read(d/f'{state}_dx{length}_n{n:.10f}_full_cfl{cfl:g}'/'result.json')
      if not r:continue
      spin=r.get('spinup_audit') or {};steady_cap='Critical-flow limiter active' in r.get('error','') and spin.get('limited_face_steps',0)>0 and spin.get('clipping_volume_m3',1)<1e-6
      if r['valid'] or cfl==.0875 or steady_cap:
       grid_done+=1;grid_valid+=int(r['valid']);break
  rows.append(dict(example=ex,baseline='accepted' if base and base['valid'] else 'rejected' if base else 'pending',calibration=cal['status'] if cal else 'pending',selected_n=cal.get('selected_n') if cal else None,timing_error_minutes=(cal.get('final') or {}).get('timing_error_minutes') if cal else None,frontier_done=(d/'boundary_refinement.json').exists(),transfer_attempts=len((transfer or {}).get('runs',[])),transfer_accepted=sum(r['valid'] for r in (transfer or {}).get('runs',[])),common_cfl_done=(d/'common_cfl_check.json').exists(),grid_done=grid_done,grid_accepted=grid_valid,grid_expected=9 if ex in b.SUBSET else 0))
 df=pd.DataFrame(rows);b.save(b.OUT/'progress.json',dict(updated_utc=str(pd.Timestamp.now(tz='UTC')),cases=rows));print(df.to_string(index=False))
if __name__=='__main__':main()
