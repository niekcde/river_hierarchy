"""Check global initial continuity as well as the solver's local-node criterion."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import broad_study as b

def main():
 rows=[];seen=set()
 for ex in b.CASES:
  rs=[]
  for name in ['baseline.json','calibration.json','transfer.json']:
   p=b.OUT/ex/name
   if not p.exists():continue
   r=json.loads(p.read_text())
   rs.extend(r.get('runs',[]) if name=='transfer.json' else [r['final']] if name=='calibration.json' and 'final' in r else [r] if name=='baseline.json' else [])
  for r in rs:
   if not r.get('valid') or r['folder'] in seen:continue
   seen.add(r['folder']);f=b.frame(ex,r['state']);cells=b.v.grid(f,f"max{r['length']}");m=b.build(cells,r['n']);d=np.load(Path(r['folder'])/'initial.npz');q=d['q'];Q=float(b.forcing(ex).forcing_cms.iloc[0]);terminal=m['b']==len(cells);out=float(np.sum(q[terminal]*m['wf'][terminal]));relative=abs(out-Q)/max(Q,1)
   rows.append(dict(example=ex,state=r['state'],n=r['n'],initial_inflow_cms=Q,initial_outflow_cms=out,relative_initial_global_imbalance=relative,maximum_local_imbalance_cms=r['spinup_audit']['final_node_imbalance_cms'],folder=r['folder']))
 df=pd.DataFrame(rows);df.to_csv(b.OUT/'initial_state_checks.csv',index=False)
 print('Initial states',len(df),'largest relative global imbalance',df.relative_initial_global_imbalance.max())
if __name__=='__main__':main()
