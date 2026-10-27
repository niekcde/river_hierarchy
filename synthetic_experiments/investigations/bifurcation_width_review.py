"""Read-only junction width audit: unsegmented base hydraulic links, one case each."""
from pathlib import Path
import json,pandas as pd,numpy as np
root=Path(__file__).resolve().parents[2];out=root/'synthetic_experiments/outputs/bifurcation_width_review_20261027';out.mkdir(exist_ok=True)
rows=[]
for c in json.loads((root/'RAPID/outputs/event_window_plan_20261013/plan.json').read_text())['cases']:
 ex=c['example'];original=next(j for j in c['jobs'] if j['state_id']=='S000_base')['parameters'];new=root/f'RAPID/outputs/uncapped_event_window_20261020/parameters/{ex}/S000_base/parameters.parquet';path=new if new.exists() else Path(original);f=pd.read_parquet(path)
 for node,outs in f.groupby('u'):
  ins=f[f.v==node]
  if len(outs)<2 or len(ins)!=1:continue
  rows.append(dict(example=ex,node=str(node),n_out=len(outs),incoming_width_m=float(ins.width.sum()),outgoing_width_sum_m=float(outs.width.sum()),ratio=float(outs.width.sum()/ins.width.sum()),incoming_length_m=float(ins.link_length_m.iloc[0]),outgoing_widths_m=json.dumps(outs.width.tolist()),source=str(path)))
r=pd.DataFrame(rows);r.to_csv(out/'junctions.csv',index=False)
s=r.groupby('example').agg(n=('ratio','size'),median=('ratio','median'),p25=('ratio',lambda s:s.quantile(.25)),p75=('ratio',lambda s:s.quantile(.75)),fraction_above_one=('ratio',lambda s:(s>1).mean()));s.to_csv(out/'network_summary.csv');print(s.round(3).to_string())
for label,g in [('all_one_in_multi_out',r),('one_in_two_out',r[r.n_out==2])]:print(label,len(g),g.ratio.quantile([0,.1,.25,.5,.75,.9,1]).to_dict(),'above1',(g.ratio>1).mean())
print('ratios',[(r, r**(-.6),r**(-.4),r**.4,1/r) for r in [1,1.2,1.4]])
