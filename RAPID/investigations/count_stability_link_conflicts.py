"""Minimum interval changes needed for a common integer timestep, fixed X/segmentation."""
from pathlib import Path
import pandas as pd,numpy as np
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'RAPID/assets/bounded_robust_slopes_20261002'
d=pd.read_csv(OUT/'stability_link_details.csv',dtype={'example':str});summaries=[];conflicts=[]
representative={'05':180,'11':140,'13':180,'15':162,'22':140,'41':150,'46':185}
for ex,g in d.groupby('example'):
 counts=[]
 for dt in range(max(1,int(np.ceil(g.dt_lower_s.min()))),int(np.floor(g.dt_upper_s.max()))+1):
  ns=int((g.dt_lower_s>dt+1e-9).sum());nf=int((g.dt_upper_s<dt-1e-9).sum());counts.append((dt,ns,nf))
 minimum=min(a+b for _,a,b in counts);optimal=[dt for dt,a,b in counts if a+b==minimum];dt=representative[ex];assert dt in optimal
 slow=g[g.dt_lower_s>dt+1e-9];fast=g[g.dt_upper_s<dt-1e-9]
 summaries.append(dict(example=ex,n_links=len(g),slow_side_only_count=int((g.dt_lower_s>g.dt_upper_s.min()).sum()),fast_side_only_count=int((g.dt_upper_s<g.dt_lower_s.max()).sum()),minimum_changed_intervals=minimum,representative_dt_s=dt,slow_count_at_dt=len(slow),fast_count_at_dt=len(fast),slow_ids=';'.join(map(str,slow.id_link)),fast_ids=';'.join(map(str,fast.id_link)),all_optimal_integer_timesteps=';'.join(map(str,optimal))))
 for side,frame in [('slow',slow),('fast',fast)]:
  for z in frame.to_dict('records'):z.update(conflict_side=side,representative_dt_s=dt);conflicts.append(z)
pd.DataFrame(summaries).to_csv(OUT/'stability_conflict_counts.csv',index=False);pd.DataFrame(conflicts).to_csv(OUT/'minimum_interval_change_links.csv',index=False)
print(pd.DataFrame(summaries).drop(columns=['all_optimal_integer_timesteps']).to_string(index=False))
