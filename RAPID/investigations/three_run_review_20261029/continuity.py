"""Fixed Manning depth >5m and adjacent-reach depth discontinuity screening.

Depth differences are NOT mass-balance violations. Junctions are separated from
one-in/one-out nodes. New/amplified flags compare to parent maxima at same node.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from review import SOURCE,DEST,RUNS,digest

def edges(f):
 cols=['id_link','u','v','depth_m','width','slope','reference_discharge_cms']
 incoming=f[cols].rename(columns={x:'up_'+x for x in cols if x!='v'}).rename(columns={'v':'node'})
 outgoing=f[cols].rename(columns={x:'down_'+x for x in cols if x!='u'}).rename(columns={'u':'node'})
 d=incoming.merge(outgoing,on='node')
 d['junction']=d.node.map(f.groupby('v').size()).gt(1)|d.node.map(f.groupby('u').size()).gt(1)
 d['depth_jump_m']=d.down_depth_m-d.up_depth_m
 d['abs_depth_jump_m']=d.depth_jump_m.abs()
 d['depth_ratio']=np.maximum(d.down_depth_m,d.up_depth_m)/np.minimum(d.down_depth_m,d.up_depth_m)
 return d

def main():
 rows=[];flagged=[];parent_edges=[]
 for exdir in sorted((SOURCE/RUNS[2]/'states').iterdir()):
  pf=pd.read_parquet(exdir/'S000_base/parameters.parquet');pe=edges(pf)
  pn=pe.groupby('node').agg(parent_node_max_jump_m=('abs_depth_jump_m','max'),parent_node_max_ratio=('depth_ratio','max'))
  pp=pe.set_index(['up_id_link','down_id_link'])[['abs_depth_jump_m','depth_ratio']].rename(columns={'abs_depth_jump_m':'parent_pair_jump_m','depth_ratio':'parent_pair_ratio'})
  parent_edges.append(pe.assign(example=exdir.name))
  for folder in sorted(exdir.iterdir()):
   if folder.name=='single_centerline':continue
   f=pd.read_parquet(folder/'parameters.parquet');d=edges(f).merge(pn,left_on='node',right_index=True,how='left').merge(pp,left_on=['up_id_link','down_id_link'],right_index=True,how='left')
   d['new_or_amplified_at_node']=(d.abs_depth_jump_m>d.parent_node_max_jump_m+1)|(d.parent_node_max_jump_m.isna())
   d['matches_parent_pair']=np.isclose(d.abs_depth_jump_m,d.parent_pair_jump_m,rtol=1e-9,atol=1e-8)&np.isclose(d.depth_ratio,d.parent_pair_ratio,rtol=1e-9,atol=1e-8)
   h=f.depth_m;ratio=h/f.width
   row=dict(example=exdir.name,state=folder.name,links=len(f),links_depth_gt5=int(h.gt(5).sum()),links_depth_gt5_width_le30=int((h.gt(5)&f.width.le(30)).sum()),links_ratio_gt_1over6=int(ratio.gt(1/6).sum()),
    max_jump_m=float(d.abs_depth_jump_m.max()),max_ratio=float(d.depth_ratio.max()))
   for label,g in [('all',d),('serial',d[~d.junction]),('junction',d[d.junction])]:
    row[label+'_pairs']=len(g)
    for threshold in [2,5,10]:row[label+'_jump_gt'+str(threshold)]=int(g.abs_depth_jump_m.gt(threshold).sum())
    row[label+'_ratio_gt2']=int(g.depth_ratio.gt(2).sum())
    severe=g.abs_depth_jump_m.gt(5)&g.depth_ratio.gt(2)
    row[label+'_jump_gt5_ratio_gt2']=int(severe.sum())
    row[label+'_new_amplified_severe']=int((severe&g.new_or_amplified_at_node).sum())
   rows.append(row)
   flags=d[(d.abs_depth_jump_m>5)|(d.depth_ratio>2)].copy();flags['example']=exdir.name;flags['state']=folder.name;flagged.append(flags)
  print('Continuity',exdir.name,flush=True)
 ds=pd.DataFrame(rows);ds.to_csv(DEST/'depth5_and_continuity_by_state.csv',index=False)
 pd.concat(flagged,ignore_index=True).to_parquet(DEST/'flagged_depth_connections.parquet',index=False)
 pd.concat(parent_edges,ignore_index=True).to_csv(DEST/'parent_depth_connections.csv',index=False)
 sums=[]
 for group,g in [('parents',ds[ds.state.eq('S000_base')]),('collapses',ds[ds.state.ne('S000_base')])]:
  for col in [c for c in ds if c.startswith('links_') or ('_jump_' in c and not c.startswith('max_')) or '_ratio_gt' in c or '_new_amplified' in c]:
   sums.append(dict(group=group,criterion=col,states=int(g[col].gt(0).sum()),occurrences=int(g[col].sum()),total_states=len(g)))
 pd.DataFrame(sums).to_csv(DEST/'depth5_continuity_counts.csv',index=False)
 print(pd.DataFrame(sums).to_string(index=False),flush=True)
 (DEST/'continuity_method.json').write_text(json.dumps(dict(script_sha256=digest(__file__),
 thresholds='Screen only: >5 m depth, >5 m adjacent depth jump plus >2 depth ratio; not physical validity cutoffs.',
 new_amplified='Adjacent depth jump exceeds the maximum parent jump at the same node by >1m, or node absent in parent. Junction comparisons use every incoming/outgoing reach pair.',
 caveats='Local normal-depth estimates, not computed water-surface elevations. A jump does not imply loss of discharge continuity. Repeated connections across states are occurrences, not independent river locations.'),indent=2)+'\n')
if __name__=='__main__':main()
