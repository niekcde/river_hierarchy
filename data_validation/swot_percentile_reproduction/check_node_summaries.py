from pathlib import Path
import pandas as pd
import numpy as np
import pyarrow.parquet as pq
P=Path(__file__).resolve().parent
s=pd.read_csv(P/'selected_sword_reaches.csv').set_index('reach_id')
n=pd.read_parquet(P/'sample_node_observations.parquet')
g=n.loc[(n.wse_sm_u<=1)&(n.dark_frac<=.5)&n.xtrk_dist.abs().between(10000,60000)&(n.wse_sm>0)].copy()
g['width']=g.width.where(g.width>0)
fields=[f'{v}_obs_p{q}' for v in ['wse','width'] for q in [10,50,90]]
rows=[];coverage=[];nodechecks=[]
for rid,r in s.iterrows():
 sn=pq.read_table(f'/Volumes/PhD/SWORD/v17c/sword_{r.region}_v17c_nodes.parquet',columns=['reach_id','node_id','n_obs']+fields,filters=[('reach_id','=',rid)]).to_pandas().set_index('node_id')
 sn[fields]=sn[fields].where((sn[fields]>-9999)&np.isfinite(sn[fields]))
 gn=g.loc[g.reach_id==rid].copy();gn.node_id=gn.node_id.astype('int64')
 local=gn.groupby('node_id')[['wse_sm','width']].quantile([.1,.5,.9]).unstack()
 local.columns=[f'{"wse" if v=="wse_sm" else v}_obs_p{int(q*100)}' for v,q in local.columns]
 for source,df in [('sword_nodes',sn[fields]),('local_nodes',local)]:
  for agg in ['mean','median']:
   values=df.agg(agg)
   rec=dict(reach_id=rid,source=source,aggregation=agg)
   for f in fields: rec[f]=values[f];rec[f'delta_{f}']=values[f]-r[f]
   rows.append(rec)
 joined=local.join(sn[fields],lsuffix='_local',rsuffix='_sword')
 for f in fields:
  delta=joined[f+'_local']-joined[f+'_sword']
  nodechecks.append(dict(reach_id=rid,field=f,compared=int(delta.notna().sum()),exact=int((delta.abs()<1e-6).sum()),median_absolute_difference=delta.abs().median()))
 validtime=pd.to_datetime(n.loc[n.reach_id==rid,'time_str'],errors='coerce',utc=True)
 coverage.append(dict(reach_id=rid,sword_reach_n_obs=r.n_obs,sword_node_n_obs_min=sn.n_obs.min(),sword_node_n_obs_max=sn.n_obs.max(),sword_node_n_obs_mean=sn.n_obs.mean(),sword_node_n_obs_median=sn.n_obs.median(),local_filtered_node_rows=len(gn),local_filtered_nodes=gn.node_id.nunique(),local_filtered_passes=len(gn[['cycle','pass']].drop_duplicates()),first_time=str(validtime.min()),last_time=str(validtime.max())))
pd.DataFrame(rows).to_csv(P/'node_summary_comparison.csv',index=False)
pd.DataFrame(nodechecks).to_csv(P/'node_percentile_comparison.csv',index=False)
pd.DataFrame(coverage).to_csv(P/'coverage_comparison.csv',index=False)
print(pd.DataFrame(coverage).to_string(index=False))
for (source,agg),d in pd.DataFrame(rows).groupby(['source','aggregation']):
 print(source,agg,d[['delta_'+f for f in fields]].abs().median().to_dict())
print(pd.read_csv(P/'comparison.csv').query('method=="node_pooled_repository"')[['reach_id','sword_wse_p50','wse_p50','sword_width_p50','width_p50']].to_string(index=False))
