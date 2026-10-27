"""Compare six existing example reaches with local SWORD v17c; read-only sources."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import json
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

OUT=Path(__file__).resolve().parent
BASE=OUT.parent/'swot_sword_17c'
SWOT=Path('/Volumes/PhD/SWOT/RiverSP_D_parq')
SWORD=Path('/Volumes/PhD/SWORD/v17c')
fields=[f'{v}_obs_p{q}' for v in ['wse','width'] for q in [10,50,90]]
assoc=pd.read_csv(BASE/'example_reaches.csv')
cache=pd.read_parquet(BASE/'target_observations.parquet')
parts=[]
for region in ['AS','SA']:
 d=pq.read_table(SWORD/f'sword_{region}_v17c_reaches.parquet',columns=['reach_id','river_name','slope','n_nodes','n_obs','swot_obs_source']+fields,filters=[('reach_id','in',assoc.loc[assoc.region==region,'reach_id'].tolist())]).to_pandas()
 d=d.loc[(d[fields]>-9999).all(axis=1)&d.reach_id.isin(cache.loc[cache.passes_repository_filter,'reach_id'])].sort_values('slope')
 d=d.iloc[[0,len(d)//2,len(d)-1]].copy();d['region']=region;parts.append(d)
sword=pd.concat(parts,ignore_index=True)
sword.to_csv(OUT/'selected_sword_reaches.csv',index=False)
ids=sword.reach_id.tolist()
# Reuse the earlier exhaustive archive scan to identify every node granule containing these IDs.
names=sorted(cache.loc[cache.reach_id.isin(ids),'source_file'].unique())

def read(name,kind):
 p=SWOT/kind/name.replace('_Node_','_Reach_') if kind=='reach' else SWOT/kind/name
 if not p.exists(): raise FileNotFoundError(p)
 cols=['reach_id','time_str','cycle','pass','crid','version','wse','wse_u','width','dark_frac','xtrk_dist']
 if kind=='node': cols+=['node_id','wse_sm','wse_sm_u']
 d=pq.read_table(p,columns=cols,filters=[('reach_id','in',[str(x) for x in ids])]).to_pandas()
 d['reach_id']=d.reach_id.astype('int64');d['source_file']=p.name
 return d
with ThreadPoolExecutor(max_workers=8) as pool:
 node=pd.concat(pool.map(lambda n:read(n,'node'),names),ignore_index=True)
 reach=pd.concat(pool.map(lambda n:read(n,'reach'),names),ignore_index=True)
node.to_parquet(OUT/'sample_node_observations.parquet',index=False)
reach.to_parquet(OUT/'sample_reach_observations.parquet',index=False)
assert len(node)==cache.reach_id.isin(ids).sum()
rows=[]
def compare(d,method,wse='wse_sm',passagg=None):
 d=d.copy();d['wse_value']=d[wse].where(np.isfinite(d[wse])&(d[wse]>-9999))
 d['width_value']=d.width.where(np.isfinite(d.width)&(d.width>0))
 for rid,g in d.groupby('reach_id'):
  ref=sword.set_index('reach_id').loc[rid]
  if passagg:
   vals=g.groupby(['cycle','pass'])[['wse_value','width_value']].agg(passagg)
  else: vals=g[['wse_value','width_value']]
  rec=dict(reach_id=rid,method=method,n_rows=len(g),n_samples=len(vals),n_nodes=g.node_id.nunique() if 'node_id' in g else np.nan,sword_n_obs=ref.n_obs)
  for v in ['wse','width']:
   rec[f'n_valid_{v}']=int(vals[f'{v}_value'].notna().sum())
   for q in [10,50,90]:
    k=f'{v}_p{q}';rec[k]=vals[f'{v}_value'].quantile(q/100);rec[f'sword_{k}']=ref[f'{v}_obs_p{q}'];rec[f'delta_{k}']=rec[k]-rec[f'sword_{k}']
  rows.append(rec)
ng=node.loc[(node.wse_sm_u<=1)&(node.dark_frac<=.5)&node.xtrk_dist.abs().between(10000,60000)&(node.wse_sm>0)].copy()
compare(ng,'node_pooled_repository')
compare(ng,'node_pass_mean_repository',passagg='mean')
compare(ng,'node_pass_median_repository',passagg='median')
compare(ng.drop_duplicates(['reach_id','node_id','cycle','pass','time_str']),'node_pooled_deduplicated')
compare(node,'node_pooled_valid_only')
# Reach product has wse_u instead of wse_sm_u: this is an analogous, not identical, filter.
rg=reach.loc[(reach.wse_u<=1)&(reach.dark_frac<=.5)&reach.xtrk_dist.abs().between(10000,60000)&(reach.wse>0)]
compare(rg,'reach_analogous_filter','wse')
compare(reach,'reach_valid_only','wse')
compare(reach.loc[(reach.wse_u<=1)&(reach.dark_frac<=.5)&(reach.wse>0)],'reach_no_xtrack','wse')
result=pd.DataFrame(rows)
result.to_csv(OUT/'comparison.csv',index=False)
meta=dict(node_files=len(names),node_rows=len(node),reach_rows=len(reach),node_date_min=node.time_str.min(),node_date_max=node.time_str.max(),crids=node.crid.value_counts().to_dict(),versions=node.version.value_counts().to_dict(),note='Node granules selected via prior exhaustive scan; reach comparison restricted to counterpart granules. Valid-only methods exclude fills, not quality-screened. Exact agreement tolerance 1e-6 m. Local SWORD files, not freshly downloaded release.')
(OUT/'metadata.json').write_text(json.dumps(meta,indent=2,default=str)+'\n')
print(sword.to_string(index=False))
print(json.dumps(meta,indent=2,default=str))
for method,d in result.groupby('method'):
 print(method, 'median abs differences:',d[[f'delta_{v}_p{q}' for v in ['wse','width'] for q in [10,50,90]]].abs().median().round(6).to_dict())
