import sys,json
sys.path.insert(0,'RAPID/investigations/cap_investigation_20261020')
from audit import OUT,read,SOURCES,save
import pandas as pd,numpy as np
jobs=read(OUT/'jobs.json');rows=[];detail=[]
for j in jobs:
 f=read(j['parameters']);raw=f.rapid_celerity_mps_raw;cap=raw>f.celerity_max_mps
 cohort='parent' if j['state_id']=='S000_base' else 'segmented' if j['state_id']=='all_closed_units_collapsed' else 'uniform' if j['state_id']=='single_centerline' else 'partial_child'
 rows.append(dict(example=j['example'],cohort=cohort,state=j['state_id'],links=len(f),capped=int(cap.sum()),cells=int(f['count'].sum()),capped_cells=int(f.loc[cap,'count'].sum())))
 if cap.any():
  x=f.loc[cap,['id_link','rapid_celerity_mps_raw','link_length_m','width','depth_m','slope','n_manning','count']].copy();x['example']=j['example'];x['cohort']=cohort;x['state']=j['state_id'];detail.append(x)
d=pd.DataFrame(rows);a=pd.concat(detail,ignore_index=True)
d.to_csv(OUT/'link_counts_by_state.csv',index=False);a.to_parquet(OUT/'upper_capped_link_occurrences.parquet',index=False)
s=a.groupby(['example','cohort']).rapid_celerity_mps_raw.agg(['size','min','median',lambda x:x.quantile(.95),'max']);s.to_csv(OUT/'raw_celerity_ranges.csv')
print(d.groupby('cohort')[['links','capped','cells','capped_cells']].sum().to_string());print(s.to_string());print('PARENT NETWORK TOTALS');print(d[d.cohort=='parent'].to_string(index=False))
print('ALL STATES bins');print(pd.crosstab(a.example,pd.cut(a.rapid_celerity_mps_raw,[5,6,8,10,12,14],right=True)).to_string())
save(OUT/'link_range_sources.json',SOURCES)
