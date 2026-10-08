"""Hydraulic ranges of unique materialized geometries; no routing or mutation."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from full_signed_routing_validation import *
rows=[]
inv=pd.read_csv(OUT/'inventory.csv',dtype={'example':str})
for r in inv[inv.status=='ready'].drop_duplicates(['example','fingerprint']).itertuples():
 f=pd.read_parquet(r.frame)
 rows.append(dict(example=r.example,fingerprint=r.fingerprint,state=r.state,min_slope=f.slope.min(),max_slope=f.slope.max(),min_depth=f.depth_m.min(),max_depth=f.depth_m.max(),min_raw_celerity=f.rapid_celerity_mps_raw.min(),max_raw_celerity=f.rapid_celerity_mps_raw.max(),low_cap_links=int((f.rapid_celerity_mps_raw<.05).sum()),high_cap_links=int((f.rapid_celerity_mps_raw>5).sum())))
pd.DataFrame(rows).to_csv(OUT/'candidate_hydraulic_ranges.csv',index=False)
print(pd.DataFrame(rows).drop(columns=['example','fingerprint','state']).agg(['min','max']).to_string())
