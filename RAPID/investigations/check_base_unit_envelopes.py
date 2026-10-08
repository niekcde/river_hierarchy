from pathlib import Path
import pandas as pd,numpy as np
O=Path('RAPID/assets/data_informed_base_slopes_20261002');rows=[]
for d in sorted(O.iterdir()):
 if not d.is_dir() or not (d/'summary.json').exists():continue
 f=pd.read_parquet(d/'reference_profile.parquet');nn=pd.read_parquet(d/'candidate_node_hydraulic_context.parquet').set_index('id_node');ll=pd.read_parquet(d/'candidate_link_hydraulic_context.parquet');ss=pd.read_csv(d/'section_constraints.csv');uu=pd.read_csv(d/'unit_solver_audit.csv');x=f.reference_section_dist_out_m.to_numpy();dx=np.diff(x);mid=(x[:-1]+x[1:])/2;T=np.diff(f.fitted_wse_m)/dx
 for r in ss[ss.used].itertuples():T[(mid>=r.lo)&(mid<=r.hi)]=r.p75
 lows={};targets={};worst=0
 for v in uu.itertuples():
  g=ll[(ll.ownership_kind==v.kind)&(ll.ownership_domain_id==v.domain)]
  for r in g.itertuples():
   lo,hi=sorted([nn.loc[r.id_us_node,'reference_section_dist_out_m'],nn.loc[r.id_ds_node,'reference_section_dist_out_m']]);w=np.maximum(0,np.minimum(x[1:],hi)-np.maximum(x[:-1],lo));target=float(np.average(T,weights=w)) if w.sum()>0 else float(T[min(len(T)-1,max(0,np.searchsorted(x,lo)-1))]);bound=v.max_feasible_target_fraction*target;targets[r.id_link]=target;lows[r.id_link]=bound;worst=max(worst,(bound-r.slope_used)*r.actual_length_m)
 ll['unit_target_slope']=ll.id_link.map(targets);ll['unit_lower_envelope_slope']=ll.id_link.map(lows);ll.to_parquet(d/'candidate_link_hydraulic_context.parquet',index=False)
 rows.append(dict(example=d.name,max_unit_lower_bound_violation_m=worst))
pd.DataFrame(rows).to_csv(O/'unit_bound_validation.csv',index=False);print(pd.DataFrame(rows).to_string(index=False))
