"""Materialize common .05–5 m/s diagnostic; select minimal-change interval examples."""
from pathlib import Path
import json
import numpy as np,pandas as pd
from common_celerity_cap_audit import interval_conflicts
ROOT=Path(__file__).resolve().parents[2];BASE=ROOT/'RAPID/assets/bounded_robust_slopes_20261002';OUT=BASE/'common_caps';d=pd.read_csv(OUT/'all_base_links.csv',dtype={'example':str});details=[];counts=[];changes=[]
for ex,g in d.groupby('example'):
 g=g.copy();g['celerity_mps']=g.celerity_raw_mps.clip(.05,5);g['K_seconds']=g.effective_length_m/g.celerity_mps;g['dt_lower_s']=.2*g.K_seconds;g['dt_upper_s']=1.8*g.K_seconds;lo=g.dt_lower_s.to_numpy();hi=g.dt_upper_s.to_numpy();dt,slow,fast=interval_conflicts(lo,hi);minimum=int((slow|fast).sum());L=g.actual_length_m.to_numpy();best=None
 # Among timesteps with minimum conflict count, minimize length-weighted log
 # celerity adjustment needed to enter every interval. Hypothetical, not applied.
 for trial in range(max(1,int(np.ceil(lo.min()))),int(np.floor(hi.max()))+1):
  sl=lo>trial+1e-9;fa=hi<trial-1e-9
  if int((sl|fa).sum())!=minimum:continue
  logchange=np.maximum(np.log(lo/trial),0)+np.maximum(np.log(trial/hi),0);cost=float(L@logchange)
  if best is None or cost<best[0]-1e-9:best=(cost,trial,sl,fa)
 cost,dt,slow,fast=best;nodes=pd.read_parquet(BASE/ex/'floor_0.05/old/nodes.parquet').set_index('id_node');g['node_us_reference_km']=g.id_us_node.map(nodes.reference_section_dist_out_m)/1000;g['node_ds_reference_km']=g.id_ds_node.map(nodes.reference_section_dist_out_m)/1000;g['too_slow_for_current_fastest']=lo>hi.min();g['too_fast_for_current_slowest']=hi<lo.max();g['controls_lower_bound']=np.isclose(lo,lo.max());g['controls_upper_bound']=np.isclose(hi,hi.min());g['cap_active']=~np.isclose(g.celerity_raw_mps,g.celerity_mps);g['diagnostic_dt_s']=dt
 counts.append(dict(example=ex,n_links=len(g),feasible=minimum==0,clipped_links=int(g.cap_active.sum()),minimum_changed_intervals=minimum,representative_dt_s=dt,slow_count_at_dt=int(slow.sum()),fast_count_at_dt=int(fast.sum()),slow_ids=';'.join(map(str,g.loc[slow,'id_link'])),fast_ids=';'.join(map(str,g.loc[fast,'id_link'])),exception_length_pct=100*L[slow|fast].sum()/L.sum(),minimum_log_change_cost=cost))
 for side,mask in [('slow',slow),('fast',fast)]:
  for z in g.loc[mask].to_dict('records'):z.update(conflict_side=side,representative_dt_s=dt);changes.append(z)
 details.extend(g.to_dict('records'))
full=pd.DataFrame(details);full.to_csv(OUT/'stability_link_details.csv',index=False);full[full.cap_active].to_csv(OUT/'cap_exception_links.csv',index=False);pd.DataFrame(counts).to_csv(OUT/'stability_conflict_counts.csv',index=False);pd.DataFrame(changes).to_csv(OUT/'minimum_interval_change_links.csv',index=False)
refs=pd.read_csv(BASE/'hydraulic_references.csv',dtype={'example':str});refs.to_csv(OUT/'hydraulic_references.csv',index=False)
(OUT/'selected_diagnostic.json').write_text(json.dumps(dict(cap_min_mps=.05,cap_max_mps=5.,alpha=.65,n_manning=.035,scope='All 16 base networks, stored BASED depth, selected robust slope candidate, 200m effective segmentation, X=.1',rationale='Broad diagnostic guardrail changes 3/5315 original links; these are explicit review candidates, not evidence of physical errors. Does not force stability.',excel_caps_used=False,local_hydraulic_edits_applied=False,production_changed=False,timestep_tiebreak='Minimum conflicting original links, then minimum physical-length-weighted absolute log celerity change; illustrative only, not applied; no forcing-clock divisibility'),indent=2))
print(pd.DataFrame(counts).drop(columns=['slow_ids','fast_ids']).to_string(index=False))
