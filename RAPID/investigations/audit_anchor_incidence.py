"""Additional boundary-incidence and previous comparison-domain checks."""
from pathlib import Path
import pandas as pd,numpy as np
from audit_all_anchoring import OUT,ids

def run():
 a=pd.read_csv(OUT/'transition_audit.csv',dtype={'example':str}).fillna('');rows=[]
 refresh=Path('network_variants/output').resolve().parent/'hydraulic_refresh/shared-head-anchored-depth-signed-inflow-20261003'
 for ex,group in a.groupby('example',sort=False):
  inv=pd.read_csv(refresh/ex/'inventory.csv');br=inv[inv.state.eq('S000_base')].iloc[0];base=pd.read_parquet(br.frame);frames={}
  for r in group.itertuples():
   if r.fingerprint not in frames:frames[r.fingerprint]=pd.read_parquet(r.frame_path)
   f=frames[r.fingerprint];selected=set(ids(r.selected_parent_link_ids));outside=base[~base.id_link.isin(selected)]
   # Exclude exact endpoint/length matches to exterior parent links. Changed
   # exterior links remain in this candidate replacement; a pass is necessary,
   # not sufficient. Separate exterior screen handles that ambiguity.
   lookup={}
   for z in outside.itertuples():lookup.setdefault((z.u,z.v),[]).append(z.link_length_m)
   keep=[not any(np.isclose(z.link_length_m,L,rtol=1e-8,atol=1e-5) for L in lookup.get((z.u,z.v),[])) for z in f.itertuples()]
   replacement=f[keep];bad_in=sorted(set(ids(r.entry_nodes))-set(replacement.u));bad_out=sorted(set(ids(r.exit_nodes))-set(replacement.v))
   p=refresh/ex/'selected_domain_resistance'/r.family/r.state/'domain_links.csv';d=pd.read_csv(p);types=','.join(sorted(set(d.comparison_domain_type.dropna()))) if 'comparison_domain_type' in d else 'not_recorded';parentids=set(d.loc[d.domain_role.eq('parent'),'id_link'].astype(int))
   rows.append(dict(example=ex,family=r.family,state=r.state,fingerprint=r.fingerprint,upstream_missing_replacement_outgoing=','.join(map(str,bad_in)),downstream_missing_replacement_incoming=','.join(map(str,bad_out)),replacement_incidence_pass=not(bad_in or bad_out),comparison_domain_type=types,comparison_domain_differs_from_selected=parentids!=selected,comparison_added_parent_links=','.join(map(str,sorted(parentids-selected))),comparison_removed_selected_links=','.join(map(str,sorted(selected-parentids)))))
  print(ex,'incidence checked',len(rows),flush=True)
 pd.DataFrame(rows).to_csv(OUT/'incidence_and_comparison_domains.csv',index=False)
if __name__=='__main__':run()
