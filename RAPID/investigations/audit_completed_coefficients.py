"""Count signed current-inflow coefficients from completed production inputs.

A source segment's external input uses a+b, so it has no negative current-
inflow contribution even if its nominal a is negative. Count that separately.
"""
from pathlib import Path
import argparse,json
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[2]
DEFAULT=Path('/Volumes/PhD/river_hierarchy/network_variants/hydraulic_refresh/shared-head-anchored-depth-signed-inflow-20261003')

def audit(root, destination):
    destination.mkdir(parents=True,exist_ok=True);rows=[];bases=[];alias_counts=[]
    for directory in sorted(root.iterdir()):
        if not directory.is_dir() or not (directory/'inventory.csv').exists():continue
        inv=pd.read_csv(directory/'inventory.csv',dtype={'example':str});ready=inv[inv.status.eq('ready')];unique=ready.drop_duplicates('fingerprint');state_lookup={}
        for rec in unique.itertuples():
            run=directory/rec.fingerprint;completed=json.loads((run/'completed.json').read_text());dt=completed['routing_dt_s'];assert completed['status']=='routed'
            f=pd.read_parquet(rec.frame);K=f.rapid_k.to_numpy();n=f['count'].to_numpy(dtype=int);a=(dt-.2*K)/(dt+1.8*K);c=(1.8*K-dt)/(dt+1.8*K);neg=a<0
            roots=set(f.id_us_node)-set(f.id_ds_node);external=f.id_us_node.isin(roots).to_numpy();internal=n-external.astype(int)
            effective=neg&(internal>0)
            qa=json.loads((run/'run/Qout.audit.json').read_text())
            row=dict(example=directory.name,state=rec.state,family=rec.family,fingerprint=rec.fingerprint,is_base=rec.state=='S000_base',dt_s=dt,n_links=len(f),n_segments=int(n.sum()),nominal_negative_links=int(neg.sum()),nominal_negative_segments=int(n[neg].sum()),negative_links=int(effective.sum()),negative_internal_segments=int(internal[neg].sum()),minimum_a=float(a.min()),minimum_c=float(c.min()),maximum_c=float(c.max()),minimum_substep_flow_cms=qa['minimum_substep_flow_cms'],maximum_substep_flow_cms=qa['maximum_substep_flow_cms'],negative_reported_flow_count=qa['negative_reported_flow_count'])
            rows.append(row);state_lookup[rec.fingerprint]=row
            if row['is_base']:
                link=f[['id_link','id_us_node','id_ds_node','count','rapid_k','rapid_celerity_mps','slope']].copy();link.insert(0,'example',directory.name);link['dt_s']=dt;link['a']=a;link['c']=c;link['negative_coefficient_applied']=effective;link['negative_internal_segments']=np.where(neg,internal,0);bases.append(link)
        alias_counts.append(dict(example=directory.name,ready_inventory_entries=len(ready),entries_with_negative_coefficients=sum(state_lookup[fp]['negative_links']>0 for fp in ready.fingerprint)))
    states=pd.DataFrame(rows);base=states[states.is_base].copy();states.to_csv(destination/'states.csv',index=False);base.to_csv(destination/'base_networks.csv',index=False);pd.concat(bases,ignore_index=True).to_csv(destination/'base_links.csv',index=False);pd.DataFrame(alias_counts).to_csv(destination/'inventory_alias_counts.csv',index=False)
    summary=dict(unique_states=len(states),unique_states_with_negative_coefficients=int(states.negative_links.gt(0).sum()),examples=int(states.example.nunique()),examples_with_negative_coefficients=int(states[states.negative_links.gt(0)].example.nunique()),base_links=int(base.n_links.sum()),base_links_with_negative_coefficients=int(base.negative_links.sum()),base_segments=int(base.n_segments.sum()),base_negative_internal_segments=int(base.negative_internal_segments.sum()),all_state_link_occurrences=int(states.n_links.sum()),negative_state_link_occurrences=int(states.negative_links.sum()),all_state_segment_occurrences=int(states.n_segments.sum()),negative_internal_segment_occurrences=int(states.negative_internal_segments.sum()),minimum_current_inflow_coefficient=float(states.minimum_a.min()),minimum_substep_flow_cms=float(states.minimum_substep_flow_cms.min()),states_with_negative_native_flow=int(states.minimum_substep_flow_cms.lt(0).sum()),negative_reported_flow_count=int(states.negative_reported_flow_count.sum()),all_previous_outflow_coefficients_in_bounds=bool((states.minimum_c>=0).all() and (states.maximum_c<1).all()),root_external_input_note='Exclude first source segments: held external forcing is multiplied by a+b, not a separately.',source=str(root))
    (destination/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2));return states

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=DEFAULT);p.add_argument('--output',type=Path,default=ROOT/'RAPID/validation/hydraulic_policy_20261003/completed_coefficients_20261005');args=p.parse_args();audit(args.root,args.output)
