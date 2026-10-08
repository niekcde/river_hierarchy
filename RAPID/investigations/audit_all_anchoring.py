"""Strict declared-boundary/exterior audit of completed hierarchy inputs.

Read-only sources; does not silently expand a failed domain to surviving ports.
All completed transitions currently have S000_base parents, checked explicitly.
"""
from pathlib import Path
import sys,json,re,time
from collections import defaultdict
import numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'RAPID/outputs/anchoring_audit_20261008'
EXAMPLES=['02','03','05','13','15','21','22','37','41','45','46','48','54','55','56']
def ids(x):
 if pd.isna(x) or not str(x).strip():return []
 return [int(float(v)) for v in re.split('[,; ]+',str(x).strip('[]() ')) if v]
def audit():
 OUT.mkdir(parents=True,exist_ok=True);rows=[];details=[];sources=[]
 old=pd.read_csv(ROOT/'RAPID/outputs/island_reporting_audit_20261008/island_anchor_scan.csv',dtype={'example':str}).set_index(['example','state'])
 for ex in EXAMPLES:
  run=ROOT/'RAPID/outputs/cunge_full_hierarchy_pilot/02' if ex=='02' else Path('/Volumes/PhD/river_hierarchy/cunge_full_hierarchy_runs')/ex
  reg=pd.read_csv(run/'state_run_registry.csv');br=reg[reg.state.eq('S000_base')].iloc[0];base=pd.read_parquet(br.frame).set_index('id_link',drop=False);bs=Path(br.manifest).parent.parent;membership=pd.read_csv(bs/'variant/summary/unit_link_membership.csv');units=pd.read_csv(bs/'hierarchy/unit_summary.csv').set_index('unit_id');registries={};frames={};manifests={}
  for rr in reg[~reg.state.eq('S000_base')].itertuples():
   d=dict(example=ex,family=rr.family,state=rr.state,fingerprint=rr.fingerprint,parent_fingerprint=br.fingerprint,technical_alias=str(rr.state).startswith('.'),error='',source_manifest=rr.manifest)
   try:
    if rr.parent_state!='S000_base':raise ValueError('Non-base parent requires explicit parent unit/ID mapping')
    if rr.family=='island':
     ar=old.loc[(ex,rr.state)];selected=[int(ar.anchor_unit_id)];d['island_target_domain_mismatch']=bool(ar.coverage_fraction<.99)
    else:
     experiment=Path(rr.manifest).parent.parent.parents[1];key=str(experiment)
     if key not in registries:registries[key]=pd.read_csv(experiment/'state_registry.csv').set_index('state_id')
     canonical=re.sub(r'^\.(.*?)\.promotion-.*$',r'\1',rr.state);sr=registries[key].loc[canonical];selected=ids(sr.selected_unit_ids);d['island_target_domain_mismatch']=False
    selected_links=set(membership.loc[membership.unit_id.isin(selected),'id_link'].astype(int));sel=base.loc[base.index.isin(selected_links)];outside=base.loc[~base.index.isin(selected_links)]
    if not selected_links or len(sel)!=len(selected_links):raise ValueError('Selected parent links unresolved')
    sn=set(sel.u)|set(sel.v);external_incident=set(outside.u)|set(outside.v);boundary=(sn&external_incident)|(set(sel.u)-set(sel.v))|(set(sel.v)-set(sel.u));entry=(set(sel.u)-set(sel.v))|(sn&set(outside.v));exit=(set(sel.v)-set(sel.u))|(sn&set(outside.u))
    if rr.fingerprint not in frames:frames[rr.fingerprint]=pd.read_parquet(rr.frame)
    f=frames[rr.fingerprint];nodes=set(f.u)|set(f.v);missing=sorted(boundary-nodes)
    lookup=defaultdict(list)
    for cr in f.itertuples():lookup[int(cr.u),int(cr.v)].append(cr)
    missing_links=[];changed=[];maxwidth=0.;maxslope=0.;ambiguous=[]
    for pr in outside.itertuples():
     cand=[cr for cr in lookup[int(pr.u),int(pr.v)] if np.isclose(cr.link_length_m,pr.link_length_m,rtol=1e-8,atol=1e-5)]
     if not cand:missing_links.append(int(pr.id_link));continue
     if len(cand)>1:ambiguous.append(int(pr.id_link));continue
     cr=cand[0];fields=[k for k in ['width','slope','depth_m','head_us','head_ds'] if not np.isclose(getattr(cr,k),getattr(pr,k),rtol=1e-8,atol=1e-8)]
     maxwidth=max(maxwidth,abs(cr.width/pr.width-1));maxslope=max(maxslope,abs(cr.slope/pr.slope-1))
     if fields:changed.append(int(pr.id_link));details.append(dict(example=ex,family=rr.family,state=rr.state,fingerprint=rr.fingerprint,parent_link_id=int(pr.id_link),child_link_id=int(cr.id_link),kind='exterior_property_changed',changed_fields=','.join(fields),parent_width=pr.width,child_width=cr.width,parent_slope=pr.slope,child_slope=cr.slope))
    if rr.manifest not in manifests:manifests[rr.manifest]=json.loads(Path(rr.manifest).read_text())
    cm=manifests[rr.manifest];vp=Path(cm['input_paths']['directed_links']).parents[1]/'summary/variant_manifest.json';vm=json.loads(vp.read_text()) if vp.exists() else {};backend=vm.get('execution_mode_used',vm.get('execution_mode','unknown'))
    d.update(selected_unit_ids=','.join(map(str,selected)),selected_parent_link_ids=','.join(map(str,sorted(selected_links))),required_boundary_nodes=','.join(map(str,sorted(boundary))),entry_nodes=','.join(map(str,sorted(entry))),exit_nodes=','.join(map(str,sorted(exit))),declared_domain_closed=len(entry)==1 and len(exit)==1,missing_boundary_nodes=','.join(map(str,missing)),missing_boundary_count=len(missing),outside_link_count=len(outside),outside_links_not_retained_count=len(missing_links),outside_links_not_retained=','.join(map(str,missing_links)),outside_property_changed_count=len(changed),outside_property_changed_ids=','.join(map(str,changed)),outside_ambiguous_count=len(ambiguous),outside_max_width_relative_change=maxwidth,outside_max_slope_relative_change=maxslope,execution_backend=backend,fallback_reason=vm.get('fallback_reason',''),child_links_path=cm['input_paths']['directed_links'],child_nodes_path=cm['input_paths']['directed_nodes'],base_links_path=json.loads(Path(br.manifest).read_text())['input_paths']['directed_links'],base_nodes_path=json.loads(Path(br.manifest).read_text())['input_paths']['directed_nodes'],variant_manifest=str(vp),frame_path=rr.frame)
    d['strict_anchor_and_exterior_pass']=not(missing or missing_links or changed or ambiguous or d['island_target_domain_mismatch'])
   except Exception as e:d['error']=str(e);d['strict_anchor_and_exterior_pass']=False
   rows.append(d)
  pd.DataFrame(rows).to_csv(OUT/'transition_audit.csv',index=False);pd.DataFrame(details).to_csv(OUT/'exterior_property_changes.csv',index=False);print(ex,'rows',len(rows),'missing anchors',sum(z.get('missing_boundary_count',0)>0 for z in rows if z['example']==ex),'errors',sum(bool(z['error']) for z in rows if z['example']==ex),flush=True)
 a=pd.DataFrame(rows);print(a.groupby('example').agg(entries=('state','size'),missing_anchors=('missing_boundary_count',lambda v:(v>0).sum()),outside_not_retained=('outside_links_not_retained_count',lambda v:(v>0).sum()),outside_property_changes=('outside_property_changed_count',lambda v:(v>0).sum()),strict_pass=('strict_anchor_and_exterior_pass','sum')).to_string())
if __name__=='__main__':audit()
