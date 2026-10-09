"""Recompute affected trial widths/depths and test fixed-exterior heads."""
from pathlib import Path
import sys,json
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'RAPID/src'))
from rapid_tools import collapse_creation_review as review
from network_variants.width_trim_selection import select_width_trim
from network_variants.shared_head_reconstruction import reconstruct_shared_heads
OUT=ROOT/'RAPID/outputs/boundary_contract_repair_20261008/review_revision'
BASE=ROOT/'RAPID/assets/bounded_robust_slopes_20261002'

def derive(ex,state,folder):
 row=review.case_row(ex,state);geo=gpd.read_parquet(folder/'trial_links.parquet');old=pd.read_parquet(row.frame_path).set_index('id_link');par=pd.read_parquet(Path(row.frame_path).with_name(row.parent_fingerprint+'.parquet')).set_index('id_link');contract=json.loads((folder/'contract.json').read_text())
 exterior=json.loads((folder/'exterior_paths.json').read_text());exact={lid:e['parent_link'] for e in exterior if e['geometry_exact'] for lid in e['child_links']}
 counts=geo.groupby('source_link').size();split=set(counts[counts>1].index)
 changed=geo[geo.source_link.isin(split)|~geo.edit_kind.eq('retained_or_split')]
 sources=set(s for v in changed.source_links for s in json.loads(v));parts=[];wd=Path(row.variant_manifest).parents[1]/'widths'
 if sources:
  pp=wd/'link_width_samples.parquet';cp=wd/'link_width_samples.csv'
  if pp.exists():samples=pd.read_parquet(pp,filters=[('id_link','in',list(sources))])
  else:
   for chunk in pd.read_csv(cp,chunksize=100000):
    keep=chunk[chunk.id_link.isin(sources)]
    if len(keep):parts.append(keep)
   samples=pd.concat(parts,ignore_index=True) if parts else pd.DataFrame()
 else:samples=pd.DataFrame()
 assigned={}
 # Assign each saved source sample to exactly one resulting fragment.
 for source in sources:
  subset=samples[samples.id_link.eq(source)] if len(samples) else pd.DataFrame();pieces=geo[geo.source_links.map(lambda s:source in json.loads(s))]
  for record in subset.to_dict('records'):
   pt=Point(record['x'],record['y']);distances=pieces.geometry.distance(pt);idx=distances.idxmin();lid=int(pieces.loc[idx].id_link)
   record['_position']=pieces.loc[idx].geometry.project(pt);assigned.setdefault(lid,[]).append(record)
 records=[]
 for r in geo.itertuples():
  ss=json.loads(r.source_links);width=float(old.loc[ss[0]].width);status='saved_unsplit_width';n=0;trim=''
  parent_id=exact.get(r.id_link)
  if pd.notna(r.restored_parent_link):parent_id=int(r.restored_parent_link)
  if parent_id is not None:width=float(par.loc[parent_id].width);status='parent_exact_geometry_width'
  elif r.id_link in set(changed.id_link):
   group=pd.DataFrame(assigned.get(r.id_link,[]));width=np.nan;status='no_valid_fragment_samples'
   if len(group):
    group=group.sort_values('_position').drop_duplicates(['x','y']).reset_index(drop=True);group['sample_index']=range(len(group));choice=select_width_trim(pd.Series(dtype=object),group);trim=choice.method;n=choice.stop-choice.start
    if choice.valid_core:width=float(group.iloc[choice.start:choice.stop].width_wet.mean());status='retrimmed_saved_samples_pending_width_QC'
  records.append(dict(id_link=int(r.id_link),u=int(r.u),v=int(r.v),link_length_m=float(r.geometry.length),width=width,width_status=status,sample_count=n,trim_method=trim,source_links=r.source_links))
 f=pd.DataFrame(records);refs=pd.read_csv(BASE/'hydraulic_references.csv',dtype={'example':str}).set_index('example').loc[ex];f['depth_m']=refs.stored_depth_m*(f.width/refs.reference_width_m)**.65
 nodes=pd.read_parquet(BASE/ex/'floor_0.05/old/nodes.parquet');bl=pd.read_parquet(BASE/ex/'floor_0.05/old/links.parquet');ext=bl[~bl.id_link.isin(contract['selected_parent_links'])];fixed=(set(ext.id_us_node)|set(ext.id_ds_node)|set(contract['reviewed_required_boundary_nodes']))-set(contract['reviewed_internal_parent_nodes'])-set(contract['reviewed_provisional_parent_nodes'])
 f.to_csv(folder/'width_depth_checks.csv',index=False);result=dict(example=ex,state=state,widths_valid=bool(np.isfinite(f.width).all() and f.width.gt(0).all()),invalid_width_links=f.loc[~np.isfinite(f.width)|f.width.le(0),'id_link'].tolist(),routing_ready=False,width_QC_pending=True)
 try:
  checked,qa=reconstruct_shared_heads(f,bl,nodes,nodes.set_index('id_node')['head'],fixed_node_ids=fixed);checked.to_parquet(folder/'trial_fragment_attributes.parquet',index=False);result.update(fixed_exterior_heads_passed=True,positive_slopes=bool(checked.slope.gt(0).all()),head_qa=qa)
 except ValueError as error:result.update(fixed_exterior_heads_passed=False,head_error=str(error))
 (folder/'attribute_checks.json').write_text(json.dumps(result,indent=2));return result

def main():
 results=[]
 for path in sorted(OUT.glob('[0-9][0-9]/*/trial_links.parquet')):
  ex,state=path.parent.parent.name,path.parent.name;result=derive(ex,state,path.parent);results.append(result);print(ex,state,result['widths_valid'],result['fixed_exterior_heads_passed'],flush=True)
 (OUT/'attribute_results.json').write_text(json.dumps(results,indent=2))
if __name__=='__main__':main()
