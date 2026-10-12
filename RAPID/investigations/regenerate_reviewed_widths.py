"""Geometry-scoped width regeneration using the existing production QC model."""
from pathlib import Path
import json
from functools import lru_cache
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from shapely.ops import unary_union
from regenerate_reviewed_hydraulics import ROOT,save
from network_variants.width_trim_selection import apply_authoritative_width_trim_selection
from network_variants.routing_width_application import apply_routing_width_model
from network_variants.width_samples import resolve_width_samples_path
from network_variants.variant_generation import _normalize_width_boolean_fields

@lru_cache(maxsize=12)
def read_links(path):return gpd.read_file(path)

@lru_cache(maxsize=16)
def parent(ex,path,frame):
 return read_links(path),pd.read_parquet(frame).set_index('id_link')

def samples(path,ids):
 p=resolve_width_samples_path(Path(path).parent)
 if p.suffix=='.parquet':return pd.read_parquet(p,filters=[('id_link','in',list(map(int,ids)))])
 chunks=[v[v.id_link.isin(ids)] for v in pd.read_csv(p,chunksize=100000,low_memory=False)]
 return pd.concat(chunks,ignore_index=True)

def sampled_summary(row,group):
 row=dict(row);group=group.sort_values('_position').drop_duplicates(['x','y']).reset_index(drop=True).copy()
 group['id_link']=int(row['id_link']);group['sample_index']=range(len(group))
 for family in ['wet','total','dry']:
  vals=pd.to_numeric(group['width_'+family],errors='coerce');valid=vals[np.isfinite(vals)&(vals.ge(0) if family=='dry' else vals.gt(0))]
  if valid.empty:raise ValueError(f'No positive {family} measurements for link {row["id_link"]}')
  for stat,value in [('adj',valid.mean()),('med',valid.median()),('p05',valid.quantile(.05)),('p50',valid.quantile(.5)),('p95',valid.quantile(.95))]:row[f'wid_{stat}_{family}']=float(value)
  row['wid_pix_'+family]=','.join(map(str,vals))
 row['n_invalid_width_samples']=int((~np.isfinite(group.width_wet)|group.width_wet.le(0)).sum())
 row['wid_pix']=','.join(map(str,group.get('dt_width_global',group.width_total)))
 row['width_trim_method']='';row['width_method']='resampled_source_profile_fragment'
 return row,group

def new_island_widths(geo,nodes,record,changed,folder):
 from network_variants.local_patch_products import _attach_changed_link_pixels,_write_local_state_bank_labels
 from network_variants.variant_generation import compute_width_families
 src=ROOT/'RAPID/outputs/island_assembly_fix_20261008'/record['state']
 paths=json.loads((src/'fresh_raw/local_patch_variant_manifest.json').read_text())['output_paths']
 cropped=geo[geo.id_link.isin(changed)][['id_link','id_us_node','id_ds_node','geometry']].copy()
 cropped,_=_attach_changed_link_pixels(cropped,changed_link_ids=set(changed),collapsed_mask_path=paths['local_collapsed_mask'])
 labels,_=_write_local_state_bank_labels(base_labels_path=paths['boundary_component_labels'],origin_mask_path=paths['local_origin_mask'],output_path=folder/'state_bank_labels.tif')
 result,ss=compute_width_families(cropped,collapsed_mask_path=paths['local_collapsed_mask'],wet_reference_mask_path=paths['local_base_mask'],origin_mask_path=paths['local_origin_mask'],rivgraph_distance_path=paths.get('raw_rivgraph_distance'),bank_labels_path=labels)
 # The expensive measurement remains a reusable intermediate on failure.
 result.to_parquet(folder/'fresh_measured_widths.parquet',index=False);ss.to_parquet(folder/'fresh_width_samples.parquet',index=False)
 return result,ss

def widths(r,release,folder):
 geo,nodes,contract,_,_=release.load(r['example'],r['family'],r['state'])
 old=read_links(r['links']).set_index('id_link',drop=False);oldf=pd.read_parquet(r['original_frame']).set_index('id_link')
 pr=release.lookup(r['example'],'indep','S000_base');par,pf=parent(r['example'],pr['links'],pr['original_frame']);par=par.set_index('id_link',drop=False)
 # Geometry, not numeric ID, establishes exact inheritance.
 parent_geoms={g.normalize().wkb:int(i) for i,g in par.geometry.items()}
 old_geoms={g.normalize().wkb:int(i) for i,g in old.geometry.items()}
 rows=[];relations={};changed=[]
 for link in geo.itertuples():
  lid=int(link.id_link);key=link.geometry.normalize().wkb
  if key in parent_geoms:
   source=parent_geoms[key];row=par.loc[source].to_dict();row['wid_adj_routing']=float(pf.loc[source].width);status='exact_parent_geometry';kind='parent';sources=[source]
  elif key in old_geoms:
   source=old_geoms[key];row=old.loc[source].to_dict();row['wid_adj_routing']=float(oldf.loc[source].width);status='unchanged_child_geometry';kind='child';sources=[source]
  else:
   value=getattr(link,'source_links',None)
   sources=json.loads(value) if isinstance(value,str) and value.startswith('[') else [int(link.source_link)]
   sources=[int(s) for s in sources if s in old.index]
   row=old.loc[sources[0]].to_dict() if sources else {}
   status='changed_geometry_requires_measurement';kind='changed';changed.append(lid)
  row['id_nodes']=f'{int(link.id_us_node)},{int(link.id_ds_node)}'
  row.update(id_link=lid,id_us_node=int(link.id_us_node),id_ds_node=int(link.id_ds_node),u=int(link.u),v=int(link.v),geometry=link.geometry,len=float(link.geometry.length),link_length_m=float(link.geometry.length),width_regeneration_method=status)
  # Membership of unaffected links remains context. Recompute membership for
  # genuinely new island assemblies; splits preserve the source memberships.
  if r['decision']=='island_reassembled':row.pop('n_units',None)
  rows.append(row);relations[lid]=(kind,sources)
 enriched=gpd.GeoDataFrame(rows,geometry='geometry',crs=geo.crs)
 measured_samples=[];needs=set(changed)
 fresh=r['decision']=='island_reassembled'
 if fresh and changed:
  if (folder/'fresh_measured_widths.parquet').exists():new=gpd.read_parquet(folder/'fresh_measured_widths.parquet');ss=pd.read_parquet(folder/'fresh_width_samples.parquet')
  else:new,ss=new_island_widths(geo,nodes,r,changed,folder)
  for row in new.to_dict('records'):
   ix=enriched.index[enriched.id_link.eq(row['id_link'])][0]
   for k,v in row.items():
    if k!='geometry':enriched.loc[ix,k]=v
   enriched.loc[ix,'width_regeneration_method']='fresh_two_bank_measurement'
  measured_samples.append(ss)
 elif changed:
  source_ids={s for lid in changed for s in relations[lid][1]};ss=samples(r['links'],source_ids)
  assigned={lid:[] for lid in changed}
  # Assign each source sample uniquely to a resulting fragment. This avoids
  # double counting boundary pixels on two adjacent fragments.
  for source in source_ids:
   group=ss[ss.id_link.eq(source)]
   pieces=geo[geo.id_link.isin([lid for lid in changed if source in relations[lid][1]])]
   # Exclude samples belonging to exact inherited fragments of the same source.
   inherited=geo[geo.source_link.eq(source)&~geo.id_link.isin(changed)]
   for sample in group.to_dict('records'):
    pt=Point(sample['x'],sample['y']);d=pieces.geometry.distance(pt);idx=d.idxmin();line=pieces.loc[idx].geometry
    if len(inherited) and float(inherited.geometry.distance(pt).min())<float(d.loc[idx])-1e-7:continue
    # Projection must land on the link, not just its endpoint: samples from
    # discarded exterior portions are not allowed to influence a fragment.
    position=line.project(pt)
    if pt.distance(line)>max(1.,np.sqrt(2)*float(contract.get('raster_step_estimate_m',10))):continue
    sample['_position']=position;assigned[int(pieces.loc[idx].id_link)].append(sample)
  for lid,values in assigned.items():
   if not values:raise ValueError(f'No source-profile support for changed link {lid}')
   ix=enriched.index[enriched.id_link.eq(lid)][0]
   row,group=sampled_summary(enriched.loc[ix],pd.DataFrame(values))
   for k,v in row.items():
    if k!='geometry':enriched.loc[ix,k]=v
   enriched.loc[ix,'width_regeneration_method']='source_profile_retrimmed'
   measured_samples.append(group)
 if changed:
  measured=pd.concat(measured_samples,ignore_index=True)
  subset=enriched[enriched.id_link.isin(changed)].copy()
  selected,marked,_=apply_authoritative_width_trim_selection(subset,measured)
  for row in selected.to_dict('records'):
   ix=enriched.index[enriched.id_link.eq(row['id_link'])][0]
   for k,v in row.items():
    if k!='geometry':enriched.loc[ix,k]=v
  measured_samples=[marked]
  # Endpoint measurements for every adjacent fixed link provide the exact
  # production context for scoped QC; distant links are not candidates.
  touched=set(subset.id_us_node)|set(subset.id_ds_node)
  neighbors=enriched[(enriched.id_us_node.isin(touched)|enriched.id_ds_node.isin(touched))&~enriched.id_link.isin(changed)]
  for kind,path in [('parent',pr['links']),('child',r['links'])]:
   ids={relations[int(lid)][1][0] for lid in neighbors.id_link if relations[int(lid)][0]==kind}
   if not ids:continue
   ss=samples(path,ids)
   for lid in neighbors.id_link:
    if relations[int(lid)][0]!=kind:continue
    source=relations[int(lid)][1][0];part=ss[ss.id_link.eq(source)].copy();part['id_link']=int(lid);measured_samples.append(part)
  scope_samples=pd.concat(measured_samples,ignore_index=True)
  # Ensure trim-support provenance is not accidentally NaN on inherited samples.
  for col in ['used_in_selected_width']:
   if col in scope_samples:scope_samples[col]=scope_samples[col].fillna(False).astype(bool)
  enriched=_normalize_width_boolean_fields(enriched)
  enriched,audit,summary=apply_routing_width_model(enriched,scope_samples,nodes,example_id=r['example'],variant_id=r['state'],apply_known_base_exclusions=False,evaluation_link_ids=changed)
  audit.to_csv(folder/'routing_width_audit.csv',index=False);save(folder/'routing_width_model.json',summary)
  marked.to_parquet(folder/'changed_width_samples.parquet',index=False)
 enriched=_normalize_width_boolean_fields(enriched)
 if 'width_trim_valid_core' in enriched:
  enriched['width_trim_valid_core']=enriched.width_trim_valid_core.map(lambda v:pd.NA if pd.isna(v) else str(v).lower() in ['true','1','1.0']).astype('boolean')
 result=pd.DataFrame(enriched[['id_link','id_us_node','id_ds_node','u','v','link_length_m','wid_adj_routing','wid_adj_wet','width_regeneration_method']]).rename(columns={'wid_adj_routing':'width'})
 for col in ['width_trim_method','width_trim_valid_core','width_model_prediction','width_model_excluded']:
  if col in enriched:result[col]=enriched[col]
 if not np.isfinite(result.width).all() or (result.width<=0).any():raise ValueError('Nonpositive regenerated routing widths')
 # Preserve full measured/QC fields for integration into the master GeoPackages.
 previous=folder/'widths.parquet'
 if previous.exists():
  prior=pd.read_parquet(previous).set_index('id_link')['width']
  np.testing.assert_allclose(result.set_index('id_link')['width'].sort_index(),prior.sort_index(),rtol=0,atol=0)
 enriched.to_file(folder/'widths_enriched.gpkg',driver='GPKG',layer='links',index=False)
 result.to_parquet(folder/'widths.parquet',index=False)
 # Preserve numerical fixed-width identity, not merely rounded equality.
 for row in result.itertuples():
  kind,sources=relations[row.id_link]
  if kind=='parent':assert row.width==float(pf.loc[sources[0]].width)
  if kind=='child':assert row.width==float(oldf.loc[sources[0]].width)
 invalid_core=result[result.id_link.isin(changed)&result.get('width_trim_valid_core',pd.Series(True,index=result.index)).eq(False)]
 report=dict(example=r['example'],family=r['family'],state=r['state'],geometry_id=r['geometry_id'],status='widths_regenerated',changed_width_links=changed,untrimmed_fallback_links=invalid_core.id_link.tolist(),positive_widths=True,unchanged_widths_preserved_exactly=True,routing_ready=False)
 save(folder/'width_result.json',report);return report
