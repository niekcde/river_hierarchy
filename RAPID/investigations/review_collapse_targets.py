"""Read-only target-island census of completed unit and island interventions.

Every frozen island has a dry-pixel witness tested in every output mask: a fully
filled island cannot escape this screen. Full pixel coverage is then verified.
Partial edits are checked on witnessed islands and all islands intersecting the
selected parent bounds. Partial edits elsewhere / bank edits are not certified.
No production exclusions or deletions are performed.
"""
from pathlib import Path
import sys,json
import xml.etree.ElementTree as ET
from affine import Affine
from functools import lru_cache
import numpy as np,pandas as pd,geopandas as gpd,rasterio
from rasterio.features import geometry_mask
from rasterio.windows import from_bounds
from shapely.ops import polygonize,unary_union
from shapely.geometry import Point
ROOT=Path(__file__).resolve().parents[2];AUDIT=ROOT/'RAPID/outputs/anchoring_audit_20261008';OUT=ROOT/'RAPID/outputs/collapse_creation_review_20261008';SOURCE=Path('/Volumes/PhD/river_hierarchy/network_variants/output')
def ids(x):return [] if pd.isna(x) else [int(v) for v in str(x).split(',') if v]
def catalog_path(ex):return SOURCE/(f'{ex}_island_direction130__anchored' if ex=='15' else f'{ex}_island__anchored')/'setup/island_candidate_geometries.gpkg'
def target_group_key(example,parent,targets):
 return (str(example),str(parent),tuple(sorted(targets))) if len(targets)==1 else None

def run(examples=None):
 OUT.mkdir(parents=True,exist_ok=True);a=pd.read_csv(AUDIT/'review_registry.csv',dtype={'example':str});records=[];coverage=[]
 for ex,group in a.groupby('example',sort=False):
  if examples and ex not in examples:continue
  dest=OUT/ex;dest.mkdir(exist_ok=True)
  if (dest/'targets.csv').exists() and (dest/'coverage.csv').exists() and pd.read_csv(dest/'targets.csv').target_error.isna().all():
   print(ex,'using completed target census',flush=True);continue
  first=group.iloc[0];islands=gpd.read_file(catalog_path(ex)).sort_values('island_id').reset_index(drop=True);base_links=gpd.read_file(first.base_links_path).set_index('id_link');base_manifest=json.loads((Path(first.base_links_path).parents[1]/'summary/variant_manifest.json').read_text());base_path=base_manifest['output_paths']['collapsed_mask'];local=[];local_cov=[]
  with rasterio.open(base_path) as base:
   assert islands.crs==base.crs==base_links.crs
   @lru_cache(maxsize=8)
   def pixels(j):
    geom=islands.iloc[j].geometry;w=from_bounds(*geom.bounds,transform=base.transform).round_offsets().round_lengths();before=base.read(1,window=w);valid=geometry_mask([geom],before.shape,base.window_transform(w),invert=True)&(before==0);return w,valid
   witnesses=[]
   for j in range(len(islands)):
    w,valid=pixels(j);rr,cc=np.nonzero(valid)
    if not len(rr):raise ValueError(f'{ex}: island {islands.iloc[j].island_id} has no original dry pixels')
    k=len(rr)//2;witnesses.append(base.xy(int(w.row_off)+int(rr[k]),int(w.col_off)+int(cc[k])))
   cache={}
   for ix,r in enumerate(group.itertuples()):
    row=dict(example=ex,family=r.family,state=r.state,fingerprint=r.fingerprint,parent_fingerprint=r.parent_fingerprint,source_manifest=r.source_manifest,variant_manifest=r.variant_manifest,target_error='')
    try:
     vp=Path(r.variant_manifest);variant=vp.parents[1]
     if vp.exists():m=json.loads(vp.read_text())
     else:
      alt=variant.parent/'island_run_manifest.json';am=json.loads(alt.read_text());m={'output_paths':am['final_paths']}
     mask=m['output_paths']['collapsed_mask'];row['original_mask_path']=mask
     if Path(mask).suffix=='.vrt' and Path(mask).exists():
      tree=ET.parse(mask);modified=False
      for source in tree.findall('.//SimpleSource')+tree.findall('.//ComplexSource'):
       el=source.find('SourceFilename');sp=Path(el.text)
       if el.get('relativeToVRT')=='1':sp=Path(mask).parent/sp
       if not sp.exists():
        replacement=Path(mask).parent/sp.name
        if not replacement.exists():continue
        dst=source.find('DstRect');src=source.find('SrcRect')
        with rasterio.open(replacement) as patch:
         expected=base.transform*Affine.translation(float(dst.get('xOff'))-float(src.get('xOff')),float(dst.get('yOff'))-float(src.get('yOff')))
         assert patch.crs==base.crs and patch.transform.almost_equals(expected), 'Relocated VRT patch grid mismatch'
         assert patch.width==int(source.find('SourceProperties').get('RasterXSize')) and patch.height==int(source.find('SourceProperties').get('RasterYSize'))
        el.text=str(replacement);el.set('relativeToVRT','0');modified=True
       elif el.get('relativeToVRT')=='1':el.text=str(sp.resolve());el.set('relativeToVRT','0')
      if modified:
       vd=dest/'resolved_vrts';vd.mkdir(exist_ok=True);resolved=vd/(r.family+'_'+r.state+'.vrt');tree.write(resolved,encoding='utf-8',xml_declaration=True);mask=str(resolved)
     row['mask_path']=mask
     selected=base_links.loc[ids(r.selected_parent_link_ids)];faces=unary_union(list(polygonize(unary_union(selected.geometry))));bounds=selected.geometry.union_all().envelope
     nearby=set(islands.sindex.query(bounds,predicate='intersects').tolist());inferred=[]
     for j in nearby:
      geo=islands.iloc[j].geometry
      if not faces.is_empty and geo.intersection(faces).area/geo.area>=.999:inferred.append(str(islands.iloc[j].island_id))
     if mask not in cache:
      try:
       with rasterio.open(mask) as child:
        assert child.crs==base.crs and child.transform.almost_equals(base.transform) and child.shape==base.shape, 'Raster grid mismatch'
        samples=np.array([v[0] for v in child.sample(witnesses)])
        if not np.isin(samples,[0,1]).all():raise ValueError('Unexpected binary mask witness values')
        wet=set(np.where(samples>0)[0]);cache[mask]={'wet':wet,'verified':{},'evidence':'saved_collapsed_raster','edit':None}
      except (rasterio.errors.RasterioError,FileNotFoundError):
       candidates=[variant/'summary/collapse_edit_geometries.gpkg',variant/'mask/collapse_edit_geometries.gpkg']
       ep=next((p for p in candidates if p.exists()),None)
       if ep is None:raise FileNotFoundError('Neither readable collapsed raster nor retained exact pixel edits')
       edits=gpd.read_file(ep)
       exact=edits[edits.geometry_role.eq('edit')&edits.action.eq('add')] if 'geometry_role' in edits else edits.iloc[0:0]
       if exact.empty:raise ValueError('Retained geometry lacks exact edit/add polygons; fill footprints cannot certify actual targets')
       if edits.action.eq('remove').any():raise ValueError('Removal edit requires explicit reconstruction')
       assert exact.crs==base.crs
       geom=exact.geometry.union_all();wet={j for j,xy in enumerate(witnesses) if geom.covers(Point(*xy))}
       cache[mask]={'wet':wet,'verified':{},'evidence':'retained_exact_pixel_edit_polygons','edit':geom,'edit_path':str(ep)}
     cached=cache[mask];row['target_evidence']=cached['evidence'];row['retained_edit_path']=cached.get('edit_path','')
     if cached['edit'] is not None:
      nearby=nearby|set(islands.sindex.query(cached['edit'],predicate='intersects').tolist())
     to_check=(cached['wet']|nearby)-set(cached['verified'])
     if cached['edit'] is None:
      with rasterio.open(mask) as child:
       for j in sorted(to_check):
        w,valid=pixels(j);after=child.read(1,window=w);n=int(valid.sum());k=int(((after>0)&valid).sum());cached['verified'][j]=(n,k)
     else:
      for j in sorted(to_check):
       w,valid=pixels(j);added=geometry_mask([cached['edit']],valid.shape,base.window_transform(w),invert=True);n=int(valid.sum());k=int((added&valid).sum());cached['verified'][j]=(n,k)
     full=[];partial=[]
     for j,(n,k) in cached['verified'].items():
      if k==n:full.append(str(islands.iloc[j].island_id))
      elif k:partial.append(str(islands.iloc[j].island_id))
      if k:local_cov.append(dict(example=ex,family=r.family,state=r.state,island_id=str(islands.iloc[j].island_id),original_dry_pixels=n,filled_pixels=k,filled_fraction=k/n))
     full=sorted(full);partial=sorted(partial);declared=[r.state] if r.family=='island' else sorted(inferred)
     row.update(declared_target_islands=';'.join(declared),declared_target_source='island_state_identity' if r.family=='island' else 'inferred_from_selected_parent_faces_99.9pct_containment',actual_full_target_islands=';'.join(full),actual_target_count=len(full),partial_target_islands=';'.join(partial),partial_target_count=len(partial),target_set_agrees=declared==full,one_island_intervention=len(full)==1 and not partial,all_catalog_full_targets_screened=True,partial_edit_scope='witnessed islands plus selected-parent bounding-box islands; outside partial edits not certified',original_anchor_domain_wrong=bool(r.island_target_domain_mismatch),original_exact_anchor_missing=bool(r.missing_exact_anchor),original_large_anchor_flag=bool(r.anchor_displacement_over_100m),catalog_path=str(catalog_path(ex)))
    except Exception as e:row['target_error']=repr(e)
    local.append(row)
    if (ix+1)%50==0:print(ex,ix+1,'/',len(group),'target states',flush=True)
   pd.DataFrame(local).to_csv(dest/'targets.csv',index=False);pd.DataFrame(local_cov,columns=['example','family','state','island_id','original_dry_pixels','filled_pixels','filled_fraction']).to_csv(dest/'coverage.csv',index=False)
   print(ex,'complete',len(local),'errors',sum(bool(z['target_error']) for z in local),flush=True)
 finalize()
def finalize():
 files=sorted(OUT.glob('[0-9][0-9]/targets.csv'));a=pd.concat([pd.read_csv(p,dtype={'example':str}) for p in files],ignore_index=True);c=pd.concat([pd.read_csv(p.parent/'coverage.csv',dtype={'example':str}) for p in files],ignore_index=True);pairs=[]
 if 'target_evidence' in a:a.loc[a.target_error.isna(),'target_evidence']=a.loc[a.target_error.isna(),'target_evidence'].fillna('saved_collapsed_raster')
 original_actions=pd.read_csv(AUDIT/'island_reuse_and_domain_counts.csv',dtype={'example':str}).set_index(['example','state']).execution_action.to_dict()
 # Target equivalence is a flag for shared intervention, not permission to retain a bad output.
 for (ex,parent,target),g in a[a.target_error.isna()&a.actual_target_count.eq(1)].groupby(['example','parent_fingerprint','actual_full_target_islands']):
  units=g[g.family.eq('indep')];islands=g[g.family.eq('island')]
  for ir in islands.itertuples():
   for ur in units.itertuples():
    pairs.append(dict(example=ex,island_state=ir.state,independent_state=ur.state,target_island=target,parent_fingerprint=parent,island_fingerprint=ir.fingerprint,independent_fingerprint=ur.fingerprint,same_output=ir.fingerprint==ur.fingerprint,island_separately_generated=original_actions[(ex,ir.state)]=='run_island_state',already_reused=Path(ir.source_manifest).resolve()==Path(ur.source_manifest).resolve(),island_source_manifest=ir.source_manifest,independent_source_manifest=ur.source_manifest,island_partial_count=ir.partial_target_count,independent_partial_count=ur.partial_target_count,island_domain_wrong=ir.original_anchor_domain_wrong,unit_exact_anchor_missing=ur.original_exact_anchor_missing,unit_large_anchor_flag=ur.original_large_anchor_flag,action='already_reused' if Path(ir.source_manifest).resolve()==Path(ur.source_manifest).resolve() else 'duplicate_target_flag_choose_valid_canonical_after_creation_review',full_mask_equivalence_verified=False))
 dup=pd.DataFrame(pairs);a.to_csv(OUT/'target_inventory.csv',index=False);c.to_csv(OUT/'target_pixel_coverage.csv',index=False);dup.to_csv(OUT/'island_independent_target_pairs.csv',index=False)
 policy_path=OUT/'review_policy.json'
 policy=json.loads(policy_path.read_text()) if policy_path.exists() else {}
 retain_indep=policy.get('canonical_duplicate_family')=='indep'
 if retain_indep:
  dup.loc[dup.island_separately_generated,'action']='retain_independent_canonical_repair_anchoring_separately'
  dup.to_csv(OUT/'island_independent_target_pairs.csv',index=False)
 geometry_scan=pd.read_csv(ROOT/'RAPID/outputs/island_reporting_audit_20261008/island_geometry_scan.csv',dtype={'example':str}).set_index(['example','state']).pure_geometry_subset.to_dict()
 status=[]
 for ir in a[a.family.eq('island')].itertuples():
  ps=dup[(dup.example==ir.example)&(dup.island_state==ir.state)] if len(dup) else dup
  if pd.notna(ir.target_error):decision='target_audit_error'
  elif ir.actual_target_count!=1 or ir.actual_full_target_islands!=ir.state or ir.partial_target_count:decision='incorrect_or_ambiguous_island_edit'
  elif retain_indep and len(ps) and original_actions[(ir.example,ir.state)]=='run_island_state':decision='retain_independent_canonical_repair_anchoring_separately'
  elif ir.original_anchor_domain_wrong:decision='wrong_processing_domain_review_creation'
  elif original_actions[(ir.example,ir.state)]=='alias_existing_unit_state':decision='already_reused_review_source_unit'
  elif len(ps):decision='duplicate_target_flag_review_canonical_output'
  else:decision='distinct_island_intervention_review_creation'
  manual_path=catalog_path(ir.example).parent/'manual_island_review_adjustment_41_099_100.json'
  if manual_path.exists():
   manual=json.loads(manual_path.read_text())
   if ir.state==manual.get('retained_island_id') and set(str(ir.actual_full_target_islands).split(';'))=={manual.get('retained_island_id'),manual.get('absorbed_component_island_id')}:
    decision=('approved_manual_composite_retain' if ir.state in policy.get('approved_manual_composites',[]) else 'documented_manual_composite_review_single_island_rule')
  status.append(dict(example=ir.example,state=ir.state,status=decision,known_pure_branch_deletion=bool(geometry_scan.get((ir.example,ir.state),False)),wrong_processing_domain=bool(ir.original_anchor_domain_wrong),exact_anchor_identity_missing=bool(ir.original_exact_anchor_missing),large_anchor_displacement=bool(ir.original_large_anchor_flag),canonical_family='indep' if retain_indep and len(ps) else 'island',anchoring_repair_separate=bool(retain_indep and len(ps)),matching_independent_states=';'.join(ps.independent_state) if len(ps) else '',actual_full_target_islands=ir.actual_full_target_islands,partial_target_islands=ir.partial_target_islands))
 pd.DataFrame(status).to_csv(OUT/'island_review_queue.csv',index=False)
 print('Census rows',len(a),'errors',a.target_error.notna().sum(),flush=True)
if __name__=='__main__':run(sys.argv[1:] or None)
