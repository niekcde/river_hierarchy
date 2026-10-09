"""Recover true island ports and independently validate new isolated assemblies."""
from pathlib import Path
import sys,json
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.features import geometry_mask
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'RAPID/src'))
from rapid_tools import collapse_creation_review as review
from rapid_tools.anchor_repair_trial import repair,topology
from rapid_tools.reviewed_geometry_edits import restore_short_parent_connector
from network_variants.boundary_contract_repair import orient,propose_repairs,paired_exterior_cuts,validate_result
from network_variants.shared_head_reconstruction import match_child_nodes
from shapely.geometry import Point
from shapely import normalize
OUT=ROOT/'RAPID/outputs/island_assembly_fix_20261008'

def run(state):
 d=OUT/state;meta=json.loads((d/'assembly_result.json').read_text());co=json.loads((d/'actual_island_contract.json').read_text());row=review.case_row('41',state)
 p=review.vectors(row.base_links_path,'links');pn=review.vectors(row.base_nodes_path,'nodes');p=orient(p,pn);nodes=gpd.read_file(meta['output_paths']['merged_nodes']);before=orient(gpd.read_file(meta['output_paths']['merged_links']),nodes)
 mp,_,_=match_child_nodes(pn,nodes,p.assign(actual_length_m=p.length),before.assign(link_length_m=before.length))
 # Geometric proximity supplies candidate claims only; exterior topology must corroborate them.
 claims=[]
 for n in co['boundary_nodes']:
  pt=pn[pn.id_node.eq(n)].geometry.iloc[0]
  for i,dist in nodes.distance(pt).items():
   if dist<=17.9030615:claims.append(dict(child_id_node=int(nodes.loc[i].id_node),matched_parent_node_id=n,match_within_tolerance=True))
 proposal=propose_repairs(p,pn,before,nodes,mp,co,pd.DataFrame(claims),geometry_tolerance_m=17.9030615)
 pp=pn.copy()
 for n,c in proposal['cuts'].items():pp.loc[pp.id_node.eq(n),'geometry']=Point(c['x'],c['y'])
 after,an,_=repair(before,nodes,pp,mp,co['boundary_nodes'],cuts=proposal['cuts'],correspondences=proposal['matches'],cut_link_ids={n:c['link_id'] for n,c in proposal['cuts'].items()});mapping=dict(zip(an.id_node,an.parent_identity))
 pairs=paired_exterior_cuts(p,pn,after,an,mapping,co,geometry_tolerance_m=17.9030615)
 for pair in pairs:
  pp=pn.copy();cuts=pair['cuts']
  for n,c in cuts.items():pp.loc[pp.id_node.eq(n),'geometry']=Point(c['x'],c['y'])
  oldsource=after.set_index('id_link').source_link.to_dict();after,an,_=repair(after,an,pp,mapping,co['boundary_nodes'],cuts=cuts,cut_link_ids={n:c['link_id'] for n,c in cuts.items()});after['source_link']=after.source_link.map(oldsource);mapping=dict(zip(an.id_node,an.parent_identity));proposal['cuts'].update(cuts)
 qa=validate_result(p,before,after,mapping,co,geometry_tolerance_m=17.9030615)
 exterior_restorations=[];pre_restore=after.copy();allowed_changes=set()
 for item in qa['exterior']:
  if item['geometry_exact'] or not item['path_unique'] or len(item['child_links'])!=1:continue
  lid=item['child_links'][0];pr=p[p.id_link.eq(item['parent_link'])].iloc[0];cr=after[after.id_link.eq(lid)].iloc[0]
  pp=pn.set_index('id_node').geometry;np_=an.set_index('id_node').geometry
  # Restore a subpixel displaced downstream interface only when its upstream
  # endpoint is exact and the complete exterior reach is uniquely identified.
  assert np_.loc[cr.id_us_node].distance(pp.loc[pr.id_us_node])<1e-5
  assert np_.loc[cr.id_ds_node].distance(pp.loc[pr.id_ds_node])<=4.475766
  raw=int(cr.id_ds_node);allowed_changes.update(after.loc[after.id_us_node.eq(raw)|after.id_ds_node.eq(raw),'id_link'].astype(int))
  after,an,shift=restore_short_parent_connector(after,an,p,pn,raw,int(pr.id_ds_node),int(lid),int(pr.id_link))
  exterior_restorations.append(dict(parent_link=int(pr.id_link),child_link=int(lid),parent_node=int(pr.id_ds_node),raw_node=raw,node_shift_m=shift))
 if exterior_restorations:
  prior=pre_restore.set_index('id_link');changed={int(r.id_link) for r in after.itertuples() if not r.geometry.equals_exact(prior.loc[r.id_link].geometry,1e-8)}
  assert changed<=allowed_changes
  qa=validate_result(p,before,after,mapping,co,geometry_tolerance_m=17.9030615)
  qa['intentional_exterior_restoration']=exterior_restorations;qa['geometry_changes_confined_to_restored_boundary_incidence']=True
 geom=topology(after);ptop=topology(p)
 explicit={int(c):int(n) for c,n in mapping.items() if n in co['boundary_nodes']}
 regen,_,_=match_child_nodes(pn,an,p.assign(actual_length_m=p.length),after.assign(link_length_m=after.length),validated_correspondences=explicit)
 assert all(regen[c]==n for c,n in explicit.items())
 duplicate_count=int(pd.Series([normalize(g).wkb for g in after.geometry]).duplicated().sum())
 vm=json.loads((d/'fresh_raw/local_patch_variant_manifest.json').read_text());island=review.catalog('41').set_index('island_id').loc[state].geometry
 with rasterio.open(vm['output_paths']['local_base_mask']) as src:
  a=src.read(1);tr=src.transform
  with rasterio.open(vm['output_paths']['local_collapsed_mask']) as ds:b=ds.read(1);assert ds.transform==tr and b.shape==a.shape
 target=geometry_mask([island],a.shape,tr,invert=True,all_touched=True);added=(a==0)&(b>0);target_dry=target&(a==0)
 raster_qa=dict(added_water_pixels=int(added.sum()),added_water_outside_target=int((added&~target).sum()),target_dry_pixels=int(target_dry.sum()),target_dry_pixels_filled=int((target_dry&(b>0)).sum()),removed_water_pixels=int(((a>0)&(b==0)).sum()))
 raster_ok=raster_qa['added_water_outside_target']==0 and raster_qa['target_dry_pixels']==raster_qa['target_dry_pixels_filled'] and raster_qa['removed_water_pixels']==0
 accepted=bool(qa['anchor_pass'] and qa['topology_preserved'] and (qa['geometry_preserved'] or bool(exterior_restorations)) and qa['exterior_exact'] and geom['dag'] and geom['components']==1 and geom['inlets']==ptop['inlets'] and geom['outlets']==ptop['outlets'] and ptop['cycle_rank']-geom['cycle_rank']==1 and duplicate_count==0 and raster_ok)
 result=dict(state=state,geometry_accepted=accepted,exterior_restorations=exterior_restorations,topology=geom,cycle_reduction=ptop['cycle_rank']-geom['cycle_rank'],boundary_nodes=co['boundary_nodes'],boundary_checks=qa['boundaries'],exterior_exact=qa['exterior_exact'],exterior_comparison_failures=[x['parent_link'] for x in qa['exterior'] if not x['path_unique']],exterior_nonexact=[x['parent_link'] for x in qa['exterior'] if not x['geometry_exact']],added_boundary_cuts=len(proposal['cuts']),restored_boundary_identities=len(proposal['matches']),duplicate_geometry_count=duplicate_count,raster_check=raster_qa,matching_preserves_boundary_ids=True,production_changed=False,routing_ready=False,hydraulic_attributes='Not regenerated; these are geometry review products')
 (d/'validation.json').write_text(json.dumps(qa,indent=2));(d/'review_result.json').write_text(json.dumps(result,indent=2));(d/'validated_node_correspondences.json').write_text(json.dumps(explicit,indent=2));(d/'boundary_repairs.json').write_text(json.dumps(proposal,indent=2))
 after.to_parquet(d/'fixed_links.parquet',index=False);an.to_parquet(d/'fixed_nodes.parquet',index=False);after.to_file(d/'fixed_links.gpkg',driver='GPKG');an.to_file(d/'fixed_nodes.gpkg',driver='GPKG')
 print(json.dumps(result),flush=True);return result

if __name__=='__main__':
 import argparse
 parser=argparse.ArgumentParser();parser.add_argument('--states',nargs='*',default=['island_41_005','island_41_055','island_41_075','island_41_138']);args=parser.parse_args()
 for state in args.states:run(state)
