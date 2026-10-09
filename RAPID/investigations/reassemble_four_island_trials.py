"""Reassemble frozen island masks from raw RivGraph using actual island face links."""
from pathlib import Path
from dataclasses import fields,replace
import sys,json,traceback
import pandas as pd
import networkx as nx
import geopandas as gpd
from shapely.ops import polygonize,unary_union
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'RAPID/src'))
from rapid_tools import collapse_creation_review as review
from network_variants.boundary_contract_repair import derive_contract
from network_variants.local_patch import LocalPatchContract,LocalPatchCollar,plan_local_patch
from network_variants.local_patch_merge import merge_local_patch_variant
from network_variants.local_patch_raster import LocalPatchRasterResult,LocalPatchRasterAttempt,BoundaryCrossing,MainBankCut
from network_variants.local_patch_variant import LocalPatchVariantResult,materialize_local_patch_variant
from network_variants.local_patch_raster import build_local_patch_base_raster
from network_variants.local_patch_execution import _selected_units_from_hierarchy_products
OUT=ROOT/'RAPID/outputs/island_assembly_fix_20261008'
STATES=['island_41_005','island_41_055','island_41_075','island_41_138']

def unpack(cls,data):return cls(**{f.name:data[f.name] for f in fields(cls) if f.name in data})

def run(state):
 row=review.case_row('41',state);manifest=json.loads(Path(row.variant_manifest).read_text());attempt=Path(manifest['local_patch']['attempt_dir']);p=review.vectors(row.base_links_path,'links');cat=review.catalog('41').set_index('island_id');island=cat.loc[state].geometry
 faces=[f for f in polygonize(unary_union(p.geometry)) if f.covers(island.representative_point())];assert len(faces)==1
 face=faces[0];assert face.intersection(island).area/island.area>.999
 selected=p[p.geometry.map(lambda g:face.boundary.intersection(g).length/g.length>.99)]
 spec=derive_contract(p,selected.id_link.astype(int).tolist());g=nx.DiGraph();g.add_edges_from(zip(selected.id_us_node,selected.id_ds_node));sources=[n for n in g if g.in_degree(n)==0];sinks=[n for n in g if g.out_degree(n)==0];assert len(sources)==len(sinks)==1
 source,sink=int(sources[0]),int(sinks[0]);inside=set(g.nodes);core=p[p.id_us_node.isin(inside)&p.id_ds_node.isin(inside)]
 collars=[]
 for role,n,col in [('upstream',source,'id_ds_node'),('downstream',sink,'id_us_node')]:
  candidates=p[p[col].eq(n)&~p.id_link.isin(core.id_link)];assert len(candidates)==1,(state,role,len(candidates));r=candidates.iloc[0]
  collars.append(LocalPatchCollar(role=role,base_link_id=int(r.id_link),base_upstream_node_id=int(r.id_us_node),base_downstream_node_id=int(r.id_ds_node),domain_boundary_node_id=n,outer_splice_anchor_node_id=int(r.id_us_node if role=='upstream' else r.id_ds_node)))
 contract=LocalPatchContract(example_id='41',selection_type='island',selected_unit_ids=tuple(manifest['selected_unit_ids']),domain_type='selected_group',core_link_ids=tuple(core.id_link.astype(int)),selected_link_ids=tuple(selected.id_link.astype(int)),upstream_collar=collars[0],downstream_collar=collars[1],anchored_source_node_id=source,anchored_sink_node_id=sink,anchored_upstream_boundary_node_ids=tuple(spec['entry_nodes']),anchored_downstream_boundary_node_ids=tuple(spec['exit_nodes']))
 rm=manifest['local_patch']['raster'];attempts=[]
 for a in rm['attempts']:
  a=dict(a);a['crossings']=tuple(unpack(BoundaryCrossing,z) for z in a['crossings']);a['bank_cuts']=tuple(unpack(MainBankCut,z) for z in a['bank_cuts']);attempts.append(unpack(LocalPatchRasterAttempt,a))
 raster=unpack(LocalPatchRasterResult,dict(rm,attempts=tuple(attempts)))
 vm=json.loads((attempt/'variant/local_patch_variant_manifest.json').read_text());variant=unpack(LocalPatchVariantResult,vm)
 dest=OUT/state;dest.mkdir(parents=True,exist_ok=True)
 # The old crop cuts crossed the larger island intervention. Regenerate a
 # correctly bounded crop using the smallest complete unit enclosing its face.
 base=Path(row.base_links_path).parents[2];membership=pd.read_csv(base/'variant/summary/unit_link_membership.csv');summary=pd.read_csv(base/'hierarchy/unit_summary.csv')
 groups={int(k):set(v.id_link.astype(int)) for k,v in membership.groupby('unit_id')}
 containing=sorted((len(edges),uid) for uid,edges in groups.items() if set(spec['selected_parent_links'])<=edges)
 assert containing
 context_unit=containing[0][1]
 fullp=gpd.read_file(row.base_links_path);fulln=gpd.read_file(row.base_nodes_path)
 planning=plan_local_patch(example_id='41',full_links=fullp,unit_summary=summary,unit_link_membership=membership,bubble_edge_resistance=pd.read_csv(base/'hierarchy/bubble_edge_resistance.csv'),selected_unit_ids=[context_unit],selection_type='island')
 assert planning.contract is not None,planning.fallback_reason
 (dest/'computational_planning.json').write_text(json.dumps(planning.to_manifest(),indent=2))
 fresh=dest/'fresh_raw/local_patch_variant_manifest.json'
 if fresh.exists():variant=unpack(LocalPatchVariantResult,json.loads(fresh.read_text()))
 else:
  raster=build_local_patch_base_raster(contract=planning.contract,full_links=fullp,full_nodes=fulln,cleaned_mask_path=base/'variant/mask/41__S000_base_collapsed.tif',boundary_component_labels_path=next((base.parents[1]/'boundary_components').glob('*land_water_labels_with_snapped_buffered_border_cut.tif')),output_dir=dest/'fresh_raster',reference_centerline_path=ROOT/'rivgraph_centerline/outputs/runs/41/rivgraph/41_centerline.gpkg')
  assert raster.is_safe,raster.fallback_reason
  units=_selected_units_from_hierarchy_products(selected_unit_ids=[context_unit],unit_summary=summary,unit_link_membership=membership,parent_links=fullp)
  variant=materialize_local_patch_variant(contract=planning.contract,raster_result=raster,reviewed_links_path=row.base_links_path,reviewed_nodes_path=row.base_nodes_path,exit_sides='NS',output_dir=dest/'fresh_raw',single_thread=True,collapse_geometry=island,collapse_component_id=state,collapse_source_unit_ids=tuple(manifest['explicit_collapse_geometry']['source_unit_ids']),network_analyzer=lambda *args,**kwargs:(None,list(units.values()),None))
  (dest/'fresh_raster_result.json').write_text(json.dumps(raster.to_manifest(),indent=2))
 if fresh.exists():
  rm=json.loads((dest/'fresh_raster_result.json').read_text());aa=[]
  for a in rm['attempts']:
   a=dict(a);a['crossings']=tuple(unpack(BoundaryCrossing,z) for z in a['crossings']);a['bank_cuts']=tuple(unpack(MainBankCut,z) for z in a['bank_cuts']);aa.append(unpack(LocalPatchRasterAttempt,a))
  raster=unpack(LocalPatchRasterResult,dict(rm,attempts=tuple(aa)))
 (dest/'actual_island_contract.json').write_text(json.dumps(dict(spec,assembly_contract=contract.to_manifest(),island_id=state,source_variant_manifest=row.variant_manifest,raw_variant_manifest=str(attempt/'variant/local_patch_variant_manifest.json')),indent=2))
 try:
  result=merge_local_patch_variant(contract=contract,raster_result=raster,variant_result=variant,parent_links_path=row.base_links_path,parent_nodes_path=row.base_nodes_path,output_dir=dest/'assembly',replacement_parent_link_ids=tuple(spec['selected_parent_links']),upstream_boundary_node_ids=tuple(spec['entry_nodes']),downstream_boundary_node_ids=tuple(spec['exit_nodes']),expected_cycle_reduction_override=1,include_exterior_direction_scaffold=True)
 except ValueError as error:
  (dest/'face_only_attempt_failure.txt').write_text(str(error)+'\n')
  # A full cut-to-cut crop may include surviving branches within the enclosing
  # unit. Assemble that complete computational interval, then explicitly check
  # and anchor the actual island boundary (the raster edit remains one island).
  try:
   result=merge_local_patch_variant(contract=planning.contract,raster_result=raster,variant_result=variant,parent_links_path=row.base_links_path,parent_nodes_path=row.base_nodes_path,output_dir=dest/'assembly_context',expected_cycle_reduction_override=1,include_exterior_direction_scaffold=True)
  except ValueError as context_error:
   (dest/'context_attempt_failure.txt').write_text(str(context_error)+'\n')
   from network_variants.local_patch_realized import assemble_local_patch_realized_graph
   from network_variants.local_patch_anchored import anchor_local_patch_realized_graph
   from network_variants.local_patch_merge import LocalPatchMergeResult
   realized=assemble_local_patch_realized_graph(contract=planning.contract,raster_result=raster,variant_result=variant,parent_links_path=row.base_links_path,parent_nodes_path=row.base_nodes_path,output_dir=dest/'corrected_realized',expected_cycle_reduction_override=1)
   units=_selected_units_from_hierarchy_products(selected_unit_ids=[context_unit],unit_summary=summary,unit_link_membership=membership,parent_links=fullp)
   anchored=anchor_local_patch_realized_graph(realized_result=realized,parent_links_path=row.base_links_path,parent_nodes_path=row.base_nodes_path,selected_unit_ids=[context_unit],units_by_id=units,output_dir=dest/'corrected_anchored',match_tolerance_m=5.59470671875)
   result=LocalPatchMergeResult(status='complete_corrected_realized_and_anchored',output_paths=dict(merged_links=anchored.output_paths['anchored_links'],merged_nodes=anchored.output_paths['anchored_nodes']),summary=dict(anchored.summary))


 (dest/'assembly_result.json').write_text(json.dumps(result.to_manifest(),indent=2));return result.to_manifest()

def main():
 import argparse
 parser=argparse.ArgumentParser();parser.add_argument('--states',nargs='*',default=STATES);args=parser.parse_args();OUT.mkdir(parents=True,exist_ok=True)
 for state in args.states:
  try:r=run(state);print(state,r['status'],flush=True)
  except Exception as e:
   d=OUT/state;d.mkdir(exist_ok=True);(d/'failed_attempt.txt').write_text(traceback.format_exc());print(state,type(e).__name__,str(e),flush=True)
if __name__=='__main__':main()
