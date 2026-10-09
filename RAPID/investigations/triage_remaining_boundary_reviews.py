"""Evidence-based triage; does not clear production checks or alter geometry."""
from pathlib import Path
import sys,json
import pandas as pd
import geopandas as gpd
import networkx as nx
from shapely.ops import polygonize,unary_union
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'RAPID/src'))
from rapid_tools import collapse_creation_review as review
from rapid_tools.anchor_repair_trial import node_mapping
OUT=ROOT/'RAPID/outputs/boundary_contract_repair_20261008'
DEST=OUT/'review_revision';DEST.mkdir(exist_ok=True)

def run():
 q=pd.read_csv(OUT/'remaining_visual_review_queue.csv',dtype={'example':str});records=[]
 for r in q.itertuples():
  folder=OUT/'full'/r.example/r.state;v=json.loads((folder/'validation.json').read_text());contract=json.loads((folder/'contract.json').read_text());row=review.case_row(r.example,r.state)
  if (folder/'trial_links.parquet').exists():c=gpd.read_parquet(folder/'trial_links.parquet');mp=dict(zip(gpd.read_parquet(folder/'trial_nodes.parquet').id_node,gpd.read_parquet(folder/'trial_nodes.parquet').parent_identity))
  else:c=review.vectors(row.child_links_path,'links');mp=node_mapping(pd.read_parquet(row.frame_path))
  parent=review.vectors(row.base_links_path,'links');sel=parent[parent.id_link.isin(contract['selected_parent_links'])];ext={x for e in v['exterior'] if e['path_unique'] for x in e['child_links']};g=nx.DiGraph();g.add_edges_from((mp[int(z.id_us_node)],mp[int(z.id_ds_node)]) for z in c.itertuples())
  sources=[n for n in g if g.in_degree(n)==0];sinks=[n for n in g if g.out_degree(n)==0]
  reached=set().union(*({s}|nx.descendants(g,s) for s in sources));drains=set().union(*({s}|nx.ancestors(g,s) for s in sinks))
  for b in v['boundaries']:
   if b['replacement_incidence']:continue
   p=b['parent_node'];raw=next(n for n,pn in mp.items() if pn==p)
   inside=c[~c.id_link.isin(ext)]
   records.append(dict(example=r.example,state=r.state,parent_node=p,identity_present=b['present'],exterior_supported=b['exterior_supported'],parent_selected_in=int(sel.id_ds_node.eq(p).sum()),parent_selected_out=int(sel.id_us_node.eq(p).sum()),child_replacement_in=int(inside.id_ds_node.eq(raw).sum()),child_replacement_out=int(inside.id_us_node.eq(raw).sum()),reachable_from_inlet=p in reached,reaches_outlet=p in drains,components=nx.number_weakly_connected_components(g),dag=nx.is_directed_acyclic_graph(g),changed_entry_exit_pairs=sum(x['expected']!=x['actual'] for x in v['replacement_connections']),assessment='overstrict_role_requirement_absent_even_in_parent' if (not sel.id_us_node.eq(p).any() or not sel.id_ds_node.eq(p).any()) else 'boundary_present_exterior_supported_internal_flow_role_changed',manual_anchor_review_needed=False))
 pd.DataFrame(records).to_csv(DEST/'connection_flag_triage.csv',index=False)
 iq=pd.read_csv(review.OUT/'island_review_queue.csv',dtype={'example':str});iq=iq[iq.status.eq('wrong_processing_domain_review_creation')];islands=review.catalog('41').set_index('island_id');records=[]
 for r in iq.itertuples():
  row=review.case_row(r.example,r.state);m=json.loads(Path(row.variant_manifest).read_text());stages=review.stage_paths(row);island=islands.loc[r.state].geometry;region=island.buffer(150);parent=review.vectors(row.base_links_path,'links');near=parent[parent.intersects(region)]
  # A parent face can enclose more than one raster island. Retain IDs, not nearest unit.
  enclosing=[face for face in polygonize(unary_union(parent.geometry)) if face.covers(island.representative_point())]
  face=min(enclosing,key=lambda x:x.area) if enclosing else None
  border=parent[parent.geometry.apply(lambda g:face.boundary.intersection(g).length/g.length>.99)] if face is not None else parent.iloc[:0]
  realized=review.vectors(stages[1][1],'links');final=review.vectors(row.child_links_path,'links');ru=unary_union(realized.geometry);fu=unary_union(final.geometry);pu=unary_union(parent.geometry)
  real_local=ru.intersection(region);final_local=fu.intersection(region)
  rnew=real_local.difference(pu.buffer(.01)).length;fnew=final_local.difference(pu.buffer(.01)).length
  lost=real_local.difference(fu.buffer(.01)).length
  retained_border=[int(v.id_link) for v in border.itertuples() if v.geometry.difference(fu.buffer(.01)).length<.02]
  records.append(dict(example=r.example,state=r.state,actual_target_island=r.actual_full_target_islands,explicit_source_units=m.get('explicit_collapse_geometry',{}).get('source_unit_ids'),processing_selected_units=m.get('selected_unit_ids'),inferred_parent_face_links=border.id_link.astype(int).tolist(),parent_face_found=face is not None,realized_new_line_in_target_buffer_m=rnew,final_new_line_in_target_buffer_m=fnew,realized_line_lost_at_anchoring_in_target_buffer_m=lost,retained_parent_face_links=retained_border,known_pure_branch_deletion=bool(r.known_pure_branch_deletion),stage_assessment='realized_geometry_removed_or_replaced_during_anchoring' if lost>50 else 'limited_realized_to_anchored_change',action='rebuild_anchoring_from_actual_target_domain_and_saved_realized_graph' if lost>50 else 'replace_wrong_unit_boundary_audit_with_actual_island_boundary_check',production_changed=False))
  print(r.state,round(rnew),round(fnew),round(lost),flush=True)
 pd.DataFrame(records).to_csv(DEST/'island_creation_triage.csv',index=False)
 # Concrete non-destructive exclusion/reuse plan for the agreed duplicate policy.
 pairs=pd.read_csv(review.OUT/'island_independent_target_pairs.csv',dtype={'example':str});dup=pairs[pairs.island_separately_generated.eq(True)].copy();dup['planned_action']='exclude_separate_island_output_retain_independent_canonical';dup['production_applied']=False;dup.to_csv(DEST/'island_duplicate_resolution_plan.csv',index=False)
 summary=dict(connection_cases=len(q),connection_flagged_nodes=len(pd.read_csv(DEST/'connection_flag_triage.csv')),island_creation_cases=len(records),duplicates=len(dup),production_changed=False)
 (DEST/'triage_summary.json').write_text(json.dumps(summary,indent=2));print(summary)
if __name__=='__main__':run()
