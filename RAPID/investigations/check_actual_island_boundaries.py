"""Reassess the eight preserved island graphs using actual enclosing parent faces."""
from pathlib import Path
import sys,json
import pandas as pd
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'RAPID/src'))
from rapid_tools import collapse_creation_review as review
from rapid_tools.anchor_repair_trial import node_mapping,repair,topology
from network_variants.shared_head_reconstruction import match_child_nodes
from network_variants.boundary_contract_repair import derive_contract,propose_repairs,orient,validate_result,paired_exterior_cuts
from shapely.geometry import Point
OUT=ROOT/'RAPID/outputs/boundary_contract_repair_20261008/review_revision'

def run():
 triage=pd.read_csv(OUT/'island_creation_triage.csv',dtype={'example':str});records=[]
 for r in triage.itertuples():
  row=review.case_row(r.example,r.state);parent=review.vectors(row.base_links_path,'links');pn=review.vectors(row.base_nodes_path,'nodes');parent=orient(parent,pn)
  selected=json.loads(r.inferred_parent_face_links);assert selected
  contract=derive_contract(parent,selected,target_ok=True);frame=pd.read_parquet(row.frame_path);mp=node_mapping(frame);cn=review.vectors(row.child_nodes_path,'nodes');child=review.vectors(row.child_links_path,'links')[['id_link','geometry']].merge(frame[['id_link','id_us_node','id_ds_node']],on='id_link');child=orient(child,cn)
  if r.known_pure_branch_deletion:
   stage=review.stage_paths(row)[1];cn=review.vectors(stage[2],'nodes');child=orient(review.vectors(stage[1],'links'),cn)
   mp,_,_=match_child_nodes(pn,cn,parent.assign(actual_length_m=parent.length),child.assign(link_length_m=child.length))
  nm_path=Path(row.variant_manifest).parents[1]/'matching/node_match.csv';nm=pd.read_csv(nm_path) if nm_path.exists() and not r.known_pure_branch_deletion else pd.DataFrame()
  tolerance=17.903061497956514
  proposal=propose_repairs(parent,pn,child,cn,mp,contract,nm,geometry_tolerance_m=tolerance)
  points=pn.copy()
  for p,cut in proposal['cuts'].items():points.loc[points.id_node.eq(p),'geometry']=Point(cut['x'],cut['y'])
  out,nodes,actions=repair(child,cn,points,mp,contract['boundary_nodes'],cuts=proposal['cuts'],correspondences=proposal['matches'],cut_link_ids={p:c['link_id'] for p,c in proposal['cuts'].items()})
  mapping=dict(zip(nodes.id_node,nodes.parent_identity))
  paired=paired_exterior_cuts(parent,pn,out,nodes,mapping,contract,geometry_tolerance_m=tolerance)
  for pair in paired:
   cuts=pair['cuts'];pp=pn.copy()
   for p,c in cuts.items():pp.loc[pp.id_node.eq(p),'geometry']=Point(c['x'],c['y'])
   original_source=out.set_index('id_link').source_link.to_dict()
   out,nodes,extra=repair(out,nodes,pp,mapping,contract['boundary_nodes'],cuts=cuts,cut_link_ids={p:c['link_id'] for p,c in cuts.items()})
   out['source_link']=out.source_link.map(original_source).astype(int);mapping=dict(zip(nodes.id_node,nodes.parent_identity));proposal['cuts'].update(cuts);proposal['records'].append(dict(decision='paired_exterior_supported_cuts',**pair))
  proposal['unresolved']=sorted(set(contract['boundary_nodes'])-set(mapping.values()))
  v=validate_result(parent,child,out,mapping,contract,geometry_tolerance_m=tolerance)
  assert v['topology_preserved'] and v['geometry_preserved']
  folder=OUT/'island_boundary_trials'/r.state;folder.mkdir(parents=True,exist_ok=True)
  (folder/'contract.json').write_text(json.dumps(contract,indent=2));(folder/'proposal.json').write_text(json.dumps(proposal,indent=2));(folder/'validation.json').write_text(json.dumps(v,indent=2))
  explicit={int(c):int(p) for c,p in mapping.items() if p in contract['boundary_nodes']}
  regenerated,_,_=match_child_nodes(pn,nodes,parent.assign(actual_length_m=parent.length),out.assign(link_length_m=out.length),validated_correspondences=explicit)
  assert all(regenerated[c]==p for c,p in explicit.items())
  (folder/'validated_node_correspondences.json').write_text(json.dumps(explicit,indent=2))
  topology_qa=topology(out);assert topology_qa['dag'] and topology_qa['components']==1
  out.to_parquet(folder/'trial_links.parquet',index=False);nodes.to_parquet(folder/'trial_nodes.parquet',index=False)
  result=dict(state=r.state,input_stage='saved_realized_before_faulty_anchoring' if r.known_pure_branch_deletion else 'current_anchored',actual_parent_face_links=selected,actual_boundary_nodes=contract['boundary_nodes'],new_cuts=len(proposal['cuts']),restored_identities=len(proposal['matches']),unresolved=proposal['unresolved'],anchor_pass=v['anchor_pass'],missing_nodes=[b['parent_node'] for b in v['boundaries'] if not b['present']],unsupported_exterior_nodes=[b['parent_node'] for b in v['boundaries'] if not b['exterior_supported']],role_flags=[b['parent_node'] for b in v['boundaries'] if not b['replacement_incidence']],routing_ready=False)
  result.update(matching_preserves_boundary_identities=True,topology=topology_qa,cycle_reduction_from_parent=topology(parent)['cycle_rank']-topology_qa['cycle_rank'],production_changed=False,fingerprint=row.fingerprint)
  result['overall_geometry_pass']=bool(result['anchor_pass'] and result['cycle_reduction_from_parent']==1)
  result['status']='geometry_trial_pass' if result['overall_geometry_pass'] else 'blocked_realized_splice_retains_extra_branch'
  result['remaining_requirement']='Reassemble the local RivGraph product against the actual island boundary; do not promote the saved merged realized graph' if not result['overall_geometry_pass'] else 'Regenerate affected hydraulic attributes before routing'
  (folder/'result.json').write_text(json.dumps(result,indent=2))
  records.append(result);print(result,flush=True)
 (OUT/'actual_island_boundary_results.json').write_text(json.dumps(records,indent=2))
if __name__=='__main__':run()
