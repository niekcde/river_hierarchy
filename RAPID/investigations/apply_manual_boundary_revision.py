"""Apply explicit human decisions in a separate, reproducible geometry trial."""
from pathlib import Path
import sys,json
import numpy as np
import pandas as pd
import geopandas as gpd
import networkx as nx
from shapely.geometry import Point
from shapely.ops import unary_union
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'RAPID/src'))
from rapid_tools import collapse_creation_review as review
from rapid_tools.anchor_repair_trial import node_mapping,repair,topology
from rapid_tools.reviewed_geometry_edits import merge_degree_two,restore_short_parent_connector
from network_variants.boundary_contract_repair import orient,PathIndex
from network_variants.shared_head_reconstruction import match_child_nodes
OUT=ROOT/'RAPID/outputs/boundary_contract_repair_20261008';DEST=OUT/'review_revision'
SPECS={
 ('22','S262_unit_287'):dict(cuts={573:857,575:857},internal=[574,589]),
 ('41','S061_unit_57'):dict(cuts={110:157,114:173,127:177,128:179},internal=[112,115]),
 ('37','S067_unit_303'):dict(restore=True),
 ('37','S289_unit_143'):dict(merge=True,internal=[296]),
 ('37','S051_unit_256'):dict(provisional=[509]),
 ('22','S135_unit_211'):dict(provisional=[418]),
}

def run_case(ex,state,spec):
 row=review.case_row(ex,state);oldfolder=OUT/'full'/ex/state;dest=DEST/ex/state;dest.mkdir(parents=True,exist_ok=True)
 parent=review.vectors(row.base_links_path,'links');pn=review.vectors(row.base_nodes_path,'nodes');parent=orient(parent,pn)
 original_nodes=review.vectors(row.child_nodes_path,'nodes');frame=pd.read_parquet(row.frame_path)
 original=review.vectors(row.child_links_path,'links')[['id_link','geometry']].merge(frame[['id_link','id_us_node','id_ds_node']],on='id_link',validate='one_to_one');original=orient(original,original_nodes)
 if (oldfolder/'trial_links.parquet').exists():
  before=gpd.read_parquet(oldfolder/'trial_links.parquet');nodes=gpd.read_parquet(oldfolder/'trial_nodes.parquet')
 else:
  before,nodes,_=repair(original,original_nodes,pn,node_mapping(frame),[])
 before=orient(before,nodes);mapping=dict(zip(nodes.id_node.astype(int),nodes.parent_identity.astype(int)));contract=json.loads((oldfolder/'contract.json').read_text())
 pp=pn.copy();cuts=spec.get('cuts',{});cut_evidence=[]
 for p,lid in cuts.items():
  line=before[before.id_link.eq(lid)].geometry.iloc[0];pt=pp.loc[pp.id_node.eq(p),'geometry'].iloc[0];projected=line.interpolate(line.project(pt));dist=pt.distance(projected)
  assert dist<=contract['geometry_tolerance_m'];pp.loc[pp.id_node.eq(p),'geometry']=projected
  cut_evidence.append(dict(parent_node=p,child_source_link=lid,projection_offset_m=dist))
 after,an,actions=repair(before,nodes,pp,mapping,contract['boundary_nodes'],cuts=cuts,cut_link_ids=cuts)
 # repair assigns source relative to its input; recover ORIGINAL source lineage.
 sources=before.set_index('id_link').source_link.to_dict()
 after['source_link']=after.source_link.map(sources).astype(int)
 after['source_links']=after.source_link.map(lambda x:json.dumps([int(x)]));after['edit_kind']='retained_or_split';after['restored_parent_link']=np.nan
 if spec.get('merge'):
  raw=int(an.loc[an.parent_identity.eq(296),'id_node'].iloc[0]);after,an=merge_degree_two(after,an,raw,989)
 if spec.get('restore'):
  after,an,shift=restore_short_parent_connector(after,an,parent,pn,653,598,984,900);actions.append(dict(action='junction_relocated_and_parent_connector_restored',raw_node=653,parent_node=598,displacement_m=shift))
 mp=dict(zip(an.id_node.astype(int),an.parent_identity.astype(int)));after=orient(after,an);after['geometry_length_m']=after.length
 # Existing terminal IDs, components, directedness and cycle rank must be conserved.
 a,b=topology(before),topology(after);invariants={k:a[k]==b[k] for k in ['components','cycle_rank','dag','inlets','outlets']};assert all(invariants.values()),invariants
 assert not an.id_node.duplicated().any() and not an.parent_identity.duplicated().any()
 graph=nx.DiGraph();graph.add_edges_from(zip(after.id_us_node,after.id_ds_node))
 reachable=set().union(*({s}|nx.descendants(graph,s) for s in b['inlets']));drains=set().union(*({s}|nx.ancestors(graph,s) for s in b['outlets']));assert set(graph)<=reachable&drains
 # Ensure directed reachability among EVERY retained original raw node is unchanged.
 g0=nx.DiGraph();g0.add_edges_from(zip(before.id_us_node,before.id_ds_node));common=set(g0)&set(graph)
 reach_same=all((nx.descendants(g0,n)&common)==(nx.descendants(graph,n)&common) for n in common);assert reach_same
 removed=set(spec.get('internal',[]));provisional=set(spec.get('provisional',[]));required=set(contract['boundary_nodes'])-removed-provisional
 missing=required-set(mp.values());assert not missing,missing
 assert not removed&set(mp.values()),'Reviewed internal parent identity still retained'
 explicit={c:p for c,p in mp.items() if p in set(pn.id_node)}
 regenerated,_,qa=match_child_nodes(pn,an,parent.assign(actual_length_m=parent.length),after.assign(link_length_m=after.length),validated_correspondences=explicit)
 # Provisional interfaces must remain unmatched, and deliberate internal identities stay absent.
 reintroduced=(removed|provisional)&set(regenerated.values())
 preserved=all(regenerated[c]==p for c,p in explicit.items());assert preserved
 index=PathIndex(after,mp);exterior=[]
 for pr in parent[~parent.id_link.isin(contract['selected_parent_links'])].itertuples():
  matches=index.compare(pr,contract['geometry_tolerance_m']);exterior.append(dict(parent_link=int(pr.id_link),path_unique=len(matches)==1,geometry_exact=len(matches)==1 and matches[0]['hausdorff_m']<1e-5 and matches[0]['length_error_m']<1e-5,**(matches[0] if len(matches)==1 else {'child_links':[]})))
 after.to_parquet(dest/'trial_links.parquet',index=False);an.to_parquet(dest/'trial_nodes.parquet',index=False)
 contract.update(reviewed_internal_parent_nodes=sorted(removed),reviewed_provisional_parent_nodes=sorted(provisional),reviewed_required_boundary_nodes=sorted(required),reviewed_decisions=spec,original_automated_contract_preserved_as_evidence=True)
 (dest/'contract.json').write_text(json.dumps(contract,indent=2));(dest/'exterior_paths.json').write_text(json.dumps(exterior,indent=2));(dest/'validated_node_correspondences.json').write_text(json.dumps(explicit,indent=2))
 # Compare only changed source groups; whole-network Hausdorff is quadratic.
 changed_sources={s for z in after.itertuples() if z.edit_kind!='retained_or_split' for s in json.loads(z.source_links)}|set(before.source_link[before.source_link.isin(after.source_link[after.source_link.duplicated(keep=False)])])
 old_local=before[before.source_link.isin(changed_sources)];new_local=after[after.source_links.map(lambda ss:bool(set(json.loads(ss))&changed_sources))]
 geometry_difference=unary_union(old_local.geometry).hausdorff_distance(unary_union(new_local.geometry)) if len(old_local) else 0.
 result=dict(example=ex,state=state,fingerprint=row.fingerprint,reviewed_geometry_pass=True,reviewed_required_nodes_present=True,invariants=invariants,all_retained_node_reachability_preserved=reach_same,geometry_hausdorff_change_m=geometry_difference,total_length_change_m=float(after.length.sum()-before.length.sum()),new_cuts=cut_evidence,actions=actions,provisional_nodes=sorted(provisional),internal_nodes=sorted(removed),matching_preserves_explicit_identities=preserved,matching_reintroduced_waived_nodes=sorted(reintroduced),routing_ready=False,production_changed=False,remaining_exterior_comparison_failures=[r['parent_link'] for r in exterior if not r['path_unique']])
 (dest/'result.json').write_text(json.dumps(result,indent=2));return result

def main():
 import os
 (DEST/'manual_revision.pid').write_text(str(os.getpid()))
 results=[]
 for (ex,state),spec in SPECS.items():
  result=run_case(ex,state,spec);results.append(result);print(ex,state,'passed',result['matching_reintroduced_waived_nodes'],flush=True)
 (DEST/'manual_edit_results.json').write_text(json.dumps(results,indent=2))
if __name__=='__main__':main()
