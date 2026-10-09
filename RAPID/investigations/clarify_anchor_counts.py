"""Separate exact-coordinate identity failures from projected-cut candidates.

Nearest node distance equals nearest line distance implies that a child node
exists at a closest network location. This is NOT a topology/lineage certificate.
No source graph or production registry is modified.
"""
from pathlib import Path
import pandas as pd,numpy as np,json
OUT=Path(__file__).resolve().parents[2]/'RAPID/outputs/anchoring_audit_20261008'
def run():
 a=pd.read_csv(OUT/'review_registry.csv',dtype={'example':str});b=pd.read_csv(OUT/'boundary_geometry_details.csv',dtype={'example':str})
 b['node_at_nearest_network_projection']=np.isclose(b.nearest_child_node_distance_m,b.nearest_child_line_distance_m,rtol=0,atol=1e-5)
 keys=['example','family','state','fingerprint']
 g=b.groupby(keys).agg(all_projection_nodes=('node_at_nearest_network_projection','all'),exact_missing=('missing_canonical_node','any')).reset_index().merge(a[keys+['anchor_displacement_over_100m']],on=keys,validate='one_to_one')
 u=g[g.family.ne('island')].groupby(['example','fingerprint']).agg(exact_missing=('exact_missing','any'),all_projection_nodes=('all_projection_nodes','all'),over_100m=('anchor_displacement_over_100m','any'),states=('state',lambda x:';'.join(x))).reset_index()
 u['review_class']=np.select([~u.exact_missing,u.all_projection_nodes,u.over_100m],['exact_boundary_nodes_present','displaced_projection_nodes_present','projection_cut_unresolved_over_100m'],default='projection_cut_unresolved_up_to_100m')
 u.to_csv(OUT/'unit_cut_classification.csv',index=False);u[~u.all_projection_nodes].to_csv(OUT/'unit_cut_review_131.csv',index=False)
 summary=u.groupby(['example','review_class']).size().unstack(fill_value=0);summary.to_csv(OUT/'unit_cut_counts_by_network.csv')
 frames=[]
 for ex in a.loc[a.family.eq('island'),'example'].unique():
  suffix='_direction130' if ex=='15' else ''
  path=Path('/Volumes/PhD/river_hierarchy/network_variants/output')/f'{ex}_island{suffix}__anchored/island_analysis_manifest_strict.csv'
  f=pd.read_csv(path);f['example']=ex;frames.append(f)
 island=a[a.family.eq('island')][['example','state','island_target_domain_mismatch']].merge(pd.concat(frames),left_on=['example','state'],right_on=['example','island_id'],validate='one_to_one')
 island[['example','state','execution_action','equivalence_status','source_unit_id','source_state_dir','analysis_state_dir','mask_equal','anchor_equal','bank_labels_equal','island_target_domain_mismatch']].to_csv(OUT/'island_reuse_and_domain_counts.csv',index=False)
 counts=dict(unit_outputs=len(u),unit_classes=u.review_class.value_counts().to_dict(),island_states=len(island),island_actions=island.execution_action.value_counts().to_dict(),wrong_island_domains=int(island.island_target_domain_mismatch.sum()))
 (OUT/'clarified_counts.json').write_text(json.dumps(counts,indent=2));print(json.dumps(counts,indent=2));print(summary.to_string())
 (OUT/'COUNT_CLARIFICATION.md').write_text('''# Clarification of anchoring and island counts

The previous 1,134 exact-coordinate flags and 45 >100 m flags span unit, compound and island entries. Neither is a count of proven failed realized-to-anchored cuts.

## Unit / compound outputs, deduplicated across those two families

- 1,903 distinct outputs.
- 948 retain all required boundary coordinates exactly.
- 824 fail exact coordinate identity but have a child node at a nearest network projection for every required boundary. These are displaced-node candidates, not proven missed cuts. The 05/island_05_038 alias of S030_unit_37 is a concrete example: node 90 exists 7.019 m from parent boundary 73, has one incoming and one outgoing link (131 and 134), and both nearest-line and nearest-node distances are 7.019 m. Thus a graph break exists there, while the exact-coordinate hydraulic matcher fails to recognize the original identity.
- 131 have at least one boundary whose nearest network projection is not a child node: 35 with nearest-node distance >100 m and 96 with all distances <=100 m. These need correspondence and cut-location review. A nearby shifted cut elsewhere on the same line can still represent the boundary; conversely a nearby node can be the wrong junction. This geometric classification alone cannot establish the exact total of incorrect cuts.

Projection-node presence was inferred from equality of minimum node and minimum line distances within 1e-5 m, using the completed vector audit. It proves presence at a closest location, not correct topology, unique identity, or correct transition-stage provenance. The audit inspected final outputs; attributing every defect specifically to realized-to-anchored processing requires stage comparison. No generic 7 m or 100 m acceptance tolerance has been adopted.

## Islands

- 1,552 routed island entries.
- 1,261 already alias/reuse an independent unit output. These are not duplicate generated graphs.
- 291 have separate island runs. This is not a count of incorrect or unnecessary runs.
- 18 recorded source domains are wrong, all in network 41: 005, 035, 036, 040, 041, 044, 046, 047, 050, 055, 062, 075, 110, 117, 119, 121, 124, 138.
- Four of those 18 have confirmed pure branch-deletion outputs: 005, 055, 075, 138. They are included in the 18, not additional cases. Four is the confirmed pure-deletion count, not a proof that cleanup did no damage in the other fourteen.
- Island 41_006 and base unit 4 are a confirmed same-intended-target pair generated separately, with equal added pixels in the inspected compound window. Their final geometries differ because the independent output loses its cuts. Island 006 itself retains its cuts. Full-network mask equivalence and a complete new target-identity-based duplicate census have not been established. Old equivalence flags are not sufficient: 006 is currently recorded as mask_equal=False even though the inspected target-window edits agree.

Wrong-domain selection and failure to reuse an equivalent target are related identity/domain problems but different failure modes. A legitimate alias also inherits any defect in its source unit graph. These island and unit counts therefore overlap and must not be added to estimate affected unique simulations.

## Review files

- unit_cut_classification.csv: all 1,903 unit/compound outputs and their states.
- unit_cut_review_131.csv: unresolved projected cuts, with large-distance priority flag.
- unit_cut_counts_by_network.csv: per-network classification.
- island_reuse_and_domain_counts.csv: all 1,552 island identities, sources, reuse status and domain flags.

This clarification supersedes any interpretation of the broad exact-coordinate flags as confirmed missing cuts. Production inputs/results remain unchanged.
''')
if __name__=='__main__':run()
