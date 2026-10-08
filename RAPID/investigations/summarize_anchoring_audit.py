"""Summarize all-state anchoring audit without mutating scientific results."""
from pathlib import Path
import json
import pandas as pd
from audit_all_anchoring import OUT

def run():
 keys=['example','family','state','fingerprint']
 a=pd.read_csv(OUT/'transition_audit.csv',dtype={'example':str});g=pd.read_csv(OUT/'geometry_verification.csv',dtype={'example':str});b=pd.read_csv(OUT/'boundary_geometry_details.csv',dtype={'example':str});i=pd.read_csv(OUT/'incidence_and_comparison_domains.csv',dtype={'example':str})
 assert len(a)==len(g)==len(i)==3477
 assert a.error.isna().all() and g.geometry_error.isna().all()
 a=a.merge(g,on=keys,validate='one_to_one',suffixes=('','_geometry')).merge(i,on=keys,validate='one_to_one')
 agg=b.groupby(keys).agg(max_anchor_displacement_m=('nearest_child_node_distance_m','max'),missing_anchor_on_existing_line=('classification',lambda x:sum(x=='node_missing_line_passes_original_anchor')),coordinate_present_mapping_review=('classification',lambda x:sum(x=='coordinate_present_mapping_requires_review'))).reset_index()
 a=a.merge(agg,on=keys,validate='one_to_one')
 a['missing_exact_anchor']=a.missing_boundary_count.gt(0);a['anchor_displacement_over_100m']=a.max_anchor_displacement_m.gt(100);a['exterior_geometry_changed']=a.exterior_geometry_not_retained_count.gt(0);a['exterior_width_change_over_1pct']=a.outside_max_width_relative_change.gt(.01);a['exterior_slope_change_over_1pct']=a.outside_max_slope_relative_change.gt(.01);a['explicit_contract_anchor_missing']=a.missing_explicit_contract_boundary_nodes.notna();a['incidence_failure']=~a.replacement_incidence_pass
 a['geometry_anchor_domain_pass']=~(a.missing_exact_anchor|a.exterior_geometry_changed|a.island_target_domain_mismatch|a.incidence_failure)
 a['strict_anchor_and_exterior_pass']=a.strict_anchor_and_exterior_pass&a.geometry_anchor_domain_pass
 flags=['missing_exact_anchor','anchor_displacement_over_100m','exterior_geometry_changed','exterior_width_change_over_1pct','exterior_slope_change_over_1pct','island_target_domain_mismatch','explicit_contract_anchor_missing','incidence_failure']
 unique=a.groupby(['example','fingerprint']).agg(**{f:(f,'max') for f in flags},geometry_anchor_domain_pass=('geometry_anchor_domain_pass','min'),strict_anchor_and_exterior_pass=('strict_anchor_and_exterior_pass','min'),logical_entries=('state','size'),states=('state',lambda x:';'.join(x))).reset_index()
 summary=unique.groupby('example').agg(unique_pairs=('fingerprint','size'),missing_exact_anchor=('missing_exact_anchor','sum'),over_100m=('anchor_displacement_over_100m','sum'),exterior_geometry_changed=('exterior_geometry_changed','sum'),geometry_anchor_domain_pass=('geometry_anchor_domain_pass','sum'),strict_pass=('strict_anchor_and_exterior_pass','sum')).reset_index()
 a.to_csv(OUT/'review_registry.csv',index=False);unique.to_csv(OUT/'unique_pair_review.csv',index=False);summary.to_csv(OUT/'network_summary.csv',index=False)
 # Known positive/negative controls establish selected-domain distinction.
 def case(ex,state):return a[(a.example==ex)&a.state.eq(state)].iloc[0]
 assert case('41','S016_unit_4').missing_boundary_nodes=='5,7'
 assert case('41','S076_unit_3').geometry_anchor_domain_pass
 assert case('41','island_41_005').island_target_domain_mismatch
 assert not case('41','island_41_006').missing_exact_anchor
 assert not case('41','island_41_006').exterior_geometry_changed
 assert case('13','S039_unit_33').missing_anchor_on_existing_line>0
 qa=dict(logical_transitions=len(a),unique_physical_pairs=len(unique),networks=a.example.nunique(),source_errors=0,geometry_errors=0,known_case_controls_passed=True,coordinate_present_mapping_review=int(a.coordinate_present_mapping_review.sum()),logical_flag_counts={f:int(a[f].sum()) for f in flags},unique_flag_counts={f:int(unique[f].sum()) for f in flags},geometry_anchor_domain_pass_unique=int(unique.geometry_anchor_domain_pass.sum()),strict_pass_unique=int(unique.strict_anchor_and_exterior_pass.sum()))
 (OUT/'validation.json').write_text(json.dumps(qa,indent=2))
 miss=b[b.missing_canonical_node].drop_duplicates(['example','fingerprint','parent_node_id'])
 top=miss.sort_values('nearest_child_node_distance_m',ascending=False).head(20)
 top.to_csv(OUT/'largest_anchor_displacements.csv',index=False)
 text=f'''# Completed-network anchoring and exterior-preservation audit — 8 October 2026

## Outcome and scope

Audited all **{len(a):,} logical parent–child entries**, representing **{len(unique):,} distinct physical child fingerprints across 15 networks**. All parents are S000_base. Networks: {', '.join(summary.example)}. Network 11 is excluded from the retained completed hierarchy set. Every state/fingerprint/frame matches the latest shared-head hydraulic-refresh inventory. No production inputs, calibration, outputs, exclusions or routing results were changed.

This is a strict **original selected-domain preservation audit**, not a new hydrograph accuracy test. It finds problems beyond case 41. A missing exact-coordinate anchor is a review flag, not automatically a gross collapse error: it includes small raster shifts as well as lost splits. Thresholds of 100 m and 1% below prioritize review; they are not scientific acceptance criteria.

## Unique physical pairs

Aliases remain individually auditable in review_registry.csv. For this table, a pair fails if any associated selected domain fails; identical output fingerprints can have different target metadata.

{summary.to_markdown(index=False)}

- Missing one or more exact original boundary nodes: **{qa['unique_flag_counts']['missing_exact_anchor']}** unique pairs.
- At least one required boundary more than 100 m from every child node: **{qa['unique_flag_counts']['anchor_displacement_over_100m']}** unique pairs.
- At least one original exterior link geometry not retained as a complete child link: **{qa['unique_flag_counts']['exterior_geometry_changed']}** unique pairs. This includes merge/split changes even where the same channel trace survives.
- Geometry, exact anchors, necessary incidence and known target-domain checks all pass: **{qa['geometry_anchor_domain_pass_unique']}** unique pairs.
- Those checks plus strict exterior hydraulic-value preservation all pass: **{qa['strict_pass_unique']}** unique pairs. This is deliberately stricter than the current shared-head policy; do not interpret every additional flag as an implementation bug.

## What was checked

1. Resolve intended parent links from original state/unit membership. Island selections use the recorded source unit, with the earlier exhaustive target-containment audit carried forward separately. This is not a new all-state raster-target audit.
2. Derive every selected-domain interface from parent connectivity, including interfaces to unselected branches. Do not test only nominal inlet/outlet or require an open compound member to have a single pair of ports.
3. Check canonical physical node identities, then independently inspect original projected GeoPackages and nearest child-node/line distances. Raw child numeric IDs are not trusted.
4. Compare **every** original exterior reach, including reaches missing from the child. Endpoint/length correspondence checks hydraulic attributes; normalized vector shapes plus topological equality check geometry independently of raw IDs and harmless vertex resampling.
5. Check required incoming/outgoing incidence of candidate replacement links. This is necessary but not sufficient: an altered exterior edge can belong to the candidate set. The separate exterior checks prevent treating that as a clean pass.
6. Read recorded local-patch contracts where available, and separately record existing enlarged comparison domains. Missing contract metadata is not evidence of a clean state.
7. Confirm known positive and negative case-41 controls and an independently found missing split in case 13. All source and geometry reads completed without errors.

## Why the flags differ in severity

For missing anchors, nearest-child-node distance has median **{miss.nearest_child_node_distance_m.median():.2f} m**, 95th percentile **{miss.nearest_child_node_distance_m.quantile(.95):.2f} m**, and maximum **{miss.nearest_child_node_distance_m.max():.2f} m** (deduplicated by example/fingerprint/original anchor). Small movements often resemble raster-scale relocation; this audit does not accept them automatically or prove their routing impact. There are **{len(miss[miss.classification.eq('node_missing_line_passes_original_anchor')])}** distinct missing boundary instances where a child line still passes through the original point: the required graph cut is absent at that exact point. That category includes both nearby shifted cuts and clearly merged reaches; use the distances to distinguish them.

Exterior slopes and heads require separate interpretation. reconstruct_shared_heads deliberately re-solves interiors of affected hydraulic ownership domains, which can exceed the selected collapse domain. Therefore a surviving geometric reach can acquire different slope/head values without a vector-generation bug. Most individual flagged slope changes may be small; the registry records maximum relative changes and separates width changes from slope changes. An arbitrary absolute head offset alone does not change routing; slope and depth do. We must explicitly decide whether the scientific comparison freezes the selected-domain exterior or allows the larger reconstructed domain.

## Case 41 retained findings

- **S016_unit_4**: original boundaries **5 and 7 missing**; exterior connectors **4 and 8** are no longer separate reaches. This is the small independent collapse, so removal of those boundaries is inappropriate for the intended frozen exterior comparison.
- **S076_unit_3**: the whole compound collapse retains its required outer interface. Removal of internal nodes 5 and 7 is expected here; it is not misclassified as an anchoring failure.
- **island_41_005**: exact-node checks alone would pass, but the wrong small-unit target domain remains flagged. The prior raster/vector trace showed that the actual large-island target was filled and its new channel subsequently removed by cleanup. The result was deletion of old link 7. Target correctness and anchoring correctness are independent tests.
- **island_41_006**: small-island geometry/anchors are retained. Exterior connector slopes nevertheless change under shared-head reconstruction; this prevents a strict hydraulic-exterior pass, but does not make it the same geometric defect as S016.
- The prior complete island scan's **18 incorrectly containing source domains, all in 41**, remain flagged. Four were confirmed pure branch deletion; the other fourteen require their own repair assessment. That earlier target-containment result is not expanded into an unsupported claim that all other raster targets are perfect.

## Mechanisms and next repair gates

The legacy anchor transform in network_variants/variant_generation.py is conditional on recovered parent paths/overlap cuts. It can return the graph with zero splits, and can place a split at a projected point on the child line. Neither behavior alone certifies the exact original interface. Current coordinate-based hydraulic matching deliberately treats displaced coordinates as new nodes. These two conventions can disagree. The local-patch merge has an incidence check, but a check at an intermediate stage does not certify the final graph after anchoring, pruning, width processing and fallback paths. The historical output manifests span several execution modes/versions; the code inspection identifies mechanisms, not a proof that one current line caused every historical flag.

Use a backend-independent **final artifact gate**, after every fallback and cleanup, checking intended target identities, all required interface locations and directions, and every exterior reach. Permit internal-node removal only inside the selected domain. Fail explicitly when required splits cannot be established; do not silently accept a connected DAG with the desired loop count. Preserve parent exterior geometries and hydraulic values if that is the intended experiment, or explicitly declare and audit a larger allowed reconstruction domain. An enlarged surviving-anchor analysis domain can make resistance comparable, but cannot restore a missing graph cut or prove the original collapse implementation correct.

Repair priorities: (1) the 18 wrong island domains in 41 and confirmed branch-deletion cases; (2) large missing splits, including S016 in 41 and the large-displacement cases listed separately; (3) reconcile projected/raster-shifted anchor identity with the exact-coordinate hydraulic policy; (4) resolve exterior hydraulic inheritance/domain policy. Regenerate and reroute only after those artifacts pass the gate. This audit does not estimate hydrograph changes or prescribe a blanket rerun of every flagged state.

## Review artifacts

- review_registry.csv: one row per logical transition, complete flags, IDs, source paths, distances, contract and comparison-domain information.
- unique_pair_review.csv / network_summary.csv: deduplicated counts.
- boundary_geometry_details.csv / largest_anchor_displacements.csv: original node, nearest child node, line distance and classification.
- exterior_geometry_details.csv: original exterior links no longer retained whole.
- exterior_property_changes.csv.gz: matched exterior link changes, with widths/slopes and changed-field names.
- incidence_and_comparison_domains.csv: interface direction checks and existing domain expansions.
- validation.json: scope and control assertions.

Reproduce with audit_all_anchoring.py, verify_anchoring_geometry.py, audit_anchor_incidence.py, then summarize_anchoring_audit.py (all under RAPID/investigations). These scripts write only investigation outputs and do not alter production.
'''
 (OUT/'REPORT.md').write_text(text)
 print(json.dumps(qa,indent=2));print(summary.to_string(index=False))
 if (OUT/'exterior_property_changes.csv').exists():
  import gzip,shutil
  with (OUT/'exterior_property_changes.csv').open('rb') as src,gzip.open(OUT/'exterior_property_changes.csv.gz','wb') as dst:shutil.copyfileobj(src,dst)
  (OUT/'exterior_property_changes.csv').unlink()
if __name__=='__main__':run()
