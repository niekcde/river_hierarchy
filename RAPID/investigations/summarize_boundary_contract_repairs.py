"""Publish the full repair inventory, alias dependencies and focused review queue."""
from pathlib import Path
import sys,json,hashlib,html
from collections import Counter
import pandas as pd
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'RAPID/outputs/boundary_contract_repair_20261008'


def main():
    registry=pd.read_csv(ROOT/'RAPID/outputs/anchoring_audit_20261008/review_registry.csv',dtype={'example':str})
    records=[];boundaries=[];exterior=[];duplicates=[];tolerances=[];cut_keys=set();map_keys=set();head_diagnostics=[]
    for file in sorted((OUT/'full').glob('*/*/result.json')):
        result=json.loads(file.read_text());folder=file.parent
        validation=folder/'validation.json';proposal=folder/'repair_proposal.json'
        if validation.exists():
            v=json.loads(validation.read_text());assignments={}
            for row in v['exterior']:
                if row['path_unique']:
                    for c in row['child_links']:assignments.setdefault(c,[]).append(row['parent_link'])
                if not row['geometry_exact']:exterior.append(dict(example=result['example'],state=result['state'],**row))
            collisions={c:ps for c,ps in assignments.items() if len(set(ps))>1}
            if collisions:duplicates.append(dict(example=result['example'],state=result['state'],collisions=collisions))
            result['exterior_path_reuse']=bool(collisions)
            if collisions:result['status']='unresolved_exterior_path_reuse';result['anchor_pass']=False
            for b in v['boundaries']:
                if not (b['present'] and b['exterior_supported'] and b['replacement_incidence']):boundaries.append(dict(example=result['example'],state=result['state'],fingerprint=result['fingerprint'],**b))
            failed=[b for b in v['boundaries'] if not (b['present'] and b['exterior_supported'] and b['replacement_incidence'])]
            result['failed_boundary_checks']=','.join(str(b['parent_node']) for b in failed)
            if failed and not result.get('reason'):
                result['reason']=';'.join(sorted({b['reason'] or 'missing_replacement_incidence' for b in failed}))
        if proposal.exists():
            p=json.loads(proposal.read_text());contract=json.loads((folder/'contract.json').read_text())
            cut_keys.update((result['example'],result['fingerprint'],int(k)) for k in p['cuts'])
            map_keys.update((result['example'],result['fingerprint'],int(k),int(val)) for k,val in p['matches'].items())
            for action in p['records']:
                if action['decision']=='exterior_supported_saved_correspondence':tolerances.append(dict(example=result['example'],state=result['state'],fingerprint=result['fingerprint'],**action,displacement_pixels=action['displacement_m']/contract['raster_step_estimate_m']))
        attrs=folder/'fragment_result.json'
        if attrs.exists():
            a=json.loads(attrs.read_text());result['attribute_status']=a['status'];result['fragment_widths_valid']=a.get('widths_valid',False);result['attribute_reason']=a.get('reason','')
        if (folder/'strict_head_failure.json').exists():head_diagnostics.append(json.loads((folder/'strict_head_failure.json').read_text()))
        if (folder/'exterior_attributes_check.json').exists():
            result['exact_exterior_attributes_preserved']=json.loads((folder/'exterior_attributes_check.json').read_text())['all_preserved']
        file.write_text(json.dumps(result,indent=2))
        records.append(result)
    f=pd.DataFrame(records);f.to_csv(OUT/'full/state_results.csv',index=False)
    rank={'error':0,'blocked_contract':1,'unresolved_exterior_path_reuse':2,'partial_repair_unresolved':3,'unresolved':4,'repaired_anchor_pass':5,'unchanged_anchor_pass':6}
    f['_rank']=f.status.map(rank);unique=f.sort_values('_rank').drop_duplicates(['example','fingerprint']).drop(columns='_rank')
    unique.to_csv(OUT/'distinct_output_results.csv',index=False)
    pd.DataFrame(boundaries).to_csv(OUT/'unresolved_boundaries.csv',index=False)
    pd.DataFrame(exterior).to_json(OUT/'nonexact_exterior_paths.json',orient='records',indent=2)
    pd.DataFrame(tolerances).to_json(OUT/'accepted_correspondence_distances.json',orient='records',indent=2)
    # Existing aliases inherit source status; no target-based duplicate is promoted here.
    sources={}
    for r in registry[registry.family.isin(['indep','compound_bubbles'])].itertuples():sources.setdefault(str(Path(r.source_manifest).resolve()),[]).append((r.example,r.state))
    statuses=f.set_index(['example','state']);aliases=[]
    for r in registry[registry.family.eq('island')].itertuples():
        candidates=sources.get(str(Path(r.source_manifest).resolve()),[])
        if candidates:
            linked=[statuses.loc[k].status for k in candidates if k in statuses.index]
            aliases.append(dict(example=r.example,island_state=r.state,source_states=';'.join(s for _,s in candidates),source_statuses=';'.join(linked),relationship='existing_source_alias',applied=False))
        else:aliases.append(dict(example=r.example,island_state=r.state,source_states='',source_statuses='',relationship='separately_generated_island_not_modified',applied=False))
    pd.DataFrame(aliases).to_csv(OUT/'island_alias_dependencies.csv',index=False)
    policy=json.loads((ROOT/'RAPID/outputs/collapse_creation_review_20261008/review_policy.json').read_text())
    unresolved=f[~f.status.isin(['repaired_anchor_pass','unchanged_anchor_pass'])].drop(columns='_rank')
    unresolved.to_csv(OUT/'priority_review_queue.csv',index=False)
    connectivity=f[f.get('replacement_connectivity_review',pd.Series(False,index=f.index)).eq(True)].drop(columns='_rank')
    connectivity.to_csv(OUT/'replacement_connectivity_review.csv',index=False)
    hydraulic=f[f.get('attribute_status',pd.Series('',index=f.index)).eq('blocked') | f.get('fragment_widths_valid',pd.Series(True,index=f.index)).eq(False)].drop(columns='_rank')
    hydraulic.to_csv(OUT/'hydraulic_review_queue.csv',index=False)
    provenance={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [ROOT/'network_variants/boundary_contract_repair.py',ROOT/'network_variants/shared_head_reconstruction.py',ROOT/'RAPID/investigations/run_boundary_contract_repairs.py',ROOT/'RAPID/investigations/derive_repaired_fragment_attributes.py',ROOT/'RAPID/outputs/anchoring_audit_20261008/review_registry.csv']}
    summary=dict(logical_states=len(f),distinct_source_outputs=len(unique),logical_status_counts=f.status.value_counts().to_dict(),distinct_status_counts=unique.status.value_counts().to_dict(),
        logical_new_cuts=int(f.new_cuts.sum()),logical_restored_identities=int(f.restored_identities.sum()),
        distinct_new_boundary_cuts=len(cut_keys),distinct_restored_identities=len(map_keys),
        distinct_any_repaired=int(f[(f.new_cuts+f.restored_identities).gt(0)][['example','fingerprint']].drop_duplicates().shape[0]),
        exterior_geometry_exact_distinct=int(unique.exterior_exact.fillna(False).sum()),
        replacement_connectivity_review_states=len(connectivity),
        replacement_connectivity_review_distinct=int(connectivity[['example','fingerprint']].drop_duplicates().shape[0]),
        attribute_status_counts=f.attribute_status.value_counts().to_dict(),hydraulic_review_states=len(hydraulic),
        strict_head_failures=len(head_diagnostics),current_head_policy_solves_strict_failures=sum(x['current_head_policy_passed'] for x in head_diagnostics),
        exact_exterior_attribute_checks=int(f.get('exact_exterior_attributes_preserved',pd.Series(dtype=bool)).notna().sum()),
        exact_exterior_attribute_failures=int(f.get('exact_exterior_attributes_preserved',pd.Series(dtype=bool)).eq(False).sum()),
        existing_island_aliases=sum(x['relationship']=='existing_source_alias' for x in aliases),exterior_path_reuse_failures=duplicates,
        all_states_accounted_for=len(f)==int(registry.family.isin(['indep','compound_bubbles']).sum()),production_promoted=False,
        geometry_envelope='Four estimated raster steps, secondary to saved correspondence and directed exterior-path evidence; not a hydraulic calibration parameter',
        review_policy=policy,provenance=provenance)
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2))
    per_network=unique.groupby('example').agg(outputs=('state','size'),anchor_pass=('anchor_pass','sum'),exterior_exact=('exterior_exact','sum'),new_cuts=('new_cuts','sum'),restored_identities=('restored_identities','sum'))
    per_network.to_csv(OUT/'by_network.csv')
    intro='''<!doctype html><meta charset="utf-8"><style>body{font:16px system-ui;margin:40px;max-width:1400px}table{border-collapse:collapse}td,th{padding:7px;border:1px solid #ddd;text-align:left}p{max-width:1100px;line-height:1.5}code{background:#eee;padding:2px}</style><h1>Boundary-contract repair trial</h1><p>Targets are cross-checked against the saved target census. Boundary contracts are derived from selected parent links and their exterior connections. Existing saved node correspondences need directed exterior-path evidence; unresolved correspondence claims are not replaced by arbitrary new cuts. Accepted IDs persist through the hydraulic matcher. Exterior validation supports chains of subdivided links.</p><p><b>Production data and calibration are unchanged.</b> All repaired geometry is isolated here. Anchor pass, exact exterior preservation, and hydraulic readiness are separate checks. Trial attributes freeze original exterior heads and inherit parent exterior widths only on exact recovered geometry. Interior split widths are re-trimmed from their own samples. Missing trim support remains blocked; learned width QC and calibrated Cunge preparation have not been rerun.</p><p>Four estimated raster steps form a secondary geometry envelope. Topology and recorded correspondence are required; distance alone is insufficient. Target census coverage limitations are retained in each contract. Independent states remain canonical, and the approved island 41_099 / tiny-island absorption is retained. Existing island aliases are listed as dependencies, not rewritten.</p>'''
    intro+=f'<p><b>{len(f)} logical states / {len(unique)} distinct source outputs.</b> Where aliases differ, the distinct-output table uses the least-resolved status.</p>'
    intro+=unique.status.value_counts().rename('Distinct outputs').to_frame().to_html()
    intro+='<h2>By network</h2>'+per_network.to_html()+'<h2>Unresolved states</h2>'+unresolved[['example','state','status','reason','unresolved_boundaries']].to_html(index=False)
    intro+=f'<h2>Separate creation and hydraulic checks</h2><p>{len(connectivity)} states have different entry-to-exit reachability from the parent selected domain. This is a separate review flag for multi-boundary collapse behaviour, not automatically a missed-anchor defect. No simple one-entry/one-exit case shows this difference. Splitting and identity repair preserve the existing child graph connections.</p><p>{len(hydraulic)} repaired states require hydraulic review because fixed exterior heads cannot be satisfied or fragment width support is insufficient. A diagnostic current-policy head solve is recorded for strict head failures; it is never silently substituted.</p>'
    intro+='''<p><a href="priority_review_queue.csv">Anchor/target review queue</a> · <a href="unresolved_boundaries.csv">Boundary failures</a> · <a href="replacement_connectivity_review.csv">Multi-boundary connectivity review</a> · <a href="hydraulic_review_queue.csv">Hydraulic review</a> · <a href="island_alias_dependencies.csv">Island dependencies</a> · <a href="summary.json">Summary and provenance</a></p><p>Open RAPID/notebooks/rapid_boundary_contract_review.ipynb for before/after Plotly maps of any state.</p>'''
    (OUT/'index.html').write_text(intro)
    print(json.dumps({k:v for k,v in summary.items() if k not in ['review_policy','provenance']},indent=2))
if __name__=='__main__':main()
