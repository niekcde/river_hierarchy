"""Reproducible isolated boundary-contract repair and validation inventory."""
from pathlib import Path
import sys,json,argparse,traceback
from functools import lru_cache
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'RAPID/src'))
from network_variants.boundary_contract_repair import derive_contract,propose_repairs,validate_result,orient,ids
from network_variants.shared_head_reconstruction import match_child_nodes
from rapid_tools.anchor_repair_trial import repair,node_mapping
OUT=ROOT/'RAPID/outputs/boundary_contract_repair_20261008'
AUDIT=ROOT/'RAPID/outputs/anchoring_audit_20261008'
REVIEWED_CUTS={('41','S016_unit_4'):[5,7],('02','S023_unit_12'):[27],('02','S026_unit_16'):[32],('13','S039_unit_33'):[64],('15','S135_unit_144'):[281]}
REVIEWED_MATCHES={('02','S005_unit_25'):{56:50},('13','S039_unit_33'):{64:66},('02','S004_unit_5'):{58:7}}

@lru_cache(maxsize=24)
def vectors(path,kind):
    cols=['id_node','geometry'] if kind=='nodes' else ['id_link','id_us_node','id_ds_node','geometry']
    return gpd.read_file(path,columns=cols)

@lru_cache(maxsize=24)
def parent_inputs(links_path,nodes_path):
    links=vectors(links_path,'links');nodes=vectors(nodes_path,'nodes');links=orient(links,nodes)
    steps=np.concatenate([np.hypot(*np.diff(np.asarray(g.coords),axis=0).T) for g in links.geometry])
    resolution=float(np.quantile(steps[steps>1e-5],.05))
    membership=pd.read_csv(Path(links_path).parents[2]/'variant/summary/unit_link_membership.csv')
    return links,nodes,membership,resolution


def process(row,target,out,*,tolerance_pixels=4):
    parent,pn,membership,resolution=parent_inputs(row.base_links_path,row.base_nodes_path)
    chosen=membership[membership.unit_id.isin(ids(row.selected_unit_ids))].id_link.astype(int).tolist()
    manifest=json.loads(Path(row.variant_manifest).read_text())
    declared=ids(row.explicit_contract_boundary_nodes) if pd.notna(row.explicit_contract_boundary_nodes) else None
    target_ok=(target.target_set_agrees == True and pd.notna(target.actual_full_target_islands)
               and target.actual_target_count > 0 and target.partial_target_count == 0
               and str(target.target_error).strip() in ['', 'nan'])
    contract=derive_contract(parent,chosen,declared_boundaries=declared,target_ok=target_ok)
    if set(chosen)!=set(ids(row.selected_parent_link_ids)):contract['valid']=False;contract['membership_mismatch']=True
    if 'selected_unit_ids' in manifest and set(ids(manifest['selected_unit_ids']))!=set(ids(row.selected_unit_ids)):
        contract['valid']=False;contract['manifest_selection_mismatch']=True
    if set(contract['boundary_nodes'])!=set(ids(row.required_boundary_nodes)):contract['valid']=False;contract['audit_boundary_mismatch']=True
    contract.update(declared_target_islands=str(target.declared_target_islands),actual_target_islands=str(target.actual_full_target_islands),
                    selected_unit_ids=ids(row.selected_unit_ids),partial_edit_scope=target.partial_edit_scope,
                    geometry_tolerance_m=resolution*tolerance_pixels,raster_step_estimate_m=resolution,geometry_tolerance_pixels=tolerance_pixels)
    dest=out/row.example/row.state;dest.mkdir(parents=True,exist_ok=True)
    (dest/'contract.json').write_text(json.dumps(contract,indent=2))
    base=dict(example=row.example,state=row.state,family=row.family,fingerprint=row.fingerprint,contract_valid=contract['valid'],output=str(dest),production_promoted=False)
    if not contract['valid']:return dict(base,status='blocked_contract',reason='target_or_boundary_contract_mismatch',new_cuts=0,restored_identities=0,anchor_pass=False,exterior_exact=False)
    frame=pd.read_parquet(row.frame_path);mapping=node_mapping(frame)
    child=vectors(row.child_links_path,'links')[['id_link','geometry']].merge(frame[['id_link','id_us_node','id_ds_node']],on='id_link',validate='one_to_one')
    cn=vectors(row.child_nodes_path,'nodes');child=orient(child,cn)
    matches_path=Path(row.variant_manifest).parents[1]/'matching/node_match.csv'
    nm=pd.read_csv(matches_path) if matches_path.exists() else pd.DataFrame()
    key=(row.example,row.state)
    proposal=propose_repairs(parent,pn,child,cn,mapping,contract,nm,geometry_tolerance_m=resolution*tolerance_pixels,
                            reviewed_matches=REVIEWED_MATCHES.get(key),reviewed_cuts=REVIEWED_CUTS.get(key,()))
    ppoints=pn.copy()
    for p,spec in proposal['cuts'].items():ppoints.loc[ppoints.id_node.eq(p),'geometry']=Point(spec['x'],spec['y'])
    after,an,actions=repair(child,cn,ppoints,mapping,contract['boundary_nodes'],cuts=proposal['cuts'],correspondences=proposal['matches'],
                            cut_link_ids={p:v['link_id'] for p,v in proposal['cuts'].items()})
    corrected=dict(zip(an.id_node.astype(int),an.parent_identity.astype(int)))
    validation=validate_result(parent,child,after,corrected,contract,geometry_tolerance_m=resolution*tolerance_pixels)
    if not validation['topology_preserved'] or not validation['geometry_preserved']:raise ValueError('Geometry/topology invariant violated')
    explicit={int(c):int(p) for c,p in corrected.items() if p in contract['boundary_nodes'] and (c not in mapping or mapping[c]!=p)}
    # Exercise the actual hydraulic matcher with the persisted identities.
    regenerated,unknown,matching_qa=match_child_nodes(pn,an,parent.assign(actual_length_m=parent.length),after.assign(link_length_m=after.length),validated_correspondences=explicit)
    for p in contract['boundary_nodes']:
        if p in corrected.values():
            raw=next(c for c,v in corrected.items() if v==p)
            if regenerated[raw]!=p:raise ValueError('Hydraulic rematching discarded boundary identity')
    proposal['hydraulic_matching_qa']=matching_qa
    (dest/'repair_proposal.json').write_text(json.dumps(proposal,indent=2))
    (dest/'validated_node_correspondences.json').write_text(json.dumps(explicit,indent=2))
    (dest/'validation.json').write_text(json.dumps(validation,indent=2))
    pd.DataFrame(validation['exterior']).to_json(dest/'exterior_paths.json',orient='records',indent=2)
    changed=bool(proposal['cuts'] or proposal['matches'])
    if changed:
        after.to_parquet(dest/'trial_links.parquet',index=False);an.to_parquet(dest/'trial_nodes.parquet',index=False)
    status=('repaired_anchor_pass' if changed else 'unchanged_anchor_pass') if validation['anchor_pass'] else ('partial_repair_unresolved' if changed else 'unresolved')
    return dict(base,status=status,reason='',new_cuts=len(proposal['cuts']),restored_identities=len(proposal['matches']),anchor_pass=validation['anchor_pass'],
        exterior_exact=validation['exterior_exact'],exterior_path_failures=sum(not z['path_unique'] for z in validation['exterior']),
        unresolved_boundaries=','.join(map(str,proposal['unresolved'])),routing_ready=False,
        source_manifest=row.source_manifest,source_frame=row.frame_path,
        geometry_preserved=validation['geometry_preserved'],topology_preserved=validation['topology_preserved'])


def process_task(task):
    from types import SimpleNamespace
    data,target,out,tolerance,resume=task
    row=SimpleNamespace(**data);out=Path(out);dest=out/row.example/row.state/'result.json'
    if resume and dest.exists() and (dest.parent/'contract.json').exists():
        cached=json.loads(dest.read_text());saved_contract=json.loads((dest.parent/'contract.json').read_text())
        if cached['status']!='error' and cached['fingerprint']==row.fingerprint and saved_contract.get('geometry_tolerance_pixels')==tolerance:
            return cached
    try:result=process(row,pd.Series(target),out,tolerance_pixels=tolerance)
    except Exception as error:
        result=dict(example=row.example,state=row.state,family=row.family,fingerprint=row.fingerprint,status='error',reason=str(error),new_cuts=0,restored_identities=0,anchor_pass=False,exterior_exact=False)
        dest.parent.mkdir(parents=True,exist_ok=True)
        (dest.parent/'error.txt').write_text(traceback.format_exc())
    dest.parent.mkdir(exist_ok=True,parents=True);dest.write_text(json.dumps(result,indent=2))
    return result


def main():
    from concurrent.futures import ProcessPoolExecutor
    parser=argparse.ArgumentParser();parser.add_argument('--reviewed-only',action='store_true');parser.add_argument('--examples',nargs='*');parser.add_argument('--tolerance-pixels',type=float,default=4);parser.add_argument('--resume',action='store_true');parser.add_argument('--workers',type=int,default=1);args=parser.parse_args()
    out=OUT/('reviewed' if args.reviewed_only else 'full');out.mkdir(exist_ok=True,parents=True)
    registry=pd.read_csv(AUDIT/'review_registry.csv',dtype={'example':str})
    target=pd.read_csv(ROOT/'RAPID/outputs/collapse_creation_review_20261008/target_inventory.csv',dtype={'example':str}).set_index(['example','state'])
    scope=registry[registry.family.isin(['indep','compound_bubbles'])]
    if args.reviewed_only:scope=scope[scope.apply(lambda r:(r.example,r.state) in set(REVIEWED_CUTS)|set(REVIEWED_MATCHES)|{('02','CB001_bubble_5'),('02','CB005_bubble_6')},axis=1)]
    if args.examples:scope=scope[scope.example.isin(args.examples)]
    tasks=[(row._asdict(),target.loc[(row.example,row.state)].to_dict(),str(out),args.tolerance_pixels,args.resume) for row in scope.itertuples(index=False)]
    results=[]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for i,result in enumerate(pool.map(process_task,tasks,chunksize=1)):
            results.append(result)
            if args.reviewed_only or (i+1)%25==0:print(i+1,len(scope),result['example'],result['state'],result['status'],result['new_cuts'],result['restored_identities'],flush=True)
            if (i+1)%25==0:pd.DataFrame(results).to_csv(out/'state_results.csv',index=False)
    f=pd.DataFrame(results);f.to_csv(out/'state_results.csv',index=False)
    print(f.status.value_counts().to_string(),flush=True)
if __name__=='__main__':main()
