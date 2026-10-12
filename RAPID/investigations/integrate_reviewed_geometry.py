"""Assemble and activate the reviewed geometry release; do not route stale attributes."""
from pathlib import Path
import sys, json, shutil, argparse, os
from functools import lru_cache
import pandas as pd
import geopandas as gpd
import networkx as nx
import numpy as np
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from network_variants.reviewed_geometry import ACTIVE, ReviewedGeometry, digest
SOURCE = ROOT/'RAPID/outputs/boundary_contract_repair_20261008'
REV = SOURCE/'review_revision'
ISLAND = ROOT/'RAPID/outputs/island_assembly_fix_20261008'
ASSETS = ROOT/'RAPID/assets/bounded_robust_slopes_20261002/child_head_reconstruction_20261003'
OUT = ROOT/'RAPID/assets/reviewed_geometry_20261009'


def read(path):
    return json.loads(Path(path).read_text())


def write(path, data):
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False)+'\n')


def truth(value):
    return value is True or str(value).lower() == 'true'


@lru_cache(maxsize=None)
def make_bundle(folder, stage):
    folder = Path(folder)
    island = stage == 'island_reassembled'
    lp = folder/('fixed_links.parquet' if island else 'trial_links.parquet')
    npth = folder/('fixed_nodes.parquet' if island else 'trial_nodes.parquet')
    if not lp.exists():
        return None
    cp = folder/('actual_island_contract.json' if island else 'contract.json')
    mp = folder/'validated_node_correspondences.json'
    files = {'links.parquet': lp, 'nodes.parquet': npth, 'contract.json': cp,
             'validated_node_correspondences.json': mp}
    hashes = {name: digest(path) for name, path in files.items()}
    import hashlib
    identity = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()[:24]
    dest = OUT/'objects'/identity
    dest.mkdir(parents=True, exist_ok=True)
    for name, path in files.items():
        if not (dest/name).exists():
            shutil.copy2(path, dest/name)
        if digest(dest/name) != hashes[name]:
            raise ValueError(f'Corrupted release object {dest/name}')
    links, nodes = gpd.read_parquet(lp), gpd.read_parquet(npth)
    if links.crs != nodes.crs or links.crs is None:
        raise ValueError(f'CRS mismatch: {folder}')
    if links.id_link.duplicated().any() or nodes.id_node.duplicated().any():
        raise ValueError(f'Duplicate IDs: {folder}')
    if not np.isfinite(links.length).all() or (links.length <= 0).any():
        raise ValueError(f'Invalid length: {folder}')
    if not links.geometry.is_valid.all() or not nodes.geometry.is_valid.all():
        raise ValueError(f'Invalid geometry: {folder}')
    endpoints = set(links.id_us_node)|set(links.id_ds_node)
    if not endpoints <= set(nodes.id_node):
        raise ValueError(f'Unknown endpoint: {folder}')
    g = nx.DiGraph();g.add_edges_from(zip(links.id_us_node, links.id_ds_node))
    if not nx.is_directed_acyclic_graph(g) or nx.number_weakly_connected_components(g) != 1:
        raise ValueError(f'Disconnected/cyclic graph: {folder}')
    points = nodes.set_index('id_node').geometry
    from shapely.geometry import Point
    maximum_endpoint_error = max(max(Point(r.geometry.coords[0]).distance(points[r.id_us_node]), Point(r.geometry.coords[-1]).distance(points[r.id_ds_node])) for r in links.itertuples())
    if maximum_endpoint_error > 1e-5:
        raise ValueError(f'Geometry endpoints differ from nodes by {maximum_endpoint_error}m: {folder}')
    contract = read(cp)
    required = set(contract.get('reviewed_required_boundary_nodes', contract['boundary_nodes']))
    identities = dict(zip(nodes.id_node.astype(int), nodes.parent_identity.astype(int)))
    if required - set(identities.values()):
        raise ValueError(f'Missing required parent identity: {folder}')
    explicit = {int(k): int(v) for k,v in read(mp).items()}
    if any(identities.get(k) != v for k,v in explicit.items()):
        raise ValueError(f'Conflicting persisted correspondence: {folder}')
    forbidden = set(contract.get('reviewed_internal_parent_nodes', []))|set(contract.get('reviewed_provisional_parent_nodes', []))
    if forbidden & set(identities.values()):
        raise ValueError(f'Waived identity reintroduced: {folder}')
    # Store complete canonical identity map, including child-only IDs, separately
    # from validated original-parent matches. Consumers must not rematch by distance.
    write(dest/'node_identities.json', identities)
    hashes['node_identities.json'] = digest(dest/'node_identities.json')
    evidence = {}
    for name in ['validation.json', 'result.json', 'review_result.json', 'exterior_paths.json', 'boundary_repairs.json', 'fragment_result.json', 'strict_head_failure.json']:
        if (folder/name).exists():
            shutil.copy2(folder/name, dest/name)
            evidence[name] = digest(dest/name)
    metadata = dict(geometry_id=identity, source_folder=str(folder), stage=stage,
                    checksums=hashes, evidence_checksums=evidence, links=len(links), nodes=len(nodes),
                    required_boundaries=len(required), maximum_endpoint_error_m=maximum_endpoint_error,
                    directed_acyclic=True, components=1, geometry_validated=True,
                    hydraulics_ready=False)
    write(dest/'bundle.json', metadata)
    return dict(bundle=str(dest), geometry_id=identity, requires_hydraulic_regeneration=True)


def build():
    if (OUT/'manifest.json').exists():
        raise ValueError('This release is immutable and may be referenced by later releases; build a new release to revise it.')
    OUT.mkdir(parents=True, exist_ok=True)
    audit = pd.read_csv(ROOT/'RAPID/outputs/anchoring_audit_20261008/review_registry.csv', dtype={'example':str})
    audits = {(r.example,r.family,r.state):r for r in audit.itertuples()}
    full = pd.read_csv(SOURCE/'full/state_results.csv', dtype={'example':str})
    fullmap = {(r.example,r.state):r for r in full.itertuples()}
    manual = {(r['example'],r['state']):r for r in read(REV/'manual_edit_results.json')}
    accepted = {(r['example'],r['state']) for r in map(json.loads,(SOURCE/'boundary_manual_reviews.jsonl').read_text().splitlines()) if r['decision'] in ['visual_acceptance','visual_no_problem_question']}
    triage = pd.read_csv(REV/'connection_flag_triage.csv', dtype={'example':str})
    if not (triage.identity_present.all() and triage.exterior_supported.all() and triage.dag.all()):
        raise ValueError('Connection triage contains failed cases')
    triaged = set(zip(triage.example,triage.state))
    islands = {r['state']:r for r in read(REV/'actual_island_boundary_results.json')}
    duplicate = pd.read_csv(REV/'island_duplicate_resolution_plan.csv', dtype={'example':str})
    dupmap = {(r.example,r.island_state):r.independent_state for r in duplicate.itertuples()}
    aliases = pd.read_csv(SOURCE/'island_alias_dependencies.csv', dtype={'example':str})
    aliasmap = {(r.example,r.island_state):r.source_states for r in aliases[aliases.relationship.eq('existing_source_alias')].itertuples()}
    source_inventories = sorted(ASSETS.glob('*/inventory.csv'))
    inventory = pd.concat([pd.read_csv(p,dtype={'example':str}) for p in source_inventories],ignore_index=True).fillna('')
    records = []
    for row in inventory.to_dict('records'):
        ex, family, state = row['example'],row['family'],row['state'];key = (ex,family,state);pair=(ex,state)
        record = dict(example=ex,family=family,state=state,parent_state=row['parent_state'],
                      original_manifest=row['manifest'],original_frame=row['frame'],original_fingerprint=row['fingerprint'],
                      geometry_status='retained',decision='unchanged_source',requires_hydraulic_regeneration=False,
                      separate_observation_excluded=False)
        a = audits.get(key)
        if a is not None:
            record.update(links=a.child_links_path,nodes=a.child_nodes_path)
        elif row['status']=='ready':
            paths=read(row['manifest'])['input_paths']
            record.update(links=paths['directed_links'],nodes=paths['directed_nodes'])
        if row['status']=='excluded_topology_review' or ex=='11':
            record.update(geometry_status='excluded',decision='existing_topology_exclusion' if ex!='11' else 'example11_excluded')
        elif family=='island' and pair in dupmap:
            record.update(canonical_key=[ex,'indep',dupmap[pair]],decision='duplicate_use_independent',separate_observation_excluded=True)
        elif family=='island' and pair in aliasmap:
            target=aliasmap[pair]
            candidates=[k for k in audits if k[0]==ex and k[2]==target and k[1]!='island']
            if len(candidates)!=1:raise ValueError(f'Ambiguous existing alias {pair}: {candidates}')
            record.update(canonical_key=list(candidates[0]),decision='existing_source_alias')
        elif family=='island' and state in islands:
            folder=ISLAND/state
            if (folder/'review_result.json').exists():
                if not read(folder/'review_result.json')['geometry_accepted']:raise ValueError(state)
                stage='island_reassembled'
            else:
                if not islands[state]['overall_geometry_pass']:raise ValueError(state)
                folder=REV/'island_boundary_trials'/state;stage='island_boundary_repaired'
            record.update(make_bundle(str(folder),stage));record.update(decision=stage,geometry_status='accepted')
        elif pair in fullmap:
            result=fullmap[pair];folder=SOURCE/'full'/ex/state;stage='automated_anchor_repair'
            record['historical_automatic_status']=result.status
            if pair in manual:
                m=manual[pair]
                if not m['reviewed_geometry_pass'] or m['matching_reintroduced_waived_nodes']:raise ValueError(pair)
                folder=REV/ex/state;stage='manual_boundary_revision'
            elif result.status not in ['repaired_anchor_pass','unchanged_anchor_pass']:
                if pair in accepted:stage='visually_accepted_boundary_interpretation'
                elif pair in triaged:stage='validated_connection_interpretation'
                else:raise ValueError(f'Unresolved geometry review {pair}')
                # Waivers concern role interpretation only, never missing identity.
                validation=read(folder/'validation.json')
                if any(not b['present'] for b in validation['boundaries']):
                    raise ValueError(f'Visual/triage acceptance cannot waive absent boundary {pair}')
            contract_path=folder/'contract.json'
            contract_id=digest(contract_path)
            frozen_contract=OUT/'contracts'/f'{contract_id}.json'
            frozen_contract.parent.mkdir(exist_ok=True)
            if not frozen_contract.exists():shutil.copy2(contract_path,frozen_contract)
            record.update(contract_path=str(frozen_contract),contract_sha256=contract_id)
            record['acceptance_basis'] = stage
            record['historical_exterior_exact'] = truth(result.exterior_exact)
            bundle=make_bundle(str(folder),stage)
            if bundle:record.update(bundle)
            record.update(decision=stage if bundle or stage!='automated_anchor_repair' else 'unchanged_anchor_pass',geometry_status='accepted',review_folder=str(folder))
        records.append(record)
    keyed={(r['example'],r['family'],r['state']):r for r in records}
    if len(keyed)!=len(records):raise ValueError('Duplicate logical records')
    for record in records:
        if record.get('canonical_key'):
            target=keyed[tuple(record['canonical_key'])]
            if target.get('canonical_key') or target['geometry_status']=='excluded':raise ValueError('Invalid canonical alias target')
            for name in ['bundle','geometry_id','requires_hydraulic_regeneration']:
                if name in target:record[name]=target[name]
            record['geometry_status']='canonical_alias'
    # Each object is immutable and copied once. Original large rasters/width
    # files are referenced, not duplicated. Routing is explicitly not activated.
    inputs = {str(p):digest(p) for p in source_inventories+[SOURCE/'full/state_results.csv',SOURCE/'boundary_manual_reviews.jsonl',REV/'manual_edit_results.json',REV/'connection_flag_triage.csv',REV/'island_duplicate_resolution_plan.csv',ISLAND/'summary.json']}
    manifest=dict(schema_version=1,release=OUT.name,records=records,input_checksums=inputs,
                  geometry_integration_complete=True,hydraulic_regeneration_complete=False,routing_ready=False,
                  tiny_island_absorption_retained={'island_41_099':['island_41_100']})
    write(OUT/'manifest.json',manifest)
    release=ReviewedGeometry(OUT/'manifest.json')
    for r in records:
        c=release.canonical(r)
        if c.get('bundle') and not Path(c['bundle']).is_dir():raise ValueError('Missing canonical object')
    summary=dict(logical_states=len(records),geometry_objects=len(list((OUT/'objects').glob('*/bundle.json'))),
                 decisions=pd.Series([r['decision'] for r in records]).value_counts().to_dict(),
                 duplicate_island_entries=sum(r['separate_observation_excluded'] for r in records),
                 existing_island_aliases=sum(r['decision']=='existing_source_alias' for r in records),
                 states_requiring_hydraulic_regeneration=sum(r['requires_hydraulic_regeneration'] and not r['separate_observation_excluded'] for r in records),
                 unresolved_geometry_reviews=0,legacy_geometry_files_overwritten=False,routing_ready=False,
                 remaining_hydraulic_issues=[x for x in read(REV/'summary.json')['remaining_work'] if not x.startswith(('Integrate reviewed','Apply the independent'))])
    write(OUT/'summary.json',summary)
    pd.DataFrame(records).to_csv(OUT/'state_registry.csv',index=False)
    for ex,group in pd.DataFrame(records).groupby('example'):
        d=OUT/ex;d.mkdir(exist_ok=True)
        group.to_csv(d/'geometry_registry.csv',index=False)
        # A preparation inventory makes pending attributes explicit. Keep the
        # original frame in original_frame for provenance, never in frame for a
        # changed geometry that has not been hydraulically regenerated.
        prepared=inventory[inventory.example.eq(ex)].copy()
        for idx,row in prepared.iterrows():
            r=keyed[(ex,row.family,row.state)];canonical=release.canonical(r)
            prepared.loc[idx,'geometry_release']=OUT.name
            prepared.loc[idx,'geometry_id']=canonical.get('geometry_id','')
            prepared.loc[idx,'original_frame']=row.frame
            prepared.loc[idx,'canonical_family']=canonical['family']
            prepared.loc[idx,'canonical_state']=canonical['state']
            prepared.loc[idx,'geometry_bundle']=canonical.get('bundle','')
            if r['geometry_status']=='excluded':
                prepared.loc[idx,'status']='excluded_topology_review'
            elif r['separate_observation_excluded']:
                prepared.loc[idx,'status']='excluded_duplicate'
                prepared.loc[idx,'frame']=''
            elif canonical['requires_hydraulic_regeneration']:
                prepared.loc[idx,'status']='pending_reviewed_geometry_hydraulics'
                prepared.loc[idx,'frame']=''
        prepared.to_csv(d/'inventory.csv',index=False)
    return summary


def activate():
    # Snapshot the previous pointer for rollback; replace only after all checks.
    validation=read(OUT/'validation.json')
    if validation.get('manifest_sha256') != digest(OUT/'manifest.json'):
        raise ValueError('Run validate_reviewed_geometry_integration.py on this release before activation.')
    ACTIVE.parent.mkdir(parents=True,exist_ok=True)
    if ACTIVE.exists():
        previous=OUT/'previous_active.json'
        if not previous.exists():shutil.copy2(ACTIVE,previous)
    pointer=dict(manifest=str(OUT/'manifest.json'),sha256=digest(OUT/'manifest.json'),
                 release=OUT.name,stage='geometry_integrated_hydraulics_pending')
    temp=ACTIVE.with_suffix('.tmp');write(temp,pointer);os.replace(temp,ACTIVE)
    return pointer


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--activate',action='store_true');args=parser.parse_args()
    summary=build()
    if args.activate:
        import subprocess
        subprocess.run([sys.executable,str(Path(__file__).with_name('validate_reviewed_geometry_integration.py'))],check=True)
        summary['activation']=activate()
    print(json.dumps(summary,indent=2))
