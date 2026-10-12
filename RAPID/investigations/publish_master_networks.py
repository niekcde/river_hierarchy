"""Prepare a reviewable installation manifest, then install it with rollback."""
from integrate_master_networks import *
import pyogrio

@lru_cache(maxsize=4)
def sample_table(path):
 p=Path(path)
 return pd.read_parquet(p) if p.suffix=='.parquet' else pd.read_csv(p,low_memory=False)

def stage_samples(o, record):
 from network_variants.width_samples import resolve_width_samples_path
 d=STAGE/'objects'/o['geometry_id'];widths=json.loads((HYD/'objects'/o['geometry_id']/'width_result.json').read_text())
 if not widths['changed_width_links']:return None
 target=resolve_width_samples_path(Path(record['links']).parent).resolve();out=d/target.name
 if out.exists():return out,target
 geo=gpd.read_parquet(Path(record['bundle'])/'links.parquet')
 old=gpd.read_file(record['links']);parent=active_geometry().lookup(record['example'],'indep','S000_base');par=gpd.read_file(parent['links'])
 oldmap={g.normalize().wkb:int(i) for i,g in zip(old.id_link,old.geometry)};parmap={g.normalize().wkb:int(i) for i,g in zip(par.id_link,par.geometry)}
 changed=set(widths['changed_width_links']);parts=[pd.read_parquet(HYD/'objects'/o['geometry_id']/'changed_width_samples.parquet')]
 sourcegroups={'parent':{},'child':{}}
 for row in geo.itertuples():
  if row.id_link in changed:continue
  key=row.geometry.normalize().wkb
  kind,src=('parent',parmap[key]) if key in parmap else ('child',oldmap[key])
  sourcegroups[kind].setdefault(src,[]).append(int(row.id_link))
 for kind,path in [('parent',resolve_width_samples_path(Path(parent['links']).parent)),('child',target)]:
  table=sample_table(str(path))
  for src,ids in sourcegroups[kind].items():
   part=table[table.id_link.eq(src)]
   for lid in ids:
    x=part.copy();x['id_link']=lid;parts.append(x)
 result=pd.concat(parts,ignore_index=True);assert set(result.id_link)<=set(geo.id_link)
 assert changed<=set(result.id_link)
 if out.suffix=='.parquet':result.to_parquet(out,index=False)
 else:result.to_csv(out,index=False)
 return out,target

def finalize(objects_only=False):
 release=active_geometry()
 source=[json.loads(p.read_text()) for p in (STAGE/'objects').glob('*/staged.json')] if objects_only else json.loads((STAGE/'objects_index.json').read_text())
 objects={tuple(o['key']):o for o in source}
 actions={};new_records=[];inventory=[]
 def action(source,target):
  source=Path(source);target=safe_target(target);sha=digest(source)
  if str(target) in actions:
   assert actions[str(target)]['sha256']==sha, target
  actions[str(target)]=dict(source=str(source),target=str(target),sha256=sha)
 def stage_json(name,target,data):
  p=STAGE/'metadata'/name;save(p,data);action(p,target)
 for i,(key,o) in enumerate(objects.items()):
  d=STAGE/'objects'/o['geometry_id'];f=pd.read_parquet(d/'hydraulic_frame.parquet')
  if not o['changed_geometry']:
   existing=pyogrio.read_dataframe(o['paths']['links'],columns=['id_link','wid_adj_routing','id_us_node','id_ds_node'],read_geometry=False).set_index('id_link').loc[f.id_link]
   np.testing.assert_allclose(existing.wid_adj_routing,f.width,rtol=0,atol=0)
   assert np.array_equal(existing.id_us_node,f.id_us_node) and np.array_equal(existing.id_ds_node,f.id_ds_node)
  # Refresh endpoint lineage where reviewed anchoring changes node identities.
  updated=d/'links.gpkg'
  if o['changed_geometry'] and not (d/'lineage_standardized.json').exists():
   lf=gpd.read_file(d/'links.gpkg');parent_ids=set(base_nodes(key[0]).index)
   lf['matched_parent_us_node']=lf.u.where(lf.u.isin(parent_ids))
   lf['matched_parent_ds_node']=lf.v.where(lf.v.isin(parent_ids))
   if o['changed_geometry']:
    ids=json.loads((HYD/'objects'/o['geometry_id']/'width_result.json').read_text())['changed_width_links'];sel=lf.id_link.isin(ids)
    lf.loc[sel,'lineage_type']='reviewed_geometry_replacement'
    lf.loc[sel,'lineage_method']='reviewed_source_geometry_and_parent_identities'
    for col in lf.columns:
     if 'overlap_fraction' in col or 'overlap_length' in col or col=='matched_parent_node_path':lf.loc[sel,col]=None
   temp=d/'updated_links.gpkg';lf.to_file(temp,layer='links',driver='GPKG',index=False);os.replace(temp,updated)
   columns=[c for c in lf if c.startswith(('matched_parent','dominant_parent','lineage','source_link','width_regeneration'))]
   lf[['id_link',*columns]].to_csv(d/'link_lineage.csv',index=False)
   save(d/'lineage_standardized.json',dict(complete=True))
  # Supply the established node-context column names using current topology.
  nf=gpd.read_file(d/'nodes.gpkg');lc=pd.read_csv(d/'link_hydraulic_context.csv')
  from network_variants.hydraulic_context import _compute_network_dist_out
  distances=_compute_network_dist_out(gpd.GeoDataFrame(lc),nf,lengths=lc.link_length_m).set_index('id_node')
  nc=pd.read_csv(d/'node_hydraulic_context.csv')
  for col in ['node_network_dist_out_m','node_network_dist_source']:nc[col]=nc.id_node.map(distances[col])
  for src,dest in [('sword_wse','node_wse_raw'),('sword_wse_field','node_wse_field'),('sword_wse_fill_method','node_wse_fill_method'),('sword_node_id','node_sword_node_id'),('sword_dist_out','node_sword_dist_out')]:
   nc[dest]=nc[src] if src in nc else np.nan
  reference=base(key[0])[1].set_index('id_node')
  nc['node_reference_section_dist_out_m']=nc.parent_identity.map(reference.reference_section_dist_out_m)
  nc['node_reference_section_dist_source']='reviewed_parent_identity'
  nc.to_csv(d/'node_hydraulic_context.csv',index=False)
  metadata=json.loads((d/'hydraulic_context_manifest.json').read_text())
  refs=base(key[0])[2]
  metadata['section_slope_summary']=dict(slope=float(refs.new_global_slope),source_method='bounded_robust_reference_profile',profile_mode='reference_section')
  metadata['head_concepts']=dict(observed_node_wse='node_wse_raw',derived_heads='rapid_wse_for_slope',slope_policy='existing_domain_policy')
  save(d/'hydraulic_context_manifest.json',metadata)
  # Retain GeoPackage layer names so existing QGIS references still open.
  for a in o['actions']:
   src=Path(a['source']);target=Path(a['target'])
   if src.name=='links.gpkg':src=updated
   if src.suffix=='.gpkg':
    if not o['changed_geometry']:
     continue
    expected=pyogrio.list_layers(target)[0][0]
    out=d/(target.stem+'.gpkg')
    if not out.exists():
     data=gpd.read_file(src)
     if str(target)==o['paths']['directed_links']:
      cols=[c for c in data if not c.startswith(('wid_pix','width_','qc_','opposition_','comparison_','adaptive_','routing_context_')) and c not in ['idx_link','cldists','clangs']]
      data=data[cols]
     data.to_file(out,layer=expected,driver='GPKG',index=False)
    assert pyogrio.list_layers(out)[0][0]==expected
    action(out,target)
   else:action(src,target)
  if o['changed_geometry']:
   extra=stage_samples(o,release.records[key])
   if extra:action(*extra)
  # Historic auxiliary artifacts remain evidence, with explicit current-state metadata.
  state=Path(o['paths']['manifest']).parent.parent
  for rel in ['variant/summary/variant_manifest.json','variant/directed/direction_validation_report.json','variant/widths/width_manifest.json','variant/local_patch_variant_manifest.json']:
   target=state/rel
   if target.exists():
    data=json.loads(target.read_text());data['master_integration']=dict(release=RELEASE,geometry_id=o['geometry_id'],hydraulic_context=o['paths']['manifest'],previous_diagnostics_historical=True)
    if 'n_links' in data:data['n_links']=o['n_links']
    if isinstance(data.get('output_paths'),dict):
     for field,value in [('directed_links',o['paths']['directed_links']),('directed_nodes',o['paths']['nodes']),('links_with_width_families',o['paths']['links'])]:data['output_paths'][field]=value
    stage_json(f"objects/{o['geometry_id']}/{rel.replace('/','_')}",target,data)
  if (i+1)%100==0:print('finalized objects',i+1,flush=True)
 if objects_only:
  print('COMPATIBLE OBJECT FILES COMPLETE',len(objects),flush=True);return
 # Prepare records for a master-local resolver, without references to staging products.
 for old in release.records.values():
  r=dict(old);canonical=release.canonical(old);key=(canonical['example'],canonical['family'],canonical['state'])
  if old['geometry_status']=='excluded':new_records.append(r);continue
  o=objects[key];paths=o['paths'];r['source_bundle']=r.pop('bundle',None)
  r.update(geometry_id=o['geometry_id'],links=paths['links'],nodes=paths['nodes'],original_frame=paths['frame'],original_manifest=paths['manifest'],original_fingerprint=o['fingerprint'],master_products_published=True)
  r.pop('retained_source_checksums',None)
  r['retained_source_checksums']={p:(actions[p]['sha256'] if p in actions else digest(p)) for p in [paths['links'],paths['nodes'],paths['frame'],str(Path(paths['manifest']).parent/'link_hydraulic_context.csv'),str(Path(paths['manifest']).parent/'node_hydraulic_context.csv')]}
  cp=str(Path(paths['manifest']).parent/'reviewed_boundary_contract.json')
  if cp in actions:r['contract_path']=cp;r['contract_sha256']=actions[cp]['sha256']
  elif r.get('contract_path'):
   target=Path(paths['manifest']).parent/'reviewed_boundary_contract.json';action(r['contract_path'],target);r['contract_path']=str(target);r['contract_sha256']=actions[str(target)]['sha256']
  new_records.append(r)
 # Inventory retains all logical states, while physical files follow canonical states.
 for old in release.records.values():
  canonical=release.canonical(old);key=(canonical['example'],canonical['family'],canonical['state']);o=objects.get(key)
  excluded=old['geometry_status']=='excluded';duplicate=old.get('separate_observation_excluded',False)
  row=dict(example=old['example'],family=old['family'],state=old['state'],parent_state=old.get('parent_state','S000_base'),
    status='excluded_topology_review' if excluded else 'excluded_duplicate' if duplicate else 'ready',geometry_release=RELEASE,
    geometry_id=o['geometry_id'] if o else '',fingerprint=o['fingerprint'] if o else '',frame=o['paths']['frame'] if o else '',
    manifest=o['paths']['manifest'] if o else '',context_alias=bool(old.get('canonical_key')),canonical_family=canonical['family'],canonical_state=canonical['state'])
  inventory.append(row)
 inv=pd.DataFrame(inventory);inv.to_csv(STAGE/'hydraulic_inventory.csv',index=False);action(STAGE/'hydraulic_inventory.csv',MASTER/'hydraulic_inventory.csv')
 registry_checks=[]
 for (ex,family),group in inv.groupby(['example','family']):
  suffix='_direction130' if ex=='15' else '';folder=MASTER/f'{ex}_{family}{suffix}__anchored'
  assert folder.is_dir(),folder
  group.to_csv(STAGE/f'inventory_{ex}_{family}.csv',index=False);action(STAGE/f'inventory_{ex}_{family}.csv',folder/'hydraulic_inventory.csv')
  bystate=group.set_index('state')
  names=['state_registry.csv'] if family!='island' else ['island_hydraulic_context_registry.csv','island_observation_manifest.csv','island_analysis_manifest.csv','island_analysis_manifest_strict.csv']
  for name in names:
   target=folder/name
   if not target.exists():continue
   table=pd.read_csv(target,low_memory=False).fillna('');idcol='state_id' if family!='island' else 'island_id'
   for idx,row in table.iterrows():
    state=str(row[idcol])
    if state not in bystate.index:continue
    entry=bystate.loc[state];canonical=release.canonical(release.lookup(ex,family,state));o=objects.get((canonical['example'],canonical['family'],canonical['state']))
    for col,val in [('geometry_release',RELEASE),('geometry_id',entry.geometry_id),('hydraulic_frame_path',entry.frame),('hydraulic_fingerprint',entry.fingerprint),('hydraulic_status',entry.status),('canonical_family',entry.canonical_family),('canonical_state',entry.canonical_state),('rapid_preparation_status','requires_regeneration'),('routing_results_status','historical')]:table.loc[idx,col]=val
    if not o:
     if family!='island':table.loc[idx,'status']='excluded_topology_review'
     continue
    p=o['paths'];h=Path(p['manifest']).parent;sd=h.parent;variant=sd/'variant'
    if family!='island':
     for col,val in [('directed_links_path',p['directed_links']),('analysis_links_path',p['links']),('directed_nodes_path',p['nodes']),('variant_output_dir',str(variant)),('hydraulic_context_output_dir',str(h))]:table.loc[idx,col]=val
     if o['changed_geometry']:
      table.loc[idx,'hierarchy_status']='requires_refresh'
      table.loc[idx,'equivalent_metrics_status']='requires_refresh'
    elif name=='island_hydraulic_context_registry.csv':
     for col,val in [('hydraulic_context_dir',str(h)),('link_hydraulic_context_path',str(h/'link_hydraulic_context.csv')),('node_hydraulic_context_path',str(h/'node_hydraulic_context.csv')),('hydraulic_context_manifest_path',p['manifest'])]:table.loc[idx,col]=val
     table.loc[idx,'hydraulic_context_status']='excluded_from_primary_analysis' if entry.status.startswith('excluded') else 'ready_reference' if entry.context_alias else 'ready_generated'
     table.loc[idx,'hydraulic_context_action']='excluded_duplicate_use_independent' if entry.status=='excluded_duplicate' else 'reference_existing_independent_state' if entry.context_alias else 'master_integrated'
    else:
     for col,val in [('analysis_state_dir',str(sd)),('canonical_state_dir',str(sd)),('analysis_variant_dir',str(variant)),('analysis_directed_links_path',p['directed_links']),('analysis_directed_nodes_path',p['nodes']),('analysis_routing_width_links_path',p['links']),('analysis_product_manifest_path',p['manifest'])]:table.loc[idx,col]=val
     if entry.context_alias:
      table.loc[idx,'source_state_dir']=str(sd);table.loc[idx,'execution_action']='alias_existing_unit_state';table.loc[idx,'analysis_backend']='independent_unit_alias'
      for col in ['local_patch_variant_dir','local_patch_directed_links_path','local_patch_directed_nodes_path','local_patch_routing_width_links_path','local_patch_product_manifest_path']:
       if col in table:table.loc[idx,col]=''
     table.loc[idx,'observation_status']='excluded_duplicate' if entry.status=='excluded_duplicate' else 'ready_reference' if entry.context_alias else 'ready_master_integrated'
     table.loc[idx,'analysis_execution_status']='excluded_duplicate' if entry.status=='excluded_duplicate' else 'ready_alias_reference' if entry.context_alias else 'ready_direct_local_patch'
     table.loc[idx,'separate_observation_excluded']=entry.status=='excluded_duplicate'
     if entry.status=='excluded_duplicate':
      table.loc[idx,'equivalence_status']='reviewed_same_target_island_use_independent'
      if 'strict_analysis_status' in table:table.loc[idx,'strict_analysis_status']='exclude_duplicate_use_independent'
   out=STAGE/'registries'/folder.name/name;out.parent.mkdir(parents=True,exist_ok=True);table.to_csv(out,index=False);action(out,target)
   registry_checks.append(dict(path=str(target),rows=len(table),registered_states=int(table[idcol].isin(bystate.index).sum())))
   sidecar=target.with_suffix('.json')
   if sidecar.exists():
    data=json.loads(sidecar.read_text())
    if 'n_included' in data:data['n_included']=int(table.hydraulic_status.eq('ready').sum())
    if 'n_excluded' in data:data['n_excluded']=int(table.hydraulic_status.astype(str).str.startswith('excluded').sum())
    data['n_excluded_duplicates']=int(table.hydraulic_status.eq('excluded_duplicate').sum())
    data['master_integration']=dict(release=RELEASE,registry_csv=str(target),hydraulic_inventory=str(folder/'hydraulic_inventory.csv'),previous_execution_statistics_historical=True)
    stage_json('registries/'+folder.name+'/'+sidecar.name,sidecar,data)
  stage_json(f'registries/{folder.name}/master_integration.json',folder/'master_integration.json',dict(release=RELEASE,hydraulic_inventory=str(folder/'hydraulic_inventory.csv'),attributes_current=True,rapid_preparation='pending_separate_task',previous_routing_results='historical',previous_hierarchy_metrics='refresh_required_for_changed_geometry'))
 manifest=dict(schema_version='master-network-release-v1',release=RELEASE,records=new_records,geometry_integration_complete=True,hydraulic_regeneration_complete=True,routing_ready=False,master_root=str(MASTER),supersedes=str(release.manifest),supersedes_sha256=digest(release.manifest),tiny_island_absorption_retained=True)
 stage_json('master_manifest.json',MASTER/'reviewed_master_manifest.json',manifest)
 # Capture the exact pre-installation file contents for drift detection and rollback.
 for i,a in enumerate(actions.values()):
  target=Path(a['target']);a['previous_sha256']=digest(target) if target.exists() else None;a['previous_bytes']=target.stat().st_size if target.exists() else 0
  a['bytes']=Path(a['source']).stat().st_size
  a['sidecars']={str(target)+suffix:digest(str(target)+suffix) for suffix in ['-wal','-shm'] if target.suffix=='.gpkg' and Path(str(target)+suffix).exists()}
  if (i+1)%1000==0:print('prepared file actions',i+1,flush=True)
 plan=dict(release=RELEASE,master_root=str(MASTER),backup_root=str(BACKUP),physical_states=len(objects),repaired_geometries=sum(o['changed_geometry'] for o in objects.values()),logical_states=len(inv),status_counts=inv.status.value_counts().to_dict(),actions=list(actions.values()),registry_checks=registry_checks)
 save(STAGE/'installation_plan.json',plan)
 print(json.dumps({k:v for k,v in plan.items() if k not in ['actions','registry_checks']},indent=2),flush=True)
 print('FILES',len(actions),'NEW BYTES',sum(a['bytes'] for a in actions.values()),'BACKUP BYTES',sum(a['previous_bytes'] for a in actions.values()),flush=True)


def publish():
 plan=json.loads((STAGE/'installation_plan.json').read_text());assert plan['master_root']==str(MASTER)
 assert not (BACKUP/'completion.json').exists(),'Already published'
 # Fully verify staged files and source drift before the first external write.
 for a in plan['actions']:
  target=safe_target(a['target']);assert digest(a['source'])==a['sha256']
  assert (digest(target) if target.exists() else None)==a['previous_sha256'],f'Source changed: {target}'
  for side,sha in a.get('sidecars',{}).items():assert digest(side)==sha,f'SQLite sidecar changed: {side}'
 BACKUP.mkdir(parents=True,exist_ok=True);save(BACKUP/'installation_plan.json',plan)
 # Back up every file before replacing any; verify each backup.
 for i,a in enumerate(plan['actions']):
  target=Path(a['target']);backup=BACKUP/'files'/target.relative_to(MASTER)
  if target.exists():
   backup.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(target,backup);assert digest(backup)==a['previous_sha256']
  # Preserve SQLite sidecars separately, then remove them at installation.
  if target.suffix=='.gpkg':
   for suffix in ['-wal','-shm']:
    side=Path(str(target)+suffix)
    if side.exists():shutil.copy2(side,Path(str(backup)+suffix))
  if (i+1)%1000==0:print('backed up',i+1,flush=True)
 written=[]
 try:
  for i,a in enumerate(plan['actions']):
   target=Path(a['target']);target.parent.mkdir(parents=True,exist_ok=True)
   assert (digest(target) if target.exists() else None)==a['previous_sha256'],f'Source changed during publication: {target}'
   for side,sha in a.get('sidecars',{}).items():assert digest(side)==sha,f'SQLite sidecar changed during publication: {side}'
   temp=target.with_name(target.name+'.integration-tmp');shutil.copy2(a['source'],temp)
   if target.suffix=='.gpkg':
    for suffix in ['-wal','-shm']:Path(str(target)+suffix).unlink(missing_ok=True)
   os.replace(temp,target);written.append(a);assert digest(target)==a['sha256']
   if (i+1)%1000==0:print('installed',i+1,flush=True)
 except BaseException:
  for a in reversed(written):
   target=Path(a['target']);backup=BACKUP/'files'/target.relative_to(MASTER)
   if a['previous_sha256'] is None:target.unlink(missing_ok=True)
   else:shutil.copy2(backup,target)
   if target.suffix=='.gpkg':
    for suffix in ['-wal','-shm']:
     b=Path(str(backup)+suffix)
     if b.exists():shutil.copy2(b,Path(str(target)+suffix))
  save(BACKUP/'rollback.json',dict(rolled_back=True,written_files=len(written)));raise
 save(BACKUP/'completion.json',dict(release=RELEASE,installed_files=len(written),installed_bytes=sum(a['bytes'] for a in written),all_checksums_verified=True))
 print('INSTALLATION COMPLETE',len(written),flush=True)

def rollback():
 plan=json.loads((BACKUP/'installation_plan.json').read_text())
 assert (BACKUP/'installation_plan.json').exists(), 'No publication backup to roll back'
 for a in plan['actions']:
  target=safe_target(a['target']);actual=digest(target) if target.exists() else None
  assert actual in [a['sha256'],a['previous_sha256']],f'Product changed after publication: {target}'
  if a['previous_sha256'] is not None:assert digest(BACKUP/'files'/target.relative_to(MASTER))==a['previous_sha256']
 for a in reversed(plan['actions']):
  target=Path(a['target']);backup=BACKUP/'files'/target.relative_to(MASTER)
  if a['previous_sha256'] is None:target.unlink(missing_ok=True)
  else:shutil.copy2(backup,target)
  for suffix in ['-wal','-shm']:
   old=Path(str(backup)+suffix)
   if old.exists():shutil.copy2(old,Path(str(target)+suffix))
 pointer=ROOT/'network_variants/config/reviewed_geometry.json'
 if (STAGE/'previous_active_pointer.json').exists():shutil.copy2(STAGE/'previous_active_pointer.json',pointer)
 save(BACKUP/'rollback.json',dict(rolled_back=True,restored_files=len(plan['actions'])))
 print('MASTER ROLLBACK COMPLETE',flush=True)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('action',choices=['finalize','publish','prepare-objects','rollback']);a=p.parse_args()
 if a.action=='publish':publish()
 elif a.action=='rollback':rollback()
 else:finalize(objects_only=a.action=='prepare-objects')
