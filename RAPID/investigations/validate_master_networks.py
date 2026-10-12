"""Read installed master products and registries before activating the pointer."""
from integrate_master_networks import *
from network_variants.reviewed_geometry import ReviewedGeometry, _cached_digest
from types import SimpleNamespace

def main():
 plan=json.loads((STAGE/'installation_plan.json').read_text());release=ReviewedGeometry(MASTER/'reviewed_master_manifest.json')
 for a in plan['actions']:
  stat=Path(a['target']).stat();assert _cached_digest(a['target'],stat.st_mtime_ns,stat.st_size)==a['sha256'],a['target']
 inv=pd.read_csv(MASTER/'hydraulic_inventory.csv',dtype={'example':str}).fillna('')
 assert len(inv)==3554;release.assert_hydraulics_current(inv)
 objects=json.loads((STAGE/'objects_index.json').read_text());total=0
 for i,o in enumerate(objects):
  r=release.lookup(*o['key']);p=o['paths'];f=pd.read_parquet(p['frame'])
  links=gpd.read_file(p['links'],columns=['id_link','id_us_node','id_ds_node','wid_adj_routing']);nodes=gpd.read_file(p['nodes'],columns=['id_node']);context=pd.read_csv(Path(p['manifest']).parent/'link_hydraulic_context.csv')
  assert set(links.id_link)==set(f.id_link)==set(context.id_link)
  l=links.set_index('id_link').loc[f.id_link];h=context.set_index('id_link').loc[f.id_link]
  assert np.array_equal(l.id_us_node,f.id_us_node) and np.array_equal(l.id_ds_node,f.id_ds_node)
  np.testing.assert_allclose(l.wid_adj_routing,f.width,atol=0,rtol=0);np.testing.assert_allclose(h.slope_used,f.slope,rtol=1e-12)
  np.testing.assert_allclose(l.geometry.length,f.link_length_m,atol=1e-5,rtol=1e-8)
  heads=pd.read_csv(Path(p['manifest']).parent/'node_hydraulic_context.csv').set_index('id_node').rapid_wse_for_slope
  np.testing.assert_allclose(f.id_us_node.map(heads),f.head_us,atol=1e-12,rtol=1e-12)
  np.testing.assert_allclose(f.id_ds_node.map(heads),f.head_ds,atol=1e-12,rtol=1e-12)
  assert (f.slope>0).all() and np.isfinite(f.slope).all()
  release.assert_preparation_current(SimpleNamespace(directed_nodes_path=Path(p['nodes']),link_widths_path=Path(p['links']),hydraulic_context_output_dir=Path(p['manifest']).parent,geometry_release=RELEASE,geometry_id=o['geometry_id']))
  total+=len(f)
  if (i+1)%100==0:print('validated master objects',i+1,flush=True)
 # Exercise the existing registry resolver, not only the new aggregate inventory.
 from rapid_tools.registry import iter_preparable_states
 contexts=0
 for entry in plan['registry_checks']:
  path=Path(entry['path']);table=pd.read_csv(path)
  if path.name!='state_registry.csv':continue
  for context in iter_preparable_states(table):
   release.assert_preparation_current(context);contexts+=1
   assert context.link_widths_path.exists() and context.directed_nodes_path.exists()
 # The reviewed duplicate island resolves to independent unit 4 in ordinary files.
 island=pd.read_csv(MASTER/'41_island__anchored/island_hydraulic_context_registry.csv')
 row=island[island.island_id.eq('island_41_006')].iloc[0]
 assert row.hydraulic_context_status=='excluded_from_primary_analysis'
 assert 'S016_unit_4' in row.hydraulic_context_dir
 result=dict(release=RELEASE,all_installed_checksums_match=True,physical_states=len(objects),logical_states=len(inv),link_rows_checked=total,normal_registry_contexts=contexts,
             alias_and_exclusion_checks_passed=True,geometry_hydraulics_consistent=True,positive_slopes=True,ready_for_separate_rapid_preparation=True,routing_run=False)
 save(STAGE/'validation.json',result);print(json.dumps(result,indent=2),flush=True)
 # Activation is the last, reversible local step after installed validation.
 pointer=ROOT/'network_variants/config/reviewed_geometry.json'
 if not (STAGE/'previous_active_pointer.json').exists():shutil.copy2(pointer,STAGE/'previous_active_pointer.json')
 target=MASTER/'reviewed_master_manifest.json'
 temporary=pointer.with_suffix('.integration.tmp')
 save(temporary,dict(manifest=str(target),sha256=digest(target),release=RELEASE,stage='master_geometry_and_hydraulics_integrated_rapid_prep_pending'))
 os.replace(temporary,pointer)
 save(STAGE/'activation.json',dict(active=True,master_manifest=str(target),backup_root=str(BACKUP),validation=str(STAGE/'validation.json')))
 print('MASTER ACTIVATED',flush=True)

if __name__=='__main__':main()
