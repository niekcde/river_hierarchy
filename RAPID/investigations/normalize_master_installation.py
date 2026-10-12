"""Remove historical routing columns before publishing physical master tables."""
from integrate_master_networks import *

def main():
 plan=json.loads((STAGE/'installation_plan.json').read_text());actions={a['target']:a for a in plan['actions']}
 keep=['id_link','id_us_node','id_ds_node','u','v','link_length_m','width','wid_adj_wet','depth_m','kb_local','head_us','head_ds','slope','width_regeneration_method','geometry_id','geometry_release','hydraulic_head_policy']
 count=0
 correspondence_paths={}
 for o in json.loads((STAGE/'objects_index.json').read_text()):
  p=o['paths'];a=actions[p['frame']];source=Path(a['source']);f=pd.read_parquet(source)
  cols=[c for c in keep if c in f]
  if set(f.columns)!=set(cols):f[cols].to_parquet(source,index=False);count+=1
  a['sha256']=digest(source);a['bytes']=source.stat().st_size
  target=str(Path(p['manifest']).parent/'link_hydraulic_context.csv');a=actions[target];src=Path(a['source']);g=pd.read_csv(src)
  cols=[c for c in [*keep,'slope_used','slope_adjusted','slope_source_method','slope_reason','link_length_source','wse_us','wse_ds'] if c in g]
  if set(g.columns)!=set(cols):g[cols].to_csv(src,index=False)
  a['sha256']=digest(src);a['bytes']=src.stat().st_size
  if o['changed_geometry']:
   record=active_geometry().lookup(*o['key'])
   src=STAGE/'objects'/o['geometry_id']/'validated_node_correspondences.json'
   shutil.copy2(Path(record['bundle'])/'validated_node_correspondences.json',src)
   target=str(Path(p['manifest']).parent/src.name)
   actions[target]=dict(source=str(src),target=target,sha256=digest(src),bytes=src.stat().st_size,previous_sha256=digest(target) if Path(target).exists() else None,previous_bytes=Path(target).stat().st_size if Path(target).exists() else 0,sidecars={})
   correspondence_paths[o['geometry_id']]=target
 manifest_action=actions[str(MASTER/'reviewed_master_manifest.json')];mp=Path(manifest_action['source']);manifest=json.loads(mp.read_text())
 for r in manifest['records']:
  if r.get('separate_observation_excluded'):
   old=active_geometry().lookup(r['example'],r['family'],r['state'])
   target=str(Path(old['original_manifest']).resolve().parent.parent/'geometry_integration_status.json')
   src=STAGE/'metadata'/'retired_duplicates'/f"{r['example']}_{r['state']}.json"
   save(src,dict(status='historical_duplicate_not_an_active_observation',canonical_key=r['canonical_key'],canonical_hydraulic_manifest=r['original_manifest'],release=RELEASE))
   actions[target]=dict(source=str(src),target=target,sha256=digest(src),bytes=src.stat().st_size,previous_sha256=digest(target) if Path(target).exists() else None,previous_bytes=Path(target).stat().st_size if Path(target).exists() else 0,sidecars={})
  if r.get('geometry_id') in correspondence_paths:
   target=correspondence_paths[r['geometry_id']]
   r['validated_correspondences_path']=target
   r.setdefault('retained_source_checksums',{})[target]=actions[target]['sha256']
  for path in r.get('retained_source_checksums',{}):
   if path in actions:r['retained_source_checksums'][path]=actions[path]['sha256']
 save(mp,manifest);manifest_action['sha256']=digest(mp);manifest_action['bytes']=mp.stat().st_size
 plan['actions']=list(actions.values());plan['routing_parameter_columns_removed_from_retained_frames']=count
 save(STAGE/'installation_plan.json',plan)
 print('Normalized physical-only tables:',len(json.loads((STAGE/'objects_index.json').read_text())),'removed historic routing columns from',count,flush=True)

if __name__=='__main__':main()
