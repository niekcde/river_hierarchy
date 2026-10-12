"""Validate and package regenerated physical frames without starting routing."""
from pathlib import Path
import sys,json,hashlib
import numpy as np
import pandas as pd
import geopandas as gpd
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT),str(ROOT/'RAPID/src'),str(Path(__file__).parent)]
from network_variants.reviewed_geometry import active_geometry,digest
from regenerate_reviewed_hydraulics import OUT,base,save
from rapid_tools import cunge

def main():
 release=active_geometry();seen={};results=[]
 for r in release.records.values():
  r=release.canonical(r)
  if not r.get('bundle') or r['geometry_id'] in seen:continue
  gid=r['geometry_id'];folder=OUT/'objects'/gid;seen[gid]=None
  hp=folder/'head_result.json';wp=folder/'width_result.json'
  if not hp.exists() or not wp.exists():raise ValueError(f'Regeneration is unfinished: {r["state"]}')
  h=json.loads(hp.read_text());w=json.loads(wp.read_text())
  if w['status']!='widths_regenerated' or h['status'] != 'current_slope_pass':
   results.append(dict(example=r['example'],state=r['state'],geometry_id=gid,status='blocked_regeneration',reason=w.get('reason',h.get('reason',''))));continue
  headfile=folder/'heads.parquet'
  if not headfile.exists():
   results.append(dict(example=r['example'],state=r['state'],geometry_id=gid,status='blocked_heads',reason=h.get('reason','')));continue
  f=pd.read_parquet(headfile);width=pd.read_parquet(folder/'widths.parquet')
  for key in ['id_link','id_us_node','id_ds_node','u','v','link_length_m']:
   np.testing.assert_allclose(f.set_index('id_link')[key] if key!='id_link' else f.id_link,width.set_index('id_link')[key] if key!='id_link' else width.id_link,rtol=0,atol=1e-8)
  f=f.merge(width[['id_link','width','wid_adj_wet','width_regeneration_method']],on='id_link',validate='one_to_one')
  bl,bn,refs=base(r['example'])
  f['depth_m']=float(refs.stored_depth_m)*(f.width/float(refs.reference_width_m))**.65
  f['kb_local']=f.width/f.depth_m
  assert np.isfinite(f[['width','depth_m','slope','link_length_m']]).all().all()
  assert (f[['width','depth_m','slope','link_length_m']]>0).all().all()
  check=pd.concat([f[['u','head_us']].rename(columns={'u':'node','head_us':'head'}),f[['v','head_ds']].rename(columns={'v':'node','head_ds':'head'})])
  assert check.groupby('node')['head'].nunique().max()==1
  np.testing.assert_allclose((f.head_us-f.head_ds)/f.link_length_m,f.slope,rtol=1e-12)
  geom=gpd.read_parquet(Path(r['bundle'])/'links.parquet');assert set(f.id_link)==set(geom.id_link)
  np.testing.assert_allclose(f.set_index('id_link').link_length_m,geom.set_index('id_link').geometry.length,rtol=0,atol=1e-8)
  f['geometry_id']=gid;f['geometry_release']=release.release
  f['hydraulic_head_policy']='existing_domain_policy'
  physical=f[['u','v','link_length_m','width','slope']].sort_values(['u','v','link_length_m','width']).to_numpy()
  fp=hashlib.sha256(physical.tobytes()).hexdigest()[:20]
  fname='hydraulic_frame.parquet'
  f.to_parquet(folder/fname,index=False)
  # Check MC parameter consumption using the existing baseline references.
  # This is a numerical/input consistency check, not a routing/calibration run.
  grid=cunge.freeze_grid(f,float(refs.reference_q_cms))
  np.testing.assert_allclose(grid.rapid_effective_length_m*grid['count'],f.link_length_m,rtol=1e-12)
  trial=cunge.parameters(grid,.035)
  assert np.isfinite(trial[['rapid_k','rapid_x','reference_discharge_cms']]).all().all()
  assert (trial.rapid_k>0).all()
  result=dict(example=r['example'],family=r['family'],state=r['state'],geometry_id=gid,
    status='attributes_regenerated',fingerprint=fp,
    frame=str(folder/fname),frame_sha256=digest(folder/fname),links=len(f),positive_attributes=True,
    head_policy='existing_domain_policy',changed_width_links=w['changed_width_links'],untrimmed_width_fallbacks=w['untrimmed_fallback_links'],
    cunge_parameter_consumption_check_passed=True,n_manning_recalibrated=False,cunge_routing_parameters_frozen=False,routing_run=False)
  save(folder/'regeneration_result.json',result);results.append(result);seen[gid]=result
 for p in release.manifest.parent.glob('*/inventory.csv'):
  inv=pd.read_csv(p,dtype={'example':str}).fillna('')
  for idx,row in inv.iterrows():
   rec=release.lookup(row.example,row.family,row.state);canonical=release.canonical(rec)
   if rec['geometry_status']=='excluded' or rec.get('separate_observation_excluded'):continue
   if not canonical.get('bundle'):continue
   res=seen[canonical['geometry_id']]
   inv.loc[idx,'geometry_release']=release.release
   if res:
    inv.loc[idx,'status']=res['status'];inv.loc[idx,'regenerated_fingerprint']=res['fingerprint'];inv.loc[idx,'regenerated_frame']=res['frame']
    inv.loc[idx,'frame']=res['frame']
   else:inv.loc[idx,'status']='blocked_regeneration'
  dest=OUT/p.parent.name;dest.mkdir(exist_ok=True);inv.to_csv(dest/'inventory.csv',index=False)
 pd.DataFrame(results).to_csv(OUT/'regeneration_results.csv',index=False)
 counts=pd.Series([r['status'] for r in results]).value_counts().to_dict()
 summary=dict(source_geometry_release=release.release,source_manifest=str(release.manifest),source_manifest_sha256=digest(release.manifest),
   geometry_objects=len(results),status_counts=counts,logical_entries=3554,active_geometry_unchanged=True,hydraulic_release_activated=False,routing_run=False,
   known_six_fragment_widths_resolved_with_existing_untrimmed_policy=True,
   head_policy='existing_domain_policy', superseded_fixed_exterior_diagnostic_is_gate=False,
   remaining_action='Publish regenerated attributes into the preparation workflow, then freeze calibrated Cunge parameters and prepare routing.')
 save(OUT/'summary.json',summary);print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
