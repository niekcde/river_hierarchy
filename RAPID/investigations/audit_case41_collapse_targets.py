"""Compare declared collapse targets with raster additions and final anchors."""
from pathlib import Path
import json,hashlib
import numpy as np,pandas as pd,geopandas as gpd,rasterio
from rasterio.features import geometry_mask
from rasterio.windows import from_bounds
ROOT=Path(__file__).resolve().parents[2];R=Path('/Volumes/PhD/river_hierarchy/network_variants/output');OUT=ROOT/'RAPID/outputs/temporal_and_target_audit_20261008';B=R/'41_indep__anchored/states/S000_base/variant';isl=gpd.read_file(R/'41_island__anchored/setup/island_candidate_geometries.gpkg');isl=isl[isl.island_id.isin(['island_41_005','island_41_006'])];base=gpd.read_file(B/'widths/links_with_width_families.gpkg');region=base[base.id_link.isin([4,5,6,7,8])].geometry.union_all().buffer(200);paths={s:R/'41_indep__anchored/states'/s/'variant' for s in ['S016_unit_4','S076_unit_3']};paths.update({s:R/'41_island__anchored/island_reruns'/s/'local_patch/variant' for s in ['island_41_005','island_41_006']});declared={'S016_unit_4':['island_41_006'],'S076_unit_3':['island_41_005','island_41_006'],'island_41_005':['island_41_005'],'island_41_006':['island_41_006']};required={'S016_unit_4':[5,7],'S076_unit_3':[4,6],'island_41_005':[4,6],'island_41_006':[5,7]};reg=pd.read_csv('/Volumes/PhD/river_hierarchy/cunge_full_hierarchy_runs/41/state_run_registry.csv');records=[]
with rasterio.open(next((B/'mask').glob('*collapsed.tif'))) as src:
 w=from_bounds(*region.bounds,transform=src.transform).round_offsets().round_lengths();before=src.read(1,window=w)>0;tr=src.window_transform(w);bounds=src.window_bounds(w);crs=src.crs
for state,p in paths.items():
 m=json.loads((p/'summary/variant_manifest.json').read_text())
 with rasterio.open(m['output_paths']['collapsed_mask']) as src:
  assert src.crs==crs
  cw=from_bounds(*bounds,transform=src.transform).round_offsets().round_lengths();after=src.read(1,window=cw)>0;assert after.shape==before.shape;assert src.window_transform(cw).almost_equals(tr)
 added=after&~before;actual=[];partial=[];counts={}
 for ir in isl.itertuples():
  pix=geometry_mask([ir.geometry],before.shape,tr,invert=True)&~before;n=int(pix.sum());k=int((pix&added).sum());counts[ir.island_id]=dict(original_dry_pixels=n,added_pixels=k,coverage=k/n)
  if k==n:actual.append(ir.island_id)
  elif k:partial.append(ir.island_id)
 row=reg[reg.state.eq(state)].iloc[0];f=pd.read_parquet(row.frame);node_set=set(f.u)|set(f.v)
 records.append(dict(state=state,declared_target_islands=declared[state],actual_fully_filled_islands=sorted(actual),partially_filled_islands=partial,target_set_matches=sorted(actual)==sorted(declared[state]) and not partial,coverage=counts,section_added_pixels=int(added.sum()),section_removed_pixels=int((before&~after).sum()),section_added_mask_sha256=hashlib.sha256(added.tobytes()).hexdigest(),required_original_anchor_nodes=required[state],anchors_present_in_hydraulic_input=all(n in node_set for n in required[state]),missing_original_anchor_nodes=[n for n in required[state] if n not in node_set],hydraulic_fingerprint=row.fingerprint,known_graph_failure=state=='island_41_005',scope='full compound section raster window; not a full-network pixel-equivalence certificate'))
(OUT/'41_collapse_target_registry.json').write_text(json.dumps(records,indent=2));pd.DataFrame([{k:v for k,v in z.items() if k!='coverage'} for z in records]).to_csv(OUT/'41_collapse_target_registry.csv',index=False);print(json.dumps(records,indent=2))
