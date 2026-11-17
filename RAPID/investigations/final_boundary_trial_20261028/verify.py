from pathlib import Path
import sys,json,importlib.util
ROOT=Path(__file__).resolve().parents[3];sys.path[:0]=[str(ROOT),str(ROOT/'RAPID/investigations/member_boundary_review_20261027')]
import pandas as pd,geopandas as gpd
from shapely.geometry import Point
from shapely.ops import unary_union
from network_variants.reviewed_geometry import ReviewedGeometry
OUT=ROOT/'RAPID/outputs/final_boundary_trial_20261028';release=ReviewedGeometry()
def main():
 spec=importlib.util.spec_from_file_location('previous_verifier',ROOT/'RAPID/investigations/accepted_boundary_trial_20261027/verify.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);module.OUT=OUT;module.main()
 review=json.loads((OUT/'residual_decisions.json').read_text());bycase={c['key']:c['reaches'] for c in review['cases']};cases=json.loads((OUT/'case_results.json').read_text());checks=[];crossings=[];cache={}
 for row in cases:
  key=row['key'];ex,state=key.split('/');dest=OUT/key;before=gpd.read_parquet(dest/'before_residual_geometry.parquet').set_index('id_link');after=gpd.read_parquet(dest/'geometry_candidate.parquet').set_index('id_link');before=before.loc[after.index]
  if ex not in cache:cache[ex]=gpd.read_file(release.lookup(ex,'indep','S000_base')['links']).set_index('id_link')
  for z in bycase.get(key,[]):
   a=after.loc[z['child_link']].geometry;b=before.loc[z['child_link']].geometry
   if z['decision']=='Accept restoration trial':
    ref=cache[ex].loc[z['parent_link']].geometry;assert a.hausdorff_distance(ref)<1e-6 and abs(a.length-ref.length)<1e-6
   else:assert a.equals_exact(b,1e-9)
  changed=[i for i in after.index if not after.loc[i].geometry.equals_exact(before.loc[i].geometry,1e-9)];pairs=set();new_cross=[]
  for lid in changed:
   a=after.loc[lid];assert a.geometry.is_valid and a.geometry.length>0
   if not a.geometry.is_simple and before.loc[lid].geometry.is_simple:new_cross.append(dict(type='new_self_intersection',link=int(lid)))
   nearby=set(after.index[after.sindex.query(a.geometry)])|set(before.index[before.sindex.query(before.loc[lid].geometry)])
   for other in nearby:
    if other==lid:continue
    pair=tuple(sorted([int(lid),int(other)]))
    if pair in pairs:continue
    pairs.add(pair);b=after.loc[other];new=a.geometry.intersection(b.geometry);old=before.loc[lid].geometry.intersection(before.loc[other].geometry)
    if new.is_empty:continue
    shared={int(a.u),int(a.v)}&{int(b.u),int(b.v)};joints=[]
    for node in shared:joints.append(Point(a.geometry.coords[0] if node==a.u else a.geometry.coords[-1]).buffer(.001))
    allowed=unary_union(joints+[old.buffer(.001)])
    extra=new.difference(allowed)
    if not extra.is_empty:new_cross.append(dict(type='new_nonshared_intersection',links=list(pair),geometry_type=extra.geom_type,overlap_length_m=float(extra.length)))
  for c in new_cross:crossings.append(dict(key=key,**c))
  checks.append(dict(key=key,changed_geometry_links=len(changed),review_decisions_honoured=True,new_intersection_flags=len(new_cross)))
 (OUT/'residual_geometry_verification.json').write_text(json.dumps(checks,indent=2));(OUT/'new_intersection_flags.json').write_text(json.dumps(crossings,indent=2));print('Residual decisions honoured in',len(checks),'cases; new intersection flags:',len(crossings))
if __name__=='__main__':main()
