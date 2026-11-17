"""Independent checks of saved candidate artifacts, including full-network LP."""
from pathlib import Path
import sys,json
ROOT=Path(__file__).resolve().parents[3];sys.path[:0]=[str(ROOT),str(ROOT/'RAPID/investigations/member_boundary_review_20261027')]
import numpy as np,pandas as pd,geopandas as gpd
from scipy.optimize import linprog
from scipy.sparse import lil_matrix
from shapely.geometry import Point
from feasibility import head_series
from network_variants.reviewed_geometry import ReviewedGeometry
OUT=ROOT/'RAPID/outputs/accepted_boundary_trial_20261027_v4';release=ReviewedGeometry()
def main():
 rows=json.loads((OUT/'case_results.json').read_text());checks=[];cache={}
 for row in rows:
  ex,state=row['key'].split('/');dest=OUT/ex/state;f=gpd.read_parquet(dest/'geometry_candidate.parquet');bc=json.loads((dest/'boundary_contract.json').read_text())
  if ex not in cache:
   pr=release.lookup(ex,'indep','S000_base');p=pd.read_parquet(pr['original_frame']);cache[ex]=p,head_series(p),gpd.read_file(pr['links']).set_index('id_link')
  p,h,pg=cache[ex];restored=0
  for q in f[f.id_link<0].itertuples():
   ref=pg.loc[-q.id_link].geometry
   assert ref.hausdorff_distance(q.geometry)<1e-6 and abs(ref.length-q.geometry.length)<1e-6;restored+=1
  points={}
  for q in f.itertuples():
   for v,xy in [(q.u,q.geometry.coords[0]),(q.v,q.geometry.coords[-1])]:
    if v in points:assert Point(points[v]).distance(Point(xy))<1e-6
    points[v]=xy
  if row['status']!='solved':checks.append(dict(key=row['key'],geometry_checked=True,restored_reaches=restored,head_check='not solved; excluded'));continue
  f=pd.read_parquet(dest/'slope_candidate.parquet');heads=head_series(f);primary=json.loads((dest/'primary_head_fit.json').read_text());external=p[~p.id_link.isin(bc['approved_scope'])];fixed=set(external.u)|set(external.v)|set(bc['boundary_nodes']);released=set(primary['candidates']);frozen=fixed-released
  assert max(abs(heads.loc[list(frozen)]-h.loc[list(frozen)]),default=0)<1e-9
  np.testing.assert_allclose(f.slope,(f.head_us-f.head_ds)/f.link_length_m,rtol=1e-12,atol=1e-15)
  drop=np.maximum(f.floor.to_numpy()*f.link_length_m.to_numpy(),1e-10);actual=(f.head_us-f.head_ds).to_numpy();assert (actual>0).all() and np.max(drop-actual)<1e-8
  free=sorted(set(heads.index)-fixed);ix={v:i for i,v in enumerate(free)};A=lil_matrix((len(f),len(free)));b=-drop.copy()
  for j,q in enumerate(f.itertuples()):
   for v,s in [(q.u,1),(q.v,-1)]:
    if v in ix:A[j,ix[v]]=-s
    else:b[j]+=s*heads.loc[v]
  if free:
   result=linprog(np.zeros(len(free)),A_ub=A.tocsr(),b_ub=b,bounds=[(None,None)]*len(free),method='highs',options={'primal_feasibility_tolerance':1e-9});assert result.success,(row['key'],result.message)
  else:assert np.min(b)>=-1e-8
  checks.append(dict(key=row['key'],geometry_checked=True,restored_reaches=restored,full_network_lp_feasible=True,max_floor_deficit_m=float(max(0,np.max(drop-actual))),all_unreleased_exterior_heads_fixed=True))
 (OUT/'independent_verification.json').write_text(json.dumps(checks,indent=2));print('Verified',len(checks),'geometry candidates;',sum(c.get('full_network_lp_feasible',False) for c in checks),'full-network LP checks')
if __name__=='__main__':main()
