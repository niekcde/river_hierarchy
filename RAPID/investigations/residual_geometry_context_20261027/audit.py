from pathlib import Path
import sys,json
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
import pandas as pd,geopandas as gpd,pyogrio
from shapely.geometry import Point
from network_variants.reviewed_geometry import ReviewedGeometry
BASE=ROOT/'RAPID/outputs/accepted_boundary_trial_20261027_v4';OUT=ROOT/'RAPID/outputs/residual_geometry_context_20261027';OUT.mkdir(exist_ok=False)
r=ReviewedGeometry();data=pd.read_csv(BASE/'remaining_exterior_geometry_changes.csv');results=[];cache={}
for key,group in data.groupby('key'):
 ex,state=key.split('/');dest=BASE/ex/state;f=gpd.read_parquet(dest/'geometry_candidate.parquet').set_index('id_link');bc=json.loads((dest/'boundary_contract.json').read_text());boundary=set(bc['boundary_nodes'])
 if ex not in cache:
  p=r.lookup(ex,'indep','S000_base');fr=pd.read_parquet(p['original_frame']).set_index('id_link');ns=pyogrio.read_dataframe(p['nodes']).set_index('id_node');mp=dict(zip(fr.u,fr.id_us_node))|dict(zip(fr.v,fr.id_ds_node));cache[ex]=fr,{v:ns.loc[raw].geometry for v,raw in mp.items()}
 parent,points=cache[ex]
 for row in group.itertuples():
  q=f.loc[row.child_link];moved=[]
  for node,xy in [(int(q.u),q.geometry.coords[0]),(int(q.v),q.geometry.coords[-1])]:
   d=Point(xy).distance(points[node])
   if d>1e-6:moved.append(dict(node=node,displacement_m=d,selected_boundary=node in boundary))
  results.append(dict(key=key,parent_link=int(row.parent_link),child_link=int(row.child_link),length_delta_m=float(row.length_delta_m),hausdorff_m=float(row.hausdorff_m),moved_endpoints=moved,all_moved_endpoints_are_boundaries=bool(moved) and all(v['selected_boundary'] for v in moved),solved=key!='37/S289_unit_143'))
(OUT/'endpoint_audit.json').write_text(json.dumps(results,indent=2));print('reach instances',len(results),'solved',sum(z['solved'] for z in results),'all moved endpoints boundaries',sum(z['all_moved_endpoints_are_boundaries'] for z in results))
for z in results:
 if not z['all_moved_endpoints_are_boundaries']:print('OTHER',z)
print('boundary occurrences',sum(len(z['moved_endpoints']) for z in results),'unique case/node',len({(z['key'],v['node']) for z in results for v in z['moved_endpoints']}))
