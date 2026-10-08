"""Confirm broad anchoring screen against original projected vector geometry."""
from pathlib import Path
import json
from functools import lru_cache
import numpy as np,pandas as pd,geopandas as gpd
from scipy.spatial import cKDTree
from shapely import normalize,to_wkb
from audit_all_anchoring import OUT,ids
@lru_cache(maxsize=16)
def base_vectors(lp,npth):
 l=gpd.read_file(lp,columns=['id_link','geometry']).set_index('id_link');n=gpd.read_file(npth,columns=['id_node','geometry']).set_index('id_node');return l,n
@lru_cache(maxsize=64)
def child_vectors(lp,npth):
 l=gpd.read_file(lp,columns=['id_link','geometry']).set_index('id_link');n=gpd.read_file(npth,columns=['id_node','geometry']).set_index('id_node');return l,n

def run():
 a=pd.read_csv(OUT/'transition_audit.csv',dtype={'example':str}).fillna('');rows=[];anchors=[];exterior=[]
 for ex,group in a.groupby('example',sort=False):
  for r in group.itertuples():
   d=dict(example=ex,family=r.family,state=r.state,fingerprint=r.fingerprint,geometry_error='')
   try:
    bl,bn=base_vectors(r.base_links_path,r.base_nodes_path);cl,cn=child_vectors(r.child_links_path,r.child_nodes_path)
    assert bl.crs==bn.crs==cl.crs==cn.crs and bl.crs.is_projected
    tree=cKDTree(np.array([(p.x,p.y) for p in cn.geometry]));selected=set(ids(r.selected_parent_link_ids));missing=set(ids(r.missing_boundary_nodes));boundary=ids(r.required_boundary_nodes)
    vm=json.loads(Path(r.variant_manifest).read_text()) if Path(r.variant_manifest).exists() else {}
    contract=vm.get('local_patch',{}).get('planning',{}).get('contract',{});cb=set(contract.get('anchored_upstream_boundary_node_ids',[]))|set(contract.get('anchored_downstream_boundary_node_ids',[]))
    for node in boundary:
     p=bn.loc[node].geometry;dist,ix=tree.query([p.x,p.y]);line_dist=float(cl.geometry.distance(p).min()) if node in missing else 0.
     anchors.append(dict(example=ex,family=r.family,state=r.state,fingerprint=r.fingerprint,parent_node_id=node,missing_canonical_node=node in missing,nearest_child_node_id=int(cn.index[ix]),nearest_child_node_distance_m=float(dist),nearest_child_line_distance_m=line_dist,explicit_contract_anchor=node in cb,classification=('retained' if node not in missing else 'coordinate_present_mapping_requires_review' if dist<=1e-5 else 'node_missing_line_passes_original_anchor' if line_dist<=1e-5 else 'node_and_original_line_location_not_retained')))
    hashes=set(to_wkb(normalize(cl.geometry.values)));absent=[]
    for lid in set(bl.index)-selected:
     geom=bl.loc[lid].geometry
     if to_wkb(normalize(geom)) not in hashes:
      # Equality allows a harmless change in vertex sampling on an identical line.
      near=cl.iloc[list(cl.sindex.query(geom,predicate='intersects'))]
      if not any(geom.equals(g) for g in near.geometry):absent.append(int(lid))
    exterior.extend(dict(example=ex,family=r.family,state=r.state,fingerprint=r.fingerprint,parent_link_id=lid) for lid in absent)
    d.update(exterior_geometry_not_retained_count=len(absent),exterior_geometry_not_retained_ids=','.join(map(str,sorted(absent))),explicit_contract_boundary_nodes=','.join(map(str,sorted(cb))),missing_explicit_contract_boundary_nodes=','.join(map(str,sorted(cb&missing))),contract_found=bool(contract),contract_boundary_matches_selected=cb==set(boundary),execution_backend=vm.get('execution_mode_used','unknown'))
   except Exception as e:d['geometry_error']=repr(e)
   rows.append(d)
  pd.DataFrame(rows).to_csv(OUT/'geometry_verification.csv',index=False);pd.DataFrame(anchors).to_csv(OUT/'boundary_geometry_details.csv',index=False);pd.DataFrame(exterior).to_csv(OUT/'exterior_geometry_details.csv',index=False)
  print(ex,'geometry verified',len(rows),'errors',sum(bool(x['geometry_error']) for x in rows),flush=True)
if __name__=='__main__':run()
