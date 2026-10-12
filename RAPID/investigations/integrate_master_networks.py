"""Stage, verify and transactionally install reviewed products into the master tree.

No RAPID preparation or routing is performed. All overwritten files are backed up.
"""
from pathlib import Path
import sys,json,hashlib,shutil,os,argparse
from functools import lru_cache
import numpy as np
import pandas as pd
import geopandas as gpd
import networkx as nx
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'RAPID/src'),str(Path(__file__).parent)]
from network_variants.reviewed_geometry import active_geometry,digest
from regenerate_reviewed_hydraulics import base
MASTER=(ROOT/'network_variants/output').resolve()
STAGE=ROOT/'RAPID/assets/master_integration_20261009'
HYD=ROOT/'RAPID/assets/regenerated_hydraulics_20261009'
RELEASE='master_integrated_20261009'
BACKUP=MASTER/'_integration_backups'/RELEASE

def save(p,x):
 p.parent.mkdir(parents=True,exist_ok=True)
 p.write_text(json.dumps(x,indent=2,default=lambda v:v.item() if hasattr(v,'item') else str(v))+'\n')

def safe_target(p):
 p=Path(p).resolve();p.relative_to(MASTER);return p

@lru_cache(maxsize=16)
def base_nodes(ex):
 r=active_geometry().lookup(ex,'indep','S000_base')
 return gpd.read_file(r['nodes']).set_index('id_node')

def clean_gdf(f):
 f=f.copy()
 for col in f.columns:
  if col=='geometry':continue
  if f[col].dtype==object:
   f[col]=f[col].map(lambda v: json.dumps(v.tolist() if isinstance(v,np.ndarray) else v) if isinstance(v,(list,dict,tuple,np.ndarray)) else v)
 return f

def stage_object(r):
 gid=r.get('geometry_id') or hashlib.sha256(json.dumps(r.get('retained_source_checksums',dict(links=r['links'],nodes=r['nodes'])),sort_keys=True).encode()).hexdigest()[:24];d=STAGE/'objects'/gid;done=d/'staged.json'
 if done.exists():return json.loads(done.read_text())
 d.mkdir(parents=True,exist_ok=True)
 release=active_geometry();changed=bool(r.get('bundle'))
 if changed:
  geo,nodes,contract,matches,_=release.load(r['example'],r['family'],r['state'])
  links=gpd.read_file(HYD/'objects'/gid/'widths_enriched.gpkg')
  f=pd.read_parquet(HYD/'objects'/gid/'hydraulic_frame.parquet')
 else:
  links=gpd.read_file(r['links']);nodes=gpd.read_file(r['nodes']);geo=links.copy();contract={};matches={}
  f=pd.read_parquet(r['original_frame'])
 old_nodes=gpd.read_file(r['nodes']).set_index('id_node')
 bn=base_nodes(r['example']);bl,refnodes,refs=base(r['example'])
 assert set(links.id_link)==set(f.id_link)
 f=f.set_index('id_link').loc[links.id_link].reset_index()
 np.testing.assert_allclose(links.geometry.length,f.link_length_m,rtol=1e-8,atol=1e-5)
 for col in ['id_us_node','id_ds_node']:
  assert np.array_equal(links[col].to_numpy(),f[col].to_numpy())
 # Replace physical attributes; preserve frozen measured widths and QC provenance.
 for col in ['u','v','link_length_m','width','depth_m','kb_local','head_us','head_ds','slope']:
  links[col]=f[col].to_numpy()
 links['wid_adj_routing']=f.width.to_numpy();links['len']=f.link_length_m.to_numpy()
 links['slope_used']=f.slope.to_numpy();links['geometry_id']=gid;links['geometry_release']=RELEASE
 pairs=pd.concat([f[['id_us_node','u','head_us']].rename(columns={'id_us_node':'raw','u':'parent','head_us':'head'}),f[['id_ds_node','v','head_ds']].rename(columns={'id_ds_node':'raw','v':'parent','head_ds':'head'})]).drop_duplicates()
 assert not pairs.raw.duplicated().any();assert not pairs.parent.duplicated().any()
 mapping=pairs.set_index('raw');assert set(nodes.id_node)==set(mapping.index)
 graph=nx.DiGraph();graph.add_edges_from(zip(links.id_us_node,links.id_ds_node));assert nx.is_directed_acyclic_graph(graph)
 incoming={n:[] for n in graph};outgoing={n:[] for n in graph}
 for lid,u,v in zip(links.id_link,links.id_us_node,links.id_ds_node):
  incoming[v].append(int(lid));outgoing[u].append(int(lid))
 roots=[n for n in graph if not incoming[n]];ends=[n for n in graph if not outgoing[n]];assert len(roots)==len(ends)==1
 rows=[]
 for node in nodes.itertuples():
  raw=int(node.id_node);parent=int(mapping.loc[raw,'parent']);prior=old_nodes.loc[raw].to_dict() if raw in old_nodes.index and old_nodes.loc[raw].geometry.equals(node.geometry) else {}
  if not prior and parent in bn.index:
   prior={k:v for k,v in bn.loc[parent].to_dict().items() if k.startswith('sword_')}
  prior.update(id_node=raw,geometry=node.geometry,parent_identity=parent,matched_parent_node_id=parent if parent in bn.index else None,
               id_links=','.join(map(str,incoming[raw]+outgoing[raw])),n_links=len(incoming[raw])+len(outgoing[raw]),
               in_degree=len(incoming[raw]),out_degree=len(outgoing[raw]),is_inlet=raw==roots[0],is_outlet=raw==ends[0],
               type_io='inlet' if raw==roots[0] else 'outlet' if raw==ends[0] else 'internal',
               rapid_wse_for_slope=float(mapping.loc[raw,'head']),rapid_wse_source_method='existing_domain_policy',geometry_id=gid,geometry_release=RELEASE)
  for col in ['sword_wse','sword_wse_field','sword_wse_fill_method']:prior.setdefault(col,None)
  prior['match_distance']=float(node.geometry.distance(bn.loc[parent].geometry)) if parent in bn.index else None
  prior['match_within_tolerance']=bool(parent in bn.index and prior['match_distance']<=1e-5)
  rows.append(prior)
 nodes=gpd.GeoDataFrame(rows,geometry='geometry',crs=geo.crs)
 links['id_nodes']=[f'{int(u)},{int(v)}' for u,v in zip(links.id_us_node,links.id_ds_node)];links['n_nodes']=2
 links['is_inlet']=links.id_us_node.eq(roots[0]);links['is_outlet']=links.id_ds_node.eq(ends[0])
 links['link_conn']=[','.join(map(str,sorted(set(incoming[u]+outgoing[u]+incoming[v]+outgoing[v])-{int(lid)}))) for lid,u,v in zip(links.id_link,links.id_us_node,links.id_ds_node)]
 # Pixel indices and length-derived diagnostics on cut/assembled links must not pretend to be current.
 if changed:
  changed_ids=json.loads((HYD/'objects'/gid/'width_result.json').read_text())['changed_width_links']
  for col in ['idx_link','len_adj','sinuosity','cldists','clangs','maxang','raw_id_nodes']:
   if col in links:links.loc[links.id_link.isin(changed_ids),col]=None
 links=clean_gdf(links);nodes=clean_gdf(nodes)
 links.to_file(d/'links.gpkg',layer='links',driver='GPKG',index=False)
 nodes.to_file(d/'nodes.gpkg',layer='nodes',driver='GPKG',index=False)
 f['geometry_release']=RELEASE;f['geometry_id']=gid;f['hydraulic_head_policy']='existing_domain_policy'
 f.to_parquet(d/'hydraulic_frame.parquet',index=False)
 # Deliberately write current hydraulic fields, not stale diagnostics from the previous solve.
 lc=f.copy();lc['slope_used']=lc.slope;lc['slope_adjusted']=True;lc['slope_source_method']='existing_domain_policy'
 lc['slope_reason']='reviewed_geometry_shared_head_reconstruction';lc['link_length_source']='geometry'
 lc['wse_us']=lc.head_us;lc['wse_ds']=lc.head_ds
 lc.to_csv(d/'link_hydraulic_context.csv',index=False)
 nc=nodes.drop(columns='geometry').copy();nc.to_csv(d/'node_hydraulic_context.csv',index=False)
 nm=nodes[['id_node','matched_parent_node_id','match_distance','match_within_tolerance']].rename(columns={'id_node':'child_id_node'})
 nm['match_method']='reviewed_parent_identity';nm.to_csv(d/'node_match.csv',index=False)
 lineage_columns=[c for c in links if c.startswith(('matched_parent','dominant_parent','lineage','source_link','width_regeneration'))]
 links[['id_link',*lineage_columns]].to_csv(d/'link_lineage.csv',index=False)
 if contract:save(d/'reviewed_boundary_contract.json',contract)
 target_h=Path(r['original_manifest']).resolve().parent;target_links=safe_target(r['links']);target_nodes=safe_target(r['nodes'])
 directed=target_nodes.with_name(target_nodes.name.replace('_nodes.gpkg','_links.gpkg'))
 assert directed!=target_nodes and directed.exists()
 paths={'links':str(target_links),'nodes':str(target_nodes),'directed_links':str(directed),'frame':str(target_h/'hydraulic_frame.parquet'),'manifest':str(target_h/'hydraulic_context_manifest.json')}
 fingerprint=hashlib.sha256(f[['u','v','link_length_m','width','slope']].sort_values(['u','v','link_length_m','width']).to_numpy().tobytes()).hexdigest()[:20]
 meta=dict(schema_version='master-network-hydraulics-v1',state_id=r['state'],geometry_release=RELEASE,geometry_id=gid,
           input_paths=dict(directed_links=str(target_links),directed_nodes=str(target_nodes)),
           output_paths=dict(link_hydraulic_context=str(target_h/'link_hydraulic_context.csv'),node_hydraulic_context=str(target_h/'node_hydraulic_context.csv'),hydraulic_frame=paths['frame']),
           n_links=len(links),n_nodes=len(nodes),head_policy='existing_domain_policy',fingerprint=fingerprint,
           hydraulic_attributes_current=True,rapid_preparation_status='requires_regeneration',routing_results_status='historical',
           hierarchy_metrics_status='requires_refresh' if changed else 'unchanged_geometry',
           provenance=dict(source_geometry_release=release.release,source_geometry_id=gid,source_frame=str(HYD/'objects'/gid/'hydraulic_frame.parquet') if changed else r['original_frame']))
 save(d/'hydraulic_context_manifest.json',meta)
 actions=[]
 def add(source,target):actions.append(dict(source=str(d/source),target=str(safe_target(target)),sha256=digest(d/source)))
 add('links.gpkg',target_links);add('links.gpkg',directed);add('nodes.gpkg',target_nodes)
 for name in ['hydraulic_frame.parquet','link_hydraulic_context.csv','node_hydraulic_context.csv','hydraulic_context_manifest.json']:add(name,target_h/name)
 if changed:
  add('node_match.csv',target_links.parent.parent/'matching/node_match.csv');add('link_lineage.csv',target_links.parent.parent/'matching/link_lineage.csv')
  if contract:add('reviewed_boundary_contract.json',target_h/'reviewed_boundary_contract.json')
  save(d/'geometry_integration_status.json',dict(geometry_release=RELEASE,geometry_id=gid,geometry_current=True,
       matching_authority='reviewed parent identities in matching/node_match.csv',historical_matching_files=['link_match.csv','node_sword_match.csv'],
       hierarchy_and_equivalent_metrics='require_refresh',rapid_prep_and_runs='historical_require_regeneration'))
  add('geometry_integration_status.json',target_h.parent/'geometry_integration_status.json')
 # Validate the actual on-disk formats before authorizing any replacement.
 read=gpd.read_file(d/'links.gpkg');nn=gpd.read_file(d/'nodes.gpkg')
 assert set(read.id_link)==set(f.id_link) and set(nn.id_node)==set(mapping.index)
 np.testing.assert_allclose(read.wid_adj_routing,f.width,rtol=0,atol=0)
 assert read.geometry.to_wkb().tolist()==links.geometry.to_wkb().tolist()
 assert (lc.slope_used>0).all() and np.isfinite(lc.slope_used).all()
 result=dict(key=[r['example'],r['family'],r['state']],geometry_id=gid,changed_geometry=changed,paths=paths,actions=actions,fingerprint=fingerprint,n_links=len(f),validated=True)
 save(done,result);return result


def stage():
 STAGE.mkdir(parents=True,exist_ok=True);release=active_geometry();objects={}
 for i,r in enumerate(release.records.values()):
  if r.get('canonical_key') or r['geometry_status']=='excluded':continue
  result=stage_object(r);objects[tuple(result['key'])]=result
  if len(objects)%25==0:print('staged',len(objects),r['example'],r['state'],flush=True)
 save(STAGE/'objects_index.json',list(objects.values()))
 print('OBJECTS COMPLETE',len(objects),flush=True)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('action',choices=['stage']);a=p.parse_args();stage()
