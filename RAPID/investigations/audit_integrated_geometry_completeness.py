"""Independent completeness/semantic audit, beyond release-count validation."""
from pathlib import Path
import sys,json
from collections import Counter
import pandas as pd
import geopandas as gpd
import networkx as nx
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT),str(ROOT/'RAPID/src')]
from network_variants.reviewed_geometry import active_geometry,digest
import argparse
parser=argparse.ArgumentParser();parser.add_argument('--manifest',type=Path);args=parser.parse_args()
from network_variants.reviewed_geometry import ReviewedGeometry
release=ReviewedGeometry(args.manifest) if args.manifest else active_geometry();OUT=release.manifest.parent/'completeness_audit';OUT.mkdir(exist_ok=True)
records=list(release.records.values());errors=[];warnings=[];checks=[]

def fail(kind,record,detail):
    errors.append(dict(kind=kind,example=record.get('example'),state=record.get('state'),detail=str(detail)))

# Check CURRENT experiment registries, not just the frozen input inventory.
current=set()
for ex in sorted({r['example'] for r in records}):
 suffix='_direction130' if ex=='15' else ''
 for family in ['indep','compound_bubbles','island']:
  folder=ROOT/'network_variants/output'/f'{ex}_{family}{suffix}__anchored'
  p=folder/('island_hydraulic_context_registry.csv' if family=='island' else 'state_registry.csv')
  if not p.exists():
   if any(r['example']==ex and r['family']==family for r in records):
    fail('missing_current_registry',dict(example=ex,state=family),p)
   continue
  f=pd.read_csv(p)
  column='island_id' if family=='island' else 'state_id'
  current.update((ex,family,str(v)) for v in f[column])
expected=set(release.records)
for key in current-expected:fail('current_state_missing_from_release',dict(example=key[0],state=key[2]),key)
for key in expected-current:fail('release_state_missing_from_current_registry',dict(example=key[0],state=key[2]),key)
# Every parent reference must resolve in the logical hierarchy.
for r in records:
 p=r.get('parent_state')
 if p and (r['example'],r['family'],p) not in expected:
  if p!='S000_base' or (r['example'],'indep',p) not in expected:fail('missing_parent',r,p)
# Check targets independently of alias selection, including retained absorption.
targets=pd.read_csv(ROOT/'RAPID/outputs/collapse_creation_review_20261008/target_inventory.csv',dtype={'example':str}).fillna('')
targets={(r.example,r.family,r.state):r for r in targets.itertuples()}
for r in records:
 if not r.get('canonical_key'):continue
 c=release.canonical(r);a=targets.get((r['example'],r['family'],r['state']));b=targets.get((c['example'],c['family'],c['state']))
 if a is None or b is None:fail('alias_missing_target_evidence',r,c['state']);continue
 if a.actual_full_target_islands != b.actual_full_target_islands or int(a.actual_target_count)!=1 or int(b.actual_target_count)!=1:
  fail('alias_target_mismatch',r,dict(island=a.actual_full_target_islands,canonical=b.actual_full_target_islands))
 if int(a.partial_target_count) or int(b.partial_target_count):fail('alias_partial_target',r,c['state'])
 if r.get('separate_observation_excluded') and c['family']!='indep':fail('duplicate_not_independent_canonical',r,c['family'])
# Recompute topology from every integrated geometry; check endpoints/identity,
# terminal preservation, direction and the intended number of removed loops.
seen=set();topology=[]
for index,r in enumerate(records):
 c=release.canonical(r)
 if not c.get('bundle') or c['geometry_id'] in seen:continue
 seen.add(c['geometry_id'])
 links,nodes,contract,matches,_=release.load(c['example'],c['family'],c['state'])
 mp=dict(zip(nodes.id_node.astype(int),nodes.parent_identity.astype(int)))
 if nodes.parent_identity.duplicated().any():fail('duplicate_parent_identity',c,'not injective')
 if 'u' in links and ((links.id_us_node.map(mp)!=links.u).any() or (links.id_ds_node.map(mp)!=links.v).any()):fail('canonical_endpoint_disagreement',c,'u/v differ from parent_identity')
 if links.geometry.normalize().to_wkb().duplicated().any():fail('duplicate_link_geometry',c,'exact duplicate')
 g=nx.MultiDiGraph();g.add_edges_from((mp[int(a)],mp[int(b)]) for a,b in zip(links.id_us_node,links.id_ds_node))
 terminal=lambda graph: (sorted(n for n in graph if graph.in_degree(n)==0),sorted(n for n in graph if graph.out_degree(n)==0))
 if list(nx.selfloop_edges(g)):fail('self_loop',c,'same canonical endpoints')
 if not nx.is_directed_acyclic_graph(g):fail('directed_cycle',c,'cycle')
 if nx.number_weakly_connected_components(g)!=1:fail('disconnected',c,'multiple components')
 old=pd.read_parquet(c['original_frame']);og=nx.MultiDiGraph();og.add_edges_from(zip(old.u,old.v))
 if terminal(g)!=terminal(og):fail('changed_terminals',c,dict(old=terminal(og),new=terminal(g)))
 rank=lambda graph:graph.number_of_edges()-graph.number_of_nodes()+nx.number_weakly_connected_components(graph)
 if rank(g)!=rank(og):warnings.append(dict(kind='cycle_rank_change_from_old_child',example=c['example'],state=c['state'],old=rank(og),new=rank(g)))
 needed=set(contract.get('reviewed_required_boundary_nodes',contract['boundary_nodes']))
 forbidden=set(contract.get('reviewed_internal_parent_nodes',[]))|set(contract.get('reviewed_provisional_parent_nodes',[]))
 if needed-set(mp.values()):fail('missing_anchor',c,needed-set(mp.values()))
 if forbidden&set(mp.values()):fail('waived_node_reintroduced',c,forbidden&set(mp.values()))
 topology.append(dict(example=c['example'],state=c['state'],geometry_id=c['geometry_id'],nodes=len(g),links=g.number_of_edges(),old_cycle_rank=rank(og),new_cycle_rank=rank(g),inlet=terminal(g)[0],outlet=terminal(g)[1]))
 if len(seen)%100==0:print('Checked',len(seen),'geometry objects',flush=True)
# Audit the retained geometries as well: these were only existence-checked by
# the first integration validator. Ensure canonical identities and graph agree.
retained_seen=set();retained_checks=[]
for r in records:
 c=release.canonical(r)
 if c.get('bundle') or c['geometry_status']=='excluded':continue
 key=(c['links'],c['nodes'],c['original_frame'])
 if key in retained_seen:continue
 retained_seen.add(key)
 try:
  links,nodes,contract,matches,_=release.load(c['example'],c['family'],c['state'])
  if nodes.parent_identity.isna().any():fail('retained_unmapped_node',c,int(nodes.parent_identity.isna().sum()))
  if nodes.parent_identity.duplicated().any():fail('retained_duplicate_identity',c,'not injective')
  if contract:
   required=set(contract.get('reviewed_required_boundary_nodes',contract['boundary_nodes']))
   if required-set(nodes.parent_identity):fail('retained_missing_anchor',c,required-set(nodes.parent_identity))
  g=nx.MultiDiGraph();g.add_edges_from(zip(links.u,links.v))
  old=pd.read_parquet(c['original_frame']);og=nx.MultiDiGraph();og.add_edges_from(zip(old.u,old.v))
  if Counter(g.edges())!=Counter(og.edges()):fail('retained_frame_topology_disagreement',c,'raw geometry vs hydraulic identity frame')
  if not nx.is_directed_acyclic_graph(g) or nx.number_weakly_connected_components(g)!=1:fail('retained_topology_invalid',c,'cycle or disconnected')
  retained_checks.append(dict(example=c['example'],state=c['state'],nodes=len(nodes),links=len(links)))
 except Exception as e:fail('retained_load_error',c,e)
 if len(retained_seen)%100==0:print('Checked',len(retained_seen),'retained geometry sources',flush=True)
pd.DataFrame(retained_checks).to_csv(OUT/'retained_geometry_checks.csv',index=False)
# Ensure exported inventories match canonical selection and stale status.
for p in release.manifest.parent.glob('*/inventory.csv'):
 f=pd.read_csv(p,dtype={'example':str}).fillna('')
 for row in f.to_dict('records'):
  r=release.lookup(row['example'],row['family'],row['state']);c=release.canonical(r)
  if row['canonical_state']!=c['state'] or row['canonical_family']!=c['family']:fail('exported_alias_mismatch',r,row)
  if r.get('separate_observation_excluded') and row['status']!='excluded_duplicate':fail('duplicate_still_in_primary_population',r,row['status'])
  if r['geometry_status']=='excluded' and row['status']!='excluded_topology_review':fail('exclusion_not_applied',r,row['status'])
  if c.get('requires_hydraulic_regeneration') and not r.get('separate_observation_excluded') and (row['frame'] or row['status']!='pending_reviewed_geometry_hydraulics'):fail('stale_frame_exposed',r,row['frame'])
# Cross-check original human review ledger coverage (not only hard-coded examples).
ledger=ROOT/'RAPID/outputs/boundary_contract_repair_20261008/boundary_manual_reviews.jsonl'
for decision in map(json.loads,ledger.read_text().splitlines()):
 r=release.lookup(decision['example'],'indep',decision['state'])
 if r is None or r['geometry_status']!='accepted':fail('manual_review_not_integrated',decision,'missing or unaccepted')
summary=dict(current_registry_states=len(current),release_states=len(expected),verified_repaired_geometries=len(seen),
             verified_retained_geometry_sources=len(retained_seen),aliases_checked=sum(bool(r.get('canonical_key')) for r in records),manual_review_records_checked=len(ledger.read_text().splitlines()),
             errors=errors,warnings=warnings,geometry_and_coverage_pass=not errors,routing_ready=False)
pd.DataFrame(topology).to_csv(OUT/'geometry_topology.csv',index=False)
(OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2))
