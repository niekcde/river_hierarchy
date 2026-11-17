from pathlib import Path
import sys,json
ROOT=Path(__file__).resolve().parents[3];sys.path[:0]=[str(ROOT),str(ROOT/'RAPID/investigations/frozen_floor_review_20261027')]
from check import release,frame,js,outside_paths
import pyogrio,networkx as nx
from shapely.ops import unary_union
OUT=ROOT/'RAPID/outputs/review_notes_20261027';cases=js(OUT/'evidence.json');out=[]
for c in cases:
 ex,state=c['key'].split('/');r=c['record'];f=frame(r['original_frame']);fi=f.set_index('id_link');g=pyogrio.read_dataframe(r['links']).set_index('id_link');br=release.lookup(ex,'indep','S000_base');pg=pyogrio.read_dataframe(br['links']).set_index('id_link')
 G=nx.MultiDiGraph()
 for q in f.itertuples():G.add_edge(int(q.u),int(q.v),key=int(q.id_link))
 chains=[]
 for e in c['exterior']:
  a,b=e['u'],e['v'];found=[]
  if a in G and b in G:
   for path in nx.all_simple_edge_paths(G,a,b,cutoff=4):
    ids=[v[2] for v in path];lines=list(g.loc[ids].geometry);union=unary_union(lines);pgeom=pg.loc[e['parent_link']].geometry
    found.append(dict(child_links=ids,nodes=[path[0][0]]+[q[1] for q in path],length=float(fi.loc[ids].link_length_m.sum()),length_difference=float(fi.loc[ids].link_length_m.sum()-e['length']),hausdorff=float(pgeom.hausdorff_distance(union)),simple=[bool(v.is_simple) for v in lines],closed=[bool(v.is_closed) for v in lines],overlap_within_chain_m=float(sum(v.length for v in lines)-union.length)))
  chains.append(dict(parent_link=e['parent_link'],paths=found))
 # October 3 hydraulic snapshot predates October 8 boundary repair. Compare raw IDs and lengths.
 histories=[v for k,v in c['history'].items() if k.endswith('/result.json') and v.get('source_frame')]
 prior=[]
 if histories:
  path=Path(histories[0]['source_frame'])
  if path.exists():
   old=frame(path);oldi=old.set_index('id_link');raw=set(old.id_us_node)|set(old.id_ds_node)
   for node in c['nodes']:
    if not node['child_present']:continue
    n=node['child_raw'];matches=[]
    for q in f[(f.id_us_node==n)|(f.id_ds_node==n)].itertuples():
     if q.id_link in oldi.index:
      q0=oldi.loc[q.id_link];matches.append(dict(link=int(q.id_link),same_raw_endpoints=bool(q.id_us_node==q0.id_us_node and q.id_ds_node==q0.id_ds_node),length_delta=float(q.link_length_m-q0.link_length_m)))
    prior.append(dict(node=node['node'],raw=n,already_present_oct3=n in raw,incident_links=matches))
 # Specific overlap measurements and edge evidence
 extra={}
 for lids in ([580,620],[578,580],[578,620]) if c['key']=='15/S113_unit_189' else []:
  extra[str(lids)]=float(g.loc[lids[0]].geometry.intersection(g.loc[lids[1]].geometry).length)
 if c['key']=='22/S244_unit_262':
  extra['nodes']=c['nodes'];extra['edges']=c['exterior']
 if c['key']=='37/S067_unit_303':
  paths=outside_paths(r,frame(br['original_frame']),f,c['selected']);extra['all_exterior_unique']=all(p.get('path_unique') for p in paths);extra['nonexact_exterior']=[p for p in paths if not p.get('geometry_exact')]
 out.append(dict(key=c['key'],chains=chains,pre_repair_evidence=prior,extra=extra))
(OUT/'followup.json').write_text(json.dumps(out,indent=2))
for c in out:
 if c['key'].startswith('02/') and c['key'] not in ['02/S004_unit_5','02/S026_unit_16']:continue
 print(c['key'],json.dumps(c))
