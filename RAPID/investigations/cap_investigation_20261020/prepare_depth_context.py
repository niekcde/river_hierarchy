"""Prepare optional explicit depth descriptors; no fits and no accepted model changes."""
import json,hashlib
from pathlib import Path
import pandas as pd,numpy as np
from slope_ratio_audit import ROOT,OUT,EXT,SC,read,SOURCES,validate_domain,allocation_tables
DEST=OUT/'depth_context_20261023'
def main():
 DEST.mkdir(exist_ok=True)
 d=read(OUT/'augmented_tests/input.parquet');meta=read(SC/'geometry_responses.parquet').set_index(['example','state']);graphs=read(SC/'domain_graphs.csv').set_index(['example','state','scope']);jobs={}
 for path in [ROOT/'RAPID/outputs/event_window_plan_20261013/plan.json',EXT/'routing_plan.json',OUT/'routing_plan.json']:
  for case in read(path)['cases']:
   for j in case['jobs']:
    for name in set([j['state_id'],*j.get('aliases',[])]):jobs[case['example'],name]=j
 parents={ex:read(jobs[ex,'S000_base']['parameters']) for ex in d.example.unique()};cache={};rows=[]
 for ex,group in d.groupby('example'):
  for r in group.itertuples():
   row=dict(example=ex,state=r.state,cohort=r.cohort)
   if r.cohort=='segmented':
    line=read(EXT/f'variants/{ex}/{r.state}/lineage.csv');line=line[line.role.eq('collapsed_block')]
   else:g=graphs.loc[(ex,r.state,meta.loc[(ex,r.state),'B_scope'])]
   for side,full in [('parent',parents[ex]),('child',read(jobs[ex,r.state]['parameters']))]:
    if r.cohort=='segmented':
     f=full[full.id_link.isin(line[side+'_link_id'])];w=f.reference_flow_fraction*f.link_length_m
    else:
     key=g[side+'_graph']
     if key not in cache:
      geom=read(SC/'graphs'/f'{key}.parquet');cache[key]=(geom,allocation_tables(geom)[1].set_index('id_link'))
     geom,a=cache[key];f,_=validate_domain(full,geom);f=f.set_index('id_link').loc[a.index];w=a.mean_length_contribution_m
    np.testing.assert_allclose(w.sum(),getattr(r,'B_'+side+'_route_length_mean'),rtol=1e-10,atol=1e-7)
    h=f.depth_m.to_numpy();assert np.isfinite(h).all() and (h>0).all()
    row[side+'_mean_depth_m']=float(np.average(h,weights=w));factor=h/f.width.to_numpy()**.65
    row[side+'_depth_width_scale']=float(factor.mean());row[side+'_depth_width_scale_cv']=float(factor.std()/factor.mean())
    assert row[side+'_depth_width_scale_cv']<1e-10
   row['parent_depth_nd']=row['parent_mean_depth_m']/1.;row['depth_ratio']=row['child_mean_depth_m']/row['parent_mean_depth_m'];rows.append(row)
  print(ex,'depth descriptors prepared',flush=True)
 q=pd.DataFrame(rows);assert len(q)==2185 and not q.duplicated(['example','state']).any();q.to_parquet(DEST/'depth_candidates.parquet',index=False)
 for path,h in SOURCES.items():assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==h
 (DEST/'manifest.json').write_text(json.dumps(dict(sources=SOURCES,script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),rows=len(q),definition='Visit-length-weighted parent/child mean derived depth. B outer domain for regular cases; selected edited footprint for segmented. Parent depth /1m and child/parent ratio prepared as separate OPTIONAL inputs.',provenance='Stored derived depth follows network reference anchor times width^.65, verified numerically. It is not independently observed reach depth. No hydrograph response used in feature calculation.',fits=0,production_modified=False),indent=2)+'\n')
if __name__=='__main__':main()
