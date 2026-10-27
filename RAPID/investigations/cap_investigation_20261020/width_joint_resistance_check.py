"""Three prespecified single-feature additions to the frozen nine-input panel."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
import pandas as pd
import networkx as nx
from slope_ratio_audit import ROOT,OUT,EXT,SC,read,SOURCES,validate_domain,allocation_tables
from tune_nine_inputs import b,COLUMNS
DEST=OUT/'width_joint_resistance_20261023'
PANELS={'parent_width':'parent_width_nd','joint_geometry':'delta_G','resistance':'resistance_log'}
def prepare():
 DEST.mkdir(exist_ok=True)
 if (DEST/'preparation.json').exists():
  m=json.loads((DEST/'preparation.json').read_text())
  assert m['script_hash']==hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
  for path,h in m['sources'].items():assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==h
  return pd.read_parquet(DEST/'input.parquet'),m
 d=read(OUT/'augmented_tests/input.parquet');linked=read(SC/'linked_responses.parquet').set_index(['example','state'])
 graphs=read(SC/'domain_graphs.csv').set_index(['example','state','scope']);sol=read(SC/'resistance_solver_audit.csv');sol=sol[sol.alpha.eq(1.)].set_index('graph')
 jobs={}
 for plan in [ROOT/'RAPID/outputs/event_window_plan_20261013/plan.json',EXT/'routing_plan.json',OUT/'routing_plan.json']:
  for case in read(plan)['cases']:
   for j in case['jobs']:
    for name in set([j['state_id'],*j.get('aliases',[])]):jobs[case['example'],name]=j
 parents={ex:read(jobs[ex,'S000_base']['parameters']) for ex in d.example.unique()};cache={};rows=[];res_audit=[]
 for ex,group in d.groupby('example'):
  for r in group.itertuples():
   fulls={'parent':parents[ex],'child':read(jobs[ex,r.state]['parameters'])};z=dict(example=ex,state=r.state)
   if r.cohort=='segmented':
    line=read(EXT/f'variants/{ex}/{r.state}/lineage.csv');line=line[line.role.eq('collapsed_block')]
    pg=graphs.loc[(ex,'fully_collapsed','N'),'parent_graph'];saved=read(SC/'graphs'/f'{pg}.parquet');parent=fulls['parent'];validate_domain(parent,saved)
    np.testing.assert_allclose(parent.set_index('id_link').loc[saved.id_link,['link_length_m','width']],saved[['link_length_m','width']],rtol=1e-12)
    s=sol.loc[pg];assert bool(s.success) and s.graph_sha256==SOURCES[str(SC/'graphs'/f'{pg}.parquet')]
    child=fulls['child'];graph=nx.MultiDiGraph();graph.add_edges_from(zip(child.id_us_node,child.id_ds_node));assert nx.is_directed_acyclic_graph(graph) and nx.is_weakly_connected(graph) and all(graph.in_degree(n)<=1 and graph.out_degree(n)<=1 for n in graph)
    kp=float(s.K);kc=float((child.link_length_m/child.width**5).sum());z['resistance_log']=-np.log10(kc/kp)
    res_audit.append(dict(example=ex,state=r.state,scope='full network including connectors',K_parent=kp,K_child=kc,parent_energy_gap=s.relative_energy_gap,child_method='exact serial sum',parent_graph=pg))
   else:
    rr=linked.loc[(ex,r.state)];g=graphs.loc[(ex,r.state,rr.B_scope)];assert bool(rr.B_K_valid_a1) and np.isfinite(rr.B_K_log_a1)
    z['resistance_log']=float(rr.B_K_log_a1)
   for side,full in fulls.items():
    if r.cohort=='segmented':
     f=full[full.id_link.isin(line[side+'_link_id'])];w=f.reference_flow_fraction*f.link_length_m
    else:
     key=g[side+'_graph']
     if key not in cache:
      geom=read(SC/'graphs'/f'{key}.parquet');cache[key]=(geom,allocation_tables(geom)[1].set_index('id_link'))
     geom,alloc=cache[key];f,_=validate_domain(full,geom);f=f.set_index('id_link').loc[alloc.index]
     np.testing.assert_allclose(f[['link_length_m','width']],geom.set_index('id_link').loc[alloc.index,['link_length_m','width']],rtol=1e-10,atol=1e-7);w=alloc.mean_length_contribution_m
     ss=sol.loc[key];assert bool(ss.success) and ss.graph_sha256==SOURCES[str(SC/'graphs'/f'{key}.parquet')]
     z[side+'_K']=float(ss.K)
    assert (f.slope>0).all()
    z[side+'_G']=float((w/1000*(f.width/100)**(-13/30)/np.sqrt(f.slope)).sum())
    if side=='parent':
     wm=float(np.average(f.width,weights=f.link_length_m));np.testing.assert_allclose(wm,r.B_parent_width_mean,rtol=1e-10)
     z['parent_width_nd']=wm/100
   if r.cohort!='segmented':np.testing.assert_allclose(z['resistance_log'],-np.log10(z['child_K']/z['parent_K']),rtol=1e-10,atol=1e-10)
   z['delta_G']=z['child_G']-z['parent_G'];rows.append(z)
  print(ex,'prepared',len(group),'cases',flush=True)
 d=d.merge(pd.DataFrame(rows),on=['example','state'],how='left',validate='one_to_one');assert len(d)==2185 and np.isfinite(d[COLUMNS+list(PANELS.values())]).all().all()
 d.to_parquet(DEST/'input.parquet',index=False);pd.DataFrame(res_audit).to_csv(DEST/'segmented_resistance_audit.csv',index=False)
 for path,h in SOURCES.items():assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==h
 m=dict(sources=SOURCES,script_hash=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),rows=len(d),unmatched=0,definitions=dict(parent_width_nd='length-weighted parent width / 100 m; B domain for regular, edited footprint for segmented',delta_G='child-parent sum r_e*(L_e/1km)*(W_e/100m)^(-13/30)/sqrt(s_e); same domain conventions as parent width',resistance_log='-log10(Kchild/Kparent), Ke=L/W^5; B outer domain for regular, full network INCLUDING unchanged connectors for segmented. Cached parent/unit-flow solutions validated against active geometry. Segmented child exact series sum.'),caveat='All new features geometric; joint exponent derives from current assumed depth-width exponent .65. No calibrated roughness or modeled depths used. Different scope conventions explicitly retained; this is not a domain-choice comparison.')
 (DEST/'preparation.json').write_text(json.dumps(m,indent=2)+'\n');return d,m

def summarize(d):
 old=pd.read_parquet(OUT/'nine_input_tuning/predictions.parquet');old=old[old.variant.eq('same_grid')].copy();old['panel']='baseline'
 rows=[]
 for p in sorted((DEST/'folds').glob('*/*.json')):rows.extend(json.loads(p.read_text())['rows'])
 pred=pd.concat([old[['example','state','cohort','centroid_shift_hours','predicted_hours','panel']],pd.DataFrame(rows)],ignore_index=True)
 assert len(pred)==4*2185 and not pred.duplicated(['panel','example','state']).any()
 pred.to_parquet(DEST/'predictions.parquet',index=False);scores=[]
 for (panel,cohort),g in pred.groupby(['panel','cohort']):
  for subset,z in [('all',g),('without54',g[g.example.ne('54')])]:
   w=1/z.groupby('example').example.transform('size').to_numpy();y=z.centroid_shift_hours.to_numpy();pr=z.predicted_hours.to_numpy()
   scores.append(dict(panel=panel,cohort=cohort,subset=subset,n=len(z),r2=1-np.sum(w*(y-pr)**2)/np.sum(w*(y-np.average(y,weights=w))**2),mae_minutes=60*np.average(abs(y-pr),weights=w),relative_error_pct=100*np.average(abs(y-pr)/np.maximum(abs(y),1/60),weights=w)))
 s=pd.DataFrame(scores);s.to_csv(DEST/'scores.csv',index=False)
 s[s.subset.eq('all')].groupby('panel').relative_error_pct.mean().rename('equal_cohort_objective_pct').to_csv(DEST/'objective.csv')
 pred['abs_error_min']=60*abs(pred.predicted_hours-pred.centroid_shift_hours)
 pred.groupby(['panel','cohort','example']).abs_error_min.mean().rename('mae_minutes').to_csv(DEST/'network_errors.csv')
 priorities=pd.read_csv(OUT/'outlier_diagnosis_20261023/priority_cases.csv',dtype={'example':str})[['example','state']]
 pred.merge(priorities,on=['example','state'],validate='many_to_one').to_csv(DEST/'priority_predictions.csv',index=False)
 pred[pred.cohort.eq('segmented')].to_csv(DEST/'segmented_predictions.csv',index=False)
 print(s[s.subset.eq('all')].to_string(index=False),flush=True)

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--run',action='store_true');ap.add_argument('--jobs',type=int,default=4);args=ap.parse_args();d,prep=prepare()
 choices=json.loads((OUT/'nine_input_tuning/selections.json').read_text());choices=[r for r in choices if r['variant']=='same_grid'];assert len(choices)==15
 design=dict(preparation=prep,choices=choices,panels=PANELS,baseline_columns=COLUMNS,seed=b.engine.SEED,validation='Add exactly one feature per panel. Reuse nine-input baseline training-only selected settings for each whole-network holdout. Equal-network training weights; no retuning; no augmentation.',fits=45,production_modified=False)
 fp=hashlib.sha256(json.dumps(design,sort_keys=True).encode()).hexdigest();(DEST/'design.json').write_text(json.dumps(design,indent=2)+'\n')
 if not args.run:return
 for panel,col in PANELS.items():
  b.COLUMNS=COLUMNS+[col]
  for choice in choices:
   ex=choice['held'];path=DEST/'folds'/panel/f'{ex}.json';path.parent.mkdir(parents=True,exist_ok=True)
   if path.exists():assert json.loads(path.read_text())['fingerprint']==fp;continue
   tr=d[d.example.ne(ex)];te=d[d.example.eq(ex)];assert tr.example.nunique()==14
   model=b.fit(tr,choice['params'],jobs=args.jobs);pr=b.forest.predict(model,te);assert np.isfinite(pr).all()
   rows=[dict(example=r.example,state=r.state,cohort=r.cohort,centroid_shift_hours=r.centroid_shift_hours,predicted_hours=float(v),panel=panel) for r,v in zip(te.itertuples(),pr)]
   path.write_text(json.dumps(dict(fingerprint=fp,rows=rows),indent=2)+'\n');print(panel,ex,'saved',flush=True)
 b.COLUMNS=list(COLUMNS);summarize(d)
 (DEST/'completion.json').write_text(json.dumps(dict(complete=True,fingerprint=fp,fits=45,production_modified=False),indent=2)+'\n')
if __name__=='__main__':main()
