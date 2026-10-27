"""Requested six panels: depth-free length/slope change, explicit-depth resistance.

Delta G_s is always an addition. Only mean-width ratio is replaced in panels 3/5.
Resistance uses existing wide-Chezy Ke=L/(W^2 h^3), NOT RAPID travel-time K.
"""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
import pandas as pd
import networkx as nx
from slope_ratio_audit import ROOT,OUT,EXT,SC,read,SOURCES
from tune_nine_inputs import b,COLUMNS
DEST=OUT/'six_panel_geometry_depth_resistance_20261023'
WIDTH='log1p__B_ratio_width_mean';R='resistance_explicit_depth';G='delta_G_s'
REPLACED=[R if c==WIDTH else c for c in COLUMNS]
PANELS={'1_current':list(COLUMNS),'2_current_plus_Gs':list(COLUMNS)+[G],
 '3_replace_width_with_Rh':REPLACED,'4_current_plus_Rh':list(COLUMNS)+[R],
 '5_replace_width_with_Rh_plus_Gs':REPLACED+[G],
 '6_current_plus_Rh_plus_Gs':list(COLUMNS)+[R,G]}
def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def verify_sources(m):
 for path,h in m['sources'].items():assert digest(path)==h,path

def prepare():
 DEST.mkdir(exist_ok=True)
 if (DEST/'preparation.json').exists():
  m=json.loads((DEST/'preparation.json').read_text());assert m['script_sha256']==digest(__file__);verify_sources(m)
  return pd.read_parquet(DEST/'input.parquet'),m
 d=read(OUT/'augmented_tests/input.parquet')
 slope=read(OUT/'slope_ratio_audit_20261023/input_and_residuals.parquet')
 dep=read(OUT/'depth_context_20261023/depth_candidates.parquet')
 for path in [OUT/'slope_ratio_audit_20261023/manifest.json',OUT/'depth_context_20261023/manifest.json',OUT/'width_joint_resistance_20261023/preparation.json']:
  verify_sources(read(path))
 d=d.merge(slope[['example','state','parent_length_km','child_length_km','child_slope_slow']],on=['example','state'],how='left',validate='one_to_one')
 d=d.merge(dep.drop(columns='cohort'),on=['example','state'],how='left',validate='one_to_one')
 assert len(d)==2185 and d.example.nunique()==15
 d['parent_G_s']=d.parent_length_km*d.parent_slope_slow
 d['child_G_s']=d.child_length_km*d.child_slope_slow
 d[G]=d.child_G_s-d.parent_G_s
 np.testing.assert_allclose(d.parent_length_km,d.B_parent_route_length_mean/1000,rtol=1e-10)
 np.testing.assert_allclose(d.child_length_km,d.B_child_route_length_mean/1000,rtol=1e-10)
 np.testing.assert_allclose(d.parent_depth_width_scale,d.child_depth_width_scale,rtol=1e-10)
 linked=read(SC/'linked_responses.parquet').set_index(['example','state']);graphs=read(SC/'domain_graphs.csv').set_index(['example','state','scope'])
 sol=read(SC/'resistance_solver_audit.csv');sol=sol[np.isclose(sol.alpha,.65)].set_index('graph')
 jobs={}
 for plan in [EXT/'routing_plan.json',OUT/'routing_plan.json']:
  for case in read(plan)['cases']:
   for j in case['jobs']:jobs[case['example'],j['state_id']]=j
 audit=[];values=[];cache={}
 def solve_record(key):
  if key not in cache:
   record=sol.loc[key];path=SC/'graphs'/f'{key}.parquet';read(path)
   assert bool(record.success) and record.graph_sha256==SOURCES[str(path)]
   cache[key]=record
  return cache[key]
 for row in d.itertuples():
  ex,state=row.example,row.state;ap=row.parent_depth_width_scale;ac=row.child_depth_width_scale
  if row.cohort=='original':
   lr=linked.loc[(ex,state)];gr=graphs.loc[(ex,state,lr.B_scope)]
   sp,sc=solve_record(gr.parent_graph),solve_record(gr.child_graph)
   kp=float(sp.K)/ap**3;kc=float(sc.K)/ac**3
   expected=float(lr.B_K_log_a065)
   scope=str(lr.B_scope)
  else:
   gr=graphs.loc[(ex,'fully_collapsed','N')];sp=solve_record(gr.parent_graph)
   parent=read(jobs[ex,'S000_base']['parameters']);child=read(jobs[ex,state]['parameters'])
   for f,a in [(parent,ap),(child,ac)]:np.testing.assert_allclose(f.depth_m,a*f.width**.65,rtol=1e-10)
   old=read(SC/'graphs'/f'{gr.parent_graph}.parquet')
   pd.testing.assert_frame_equal(parent.set_index('id_link')[['id_us_node','id_ds_node','link_length_m','width']].sort_index(),old.set_index('id_link')[['id_us_node','id_ds_node','link_length_m','width']].sort_index(),check_dtype=False,rtol=1e-10,atol=1e-9)
   graph=nx.MultiDiGraph();graph.add_edges_from(zip(child.id_us_node,child.id_ds_node));assert nx.is_directed_acyclic_graph(graph) and nx.is_weakly_connected(graph) and all(graph.in_degree(n)<=1 and graph.out_degree(n)<=1 for n in graph)
   kp=float(sp.K)/ap**3;kc=float((child.link_length_m/(child.width**2*child.depth_m**3)).sum())
   expected=-np.log10(float((child.link_length_m/child.width**3.95).sum())/float(sp.K));scope='N including connectors'
  value=-np.log10(kc/kp);assert np.isfinite(value) and kp>0 and kc>0
  np.testing.assert_allclose(value,expected,atol=1e-10,rtol=1e-10);values.append(value)
  audit.append(dict(example=ex,state=state,scope=scope,K_parent_explicit_depth=kp,K_child_explicit_depth=kc,resistance_log=value,normalized_width_law_log=expected,parent_depth_width_scale=ap,child_depth_width_scale=ac))
 d[R]=values;assert np.isfinite(d[list(dict.fromkeys(sum(PANELS.values(),[])))]).all().all()
 d.to_parquet(DEST/'input.parquet',index=False);pd.DataFrame(audit).to_csv(DEST/'explicit_depth_resistance_audit.csv',index=False)
 verify_sources(dict(sources=SOURCES))
 m=dict(sources=SOURCES,script_sha256=digest(__file__),rows=len(d),unmatched=0,
  Gs='sum r_e*(L_e/1km)/sqrt(s_e); delta=child-parent; B comparison domain for regular, edited footprint excluding connectors for segmented. No depth or depth-derived width exponent; r_e remains width-based.',
  resistance='Ke=L/(W^2 h^3), wide-Chezy law with common coefficient cancelling; B outer domain for regular, full N including connectors for segmented. Actual derived h=a W^.65; alpha=.65 unit-flow solves rescaled by a^-3 are algebraically exact. Since a is identical parent/child, absolute depth scale cancels in resistance ratio.',
  depth='Explicit derived depths still inherit .65 within-network distribution. Parent mean depth /1m and child/parent ratio prepared but NOT used as extra predictors in these six panels.',
  production_modified=False)
 (DEST/'preparation.json').write_text(json.dumps(m,indent=2)+'\n');return d,m

def summarize(design,fp):
 old=read(OUT/'nine_input_tuning/predictions.parquet');old=old[old.variant.eq('same_grid')].copy();old['panel']='1_current'
 rows=[]
 for path in sorted((DEST/'folds').glob('*/*.json')):
  x=json.loads(path.read_text());assert x['fingerprint']==fp;rows.extend(x['rows'])
 cols=['example','state','cohort','centroid_shift_hours','predicted_hours','panel']
 p=pd.concat([old[cols],pd.DataFrame(rows)],ignore_index=True);assert len(p)==6*2185 and not p.duplicated(['panel','example','state']).any()
 p.to_parquet(DEST/'predictions.parquet',index=False);scores=[];network=[]
 for (panel,cohort),g in p.groupby(['panel','cohort']):
  for subset,z in [('all',g),('without54',g[g.example.ne('54')])]:
   w=1/z.groupby('example').example.transform('size').to_numpy();y=z.centroid_shift_hours.to_numpy();pred=z.predicted_hours.to_numpy()
   scores.append(dict(panel=panel,cohort=cohort,subset=subset,n=len(z),r2=1-np.sum(w*(pred-y)**2)/np.sum(w*(y-np.average(y,weights=w))**2),mae_minutes=60*np.average(abs(pred-y),weights=w),relative_error_pct=100*np.average(abs(pred-y)/np.maximum(abs(y),1/60),weights=w)))
  for ex,z in g.groupby('example'):
   err=z.predicted_hours-z.centroid_shift_hours
   network.append(dict(panel=panel,cohort=cohort,example=ex,mae_minutes=60*abs(err).mean(),relative_error_pct=100*np.mean(abs(err)/np.maximum(abs(z.centroid_shift_hours),1/60))))
 s=pd.DataFrame(scores);s.to_csv(DEST/'scores.csv',index=False);pd.DataFrame(network).to_csv(DEST/'network_scores.csv',index=False)
 s[s.subset.eq('all')].groupby('panel').relative_error_pct.mean().rename('equal_cohort_objective_pct').to_csv(DEST/'objective.csv')
 priority=read(OUT/'outlier_diagnosis_20261023/priority_cases.csv')[['example','state']]
 p.merge(priority,on=['example','state'],validate='many_to_one').to_csv(DEST/'priority_predictions.csv',index=False)
 p[p.cohort.eq('segmented')].to_csv(DEST/'segmented_predictions.csv',index=False)
 base=s[s.panel.eq('1_current')&s.subset.eq('all')].set_index('cohort');np.testing.assert_allclose(base.loc['original','r2'],.6386142466697977,atol=1e-12);np.testing.assert_allclose(base.loc['segmented','r2'],.5099587362122324,atol=1e-12)
 print(s[s.subset.eq('all')].to_string(index=False),flush=True)

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--run',action='store_true');ap.add_argument('--jobs',type=int,default=4);args=ap.parse_args();d,prep=prepare()
 selections=read(OUT/'nine_input_tuning/selections.json');choices=[r for r in selections if r['variant']=='same_grid'];assert len(choices)==15
 deps={str(Path(mod.__file__)):digest(mod.__file__) for mod in [b,b.engine,b.forest]}
 design=dict(preparation=prep,panels=PANELS,choices=choices,dependencies=deps,seed=b.engine.SEED,fit_count=75,
  validation='No retuning. Identical accepted baseline fold-selected settings for every panel; whole-network holdout, equal-network training weights, training-only preprocessing, fixed seed, full-centroid target and 0.5% window. No augmented states.',
  warning='New Gs and explicit-depth resistance differ from previous deltaG(width^-13/30) and Ke=L/W^5; previous scores cannot stand in for these panels.',production_modified=False)
 fp=hashlib.sha256(json.dumps(design,sort_keys=True).encode()).hexdigest();(DEST/'design.json').write_text(json.dumps(design,indent=2)+'\n')
 if not args.run:return
 for panel,cols in PANELS.items():
  if panel=='1_current':continue
  b.COLUMNS=list(cols)
  for choice in choices:
   ex=choice['held'];path=DEST/'folds'/panel/f'{ex}.json';path.parent.mkdir(parents=True,exist_ok=True)
   if path.exists():assert json.loads(path.read_text())['fingerprint']==fp;continue
   tr=d[d.example.ne(ex)];te=d[d.example.eq(ex)];assert tr.example.nunique()==14
   artifact=b.fit(tr,choice['params'],jobs=args.jobs);pred=b.forest.predict(artifact,te);assert np.isfinite(pred).all()
   rows=[dict(example=r.example,state=r.state,cohort=r.cohort,centroid_shift_hours=r.centroid_shift_hours,predicted_hours=float(v),panel=panel) for r,v in zip(te.itertuples(),pred)]
   path.write_text(json.dumps(dict(fingerprint=fp,rows=rows),indent=2)+'\n');print(panel,ex,'saved',flush=True)
 b.COLUMNS=list(COLUMNS);summarize(design,fp);verify_sources(prep)
 (DEST/'completion.json').write_text(json.dumps(dict(complete=True,fingerprint=fp,fits=75,production_modified=False),indent=2)+'\n')
if __name__=='__main__':main()
