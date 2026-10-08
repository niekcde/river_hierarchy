"""Fixed geometric benchmarks on refreshed real cases and frozen synthetic transfer."""
from pathlib import Path
import sys,json,itertools
import numpy as np
import pandas as pd
import networkx as nx
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import r2_score,mean_absolute_error
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'RAPID/src'))
from rapid_tools.refreshed_analysis import build_refreshed_analysis_frame,DEFAULT_REFRESH_ROOT
OUT=ROOT/'RAPID/validation/real_predictors_20261005'
R='resistance_log';LF='length_fraction';E='width_fraction';L='parent_equivalent_length';EL='parent_elongation';D='parent_dominant_width_fraction';CV='parent_path_length_cv';A='signed_length_width_association';S='domain_reference_slope'
FULL=[R,LF,E,L,EL,D,CV,A,S];SELECTED=[R,E,L,D,A,S];CHANGE=['log_equivalent_length_ratio','log_elongation_ratio','change_path_length_cv','change_width_dominance'];CONTEXT=['upstream_shortest_length_m','downstream_shortest_length_m']
SETS={**{c:[c] for c in ['omega']+FULL},'components':[R,LF,E],'selected_six':SELECTED,'all_geometry':FULL,'selected_plus_change':SELECTED+CHANGE,'selected_plus_change_context':SELECTED+CHANGE+CONTEXT,'omega_times_dominance':['omega_times_dominance'],'omega_times_association':['omega_times_association'],'omega_times_slope':['omega_times_slope']}
Y=['lag_shift_pct','event_duration_change_pct','peak_excess_change_pct','interpolated_lag_shift_pct']

def estimator(kind):
 if kind=='linear':return make_pipeline(StandardScaler(),Ridge(alpha=1.))
 return HistGradientBoostingRegressor(max_iter=80,max_leaf_nodes=7,min_samples_leaf=20,l2_regularization=1.,early_stopping=False,random_state=20260929)

def geometry(f,ids):
 f=f[f.id_link.isin(ids)];g=nx.MultiDiGraph()
 for r in f.itertuples():g.add_edge(r.id_us_node,r.id_ds_node,key=int(r.id_link),length=r.link_length_m,width=r.width)
 sources=[n for n in g if g.in_degree(n)==0];sinks=[n for n in g if g.out_degree(n)==0]
 if len(sources)!=1 or len(sinks)!=1 or not nx.is_directed_acyclic_graph(g):raise ValueError('Domain needs one source/sink and no cycle')
 u,v=sources[0],sinks[0];counts={u:1}
 for node in nx.topological_sort(g):
  for _,target,key in g.out_edges(node,keys=True):counts[target]=min(10001,counts.get(target,0)+counts.get(node,0))
 if counts.get(v,0)>10000:raise ValueError('Path enumeration exceeds 10000')
 paths=list(nx.all_simple_edge_paths(g,u,v))
 if not paths or len(paths)>10000:raise ValueError('Path enumeration exceeds 10000 or empty')
 if set(e for p in paths for e in p)!=set(g.edges(keys=True)):raise ValueError('Off-path domain edges')
 lengths=[];widths=[]
 for p in paths:
  ls=np.array([g.edges[e]['length'] for e in p]);ws=np.array([g.edges[e]['width'] for e in p]);lengths.append(ls.sum());widths.append(ls.sum()/sum(ls/ws))
 ls=np.array(lengths);ws=np.array(widths);leq=np.average(ls,weights=ws)
 heads={**dict(zip(f.id_us_node,f.head_us)),**dict(zip(f.id_ds_node,f.head_ds))}
 return dict(equivalent_length=leq,elongation=leq/ws.sum(),dominant_width_fraction=ws.max()/ws.sum(),path_length_cv=ls.std()/ls.mean(),association=leq/ls.mean()-1,slope=(heads[u]-heads[v])/ls.min(),paths=len(paths)),u,v

def build():
 ids=sorted(p.name for p in DEFAULT_REFRESH_ROOT.iterdir() if (p/'inventory.csv').exists())
 if '--reuse-analysis' in sys.argv:d=pd.read_parquet(OUT/'analysis.parquet')
 else:
  d=build_refreshed_analysis_frame(ids);d.to_parquet(OUT/'analysis.parquet',index=False)
 records=[];issues=[]
 for ex,g in d[~d.technical_alias].groupby('example_id'):
  inv=pd.read_csv(DEFAULT_REFRESH_ROOT/ex/'inventory.csv');ready=inv[inv.status.eq('ready')].drop_duplicates('fingerprint').set_index('fingerprint');b=ready[ready.state.eq('S000_base')].iloc[0];parent=pd.read_parquet(b.frame);graph=nx.MultiDiGraph()
  for r in parent.itertuples():graph.add_edge(r.id_us_node,r.id_ds_node,key=r.id_link,length=r.link_length_m)
  inlet=next(n for n in graph if graph.in_degree(n)==0);outlet=next(n for n in graph if graph.out_degree(n)==0)
  up=nx.single_source_dijkstra_path_length(graph,inlet,weight='length');down=nx.single_source_dijkstra_path_length(graph.reverse(),outlet,weight='length');cache={};pcache={};ccache={}
  for r in g.itertuples():
   try:
    pids=[int(i) for i in r.matched_base_link_ids.split(',')];cids=[int(i) for i in r.child_link_ids.split(',')]
    if r.matched_base_link_ids not in pcache:
     try:pcache[r.matched_base_link_ids]=geometry(parent,pids)
     except ValueError as err:pcache[r.matched_base_link_ids]=err
    if isinstance(pcache[r.matched_base_link_ids],ValueError):raise pcache[r.matched_base_link_ids]
    p,u,v=pcache[r.matched_base_link_ids]
    if r.fingerprint not in cache:cache[r.fingerprint]=pd.read_parquet(ready.loc[r.fingerprint,'frame'])
    ckey=(r.fingerprint,r.child_link_ids)
    if ckey not in ccache:ccache[ckey]=geometry(cache[r.fingerprint],cids)
    c,_,_=ccache[ckey];row=r._asdict();row.update({f'parent_{k}':val for k,val in p.items()});row.update({f'child_{k}':val for k,val in c.items()})
    row.update(resistance_log=r.neg_log10_lambda,omega=r.len_fraction_width_weighted_log_lambda,signed_length_width_association=p['association'],domain_reference_slope=p['slope'],upstream_shortest_length_m=up[u],downstream_shortest_length_m=down[v],lag_shift_pct=r.lag_to_inflow_peak_pct_shift_vs_base)
    row.update(log_equivalent_length_ratio=np.log(c['equivalent_length']/p['equivalent_length']),log_elongation_ratio=np.log(c['elongation']/p['elongation']),change_path_length_cv=c['path_length_cv']-p['path_length_cv'],change_width_dominance=c['dominant_width_fraction']-p['dominant_width_fraction']);records.append(row)
   except (ValueError,KeyError,nx.NetworkXError) as err:issues.append(dict(example_id=ex,state_id=r.state_id,reason=str(err)))
  print(ex,'geometry complete',flush=True)
 pd.DataFrame(issues,columns=['example_id','state_id','reason']).to_csv(OUT/'excluded_geometry.csv',index=False)
 f=pd.DataFrame(records);f['comparison_alias']=f.duplicated(['example_id','fingerprint','matched_base_link_ids','child_link_ids']);f.to_parquet(OUT/'features_all.parquet',index=False)
 return f[~f.comparison_alias].copy()

def products(d):
 for label,col in [('dominance',D),('association',A),('slope',S)]:d['omega_times_'+label]=d.omega*d[col]
 return d

def score_rows(obs,pred,metadata,**tags):
 for scope,key in [('pooled',None),('example_id','example_id'),('mode','mode')]:
  groups=[('all',np.ones(len(obs),bool))] if key is None else [(str(v),(metadata[key]==v).to_numpy()) for v in sorted(metadata[key].unique())]
  for label,m in groups:
   yield dict(**tags,scope=scope,group=label,n=int(m.sum()),r2=r2_score(obs[m],pred[m]) if m.sum()>1 and np.ptp(obs[m])>0 else np.nan,mae=mean_absolute_error(obs[m],pred[m]))

def main():
 OUT.mkdir(parents=True,exist_ok=True);d=products(build());d.to_parquet(OUT/'features.parquet',index=False)
 scores=[];predictions=[];corr=[]
 for feature in ['omega']+FULL+CHANGE+CONTEXT:
  for y in Y:
   for ex,g in [('all',d)]+list(d.groupby('example_id')):
    corr.append(dict(feature=feature,response=y,example_id=ex,n=len(g),spearman=g[feature].corr(g[y],method='spearman') if g[feature].nunique()>1 else np.nan))
 pd.DataFrame(corr).to_csv(OUT/'correlations.csv',index=False)
 for label,cols in SETS.items():
  for y in Y:
   valid=d[np.isfinite(d[cols+[y]]).all(axis=1)].reset_index(drop=True);obs=valid[y].to_numpy()
   for kind in ['linear','boosted']:
    pred=np.full(len(valid),np.nan);null=pred.copy()
    for ex in valid.example_id.unique():
     te=valid.example_id.eq(ex);tr=~te;fit=estimator(kind).fit(valid.loc[tr,cols],valid.loc[tr,y]);pred[te]=fit.predict(valid.loc[te,cols]);null[te]=valid.loc[tr,y].mean()
    scores.extend(score_rows(obs,pred,valid,validation='leave_river_out',feature_set=label,model=kind,response=y))
    p=valid[['example_id','state_id','mode']].copy();p=p.assign(feature_set=label,model=kind,response=y,observed=obs,predicted=pred,null_prediction=null);predictions.append(p)
   if label=='omega':scores.extend(score_rows(obs,null,valid,validation='leave_river_out',feature_set='training_mean',model='baseline',response=y))
  pd.DataFrame(scores).to_csv(OUT/'scores.csv',index=False);print('scored',label,flush=True)
 pd.concat(predictions).to_parquet(OUT/'predictions.parquet',index=False)
 synthpath=ROOT/'synthetic_experiments/outputs/hydraulic_alpha065_20260930/full_grid_island/current_analysis/compact_predictors/features.parquet';s=products(pd.read_parquet(synthpath));train=s[s.split.eq('development')];test=s[~s.split.eq('development')];transfer=[];ranges=[]
 for col in FULL:
  ranges.append(dict(feature=col,synthetic_min=train[col].min(),synthetic_max=train[col].max(),real_min=d[col].min(),real_max=d[col].max(),outside_fraction=float(((d[col]<train[col].min())|(d[col]>train[col].max())).mean())))
 for label in ['omega','components','selected_six','selected_plus_change_context']:
  cols=SETS[label]
  for y in Y:
   sy='lag_shift_pct' if y=='interpolated_lag_shift_pct' else y;fit=estimator('boosted').fit(train[cols],train[sy])
   real=d[np.isfinite(d[cols+[y]]).all(axis=1)];pred=fit.predict(real[cols]);transfer.extend(score_rows(real[y].to_numpy(),pred,real,validation='frozen_synthetic_to_real',feature_set=label,model='boosted',response=y))
   if y!='interpolated_lag_shift_pct':transfer.append(dict(validation='synthetic_geometry_holdout',feature_set=label,model='boosted',response=y,scope='pooled',group='all',n=len(test),r2=r2_score(test[y],fit.predict(test[cols])),mae=mean_absolute_error(test[y],fit.predict(test[cols]))))
 pd.DataFrame(transfer).to_csv(OUT/'transfer_scores.csv',index=False);pd.DataFrame(ranges).to_csv(OUT/'feature_ranges.csv',index=False)
 (OUT/'provenance.json').write_text(json.dumps(dict(source=str(DEFAULT_REFRESH_ROOT),synthetic_source=str(synthpath),feature_sets=SETS,responses=Y,comparisons=len(d),excluded_geometry=len(pd.read_csv(OUT/'excluded_geometry.csv')),rules='Fixed models and groups; no tuning on real scores. Whole rivers held out. Identical fingerprint+domain aliases deduplicated. All eligible domains included, no edge-compound filtering. Refined lag is a sensitivity: frozen synthetic training target remains sampled lag. Peak excess uses parent max(initial,final). Path descriptors enumerate <=10000 source-sink paths; exclusions explicit. Parent/child domain measures are not claims about independent parallel physical channels.'),indent=2))
if __name__=='__main__':main()
