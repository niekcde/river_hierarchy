"""Focused CB38/CB28 audit; run from repository root with PYTHONPATH=.:RAPID/src.

No refitting or production mutations. Travel time and celerity are model
explanations, not proposed hydrography-only predictors. Path probabilities use
width splits and confluence accumulation within the validated collapse domain.
"""
import json
from pathlib import Path
import pandas as pd, numpy as np, networkx as nx
from rapid_tools.current_review import build_current_analysis_frame, load_pair
from rapid_tools.current_domains import load_domain_pair, allocation_tables
from rapid_tools.common_grid import route_interval_means
out=Path('RAPID/outputs/hydrograph_case_checks_20261019')
out.mkdir(parents=True,exist_ok=True)
a=build_current_analysis_frame(['55']); summaries=[]
for st in ['CB001_bubble_38','CB024_bubble_28']:
 r=a[a.state_id.eq(st)].iloc[0];p,c,meta=load_domain_pair(r)
 for side,f,key in [('parent',p,'parent_parameters_parquet'),('child',c,'parameters_parquet')]:
  nodes,edges=allocation_tables(f);e=edges.merge(f,on='id_link',validate='one_to_one');g=nx.DiGraph();g.add_edges_from(zip(e.u_x,e.v_x));order=list(nx.topological_sort(g));best={order[0]:1.};paths={order[0]:[]}
  for u in order:
   for row in e[e.u_x.eq(u)].itertuples():
    pr=best[u]*row.p_outgoing
    if pr>best.get(row.v_x,-1):best[row.v_x]=pr;paths[row.v_x]=paths[u]+[row.id_link]
  result=json.loads((Path(r[key]).parent.parent/'run/result.json').read_text());dt=result['native_dt_s']
  k=f.rapid_k;x=f.rapid_x;den=dt+2*k*(1-x);decay=(2*k*(1-x)-dt)/den
  summary=dict(state=st,side=side,links=len(f),total_length_m=float(f.link_length_m.sum()),mean_length_m=float((e.r_edge*e.link_length_m).sum()),mean_travel_minutes=float((e.r_edge*e.link_length_m/e.rapid_celerity_mps).sum()/60),dominant_path_probability=best[order[-1]],dominant_path_links=paths[order[-1]],celerity_min=float(f.rapid_celerity_mps.min()),celerity_median=float(f.rapid_celerity_mps.median()),celerity_max=float(f.rapid_celerity_mps.max()),capped=int(f.rapid_celerity_capped.sum()),slope_min=float(f.slope.min()),slope_max=float(f.slope.max()),x_min=float(x.min()),x_median=float(x.median()),x_max=float(x.max()),decay_min=float(decay.min()),decay_max=float(decay.max()),n=f.n_manning.unique().tolist(),native_dt=dt,minimum_native_flow=result['minimum_native_flow_cms'])
  summaries.append(summary)
  print(json.dumps(summary))
 if st=='CB001_bubble_38':
  parent,child,forcing=load_pair(r)
  publication=json.loads((Path(r.parameters_parquet).parent.parent/'publication.json').read_text())
  forcing=pd.read_parquet(publication['run_forcing']);print('forcing columns',forcing.columns.tolist())
  trials=[]
  for side,key,saved in [('parent','parent_parameters_parquet',parent),('child','parameters_parquet',child)]:
   full=pd.read_parquet(r[key]);res=json.loads((Path(r[key]).parent.parent/'run/result.json').read_text())
   for factor in [1,2]:
    val,audit=route_interval_means(full,forcing.forcing_cms.to_numpy(),res['nsub']*factor,res['native_dt_s']/factor,initial_flow=res['initial_cms'])
    if factor==1: np.testing.assert_allclose(val,saved.q_outlet_cms.to_numpy(),rtol=0,atol=1e-8)
    trials.append(dict(side=side,factor=factor,peak_index=int(np.argmax(val)),peak_cms=float(max(val)),max_difference_saved=float(np.max(np.abs(val-saved.q_outlet_cms.to_numpy()))),**audit))
  print('TRIALS',trials);(out/'cb38_timestep_checks.json').write_text(json.dumps(trials,indent=2))
(out/'domain_routing_checks.json').write_text(json.dumps(summaries,indent=2))
