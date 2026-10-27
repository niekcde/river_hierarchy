"""Focused network 37 CB001 bubble 3 audit; run from repository root with PYTHONPATH=.:RAPID/src.

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
import importlib.util
cap_spec=importlib.util.spec_from_file_location('cap_check',Path(__file__).with_name('check_cap_centroids_20261019.py'))
cap_module=importlib.util.module_from_spec(cap_spec);cap_spec.loader.exec_module(cap_module)
state_metrics=pd.read_parquet('RAPID/outputs/hydrograph_response/state_metrics.parquet')
out=Path('RAPID/outputs/hydrograph_case37_checks_20261019')
out.mkdir(parents=True,exist_ok=True)
a=build_current_analysis_frame(['37']); summaries=[]
for st in ['CB001_bubble_3']:
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
 if st=='CB001_bubble_3':
  parent_meta=state_metrics[state_metrics.example.eq('37')&state_metrics.state.eq('S000_base')].iloc[0]
  child_meta=state_metrics[state_metrics.example.eq('37')&state_metrics.state.eq(st)].iloc[0]
  parent=pd.read_parquet(parent_meta.outlet);child=pd.read_parquet(child_meta.outlet)
  forcing=pd.read_parquet(Path(parent_meta.outlet).parent.parent/'forcing.parquet')
  trials=[]
  for side,key,saved in [('parent','parent_parameters_parquet',parent),('child','parameters_parquet',child)]:
   full=pd.read_parquet(r[key]);res=json.loads((Path(r[key]).parent.parent/'run/result.json').read_text())
   for factor in [1,2]:
    meta=parent_meta if side=='parent' else child_meta
    initial=json.loads((Path(meta.outlet).parent/'completion.json').read_text())['audit']['initial_outlet_cms']
    val,audit=route_interval_means(full,forcing.forcing_cms.to_numpy(),res['nsub']*factor,res['native_dt_s']/factor,initial_flow=initial)
    if factor==1: np.testing.assert_allclose(val,saved.q_outlet_cms.to_numpy(),rtol=0,atol=1e-8)
    sr=state_metrics[state_metrics.example.eq('37')&state_metrics.state.eq('S000_base' if side=='parent' else st)].iloc[0]
    t=pd.to_datetime(saved.time_utc,utc=True).astype('int64').to_numpy()/1e9
    tt=np.r_[sr.rise_seconds,t[(t>sr.rise_seconds)&(t<sr.fall_seconds)],sr.fall_seconds]
    excess=np.interp(tt,t,val)-sr.baseline_cms
    cap_time=cap_module.cap_centroid(tt,excess,.8)['cap_centroid_seconds']
    trials.append(dict(cap_centroid_seconds=cap_time,side=side,factor=factor,peak_index=int(np.argmax(val)),peak_cms=float(max(val)),max_difference_saved=float(np.max(np.abs(val-saved.q_outlet_cms.to_numpy()))),**audit))
  print('TRIALS',trials);(out/'bubble37_timestep_checks.json').write_text(json.dumps(trials,indent=2))
(out/'domain_routing_checks.json').write_text(json.dumps(summaries,indent=2))
