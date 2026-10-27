"""Audit segmented timing, route-time decomposition and fitted leaf neighborhoods."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from geometry_audit import OUT,pilot
from rapid_tools.hydrograph_response import distribution
import segmented_search as search
DEST=OUT/'segmented_diagnostics'

def main():
 DEST.mkdir(exist_ok=True);plan=pilot.read(OUT/'routing_plan.json');responses=pilot.read(OUT/'responses.parquet');metrics=pilot.read(OUT/'state_metrics.parquet');summary=[];blocks=[];checks=[]
 for case in plan['cases']:
  ex=case['example'];jobs={j['state_id']:j for j in case['jobs']};p=pilot.read(jobs['S000_base']['parameters']);c=pilot.read(jobs['all_closed_units_collapsed']['parameters']);line=pilot.read(OUT/f'variants/{ex}/all_closed_units_collapsed/lineage.csv')
  for f in [p,c]:
   alloc=pilot.allocation_tables(f)[1].set_index('id_link').r_edge
   np.testing.assert_allclose(f.reference_flow_fraction,alloc.loc[f.id_link],rtol=1e-10,atol=1e-12)
   np.testing.assert_allclose(f['count']*f.rapid_k,f.link_length_m/f.rapid_celerity_mps,rtol=1e-10)
   np.testing.assert_allclose(f.slope,(f.head_us-f.head_ds)/f.link_length_m,rtol=1e-10,atol=1e-12)
  for role,g in line.groupby('role'):
   if role=='unchanged':
    a=p.set_index('id_link').loc[g.parent_link_id];b=c.set_index('id_link').loc[g.child_link_id]
    cols=['link_length_m','width','depth_m','slope','rapid_celerity_mps','rapid_k','rapid_x','count','reference_flow_fraction'];np.testing.assert_allclose(a[cols],b[cols],rtol=1e-12,atol=1e-12)
  for state,g in line[line.role.eq('collapsed_block')].groupby('block_state'):
   a=p[p.id_link.isin(g.parent_link_id)];b=c[c.id_link.isin(g.child_link_id)];assert len(b)==1;b=b.iloc[0]
   lp=float((a.reference_flow_fraction*a.link_length_m).sum());tp=float((a.reference_flow_fraction*a.link_length_m/a.rapid_celerity_mps).sum());lc=float(b.link_length_m);tc=lc/b.rapid_celerity_mps
   length=(lc-lp)*tp/lp;speed=lc*(1/b.rapid_celerity_mps-tp/lp);assert np.isclose(length+speed,tc-tp)
   blocks.append(dict(example=ex,state=state,parent_length_km=lp/1000,child_length_km=lc/1000,parent_effective_speed=lp/tp,child_speed=b.rapid_celerity_mps,parent_time_hours=tp/3600,child_time_hours=tc/3600,delta_hours=(tc-tp)/3600,length_term_hours=length/3600,speed_term_hours=speed/3600,child_width=b.width,child_depth=b.depth_m,child_slope=b.slope,child_head_drop=b.head_us-b.head_ds,child_speed_capped=bool(b.rapid_celerity_capped)))
  tp=(p.reference_flow_fraction*p.link_length_m/p.rapid_celerity_mps).sum()/3600;tc=(c.reference_flow_fraction*c.link_length_m/c.rapid_celerity_mps).sum()/3600
  r=responses[(responses.example==ex)&responses.cohort.eq('segmented')&responses.window.eq('0.5%')].iloc[0]
  summary.append(dict(example=ex,parent_time_hours=tp,child_time_hours=tc,delta_time_hours=tc-tp,centroid_shift_hours=r.centroid_shift_hours,peak_shift_hours=r.peak_shift_hours,initial_cms=case['initial_cms'],final_cms=case['final_cms'],baseline_cms=r.baseline_cms,n_manning=float(p.n_manning.iloc[0]),beta=float(p.diffusion_multiplier.iloc[0])))
  if ex=='22':
   for window in ['0.5%','1%','2%']:
    ms=metrics[(metrics.example==ex)&(metrics.window==window)].set_index('state');vals={}
    for state in ['S000_base','all_closed_units_collapsed']:
     row=ms.loc[state];curve=pilot.read(row.outlet);t=curve.time_utc.astype('int64').to_numpy()/1e9;q=curve.q_outlet_cms.to_numpy();mask=(t>=row.rise_seconds)&(t<=row.fall_seconds);m=distribution(t[mask],q[mask]-row.baseline_cms);peak=float(t[mask][np.argmax(q[mask])]);np.testing.assert_allclose(m['centroid_seconds'],row.centroid_seconds,atol=1e-6,rtol=0);assert peak==row.peak_seconds;vals[state]=(m['centroid_seconds'],peak)
    shift=(vals['all_closed_units_collapsed'][0]-vals['S000_base'][0])/3600;rr=responses[(responses.example==ex)&responses.cohort.eq('segmented')&responses.window.eq(window)].iloc[0];assert np.isclose(shift,rr.centroid_shift_hours)
    checks.append(dict(window=window,recomputed_centroid_shift_hours=shift,recomputed_peak_shift_hours=(vals['all_closed_units_collapsed'][1]-vals['S000_base'][1])/3600))
 summary=pd.DataFrame(summary);blocks=pd.DataFrame(blocks);summary.to_csv(DEST/'routing_summary.csv',index=False);blocks.to_csv(DEST/'block_attribution.csv',index=False);pd.DataFrame(checks).to_csv(DEST/'response22_recomputed.csv',index=False)
 for r in summary.itertuples():assert np.isclose(blocks[blocks.example==r.example].delta_hours.sum(),r.delta_time_hours,atol=1e-10)
 # Refit exact outer models to inspect geometric neighborhoods, not to retune.
 d=pd.read_parquet(search.OUT/'input.parquet');selections=json.loads((search.OUT/'outer_selections.json').read_text());saved=pd.read_parquet(search.OUT/'predictions.parquet');neighbors=[];leafrows=[]
 for ex in ['22','46','55']:
  choice=next(r for r in selections if r['held_example']==ex and r['selection']=='joint')['candidate'];tr=d[d.example!=ex].copy();te=d[(d.example==ex)&d.cohort.eq('segmented')];artifact=search.fit(tr,choice,4);pred=search.predict(artifact,te)[0];ref=saved[(saved.variant=='tuned_joint')&(saved.example==ex)&saved.cohort.eq('segmented')].predicted_hours.iloc[0];assert np.isclose(pred,ref,rtol=1e-12,atol=1e-12)
  xt=artifact['preprocessor'].transform(search.transformed(tr,choice['size']));x=artifact['preprocessor'].transform(search.transformed(te,choice['size']));dist=np.sqrt(((xt-x)**2).sum(axis=1));near=tr.copy();near['standardized_distance']=dist;near['held_example']=ex;neighbors.extend(near.nsmallest(8,'standardized_distance').to_dict('records'))
  weights=search.b.engine.weights(tr);leafweight=np.zeros(len(tr));encounters=np.zeros(len(tr))
  for tree in artifact['estimator'].estimators_:
   leaf=tree.apply(x)[0];mask=tree.apply(xt)==leaf;encounters+=mask;leafweight[mask]+=weights[mask]/weights[mask].sum()/len(artifact['estimator'].estimators_)
  # Geometric co-leaf proximity on all training rows, not bootstrap-aware SHAP/influence.
  z=tr[['example','state','cohort','centroid_shift_hours','domain_size','domain_size_edited']].copy();z['co_leaf_weight']=leafweight;z['tree_encounter_fraction']=encounters/len(artifact['estimator'].estimators_);z['held_example']=ex;leafrows.extend(z.to_dict('records'))
 pd.DataFrame(neighbors).to_csv(DEST/'geometric_neighbors.csv',index=False);pd.DataFrame(leafrows).to_parquet(DEST/'leaf_neighborhoods.parquet',index=False)
 pilot.save(DEST/'manifest.json',dict(sources=pilot.SOURCES,notes=['Route-time diagnostic uses calibrated model speeds, not proposed hydrography-only predictors','Block length/speed decomposition is an algebraic attribution, not a causal intervention','Co-leaf weights use all training rows and original sample weights, not bootstrap multiplicities; not exact prediction attribution','Original frozen model unchanged']))
 print(summary.round(4).to_string(index=False));print(blocks[blocks.example=='22'].sort_values('delta_hours').head(8).round(4).to_string(index=False))
if __name__=='__main__':main()
