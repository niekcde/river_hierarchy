"""Exclusion diagnostics and matched forest-neighbor hydraulic context; no fits."""
import json
import numpy as np
import pandas as pd
from slope_ratio_audit import ROOT,OUT,EXT,SC,read,SOURCES,validate_domain,allocation_tables
import hashlib
from pathlib import Path
DEST=OUT/'dominant_outlier_context_20261023'
TARGETS=[('22','CB001_bubble_9'),('22','CB002_bubble_11'),*[(e,'all_closed_units_collapsed') for e in ['22','46','55']]]
def main():
 DEST.mkdir(exist_ok=True)
 d=read(OUT/'outlier_diagnosis_20261023/ranked_population.parquet');scores=[]
 for cohort,keys in [('original',TARGETS[:2]),('segmented',TARGETS[2:])]:
  g=d[d.cohort.eq(cohort)]
  selections=[('all',[]),*[(f'without_{e}_{s}',[(e,s)]) for e,s in keys],('without_all_priority',keys)]
  for label,remove in selections:
   z=g[~pd.MultiIndex.from_frame(g[['example','state']]).isin(remove)]
   w=1/z.groupby('example').example.transform('size').to_numpy();y=z.centroid_shift_hours.to_numpy();pred=z.predicted_hours.to_numpy();sse=np.sum(w*(pred-y)**2);sst=np.sum(w*(y-np.average(y,weights=w))**2)
   scores.append(dict(cohort=cohort,subset=label,n=len(z),networks=z.example.nunique(),r2=1-sse/sst,mae_minutes=60*np.average(abs(pred-y),weights=w),relative_error_pct=100*np.average(abs(pred-y)/np.maximum(abs(y),1/60),weights=w),weighted_sse=sse,weighted_sst=sst))
 pd.DataFrame(scores).to_csv(DEST/'exclusion_scores.csv',index=False)
 n=read(OUT/'nine_error_audit/bootstrap_leaf_neighbors.csv')
 for col in ['held_example','neighbor_example']:n[col]=n[col].astype(str).str.zfill(2)
 n=n[pd.MultiIndex.from_frame(n[['held_example','held_state']]).isin(TARGETS)].groupby(['held_example','held_state'],sort=False).head(3)
 assert len(n)==15 and len(n.groupby(['held_example','held_state']))==5
 keys=set(TARGETS)|set(zip(n.neighbor_example,n.neighbor_state))
 # CSV example columns require explicit string normalization for joins.
 keys={(str(e).zfill(2),s) for e,s in keys}
 jobs={}
 for plan in [ROOT/'RAPID/outputs/event_window_plan_20261013/plan.json',EXT/'routing_plan.json',OUT/'routing_plan.json']:
  for case in read(plan)['cases']:
   for j in case['jobs']:
    for name in set([j['state_id'],*j.get('aliases',[])]):jobs[case['example'],name]=j
 meta=read(SC/'geometry_responses.parquet').set_index(['example','state']);graphs=read(SC/'domain_graphs.csv').set_index(['example','state','scope']);rows=[]
 for ex,state in sorted(keys):
  record=d[d.example.eq(ex)&d.state.eq(state)].iloc[0];entry=dict(example=ex,state=state,cohort=record.cohort,actual_centroid_h=record.centroid_shift_hours,predicted_centroid_h=record.predicted_hours)
  for side,key in [('parent','S000_base'),('child',state)]:
   full=read(jobs[ex,key]['parameters'])
   if record.cohort=='segmented':
    line=read(EXT/f'variants/{ex}/{state}/lineage.csv');ids=line.loc[line.role.eq('collapsed_block'),side+'_link_id'];f=full[full.id_link.isin(ids)].copy();w=f.reference_flow_fraction*f.link_length_m
   else:
    g=graphs.loc[(ex,state,meta.loc[(ex,state),'B_scope'])];membership=read(SC/'graphs'/f'{g[side+"_graph"]}.parquet');part,_=validate_domain(full,membership);alloc=allocation_tables(part)[1].set_index('id_link');f=part.set_index('id_link').loc[alloc.index];w=alloc.mean_length_contribution_m
   length=float(w.sum()/1000);T=float((w/f.rapid_celerity_mps).sum()/3600);G=float((w/1000*(f.width/100)**(-13/30)/np.sqrt(f.slope)).sum())
   depthscale=f.depth_m/f.width**.65;assert np.std(depthscale)/np.mean(depthscale)<1e-10
   entry.update({side+'_mean_length_km':length,side+'_mean_width_m':float(np.average(f.width,weights=f.link_length_m)),side+'_slope_context':float(np.average(1/np.sqrt(f.slope),weights=w)),side+'_effective_celerity_mps':length*1000/(T*3600),side+'_travel_h':T,side+'_G':G,side+'_A_hours':T/G,side+'_depth_width_scale':float(depthscale.iloc[0]),side+'_n':float(f.n_manning.iloc[0])})
  entry['delta_G']=entry['child_G']-entry['parent_G'];entry['delta_travel_h']=entry['child_travel_h']-entry['parent_travel_h'];entry['reconstruction_error_h']=entry['parent_A_hours']*entry['delta_G']-entry['delta_travel_h'];assert abs(entry['reconstruction_error_h'])<1e-8
  entry['centroid_to_route_change_ratio']=entry['actual_centroid_h']/entry['delta_travel_h'];rows.append(entry)
 q=pd.DataFrame(rows);q.to_csv(DEST/'matched_physical_context.csv',index=False)
 contrasts=[]
 for r in n.itertuples():
  ex,ne=str(r.held_example).zfill(2),str(r.neighbor_example).zfill(2)
  a=q[q.example.eq(ex)&q.state.eq(r.held_state)].iloc[0];b=q[q.example.eq(ne)&q.state.eq(r.neighbor_state)].iloc[0]
  contrasts.append(dict(example=ex,state=r.held_state,neighbor_example=ne,neighbor_state=r.neighbor_state,bootstrap_leaf_weight=r.mean_bootstrap_leaf_weight,target_actual_h=a.actual_centroid_h,neighbor_actual_h=b.actual_centroid_h,parent_length_ratio=a.parent_mean_length_km/b.parent_mean_length_km,parent_width_ratio=a.parent_mean_width_m/b.parent_mean_width_m,delta_G_ratio=a.delta_G/b.delta_G,hydraulic_A_ratio=a.parent_A_hours/b.parent_A_hours,travel_change_ratio=a.delta_travel_h/b.delta_travel_h))
 pd.DataFrame(contrasts).to_csv(DEST/'neighbor_contrasts.csv',index=False)
 for path,h in SOURCES.items():assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==h
 (DEST/'manifest.json').write_text(json.dumps(dict(sources=SOURCES,code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),targets=TARGETS,exclusion='Fixed OOF predictions, no retraining. Recompute equal-network weights and target variance in each retained subset. Post-hoc sensitivity, not replacement headline score.',neighbors='Three highest prior bootstrap-leaf weights per selected case. Diagnostic association, not causal importance or exact prediction decomposition for median trees.',G='sum r_e*(L_e/1km)*(W_e/100m)^(-13/30)/sqrt(s_e). Within calibrated depth-width law, T=A_example*G. A is diagnostic and not a hydrography-only predictor. T is not the finite-window centroid response.',production_modified=False),indent=2)+'\n')
 print(pd.DataFrame(scores).to_string(index=False));print(pd.DataFrame(contrasts).round(4).to_string(index=False))
if __name__=='__main__':main()
