"""Bounded all-16 robust smoothing selection, gap anchoring and base allocation."""
from pathlib import Path
import sys,json,hashlib,time
from concurrent.futures import ProcessPoolExecutor
import numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT/'RAPID/src'),str(ROOT/'RAPID/investigations')]
from rapid_tools.robust_reference_profile import (prepare_problem,fit_robust_profile,training_delta,score_held_cycles,disconnected_gap_anchors)
import smoothing_floor_allocation_tests as allocation
PREV=ROOT/'RAPID/assets/smoothing_floor_allocation_20261002';ROBUST=ROOT/'RAPID/assets/robust_slope_loss_20261002';OUT=ROOT/'RAPID/assets/bounded_robust_slopes_20261002';OUT.mkdir(parents=True,exist_ok=True)
SOURCES=json.loads((PREV/'method.json').read_text())['sources']

def run(ex):
 start=time.monotonic();source=Path(SOURCES[ex]);dest=OUT/ex/'floor_0.05';dest.mkdir(parents=True,exist_ok=True)
 f=pd.read_parquet(PREV/ex/'floor_0.05/profile.parquet');o=pd.read_parquet(source/'reference_profile_observations.parquet');meta=json.loads((source/'reference_profile_audit.json').read_text())['summary'];ref=meta['reference_slope_m_per_m'];lam0=json.loads((PREV/ex/'floor_0.05/selection.json').read_text())['selected_lambda'];bridges=json.loads((source/'reference_profile_selection.json').read_text()).get('manual_adjustments',[]);gaps=disconnected_gap_anchors(f,o,.05*ref)
 (dest/'gap_anchors.json').write_text(json.dumps(gaps,indent=2));(dest/'manual_bridges.json').write_text(json.dumps(bridges,indent=2))
 cycles=np.array(sorted(o.cycle.unique()));np.random.default_rng(20261002).shuffle(cycles);foldmap={v:i%3 for i,v in enumerate(cycles)};fold=o.cycle.map(foldmap).to_numpy();(dest/'folds.json').write_text(json.dumps({str(k):v for k,v in foldmap.items()},indent=2))
 problems=[];deltas=[];squared_solutions=[];anchor_audits=[]
 for k in range(3):
  p=prepare_problem(f,o[fold!=k],ref,weight_total=len(o),gaps=gaps,bridges=bridges,initial_heights=f.fitted_wse_m.to_numpy())
  _,v,qa=fit_robust_profile(p,lam0,np.inf);problems.append(p);deltas.append(training_delta(p,v));squared_solutions.append(v);anchor_audits.append(qa)
 (dest/'training_scales.json').write_text(json.dumps(dict(deltas=deltas,squared_anchor_audits=anchor_audits,anchor_lambda=lam0),indent=2))
 records={};predictions=[]
 def evaluate(lam):
  key=f'{lam:.12g}'
  if key in records:return records[key]
  robust=[];ordinary=[];audits=[]
  try:
   for k,p in enumerate(problems):
    h,v,qa=fit_robust_profile(p,lam,deltas[k],initial=squared_solutions[k]);score=score_held_cycles(o[fold==k],p['x'],h,deltas[k]);robust.extend(score.robust.tolist());ordinary.extend(score.squared.tolist());audits.append(qa)
    for cycle,z in score.iterrows():predictions.append(dict(example=ex,lambda_value=lam,fold=k,cycle=str(cycle),huber_loss=z.robust,mse=z.squared,delta=deltas[k]))
   rec=dict(lambda_value=lam,cv_huber=float(np.mean(robust)),cv_mse=float(np.mean(ordinary)),cv_se=float(np.std(robust,ddof=1)/np.sqrt(len(robust))),fold_audits=audits,status='complete')
  except Exception as err:rec=dict(lambda_value=lam,status='failed',error=repr(err))
  records[key]=rec;(dest/'cv.json').write_text(json.dumps(records,indent=2));print(ex,'CV',lam,rec['status'],rec.get('cv_huber'),flush=True);return rec
 for factor in [.01,.1,1.,10.,100.]:evaluate(lam0*factor)
 for _ in range(3):
  good=[v for v in records.values() if v['status']=='complete'];assert good;best=min(good,key=lambda z:z['cv_huber']);lo=min(z['lambda_value'] for z in good);hi=max(z['lambda_value'] for z in good)
  if best['lambda_value']==lo:evaluate(lo/10)
  elif best['lambda_value']==hi:evaluate(hi*10)
  else:break
 good=[v for v in records.values() if v['status']=='complete'];best=min(good,key=lambda z:z['cv_huber'])
 for factor in [10**-.5,10**.5]:evaluate(best['lambda_value']*factor)
 good=[v for v in records.values() if v['status']=='complete'];minimum=min(z['cv_huber'] for z in good);best=max([z for z in good if z['cv_huber']<=minimum+1e-6*max(minimum,1e-12)],key=lambda z:z['lambda_value']);lam=best['lambda_value']
 pd.DataFrame(predictions).to_csv(dest/'held_cycle_scores.csv',index=False)
 p=prepare_problem(f,o,ref,gaps=gaps,bridges=bridges,initial_heights=f.fitted_wse_m.to_numpy());_,seed,sq_audit=fit_robust_profile(p,lam0,np.inf);delta=training_delta(p,seed);h,v,qa=fit_robust_profile(p,lam,delta,initial=seed)
 profile=f.copy();profile['previous_candidate']=f.fitted_wse_m;profile['previous_robust']=pd.read_parquet(ROBUST/ex/'floor_0.05/profile.parquet').fitted_wse_m;profile['fitted_wse_m']=h;profile['slope']=np.r_[np.nan,np.diff(h)/np.diff(p['x'])];profile.to_parquet(dest/'profile.parquet',index=False)
 obs=o.copy();obs['fitted_profile_wse_m']=h[p['ni']];obs['acquisition_stage_offset_m']=v[len(h):][p['ai']];obs['residual_m']=o.wse_m.to_numpy()-obs.fitted_profile_wse_m-obs.acquisition_stage_offset_m;obs['stage_aligned_wse_m']=o.wse_m.to_numpy()-obs.acquisition_stage_offset_m;obs['robust_weight_multiplier']=np.minimum(1,delta/np.maximum(abs(obs.residual_m),1e-15));obs.to_parquet(dest/'observations.parquet',index=False)
 pd.DataFrame({'acquisition_id':p['acqs'],'stage_offset_m':v[len(h):]}).to_csv(dest/'acquisition_offsets.csv',index=False)
 suffix='_direction130' if ex=='15' else '';hc=ROOT/'network_variants/output'/f'{ex}_indep{suffix}__anchored/states/S000_base/hydraulic_context';nodes=pd.read_csv(hc/'node_hydraulic_context.csv');links=pd.read_csv(hc/'link_hydraulic_context.csv');allocation.OUT=OUT;a=allocation.units(ex,.05,f,h,links,nodes,'old');ends=[nodes.reference_section_dist_out_m.min(),nodes.reference_section_dist_out_m.max()];drop=lambda y:float(np.diff(np.interp(ends,p['x'],y))[0])
 s=dict(example=ex,selected_lambda=lam,previous_lambda=lam0,lambda_ratio=lam/lam0,cv_huber=best['cv_huber'],cv_rmse=np.sqrt(best['cv_mse']),cv_grid_edge=lam in [min(z['lambda_value'] for z in good),max(z['lambda_value'] for z in good)],failed_cv=sum(z['status']=='failed' for z in records.values()),delta_m=delta,gap_count=len(gaps),manual_bridge_count=len(bridges),old_drop=drop(f.fitted_wse_m),new_drop=drop(h),floor_active_pct=float(100*np.sum(np.diff(p['x'])[np.diff(h)/np.diff(p['x'])<=.05*ref*(1+1e-4)+1e-10])/np.ptp(p['x'])),min_profile_slope=float(np.min(np.diff(h)/np.diff(p['x']))),**{k:z for k,z in a.items() if k!='example'},profile_solver=qa,seconds=time.monotonic()-start)
 (dest/'selection.json').write_text(json.dumps(s,indent=2));print('DONE',ex,a['below_1e5'],'lambda ratio',lam/lam0,flush=True);return s

def main():
 selected=sys.argv[1:] or list(SOURCES)
 with ProcessPoolExecutor(max_workers=2) as pool:list(pool.map(run,selected))
 rows=[json.loads(p.read_text()) for p in sorted(OUT.glob('*/floor_0.05/selection.json'))];pd.DataFrame([{k:v for k,v in row.items() if k!='profile_solver'} for row in rows]).to_csv(OUT/'summary.csv',index=False)
 (OUT/'method.json').write_text(json.dumps(dict(sources=SOURCES,method='Huber + normalized curvature; explicit floor, gauge, linear bridge and SWORD gap-head constraints',cv='Three whole-cycle folds, seed20261002. Training-only delta from squared-loss anchor at previous lambda. Fixed total base observation weight. Minimum mean held-cycle Huber loss; 1e-6 relative tie picks larger lambda. 5 initial log candidates, at most 3 edge extensions and 2 half-decade refinements.',gap='Only spatially disconnected acquisition-span components; static SWORD endpoint head difference; floor-incompatible reference fails explicitly.',manual='Retain documented case46 bridge as a linear constraint inside all fits.',production_changed=False,script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),indent=2))
if __name__=='__main__':main()
