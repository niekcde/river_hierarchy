"""Isolated five-case profile-floor experiment; no production promotion or rerouting."""
from pathlib import Path
import sys,json,hashlib,math
from dataclasses import replace
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import pandas as pd
import pyogrio
from scipy.optimize import isotonic_regression
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'RAPID/src')]
import rapid_tools.reference_profile as profile_module
from rapid_tools.reference_profile import ReferenceProfileConfig,select_reference_profile,write_reference_profile_artifacts,apply_manual_linear_profile_bridge
from network_variants.reference_profile_candidate import generate_reference_profile_candidate_context
OUT=ROOT/'RAPID/assets/profile_floor_sensitivity_20261002'
CASES={'11':52,'13':174,'21':52,'41':95,'46':71}
FLOORS=[.01,.02,.05,.07,.09]

def run_case(ex):
 # Equivalent compiled PAVA; no change to objective, floor, iterations or stopping criteria.
 original_pava=profile_module._pava_nondecreasing
 rng=np.random.default_rng(20261002)
 for size in [5,259,1419,2151,3248]:
  for values in [rng.normal(size=size),np.linspace(-3,3,size)+rng.normal(size=size),np.zeros(size)]:
   assert np.allclose(original_pava(values),isotonic_regression(values).x,rtol=1e-13,atol=1e-13)
 profile_module._pava_nondecreasing=lambda values:isotonic_regression(values).x
 exp=ROOT/f'network_variants/output/{ex}_indep__anchored';hc=exp/'states/S000_base/hydraulic_context'
 man=json.loads((hc/'hydraulic_context_manifest.json').read_text());source=Path(man['slope_refit_promotion']['reference_profile_dir'])
 sel=json.loads((source/'reference_profile_selection.json').read_text());old=pd.read_parquet(source/'reference_profile.parquet');obs=pd.read_parquet(source/'reference_profile_observations.parquet')
 nodes=pd.read_csv(hc/'node_hydraulic_context.csv').set_index('id_node');endpoints=sorted(nodes.loc[[r['id_node'] for r in sel['inputs']['endpoint_projection']],'reference_section_dist_out_m'].tolist())
 # Node CSV roundtrips can differ by <1e-9m; preserve the exact saved grid endpoints.
 grid=old.reference_section_dist_out_m.to_numpy(float)
 snapped=[float(grid[np.argmin(abs(grid-v))]) for v in endpoints]
 assert max(abs(a-b) for a,b in zip(endpoints,snapped))<1e-6
 endpoints=snapped
 adjustments=sel.get('manual_adjustments',[])
 if adjustments:
  original_fit=profile_module.fit_reference_profile
  def bridged_fit(*args,**kwargs):
   fit=original_fit(*args,**kwargs)
   for adjustment in adjustments:
    fit,_=apply_manual_linear_profile_bridge(fit,adjustment_id=adjustment['adjustment_id'],start_distance_m=adjustment['requested_start_distance_m'],end_distance_m=adjustment['requested_end_distance_m'])
   return fit
  profile_module.fit_reference_profile=bridged_fit
 cfg=ReferenceProfileConfig(**sel['config']);lam=sel['selection']['selected_lambda']
 widths=pyogrio.read_dataframe(man['input_paths']['directed_links'],columns=['id_link','wid_adj_routing'],read_geometry=False)
 oldlinks=pd.read_csv(hc/'link_hydraulic_context.csv');ref=json.loads((exp/'reference_section/reference_kb_summary.json').read_text())
 prep=exp/'rapid_prep_manifest.json';prep=json.loads(prep.read_text())['prep_config'] if prep.exists() else {}
 caplo,caphi=prep.get('min_celerity_mps',.05),prep.get('max_celerity_mps',2.524)
 out=OUT/ex;out.mkdir(parents=True,exist_ok=True);results=[]
 for eta in FLOORS:
  directory=out/f'eta_{eta:.2f}';pp=directory/'profile';cp=directory/'context';record=directory/'result.json'
  if record.exists():results.append(json.loads(record.read_text()));continue
  config=replace(cfg,minimum_slope_fraction=eta)
  if not (pp/'reference_profile_audit.json').exists():
   fit,cv,selection=select_reference_profile(old[['reference_section_dist_out_m','static_wse_m']],obs,endpoint_distances_m=endpoints,lambda_values=[lam],config=config)
   selection.update(selection_rule='fixed_reviewed_lambda_with_acquisition_CV',floor_sensitivity_only=True)
   records=[]
   for adjustment in adjustments:
    record=dict(adjustment);x=fit.profile.reference_section_dist_out_m.to_numpy();y=fit.profile.fitted_wse_m.to_numpy()
    record['start_head_m']=float(np.interp(record['resolved_start_distance_m'],x,y));record['end_head_m']=float(np.interp(record['resolved_end_distance_m'],x,y))
    record['bridge_slope_m_per_m']=(record['end_head_m']-record['start_head_m'])/(record['resolved_end_distance_m']-record['resolved_start_distance_m'])
    records.append(record)
   write_reference_profile_artifacts(fit,cv,selection,pp,config=config,input_metadata=dict(source_profile=str(source),endpoint_distances_m=endpoints),manual_adjustments=records)
  profile=pd.read_parquet(pp/'reference_profile.parquet');summary=json.loads((pp/'reference_profile_audit.json').read_text())['summary'];cv=pd.read_csv(pp/'reference_profile_cv.csv')
  r=dict(example=ex,eta=eta,lambda_value=lam,cv_mse=float(cv.cv_mean_mse_m2.iloc[0]),cv_se=float(cv.cv_se_mse_m2.iloc[0]),rms_m=summary['weighted_rms_m'],head_loss_m=summary['fitted_head_loss_m'],outlet_head_m=summary['fitted_endpoint_head_outlet_m'],inlet_head_m=summary['fitted_endpoint_head_inlet_m'],profile_max_change_m=float(abs(profile.fitted_wse_m-old.fitted_wse_m).max()),floor_active_length_m=float(np.diff(profile.reference_section_dist_out_m)[profile.floor_active.to_numpy()[1:]].sum()),outer_iterations=summary['outer_iterations'],inner_iterations=summary['inner_iterations'])
  heads=np.interp(endpoints,profile.reference_section_dist_out_m,profile.fitted_wse_m)
  r.update(outlet_head_m=float(heads[0]),inlet_head_m=float(heads[1]),head_loss_m=float(heads[1]-heads[0]))
  if eta==.01:assert r['profile_max_change_m']<1e-6,(ex,r['profile_max_change_m'])
  try:
   if not (cp/'candidate_context_manifest.json').exists():generate_reference_profile_candidate_context(exp,pp,cp)
   links=pd.read_parquet(cp/'candidate_link_hydraulic_context.parquet')
   check=links[['id_link','candidate_slope_m_per_m']].merge(oldlinks[['id_link','slope_used']],on='id_link',validate='one_to_one')
   r['max_link_slope_change']=float(abs(check.candidate_slope_m_per_m-check.slope_used).max())
   # Use actual current width product, not potentially stale directed-file widths.
   links=links.drop(columns=['wid_adj_routing'],errors='ignore').merge(widths,on='id_link',validate='one_to_one')
   length=links.actual_length_m.to_numpy(float);slope=links.candidate_slope_m_per_m.to_numpy(float);width=links.wid_adj_routing.to_numpy(float)
   assert np.isfinite(slope).all() and (slope>0).all()
   depth=ref['reference_depth_m']*(width/ref['reference_width_m'])**.65
   raw=5/(3*.035)*depth**(2/3)*np.sqrt(slope)
   count=np.maximum(1,np.rint(length/200));effective=np.maximum(200,length/count)
   for label,c in [('uncapped',raw),('capped',np.clip(raw,caplo,caphi))]:
    k=effective/c;il=int(k.argmax());iu=int(k.argmin());low=float(.2*k[il]);high=float(1.8*k[iu])
    r.update({label+'_lower_s':low,label+'_upper_s':high,label+'_feasible':math.ceil(low-1e-9)<=math.floor(high+1e-9),label+'_slow_link':int(links.id_link.iloc[il]),label+'_fast_link':int(links.id_link.iloc[iu])})
   target=links.set_index('id_link').loc[CASES[ex]];r['target_slope']=float(target.candidate_slope_m_per_m);r['minimum_slope']=float(slope.min());r['context_status']='ok'
  except (ValueError,RuntimeError) as err:r['context_status']=str(err)
  directory.mkdir(parents=True,exist_ok=True);record.write_text(json.dumps(r,indent=2));results.append(r)
  print(ex,eta,'complete',r['context_status'],'capped stable',r.get('capped_feasible'),flush=True)
 pd.DataFrame(results).to_csv(out/'summary.csv',index=False)
 return results

def main():
 OUT.mkdir(parents=True,exist_ok=True)
 with ProcessPoolExecutor(max_workers=2) as pool:results=list(pool.map(run_case,CASES))
 d=pd.DataFrame([r for rows in results for r in rows]);d.to_csv(OUT/'summary.csv',index=False)
 (OUT/'provenance.json').write_text(json.dumps(dict(reference='PROFILE-FLOOR-2026-10-02',scope='Five profile-derived parent controllers; not all states or a production rerun',floors=FLOORS,cases=CASES,profile_method='Identical saved static grid and QC observations; reviewed per-case lambda fixed; same 3-fold acquisition CV recalculated for each eta',manual_bridge='Existing case46 370–410km bridge retained in full fits and CV fits; no new bridges introduced.',domain_floor=.01,stability='alpha=.65,n=.035, stored BASED depth anchor fixed, 200m target/floor, X=.1; with/without saved caps (.05/2.524 defaults for13); no adaptive length safeguards',baseline='eta=.01 reproduces saved profile within 1e-6m',solver_note='Equivalent scipy compiled PAVA verified against original at 1e-13; existing finite iteration settings retained. This tests the current fitting procedure; iteration counts exported, not a new convergence proof.',script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),indent=2))
 print('COMPLETE',len(d),'profile scenarios',flush=True)
if __name__=='__main__':main()
