"""Fixed-calibration flagged-depth substitutions and hydraulic-policy cross-checks."""
import os
for key in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS']:os.environ[key]='1'
from pathlib import Path
import sys,json
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[3]
sys.path[:0]=[str(ROOT/'RAPID/src'),str(ROOT/'RAPID/investigations/cap_investigation_20261020')]
import run_three_runs_control_fix as r
from rapid_tools.event_window_batch import settling_hours
SOURCE=Path('/Volumes/PhD/river_hierarchy/three_run_comparison_20261028')
DEST=ROOT/'RAPID/outputs/uncapped_event_window_20261020/three_run_review_20261029/targeted_depth_sensitivity'
SELECTION={'41':['S000_base','S130_unit_126'],'37':['S000_base','S256_unit_229','S290_unit_112']}

def update_depth(source,h):
 f=source.copy();f['depth_m']=h
 f['celerity_times_n']=5/3*f.depth_m**(2/3)*np.sqrt(f.slope)
 c=f.celerity_times_n/f.n_manning
 f['rapid_celerity_mps_raw']=c;f['rapid_celerity_mps']=c;f['rapid_celerity_capped']=False
 f['rapid_k']=f.rapid_effective_length_m/c
 f['hydraulic_length_scale_m']=f.reference_discharge_cms/(f.width*f.slope*c)
 f['rapid_x']=.5-f.diffusion_multiplier*f.hydraulic_length_scale_m/(2*f.rapid_effective_length_m)
 f['depth_policy']='diagnostic_BASED_substitution_no_recalibration'
 return f

def main():
 DEST.mkdir(parents=True,exist_ok=True);r.init_worker(DEST,'targeted')
 allmetrics=[];audits=[];parameters=[];sources={}
 for ex,states in SELECTION.items():
  case=r.CTX['cases'][ex]
  for state in states:
   originals={};originalmetrics={};receipts={}
   for name,run in [('based',r.RUNS[1]),('manning',r.RUNS[2])]:
    folder=SOURCE/run/'states'/ex/state;receipt=json.loads((folder/'completion.json').read_text());receipts[name]=receipt
    for filename in ['parameters.parquet','metrics.parquet','outlet.parquet']:
     p=folder/filename;assert r.digest(p)==receipt['files'][str(p)];sources[str(p)]=r.digest(p)
    originals[name]=pd.read_parquet(folder/'parameters.parquet')
    originalmetrics[name]=pd.read_parquet(folder/'metrics.parquet')
    allmetrics.append(originalmetrics[name].assign(model=name,clock=1))
   b=originals['based'];m=originals['manning']
   for col in ['id_link','u','v','link_length_m','width','slope','count','split_fraction','reference_discharge_cms']:
    np.testing.assert_array_equal(b[col],m[col])
   np.testing.assert_allclose(update_depth(m,m.depth_m)[['rapid_k','rapid_x','rapid_celerity_mps']],m[['rapid_k','rapid_x','rapid_celerity_mps']],rtol=1e-12)
   nB=float(b.n_manning.iloc[0]);betaB=float(b.diffusion_multiplier.iloc[0]);nM=float(m.n_manning.iloc[0]);betaM=float(m.diffusion_multiplier.iloc[0])
   modes={'manning':m,'depth_gt10_to_based':update_depth(m,np.where(m.depth_m>10,b.depth_m,m.depth_m)),
          'depth_gt5_to_based':update_depth(m,np.where(m.depth_m>5,b.depth_m,m.depth_m)),
          'all_based_depth_manning_fit':update_depth(m,b.depth_m),
          'manning_depth_based_fit':r.hydraulic_parameters(m,r.RUNS[2],nB,betaB)}
   # Cross-model attribution requires celerity cap policy not to be active.
   assert not b.rapid_celerity_capped.any()
   folder=DEST/ex/state;folder.mkdir(parents=True,exist_ok=True)
   force=pd.read_parquet(case['forcing'])
   originalout=pd.read_parquet(SOURCE/r.RUNS[2]/'states'/ex/state/'outlet.parquet')
   force=originalout.merge(force[['time_utc','forcing_cms']],on='time_utc',how='left',validate='one_to_one')
   missing=force.forcing_cms.isna();assert force.loc[missing,'phase'].eq('settling').all();force.loc[missing,'forcing_cms']=case['final_cms']
   nb=int(force.phase.eq('spinup').sum());initial=float(force.forcing_cms.iloc[0]);hold=force.loc[force.phase.eq('settling'),'interval_start_utc'].iloc[0]
   dt=receipts['manning']['native_dt_seconds'];assert dt==receipts['based']['native_dt_seconds']
   job=next(j for j in case['jobs'] if j['state']==state)
   cache={}
   for model,f in modes.items():
    for col in ['id_link','width','slope','count','split_fraction','reference_discharge_cms']:
     np.testing.assert_array_equal(f[col],m[col])
    changed=~np.isclose(f.depth_m,m.depth_m,rtol=1e-12,atol=1e-12)
    alpha_violations=int((f.rapid_effective_length_m>f.cunge_alpha*f.hydraulic_length_scale_m*(1+1e-12)).sum())
    f.to_parquet(folder/(model+'_parameters.parquet'),index=False)
    parameters.append(dict(example=ex,state=state,model=model,n=float(f.n_manning.iloc[0]),beta=float(f.diffusion_multiplier.iloc[0]),
     changed_depth_reaches=int(changed.sum()),maximum_depth_m=float(f.depth_m.max()),alpha_bound_violations=alpha_violations,
     transport_mean_minutes=float((f['count']*f.rapid_k*f.reference_flow_fraction).sum()/60)))
    for clock in [1,2]:
     if model=='manning' and clock==1:continue
     key=(r.stamp(f[['rapid_k','rapid_x']].to_numpy().tolist()),clock)
     if key in cache:
      allmetrics.append(cache[key].assign(model=model));continue
     if not changed.any() and model.startswith('depth_') and clock==1:
      allmetrics.append(originalmetrics['manning'].assign(model=model,clock=clock));continue
     print('Routing',ex,state,model,clock,flush=True)
     q,audit=r.solve_wave(f,force.forcing_cms.to_numpy()[nb:],dt/clock,initial)
     _,settled=settling_hours(force.time_utc.iloc[nb:],q,hold,case['final_cms'],case['tolerance_cms'])
     if not settled:raise RuntimeError('Extend diagnostic tail before interpreting results')
     q=np.r_[np.full(nb,initial),q]
     metrics=r.metrics_for(ex,state,job['aliases'],force,q).assign(model=model,clock=clock)
     allmetrics.append(metrics);cache[key]=metrics
     pd.DataFrame({'time_utc':force.time_utc,'q_outlet_cms':q}).to_parquet(folder/f'{model}_clock{clock}_outlet.parquet',index=False)
     audits.append(dict(example=ex,state=state,model=model,clock=clock,dt_seconds=dt/clock,settled=settled,**audit))
   pd.concat(allmetrics).to_csv(DEST/'metrics.csv',index=False)
   pd.DataFrame(parameters).to_csv(DEST/'parameter_summary.csv',index=False)
   pd.DataFrame(audits).to_csv(DEST/'routing_audit.csv',index=False)
 metrics=pd.concat(allmetrics);responses=[]
 for (ex,model,clock,window),g in metrics.groupby(['example','model','clock','window']):
  parent=g[g.state=='S000_base'].iloc[0]
  for child in g[g.state!='S000_base'].itertuples():
   responses.append(dict(example=ex,state=child.state,model=model,clock=clock,window=window,
    centroid_shift_min=(child.centroid_seconds-parent.centroid_seconds)/60,
    peak_shift_min=(child.peak_seconds-parent.peak_seconds)/60,
    peak_excess_change_pct=100*(child.peak_excess_cms/parent.peak_excess_cms-1)))
 resp=pd.DataFrame(responses);resp.to_csv(DEST/'paired_responses.csv',index=False)
 r.write_json(DEST/'completion.json',dict(selection=SELECTION,source_hashes=sources,script_sha256=r.digest(__file__),
  qualifications=['Diagnostic depth substitutions relax the original Manning Q-depth relation; not corrected physical depths.',
  'Flags selected from original fixed-Manning depths separately in parent and child; same threshold applied to each.',
  'Cross-fitted Manning depths are recomputed with alternate n, as the hydraulic formula requires.',
  'No calibration, production mutations, spatial refinement or variable-depth runs.',
  'Alpha accuracy flags reported explicitly; routing timestep and discharge validity enforced.']))
 print(resp.query("window=='0.5%' and clock==1").to_string(index=False),flush=True)

if __name__=='__main__':main()
