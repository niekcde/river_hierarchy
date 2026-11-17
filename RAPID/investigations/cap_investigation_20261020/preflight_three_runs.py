"""Read-only three-run preflight. Computes diagnostics, never calibrates/routes."""
from pathlib import Path
import json,math,hashlib,shutil
import numpy as np
import pandas as pd
from manning_calibrated_network import ROOT,OUT
from rapid_tools import cunge
from rapid_tools.reference_section import predict_based_depth
from staged_routing_merge import assemble

SETUP=OUT/'three_run_setup_20261028'
DEST=SETUP/'preflight_20261028'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
 DEST.mkdir(exist_ok=True);sources={};issues=[];rows=[];parents=[];restored=[]
 def read(p):
  p=Path(p);sources[str(p)]=sha(p)
  return pd.read_parquet(p) if p.suffix=='.parquet' else json.loads(p.read_text())
 design=read(SETUP/'run_setup_v2.json');frozen=read(SETUP/'manifest.json');amend=read(SETUP/'input_policy_amendment.json')
 assert sha(SETUP/'run_setup_v2.json')==amend['active_setup_sha256']
 for p,h in frozen['output_hashes'].items():assert sha(p)==h,p
 for p,h in frozen['source_hashes'].items():
  if sha(p)!=h:issues.append(dict(case='GLOBAL',stage='frozen_source',error='Changed since freeze: '+p))
 vals=pd.read_csv(SETUP/'frozen_reference_discharges.csv',dtype={'example':str}).set_index('example')
 plan=read(OUT/'manning_all_networks_preparation_20261027/run_specification.json')
 index=read(design['shared']['staged_slope_index']);stage={x['case']:x for x in index['cases']}
 source_model=ROOT/'RAPID/assets/based_model_v2.ubj';sources[str(source_model)]=sha(source_model)
 for case in plan['cases']:
  ex=case['example'];v=vals.loc[ex];pj=next(j for j in case['jobs'] if j['state']=='S000_base');base=read(pj['source_parameters']);bi=base.set_index('id_link')
  force=read(case['forcing']);event=force[force.phase.eq('input_event')];spin=force[force.phase.eq('spinup')]
  assert np.allclose(spin.forcing_cms,v.spinup_cms) and np.isclose(event.forcing_cms.max(),v.inflow_event_peak_cms)
  assert np.isclose(v.midpoint_Qref_cms,(v.spinup_cms+v.inflow_event_peak_cms)/2)
  suffix='_direction130' if ex=='15' else ''
  kb=read(ROOT/f'network_variants/output/{ex}_indep{suffix}__anchored/reference_section/reference_kb_summary.json')
  old_anchor=kb['reference_depth_m'];widthref=kb['reference_width_m']
  predicted_old=predict_based_depth(discharge_cms=kb['reference_discharge_cms'],width_m=widthref,slope=kb['reference_slope'])
  new_anchor=predict_based_depth(discharge_cms=v.midpoint_Qref_cms,width_m=widthref,slope=kb['reference_slope'])
  expected=old_anchor*(base.width/widthref)**.65
  parent_anchor_matches=bool(np.allclose(expected,base.depth_m,rtol=1e-9,atol=1e-10))
  reproduced=cunge.parameters(base,float(base.n_manning.iloc[0]),beta=float(base.diffusion_multiplier.iloc[0]))
  reproduction=bool(np.allclose(reproduced[['rapid_k','rapid_x','rapid_celerity_mps']],base[['rapid_k','rapid_x','rapid_celerity_mps']],rtol=1e-10,atol=1e-10))
  unchanged_hash=sha(pj['source_parameters'])==pj['source_sha256']
  parents.append(dict(example=ex,parent_source_unchanged=unchanged_hash,parent_not_indexed=ex+'/S000_base' not in stage,
   old_BASED_anchor_reproduced=bool(np.isclose(predicted_old,old_anchor,rtol=1e-8)),parent_depth_rule_matches=parent_anchor_matches,
   existing_K_X_c_reproduced=reproduction,old_depth_anchor=old_anchor,midpoint_depth_anchor=new_anchor,
   n=float(base.n_manning.iloc[0]),beta=float(base.diffusion_multiplier.iloc[0])))
  for j in case['jobs']:
   key=ex+'/'+j['state'];phase='source'
   try:
    src=read(j['source_parameters']);assert sha(j['source_parameters'])==j['source_sha256'],'Prepared source parameter fingerprint changed'
    f=src.copy();added=0;lengthchanges=0;nodechanges=0
    if key in stage:
     phase='indexed_merge';item=stage[key];s=read(item['slope_file']);assert sha(item['slope_file'])==item['slope_sha256']
     contract=read(item['contract_file']);audit=read(item['audit_file']);assert s.id_link.is_unique and src.id_link.is_unique
     assert s.slope.gt(0).all() and np.isfinite(s[['slope','head_us','head_ds','link_length_m']]).all().all()
     np.testing.assert_allclose(s.slope,(s.head_us-s.head_ds)/s.link_length_m,rtol=1e-8,atol=1e-12)
     heads=pd.concat([s[['u','head_us']].rename(columns={'u':'node','head_us':'head'}),s[['v','head_ds']].rename(columns={'v':'node','head_ds':'head'})])
     assert (heads.groupby('node')['head'].agg(lambda a: a.max()-a.min())<1e-8).all(),'Shared heads inconsistent'
     assert audit['independent_checks']['shared_heads'] and audit['independent_checks']['unreleased_exterior_heads_fixed']
     assert audit['independent_checks']['max_floor_deficit_m']<1e-8
     if item['status']=='verified_fixed':assert audit['independent_checks']['changed_boundary_heads']==0
     if 'accepted_geometry_candidate' in item:
      geo=read(item['accepted_geometry_candidate']);assert sha(item['accepted_geometry_candidate'])==item['accepted_geometry_sha256']
      a=s.set_index('id_link');g=geo.set_index('id_link');assert set(a.index)==set(g.index)
      np.testing.assert_allclose(a.loc[g.index,['u','v','link_length_m']],g[['u','v','link_length_m']],rtol=1e-10,atol=1e-7)
     else:assert set(s.id_link)==set(src.id_link),'Unindexed geometry change'
     actions=read(Path(item['accepted_geometry_candidate']).parent/'geometry_actions.json') if 'accepted_geometry_candidate' in item else []
     f,lineage,lengthchanges,nodechanges=assemble(src,base,s,actions)
     added=len(lineage)
     restored.extend(dict(case=key,**r) for r in lineage)
    phase='topology_allocation'
    order,edges,root,end=cunge.topology(f)
    shares={root:1.};alloc=np.zeros(len(f));split=np.zeros(len(f));width=f.width.to_numpy();vv=f.v.to_numpy()
    for u in order:
     ids=edges.get(u,[]);total=width[ids].sum()
     for i in ids:
      split[i]=width[i]/total;alloc[i]=shares[u]*split[i];shares[vv[i]]=shares.get(vv[i],0.)+alloc[i]
    assert np.isclose(shares[end],1,rtol=0,atol=1e-10) and (alloc>0).all()
    assert (f['count']>=1).all() and np.allclose(f['count'],f['count'].astype(int))
    phase='clock_envelope';runbounds={}
    for run in ['based_legacy','based_midpoint','manning_midpoint']:
     q=alloc*(v.legacy_routing_Qref_cms if run=='based_legacy' else v.midpoint_Qref_cms)
     if run=='based_legacy':depth=f.depth_m.to_numpy();ns=[float(base.n_manning.iloc[0])];betas=[float(base.diffusion_multiplier.iloc[0])]
     else:depth=new_anchor*(f.width.to_numpy()/widthref)**.65;ns=[.01,.1];betas=[.5,2.]
     upper=np.inf;minc=np.inf;maxc=0.
     for n in ns:
      h=(n*q/(width*np.sqrt(f.slope)))**.6 if run=='manning_midpoint' else depth
      c=5/(3*n)*h**(2/3)*np.sqrt(f.slope)
      if run!='manning_midpoint':c=np.clip(c,f.celerity_min_mps,f.celerity_max_mps)
      assert np.isfinite(c).all() and (c>0).all()
      dx=f.rapid_effective_length_m.to_numpy();k=dx/c
      for beta in betas:
       x=.5-beta*q/(2*width*f.slope*c*dx)
       ub=float((2*k*(1-x)).min());assert ub>0 and np.isfinite(ub);upper=min(upper,ub)
      minc=min(minc,float(c.min()));maxc=max(maxc,float(c.max()))
     dt=30/math.ceil(30/min(float(j['native_dt_seconds']),.99*upper))
     runbounds[run+'_dt_seconds']=dt;runbounds[run+'_needs_smaller_dt']=dt<float(j['native_dt_seconds'])-1e-10
    rows.append(dict(example=ex,state=j['state'],indexed=key in stage,status='passed',links=len(f),new_parent_restorations=added,
     changed_length_rows=lengthchanges,changed_endpoint_rows=nodechanges,old_dt_seconds=j['native_dt_seconds'],**runbounds))
   except Exception as exc:
    issues.append(dict(case=key,stage=phase,error=str(exc)));rows.append(dict(example=ex,state=j['state'],indexed=key in stage,status='blocked'))
  print('Checked',ex,'states',len(case['jobs']),'issues',len(issues),flush=True)
 pd.DataFrame(rows).to_csv(DEST/'state_checks.csv',index=False);pd.DataFrame(parents).to_csv(DEST/'parent_checks.csv',index=False)
 pd.DataFrame(restored).to_csv(DEST/'restored_link_attribute_lineage.csv',index=False)
 free=shutil.disk_usage('/Volumes/PhD').free/2**30
 summary=dict(states=len(rows),passed=sum(r['status']=='passed' for r in rows),indexed_cases=len(stage),issues=issues,
  parent_checks_all_pass=bool(pd.DataFrame(parents)[['parent_source_unchanged','parent_not_indexed','old_BASED_anchor_reproduced','parent_depth_rule_matches','existing_K_X_c_reproduced']].all().all()),
  drive_free_GiB=free,working_budget_GiB=20,minimum_reserve_GiB=20,space_pass=free>=40,
  clock_check='Analytical previous-outflow bound, using inherited cell counts and lengths. Midpoint methods checked over n/beta bound corners; adjust clocks as listed. This is not a hydrograph or spatial convergence check.',
  retained_policy='Indexed inputs only; nonindexed slopes and geometry retained. Existing boundary-flex/length exceptions retained.',
  routing_runs=0,calibration_runs=0,production_modified=False)
 for p,h in sources.items():
  if sha(p)!=h:summary['issues'].append(dict(case='GLOBAL',stage='source_mutation',error=p))
 summary['preflight_pass']=summary['passed']==2215 and not summary['issues'] and summary['parent_checks_all_pass'] and summary['space_pass']
 (DEST/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
 sources[str(Path(__file__).resolve())]=sha(__file__)
 sources[str(Path(__file__).with_name('staged_routing_merge.py').resolve())]=sha(Path(__file__).with_name('staged_routing_merge.py'))
 (DEST/'manifest.json').write_text(json.dumps(dict(source_hashes=sources,output_hashes={str(p):sha(p) for p in DEST.iterdir() if p.is_file() and p.name!='manifest.json'}),indent=2)+'\n')
 print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
