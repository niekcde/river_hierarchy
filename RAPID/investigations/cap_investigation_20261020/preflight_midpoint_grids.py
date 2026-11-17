"""Shared midpoint-grid sizing and clock envelope; does not route or calibrate."""
import json,math
import numpy as np
import pandas as pd
from preflight_three_runs import ROOT,OUT,SETUP,DEST,sha
from rapid_tools import cunge
from rapid_tools.reference_section import predict_based_depth
from staged_routing_merge import assemble

def main():
 plan=json.loads((OUT/'manning_all_networks_preparation_20261027/run_specification.json').read_text())
 index=json.loads((ROOT/'RAPID/outputs/compound_member_slopes_final_20261028/staged_slope_index.json').read_text());stages={x['case']:x for x in index['cases']}
 values=pd.read_csv(SETUP/'frozen_reference_discharges.csv',dtype={'example':str}).set_index('example');rows=[];issues=[]
 for case in plan['cases']:
  ex=case['example'];qmid=values.loc[ex,'midpoint_Qref_cms'];base=pd.read_parquet(next(j['source_parameters'] for j in case['jobs'] if j['state']=='S000_base')).set_index('id_link')
  suffix='_direction130' if ex=='15' else '';kb=json.loads((ROOT/f'network_variants/output/{ex}_indep{suffix}__anchored/reference_section/reference_kb_summary.json').read_text())
  anchor=predict_based_depth(discharge_cms=qmid,width_m=kb['reference_width_m'],slope=kb['reference_slope'])
  for job in case['jobs']:
   key=ex+'/'+job['state']
   try:
    src=pd.read_parquet(job['source_parameters']);f=src.copy()
    if key in stages:
     item=stages[key];slopes=pd.read_parquet(item['slope_file'])
     actions=json.loads((__import__('pathlib').Path(item['accepted_geometry_candidate']).parent/'geometry_actions.json').read_text()) if 'accepted_geometry_candidate' in item else []
     f,_,_,_=assemble(src,base.reset_index(),slopes,actions)
    order,edges,root,end=cunge.topology(f);flow={root:1.};alloc=np.zeros(len(f));w=f.width.to_numpy();v=f.v.to_numpy();S=f.slope.to_numpy();L=f.link_length_m.to_numpy()
    for u in order:
     ids=edges.get(u,[]);total=w[ids].sum()
     for i in ids:alloc[i]=flow[u]*w[i]/total;flow[v[i]]=flow.get(v[i],0)+alloc[i]
    assert np.isclose(flow[end],1)
    q=qmid*alloc;hbased=anchor*(w/kb['reference_width_m'])**.65
    cb=np.clip(5/(3*.01)*hbased**(2/3)*np.sqrt(S),f.celerity_min_mps,f.celerity_max_mps)
    hm=(.01*q/(w*np.sqrt(S)))**.6;cm=5/(3*.01)*hm**(2/3)*np.sqrt(S)
    old=f['count'].to_numpy().astype(int);alpha=float(f.cunge_alpha.iloc[0]);required=np.maximum(np.ceil(L/(alpha*q/(w*S*cb))),np.ceil(L/(alpha*q/(w*S*cm)))).astype(int)
    counts=np.maximum(old,required);dx=L/counts
    uppers=[]
    for c in [cb,cm]:
     # Smallest bound is at minimum n and beta for both depth formulations.
     k=dx/c;x=.5-.5*q/(2*w*S*c*dx);uppers.append(float((2*k*(1-x)).min()))
    upper=min(uppers);dt=30/math.ceil(30/min(job['native_dt_seconds'],.99*upper))
    assert np.isfinite(upper) and upper>0 and dt>0
    rows.append(dict(example=ex,state=job['state'],old_cells=int(old.sum()),shared_midpoint_cells=int(counts.sum()),
       refined_links=int((counts>old).sum()),shared_dt_seconds=dt,old_dt_seconds=job['native_dt_seconds'],
       timestep_refined=bool(dt<job['native_dt_seconds']-1e-10)))
   except Exception as e:issues.append(dict(case=key,error=str(e)))
  print('Grid envelope',ex,flush=True)
 d=pd.DataFrame(rows);d.to_csv(DEST/'shared_midpoint_grid_checks.csv',index=False)
 summary=dict(cases=len(d),issues=issues,pass_check=len(d)==2215 and not issues,
  cases_requiring_finer_grid=int(d.refined_links.gt(0).sum()),cases_requiring_smaller_clock=int(d.timestep_refined.sum()),
  original_cells=int(d.old_cells.sum()),shared_midpoint_cells=int(d.shared_midpoint_cells.sum()),
  minimum_dt_seconds=float(d.shared_dt_seconds.min()),
  prescription='For runs02/03 use a shared per-state grid: maximum of inherited counts and both formulations Cunge alpha length-scale counts at n=0.01. Use common clock safe at n=0.01,beta=0.5. Final calibration results still require waveform and spatial-convergence checks.',
  code_sha256=sha(__file__),routing_runs=0)
 (DEST/'midpoint_grid_summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
