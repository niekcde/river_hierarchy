import sys,json
from pathlib import Path
import numpy as np,pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parent))
import cunge_reference_flow_sensitivity as s
results=[]
for ex,state in [('41','island_41_005'),('46','S075_unit_16')]:
 meta=json.loads((s.OUT/'provenance.json').read_text())
 alias=pd.read_csv(s.OUT/'state_aliases.csv',dtype={'example':str})
 key=alias[(alias.example==ex)&(alias.state==state)].iloc[0]['key']; item=next(x for x in meta['sources'] if x['key']==key)
 r=pd.read_csv(s.OUT/'sensitivity_results.csv',dtype={'example':str}); r=r[(r.example==ex)&(r.state==state)]; dt=float(r.iloc[0].dt_s); n=float(r.iloc[0].n_manning)
 m=json.loads((Path(item['source_calibration'])/'case.json').read_text());step=m['forcing_step_s'];nsub=round(step/dt)
 parent=s.cunge.freeze_grid(pd.read_parquet(item['parent_frame']),m['reference_discharge_cms']);child=s.cunge.freeze_grid(pd.read_parquet(item['child_frame']),m['reference_discharge_cms']);_,shared,_=s.conserved(parent,child)
 x=child.copy();x.loc[shared._ci,'reference_discharge_cms']=shared.reference_discharge_cms_p.to_numpy()
 hold,changes=s.held_splits(parent,child,shared);variants={'actual':child,'X_only':x}
 if changes:variants['parent_bifurcation_splits']=hold
 count=np.maximum.reduce([np.maximum(1,np.ceil(g.link_length_m/(2*g.reference_discharge_cms/(g.width*g.slope*np.clip(g.celerity_times_n/.01,.05,5))))).astype(int).to_numpy() for g in variants.values()])
 frames={}
 for name,g in variants.items():
  g=g.copy();g['count']=count;g['rapid_effective_length_m']=g.link_length_m/count;frames[name]=s.cunge.parameters(g,n)
 wave=pd.read_parquet(s.OUT/ex/key/'waveforms.parquet');flow=wave.inflow_cms.to_numpy()
 details=[]
 for row in shared.itertuples(index=False):
  # Key merge preserves physical parameters, so identify by exact keys.
  sel=np.ones(len(child),dtype=bool)
  for k in s.KEYS:sel &= child[k].eq(getattr(row,k)).to_numpy()
  i=np.flatnonzero(sel)[0]
  if abs(row.reference_discharge_cms_c-row.reference_discharge_cms_p)>1e-7:
   details.append(dict(u=int(row.u),v=int(row.v),Q_parent=row.reference_discharge_cms_p,Q_child=row.reference_discharge_cms_c,width_m=float(child.iloc[i].width),X_parent_Q=float(frames['X_only'].iloc[i].rapid_x),X_child_Q=float(frames['actual'].iloc[i].rapid_x),K_s=float(frames['actual'].iloc[i].rapid_k)))
 (s.OUT/ex/key/'affected_coefficient_details.json').write_text(json.dumps(details,indent=2))
 for factor in [1,2]:
  runs={}
  for name,f in frames.items():
   native,audit=s.cunge.route(f,flow,nsub*factor,dt/factor,return_native=True)
   averaged=native.reshape(-1,nsub*factor).mean(axis=1)
   runs[name]=(native.argmax()*dt/factor,averaged.argmax()*step,averaged,audit)
  for name,(tp,tavg,q,audit) in runs.items():
   actual=runs['actual']; results.append(dict(example=ex,state=state,variant=name,dt_s=dt/factor,native_peak_difference_s=tp-actual[0],saved_peak_difference_s=tavg-actual[1],maximum_wave_difference_pct=100*np.max(abs(q-actual[2]))/np.ptp(flow),minimum_native_flow_cms=audit['minimum_native_flow_cms']))
  print(ex,factor,results[-len(runs):],flush=True)
pd.DataFrame(results).to_csv(s.OUT/'focused_clock_check.csv',index=False)
