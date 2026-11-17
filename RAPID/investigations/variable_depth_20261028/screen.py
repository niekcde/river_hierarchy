"""All-state hydraulic envelope on unchanged midpoint grids; no routing/calibration."""
import json,math
import numpy as np
import pandas as pd
from prototype import r,DEST,parameters,HERE

def main():
 DEST.mkdir(exist_ok=True);r.init_worker(DEST,'variable-depth-preflight');rows=[];details=[]
 for ex,case in r.CTX['cases'].items():
  qvalues=np.r_[pd.read_parquet(case['forcing']).forcing_cms,pd.read_parquet(case['calibration_forcing']).forcing_cms]
  qmin=float(np.min(qvalues));qmax=float(np.max(qvalues))
  if qmin<=0:raise ValueError('Dry-flow closure not implemented: '+ex)
  for job in case['jobs']:
   f,dt,_=r.prepare_frame(ex,job['state'],r.RUNS[2]);w=f.width.to_numpy();s=f.slope.to_numpy();dx=f.rapid_effective_length_m.to_numpy()
   low=qmin*f.reference_flow_fraction.to_numpy();high=qmax*f.reference_flow_fraction.to_numpy()
   # U=2K(1-X)=dx/c + beta*Q/(B*S*c^2). Minimum n,beta
   # minimize U; over Q include the exact interior stationary point.
   coef=5/(3*.01)*(.01/(w*np.sqrt(s)))**.4*np.sqrt(s)
   critical=np.clip((2*dx*w*s*coef/.5)**(1/.6),low,high)
   upper=dx/(coef*critical**.4)+.5*critical/(w*s*(coef*critical**.4)**2)
   safe=60/math.ceil(60/min(dt,.99*float(upper.min())))
   hmin=(.01*low/(w*np.sqrt(s)))**.6;hmax=(.1*high/(w*np.sqrt(s)))**.6
   alpha=dx/(low/(w*s*coef*low**.4))
   rows.append(dict(example=ex,state=job['state'],cells=int(f['count'].sum()),current_dt_seconds=dt,
      envelope_dt_seconds=safe,requires_smaller_dt=bool(safe<dt-1e-10),depth_min_m=float(hmin.min()),depth_max_m=float(hmax.max()),
      depth_width_ratio_max=float((hmax/w).max()),max_alpha_low_flow=float(alpha.max()),
      links_exceeding_current_alpha=int((alpha>f.cunge_alpha.to_numpy()*(1+1e-10)).sum()),network_qmin_cms=qmin,network_qmax_cms=qmax))
   if job['state']=='S000_base':
    details.append(pd.DataFrame(dict(example=ex,id_link=f.id_link,width_m=w,slope=s,depth_min_m=hmin,depth_max_m=hmax,depth_width_ratio_max=hmax/w)))
  print('Screened',ex,flush=True)
 d=pd.DataFrame(rows);d.to_csv(DEST/'hydraulic_envelope.csv',index=False);pd.concat(details).to_csv(DEST/'parent_depth_envelopes.csv',index=False)
 summary=dict(states=len(d),smaller_timestep_states=int(d.requires_smaller_dt.sum()),minimum_screened_dt_seconds=float(d.envelope_dt_seconds.min()),
  states_exceeding_alpha_at_low_flow=int(d.links_exceeding_current_alpha.gt(0).sum()),max_depth_m=float(d.depth_max_m.max()),max_depth_width_ratio=float(d.depth_width_ratio_max.max()),
  limitations='Discharge envelope assumes steady branch allocation of observed input extrema; runtime overshoot and coefficient checks remain required. Alpha exceedance is spatial-accuracy evidence, not proof of instability. Depth envelope spans n=.01..0.1, not calibrated results. No grids changed.',
  source_hashes={str(p):r.digest(p) for p in [HERE/'screen.py',HERE/'prototype.py',HERE/'engine.cpp']})
 r.write_json(DEST/'screen_summary.json',summary);print(json.dumps(summary,indent=2),flush=True)
if __name__=='__main__':main()
