"""Uncalibrated parent pilots with existing midpoint cells, fixed legacy n/beta."""
import json,time
import numpy as np
import pandas as pd
from prototype import r,DEST,engine,route,parameters,HERE

def main():
 r.init_worker(DEST,'conservative-pilot');fun=engine('conservative.cpp');rows=[]
 envelope=pd.read_csv(DEST/'hydraulic_envelope.csv',dtype={'example':str}).set_index(['example','state'])
 for ex in ['05','54']:
  case=r.CTX['cases'][ex];f,dt,_=r.prepare_frame(ex,'S000_base',r.RUNS[2]);base,_,_=r.parent_inputs(ex)
  n=float(base.n_manning.iloc[0]);beta=float(base.diffusion_multiplier.iloc[0]);f=parameters(f,n,beta)
  dt=float(envelope.loc[(ex,'S000_base'),'envelope_dt_seconds'])
  data=pd.read_parquet(case['calibration_forcing']);meta=next(x for x in json.loads(open(case['calibration_metadata']).read()) if x['method']=='cunge')
  initial=meta['initial_cms'];prev=None
  for divisor in [1,2]:
   begin=time.monotonic()
   try:
    q,a,links=route(fun,f,data.forcing_cms.to_numpy(),dt/divisor,initial,beta)
    links.to_csv(DEST/f'conservative_{ex}_dt{divisor}_link_audit.csv',index=False)
    pd.DataFrame(dict(time_utc=data.time_utc,q_outlet_cms=q)).to_parquet(DEST/f'conservative_{ex}_dt{divisor}.parquet',index=False)
    rec=dict(example=ex,n=n,beta=beta,dt=dt/divisor,cells=int(f['count'].sum()),status='completed',seconds=time.monotonic()-begin,**a)
    if prev is not None:rec.update(peak_time_change_minutes=int(np.argmax(q)-np.argmax(prev)),max_wave_change_cms=float(abs(q-prev).max()),peak_relative_change=float(abs(q.max()/prev.max()-1)))
    prev=q
   except Exception as e:rec=dict(example=ex,dt=dt/divisor,status='failed',error=str(e),seconds=time.monotonic()-begin)
   rows.append(rec);pd.DataFrame(rows).to_csv(DEST/'conservative_parent_pilots.csv',index=False);print(json.dumps(rec),flush=True)
 r.write_json(DEST/'conservative_pilot_provenance.json',dict(calibrated=False,grid_changed=False,source_hashes={str(p):r.digest(p) for p in [HERE/'conservative.cpp',HERE/'prototype.py',HERE/'pilot_conservative.py']}))
if __name__=='__main__':main()
