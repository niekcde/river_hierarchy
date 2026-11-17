"""Trace extreme screening depths to unchanged inputs and roughness assumptions."""
import pandas as pd,numpy as np
from prototype import r,DEST

def main():
 r.init_worker(DEST,'depth-trace');screen=pd.read_csv(DEST/'hydraulic_envelope.csv',dtype={'example':str})
 chosen=screen[screen.state.ne('single_centerline')].nlargest(6,'depth_max_m')
 chosen=pd.concat([chosen,screen.nlargest(3,'depth_width_ratio_max'),screen[screen.state.eq('S000_base')]])
 rows=[]
 for rec in chosen.drop_duplicates(['example','state']).itertuples():
  f,dt,job=r.prepare_frame(rec.example,rec.state,r.RUNS[2]);src=pd.read_parquet(job['source_parameters']).set_index('id_link')
  base,_,_=r.parent_inputs(rec.example);oldn=float(base.n_manning.iloc[0])
  q=rec.network_qmax_cms*f.reference_flow_fraction.to_numpy();h=(.1*q/(f.width*np.sqrt(f.slope)))**.6
  ids=set(h.nlargest(2).index)|set((h/f.width).nlargest(2).index)
  for i in ids:
   a=f.iloc[i];orig=src.loc[a.id_link]
   for n in [.01,oldn,.035,.1]:
    depth=(n*q[i]/(a.width*np.sqrt(a.slope)))**.6
    rows.append(dict(example=rec.example,state=rec.state,id_link=int(a.id_link),n=n,legacy_reference_n=oldn,
       width_m=a.width,slope=a.slope,previous_slope=orig.slope,previous_width_m=orig.width,
       previous_depth_m=orig.depth_m,peak_allocated_q_cms=q[i],reference_fraction=a.reference_flow_fraction,
       depth_at_peak_m=depth,depth_width_ratio=depth/a.width,
       fixed_midpoint_depth_m=(n*a.reference_discharge_cms/(a.width*np.sqrt(a.slope)))**.6,
       source_parameters=job['source_parameters']))
 d=pd.DataFrame(rows);d.to_csv(DEST/'extreme_depth_trace.csv',index=False)
 print(d[d.n.eq(.1)].nlargest(6,'depth_at_peak_m')[['example','state','id_link','width_m','slope','previous_slope','peak_allocated_q_cms','depth_at_peak_m','previous_depth_m']].to_string(index=False))
 key=d[(d.example=='22')&(d.state=='S122_unit_216')];print(key.to_string(index=False))
if __name__=='__main__':main()
