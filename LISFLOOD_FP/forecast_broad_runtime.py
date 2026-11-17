"""Approximate full-run work from cell count and initial normal-depth Courant rates.
Measured trial wall times include subprocess overhead/spinup and concurrent load.
This is an order-of-magnitude forecast, not a benchmark or a completion promise.
"""
import json
import numpy as np
import pandas as pd
import broad_study as b

def work(f,n,Q):
 counts=np.ceil(f.link_length_m.to_numpy()/100)
 dx=f.link_length_m.to_numpy()/counts
 h=(Q*f.reference_flow_fraction.to_numpy()*n/(f.width.to_numpy()*np.sqrt(f.slope.to_numpy())))**.6
 return float(counts.sum()*np.max(np.sqrt(9.80665*h)/dx))

def main():
 rows=[]
 index_path=b.OUT/'runtime_geometry_index.csv'
 index=pd.read_csv(index_path,dtype={'example':str}) if index_path.exists() else None
 for ex,c in b.CASES.items():
  cp=b.OUT/ex/'calibration.json';bp=b.OUT/ex/'baseline.json'
  cal=json.loads(cp.read_text()) if cp.exists() else {}
  ref=cal.get('final',{})
  if not ref.get('valid') and bp.exists():ref=json.loads(bp.read_text())
  if not ref.get('valid'):continue
  n=ref['n'];Q=float(b.forcing(ex).forcing_cms.iloc[0]);base=b.frame(ex,'S000_base');bw=work(base,n,Q)
  for j in c['jobs']:
   if index is not None:
    ix=index[index.example.eq(ex)].set_index('state');r=ix.loc[j['state_id']]
    ratio=float(r.normalized_work/ix.loc['S000_base','normalized_work']);cells=int(r.cells100)
   else:
    f=pd.read_parquet(j['parameters']);ratio=work(f,n,Q)/bw;cells=int(np.ceil(f.link_length_m/100).sum())
   rows.append(dict(example=ex,state=j['state_id'],calibration_status=cal.get('status','pending'),n=n,cells100=cells,work_ratio_to_original=ratio,original_measured_seconds=ref['seconds'],estimated_seconds=ref['seconds']*ratio))
 df=pd.DataFrame(rows);df.to_csv(b.OUT/'runtime_forecast_states.csv',index=False)
 if not len(df):return
 g=df.groupby('example').agg(states=('state','size'),estimated_serial_hours=('estimated_seconds',lambda x:x.sum()/3600),same_original_cost_hours=('original_measured_seconds',lambda x:x.sum()/3600),original_measured_seconds=('original_measured_seconds','first'))
 g.to_csv(b.OUT/'runtime_forecast_examples.csv')
 print(g.to_string());print('TOTAL estimated serial hours',df.estimated_seconds.sum()/3600,'same-original-cost hours',df.original_measured_seconds.sum()/3600)
if __name__=='__main__':main()
