import json
from pathlib import Path
import pandas as pd,numpy as np
root=Path(__file__).resolve().parents[2]; out=root/'synthetic_experiments/outputs/depth_reference_review_20261027'; rows=[]
for c in json.loads((root/'RAPID/outputs/event_window_plan_20261013/plan.json').read_text())['cases']:
 f=pd.read_parquet(c['forcing']);time=pd.to_datetime(f.time_utc,utc=True);mask=(time>=pd.Timestamp(c['window']['start_utc']))&(time<=pd.Timestamp(c['window']['end_utc']));q=f.loc[mask,'forcing_cms'];p=Path(c['forcing']).parents[2]/'reference_section/reference_kb_summary.json';meta=json.loads(p.read_text()) if p.exists() else {}
 peak=q.max();base=c['initial_cms'];end=c['final_cms']; rows.append(dict(example=c['example'],initial_cms=base,final_cms=end,event_min_cms=q.min(),event_peak_cms=peak,peak_start_ratio=peak/base,peak_end_ratio=peak/end,original_forcing_peak_cms=f.forcing_cms.max(),based_reference_discharge_cms=meta.get('reference_discharge_cms'),based_depth_m=meta.get('reference_depth_m'),based_source=str(p)))
r=pd.DataFrame(rows);r.to_csv(out/'real_forcing_reference_audit.csv',index=False);print(r.drop(columns=['based_source']).round(3).to_string(index=False));print('quantiles',r.peak_start_ratio.quantile([0,.25,.5,.75,1]).to_dict())
