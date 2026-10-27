import importlib.util,json
from pathlib import Path
import pandas as pd,numpy as np
root=Path(__file__).resolve().parents[2];sp=importlib.util.spec_from_file_location('p',root/'synthetic_experiments/investigations/cunge_pilot.py');m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m)
s=json.loads(m.CONFIG.read_text());a=json.loads((m.OUT/'narrow_long/audit.json').read_text());frames,_,_=m.prepare(a['parameters'],s);t,flow,_=m.forcing(s,96);f=frames['simple/original'];rows=[]
for width in [475,25]:
 for variant in ['original','fourfold_refinement','manning_consistent_diffusivity_only']:
  g=f[(f.width==500)|(f.width==width)].copy();g['split_fraction']=1.;g['reference_flow_fraction']=1.
  if variant=='fourfold_refinement':
   g['count']*=4;g['rapid_effective_length_m']/=4;g['rapid_k']/=4;g['rapid_x']=.5-4*(.5-g.rapid_x)
  if variant=='manning_consistent_diffusivity_only':
   qm=.6*g.rapid_celerity_mps*g.width*g.depth_m
   g['rapid_x']=.5*(1-qm/(g.width*g.slope*g.rapid_celerity_mps*g.rapid_effective_length_m))
  q,_=m.cunge.route(g,flow,4,15)
  rows.append(dict(width=width,variant=variant,peak_hours=float(t[q.argmax()]/3600),peak_cms=float(q.max())))
result=pd.DataFrame(rows);print(result.to_string(index=False));result.to_csv(root/'synthetic_experiments/outputs/narrow_long_timing_diagnostic_20261026/diffusion_sensitivity.csv',index=False)
