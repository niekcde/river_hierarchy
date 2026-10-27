"""Independent refinement checks for the integrated conservative solver."""
from pathlib import Path
import sys,json
from dataclasses import replace
root=Path(__file__).resolve().parents[2]
for path in (root,root/'RAPID/src',root/'synthetic_experiments/src'):sys.path.insert(0,str(path))
import numpy as np,pandas as pd
from synthetic_experiments.networks import make_states
from synthetic_experiments.hydraulics import HydraulicSettings,prepare,forcing,route,summarize
config=json.loads((root/'synthetic_experiments/configs/depth_comparison_20261027.json').read_text())
p={**config['base_geometry'],**config['cases']['narrow_long']};g,pairs=make_states(p)
base=HydraulicSettings(method='variable_manning',tail_hours=192)
rows=[]
for name in ['simple/original','simple/full','nested/original','nested/main_island_only']:
 for test,s in [('reference',base),('half_dt',replace(base,dt_seconds=7.5)),('half_dx',replace(base,segment_length_m=100))]:
  t,q=forcing(s);w,a=route(prepare(g[name],p,s),q,s)
  stats=summarize(t,w,s.baseflow_cms);rows.append(dict(state=name,test=test,**stats,**a))
f=pd.DataFrame(rows);baseline=f[f.test=='reference'][['state','centroid_seconds','peak_seconds']]
f=f.merge(baseline,on='state',suffixes=('','_reference'))
f['centroid_change_seconds']=f.centroid_seconds-f.centroid_seconds_reference
f['peak_change_seconds']=f.peak_seconds-f.peak_seconds_reference
out=root/'synthetic_experiments/outputs/depth_methods_verification_20261027';out.mkdir(exist_ok=True)
f.to_csv(out/'refinement.csv',index=False)
assert f.network_mass_residual_m3.abs().max()<1
assert f.centroid_change_seconds.abs().max()<60
assert f.peak_change_seconds.abs().max()<=30
print(f[['state','test','centroid_change_seconds','peak_change_seconds','network_mass_residual_m3']].round(5).to_string(index=False))
