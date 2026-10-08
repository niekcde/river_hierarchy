import sys,json
from pathlib import Path
import numpy as np,pandas as pd
R=Path.cwd();sys.path.insert(0,str(R/'RAPID/src'))
from rapid_tools.reference_profile import _curvature_penalty
O=R/'RAPID/assets/data_informed_base_slopes_20261002';sources=json.loads((O/'method.json').read_text())['sources'];rows=[]
for ex,path in sources.items():
 p=Path(path);f=pd.read_parquet(p/'reference_profile.parquet');o=pd.read_parquet(p/'reference_profile_observations.parquet');a=json.loads((p/'reference_profile_audit.json').read_text())['summary'];x=f.reference_section_dist_out_m.to_numpy();C=_curvature_penalty(x)
 for label,h in [('saved_profile',f.fitted_wse_m.to_numpy()),('converged_equal_observation',pd.read_parquet(O/ex/'control_equal_observation_profile.parquet').fitted_wse_m.to_numpy())]:
  v=o.wse_m.to_numpy()-np.interp(o.reference_section_dist_out_m,x,h);offset=pd.Series(v).groupby(o.acquisition_id.to_numpy()).transform('mean').to_numpy();res=v-offset;center=h-h.mean();data=float(res@res);smooth=float(a['lambda_value']*(center@(C@center)));rows.append(dict(example=ex,variant=label,data_sse=data,curvature_penalty=smooth,total_objective=data+smooth))
pd.DataFrame(rows).to_csv(O/'objective_controls.csv',index=False)
print(pd.DataFrame(rows).pivot(index='example',columns='variant',values='total_objective').to_string())
