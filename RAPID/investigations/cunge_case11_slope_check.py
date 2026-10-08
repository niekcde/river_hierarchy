"""Case 11 acquisition slopes and isolated link-54 slope sensitivities.

Sensitivity edits are diagnostic only, not a coherent revised shared-head profile.
No production inputs or calibration parameters are changed.
"""
import sys,json
from pathlib import Path
import numpy as np,pandas as pd
from scipy.stats import theilslopes
root=Path(__file__).resolve().parents[2];sys.path[:0]=[str(root),str(root/'RAPID/src')]
from rapid_tools.cunge import parameters,route
out=root/'RAPID/outputs/cunge_calibration_alpha2';base=root/'RAPID/assets/bounded_robust_slopes_20261002/11/floor_0.05'
nd=pd.read_parquet(base/'old/nodes.parquet').set_index('id_node');bounds=sorted(nd.loc[[36,37],'reference_section_dist_out_m'].tolist());print('segment distance bounds',bounds)
o=pd.read_parquet(base/'observations.parquet');oo=o[o.reference_section_dist_out_m.between(*bounds)];rows=[]
for name,g in oo.groupby('acquisition_id'):
 x=g.reference_section_dist_out_m.to_numpy();y=g.wse_m.to_numpy()
 if len(g)<8 or np.ptp(x)<5000:continue
 slope,intercept,lo,hi=theilslopes(y,x)
 rows.append(dict(acquisition=name,date=str(g.time.min()),pass_id=int(g['pass'].iloc[0]),n=len(g),span_km=np.ptp(x)/1000,slope_m_km=slope*1000,low=lo*1000,high=hi*1000))
a=pd.DataFrame(rows);print(a.to_string(index=False));a.to_csv(out/'case_11_pass_slope_diagnostic.csv',index=False)
r=pd.read_csv(out/'comparison.csv',dtype={'example':str});rr=r[r.example.eq('11')&r.method.eq('cunge')].iloc[0];p=Path(rr.output_dir);grid=pd.read_parquet(p/'frozen_grid.parquet');w=pd.read_parquet(p/'cunge_best_wave.parquet');flow=w.q_inflow_cms.to_numpy();runs=[]
for factor in [.5,1.,2.,4.]:
 g=grid.copy();g.loc[g.id_link.eq(54),'slope']*=factor;g['celerity_times_n']=5/3*g.depth_m**(2/3)*np.sqrt(g.slope)
 f=parameters(g,float(rr.n_manning));q,audit=route(f,flow,69,10);link=f[f.id_link.eq(54)].iloc[0]
 runs.append(dict(slope_factor=factor,celerity=link.rapid_celerity_mps,K_h=link.rapid_k/3600,X=link.rapid_x,network_peak_lag_h=(q.argmax()-flow.argmax())*690/3600,min_flow=audit['minimum_native_flow_cms']))
print('ROUTING',json.dumps(runs,indent=2));pd.DataFrame(runs).to_csv(out/'case_11_local_slope_sensitivity.csv',index=False)
