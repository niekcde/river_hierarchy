"""Read-only SWOT-support audit for five refreshed profile-derived controllers. Run from repository root."""
from pathlib import Path
import json
import numpy as np,pandas as pd
out=Path('RAPID/assets/low_slope_support_20261002');rows=[]
for ex,link in [('11',52),('13',174),('21',52),('41',95),('46',71)]:
 root=Path('network_variants/output')/f'{ex}_indep__anchored/states/S000_base/hydraulic_context'
 m=json.loads((root/'hydraulic_context_manifest.json').read_text());p=Path(m['slope_refit_promotion']['reference_profile_dir'])
 f=pd.read_parquet(p/'reference_profile.parquet').sort_values('reference_section_dist_out_m');o=pd.read_parquet(p/'reference_profile_observations.parquet')
 nodes=pd.read_csv(root/'node_hydraulic_context.csv').set_index('id_node');links=pd.read_csv(root/'link_hydraulic_context.csv');r=links.set_index('id_link').loc[link]
 a,b=sorted([nodes.loc[r.id_us_node,'reference_section_dist_out_m'],nodes.loc[r.id_ds_node,'reference_section_dist_out_m']]);x=f.reference_section_dist_out_m.to_numpy();y=f.fitted_wse_m.to_numpy();s=np.diff(y)/np.diff(x)
 audit=json.loads((p/'reference_profile_audit.json').read_text())['summary'];floor=audit['minimum_slope_m_per_m']
 mask=(x[:-1]<b)&(x[1:]>a);ii=np.where(mask)[0];mid=(a+b)/2;j=min(len(s)-1,max(0,np.searchsorted(x,mid)-1));l=h=j
 # Flat neighborhood defined explicitly: slope at most twice the configured profile floor.
 low=s<=2*floor*(1+1e-6)
 if low[j]:
  while l>0 and low[l-1]:l-=1
  while h<len(s)-1 and low[h+1]:h+=1
 else:l=h=j
 rec=dict(example=ex,link=link,length_m=r.link_length_m,projected_span_m=b-a,x_min=a,x_max=b,slope=r.slope_used,profile_floor=floor,profile_span_mean_slope=(np.interp(b,x,y)-np.interp(a,x,y))/(b-a),profile_head_difference_m=np.interp(b,x,y)-np.interp(a,x,y),mapped_head_difference_m=r.candidate_head_us_m-r.candidate_head_ds_m,flat_neighborhood_length_m=x[h+1]-x[l] if low[j] else 0,flat_x_min=x[l],flat_x_max=x[h+1],profile_match_max_error_m=max(abs(np.interp(z,x,y)-nodes.loc[n,'profile_head_m']) for n,z in [(r.id_us_node,nodes.loc[r.id_us_node,'reference_section_dist_out_m']),(r.id_ds_node,nodes.loc[r.id_ds_node,'reference_section_dist_out_m'])]))
 for label,left,right in [('span',a,b),('local_2km',a-1000,b+1000),('flat',x[l],x[h+1])]:
  obs=o[o.reference_section_dist_out_m.between(left,right)];grid=f[f.reference_section_dist_out_m.between(left,right)]
  rec.update({label+'_observations':len(obs),label+'_observed_positions':obs.reference_section_dist_out_m.nunique(),label+'_grid_positions':len(grid),label+'_acquisitions':obs.acquisition_id.nunique(),label+'_median_uncertainty_m':obs.wse_uncertainty_m.median(),label+'_residual_rms_m':np.sqrt(np.mean(obs.residual_m**2))})
 # A separate absolute threshold makes broad low-slope extent comparable across cases.
 ll=hh=j
 if s[j]<1e-5:
  while ll>0 and s[ll-1]<1e-5:ll-=1
  while hh<len(s)-1 and s[hh+1]<1e-5:hh+=1
  rec['below_1e5_neighborhood_length_m']=x[hh+1]-x[ll]
 else:rec['below_1e5_neighborhood_length_m']=0
 rows.append(rec)
 print(ex, json.dumps(rec,default=float),flush=True)
pd.DataFrame(rows).to_csv(out/'support.csv',index=False)
