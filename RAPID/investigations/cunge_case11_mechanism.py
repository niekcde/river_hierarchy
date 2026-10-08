"""Case 11 mechanism diagnostics at identical n and K; no calibration changes.

Diffusivity scalings are causal sensitivity probes, not proposed parameter fits.
The unit step is deliberately abrupt; the relaxed fixed-X scheme can undershoot.
Analytic response moments do not describe water-particle travel times.
"""
from pathlib import Path
import sys,json
import pandas as pd,numpy as np
root=Path(__file__).resolve().parents[2];sys.path.insert(0,str(root));sys.path.insert(0,str(root/'RAPID/src'))
from rapid_tools.cunge import parameters,topology,route
out=root/'RAPID/outputs/cunge_calibration_alpha2';r=pd.read_csv(out/'comparison.csv',dtype={'example':str});r=r[r.example.eq('11')].set_index('method');p=Path(r.loc['cunge','output_dir']);g=pd.read_parquet(p/'frozen_grid.parquet');dt=10.;step=690.;nsub=69
order,edges,rootnode,end=topology(g)
# path lengths and flow-weighted means via DAG sums
stats={rootnode:np.array([1.,0.,0.,0.,0.])};mins={rootnode:0};maxs={rootnode:0}
for u in order:
 for i in edges.get(u,[]):
  row=g.iloc[i];v=row.v;s=row.split_fraction;L=row.link_length_m
  st=stats[u]; new=np.array([st[0],st[1]+st[0]*L,0,0,0])*s
  stats[v]=stats.get(v,np.zeros(5))+new
  mins[v]=min(mins.get(v,np.inf),mins[u]+L);maxs[v]=max(maxs.get(v,0),maxs[u]+L)
print('LENGTH',dict(total_km=g.link_length_m.sum()/1000,min_path_km=mins[end]/1000,max_path_km=maxs[end]/1000,weighted_path_km=stats[end][1]/1000,links=len(g),cells=int(g['count'].sum())))
def metrics(f):
 sig={rootnode:np.array([1.,0.,0.,1.])}
 for u in order:
  for i in edges.get(u,[]):
   row=f.iloc[i];v=row.v;s=row.split_fraction;k=row.rapid_k;x=row.rapid_x;cnt=row['count'];mu=cnt*k;var=cnt*k*k*(1-2*x)
   a=(dt-2*k*x)/(dt+2*k*(1-x));old=sig[u]
   val=s*np.array([old[0],old[1]+old[0]*mu,old[2]+2*mu*old[1]+old[0]*(var+mu*mu),old[3]*a**cnt])
   sig[v]=sig.get(v,np.zeros(4))+val
 st=sig[end];return dict(mean_h=st[1]/3600,std_h=np.sqrt(st[2]-st[1]**2)/3600,same_native_step_gain=st[3],c_min=f.rapid_celerity_mps.min(),c_max=f.rapid_celerity_mps.max(),x_min=f.rapid_x.min(),x_max=f.rapid_x.max())
for method in ['fixed_x','cunge']:
 print(method,'own_n',metrics(parameters(g,r.loc[method,'n_manning'],method=method)))
f=parameters(g,r.loc['cunge','n_manning']);fb=parameters(g,r.loc['cunge','n_manning'],method='fixed_x');np.testing.assert_array_equal(f.rapid_k,fb.rapid_k)
print('fixed_x SAME n',metrics(fb))
f['Q_manning']=f.width*f.depth_m*.6*f.rapid_celerity_mps_raw
f['Q_ratio']=f.reference_discharge_cms/f.Q_manning
f['weighted_mean_h']=f.reference_flow_fraction*f['count']*f.rapid_k/3600
f['weighted_var_h2']=f.reference_flow_fraction*f['count']*f.rapid_k**2*(1-2*f.rapid_x)/3600**2
print(f.sort_values('weighted_var_h2',ascending=False)[['id_link','link_length_m','width','slope','depth_m','reference_discharge_cms','rapid_celerity_mps','rapid_x','Q_ratio','weighted_mean_h','weighted_var_h2']].head(8).to_string(index=False))
print('ratios',f.Q_ratio.describe().to_dict())
# Diagnostic runs use identical n, K, forcing, timestep and grid; vary X only.
w=pd.read_parquet(p/'cunge_best_wave.parquet'); flow=w.q_inflow_cms.to_numpy();t=w.time_seconds.to_numpy()
trials=[]
for label,fac in [('fixed_x',None),('cunge_D_1',1),('cunge_D_0.1',.1),('cunge_D_0.01',.01),('cunge_Q_manning',-1)]:
 ff=f.copy()
 if fac is None:ff=fb.copy()
 else:
  factor=(1/f.Q_ratio) if fac==-1 else fac
  ff['rapid_x']=.5*(1-factor*f.hydraulic_length_scale_m/f.rapid_effective_length_m)
 q,audit=route(ff,flow,nsub,dt)
 trials.append(dict(trial=label,lag_h=(t[q.argmax()]-t[flow.argmax()])/3600,peak_cms=q.max(),min_native=audit['minimum_native_flow_cms'],**metrics(ff)))
print('ABLATIONS',json.dumps(trials,indent=2))
pd.DataFrame(trials).to_csv(out/'case_11_mechanism_diagnostic.csv',index=False)
f.to_csv(out/'case_11_link_diagnostic.csv',index=False)
# Positive unit-step response, not an interpretation of parcel travel time.
stepflow=np.ones(len(flow));stepflow[0]=0.
step_rows=[]
for method,ff in [('fixed_x',fb),('cunge',f)]:
 response,audit=route(ff,stepflow,nsub,dt)
 row=dict(method=method)
 for hours in [0,1,6,8,24,48,72,168,720]:
  j=1+int(hours*3600/step);row[f'response_at_{hours}h']=float(response[j])
 for level in [.1,.5,.9,.95,.99]:
  ids=np.flatnonzero(response>=level);row[f'time_to_{level}_h']=float((ids[0]-1)*step/3600) if len(ids) else None
 step_rows.append(row)
print('STEP',json.dumps(step_rows,indent=2));pd.DataFrame(step_rows).to_csv(out/'case_11_step_response_diagnostic.csv',index=False)
rr=f.loc[f.id_link.eq(54)].iloc[0];print('LINK54',dict(count=int(rr['count']),K_h=rr.rapid_k/3600,X=rr.rapid_x,D_m2s=rr.hydraulic_diffusivity_m2_s,ld_km=rr.hydraulic_length_scale_m/1000,a=(dt-2*rr.rapid_k*rr.rapid_x)/(dt+2*rr.rapid_k*(1-rr.rapid_x)),relaxation_days=rr.rapid_k*(1-rr.rapid_x)/86400,Q_manning=rr.Q_manning,slope=rr.slope))
