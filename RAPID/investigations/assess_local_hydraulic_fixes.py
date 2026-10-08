"""Read-only local-change magnitudes and modest shared n/alpha sensitivity."""
from pathlib import Path
import sys
import numpy as np,pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parent))
from common_celerity_cap_audit import interval_conflicts
ROOT=Path(__file__).resolve().parents[2];BASE=ROOT/'RAPID/assets/bounded_robust_slopes_20261002';RUN=BASE/'common_caps';OUT=RUN/'fix_options';OUT.mkdir(exist_ok=True)
d=pd.read_csv(RUN/'stability_link_details.csv',dtype={'example':str});ref=pd.read_csv(BASE/'hydraulic_references.csv',dtype={'example':str}).set_index('example');choices=pd.read_csv(RUN/'stability_conflict_counts.csv',dtype={'example':str}).set_index('example');rows=[];sensitivity=[];network=[]
for ex,g in d.groupby('example'):
 a=ref.loc[ex];dt=float(choices.loc[ex,'representative_dt_s']);c=g.celerity_mps.to_numpy();raw=g.celerity_raw_mps.to_numpy();L=g.effective_length_m.to_numpy();target=np.clip(c,.2*L/dt,1.8*L/dt);required=~np.isclose(target,c,rtol=1e-10);K=L/c;network.append(dict(example=ex,X_continuous_max=float(K.min()/(K.min()+K.max()))))
 for pos in np.where(required)[0]:
  z=g.iloc[pos].to_dict();r=target[pos]/raw[pos];z.update(target_celerity_mps=target[pos],change_pct_from_used=100*(target[pos]/c[pos]-1),n_equivalent=.035/r,width_factor_equivalent=r**(3/(2*.65)),depth_factor_equivalent=r**1.5,slope_factor_equivalent=r*r,alpha_equivalent=.65+1.5*np.log(r)/np.log(z['wid_adj_routing']/a.reference_width_m),equivalent_length_target=float(np.clip(L[pos],dt*c[pos]/1.8,dt*c[pos]/.2)));rows.append(z)
 for alpha in [.60,.625,.65,.675,.71]:
  depth=a.stored_depth_m*(g.wid_adj_routing.to_numpy()/a.reference_width_m)**alpha
  for n in [.025,.030,.035,.040,.050]:
   cr=5/(3*n)*depth**(2/3)*np.sqrt(g.slope.to_numpy());cc=np.clip(cr,.05,5);kk=L/cc;t,sl,fa=interval_conflicts(.2*kk,1.8*kk);sensitivity.append(dict(example=ex,alpha=alpha,n=n,feasible=not (sl|fa).any(),minimum_exceptions=int((sl|fa).sum()),clipped_links=int((cr!=cc).sum()),continuous_feasible=.2*kk.max()<=1.8*kk.min()))
pd.DataFrame(rows).to_csv(OUT/'local_equivalent_changes.csv',index=False);s=pd.DataFrame(sensitivity);s.to_csv(OUT/'n_alpha_sensitivity.csv',index=False);summary=s.groupby(['alpha','n']).agg(passing=('feasible','sum'),remaining_intervals=('minimum_exceptions','sum'),clipped_links=('clipped_links','sum')).reset_index();summary.to_csv(OUT/'n_alpha_summary.csv',index=False);pd.DataFrame(network).to_csv(OUT/'X_feasibility_limits.csv',index=False)
local=pd.DataFrame(rows);print(local.groupby('example').agg(n_links=('id_link','size'),min_change=('change_pct_from_used','min'),max_change=('change_pct_from_used','max'),n_min=('n_equivalent','min'),n_max=('n_equivalent','max')).to_string());print(summary.to_string(index=False));print(s.groupby('example').feasible.any().to_string());print(pd.DataFrame(network).to_string(index=False))
