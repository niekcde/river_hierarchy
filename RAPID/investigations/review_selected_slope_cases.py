"""Review the selected 0.05-floor/old-allocation candidate against saved cycle observations."""
from pathlib import Path
import json
import numpy as np,pandas as pd
from scipy.stats import theilslopes
ROOT=Path(__file__).resolve().parents[2];RUN=ROOT/'RAPID/assets/smoothing_floor_allocation_20261002';OUT=ROOT/'RAPID/assets/selected_slope_review_20261002';OUT.mkdir(parents=True,exist_ok=True)
sources=json.loads((RUN/'method.json').read_text())['sources'];case_rows=[];windows=[];lowrows=[];endfits=[]
for ex,p in sources.items():
 p=Path(p);f=pd.read_parquet(RUN/ex/'floor_0.05/profile.parquet');g=pd.read_parquet(RUN/ex/'floor_0.01/profile.parquet');o=pd.read_parquet(p/'reference_profile_observations.parquet');n=pd.read_parquet(RUN/ex/'floor_0.05/old/nodes.parquet');links=pd.read_parquet(RUN/ex/'floor_0.05/old/links.parquet');meta=json.loads((p/'reference_profile_audit.json').read_text())['summary'];floor=.05*meta['reference_slope_m_per_m']
 x=f.reference_section_dist_out_m.to_numpy();old=f.old_profile.to_numpy();new=f.fitted_wse_m.to_numpy();small=g.fitted_wse_m.to_numpy();dx=np.diff(x);s=np.diff(new)/dx;active=s<=floor*(1+1e-4)+1e-10
 lo=float(n.reference_section_dist_out_m.min());hi=float(n.reference_section_dist_out_m.max());ends=[lo,hi];oh=np.interp(ends,x,old);nh=np.interp(ends,x,new);sh=np.interp(ends,x,small)
 # Compare after fitting each acquisition's offset independently; residual diagnostics are in-sample.
 oo=o.copy()
 for label,y in [('old',old),('new',new),('small_floor',small)]:
  r=oo.wse_m.to_numpy()-np.interp(oo.reference_section_dist_out_m,x,y);oo[label+'_res']=r-pd.Series(r).groupby(oo.acquisition_id.to_numpy()).transform('mean').to_numpy()
 summary=dict(example=ex,outlet_km=lo/1000,inlet_km=hi/1000,old_outlet=oh[0],new_outlet=nh[0],old_inlet=oh[1],new_inlet=nh[1],outlet_change=nh[0]-oh[0],inlet_change=nh[1]-oh[1],old_drop=oh[1]-oh[0],new_drop=nh[1]-nh[0],refit_at_old_floor_drop=sh[1]-sh[0],added_drop_from_floor=(nh[1]-nh[0])-(sh[1]-sh[0]),floor=floor,floor_active_length_km=float(dx[active].sum()/1000),floor_active_length_pct=100*dx[active].sum()/dx.sum(),low_links=int((links.slope<1e-5).sum()),min_link_slope=links.slope.min())
 for label,e in [('outlet',lo),('inlet',hi)]:
  z=o[abs(o.reference_section_dist_out_m-e)<=5000];summary[label+'_cycles_5km']=z.cycle.nunique();summary[label+'_positions_5km']=z.reference_section_dist_out_m.nunique();summary[label+'_nearest_observation_m']=float(abs(o.reference_section_dist_out_m-e).min())
 # Same-acquisition endpoint drops: avoid combining arbitrary stage offsets from different passes.
 drops=[]
 for aid,z in o.groupby('acquisition_id'):
  fits=[]
  for e in ends:
   q=z[abs(z.reference_section_dist_out_m-e)<=5000].groupby('reference_section_dist_out_m').wse_m.mean().sort_index()
   if len(q)<4 or np.ptp(q.index)<1500:break
   slope,intercept,_,_=theilslopes(q.to_numpy(),q.index.to_numpy());fits.append((slope*e+intercept,len(q),float(abs(q.index-e).min())))
  if len(fits)==2:
   drop=fits[1][0]-fits[0][0];drops.append((z.cycle.iloc[0],drop));endfits.append(dict(example=ex,acquisition_id=aid,cycle=z.cycle.iloc[0],drop_m=drop,outlet_nearest_m=fits[0][2],inlet_nearest_m=fits[1][2]))
 cs=pd.DataFrame(drops,columns=['cycle','drop']).groupby('cycle')['drop'].median()
 summary.update(endpoint_drop_cycles=len(cs),endpoint_drop_p10=cs.quantile(.1),endpoint_drop_p50=cs.quantile(.5),endpoint_drop_p90=cs.quantile(.9));case_rows.append(summary)
 edges=np.r_[np.arange(lo,hi,10000),hi]
 for left,right in zip(edges[:-1],edges[1:]):
  z=oo[oo.reference_section_dist_out_m.between(left,right)];cv=[]
  for aid,zz in z.groupby('acquisition_id'):
   q=zz.groupby('reference_section_dist_out_m').wse_m.mean().sort_index();xx=q.index.to_numpy()
   if len(q)<6 or np.ptp(xx)/(right-left)<.7 or np.diff(xx).max()/(right-left)>.3:continue
   cv.append(dict(cycle=zz.cycle.iloc[0],observed=theilslopes(q.to_numpy(),xx).slope,old=theilslopes(np.interp(xx,x,old),xx).slope,new=theilslopes(np.interp(xx,x,new),xx).slope))
  v=pd.DataFrame(cv);v=v.groupby('cycle').median() if len(v) else pd.DataFrame(columns=['observed','old','new'])
  overlaps=np.maximum(0,np.minimum(x[1:],right)-np.maximum(x[:-1],left));rec=dict(example=ex,lo=left,hi=right,n_cycles=len(v),obs_p10=v.observed.quantile(.1),obs_median=v.observed.median(),obs_p90=v.observed.quantile(.9),old_median=v.old.median(),new_median=v.new.median(),floor_fraction=np.sum(overlaps[active])/np.sum(overlaps),drop_old=np.interp(right,x,old)-np.interp(left,x,old),drop_new=np.interp(right,x,new)-np.interp(left,x,new))
  for name in ['old','new','small_floor']:
   values=z.assign(squared=z[name+'_res']**2).groupby('cycle').squared.mean();rec[name+'_cycle_rmse']=float(np.sqrt(values.mean()))
  windows.append(rec)
 pos=n.set_index('id_node').reference_section_dist_out_m
 for r in links[links.slope<1e-5].itertuples():
  left,right=sorted([pos[r.id_us_node],pos[r.id_ds_node]]);overlap=np.maximum(0,np.minimum(x[1:],right)-np.maximum(x[:-1],left));record=dict(example=ex,link=r.id_link,kind=r.ownership_kind,domain=r.ownership_domain_id,lo=left,hi=right,length_m=r.actual_length_m,slope=r.slope,old_slope=r.old_slope,drop_mm=r.slope*r.actual_length_m*1000,profile_slope=(np.interp(right,x,new)-np.interp(left,x,new))/(right-left) if right>left else np.nan,floor_fraction=np.sum(overlap[active])/np.sum(overlap) if overlap.sum()>0 else np.nan)
  for label,a,b in [('exact',left,right),('expanded',max(x.min(),left-1000),min(x.max(),right+1000))]:
   data=[]
   for aid,z in o[o.reference_section_dist_out_m.between(a,b)].groupby('acquisition_id'):
    q=z.groupby('reference_section_dist_out_m').wse_m.mean().sort_index();xx=q.index.to_numpy()
    if b<=a or len(q)<4 or np.ptp(xx)/(b-a)<.5:continue
    data.append((z.cycle.iloc[0],theilslopes(q.to_numpy(),xx).slope,theilslopes(np.interp(xx,x,new),xx).slope))
   v=pd.DataFrame(data,columns=['cycle','observed','fitted']).groupby('cycle').median();record.update({label+'_cycles':len(v),label+'_p10':v.observed.quantile(.1),label+'_median':v.observed.median(),label+'_p90':v.observed.quantile(.9),label+'_fit':v.fitted.median(),label+'_fraction_positive':float((v.observed>0).mean()) if len(v) else np.nan})
  lowrows.append(record)
 print(ex,'reviewed',flush=True)
pd.DataFrame(case_rows).to_csv(OUT/'cases.csv',index=False);pd.DataFrame(windows).to_csv(OUT/'windows.csv',index=False);pd.DataFrame(lowrows).to_csv(OUT/'low_links.csv',index=False);pd.DataFrame(endfits).to_csv(OUT/'endpoint_cycle_drops.csv',index=False)
print('DONE')
