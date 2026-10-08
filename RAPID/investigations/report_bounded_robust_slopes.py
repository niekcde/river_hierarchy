"""Compare controlled robust-loss profiles with cycle slopes and squared-loss controls."""
from pathlib import Path
import sys,json,numpy as np,pandas as pd
from scipy.stats import theilslopes
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'RAPID/assets/bounded_robust_slopes_20261002';PREV=ROOT/'RAPID/assets/smoothing_floor_allocation_20261002';sources=json.loads((OUT/'method.json').read_text())['sources'];windows=pd.read_csv(ROOT/'RAPID/assets/selected_slope_review_20261002/windows.csv',dtype={'example':str});rows=[];priority=[]
if len(sys.argv)>1:
 selected=set(sys.argv[1:]);rows=pd.read_csv(OUT/'windows.csv',dtype={'example':str}).query('example not in @selected').to_dict('records');priority=pd.read_csv(OUT/'priority.csv',dtype={'example':str}).query('example not in @selected').to_dict('records');sources={k:v for k,v in sources.items() if k in selected}
for ex,source in sources.items():
 f=pd.read_parquet(OUT/ex/'floor_0.05/profile.parquet');o=pd.read_parquet(Path(source)/'reference_profile_observations.parquet');x=f.reference_section_dist_out_m.to_numpy();h=f.fitted_wse_m.to_numpy();old=f.previous_candidate.to_numpy()
 selections=[('window',z.lo,z.hi) for z in windows[windows.example==ex].itertuples()]
 if ex=='13':selections.append(('priority',63559.376602,66166.728978))
 if ex=='46':selections.extend([('priority',73649.,83143.),('priority',425713.,428445.)])
 for kind,a,b in selections:
  cv=[]
  for aid,z in o[o.reference_section_dist_out_m.between(a,b)].groupby('acquisition_id'):
   q=z.groupby('reference_section_dist_out_m').wse_m.mean().sort_index();xx=q.index.to_numpy();minimum=6 if kind=='window' else 4;coverage=.7 if kind=='window' else .5
   if len(q)<minimum or np.ptp(xx)/(b-a)<coverage or (kind=='window' and np.diff(xx).max()/(b-a)>.3):continue
   cv.append(dict(cycle=z.cycle.iloc[0],obs=theilslopes(q.to_numpy(),xx).slope,old=theilslopes(np.interp(xx,x,old),xx).slope,new=theilslopes(np.interp(xx,x,h),xx).slope))
  if not cv:continue
  v=pd.DataFrame(cv).groupby('cycle').median();rec=dict(example=ex,lo=a,hi=b,n_cycles=len(v),obs_p10=v.obs.quantile(.1),obs_median=v.obs.median(),obs_p90=v.obs.quantile(.9),old_median=v.old.median(),new_median=v.new.median());(rows if kind=='window' else priority).append(rec)
 print(ex,'reviewed',flush=True)
w=pd.DataFrame(rows);w['old_outside']=(w.old_median<w.obs_p10)|(w.old_median>w.obs_p90);w['new_outside']=(w.new_median<w.obs_p10)|(w.new_median>w.obs_p90);w.to_csv(OUT/'windows.csv',index=False);pd.DataFrame(priority).to_csv(OUT/'priority.csv',index=False)
w[w.n_cycles>=5].groupby('example').agg(n_windows=('new_outside','size'),old_outside=('old_outside','sum'),new_outside=('new_outside','sum')).to_csv(OUT/'window_summary.csv')
