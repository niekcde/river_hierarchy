from pathlib import Path
import json,numpy as np,pandas as pd,matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
r=Path('RAPID/assets/smoothing_floor_allocation_20261002');out=Path('RAPID/assets/selected_slope_review_20261002');src=json.loads((r/'method.json').read_text())['sources'];w=pd.read_csv(out/'windows.csv',dtype={'example':str})
fig,axes=plt.subplots(8,2,figsize=(16,26),constrained_layout=True)
for ax,(ex,p) in zip(axes.flat,src.items()):
 f=pd.read_parquet(r/ex/'floor_0.05/profile.parquet');o=pd.read_parquet(Path(p)/'reference_profile_observations.parquet');x=f.reference_section_dist_out_m.to_numpy();y=f.fitted_wse_m.to_numpy();res=o.wse_m-np.interp(o.reference_section_dist_out_m,x,y);offset=res.groupby(o.acquisition_id).transform('mean');o['aligned']=o.wse_m-offset;e=o.groupby('reference_section_dist_out_m').aligned.quantile([.1,.9]).unstack();ax.fill_between(e.index/1000,e[.1],e[.9],color='gray',alpha=.25,label='Aligned observations P10–P90');ax.plot(x/1000,f.old_profile,lw=1,label='Previous');ax.plot(x/1000,y,lw=1,label='Candidate');ax.set(title='Case '+ex,xlabel='Distance to outlet (km)',ylabel='WSE (m)');ax.grid(alpha=.2)
axes.flat[0].legend(fontsize=8);fig.savefig(out/'all_profiles.png',dpi=125);plt.close(fig)
fig,axes=plt.subplots(8,2,figsize=(16,26),constrained_layout=True)
for ax,(ex,z) in zip(axes.flat,w.groupby('example',sort=False)):
 z=z[z.n_cycles>=5];x=(z.lo+z.hi)/2000;ax.vlines(x,z.obs_p10*1e5,z.obs_p90*1e5,color='gray',alpha=.7,label='Cycle P10–P90');ax.scatter(x,z.obs_median*1e5,s=14,color='black',label='Cycle median');ax.plot(x,z.old_median*1e5,'o-',ms=3,lw=1,label='Previous');ax.plot(x,z.new_median*1e5,'o-',ms=3,lw=1,label='Candidate');ax.set(title='Case '+ex,xlabel='Distance to outlet (km)',ylabel='Slope (×10⁻⁵ m/m)');ax.grid(alpha=.2)
axes.flat[0].legend(fontsize=8);fig.savefig(out/'all_local_slopes.png',dpi=125);plt.close(fig)
fig,axes=plt.subplots(3,2,figsize=(14,12),constrained_layout=True)
for ax,(ex,a,b) in zip(axes.flat,[('21',75,115),('22',105,125),('13',62,68),('46',73,84),('46',424,431),('46',267,276)]):
 f=pd.read_parquet(r/ex/'floor_0.05/profile.parquet');o=pd.read_parquet(Path(src[ex])/'reference_profile_observations.parquet');o=o[o.reference_section_dist_out_m.between(a*1000,b*1000)];xx=f.reference_section_dist_out_m.to_numpy();new=f.fitted_wse_m.to_numpy();old=f.old_profile.to_numpy();ref=np.interp(a*1000,xx,new)
 for aid,z in o.groupby('acquisition_id'):
  z=z.sort_values('reference_section_dist_out_m');x=z.reference_section_dist_out_m.to_numpy();y=z.wse_m.to_numpy();offset=np.mean(y-np.interp(x,xx,new));ax.plot(x/1000,y-offset-ref,color='gray',alpha=.2,lw=.7)
 mask=(xx>=a*1000)&(xx<=b*1000);ax.plot(xx[mask]/1000,new[mask]-ref,color='red',label='Candidate');ax.plot(xx[mask]/1000,old[mask]-np.interp(a*1000,xx,old),color='blue',label='Previous (aligned at left)');ax.set(title=f'Case {ex}: {a}–{b} km',xlabel='Distance to outlet (km)',ylabel='Relative WSE (m)');ax.grid(alpha=.2)
axes.flat[0].legend();fig.savefig(out/'priority_sections.png',dpi=140);plt.close(fig)
