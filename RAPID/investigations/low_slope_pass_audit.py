"""Counts refreshed parent low slopes and inspects case11 raw acquisition slopes."""
from pathlib import Path
import json
import numpy as np,pandas as pd
from scipy.stats import linregress,theilslopes
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'RAPID/assets/low_slope_pass_audit_20261002';OUT.mkdir(parents=True,exist_ok=True)
EX=['02','03','05','11','13','15','21','22','37','41','45','46','48','54','55','56']
counts=[];lowlinks=[]
for ex in EX:
 suffix='_direction130' if ex=='15' else ''
 p=ROOT/f'network_variants/output/{ex}_indep{suffix}__anchored/states/S000_base/hydraulic_context/link_hydraulic_context.csv'
 d=pd.read_csv(p);length=d.actual_length_m if 'actual_length_m' in d else d.link_length_m
 for threshold in [1e-6,1e-5,1e-4]:
  mask=d.slope_used<threshold
  counts.append(dict(example=ex,threshold=threshold,total_links=len(d),low_links=int(mask.sum()),low_link_pct=100*mask.mean(),total_length_km=length.sum()/1000,low_length_km=length[mask].sum()/1000,low_length_pct=100*length[mask].sum()/length.sum()))
 lowlinks.append(d[d.slope_used<1e-5].assign(example=ex))
pd.DataFrame(counts).to_csv(OUT/'counts.csv',index=False);pd.concat(lowlinks,ignore_index=True).to_csv(OUT/'low_links.csv',index=False)
p=ROOT/'network_variants/output/_slope_refit_candidates/profiles/11';o=pd.read_parquet(p/'reference_profile_observations.parquet');f=pd.read_parquet(p/'reference_profile.parquet')
acq=[];rows=[]
windows={'flat_region_39.9_42.54':(39902.90292294653,42542.259934657304),'downstream_of_break_40_44':(40000,44000),'upstream_of_break_44_48':(44000,48000),'wide_38_44':(38000,44000)}
for ordinal,(aid,g) in enumerate(o.groupby('acquisition_id',sort=False),1):
 acq.append(dict(display_ordinal=ordinal,acquisition_id=aid,time=str(g.time.min()),cycle=g.cycle.iloc[0],orbit_pass=g['pass'].iloc[0]))
 for name,(lo,hi) in windows.items():
  z=g[g.reference_section_dist_out_m.between(lo,hi)].groupby('reference_section_dist_out_m',as_index=False).wse_m.mean()
  if len(z)<3:continue
  x=z.reference_section_dist_out_m.to_numpy();y=z.wse_m.to_numpy();fit=linregress(x,y);robust=theilslopes(y,x)
  rows.append(dict(display_ordinal=ordinal,acquisition_id=aid,window=name,n_positions=len(z),span_m=x.max()-x.min(),ols_slope=fit.slope,ols_r2=fit.rvalue**2,ols_stderr=fit.stderr,theil_slope=robust.slope,fitted_profile_slope_same_positions=linregress(x,np.interp(x,f.reference_section_dist_out_m,f.fitted_wse_m)).slope))
pd.DataFrame(acq).to_csv(OUT/'acquisitions.csv',index=False);pd.DataFrame(rows).to_csv(OUT/'pass_slopes.csv',index=False)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig,axs=plt.subplots(2,1,figsize=(12,8),constrained_layout=True)
for ordinal,(aid,g) in enumerate(o.groupby('acquisition_id',sort=False),1):
 z=g[g.reference_section_dist_out_m.between(37000,49000)].sort_values('reference_section_dist_out_m')
 label=f'{ordinal}: cycle {int(g.cycle.iloc[0])}'
 axs[0].plot(z.reference_section_dist_out_m/1000,z.stage_aligned_wse_m,'.-',lw=1,alpha=.7,label=label)
 # Subtract each pass mean in the shared flat window to compare slopes without global-offset distraction.
 mask=z.reference_section_dist_out_m.between(*windows['flat_region_39.9_42.54'])
 zz=z[mask]
 axs[1].plot(zz.reference_section_dist_out_m/1000,zz.wse_m-zz.wse_m.mean(),'.-',lw=1.3,label=label)
for ax in axs:
 ax.axvspan(41.949565,42.306702,color='gray',alpha=.2,label='link52 projected span')
 ax.grid(alpha=.15);ax.set(xlabel='Distance to outlet (km)')
z=f[f.reference_section_dist_out_m.between(37000,49000)]
axs[0].plot(z.reference_section_dist_out_m/1000,z.fitted_wse_m,'k-',lw=3,label='Promoted profile')
axs[0].axvline(44,color='black',ls='--');axs[0].set(ylabel='Stage-aligned WSE (m)',title='Case 11: individual acquisitions and promoted profile')
zz=f[f.reference_section_dist_out_m.between(*windows['flat_region_39.9_42.54'])]
axs[1].plot(zz.reference_section_dist_out_m/1000,zz.fitted_wse_m-zz.fitted_wse_m.mean(),'k-',lw=3)
axs[1].set(ylabel='WSE relative to each trace mean (m)',title='Flat region: pass slopes compared on the same spatial window')
axs[0].legend(ncol=4,fontsize=8)
fig.savefig(OUT/'case11_passes.png',dpi=170);plt.close(fig)
print('Saved',OUT)
