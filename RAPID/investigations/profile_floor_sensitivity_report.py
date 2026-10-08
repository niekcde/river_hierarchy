from pathlib import Path
import pandas as pd,numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=Path('RAPID/assets/profile_floor_sensitivity_20261002');d=pd.read_csv(p/'summary.csv',dtype={'example':str});base=d[d.eta==.01].set_index('example')
for metric in ['cv_mse','rms_m','head_loss_m','target_slope']:
 d[metric+'_change_pct']=100*(d[metric]/d.example.map(base[metric])-1)
d['floor_active_length_change_km']=(d.floor_active_length_m-d.example.map(base.floor_active_length_m))/1000
assert len(d)==25
assert d[d.eta==.01].profile_max_change_m.max()<1e-6
print(d[['example','eta','cv_mse_change_pct','rms_m_change_pct','head_loss_m_change_pct','target_slope','capped_feasible','capped_slow_link','context_status']].round(5).to_string(index=False))
d.to_csv(p/'comparison.csv',index=False)
fig,axs=plt.subplots(2,2,figsize=(11,7),constrained_layout=True)
for ex,g in d.groupby('example'):
 for ax,y,title in zip(axs.flat,['cv_mse_change_pct','head_loss_m_change_pct','target_slope','floor_active_length_change_km'],['Held-out acquisition MSE change (%)','Total fitted head-loss change (%)','Original problem-link slope (m/m)','Additional profile length at floor (km)']):
  ax.plot(g.eta,g[y],'-o',label=ex);ax.set(xlabel='Profile minimum-slope fraction',title=title,xticks=[.01,.02,.05,.07,.09]);ax.grid(alpha=.2)
axs[1,0].set_yscale('log');axs[0,0].legend(title='Case',ncol=3)
fig.suptitle('Profile-floor sensitivity · reviewed smoothing held fixed · domain floor unchanged')
fig.savefig(p/'floor_sensitivity.png',dpi=170);plt.close(fig)
