from pathlib import Path
import numpy as np
import pandas as pd
from scipy.io import savemat
import broad_study as b
root=b.OUT/'investigation';dest=root/'first_clipping48'
src=root/'limiter_retained_48_n0.0207996059_dx50/_input_29515.npz'
d=dict(np.load(src));p=dict(np.load(dest/'first_failure.npz'))
f=pd.read_parquet(root/'current_mc/states/48/S000_base/parameters.parquet');cells=b.v.grid(f,'max50');m=b.build(cells,float(d['n']))
assert all(np.array_equal(m[k],d[k]) for k in ['z','w','dx','a','b','wf','dxf'])
d.setdefault('monitor_face',np.array(len(d['a'])-1))
for k in ['a','b','source','monitor_face','sample_indices']:d[k]=d[k].astype(float)
savemat(dest/'input.mat',d,oned_as='column')
s=p['step'];d.update(h=p['pre_h'],q=p['pre_q'],times=np.array([0.,s[1]]),forcing=np.array([s[2],s[2]]),sample_indices=np.array([0.,1.]))
(dest/'one_step').mkdir(exist_ok=True);savemat(dest/'one_step/input.mat',d,oned_as='column')
i=int(np.argmin(p['trial']));print('CELL',i,cells.iloc[i].to_dict());print('STEP',s,'h_before',p['pre_h'][i],'trial',p['trial'][i],'available_m3',p['pre_h'][i]*m['w'][i]*m['dx'][i],'net_loss_m3',-s[1]*p['balance'][i])
for j in np.where((m['a']==i)|(m['b']==i))[0]:
 print('FACE',j,'a,b',m['a'][j],m['b'][j],'Q',p['new_q'][j]*m['wf'][j],'hf',p['hf'][j],'wf',m['wf'][j],'oldQ',p['pre_q'][j]*m['wf'][j])
print('initial_depth',np.min(np.load(src)['h']),np.max(np.load(src)['h']))
