"""Example-37 total variation diagnostic; no forcing/metric changes."""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'RAPID/src'))
from rapid_tools.hydrograph_review import base_directory,event_clock
OUT=ROOT/'RAPID/validation/hydrograph_mean_spread_20261005'
f=pd.read_parquet(ROOT/'RAPID/validation/all_pairs_lag_20261005/features.parquet')
s=f[f.example_id.eq('37')&f.state_id.isin(['CB001_bubble_3','S274_unit_280','island_37_272'])]
h=pd.read_csv(base_directory('37')/'run/outlet_hydrograph.csv');clock,_=event_clock('37',h);t=h.time_seconds.to_numpy();dt=np.diff(t)
np.testing.assert_allclose(dt,dt[0],atol=1e-8,rtol=0)
series={'forcing':h.q_inflow_cms.to_numpy(),'parent':h.q_outlet_cms.to_numpy()}
for _,r in s.iterrows():
 c=pd.read_csv(Path(r.refreshed_run_dir)/'outlet_hydrograph.csv');np.testing.assert_allclose(c.time_seconds,t);np.testing.assert_allclose(c.q_inflow_cms,h.q_inflow_cms);series[r.state_id]=c.q_outlet_cms.to_numpy()
g=pd.read_csv(OUT/'gauge_means.csv',dtype={'example':str}).set_index('example').loc['37'];lo,hi=pd.to_datetime(g.start_utc),pd.to_datetime(g.end_utc)
rows=[]
for window,mask in [('full_routed_event',np.ones(len(t),bool)),('shared_gauge_window',clock.between(lo,hi).to_numpy())]:
 for stride in [1,2,4,8]:
  ids=np.flatnonzero(mask)[::stride];p=series['parent'][ids];Aparent=p.max()-max(p[0],p[-1]);assert Aparent>0
  for state,q in series.items():
   z=q[ids];A=z.max()-max(z[0],z[-1]);tv=np.abs(np.diff(z)).sum();minimum=2*z.max()-z[0]-z[-1]
   rows.append(dict(window=window,stride=stride,sample_minutes=dt[0]*stride/60,state=state,n=len(z),peak_cms=z.max(),reference_cms=max(z[0],z[-1]),amplitude_cms=A,tv_cms=tv,tv_over_parent_amplitude=tv/Aparent,tv_over_own_amplitude=tv/A,monotone_rise_fall_tv_cms=minimum,excess_variation_cms=tv-minimum,excess_over_parent_amplitude=(tv-minimum)/Aparent))
d=pd.DataFrame(rows)
for (win,stride),z in d.groupby(['window','stride']):
 b=z[z.state.eq('parent')].iloc[0]
 for col in ['tv_cms','tv_over_own_amplitude','excess_variation_cms']:
  d.loc[z.index,col+'_change_vs_parent_pct']=100*(z[col]/b[col]-1)
d.to_csv(OUT/'example37_total_variation.csv',index=False)
print(d[d.stride.eq(1)][['window','state','sample_minutes','amplitude_cms','tv_cms','tv_cms_change_vs_parent_pct','tv_over_own_amplitude','excess_variation_cms','excess_over_parent_amplitude']].round(4).to_string(index=False))
print(d[d.state.eq('CB001_bubble_3')][['window','sample_minutes','tv_cms_change_vs_parent_pct','tv_over_own_amplitude_change_vs_parent_pct']].round(3).to_string(index=False))
assert np.isfinite(d.select_dtypes('number')).all().all()
assert (d.excess_variation_cms>=-1e-8).all()
