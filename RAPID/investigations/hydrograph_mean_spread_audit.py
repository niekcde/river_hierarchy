"""Descriptive gauge means and example-37 discharge SD; no routing/Excel edits."""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'RAPID/src'))
from rapid_tools.analysis import _load_gauge_quality_sheet,_load_station_timeseries_for_observed_lag
from rapid_tools.hydrograph_review import base_directory,event_clock
OUT=ROOT/'RAPID/validation/hydrograph_mean_spread_20261005';OUT.mkdir(parents=True,exist_ok=True)
f=pd.read_parquet(ROOT/'RAPID/validation/all_pairs_lag_20261005/features.parquet')
quality=_load_gauge_quality_sheet();rows=[];windows={}
for ex in sorted(f.example_id.unique()):
 r=quality.loc[pd.to_numeric(quality.Example,errors='coerce').eq(int(ex))].iloc[0]
 start,end=pd.to_datetime(r.Period_start,utc=True),pd.to_datetime(r.Period_end,utc=True)
 gs=[]
 for station in [r.Up,r.Down]:
  g=_load_station_timeseries_for_observed_lag(str(station)).copy();g['time']=pd.to_datetime(g.time,utc=True)
  g=g.loc[g.time.between(start,end)&np.isfinite(g.discharge_cms)].sort_values('time').drop_duplicates('time');gs.append(g)
 lo=max(g.time.min() for g in gs);hi=min(g.time.max() for g in gs);assert lo<hi
 windows[ex]=(lo,hi);rec=dict(example=ex,start_utc=lo,end_utc=hi,window_hours=(hi-lo).total_seconds()/3600)
 for side,g in zip(['upstream','downstream'],gs):
  z=g[g.time.between(lo,hi)];t=(z.time-z.time.iloc[0]).dt.total_seconds().to_numpy();q=z.discharge_cms.to_numpy();dt=np.diff(t)
  rec.update({side+'_n':len(q),side+'_mean_cms':q.mean(),side+'_std_cms':q.std(ddof=0),side+'_median_step_minutes':np.median(dt)/60,side+'_max_gap_hours':dt.max()/3600,side+'_time_weighted_mean_cms':np.trapz(q,t)/(t[-1]-t[0])})
 rec['mean_difference_cms']=rec['downstream_mean_cms']-rec['upstream_mean_cms'];rec['mean_ratio_down_up']=rec['downstream_mean_cms']/rec['upstream_mean_cms'];rec['mean_difference_pct']=100*(rec['mean_ratio_down_up']-1)
 rows.append(rec)
a=pd.DataFrame(rows);a.to_csv(OUT/'gauge_means.csv',index=False)
s=f[f.example_id.eq('37')&f.state_id.isin(['CB001_bubble_3','S274_unit_280','island_37_272'])]
parent=pd.read_csv(base_directory('37')/'run/outlet_hydrograph.csv');clock,_=event_clock('37',parent)
series=[('forcing',parent.q_inflow_cms.to_numpy()),('parent',parent.q_outlet_cms.to_numpy())]
for _,r in s.iterrows():
 h=pd.read_csv(Path(r.refreshed_run_dir)/'outlet_hydrograph.csv');np.testing.assert_allclose(h.time_seconds,parent.time_seconds);np.testing.assert_allclose(h.q_inflow_cms,parent.q_inflow_cms);series.append((r.state_id,h.q_outlet_cms.to_numpy()))
lo,hi=windows['37'];records=[]
for window,mask in [('full_routed_event',np.ones(len(parent),dtype=bool)),('shared_gauge_window',clock.between(lo,hi).to_numpy())]:
 base_std=parent.q_outlet_cms.to_numpy()[mask].std(ddof=0)
 for name,q in series:
  z=q[mask];records.append(dict(window=window,state=name,n=len(z),mean_cms=z.mean(),std_cms=z.std(ddof=0),cv=z.std(ddof=0)/z.mean(),std_change_vs_parent_pct=100*(z.std(ddof=0)/base_std-1)))
b=pd.DataFrame(records);b.to_csv(OUT/'example37_spread.csv',index=False)
print(a[['example','upstream_mean_cms','downstream_mean_cms','mean_difference_pct','upstream_max_gap_hours','downstream_max_gap_hours']].round(2).to_string(index=False));print(b.round(4).to_string(index=False))
