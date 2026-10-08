"""Current timing, response magnitudes and comparison-cohort inventory."""
from pathlib import Path
import sys,json
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'RAPID/src'))
from rapid_tools.hydrograph_review import base_directory,event_clock
from rapid_tools.refreshed_analysis import interpolated_peak
SOURCE=ROOT/'RAPID/validation/all_pairs_lag_20261005'
OUT=ROOT/'RAPID/validation/all_pairs_metric_inventory_20261005';OUT.mkdir(parents=True,exist_ok=True)
d=pd.read_parquet(SOURCE/'features.parquet');c=d.copy()
cohort=pd.concat([d.groupby('example_id').size().rename('plotted'),c.groupby('example_id').size().rename('path_geometry')],axis=1);cohort['excluded']=cohort.plotted-cohort.path_geometry;cohort.to_csv(OUT/'cohorts.csv')
gauge=pd.read_csv(ROOT/'RAPID/validation/hydrograph_match_review_20261005/summary.csv',dtype={'example':str}).set_index('example');rows=[];excess={}
for ex in sorted(d.example_id.unique()):
 run=base_directory(ex)/'run';h=pd.read_csv(run/'outlet_hydrograph.csv');m=json.loads((run/'hydrograph_metrics.json').read_text());peak,status=interpolated_peak(h.time_seconds,h.q_outlet_cms);assert status=='ok'
 lag=(peak-m['inflow_peak_time_seconds'])/3600;g=gauge.loc[ex]
 excess[ex]=h.q_outlet_cms.max()-max(h.q_outlet_cms.iloc[0],h.q_outlet_cms.iloc[-1])
 clock,_=event_clock(ex,h);model_utc=clock.iloc[0]+pd.to_timedelta(peak-h.time_seconds.iloc[0],unit='s')
 up=pd.to_datetime(g.upstream_peak_first_utc,utc=True);down=pd.to_datetime(g.downstream_peak_first_utc,utc=True)
 upmid=up+(pd.to_datetime(g.upstream_peak_last_utc,utc=True)-up)/2
 downmid=down+(pd.to_datetime(g.downstream_peak_last_utc,utc=True)-down)/2
 observed=(down-up).total_seconds()/3600
 rows.append(dict(example=ex,observed_first_peak_lag_h=observed,model_interpolated_lag_h=lag,lag_difference_h=lag-observed,model_downstream_peak_error_h=(model_utc-down).total_seconds()/3600,observed_span_midpoint_lag_h=(downmid-upmid).total_seconds()/3600,model_downstream_span_midpoint_error_h=(model_utc-downmid).total_seconds()/3600,observed_up_tie_span_h=g.up_equal_max_span_h,observed_down_tie_span_h=g.down_equal_max_span_h,model_interpolated_peak_utc=model_utc,observed_up_peak_first_utc=up,observed_down_peak_first_utc=down))
lag=pd.DataFrame(rows);lag.to_csv(OUT/'network_lags.csv',index=False)
d['duration_change_minutes']=(d.child_event_duration_hours-d.parent_event_duration_hours)*60
d['peak_excess_change_cms']=d.peak_excess_change_pct/100*d.example_id.map(excess)
d['timing_change_minutes']=d.interpolated_peak_shift_hours*60
metrics=['interpolated_lag_shift_pct','timing_change_minutes','event_duration_change_pct','peak_excess_change_pct','duration_change_minutes','peak_excess_change_cms'];summaries=[]
for ex,g in [('all',d),*list(d.groupby('example_id'))]:
 for metric in metrics:
  z=g[metric];a=z.abs();summaries.append(dict(example=ex,metric=metric,n=len(z),signed_min=z.min(),signed_median=z.median(),signed_max=z.max(),median_absolute=a.median(),p90_absolute=a.quantile(.9),p95_absolute=a.quantile(.95),p99_absolute=a.quantile(.99),max_absolute=a.max(),fraction_absolute_below_0p1=a.lt(.1).mean(),fraction_absolute_below_1=a.lt(1).mean()))
s=pd.DataFrame(summaries);s.to_csv(OUT/'response_magnitudes.csv',index=False)
d[['example_id','mode','state_id','plot_category']+metrics].to_csv(OUT/'comparison_responses.csv',index=False)
print(lag[['example','observed_first_peak_lag_h','model_interpolated_lag_h','lag_difference_h']].round(2).to_string(index=False));print(s[s.example.eq('all')].round(6).to_string(index=False))
