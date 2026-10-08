"""Trace observed first maxima, forcing maxima and refreshed model peaks in UTC."""
from pathlib import Path
import sys,json
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'RAPID/src'))
from rapid_tools.hydrograph_review import base_directory,event_clock
from rapid_tools.analysis import _load_gauge_quality_sheet,_load_station_timeseries_for_observed_lag
from rapid_tools.refreshed_analysis import interpolated_peak
OUT=ROOT/'RAPID/validation/peak_definitions_03_05_20261005'

def main():
 OUT.mkdir(parents=True,exist_ok=True);quality=_load_gauge_quality_sheet();rows=[];ties=[];contrasts=[]
 fig,axes=plt.subplots(2,1,figsize=(12,8),constrained_layout=True)
 for ex,ax in zip(['03','05'],axes):
  r=quality[pd.to_numeric(quality.Example,errors='coerce').eq(int(ex))].iloc[0];start,end=pd.to_datetime(r.Period_start,utc=True),pd.to_datetime(r.Period_end,utc=True);series={};peaks={}
  for side,station in [('upstream',r.Up),('downstream',r.Down)]:
   g=_load_station_timeseries_for_observed_lag(str(station));g=g[g.time.between(start,end)];series[side]=(pd.DatetimeIndex(g.time),g.discharge_cms.to_numpy())
  h=pd.read_csv(base_directory(ex)/'run/outlet_hydrograph.csv');utc,_=event_clock(ex,h);series['forcing']=(pd.DatetimeIndex(utc),h.q_inflow_cms.to_numpy());series['model']=(pd.DatetimeIndex(utc),h.q_outlet_cms.to_numpy())
  for name,(t,q) in series.items():
   i=int(np.argmax(q));maxima=t[q==q[i]];peaks[name]=maxima
   rows.append(dict(example=ex,series=name,peak_cms=q[i],first_maximum_utc=maxima.min(),last_maximum_utc=maxima.max(),n_equal_maxima=len(maxima)))
   ties.extend(dict(example=ex,series=name,time_utc=stamp,peak_cms=q[i]) for stamp in maxima)
  obs=(peaks['downstream'].min()-peaks['upstream'].min()).total_seconds()/3600;model=(peaks['model'].min()-peaks['forcing'].min()).total_seconds()/3600
  refined,status=interpolated_peak(h.time_seconds,h.q_outlet_cms);rt=utc.iloc[0]+pd.Timedelta(seconds=refined-h.time_seconds.iloc[0])
  contrasts.append(dict(example=ex,observed_first_max_lag_h=obs,model_sampled_lag_h=model,model_minus_observed_lag_h=model-obs,relative_lag_error_pct=100*(model/obs-1),direct_outlet_peak_error_h=(peaks['model'].min()-peaks['downstream'].min()).total_seconds()/3600,forcing_minus_observed_up_peak_h=(peaks['forcing'].min()-peaks['upstream'].min()).total_seconds()/3600,observed_tie_lag_min_h=(peaks['downstream'].min()-peaks['upstream'].max()).total_seconds()/3600,observed_tie_lag_max_h=(peaks['downstream'].max()-peaks['upstream'].min()).total_seconds()/3600,model_interpolated_peak_utc=rt,interpolation_status=status))
  left=peaks['upstream'].min()-pd.Timedelta(hours=18 if ex=='03' else 36);right=peaks['downstream'].max()+pd.Timedelta(hours=18 if ex=='03' else 36)
  colors=dict(upstream='#2563eb',downstream='#111827',forcing='#93c5fd',model='#dc2626')
  for name,(t,q) in series.items():
   m=(t>=left)&(t<=right);ax.plot(t[m],q[m],label={'upstream':'Observed upstream','downstream':'Observed downstream','forcing':'Applied forcing','model':'Model outlet'}[name],color=colors[name],lw=1.8,ls=':' if name=='forcing' else '-')
   if name!='forcing':ax.scatter(peaks[name],np.repeat(q.max(),len(peaks[name])),s=18,color=colors[name],zorder=4)
  if ex=='03':ax.axvspan(peaks['downstream'].min(),peaks['downstream'].max(),alpha=.08,color='gray',label='Observed maximum plateau')
  ax.set(title=f'Case {ex}: dots mark all equal maxima; no temporal alignment',ylabel='Discharge (m³/s)',xlabel='UTC');ax.xaxis.set_major_formatter(mdates.DateFormatter('%d %b\n%H:%M'));ax.legend(fontsize=8);ax.grid(alpha=.15)
 pd.DataFrame(rows).to_csv(OUT/'peaks.csv',index=False);pd.DataFrame(ties).to_csv(OUT/'all_equal_maxima.csv',index=False);pd.DataFrame(contrasts).to_csv(OUT/'lag_calculations.csv',index=False);fig.savefig(OUT/'identified_peaks.png',dpi=160)
 print(pd.DataFrame(contrasts).to_string(index=False))
if __name__=='__main__':main()
