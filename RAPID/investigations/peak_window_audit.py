"""All-case peak-window diagnostics; no rerouting or primary-metric replacement."""
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
from rapid_tools.refreshed_analysis import DEFAULT_REFRESH_ROOT
OUT=ROOT/'RAPID/validation/peak_windows_20261005'

def main():
 OUT.mkdir(parents=True,exist_ok=True);quality=_load_gauge_quality_sheet();summaries=[];windows=[];ties=[];notes=[];manual=[]
 examples=sorted(p.name for p in DEFAULT_REFRESH_ROOT.iterdir() if (p/'inventory.csv').exists())
 for ex in examples:
  qr=quality[pd.to_numeric(quality.Example,errors='coerce').eq(int(ex))].iloc[0];start,end=pd.to_datetime(qr.Period_start,utc=True),pd.to_datetime(qr.Period_end,utc=True);series={};peaks={};stats={}
  for name,station in [('upstream',qr.Up),('downstream',qr.Down)]:
   g=_load_station_timeseries_for_observed_lag(str(station));g=g[g.time.between(start,end)];series[name]=(pd.DatetimeIndex(g.time),g.discharge_cms.to_numpy())
  h=pd.read_csv(base_directory(ex)/'run/outlet_hydrograph.csv');utc,_=event_clock(ex,h)
  series['forcing']=(pd.DatetimeIndex(utc),h.q_inflow_cms.to_numpy());series['model']=(pd.DatetimeIndex(utc),h.q_outlet_cms.to_numpy())
  for name,(t,q) in series.items():
   equal=t[q==q.max()];peaks[name]=(equal.min(),equal.max());step=np.median(np.diff(t.asi8))/1e9;blocks=1+int(np.sum(np.diff(equal.asi8)/1e9>1.5*step))
   stats[name]=dict(count=len(equal),blocks=blocks,span_h=(equal.max()-equal.min()).total_seconds()/3600)
   ties.extend(dict(example=ex,series=name,time_utc=a,peak_cms=q.max()) for a in equal)
   # Descriptive absolute-Q thresholds; not event-duration thresholds or confidence bounds.
   for fraction in [1.,.995,.99]:
    ix=np.flatnonzero(q>=fraction*q.max());breaks=np.flatnonzero((np.diff(ix)>1)|(np.diff(t[ix].asi8)/1e9>1.5*step))+1
    for k,block in enumerate(np.split(ix,breaks),1):
     windows.append(dict(example=ex,series=name,fraction_of_max=fraction,block=k,start_utc=t[block[0]],end_utc=t[block[-1]],samples=len(block)))
  u0,u1=peaks['upstream'];d0,d1=peaks['downstream'];i0,i1=peaks['forcing'];mp=peaks['model'][0]
  hours=lambda delta:delta.total_seconds()/3600
  obs=hours(d0-u0);lo=hours(d0-u1);hi=hours(d1-u0);midobs=hours((d0+(d1-d0)/2)-(u0+(u1-u0)/2));model=hours(mp-i0);lastmodel=hours(mp-i1)
  row=dict(example=ex,observed_first_lag_h=obs,observed_tie_span_min_lag_h=lo,observed_tie_span_max_lag_h=hi,observed_span_midpoint_lag_h=midobs,model_first_forcing_lag_h=model,direct_error_first_h=hours(mp-d0),direct_error_downstream_span_midpoint_h=hours(mp-(d0+(d1-d0)/2)),model_inside_downstream_max_span=bool(d0<=mp<=d1),up_equal_max_span_h=stats['upstream']['span_h'],down_equal_max_span_h=stats['downstream']['span_h'],up_max_blocks=stats['upstream']['blocks'],down_max_blocks=stats['downstream']['blocks'],forcing_max_span_h=stats['forcing']['span_h'],relative_parent_child_metric_scale_min=1.,relative_parent_child_metric_scale_max=model/lastmodel if lastmodel>0 else np.nan,gauge_speed_scale_min=obs/hi if hi>0 else np.nan,gauge_speed_scale_max=obs/lo if lo>0 else np.nan,quality_reason=str(qr.get('Reason','')))
  summaries.append(row)
  fig,ax=plt.subplots(figsize=(11,4.5),constrained_layout=True);colors=dict(upstream='#2563eb',downstream='#111827',forcing='#93c5fd',model='#dc2626')
  margin=pd.Timedelta(hours=max(12,abs(obs)*.3));left=min(u0,i0,d0,mp)-margin;right=max(u1,d1,mp)+margin
  for name,(t,q) in series.items():
   mask=(t>=left)&(t<=right);ax.plot(t[mask],q[mask],label=name,color=colors[name],ls=':' if name=='forcing' else '-')
   equal=t[q==q.max()];ax.scatter(equal,np.full(len(equal),q.max()),s=15,color=colors[name])
  if d1>d0:ax.axvspan(d0,d1,color='gray',alpha=.10,label='Downstream equal-max span (may contain gaps)')
  if ex=='05':
   a=pd.Timestamp('2024-01-07T19:00:00Z');b=pd.Timestamp('2024-01-08T09:00:00Z');t,q=series['downstream'];sel=(t>=a)&(t<=b);center=a+(b-a)/2
   manual.append(dict(example=ex,start_utc=a,end_utc=b,midpoint_utc=center,minimum_q=float(q[sel].min()),maximum_q=float(q[sel].max()),range_pct_of_event_max=float(100*np.ptp(q[sel])/q.max()),model_error_to_midpoint_h=hours(mp-center),model_inside_window=bool(a<=mp<=b),status='User-proposed first shoulder/window; not adopted as replacement peak or attributed to a tributary'))
   ax.axvspan(a,b,color='#16a34a',alpha=.12,label='User-proposed earlier window (exploratory)')
  ax.set(title=f'{ex}: recorded peak windows; dots = equal maxima; no time alignment',ylabel='Discharge (m³/s)',xlabel='UTC');ax.xaxis.set_major_formatter(mdates.DateFormatter('%d %b\n%H:%M'));ax.legend(fontsize=8);ax.grid(alpha=.15);fig.savefig(OUT/f'{ex}_peak_windows.png',dpi=150);plt.close(fig)
  print(ex,'complete',flush=True)
 pd.DataFrame(summaries).to_csv(OUT/'summary.csv',index=False);pd.DataFrame(windows).to_csv(OUT/'near_peak_windows.csv',index=False);pd.DataFrame(ties).to_csv(OUT/'equal_maxima.csv',index=False);pd.DataFrame(manual).to_csv(OUT/'exploratory_windows.csv',index=False)
 (OUT/'provenance.json').write_text(json.dumps(dict(date='2026-10-05',examples=examples,source=str(DEFAULT_REFRESH_ROOT),primary_metrics_changed=False,routing_changed=False,threshold_note='99 and 99.5 percent of absolute maximum discharge are descriptive sensitivity windows, not accepted peak times or confidence intervals. Keep disconnected blocks separate.',midpoint_note='Span midpoint is a diagnostic only; for separated equal maxima it can fall below the maximum.',speed_note='At fixed model r and gauge distance, the gauge-scaled speed-change magnitude is proportional to 1/observed lag. Model-only speed is unaffected by observed peak selection.'),indent=2))
if __name__=='__main__':main()
