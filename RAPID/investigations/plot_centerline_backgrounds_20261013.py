"""Offline 15-panel base/centerline/gauge/background comparison; no model changes."""
from pathlib import Path
import base64,gzip,hashlib,json,re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from plotly.offline import get_plotlyjs
from export_calibration_overview import ROOT,display_frame

OUT=ROOT/'RAPID/outputs/centerline_background_review_20261013'
OUT.mkdir(exist_ok=True)
REVIEW=ROOT/'RAPID/outputs/final_straight_background_review_20261013'

def clean(q):return [float(v) if np.isfinite(v) else None for v in q]

def main():
 old=(ROOT/'RAPID/outputs/calibration_review/15_network_overview.html').read_text()
 previous=json.loads(gzip.decompress(base64.b64decode(re.search(r'<script id="payload"[^>]*>(.*?)</script>',old,re.S)[1])))
 anchors={c['example']:c for c in json.loads((REVIEW/'candidate_definitions.json').read_text())['candidates']}
 extra={c['example']:c for c in json.loads((REVIEW/'single_reach_additional_adjustments.json').read_text())['cases']}
 panels=[];fig,axes=plt.subplots(5,3,figsize=(18,20));sources={}
 for oldp,ax in zip(previous['panels'],axes.flat):
  ex=oldp['example'];c=anchors[ex];e=extra[ex];f,source=display_frame(ex);origin=pd.Timestamp(oldp['origin'])
  center=ROOT/f'network_variants/output/{ex}_indep__anchored/reference_section/rapid/cunge/run/outlet.parquet'
  ref=pd.read_parquet(center);series=[]
  def add(times,q,name,color,dash='solid',key=None,flags=None):
   times=pd.DatetimeIndex(times);x=((times-origin).total_seconds()/86400).to_numpy();q=np.asarray(q,dtype=float)
   qmain=q.copy()
   if flags is not None:qmain[np.asarray(flags,bool)]=np.nan
   tr=dict(x=x.tolist(),y=clean(qmain),name=name,key=key or name,line=dict(color=color,dash=dash,width=1.7),customdata=np.datetime_as_string(times.tz_convert('UTC').tz_localize(None).to_numpy(),unit='s').tolist(),mode='lines',type='scatter',connectgaps=False)
   series.append(tr)
   if flags is not None and np.any(flags):
    near=np.asarray(flags,bool).copy();near[1:]|=flags[:-1];near[:-1]|=flags[1:]
    series.append(dict(tr,y=clean(np.where(near,q,np.nan)),line=dict(color=color,dash='dot',width=1),showlegend=False,opacity=.5))
   np.testing.assert_allclose(np.array([np.nan if v is None else v for v in tr['y']]),qmain,equal_nan=True)
  add(f.time_utc,f.upstream_cms,'Upstream gauge','#2580a5','solid','upstream',f.upstream_gap.to_numpy())
  add(f.time_utc,f.downstream_cms,'Downstream gauge','#343a40','solid','downstream',f.downstream_gap.to_numpy())
  add(f.time_utc,f.calibrated_cunge_cms,'Base model','#ce3d45',key='base')
  add(ref.time_utc,ref.q_outlet_cms,'Single centerline','#e38b15',key='centerline')
  times=pd.to_datetime([c['fixed_start_utc'],c['fixed_end_utc']],utc=True,format='ISO8601')
  add(times,[c['candidate_start_cms'],c['candidate_end_cms']],'Accepted background','#9155b5','dash','accepted')
  add(times,[e['new_start_cms'],e['new_end_cms']],'Proposed background incl. centerline','#168a65','dash','proposed')
  event=((times-origin).total_seconds()/86400).tolist()
  full=[min(float(s['x'][0]) for s in series),max(float(s['x'][-1]) for s in series)]
  p=dict(example=ex,origin=origin.isoformat(),series=series,offset=oldp['normalization']['downstream_offset_cms'],ranges=dict(full=full,event=event,peak=oldp['peak_x']),volume_increase_pct=e['parent_excess_volume_increase_pct'])
  panels.append(p)
  print(ex,flush=True)
  for tr in series:
   y=np.array([np.nan if v is None else v for v in tr['y']]);ax.plot(tr['x'],y,color=tr['line']['color'],ls='--' if tr['line']['dash']=='dash' else ':' if tr['line']['dash']=='dot' else '-',lw=1,label=tr['name'] if tr.get('showlegend',True) else '_nolegend_')
  ax.set(title=f'{ex} · additional base excess volume +{p["volume_increase_pct"]:.2f}%',xlim=event,xlabel=f'Days from {origin:%Y-%m-%d} UTC',ylabel='Discharge (m³/s)');ax.grid(alpha=.2)
  # Match event view scale to visible data, including interpolated background endpoints.
  values=[v for tr in series for x,v in zip(tr['x'],tr['y']) if v is not None and event[0]<=x<=event[1]]
  lo,hi=min(values),max(values);pad=max((hi-lo)*.07,1);ax.set_ylim(lo-pad,hi+pad)
  sources[ex]=dict(**source,centerline=str(center.resolve()),centerline_sha256=hashlib.sha256(center.read_bytes()).hexdigest(),gauge_wave_sha256=hashlib.sha256((ROOT/f'RAPID/outputs/bounded_beta_calibration_20261009/{ex}/wave.parquet').read_bytes()).hexdigest())
 handles,labels=axes.flat[0].get_legend_handles_labels();fig.legend(handles,labels,loc='upper center',ncol=3,bbox_to_anchor=(.5,.978),fontsize=10)
 fig.suptitle('Base and single-centerline hydrographs, gauges and shared backgrounds\nGauges in original discharge units; proposed lines are not adopted',fontsize=17,y=.999)
 fig.tight_layout(rect=(0,0,1,.945));fig.savefig(OUT/'15_network_centerline_backgrounds.png',dpi=140);plt.close(fig)
 packed=base64.b64encode(gzip.compress(json.dumps(panels,separators=(',',':'),allow_nan=False).encode())).decode()
 script=Path(__file__).with_suffix('.js').read_text()
 html='''<!doctype html><meta charset="utf-8"><title>15 networks — centerline background review</title><style>body{font:15px Arial;margin:22px;color:#24303a}p{line-height:1.5;max-width:1300px}.controls{position:sticky;top:0;z-index:5;background:#eef2f5;padding:14px;display:flex;gap:20px;flex-wrap:wrap}select{padding:6px}#plot{width:100%}</style><h1>Base, single centerline, gauges and background levels</h1><p>Red: base model. Orange: fully collapsed single-centerline model. Blue: upstream gauge. Black: downstream gauge. Purple dashed: accepted shared background. Green dashed: proposed shared background that also includes the single centerline; <b>not adopted</b>. Background endpoints mark the selected event boundaries. No discharge is clipped.</p><p>Gauges default to original discharge values, without time or level alignment. The optional downstream alignment adds the same constant offset used in the calibration review; hover shows original and displayed Q. Dotted gauge segments flag long sampling intervals. Gauge series stop at their available support; full view retains both models’ complete saved tails. Event uses the selected response window; Peak uses the calibration crest window. Source clocks for 21/22 remain provisional.</p><div class="controls"><label>Network <select id="network"><option value="all">All 15 networks</option></select></label><label>View <select id="view"><option value="full">Full view</option><option value="event">Event</option><option value="peak">Peak</option></select></label><label>Downstream gauge <select id="gauge"><option value="raw">Original discharge</option><option value="aligned">Baseline aligned (display only)</option></select></label></div><p id="note"></p><div id="plot"></div><p><a href="15_network_centerline_backgrounds.png">Download 15-panel event image</a>. Click legend items to toggle curves. All saved one-minute values are retained in this offline HTML.</p><script>LIBRARY</script><script id="payload" type="application/octet-stream">PAYLOAD</script><script>SCRIPT</script>'''
 path=OUT/'15_network_centerline_backgrounds.html';path.write_text(html.replace('LIBRARY',get_plotlyjs()).replace('PAYLOAD',packed).replace('SCRIPT',script))
 (OUT/'manifest.json').write_text(json.dumps(dict(panels=len(panels),sources=sources,views=['full','event','peak'],raw_gauges_default=True,baselines_adopted=False,html_sha256=hashlib.sha256(path.read_bytes()).hexdigest()),indent=2))
 print(path)
if __name__=='__main__':main()
