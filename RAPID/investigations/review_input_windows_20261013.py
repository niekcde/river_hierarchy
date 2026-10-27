"""Review proposed INPUT event windows against saved curves; changes no runs/metrics."""
from pathlib import Path
import base64,gzip,json,re
import pandas as pd
from plotly.offline import get_plotlyjs
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'RAPID/outputs/input_window_review_20261013'
# Nearby meaningful minima where requested; explicit fixed ends retained.
SELECTIONS={
'02':('2025-07-31 02:00:30','2025-08-02 03:30:30','First major pulse; end remains above start and excludes subsequent input rises.'),
'03':('2023-02-14 18:59:30','2023-02-19 02:44:30','Second pre-event minimum; end is a local dip before a rebound, not final recession minimum.'),
'05':('2024-01-02 12:29:30','2024-01-12 19:07:30','Prominent pre-event and Jan 12 post-event minima; tiny intervening ripples ignored.'),
'13':('2024-10-02 00:29:30','2024-10-07 08:59:30','Last tied pre-rise minimum; end before renewed rise.'),
'15':('2026-04-01 11:42:30','2026-04-12 22:57:30','Representative points on minimum plateaus; exact seconds are not unique physical boundaries.'),
'21':(None,None,'Retain full observation-supported input, including secondary rise. Source clock remains provisional.'),
'22':('2025-08-17 02:34:30','2025-08-22 13:00:30','Requested end retained: not a distinct minimum. Earlier dip Aug 22 06:14:30; recession continues. Source clock provisional.'),
'37':('2014-06-25 18:30:30','2014-06-30 15:29:30','Nearby input minima.'),
'41':('2024-07-05 16:59:30','2024-07-27 20:14:30','First nearby tied minimum; final substantial dip before renewed rise.'),
'45':('2024-06-13 16:49:30','2024-06-24 13:46:30','Endpoint moved to nearby Q=1066.032 per user review; recession continues afterwards.'),
'46':('2024-08-24 21:35:30','2024-09-03 00:00:30','Requested plateau end retained. Current parent peak follows this input endpoint; output must extend.'),
'48':('2025-06-02 03:29:30','2025-06-03 06:00:30','Small June 2 peak; nearby postpeak minimum approximately one hour before requested 07:00.'),
'54':(None,'2023-06-26 00:00:30','Original supported input start; requested end retained. Nearby lower dip Jun 26 03:00:30.'),
'55':(None,'2025-04-19 00:00:30','Original supported input start; requested end retained. Nearby dip Apr 18 22:54:30.'),
'56':('2025-06-24 06:09:30','2025-07-13 00:00:30','Requested end retained on continuing recession. Main later minimum Jul 14 02:54:30.')}
def main():
 OUT.mkdir(exist_ok=True)
 source=ROOT/'RAPID/outputs/centerline_background_review_20261013/15_network_centerline_backgrounds.html'
 panels=json.loads(gzip.decompress(base64.b64decode(re.search(r'<script id="payload"[^>]*>(.*?)</script>',source.read_text(),re.S)[1])))
 records=[]
 for p in panels:
  ex=p['example'];start,end,note=SELECTIONS[ex]
  reg=pd.read_csv(ROOT/f'network_variants/output/{ex}_indep__anchored/rapid_run_registry.csv');r=reg[reg.state_id.eq('S000_base')].iloc[0]
  f=pd.read_parquet(r.forcing_parquet);f=f[(~f.forcing_endpoint_hold)&f.upstream_cms.notna()].reset_index(drop=True)
  bounds=[pd.Timestamp(start,tz='UTC') if start else f.time_utc.iloc[0],pd.Timestamp(end,tz='UTC') if end else f.time_utc.iloc[-1]]
  indices=[(f.time_utc-t).abs().idxmin() for t in bounds];pts=f.loc[indices];ts=pts.time_utc.tolist();q=pts.upstream_cms.tolist()
  assert ts[0]<ts[1] and all(abs(a-b)<=pd.Timedelta(minutes=1) for a,b in zip(ts,bounds))
  origin=pd.Timestamp(p['origin']);x=[(t-origin).total_seconds()/86400 for t in ts]
  p['series']=[s for s in p['series'] if s['key'] in ['upstream','downstream','base','centerline']]
  p['series'].append(dict(x=x,y=q,name='Input boundaries',key='boundaries',mode='markers',type='scatter',line=dict(color='#198754'),customdata=[t.isoformat() for t in ts]))
  span=x[1]-x[0];p['ranges']['event']=[x[0]-.08*span,x[1]+.15*span];p['bounds']=x;p['note']=note
  p['ranges']['start']=[x[0]-.5,x[0]+.5];p['ranges']['end']=[x[1]-.5,x[1]+.5]
  records.append(dict(example=ex,start_utc=ts[0].isoformat(),start_cms=q[0],end_utc=ts[1].isoformat(),end_cms=q[1],note=note))
 table=pd.DataFrame(records);table.to_csv(OUT/'proposed_input_windows.csv',index=False)
 (OUT/'proposed_input_windows.json').write_text(json.dumps(dict(status='review_only',time_basis='UTC; saved one-minute interval means, not native measurement precision',windows=records),indent=2))
 script=Path(__file__).with_suffix('.js').read_text()
 packed=base64.b64encode(gzip.compress(json.dumps(panels,separators=(',',':'),allow_nan=False).encode())).decode()
 html='''<!doctype html><meta charset="utf-8"><title>Input event windows</title><style>body{font:15px Arial;margin:22px;color:#24303a}p{max-width:1300px;line-height:1.5}.controls{position:sticky;top:0;background:#edf2f5;padding:14px;z-index:5}select{margin-right:25px;padding:6px}td,th{padding:8px;text-align:left}</style><h1>15 proposed input-event windows — review only</h1><p>Green markers and shading identify the proposed INPUT interval. Red is the saved parent, orange the single centerline, blue the upstream gauge, black the downstream gauge. Output response windows must extend beyond input endpoints as needed. Existing model curves have NOT been rerun with cropped forcing: later input events can affect their tails. No baseline or response values have been changed.</p><p>All times UTC. One-minute interval means do not imply minute-resolution source measurements. Nearby minima are review candidates; fixed endpoints are distinguished in the table. Gauges are unshifted. Source clocks for 21/22 remain provisional.</p><div class="controls">Network <select id="network"><option value="all">All 15</option></select>View <select id="view"><option value="event">Proposed event + context</option><option value="full">Full view</option><option value="peak">Peak</option><option value="start">Start ±12 hours</option><option value="end">End ±12 hours</option></select></div><p id="note"></p><div id="plot"></div>TABLE<script>LIBRARY</script><script id="payload" type="application/octet-stream">PAYLOAD</script><script>SCRIPT</script>'''
 (OUT/'15_input_event_windows.html').write_text(html.replace('TABLE',table.to_html(index=False,float_format=lambda x:f'{x:.3f}')).replace('LIBRARY',get_plotlyjs()).replace('PAYLOAD',packed).replace('SCRIPT',script))
 print(table.to_string(index=False));print(OUT/'15_input_event_windows.html')
if __name__=='__main__':main()
