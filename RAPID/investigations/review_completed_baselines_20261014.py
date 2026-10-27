"""Two Plotly reviews and cutoff sensitivity; no active metric changes."""
from pathlib import Path
import sys,json,gzip,base64,re
import numpy as np,pandas as pd
from plotly.offline import get_plotlyjs
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT),str(ROOT/'RAPID/src')]
from rapid_tools.hydrograph_response import distribution
OUT=ROOT/'RAPID/outputs/completed_event_review_20261014'
RUN=ROOT/'RAPID/outputs/event_window_runs_20261013'

def main():
 OUT.mkdir(exist_ok=True)
 plan=json.loads((ROOT/'RAPID/outputs/event_window_plan_20261013/plan.json').read_text())
 old=(ROOT/'RAPID/outputs/centerline_background_review_20261013/15_network_centerline_backgrounds.html').read_text()
 previous={p['example']:p for p in json.loads(gzip.decompress(base64.b64decode(re.search(r'<script id="payload"[^>]*>(.*?)</script>',old,re.S)[1])))}
 panels=[];summaries=[];metrics=[]
 for c in plan['cases']:
  ex=c['example'];done=json.loads((RUN/ex/'completion.json').read_text());dest=Path(done['final_directory']);forcing=pd.read_parquet(dest/'forcing.parquet');t=forcing.time_utc.astype('int64').to_numpy()/1e9
  eventids=np.flatnonzero(forcing.phase.eq('input_event'));start=int(eventids[0]);end=int(eventids[-1]);cuts={k:end+1 for k in ['1pct','0.5pct','0.1pct']};waves={};minimum=min(c['initial_cms'],c['final_cms']);minstate='endpoint equilibrium'
  lo=np.full(len(t),np.inf);hi=np.full(len(t),-np.inf)
  for j in c['jobs']:
   f=pd.read_parquet(dest/j['state_id']/'outlet.parquet');assert np.array_equal(f.time_utc.astype('int64').to_numpy()/1e9,t)
   q=f.q_outlet_cms.to_numpy();waves[j['state_id']]=q;lo=np.minimum(lo,q);hi=np.maximum(hi,q)
   if q[start:].min()<minimum:minimum=float(q[start:].min());minstate=j['state_id']
   for k,mult in [('1pct',10),('0.5pct',5),('0.1pct',1)]:
    ids=np.flatnonzero((np.arange(len(q))>end)&(abs(q-c['final_cms'])>mult*c['tolerance_cms']))
    if len(ids):cuts[k]=max(cuts[k],int(ids[-1]+1))
  cuts['full']=len(t)-1;assert cuts['0.1pct']<len(t)-5760+2
  for state,q in waves.items():
   for mode,cut in cuts.items():
    assert start<cut<len(t)
    m=distribution(t[start:cut+1],q[start:cut+1]-minimum)
    metrics.append(dict(example=ex,state=state,window=mode,baseline_cms=minimum,cutoff_utc=pd.Timestamp(t[cut],unit='s',tz='UTC').isoformat(),peak_cms=float(q[start:cut+1].max()),peak_seconds=float(t[start+np.argmax(q[start:cut+1])]),**m))
  origin=pd.Timestamp(previous[ex]['origin']);epoch=origin.timestamp();x=(t-epoch)/86400
  def series(q,name,key,color,dash='solid'):
   return dict(x=x.tolist(),y=np.asarray(q).tolist(),name=name,key=key,type='scatter',mode='lines',line=dict(color=color,width=1.5,dash=dash),customdata=forcing.time_utc.dt.strftime('%Y-%m-%d %H:%M:%S UTC').tolist())
  oldp=previous[ex]
  gauges=[s for s in oldp['series'] if s['key'] in ['upstream','downstream']]
  oldbase=[dict(s,name='Original calibrated parent',key='original') for s in oldp['series'] if s['key']=='base']
  base=series(waves['S000_base'],'Rerun parent','base','#ce3d45')
  center=series(waves['single_centerline'],'Rerun single centerline','centerline','#e38b15')
  applied=series(forcing.forcing_cms,'Applied input (held outside event)','forcing','#2580a5','dash')
  bands=[series(lo,'All-state minimum envelope','lower','#aab5bf'),series(hi,'All-state range','upper','#aab5bf')]
  bands[0]['line']['width']=0;bands[0]['showlegend']=False
  bands[1].update(fill='tonexty',fillcolor='rgba(90,110,130,.16)');bands[1]['line']['width']=0
  baseline=dict(x=[x[start],x[cuts['0.1pct']]],y=[minimum,minimum],customdata=['Shared minimum reference']*2,name='Proposed minimum baseline',key='baseline',type='scatter',mode='lines',line=dict(color='#9155b5',width=2,dash='dash'))
  summary=dict(example=ex,baseline_cms=minimum,minimum_state=minstate,final_equilibrium_cms=c['final_cms'],final_excess_cms=c['final_cms']-minimum,cutoff_utc=pd.Timestamp(t[cuts['0.1pct']],unit='s',tz='UTC').isoformat(),cutoff_days_after_input=(t[cuts['0.1pct']]-t[end])/86400,full_tail_days=done['final_tail_hours']/24,removed_padding_days=(t[-1]-t[cuts['0.1pct']])/86400)
  summaries.append(summary)
  panels.append(dict(example=ex,origin=origin.isoformat(),offset=oldp['offset'],calibration_series=gauges+oldbase+[applied,base,center],baseline_series=bands+gauges+[applied,base,center,baseline],ranges=dict(full=[float(x[0]),float(x[-1])],event=[float(x[start]-.05*(x[cuts['0.1pct']]-x[start])),float(x[cuts['0.1pct']]+.05*(x[cuts['0.1pct']]-x[start]))],peak=oldp['ranges']['peak']),input_bounds=[float(x[start]),float(x[end])],cutoffs={k:float(x[v]) for k,v in cuts.items()},summary=summary))
  print(ex,'baseline',round(minimum,3),'cut',summary['cutoff_days_after_input'],'days',flush=True)
 states=pd.DataFrame(metrics);states.to_parquet(OUT/'cutoff_sensitivity_states.parquet',index=False)
 rows=[]
 for (ex,mode),g in states.groupby(['example','window']):
  p=g[g.state.eq('S000_base')].iloc[0]
  for r in g[g.state.ne('S000_base')].itertuples():
   rows.append(dict(example=ex,state=r.state,window=mode,peak_shift_hours=(r.peak_seconds-p.peak_seconds)/3600,peak_excess_change_pct=100*((r.peak_cms-r.baseline_cms)/(p.peak_cms-p.baseline_cms)-1),centroid_shift_hours=(r.centroid_seconds-p.centroid_seconds)/3600,duration80_change_pct=100*(r.duration80_hours/p.duration80_hours-1),asymmetry_change=r.asymmetry-p.asymmetry,volume_change_pct=100*(r.volume_m3/p.volume_m3-1)))
 pd.DataFrame(rows).to_parquet(OUT/'cutoff_sensitivity_responses.parquet',index=False)
 sensitivity=[]
 for ex,g in states[states.state.eq('S000_base')].groupby('example'):
  a=g[g.window.eq('0.1pct')].iloc[0];b=g[g.window.eq('full')].iloc[0];z=g[g.window.eq('1pct')].iloc[0]
  sensitivity.append(dict(example=ex,parent_volume_full_vs_cut_pct=100*(b.volume_m3/a.volume_m3-1),parent_D80_full_vs_cut_pct=100*(b.duration80_hours/a.duration80_hours-1),parent_centroid_full_vs_cut_hours=(b.centroid_seconds-a.centroid_seconds)/3600,parent_asymmetry_full_minus_cut=b.asymmetry-a.asymmetry,parent_D80_1pct_vs_cut_pct=100*(z.duration80_hours/a.duration80_hours-1)))
 sensitivity=pd.DataFrame(sensitivity);sensitivity.to_csv(OUT/'parent_cutoff_sensitivity.csv',index=False)
 table=pd.DataFrame(summaries);table.to_csv(OUT/'baseline_proposal.csv',index=False)
 packed=base64.b64encode(gzip.compress(json.dumps(panels,separators=(',',':'),allow_nan=False).encode())).decode()
 js=Path(__file__).with_suffix('.js').read_text()
 for mode,title in [('calibration','Calibration review: original and rerun hydrographs'),('baseline','Baseline proposal and common analysis cutoff')]:
  note=('Original calibrated parent and current rerun use the same calibrated parameters. Cropping the forcing and resetting initial equilibrium can change the response; these reruns have not been recalibrated. Downstream alignment is a display-only constant offset.' if mode=='calibration' else 'Review proposal only. Purple baseline = minimum across all routed states and their initial/final equilibrium levels; gauges excluded. Dashed vertical cutoff = last settling time across all states at 0.1% of input range. Initial spin-up, 96-hour confirmation and extra rounded padding are excluded from the proposed analysis. Gray shading spans every model state. The cutoff is operational, not a physical separation of event flow.')
  html='''<!doctype html><meta charset="utf-8"><title>TITLE</title><style>body{font:15px Arial;margin:22px;color:#24303a}p{max-width:1400px;line-height:1.5}.controls{position:sticky;top:0;background:#edf2f5;padding:14px;z-index:5}select{padding:6px;margin-right:20px}td,th{padding:8px;text-align:left}</style><h1>TITLE</h1><p>NOTE</p><div class="controls">Network <select id="network"><option value="all">All 15</option></select>View <select id="view"><option value="event">Event / proposed analysis</option><option value="full">Full simulation</option><option value="peak">Peak</option></select>Downstream gauge <select id="gauge"><option value="raw">Original discharge</option><option value="aligned">Original calibration offset</option></select></div><p id="note"></p><div id="plot"></div>TABLE<script>LIBRARY</script><script id="payload" type="application/octet-stream">PAYLOAD</script><script>const REVIEW_MODE="MODE";SCRIPT</script>'''
  html=html.replace('TITLE',title).replace('NOTE',note).replace('TABLE',table.to_html(index=False,float_format=lambda v:f'{v:.3f}')).replace('"MODE"',json.dumps(mode)).replace('SCRIPT',js).replace('LIBRARY',get_plotlyjs()).replace('PAYLOAD',packed)
  if mode=='calibration':
   html=html.replace('<option value="raw">Original discharge</option>', '<option value="normalized">Peak matched to input</option><option value="raw">Original discharge</option>')
   html=html.replace(note, 'Red: Full Complexity; orange: single centerline. Both use the existing calibrated parameters with the cropped event forcing and reset initial equilibrium. Downstream peak matching multiplies observed discharge by input peak / downstream peak across the saved series, without shifting time or subtracting a baseline. Raw discharge and the original constant-offset display remain available.')
  warning='<h2>Cutoff sensitivity — parent hydrographs</h2><p>Full versus cutoff compares the entire saved tail with the proposed 0.1% common settling cutoff, holding the baseline fixed. Positive final excess means even an exactly constant plateau adds volume. The 1% comparison also tests the cutoff choice itself. These are diagnostic results; the baseline and analysis window are not adopted.</p>'+sensitivity.to_html(index=False,float_format=lambda v:f'{v:.3f}')
  if mode=='baseline':html=html.replace('<script>'+get_plotlyjs(),warning+'<script>'+get_plotlyjs(),1)
  (OUT/f'15_{mode}_review.html').write_text(html)
 (OUT/'manifest.json').write_text(json.dumps(dict(status='review_proposal_not_adopted',states=len(states)//4,windows=['1pct','0.5pct','0.1pct','full'],source_plan=str(ROOT/'RAPID/outputs/event_window_plan_20261013/plan.json')),indent=2))
if __name__=='__main__':
 main()
 from update_baseline_locations_20261014 import main as update_locations
 update_locations()
