"""Provisional, user-directed plateau review; never changes routing or response data."""
from pathlib import Path
import sys,json,re,gzip,base64,hashlib,html
import numpy as np
import pandas as pd
from scipy.signal import find_peaks
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT),str(ROOT/'RAPID/src'),str(Path(__file__).parent)]
from export_calibration_overview import display_frame
OUT=ROOT/'RAPID/outputs/calibration_review'
source=OUT/'15_network_overview.html';original=source.read_text()
payload=json.loads(gzip.decompress(base64.b64decode(re.search(r'<script id="payload"[^>]*>(.*?)</script>',original,re.S)[1])))
rows=[];starts=[];frames={};review={}
def iso(t):return pd.Timestamp(t).isoformat()
for panel in payload['panels']:
 ex=panel['example'];f,provenance=display_frame(ex);frames[ex]=f;q=f.calibrated_cunge_cms.to_numpy();pk=int(np.nanargmax(q));origin=pd.Timestamp(panel['origin'])
 minima,_=find_peaks(-q,prominence=.001*(np.nanmax(q)-np.nanmin(q)),distance=60)
 post=[int(i) for i in minima if i>pk];firstmin=post[0] if post else len(f)-1
 candidates=[]
 def candidate(key,label,col,kind,a=None,b=None):
  if kind=='final':ind=len(f)-1;value=float(f[col].iloc[ind]);lo=hi=f.time_utc.iloc[ind]
  elif kind=='minimum':
   lo=pd.Timestamp(a);hi=pd.Timestamp(b);eligible=[i for i in minima if lo<=f.time_utc.iloc[i]<=hi] if col=='calibrated_cunge_cms' else []
   if eligible:ind=min(eligible,key=lambda i:q[i] if ex=='48' else abs(f.time_utc.iloc[i]-(lo+(hi-lo)/2)))
   else:
    w=f[f.time_utc.between(lo,hi)];ind=int(w[col].idxmin())
   value=float(f[col].iloc[ind])
  else:
   lo=pd.Timestamp(a);hi=pd.Timestamp(b);w=f[f.time_utc.between(lo,hi)];value=float(w[col].median());ind=int(w.index[len(w)//2])
  ind=int(ind)
  win=f[f.time_utc.between(lo,hi)][col]
  candidates.append(dict(key=key,label=label,source=col,kind=kind,raw_cms=value,reference_utc=iso(f.time_utc.iloc[ind]),window_start_utc=iso(lo),window_end_utc=iso(hi),window_min_cms=float(win.min()),window_max_cms=float(win.max()),reference_index=ind))
 # User's explicit start time for 02 is retained; 48 is a nearby model minimum.
 si=0;start_rule='Full-view equilibrium start'
 if ex=='02':
  stamp=pd.Timestamp('2025-07-31T02:00:00Z');startq=float(np.interp(stamp.value,f.time_utc.astype('int64'),q));start_rule='User exact start time; model Q interpolated'
 elif ex=='48':
  target=pd.Timestamp('2025-06-02T05:00:00Z');eligible=[i for i in minima if abs(f.time_utc.iloc[i]-target)<=pd.Timedelta(hours=12) and i<pk];si=min(eligible,key=lambda i:q[i]);stamp=f.time_utc.iloc[si];startq=float(q[si]);start_rule='Deepest model minimum within ±12 h of requested 06-02 05:00; candidate, not exact requested time'
 else:stamp=f.time_utc.iloc[0];startq=float(q[0])
 start=dict(example=ex,start_utc=iso(stamp),start_cms=startq,rule=start_rule,x=(stamp-origin).total_seconds()/86400)
 starts.append(start)
 default='final';raise_pct=0;note='Final held-forcing value. A crossing is not proof of permanent settlement.'
 if ex in ['02','15','45','48','56']:
  windows={'02':('2025-08-03T00:00Z','2025-08-03T20:00Z'),'15':('2026-04-12T16:00Z','2026-04-13T16:00Z'),'45':('2024-06-26T13:00Z','2024-06-27T13:00Z'),'48':('2025-06-02T19:00Z','2025-06-03T19:00Z'),'56':('2025-07-14T00:00Z','2025-07-15T00:00Z')}
  a,b=windows[ex];candidate('model_min','Nearby model minimum','calibrated_cunge_cms','minimum',a,b)
  candidate('input_min','Input minimum in requested neighbourhood','forcing_cms','minimum',a,b)
  default='input_min' if ex=='15' else 'model_min';raise_pct=1 if ex=='15' else 0
  note='Local minimum candidate; not evidence of a stable plateau. Input and model alternatives retained.'
 elif ex=='13':
  candidate('input_dip','First input dip','forcing_cms','minimum','2024-10-06T18:00Z','2024-10-08T00:00Z')
  candidate('model_min','Post-event model minimum','calibrated_cunge_cms','minimum','2024-10-08T00:00Z','2024-10-11T00:00Z')
  candidate('final','Subsequent higher held plateau','forcing_cms','final');default='model_min'
  note='Alternatives remain unapproved. The higher final level is crossed on recession, then approached again from below after the dip; crossing is not settlement.'
 elif ex in ['46','54','55']:
  windows={'46':('2024-09-02T00:00Z','2024-09-03T00:00Z'),'54':('2023-06-26T00:00Z','2023-06-28T00:00Z'),'55':('2025-04-19T00:00Z','2025-04-20T00:00Z')}
  candidate('input_plateau','Median input level in requested date window','forcing_cms','median',*windows[ex]);candidate('final','Final held-forcing plateau','forcing_cms','final');default='input_plateau';raise_pct=1 if ex=='55' else 0
  note='Input window defines a discharge level, not a model end time. Show the later falling-limb crossing.'
  if ex=='46':note+=' Intervening input changes mean this is not proof of isolation of the original pulse.'
  if ex=='55':note+=' Reject any crossing after the first model recession minimum; the subsequent rise cannot close this event.'
 else:
  candidate('final','Final held-forcing plateau','forcing_cms','final');raise_pct=1 if ex=='03' else (0.5 if ex in ['21','22'] else 0)
 if ex in ['21','22']:
  final=float(f.forcing_cms.iloc[-1]);relative=100*(.005*final)/(q[pk]-final)
  note+=f' The +0.5% plateau removes a weak late tail from the selected event window, not from the saved hydrograph. Its increase is {relative:.3f}% of the peak above the final discharge. Saved final residual above the raw level is {q[-1]-final:.6g} m³/s. Tail physicality is not established by this boundary choice.'
 # Each candidate gets all four raw/raised values. Only downward crossings qualify.
 for c in candidates:
  stop=len(f)-1 if c['key']=='final' and ex not in ['13','55'] else firstmin
  if c['key']=='model_min':stop=c['reference_index']
  if ex in ['15','48']:stop=next(v['reference_index'] for v in candidates if v['key']=='model_min')
  c['levels']=[]
  for pct in [0,0.5,1,2]:
   level=c['raw_cms']*(1+pct/100);inds=np.arange(pk+1,stop+1);cross=inds[(q[inds-1]>level)&(q[inds]<=level)]
   crossing=None;cx=None;status='not reached within checked recession / full-tail window'
   if len(cross):
    j=int(cross[0]);fraction=float((q[j-1]-level)/(q[j-1]-q[j]));fraction=float(np.clip(fraction,0,1));time=f.time_utc.iloc[j-1]+pd.Timedelta(seconds=60*fraction);crossing=iso(time);cx=(time-origin).total_seconds()/86400;status='falling-limb crossing' if pct or c['key']!='model_min' else 'touches selected model minimum'
   result=dict(example=ex,candidate=c['key'],label=c['label'],source=c['source'],raw_cms=c['raw_cms'],raise_pct=pct,plateau_cms=level,reference_utc=c['reference_utc'],window_start_utc=c['window_start_utc'],window_end_utc=c['window_end_utc'],window_min_cms=c['window_min_cms'],window_max_cms=c['window_max_cms'],crossing_utc=crossing,crossing_x=cx,status=status,recession_limit_utc=iso(f.time_utc.iloc[stop]),display_default=c['key']==default and pct==raise_pct)
   rows.append(result);c['levels'].append(result)
 review[ex]=dict(selection_status='Tail-sensitivity choice for review' if ex in ['21','22'] else 'Accepted in user review; 05 raw preference verified',start=start,candidates=candidates,default_candidate=default,default_raise_pct=raise_pct,note=note,provenance=provenance)
 panel['review']=review[ex]
 # Keep input/model only: baseline alignment of observations is not a baseline selection.
 panel['series']=[s for s in panel['series'] if s['legendgroup'] in ['curve_0','curve_3']]
 for mode in ['full','event','peak']:
  low=min([panel[mode+'_y'][0],startq]+[c['raw_cms'] for c in candidates]);high=max([panel[mode+'_y'][1],startq]+[c['raw_cms']*1.02 for c in candidates]);pad=(high-low)*.03;panel[mode+'_y']=[low-pad,high+pad]
pd.DataFrame(rows).to_csv(OUT/'plateau_candidates_20261013.csv',index=False);pd.DataFrame(starts).to_csv(OUT/'plateau_starts_20261013.csv',index=False)
pd.DataFrame(rows).query('display_default').merge(pd.DataFrame(starts),on='example').to_csv(OUT/'reviewed_plateau_selections_20261013.csv',index=False)
(OUT/'plateau_review_20261013.json').write_text(json.dumps(dict(status='Reviewed reference choices; 21/22 tail sensitivity remains under review; no event background B(t) assigned',rule='Raised plateau = raw reference discharge × (1 + percentage/100); this is the plateau itself, per user instruction; include 0.5%, 1% and 2% alternatives. Percentages refer to total reference discharge.',crossing_rule='First post-crest downward crossing; bounded candidates stop at their recession minimum, full-tail final references use complete output; no rising-limb crossings',minima_rule='Model local minima: prominence 0.1% of full model range, at least 60 minutes apart; closest within disclosed user neighbourhood, except 48 uses the deepest minimum to avoid crest ripples. Median windows are provisional.',cases=review),indent=2))
# Preserve user criteria but correct the explicitly rejected interpretation.
criteria_path=OUT/'provisional_event_boundaries_20261013.json';criteria=json.loads(criteria_path.read_text());criteria['rules']['tolerance']='Raise the plateau itself by 0.5%, 1% or 2% of its raw discharge, retaining that as the selected plateau. Latest review: raw for 05; +0.5% for 21 and 22 as tail-sensitivity choices.'
for r in criteria['records']:
 if 'settlement_tolerance' in r['end']:r['end']['plateau_raise']='User-approved interpretation: raise plateau value itself by 1% or 2%; candidates checked separately';r['end'].pop('settlement_tolerance',None)
criteria_path.write_text(json.dumps(criteria,indent=2)+'\n')
# Use the existing offline Plotly library, but a dedicated review interface.
library=original.split('<script>',1)[1].split('</script>',1)[0]
packed=base64.b64encode(gzip.compress(json.dumps(payload,separators=(',',':'),allow_nan=False).encode(),compresslevel=6,mtime=0)).decode()
script=r'''
let payload,current='all',mode='full',raise='default';const choices={};
const byId=id=>document.getElementById(id);
function selected(p){const r=p.review;const c=r.candidates.find(c=>c.key===(choices[p.example]||r.default_candidate));return {c,l:c.levels.find(l=>l.raise_pct===(raise==='default'?r.default_raise_pct:Number(raise)))};}
function chooseSource(){const el=byId('reference');el.innerHTML='';el.disabled=current==='all';if(current==='all'){el.add(new Option('Per-network proposed reference','default'));return;}const p=payload.panels.find(p=>p.example===current);p.review.candidates.forEach(c=>el.add(new Option(c.label,c.key)));el.value=choices[current]||p.review.default_candidate;}
const fmt=x=>x===null?'—':Number(x).toFixed(3);
function row(p,c,l){return `<tr><td>${p.example}</td><td>${c.label}</td><td>${fmt(c.raw_cms)}</td><td>${l.raise_pct}%</td><td>${fmt(l.plateau_cms)}</td><td>${l.crossing_utc||'Not reached on selected recession'}</td><td>${l.status}</td></tr>`;}
async function render(){
const all=current==='all',panels=all?payload.panels:payload.panels.filter(p=>p.example===current);let data=[],seen=new Set();
panels.forEach((p,i)=>{const suffix=all&&i>0?String(i+1):'',r=p.review,{c,l}=selected(p);let traces=[...p.series];
traces.push({type:'scatter',mode:'lines',x:p.full_x,y:[r.start.start_cms,r.start.start_cms],name:'Start reference',legendgroup:'start',line:{color:'#15976e',dash:'dash'},hovertemplate:'Start reference: %{y:.3f} m³/s<extra></extra>'});
traces.push({type:'scatter',mode:'lines',x:p.full_x,y:[l.plateau_cms,l.plateau_cms],name:'End plateau',legendgroup:'end',line:{color:'#9b59b6',dash:'dash'},hovertemplate:c.label+' · '+l.raise_pct+'% raise<br>%{y:.3f} m³/s<extra></extra>'});
traces.push({type:'scatter',mode:'markers',x:[r.start.x],y:[r.start.start_cms],name:'Start selection',legendgroup:'start',showlegend:false,marker:{color:'#15976e',size:9},hovertemplate:r.start.start_utc+'<br>'+r.start.rule+'<extra></extra>'});
if(l.crossing_x!==null)traces.push({type:'scatter',mode:'markers',x:[l.crossing_x],y:[l.plateau_cms],name:'Recession crossing',legendgroup:'cross',marker:{color:'#9b59b6',size:10,symbol:'diamond'},hovertemplate:l.crossing_utc+'<br>'+l.status+'<extra></extra>'});
traces.forEach(t=>{let show=t.showlegend!==false&&!seen.has(t.legendgroup);if(show)seen.add(t.legendgroup);data.push({...t,showlegend:show,xaxis:'x'+suffix,yaxis:'y'+suffix});});});
let layout;
if(all){layout=JSON.parse(JSON.stringify(payload.layouts[mode]));layout.shapes=[];layout.annotations.forEach((a,i)=>{a.text='Network '+panels[i].example;});panels.forEach((p,i)=>{const key='yaxis'+(i?i+1:'');layout[key].range=p[mode+'_y'];});}
else {const p=panels[0];layout={height:650,margin:{l:80,r:30,t:80,b:70},title:{text:'Network '+p.example+' · plateau review'},xaxis:{range:p[mode+'_x'],title:{text:'Days from '+p.origin+' (UTC)'}},yaxis:{range:p[mode+'_y'],title:{text:'Discharge (m³/s)'}},legend:{orientation:'h',y:1.12}};}
layout.paper_bgcolor='white';layout.plot_bgcolor='white';layout.dragmode='zoom';layout.hovermode='closest';byId('plot').style.height=layout.height+'px';
await Plotly.react(byId('plot'),data,layout,{responsive:true,scrollZoom:true,displaylogo:false});
byId('context').textContent=all?'Provisional candidates. Select a network to compare reference levels and view all crossing checks.':panels[0].review.note+' Start: '+panels[0].review.start.start_utc+' — '+panels[0].review.start.rule;
byId('values').innerHTML=panels.map(p=>{if(all){const {c,l}=selected(p);return row(p,c,l);}return p.review.candidates.map(c=>c.levels.map(l=>row(p,c,l)).join('')).join('');}).join('');
byId('status').textContent=all?'15 panels · provisional plateau review':'Network '+current+' · provisional plateau review';
}
(async()=>{try{const bytes=Uint8Array.from(atob(byId('payload').textContent.trim()),c=>c.charCodeAt(0));payload=JSON.parse(await new Response(new Blob([bytes]).stream().pipeThrough(new DecompressionStream('gzip'))).text());payload.panels.forEach(p=>byId('panel').add(new Option('Network '+p.example,p.example)));chooseSource();
byId('panel').addEventListener('change',e=>{current=e.target.value;chooseSource();render();});byId('view').addEventListener('change',e=>{mode=e.target.value;render();});byId('reference').addEventListener('change',e=>{choices[current]=e.target.value;render();});byId('raise').addEventListener('change',e=>{raise=e.target.value;render();});byId('reset').addEventListener('click',render);await render();}catch(e){byId('status').textContent='Could not load: '+e.message;}})();
'''
document='''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>15-network plateau review</title><style>body{margin:0;font:14px Arial;color:#202a34}header,section{padding:16px 26px}h1{font-size:24px}p{max-width:1400px;line-height:1.5}.controls{position:sticky;top:0;background:#f4f6f9;padding:12px 26px;z-index:10;display:flex;gap:12px;flex-wrap:wrap;align-items:center}select,button{font:inherit;padding:7px}#context{padding:0 26px}table{border-collapse:collapse;width:100%;font-size:12px}th,td{padding:8px;border-bottom:1px solid #d9dfe5;text-align:left}th{background:#f4f6f9}.table{overflow:auto}</style><script>LIBRARY</script></head><body><header><h1>15-network plateau review</h1><p>Blue: applied input. Red: current base Cunge. Green: start reference. Purple: proposed end plateau; diamond: the first qualifying model recession crossing. All times are UTC and values are m³/s. This separate review copy records the accepted references and the latest tail-sensitivity choices; no event-volume background B(t) has been assigned.</p><p><b>Raising the plateau means raising its discharge value itself:</b> raw level × (1 + raise/100). Defaults retain the accepted choices: raw for 05; +0.5% for 21 and 22 as tail-sensitivity choices; +1% for 03, 15 and 55. Other cases retain their reviewed raw candidate. Accepted reference levels do not define a full background curve B(t). A crossing does not imply permanent settlement. Held-forcing tails are model extensions, not new observations.</p><p>For 13, compare the input dip, model minimum and higher subsequent plateau. For 46, compare the September 2 input level with the final level: the input reference date is not the model end date. For 55, crossings after the first recession minimum are rejected. Nearby extrema and date-window medians remain provisional; see the table and selected-network context.</p></header><div class="controls"><label>Show <select id="panel"><option value="all">All 15 networks</option></select></label><label>View <select id="view"><option value="full" selected>Full view</option><option value="event">Event</option><option value="peak">Peak</option></select></label><label>End reference <select id="reference"></select></label><label>Plateau raise <select id="raise"><option value="default">Proposed per network</option><option value="0">Raw (0%)</option><option value="0.5">+0.5%</option><option value="1">+1%</option><option value="2">+2%</option></select></label><button id="reset">Reset view</button><span id="status">Loading…</span></div><p id="context"></p><div id="plot"></div><section><h2>Checked candidate values</h2><p>All networks shows the displayed choice for each panel. Select a network to see every candidate and all four plateau values. Bounded-event candidates stop at their recession minimum; full-tail final references search the complete output, including secondary shoulders. Only downward crossings qualify.</p><div class="table"><table><thead><tr><th>Network</th><th>End reference</th><th>Raw Q</th><th>Raise</th><th>Plateau Q</th><th>Model crossing UTC</th><th>Check</th></tr></thead><tbody id="values"></tbody></table></div></section><script id="payload" type="application/octet-stream">PAYLOAD</script><script>SCRIPT</script></body></html>'''
path=OUT/'15_network_plateau_review.html';path.write_text(document.replace('LIBRARY',library).replace('PAYLOAD',packed).replace('SCRIPT',script))
# Scientific and structural checks independent of the HTML controls.
assert len(payload['panels'])==15
for row in rows:
 assert np.isclose(row['plateau_cms'],row['raw_cms']*(1+row['raise_pct']/100))
 if row['crossing_utc']:
  f=frames[row['example']];t=pd.Timestamp(row['crossing_utc']);q=f.calibrated_cunge_cms.to_numpy();pk=int(np.nanargmax(q))
  assert f.time_utc.iloc[pk]<t<=pd.Timestamp(row['recession_limit_utc'])
  assert np.isclose(np.interp(t.value,f.time_utc.astype('int64'),q),row['plateau_cms'],atol=1e-6)
manifest=dict(status='provisional',source_html=str(source),source_html_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),html_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),networks=15,candidate_rows=len(rows),crossing_and_percentage_checks='passed',source_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
(OUT/'plateau_review_manifest_20261013.json').write_text(json.dumps(manifest,indent=2))
print(pd.DataFrame(rows).query('display_default')[['example','candidate','plateau_cms','crossing_utc','status']].to_string(index=False));print(path)
