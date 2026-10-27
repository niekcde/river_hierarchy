"""Update the existing baseline HTML with 0.5/1/2% shared response windows."""
from pathlib import Path
import json,re,gzip,base64
import numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'RAPID/outputs/completed_event_review_20261014'

def main():
 path=OUT/'15_baseline_review.html';html=path.read_text()
 panels=json.loads(gzip.decompress(base64.b64decode(re.search(r'<script id="payload"[^>]*>(.*?)</script>',html,re.S)[1])))
 plan=json.loads((ROOT/'RAPID/outputs/event_window_plan_20261013/plan.json').read_text());rows=[]
 for c,p in zip(plan['cases'],panels):
  ex=c['example'];assert ex==p['example'];done=json.loads((ROOT/f'RAPID/outputs/event_window_runs_20261013/{ex}/completion.json').read_text());folder=Path(done['final_directory'])
  forcing=pd.read_parquet(folder/'forcing.parquet');t=forcing.time_utc.astype('int64').to_numpy()/1e9;event=np.flatnonzero(forcing.phase.eq('input_event'));first,last=event[0],event[-1];levels={}
  for label,mult in [('0.5%',5),('1%',10),('2%',20)]:levels[label]=dict(tolerance=c['tolerance_cms']*mult,start=len(t),end=last+1,start_state=None,end_state=None)
  for j in c['jobs']:
   f=pd.read_parquet(folder/j['state_id']/'outlet.parquet');q=f.q_outlet_cms.to_numpy();assert np.array_equal(f.time_utc.astype('int64').to_numpy()/1e9,t)
   for label,v in levels.items():
    departure=abs(q[first:]-c['initial_cms'])>v['tolerance'];cumulative=np.r_[0,np.cumsum(departure)];runs=np.flatnonzero(cumulative[30:]-cumulative[:-30]==30)
    if not len(runs):raise ValueError(f'No sustained onset: {ex}/{j["state_id"]}/{label}')
    onset=first+int(runs[0]);assert np.all(departure[runs[0]:runs[0]+30])
    if onset<v['start']:v.update(start=onset,start_state=j['state_id'])
    outside=np.flatnonzero((np.arange(len(q))>last)&(abs(q-c['final_cms'])>v['tolerance']))
    end=int(outside[-1]+1) if len(outside) else last+1
    assert end<len(q) and np.all(abs(q[end:]-c['final_cms'])<=v['tolerance'])
    if end>=v['end']:v.update(end=end,end_state=j['state_id'])
  origin=pd.Timestamp(p['origin']).timestamp();p['threshold_windows']={}
  for label,v in levels.items():
   a,b=v['start'],v['end'];assert first<=a<b<len(t)
   row=dict(example=ex,threshold=label,tolerance_cms=v['tolerance'],start_utc=forcing.time_utc.iloc[a].isoformat(),end_utc=forcing.time_utc.iloc[b].isoformat(),start_hours_after_input_start=(t[a]-t[first])/3600,end_days_after_input_end=(t[b]-t[last])/86400,start_state=v['start_state'],end_state=v['end_state'],onset_sustained_minutes=30)
   rows.append(row);p['threshold_windows'][label]=dict(start=(t[a]-origin)/86400,end=(t[b]-origin)/86400,**row)
  a=min(v['start'] for v in p['threshold_windows'].values());b=max(v['end'] for v in p['threshold_windows'].values());span=b-a
  p['ranges']['event']=[a-.06*span,b+.06*span]
  starts=[v['start'] for v in p['threshold_windows'].values()];ends=[v['end'] for v in p['threshold_windows'].values()]
  p['ranges']['start']=[min(starts)-.25,max(starts)+.5];p['ranges']['end']=[min(ends)-max(.5,.15*(max(ends)-min(ends))),max(ends)+max(.5,.15*(max(ends)-min(ends)))]
  for s in p['baseline_series']:
   if s['key']=='baseline':s['x']=[a,b]
  print(ex,{k:round(v['end_days_after_input_end'],3) for k,v in p['threshold_windows'].items()},flush=True)
 pd.DataFrame(rows).to_csv(OUT/'threshold_window_locations.csv',index=False)
 packed=base64.b64encode(gzip.compress(json.dumps(panels,separators=(',',':'),allow_nan=False).encode())).decode()
 html=re.sub(r'(<script id="payload"[^>]*>).*?(</script>)',lambda m:m[1]+packed+m[2],html,flags=re.S)
 js=(ROOT/'RAPID/investigations/review_completed_baselines_20261014.js').read_text()
 html=re.sub(r'<script>const REVIEW_MODE="baseline";.*?</script>',lambda m:'<script>const REVIEW_MODE="baseline";'+js+'</script>',html,flags=re.S)
 html=html.replace('Dashed vertical cutoff = last settling time across all states at 0.1% of input range.','Colored markers show three candidate shared windows: 0.5% (blue), 1% (green), 2% (magenta). Dotted colored lines mark onset; dashed colored lines mark the end. Each threshold is a percentage of the same input discharge range. Onset is the earliest 30-minute sustained absolute departure from the initial plateau across all states. End is after the last departure from the final plateau, checking all remaining saved output. The 30-minute persistence is a review convention, not a finalized definition.')
 if '<option value="start">' not in html:
  html=html.replace('<option value="peak">Peak</option>','<option value="peak">Peak</option><option value="start">Onset close-up</option><option value="end">End close-up</option>')
 table='<h2>Candidate onset and end locations</h2>'+pd.DataFrame(rows).to_html(index=False,float_format=lambda v:f'{v:.4f}')+'<p>The older sensitivity table below still compares the input-start / 0.1% cutoff calculation with full saved output. It has not been recalculated using these candidate onset windows.</p>'
 html=re.sub(r'<h2>Candidate onset and end locations</h2>.*?(?=<h2>Cutoff sensitivity — parent hydrographs</h2>)','',html,flags=re.S)
 html=html.replace('<h2>Cutoff sensitivity — parent hydrographs</h2>',table+'<h2>Cutoff sensitivity — parent hydrographs</h2>')
 path.write_text(html)
if __name__=='__main__':main()
