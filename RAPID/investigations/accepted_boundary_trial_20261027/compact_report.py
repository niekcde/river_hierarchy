"""Simplify display polylines only; leave candidates and audit metrics intact."""
from pathlib import Path
import json
from shapely.geometry import LineString
ROOT=Path(__file__).resolve().parents[3];p=ROOT/'RAPID/outputs/accepted_boundary_trial_20261027_v4/review_results.html';s=p.read_text();prefix,data=s.rsplit('const cases=',1);raw,suffix=data.split(';const el=',1);cases=json.loads(raw)
for case in cases:
 for trace in case['figure']['data']:
  xx=[];yy=[];part=[]
  def flush():
   if len(part)>1:
    for x,y in LineString(part).simplify(.005).coords:xx.append(round(x,6));yy.append(round(y,6))
    xx.append(None);yy.append(None)
   part.clear()
  for x,y in zip(trace['x'],trace['y']):
   if x is None or y is None:flush()
   else:part.append((x,y))
  flush();trace['x']=xx;trace['y']=yy
prefix=prefix.replace('Zoom and pan to inspect.','Display lines are simplified at 5 m for responsiveness; exact geometry checks and saved candidates are unchanged. Zoom and pan to inspect.')
p.write_text(prefix+'const cases='+json.dumps(cases,separators=(',',':')).replace('</','<\\/')+';const el='+suffix)
print('Compact report MB',round(p.stat().st_size/1e6,2))
