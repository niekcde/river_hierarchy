import importlib.util,json
from pathlib import Path
import numpy as np,pandas as pd
root=Path(__file__).resolve().parents[2];spec=importlib.util.spec_from_file_location('pilot',root/'synthetic_experiments/investigations/cunge_pilot.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
s=json.loads(m.CONFIG.read_text()); a=json.loads((m.OUT/'narrow_long/audit.json').read_text()); frames,_,_=m.prepare(a['parameters'],s)
out=root/'synthetic_experiments/outputs/narrow_long_timing_diagnostic_20261026'
f=frames['simple/original']; child=frames['simple/full']; rows=[]
for state,g in [('parent',f),('child',child)]:
 g=g.copy();g['travel_minutes']=g.link_length_m/g.rapid_celerity_mps/60;g.to_csv(out/f'{state}_links.csv',index=False)
def stats(t,q):
 e=q-100; vol=np.trapz(e,t);return {'centroid_hours':float(np.trapz(t*e,t)/vol/3600),'peak_hours':float(t[np.argmax(q)]/3600),'volume_m3':float(vol),'peak_cms':float(q.max())}
for tail in [768,1536,3072]:
 t,forcing,_=m.forcing(s,tail); waves={}
 for state,g in [('parent',f),('child',child)]:
  wave,audit=m.cunge.route(g,forcing,4,15);waves[state]=wave;rows.append(dict(tail_hours=tail,state=state,**stats(t,wave)))
 if tail==3072:
  components={}
  for label,width in [('broad',475),('narrow',25)]:
   g=f[(f.width==500)|(f.width==width)].copy();g['split_fraction']=1.;g['reference_flow_fraction']=1.
   q,_=m.cunge.route(g,forcing,4,15);components[label]=q
   rows.append(dict(tail_hours=tail,state=label+'_path_normalized',**stats(t,q)))
  np.testing.assert_allclose(.95*components['broad']+.05*components['narrow'],waves['parent'],atol=1e-8,rtol=1e-10)
  keep=t<8*86400
  pd.DataFrame({'time_seconds':t[keep],'parent':waves['parent'][keep],'child':waves['child'][keep],'broad_normalized':components['broad'][keep],'narrow_normalized':components['narrow'][keep]}).to_csv(out/'peak_components.csv',index=False)
pd.DataFrame(rows).to_csv(out/'timing_checks.csv',index=False);print(pd.DataFrame(rows).to_string(index=False))
expected={state:float(sum(g.reference_flow_fraction*g.link_length_m/g.rapid_celerity_mps)/60) for state,g in [('parent',f),('child',child)]};(out/'infinite_tail_travel_minutes.json').write_text(json.dumps(expected,indent=2));print(expected)
