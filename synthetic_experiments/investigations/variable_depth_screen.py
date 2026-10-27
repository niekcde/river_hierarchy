"""Exploratory conventional three-point Cunge; not production or mass-corrected.
Compile companion C++ to /tmp/variable_depth_screen.dylib before execution.
"""
import ctypes,importlib.util,json
from pathlib import Path
import numpy as np,pandas as pd
root=Path(__file__).resolve().parents[2];spec=importlib.util.spec_from_file_location('p',root/'synthetic_experiments/investigations/cunge_pilot.py');p=importlib.util.module_from_spec(spec);spec.loader.exec_module(p)
out=root/'synthetic_experiments/outputs/depth_reference_review_20261027';out.mkdir(exist_ok=True)
lib=ctypes.CDLL('/tmp/variable_depth_screen.dylib');func=lib.cell
vec=np.ctypeslib.ndpointer(dtype=np.float64,flags='C_CONTIGUOUS');func.argtypes=[vec,vec,ctypes.c_long]+[ctypes.c_double]*6;func.restype=None
s=json.loads(p.CONFIG.read_text()); rows=[]; checks=[]
def route(g,flow,dt):
 ns=round(60/dt);order,edges,source,sink=p.cunge.topology(g);signals={source:np.repeat(flow,ns)};minq=1e100
 for u in order:
  sig=signals.pop(u)
  if u==sink:return sig.reshape(-1,ns).mean(axis=1),minq
  for i in edges[u]:
   r=g.iloc[i];v=np.ascontiguousarray(sig*r.split_fraction);eq=float(flow[0]*r.reference_flow_fraction)
   for _ in range(int(r['count'])):
    w=np.empty_like(v);func(v,w,len(v),eq,r.width,r.slope,s['n_manning'],r.rapid_effective_length_m,dt);v=w;minq=min(minq,float(v.min()))
   signals[r.v]=signals.get(r.v,0.)+v

def stat(t,w,base,flow):
 e=w-base;vol=np.trapz(e,t);return dict(peak_hours=float(t[w.argmax()]/3600),centroid_hours=float(np.trapz(t*e,t)/vol/3600),volume_error_pct=float(100*(vol/np.trapz(flow-base,t)-1)),minimum_cms=float(w.min()),tail_excess_cms=float(w[-1]-base))
for case in ['baseline','narrow_long','dominant_long']:
 params={**s['base_geometry'],**s['cases'][case]};frames,_,_=p.prepare(params,s)
 for ratio in [2,10]:
  settings={**s,'baseflow_cms':s['peak_cms']/ratio};t,flow,_=p.forcing(settings,384);waves={}
  for state in ['simple/original','simple/full']:
   original=frames[state]
   for method in ['fixed_BASED_peak','fixed_Manning_mid','fixed_Manning_peak','variable_Manning_3point']:
    g=original.copy()
    if method.startswith('fixed_Manning'):
     total=s['peak_cms'] if method.endswith('peak') else .5*(s['peak_cms']+settings['baseflow_cms'])
     g['reference_discharge_cms']=total*g.reference_flow_fraction
     g['depth_m']=(s['n_manning']*g.reference_discharge_cms/(g.width*np.sqrt(g.slope)))**.6
     g['celerity_times_n']=(5/3)*g.depth_m**(2/3)*np.sqrt(g.slope)
     g=p.cunge.parameters(g,s['n_manning'])
    if method.startswith('variable'):
     w,minimum=route(g,flow,15)
    else:w,_=p.cunge.route(g,flow,4,15)
    assert np.isfinite(w).all() and w.min()>0
    record=dict(case=case,peak_start_ratio=ratio,state=state,method=method,**stat(t,w,settings['baseflow_cms'],flow))
    if not method.startswith('variable'):
     record['exact_linear_centroid_hours']=40+float(sum(g.reference_flow_fraction*g.link_length_m/g.rapid_celerity_mps))/3600
    rows.append(record);waves[state,method]=w
    if method.startswith('variable') and ratio==2:
     wd,_=route(g,flow,7.5)
     gf=g.copy();gf['count']*=2;gf['rapid_effective_length_m']/=2
     wx,_=route(gf,flow,15)
     for label,z in [('half_dt',wd),('half_dx',wx)]:
      checks.append(dict(case=case,state=state,check=label,max_difference_cms=float(np.max(abs(z-w))),**stat(t,z,settings['baseflow_cms'],flow)))
  keep=t<7*86400
  pd.DataFrame({'time_seconds':t[keep],'inflow':flow[keep],**{a+'__'+b:w[keep] for (a,b),w in waves.items()}}).to_parquet(out/f'{case}_ratio{ratio}_hydrographs.parquet')
r=pd.DataFrame(rows);r.to_csv(out/'variable_depth_screen.csv',index=False);pd.DataFrame(checks).to_csv(out/'variable_depth_refinement.csv',index=False)
results=[]
for keys,g in r.groupby(['case','peak_start_ratio','method']):
 a=g[g.state=='simple/original'].iloc[0];b=g[g.state=='simple/full'].iloc[0];results.append(dict(case=keys[0],ratio=keys[1],method=keys[2],peak_shift_min=60*(b.peak_hours-a.peak_hours),centroid_shift_min=60*((b.exact_linear_centroid_hours-a.exact_linear_centroid_hours) if keys[2].startswith('fixed') else b.centroid_hours-a.centroid_hours),max_abs_volume_error_pct=g.volume_error_pct.abs().max()))
r=pd.DataFrame(results);r.to_csv(out/'pair_screen.csv',index=False);print(r.round(3).to_string(index=False))
