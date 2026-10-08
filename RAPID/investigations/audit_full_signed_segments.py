"""Every base subreach, fixed K, native recurrence; all 16 including pulse-only cases."""
from pathlib import Path
import sys,json
sys.path.insert(0,str(Path(__file__).resolve().parent))
from route_full_signed_validation import *
import signed_coefficient_routing_check as audit

def run(ex):
 dest=OUT/ex;inv=pd.read_csv(dest/'inventory.csv');r=inv[(inv.family=='indep')&(inv.state=='S000_base')].iloc[0];f=pd.read_parquet(r.frame)
 if ex in ['11','46']:
  previous=BASE/'common_caps/signed_coefficient_check';rows=json.loads((previous/'segment_validation.json').read_text());row=next(x for x in rows if x['example']==ex)
  import shutil
  shutil.copyfile(previous/f'{ex}_every_segment_audit.csv',dest/'every_segment_audit.csv')
  row['reuse']='Exact same base hydraulic frame: checked below'
  from check_explicit_celerity_regularization import load_frames
  old,_=load_frames(ex);a=old['S000_base','original'].sort_values('id_link');b=f.sort_values('id_link')
  assert np.allclose(a.rapid_k,b.rapid_k,rtol=1e-10) and np.array_equal(a['count'],b['count'])
 else:
  _,_,_,_,_,_,fdt=grid_input(ex,r.max_travel_seconds,120.)
  dt=max(x for x in range(1,fdt+1) if fdt%x==0 and x<.99*r.upper_dt_seconds)
  if ex in ['13','45','48']:
   end=4*86400+max(10*86400,3*r.max_travel_seconds);t=np.arange(int(np.ceil(end/dt)))*dt
   def integral(x):
    a=np.clip(x-86400,0,12*3600);b=np.clip(x-86400-12*3600,0,24*3600)
    return a*a/(24*3600)+b-b*b/(48*3600)
   q0=qe=1.;inp=1+(integral(t+dt)-integral(t))/dt
  else:t,inp,q0,qe,_=audit.forcing(ex,dt,r.max_travel_seconds)
  audit.OUT=dest;audit.CLOCKS[ex]=[dt]
  row,_=audit.audit_segments(ex,f,dt,t,inp,q0,qe)
  (dest/f'{ex}_every_segment_audit.csv').rename(dest/'every_segment_audit.csv')
 row['forcing_kind']='standard_triangle_no_saved_event' if ex in ['13','45','48'] else 'saved_actual_event'
 (dest/'segment_validation.json').write_text(json.dumps(row,indent=2))
 print('SEGMENTS COMPLETE',ex,flush=True)

if __name__=='__main__':
 from concurrent.futures import ProcessPoolExecutor
 with ProcessPoolExecutor(max_workers=2) as pool:list(pool.map(run,sys.argv[1:] or EXAMPLES))
