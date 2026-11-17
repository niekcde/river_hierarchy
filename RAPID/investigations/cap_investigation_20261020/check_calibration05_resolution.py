"""Read-only production audit: example 05 BASED-midpoint calibration resolution."""
from pathlib import Path
import ast,json
import numpy as np
import pandas as pd
from scipy.signal import find_peaks
import run_three_runs as r
from check_peak_resolution import crest

out=Path('/Volumes/PhD/river_hierarchy/three_run_comparison_20261028')
r.init_worker(out,'diagnostic-only')
ex='05';run=r.RUNS[1];case=r.CTX['cases'][ex]
dest=r.SETUP/'calibration05_resolution_diagnosis';dest.mkdir(exist_ok=True)
profile=pd.read_csv(out/run/'calibration'/ex/'beta_profile.csv')
best=min(profile.to_dict('records'),key=lambda x:(x['height_distance_cms'],abs(np.log(x['beta'])),abs(x['timing_error_minutes'])))
grid,dt,_=r.prepare_frame(ex,'S000_base',run)
data=pd.read_parquet(case['calibration_forcing']);meta=next(x for x in json.loads(Path(case['calibration_metadata']).read_text()) if x['method']=='cunge')
valid=data.downstream_cms.notna().to_numpy();times=data.time_utc.astype('int64').to_numpy()/1e9
goal=pd.Timestamp(case['timing_target']['target_utc']).timestamp()
archive=r.BUNDLE/'source_code/calibrate_common_grid_height.py'
node=next(x for x in ast.parse(archive.read_text()).body if isinstance(x,ast.FunctionDef) and x.name=='feature')
scope={'np':np,'pd':pd,'find_peaks':find_peaks};exec(compile(ast.Module(body=[node],type_ignores=[]),str(archive),'exec'),scope)
rows=[]
for sub,clock in [(1,1),(1,2),(2,2),(4,4),(8,8)]:
 g=grid.copy();g['count']*=sub;g['rapid_effective_length_m']=g.link_length_m/g['count']
 q,audit=r.solve_wave(r.hydraulic_parameters(g,run,best['n_manning'],best['beta']),data.forcing_cms.to_numpy(),dt/clock,meta['initial_cms'])
 tm,h,kind,_=scope['feature'](q[valid],times[valid],ex,case['timing_target'])
 c=crest(q[valid],times[valid],meta['initial_cms'])
 peaks,_=find_peaks(q[valid],prominence=np.ptp(q[valid])*.001,distance=30)
 peaks=sorted(peaks,key=lambda i:q[valid][i],reverse=True)[:3]
 row=dict(subdivision=sub,clock_divisor=clock,native_dt_seconds=dt/clock,timing_error_minutes=(tm-goal)/60,feature_kind=kind,**c,
 prominent_peaks=[dict(time=str(pd.to_datetime(times[valid][i],unit='s',utc=True)),q=float(q[valid][i])) for i in peaks])
 rows.append(row)
 pd.DataFrame(dict(time_utc=data.time_utc,q_outlet_cms=q)).to_parquet(dest/f'wave_{sub}_{clock}.parquet',index=False)
 print(json.dumps(row),flush=True)
r.write_json(dest/'results.json',dict(selected=best,timing_target=case['timing_target'],height_target=case['height_target'],results=rows,
 sources={str(p):r.digest(p) for p in [Path(__file__),Path(r.__file__),Path(case['calibration_forcing']),out/run/'calibration'/ex/'beta_profile.csv']}))
pd.DataFrame([{k:v for k,v in row.items() if k!='prominent_peaks'} for row in rows]).to_csv(dest/'results.csv',index=False)
