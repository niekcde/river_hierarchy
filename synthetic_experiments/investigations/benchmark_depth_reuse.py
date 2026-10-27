"""Isolated benchmark: production routing unchanged; exact-input on-disk cache."""
from pathlib import Path
import sys,json,time,inspect,copy,math
from dataclasses import asdict
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[2]
for p in (ROOT,ROOT/'RAPID/src',ROOT/'synthetic_experiments/src'):sys.path.insert(0,str(p))
import synthetic_experiments.depth_pipeline as dp
from synthetic_experiments.networks import digest,make_states,canonical_network
OUT=ROOT/'synthetic_experiments/outputs/depth_reuse_benchmark_20261027';OUT.mkdir(exist_ok=True)
if (OUT/'results.json').exists():raise RuntimeError('Existing benchmark: do not overwrite measured results')
c=json.loads((ROOT/'synthetic_experiments/configs/depth_comparison_20261027.json').read_text());source=dp.provenance()
settings={k:v for k,v in c.items() if k in dp.HydraulicSettings.__dataclass_fields__}
original=dp.route;results=[];callrows=[]
# Identical production function except tail growth, confined to this investigation.
namespace=dict(dp.__dict__)
exec(inspect.getsource(dp.run_group).replace('tail*=2','tail*=1.5'),namespace)
small_tail=namespace['run_group']
scenarios=[('typical',dict(length_multiplier=1.25,branch_width_factor=1.1)),('long_low_slope',dict(total_length_m=40000,total_width_m=1000,reference_slope=5e-5,length_multiplier=1.5,outer_a_fraction=.9,branch_width_factor=1.2))]
jobs=[]
for label,changes in scenarios:
 for inner in [.2,.8]:
  for method in c['methods']:
   p=dict(c['base_geometry'],**changes);p['inner_fraction']=inner
   jobs.append(dict(id=digest(dict(p=p,method=method)),case=label,parameters=p,settings=dict(settings,method=method)))
for variant in ['baseline','disk_cache']:
 counters=dict(hits=0,misses=0)
 def wrapped(frame,flow,s):
  # All prepared columns, settings (including tail), source and forcing bytes.
  import hashlib
  key=digest(dict(frame=frame.to_json(orient='split',double_precision=15),settings=asdict(s),source=source,flow=hashlib.sha256(flow.tobytes()).hexdigest()))
  folder=OUT/'cache';folder.mkdir(exist_ok=True);meta=folder/(key+'.json');wave=folder/(key+'.npy')
  if variant=='disk_cache' and meta.exists() and wave.exists():
   counters['hits']+=1;return np.load(wave),json.loads(meta.read_text())
  counters['misses']+=1;t=time.perf_counter();w,a=original(frame,flow,s)
  callrows.append(dict(variant=variant,method=s.method,tail=s.tail_hours,cells=int(frame['count'].sum()),samples=len(flow),seconds=time.perf_counter()-t))
  if variant=='disk_cache':np.save(wave,w);meta.write_text(json.dumps(a))
  return w,a
 dp.route=wrapped
 for job in jobs:
  folder=OUT/variant/job['id'];t=time.perf_counter();before=copy.copy(counters)
  print(variant,job['case'],job['parameters']['inner_fraction'],job['settings']['method'],flush=True)
  response=dp.run_group(job,source,c['convergence'],folder)
  elapsed=time.perf_counter()-t
  if variant=='disk_cache':
   prior=pd.read_csv(OUT/'baseline'/job['id']/'responses.csv');pd.testing.assert_frame_equal(response,prior,check_dtype=False,rtol=1e-12,atol=1e-12)
   a=pd.read_parquet(OUT/'baseline'/job['id']/'hydrographs.parquet');b=pd.read_parquet(folder/'hydrographs.parquet');pd.testing.assert_frame_equal(a,b,check_exact=True)
  history=json.loads((folder/'completion.json').read_text())['convergence_history']
  row=dict(variant=variant,case=job['case'],inner=job['parameters']['inner_fraction'],method=job['settings']['method'],seconds=elapsed,hits=counters['hits']-before['hits'],misses=counters['misses']-before['misses'],tails=[h['tail_hours'] for h in history]);results.append(row);print(row,flush=True)
  (OUT/'progress.json').write_text(json.dumps(results,indent=2))
dp.route=original
for job in jobs:
 if job['settings']['method']!='variable_manning' or job['parameters']['inner_fraction']!=.2:continue
 folder=OUT/'smaller_tail'/job['id'];t=time.perf_counter();response=small_tail(job,source,c['convergence'],folder);elapsed=time.perf_counter()-t
 baseline=pd.read_csv(OUT/'baseline'/job['id']/'responses.csv');history=json.loads((folder/'completion.json').read_text())['convergence_history']
 row=dict(variant='smaller_tail',case=job['case'],method='variable_manning',seconds=elapsed,tails=[h['tail_hours'] for h in history],max_centroid_shift_difference_seconds=float(abs(response.centroid_shift_minutes-baseline.centroid_shift_minutes).max()*60),max_peak_shift_difference_seconds=float(abs(response.peak_shift_minutes-baseline.peak_shift_minutes).max()*60))
 results.append(row);print(row,flush=True);(OUT/'progress.json').write_text(json.dumps(results,indent=2))
(OUT/'results.json').write_text(json.dumps(results,indent=2));pd.DataFrame(callrows).to_csv(OUT/'routing_calls.csv',index=False)
# Cell-weighted exact-state reuse across the real full grid, not just this small panel.
unique={};total=0
for i,(_,p) in enumerate(dp.configurations(c,True)):
 local={}
 for g in make_states(p)[0].values():
  net=canonical_network(g);key=digest(dict(network=net,reference_width=p['total_width_m'],reference_slope=p['reference_slope']));local[key]=sum(math.ceil(e[2]/200) for e in net['edges'])
 total+=sum(local.values());unique.update(local)
print('weighted census',total,sum(unique.values()),flush=True)
(OUT/'weighted_census.json').write_text(json.dumps(dict(per_method_cells_without_global_cache=total,per_method_cells_with_global_cache=sum(unique.values()),remaining_fraction=sum(unique.values())/total),indent=2))
