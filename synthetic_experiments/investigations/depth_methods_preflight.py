"""Read-only grid census plus six isolated routing benchmarks; never runs full grid."""
from pathlib import Path
import sys,json,time,math,shutil
from dataclasses import fields
from collections import Counter
ROOT=Path(__file__).resolve().parents[2]
for p in (ROOT,ROOT/'RAPID/src',ROOT/'synthetic_experiments/src'):sys.path.insert(0,str(p))
from synthetic_experiments.depth_pipeline import configurations,run_group,provenance
from synthetic_experiments.hydraulics import HydraulicSettings
from synthetic_experiments.networks import make_states,canonical_network,digest
OUT=ROOT/'synthetic_experiments/outputs/depth_methods_preflight_20261027';OUT.mkdir(exist_ok=True)
c=json.loads((ROOT/'synthetic_experiments/configs/depth_comparison_20261027.json').read_text())
source=provenance();unique=set();refs=0;local=0;cells=0;hist=Counter();start=time.perf_counter()
for i,(name,p) in enumerate(configurations(c,True)):
 graphs,_=make_states(p);keys={}
 for g in graphs.values():
  net=canonical_network(g)
  # Explicit reference width/slope retain BASED reference-depth dependencies.
  key=digest(dict(network=net,reference_width=p['total_width_m'],reference_slope=p['reference_slope']))
  keys[key]=sum(math.ceil(e[2]/c['segment_length_m']) for e in net['edges'])
 refs+=len(graphs);local+=len(keys);cells+=sum(keys.values());unique.update(keys);hist[len(keys)]+=1
 if (i+1)%1000==0:print('census',i+1,flush=True)
census=dict(geometries=i+1,groups=3*(i+1),comparisons=21*(i+1),state_references_all_methods=3*refs,within_group_unique_all_methods=3*local,global_exact_keys_all_methods=3*len(unique),cell_passes_per_method_per_tail_attempt=cells,within_group_counts=dict(hist),census_seconds=time.perf_counter()-start,free_bytes=shutil.disk_usage(ROOT).free)
(OUT/'census.json').write_text(json.dumps(census,indent=2));print(census,flush=True)
settings={k:v for k,v in c.items() if k in {f.name for f in fields(HydraulicSettings)}}
bench=[]
for label,updates in [('typical',dict(length_multiplier=1.25,branch_width_factor=1.1)),('long_low_slope',dict(total_length_m=40000,total_width_m=1000,reference_slope=5e-5,length_multiplier=1.5,outer_a_fraction=.9,inner_fraction=.2,branch_width_factor=1.2))]:
 for method in c['methods']:
  p=dict(c['base_geometry'],**updates);s=dict(settings,method=method);job=dict(id=digest(dict(p=p,s=s)),case=label,parameters=p,settings=s)
  folder=OUT/'benchmarks'/f'{label}_{method}'
  if (folder/'completion.json').exists():raise RuntimeError('Benchmark output exists: use saved timings, do not time a resume')
  t=time.perf_counter();print('benchmark',label,method,flush=True)
  try:
   run_group(job,source,c['convergence'],folder)
   elapsed=time.perf_counter()-t;done=json.loads((folder/'completion.json').read_text())
   row=dict(case=label,method=method,seconds=elapsed,bytes=sum(f.stat().st_size for f in folder.iterdir()),unique_states=done['n_unique_states'],tail_hours=[h['tail_hours'] for h in done['convergence_history']],status='complete')
  except Exception as e:row=dict(case=label,method=method,seconds=time.perf_counter()-t,status='failed',error=str(e))
  bench.append(row);(OUT/'benchmarks.json').write_text(json.dumps(bench,indent=2));print(row,flush=True)
