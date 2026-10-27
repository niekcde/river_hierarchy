"""Input-only full factorial preflight plus measured pilot cost, no batch routing."""
from pathlib import Path
from itertools import product
from functools import lru_cache
import time,json,hashlib
import numpy as np
import pandas as pd
import networkx as nx
import cunge_pilot as pilot

ROOT=pilot.ROOT
OUT=ROOT/'synthetic_experiments/outputs/cunge_grid_preflight_20261026'
GRID=ROOT/'synthetic_experiments/configs/grid.json'
# BASED is identical for every repeated width/slope/Q anchor.
pilot.predict_based_depth=lru_cache(None)(pilot.predict_based_depth)


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def fingerprint(f,dt,settings):
    cols=['u','v','link_length_m','width','slope','depth_m','count','rapid_k','rapid_x','reference_flow_fraction']
    return hashlib.sha256(json.dumps(dict(frame=f[cols].round(12).to_dict('list'),dt=dt,settings=settings),sort_keys=True).encode()).hexdigest()[:24]


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    cfg=json.loads(pilot.CONFIG.read_text());grid=json.loads(GRID.read_text())['grid']
    sources={str(p):sha(p) for p in [Path(__file__),Path(pilot.__file__),pilot.CONFIG,GRID]}
    benchmark=[]
    for name,change in cfg['cases'].items():
        frames,_,_=pilot.prepare(dict(cfg['base_geometry'],**change),cfg)
        dt=pilot.shared_clock(list(frames.values()),[60],cfg['maximum_native_dt_seconds'])
        audit=json.loads((pilot.OUT/name/'audit.json').read_text());tail=audit['tail_hours']
        start=time.perf_counter();_,_,_,_,ok=pilot.run_group(frames,cfg,dt,tail)
        elapsed=time.perf_counter()-start
        horizon=(cfg['spinup_hours']+cfg['rise_hours']+cfg['fall_hours']+tail)*3600
        work=sum(int(f['count'].sum()) for f in frames.values())*horizon/dt
        benchmark.append(dict(configuration=name,wall_seconds=elapsed,native_cell_steps=work,seconds_per_cell_step=elapsed/work,tail_hours=tail,settled_and_volume_passed=bool(ok)))
    pd.DataFrame(benchmark).to_csv(OUT/'runtime_benchmark.csv',index=False)
    rows=[];failures=[];jobs={};states=[]
    for index,values in enumerate(product(*grid.values())):
        pars=dict(zip(grid,values));ident=f'grid_{index:04d}'
        try:
            frames,pairs,depth=pilot.prepare(pars,cfg)
            dt=pilot.shared_clock(list(frames.values()),[60],cfg['maximum_native_dt_seconds'])
            max_path_k=0;total_sections=0;max_speed=0;min_x=np.inf;max_x=-np.inf
            for state,f in frames.items():
                g=nx.MultiDiGraph()
                for r in f.itertuples():g.add_edge(r.u,r.v,K=r.rapid_k*r.count)
                distance={}
                for u in nx.topological_sort(g):distance[u]=max((distance[v]+d['K'] for v,_,d in g.in_edges(u,data=True)),default=0.)
                max_path_k=max(max_path_k,max(distance.values()))
                sections=int(f['count'].sum());total_sections+=sections
                max_speed=max(max_speed,float(f.rapid_celerity_mps.max()));min_x=min(min_x,float(f.rapid_x.min()));max_x=max(max_x,float(f.rapid_x.max()))
                key=fingerprint(f,dt,cfg)
                jobs.setdefault(key,dict(job_id=key,sections=sections,dt_seconds=dt,physical_links=len(f)))
                states.append(dict(configuration=ident,state=state,job_id=key))
            rows.append(dict(configuration=ident,**pars,reference_depth_m=depth,state_references=len(frames),comparisons=len(pairs),dt_seconds=dt,total_sections=total_sections,max_path_K_hours=max_path_k/3600,max_celerity_mps=max_speed,min_X=min_x,max_X=max_x,status='input_preflight_passed'))
        except Exception as e:
            failures.append(dict(configuration=ident,parameters=pars,error=str(e)))
        if index%100==0:
            pd.DataFrame(rows).to_csv(OUT/'configurations.csv',index=False)
            print(index+1,'configurations;',len(jobs),'provisional unique states;',len(failures),'failures',flush=True)
    d=pd.DataFrame(rows);d.to_csv(OUT/'configurations.csv',index=False)
    pd.DataFrame(states).to_csv(OUT/'state_references.csv',index=False)
    j=pd.DataFrame(jobs.values());j.to_csv(OUT/'provisional_jobs.csv',index=False)
    speed=[b['seconds_per_cell_step'] for b in benchmark]
    forecasts=[]
    # Conditional horizons, not assertions that every state settles at these tails.
    for tail in [96,192,768,1536]:
        horizon=(cfg['spinup_hours']+cfg['rise_hours']+cfg['fall_hours']+tail)*3600
        work=float((j.sections*horizon/j.dt_seconds).sum())
        forecasts.append(dict(tail_hours=tail,unique_routing_hours_fast=work*min(speed)/3600,unique_routing_hours_slow=work*max(speed)/3600,
            uncompressed_time_and_outlet_GiB=len(j)*horizon/60*16/2**30,
            max_one_signal_MiB=float((horizon/j.dt_seconds*8).max()/2**20)))
    pd.DataFrame(forecasts).to_csv(OUT/'conditional_cost.csv',index=False)
    assert all(sha(p)==h for p,h in sources.items())
    pilot.save(OUT/'completion.json',dict(complete=True,planned_configurations=int(np.prod([len(v) for v in grid.values()])),passed_configurations=len(d),failures=failures,comparison_references=int(d.comparisons.sum()),state_references=len(states),provisional_unique_jobs=len(j),minimum_dt_seconds=float(d.dt_seconds.min()),maximum_path_K_hours=float(d.max_path_K_hours.max()),source_sha256=sources,
        qualification='Input-only feasibility, not full-grid numerical convergence or settlement. Deduplication provisional: final shared tail, output and implementation identity must enter job IDs. Runtime forecasts measure in-memory pilot routing only; exclude I/O, repeated extensions, preflight, descriptors and convergence repeats. One-signal memory excludes simultaneous graph branches. Four constant-tail scenarios are conditional estimates, not completion deadlines. No full factorial routed.'))
    print('COMPLETE',len(d),'passed',len(failures),'failed',len(j),'provisional unique jobs',flush=True)

if __name__=='__main__':main()
