"""Plan/run the current three-method synthetic experiment, separate from legacy.

python -m synthetic_experiments.depth_pipeline --config ... --output ... [--run]
Default is a plan only. Completed groups can resume only with identical inputs
and source hashes. Full factorial is opt-in; selected pilot geometries are default.
"""
import argparse
from dataclasses import asdict, fields, replace
import hashlib
import itertools
import json
import multiprocessing
import shutil
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from pathlib import Path

import numpy as np
import pandas as pd

from .hydraulics import HydraulicSettings, METHODS, prepare, forcing, route, summarize, reported
from .networks import make_states, canonical_network, digest
from .state_cache import cached_state


def save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');tmp.replace(path)


def provenance():
    import scipy
    import rapid_tools.cunge as shared
    from rapid_tools.reference_section import DEFAULT_BASED_MODEL_PATH
    folder=Path(__file__).parent
    files=[folder/name for name in ('depth_pipeline.py','state_cache.py','hydraulics.py','_mct.cpp','networks.py','design.py','collapse.py','plotting.py')]+list((folder/'core').glob('*.py'))+[Path(shared.__file__),Path(DEFAULT_BASED_MODEL_PATH)]
    return dict(files={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},numpy=np.__version__,scipy=scipy.__version__)


def configurations(config,factorial=False):
    if factorial:
        levels=config['grid'];names=list(levels)
        return [(f'grid_{i:05}',dict(config['base_geometry'],**dict(zip(names,v))))
                for i,v in enumerate(itertools.product(*(levels[k] for k in names)))]
    return [(name,dict(config['base_geometry'],**updates,branch_width_factor=factor))
            for name,updates in config['cases'].items() for factor in config['branch_width_factors']]


def plan(config,output,*,factorial=False,cases=None,methods=None):
    chosen=configurations(config,factorial)
    if cases is not None:
        unknown=set(cases)-{x[0] for x in chosen}
        if unknown:raise ValueError(f'Unknown cases: {unknown}')
        chosen=[x for x in chosen if x[0] in cases]
    selected=methods or config['methods']
    if not set(selected)<=set(METHODS):raise ValueError('Unknown methods')
    names={f.name for f in fields(HydraulicSettings)}
    settings={k:v for k,v in config.items() if k in names and k!='method'}
    source=provenance();jobs=[]
    for name,parameters in chosen:
        for method in selected:
            s=HydraulicSettings(**dict(settings,method=method))
            identity=digest(dict(parameters=parameters,settings=asdict(s),source=source,
                                 convergence=config['convergence']))
            jobs.append(dict(id=identity,case=name,parameters=parameters,settings=asdict(s)))
    manifest=dict(schema='synthetic-depth-comparison-v1',source=source,config=config,jobs=jobs,
                  comparisons_per_group=7,n_groups=len(jobs),n_comparisons=7*len(jobs),factorial=factorial)
    output=Path(output);old=output/'manifest.json'
    if old.exists() and json.loads(old.read_text())!=manifest:
        raise ValueError('Output manifest differs; use a new output directory (no mixed scientific provenance)')
    save(old,manifest);return manifest


def run_group(job,source,convergence,folder,cache_dir=None):
    folder=Path(folder);complete=folder/'completion.json'
    identity=digest(dict(job=job,source=source,convergence=convergence))
    if complete.exists():
        status=json.loads(complete.read_text())
        if status['identity']!=identity:raise ValueError('Group provenance mismatch')
        required=['responses.csv','state_metrics.csv','hydrographs.parquet','parameters.csv']
        if not all((folder/name).exists() for name in required):raise ValueError('Incomplete saved group')
        return pd.read_csv(folder/'responses.csv')
    params=job['parameters'];s=HydraulicSettings(**job['settings'])
    graphs,pairs=make_states(params)
    frames={state:prepare(g,params,s) for state,g in graphs.items()}
    aliases={state:digest(canonical_network(graph)) for state,graph in graphs.items()}
    history=[];previous=None;tail=s.tail_hours
    while tail<=convergence['maximum_tail_hours']:
        current=replace(s,tail_hours=tail);t,q=forcing(current);input_stats=summarize(t,q,s.baseflow_cms)
        waves={};metrics={};cached={}
        for state,frame in frames.items():
            key=aliases[state]
            if key not in cached:
                def calculate():
                    w,a=route(frame,q,current);stats=summarize(t,w,s.baseflow_cms)
                    if stats['minimum_excess_cms'] < -1e-7*(s.peak_cms-s.baseflow_cms):
                        raise ValueError('Substantive below-plateau dip; reject this configuration')
                    stats.update(a)
                    stats['volume_error_fraction']=stats['volume_m3']/input_stats['volume_m3']-1
                    if s.method!='variable_manning':
                        exact=input_stats['centroid_seconds']+float(sum(frame.reference_flow_fraction*frame.link_length_m/frame.rapid_celerity_mps))
                        stats['centroid_convergence_error_seconds']=abs(stats['centroid_seconds']-exact)
                    return reported(t,w,current).discharge_cms.to_numpy(),stats
                if cache_dir is None:
                    cached[key]=calculate()
                else:
                    frame_hash=hashlib.sha256(pd.util.hash_pandas_object(frame,index=True).values.tobytes()).hexdigest()
                    cache_key=digest(dict(source=source,settings=asdict(current),frame=frame_hash,
                                         columns=list(frame.columns),dtypes=[str(x) for x in frame.dtypes],
                                         forcing=hashlib.sha256(q.tobytes()).hexdigest()))
                    cached[key]=cached_state(cache_dir,cache_key,calculate)
            waves[state],metrics[state]=cached[key]
        if s.method=='variable_manning':
            for state in metrics:
                metrics[state]=dict(metrics[state],centroid_convergence_error_seconds=(
                    abs(metrics[state]['centroid_seconds']-previous[state]) if previous else None))
        volume=max(abs(m['volume_error_fraction']) for m in metrics.values())
        mass=max(abs(m['network_mass_residual_m3'])/input_stats['volume_m3'] for m in metrics.values())
        centroid=max((m['centroid_convergence_error_seconds'] if m['centroid_convergence_error_seconds'] is not None else 1e99) for m in metrics.values())
        final=max(abs(m['final_excess_cms']) for m in metrics.values())/(s.peak_cms-s.baseflow_cms)
        history.append(dict(tail_hours=tail,max_volume_error=volume,max_mass_residual_fraction=mass,
                            max_centroid_error_seconds=centroid,max_final_excess_fraction=final))
        if mass>convergence['mass_balance_tolerance']:raise ValueError(f'Mass-balance failure {mass}')
        if volume<convergence['volume_tolerance'] and centroid<convergence['centroid_seconds'] and final<convergence['final_excess_fraction']:break
        previous={state:m['centroid_seconds'] for state,m in metrics.items()};tail*=2
    else:raise ValueError('Tail/centroid did not converge; increase maximum_tail_hours, do not publish truncated response')
    folder.mkdir(parents=True,exist_ok=True)
    waveframe=pd.DataFrame(dict(time_seconds=reported(t,q,current).time_seconds,inflow_cms=reported(t,q,current).discharge_cms))
    for state,w in waves.items():waveframe[state]=w
    waveframe.to_parquet(folder/'hydrographs.parquet',index=False)
    pd.DataFrame([dict(state=state,**m) for state,m in metrics.items()]).to_csv(folder/'state_metrics.csv',index=False)
    pd.concat([f.assign(state=k) for k,f in frames.items()]).to_csv(folder/'parameters.csv',index=False)
    rows=[]
    for pair in pairs:
        p,c=metrics[pair['parent_state']],metrics[pair['child_state']]
        lag=p['centroid_seconds']-input_stats['centroid_seconds']
        rows.append(dict(**pair,case=job['case'],method=s.method,**params,
            centroid_shift_minutes=(c['centroid_seconds']-p['centroid_seconds'])/60,
            peak_shift_minutes=(c['peak_seconds']-p['peak_seconds'])/60,
            centroid_lag_change_pct=100*(c['centroid_seconds']-p['centroid_seconds'])/lag if lag>0 else None,
            tail_hours=tail,group_id=job['id']))
    response=pd.DataFrame(rows);response.to_csv(folder/'responses.csv',index=False)
    save(complete,dict(identity=identity,job=job,status='complete',convergence_history=history,
        response_definition='native endpoint peak; full excess-volume centroid; common forcing and shared tail within group',
        report_definition='exact trapezoidal interval means; native timing metrics saved separately',
        n_unique_states=len(set(aliases.values())),aliases=aliases))
    return response


def _run_job(args):
    job,source,convergence,output,min_free_gb=args
    if shutil.disk_usage(output).free < min_free_gb*1024**3:
        raise RuntimeError(f'Less than {min_free_gb:g} GiB free; free space and resume the same command')
    folder=Path(output)/'groups'/job['id']
    result=run_group(job,source,convergence,folder,Path(output)/'state_cache')
    import resource,sys
    rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    save(folder/'worker_resources.json',dict(peak_rss_bytes=int(rss if sys.platform=='darwin' else rss*1024)))
    return result


def run(manifest,output,workers=1,min_free_gb=5):
    if workers<1:raise ValueError('workers must be positive')
    if manifest['source']!=provenance():raise ValueError('Source changed after planning')
    output=Path(output)
    if manifest.get('factorial') and shutil.disk_usage(output).free < 40*1024**3:
        raise RuntimeError('Full factorial requires at least 40 GiB free at launch; free space or choose a larger output volume')
    # Compile once before spawning: workers only load the finished library.
    if any(j['settings']['method']=='variable_manning' for j in manifest['jobs']):
        from .hydraulics import _kernel
        _kernel()
    args=[(job,manifest['source'],manifest['config']['convergence'],str(output),min_free_gb) for job in manifest['jobs']]
    def collect(iterator):
        rows=[]
        for i,result in enumerate(iterator,1):
            rows.append(result)
            print(f"Completed {i}/{len(args)} groups",flush=True)
            if i%100==0:pd.concat(rows).to_csv(output/'responses.csv',index=False)
        pd.concat(rows).to_csv(output/'responses.csv',index=False)
        return len(rows)
    if workers==1:
        count=collect(map(_run_job,args))
    else:
        with ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('spawn')) as pool:
            def bounded_results():
                pending=set();remaining=iter(args)
                for _ in range(workers):
                    item=next(remaining,None)
                    if item is not None:pending.add(pool.submit(_run_job,item))
                try:
                    while pending:
                        done,pending=wait(pending,return_when=FIRST_COMPLETED)
                        for future in done:
                            yield future.result()
                            item=next(remaining,None)
                            if item is not None:pending.add(pool.submit(_run_job,item))
                finally:
                    for future in pending:future.cancel()
            count=collect(bounded_results())
    save(output/'completion.json',dict(status='complete',n_groups=count,n_comparisons=7*count,
        source=manifest['source']))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--run',action='store_true');parser.add_argument('--factorial',action='store_true')
    parser.add_argument('--case',action='append');parser.add_argument('--method',action='append',choices=METHODS)
    parser.add_argument('--workers',type=int,default=1,help='Local worker processes; use 2 for two CPU workers')
    parser.add_argument('--minimum-free-gb',type=float,default=5,help='Stop before new groups if disk space falls below this many GiB')
    a=parser.parse_args();config=json.loads(a.config.read_text())
    manifest=plan(config,a.output,factorial=a.factorial,cases=a.case,methods=a.method)
    print(f"Planned {manifest['n_groups']} groups / {manifest['n_comparisons']} parent-child comparisons",flush=True)
    if a.run:run(manifest,a.output,workers=a.workers,min_free_gb=a.minimum_free_gb)

if __name__=='__main__':main()
