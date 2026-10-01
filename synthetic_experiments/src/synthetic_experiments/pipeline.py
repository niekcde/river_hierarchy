"""Plan, resume, and analyse controlled synthetic routing experiments."""
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import itertools
import json
from pathlib import Path
import shutil
import time
import traceback

import numpy as np
import pandas as pd

from .networks import make_states, canonical_network, digest
from .routing import Settings, schedule, route

SCHEMA='synthetic-grid-v2'


def write_json(path,value):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
    temporary.replace(path)


def source_fingerprint():
    """Invalidate cached jobs when scientific implementation/dependencies change."""
    import hashlib, importlib.metadata
    import rapid_tools.engine, rapid_tools.prep, rapid_tools.k_values
    import hierarchy_level_definition.unit_detection.bifurcation_confluence_units as detection
    import hierarchy_level_definition.metrics.equivalent_resistance as eq
    import network_variants.equivalent_transition_domain as domains
    paths=list(Path(__file__).parent.rglob('*.py'))+[Path(m.__file__) for m in
        (rapid_tools.engine,rapid_tools.prep,rapid_tools.k_values,detection,eq,domains)]
    versions={name:importlib.metadata.version(name) for name in ('numpy','scipy','pandas','networkx','geopandas','shapely')}
    return digest(dict(files={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},versions=versions))


def routing_provenance():
    """Execution-only identity for exact job reuse across analysis extensions."""
    import hashlib, inspect, importlib.metadata
    from . import routing
    import rapid_tools.engine as engine, rapid_tools.prep as prep, rapid_tools.k_values as k_values
    return dict(files={m.__name__:hashlib.sha256(Path(m.__file__).read_bytes()).hexdigest()
                       for m in (routing,engine,prep,k_values)},
                digest_source=inspect.getsource(digest),
                versions={name:importlib.metadata.version(name) for name in
                          ('numpy','scipy','pandas','networkx','geopandas','shapely')})


def reuse_runs(output, source_output):
    """Copy only complete jobs with identical execution inputs and routing code.

    Legacy compatibility is explicitly recorded before any source changes;
    matching geometry alone is never sufficient evidence for reusing a run.
    Original outputs are neither edited nor hard-linked to the new directory.
    """
    output,source_output=Path(output),Path(source_output)
    if output.resolve()==source_output.resolve():
        raise ValueError('Reuse requires a different source output directory')
    target=verify_plan(output)
    source=json.loads((source_output/'manifest.json').read_text())
    routing_source=digest(routing_provenance())
    source_routing=source.get('routing_source')
    if source_routing is None:
        registry=Path(__file__).resolve().parents[2]/'configs'/'routing_compatibility.json'
        record=json.loads(registry.read_text()).get(source['source'],{})
        provenance=record.get('routing_provenance')
        if provenance and digest(provenance)==record.get('routing_source'):
            source_routing=record['routing_source']
    if source_routing!=routing_source or target['routing_source']!=routing_source:
        raise ValueError('Routing implementation compatibility is not established; reroute instead')
    identity=lambda spec:digest({k:spec[k] for k in ('network','settings','clock')})
    available={}
    for run_id in source['run_ids']:
        spec=json.loads((source_output/'jobs'/f'{run_id}.json').read_text())
        if spec['source']!=source['source']:
            raise ValueError('Source job provenance differs from source manifest')
        folder=source_output/'runs'/run_id
        if not all((folder/name).exists() for name in ('metrics.json','hydrograph.parquet','segments.csv')):
            continue
        metrics=json.loads((folder/'metrics.json').read_text())
        if metrics.get('run_id')==run_id:
            available[identity(spec)]=(run_id,folder,metrics)
    reused=[]; already_complete=0
    for run_id in target['run_ids']:
        destination=output/'runs'/run_id
        if all((destination/name).exists() for name in ('metrics.json','hydrograph.parquet','segments.csv')):
            if json.loads((destination/'metrics.json').read_text()).get('run_id')==run_id:
                already_complete+=1
                continue
        spec=json.loads((output/'jobs'/f'{run_id}.json').read_text())
        match=available.get(identity(spec))
        if match is None:
            continue
        old_id,folder,metrics=match
        destination.mkdir(parents=True,exist_ok=True)
        for name in ('hydrograph.parquet','segments.csv'):
            temporary=destination/(name+'.tmp')
            shutil.copy2(folder/name,temporary)
            temporary.replace(destination/name)
        write_json(destination/'metrics.json',dict(metrics,run_id=run_id,
            reused_from=dict(output=str(source_output.resolve()),run_id=old_id,source=source['source'])))
        (destination/'failure.json').unlink(missing_ok=True)
        reused.append(dict(run_id=run_id,source_run_id=old_id))
    report=dict(source_output=str(source_output.resolve()),source_plan=source['plan_id'],
                target_plan=target['plan_id'],routing_source=routing_source,
                reused=len(reused),already_complete=already_complete,
                remaining=target['n_unique_runs']-len(reused)-already_complete,jobs=reused)
    write_json(output/'reuse_status.json',report)
    print({k:v for k,v in report.items() if k!='jobs'},flush=True)
    return report


def plan(config_path,output):
    output=Path(output); config=json.loads(Path(config_path).read_text())
    settings=Settings(**config['routing'])
    if 'cases' in config:
        cases=config['cases']
        if not cases: raise ValueError('Empty cases')
    else:
        grid=config['grid']; keys=list(grid)
        if not grid or any(not values for values in grid.values()): raise ValueError('Empty grid')
        cases=(dict(zip(keys,values)) for values in itertools.product(*(grid[k] for k in keys)))
    fingerprint=source_fingerprint()
    execution_provenance=routing_provenance()
    routing_source=digest(execution_provenance)
    config_key=digest(dict(config=config,source=fingerprint,schema=SCHEMA))
    existing=output/'manifest.json'
    if existing.exists():
        if json.loads(existing.read_text())['plan_id']!=config_key:
            raise ValueError('Output belongs to another config/code revision; use a new output directory')
        return json.loads(existing.read_text())
    runs={}; configurations=[]; pairs=[]; states_rows=[]
    stats_cache={}
    for i,params in enumerate(cases):
        config_id=digest(params)
        states,comparisons=make_states(params)
        canonical={name:canonical_network(graph) for name,graph in states.items()}
        clock=schedule(canonical.values(),settings,cache=stats_cache)
        mapping={}
        for name,network in canonical.items():
            spec=dict(schema=SCHEMA,source=fingerprint,routing_source=routing_source,network=network,settings=asdict(settings),
                      clock={k:clock[k] for k in ('dt_seconds','buffer_hours')})
            run_id=digest(spec); mapping[name]=run_id
            runs.setdefault(run_id,dict(run_id=run_id,**spec))
            states_rows.append(dict(configuration_id=config_id,state_id=name,run_id=run_id))
        configurations.append(dict(configuration_id=config_id,parameters=params,clock=clock))
        for pair in comparisons:
            pairs.append(dict(configuration_id=config_id,**pair,parent_run_id=mapping[pair['parent_state']],
                              child_run_id=mapping[pair['child_state']]))
        if (i+1)%100==0: print(f'Planned {i+1} configurations; {len(runs)} unique routing jobs',flush=True)
    output.mkdir(parents=True,exist_ok=True)
    for run in runs.values(): write_json(output/'jobs'/f'{run["run_id"]}.json',run)
    pd.DataFrame(states_rows).to_csv(output/'states.csv',index=False)
    pd.DataFrame(pairs).to_csv(output/'comparisons.csv',index=False)
    write_json(output/'configurations.json',configurations)
    manifest=dict(schema=SCHEMA,plan_id=config_key,source=fingerprint,config=config,
                  routing_source=routing_source,routing_provenance=execution_provenance,
                  created_utc=datetime.now(timezone.utc).isoformat(),n_configurations=len(configurations),
                  n_state_references=len(states_rows),n_unique_runs=len(runs),n_comparisons=len(pairs),
                  run_ids=list(runs),numerical_policy='shared dt and forcing window within each configuration')
    write_json(existing,manifest)
    return manifest


def verify_plan(output):
    manifest=json.loads((Path(output)/'manifest.json').read_text())
    if manifest['source']!=source_fingerprint():
        raise ValueError('Code changed after planning; create a new plan/output directory')
    return manifest


def run_jobs(output,limit=None):
    output=Path(output); manifest=verify_plan(output); attempted=0; completed=0; failed=0
    for run_id in manifest['run_ids']:
        destination=output/'runs'/run_id
        done=destination/'metrics.json'; hydro=destination/'hydrograph.parquet'
        if done.exists() and hydro.exists() and (destination/'segments.csv').exists():
            if json.loads(done.read_text()).get('run_id')==run_id:
                completed+=1; continue
        if limit is not None and attempted>=limit: break
        attempted+=1
        spec=json.loads((output/'jobs'/f'{run_id}.json').read_text())
        destination.mkdir(parents=True,exist_ok=True)
        try:
            data,metrics,segments=route(spec['network'],Settings(**spec['settings']),spec['clock'])
            data.to_parquet(hydro.with_suffix('.tmp'),index=False)
            hydro.with_suffix('.tmp').replace(hydro)
            segments.drop(columns='geometry').to_csv(destination/'segments.csv',index=False)
            write_json(done,dict(run_id=run_id,**metrics))
            (destination/'failure.json').unlink(missing_ok=True)
            completed+=1
            print(f'{completed}/{manifest["n_unique_runs"]}: {run_id} ({metrics["wall_seconds"]:.2f}s)',flush=True)
        except Exception as exc:
            failed+=1
            write_json(destination/'failure.json',dict(run_id=run_id,error=str(exc),traceback=traceback.format_exc()))
            print(f'FAILED {run_id}: {exc}',flush=True)
    write_json(output/'run_status.json',dict(completed=completed,attempted=attempted,failed=failed))
    if failed: raise RuntimeError(f'{failed} jobs failed; see runs/*/failure.json')


def analyse(output):
    from .resistance import predictor
    output=Path(output); manifest=verify_plan(output)
    configurations=json.loads((output/'configurations.json').read_text())
    pairs=pd.read_csv(output/'comparisons.csv'); rows=[]; missing=[]
    settings=Settings(**manifest['config']['routing'])
    for config in configurations:
        config_id=config['configuration_id']; states,_=make_states(config['parameters'])
        for pair in pairs.loc[pairs.configuration_id.eq(config_id)].to_dict('records'):
            paths=[output/'runs'/pair[k]/'metrics.json' for k in ('parent_run_id','child_run_id')]
            if not all(p.exists() for p in paths): missing.append(pair); continue
            parent,child=(json.loads(p.read_text()) for p in paths)
            hydro0,hydro1=(pd.read_parquet(p.parent/'hydrograph.parquet') for p in paths)
            if not np.array_equal(hydro0.time_seconds,hydro1.time_seconds): raise ValueError('Paired times differ')
            lag0,lag1=parent['peak_lag_seconds'],child['peak_lag_seconds']
            response=100*(lag1/lag0-1) if lag0>0 else np.nan
            event=hydro0.time_seconds>=parent['buffer_hours']*3600
            difference=hydro1.outlet_cms[event].to_numpy()-hydro0.outlet_cms[event].to_numpy()
            scale=settings.peak_cms-settings.baseflow_cms
            row=dict(**pair,**config['parameters'],example_id=config_id,state_id=pair['child_state'],
                     parent_lag_seconds=lag0,child_lag_seconds=lag1,lag_shift_seconds=lag1-lag0,
                     lag_shift_pct=response,peak_change_pct=100*(child['peak_cms']/parent['peak_cms']-1),
                     hydrograph_rmse_cms=float(np.sqrt(np.mean(difference**2))),
                     hydrograph_normalized_l1=float(np.sum(np.abs(difference))*parent['dt_seconds']/parent['event_volume_in_m3']),
                     peak_speed_change_mps=(config['parameters']['total_length_m']*(1/lag1-1/lag0)) if min(lag0,lag1)>0 else np.nan,
                     parent_capped_segments=parent['n_celerity_capped'],child_capped_segments=child['n_celerity_capped'],
                     parent_volume_error=parent['volume_error_fraction'],child_volume_error=child['volume_error_fraction'],
                     timing_resolution_seconds=parent['dt_seconds'])
            row.update(predictor(states[pair['parent_state']],states[pair['child_state']],pair,settings))
            rows.append(row)
    frame=pd.DataFrame(rows)
    frame.to_csv(output/'results.csv',index=False)
    pd.DataFrame(missing).to_csv(output/'pending_comparisons.csv',index=False)
    write_json(output/'analysis_status.json',dict(completed_comparisons=len(rows),pending_comparisons=len(missing)))
    if not frame.empty: plots(output,frame)
    print(f'Analysed {len(rows)} comparisons; {len(missing)} pending')
    return frame


def plots(output,frame):
    from rapid_tools.response_diagnostics import make_signed_magnitude_scatter, signed_scatter_html
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    figures=output/'figures'; figures.mkdir(exist_ok=True)
    for response,y,label in [('Lag','lag_shift_pct','Lag shift (%)'),('Speed','peak_speed_change_mps','Modeled apparent speed change (m/s)')]:
        for suffix,logx,logy in [('signed',False,False),('log-y',False,True),('log-xy',True,True)]:
            fig=make_signed_magnitude_scatter(frame,x='resistance_predictor',y=y,log_x=logx,log_y=logy,response_name=response,y_label=label)
            (figures/f'{response.lower()}-{suffix}.html').write_text(signed_scatter_html(fig))
        fig=make_signed_magnitude_scatter(frame,x='length_exposure_lambda',y=y,log_x=True,log_y=True,response_name=response,y_label=label,x_label='Length fraction × exposure × resistance ratio')
        (figures/f'{response.lower()}-raw-ratio.html').write_text(signed_scatter_html(fig))
    baseline=dict(total_width_m=500,reference_slope=.0001,outer_a_fraction=.8,inner_fraction=.5,
                  length_multiplier=1.,total_length_m=20000,internal_location='central')
    factor_figure=make_subplots(rows=len(baseline),cols=1,subplot_titles=list(baseline))
    for i,factor in enumerate(baseline,1):
        subset=frame
        for other,value in baseline.items():
            if other!=factor:
                subset=subset.loc[subset[other].eq(value)]
        for state,part in subset.groupby('child_state'):
            if factor=='internal_location':
                part=part.assign(_order=part[factor].map({'upstream':0,'central':1,'downstream':2})).sort_values('_order')
            else:
                part=part.sort_values(factor)
            factor_figure.add_trace(go.Scatter(x=part[factor],y=part.lag_shift_pct,mode='lines+markers',
                name=state,legendgroup=state,showlegend=i==1),row=i,col=1)
        factor_figure.update_yaxes(title_text='Lag shift (%)',row=i,col=1)
    factor_figure.update_layout(height=250*len(baseline),template='plotly_white',
                               title='One-factor slices through the baseline configuration')
    factor_figure.write_html(figures/'factor-response.html')
    # Representative parent/child hydrographs (one configuration); never fabricate routed outputs.
    first=frame.loc[frame.configuration_id.eq(frame.iloc[0].configuration_id)]
    fig=make_subplots(rows=len(first),cols=1,subplot_titles=first.child_state.tolist())
    for i,row in enumerate(first.itertuples(),1):
        for key,label in [('parent_run_id','Original'),('child_run_id','Collapsed')]:
            h=pd.read_parquet(output/'runs'/getattr(row,key)/'hydrograph.parquet')
            metrics=json.loads((output/'runs'/getattr(row,key)/'metrics.json').read_text())
            fig.add_trace(go.Scatter(x=h.time_seconds/3600-metrics['buffer_hours'],y=h.outlet_cms,name=label,legendgroup=label,showlegend=i==1),row=i,col=1)
        fig.update_xaxes(title_text='Hours since pulse begins',range=[-6,96],row=i,col=1)
        fig.update_yaxes(title_text='Discharge (m³/s)',row=i,col=1)
    fig.update_layout(height=260*len(first),template='plotly_white')
    fig.write_html(figures/'hydrographs.html')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['plan','run','analyse','all','validate','reuse'])
    parser.add_argument('--config',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--limit',type=int,help='Maximum unfinished jobs to attempt; rerun to resume')
    parser.add_argument('--reuse-from',type=Path,help='Copy exactly matching completed routing jobs from a compatible prior plan')
    args=parser.parse_args()
    if args.limit is not None and args.limit < 1: parser.error('--limit must be positive')
    if args.command=='reuse' and args.reuse_from is None: parser.error('--reuse-from is required for reuse')
    if args.reuse_from is not None and args.command not in ('plan','all','reuse'):
        parser.error('--reuse-from is supported with plan, all, or reuse')
    if args.command == 'validate':
        if args.config is None: parser.error('--config is required for validate')
        from .validation import validate_baseline
        validate_baseline(args.config,args.output)
        print('Wrote numerical_validation.json')
        return
    if args.command in ('plan','all'):
        if args.config is None: parser.error('--config is required for plan/all')
        m=plan(args.config,args.output)
        print({k:v for k,v in m.items() if k.startswith('n_')})
    if args.reuse_from is not None: reuse_runs(args.output,args.reuse_from)
    if args.command in ('run','all'): run_jobs(args.output,args.limit)
    if args.command in ('analyse','all'): analyse(args.output)

if __name__=='__main__': main()
