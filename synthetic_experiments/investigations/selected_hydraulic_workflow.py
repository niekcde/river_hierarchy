"""Versioned selected-alpha synthetic grid and extension; preserve legacy outputs.

CLI: plan | run | analyse | all. Restartable jobs, bounded process workers,
full settings in each job, original geometry and validation partitions retained.
"""
from pathlib import Path
from dataclasses import asdict
from types import FunctionType
from concurrent.futures import ProcessPoolExecutor,as_completed
from functools import lru_cache
import argparse,json,hashlib,os,sys,math
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from width_depth_scaling import ROOT,SELECTED,SELECTED_PATH,adjusted,compact,predictor,routing,make_states,canonical_network,digest
from synthetic_experiments.pipeline import write_json,source_fingerprint
from rapid_tools.reference_section import predict_based_depth,DEFAULT_BASED_MODEL_PATH
from embedded_width_length_extension import embed
from segmentation_sensitivity import run_frame
from event_duration_verification import duration
from event_shape_metrics import volume_shape
VERSION=ROOT/'synthetic_experiments/outputs/hydraulic_alpha065_20260930'
OLD=ROOT/'synthetic_experiments/outputs'


def implementation():
    names=['selected_hydraulic_workflow.py','width_depth_scaling.py','segmentation_sensitivity.py','embedded_width_length_extension.py']
    return dict(source=source_fingerprint(),files={n:hashlib.sha256((Path(__file__).parent/n).read_bytes()).hexdigest() for n in names},selection=SELECTED,model_sha256=hashlib.sha256(DEFAULT_BASED_MODEL_PATH.read_bytes()).hexdigest())


@lru_cache(None)
def reference(w,slope):
    with threadpool_limits(limits=1):d=predict_based_depth(discharge_cms=1000,width_m=w,slope=slope)
    return w/d


def settings(params):
    return routing.Settings(kb=reference(params['total_width_m'],params['reference_slope']),n_manning=SELECTED['n_manning'],x=SELECTED['x'],target_length_m=SELECTED['target_length_m'],effective_length_floor_m=SELECTED['effective_length_floor_m'])


def clock_for(nets,s,w):
    env=dict(routing.schedule.__globals__);env['compact_k']=lambda net,sett:compact(net,sett,w,SELECTED['depth_alpha'])
    return FunctionType(routing.schedule.__code__,env,'selected_schedule',routing.schedule.__defaults__)(nets,s)


def configurations(dataset):
    if dataset=='full_grid_island':return json.loads((OLD/dataset/'configurations.json').read_text())
    cfg=json.loads((ROOT/'synthetic_experiments/configs/embedded_extension.json').read_text())
    base={k:cfg[k] for k in ['total_width_m','reference_slope','inner_fraction','internal_location']}
    return [dict(configuration_id=r['configuration_id'],parameters=dict(base,total_length_m=20000,outer_a_fraction=r['outer_a_fraction'],length_multiplier=r['length_multiplier']),network_extent_m=r['network_extent_m'],is_pilot=r['is_pilot']) for r in pd.read_csv(OLD/dataset/'configurations.csv').to_dict('records')]


def plan(dataset):
    out=VERSION/dataset;out.mkdir(parents=True,exist_ok=True);impl=implementation();pid=digest(dict(dataset=dataset,implementation=impl,configs=configurations(dataset)))
    if (out/'manifest.json').exists():
        manifest=json.loads((out/'manifest.json').read_text());assert manifest['plan_id']==pid,'Implementation changed; use another version directory'
        return manifest
    jobs={};pairs=[];state_rows=[];configs=[]
    for idx,c in enumerate(configurations(dataset)):
        params=c['parameters'];w=params['total_width_m'];s=settings(params);graphs,comparisons=make_states(params)
        if 'network_extent_m' in c:graphs={k:embed(g,c['network_extent_m'],params['reference_slope']) for k,g in graphs.items()}
        nets={k:canonical_network(g) for k,g in graphs.items()};clock=clock_for(nets.values(),s,w)
        runids={}
        for state,net in nets.items():
            job=dict(network=net,settings=asdict(s),clock={k:clock[k] for k in ['dt_seconds','buffer_hours']},hydraulic_reference=dict(width_m=w,depth_alpha=SELECTED['depth_alpha']),source=digest(impl))
            rid=digest(job);jobs[rid]=dict(run_id=rid,**job);runids[state]=rid
            state_rows.append(dict(configuration_id=c['configuration_id'],state_id=state,run_id=rid))
        configs.append(dict(c,clock=clock,settings=asdict(s)))
        for pair in comparisons:
            row=dict(configuration_id=c['configuration_id'],**params,**pair,parent_run_id=runids[pair['parent_state']],child_run_id=runids[pair['child_state']])
            if 'network_extent_m' in c:
                shift=(c['network_extent_m']-20000)/2
                row.update(network_extent_m=c['network_extent_m'],is_pilot=c['is_pilot'],outer_unit_span_m=10000,approach_length_m=(c['network_extent_m']-10000)/2,exit_length_m=(c['network_extent_m']-10000)/2,total_head_drop_m=c['network_extent_m']*params['reference_slope'])
                row['selection_start_m']+=shift;row['selection_end_m']+=shift
            pairs.append(row)
        if idx%500==0:print('PLAN',dataset,idx,len(jobs),flush=True)
    for rid,job in jobs.items():write_json(out/'jobs'/f'{rid}.json',job)
    pd.DataFrame(pairs).to_csv(out/'comparisons.csv',index=False);pd.DataFrame(pairs).to_csv(out/'plan.csv',index=False)
    pd.DataFrame(state_rows).to_csv(out/'states.csv',index=False);write_json(out/'configurations.json',configs)
    manifest=dict(plan_id=pid,source=digest(impl),implementation=impl,config=dict(routing=asdict(routing.Settings(n_manning=.035)),selected_hydraulic_reference=SELECTED),dataset=dataset,n_configurations=len(configs),n_comparisons=len(pairs),n_unique_runs=len(jobs),run_ids=list(jobs),status='planned',note='Per-configuration BASED kb is stored in each job; config routing kb is not a global applied kb.')
    write_json(out/'manifest.json',manifest);return manifest


def worker(task):
    out,rid=task;out=Path(out);folder=out/'runs'/rid;job=json.loads((out/'jobs'/f'{rid}.json').read_text())
    try:
        s=routing.Settings(**job['settings']);ref=job['hydraulic_reference'];frame=adjusted(routing.prepare(job['network'],s),s,ref['width_m'],ref['depth_alpha'])
        h,mt,frame=run_frame(job['network'],frame,s,job['clock'])
        folder.mkdir(parents=True,exist_ok=True);h.to_parquet(folder/'hydrograph.parquet',index=False)
        frame.drop(columns='geometry',errors='ignore').to_csv(folder/'segments.csv',index=False)
        write_json(folder/'metrics.json',dict(run_id=rid,**mt));return rid,None
    except Exception as exc:
        import traceback
        write_json(folder/'failure.json',dict(error=str(exc),traceback=traceback.format_exc()));return rid,str(exc)


def run(dataset,workers):
    out=VERSION/dataset;m=plan(dataset);todo=[]
    for rid in m['run_ids']:
        folder=out/'runs'/rid
        if all((folder/f).exists() for f in ['metrics.json','hydrograph.parquet','segments.csv']):
            mt=json.loads((folder/'metrics.json').read_text());assert mt['run_id']==rid
            assert max(abs(mt['volume_error_fraction']),mt['pre_event_relative_error'],mt['final_relative_error'])<1e-4
        else:todo.append((str(out),rid))
    done=len(m['run_ids'])-len(todo);failed=[]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures=[pool.submit(worker,t) for t in todo]
        for f in as_completed(futures):
            rid,error=f.result();done+=1
            if error:failed.append((rid,error));print('FAIL',rid,error,flush=True)
            if done%100==0:print('RUN',dataset,done,'/',len(m['run_ids']),flush=True)
    write_json(out/'run_status.json',dict(completed=done-len(failed),failed=failed));assert not failed
    m['status']='routing_complete';write_json(out/'manifest.json',m)


@lru_cache(maxsize=512)
def load(out,rid):
    folder=Path(out)/'runs'/rid;mt=json.loads((folder/'metrics.json').read_text());h=pd.read_parquet(folder/'hydrograph.parquet')
    onset=round(mt['buffer_hours']*3600/mt['dt_seconds'])*mt['dt_seconds'];mask=h.time_seconds>=onset
    t=h.time_seconds[mask].to_numpy()-onset;q=h.outlet_cms[mask].to_numpy()-100
    from scipy.signal import find_peaks
    shape=dict(peak_excess_cms=float(q.max()),screened_peak_count=len(find_peaks(q,prominence=9)[0]),**volume_shape(t,q,mt['dt_seconds']))
    return h,mt,shape,t,q


def analyse(dataset):
    out=VERSION/dataset;m=plan(dataset);rows=[];pairs=pd.read_csv(out/'comparisons.csv')
    for idx,c in enumerate(json.loads((out/'configurations.json').read_text())):
        params=c['parameters'];graphs,comparisons=make_states(params);s=routing.Settings(**c['settings']);bychild={p['child_state']:p for p in comparisons}
        for r in pairs[pairs.configuration_id==c['configuration_id']].to_dict('records'):
            ph,pm,ps,pt,pq=load(str(out),r['parent_run_id']);ch,cm,cs,ct,cq=load(str(out),r['child_run_id'])
            assert np.array_equal(ph.time_seconds,ch.time_seconds)
            original_pair=bychild[r['child_state']]
            pred=predictor(graphs[r['parent_state']],graphs[r['child_state']],original_pair,s,params['total_width_m'],SELECTED['depth_alpha'])
            if 'network_extent_m' in c:
                shift=(c['network_extent_m']-20000)/2;domain_length=pred['length_fraction']*pred['parent_total_channel_length_m']
                pred['domain_start_m']+=shift;pred['domain_end_m']+=shift
                for side in ['parent','child']:
                    g=embed(graphs[r[side+'_state']],c['network_extent_m'],params['reference_slope'])
                    pred[side+'_total_channel_length_m']=sum(d['length_m'] for *_,d in g.edges(data=True));pred[side+'_sum_wl']=sum(d['length_m']*d['w'] for *_,d in g.edges(data=True))
                pred['length_fraction']=domain_length/pred['parent_total_channel_length_m'];pred['resistance_predictor']=pred['length_fraction']*pred['width_fraction']*(-np.log10(pred['Lambda_matched']));pred['length_exposure_lambda']=pred['length_fraction']*pred['width_fraction']*pred['Lambda_matched']
            ds={}
            for tol,label in [(.005,'005'),(.01,'010'),(.02,'020')]:
                threshold=tol*ps['peak_excess_cms'];p=duration(pt+pm['dt_seconds']/2,pq+100,100,100,threshold,threshold);chd=duration(ct+cm['dt_seconds']/2,cq+100,100,100,threshold,threshold)
                ds['duration_change_'+label+'_pct']=100*(chd['duration_seconds']/p['duration_seconds']-1)
            lag0=pm['peak_lag_seconds'];lag1=cm['peak_lag_seconds'];dt=pm['dt_seconds'];delta=cq-pq
            row=dict(r,**pred,example_id=r['configuration_id'],state_id=r['child_state'],parent_lag_seconds=lag0,child_lag_seconds=lag1,lag_shift_seconds=lag1-lag0,lag_shift_pct=100*(lag1/lag0-1),
                peak_change_pct=100*(cm['peak_cms']/pm['peak_cms']-1),peak_excess_change_pct=100*(cs['peak_excess_cms']/ps['peak_excess_cms']-1),temporal_spread_change_pct=100*(cs['temporal_sd_seconds']/ps['temporal_sd_seconds']-1),event_duration_change_pct=ds['duration_change_010_pct'],
                volume_quantile_duration_change_pct=100*(cs['duration_t01_t99_seconds']/ps['duration_t01_t99_seconds']-1),hydrograph_rmse_cms=float(np.sqrt(np.mean(delta**2))),hydrograph_normalized_l1=float(np.sum(abs(delta))*dt/pm['event_volume_in_m3']),
                peak_speed_change_mps=params['total_length_m']*(1/lag1-1/lag0),timing_resolution_seconds=dt,dt_seconds=dt,within_one_dt=abs(lag1-lag0)<=dt,omega=pred['resistance_predictor'],depth_alpha=SELECTED['depth_alpha'],reference_kb=s.kb,n_manning=s.n_manning,**ds)
            for side,mt,shape in [('parent',pm,ps),('child',cm,cs)]:
                row.update({side+'_'+k:v for k,v in shape.items()});row.update({side+'_capped_segments':mt['n_celerity_capped'],side+'_volume_error':mt['volume_error_fraction'],side+'_n_celerity_capped':mt['n_celerity_capped'],side+'_n_length_floored':mt['n_length_floored']})
            row['any_cap']=row['parent_capped_segments']+row['child_capped_segments']>0;row['any_floor']=row['parent_n_length_floored']+row['child_n_length_floored']>0
            rows.append(row)
        if idx%300==0:print('ANALYSE',dataset,idx,flush=True)
    d=pd.DataFrame(rows);assert len(d)==m['n_comparisons'];d.to_csv(out/'results.csv',index=False)
    if dataset=='embedded_width_length_extension':d.to_csv(out/'full_results.csv',index=False)
    else:
        # Reuse only geometry descriptors and original group/fold assignments.
        old=pd.read_parquet(OLD/dataset/'factor_analysis/geometry_predictors/slope_path_arrangement/features.parquet')
        descriptor_names=['equivalent_length','equivalent_width','elongation','path_length_cv','path_length_range_norm','effective_n_paths_width','dominant_width_fraction','topologic_complexity_score','branching_density_by_length','n_valid_paths','width_entropy','width_evenness']
        cols=['configuration_id','child_state','geometry_group','split','development_fold','parent_path_length_mean','signed_length_width_association','domain_reference_slope','domain_head_loss_m']+[prefix+'_'+v for prefix in ['parent','child','delta'] for v in descriptor_names]
        features=d.merge(old[cols],on=['configuration_id','child_state'],validate='one_to_one')
        features['resistance_log']=-np.log10(features.Lambda_matched)
        features['length_exposure']=features.length_fraction*features.width_fraction;features['length_resistance']=features.length_fraction*features.resistance_log;features['exposure_resistance']=features.width_fraction*features.resistance_log
        (out/'current_analysis').mkdir(exist_ok=True);features.to_parquet(out/'current_analysis/current_comparisons.parquet',index=False)
        write_json(out/'current_analysis/geometry_reuse.json',dict(source=str(OLD/dataset/'factor_analysis/geometry_predictors/slope_path_arrangement/features.parquet'),columns=cols,source_sha256=hashlib.sha256((OLD/dataset/'factor_analysis/geometry_predictors/slope_path_arrangement/features.parquet').read_bytes()).hexdigest(),note='Unchanged geometry and exact previous splits only; all routing responses and hydraulic resistance recomputed.'))
    m['status']='analysis_complete';write_json(out/'manifest.json',m)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['plan','run','analyse','all']);parser.add_argument('--workers',type=int,default=4);args=parser.parse_args()
    for dataset in ['full_grid_island','embedded_width_length_extension']:
        if args.command in ['plan','all']:plan(dataset)
        if args.command in ['run','all']:run(dataset,args.workers)
        if args.command in ['analyse','all']:analyse(dataset)
if __name__=='__main__':main()
