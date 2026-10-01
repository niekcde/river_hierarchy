"""Embed fixed synthetic units in longer networks; extend width and length range.

--phase plan validates every geometry and writes a deduplicated routing plan.
--phase pilot routes nine boundary/anchor configurations (63 comparisons).
--phase full resumes all 140 configurations (980 comparisons).
The original package/grid are untouched; only serial approach/exit lengths grow.
"""
from pathlib import Path
from dataclasses import asdict
from itertools import product
from collections import defaultdict
import sys,json,copy,argparse,hashlib,time
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[2]
for path in (ROOT,ROOT/'synthetic_experiments/src',ROOT/'RAPID/src'):sys.path.insert(0,str(path))
from synthetic_experiments.networks import make_states,canonical_network,validate_graph,digest
from synthetic_experiments.routing import Settings,schedule,route
from synthetic_experiments.resistance import predictor
from synthetic_experiments.pipeline import source_fingerprint
from event_duration_verification import duration
from event_shape_metrics import volume_shape
CONFIG=ROOT/'synthetic_experiments/configs/embedded_extension.json'
OUT=ROOT/'synthetic_experiments/outputs/embedded_width_length_extension'


def embed(graph,extent,slope):
    """Retain all internal geometry and change only serial approach/exit lengths."""
    old=20000.;shift=(extent-old)/2
    if extent<old:raise ValueError('This extension adds exterior network length only')
    g=copy.deepcopy(graph)
    for node,data in g.nodes(data=True):
        x=graph.nodes[node]['x']
        data['x']=0. if np.isclose(x,0) else extent if np.isclose(x,old) else x+shift
        data['head_m']=slope*(extent-data['x'])
    for u,v,k,data in g.edges(keys=True,data=True):
        before=graph.edges[u,v,k]
        oldspan=graph.nodes[v]['x']-graph.nodes[u]['x'];newspan=g.nodes[v]['x']-g.nodes[u]['x']
        data['length_m']=before['length_m']+(newspan-oldspan)
        data['length_base_m']=newspan
        data['head_drop_m']=g.nodes[u]['head_m']-g.nodes[v]['head_m']
        data['slope']=data['head_drop_m']/data['length_m']
        data['length_factor']=data['length_m']/newspan
        if graph.nodes[u]['x']>=5000 and graph.nodes[v]['x']<=15000:
            assert np.isclose(data['length_m'],before['length_m']) and data['w']==before['w']
            assert np.isclose(data['slope'],before['slope'],rtol=1e-10)
    validate_graph(g)
    # Unique inlet/outlet and total boundary head follow the prescribed slope.
    inlet=next(n for n in g if g.in_degree(n)==0);outlet=next(n for n in g if g.out_degree(n)==0)
    assert np.isclose(g.nodes[inlet]['head_m']-g.nodes[outlet]['head_m'],slope*extent)
    return g


def plan():
    config=json.loads(CONFIG.read_text());settings=Settings();source=source_fingerprint()
    implementation=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    OUT.mkdir(parents=True,exist_ok=True);(OUT/'runs').mkdir(exist_ok=True)
    jobs={};rows=[];configrows=[]
    for a,m in product(config['outer_a_fraction'],config['length_multiplier']):
        params={k:config[k] for k in ['total_width_m','reference_slope','inner_fraction','internal_location']}
        params.update(total_length_m=20000,outer_a_fraction=a,length_multiplier=m)
        original,pairs=make_states(params)
        resistance={p['child_state']:predictor(original[p['parent_state']],original[p['child_state']],p,settings) for p in pairs}
        for extent in config['network_extent_m']:
            states={key:embed(g,extent,config['reference_slope']) for key,g in original.items()}
            nets={key:canonical_network(g) for key,g in states.items()}
            clock=schedule(nets.values(),settings)
            cid=f'a{a:g}_m{m:g}_L{extent//1000:d}km'
            pilot=(a in [.05,.95] and m in [1,3] and extent in [20000,800000]) or (a==.8 and m==1 and extent==20000)
            ids={}
            for key,net in nets.items():
                job=dict(network=net,settings=asdict(settings),clock=clock,source=source,implementation=implementation)
                rid=digest(job);jobs[rid]=job;ids[key]=rid
            configrows.append(dict(configuration_id=cid,outer_a_fraction=a,length_multiplier=m,network_extent_m=extent,is_pilot=pilot,dt_seconds=clock['dt_seconds']))
            for pair in pairs:
                parent,child=states[pair['parent_state']],states[pair['child_state']]
                r=dict(resistance[pair['child_state']]);domain_length=r['length_fraction']*r['parent_total_channel_length_m']
                # Hydraulic comparison is the unchanged original closed domain.
                # Do not rerun predictor's .25/.75 outer-boundary inference on
                # the embedded network: that would select the wrong domain.
                shift=(extent-20000)/2
                r['domain_start_m']+=shift;r['domain_end_m']+=shift
                for side,g in [('parent',parent),('child',child)]:
                    r[side+'_total_channel_length_m']=sum(x['length_m'] for *_,x in g.edges(data=True))
                    r[side+'_sum_wl']=sum(x['w']*x['length_m'] for *_,x in g.edges(data=True))
                r['length_fraction']=domain_length/r['parent_total_channel_length_m']
                r['resistance_predictor']=r['length_fraction']*r['width_fraction']*(-np.log10(r['Lambda_matched']))
                r['length_exposure_lambda']=r['length_fraction']*r['width_fraction']*r['Lambda_matched']
                newpair=dict(pair,selection_start_m=pair['selection_start_m']+shift,selection_end_m=pair['selection_end_m']+shift)
                rows.append(dict(configuration_id=cid,is_pilot=pilot,outer_a_fraction=a,length_multiplier=m,
                    network_extent_m=extent,total_width_m=config['total_width_m'],reference_slope=config['reference_slope'],
                    inner_fraction=config['inner_fraction'],internal_location=config['internal_location'],
                    outer_unit_span_m=10000,approach_length_m=(extent-10000)/2,exit_length_m=(extent-10000)/2,
                    total_head_drop_m=config['reference_slope']*extent,parent_run_id=ids[pair['parent_state']],child_run_id=ids[pair['child_state']],
                    **newpair,**r))
    d=pd.DataFrame(rows)
    assert len(d)==980 and d[d.is_pilot].shape[0]==63
    assert not d.duplicated(['configuration_id','child_state']).any()
    # Width/multiplier fixed: exact local resistance and exposure stay fixed;
    # larger exterior extent monotonically lowers length fraction and omega.
    for _,g in d.groupby(['outer_a_fraction','length_multiplier','child_state']):
        g=g.sort_values('network_extent_m')
        assert np.allclose(g.Lambda_matched,g.Lambda_matched.iloc[0])
        assert np.allclose(g.width_fraction,g.width_fraction.iloc[0])
        assert g.length_fraction.diff().dropna().lt(0).all()
        if g.resistance_predictor.iloc[0]>0:assert g.resistance_predictor.diff().dropna().lt(0).all()
    plan_id=digest(dict(config=config,source=source,implementation=implementation))
    manifest=dict(plan_id=plan_id,config=config,settings=asdict(settings),source=source,implementation=implementation,
                  n_configurations=len(configrows),n_comparisons=len(d),n_unique_jobs=len(jobs),
                  predictor_method='solve unchanged original matched domain; recompute network length fraction after embedding; preserve domain head drop, widths, lengths and exposure',
                  phase='planned')
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2))
    (OUT/'jobs.json').write_text(json.dumps(jobs))
    d.to_csv(OUT/'plan.csv',index=False);pd.DataFrame(configrows).to_csv(OUT/'configurations.csv',index=False)
    print('Planned',len(configrows),'configurations',len(d),'comparisons',len(jobs),'unique jobs',flush=True)
    return d,jobs,manifest


def postprocess(selected,jobs):
    cache={}
    def load(rid):
        if rid not in cache:
            folder=OUT/'runs'/rid;mt=json.loads((folder/'metrics.json').read_text());h=pd.read_parquet(folder/'hydrograph.parquet')
            dt=mt['dt_seconds'];onset=round(mt['buffer_hours']*3600/dt)*dt;event=h.time_seconds.to_numpy()>=onset
            t=h.time_seconds.to_numpy()[event]-onset+dt/2;q=h.outlet_cms.to_numpy()[event]
            cache[rid]=(t,q,mt)
        return cache[rid]
    rows=[]
    for r in selected.to_dict('records'):
        tp,qp,p=load(r['parent_run_id']);tc,qc,c=load(r['child_run_id']);base=jobs[r['parent_run_id']]['settings']['baseflow_cms']
        amp=qp.max()-base
        for fraction,label in [(.005,'005'),(.01,'010'),(.02,'020')]:
            a=duration(tp,qp,base,base,fraction*amp,fraction*amp);b=duration(tc,qc,base,base,fraction*amp,fraction*amp)
            r['duration_change_'+label+'_pct']=100*(b['duration_seconds']/a['duration_seconds']-1)
        r.update(omega=r['resistance_predictor'],lag_shift_pct=100*(c['peak_lag_seconds']/p['peak_lag_seconds']-1),
            lag_shift_seconds=c['peak_lag_seconds']-p['peak_lag_seconds'],parent_lag_seconds=p['peak_lag_seconds'],
            event_duration_change_pct=r['duration_change_010_pct'],peak_excess_change_pct=100*((qc.max()-base)/amp-1),
            parent_capped_segments=p['n_celerity_capped'],child_capped_segments=c['n_celerity_capped'],
            parent_floored_segments=p['n_length_floored'],child_floored_segments=c['n_length_floored'],
            timing_resolution_seconds=max(p['dt_seconds'],c['dt_seconds']),
            max_volume_error=max(abs(p['volume_error_fraction']),abs(c['volume_error_fraction'])))
        r['within_one_dt']=abs(r['lag_shift_seconds'])<=r['timing_resolution_seconds'];rows.append(r)
    return pd.DataFrame(rows)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--phase',choices=['plan','pilot','full'],default='pilot');args=parser.parse_args()
    d,jobs,manifest=plan()
    if args.phase=='plan':return
    selected=d[d.is_pilot].copy() if args.phase=='pilot' else d
    needed=sorted(set(selected.parent_run_id)|set(selected.child_run_id));start=time.perf_counter()
    for number,rid in enumerate(needed,1):
        folder=OUT/'runs'/rid;folder.mkdir(exist_ok=True)
        if not ((folder/'metrics.json').exists() and (folder/'hydrograph.parquet').exists()):
            job=jobs[rid];h,mt,segments=route(job['network'],Settings(**job['settings']),job['clock'])
            h.to_parquet(folder/'hydrograph.parquet',index=False)
            (folder/'metrics.json').write_text(json.dumps(mt,indent=2))
        if number%5==0:print(args.phase,number,'/',len(needed),'jobs;',round(time.perf_counter()-start,1),'s',flush=True)
    result=postprocess(selected,jobs);result.to_csv(OUT/f'{args.phase}_results.csv',index=False)
    manifest.update(phase=args.phase+'_complete',completed_comparisons=len(result),completed_unique_jobs=len(needed),wall_seconds=time.perf_counter()-start)
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2))
    print(result.groupby('child_state')[['omega','lag_shift_pct','event_duration_change_pct','peak_excess_change_pct']].agg(['min','max']).round(4).to_string(),flush=True)
    print(args.phase,'complete;',len(result),'comparisons;',round(time.perf_counter()-start,1),'s',flush=True)
if __name__=='__main__':main()
