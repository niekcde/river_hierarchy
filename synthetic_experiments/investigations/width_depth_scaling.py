"""Diagnostic BASED-anchored width/depth scaling; production defaults unchanged.

alpha=1 reproduces constant kb. Alpha changes depth, never allocation width.
Preflight original factorial grid and stored real base networks; route the seven
focused geometries and recompute matched Chezy resistance with local kb.
"""
from pathlib import Path
from dataclasses import replace, asdict
from types import FunctionType
import sys, json, math, hashlib
import numpy as np
import pandas as pd
import geopandas as gpd
from threadpoolctl import threadpool_limits
ROOT=Path(__file__).resolve().parents[2]
for p in [ROOT,ROOT/'synthetic_experiments/src',ROOT/'RAPID/src']: sys.path.insert(0,str(p))
from synthetic_experiments import routing, resistance
from synthetic_experiments.networks import make_states,canonical_network,digest
from synthetic_experiments.pipeline import write_json,source_fingerprint
from rapid_tools.reference_section import predict_based_depth,DEFAULT_BASED_MODEL_PATH
from rapid_tools.prep import _compute_subreach_count
from rapid_tools.k_values import compute_dt_bounds
from segmentation_sensitivity import run_frame
from manning_sensitivity import configurations,Y
from event_duration_verification import duration
OUT=ROOT/'synthetic_experiments/outputs/width_depth_scaling'
SELECTED_PATH=ROOT/'synthetic_experiments/configs/selected_hydraulic_reference.json'
SELECTED=json.loads(SELECTED_PATH.read_text())
ALPHAS=sorted(set([1.,.8,.71,.6,.5,float(SELECTED['depth_alpha'])]),reverse=True)


def adjusted(frame,s,wref,alpha):
    frame=frame.copy()
    width=frame.width.to_numpy()
    depth=wref/s.kb*(width/wref)**alpha
    raw=5/(3*s.n_manning)*np.sqrt(frame.slope_used.to_numpy())*depth**(2/3)
    c=np.clip(raw,s.min_celerity_mps,s.max_celerity_mps)
    frame['diagnostic_depth_m']=depth
    frame['rapid_kb_value']=width/depth
    frame['rapid_celerity_mps_raw']=raw
    frame['rapid_celerity_mps']=c
    frame['rapid_celerity_capped']=~np.isclose(raw,c)
    frame['rapid_k']=frame.rapid_effective_length_m/c
    frame['rapid_branch_width_m']=width
    return frame


def compact(net,s,wref,alpha):
    return adjusted(routing.compact_k(net,s),s,wref,alpha)


def predictor(parent,child,pair,s,wref,alpha):
    # Retain the actual widths for exposure, sum WL and selection; only change
    # local kb passed into the existing resistance equation/solver.
    environment=dict(resistance.solve.__globals__)
    def local_k(length,width,cz,kb):
        return resistance.compute_K_chezy(length,width,cz,kb*(np.asarray(width)/wref)**(1-alpha))
    environment['compute_K_chezy']=local_k
    solve=FunctionType(resistance.solve.__code__,environment,'solve_local_kb')
    env=dict(resistance.predictor.__globals__);env['solve']=solve
    return FunctionType(resistance.predictor.__code__,env,'predictor_local_kb')(parent,child,pair,s)


def width_shares(net):
    import networkx as nx
    g=nx.MultiDiGraph()
    for i,(u,v,L,w,slope) in enumerate(net['edges']):g.add_edge(u,v,key=i,width=w)
    shares={n:0. for n in g};shares[0]=1.;flows={}
    for u in nx.topological_sort(g):
        outgoing=list(g.out_edges(u,keys=True,data=True));total=sum(d['width'] for *_,d in outgoing)
        for _,v,i,d in outgoing:
            f=shares[u]*d['width']/total;flows[i]=f;shares[v]+=f
    return np.array([flows[i] for i in range(len(net['edges']))])


def edge_hydraulics(length,width,slope,s,wref,alpha):
    depth=wref/s.kb*(width/wref)**alpha
    raw=5/(3*s.n_manning)*np.sqrt(slope)*depth**(2/3)
    c=np.clip(raw,s.min_celerity_mps,s.max_celerity_mps)
    counts=np.array([_compute_subreach_count(float(L),s.target_length_m) for L in length])
    effective=np.maximum(length/counts,s.effective_length_floor_m)
    return dict(depth=depth,raw=raw,c=c,counts=counts,effective=effective,K=effective/c)


def bound_row(length,width,slope,s,wref,alpha):
    h=edge_hydraulics(length,width,slope,s,wref,alpha);K=h['K'];depth=h['depth']
    lo,hi=2*s.x*float(K.max()),2*(1-s.x)*float(K.min())
    slow=int(K.argmax());fast=int(K.argmin())
    return dict(dt_lower_seconds=lo,dt_upper_seconds=hi,feasible=bool(math.ceil(lo+1e-8)<=math.floor(hi)),min_depth_m=float(depth.min()),max_depth_m=float(depth.max()),
        min_width_over_reference=float(width.min()/wref),slow_edge=slow,fast_edge=fast,
        slow_width_m=float(width[slow]),slow_slope=float(slope[slow]),slow_depth_m=float(depth[slow]),slow_celerity_mps=float(h['c'][slow]),slow_effective_length_m=float(h['effective'][slow]),
        fast_width_m=float(width[fast]),fast_slope=float(slope[fast]),fast_depth_m=float(depth[fast]),fast_celerity_mps=float(h['c'][fast]),fast_effective_length_m=float(h['effective'][fast]))


def preflight():
    rows=[];state_rows=[];pair_rows=[];ref_cache={}
    configs=json.loads((ROOT/'synthetic_experiments/outputs/full_grid_island/configurations.json').read_text())
    # Include the integrated low-impact extension where configurations are available.
    extra=ROOT/'synthetic_experiments/outputs/embedded_width_length_extension/configurations.json'
    sources=[('full_grid_island',configs)]
    if extra.exists():sources.append(('embedded_width_length_extension',json.loads(extra.read_text())))
    extra_csv=extra.with_suffix('.csv')
    if not extra.exists() and extra_csv.exists():
        cfg=json.loads((ROOT/'synthetic_experiments/configs/embedded_extension.json').read_text())
        base={k:cfg[k] for k in ['total_width_m','reference_slope','inner_fraction','internal_location']}
        extra_configs=[dict(configuration_id=r['configuration_id'],parameters=dict(base,total_length_m=20000,outer_a_fraction=r['outer_a_fraction'],length_multiplier=r['length_multiplier']),embedded_extent_m=r['network_extent_m']) for r in pd.read_csv(extra_csv).to_dict('records')]
        sources.append(('embedded_width_length_extension',extra_configs))
    for dataset,configs in sources:
        for idx,config in enumerate(configs):
            params=config['parameters'];w=params['total_width_m'];slope=params['reference_slope']
            key=(w,slope)
            if key not in ref_cache:
                with threadpool_limits(limits=1):ref_cache[key]=predict_based_depth(discharge_cms=1000,width_m=w,slope=slope)
            s=routing.Settings(kb=w/ref_cache[key],n_manning=.035)
            states,pairs=make_states(params)
            if 'embedded_extent_m' in config:
                from embedded_width_length_extension import embed
                states={k:embed(g,config['embedded_extent_m'],slope) for k,g in states.items()}
            nets={k:canonical_network(net) for k,net in states.items()}
            edges=np.array([e for net in nets.values() for e in net['edges']])
            for alpha in ALPHAS:
                identity=dict(dataset=dataset,configuration_id=config['configuration_id'],alpha=alpha,embedded_extent_m=config.get('embedded_extent_m',params['total_length_m']),**params)
                rows.append(dict(**identity,**bound_row(edges[:,2],edges[:,3],edges[:,4],s,w,alpha)))
                sb={}
                for state,net in nets.items():
                    e=np.array(net['edges']);sb[state]=bound_row(e[:,2],e[:,3],e[:,4],s,w,alpha)
                    state_rows.append(dict(**identity,state=state,network_id=digest(net),reference_kb=s.kb,**sb[state]))
                for pair in pairs:
                    bp=sb[pair['parent_state']];bc=sb[pair['child_state']]
                    lo=max(bp['dt_lower_seconds'],bc['dt_lower_seconds']);hi=min(bp['dt_upper_seconds'],bc['dt_upper_seconds'])
                    pair_rows.append(dict(**identity,**pair,dt_lower_seconds=lo,dt_upper_seconds=hi,
                        feasible=math.ceil(lo+1e-8)<=math.floor(hi),parent_feasible=bp['feasible'],child_feasible=bc['feasible']))
            if idx%500==0:print('preflight',dataset,idx,flush=True)
    pd.DataFrame(rows).to_csv(OUT/'grid_preflight.csv',index=False)
    pd.DataFrame(state_rows).to_csv(OUT/'state_preflight.csv',index=False)
    pd.DataFrame(pair_rows).to_csv(OUT/'pair_preflight.csv',index=False)
    real=[];real_edges=[];real_sources=[]
    for ex in ['03','21','55','56']:
        root=ROOT/f'network_variants/output/{ex}_indep__anchored'
        ref=json.loads((root/'reference_section/reference_kb_summary.json').read_text())
        links=gpd.read_file(root/'states/S000_base/variant/widths/links_with_width_families.gpkg',ignore_geometry=True)
        slopes=pd.read_csv(root/'states/S000_base/hydraulic_context/link_hydraulic_context.csv')
        d=slopes[['id_link','link_length_m','slope_used']].merge(links[['id_link','wid_adj_routing']],on='id_link',validate='one_to_one')
        assert np.isfinite(d.to_numpy()).all() and (d[['link_length_m','slope_used','wid_adj_routing']]>0).all().all()
        s=routing.Settings(kb=ref['kb_value'],n_manning=.035)
        source_paths=[root/'reference_section/reference_kb_summary.json',root/'states/S000_base/variant/widths/links_with_width_families.gpkg',root/'states/S000_base/hydraulic_context/link_hydraulic_context.csv']
        real_sources.extend(dict(example=ex,path=str(p.relative_to(ROOT)),sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in source_paths)
        for alpha in ALPHAS:
            b=bound_row(d.link_length_m.to_numpy(),d.wid_adj_routing.to_numpy(),d.slope_used.to_numpy(),s,ref['reference_width_m'],alpha)
            real.append(dict(example=ex,alpha=alpha,n_links=len(d),reference_width_m=ref['reference_width_m'],reference_depth_m=ref['reference_depth_m'],reference_slope=ref['reference_slope'],
                width_refresh_marker=(root/'rapid_width_refresh_required.json').exists(),
                slow_link_id=int(d.iloc[b['slow_edge']].id_link),fast_link_id=int(d.iloc[b['fast_edge']].id_link),**b))
            h=edge_hydraulics(d.link_length_m.to_numpy(),d.wid_adj_routing.to_numpy(),d.slope_used.to_numpy(),s,ref['reference_width_m'],alpha)
            for i,r in enumerate(d.to_dict('records')):
                real_edges.append(dict(example=ex,alpha=alpha,**r,width_over_reference=r['wid_adj_routing']/ref['reference_width_m'],slope_over_reference=r['slope_used']/ref['reference_slope'],
                    depth_m=h['depth'][i],celerity_raw_mps=h['raw'][i],celerity_mps=h['c'][i],segment_count=h['counts'][i],effective_length_m=h['effective'][i],K_seconds=h['K'][i],
                    controls_lower=i==b['slow_edge'],controls_upper=i==b['fast_edge'],blocks_fast_bound=2*s.x*h['K'][i]>b['dt_upper_seconds']))
    pd.DataFrame(real).to_csv(OUT/'real_base_preflight.csv',index=False)
    pd.DataFrame(real_edges).to_csv(OUT/'real_edge_preflight.csv',index=False)
    write_json(OUT/'real_sources.json',real_sources)


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    source=source_fingerprint();script_hash=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    manifest=dict(status='running',reference='HYD-2026-09-29-WIDTH-DEPTH',alphas=ALPHAS,source=source,script_sha256=script_hash,
        selected_synthetic_alpha=SELECTED['depth_alpha'],selection_reference=SELECTED['decision_reference'],
        selection_config_sha256=hashlib.sha256(SELECTED_PATH.read_bytes()).hexdigest(),real_case_decision=SELECTED['real_case_decision'],
        based_model_sha256=hashlib.sha256(DEFAULT_BASED_MODEL_PATH.read_bytes()).hexdigest(),
        formula='D=Dref*(W/Wref)^alpha; kb=kbref*(W/Wref)^(1-alpha)',
        controls='n=.035, X=.1, target/floor=200m, original forcing and width allocation; one reference per configuration shared by all states',
        resistance='Recomputed matched Chezy resistance with local kb; constant Cz retained. This is not Manning-equivalent resistance.',
        real_scope='Stored base networks 03/21/55/56 only; nominal 200m target/floor and n=.035; no real rerouting or full parent/child schedule. Width-refresh markers retained.',
        routing_scope='Seven focused configurations; each exponent has a shared all-state clock. Additional common-clock controls where feasible.',
        literature='Li & Limaye 2025, doi:10.1029/2024JF008196 eq9 exponent .71. BASED anchoring is our modelling assumption; .5 is stronger moderation sensitivity, not calibrated.')
    write_json(OUT/'manifest.json',manifest)
    preflight()
    references=pd.read_csv(ROOT/'synthetic_experiments/outputs/manning_sensitivity/references.csv').set_index('case')
    results=[];state_rows=[];edge_rows=[];failures=[];bounds=[];parity=0
    for case,params in configurations():
        wref=params['total_width_m'];s=routing.Settings(kb=float(references.loc[case,'kb']),n_manning=.035)
        states,pairs=make_states(params);nets={k:canonical_network(v) for k,v in states.items()}
        baseframes={k:routing.prepare(n,s) for k,n in nets.items()}
        frames={alpha:{k:adjusted(f,s,wref,alpha) for k,f in baseframes.items()} for alpha in ALPHAS}
        clocks={}
        for alpha in ALPHAS:
            env=dict(routing.schedule.__globals__);env['compact_k']=lambda n,_s:compact(n,_s,wref,alpha)
            schedule=FunctionType(routing.schedule.__code__,env,'schedule_scaled',routing.schedule.__defaults__)
            bs=[compute_dt_bounds(f.rapid_k,x_value=s.x) for f in frames[alpha].values()]
            lo=max(b[0] for b in bs);hi=min(b[1] for b in bs)
            bounds.append(dict(case=case,alpha=alpha,dt_lower_seconds=lo,dt_upper_seconds=hi,feasible=math.ceil(lo+1e-8)<=math.floor(hi)))
            for state,net in nets.items():
                f=compact(net,s,wref,alpha);share=width_shares(net)
                depth=f.diagnostic_depth_m.to_numpy();q=share*s.peak_cms
                qm=f.width.to_numpy()*depth**(5/3)*np.sqrt(f.slope_used.to_numpy())/s.n_manning
                for i,r in enumerate(f.to_dict('records')):
                    edge_rows.append(dict(case=case,state=state,alpha=alpha,edge=i,**r,reference_q_allocated_cms=q[i],manning_implied_q_cms=qm[i],qm_over_allocated=qm[i]/q[i],velocity_allocated_mps=q[i]/(r['width']*depth[i])))
            try:clocks[alpha]=schedule(nets.values(),s)
            except ValueError as exc:
                if 'No common stable integer timestep' not in str(exc):raise
                failures.append(dict(case=case,alpha=alpha,reason=str(exc)));print(case,alpha,'INFEASIBLE',flush=True)
        # Identical clocks isolate hydraulic scaling from changes in timestep.
        common_lo=max(c['dt_min_seconds'] for c in clocks.values());common_hi=min(c['dt_max_seconds'] for c in clocks.values())
        shared=None
        if math.ceil(common_lo+1e-8)<=math.floor(common_hi):
            shared=dict(next(iter(clocks.values())),dt_seconds=math.ceil(common_lo+1e-8),dt_min_seconds=common_lo,dt_max_seconds=common_hi,
                buffer_hours=max(c['buffer_hours'] for c in clocks.values()),max_path_k_seconds=max(c['max_path_k_seconds'] for c in clocks.values()))
        for mode in ['native_clock','common_clock']:
            if mode=='common_clock' and shared is None:continue
            for alpha,ownclock in clocks.items():
                clock=ownclock if mode=='native_clock' else shared;outputs={}
                for state,net in nets.items():
                    frame=frames[alpha][state]
                    job=dict(network=net,settings=asdict(s),reference_width_m=wref,alpha=alpha,clock=clock,source=source,script_sha256=script_hash)
                    rid=digest(job);folder=OUT/'runs'/rid
                    if (folder/'metrics.json').exists():
                        assert json.loads((folder/'job.json').read_text())==job
                        mt=json.loads((folder/'metrics.json').read_text());h=pd.read_parquet(folder/'hydrograph.parquet')
                    else:
                        h,mt,_=run_frame(net,frame,s,clock)
                        folder.mkdir(parents=True,exist_ok=True);write_json(folder/'job.json',job)
                        h.to_parquet(folder/'hydrograph.parquet',index=False);write_json(folder/'metrics.json',mt)
                    outputs[state]=(rid,h,mt)
                    state_rows.append(dict(case=case,alpha=alpha,clock_mode=mode,state=state,run_id=rid,**mt))
                    if alpha==1 and mode=='native_clock' and case=='baseline' and state in ['centerline/original','simple/original','nested/original']:
                        h0,m0,_=routing.route(net,s,clock)
                        assert np.allclose(frame.rapid_k,baseframes[state].rapid_k,rtol=1e-12)
                        assert np.allclose(h0.outlet_cms,h.outlet_cms,rtol=1e-10,atol=1e-8);parity+=1
                for pair in pairs:
                    pid,ph,pm=outputs[pair['parent_state']];cid,ch,cm=outputs[pair['child_state']]
                    amp=pm['peak_cms']-s.baseflow_cms;spans={}
                    for tolerance in [.005,.01,.02]:
                        ds=[]
                        for h in [ph,ch]:
                            t=h.time_seconds.to_numpy()-routing.buffer_seconds(clock)+clock['dt_seconds']/2;mask=t>=0
                            ds.append(duration(t[mask],h.outlet_cms.to_numpy()[mask],s.baseflow_cms,s.baseflow_cms,tolerance*amp,tolerance*amp)['duration_seconds'])
                        spans[tolerance]=ds
                    pred=predictor(states[pair['parent_state']],states[pair['child_state']],pair,s,wref,alpha)
                    if alpha==1:
                        old=resistance.predictor(states[pair['parent_state']],states[pair['child_state']],pair,s)
                        assert np.isclose(pred['Lambda_matched'],old['Lambda_matched'],rtol=1e-9)
                    results.append(dict(case=case,alpha=alpha,clock_mode=mode,**params,**pair,**pred,parent_run_id=pid,child_run_id=cid,dt_seconds=clock['dt_seconds'],
                        lag_shift_pct=100*(cm['peak_lag_seconds']/pm['peak_lag_seconds']-1),
                        event_duration_change_pct=100*(spans[.01][1]/spans[.01][0]-1),
                        peak_excess_change_pct=100*((cm['peak_cms']-s.baseflow_cms)/amp-1),
                        duration_change_005_pct=100*(spans[.005][1]/spans[.005][0]-1),duration_change_020_pct=100*(spans[.02][1]/spans[.02][0]-1)))
                print(case,alpha,mode,'complete dt',clock['dt_seconds'],flush=True)
        pd.DataFrame(results).to_csv(OUT/'results_partial.csv',index=False)
    d=pd.DataFrame(results);st=pd.DataFrame(state_rows)
    assert not d.duplicated(['case','alpha','clock_mode','child_state']).any()
    assert len(d.query("clock_mode=='native_clock'"))+7*len(failures)==7*len(ALPHAS)*7
    old=pd.read_csv(ROOT/'synthetic_experiments/outputs/manning_sensitivity/results.csv').query("variant=='based_n0035'")
    match=d.query("alpha==1 and clock_mode=='native_clock'").merge(old,on=['case','child_state'],suffixes=('_new','_old'),validate='one_to_one')
    assert len(match)==42
    for y in Y:assert np.allclose(match[y+'_new'],match[y+'_old'],atol=1e-7),y
    for name,frame in [('results',d),('states',st),('edges',pd.DataFrame(edge_rows)),('bounds',pd.DataFrame(bounds)),('infeasible',pd.DataFrame(failures))]:frame.to_csv(OUT/f'{name}.csv',index=False)
    manifest.update(status='complete_with_infeasible' if failures else 'complete',comparisons=len(d),native_comparisons=len(d.query("clock_mode=='native_clock'")),
        unique_runs=st.run_id.nunique(),parity_hydrographs=parity,previous_comparisons_reproduced=len(match),max_volume_error=float(st.volume_error_fraction.abs().max()),infeasible=len(failures))
    write_json(OUT/'manifest.json',manifest);print('COMPLETE',manifest,flush=True)
if __name__=='__main__':main()
