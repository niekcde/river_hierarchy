"""Compare uniform/refined Muskingum routing; no changes to production code.

Use the existing route function with an isolated prepare binding so all solver,
forcing and QA logic is identical; frames vary only in segmentation/safeguards.
"""
from pathlib import Path
from dataclasses import replace,asdict
from types import FunctionType
import sys,json,math,hashlib
import numpy as np
import pandas as pd
import networkx as nx
ROOT=Path(__file__).resolve().parents[2]
for p in [ROOT,ROOT/'synthetic_experiments/src',ROOT/'RAPID/src']:sys.path.insert(0,str(p))
from synthetic_experiments.networks import make_states,canonical_network,digest
from synthetic_experiments import routing
from synthetic_experiments.pipeline import write_json,source_fingerprint
from rapid_tools.k_values import compute_k_values,compute_dt_bounds,apply_stability_effective_length_cap
from rapid_tools.prep import _compute_subreach_count
from event_duration_verification import duration
OUT=ROOT/'synthetic_experiments/outputs/segmentation_sensitivity'
Y=['lag_shift_pct','event_duration_change_pct','peak_excess_change_pct']

def build_frame(net,s,target,adaptive_dt=None):
    rows=[];nextnode=len(net['nodes'])
    for edge,(u,v,L,w,slope) in enumerate(net['edges']):
        count=_compute_subreach_count(L,target)
        c=np.clip(5/(3*s.n_manning)*(w/s.kb)**(2/3)*np.sqrt(slope),s.min_celerity_mps,s.max_celerity_mps)
        if adaptive_dt is not None:count=max(count,int(math.ceil(L/c/(adaptive_dt/(2*s.x)))))
        nodes=[u]+list(range(nextnode,nextnode+count-1))+[v];nextnode+=count-1
        for j in range(count):
            rows.append(dict(edge=edge,id_us_node=nodes[j],id_ds_node=nodes[j+1],link_length_m=L/count,width=w,
                slope_used=slope,is_inlet=(u==0 and j==0),is_outlet=(v==len(net['nodes'])-1 and j==count-1)))
    config=replace(routing.k_config(s),min_effective_length_m=target if adaptive_dt is None else None)
    frame=compute_k_values(pd.DataFrame(rows),config=config)
    frame['rapid_branch_width_m']=frame.width
    frame['physical_k']=frame.link_length_m/frame.rapid_celerity_mps
    assert np.allclose(frame.groupby('edge').link_length_m.sum(),[e[2] for e in net['edges']])
    return frame

def bounds(frames,x):
    b=[compute_dt_bounds(f.rapid_k,x_value=x) for f in frames.values()]
    return max(a for a,b in b),min(b for a,b in b)

def moments(net,frame,x):
    graph=nx.MultiDiGraph()
    for i,(u,v,L,w,slope) in enumerate(net['edges']):graph.add_edge(u,v,key=i,width=w)
    ps=[]
    for path in nx.all_simple_edge_paths(graph,0,len(net['nodes'])-1):
        probability=1.;mean=0.;var=0.
        for u,v,key in path:
            probability*=graph.edges[u,v,key]['width']/sum(d['width'] for *_,d in graph.out_edges(u,keys=True,data=True))
            ks=frame.loc[frame.edge==key,'rapid_k'];mean+=ks.sum();var+=(1-2*x)*(ks**2).sum()
        ps.append((probability,mean,var))
    assert np.isclose(sum(p for p,m,v in ps),1)
    mean=sum(p*m for p,m,v in ps)
    within=sum(p*v for p,m,v in ps);between=sum(p*(m-mean)**2 for p,m,v in ps)
    return dict(path_mean_k_seconds=mean,within_path_variance_seconds2=within,between_path_variance_seconds2=between,
        max_path_k_seconds=max(m for p,m,v in ps))

def run_frame(net,frame,s,clock):
    environment=dict(routing.route.__globals__);environment['prepare']=lambda _net,_s:frame.copy()
    isolated=FunctionType(routing.route.__code__,environment,'route_prescribed_frame')
    return isolated(net,s,clock)

def observed_moments(h,mt,s):
    onset=round(mt['buffer_hours']*3600/mt['dt_seconds'])*mt['dt_seconds']
    mask=h.time_seconds>=onset
    t=h.time_seconds.to_numpy()[mask]+mt['dt_seconds']/2
    means=[];vars=[]
    for col in ['inlet_cms','outlet_cms']:
        excess=h[col].to_numpy()[mask]-s.baseflow_cms
        assert excess.min()>-1e-5
        excess=np.maximum(excess,0);mean=np.sum(t*excess)/excess.sum()
        means.append(mean);vars.append(np.sum((t-mean)**2*excess)/excess.sum())
    return dict(centroid_lag_seconds=means[1]-means[0],added_variance_seconds2=vars[1]-vars[0],outlet_temporal_sd_seconds=np.sqrt(vars[1]))

def main():
    OUT.mkdir(parents=True,exist_ok=True);source=source_fingerprint();failed=[];rows=[];runs=[];jobs=set()
    base=dict(total_width_m=500,reference_slope=1e-4,outer_a_fraction=.8,inner_fraction=.5,length_multiplier=1.25,total_length_m=20000,internal_location='central')
    cases=[('baseline',base),('long_narrow',dict(base,outer_a_fraction=.05,length_multiplier=3.)),('long_dominant',dict(base,outer_a_fraction=.95,length_multiplier=3.))]
    kb=float(pd.read_csv(ROOT/'synthetic_experiments/outputs/manning_sensitivity/references.csv').query("case=='baseline'").kb.iloc[0])
    s=routing.Settings(kb=kb,n_manning=.035)
    manifest=dict(status='running',source=source,settings=asdict(s),cases=cases,
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        note='Uniform target/floor 200,100,50m; adaptive maximum 200,100,50m with no effective floor; real safeguard at 200m; additional timestep-only controls.',
        adaptive_rule='dt=floor(0.8*minimum upper bound of uniform frames); subdivide each edge sufficiently for lower bound, preserving total physical length and summed physical K',
        theory='Constant-coefficient path mean=sum K; within-path variance=(1-2X)sum K^2; between-path variance from width-weighted path means. Moment theory is diagnostic; measured moments expose discrete boundary timing offsets.')
    write_json(OUT/'manifest.json',manifest)
    for case,params in cases:
        states,pairs=make_states(params);nets={k:canonical_network(v) for k,v in states.items()}
        variants=[]
        for target in [200,100,50]:
            uniform={k:build_frame(net,s,target) for k,net in nets.items()}
            lo,hi=bounds(uniform,s.x);dt=math.floor(.8*hi)
            variants.append((f'uniform_{target}',uniform,dt))
            adaptive={k:build_frame(net,s,target,adaptive_dt=dt) for k,net in nets.items()}
            variants.append((f'adaptive_{target}',adaptive,dt))
            lower,upper=bounds(adaptive,s.x)
            small=math.ceil(lower+1e-8)
            if small!=dt:variants.append((f'adaptive_{target}_small_dt',adaptive,small))
            if target==200:
                # Use exactly the current real-workflow helper and retain 200m floor.
                safe={k:apply_stability_effective_length_cap(f,x_value=s.x,min_celerity_mps=s.min_celerity_mps,safety_margin_seconds=10.) for k,f in uniform.items()}
                safelo,safehi=bounds(safe,s.x)
                variants.append(('real_safeguard_200',safe,math.ceil(safelo+1e-8)))
        for variant,frames,dt in variants:
            lo,hi=bounds(frames,s.x)
            if math.ceil(lo+1e-8)>math.floor(hi):
                failed.append(dict(case=case,variant=variant,dt_min_seconds=lo,dt_max_seconds=hi,reason='No common nonnegative-coefficient timestep'))
                print(case,variant,'INFEASIBLE',round(lo,2),round(hi,2),flush=True);continue
            if not lo<=dt<=hi:dt=math.ceil(lo+1e-8)
            theory={k:moments(nets[k],f,s.x) for k,f in frames.items()}
            travel=max(v['max_path_k_seconds'] for v in theory.values())
            clock=dict(dt_seconds=dt,dt_min_seconds=lo,dt_max_seconds=hi,buffer_hours=math.ceil(max(48,3*travel/3600)),max_path_k_seconds=travel)
            data={}
            for state,frame in frames.items():
                identity=dict(net=nets[state],settings=asdict(s),clock=clock,frame=frame[['edge','id_us_node','id_ds_node','link_length_m','width','rapid_k']].to_dict('records'),source=source)
                rid=digest(identity);jobs.add(rid);folder=OUT/'runs'/rid
                if (folder/'metrics.json').exists():
                    mt=json.loads((folder/'metrics.json').read_text());h=pd.read_parquet(folder/'hydrograph.parquet')
                else:
                    h,mt,_=run_frame(nets[state],frame,s,clock)
                    mt.update(observed_moments(h,mt,s))
                    folder.mkdir(parents=True,exist_ok=True);h.to_parquet(folder/'hydrograph.parquet',index=False)
                    write_json(folder/'metrics.json',mt);write_json(folder/'job.json',identity)
                mt.update(observed_moments(h,mt,s))
                write_json(folder/'metrics.json',mt)
                # Physical K preserved by adaptive subdivision. Uniform floors audited separately.
                physical=frame.groupby('edge').physical_k.sum()
                actual=frame.groupby('edge').rapid_k.sum()
                if variant.startswith('adaptive'):assert np.allclose(physical,actual)
                inlet_k=float(frame.loc[frame.is_inlet,'rapid_k'].iloc[0])
                # RAPID treats the external interval-mean inflow without the
                # delayed-inflow numerator used at internal connections.
                expected_mean=theory[state]['path_mean_k_seconds']-s.x*inlet_k-dt/2
                expected_var=theory[state]['within_path_variance_seconds2']+theory[state]['between_path_variance_seconds2']+(s.x*inlet_k)**2-dt**2/4
                assert np.isclose(mt['centroid_lag_seconds'],expected_mean,rtol=1e-7,atol=.01)
                assert np.isclose(mt['added_variance_seconds2'],expected_var,rtol=1e-7,atol=.1)
                row=dict(case=case,variant=variant,state=state,run_id=rid,**mt,**theory[state],
                    predicted_discrete_centroid_lag_seconds=expected_mean,predicted_discrete_added_variance_seconds2=expected_var,
                    min_segment_length_m=frame.link_length_m.min(),max_segment_length_m=frame.link_length_m.max(),
                    max_edge_k_relative_change=float(np.max(abs(actual/physical-1))))
                runs.append(row);data[state]=(h,mt,rid)
            # Adapter parity test on the standard baseline 200m network, same clock.
            if case=='baseline' and variant=='uniform_200':
                for state in ['centerline/original','simple/original','nested/original']:
                    h0,m0,f0=routing.route(nets[state],s,clock)
                    assert np.allclose(h0.outlet_cms,data[state][0].outlet_cms,rtol=1e-11,atol=1e-8)
            for pair in pairs:
                ph,pm,pid=data[pair['parent_state']];ch,cm,cid=data[pair['child_state']]
                amp=pm['peak_cms']-s.baseflow_cms;spans=[]
                for h,mt in [(ph,pm),(ch,cm)]:
                    t=h.time_seconds.to_numpy()-mt['buffer_hours']*3600+dt/2;mask=t>=0
                    spans.append(duration(t[mask],h.outlet_cms.to_numpy()[mask],s.baseflow_cms,s.baseflow_cms,.01*amp,.01*amp)['duration_seconds'])
                rows.append(dict(case=case,variant=variant,**pair,parent_run_id=pid,child_run_id=cid,dt_seconds=dt,
                    lag_shift_pct=100*(cm['peak_lag_seconds']/pm['peak_lag_seconds']-1),
                    event_duration_change_pct=100*(spans[1]/spans[0]-1),
                    peak_excess_change_pct=100*((cm['peak_cms']-s.baseflow_cms)/amp-1),
                    parent_duration_hours=spans[0]/3600,child_duration_hours=spans[1]/3600,
                    parent_centroid_lag_hours=pm['centroid_lag_seconds']/3600,child_centroid_lag_hours=cm['centroid_lag_seconds']/3600,
                    parent_peak_lag_hours=pm['peak_lag_seconds']/3600,child_peak_lag_hours=cm['peak_lag_seconds']/3600,
                    parent_peak_cms=pm['peak_cms'],child_peak_cms=cm['peak_cms']))
            print(case,variant,'complete',dt,'s',flush=True)
    d=pd.DataFrame(rows);r=pd.DataFrame(runs)
    d.to_csv(OUT/'comparisons.csv',index=False);r.to_csv(OUT/'run_diagnostics.csv',index=False)
    pd.DataFrame(failed).to_csv(OUT/'infeasible.csv',index=False)
    manifest.update(status='complete',comparisons=len(d),unique_runs=len(jobs),infeasible_variants=len(failed),
        adapter_parity='three baseline states matched stock route at identical clock',max_volume_error=float(r.volume_error_fraction.abs().max()))
    write_json(OUT/'manifest.json',manifest)
    print('COMPLETE',len(d),'comparisons',len(jobs),'unique routes',flush=True)
if __name__=='__main__':main()
