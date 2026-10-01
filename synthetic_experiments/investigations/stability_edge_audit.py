"""Locate exact edge controllers of the focused synthetic timestep bounds.
Read-only geometry/K computation; no routing or changed settings.
"""
from pathlib import Path
import sys,json,hashlib
import numpy as np
import pandas as pd
import networkx as nx
ROOT=Path(__file__).resolve().parents[2]
for p in [ROOT,ROOT/'synthetic_experiments/src',ROOT/'RAPID/src']:sys.path.insert(0,str(p))
from synthetic_experiments.networks import make_states,canonical_network
from synthetic_experiments.routing import Settings,compact_k
from manning_sensitivity import configurations
OUT=ROOT/'synthetic_experiments/outputs/segmentation_sensitivity/edge_audit'

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    refs=pd.read_csv(ROOT/'synthetic_experiments/outputs/manning_sensitivity/references.csv').set_index('case')
    rows=[]
    for case,params in configurations():
        s=Settings(kb=float(refs.loc[case,'kb']),n_manning=.035)
        states,_=make_states(params)
        for name,g in states.items():
            net=canonical_network(g);f=compact_k(net,s)
            graph=nx.DiGraph();graph.add_nodes_from(range(len(net['nodes'])));graph.add_edges_from((e[0],e[1]) for e in net['edges'])
            q={n:0. for n in graph};q[0]=1.;shares={}
            for u in nx.topological_sort(graph):
                outgoing=[(i,e) for i,e in enumerate(net['edges']) if e[0]==u];total=sum(e[3] for i,e in outgoing)
                for i,e in outgoing:shares[i]=q[u]*e[3]/total;q[e[1]]+=shares[i]
            assert np.isclose(q[len(net['nodes'])-1],1)
            for i,r in enumerate(f.itertuples()):
                u,v,L,w,S=net['edges'][i];start=net['nodes'][u][0];end=net['nodes'][v][0]
                labels=[str(k) for a,b,k,d in g.edges(keys=True,data=True) if np.isclose(g.nodes[a]['x'],start) and np.isclose(g.nodes[b]['x'],end) and np.isclose(d['w'],w) and np.isclose(d['length_m'],L)]
                rows.append(dict(case=case,state=name,edge_index=i,edge_label=' / '.join(labels) if labels else 'canonical merged series',
                    start_km=start/1000,end_km=end/1000,width_m=w,length_m=L,slope=S,depth_m=w/s.kb,
                    segment_length_m=r.link_length_m,effective_segment_length_m=r.rapid_effective_length_m,segment_count=int(r.count),
                    celerity_raw_mps=r.rapid_celerity_mps_raw,celerity_used_mps=r.rapid_celerity_mps,
                    clipped=bool(r.rapid_celerity_capped),segment_K_seconds=r.rapid_k,flow_share=shares[i],
                    dt_lower_seconds=2*s.x*r.rapid_k,dt_upper_seconds=2*(1-s.x)*r.rapid_k))
    d=pd.DataFrame(rows)
    d['state_dt_lower']=d.groupby(['case','state']).dt_lower_seconds.transform('max')
    d['state_dt_upper']=d.groupby(['case','state']).dt_upper_seconds.transform('min')
    d['case_dt_lower']=d.groupby('case').dt_lower_seconds.transform('max')
    d['case_dt_upper']=d.groupby('case').dt_upper_seconds.transform('min')
    d['controls_state_lower']=np.isclose(d.dt_lower_seconds,d.state_dt_lower)
    d['controls_state_upper']=np.isclose(d.dt_upper_seconds,d.state_dt_upper)
    d['controls_case_lower']=np.isclose(d.dt_lower_seconds,d.case_dt_lower)
    d['controls_case_upper']=np.isclose(d.dt_upper_seconds,d.case_dt_upper)
    d['blocks_case_fast_bound']=d.dt_lower_seconds>d.case_dt_upper
    d.to_csv(OUT/'edges.csv',index=False)
    states=d[['case','state','state_dt_lower','state_dt_upper']].drop_duplicates()
    states['feasible_integer_dt']=np.ceil(states.state_dt_lower+1e-8)<=np.floor(states.state_dt_upper)
    states.to_csv(OUT/'states.csv',index=False)
    cases=d[['case','case_dt_lower','case_dt_upper']].drop_duplicates()
    cases['feasible_integer_dt']=np.ceil(cases.case_dt_lower+1e-8)<=np.floor(cases.case_dt_upper)
    cases.to_csv(OUT/'cases.csv',index=False)
    (OUT/'provenance.json').write_text(json.dumps(dict(reference='HYD-2026-09-29-EDGE-CONTROLLERS',
        scope='Seven focused configurations only; n=.035 and saved BASED kb; 200m segmentation/floor; X=.1; no full-grid preflight yet',
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),source_references='outputs/manning_sensitivity/references.csv'),indent=2))
    print(cases.to_string(index=False))
    print(d[(d.case=='long_narrow_A')&d.state.isin(['simple/original','nested/original','crossover/original'])&(d.blocks_case_fast_bound|d.controls_case_upper)][['state','edge_label','start_km','end_km','width_m','length_m','slope','depth_m','celerity_used_mps','segment_K_seconds','dt_lower_seconds','dt_upper_seconds','clipped','flow_share']].round(5).to_string(index=False))
if __name__=='__main__':main()
