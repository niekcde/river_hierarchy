"""Cases 11/46: shared-clock parent-child routing sensitivity, isolated outputs."""
from pathlib import Path
import sys,json,hashlib
import numpy as np
import pandas as pd
import networkx as nx
import pyogrio
from scipy.spatial import cKDTree
from scipy.fft import rfft,irfft,next_fast_len
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import factorized

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'RAPID/src'),str(ROOT/'synthetic_experiments/investigations')]
from rapid_tools.celerity_regularization import regularize_routing_celerity
from rapid_tools.engine import ccc_mat,rte_mat,mus_rte
from event_duration_verification import duration
BASE=ROOT/'RAPID/assets/bounded_robust_slopes_20261002'
OUT=BASE/'common_caps/explicit_regularization_check'
STATES={'11':['S000_base','S003_unit_18'],
        '46':['S000_base','S004_unit_24','S010_unit_105']}


def transfer(frame,dt,nfft):
    """Exact discrete Muskingum transfer on the DAG, batching frequencies.

    Equal segments on each original link are a cascade H**count. The first
    inlet segment uses RAPID's held external-input coefficient C1+C2.
    This is the same recurrence as engine.mus_rte, not pure translation.
    """
    G=nx.DiGraph(); outgoing={}
    for r in frame.itertuples():
        G.add_edge(r.u,r.v);outgoing.setdefault(r.u,[]).append(r)
    assert nx.is_directed_acyclic_graph(G)
    roots=[v for v in G if G.in_degree(v)==0];ends=[v for v in G if G.out_degree(v)==0]
    assert len(roots)==len(ends)==1
    order=list(nx.topological_sort(G));answer=np.zeros(nfft//2+1,complex)
    for start in range(0,len(answer),4096):
        f=np.arange(start,min(start+4096,len(answer)))
        z=np.exp(-2j*np.pi*f/nfft)
        signals={roots[0]:np.ones(len(f),complex)}
        for node in order:
            signal=signals.pop(node)
            if node==ends[0]:answer[start:start+len(f)]=signal;continue
            edges=outgoing[node];width=sum(r.width for r in edges)
            for r in edges:
                K=r.rapid_k;den=dt+1.8*K
                a=(dt-.2*K)/den;b=(dt+.2*K)/den;c=(1.8*K-dt)/den
                H=(a+b*z)/(1-c*z)
                if node==roots[0]:
                    Htotal=((a+b)/(1-c*z))*H**(r.count-1)
                else:Htotal=H**r.count
                value=signal*(r.width/width)*Htotal
                if r.v in signals:signals[r.v]+=value
                else:signals[r.v]=value
    assert abs(answer[0]-1)<1e-10
    return answer


def verify_transfer():
    # Split/rejoin with one branch containing a two-segment cascade.
    f=pd.DataFrame(dict(u=[0,1,1,2,3],v=[1,2,3,4,4],width=[10,3,7,3,7],
                        count=[1,2,1,1,1],rapid_k=[100,120,90,130,80]))
    edges=[];virtual=10
    for r in f.itertuples():
        u=r.u
        for i in range(r.count):
            v=r.v if i==r.count-1 else virtual
            if i<r.count-1:virtual+=1
            edges.append((u,v,r.width,r.rapid_k));u=v
    net=np.zeros((len(edges),len(edges)))
    for i,(_,v,_,_) in enumerate(edges):
        dest=[j for j,e in enumerate(edges) if e[0]==v]
        for j in dest:net[j,i]=edges[j][2]/sum(edges[k][2] for k in dest)
    mats=rte_mat(csc_matrix(net),*ccc_mat(np.array([e[3] for e in edges]),np.full(len(edges),.1),60))
    solve=factorized(mats[0]);state=np.zeros(len(edges));external=np.zeros(len(edges));q=[]
    for i in range(512):
        external[:]=0;external[0]=float(i==0)
        _,state=mus_rte(*mats,1,state,external,solve)
        q.append(sum(state[j] for j,e in enumerate(edges) if e[1]==4))
    fft=irfft(transfer(f,60,2048),2048)[:512]
    error=float(np.max(abs(fft-q)));assert error<1e-12,error
    return error


def load_frames(ex):
    exp=ROOT/f'network_variants/output/{ex}_indep__anchored'
    base_manifest=json.loads((exp/'states/S000_base/hydraulic_context/hydraulic_context_manifest.json').read_text())
    base_nodes=pyogrio.read_dataframe(base_manifest['input_paths']['directed_nodes'])
    xy=np.c_[base_nodes.geometry.x,base_nodes.geometry.y];tree=cKDTree(xy)
    base_heads=pd.read_parquet(BASE/ex/'floor_0.05/old/nodes.parquet').set_index('id_node')['head']
    adjusted=base_heads.copy()
    changes=pd.read_csv(BASE/'common_caps/local_heads_and_caps/changed_nodes.csv',dtype={'example':str})
    for r in changes[changes.example.eq(ex)&changes.clock.eq('nominal_integer')&changes.head_budget_m.eq(.05)].itertuples():
        adjusted.loc[r.id_node]=r.head_after
    refs=pd.read_csv(BASE/'hydraulic_references.csv',dtype={'example':str}).set_index('example').loc[ex]
    all_frames={};audits=[]
    for state in STATES[ex]:
        folder=exp/'states'/state/'hydraulic_context'
        manifest=json.loads((folder/'hydraulic_context_manifest.json').read_text())
        nodes=pyogrio.read_dataframe(manifest['input_paths']['directed_nodes'])
        dist,ix=tree.query(np.c_[nodes.geometry.x,nodes.geometry.y])
        if np.max(dist)>1e-5:raise ValueError(f'Unmapped child node: {ex}/{state}, {np.max(dist)}')
        mapping=dict(zip(nodes.id_node.astype(int),base_nodes.id_node.to_numpy()[ix].astype(int)))
        geometry=pyogrio.read_dataframe(manifest['input_paths']['directed_links'])
        h=pd.read_csv(folder/'link_hydraulic_context.csv')
        g=h[['id_link','id_us_node','id_ds_node','link_length_m']].merge(
            geometry[['id_link','wid_adj_routing']],on='id_link',validate='one_to_one')
        g['u']=g.id_us_node.map(mapping);g['v']=g.id_ds_node.map(mapping)
        g['width']=g.wid_adj_routing;g['count']=np.maximum(1,np.rint(g.link_length_m/200)).astype(int)
        g['rapid_effective_length_m']=np.maximum(200,g.link_length_m/g['count'])
        depth=refs.stored_depth_m*(g.width/refs.reference_width_m)**.65
        for method,heads in [('original',base_heads),('local_heads_5cm',adjusted)]:
            f=g.copy();f['slope']=(g.u.map(heads)-g.v.map(heads))/g.link_length_m
            assert (f.slope>0).all()
            f['rapid_celerity_mps_raw']=5/(3*.035)*depth**(2/3)*np.sqrt(f.slope)
            f['rapid_celerity_mps']=f.rapid_celerity_mps_raw.clip(.05,5)
            f['rapid_k']=f.rapid_effective_length_m/f.rapid_celerity_mps
            all_frames[state,method]=f
        audits.append(dict(example=ex,state=state,n_links=len(g),n_segments=int(g['count'].sum()),
                           max_node_match_distance=float(dist.max())))
    return all_frames,audits


def choose_clocks(frames,forcing_dt):
    f=pd.concat([v for (state,method),v in frames.items() if method=='original'],ignore_index=True)
    L=f.rapid_effective_length_m.to_numpy();c=f.rapid_celerity_mps.to_numpy()
    candidates=[i for i in range(1,forcing_dt+1) if forcing_dt%i==0
                and (.202*L/i<=5).all() and (1.782*L/i>=.05).all()]
    ranked=[]
    for dt in candidates:
        target=np.clip(c,.202*L/dt,1.782*L/dt)
        ranked.append((int((~np.isclose(target,c)).sum()),float(np.sum(f.link_length_m*abs(np.log(target/c)))),dt))
    ranked.sort();first=ranked[0][2]
    second=next(r[2] for r in ranked if abs(r[2]/first-1)>=.15)
    return [first,second]


def longest_travel(f):
    G=nx.DiGraph();out={}
    for r in f.itertuples():G.add_edge(r.u,r.v);out.setdefault(r.u,[]).append(r)
    dist={v:0. for v in G}
    for u in nx.topological_sort(G):
        for r in out.get(u,[]):dist[r.v]=max(dist[r.v],dist[u]+r.rapid_k*r.count)
    return max(dist.values())


def quadratic_peak(h, column):
    values=h[column].to_numpy();i=int(values.argmax())
    if i==0 or i==len(values)-1:raise ValueError('Peak on boundary')
    a,b,c=values[i-1:i+2]
    offset=.5*(a-c)/(a-2*b+c)
    assert abs(offset)<=.5+1e-5
    return float(h.time_seconds.iloc[i]+np.median(np.diff(h.time_seconds))*offset)


def main():
    OUT.mkdir(exist_ok=True);fft_error=verify_transfer();summaries=[];responses=[];audits=[];link_audits=[]
    for ex in STATES:
        frames,qa=load_frames(ex);audits.extend(qa)
        saved=pd.read_csv(ROOT/f'network_variants/output/{ex}_indep__anchored/states/S000_base/rapid/run/outlet_hydrograph.csv')
        times=saved.time_seconds.to_numpy();flow=saved.q_inflow_cms.to_numpy()
        forcing_dt=int(round(np.median(np.diff(times))))
        changed=np.flatnonzero(abs(np.diff(flow))>np.ptp(flow)*1e-8)
        onset=times[changed[0]];finish=times[changed[-1]+1]
        # Remove excessive historical spinup; initialize exactly at equilibrium.
        left=max(0,np.searchsorted(times,onset-86400));right=min(len(times)-1,np.searchsorted(times,finish)+1)
        source_t=times[left:right+1]-times[left];source_q=flow[left:right+1]
        clocks=choose_clocks(frames,forcing_dt)
        maxtravel=max(longest_travel(f) for f in frames.values())
        end=source_t[-1]+max(10*86400,8*maxtravel)
        for dt in clocks:
            n=int(np.ceil(end/dt));t=np.arange(n)*dt
            # Exact interval means of the same piecewise-linear saved forcing.
            increments=np.diff(source_t)*(source_q[:-1]+source_q[1:])/2
            cumulative=np.r_[0,np.cumsum(increments)]
            def primitive(x):
                ix=np.clip(np.searchsorted(source_t,x,side='right')-1,0,len(source_t)-2)
                dx=x-source_t[ix];s=(source_q[ix+1]-source_q[ix])/(source_t[ix+1]-source_t[ix])
                inside=cumulative[ix]+source_q[ix]*dx+.5*s*dx*dx
                return np.where(x>=source_t[-1],cumulative[-1]+(x-source_t[-1])*source_q[-1],inside)
            inflow=(primitive(t+dt)-primitive(t))/dt
            nfft=next_fast_len(2*n);input_fft=rfft(inflow-flow[0],nfft)
            for method in ['original','local_heads_5cm']:
                hydros={}
                for state in STATES[ex]:
                    f=regularize_routing_celerity(frames[state,method],dt_seconds=dt)
                    H=transfer(f,dt,nfft);out=flow[0]+irfft(input_fft*H,nfft)[:n]
                    # Initial wraparound and final settlement must be negligible.
                    error0=abs(out[0]-flow[0])/np.ptp(flow);error1=abs(out[-1]-flow[-1])/np.ptp(flow)
                    assert max(error0,error1)<1e-5,(ex,state,error0,error1)
                    assert out.min()>0 and np.isfinite(out).all()
                    key=dict(example=ex,state=state,method=method,dt_s=dt)
                    h=pd.DataFrame(dict(time_seconds=t+dt/2,inlet_cms=inflow,outlet_cms=out))
                    h.to_parquet(OUT/f'{ex}_{state}_{method}_dt{dt}.parquet',index=False)
                    hydros[state]=h
                    summaries.append(dict(**key,forcing_dt_s=forcing_dt,n_links=len(f),
                        regularized_links=int(f.rapid_celerity_regularization_applied.sum()),
                        initial_error_fraction=error0,final_error_fraction=error1,
                        peak_time_s=float(h.time_seconds.iloc[np.argmax(out)]),
                        peak_excess_cms=float(out.max()-flow[0]),peak_lag_s=float(dt*(np.argmax(out)-np.argmax(inflow)))))
                    for r in f[f.rapid_celerity_regularization_applied].itertuples():
                        link_audits.append(dict(**key,id_link=r.id_link,u=r.u,v=r.v,
                            hydraulic_celerity=r.rapid_celerity_mps_raw,common_capped=r.rapid_celerity_pre_regularization_mps,
                            routing_celerity=r.rapid_celerity_mps,correction_factor=r.rapid_celerity_regularization_factor,
                            slope=r.slope,width=r.width))
                    print('Routed',ex,state,method,dt,flush=True)
                parent=hydros['S000_base'];pq=parent.outlet_cms.to_numpy();pt=parent.time_seconds.to_numpy()
                rise=.01*(pq.max()-flow[0]);fall=.01*(pq.max()-flow[-1])
                pdur=duration(pt,pq,flow[0],flow[-1],rise,fall)
                for state in STATES[ex][1:]:
                    child=hydros[state];cq=child.outlet_cms.to_numpy()
                    cdur=duration(pt,cq,flow[0],flow[-1],rise,fall)
                    shift=float(pt[np.argmax(cq)]-pt[np.argmax(pq)])
                    lag=float(pt[np.argmax(pq)]-pt[np.argmax(inflow)])
                    interpolated_shift=quadratic_peak(child,'outlet_cms')-quadratic_peak(parent,'outlet_cms')
                    interpolated_lag=quadratic_peak(parent,'outlet_cms')-quadratic_peak(parent,'inlet_cms')
                    responses.append(dict(example=ex,state=state,method=method,dt_s=dt,
                        timing_change_hours=shift/3600,lag_shift_pct=100*shift/lag,
                        quadratic_peak_timing_change_hours=interpolated_shift/3600,
                        quadratic_peak_lag_shift_pct=100*interpolated_shift/interpolated_lag,
                        event_duration_change_pct=100*(cdur['duration_seconds']/pdur['duration_seconds']-1),
                        peak_excess_change_pct=100*((cq.max()-flow[0])/(pq.max()-flow[0])-1),
                        parent_duration_hours=pdur['duration_seconds']/3600))
    pd.DataFrame(summaries).to_csv(OUT/'routing_summary.csv',index=False)
    pd.DataFrame(responses).to_csv(OUT/'comparisons.csv',index=False)
    pd.DataFrame(audits).to_csv(OUT/'geometry_audit.csv',index=False)
    pd.DataFrame(link_audits).to_csv(OUT/'regularized_links.csv',index=False)
    (OUT/'provenance.json').write_text(json.dumps(dict(date='2026-10-02',
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        fft_engine_max_error=fft_error,alpha=.65,n=.035,X=.1,caps=[.05,5],margin=.01,
        forcing='Saved base inflow, piecewise-linear interval means; exact equilibrium initialization; extended final plateau',
        child_heads='Updated base shared node heads mapped by exact coordinates; existing child geometry and widths; resistance not recomputed',
        scope='Selected simple-unit children only; actual saved forcing-clock divisors shared across all tested states/methods',
        production_changed=False),indent=2))
    print(pd.DataFrame(responses).to_string(index=False))


if __name__=='__main__':main()
