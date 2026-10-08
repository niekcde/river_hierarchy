"""Isolated native-recurrence test of fractional clocks; no production changes.

Spatial lfilter evaluation is the same linear Muskingum recurrence on a DAG,
with held external inflow and forcing-interval averaged output. Cross-check it
against the production sparse recurrence. Preserve the complete saved forcing;
only the exactly steady initial plateau is skipped, as in production.
"""
from pathlib import Path
import sys,json,time,hashlib,math
import numpy as np
import pandas as pd
import networkx as nx
from scipy.signal import lfilter
from scipy.sparse import csc_matrix,identity
from scipy.sparse.linalg import factorized
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'RAPID/src'))
from rapid_tools.engine import ccc_mat,rte_mat,mus_rte
INV=ROOT/'RAPID/assets/bounded_robust_slopes_20261002/child_head_reconstruction_20261003'
OUT=ROOT/'RAPID/outputs/fractional_clock_check_20261004'
CLOCKS={'37':[5.,47.1875,94.375],'46':[35.,33.90625,67.8125]}


def structure(f):
    graph=nx.DiGraph();out={}
    for r in f.itertuples():
        graph.add_edge(r.u,r.v);out.setdefault(r.u,[]).append(r)
    assert nx.is_directed_acyclic_graph(graph)
    roots=[n for n in graph if graph.in_degree(n)==0];ends=[n for n in graph if graph.out_degree(n)==0]
    assert len(roots)==len(ends)==1
    return graph,out,roots[0],ends[0]


def cascade(f,flow,nsub,dt,q0,save_audit=False,return_native=False,serialized_allocations=False):
    graph,out,root,end=structure(f)
    signals={root:np.repeat(flow,nsub)};steady={root:q0};audits=[]
    for node in nx.topological_sort(graph):
        signal=signals.pop(node);initial=steady.pop(node)
        if node==end:
            return (signal if return_native else signal.reshape(-1,nsub).mean(axis=1)),audits
        width=sum(r.width for r in out[node])
        for r in out[node]:
            frac=r.width/width
            if serialized_allocations and node!=root:
                frac=float(np.float32(round(frac,6)))
            value=signal*frac;eq=initial*frac
            den=dt+1.8*r.rapid_k;a=(dt-.2*r.rapid_k)/den;b=(dt+.2*r.rapid_k)/den;c=(1.8*r.rapid_k-dt)/den
            assert 0<=c<1
            for segment in range(int(r.count)):
                b0,b1=(a+b,0.) if node==root and segment==0 else (a,b)
                value,_=lfilter([b0,b1],[1.,-c],value,zi=[(b1+c)*eq])
                if not np.isfinite(value).all():raise ValueError('Nonfinite flow')
                lo=float(value.min());hi=float(value.max())
                if lo < -1e-10*max(1.,hi):raise ValueError(f'Negative flow {r.id_link}/{segment}: {lo}')
                if save_audit:audits.append(dict(id_link=int(r.id_link),segment=segment+1,minimum=lo,maximum=hi,a=b0,c=c))
            if r.v in signals:signals[r.v]+=value;steady[r.v]+=eq
            else:signals[r.v]=value;steady[r.v]=eq
    raise AssertionError('No outlet')


def sparse_benchmark(f,flow,nsub,dt,q0,serialized_allocations=False):
    graph,out,root,end=structure(f);edges=[];virtual=-1
    for node in nx.topological_sort(graph):
        for r in out.get(node,[]):
            u=node
            for seg in range(int(r.count)):
                v=r.v if seg==r.count-1 else virtual
                if seg<r.count-1:virtual-=1
                edges.append((u,v,r.width,r.rapid_k));u=v
    incoming={};outgoing={}
    for j,(u,v,w,k) in enumerate(edges):incoming.setdefault(v,[]).append(j);outgoing.setdefault(u,[]).append(j)
    rows=[];cols=[];values=[]
    for node,parents in incoming.items():
        children=outgoing.get(node,[]);total=sum(edges[j][2] for j in children)
        for i in parents:
            for j in children:
                ratio=edges[j][2]/total
                if serialized_allocations:ratio=float(np.float32(round(ratio,6)))
                rows.append(j);cols.append(i);values.append(ratio)
    net=csc_matrix((values,(rows,cols)),shape=(len(edges),len(edges)))
    mats=rte_mat(net,*ccc_mat(np.array([e[3] for e in edges]),np.full(len(edges),.1),dt))
    external=np.zeros(len(edges));roots=outgoing[root];total=sum(edges[j][2] for j in roots)
    for j in roots:external[j]=edges[j][2]/total
    state=factorized(identity(len(edges),format='csc')-net)(external*q0);solve=factorized(mats[0]);audit={'minimum_substep_flow_cms':float(state.min()),'maximum_substep_flow_cms':float(state.max())};answer=[]
    start=time.perf_counter()
    for q in flow:
        mean,state=mus_rte(*mats,nsub,state,external*q,solve,diagnostics=audit)
        answer.append(sum(mean[j] for j in incoming[end]))
    return np.array(answer),time.perf_counter()-start,audit


def peak(t,q):
    i=int(q.argmax());assert 0<i<len(q)-1
    a,b,c=q[i-1:i+2];den=a-2*b+c
    return float(t[i]+(.5*(a-c)/den if den else 0)*np.median(np.diff(t)))


def event_duration(t,q,q0,qe,rise,fall):
    k=int(q.argmax());up=np.flatnonzero(q[:k+1]>=q0+rise);down=np.flatnonzero(q[k:]>=qe+fall)+k
    if not len(up) or not len(down):return float('nan')
    i=int(up[0]);j=int(down[-1])
    if i==0 or j==len(q)-1:return float('nan')
    start=t[i-1]+(q0+rise-q[i-1])/(q[i]-q[i-1])*(t[i]-t[i-1])
    stop=t[j]+(qe+fall-q[j])/(q[j+1]-q[j])*(t[j+1]-t[j])
    return float(stop-start)


def run(ex):
    dest=OUT/ex;dest.mkdir(parents=True,exist_ok=True)
    inv=pd.read_csv(INV/ex/'inventory.csv');good=inv[inv.status.eq('ready')].drop_duplicates('fingerprint')
    chosen={}
    def add(row,reason):chosen.setdefault(row.fingerprint,{'row':row,'reasons':[]})['reasons'].append(reason)
    add(good[(good.family=='indep')&(good.state=='S000_base')].iloc[0],'base')
    add(good.loc[good.upper_dt_seconds.idxmin()],'smallest upper timestep bound')
    add(good.loc[good.max_travel_seconds.idxmax()],'longest travel time')
    prior=pd.read_csv(INV/ex/'comparisons.csv')
    for col in ['timing_change_minutes','duration_change_pct','peak_excess_change_pct']:
        fp=prior.loc[prior[col].abs().idxmax(),'fingerprint'];add(good[good.fingerprint.eq(fp)].iloc[0],'largest absolute '+col)
    for fam in ['compound_bubbles','island']:
        eligible=good[good.family.eq(fam)]
        if len(eligible) and not any(v['row'].family==fam for v in chosen.values()):add(eligible.loc[eligible.max_travel_seconds.idxmax()],fam+' coverage')
    pd.DataFrame([dict(fingerprint=k,state=v['row'].state,family=v['row'].family,reasons='; '.join(v['reasons'])) for k,v in chosen.items()]).to_csv(dest/'selection.csv',index=False)
    h=pd.read_csv(ROOT/f'network_variants/output/{ex}_indep__anchored/states/S000_base/rapid/run/outlet_hydrograph.csv')
    t=h.time_seconds.to_numpy();flow=h.q_inflow_cms.to_numpy();forcing_dt=float(np.median(np.diff(t)));assert np.allclose(np.diff(t),forcing_dt)
    q0=float(flow[0]);qe=float(flow[-1]);start=int(np.flatnonzero(flow!=q0)[0]);active=flow[start:];amplitude=float(np.ptp(flow))
    coefficient_rows=[]
    for rec in good.itertuples():
        f=pd.read_parquet(rec.frame);K=f.rapid_k.to_numpy()
        for dt in CLOCKS[ex]:
            c=(1.8*K-dt)/(1.8*K+dt);a=(dt-.2*K)/(1.8*K+dt)
            coefficient_rows.append(dict(state=rec.state,fingerprint=rec.fingerprint,dt_s=dt,min_c=float(c.min()),max_c=float(c.max()),min_a=float(a.min()),upper_bound_pass=bool((c>=0).all() and (c<1).all())))
    pd.DataFrame(coefficient_rows).to_csv(dest/'all_state_coefficients.csv',index=False)
    assert all(r['upper_bound_pass'] for r in coefficient_rows)
    runs=[];benchmarks=[];waves={};base_fp=next(iter(chosen))
    for fp,selection in chosen.items():
        rec=selection['row'];f=pd.read_parquet(rec.frame)
        for dt in CLOCKS[ex]:
            nsub=round(forcing_dt/dt);assert abs(nsub*dt-forcing_dt)<1e-9
            if fp==base_fp:
                probe=active[:64]
                sparse,seconds,audit=sparse_benchmark(f,probe,nsub,dt,q0)
                filtered,_=cascade(f,probe,nsub,dt,q0)
                error=float(np.max(np.abs(sparse-filtered))/amplitude);assert error<1e-8,error
                benchmarks.append(dict(dt_s=dt,intervals=len(probe),seconds=seconds,projected_active_routing_seconds=seconds*len(active)/len(probe),sparse_cascade_error_fraction=error,**audit))
                pd.DataFrame(benchmarks).to_csv(dest/'sparse_benchmarks.csv',index=False)
            begin=time.perf_counter();q,audit=cascade(f,active,nsub,dt,q0,True);seconds=time.perf_counter()-begin
            full=np.r_[np.full(start,q0),q];waves[fp,dt]=full
            pd.DataFrame(dict(time_seconds=t,inflow=flow,outflow=full)).to_parquet(dest/f'{fp}_dt{dt}.parquet',index=False)
            pd.DataFrame(audit).to_parquet(dest/f'{fp}_dt{dt}_segment_audit.parquet',index=False)
            parent=waves[base_fp,dt];pdur=event_duration(t,parent,q0,qe,.01*(parent.max()-q0),.01*(parent.max()-qe))
            dur=event_duration(t,full,q0,qe,.01*(parent.max()-q0),.01*(parent.max()-qe))
            row=dict(state=rec.state,fingerprint=fp,dt_s=dt,peak_time_s=peak(t,full),peak_excess=float(full.max()-q0),event_duration_s=dur,timing_change_minutes=(peak(t,full)-peak(t,parent))/60,duration_change_pct=100*(dur/pdur-1),peak_excess_change_pct=100*((full.max()-q0)/(parent.max()-q0)-1),minimum_segment_flow=min(a['minimum'] for a in audit),final_error_fraction=abs(full[-1]-qe)/amplitude,cascade_seconds=seconds)
            reference=waves[fp,CLOCKS[ex][0]];row.update(max_wave_difference_fraction=float(np.max(np.abs(full-reference))/amplitude),peak_difference_seconds=peak(t,full)-peak(t,reference))
            runs.append(row);pd.DataFrame(runs).to_csv(dest/'runs.csv',index=False)
            print(ex,rec.state,dt,'seconds',round(seconds,2),'wave_error',row['max_wave_difference_fraction'],'peak_diff',row['peak_difference_seconds'],flush=True)
    (dest/'provenance.json').write_text(json.dumps(dict(example=ex,forcing_dt_s=forcing_dt,full_forcing_intervals=len(flow),skipped_initial_intervals=start,active_intervals=len(active),selected_states=len(chosen),total_states=len(good),clocks=CLOCKS[ex],production_changed=False,script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),note='Exact held-forcing native recurrence, all selected segments checked at every substep; sparse engine agreement benchmark; full saved event and tail retained.'),indent=2))

if __name__=='__main__':
    for ex in sys.argv[1:] or ['37','46']:run(ex)
