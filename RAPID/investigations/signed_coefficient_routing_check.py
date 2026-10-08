"""Fixed hydraulics, signed inflow coefficients: convergence and every-segment audit."""
from pathlib import Path
import sys,json,hashlib
import numpy as np
import pandas as pd
import networkx as nx
from scipy.fft import rfft,irfft,next_fast_len
from scipy.signal import lfilter
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import factorized

sys.path.insert(0,str(Path(__file__).resolve().parent))
from check_explicit_celerity_regularization import (
    ROOT,BASE,STATES,load_frames,longest_travel,transfer,quadratic_peak,duration,
    ccc_mat,rte_mat,mus_rte)
OUT=BASE/'common_caps/signed_coefficient_check'
CLOCKS={'11':[69,46,23],'46':[35,31,7]}


def forcing(ex,dt,travel):
    suffix='_direction130' if ex=='15' else ''
    p=ROOT/f'network_variants/output/{ex}_indep{suffix}__anchored/states/S000_base/rapid/run/outlet_hydrograph.csv'
    h=pd.read_csv(p);times=h.time_seconds.to_numpy();q=h.q_inflow_cms.to_numpy()
    changed=np.flatnonzero(abs(np.diff(q))>np.ptp(q)*1e-8)
    left=max(0,np.searchsorted(times,times[changed[0]]-86400))
    right=min(len(times)-1,changed[-1]+2)
    st=times[left:right+1]-times[left];sq=q[left:right+1]
    end=st[-1]+max(10*86400,3*travel)
    t=np.arange(int(np.ceil(end/dt)))*dt
    cumulative=np.r_[0,np.cumsum(np.diff(st)*(sq[:-1]+sq[1:])/2)]
    def primitive(x):
        ix=np.clip(np.searchsorted(st,x,side='right')-1,0,len(st)-2)
        dx=x-st[ix];s=(sq[ix+1]-sq[ix])/(st[ix+1]-st[ix])
        result=cumulative[ix]+sq[ix]*dx+.5*s*dx*dx
        return np.where(x>=st[-1],cumulative[-1]+(x-st[-1])*sq[-1],result)
    mean=(primitive(t+dt)-primitive(t))/dt
    return t,mean,float(q[0]),float(q[-1]),int(round(np.median(np.diff(times))))


def coefficients(f,dt):
    K=f.rapid_k.to_numpy();den=dt+1.8*K
    a=(dt-.2*K)/den;b=(dt+.2*K)/den;c=(1.8*K-dt)/den
    assert (c>=0).all() and (c<1).all()
    return a,b,c


def audit_segments(ex,f,dt,t,actual,q0,qend):
    """Time-domain recurrence for EVERY 200 m subreach, three input shapes.

    Exact initial equilibrium; no output clipping. Sum the Muskingum storage
    changes and check against integrated boundary inflow/outflow. Controlled
    pulse amplitudes are one, baseline zero, so negative perturbations cannot
    be hidden by a large baseflow.
    """
    start=86400.;rise=12*3600.;fall=24*3600.
    def tri_integral(x):
        a=np.clip(x-start,0,rise);b=np.clip(x-start-rise,0,fall)
        return a*a/(2*rise)+b-b*b/(2*fall)
    triangle=(tri_integral(t+dt)-tri_integral(t))/dt
    rectangle=(np.clip(t+dt-start,0,86400)-np.clip(t-start,0,86400))/dt
    inflow=np.c_[actual,triangle,rectangle];initial=np.array([q0,0.,0.])
    G=nx.DiGraph();outgoing={}
    for r in f.itertuples():G.add_edge(r.u,r.v);outgoing.setdefault(r.u,[]).append(r)
    roots=[v for v in G if G.in_degree(v)==0];ends=[v for v in G if G.out_degree(v)==0]
    assert len(roots)==len(ends)==1
    signals={roots[0]:inflow};steady={roots[0]:initial};shares={roots[0]:1.}
    audits=[];storage_change=np.zeros(3);outlet=None;worst=[]
    for ni,node in enumerate(nx.topological_sort(G)):
        signal=signals.pop(node);initial_node=steady.pop(node);fraction=shares.pop(node)
        if node==ends[0]:outlet=signal;continue
        total=sum(r.width for r in outgoing[node])
        for r in outgoing[node]:
            frac=r.width/total;u=signal*frac;eq=initial_node*frac;share=fraction*frac
            K=r.rapid_k;den=dt+1.8*K
            a=(dt-.2*K)/den;b=(dt+.2*K)/den;c=(1.8*K-dt)/den
            for seg in range(r.count):
                root=(node==roots[0] and seg==0)
                b0,b1=(a+b,0.) if root else (a,b)
                out,_=lfilter([b0,b1],[1,-c],u,axis=0,zi=(b1*eq+c*eq)[None,:])
                assert np.isfinite(out).all()
                # Storage associated with the actual engine recurrence. The
                # root uses held external forcing rather than trapezoidal I.
                delta=K*.9*(out[-1]-eq)
                if not root:delta+=K*.1*(u[-1]-eq)
                storage_change+=delta
                minimum=out.min(axis=0);maximum=out.max(axis=0)
                # Count local turns at an input-amplitude-scaled derivative
                # threshold; reported descriptively, not an accuracy verdict.
                amplitudes=np.array([np.ptp(actual),1.,1.])*share
                for j,name in enumerate(['actual','triangle_12h_24h','rectangle_24h']):
                    differences=np.diff(out[:,j]);active=differences[np.abs(differences)>max(amplitudes[j]*1e-9,1e-13)]
                    turns=int(np.sum(active[1:]*active[:-1]<0))
                    negative_volume=float(-np.minimum(out[:,j],0).sum()*dt)
                    audits.append(dict(example=ex,id_link=r.id_link,segment=seg+1,dt_s=dt,forcing=name,
                        branch_fraction=share,current_inflow_coefficient=b0,previous_outflow_coefficient=c,
                        minimum_cms=minimum[j],maximum_cms=maximum[j],
                        minimum_divided_by_branch_fraction=minimum[j]/share,
                        negative_volume=negative_volume,negative_volume_fraction=(negative_volume/(share*(rise+fall)/2) if j==1 else negative_volume/(share*86400) if j==2 else np.nan),
                        turning_points=turns))
                # Save worst zero-base rectangular response for illustration.
                score=minimum[2]/share
                if not worst or score<worst[0][0]:
                    worst=[(score,r.id_link,seg+1,pd.DataFrame(dict(time_seconds=t+dt/2,local_input=u[:,2]/share,local_output=out[:,2]/share)))]
                u=out
            if r.v in signals:
                signals[r.v]+=u;steady[r.v]+=eq;shares[r.v]+=share
            else:signals[r.v]=u;steady[r.v]=eq;shares[r.v]=share
        if ni%100==0:print('Segment audit',ex,'node',ni,flush=True)
    integrated_input=inflow.sum(axis=0)*dt
    previous=np.vstack([initial,outlet[:-1]])
    integrated_output=((outlet+previous)*.5).sum(axis=0)*dt
    residual=integrated_input-integrated_output-storage_change
    scale=np.maximum(np.abs(integrated_input),1.)
    assert (abs(residual/scale)<1e-8).all(),residual/scale
    tag='' if dt==CLOCKS[ex][0] else f'_dt{dt}'
    pd.DataFrame(audits).to_csv(OUT/f'{ex}{tag}_every_segment_audit.csv',index=False)
    pd.DataFrame(dict(time_seconds=t+dt/2,actual=outlet[:,0],triangle=outlet[:,1],rectangle=outlet[:,2])).to_parquet(OUT/f'{ex}{tag}_audit_outlet.parquet',index=False)
    worst[0][3].to_parquet(OUT/f'{ex}{tag}_worst_rectangle_segment.parquet',index=False)
    return dict(example=ex,dt_s=dt,n_segments=int(f['count'].sum()),
                mass_residual_relative= list(residual/scale),
                worst_rectangle_link=int(worst[0][1]),worst_rectangle_segment=int(worst[0][2]),
                worst_rectangle_relative_min=float(worst[0][0])),outlet[:,0]


def verify_signed():
    # Two serial reaches, downstream current-inflow coefficient < 0.
    f=pd.DataFrame(dict(u=[0,1],v=[1,2],width=[1,1],count=[1,1],rapid_k=[100,1000]))
    net=csc_matrix([[0,0],[1,0]])
    mats=rte_mat(net,*ccc_mat(np.array([100.,1000.]),np.array([.1,.1]),10))
    solve=factorized(mats[0]);state=np.zeros(2);external=np.zeros(2);out=[]
    for i in range(4096):
        external[0]=float(i==0);_,state=mus_rte(*mats,1,state,external,solve);out.append(state[1])
    fft=irfft(transfer(f,10,16384),16384)[:4096]
    err=float(np.max(abs(fft-out)));assert err<1e-12 and min(out)<0
    return err


def main():
    OUT.mkdir(exist_ok=True);error=verify_signed();summary=[];comparisons=[];qa=[];counts=[]
    for ex in STATES:
        loaded,_=load_frames(ex);frames={s:loaded[s,'original'] for s in STATES[ex]}
        travel=max(longest_travel(f) for f in frames.values())
        for dt in CLOCKS[ex]:
            t,inflow,q0,qend,forcing_dt=forcing(ex,dt,travel)
            assert forcing_dt%dt==0
            n=len(t);nfft=next_fast_len(2*n);spectrum=rfft(inflow-q0,nfft);hydros={}
            for state,f in frames.items():
                a,b,c=coefficients(f,dt)
                counts.append(dict(example=ex,state=state,dt_s=dt,negative_coefficient_links=int((a<0).sum()),
                    negative_coefficient_segments=int(f.loc[a<0,'count'].sum()),min_inflow_coefficient=a.min(),
                    min_previous_outflow_coefficient=c.min(),max_previous_outflow_coefficient=c.max()))
                out=q0+irfft(spectrum*transfer(f,dt,nfft),nfft)[:n]
                assert np.isfinite(out).all()
                plateau_error=max(abs(out[0]-q0),abs(out[-1]-qend))/np.ptp(inflow)
                assert plateau_error<1e-6,plateau_error
                h=pd.DataFrame(dict(time_seconds=t+dt/2,inlet_cms=inflow,outlet_cms=out))
                h.to_parquet(OUT/f'{ex}_{state}_dt{dt}.parquet',index=False);hydros[state]=h
                summary.append(dict(example=ex,state=state,dt_s=dt,minimum_outlet=out.min(),
                    peak_lag_hours=(quadratic_peak(h,'outlet_cms')-quadratic_peak(h,'inlet_cms'))/3600,
                    peak_excess=out.max()-q0,plateau_error=plateau_error))
                print('Fixed-hydraulic routing',ex,state,dt,flush=True)
            parent=hydros['S000_base'];pq=parent.outlet_cms.to_numpy();pt=parent.time_seconds.to_numpy()
            rt=.01*(pq.max()-q0);ft=.01*(pq.max()-qend);pdur=duration(pt,pq,q0,qend,rt,ft)
            for state,h in hydros.items():
                if state=='S000_base':continue
                cq=h.outlet_cms.to_numpy();cdur=duration(pt,cq,q0,qend,rt,ft)
                comparisons.append(dict(example=ex,state=state,dt_s=dt,
                    timing_change_minutes=(quadratic_peak(h,'outlet_cms')-quadratic_peak(parent,'outlet_cms'))/60,
                    duration_change_pct=100*(cdur['duration_seconds']/pdur['duration_seconds']-1),
                    peak_excess_change_pct=100*((cq.max()-q0)/(pq.max()-q0)-1)))
            if dt in [CLOCKS[ex][0],CLOCKS[ex][-1]]:
                audit,actual=audit_segments(ex,frames['S000_base'],dt,t,inflow,q0,qend)
                difference=np.max(abs(actual-parent.outlet_cms.to_numpy()))/np.ptp(inflow)
                assert difference<1e-8,difference
                audit['fft_vs_every_segment_relative_error']=float(difference);qa.append(audit)
            pd.DataFrame(summary).to_csv(OUT/'routing_summary.csv',index=False)
            pd.DataFrame(comparisons).to_csv(OUT/'comparisons.csv',index=False)
            pd.DataFrame(counts).to_csv(OUT/'coefficients.csv',index=False)
            (OUT/'segment_validation.json').write_text(json.dumps(qa,indent=2))
    (OUT/'provenance.json').write_text(json.dumps(dict(date='2026-10-02',
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),signed_impulse_engine_error=error,
        fixed='Original bounded robust heads; alpha .65; n .035; X .1; 200m segmentation; .05–5 m/s common caps',
        changed='Only dt; allow negative current-inflow coefficient; retain nonnegative previous-outflow coefficient',
        scope='All selected parent/child outlets at three clocks; every base subreach audited at largest and smallest clocks for actual forcing and two zero-base pulses',
        production_changed=False),indent=2))
    print(pd.DataFrame(summary).to_string(index=False));print(pd.DataFrame(comparisons).to_string(index=False))


if __name__=='__main__':main()
