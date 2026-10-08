"""All materialized states: shared-clock, fixed-hydraulic frequency screening.

The reporting grid is independent of solver dt. Validate its band-limited
input representation against native-step runs before interpreting screening.
"""
from pathlib import Path
import sys,json,hashlib,time
import numpy as np
import pandas as pd
import networkx as nx
from scipy.fft import rfft,irfft,next_fast_len
sys.path.insert(0,str(Path(__file__).resolve().parent))
from full_signed_routing_validation import ROOT,BASE,OUT,EXAMPLES
from check_explicit_celerity_regularization import duration,quadratic_peak


def grid_input(ex,maxtravel,step):
    suffix='_direction130' if ex=='15' else ''
    saved=ROOT/f'network_variants/output/{ex}_indep{suffix}__anchored/states/S000_base/rapid/run/outlet_hydrograph.csv'
    if saved.exists():
        h=pd.read_csv(saved);t=h.time_seconds.to_numpy();q=h.q_inflow_cms.to_numpy()
    else:
        # No observed-event run exists: explicitly labelled numerical pulse.
        t=np.arange(0,4*86400+3600,3600.)
        q=1+np.maximum(0,np.minimum((t-86400)/(12*3600),(86400+36*3600-t)/(24*3600)))
    change=np.flatnonzero(abs(np.diff(q))>np.ptp(q)*1e-8)
    left=max(0,np.searchsorted(t,t[change[0]]-86400));right=min(len(t)-1,change[-1]+2)
    st=t[left:right+1]-t[left];sq=q[left:right+1]
    end=st[-1]+max(10*86400,3*maxtravel)
    n=int(np.ceil(end/step));nfft=next_fast_len(2*n+int(np.ceil(4*maxtravel/step)))
    grid=np.arange(nfft)*step;signal=np.interp(grid,st,sq)
    # Smoothly return to the initial level outside the analyzed record, avoiding
    # a sharp artificial jump at the periodic boundary. Causal tail is checked.
    begin=n;finish=n+int(np.ceil(max(86400,maxtravel)/step))
    x=np.linspace(0,np.pi,finish-begin)
    signal[begin:finish]=q[0]+(q[-1]-q[0])*(1+np.cos(x))/2
    signal[finish:]=q[0]
    return grid[:n],signal[:n],rfft(signal-q[0]),nfft,q[0],q[-1],int(round(np.median(np.diff(t))))


def response(f,dt,nfft,step,cache=None):
    omega=2*np.pi*np.arange(nfft//2+1)/(nfft*step);z=np.exp(-1j*omega*dt)
    G=nx.DiGraph();out={}
    for r in f.itertuples():G.add_edge(r.u,r.v);out.setdefault(r.u,[]).append(r)
    roots=[v for v in G if G.in_degree(v)==0];ends=[v for v in G if G.out_degree(v)==0]
    if len(roots)!=1 or len(ends)!=1 or not nx.is_directed_acyclic_graph(G):raise ValueError('Invalid routing graph')
    signals={roots[0]:np.ones(len(z),complex)}
    for u in nx.topological_sort(G):
        signal=signals.pop(u)
        if u==ends[0]:answer=signal;continue
        total=sum(r.width for r in out[u])
        for r in out[u]:
            key=(float(r.rapid_k),int(r.count),u==roots[0])
            H=cache.get(key) if cache is not None else None
            if H is None:
                K=r.rapid_k;den=dt+1.8*K
                a=(dt-.2*K)/den;b=(dt+.2*K)/den;c=(1.8*K-dt)/den
                if c< -1e-12:raise ValueError('Upper positivity bound violated')
                single=(a+b*z)/(1-c*z)
                H=((a+b)/(1-c*z))*single**(r.count-1) if u==roots[0] else single**r.count
                # Bound cache memory; entries beyond this are calculated only.
                if cache is not None and len(cache)<max(1,128*1024**2//H.nbytes):cache[key]=H
            value=signal*(r.width/total)*H
            if r.v in signals:signals[r.v]+=value
            else:signals[r.v]=value
    assert abs(answer[0]-1)<1e-9
    # Exact interval averaging for each Fourier component. Time is reported at
    # interval centers, so no extra half-dt phase shift is required.
    return answer*np.sinc(omega*dt/(2*np.pi))


def summarize(t,q,inlet,q0,qe):
    h=pd.DataFrame(dict(time_seconds=t,outlet_cms=q,inlet_cms=inlet))
    p=quadratic_peak(h,'outlet_cms');ip=quadratic_peak(h,'inlet_cms')
    return dict(peak_time_s=p,lag_hours=(p-ip)/3600,peak_excess=q.max()-q0,
        min_flow=q.min(),max_flow=q.max(),initial_error=abs(q[0]-q0)/np.ptp(inlet),
        final_error=abs(q[-1]-qe)/np.ptp(inlet))


def run(ex,step=120.):
    dest=OUT/ex;inv=pd.read_csv(dest/'inventory.csv',dtype={'example':str})
    good=inv[inv.status.eq('ready')].drop_duplicates('fingerprint').copy()
    baseline=inv[inv.family.eq('indep')&inv.state.eq('S000_base')].iloc[0]
    if baseline.status!='ready':raise ValueError('Base not materialized')
    baseid=baseline.fingerprint
    if not inv.parent_state.eq('S000_base').all():raise ValueError('Non-base parent needs explicit pairing')
    # Parent first, then each unique geometry once; island aliases share results.
    good=pd.concat([good[good.fingerprint.eq(baseid)],good[~good.fingerprint.eq(baseid)]])
    t,inlet,spectrum,nfft,q0,qe,forcing_dt=grid_input(ex,good.max_travel_seconds.max(),step)
    upper=float(good.upper_dt_seconds.min())
    divisors=[i for i in range(1,forcing_dt+1) if forcing_dt%i==0 and i<=.99*upper]
    if not divisors:raise ValueError('No integer divisor under upper bound')
    clocks=[max(divisors)]
    for factor in [.5,.25]:
        options=[x for x in divisors if x<=factor*clocks[0]]
        if options and max(options) not in clocks:clocks.append(max(options))
    if len(clocks)<2:raise ValueError('Cannot establish timestep convergence with available integer clocks')
    rows=[];pairs=[];errors=[]
    # Two well-separated clocks for complete screen. Three-clock pilot already
    # exists; native followups can investigate states flagged by this screen.
    clocks=[clocks[0],clocks[-1]]
    for dt in clocks:
        cache={};base_q=None;base_m=None;base_d=None
        for i,r in enumerate(good.itertuples()):
            try:
                f=pd.read_parquet(r.frame);q=q0+irfft(spectrum*response(f,dt,nfft,step,cache),nfft)[:len(t)]
                if not np.isfinite(q).all():raise ValueError('Nonfinite output')
                m=summarize(t,q,inlet,q0,qe)
                if max(m['initial_error'],m['final_error'])>1e-4:raise ValueError('Censored/wrapped plateau')
                row=dict(example=ex,fingerprint=r.fingerprint,dt_s=dt,**m)
                rows.append(row)
                if r.fingerprint==baseid:
                    base_q=q;base_m=m;rise=.01*m['peak_excess'];fall=.01*(q.max()-qe)
                    base_d=duration(t,q,q0,qe,rise,fall)
                else:
                    child_d=duration(t,q,q0,qe,rise,fall)
                    pairs.append(dict(example=ex,fingerprint=r.fingerprint,dt_s=dt,
                        timing_change_minutes=(m['peak_time_s']-base_m['peak_time_s'])/60,
                        duration_change_pct=100*(child_d['duration_seconds']/base_d['duration_seconds']-1),
                        peak_excess_change_pct=100*(m['peak_excess']/base_m['peak_excess']-1)))
                # Save a compact, common 120 s output grid for every state.
                pd.DataFrame(dict(time_seconds=t,inlet_cms=inlet,outlet_cms=q)).to_parquet(dest/f'route_{r.fingerprint}_dt{dt}.parquet',index=False)
            except Exception as err:
                errors.append(dict(example=ex,fingerprint=r.fingerprint,dt_s=dt,error=str(err)))
                if r.fingerprint==baseid:raise
            if i%25==0:
                pd.DataFrame(rows).to_csv(dest/'routing_summary.csv',index=False)
                pd.DataFrame(pairs).to_csv(dest/'comparisons.csv',index=False)
                print('SCREEN',ex,dt,i,'/',len(good),flush=True)
    pd.DataFrame(rows).to_csv(dest/'routing_summary.csv',index=False)
    pd.DataFrame(pairs).to_csv(dest/'comparisons.csv',index=False)
    pd.DataFrame(errors).to_csv(dest/'routing_errors.csv',index=False)
    (dest/'routing_method.json').write_text(json.dumps(dict(reporting_step_s=step,clocks=clocks,
        forcing_kind=('standard_triangle_no_saved_event' if ex in ['13','45','48'] else 'saved_actual_event'),forcing_dt_s=forcing_dt,nfft=nfft,unique_states=len(good),base_fingerprint=baseid,
        note='Band-limited representation on common output grid; requires native-step cross-check; fixed K across clocks',
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),indent=2))
    print('SCREEN COMPLETE',ex,len(pairs),'comparisons',len(errors),'errors',flush=True)


if __name__=='__main__':
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(max_workers=2) as pool:list(pool.map(run,sys.argv[1:] or EXAMPLES))
