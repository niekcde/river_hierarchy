"""Conservative internal-head sensitivity, followed by explicit routing caps.

Preserves every unit boundary, all direct-profile nodes, slopes' existing
floors, and stability of previously stable links at a fixed candidate clock.
No production files are modified. Head budgets are sensitivity assumptions.
"""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
from scipy.optimize import linprog
from scipy import sparse

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT/'RAPID/assets/bounded_robust_slopes_20261002'
OUT = BASE/'common_caps/local_heads_and_caps'
MARGIN = .01


def adjust_unit(g, fixed, bad, slope_lo, slope_hi, budget):
    endpoints = set(g.id_us_node)|set(g.id_ds_node)
    implicated = set(g.loc[bad,'id_us_node'])|set(g.loc[bad,'id_ds_node'])
    free = sorted((endpoints-fixed)&implicated)
    if not free or not bad.any() or budget == 0:
        return {}, 0.
    index = {v:i for i,v in enumerate(free)}
    m, n = len(g), len(free)
    C = np.zeros((m,n))
    for i,r in enumerate(g.itertuples()):
        if r.id_us_node in index: C[i,index[r.id_us_node]] += 1
        if r.id_ds_node in index: C[i,index[r.id_ds_node]] -= 1
    length = g.actual_length_m.to_numpy()
    drop = (g.head_us-g.head_ds).to_numpy()
    floor = np.minimum(g.lower_bound.fillna(0).to_numpy()*length, drop)
    low, high = slope_lo*length, slope_hi*length
    # Variables: node shifts, lower/upper drop-deficit slack, |node shift|.
    nv = 2*n+2*m
    rows, rhs = [], []
    def add(a, b): rows.append(a); rhs.append(b)
    for i in range(m):
        # Never sacrifice another link by moving its drop farther from its
        # feasible interval; this also protects already-low slopes.
        guard_low=max(floor[i],min(drop[i],low[i]))
        a=np.zeros(nv); a[:n]=-C[i]; add(a,drop[i]-guard_low)
        if np.isfinite(high[i]):
            guard_high=max(drop[i],high[i])
            a=np.zeros(nv); a[:n]=C[i]; add(a,guard_high-drop[i])
        a=np.zeros(nv); a[:n]=-C[i]; a[n+i]=-1; add(a,drop[i]-low[i])
        if np.isfinite(high[i]):
            a=np.zeros(nv); a[:n]=C[i]; a[n+m+i]=-1; add(a,high[i]-drop[i])
    for j in range(n):
        a=np.zeros(nv); a[j]=1; a[n+2*m+j]=-1; add(a,0.)
        a=np.zeros(nv); a[j]=-1; a[n+2*m+j]=-1; add(a,0.)
    bounds = [(-budget,budget)]*n
    bounds += [(0,None) if b else (0,0) for b in bad]*2
    bounds += [(0,None)]*n
    objective=np.zeros(nv)
    # Normalize deficits by target drop to avoid favoring physically long links.
    objective[n:n+m]=1/np.maximum(low,1e-8)
    objective[n+m:n+2*m]=1/np.maximum(np.where(np.isfinite(high),high,1),1e-8)
    A=sparse.csr_matrix(np.array(rows)); rhs=np.array(rhs)
    first=linprog(objective,A_ub=A,b_ub=rhs,bounds=bounds,method='highs')
    if not first.success: raise RuntimeError(first.message)
    # Lexicographic second objective: the smallest total node movement.
    second_objective=np.zeros(nv); second_objective[n+2*m:]=1
    second=linprog(second_objective,A_ub=sparse.vstack([A,sparse.csr_matrix(objective[None,:])]),
                   b_ub=np.r_[rhs,first.fun+1e-9],bounds=bounds,method='highs')
    if not second.success: raise RuntimeError(second.message)
    return dict(zip(free,second.x[:n])), float(first.fun)


def main():
    OUT.mkdir(exist_ok=True)
    source=BASE/'common_caps/stability_link_details.csv'
    data=pd.read_csv(source,dtype={'example':str})
    clocks=pd.read_csv(BASE/'common_caps/stability_conflict_counts.csv',dtype={'example':str}).set_index('example')
    summaries=[]; link_frames=[]; node_rows=[]
    for ex, source_g in data.groupby('example'):
        g=source_g.reset_index(drop=True)
        nodes=pd.read_parquet(BASE/ex/'floor_0.05/old/nodes.parquet').set_index('id_node')
        units=pd.read_csv(BASE/ex/'floor_0.05/old/units.csv')
        fixed=set(units.source)|set(units.sink)
        direct=g[~g.ownership_kind.isin(['simple_unit','compound_bubble'])]
        fixed |= set(direct.id_us_node)|set(direct.id_ds_node)
        # Also preserve interfaces between different ownership domains.
        incidence={}
        for r in g.itertuples():
            domain=(r.ownership_kind,r.ownership_domain_id)
            for v in [r.id_us_node,r.id_ds_node]: incidence.setdefault(v,set()).add(domain)
        fixed |= {v for v,domains in incidence.items() if len(domains)>1}
        for clock_mode in ['nominal_integer','hourly_divisor_scenario']:
            dt0=int(clocks.loc[ex,'representative_dt_s'])
            c=g.celerity_mps.to_numpy(); L=g.effective_length_m.to_numpy()
            candidates=np.arange(max(1,int(np.ceil(.202*L.max()/5))),int(np.floor(1.782*L.min()/.05))+1)
            if clock_mode=='hourly_divisor_scenario':
                candidates=candidates[3600%candidates==0]
            target=np.clip(c[None,:],.202*L[None,:]/candidates[:,None],1.782*L[None,:]/candidates[:,None])
            count=(~np.isclose(target,c[None,:],rtol=1e-10)).sum(axis=1)
            cost=np.abs(np.log(target/c[None,:]))@g.actual_length_m.to_numpy()
            dt=int(candidates[np.lexsort((abs(candidates-dt0),cost,count))[0]])
            L=g.effective_length_m.to_numpy(); raw=g.celerity_raw_mps.to_numpy()
            c0=g.celerity_mps.to_numpy()
            lo=.202*L/dt; hi=1.782*L/dt
            assert (lo<=5).all() and (hi>=.05).all()
            slo=g.slope.to_numpy()*(lo/raw)**2
            shi=np.where(hi>=5,np.inf,g.slope.to_numpy()*(hi/raw)**2)
            slo=np.where(lo<=.05,0,slo)
            bad=(c0<lo-1e-10)|(c0>hi+1e-10)
            for budget in [0.,.025,.05,.10]:
                heads=nodes['head'].copy()
                for _, unit in g[g.ownership_kind.isin(['simple_unit','compound_bubble'])].groupby(['ownership_kind','ownership_domain_id']):
                    ix=unit.index.to_numpy()
                    shifts,_=adjust_unit(unit,fixed,bad[ix],slo[ix],shi[ix],budget)
                    for node,shift in shifts.items(): heads.loc[node]+=shift
                new_slope=(g.id_us_node.map(heads)-g.id_ds_node.map(heads)).to_numpy()/g.actual_length_m.to_numpy()
                raw_new=raw*np.sqrt(new_slope/g.slope.to_numpy())
                physical_c=np.clip(raw_new,.05,5)
                used=np.clip(physical_c,lo,hi)
                caps=~np.isclose(used,physical_c,rtol=1e-8,atol=1e-10)
                slope_changed=~np.isclose(new_slope,g.slope,rtol=1e-7,atol=1e-12)
                shifts=heads-nodes['head']
                assert shifts.loc[list(fixed)].abs().max()<1e-9
                assert shifts.abs().max()<=budget+1e-8
                assert (new_slope>0).all()
                # Do not worsen inherited solver-tolerance deviations from floors.
                effective_floor=np.minimum(g.lower_bound.fillna(0),g.slope)
                assert ((new_slope-effective_floor)*g.actual_length_m>=-1e-7).all()
                assert not (caps & ~bad).any(), 'Created new unstable link'
                old_target=np.clip(c0,lo,hi)
                assert (np.abs(np.log(used/physical_c))<=np.abs(np.log(old_target/c0))+1e-6).all(), 'Worsened a correction'
                K=L/used; denom=2*K*.9+dt
                coeff=np.array([(dt-.2*K)/denom,(dt+.2*K)/denom,(1.8*K-dt)/denom])
                assert coeff.min()>=-1e-10 and np.allclose(coeff.sum(axis=0),1.)
                key=dict(example=ex,clock=clock_mode,dt_s=dt,head_budget_m=budget)
                summaries.append(dict(**key,baseline_exceptions=int(bad.sum()),
                    changed_nodes=int((shifts.abs()>1e-8).sum()),changed_slopes=int(slope_changed.sum()),
                    max_node_shift_m=shifts.abs().max(),remaining_caps=int(caps.sum()),
                    slow_caps=int((used>physical_c+1e-8).sum()),fast_caps=int((used<physical_c-1e-8).sum()),
                    cap_length_pct=100*g.loc[caps,'actual_length_m'].sum()/g.actual_length_m.sum(),
                    largest_speedup=float((used/physical_c).max()),
                    largest_slowdown=float((physical_c/used).max()),min_coefficient=float(coeff.min())))
                out=g[['id_link','id_us_node','id_ds_node','ownership_kind','ownership_domain_id','actual_length_m','effective_length_m','slope']].copy()
                for k,v in key.items(): out[k]=v
                out['head_us_new']=g.id_us_node.map(heads); out['head_ds_new']=g.id_ds_node.map(heads)
                out['slope_new']=new_slope; out['celerity_raw_new']=raw_new
                out['celerity_common_capped']=physical_c; out['celerity_routing_candidate']=used
                out['local_cap_applied']=caps; out['slope_changed']=slope_changed
                link_frames.append(out)
                for node in heads.index:
                    if abs(shifts.loc[node])>1e-8:
                        node_rows.append(dict(**key,id_node=int(node),head_before=nodes.loc[node,'head'],head_after=heads.loc[node],shift_m=shifts.loc[node]))
        print('Completed',ex,flush=True)
    summary=pd.DataFrame(summaries);summary.to_csv(OUT/'summary.csv',index=False)
    pd.concat(link_frames,ignore_index=True).to_parquet(OUT/'candidate_links.parquet',index=False)
    pd.DataFrame(node_rows).to_csv(OUT/'changed_nodes.csv',index=False)
    (OUT/'provenance.json').write_text(json.dumps(dict(date='2026-10-02',
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),alpha=.65,n=.035,X=.1,
        caps=[.05,5],stability_margin=MARGIN,head_budgets_m=[0,.025,.05,.10],
        interpretation='Bounded numerical redistribution sensitivity, not observation-calibrated slopes',
        scope='16 base networks, no child states or hydrograph rerouting, hourly clock hypothetical',
        fixed='Longitudinal profile, unit source/sink heads, inter-domain interfaces, direct-profile nodes, segmentation, slope floors',
        free='Only interior endpoints of initially unstable links',production_changed=False),indent=2))
    print(summary.groupby(['clock','head_budget_m'])[['baseline_exceptions','changed_nodes','changed_slopes','remaining_caps']].sum().to_string())


if __name__=='__main__': main()
