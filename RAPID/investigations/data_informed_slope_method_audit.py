"""Prototype evaluation, not production promotion: cycle-derived slope targets and graph feasibility."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import networkx as nx
from scipy.stats import theilslopes
ROOT=Path(__file__).resolve().parents[2]
PRE=ROOT/'RAPID/assets/section_slope_feasibility_20261002'
OUT=ROOT/'RAPID/assets/data_informed_slope_method_20261002';OUT.mkdir(parents=True,exist_ok=True)
r=pd.read_csv(PRE/'acquisition_slopes.csv',dtype={'example':str});r.example=r.example.str.zfill(2)
s=pd.read_csv(PRE/'sections.csv',dtype={'example':str});s.example=s.example.str.zfill(2)
sources=json.loads((PRE/'sources.json').read_text());rng=np.random.default_rng(20261002)
records=[];held=[]
for row in s.itertuples():
    g=r[(r.example==row.example)&(r.section==row.section)]
    c=g.groupby('cycle').slope.median().sort_index();y=c.to_numpy();n=len(y)
    rec=dict(example=row.example,section=row.section,kind=row.kind,lo=row.lo,hi=row.hi,n_cycles=n)
    if n>=5:
        boot=np.quantile(rng.choice(y,(2000,n),replace=True),[.1,.5,.75],axis=1)
        q10,p50,p75,p90=np.quantile(y,[.1,.5,.75,.9])
        lo10=float(np.quantile(boot[0],.05));lo75,hi75=np.quantile(boot[2],[.05,.95])
        early=np.quantile(y[:n//2],.75);late=np.quantile(y[n//2:],.75)
        jack=np.array([np.quantile(np.delete(y,i),.75) for i in range(n)])
        rec.update(p10=q10,p50=p50,p75=p75,p90=p90,lower_evidence_bound=lo10,p75_boot_lo=lo75,p75_boot_hi=hi75,early_p75=early,late_p75=late,epoch_ratio=max(early,late)/min(early,late) if min(early,late)>0 else np.inf,jackknife_relative_range=np.ptp(jack)/abs(p75) if p75 else np.inf,positive_evidence=lo10>0,target_positive=p75>0,p75_over_p50=p75/p50 if p50 else np.nan)
        for q in [.5,.65,.75,.9]:
            pred=np.array([np.quantile(np.delete(y,i),q) for i in range(n)])
            err=y-pred;scale=max(np.median(np.abs(y)),1e-15)
            held.append(dict(example=row.example,section=row.section,kind=row.kind,quantile=q,n_cycles=n,mae_scaled=np.mean(abs(err))/scale,signed_bias_scaled=np.mean(pred-y)/scale,heldout_below_target=np.mean(y<pred),pinball_scaled=np.mean(np.maximum(q*err,(q-1)*err))/scale))
    records.append(rec)
a=pd.DataFrame(records);a.to_csv(OUT/'section_evidence.csv',index=False);pd.DataFrame(held).to_csv(OUT/'heldout_cycle_scores.csv',index=False)
# Shift 10 km windows by 5 km: existing centered windows alone cannot establish robust boundaries.
shifted=[]
for ex,p in sources.items():
    o=pd.read_parquet(Path(p)/'reference_profile_observations.parquet')
    original=s[(s.example==ex)&(s.kind=='10km')];xmin=original.lo.min();xmax=original.hi.max()
    for lo in np.arange(xmin+5000,xmax-9999,10000):
        hi=lo+10000;vals=[]
        for aid,g in o[o.reference_section_dist_out_m.between(lo,hi)].groupby('acquisition_id'):
            z=g.groupby('reference_section_dist_out_m').wse_m.mean().sort_index();x=z.index.to_numpy()
            if len(x)<6 or np.ptp(x)/10000<.7 or np.diff(x).max()/10000>.3:continue
            vals.append((g.cycle.iloc[0],theilslopes(z.to_numpy(),x).slope))
        c=pd.DataFrame(vals,columns=['cycle','slope']).groupby('cycle').slope.median()
        mids=(original.lo+original.hi)/2;nearest=original.iloc[np.argsort(abs(mids-(lo+5000)))[:2]]
        base=a[(a.example==ex)&a.section.isin(nearest.section)].p75.dropna()
        shifted.append(dict(example=ex,lo=lo,hi=hi,n_cycles=len(c),p75=c.quantile(.75),unshifted_neighbor_p75=base.mean(),ratio_to_unshifted=c.quantile(.75)/base.mean() if len(c) and len(base) and base.mean()>0 else np.nan))
    print(ex,'shifted windows done',flush=True)
pd.DataFrame(shifted).to_csv(OUT/'shifted_windows.csv',index=False)
# Build inspectable piecewise-linear candidates: positively evidenced windows use quantile target;
# unresolved windows explicitly retain current endpoint head drop. No arbitrary dimensional floor.
# This is a stress test: window boundaries are not final geomorphic sections.
profiles=[];domainrows=[];case_rows=[]
for ex,p in sources.items():
    suffix='_direction130' if ex=='15' else '';root=ROOT/'network_variants/output'/f'{ex}_indep{suffix}__anchored/states/S000_base/hydraulic_context'
    nodes=pd.read_csv(root/'node_hydraulic_context.csv').set_index('id_node');links=pd.read_csv(root/'link_hydraulic_context.csv')
    f=pd.read_parquet(Path(p)/'reference_profile.parquet');xx=f.reference_section_dist_out_m.to_numpy();yy=f.fitted_wse_m.to_numpy()
    for size in ['10km','20km']:
        bins=a[(a.example==ex)&(a.kind==size)].sort_values('lo').copy()
        for q in ['p50','p75']:
            edges=np.r_[bins.lo.to_numpy(),bins.hi.iloc[-1]];old=np.interp(edges,xx,yy);oldslopes=np.diff(old)/np.diff(edges)
            use=bins.positive_evidence.fillna(False).to_numpy(bool)&np.isfinite(bins[q].to_numpy(float))
            slopes=np.where(use,bins[q].to_numpy(float),oldslopes)
            heads=old[0]+np.r_[0,np.cumsum(slopes*np.diff(edges))]
            for i,v in enumerate(bins.itertuples()):profiles.append(dict(example=ex,scale=size,target=q,lo=v.lo,hi=v.hi,slope=slopes[i],evidence_based=use[i],head_lo=heads[i],head_hi=heads[i+1],old_slope=oldslopes[i]))
            case_rows.append(dict(example=ex,scale=size,target=q,evidence_length_pct=100*np.diff(edges)[use].sum()/np.ptp(edges),head_drop_ratio=(heads[-1]-heads[0])/(old[-1]-old[0]),min_evidence_slope=slopes[use].min() if use.any() else np.nan,n_evidence_below_1e5=int((slopes[use]<1e-5).sum())))
            for did,d in links[links.ownership_kind=='compound_bubble'].groupby('ownership_domain_id'):
                graph=nx.MultiDiGraph()
                for v in d.itertuples():
                    # Length-average section targets over projected span rather than midpoint transfer.
                    ux=nodes.loc[v.id_us_node,'reference_section_dist_out_m'];dx=nodes.loc[v.id_ds_node,'reference_section_dist_out_m'];lo,hi=sorted([ux,dx])
                    overlaps=np.maximum(0,np.minimum(edges[1:],hi)-np.maximum(edges[:-1],lo))
                    if overlaps.sum()>0:
                        target=float(np.average(slopes,weights=overlaps));covered=float(overlaps[use].sum()/overlaps.sum())
                    else:
                        idx=min(len(slopes)-1,max(0,np.searchsorted(edges,lo)-1));target=float(slopes[idx]);covered=float(use[idx])
                    graph.add_edge(int(v.id_us_node),int(v.id_ds_node),key=int(v.id_link),weight=target*v.actual_length_m,coverage=covered)
                src=[n for n in graph if graph.in_degree(n)==0];snk=[n for n in graph if graph.out_degree(n)==0]
                assert len(src)==len(snk)==1 and nx.is_directed_acyclic_graph(graph)
                src,snk=src[0],snk[0];dist={n:-np.inf for n in graph};dist[src]=0
                for u in nx.topological_sort(graph):
                    for _,v,k,attr in graph.out_edges(u,keys=True,data=True):dist[v]=max(dist[v],dist[u]+attr['weight'])
                longest=dist[snk];usx=nodes.loc[src,'reference_section_dist_out_m'];dsx=nodes.loc[snk,'reference_section_dist_out_m']
                for boundary in ['current','candidate']:
                    drop=float(nodes.loc[src,'profile_head_m']-nodes.loc[snk,'profile_head_m']) if boundary=='current' else float(np.interp(usx,edges,heads)-np.interp(dsx,edges,heads))
                    t=drop/longest if longest>0 else np.nan
                    # Construct certificate of feasibility for the max-min lower slopes.
                    certificate={v:drop-t*dist[v] for v in graph}
                    slack=min(certificate[u]-certificate[v]-t*z['weight'] for u,v,z in graph.edges(data=True))
                    assert slack>=-1e-8 and abs(certificate[snk])<1e-8
                    domainrows.append(dict(example=ex,domain=int(did),scale=size,target=q,boundary=boundary,n_links=len(d),all_links_evidenced=all(z['coverage']>1-1e-9 for _,_,z in graph.edges(data=True)),boundary_drop_m=drop,longest_target_path_drop_m=longest,max_feasible_target_fraction=t,certificate_min_slack_m=slack))
    print(ex,'candidate and compound checks done',flush=True)
pd.DataFrame(profiles).to_csv(OUT/'candidate_sections.csv',index=False);pd.DataFrame(case_rows).to_csv(OUT/'candidate_cases.csv',index=False);pd.DataFrame(domainrows).to_csv(OUT/'compound_maxmin.csv',index=False)
(OUT/'provenance.json').write_text(json.dumps(dict(sources=sources,seed=20261002,bootstrap_draws=2000,purpose='Read-only diagnostic; candidate section windows not final boundaries'),indent=2))
print('DONE',OUT)
