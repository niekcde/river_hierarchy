"""Read-only all-case cycle slope support and compound head-bound feasibility."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import networkx as nx
from scipy.stats import theilslopes
from scipy.optimize import linprog
from scipy.sparse import lil_matrix
ROOT=Path(__file__).resolve().parents[2]; OUT=ROOT/'RAPID/assets/section_slope_feasibility_20261002';OUT.mkdir(parents=True,exist_ok=True)
EX=['02','03','05','11','13','15','21','22','37','41','45','46','48','54','55','56']
sections=[];fits=[];domains=[];sources={}
for ex in EX:
    context=ROOT/'network_variants/output'/f'{ex}_indep{"_direction130" if ex=="15" else ""}__anchored/states/S000_base/hydraulic_context'
    m=json.loads((context/'hydraulic_context_manifest.json').read_text());p=Path(m['slope_refit_promotion']['reference_profile_dir']);sources[ex]=str(p)
    f=pd.read_parquet(p/'reference_profile.parquet').sort_values('reference_section_dist_out_m');o=pd.read_parquet(p/'reference_profile_observations.parquet')
    nodes=pd.read_csv(context/'node_hydraulic_context.csv').set_index('id_node');links=pd.read_csv(context/'link_hydraulic_context.csv')
    xmin=float(nodes.reference_section_dist_out_m.min());xmax=float(nodes.reference_section_dist_out_m.max())
    requests=[]
    for size in [10000,20000]:
        edges=np.r_[np.arange(xmin,xmax,size),xmax]
        for j,(lo,hi) in enumerate(zip(edges[:-1],edges[1:])):requests.append((f'{size//1000}km_{j:03d}',f'{size//1000}km',lo,hi))
    for rid,g in o.groupby('reach_id'):
        lo=max(xmin,g.reference_section_dist_out_m.min());hi=min(xmax,g.reference_section_dist_out_m.max())
        if hi>lo:requests.append((f'reach_{rid}','SWORD_reach',lo,hi))
    if ex=='11':requests += [('downstream_0_44','reviewed',0,44000),('downstream_0_40','reviewed',0,40000)]
    local=[]
    for name,kind,lo,hi in requests:
        span=hi-lo;rows=[]
        for aid,g in o[o.reference_section_dist_out_m.between(lo,hi)].groupby('acquisition_id',sort=False):
            z=g.groupby('reference_section_dist_out_m').wse_m.mean().sort_index();x=z.index.to_numpy();y=z.to_numpy()
            if len(x)<6 or span<1000:continue
            coverage=np.ptp(x)/span;gap=np.diff(x).max()/span
            if coverage<.7 or gap>.3:continue
            slope=theilslopes(y,x).slope
            profile=theilslopes(np.interp(x,f.reference_section_dist_out_m,f.fitted_wse_m),x).slope
            row=dict(example=ex,section=name,kind=kind,lo=lo,hi=hi,acquisition_id=aid,cycle=g.cycle.iloc[0],n_positions=len(x),coverage=coverage,max_gap_fraction=gap,slope=slope,profile_slope=profile)
            fits.append(row);rows.append(row)
        z=pd.DataFrame(rows); cyc=z.groupby('cycle')[['slope','profile_slope']].median() if len(z) else pd.DataFrame(columns=['slope','profile_slope'])
        q=cyc.slope.quantile([.1,.25,.5,.75,.9]);n=len(cyc);positive=float((cyc.slope>0).mean()) if n else np.nan
        eligible=n>=5 and positive>=.8 and q.loc[.1]>0
        # Split cycles chronologically into alternating groups: simple sensitivity screen, not independent validation.
        a=cyc.slope.iloc[::2].median();b=cyc.slope.iloc[1::2].median();split=abs(a-b)/abs(q.loc[.5]) if n>=5 and q.loc[.5]!=0 else np.nan
        rec=dict(example=ex,section=name,kind=kind,lo=lo,hi=hi,length_km=span/1000,n_cycles=n,positive_fraction=positive,q10=q.loc[.1],q25=q.loc[.25],median=q.loc[.5],q75=q.loc[.75],q90=q.loc[.9],profile_median=cyc.profile_slope.median(),split_relative_difference=split,supported=eligible,stable_supported=bool(eligible and split<=.5),profile_endpoint_slope=(np.interp(hi,f.reference_section_dist_out_m,f.fitted_wse_m)-np.interp(lo,f.reference_section_dist_out_m,f.fitted_wse_m))/span)
        local.append(rec);sections.append(rec)
    sec=pd.DataFrame(local)
    for did,d in links[links.ownership_kind=='compound_bubble'].groupby('ownership_domain_id'):
        graph=nx.DiGraph();graph.add_edges_from(zip(d.id_us_node.astype(int),d.id_ds_node.astype(int)))
        src=[v for v in graph if graph.in_degree(v)==0];snk=[v for v in graph if graph.out_degree(v)==0]
        base=dict(example=ex,domain=int(did),n_links=len(d))
        if len(src)!=1 or len(snk)!=1 or not nx.is_directed_acyclic_graph(graph):
            domains.append(dict(**base,status='unsupported boundary topology'));continue
        src,snk=src[0],snk[0];drop=nodes.loc[src,'profile_head_m']-nodes.loc[snk,'profile_head_m'];dist=nodes.loc[src,'reference_section_dist_out_m']-nodes.loc[snk,'reference_section_dist_out_m'];ref=drop/dist
        ns=list(graph);index={v:i for i,v in enumerate(ns)};A=lil_matrix((len(d),len(ns)));length=d.actual_length_m.to_numpy();lows=[];highs=[];supported=0
        for i,r in enumerate(d.itertuples()):
            A[i,index[int(r.id_us_node)]]=1;A[i,index[int(r.id_ds_node)]]=-1
            mid=(nodes.loc[r.id_us_node,'reference_section_dist_out_m']+nodes.loc[r.id_ds_node,'reference_section_dist_out_m'])/2
            candidates=sec[(sec.kind=='20km')&(sec.lo<=mid)&(sec.hi>=mid)&sec.stable_supported]
            if len(candidates):
                v=candidates.iloc[0];lows.append(v.q10);highs.append(v.q90);supported+=1
            else:lows.append(max(1e-10,.01*ref));highs.append(np.inf)
        A=A.tocsr();bounds=[(None,None)]*len(ns);bounds[index[src]]=(float(drop),float(drop));bounds[index[snk]]=(0.,0.)
        from scipy.sparse import vstack
        for label,factor,upper in [('observed_lower_only',1.,False),('observed_band',1.,True),('widened_band',.5,True)]:
            # Widen only data-informed bounds; retain existing numerical floor elsewhere.
            lower=np.array(lows);high=np.array(highs);known=np.isfinite(high)
            lower[known]*=factor;high[known]/=factor
            mat=-A;rhs=-lower*length
            if upper and known.any():mat=vstack([mat,A[known]]);rhs=np.r_[rhs,high[known]*length[known]]
            res=linprog(np.zeros(len(ns)),A_ub=mat,b_ub=rhs,bounds=bounds,method='highs')
            domains.append(dict(**base,source=src,sink=snk,boundary_drop_m=drop,supported_links=supported,supported_fraction=supported/len(d),scenario=label,feasible=bool(res.success),status=res.message))
    print(ex,len(local),'sections',flush=True)
pd.DataFrame(sections).to_csv(OUT/'sections.csv',index=False);pd.DataFrame(fits).to_csv(OUT/'acquisition_slopes.csv',index=False);pd.DataFrame(domains).to_csv(OUT/'compound_feasibility.csv',index=False);(OUT/'sources.json').write_text(json.dumps(sources,indent=2))
s=pd.DataFrame(sections);summ=[]
for (ex,kind),g in s.groupby(['example','kind']):
    summ.append(dict(example=ex,kind=kind,n_sections=len(g),supported=int(g.supported.sum()),stable_supported=int(g.stable_supported.sum()),supported_length_pct=100*g.loc[g.stable_supported,'length_km'].sum()/g.length_km.sum()))
pd.DataFrame(summ).to_csv(OUT/'coverage.csv',index=False)
print(pd.DataFrame(summ).query("kind=='20km'").to_string(index=False))
