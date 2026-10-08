"""Full isolated candidate profiles and owned-unit solves; no production writes."""
from pathlib import Path
import sys,json,time,hashlib
import numpy as np
import pandas as pd
import networkx as nx
from scipy import sparse
from scipy.stats import theilslopes
from scipy.sparse.linalg import factorized
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT/'RAPID/src'),str(ROOT)]
from rapid_tools.reference_profile import _curvature_penalty
OUT=ROOT/'RAPID/assets/data_informed_base_slopes_20261002';OUT.mkdir(parents=True,exist_ok=True)
EVID=ROOT/'RAPID/assets/data_informed_slope_method_20261002'
evidence=pd.read_csv(EVID/'section_evidence.csv',dtype={'example':str});evidence.example=evidence.example.str.zfill(2)
sources=json.loads((EVID/'provenance.json').read_text())['sources']

def qp(P,q,A,lower,upper,x0,*,tol=2e-7,maxiter=60000):
    """Convex QP via scaled ADMM; return only with explicit residual convergence."""
    P=sparse.csc_matrix(P);A=sparse.csc_matrix(A);At=A.T.tocsc();n=P.shape[0]
    rho=1.;sigma=1e-8;x=np.asarray(x0,float).copy();z=np.clip(A@x,lower,upper);u=np.zeros(len(z))
    def build():return factorized(P+rho*(At@A)+sigma*sparse.eye(n,format='csc'))
    solve=build();start=time.monotonic()
    for it in range(maxiter):
        x=solve(-q+rho*(At@(z-u))+sigma*x)
        ax=A@x;old=z.copy();z=np.clip(ax+u,lower,upper);u+=ax-z
        primal=float(np.max(abs(ax-z)));dual=float(np.max(abs(rho*(At@(z-old)))))
        epsp=tol*(1+max(np.max(abs(ax)),np.max(abs(z))))
        epsd=tol*(1+max(np.max(abs(P@x)),np.max(abs(q)),np.max(abs(At@(rho*u)))))
        if primal<=epsp and dual<=epsd:
            return x,dict(iterations=it+1,primal_residual=primal,dual_residual=dual,primal_tolerance=epsp,dual_tolerance=epsd,seconds=time.monotonic()-start)
        if (it+1)%100==0:
            ratio=(primal/epsp)/(dual/epsd+1e-15)
            new=rho*2 if ratio>10 else rho/2 if ratio<.1 else rho
            new=min(1e6,max(1e-6,new))
            if new!=rho:u*=rho/new;rho=new;solve=build()
    raise RuntimeError(f'QP did not converge: primal={primal}/{epsp}, dual={dual}/{epsd}')

def interpolate_row(grid,x,nvar):
    i=int(np.searchsorted(grid,x));row=sparse.lil_matrix((1,nvar))
    if i==0:row[0,0]=1
    elif i==len(grid):row[0,len(grid)-1]=1
    else:
        t=(x-grid[i-1])/(grid[i]-grid[i-1]);row[0,i-1]=1-t;row[0,i]=t
    return row.tocsr()

def run(ex):
    out=OUT/ex;out.mkdir(exist_ok=True);start=time.monotonic()
    p=Path(sources[ex]);suffix='_direction130' if ex=='15' else ''
    hc=ROOT/'network_variants/output'/f'{ex}_indep{suffix}__anchored/states/S000_base/hydraulic_context'
    old=pd.read_parquet(p/'reference_profile.parquet').sort_values('reference_section_dist_out_m');obs=pd.read_parquet(p/'reference_profile_observations.parquet')
    nodes=pd.read_csv(hc/'node_hydraulic_context.csv');links=pd.read_csv(hc/'link_hydraulic_context.csv');audit=json.loads((p/'reference_profile_audit.json').read_text())['summary']
    x=old.reference_section_dist_out_m.to_numpy();n=len(x);acqs=pd.Index(sorted(obs.acquisition_id.unique()));ai=acqs.get_indexer(obs.acquisition_id);ni=np.searchsorted(x,obs.reference_section_dist_out_m);assert np.allclose(x[ni],obs.reference_section_dist_out_m,atol=1e-6)
    nv=n+len(acqs);N=len(obs);idx=np.arange(N)
    design=sparse.coo_matrix((np.ones(2*N),(np.r_[idx,idx],np.r_[ni,n+ai])),shape=(N,nv)).tocsr()
    # Equal cycle contribution; acquisitions within a cycle share that total according to valid samples.
    counts=obs.groupby('cycle').wse_m.transform('size').to_numpy();w=N/(obs.cycle.nunique()*counts)
    center=float(old.fitted_wse_m.median());y=obs.wse_m.to_numpy()-center
    P=2*(design.T@sparse.diags(w)@design)+sparse.block_diag((2*float(audit['lambda_value'])*_curvature_penalty(x),sparse.csr_matrix((len(acqs),len(acqs)))))
    q=-2*(design.T@(w*y))
    sections=evidence[(evidence.example==ex)&(evidence.kind=='SWORD_reach')].copy()
    valid=sections.positive_evidence.eq(True)&sections.p75.gt(0)
    sections['used']=valid
    # Do not treat strongly curved sections as constant-gradient evidence.
    for i,r in sections.iterrows():
        endpoint=(np.interp(r.hi,x,old.fitted_wse_m)-np.interp(r.lo,x,old.fitted_wse_m))/(r.hi-r.lo)
        mask=old.reference_section_dist_out_m.between(r.lo,r.hi)
        bulk=theilslopes(old.loc[mask,'fitted_wse_m'],old.loc[mask,'reference_section_dist_out_m']).slope if mask.sum()>=3 else endpoint
        sections.loc[i,'current_endpoint_to_bulk_ratio']=endpoint/bulk if np.isfinite(bulk) and bulk>0 else np.nan
    sections['used'] &= sections.current_endpoint_to_bulk_ratio.between(.5,2.)
    mid=(x[:-1]+x[1:])/2;dx=np.diff(x);floor=np.full(n-1,float(audit['minimum_slope_m_per_m']));supported=np.zeros(n-1,bool);target=np.diff(old.fitted_wse_m)/dx
    penalty_rows=[];penalty_targets=[];penalty_weights=[]
    for i,r in sections[sections.used].iterrows():
        mask=(mid>=r.lo)&(mid<=r.hi)
        floor[mask]=r.lower_evidence_bound;target[mask]=r.p75;supported[mask]=True
        row=interpolate_row(x,r.hi,nv)-interpolate_row(x,r.lo,nv)
        nearby=obs[obs.reference_section_dist_out_m.between(r.lo,r.hi)]
        noise=float(nearby.wse_uncertainty_m.median());noise=noise if np.isfinite(noise) and noise>0 else float(audit['weighted_rms_m'])
        spread=(r.p75_boot_hi-r.p75_boot_lo)*(r.hi-r.lo)
        weight=float(r.n_cycles*noise**2/(noise**2+spread**2))
        penalty_rows.append(row);penalty_targets.append(r.p75*(r.hi-r.lo));penalty_weights.append(weight)
        sections.loc[i,'target_penalty_weight']=weight
    if penalty_rows:
        B=sparse.vstack(penalty_rows);W=sparse.diags(penalty_weights);P=P+2*B.T@W@B;q=q-2*(B.T@(np.array(penalty_weights)*penalty_targets))
    D=sparse.diags([-np.ones(n-1),np.ones(n-1)],[0,1],shape=(n-1,n));D=sparse.hstack([D,sparse.csr_matrix((n-1,len(acqs)))])
    gauge=sparse.csr_matrix((np.ones(len(acqs))/len(acqs),(np.zeros(len(acqs)),n+np.arange(len(acqs)))),shape=(1,nv))
    # Condition constraint rows in metres; individual interval lower gradients are a declared local-uniformity assumption.
    A=sparse.vstack([D,gauge]);lower=np.r_[floor*dx,0.];upper=np.r_[np.full(n-1,np.inf),0.]
    offsets=obs.groupby('acquisition_id').acquisition_stage_offset_m.first().reindex(acqs).to_numpy();mean=offsets.mean()
    initial=np.r_[old.fitted_wse_m.to_numpy()-center+mean,offsets-mean]
    sol,solver=qp(P,q,A,lower,upper,initial,tol=1e-6,maxiter=120000)
    heights=sol[:n]+center;profile=old.copy();profile['old_fitted_wse_m']=old.fitted_wse_m;profile['fitted_wse_m']=heights
    profile['interval_slope_m_per_m']=np.r_[np.nan,np.diff(heights)/dx];profile['minimum_interval_slope']=np.r_[np.nan,floor];profile['data_supported_floor']=np.r_[False,supported];profile['floor_active']=np.r_[False,np.isclose(np.diff(heights),floor*dx,atol=2e-5,rtol=1e-7)]
    profile=profile.drop(columns=[c for c in profile if c.startswith('manual_profile_adjustment')],errors='ignore')
    # All fallbacks remain explicit. No old manual bridge is silently applied after the new fit.
    # Unsupported bridge regions remain governed by old fallback floor, not the old imposed linear shape.
    profile.to_parquet(out/'reference_profile.parquet',index=False);sections.to_csv(out/'section_constraints.csv',index=False)
    obs=obs.copy();obs['old_residual_m']=obs.residual_m;obs['fitted_profile_wse_m']=heights[ni];obs['acquisition_stage_offset_m']=sol[n:][ai];obs['residual_m']=obs.wse_m-obs.fitted_profile_wse_m-obs.acquisition_stage_offset_m;obs['stage_aligned_wse_m']=obs.wse_m-obs.acquisition_stage_offset_m
    obs.to_parquet(out/'reference_profile_observations.parquet',index=False)
    pd.DataFrame({'acquisition_id':acqs,'stage_offset_m':sol[n:]}).to_parquet(out/'reference_profile_acquisition_offsets.parquet',index=False)
    nodeheads=pd.Series(np.interp(nodes.reference_section_dist_out_m,x,heights),index=nodes.id_node.astype(int));original=nodeheads.copy();nodepos=nodes.set_index('id_node').reference_section_dist_out_m
    # Derive local target at any physical edge's projected span by integrating interval targets.
    profile_target=np.where(supported,target,np.diff(heights)/dx)
    domain_audits=[]
    for (kind,did),d in links[links.ownership_kind.isin(['simple_unit','compound_bubble'])].groupby(['ownership_kind','ownership_domain_id']):
        graph=nx.MultiDiGraph();Ts=[]
        for r in d.itertuples():
            lo,hi=sorted([nodepos.loc[r.id_us_node],nodepos.loc[r.id_ds_node]]);overlap=np.maximum(0,np.minimum(x[1:],hi)-np.maximum(x[:-1],lo))
            if overlap.sum()>0:T=float(np.average(profile_target,weights=overlap))
            else:T=float(profile_target[min(len(profile_target)-1,max(0,np.searchsorted(x,lo)-1))])
            assert T>0
            Ts.append(T);graph.add_edge(int(r.id_us_node),int(r.id_ds_node),key=int(r.id_link),weight=T*r.actual_length_m)
        assert nx.is_directed_acyclic_graph(graph)
        src=[v for v in graph if graph.in_degree(v)==0];snk=[v for v in graph if graph.out_degree(v)==0];assert len(src)==len(snk)==1
        src,snk=src[0],snk[0];drop=original.loc[src]-original.loc[snk];assert drop>0,(ex,kind,did,drop)
        dist={v:-np.inf for v in graph};dist[src]=0
        for u in nx.topological_sort(graph):
            for _,v,z in graph.out_edges(u,data=True):dist[v]=max(dist[v],dist[u]+z['weight'])
        rho=drop/dist[snk];vs=list(graph);lookup={v:i for i,v in enumerate(vs)}
        C=sparse.lil_matrix((len(d),len(vs)))
        for i,r in enumerate(d.itertuples()):
            factor=drop/(Ts[i]*r.actual_length_m);C[i,lookup[int(r.id_us_node)]]=factor;C[i,lookup[int(r.id_ds_node)]]=-factor
        C=C.tocsr()
        # Nodes on any critical longest path have uniquely fixed heads at rho_max.
        to_sink={v:-np.inf for v in graph};to_sink[snk]=0.
        for u in reversed(list(nx.topological_sort(graph))):
            for _,v,z in graph.out_edges(u,data=True):to_sink[u]=max(to_sink[u],z['weight']+to_sink[v])
        initial=np.array([1-rho*dist[v]/drop for v in vs])
        fixed_ids=[lookup[v] for v in vs if abs(dist[v]+to_sink[v]-dist[snk])<=1e-10*max(1.,dist[snk])]
        free_ids=[i for i in range(len(vs)) if i not in fixed_ids]
        hh=initial.copy()
        if free_ids:
            Cf=C[:,free_ids];constant=C[:,fixed_ids]@initial[fixed_ids]
            # Constraint rows in normalized head units, avoiding huge ratios on short edges.
            edge_scale=np.array(Ts)*d.actual_length_m.to_numpy()/drop
            Af=sparse.diags(edge_scale)@Cf
            lower=(rho-constant)*edge_scale
            Pf=2*Cf.T@Cf;qf=2*(Cf.T@(constant-1))
            free,qa=qp(Pf,qf,Af,lower,np.full(len(d),np.inf),initial[free_ids],tol=1e-7,maxiter=120000)
            hh[free_ids]=free
        else:qa=dict(iterations=0,primal_residual=0.,dual_residual=0.,primal_tolerance=0.,dual_tolerance=0.,seconds=0.)
        qa['n_critical_fixed_nodes']=len(fixed_ids)
        hh[lookup[src]]=1.;hh[lookup[snk]]=0.
        for v in vs:
            if v not in [src,snk]:nodeheads.loc[v]=original.loc[snk]+drop*hh[lookup[v]]
        domain_audits.append(dict(kind=kind,domain=int(did),n_links=len(d),source=src,sink=snk,boundary_drop_m=drop,max_feasible_target_fraction=rho,**qa))
    result=links[['id_link','id_us_node','id_ds_node','actual_length_m','ownership_kind','ownership_domain_id','slope_used']].copy();result['previous_slope_used']=result.slope_used;result['candidate_head_us_m']=result.id_us_node.map(nodeheads);result['candidate_head_ds_m']=result.id_ds_node.map(nodeheads);result['slope_used']=(result.candidate_head_us_m-result.candidate_head_ds_m)/result.actual_length_m
    result['candidate_slope_m_per_m']=result.slope_used;result['slope_source_method']='candidate_data_informed_profile_maxmin_unit_v1'
    result['profile_head_us_m']=result.id_us_node.map(original);result['profile_head_ds_m']=result.id_ds_node.map(original)
    result['profile_direct_slope_m_per_m']=(result.profile_head_us_m-result.profile_head_ds_m)/result.actual_length_m
    assert np.isfinite(result.slope_used).all() and (result.slope_used>0).all(),(ex,'nonpositive slope')
    # Global potential consistency implies identical path head losses; verify per-domain boundaries remain intact.
    for v in domain_audits:
        assert abs(nodeheads.loc[v['source']]-original.loc[v['source']])<1e-7 and abs(nodeheads.loc[v['sink']]-original.loc[v['sink']])<1e-7
    nodes=nodes[['id_node','reference_section_dist_out_m','reference_section_dist_source']].copy();nodes['profile_head_m']=nodes.id_node.map(original);nodes['candidate_head_m']=nodes.id_node.map(nodeheads)
    result.to_parquet(out/'candidate_link_hydraulic_context.parquet',index=False);nodes.to_parquet(out/'candidate_node_hydraulic_context.parquet',index=False);pd.DataFrame(domain_audits).to_csv(out/'unit_solver_audit.csv',index=False)
    ends=[nodes.reference_section_dist_out_m.min(),nodes.reference_section_dist_out_m.max()];newdrop=np.diff(np.interp(ends,x,heights))[0];olddrop=np.diff(np.interp(ends,x,old.old_fitted_wse_m if 'old_fitted_wse_m' in old else old.fitted_wse_m))[0]
    summary=dict(example=ex,status='complete',n_links=len(result),n_units=len(domain_audits),n_section_targets=int(sections.used.sum()),profile_supported_length_pct=100*dx[supported].sum()/dx.sum(),profile_solver=solver,old_low_links=int((result.previous_slope_used<1e-5).sum()),new_low_links=int((result.slope_used<1e-5).sum()),old_min_slope=float(result.previous_slope_used.min()),new_min_slope=float(result.slope_used.min()),profile_min_slope=float(np.min(np.diff(heights)/dx)),head_loss_change_pct=100*(newdrop/olddrop-1),old_head_loss_m=olddrop,new_head_loss_m=newdrop,old_observation_rms_m=float(np.sqrt(np.mean(obs.old_residual_m**2))),new_observation_rms_m=float(np.sqrt(np.mean(obs.residual_m**2))),seconds=time.monotonic()-start)
    (out/'summary.json').write_text(json.dumps(summary,indent=2));print(ex,json.dumps(summary),flush=True)
    return summary

def main():
    # Analytic solver checks: bound-active scalar and equality-constrained pair.
    v,_=qp(sparse.eye(1)*2,np.array([-4.]),sparse.eye(1),np.array([0.]),np.array([1.]),np.zeros(1));assert abs(v[0]-1)<1e-5
    v,_=qp(sparse.eye(2)*2,np.zeros(2),sparse.csr_matrix([[1.,1.]]),np.array([2.]),np.array([2.]),np.zeros(2));assert np.max(abs(v-1))<1e-5
    summaries=[]
    for ex in sources:
        path=OUT/ex/'summary.json'
        if path.exists():summaries.append(json.loads(path.read_text()));continue
        try:summaries.append(run(ex))
        except Exception as err:
            record=dict(example=ex,status='failed',error=repr(err));summaries.append(record);print(ex,'FAILED',repr(err),flush=True)
    pd.DataFrame([{k:v for k,v in r.items() if k!='profile_solver'} for r in summaries]).to_csv(OUT/'summary.csv',index=False)
    (OUT/'method.json').write_text(json.dumps(dict(method='data-informed candidate v1',sources=sources,reference='SWORD reach P75 targets with bootstrap-lower-P10 interval bounds; curvature ratio gate 0.5–2; equal cycle weights; reviewed lambda retained',fallback='Existing per-case profile minimum in unsupported intervals; no claim of wholly data-derived slopes',units='Maximal common target fraction, then least-squares normalized slope objective with fixed profile boundary heads',bridge='Old manual bridges are not imposed after the new fit; section fallback status remains visible',script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),indent=2))
if __name__=='__main__':main()
