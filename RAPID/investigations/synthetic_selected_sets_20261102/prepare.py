"""Geometry-only preparation for four frozen synthetic-selected predictor sets."""
from pathlib import Path
import sys, json, hashlib, time, math, io
import numpy as np
import pandas as pd
import networkx as nx
from scipy.special import roots_laguerre
from scipy.integrate import quad
ROOT=Path(__file__).resolve().parents[3]
sys.path[:0]=[str(ROOT),str(ROOT/'RAPID/src')]
from network_variants.equivalent_transition_domain import extract_complete_directed_domain
from rapid_tools.current_domains import validate_domain
HERE=Path(__file__).resolve().parent
OUT=ROOT/'RAPID/outputs/synthetic_selected_sets_20261102';OUT.mkdir(exist_ok=True)
BASE=ROOT/'RAPID/outputs/uncapped_event_window_20261020'
SOURCE=Path('/Volumes/PhD/river_hierarchy/three_run_comparison_20261028/03_manning_midpoint')
GM=Path('/Volumes/SyntheticResults20261027/analysis/real_geometry_manning_20261030/geometry.parquet')
HASHES={};WIDTH_AUDIT=[]
def read(p):
    p=Path(p);raw=p.read_bytes();HASHES[str(p)]=hashlib.sha256(raw).hexdigest()
    if p.suffix=='.json':return json.loads(raw)
    return pd.read_parquet(io.BytesIO(raw)) if p.suffix=='.parquet' else pd.read_csv(io.BytesIO(raw),dtype={'example':str})
def active(ex,state):
    folder=SOURCE/'states'/ex/state;p=folder/'parameters.parquet'
    receipt=read(folder/'completion.json');f=read(p)
    assert HASHES[str(p)]==receipt['files'][str(p)]
    f['id_us_node']=f.u;f['id_ds_node']=f.v
    return f
def graph(f):
    f=f.reset_index(drop=True);g=nx.MultiDiGraph()
    for i,r in enumerate(f.itertuples()):g.add_edge(int(r.u),int(r.v),key=i)
    order=list(nx.topological_sort(g));roots=[u for u in g if g.in_degree(u)==0];ends=[u for u in g if g.out_degree(u)==0]
    assert len(roots)==len(ends)==1 and nx.is_weakly_connected(g)
    index={u:i for i,u in enumerate(order)}
    outgoing=[[(k,index[v]) for _,v,k in g.out_edges(u,keys=True)] for u in order]
    L=f.link_length_m.to_numpy(float);B=f.width.to_numpy(float);S=f.slope.to_numpy(float)
    assert (L>0).all() and (B>0).all() and (S>0).all()
    counts=[0]*len(order);counts[-1]=1
    for u in range(len(order)-2,-1,-1):counts[u]=sum(counts[v] for e,v in outgoing[u])
    return f,g,order,outgoing,L,B,S,counts
QUAD={n:roots_laguerre(n) for n in [32,64,128,256]}
def equivalent_width(parts,context):
    _,g,order,out,L,B,S,counts=parts
    # W_eq=sum_routes L_r/(sum_e L_e/B_e). Uniform-route expectation
    # times exact path count, evaluated through the Laplace identity 1/R.
    resistance=L/B;rmin=np.zeros(len(order))
    for u in range(len(order)-2,-1,-1):rmin[u]=min(resistance[e]+rmin[v] for e,v in out[u])
    rscale=rmin[0];lscale=L.sum()
    transitions=[]
    for u,ee in enumerate(out):
        transitions.append([(v,counts[v]/counts[u],max(0.,(resistance[e]+rmin[v]-rmin[u])/rscale),L[e]/lscale) for e,v in ee])
    def integrand(t):
        t=np.atleast_1d(t);F=np.zeros((len(order),len(t)));M=np.zeros_like(F);F[-1]=1
        for u in range(len(order)-2,-1,-1):
            for v,q,c,l in transitions[u]:
                weight=q*np.exp(-t*c)
                F[u]+=weight*F[v];M[u]+=weight*(M[v]+l*F[v])
        return M[0]
    previous=None;error=np.nan;used=None
    for n,(nodes,weights) in QUAD.items():
        value=float(weights@integrand(nodes))*lscale/rscale
        if previous is not None:
            error=abs(value-previous)/max(abs(value),1e-300)
            if error<1e-9:used=n;break
        previous=value
    if used is None:
        val,err=quad(lambda t:float(integrand(np.array([t]))[0])*math.exp(-t),0,np.inf,epsabs=1e-12,epsrel=1e-10,limit=250)
        value=val*lscale/rscale;error=err/max(abs(val),1e-300);used='adaptive'
        if error>1e-7:raise ValueError(('width quadrature did not converge',context,error))
    width=float(counts[0])*value
    assert np.isfinite(width) and width>0
    exact_error=np.nan
    if counts[0]<=1000:
        exact=sum(sum(L[e] for u,v,e in path)/sum(resistance[e] for u,v,e in path) for path in nx.all_simple_edge_paths(g,order[0],order[-1]))
        exact_error=abs(width-exact)/exact
        np.testing.assert_allclose(width,exact,rtol=1e-8,atol=1e-8)
    WIDTH_AUDIT.append(dict(context=context,n_paths=str(counts[0]),equivalent_width_m=width,
                            quadrature_nodes=used,relative_convergence=error,enumeration_relative_error=exact_error))
    return width
def describe(f,context,need_width=False):
    p=graph(f);f,g,order,out,L,B,S,counts=p
    n=len(order);visit=np.zeros(n);visit[0]=1;fractions=np.zeros(len(f));probs=[]
    for u,ee in enumerate(out):
        q=np.array([B[e] for e,v in ee]);q=q/q.sum() if len(q) else q;probs.append(q)
        for a,(e,v) in zip(q,ee):fractions[e]=visit[u]*a;visit[v]+=fractions[e]
    mu=np.zeros(n);var=np.zeros(n)
    for u in range(n-2,-1,-1):
        branches=np.array([L[e]+mu[v] for e,v in out[u]])
        mu[u]=probs[u]@branches
        var[u]=sum(a*(var[v]+(b-mu[u])**2) for a,b,(e,v) in zip(probs[u],branches,out[u]))
    bif=sum(len(ee)>1 for ee in out)
    ib=sum(g.out_degree(u)>1 for u in order if u!=order[0]);ic=sum(g.in_degree(u)>1 for u in order if u!=order[-1])
    ix=np.argsort(B,kind='stable');p90=B[ix[min(np.searchsorted(np.cumsum(L[ix]),.9*L.sum()),len(L)-1)]]
    result=dict(mean=mu[0],cv=math.sqrt(max(var[0],0))/mu[0],area=float(L@B),
                mean_slope=float(L@S/L.sum()),slow=float(np.sum(fractions*L/np.sqrt(S))/np.sum(fractions*L)),
                bifurcations=bif,complexity=math.log1p(counts[0])+math.log1p(ib)+math.log1p(ic),
                p90=float(p90),G=float(np.sum(L*B**.4*fractions**.6*S**(-.3))),n_paths=str(counts[0]),links=len(f))
    if need_width:result['eq_width']=equivalent_width(p,context)
    return result
def main():
    start=time.monotonic();records=read(BASE/'validation_group_inventory_20261023/records.parquet')
    response=read(SOURCE/'responses.csv');receipt=read(SOURCE/'completion.json')
    assert HASHES[str(SOURCE/'responses.csv')]==receipt['files'][str(SOURCE/'responses.csv')]
    response=response[response.window.eq('0.5%')&response.state.ne('fully_collapsed')].set_index(['example','state'])
    gm=read(GM).set_index(['example','state'])
    old=read(BASE/'fixed_manning_rf_20261029/input.parquet').set_index(['example','state'])
    rows=[];cache={}
    for ex,z in records.groupby('example'):
        parent=active(ex,'S000_base')
        for r in z.itertuples():
            child=active(ex,r.state)
            if r.cohort=='segmented':
                pp,cc=parent,child
                scope='whole closed physical network including unchanged connectors'
            else:
                pp=parent[parent.id_link.isin(r.descriptor_links)].copy()
                pp,(s,t)=validate_domain(parent,pp)
                cc=extract_complete_directed_domain(child,int(s),int(t));cc,_=validate_domain(child,cc)
                scope='matched closed parent B domain and child domain at the same anchors'
            key=(ex,tuple(sorted(pp.id_link)))
            if key not in cache:cache[key]=describe(pp,f'{ex}/{r.state}/parent',True)
            a=cache[key];b=describe(cc,f'{ex}/{r.state}/child')
            rr=response.loc[(ex,r.state)]
            row=dict(example=ex,state=r.state,cohort=r.cohort,edit_type=r.edit_type,scope=scope,
                centroid_shift_minutes=float(rr.centroid_shift_hours)*60,peak_shift_minutes=float(rr.peak_shift_hours)*60,
                B_delta_route_length_mean=b['mean']-a['mean'],parent_slope_slow=a['slow'],
                path_parent_equivalent_width=a['eq_width'],path_delta_topologic_complexity_score=b['complexity']-a['complexity'],
                delta_G_manning=float(gm.loc[(ex,r.state),'delta_G_manning']),B_delta_area_proxy=b['area']-a['area'],
                B_child_width_p90=b['p90'],B_parent_bifurcations=a['bifurcations'],path_parent_slope_mean=a['mean_slope'],
                B_delta_route_length_cv=b['cv']-a['cv'],B_ratio_route_length_mean=b['mean']/a['mean'],
                path_delta_slope_mean=b['mean_slope']-a['mean_slope'],parent_paths=a['n_paths'],child_paths=b['n_paths'],
                parent_links=a['links'],child_links=b['links'])
            np.testing.assert_allclose(row['centroid_shift_minutes'],60*old.loc[(ex,r.state),'centroid_shift_hours'],rtol=0,atol=1e-10)
            np.testing.assert_allclose(row['peak_shift_minutes'],60*old.loc[(ex,r.state),'peak_shift_hours'],rtol=0,atol=1e-10)
            if r.cohort=='original':
                np.testing.assert_allclose(row['B_delta_route_length_mean'],1000*old.loc[(ex,r.state),'length_delta_km'],rtol=1e-8,atol=1e-6)
                np.testing.assert_allclose(row['parent_slope_slow'],old.loc[(ex,r.state),'parent_slope_slow'],rtol=1e-9)
            rows.append(row)
        pd.DataFrame(rows).to_parquet(OUT/'input.partial.parquet',index=False)
        pd.DataFrame(WIDTH_AUDIT).to_csv(OUT/'equivalent_width_audit.csv',index=False)
        print(ex,len(z),'complete;',round(time.monotonic()-start,1),'s',flush=True)
    data=pd.DataFrame(rows)
    assert len(data)==2185 and data.example.nunique()==15 and not data.duplicated(['example','state']).any()
    assert np.isfinite(data.select_dtypes('number')).all().all()
    data.to_parquet(OUT/'input.parquet',index=False)
    for p in [Path(__file__),ROOT/'synthetic_experiments/investigations/forward_selection_20261102/results/selected_variables_by_fold.csv']:
        HASHES[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
    (OUT/'preparation.json').write_text(json.dumps(dict(reference='REAL-SYNTHETIC-SELECTED-SETS-2026-11-02',sources=HASHES,
        rows=len(data),networks=15,seconds=time.monotonic()-start,
        responses='Current completed fixed-Manning midpoint batch; existing 0.5% finite-event windows; collapse minus own parent.',
        geometry_scope='Original cases use matched B domains. Fifteen segmented full-network collapses use whole closed physical domains INCLUDING connectors, matching synthetic domain-style formulas; this differs from the older edited-only descriptor convention for those fifteen cases.',
        width='Sum of harmonic path widths evaluated using uniform-route Laplace recursion and converged quadrature; exact enumeration checked where <=1000 routes. No cap, sampling or alternative width definition.',
        GM='Existing verified whole-network geometry_Manning table, no calibrated depth or simulated flow inputs.',
        exclusions='None from established 2185-comparison population; that inventory already resolves aliases and excludes noncanonical duplicate controls.',
        output_sha256=hashlib.sha256((OUT/'input.parquet').read_bytes()).hexdigest()),indent=2))
    print('DONE',len(data),'max width',data.path_parent_equivalent_width.max(),flush=True)
if __name__=='__main__':main()
