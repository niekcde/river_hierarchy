"""Nested grouped forward search with explicit four-principle coverage.

Fixed quadratic ridge model; pooled population objective; no routing changes.
Cached sufficient statistics make all candidate subset comparisons exact for
this model without repeatedly fitting 45,927-row design matrices.
"""
import os
for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:
    os.environ[k]='2'
os.environ.setdefault('MPLCONFIGDIR','/tmp/river-similarity-mpl')
from pathlib import Path
import itertools, json, time, hashlib
import numpy as np
import pandas as pd
from scipy.linalg import cho_factor, cho_solve
from sklearn.linear_model import Ridge
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE=Path(__file__).resolve().parent
OUT=HERE/'results';OUT.mkdir(exist_ok=True)
PRE=HERE.parent/'individual_predictor_screen_20261102/results'
SOURCE=Path('/Volumes/SyntheticResults20261027/analysis/broad_descriptors_20261030/geometry_with_manning/features.parquet')
methods=['variable_manning','fixed_manning','fixed_based']
responses=['centroid_shift_minutes','peak_shift_minutes']
targets=list(itertools.product(methods,responses))
ALPHA=1e-4  # penalty = ALPHA*n on unit-variance expanded terms
TOL=.005  # smallest subset within this absolute inner R² of best path score
MAX=10
start=time.monotonic()
screen=pd.read_csv(PRE/'screened_candidates.csv')
scores=pd.read_csv(PRE/'candidate_summary.csv')
eligible=set(scores.groupby('variable').mean_type_r2.max().loc[lambda s:s>=.05].index)
assert len(eligible)==68
topology=set(screen.loc[screen.concept.eq('Topological architecture'),'screen_variable'])
assert len(topology)==6 and not eligible&topology
excluded={'path_parent_equivalent_width','path_delta_equivalent_width','path_parent_elongation','path_delta_elongation'}
excluded |= {v for v in eligible|topology if v.endswith('width_p90')}
assert excluded <= (eligible|topology)
variables=sorted((eligible|topology)-excluded)
assert len(variables)==67 and 'B_ratio_width_mean' in variables
assert not any(v.endswith('width_p90') for v in variables)
meta=screen.set_index('screen_variable').loc[variables].copy()
def category(v):
    c=meta.loc[v,'concept']
    if c=='Topological architecture':return 'topology'
    if c=='Reach slope distribution':return 'slope'
    if c=='Reach width distribution' or v.endswith('equivalent_width'):return 'width'
    if c in ['Allocated route length distribution','Equally weighted route length distribution',
             'Branch distance contrasts','Affected domain extent','Equivalent path length',
             'Total channel length','Route elongation relative to shortest route']:return 'length'
    return 'combined'
categories={v:category(v) for v in variables}
meta['coverage_category']=[categories[v] for v in variables]
meta['restored_topology_reserve']=[v in topology for v in variables]
meta.to_csv(OUT/'candidate_pool.csv')
f=pd.read_parquet(SOURCE)
frames={m:f[f.method.eq(m)].sort_values(['case','child_state']).reset_index(drop=True) for m in methods}
g=frames[methods[0]]
for m,h in frames.items():
    pd.testing.assert_frame_equal(g[['case','child_state']],h[['case','child_state']])
    np.testing.assert_allclose(g[variables],h[variables],equal_nan=True)
Y=np.column_stack([frames[m][r] for m,r in targets])
X=g[variables].to_numpy(float).copy()
if 'domain_size' in variables:X[:,variables.index('domain_size')]*=1000
assert np.isfinite(X).all() and np.isfinite(Y).all()
for j in range(len(variables)):
    tol=1e-10*max(np.max(abs(X[:,j])),1.)
    X[:,j]=np.round(X[:,j]/tol)*tol
# Pure numeric scaling cancels in training-moment standardisation of EACH
# expanded column. No outcomes or held-out-derived centre enter a fitted model.
numeric_scale=np.maximum(np.max(abs(X),axis=0),1.)
X/=numeric_scale
case_names=np.array(sorted(g.case.unique()))
rng=np.random.default_rng(20261102)
perm=rng.permutation(case_names)
cell_map={c:i%9 for i,c in enumerate(perm)}
cell=np.array([cell_map[c] for c in g.case]);outer=cell//3;inner=cell%3
pd.DataFrame({'case':case_names,'outer_fold':[cell_map[c]//3 for c in case_names],
              'inner_label':[cell_map[c]%3 for c in case_names]}).to_csv(OUT/'case_folds.csv',index=False)
assert pd.DataFrame({'case':g.case,'cell':cell}).groupby('case').cell.nunique().eq(1).all()
# Basis: intercept, x_j, x_j*x_k including squares. Coefficients are penalised
# after training-only centring/scaling of expanded terms; intercept is unpenalised.
terms=[(j,k) for j in range(len(variables)) for k in range(j,len(variables))]
term_index={t:1+len(variables)+i for i,t in enumerate(terms)}
D=1+len(variables)+len(terms)
def expand(x):
    b=np.empty((len(x),D));b[:,0]=1;b[:,1:1+len(variables)]=x
    for i,(j,k) in enumerate(terms):b[:,1+len(variables)+i]=x[:,j]*x[:,k]
    return b
stats=[]
print(f'Pool {len(variables)}; basis {D}; building nine sufficient-statistic blocks',flush=True)
for c in range(9):
    mask=cell==c;b=expand(X[mask]);y=Y[mask]
    stats.append((b.T@b,b.T@y,y.T@y))
    del b
    print(f'Block {c+1}/9; elapsed {time.monotonic()-start:.1f}s',flush=True)
def sumstat(cells):
    return tuple(sum((stats[c][i] for c in cells),start=np.zeros_like(stats[0][i])) for i in range(3))
def indices(subset):
    s=sorted(subset)
    return np.array([0]+[1+j for j in s]+[term_index[(j,k)] for j in s for k in s if j<=k])
def fit(st,ix):
    gram,xy,_=st;gg=gram[np.ix_(ix,ix)];h=xy[ix];n=gg[0,0]
    means=gg[0,1:]/n;ym=h[0]/n
    centered=gg[1:,1:]-n*np.outer(means,means)
    sd=np.sqrt(np.maximum(np.diag(centered)/n,0))
    good=sd>1e-10
    beta=np.zeros((len(ix),6))
    if good.any():
        s=sd[good];a=centered[np.ix_(good,good)]/np.outer(s,s)
        a.flat[::len(a)+1]+=ALPHA*n
        rhs=(h[1:][good]-n*means[good,None]*ym)/s[:,None]
        z=cho_solve(cho_factor(a,check_finite=False),rhs,check_finite=False)
        beta[1:][good]=z/s[:,None]
    beta[0]=ym-means@beta[1:]
    return beta
def errors(st,ix,beta):
    gram,xy,yy=st;gg=gram[np.ix_(ix,ix)]
    return np.maximum(np.diag(yy)-2*np.sum(beta*xy[ix],axis=0)+np.sum(beta*(gg@beta),axis=0),0)
def denominator(st):
    gram,xy,yy=st
    return np.maximum(np.diag(yy)-xy[0]**2/gram[0,0],1e-20)
all_steps=[];selection_rows=[];predictions=[];outer_scores=[];candidate_evals=[];checks=[];models=[]
names={i:v for i,v in enumerate(variables)}
cats={i:categories[v] for i,v in names.items()}
required={'length','width','slope','topology'}
for o in range(3):
    train_cells=[c for c in range(9) if c//3!=o]
    train=sumstat(train_cells)
    outer_test=sumstat([c for c in range(9) if c//3==o])
    cv=[]
    for k in range(3):
        cv.append((sumstat([c for c in train_cells if c%3!=k]),
                   sumstat([c for c in train_cells if c%3==k])))
    den=denominator(train)
    cache={}
    def evaluate(subset):
        key=tuple(sorted(subset))
        if key not in cache:
            ix=indices(key);sse=np.zeros(6)
            for tr,va in cv:sse+=errors(va,ix,fit(tr,ix))
            cache[key]=1-sse/den
        return cache[key]
    # Verify sufficient-statistic evaluation against direct row prediction.
    test_subset=[variables.index('B_delta_route_length_mean'),variables.index('B_parent_width_mean')]
    ix=indices(test_subset);beta=fit(cv[0][0],ix)
    mask=(outer!=o)&(inner==0)
    small=np.column_stack([np.ones(mask.sum())]+[X[mask,j] for j in sorted(test_subset)]+
                         [X[mask,j]*X[mask,k] for j in sorted(test_subset) for k in sorted(test_subset) if j<=k])
    actual=np.sum((Y[mask]-small@beta)**2,axis=0)
    np.testing.assert_allclose(errors(cv[0][1],ix,beta),actual,rtol=1e-7,atol=1e-5)
    trmask=(outer!=o)&(inner!=0)
    direct=np.column_stack([X[trmask,j] for j in sorted(test_subset)]+
                          [X[trmask,j]*X[trmask,k] for j in sorted(test_subset) for k in sorted(test_subset) if j<=k])
    means=direct.mean(axis=0);sd=direct.std(axis=0)
    reference=Ridge(alpha=ALPHA*trmask.sum()).fit((direct-means)/sd,Y[trmask])
    np.testing.assert_allclose(small@beta,reference.predict((small[:,1:]-means)/sd),rtol=1e-6,atol=1e-5)
    checks.append({'outer':o,'gram_sse_matches_direct':True,'coefficients_match_sklearn':True})
    for allow_g in [True,False]:
        pool=[i for i,v in names.items() if allow_g or v!='delta_G_manning']
        for constrained in [False,True]:
            variant=('coverage' if constrained else 'unconstrained')+('_with_G' if allow_g else '_without_G')
            for t,(method,response) in enumerate(targets):
                chosen=[];path=[]
                for step in range(1,MAX+1):
                    missing=required-{cats[i] for i in chosen}
                    options=[i for i in pool if i not in chosen and
                             (not constrained or not missing or cats[i] in missing)]
                    trials=[(evaluate(chosen+[i])[t],i) for i in options]
                    best,add=max(trials,key=lambda z:(z[0],-z[1]))
                    candidate_evals.extend(dict(outer=o,variant=variant,method=method,response=response,
                                               step=step,candidate=names[i],inner_r2=float(s)) for s,i in trials)
                    chosen.append(add)
                    # After four-group seeding, improve its representatives by
                    # two deterministic coordinate-swap passes. Preserve coverage.
                    if constrained and step==4:
                        for _ in range(2):
                            improved=False
                            for pos in range(4):
                                base=chosen[:pos]+chosen[pos+1:]
                                opts=[i for i in pool if i not in base and cats[i]==cats[chosen[pos]]]
                                sc,new=max(((evaluate(base+[i])[t],i) for i in opts),key=lambda z:(z[0],-z[1]))
                                if sc>evaluate(chosen)[t]+1e-10:
                                    chosen[pos]=new;improved=True
                            if not improved:break
                    score=float(evaluate(chosen)[t])
                    if not constrained or step>=4:path.append((score,chosen.copy()))
                    all_steps.append(dict(outer=o,variant=variant,method=method,response=response,
                                          n_predictors=step,inner_r2=score,variables='; '.join(names[i] for i in chosen)))
                best=max(s for s,ss in path)
                score,selected=next((s,ss) for s,ss in path if s>=best-TOL)
                if constrained:assert required <= {cats[i] for i in selected}
                ix=indices(selected);beta=fit(train,ix)
                sse=errors(outer_test,ix,beta)
                r2=1-sse[t]/denominator(outer_test)[t]
                mask=outer==o
                xs=X[mask]
                s=sorted(selected)
                b=np.column_stack([np.ones(mask.sum())]+[xs[:,i] for i in s]+
                                  [xs[:,i]*xs[:,j] for i in s for j in s if i<=j])
                pred=b@beta[:,t]
                models.append(dict(outer=o,variant=variant,method=method,response=response,
                    variables=[names[i] for i in s],numeric_scales=[float(numeric_scale[i]) for i in s],
                    basis='intercept; listed variables/scales; pair products i<=j in listed order',
                    coefficients=beta[:,t].tolist()))
                np.testing.assert_allclose(np.sum((Y[mask,t]-pred)**2),sse[t],rtol=1e-6,atol=1e-4)
                outer_scores.append(dict(outer=o,variant=variant,method=method,response=response,
                                         n_predictors=len(selected),inner_r2=score,outer_r2=r2,
                                         outer_mae_minutes=float(np.mean(abs(pred-Y[mask,t])))))
                for i in selected:selection_rows.append(dict(outer=o,variant=variant,method=method,response=response,
                                                             variable=names[i],category=cats[i]))
                predictions.append(pd.DataFrame(dict(case=g.loc[mask,'case'].to_numpy(),
                    child_state=g.loc[mask,'child_state'].to_numpy(),outer=o,variant=variant,
                    method=method,response=response,observed=Y[mask,t],predicted=pred)))
            print(f'Outer {o+1}/3 {variant}; {len(cache)} distinct subsets; {time.monotonic()-start:.1f}s',flush=True)
    # Explicit standalone GM baseline, selected without response search.
    ig=indices([variables.index('delta_G_manning')]);beta=fit(train,ig)
    mask=outer==o;b=np.column_stack([np.ones(mask.sum()),X[mask,variables.index('delta_G_manning')],X[mask,variables.index('delta_G_manning')]**2])
    for t,(method,response) in enumerate(targets):
        predictions.append(pd.DataFrame(dict(case=g.loc[mask,'case'].to_numpy(),child_state=g.loc[mask,'child_state'].to_numpy(),outer=o,
            variant='G_alone',method=method,response=response,observed=Y[mask,t],predicted=b@beta[:,t])))
    pd.DataFrame(all_steps).to_csv(OUT/'search_paths.csv',index=False)
    pd.DataFrame(outer_scores).to_csv(OUT/'outer_fold_scores.csv',index=False)
    pd.DataFrame(selection_rows).to_csv(OUT/'selected_variables_by_fold.csv',index=False)
    del cv,train,outer_test,cache

pred=pd.concat(predictions,ignore_index=True)
pred.to_parquet(OUT/'outer_predictions.parquet',index=False)
pd.DataFrame(candidate_evals).to_csv(OUT/'candidate_trials.csv',index=False)
pd.DataFrame(checks).to_csv(OUT/'numerical_checks.csv',index=False)
(OUT/'selected_fold_models.json').write_text(json.dumps(models,indent=2))
summary=[]
for (variant,method,response),z in pred.groupby(['variant','method','response']):
    for scope,h in [('pooled',z)]+list(z.groupby('child_state')):
        y=h.observed.to_numpy();p=h.predicted.to_numpy();threshold=4/60 if response.startswith('centroid') else .5
        resolved=bool(np.ptp(y)>threshold)
        r2=1-np.sum((y-p)**2)/np.sum((y-y.mean())**2) if resolved else np.nan
        summary.append(dict(variant=variant,method=method,response=response,scope=scope,n=len(h),
                            r2=r2,mae_minutes=float(np.mean(abs(y-p))),response_resolved=resolved))
summary=pd.DataFrame(summary);summary.to_csv(OUT/'outer_summary.csv',index=False)
freq=pd.DataFrame(selection_rows).groupby(['variant','method','response','variable','category']).size().reset_index(name='selected_folds')
freq.to_csv(OUT/'selection_frequency.csv',index=False)
pd.DataFrame(outer_scores).groupby(['variant','method','response']).n_predictors.agg(['min','median','max']).to_csv(OUT/'model_sizes.csv')
sources=[SOURCE,PRE/'candidate_summary.csv',PRE/'screened_candidates.csv',Path(__file__)]
(OUT/'provenance.json').write_text(json.dumps(dict(reference='SYN-FORWARD-NO-P90-2026-11-03',
    sources={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
    pool=len(variables),original_pool=74,excluded=sorted(excluded),screen_pass=68,topology_reserves=6,outer_folds=3,inner_folds=3,
    model='quadratic ridge; main effects, squares, pair products; expanded columns train-standardised',
    alpha_per_row=ALPHA,smallest_within_best_r2=TOL,max_predictors=MAX,
    elapsed_seconds=time.monotonic()-start,routing_runs=0),indent=2))
print(summary[summary.scope.eq('pooled')].to_string(index=False))
