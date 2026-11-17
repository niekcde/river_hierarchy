"""Fixed univariate spline benchmark, grouped by complete factorial case."""
import os
for key in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:
    os.environ[key] = '1'
os.environ.setdefault('MPLCONFIGDIR', '/tmp/river-similarity-mpl')
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr
from sklearn.preprocessing import SplineTransformer
from sklearn.linear_model import Ridge
import sklearn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
OUT=HERE/'results'; OUT.mkdir(exist_ok=True)
SOURCE=Path('/Volumes/SyntheticResults20261027/analysis/broad_descriptors_20261030/geometry_with_manning/features.parquet')
WORK=HERE.parent/'predictor_similarity_20261102/results/working_100_classes.csv'
work=pd.read_csv(WORK).fillna('')
temporary={'A095':'path_parent_head_drop','A024':'B_delta_bifurcations',
           'A004':'B_child_bifurcations','A045':'B_parent_bifurcations',
           'A062':'B_parent_split_exposure_mean'}
work['screen_variable']=[r.representative or temporary[r.affine_class] for r in work.itertuples()]
work['temporary_affine_label']=work.representative.eq('')
assert len(work)==100 and work.screen_variable.nunique()==100
concept_path=ROOT/'synthetic_experiments/notes/geometry_concept_inventory_20261102.json'
concept={v:g['name'] for g in json.loads(concept_path.read_text())['concepts'] for v in g['candidates']}
work['concept']=work.screen_variable.map(concept)
work['input_unit_note']=np.where(work.screen_variable.eq('domain_size'),'converted km to m in screen','source units; lengths m, areas m²')
work.to_csv(OUT/'screened_candidates.csv',index=False)
f=pd.read_parquet(SOURCE)
responses=['centroid_shift_minutes','peak_shift_minutes']
methods=['variable_manning','fixed_manning','fixed_based']
keys=['case','child_state']
frames={m:f[f.method.eq(m)].sort_values(keys).reset_index(drop=True) for m in methods}
g=frames[methods[0]]
cols=work.screen_variable.tolist()
assert len(g)==45927 and not g.duplicated(keys).any()
for m,h in frames.items():
    pd.testing.assert_frame_equal(g[keys],h[keys])
    np.testing.assert_allclose(g[cols],h[cols],rtol=1e-9,atol=1e-8,equal_nan=True)
Y=np.column_stack([frames[m][r].to_numpy() for m in methods for r in responses])
assert np.isfinite(Y).all()
X=g[cols].to_numpy(float).copy()
assert np.isfinite(X).all()
if 'domain_size' in cols:X[:,cols.index('domain_size')]*=1000
# Same full-column roundoff treatment used in the prior audit; no outcome cleaning.
for j in range(X.shape[1]):
    tol=1e-10*max(np.max(abs(X[:,j])),1.)
    X[:,j]=np.round(X[:,j]/tol)*tol
cases=np.array(sorted(g.case.unique()));assert len(cases)==6561
rng=np.random.default_rng(20261102)
permuted=rng.permutation(cases)
case_fold={case:i%5 for i,case in enumerate(permuted)}
fold=np.array([case_fold[c] for c in g.case])
pd.DataFrame({'case':cases,'fold':[case_fold[c] for c in cases]}).to_csv(OUT/'case_folds.csv',index=False)
scopes=['pooled']+sorted(g.child_state.unique())
metrics=[];fold_metrics=[];diagnostics=[]
def score(y,p):
    ss=float(np.sum((y-y.mean())**2))
    return (1-float(np.sum((y-p)**2))/ss if ss>0 else np.nan,
            float(np.mean(abs(y-p))),float(np.sqrt(np.mean((y-p)**2))))
for scope in scopes:
    ii=np.arange(len(g)) if scope=='pooled' else np.flatnonzero(g.child_state.eq(scope))
    yy=Y[ii];ff=fold[ii]
    for j,variable in enumerate(cols):
        xx=X[ii,j].copy()
        tol=1e-10*max(np.max(abs(xx)),1.)
        xx=np.round(xx/tol)*tol
        if np.ptp(xx)<=tol:xx[:]=xx.mean()
        pred=np.empty_like(yy);baseline=np.empty_like(yy)
        for k in range(5):
            train=ff!=k;test=~train
            assert not set(g.iloc[ii[train]].case)&set(g.iloc[ii[test]].case)
            baseline[test]=yy[train].mean(axis=0)
            xtrain=xx[train]
            scale=np.std(xtrain)
            if scale<=tol:
                pred[test]=baseline[test]
                kind='constant training predictor: mean baseline'
            else:
                center=xtrain.mean()
                xt=((xtrain-center)/scale)[:,None]
                xv=((xx[test]-center)/scale)[:,None]
                # Quantile knots from TRAINING ONLY, deduplicated for discrete predictors.
                knots=np.unique(np.quantile(xt[:,0],np.linspace(0,1,6)))
                spline=SplineTransformer(degree=3,knots=knots[:,None],include_bias=False,extrapolation='linear')
                bt=spline.fit_transform(xt);bv=spline.transform(xv)
                model=Ridge(alpha=1e-6).fit(bt,yy[train])
                pred[test]=model.predict(bv);kind=f'cubic spline, {len(knots)} unique knots'
            diagnostics.append(dict(scope=scope,variable=variable,fold=k,fit=kind,
                                    test_outside_training_range=int(((xx[test]<xtrain.min())|(xx[test]>xtrain.max())).sum())))
        assert np.isfinite(pred).all()
        for t,(method,response) in enumerate((m,r) for m in methods for r in responses):
            y=yy[:,t];p=pred[:,t];b=baseline[:,t]
            threshold=4/60 if response.startswith('centroid') else .5
            resolved=bool(np.ptp(y)>threshold)
            varying=bool(np.ptp(xx)>tol)
            rho=float(spearmanr(xx,y).statistic) if varying and np.ptp(y)>0 else np.nan
            pear=float(pearsonr(xx,y).statistic) if varying and np.ptp(y)>0 else np.nan
            r2,mae,rmse=score(y,p);br2,bmae,_=score(y,b)
            row=dict(affine_class=work.iloc[j].affine_class,variable=variable,concept=concept[variable],
                     method=method,response=response,scope=scope,n=len(y),predictor_varying=varying,
                     response_resolved=resolved,response_range_minutes=float(np.ptp(y)),
                     spearman=rho,pearson=pear,oof_r2=r2 if resolved else np.nan,
                     mae_minutes=mae,rmse_minutes=rmse,baseline_r2=br2 if resolved else np.nan,
                     baseline_mae_minutes=bmae,mae_improvement_fraction=1-mae/bmae if resolved and bmae>0 else np.nan)
            metrics.append(row)
            for k in range(5):
                mask=ff==k;fr2,fmae,frmse=score(y[mask],p[mask])
                fold_metrics.append(dict(variable=variable,method=method,response=response,scope=scope,
                                         fold=k,r2=fr2 if resolved else np.nan,mae_minutes=fmae,rmse_minutes=frmse))
    print('Completed',scope,flush=True)
metric=pd.DataFrame(metrics);metric.to_csv(OUT/'individual_scores.csv',index=False)
pd.DataFrame(fold_metrics).to_csv(OUT/'fold_scores.csv',index=False)
pd.DataFrame(diagnostics).to_csv(OUT/'fit_diagnostics.csv',index=False)
z=metric[(metric.scope!='pooled')&metric.response_resolved]
summary=z.groupby(['affine_class','variable','concept','method','response']).agg(
    mean_type_r2=('oof_r2','mean'),median_type_r2=('oof_r2','median'),
    min_type_r2=('oof_r2','min'),max_type_r2=('oof_r2','max'),
    positive_r2_types=('oof_r2',lambda a:int((a>0).sum())),n_resolved_types=('scope','size'),
    varying_types=('predictor_varying','sum'),mean_abs_spearman=('spearman',lambda a:a.abs().mean()),
    n_correlation_types=('spearman','count'),mean_abs_pearson=('pearson',lambda a:a.abs().mean()),
    mean_type_mae_minutes=('mae_minutes','mean'),mean_type_mae_improvement=('mae_improvement_fraction','mean')).reset_index()
pooled=metric[metric.scope.eq('pooled')][['variable','method','response','oof_r2','mae_minutes']].rename(columns={'oof_r2':'pooled_r2','mae_minutes':'pooled_mae_minutes'})
summary=summary.merge(pooled,on=['variable','method','response'],validate='one_to_one')
summary.to_csv(OUT/'candidate_summary.csv',index=False)
common_rows=[]
for response in responses:
    support=set.intersection(*[set(metric[(metric.method==m)&(metric.response==response)&metric.response_resolved&metric.scope.ne('pooled')].scope) for m in methods])
    common=metric[(metric.response==response)&metric.scope.isin(support)]
    cm=common.groupby(['variable','method','response']).agg(mean_type_r2=('oof_r2','mean'),n_types=('scope','nunique')).reset_index()
    common_rows.append(cm)
common_summary=pd.concat(common_rows,ignore_index=True)
common_summary.to_csv(OUT/'common_support_methods.csv',index=False)
def label(v):
    special={'delta_G_manning':'Manning ΔG','domain_size':'Affected-domain length',
             'parent_slope_slow':'Parent inverse-root slope'}
    if v in special:return special[v]
    return v.removeprefix('B_').removeprefix('path_').replace('delta_','Change in ').replace('_',' ').replace('route length','route length').capitalize()
for response in responses:
    summary[summary.response.eq(response)].sort_values(['method','mean_type_r2'],ascending=[True,False]).to_csv(OUT/f'{response}_ranking.csv',index=False)
    top=summary[(summary.method=='variable_manning')&(summary.response==response)].nlargest(20,'mean_type_r2').variable.tolist()
    table=common_summary[common_summary.response.eq(response)].pivot(index='variable',columns='method',values='mean_type_r2').reindex(index=top,columns=methods)
    fig,ax=plt.subplots(figsize=(12,9),constrained_layout=True)
    im=ax.imshow(table.to_numpy(),vmin=0,vmax=1,cmap='viridis',aspect='auto')
    ax.set(xticks=range(3),xticklabels=['Variable Manning','Fixed Manning','Fixed BASED'],
           yticks=range(20),yticklabels=[label(v) for v in top],title=response.replace('_',' ')+' · top 20 by variable Manning\nMean held-out R² · same six collapse types across all methods')
    ax.tick_params(axis='y',labelsize=9)
    for i in range(20):
        for j in range(3):
            v=table.iloc[i,j];ax.text(j,i,f'{v:.2f}',ha='center',va='center',color='white' if v<.5 else 'black',fontsize=8)
    fig.colorbar(im,ax=ax,label='Mean within-type OOF R² (colour clipped to 0–1)')
    fig.savefig(OUT/f'{response}_top20.png',dpi=160);plt.close(fig)
lines=['# Individual predictor screen — 2 November 2026','',
'Reference: SYN-INDIVIDUAL-PREDICTOR-SCREEN-2026-11-02. All 100 working affine classes evaluated; no candidates removed.','',
'## Design and interpretation','',
'Five deterministic shuffled case-group folds (seed 20261102) hold all seven collapse entries of a parameter combination together. The same folds apply across all predictors, responses and depth approaches. This evaluates interpolation to held-out combinations within the existing grid, not extrapolation to new control ranges, unseen collapse types or real networks. Controls irrelevant to a treatment can yield repeated geometries across different cases; folds do not claim unique-geometry holdout. The synthetic data have already informed prior decisions; these are comparative internal validation scores, not untouched external validation.','',
'Each predictor alone fits a cubic spline with at most six training-quantile knot locations (duplicate knots removed), linear extrapolation, and ridge alpha 1e-6. Training-only centring/scaling and knot fitting prevent fold leakage. Hyperparameters are fixed across candidates; there is no search. Constant predictors return the training response mean. One multi-output fit shares only the predictor basis, with independent coefficients for the six method/response targets. No topology indicator or second predictor is supplied.','',
'Pooled models fit all collapse types together; within-type models are fitted separately. These answer different questions. The main ranking is equal-weight mean out-of-fold R² across response-resolved types. Constant predictors remain in that average as baseline predictions; correlations are undefined for them and their available-type counts are reported. Pooled R² can reflect between-type structure and must not substitute for within-type sensitivity.','',
'The existing operational resolution screen is retained: full-range centroid change >4 s, peak change >30 s. These are screening conventions, not validated numerical error bounds. Unresolved strata still have MAE reported, but R² is not ranked. Responses are not rounded. Predictor roundoff handling follows the earlier audit. Signed correlations are retained in detailed tables. Negative R² is not clipped in tables or averages. No p-values or independent-replication confidence intervals are claimed.','',
'Five undecided affine classes use temporary labels listed in screened_candidates.csv. This does not settle real-network representatives. Domain size is converted from km to m in the analysis; other source features are preserved.','',
'Cross-method figures and common_support_methods.csv use the same six resolved treatments for all three approaches. Method-specific rankings below retain all resolved treatments: six for Manning, seven for BASED. Do not attribute differences between those unequal-support averages solely to hydraulics.','',
'## Leading standalone candidates','']
for method in methods:
    for response in responses:
        lines += [f'### {method}: {response}','', '| Candidate | Mean type R² | Worst type R² | Pooled R² | Resolved types |','|---|---:|---:|---:|---:|']
        for r in summary[(summary.method==method)&(summary.response==response)].nlargest(10,'mean_type_r2').itertuples():
            lines.append(f'| `{r.variable}` | {r.mean_type_r2:.3f} | {r.min_type_r2:.3f} | {r.pooled_r2:.3f} | {r.n_resolved_types} |')
        lines.append('')
lines += ['## Limitations and next step','',
'This fixed univariate model measures standalone usefulness, not optimal univariate predictability or added information in combinations. Poor scores do not establish that a descriptor is wrong or useless; interactions can make weak marginal variables useful. Fold variation is a partition sensitivity description, not an iid confidence interval. Next compare related candidates and documented alternatives as additions to common baselines using the same grouped splits, with subsequent selection confined to training folds. Keep real-network checks separate.','',
'Outputs: candidate_summary.csv and per-response ranking tables; individual_scores.csv for all strata; fold_scores.csv; fit_diagnostics.csv; case_folds.csv; screened_candidates.csv; two top-20 figures.']
(OUT/'report.md').write_text('\n'.join(lines)+'\n')
sources=[SOURCE,WORK,concept_path,Path(__file__)]
(OUT/'provenance.json').write_text(json.dumps(dict(reference='SYN-INDIVIDUAL-PREDICTOR-SCREEN-2026-11-02',sources={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
    sklearn=sklearn.__version__,seed=20261102,folds=5,candidates=100,geometry_rows=45927,
    model='fixed cubic spline, <=6 quantile knots, ridge alpha=1e-6',routing_runs=0),indent=2))
print(summary[summary.method.eq('variable_manning')].sort_values(['response','mean_type_r2'],ascending=[True,False]).groupby('response').head(8)[['variable','response','mean_type_r2','pooled_r2']].to_string(index=False))
