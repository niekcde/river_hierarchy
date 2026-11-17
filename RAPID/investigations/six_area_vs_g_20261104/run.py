"""Replace G by area change in the exact six-input quadratic ridge."""
import os
for key in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:
    os.environ[key]='1'
os.environ.setdefault('MPLCONFIGDIR','/tmp/no-p90-four-mpl')
from pathlib import Path
import sys,json,hashlib,time,importlib.util
import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures,StandardScaler
from sklearn.linear_model import Ridge
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'RAPID/investigations/streamlined_predictors_20261103'))
import run_comparison as base
SOURCE=ROOT/'RAPID/outputs/mean_width_compact_20261103'
OUT=ROOT/'RAPID/outputs/six_area_vs_g_20261104'
MODEL='five_mean_width_with_G'
ALPHA_PER_ROW=1e-4

def main():
    start=time.monotonic();OUT.mkdir(exist_ok=True)
    old=json.loads((SOURCE/'design.json').read_text());receipt=json.loads((SOURCE/'completion.json').read_text())
    assert receipt['complete']
    assert base.sha(SOURCE/'predictions.parquet')==receipt['files'][str(SOURCE/'predictions.parquet')]
    for p,h in old['inputs'].items():assert base.sha(p)==h,p
    columns=['B_delta_area_proxy' if c=='delta_G_manning' else c for c in old['panels'][MODEL]]+['delta_entropy'];d=pd.read_parquet(SOURCE/'real_input.parquet')
    assert len(columns)==6 and len(d)==2185 and np.isfinite(d[columns]).all().all()
    tasks=[t for t in json.loads((SOURCE/'tasks.json').read_text())['tasks'] if t['dataset']=='real' and t['model']==MODEL]
    assert len(tasks)==65
    baseline_folder=ROOT/'RAPID/outputs/entropy_complexity_g_ridge_20261104'
    assert json.loads((baseline_folder/'completion.json').read_text())['complete']
    baseline_design=json.loads((baseline_folder/'design.json').read_text())
    assert sum(a!=b for a,b in zip(columns,baseline_design['columns']))==1
    metric_path=ROOT/'RAPID/investigations/six_input_kge_20261104/run.py'
    paths=[baseline_folder/'completion.json',baseline_folder/'design.json',baseline_folder/'predictions.parquet',metric_path,Path(__file__),SOURCE/'design.json',SOURCE/'tasks.json',SOURCE/'real_input.parquet',SOURCE/'completion.json',SOURCE/'predictions.parquet']
    design=dict(reference='SIX-AREA-VS-G-2026-11-04',columns=columns,
        sources={str(p):base.sha(p) for p in paths},n_rows=len(d),fits=65,
        ridge='Degree-two polynomial: six main effects, six squares and fifteen pair products, plus unpenalized intercept. Expanded columns standardized with training-only weighted moments. Ridge penalty 1e-4 times sum of training weights, weights normalized to mean one. No tuning.',
        weights='Training unchanged: equal total per river. Primary reporting: combined equal-pair weighting. Separate cohorts and equal-river/balanced-cohort scores retained as diagnostics.',
        representation='Replace only delta_G_manning by B_delta_area_proxy in the six-input set; entropy and complexity retained. Area uses existing affected-domain footprint; G uses whole-network accumulation differences, as in prior definitions. No feature, penalty or routing search.',
        response='Fixed-depth Manning centroid shift in hours; scores MAE minutes.',
        validation=old['validation'],rf_params=old['params'],
        caveat='Real models trained on real training folds, not synthetic-to-real coefficient transfer. Exploratory comparison on reused data. Quadratic ridge penalty retained from synthetic procedure, not tuned for real data.')
    base.save(OUT/'design.json',design)
    poly=PolynomialFeatures(degree=2,include_bias=False)
    # Stateless expansion can precede splitting; all fitted scaling uses train rows.
    X=poly.fit_transform(d[columns].to_numpy(float));assert X.shape==(2185,27)
    parts=[];checks=[]
    for i,t in enumerate(tasks,1):
        tri=np.asarray(t['train_indices']);tei=np.asarray(t['test_indices'])
        tr=d.iloc[tri];te=d.iloc[tei];y=tr.centroid_shift_hours.to_numpy()
        w=base.legacy.refit.tuning.b.engine.weights(tr)
        assert np.isclose(w.mean(),1.)
        scaler=StandardScaler().fit(X[tri],sample_weight=w)
        good=scaler.var_>0
        z=scaler.transform(X[tri])[:,good];zt=scaler.transform(X[tei])[:,good]
        alpha=ALPHA_PER_ROW*w.sum()
        model=Ridge(alpha=alpha,fit_intercept=True,solver='svd').fit(z,y,sample_weight=w)
        pred=model.predict(zt);assert np.isfinite(pred).all()
        # Independent normal-equation check for this small, standardized system.
        zm=np.average(z,axis=0,weights=w);ym=np.average(y,weights=w)
        centered=z-zm
        beta=np.linalg.solve(centered.T@(w[:,None]*centered)+alpha*np.eye(z.shape[1]),centered.T@(w*(y-ym)))
        direct=ym+(zt-zm)@beta
        np.testing.assert_allclose(pred,direct,rtol=1e-7,atol=1e-7)
        checks.append(dict(strategy=t['strategy'],repeat=t['repeat'],fold=t['fold'],n_train=len(tr),n_test=len(te),terms=int(good.sum()),max_difference_hours=float(np.max(abs(pred-direct)))))
        r=te[['example','state','cohort','edit_type','centroid_shift_hours']].copy();r['predicted_hours']=pred
        r['strategy']=t['strategy'];r['repeat']=t['repeat'];r['fold']=t['fold'];r['estimator']='six_area_ridge';parts.append(r)
    baseline=pd.read_parquet(baseline_folder/'predictions.parquet')
    baseline=baseline[baseline.estimator.eq('entropy_complexity_G_ridge')].copy();baseline['estimator']='six_G_ridge'
    new=pd.concat(parts,ignore_index=True)
    keys=['strategy','repeat','example','state']
    left=new.set_index(keys).sort_index();right=baseline.set_index(keys).sort_index()
    pd.testing.assert_index_equal(left.index,right.index)
    np.testing.assert_allclose(left.centroid_shift_hours,right.centroid_shift_hours,atol=0,rtol=0)
    p=pd.concat([baseline,new],ignore_index=True)
    assert not p.duplicated(keys+['estimator']).any()
    p.to_parquet(OUT/'predictions.parquet',index=False);pd.DataFrame(checks).to_csv(OUT/'numerical_checks.csv',index=False)
    spec=importlib.util.spec_from_file_location('six_area_kge_metrics',metric_path)
    helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
    rows=[]
    for (strategy,repeat,estimator),g in p.groupby(['strategy','repeat','estimator']):
        for population,z in [('ordinary',g[g.cohort.eq('original')]),('full_network',g[g.cohort.eq('segmented')]),('all',g)]:
            for weighting in ['equal_pair','equal_river']:
                if weighting=='equal_pair':w=np.ones(len(z))
                elif population=='all':
                    counts=z.groupby('cohort').example.nunique()
                    w=1/z.groupby(['cohort','example']).example.transform('size').to_numpy()/z.cohort.map(counts).to_numpy()
                else:w=1/z.groupby('example').example.transform('size').to_numpy()
                rows.append(dict(strategy=strategy,repeat=repeat,estimator=estimator,population=population,weighting=weighting,
                    **helper.metrics(z.centroid_shift_hours.to_numpy()*60,z.predicted_hours.to_numpy()*60,w)))
    scores=pd.DataFrame(rows);scores.to_csv(OUT/'repeat_scores.csv',index=False)
    summary=scores.groupby(['strategy','population','weighting','estimator']).agg(n=('n','first'),repeats=('repeat','size'),r2=('r2','mean'),r2_min=('r2','min'),r2_max=('r2','max'),kge=('kge','mean'),kge_min=('kge','min'),kge_max=('kge','max'),mae_minutes=('mae_minutes','mean')).reset_index()
    summary.to_csv(OUT/'summary.csv',index=False)
    # Check unchanged G predictions reproduce the accepted KGE/R² table.
    prior=pd.read_csv(ROOT/'RAPID/outputs/six_input_kge_20261104/summary.csv')
    prior=prior[prior.estimator.eq('entropy_complexity_G_ridge')]
    check=summary[summary.estimator.eq('six_G_ridge')].merge(prior,on=['strategy','population','weighting'],suffixes=('_new','_old'),validate='one_to_one')
    for field in ['r2','kge','mae_minutes']:np.testing.assert_allclose(check[field+'_new'],check[field+'_old'],atol=1e-12)
    paired=scores[scores.estimator.eq('six_area_ridge')].merge(scores[scores.estimator.eq('six_G_ridge')],on=['strategy','repeat','population','weighting'],suffixes=('_area','_G'),validate='one_to_one')
    assert (paired.n_area==paired.n_G).all()
    for metric in ['r2','kge','mae_minutes']:paired['delta_'+metric]=paired[metric+'_area']-paired[metric+'_G']
    paired.to_csv(OUT/'paired_changes.csv',index=False)
    p['absolute_error_minutes']=60*abs(p.predicted_hours-p.centroid_shift_hours)
    p.groupby(['strategy','repeat','estimator','cohort','example']).absolute_error_minutes.mean().rename('mae_minutes').to_csv(OUT/'per_river_errors.csv')
    p.sort_values('absolute_error_minutes',ascending=False).groupby(['strategy','estimator']).head(10).to_csv(OUT/'largest_errors.csv',index=False)
    lines=['# Area change versus G in the exact six-input quadratic ridge','','Reference SIX-AREA-VS-G-2026-11-04. Replace only G with the existing area-proxy change. Same real targets, rows, splits, equal-river training weights, training-only weighted standardization and penalty (1e-4 per normalized training-weight sum). Both models have 27 degree-two terms plus intercept. No tuning.','','Primary reporting: all 2,185 comparisons, equal-pair weighting. R², KGE-2009 and MAE calculated per complete out-of-fold repeat, then averaged. Separate populations and equal-river/cohort-balanced scores retained in summary.csv.','','| Validation | Model | R² | KGE | MAE (min) |','|---|---|---:|---:|---:|']
    primary=summary[summary.population.eq('all')&summary.weighting.eq('equal_pair')]
    for r in primary.itertuples():lines.append(f'| {r.strategy} | {r.estimator} | {r.r2:.4f} | {r.kge:.4f} | {r.mae_minutes:.4f} |')
    lines+=['','Comparing these candidates also retains their existing domains: G accounts for whole-network allocation changes, whereas area is summed over the affected reach footprint. This is a descriptor replacement, not proof that a single exponent or hydraulic assumption causes any performance difference. Reused real validation data make this exploratory. Fixed penalty is not tuned separately to favor either descriptor. KGE retains signed response values and can be unstable with near-zero reference means. No extreme errors clipped.','']
    (OUT/'report.md').write_text('\n'.join(lines))
    for path,h in design['sources'].items():assert base.sha(path)==h,path
    base.save(OUT/'completion.json',dict(complete=True,fits=65,seconds=time.monotonic()-start,baseline_scores_reproduced=True))
    print(primary.to_string(index=False))

if __name__=='__main__':main()
