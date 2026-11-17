"""Test six-input quadratic ridge retaining both entropy and complexity with G."""
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
OUT=ROOT/'RAPID/outputs/entropy_complexity_g_ridge_20261104'
MODEL='five_mean_width_with_G'
ALPHA_PER_ROW=1e-4

def main():
    start=time.monotonic();OUT.mkdir(exist_ok=True)
    old=json.loads((SOURCE/'design.json').read_text());receipt=json.loads((SOURCE/'completion.json').read_text())
    assert receipt['complete']
    assert base.sha(SOURCE/'predictions.parquet')==receipt['files'][str(SOURCE/'predictions.parquet')]
    for p,h in old['inputs'].items():assert base.sha(p)==h,p
    columns=old['panels'][MODEL]+['delta_entropy'];d=pd.read_parquet(SOURCE/'real_input.parquet')
    assert len(columns)==6 and len(d)==2185 and np.isfinite(d[columns]).all().all()
    tasks=[t for t in json.loads((SOURCE/'tasks.json').read_text())['tasks'] if t['dataset']=='real' and t['model']==MODEL]
    assert len(tasks)==65
    ridge_source=ROOT/'RAPID/outputs/mean_width_ridge_vs_rf_20261103'
    entropy_source=ROOT/'RAPID/outputs/entropy_g_ridge_20261104'
    for folder in [ridge_source,entropy_source]:assert json.loads((folder/'completion.json').read_text())['complete']
    paths=[ridge_source/'completion.json',ridge_source/'predictions.parquet',entropy_source/'completion.json',entropy_source/'predictions.parquet',Path(__file__),SOURCE/'design.json',SOURCE/'tasks.json',SOURCE/'real_input.parquet',SOURCE/'completion.json',SOURCE/'predictions.parquet']
    design=dict(reference='ENTROPY-COMPLEXITY-G-RIDGE-2026-11-04',columns=columns,
        sources={str(p):base.sha(p) for p in paths},n_rows=len(d),fits=65,
        ridge='Degree-two polynomial: six main effects, six squares and fifteen pair products, plus unpenalized intercept. Expanded columns standardized with training-only weighted moments. Ridge penalty 1e-4 times sum of training weights, weights normalized to mean one. No tuning.',
        weights='Same equal-total-per-river normalized sample weights as RF; scoring also matches existing cohort/river convention.',
        representation='Six inputs: allocated mean route-length change, parent slope descriptor, log1p(child/parent mean-width ratio), complexity change, delta_G_manning, entropy change. Add entropy to complexity+G; retain both topology descriptors. Degree-two expansion adds the new main effect, square and five interactions.',
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
        r['strategy']=t['strategy'];r['repeat']=t['repeat'];r['fold']=t['fold'];r['estimator']='entropy_complexity_G_ridge';parts.append(r)
    rf=pd.read_parquet(SOURCE/'predictions.parquet')
    rf=rf[rf.dataset.eq('real')&rf.model.eq(MODEL)].copy();rf['estimator']='complexity_G_RF'
    prior=pd.read_parquet(ridge_source/'predictions.parquet')
    prior=prior[prior.estimator.eq('quadratic_ridge')].copy();prior['estimator']='complexity_G_ridge'
    prior_entropy=pd.read_parquet(entropy_source/'predictions.parquet')
    prior_entropy=prior_entropy[prior_entropy.estimator.eq('entropy_G_ridge')].copy();prior_entropy['estimator']='entropy_G_ridge'
    # Compare identical row/target support in every repeated validation split.
    keys=['strategy','repeat','example','state']
    new=pd.concat(parts,ignore_index=True).set_index(keys).sort_index()
    for baseline in [rf,prior,prior_entropy]:
        check=baseline.set_index(keys).sort_index()
        pd.testing.assert_index_equal(new.index,check.index)
        np.testing.assert_allclose(new.centroid_shift_hours,check.centroid_shift_hours,atol=0,rtol=0)
    p=pd.concat([rf,prior,prior_entropy,*parts],ignore_index=True)
    assert not p.duplicated(['strategy','repeat','estimator','example','state']).any()
    p.to_parquet(OUT/'predictions.parquet',index=False);pd.DataFrame(checks).to_csv(OUT/'numerical_checks.csv',index=False)
    spec=importlib.util.spec_from_file_location('ridge_rf_score_helpers',Path(base.__file__).with_name('summarize.py'))
    helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
    rows=[]
    for (strategy,repeat,estimator),g in p.groupby(['strategy','repeat','estimator']):
        for population,z in [('ordinary',g[g.cohort.eq('original')]),('full_network',g[g.cohort.eq('segmented')]),('all',g)]:
            rows.append(dict(strategy=strategy,repeat=repeat,estimator=estimator,population=population,**helper.scores(z,population=='all')))
    s=pd.DataFrame(rows);s.to_csv(OUT/'repeat_scores.csv',index=False)
    summary=s.groupby(['strategy','population','estimator']).agg(n=('n','first'),repeats=('repeat','nunique'),r2=('r2','mean'),r2_min=('r2','min'),r2_max=('r2','max'),mae_minutes=('mae_minutes','mean')).reset_index()
    summary.to_csv(OUT/'summary.csv',index=False)
    paired=[]
    for comparator in ['complexity_G_ridge','entropy_G_ridge','complexity_G_RF']:
        z=s[s.estimator.eq('entropy_complexity_G_ridge')].merge(s[s.estimator.eq(comparator)],on=['strategy','repeat','population'],suffixes=('_new','_reference'),validate='one_to_one')
        assert (z.n_new==z.n_reference).all()
        z['delta_r2']=z.r2_new-z.r2_reference;z['delta_mae_minutes']=z.mae_minutes_new-z.mae_minutes_reference;z['comparator']=comparator;paired.append(z)
    pd.concat(paired).to_csv(OUT/'paired_changes.csv',index=False)
    p['absolute_error_minutes']=60*abs(p.predicted_hours-p.centroid_shift_hours)
    p.groupby(['strategy','repeat','estimator','cohort','example']).absolute_error_minutes.mean().rename('mae_minutes').to_csv(OUT/'per_river_errors.csv')
    p.sort_values('absolute_error_minutes',ascending=False).groupby(['strategy','estimator']).head(10).to_csv(OUT/'largest_errors.csv',index=False)
    lines=['# Six-input entropy + complexity + G quadratic ridge','','Reference ENTROPY-COMPLEXITY-G-RIDGE-2026-11-04. Real-data training and validation; no synthetic-to-real transfer. New panel retains both entropy and complexity change alongside G. Targets, rows, partitions, sample weights and penalty unchanged. Completed complexity+G ridge/RF and entropy+G ridge baselines are reused.','','Inputs: '+', '.join(columns)+'.','','Ridge: degree-two main/square/interaction expansion, training-only weighted standardization, unpenalized intercept, alpha = 1e-4 × normalized training-weight sum. Twenty-seven expanded terms; no hyperparameter tuning. The width ratio is log1p transformed for both estimators, matching existing RF inputs.','','| Validation | Population | Estimator | R² | MAE (min) |','|---|---|---|---:|---:|']
    for r in summary.itertuples():lines.append(f'| {r.strategy} | {r.population} | {r.estimator} | {r.r2:.4f} | {r.mae_minutes:.4f} |')
    lines+=['','Ordinary comparisons (2,170) and full-network controls (15) are separated. Negative R² is retained and indicates larger squared error than the scored test-population mean reference. No clipping or removal of extreme predictions. This tests the existing ridge specification with entropy+complexity+G, not an optimized ridge model. No six-input RF has been fitted in this experiment. The ordinary/full-network distinction does not by itself establish a size-driven mechanism. Prior real-data inspection limits independent-validation claims.','']
    (OUT/'report.md').write_text('\n'.join(lines))
    for path,h in design['sources'].items():assert base.sha(path)==h,path
    base.save(OUT/'completion.json',dict(complete=True,fits=65,seconds=time.monotonic()-start))
    print(summary.to_string(index=False))

if __name__=='__main__':main()
