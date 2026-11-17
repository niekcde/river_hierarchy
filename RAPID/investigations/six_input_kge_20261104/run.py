"""KGE-2009 diagnostics of saved held-out centroid-shift predictions; no fitting."""
from pathlib import Path
import hashlib,json
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[3]
SOURCE=ROOT/'RAPID/outputs/entropy_complexity_g_ridge_20261104'
OUT=ROOT/'RAPID/outputs/six_input_kge_20261104'
MODEL='entropy_complexity_G_ridge'

def metrics(y,p,w):
    y=np.asarray(y,float);p=np.asarray(p,float);w=np.asarray(w,float)
    assert np.isfinite(y).all() and np.isfinite(p).all() and (w>0).all()
    w=w/w.sum();my=float(w@y);mp=float(w@p)
    sy=float(np.sqrt(w@((y-my)**2)));sp=float(np.sqrt(w@((p-mp)**2)))
    tol=1e-12*max(float(np.max(abs(y))),1e-12)
    r=float(np.clip(w@((y-my)*(p-mp))/(sy*sp),-1,1)) if sy>tol and sp>tol else np.nan
    alpha=sp/sy if sy>tol else np.nan
    beta=mp/my if abs(my)>tol else np.nan
    kge=1-float(np.sqrt((r-1)**2+(alpha-1)**2+(beta-1)**2)) if np.isfinite([r,alpha,beta]).all() else np.nan
    err=p-y
    return dict(n=len(y),kge=kge,r=r,alpha=alpha,beta=beta,
        r2=1-float(w@(err**2))/(sy**2) if sy>tol else np.nan,
        mae_minutes=float(w@abs(err)),rmse_minutes=float(np.sqrt(w@(err**2))),
        reference_mean_minutes=my,predicted_mean_minutes=mp,reference_sd_minutes=sy,predicted_sd_minutes=sp,
        mean_bias_minutes=mp-my,abs_reference_mean_over_sd=abs(my)/sy if sy>tol else np.nan,
        n_positive=int((y>0).sum()),n_negative=int((y<0).sum()),n_zero=int((y==0).sum()),
        defined=bool(np.isfinite(kge)))

def main():
    OUT.mkdir(exist_ok=True)
    assert json.loads((SOURCE/'completion.json').read_text())['complete']
    file=SOURCE/'predictions.parquet';sha=hashlib.sha256(file.read_bytes()).hexdigest()
    p=pd.read_parquet(file);rows=[];river=[]
    assert not p.duplicated(['strategy','repeat','estimator','example','state']).any()
    for (strategy,repeat,estimator),g in p.groupby(['strategy','repeat','estimator']):
        for population,z in [('ordinary',g[g.cohort.eq('original')]),('full_network',g[g.cohort.eq('segmented')]),('all',g)]:
            y=z.centroid_shift_hours.to_numpy()*60;pred=z.predicted_hours.to_numpy()*60
            for weighting in ['equal_river','equal_pair']:
                if weighting=='equal_pair':w=np.ones(len(z))
                elif population=='all':
                    counts=z.groupby('cohort').example.nunique()
                    w=1/z.groupby(['cohort','example']).example.transform('size').to_numpy()/z.cohort.map(counts).to_numpy()
                else:w=1/z.groupby('example').example.transform('size').to_numpy()
                m=metrics(y,pred,w)
                rows.append(dict(strategy=strategy,repeat=repeat,estimator=estimator,population=population,weighting=weighting,**m))
                # Invariance checks: hours/minutes or joint sign reversal must
                # not change KGE-2009. No absolute values or offsets are used.
                if m['defined']:
                    np.testing.assert_allclose(m['kge'],metrics(y/60,pred/60,w)['kge'],atol=1e-12)
                    np.testing.assert_allclose(m['kge'],metrics(-y,-pred,w)['kge'],atol=1e-12)
        if estimator==MODEL:
            for ex,z in g[g.cohort.eq('original')].groupby('example'):
                river.append(dict(strategy=strategy,repeat=repeat,example=ex,**metrics(z.centroid_shift_hours.to_numpy()*60,z.predicted_hours.to_numpy()*60,np.ones(len(z)))))
    s=pd.DataFrame(rows);s.to_csv(OUT/'repeat_scores.csv',index=False)
    # Verify weights reproduce the already reported R²; do not change the
    # evaluation population or average per-fold KGE silently.
    prior=pd.read_csv(SOURCE/'repeat_scores.csv')
    check=s[s.weighting.eq('equal_river')].merge(prior,on=['strategy','repeat','estimator','population'],suffixes=('_new','_old'),validate='one_to_one')
    np.testing.assert_allclose(check.r2_new,check.r2_old,atol=1e-12)
    np.testing.assert_allclose(check.mae_minutes_new,check.mae_minutes_old,atol=1e-12)
    test=np.array([-2.,-1.,1.,-3.]);assert np.isclose(metrics(test,test,np.ones(4))['kge'],1.)
    assert not metrics([-1.,1.],[-1.,1.],[1.,1.])['defined']
    summary=s.groupby(['strategy','population','estimator','weighting']).agg(n=('n','first'),repeats=('repeat','size'),defined_repeats=('defined','sum'),
        kge=('kge','mean'),kge_min=('kge','min'),kge_max=('kge','max'),r2=('r2','mean'),r=('r','mean'),alpha=('alpha','mean'),beta=('beta','mean'),
        mae_minutes=('mae_minutes','mean'),mean_bias_minutes=('mean_bias_minutes','mean'),reference_mean_minutes=('reference_mean_minutes','first'),reference_sd_minutes=('reference_sd_minutes','first')).reset_index()
    summary.to_csv(OUT/'summary.csv',index=False);pd.DataFrame(river).to_csv(OUT/'per_river_diagnostics.csv',index=False)
    primary=summary[summary.estimator.eq(MODEL)&summary.weighting.eq('equal_river')]
    lines=['# Six-input ridge: KGE diagnostics','','Reference SIX-INPUT-KGE-2026-11-04. No new fits. KGE calculated across saved held-out parent–child centroid-shift predictions, not across discharge time steps and not against observed gauge hydrographs.','','KGE-2009 = 1 - sqrt((r-1)^2 + (alpha-1)^2 + (beta-1)^2), alpha = prediction SD / reference SD, beta = prediction mean / reference mean. Equal component importance. Reference is routed model centroid shift; prediction is the geometric regression estimate. Signed shifts retained. No absolute-value, log or offset transformation.','','Primary scores use equal-river weighted moments, matching existing R²/MAE population weighting. This is a sample-weighted extension of the standard KGE expression, not unequal weighting of its three components. Equal-pair conventional scores are exported as sensitivity. Combined cohorts use the existing equal-cohort/equal-river weights. Compute on the complete out-of-fold prediction population for each repeat, then average repeat scores; do not average per-fold or per-river KGE.','','| Validation | Population | R² | KGE | r | SD ratio | Mean ratio | MAE (min) |','|---|---|---:|---:|---:|---:|---:|---:|']
    for z in primary.itertuples():lines.append(f'| {z.strategy} | {z.population} | {z.r2:.4f} | {z.kge:.4f} | {z.r:.4f} | {z.alpha:.4f} | {z.beta:.4f} | {z.mae_minutes:.4f} |')
    lines+=['','Suitability: supplementary diagnostic of correlation, variability ratio and mean bias. Signed responses can cancel to a near-zero reference mean, making beta unstable or undefined; the original KGE formula has no protection against this. Preserve component values and signed mean bias, and do not use generic streamflow performance thresholds to label centroid-shift predictions. KGE is not interchangeable with R². This prediction test does not validate RAPID against gauges. R² plus MAE remain primary; no fitting objective or model decision is changed by adding KGE.','',
        'Reference: [Knoben, Freer & Woods (2019), equations 2–3 and interpretation](https://hess.copernicus.org/articles/23/4323/2019/). The near-zero-mean warning follows directly from the ratio in that equation; it is especially relevant to our signed response.','']
    (OUT/'report.md').write_text('\n'.join(lines))
    assert hashlib.sha256(file.read_bytes()).hexdigest()==sha
    (OUT/'completion.json').write_text(json.dumps(dict(complete=True,reference='SIX-INPUT-KGE-2026-11-04',prediction_sha256=sha,no_new_fits=True,r2_mae_reproduction_check=True,unit_sign_and_perfect_prediction_checks=True),indent=2))
    print(primary.to_string(index=False))

if __name__=='__main__':main()
