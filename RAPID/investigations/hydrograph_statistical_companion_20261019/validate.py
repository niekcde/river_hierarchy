"""Independent numerical checks, run in base Python with statsmodels installed.

Input CSV is a lossless-enough bridge exported from the frozen parquet by the
project Python; this script never changes the frozen data or model.
"""
import json
import warnings
import numpy as np
import pandas as pd
import statsmodels.api as sm
from run import fit,predict,weights,COLS,OUT

def main():
    d=pd.read_csv('/tmp/companion_validation.csv',dtype={'example':str})
    checks=[]
    for label,g in [('all',d),('without37',d[d.example.ne('37')]),('without54',d[d.example.ne('54')])]:
        raw=g[COLS].to_numpy();mu=raw.mean(axis=0);sd=raw.std(axis=0);x=sm.add_constant((raw-mu)/sd);y=g.centroid_shift_hours
        b,info=fit(g,'mixed')
        with warnings.catch_warnings(record=True) as caught:
            ref=sm.MixedLM(y,x,groups=g.example).fit(reml=True,method='bfgs',maxiter=2000,gtol=1e-9)
        delta=float(np.max(abs(predict(g,b)-ref.predict(x))))
        assert delta<1e-4,(label,delta)
        np.testing.assert_allclose(info['network_variance_hours2'],ref.cov_re.iloc[0,0],rtol=1e-3,atol=1e-6)
        bw,_=fit(g,'wls');wls=sm.WLS(y,x,weights=weights(g)).fit();np.testing.assert_allclose(predict(g,bw),wls.predict(x),atol=1e-10)
        changed=g.assign(centroid_shift_hours=999.,example='unseen');np.testing.assert_array_equal(predict(g,b),predict(changed,b))
        checks.append(dict(cohort=label,max_prediction_difference_hours=delta,reference_converged=bool(ref.converged),warnings=[str(w.message) for w in caught]))
    # Explicit independent-row limit and fresh group IDs for resampling.
    rng=np.random.default_rng(31);synthetic=d.copy();synthetic[COLS]=rng.normal(size=(len(d),len(COLS)));synthetic['centroid_shift_hours']=synthetic[COLS].to_numpy()@np.arange(1,8)+rng.normal(size=len(d))
    bs,info=fit(synthetic,'mixed');assert info['lambda_ratio']>=0
    groups=[d[d.example.eq('02')].assign(example=f'draw_{i}') for i in range(2)]
    sample=pd.concat(groups);assert sample.example.nunique()==2
    assert np.allclose(pd.Series(weights(sample)).groupby(sample.example.to_numpy()).sum(),1)
    OUT.joinpath('independent_validation.json').write_text(json.dumps(dict(statsmodels_version=sm.__version__,checks=checks,synthetic_lambda=info['lambda_ratio'],population_prediction_invariance=True,wls_agreement=True),indent=2)+'\n')
    print('Independent validation passed',checks)
if __name__=='__main__':main()
