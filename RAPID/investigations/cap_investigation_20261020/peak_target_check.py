"""Separate peak-target RF with frozen depth-informed centroid settings."""
import argparse
import hashlib
import json
import platform
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestRegressor
from tune_nine_inputs import b
from slope_ratio_audit import OUT
from ten_input_outlier_audit import weights
from alternative_validation import save, sha

DEST=OUT/'peak_target_check_20261026'
TARGET='peak_shift_hours'


def fit(d,cols,params,jobs):
    y=d[TARGET].to_numpy(float);w=b.engine.weights(d)
    pre=b.engine.Preprocess(cols,None,None).fit(d,y)
    mean=float(np.average(y,weights=w));sd=float(np.sqrt(np.average((y-mean)**2,weights=w)))
    assert sd>0 and pre.kept
    rf=RandomForestRegressor(**params,random_state=b.engine.SEED,n_jobs=jobs)
    rf.fit(pre.transform(d),(y-mean)/sd,sample_weight=w)
    return dict(preprocessor=pre,estimator=rf,target_mean=mean,target_sd=sd,columns=cols,
        params=params,target=TARGET,prediction_units='hours')


def scores(z,mode):
    y=z[TARGET].to_numpy();e=z.predicted_hours.to_numpy()-y;w=weights(z,mode)
    sst=np.sum(w*(y-np.average(y,weights=w))**2)
    return dict(n=len(z),networks=z.example.nunique(),r2=1-np.sum(w*e**2)/sst if sst>0 else None,
        mae_minutes=60*np.average(abs(e),weights=w),relative_error_pct=100*np.average(abs(e)/np.maximum(abs(y),1/60),weights=w))


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',action='store_true');ap.add_argument('--jobs',type=int,default=4);args=ap.parse_args()
    data_path=OUT/'alternative_validation_20261023/input.parquet'
    settings_path=OUT/'current_model_comparison_20261023/settings.json'
    baseline_path=OUT/'ten_input_tuning_20261023/predictions.parquet'
    info=json.loads(settings_path.read_text())['depth_informed_10'];cols=info['columns'];params=info['params']
    d=pd.read_parquet(data_path)
    assert len(d)==2185 and d.example.nunique()==15 and not d.duplicated(['example','state']).any()
    assert np.isfinite(d[cols+[TARGET]]).all().all()
    sources=[data_path,settings_path,baseline_path,Path(info['artifact']),Path(__file__),Path(b.engine.__file__),Path(b.forest.__file__),Path(__file__).with_name('ten_input_outlier_audit.py')]
    design=dict(target=TARGET,columns=cols,params=params,seed=b.engine.SEED,rows=len(d),networks=sorted(d.example.unique()),
        sources={str(p):sha(p) for p in sources},versions=dict(python=platform.python_version(),sklearn=sklearn.__version__,numpy=np.__version__,pandas=pd.__version__,joblib=joblib.__version__),
        method='15 whole-network held-out fits at frozen final centroid settings; no peak hyperparameter tuning. Training-only preprocessing and target standardization, equal total training weight per network. One separate all-data peak refit.',
        response='Current shared 0.5% window, global-maximum timing, child minus parent; no selective crest matching.',
        sensitivity='without54 excludes example54 only from scoring; the other folds may include54 in training. Not a 54-excluded retraining.',
        limitation='Settings and predictors selected in prior centroid investigations on these networks. Fixed-setting diagnostic, not a newly nested-tuned or independent external estimate.',production_modified=False)
    fp=hashlib.sha256(json.dumps(design,sort_keys=True).encode()).hexdigest()
    if (DEST/'design.json').exists():assert json.loads((DEST/'design.json').read_text())==design
    save(DEST/'design.json',design);print('Prepared 15 held-out peak fits + 1 all-data refit.',flush=True)
    if not args.run:return
    rows=[]
    for ex in sorted(d.example.unique()):
        path=DEST/'checkpoints'/f'{ex}.json'
        if path.exists():
            result=json.loads(path.read_text());assert result['fingerprint']==fp
        else:
            tr=d[d.example.ne(ex)];te=d[d.example.eq(ex)]
            assert set(tr.example).isdisjoint(te.example)
            a=fit(tr,cols,params,args.jobs);pred=b.forest.predict(a,te)
            assert np.isfinite(pred).all()
            result=dict(fingerprint=fp,held=ex,train_examples=sorted(tr.example.unique()),dropped=a['preprocessor'].dropped,
                rows=[dict(example=r.example,state=r.state,predicted_hours=float(v)) for r,v in zip(te.itertuples(),pred)])
            save(path,result)
        rows.extend(result['rows']);print(ex,'saved',flush=True)
    keys=['example','state'];meta=d[keys+['cohort','edit_type',TARGET,'centroid_shift_hours']]
    p=pd.DataFrame(rows).merge(meta,on=keys,how='outer',validate='one_to_one',indicator=True)
    assert p._merge.eq('both').all();p=p.drop(columns='_merge');p['variant']='peak_fixed_settings'
    p.to_parquet(DEST/'predictions.parquet',index=False)
    baseline=pd.read_parquet(baseline_path);comparisons=[p]
    for variant,name in [('fixed_settings','centroid_proxy_fixed_settings'),('nested_tuned','centroid_proxy_nested_tuned')]:
        old=baseline[baseline.variant.eq(variant)]
        z=meta.merge(old[keys+['predicted_hours','centroid_shift_hours']],on=keys,how='outer',validate='one_to_one',indicator=True,suffixes=('','_saved'))
        assert z._merge.eq('both').all()
        np.testing.assert_allclose(z.centroid_shift_hours,z.centroid_shift_hours_saved)
        z=z.drop(columns=['_merge','centroid_shift_hours_saved']);z['variant']=name;comparisons.append(z)
    combined=pd.concat(comparisons,ignore_index=True);combined.to_parquet(DEST/'comparison_predictions.parquet',index=False)
    metrics=[]
    for variant,g in combined.groupby('variant'):
        for subset,z in [('all',g),('without54',g[g.example.ne('54')])]:
            for cohort,a in [('all',z),('original',z[z.cohort.eq('original')]),('segmented',z[z.cohort.eq('segmented')])]:
                mode='equal_cohort_network' if cohort=='all' else 'equal_network'
                metrics.append(dict(variant=variant,subset=subset,cohort=cohort,weighting=mode,**scores(a,mode)))
    s=pd.DataFrame(metrics);s.to_csv(DEST/'scores.csv',index=False)
    network=[]
    for (v,c,ex),z in combined.groupby(['variant','cohort','example']):network.append(dict(variant=v,cohort=c,example=ex,**scores(z,'equal_network')))
    pd.DataFrame(network).to_csv(DEST/'network_scores.csv',index=False)
    p['error_minutes']=60*(p.predicted_hours-p[TARGET]);p['abs_error_minutes']=abs(p.error_minutes)
    for cohort,g in p.groupby('cohort'):
        g=g.copy();g['weighted_squared_error']=weights(g,'equal_network')*g.error_minutes**2
        g.sort_values('weighted_squared_error',ascending=False).head(15).to_csv(DEST/f'largest_errors_{cohort}.csv',index=False)
    a=fit(d,cols,params,args.jobs);a.update(fingerprint=fp,training_rows=len(d),training_examples=sorted(d.example.unique()),versions=design['versions'])
    tmp=DEST/'final_peak_model.joblib.tmp';joblib.dump(a,tmp,compress=3);tmp.replace(DEST/'final_peak_model.joblib')
    restored=joblib.load(DEST/'final_peak_model.joblib')
    np.testing.assert_allclose(b.forest.predict(a,d),b.forest.predict(restored,d),atol=1e-12,rtol=1e-12)
    np.testing.assert_allclose(b.forest.predict(restored,d[cols]),b.forest.predict(restored,d),atol=1e-12,rtol=1e-12)
    save(DEST/'final_peak_model.json',dict(**design,fingerprint=fp,artifact_hash=sha(DEST/'final_peak_model.joblib')))
    for f,h in design['sources'].items():assert sha(f)==h,f
    save(DEST/'completion.json',dict(complete=True,fingerprint=fp,held_out_fits=15,final_fits=1,unmatched=0,production_modified=False))
    print(s.to_string(index=False),flush=True)

if __name__=='__main__':main()
