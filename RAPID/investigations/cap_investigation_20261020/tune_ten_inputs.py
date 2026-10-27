"""Nested grouped tuning of the ten-input resistance + inverse-depth forest."""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import joblib
import numpy as np
import pandas as pd
import sklearn
import tune_nine_inputs as tuning
import augmented_tests
from ten_input_outlier_audit import score

b=tuning.b
SOURCE=tuning.OUT/'resistance_inverse_depth_panels_20261023'
DEST=tuning.OUT/'ten_input_tuning_20261023'
PANEL='5_replace_width_with_Rh_plus_Gh'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def summarize(p):
    rows=[]
    for variant,g in p.groupby('variant'):
        for subset,z in [('all',g),('without54',g[g.example.ne('54')])]:
            for cohort,a in [('all',z),('original',z[z.cohort.eq('original')]),('segmented',z[z.cohort.eq('segmented')])]:
                modes=['equal_network','equal_case','equal_cohort_network'] if cohort=='all' else ['equal_network']
                for mode in modes:
                    rows.append(dict(variant=variant,cohort=cohort,subset=subset,weighting=mode,**score(a,mode)))
    s=pd.DataFrame(rows)
    s.to_csv(DEST/'scores.csv',index=False)
    s[(s.cohort=='all') & (s.weighting=='equal_cohort_network')][['variant','subset','relative_error_pct']].to_csv(DEST/'objective.csv',index=False)
    per=[]
    for (variant,cohort,ex),g in p.groupby(['variant','cohort','example']):
        err=g.predicted_hours-g.centroid_shift_hours
        per.append(dict(variant=variant,cohort=cohort,example=ex,
            mae_minutes=60*abs(err).mean(),relative_error_pct=100*np.mean(abs(err)/np.maximum(abs(g.centroid_shift_hours),1/60))))
    pd.DataFrame(per).to_csv(DEST/'network_scores.csv',index=False)
    print(s[(s.subset=='all') & ((s.cohort!='all') | (s.weighting=='equal_cohort_network'))].to_string(index=False),flush=True)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--run',action='store_true')
    ap.add_argument('--jobs',type=int,default=4)
    args=ap.parse_args()
    DEST.mkdir(exist_ok=True)
    source_design=json.loads((SOURCE/'design.json').read_text())
    assert json.loads((SOURCE/'completion.json').read_text())['complete']
    cols=list(source_design['panels'][PANEL]);assert len(cols)==10
    grid_path=tuning.OUT/'nine_input_tuning/design.json'
    candidates=json.loads(grid_path.read_text())['candidates'];assert len(candidates)==6
    d=pd.read_parquet(SOURCE/'input.parquet')
    assert digest(SOURCE/'input.parquet')==source_design['input_hash']
    assert len(d)==2185 and d.example.nunique()==15 and not d.duplicated(['example','state']).any()
    assert np.isfinite(d[cols+['centroid_shift_hours']]).all().all()
    assert d[d.cohort.eq('segmented')].groupby('example').size().eq(1).all()
    files=[SOURCE/x for x in ['input.parquet','design.json','completion.json','predictions.parquet']]
    files += [grid_path,Path(__file__),Path(tuning.__file__),Path(augmented_tests.__file__),
        Path(__file__).with_name('ten_input_outlier_audit.py'),Path(b.__file__),Path(b.forest.__file__),Path(b.engine.__file__)]
    versions=dict(python=platform.python_version(),numpy=np.__version__,pandas=pd.__version__,sklearn=sklearn.__version__,joblib=joblib.__version__)
    design=dict(columns=cols,candidates=candidates,seed=b.engine.SEED,target='centroid_shift_hours',primary_window='0.5%',
        sources={str(x.resolve()):digest(x) for x in files},versions=versions,rows=len(d),
        validation='15 outer whole-example holdouts. Each outer-training set tunes six settings with three GroupKFold folds, never including the held example. Select lowest inner objective, index as fixed tie-break. Fresh outer fit with selected settings.',
        objective='Equal average of regular and segmented one-minute-floored relative errors, equal total network weight within each cohort. Lower is better.',
        training='Both cohorts, equal total weight per network across states. Training-only preprocessing and target standardization. Fixed ten predictors, no augmentation.',
        final='Six-setting three-fold grouped selection using all 15 examples, then fit all 2185 cases. Report outer held-out performance, never final-selection or training scores as test performance.',
        fit_count=304,caveats=['Exploratory reused examples; not independent external validation.','Only 15 independent segmented cases.','Hyperparameter search limited to the existing six configurations.','Predictors were chosen through prior exploratory comparisons on these networks.'],production_modified=False)
    fp=hashlib.sha256(json.dumps(design,sort_keys=True).encode()).hexdigest()
    if (DEST/'design.json').exists():assert json.loads((DEST/'design.json').read_text())==design,'Design changed; use a new output directory'
    tuning.save(DEST/'design.json',design)
    print('Prepared 304 fits: 270 inner, 15 outer, 18 final-selection, one all-data forest.',flush=True)
    if not args.run:return
    # Reuse the tested grouped tuning implementation with this exact schema/output.
    tuning.DEST=DEST;tuning.COLUMNS=list(cols)
    rows,selections=[],[]
    for ex in sorted(d.example.unique()):
        tr=d[d.example.ne(ex)].reset_index(drop=True);te=d[d.example.eq(ex)]
        assert tr.example.nunique()==14 and ex not in set(tr.example)
        choice=tuning.tune(tr,candidates,ex,fp,args.jobs)[0]
        for inner in choice['inner_groups']:
            assert ex not in inner['train']+inner['validation']
            assert set(inner['train']).isdisjoint(inner['validation'])
        path=DEST/'outer'/f'{ex}.json'
        saved=b.forest.checked(path,fp)
        if saved is None:
            model=tuning.fit(tr,choice['params'],args.jobs)
            pred=b.forest.predict(model,te);assert np.isfinite(pred).all()
            saved=dict(fingerprint=fp,rows=[dict(example=r.example,state=r.state,cohort=r.cohort,
                centroid_shift_hours=r.centroid_shift_hours,predicted_hours=float(v),variant='nested_tuned')
                for r,v in zip(te.itertuples(),pred)])
            tuning.save(path,saved)
        rows.extend(saved['rows']);selections.append(dict(held=ex,**choice))
        tuning.save(DEST/'selections.json',selections)
        print('Held-out',ex,'complete:',choice['name'],flush=True)
    new=pd.DataFrame(rows);assert len(new)==2185 and not new.duplicated(['example','state']).any()
    old=pd.read_parquet(SOURCE/'predictions.parquet');old=old[old.panel.eq(PANEL)].copy();old['variant']='fixed_settings'
    paired=new.merge(old,on=['example','state'],validate='one_to_one',suffixes=('_new','_old'))
    assert len(paired)==2185 and (paired.cohort_new==paired.cohort_old).all()
    np.testing.assert_array_equal(paired.centroid_shift_hours_new,paired.centroid_shift_hours_old)
    p=pd.concat([new,old[new.columns]],ignore_index=True)
    p.to_parquet(DEST/'predictions.parquet',index=False);summarize(p)
    best=tuning.tune(d.reset_index(drop=True),candidates,'all_data',fp,args.jobs)[0]
    model=tuning.fit(d,best['params'],args.jobs)
    model.update(fingerprint=fp,versions=versions,training_rows=len(d),training_examples=sorted(d.example.unique()))
    path=DEST/'final_model.joblib';tmp=DEST/'final_model.joblib.tmp'
    joblib.dump(model,tmp,compress=3);tmp.replace(path)
    restored=joblib.load(path)
    np.testing.assert_allclose(b.forest.predict(model,d),b.forest.predict(restored,d),rtol=1e-12,atol=1e-12)
    np.testing.assert_allclose(b.forest.predict(restored,d[cols]),b.forest.predict(restored,d),rtol=1e-12,atol=1e-12)
    tuning.save(DEST/'final_model.json',dict(selected=best,columns=cols,versions=versions,sha256=digest(path),
        target='centroid_shift_hours',prediction_units='hours',training_rows=len(d),training_examples=sorted(d.example.unique()),
        warning='Separate ten-input candidate artifact. Use nested held-out scores for performance, not all-data selection score. No production replacement.'))
    for path,h in design['sources'].items():assert digest(path)==h,path
    tuning.save(DEST/'completion.json',dict(complete=True,fingerprint=fp,fits=304,unmatched=0,
        serialized_predictions_verified=True,production_modified=False))
    print('Completed nested evaluation and saved final ten-input model:',DEST/'final_model.joblib',flush=True)


if __name__=='__main__':main()
