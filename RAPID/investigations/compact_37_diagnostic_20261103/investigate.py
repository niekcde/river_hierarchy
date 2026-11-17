"""Post-hoc diagnosis of example-37 compact RF errors, not model selection."""
import os
for name in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:os.environ[name]='1'
from pathlib import Path
import sys,json,hashlib,time
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'RAPID/investigations/streamlined_predictors_20261103'))
import run_comparison as experiment
OUT=ROOT/'RAPID/outputs/compact_37_diagnostic_20261103';OUT.mkdir(exist_ok=True)

def donor_weights(artifact,train,test):
    """Exact RF prediction weights including bootstrap multiplicity and sample weights."""
    pre=artifact['preprocessor'];rf=artifact['estimator']
    xt=pre.transform(train).astype(np.float32);xv=pre.transform(test).astype(np.float32)
    weights=experiment.legacy.refit.tuning.b.engine.weights(train)
    result=np.zeros((len(test),len(train)))
    for tree,sample in zip(rf.estimators_,rf.estimators_samples_):
        w=weights*np.bincount(sample,minlength=len(train))
        lt=tree.apply(xt);lv=tree.apply(xv)
        for leaf in np.unique(lv):
            a=np.flatnonzero(lv==leaf);b=np.flatnonzero((lt==leaf)&(w>0))
            result[np.ix_(a,b)]+=w[b]/w[b].sum()/len(rf.estimators_)
    np.testing.assert_allclose(result.sum(axis=1),1,atol=1e-12)
    return result

def main():
    started=time.monotonic()
    source=experiment.OUT;design=json.loads((source/'design.json').read_text())
    data=pd.read_parquet(source/'real_input.parquet')
    predictions=pd.read_parquet(source/'predictions.parquet')
    tasks=json.loads((source/'tasks.json').read_text())['tasks']
    task=next(t for t in tasks if t['dataset']=='real' and t['strategy']=='leave_one_network_out' and
              t['model']=='five_with_G' and data.iloc[t['test_indices']].example.eq('37').all())
    tr=data.iloc[task['train_indices']].reset_index(drop=True)
    te=data.iloc[task['test_indices']].reset_index(drop=True)
    assert set(tr.example).isdisjoint(te.example) and te.example.eq('37').all()
    base=design['panels']['five_with_G'];width='log1p__path_parent_equivalent_width'
    panels={m:design['panels'][m] for m in ['five_with_G','five_without_G','revised9']}
    panels.update(drop_equivalent_width=[c for c in base if c!=width],
        add_edited_domain_size=base+['domain_size_m'],
        replace_width_with_domain_size=[c for c in base if c!=width]+['domain_size_m'],
        exclude_full_network_training=base)
    # Freeze probes before their scores; all are diagnosis on an already inspected river.
    experiment.save(OUT/'design.json',dict(reference='COMPACT-37-DIAGNOSTIC-2026-11-03',panels=panels,
        params=design['params'],held_example='37',baseline_source=str(source),
        caveat='Post-hoc one-river diagnostic. This does not estimate independently selected model performance or justify replacing the full-population benchmark.',
        donor_method='Exact leaf means, including bootstrap multiplicities, training weights and inverse target scaling; verify against saved held-out predictions.',
        sources={str(source/f):experiment.sha(source/f) for f in ['real_input.parquet','design.json','tasks.json','predictions.parquet']},
        script_sha256=experiment.sha(__file__)))
    allpred=[];scores=[];donors=[];donor_cohorts=[];parity=[];importance=[]
    chosen=['S315_unit_274','S256_unit_229','island_37_275','CB001_bubble_3','all_closed_units_collapsed']
    for model,columns in panels.items():
        train=tr[tr.cohort.eq('original')].reset_index(drop=True) if model=='exclude_full_network_training' else tr
        b=experiment.legacy.refit.tuning.b;b.COLUMNS=columns
        artifact=b.fit(train,design['params'],jobs=1)
        pred=b.forest.predict(artifact,te)
        row=te[['example','state','cohort','edit_type','group_id','centroid_shift_hours','domain_size_m',width]].copy()
        row['model']=model;row['predicted_hours']=pred;row['error_minutes']=60*(pred-te.centroid_shift_hours.to_numpy())
        allpred.append(row)
        if model in ['five_with_G','five_without_G','revised9']:
            saved=predictions[(predictions.dataset=='real')&(predictions.strategy=='leave_one_network_out')&
                              predictions.model.eq(model)&predictions.example.eq('37')].set_index('state').loc[te.state]
            np.testing.assert_allclose(pred,saved.predicted_hours,rtol=1e-12,atol=1e-12)
            parity.append(dict(model=model,max_error_hours=float(abs(pred-saved.predicted_hours.to_numpy()).max())))
            weights=donor_weights(artifact,train,te)
            reconstruction=weights@train.centroid_shift_hours.to_numpy()
            np.testing.assert_allclose(reconstruction,pred,rtol=1e-10,atol=1e-11)
            parity[-1]['donor_reconstruction_max_error_hours']=float(abs(reconstruction-pred).max())
            for cohort in ['original','segmented']:
                mask=train.cohort.eq(cohort).to_numpy()
                mass=weights[:,mask].sum(axis=1)
                contribution=60*(weights[:,mask]@train.loc[mask,'centroid_shift_hours'].to_numpy())
                donor_cohorts.extend(dict(model=model,state=state,training_cohort=cohort,prediction_weight=float(w),contribution_minutes=float(c)) for state,w,c in zip(te.state,mass,contribution))
            for i,state in enumerate(te.state):
                if state not in chosen:continue
                indices=np.argsort(-weights[i])[:15]
                for j in indices:
                    t=train.iloc[j]
                    donors.append(dict(model=model,test_state=state,donor_example=t.example,donor_state=t.state,donor_cohort=t.cohort,
                        donor_shift_minutes=60*t.centroid_shift_hours,weight=weights[i,j],contribution_minutes=60*weights[i,j]*t.centroid_shift_hours))
        for c,v in zip(artifact['preprocessor'].kept,artifact['estimator'].feature_importances_):
            importance.append(dict(model=model,variable=c,impurity_importance=v))
        for population,mask in [('ordinary',te.cohort.eq('original')),('shared_large_parent',te[width].eq(63.156919) if False else te.group_id.eq('37_9485f774f602')),('full_network',te.cohort.eq('segmented'))]:
            y=te.loc[mask,'centroid_shift_hours'].to_numpy()*60;e=pred[mask]*60-y
            sst=((y-y.mean())**2).sum()
            scores.append(dict(model=model,population=population,n=int(mask.sum()),r2=1-(e**2).sum()/sst if sst>1e-20 else None,
                mae_minutes=np.mean(abs(e)),rmse_minutes=np.sqrt(np.mean(e**2)),mean_bias_minutes=e.mean()))
        print(model,'finished',flush=True)
    pd.concat(allpred,ignore_index=True).to_parquet(OUT/'diagnostic_predictions.parquet',index=False)
    pd.DataFrame(scores).to_csv(OUT/'diagnostic_scores.csv',index=False)
    pd.DataFrame(donors).to_csv(OUT/'top_training_donors.csv',index=False)
    pd.DataFrame(donor_cohorts).to_csv(OUT/'training_cohort_contributions.csv',index=False)
    pd.DataFrame(parity).to_csv(OUT/'reproduction_checks.csv',index=False)
    pd.DataFrame(importance).to_csv(OUT/'impurity_importance_diagnostic.csv',index=False)
    # Full saved validation results establish breadth and fold sensitivity.
    z=predictions[(predictions.dataset=='real')&predictions.example.eq('37')].copy()
    z['absolute_error_minutes']=60*abs(z.predicted_hours-z.centroid_shift_hours)
    z['squared_error_minutes']=(60*(z.predicted_hours-z.centroid_shift_hours))**2
    z.groupby(['strategy','model','state','cohort','edit_type']).agg(actual_hours=('centroid_shift_hours','first'),
        predicted_hours=('predicted_hours','mean'),prediction_min=('predicted_hours','min'),prediction_max=('predicted_hours','max'),
        mae_minutes=('absolute_error_minutes','mean'),mse_minutes=('squared_error_minutes','mean')).reset_index().to_csv(OUT/'state_errors_all_strategies.csv',index=False)
    ranges=[]
    for c in sorted(set(sum(panels.values(),[]))):
        for cohort in ['original','segmented']:
            a=tr.loc[tr.cohort.eq(cohort),c]
            ranges.append(dict(variable=c,training_cohort=cohort,train_min=a.min(),train_max=a.max(),test_min=te[c].min(),test_max=te[c].max(),
                               n_test_above_training_max=int((te[c]>a.max()).sum()),n_test_below_training_min=int((te[c]<a.min()).sum())))
    pd.DataFrame(ranges).to_csv(OUT/'training_support.csv',index=False)
    te.to_parquet(OUT/'example37_features.parquet',index=False)
    experiment.save(OUT/'fit_completion.json',dict(complete=True,seconds=time.monotonic()-started,diagnostic_fits=len(panels),production_modified=False))
    print(pd.DataFrame(scores).to_string(index=False))

if __name__=='__main__':main()
