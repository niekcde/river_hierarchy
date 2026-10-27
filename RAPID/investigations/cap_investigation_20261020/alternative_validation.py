"""Repeated balanced network/compound grouped validation of both saved panels."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import platform

import numpy as np
import pandas as pd
import sklearn
from tune_nine_inputs import b
from slope_ratio_audit import OUT
from ten_input_outlier_audit import score
from augmented_tests import objective

DEST=OUT/'alternative_validation_20261023'
SOURCE=OUT/'resistance_inverse_depth_panels_20261023'
GROUPS=OUT/'validation_groups_full_collapse_singletons_20261023'
SEEDS=[20261024+1009*i for i in range(5)]
STRATEGIES=['network_balanced','compound_mixed']


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(obj,indent=2)+'\n');temp.replace(path)


def signature(assign):
    return tuple(sorted(tuple(sorted(assign.index[assign.eq(k)])) for k in sorted(assign.unique())))


def balanced_groups(data,group_col,k,seed,banned=()):
    """Balance counts and response/type margins only; never inspect model errors.

    32 deterministic randomized greedy restarts. Full-collapse counts are equal
    across folds. Network grouping additionally fixes the number of networks.
    """
    groups=sorted(data[group_col].unique());ng=len(groups)
    assert ng>=k
    tmp=data.copy()
    tmp['balance_response']=tmp.response_bin.replace({'delay_over60min':'delay_5to60min'})
    resp=pd.crosstab(tmp[group_col],tmp.balance_response).reindex(groups,fill_value=0)
    types=pd.crosstab(tmp[group_col],tmp.edit_type).reindex(groups,fill_value=0)
    count=tmp.groupby(group_col).size().reindex(groups).to_numpy(float)
    seg=tmp[tmp.cohort.eq('segmented')].groupby(group_col).size().reindex(groups,fill_value=0).to_numpy(int)
    assert seg.sum()%k==0 and np.max(seg)<=1
    # Equal block weights for sample count, response margins and type margins;
    # low additional weight for group counts. Rare strata do not impose quotas.
    vectors=np.column_stack([count/count.sum(),
        resp.to_numpy()/resp.sum().to_numpy()/np.sqrt(resp.shape[1]),
        types.to_numpy()/types.sum().to_numpy()/np.sqrt(types.shape[1]),
        np.ones(ng)/ng*.25])
    cap=ng//k if group_col=='example' else ng
    if group_col=='example':assert ng%k==0
    quota=int(seg.sum()/k);rng=np.random.default_rng(seed);options=[]
    for _ in range(32):
        jitter=rng.uniform(.5,1.5,ng)
        priority=np.linalg.norm(vectors,axis=1)*jitter
        # Allocate segmented singletons first so all test/inner folds have them.
        order=sorted(range(ng),key=lambda i:(-int(seg[i]>0),-priority[i],groups[i]))
        totals=np.zeros((k,vectors.shape[1]));n=np.zeros(k,int);ns=np.zeros(k,int);assignment=np.full(ng,-1,int)
        for i in order:
            eligible=np.flatnonzero((n<cap)&(ns+seg[i]<=quota))
            assert len(eligible)>0
            costs=np.sum((totals[eligible]+vectors[i])**2-totals[eligible]**2,axis=1)
            bests=eligible[np.isclose(costs,costs.min(),rtol=0,atol=1e-14)]
            fold=int(rng.choice(bests));assignment[i]=fold
            totals[fold]+=vectors[i];n[fold]+=1;ns[fold]+=seg[i]
        assert (n>0).all() and (ns==quota).all()
        a=pd.Series(assignment,index=groups)
        sig=signature(a)
        if sig in banned:continue
        loss=float(np.sum((totals-1/k*np.sum(vectors,axis=0))**2))
        options.append((loss,a))
    assert options,'No distinct predefined-restart partition available'
    loss,a=min(options,key=lambda x:x[0])
    result=data[group_col].map(a).to_numpy(int)
    for fold in range(k):
        train=data[result!=fold];test=data[result==fold]
        assert set(train[group_col]).isdisjoint(test[group_col])
        assert set(test.cohort)=={'original','segmented'}
    return result,dict(balance_loss=loss,groups=ng,group_col=group_col,seed=seed,
        fold_groups={str(i):sorted(a.index[a.eq(i)]) for i in range(k)}),signature(a)


def prepare():
    DEST.mkdir(exist_ok=True)
    d=pd.read_parquet(SOURCE/'input.parquet')
    g=pd.read_parquet(GROUPS/'records.parquet')
    compare=d.merge(g[['example','state','cohort','centroid_shift_hours']],on=['example','state'],validate='one_to_one',suffixes=('_d','_g'))
    assert len(compare)==len(d)==len(g)==2185
    assert (compare.cohort_d==compare.cohort_g).all()
    np.testing.assert_array_equal(compare.centroid_shift_hours_d,compare.centroid_shift_hours_g)
    d=d.merge(g[['example','state','group_id','edit_type','response_bin']],on=['example','state'],validate='one_to_one').reset_index(drop=True)
    settings_path=OUT/'current_model_comparison_20261023/settings.json'
    settings=json.loads(settings_path.read_text())
    panels={model:v['columns'] for model,v in settings.items()}
    grid_path=OUT/'nine_input_tuning/design.json'
    candidates=json.loads(grid_path.read_text())['candidates'];assert len(candidates)==6
    assert d.group_id.nunique()==450 and np.isfinite(d[list(set(sum(panels.values(),[])))]).all().all()
    files=[SOURCE/'input.parquet',GROUPS/'records.parquet',GROUPS/'manifest.json',settings_path,grid_path,
        OUT/'nine_input_tuning/predictions.parquet',OUT/'ten_input_tuning_20261023/predictions.parquet',Path(__file__),
        Path(__file__).with_name('ten_input_outlier_audit.py'),Path(__file__).with_name('augmented_tests.py'),
        Path(b.__file__),Path(b.engine.__file__),Path(b.forest.__file__)]
    design=dict(sources={str(p):sha(p) for p in files},panels=panels,candidates=candidates,seeds=SEEDS,
        versions=dict(python=platform.python_version(),numpy=np.__version__,pandas=pd.__version__,sklearn=sklearn.__version__),
        estimator_seed=b.engine.SEED,rows=len(d),fits=1900,
        validation='2 strategies x 5 repetitions x 5 outer folds x 2 models. Six-setting tuning with 3 inner grouped folds following the same grouping policy as the outer split. Each repetition produces one held-out prediction per row.',
        balance='Fixed response/type margins and sample count; 32 seeded greedy restarts selected ONLY by margin balance, not errors. Merge >60min delays with >5min delays for assignment; retain detailed bins for reporting. Three full collapses per outer test fold, four per inner validation fold. Exactly three test networks for network strategy.',
        group_policy='network_balanced: whole examples, including full collapse. compound_mixed: 435 regular family groups plus 15 singleton full collapses; explicit related-state overlap exception.',
        training='Same network-balanced training weights and training-only preprocessing/target standardization as previous runs. Full centroid target, 0.5% window. No added combinations or predictors.',
        objective='Equal average of regular and segmented one-minute-floored, network-balanced relative errors. Same six candidates for both models.',
        reporting='Compute scores separately for each complete repetition, then mean/min/max across repeats. Do not ensemble predictions before primary scoring. Prior LOEO geometry same_grid used four candidates; note differing search budget.',
        limits='Known-network related-state prediction is not unseen-network transfer. Repetitions reuse data and their spread is not a confidence interval. Exploratory feature-selection history remains.',production_modified=False)
    fp=hashlib.sha256(json.dumps(design,sort_keys=True).encode()).hexdigest()
    if (DEST/'design.json').exists():assert json.loads((DEST/'design.json').read_text())==design,'Design changed'
    save(DEST/'design.json',design)
    if (DEST/'splits.json').exists():
        saved=json.loads((DEST/'splits.json').read_text());assert saved['fingerprint']==fp
        return d,design,fp,saved['tasks']
    tasks=[];audit=[];assignments=[]
    for strategy in STRATEGIES:
        group_col='example' if strategy=='network_balanced' else 'group_id'
        signatures=[]
        for repeat,seed in enumerate(SEEDS):
            fold,info,sig=balanced_groups(d,group_col,5,seed,signatures);signatures.append(sig)
            for i,r in enumerate(d.itertuples()):
                assignments.append(dict(strategy=strategy,repeat=repeat,example=r.example,state=r.state,
                    group_id=r.group_id,cohort=r.cohort,edit_type=r.edit_type,response_bin=r.response_bin,fold=int(fold[i])))
            for outer in range(5):
                train_idx=np.flatnonzero(fold!=outer);test_idx=np.flatnonzero(fold==outer)
                tr=d.iloc[train_idx].reset_index(drop=True);te=d.iloc[test_idx]
                inner,inner_info,_=balanced_groups(tr,group_col,3,seed+100000+outer)
                assert set(tr[group_col]).isdisjoint(te[group_col])
                tasks.append(dict(strategy=strategy,repeat=repeat,fold=outer,train_indices=train_idx.tolist(),
                    test_indices=test_idx.tolist(),inner_fold=inner.tolist(),outer_balance=info,inner_balance=inner_info))
                for cohort,z in te.groupby('cohort'):
                    audit.append(dict(strategy=strategy,repeat=repeat,fold=outer,cohort=cohort,n=len(z),
                        test_networks=z.example.nunique(),test_groups=z.group_id.nunique(),
                        same_network_train_cases=int(z.example.isin(tr.example).sum()),
                        train_cases=len(tr),train_networks=tr.example.nunique(),outer_balance_loss=info['balance_loss']))
        assert len(set(signatures))==5
    save(DEST/'splits.json',dict(fingerprint=fp,tasks=tasks))
    pd.DataFrame(assignments).to_csv(DEST/'outer_assignments.csv',index=False)
    pd.DataFrame(audit).to_csv(DEST/'fold_audit.csv',index=False)
    pd.DataFrame(assignments).groupby(['strategy','repeat','fold','edit_type','response_bin']).size().rename('n').to_csv(DEST/'fold_mixes.csv')
    d.to_parquet(DEST/'input.parquet',index=False)
    return d,design,fp,tasks


def fit_task(task,model_name,cols,candidates,fp):
    d=pd.read_parquet(DEST/'input.parquet')
    key=f"{task['strategy']}/r{task['repeat']}/f{task['fold']}/{model_name}"
    folder=DEST/'checkpoints'/key
    done=folder/'outer.json'
    if done.exists():
        result=json.loads(done.read_text());assert result['fingerprint']==fp;return key
    b.COLUMNS=list(cols)
    tr=d.iloc[task['train_indices']].reset_index(drop=True);te=d.iloc[task['test_indices']]
    inner=np.asarray(task['inner_fold']);ranked=[]
    group_col='example' if task['strategy']=='network_balanced' else 'group_id'
    assert set(tr[group_col]).isdisjoint(te[group_col])
    for index,candidate in enumerate(candidates):
        path=folder/f'candidate_{index}.json'
        if path.exists():
            trial=json.loads(path.read_text());assert trial['fingerprint']==fp
        else:
            predicted=np.full(len(tr),np.nan)
            for fold in range(3):
                training=tr[inner!=fold];validation=tr[inner==fold]
                assert set(training[group_col]).isdisjoint(validation[group_col])
                artifact=b.fit(training,candidate['params'],jobs=1)
                predicted[inner==fold]=b.forest.predict(artifact,validation)
            assert np.isfinite(predicted).all()
            loss,by=objective(tr,predicted)
            trial=dict(fingerprint=fp,index=index,name=candidate['name'],params=candidate['params'],score=loss,cohort_scores=by)
            save(path,trial)
        ranked.append(trial)
    best=min(ranked,key=lambda x:(x['score'],x['index']))
    artifact=b.fit(tr,best['params'],jobs=1);predicted=b.forest.predict(artifact,te)
    assert np.isfinite(predicted).all()
    rows=[dict(strategy=task['strategy'],repeat=task['repeat'],fold=task['fold'],model=model_name,
        example=r.example,state=r.state,cohort=r.cohort,edit_type=r.edit_type,centroid_shift_hours=r.centroid_shift_hours,
        predicted_hours=float(v),same_network_in_training=r.example in set(tr.example)) for r,v in zip(te.itertuples(),predicted)]
    save(done,dict(fingerprint=fp,best=best,rows=rows));return key


def summarize(fp):
    rows=[];selections=[]
    for path in sorted((DEST/'checkpoints').glob('*/*/*/*/outer.json')):
        result=json.loads(path.read_text());assert result['fingerprint']==fp
        rows.extend(result['rows']);r=result['rows'][0]
        selections.append(dict(strategy=r['strategy'],repeat=r['repeat'],fold=r['fold'],model=r['model'],**result['best']))
    assert len(selections)==100
    p=pd.DataFrame(rows);assert len(p)==2*5*2*2185
    assert not p.duplicated(['strategy','repeat','model','example','state']).any()
    assert p.groupby(['strategy','repeat','model']).size().eq(2185).all()
    p.to_parquet(DEST/'predictions.parquet',index=False);save(DEST/'selections.json',selections)
    metrics=[];types=[];networks=[]
    for (strategy,repeat,model),g in p.groupby(['strategy','repeat','model']):
        for subset,z in [('all',g),('without54',g[g.example.ne('54')])]:
            for cohort,a in [('all',z),('original',z[z.cohort.eq('original')]),('segmented',z[z.cohort.eq('segmented')])]:
                mode='equal_cohort_network' if cohort=='all' else 'equal_network'
                metrics.append(dict(strategy=strategy,repeat=repeat,model=model,cohort=cohort,subset=subset,**score(a,mode)))
        for kind,z in g.groupby('edit_type'):
            types.append(dict(strategy=strategy,repeat=repeat,model=model,edit_type=kind,**score(z,'equal_network')))
        for (cohort,ex),z in g.groupby(['cohort','example']):
            err=z.predicted_hours-z.centroid_shift_hours
            networks.append(dict(strategy=strategy,repeat=repeat,model=model,cohort=cohort,example=ex,
                mae_minutes=60*abs(err).mean(),relative_error_pct=100*np.mean(abs(err)/np.maximum(abs(z.centroid_shift_hours),1/60))))
    s=pd.DataFrame(metrics);s.to_csv(DEST/'repeat_scores.csv',index=False)
    pd.DataFrame(types).to_csv(DEST/'type_scores.csv',index=False);pd.DataFrame(networks).to_csv(DEST/'network_scores.csv',index=False)
    agg=s.groupby(['strategy','model','cohort','subset'])[['r2','mae_minutes','relative_error_pct']].agg(['mean','min','max','std'])
    agg.columns=['_'.join(c) for c in agg.columns];agg.reset_index().to_csv(DEST/'summary.csv',index=False)
    baselines=[]
    for model,folder,variant in [('geometry_9','nine_input_tuning','same_grid'),('depth_informed_10','ten_input_tuning_20261023','nested_tuned')]:
        old=pd.read_parquet(OUT/folder/'predictions.parquet');old=old[old.variant.eq(variant)]
        for cohort,z in [('all',old),('original',old[old.cohort.eq('original')]),('segmented',old[old.cohort.eq('segmented')])]:
            baselines.append(dict(strategy='leave_one_network_out',model=model,cohort=cohort,**score(z,'equal_cohort_network' if cohort=='all' else 'equal_network')))
    pd.DataFrame(baselines).to_csv(DEST/'baseline_scores.csv',index=False)
    print(agg.reset_index().query("subset=='all'").to_string(index=False),flush=True)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',action='store_true');ap.add_argument('--workers',type=int,default=4)
    args=ap.parse_args();d,design,fp,tasks=prepare()
    print('Prepared 100 model/fold jobs, 1900 forest fits; five distinct repetitions per strategy.',flush=True)
    if not args.run:return
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures=[pool.submit(fit_task,t,m,cols,design['candidates'],fp) for t in tasks for m,cols in design['panels'].items()]
        for i,f in enumerate(as_completed(futures),1):print(i,'/100 complete:',f.result(),flush=True)
    summarize(fp)
    for path,h in design['sources'].items():assert sha(path)==h,path
    save(DEST/'completion.json',dict(complete=True,fingerprint=fp,fits=1900,unmatched=0,production_modified=False))


if __name__=='__main__':main()
