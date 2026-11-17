"""Fixed-depth, fixed-RF reconciliation of synthetic and real predictor sets."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='1'
from pathlib import Path
import sys,json,hashlib,time,fcntl,argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'RAPID/investigations/matched_centroid_sets_20261103'))
import compare as previous
legacy=previous.legacy
OUT=ROOT/'RAPID/outputs/streamlined_predictors_20261103'
PRIOR=ROOT/'RAPID/outputs/matched_centroid_all_strategies_20261103'
SYN=Path('/Volumes/SyntheticResults20261027/analysis/broad_descriptors_20261030/geometry_with_manning/features.parquet')
FOLDS=ROOT/'synthetic_experiments/investigations/forward_selection_20261102/results/case_folds.csv'
OLD9=['log1p__B_ratio_route_length_mean','domain_size_m','delta_spread_normalized',
      'log1p__B_ratio_width_mean','log1p__B_ratio_width_cv','delta_entropy',
      'B_parent_dominance_mean','parent_slope_slow','B_delta_route_length_mean']
REVISED=[('B_delta_width_cv' if x=='log1p__B_ratio_width_cv' else x) for x in OLD9]
COMMON=['B_delta_route_length_mean','parent_slope_slow','log1p__path_parent_equivalent_width',
        'path_delta_topologic_complexity_score']
ALLOCATION=['delta_entropy','B_parent_dominance_mean']
PANELS={'old9':OLD9,'revised9':REVISED,
        'five_with_G':COMMON+['delta_G_manning'],'five_without_G':COMMON+['B_delta_area_proxy'],
        'seven_with_G':COMMON+['delta_G_manning']+ALLOCATION,
        'seven_without_G':COMMON+['B_delta_area_proxy']+ALLOCATION,
        'revised_without_allocation7':[x for x in REVISED if x not in ALLOCATION],
        'revised_without_spread8':[x for x in REVISED if x!='delta_spread_normalized']}
CV_ZERO_TOL=1e-12

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(path,value):legacy.refit.save(Path(path),value)

def prepare():
    OUT.mkdir(exist_ok=True)
    if (OUT/'design.json').exists():
        d=json.loads((OUT/'design.json').read_text())
        for p,h in d['sources'].items():assert sha(p)==h,p
        for p,h in d['inputs'].items():assert sha(p)==h,p
        fp=hashlib.sha256(json.dumps(d,sort_keys=True).encode()).hexdigest()
        return d,fp
    sources={str(Path(__file__)):sha(__file__),str(SYN):sha(SYN),str(FOLDS):sha(FOLDS),
             str(PRIOR/'input.parquet'):sha(PRIOR/'input.parquet'),
             str(PRIOR/'splits.json'):sha(PRIOR/'splits.json')}
    # Geometry is identical across hydraulic methods; use ONLY fixed Manning responses.
    s=pd.read_parquet(SYN);s=s[s.method.eq('fixed_manning')].copy().reset_index(drop=True)
    assert len(s)==45927 and s.groupby('case').size().eq(7).all()
    s['domain_size_m']=s.domain_size*1000
    s['delta_spread_normalized']=s.B_delta_route_length_sd/s.B_parent_route_length_mean
    s['delta_entropy']=s.B_delta_route_entropy
    for metric in ['route_length_mean','width_mean']:
        s['log1p__B_ratio_'+metric]=np.log1p(s['B_ratio_'+metric])
    s['B_delta_width_cv']=s.B_child_width_cv-s.B_parent_width_cv
    # A zero CV is meaningful; only the ratio with a zero denominator is undefined.
    valid=s.B_parent_width_cv>CV_ZERO_TOL
    s['log1p__B_ratio_width_cv']=np.nan
    s.loc[valid,'log1p__B_ratio_width_cv']=np.log1p(s.loc[valid,'B_child_width_cv']/s.loc[valid,'B_parent_width_cv'])
    s['log1p__path_parent_equivalent_width']=np.log1p(s.path_parent_equivalent_width)
    s['centroid_shift_hours']=s.centroid_shift_minutes/60
    s['example']=s.case;s['state']=s.child_state;s['group_id']=s.case
    s['cohort']='synthetic';s['edit_type']=s.child_state
    folds=pd.read_csv(FOLDS).set_index('case')
    s['outer_fold']=s.case.map(folds.outer_fold);assert s.outer_fold.notna().all()
    assert np.isfinite(s[list(set(sum([v for k,v in PANELS.items() if k!='old9'],[])))]).all().all()
    s['old9_defined']=np.isfinite(s[OLD9]).all(axis=1)
    coverage=s.groupby('child_state').agg(n=('case','size'),old9_defined=('old9_defined','sum'))
    coverage['undefined_old9']=coverage.n-coverage.old9_defined
    coverage.to_csv(OUT/'synthetic_coverage.csv')
    # Derive real CV differences from verified child/parent ratios and recomputed
    # parent width moments in the historical selected domain.
    r=pd.read_parquet(PRIOR/'input.parquet')
    records_path=legacy.refit.BASE/'validation_group_inventory_20261023/records.parquet'
    records=pd.read_parquet(records_path).set_index(['example','state'])
    sources[str(records_path)]=sha(records_path)
    r['domain_size_m']=r.domain_size*1000
    cache={};audit=[]
    for ex,z in r.groupby('example'):
        parent=legacy.refit.active(ex,'S000_base')
        for i,row in z.iterrows():
            record=records.loc[(ex,row.state)]
            if row.cohort=='segmented':
                path=legacy.refit.EXT/f'variants/{ex}/{row.state}/lineage.csv'
                lineage=pd.read_csv(path);sources[str(path)]=sha(path)
                ids=tuple(sorted(set(lineage.loc[lineage.role.eq('collapsed_block'),'parent_link_id'].astype(int))))
                f=parent
            else:
                ids=tuple(sorted(map(int,record.descriptor_links)))
                f=parent[parent.id_link.isin(ids)].copy()
            key=(ex,ids)
            if key not in cache:cache[key]=legacy.refit.describe(f,set(ids))
            a=cache[key];cv=a['width_cv'];assert cv>0
            ratio=np.expm1(row['log1p__B_ratio_width_cv'])
            r.at[i,'B_delta_width_cv']=cv*(ratio-1)
            np.testing.assert_allclose(a['dominance_mean'],row.B_parent_dominance_mean,rtol=1e-10)
            # Check directly against child geometry for a local case and each full control.
            if i==z.index[0] or row.cohort=='segmented':
                child=legacy.refit.active(ex,row.state)
                if row.cohort=='segmented':
                    cids=set(lineage.loc[lineage.role.eq('collapsed_block'),'child_link_id'].astype(int));cf=child
                else:
                    u,v=legacy.refit.terminal(f)
                    cf=legacy.refit.extract_complete_directed_domain(child,int(u),int(v));cids=set(cf.id_link)
                b=legacy.refit.describe(cf,cids)
                direct=b['width_cv']-cv
                np.testing.assert_allclose(direct,r.at[i,'B_delta_width_cv'],atol=1e-12,rtol=1e-10)
                audit.append(dict(example=ex,state=row.state,direct_delta_cv=direct,reconstructed_delta_cv=r.at[i,'B_delta_width_cv']))
        print('Prepared real width-CV differences',ex,flush=True)
    sources.update(legacy.refit.HASHES)
    assert np.isfinite(r[list(set(sum(PANELS.values(),[])))+['centroid_shift_hours']]).all().all()
    pd.DataFrame(audit).to_csv(OUT/'width_cv_reconstruction_checks.csv',index=False)
    s.to_parquet(OUT/'synthetic_input.parquet',index=False);r.to_parquet(OUT/'real_input.parquet',index=False)
    tasks=[]
    for fold in range(3):
        for support,models in [('full',[m for m in PANELS if m!='old9']),('common',['old9','revised9'])]:
            mask=np.ones(len(s),bool) if support=='full' else s.old9_defined.to_numpy()
            tr=np.flatnonzero(mask&s.outer_fold.ne(fold).to_numpy());te=np.flatnonzero(mask&s.outer_fold.eq(fold).to_numpy())
            assert set(s.iloc[tr].case).isdisjoint(s.iloc[te].case)
            for model in models:tasks.append(dict(dataset='synthetic',strategy='case_grouped',support=support,repeat=0,fold=fold,model=model,train_indices=tr.tolist(),test_indices=te.tolist()))
    old_tasks=json.loads((PRIOR/'splits.json').read_text())['tasks']
    for t in old_tasks:
        previous.nested.audit_task(r,t)
        for model in PANELS:tasks.append(dict(t,dataset='real',support='full',model=model))
    assert len(tasks)==547
    old_design=json.loads((previous.OLD/'design.json').read_text())
    candidate=next(c for c in old_design['candidates'] if c['name']=='fine_squared')
    sources[str(previous.OLD/'design.json')]=sha(previous.OLD/'design.json')
    for module in [legacy.refit.tuning.b,legacy.refit.tuning.b.engine,legacy.refit.tuning.b.forest]:
        sources[str(Path(module.__file__))]=sha(module.__file__)
    design=dict(reference='STREAMLINED-PREDICTORS-2026-11-03',panels=PANELS,params=candidate['params'],seed=20261014,
        sources=sources,inputs={str(OUT/f):sha(OUT/f) for f in ['synthetic_input.parquet','real_input.parquet']},
        primary='Fixed-depth Manning; centroid shift; real and synthetic fitted separately. Variable depth remains sensitivity, not primary selection.',
        model='One prespecified existing fine_squared RF configuration: 500 trees, depth 16, minimum leaf 2, squared error. Same fit code, random seed, train-only scaling/imputation and target scaling in both datasets. No tuning or subset search. Results are not interchangeable with the earlier tuned-RF benchmark.',
        weights='Equal total weight per real river; equal total weight per synthetic parameter combination.',
        validation='Real: exact historical 15 leave-river-out + 25 network-grouped + 25 compound-family outer splits. Synthetic: same three case-group outer folds used previously; complete seven-state combinations kept together. Different datasets necessarily have different grouping units.',
        coverage='Original width-CV ratio is undefined at parent CV <= 1e-12 (roundoff tolerance). Original nine and revised nine are both trained/evaluated on the same defined synthetic support. Other main synthetic panels use all 45927 rows. Real original-nine inputs are finite for all 2185 rows. No epsilon denominator or zero filling.',
        feature_changes='Width CV ratio replaced by child-minus-parent CV in revised9. seven_* adds entropy change and parent dominance to the existing five. Two prespecified revised9 ablations remove allocation descriptors or normalized route-spread change. All lengths converted to metres.',
        interpretation='Exploratory reconciliation after inspecting real results; no independent confirmation claim or automatic acceptance threshold. No test-driven feature expansion during this run. Joint removals measure predictive contribution, not causal importance.',
        jobs=547,synthetic_jobs=27,real_jobs=520)
    fp=hashlib.sha256(json.dumps(design,sort_keys=True).encode()).hexdigest()
    save(OUT/'tasks.json',dict(fingerprint=fp,tasks=tasks));save(OUT/'design.json',design)
    return design,fp

def fit_task(task,design,fp):
    folder=OUT/'checkpoints'/task['dataset']/task['strategy']/task['support']/f"r{task['repeat']}"/f"f{task['fold']}"/task['model']
    final=folder/'completion.json';prediction=folder/'predictions.parquet'
    if final.exists():
        receipt=json.loads(final.read_text());assert receipt['fingerprint']==fp and sha(prediction)==receipt['prediction_sha256']
        return str(final)
    d=pd.read_parquet(OUT/f"{task['dataset']}_input.parquet")
    tr=d.iloc[task['train_indices']];te=d.iloc[task['test_indices']]
    b=legacy.refit.tuning.b;b.COLUMNS=design['panels'][task['model']]
    assert np.isfinite(tr[b.COLUMNS]).all().all() and np.isfinite(te[b.COLUMNS]).all().all()
    start=time.monotonic();artifact=b.fit(tr,design['params'],jobs=1);pred=b.forest.predict(artifact,te)
    assert np.isfinite(pred).all()
    z=te[['example','state','cohort','edit_type','centroid_shift_hours']].copy()
    z['predicted_hours']=pred
    for k in ['dataset','strategy','support','repeat','fold','model']:z[k]=task[k]
    folder.mkdir(parents=True,exist_ok=True);temp=prediction.with_suffix('.tmp.parquet');z.to_parquet(temp,index=False);temp.replace(prediction)
    save(final,dict(fingerprint=fp,seconds=time.monotonic()-start,n_train=len(tr),n_test=len(te),
        prediction_sha256=sha(prediction),kept_columns=artifact['preprocessor'].kept,
        dropped_columns=artifact['preprocessor'].dropped))
    return str(final)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',action='store_true');args=ap.parse_args()
    design,fp=prepare();print('Prepared',design['jobs'],'fixed-RF fits (no inner tuning)',flush=True)
    if not args.run:return
    with (OUT/'run.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        while not (PRIOR/'completion.json').exists():
            save(OUT/'progress.json',dict(status='queued_after_existing_comparison',workers=0,total_jobs=547))
            print('Waiting for existing comparison to finish; no additional fit workers.',flush=True);time.sleep(30)
        tasks=json.loads((OUT/'tasks.json').read_text())['tasks'];start=time.monotonic()
        with ProcessPoolExecutor(max_workers=2) as pool:
            futures=[pool.submit(fit_task,t,design,fp) for t in tasks]
            for count,future in enumerate(as_completed(futures),1):
                path=future.result()
                save(OUT/'progress.json',dict(status='running',completed_jobs=count,total_jobs=len(tasks),workers=2,elapsed_seconds=time.monotonic()-start,last_checkpoint=path))
                print(count,'/',len(tasks),'completed;',round((time.monotonic()-start)/60,1),'minutes',flush=True)
        import summarize
        summarize.build()
        for p,h in design['sources'].items():assert sha(p)==h,p
        save(OUT/'completion.json',dict(complete=True,fingerprint=fp,jobs=len(tasks),elapsed_seconds=time.monotonic()-start,
             files={str(OUT/x):sha(OUT/x) for x in ['summary.csv','report.md','predictions.parquet']}))
        save(OUT/'progress.json',dict(status='complete',completed_jobs=len(tasks),total_jobs=len(tasks),elapsed_seconds=time.monotonic()-start))

if __name__=='__main__':main()
