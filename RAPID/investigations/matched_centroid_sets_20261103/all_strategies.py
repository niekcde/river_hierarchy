"""Three frozen centroid panels across all three historical validation schemes.

Wait for the already-running compound comparison; reuse it and compute only
the two missing validation approaches. No predictor or hyperparameter search
beyond the frozen six-configuration inner grid.
"""
import compare as raw
import safe_width as safe
import time
import json
import hashlib
import fcntl
import shutil
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
import pandas as pd

OUT=raw.ROOT/'RAPID/outputs/matched_centroid_all_strategies_20261103'
SOURCE=raw.legacy.refit.BASE/'fixed_manning_rf_nested_20261029'
MODELS=['geometry_normalized_9','centroid_with_G','centroid_without_G']

def save(name,value):raw.legacy.refit.save(OUT/name,value)

def prepare():
    OUT.mkdir(exist_ok=True)
    if (OUT/'design.json').exists():
        design=raw.nested.load(OUT/'design.json')
        for p,h in design['sources'].items():assert raw.legacy.refit.sha(p)==h,p
        assert raw.legacy.refit.sha(OUT/'input.parquet')==design['input_sha256']
        splits=raw.nested.load(OUT/'splits.json')
        return design,splits['fingerprint'],splits['tasks']
    old=raw.nested.load(raw.OLD/'design.json')
    design=raw.nested.load(safe.OUT/'design.json')
    historical=raw.nested.load(SOURCE/'splits.json')
    hd=raw.nested.load(SOURCE/'design.json')
    assert historical['fingerprint']==hashlib.sha256(json.dumps(hd,sort_keys=True).encode()).hexdigest()
    tasks=historical['tasks'];assert len(tasks)==65
    d=pd.read_parquet(safe.OUT/'input.parquet');h=pd.read_parquet(SOURCE/'input.parquet')
    np.testing.assert_array_equal(d[['example','state','group_id']],h[['example','state','group_id']])
    np.testing.assert_array_equal(d.centroid_shift_hours,h.centroid_shift_hours)
    for t in tasks:raw.nested.audit_task(d,t)
    compound=raw.nested.load(safe.OUT/'splits.json')['tasks']
    assert [t for t in tasks if t['strategy']=='compound_mixed']==compound
    panels={'geometry_normalized_9':old['panels']['geometry_normalized_9'],**design['panels']}
    assert np.isfinite(d[list(set(sum(panels.values(),[])))]).all().all()
    shutil.copy2(safe.OUT/'input.parquet',OUT/'input.parquet')
    paths=[Path(__file__),SOURCE/'splits.json',SOURCE/'design.json',SOURCE/'input.parquet',
           safe.OUT/'design.json',safe.OUT/'splits.json',safe.OUT/'input.parquet',raw.OLD/'design.json']
    design=dict(design,panels=panels,sources=design['sources']|{str(p):raw.legacy.refit.sha(p) for p in paths},
        validation='All three historical schemes: 15 leave-one-river-out folds; 5 repeats x 5 whole-river-grouped folds; 5 repeats x 5 compound-family-grouped folds. Inner grouping matches outer grouping. Same input populations, RF grid, weights and targets.',
        reused='75 compound-family outer results: 25 old normalized-nine and 50 corrected new-five fits. Other schemes refit all three panels; historical geometry_9 is not the same as geometry_normalized_9 and is not substituted.',
        jobs=120,forest_fits=2280,
        interpretation='Validation grouping and edit type are separate axes. Report all ordinary states, full compounds, compound members, islands, simple units and full-network controls separately. No new features or model settings.')
    fp=hashlib.sha256(json.dumps(design,sort_keys=True).encode()).hexdigest()
    save('design.json',design);save('splits.json',dict(fingerprint=fp,tasks=tasks))
    return design,fp,tasks

def fit_task(t,model,design,fp):
    raw.nested.DEST=OUT
    return raw.nested.fit_task(t,model,design,fp)

def summarize(fp,complete=False):
    inherited=safe.OUT/'predictions.parquet'
    if not inherited.exists():inherited=safe.OUT/'partial_predictions.parquet'
    p=pd.read_parquet(inherited)
    rows=[]
    for path in sorted((OUT/'checkpoints').glob('*/*/*/*/outer.json')):
        result=raw.nested.load(path);assert result['fingerprint']==fp;rows.extend(result['rows'])
    p=pd.concat([p,pd.DataFrame(rows)],ignore_index=True)
    assert not p.duplicated(['strategy','model','repeat','example','state']).any()
    records=[]
    for (strategy,repeat),whole in p.groupby(['strategy','repeat']):
        # Compare only repetitions completed for all three panels, not different
        # subsets of folds or different repeat counts across models.
        if any(len(whole[whole.model.eq(m)])!=2185 for m in MODELS):continue
        for model,g in whole.groupby('model'):
            subsets=[('ordinary',g[g.cohort.eq('original')]),('full_network',g[g.cohort.eq('segmented')]),('all',g)]
            subsets += [(str(k),z) for k,z in g[g.cohort.eq('original')].groupby('edit_type')]
            for name,z in subsets:
                records.append(dict(strategy=strategy,repeat=repeat,model=model,population=name,
                    **raw.nested.score(z,'equal_cohort_network' if name=='all' else 'equal_network')))
    scores=pd.DataFrame(records)
    scores.to_csv(OUT/'repeat_scores.csv',index=False)
    if len(scores):
        summary=scores.groupby(['strategy','population','model']).agg(
            completed_repeats=('repeat','nunique'),n=('n','first'),r2_mean=('r2','mean'),
            r2_min=('r2','min'),r2_max=('r2','max'),mae_minutes_mean=('mae_minutes','mean'))
        summary.reset_index().to_csv(OUT/'summary.csv',index=False)
        lines=['# Matched three-strategy centroid comparison — 3 November 2026','',
            'Status: '+('COMPLETE' if complete else 'IN PROGRESS — only jointly complete repeats are scored.'),'',
            'All three panels use identical splits, target, tuning grid and training procedure. New panels use log1p of equivalent width for numerical resolution. Ordinary includes simple, island, compound-member and full-compound collapses. Compound-family grouping is a validation scheme, not a restriction to full compounds.','',
            '| Validation | Population | Predictor set | Completed repeats | R² | MAE (min) |',
            '|---|---|---|---:|---:|---:|']
        for r in summary.reset_index().itertuples():
            lines.append(f'| {r.strategy} | {r.population} | {r.model} | {r.completed_repeats} | {r.r2_mean:.3f} | {r.mae_minutes_mean:.3f} |')
        lines+=['','Expected repeats: leave_one_network_out = 1 (15 held rivers); network_balanced = 5; compound_mixed = 5. Empty strategies are pending, not zero performance.','',
            'Read ordinary and full-network results separately. Pooled R² includes between-cohort variation and can hide ordinary-case failures. Edit-type scores use equal river weights within each type; subgroup R² are not additive.',
            '', 'Limitations: historical response/type-balanced partitions; exploratory preselected features; full-network singleton controls can share constituents with training states in compound-family validation; repeat ranges are not confidence intervals. No routing runs, synthetic-trained coefficients, or new feature selection.','']
        (OUT/'report.md').write_text('\n'.join(lines))
    p.to_parquet(OUT/('predictions.parquet' if complete else 'partial_predictions.parquet'),index=False)
    if complete:
        assert len(p)==2185*11*3 and len(records)==11*3*7
        for path,h in raw.nested.load(OUT/'design.json')['sources'].items():assert raw.legacy.refit.sha(path)==h,path
        save('completion.json',dict(complete=True,fingerprint=fp,new_outer_jobs=120,new_forest_fits=2280,
            files={str(OUT/n):raw.legacy.refit.sha(OUT/n) for n in ['predictions.parquet','repeat_scores.csv','summary.csv','report.md']}))

def main():
    design,fp,tasks=prepare()
    with (OUT/'run.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        while not (safe.OUT/'completion.json').exists():
            save('progress.json',dict(status='waiting_for_existing_compound_run',workers=0,new_jobs=120))
            summarize(fp)
            print('Waiting for existing two-worker compound comparison; no extra RF workers.',flush=True)
            time.sleep(30)
        receipt=raw.nested.load(safe.OUT/'completion.json');assert receipt['complete']
        assert raw.legacy.refit.sha(safe.OUT/'predictions.parquet')==receipt['files'][str(safe.OUT/'predictions.parquet')]
        missing=[t for t in tasks if t['strategy']!='compound_mixed']
        assert len(missing)==40
        start=time.monotonic()
        with ProcessPoolExecutor(max_workers=2) as pool:
            futures=[pool.submit(fit_task,t,m,design,fp) for t in missing for m in MODELS]
            for i,f in enumerate(as_completed(futures),1):
                checkpoint=f.result()
                save('progress.json',dict(status='running',completed_jobs=i,total_jobs=120,workers=2,
                    elapsed_seconds=time.monotonic()-start,last_checkpoint=checkpoint))
                summarize(fp)
                print(f'{i}/120 missing outer fits completed',flush=True)
        summarize(fp,True)
        save('progress.json',dict(status='complete',completed_jobs=120,total_jobs=120,elapsed_seconds=time.monotonic()-start))

if __name__=='__main__':main()
