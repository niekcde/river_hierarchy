"""Replace equivalent path width by mean-reach-width ratio in the frozen five."""
from pathlib import Path
import sys,json,hashlib,shutil,time,fcntl,importlib.util
from concurrent.futures import ProcessPoolExecutor,as_completed
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'RAPID/investigations/streamlined_predictors_20261103'))
import run_comparison as base
import numpy as np
import pandas as pd
SOURCE=base.OUT
OUT=ROOT/'RAPID/outputs/mean_width_compact_20261103'
PANELS={m.replace('five_','five_mean_width_'):[
    'log1p__B_ratio_width_mean' if c=='log1p__path_parent_equivalent_width' else c
    for c in base.PANELS[m]] for m in ['five_with_G','five_without_G']}

def prepare():
    OUT.mkdir(exist_ok=True)
    if (OUT/'design.json').exists():
        design=json.loads((OUT/'design.json').read_text())
        for p,h in design['sources'].items():assert base.sha(p)==h,p
        for p,h in design['inputs'].items():assert base.sha(p)==h,p
        return design,hashlib.sha256(json.dumps(design,sort_keys=True).encode()).hexdigest()
    receipt=json.loads((SOURCE/'completion.json').read_text());assert receipt['complete']
    assert base.sha(SOURCE/'predictions.parquet')==receipt['files'][str(SOURCE/'predictions.parquet')]
    old=json.loads((SOURCE/'design.json').read_text())
    tasks=[]
    for t in json.loads((SOURCE/'tasks.json').read_text())['tasks']:
        if t['model'] not in ['five_with_G','five_without_G'] or t['support']!='full':continue
        tasks.append(dict(t,model=t['model'].replace('five_','five_mean_width_')))
    assert len(tasks)==136
    for dataset in ['synthetic','real']:
        file=SOURCE/f'{dataset}_input.parquet'
        assert base.sha(file)==old['inputs'][str(file)]
        d=pd.read_parquet(file)
        assert np.isfinite(d[list(set(sum(PANELS.values(),[])))]).all().all()
        assert (np.expm1(d.log1p__B_ratio_width_mean)>0).all()
        shutil.copy2(file,OUT/file.name)
    files=[Path(__file__),Path(base.__file__),SOURCE/'design.json',SOURCE/'tasks.json',SOURCE/'predictions.parquet',SOURCE/'completion.json']
    design=dict(reference='MEAN-WIDTH-COMPACT-2026-11-03',panels=PANELS,params=old['params'],seed=old['seed'],
        sources={str(p):base.sha(p) for p in files},inputs={str(OUT/f):base.sha(OUT/f) for f in ['synthetic_input.parquet','real_input.parquet']},
        definition='Replace log1p(parent sum of harmonic path widths) with log1p(child/parent length-weighted arithmetic mean reach width). Each physical edge enters width moments once in the selected domain. Four other inputs, order, fixed RF settings, weighting, responses and splits remain unchanged.',
        validation=old['validation'],response=old['primary'],jobs=136,
        caveats='Post-hoc candidate prompted by example-37 failures. All populations retained, no tuning, no claim of independent feature-selection validation. Synthetic full-converged centroid and real finite-window centroid remain different integrations.')
    fp=hashlib.sha256(json.dumps(design,sort_keys=True).encode()).hexdigest()
    base.save(OUT/'tasks.json',dict(fingerprint=fp,tasks=tasks));base.save(OUT/'design.json',design)
    return design,fp

def fit(task,design,fp):
    base.OUT=OUT
    return base.fit_task(task,design,fp)

def report(design,fp):
    path=Path(base.__file__).with_name('summarize.py')
    spec=importlib.util.spec_from_file_location('mean_width_score_helpers',path)
    helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
    parts=[]
    for path in sorted((OUT/'checkpoints').rglob('completion.json')):
        c=json.loads(path.read_text());assert c['fingerprint']==fp
        f=path.parent/'predictions.parquet';assert base.sha(f)==c['prediction_sha256']
        parts.append(pd.read_parquet(f))
    assert len(parts)==136
    new=pd.concat(parts,ignore_index=True)
    old=pd.read_parquet(SOURCE/'predictions.parquet')
    old=old[old.model.isin(['old9','revised9','five_with_G','five_without_G'])&old.support.eq('full')]
    p=pd.concat([old,new],ignore_index=True)
    assert not p.duplicated(['dataset','strategy','support','repeat','model','example','state']).any()
    p.to_parquet(OUT/'predictions.parquet',index=False)
    rows=[]
    for group,g in p.groupby(['dataset','strategy','repeat','model']):
        dataset,strategy,repeat,model=group
        subsets=[('all',g)] if dataset=='synthetic' else [('ordinary',g[g.cohort.eq('original')]),('full_network',g[g.cohort.eq('segmented')]),('all',g)]
        for population,z in subsets:rows.append(dict(dataset=dataset,strategy=strategy,repeat=repeat,model=model,population=population,**helper.scores(z,dataset=='real' and population=='all')))
    s=pd.DataFrame(rows);s.to_csv(OUT/'repeat_scores.csv',index=False)
    summary=s.groupby(['dataset','strategy','population','model']).agg(n=('n','first'),repeats=('repeat','nunique'),
        r2=('r2','mean'),r2_min=('r2','min'),r2_max=('r2','max'),mae_minutes=('mae_minutes','mean'),rmse_minutes=('rmse_minutes','mean')).reset_index()
    summary.to_csv(OUT/'summary.csv',index=False)
    r=p[p.dataset.eq('real')].copy();r['abs_error_minutes']=60*abs(r.predicted_hours-r.centroid_shift_hours)
    r['squared_error_minutes']=(60*(r.predicted_hours-r.centroid_shift_hours))**2
    river=r.groupby(['strategy','repeat','model','cohort','example']).agg(n=('state','size'),mae_minutes=('abs_error_minutes','mean'),mse_minutes=('squared_error_minutes','mean')).reset_index()
    river['sse_share']=river.mse_minutes/river.groupby(['strategy','repeat','model','cohort']).mse_minutes.transform('sum')
    river.to_csv(OUT/'river_error_contributions.csv',index=False)
    r.sort_values('abs_error_minutes',ascending=False).groupby(['strategy','repeat','model']).head(10).to_csv(OUT/'largest_errors.csv',index=False)
    paired=[]
    for new_name,old_name in [('five_mean_width_with_G','five_with_G'),('five_mean_width_without_G','five_without_G'),('five_mean_width_with_G','revised9'),('five_mean_width_without_G','revised9')]:
        keys=['dataset','strategy','repeat','population']
        a=s[s.model.eq(new_name)].merge(s[s.model.eq(old_name)],on=keys,suffixes=('_new','_old'))
        for row in a.itertuples():
            assert row.n_new==row.n_old
            paired.append(dict(dataset=row.dataset,strategy=row.strategy,repeat=row.repeat,population=row.population,new_model=new_name,reference=old_name,delta_r2=row.r2_new-row.r2_old,delta_mae_minutes=row.mae_minutes_new-row.mae_minutes_old))
    pd.DataFrame(paired).to_csv(OUT/'paired_changes.csv',index=False)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    columns=['old9','revised9','five_with_G','five_mean_width_with_G','five_without_G','five_mean_width_without_G']
    a=summary[(summary.dataset.eq('real')&summary.population.eq('ordinary'))|summary.dataset.eq('synthetic')].copy()
    a['row']=a.dataset+' / '+a.strategy
    table=a.pivot(index='row',columns='model',values='r2').reindex(columns=columns)
    fig,ax=plt.subplots(figsize=(12,4.8),layout='constrained');im=ax.imshow(np.ma.masked_invalid(table.to_numpy()),vmin=-1,vmax=1,cmap='RdYlGn',aspect='auto')
    for i in range(len(table)):
        for j in range(len(columns)):
            v=table.iloc[i,j];ax.text(j,i,'—' if pd.isna(v) else f'{v:.3f}',ha='center',va='center',fontsize=10)
    ax.set_xticks(range(len(columns)),columns,rotation=25,ha='right');ax.set_yticks(range(len(table)),table.index)
    ax.set_title('Mean-width ratio replacement · same fixed RF and splits\nReal: ordinary comparisons; synthetic: full population')
    fig.colorbar(im,ax=ax,label='Held-out R² (colour clipped at −1)');fig.savefig(OUT/'comparison_matrix.png',dpi=180);plt.close(fig)
    lines=['# Mean-width replacement in the compact five','',
        'Reference: MEAN-WIDTH-COMPACT-2026-11-03. Completed post-hoc candidate test; fixed-depth Manning centroid shifts.','',
        'Replace only parent equivalent-path-width sum by `log(1 + mean_width_child / mean_width_parent)`, with physical-length-weighted arithmetic mean widths on the existing selected domains. Each reach enters these moments once. Both sets still contain five inputs.','',
        'RF settings, seed, preprocessing, training weights, targets and folds match the previous streamlined experiment exactly. Baseline predictions are reused from verified files; 136 new fits evaluate the two replacements. No routing reruns or tuning.','',
        '| Dataset | Validation | Population | Predictor set | R² | MAE (min) |',
        '|---|---|---|---|---:|---:|']
    for row in summary.itertuples():lines.append(f'| {row.dataset} | {row.strategy} | {row.population} | {row.model} | {row.r2:.3f} | {row.mae_minutes:.3f} |')
    lines+=['','Both new sets are defined on the complete synthetic population and all real comparisons. Original-nine synthetic results are omitted here because that ratio-based panel requires smaller common support; the earlier common-support comparison remains available.','',
        'Main real comparison is validation strategy × predictor set. Ordinary and full-network cohorts are additionally reported to avoid misleading pooled R². Per-river squared-error shares and the largest errors are exported to check whether improvements are limited to example 37.','',
        'This replacement was motivated by inspected example-37 errors and is exploratory. Reused validation data are not independent confirmation of the discovery process. Synthetic centroids integrate the full converged pulse; real centroids use the finite event window. Cross-dataset differences cannot be attributed solely to topology complexity.','']
    (OUT/'report.md').write_text('\n'.join(lines))
    return summary

def main():
    design,fp=prepare()
    with (OUT/'run.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        tasks=json.loads((OUT/'tasks.json').read_text())['tasks'];start=time.monotonic()
        with ProcessPoolExecutor(max_workers=2) as pool:
            jobs=[pool.submit(fit,t,design,fp) for t in tasks]
            for count,future in enumerate(as_completed(jobs),1):
                path=future.result()
                base.save(OUT/'progress.json',dict(status='running',completed_jobs=count,total_jobs=len(tasks),workers=2,elapsed_seconds=time.monotonic()-start,last_checkpoint=path))
                print(f'{count}/136 completed; {(time.monotonic()-start)/60:.1f} minutes',flush=True)
        summary=report(design,fp)
        for p,h in design['sources'].items():assert base.sha(p)==h,p
        base.save(OUT/'completion.json',dict(complete=True,fingerprint=fp,jobs=136,files={str(OUT/n):base.sha(OUT/n) for n in ['summary.csv','predictions.parquet','report.md','comparison_matrix.png']}))
        base.save(OUT/'progress.json',dict(status='complete',completed_jobs=136,total_jobs=136,elapsed_seconds=time.monotonic()-start))
        print(summary.to_string(index=False))

if __name__=='__main__':main()
