"""Paired fixed-RF test: child width P90 versus child/parent mean-width ratio."""
from pathlib import Path
import sys, json, hashlib, shutil, time, fcntl, importlib.util
from concurrent.futures import ProcessPoolExecutor, as_completed
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'RAPID/investigations/streamlined_predictors_20261103'))
import run_comparison as base
import pandas as pd
import numpy as np
SOURCE=ROOT/'RAPID/outputs/mean_width_compact_20261103'
OLD_GEOMETRY=ROOT/'RAPID/outputs/synthetic_selected_sets_20261102/input.parquet'
OUT=ROOT/'RAPID/outputs/width_p90_compact_20261103'

def prepare():
    OUT.mkdir(exist_ok=True)
    if (OUT/'design.json').exists():
        d=json.loads((OUT/'design.json').read_text())
        for p,h in d['sources'].items():assert base.sha(p)==h,p
        for p,h in d['inputs'].items():assert base.sha(p)==h,p
        return d,hashlib.sha256(json.dumps(d,sort_keys=True).encode()).hexdigest()
    receipt=json.loads((SOURCE/'completion.json').read_text());assert receipt['complete']
    assert base.sha(SOURCE/'predictions.parquet')==receipt['files'][str(SOURCE/'predictions.parquet')]
    old=json.loads((SOURCE/'design.json').read_text())
    panels={k.replace('mean_width','child_p90'):[('B_child_width_p90' if c=='log1p__B_ratio_width_mean' else c) for c in v] for k,v in old['panels'].items()}
    sources=[Path(__file__),Path(base.__file__),SOURCE/'design.json',SOURCE/'tasks.json',SOURCE/'completion.json',SOURCE/'predictions.parquet',OLD_GEOMETRY]
    for p,h in old['inputs'].items():assert base.sha(p)==h,p
    real=pd.read_parquet(SOURCE/'real_input.parquet')
    geo=pd.read_parquet(OLD_GEOMETRY).set_index(['example','state'])
    assert not geo.index.duplicated().any()
    idx=pd.MultiIndex.from_frame(real[['example','state']]);geo=geo.loc[idx]
    ordinary=real.cohort.eq('original').to_numpy()
    # Historical ordinary domains coincide; whole-network controls require their
    # edited footprint, not the old full-graph width quantile including connectors.
    for c in ['B_delta_route_length_mean','parent_slope_slow','path_delta_topologic_complexity_score','B_delta_area_proxy']:
        np.testing.assert_allclose(real.loc[ordinary,c],geo.loc[ordinary,c],rtol=1e-8,atol=1e-6)
    np.testing.assert_allclose(real.centroid_shift_hours*60,geo.centroid_shift_minutes,atol=1e-9,rtol=1e-10)
    real['B_child_width_p90']=geo.B_child_width_p90.to_numpy()
    audit=[]
    for i,row in real.loc[~ordinary].iterrows():
        lineage_path=base.legacy.refit.EXT/f'variants/{row.example}/{row.state}/lineage.csv'
        sources.append(lineage_path)
        lineage=pd.read_csv(lineage_path)
        ids=set(lineage.loc[lineage.role.eq('collapsed_block'),'child_link_id'].astype(int))
        f=base.legacy.refit.active(row.example,row.state)
        f=f[f.id_link.isin(ids)]
        assert len(f)==len(ids)>0
        w=f.width.to_numpy();length=f.link_length_m.to_numpy()
        assert (w>0).all() and (length>0).all()
        order=np.argsort(w,kind='stable')
        value=float(w[order[min(np.searchsorted(np.cumsum(length[order]),.9*length.sum()),len(w)-1)]])
        audit.append(dict(example=row.example,state=row.state,old_whole_network_p90=real.at[i,'B_child_width_p90'],edited_domain_p90=value,n_links=len(f)))
        real.at[i,'B_child_width_p90']=value
    pd.DataFrame(audit).to_csv(OUT/'full_network_scope_audit.csv',index=False)
    real.to_parquet(OUT/'real_input.parquet',index=False)
    shutil.copy2(SOURCE/'synthetic_input.parquet',OUT/'synthetic_input.parquet')
    for dataset in ['real','synthetic']:
        f=pd.read_parquet(OUT/f'{dataset}_input.parquet')
        assert np.isfinite(f[list(set(sum(panels.values(),[])))]).all().all()
        assert (f.B_child_width_p90>0).all()
    tasks=[dict(t,model=t['model'].replace('mean_width','child_p90')) for t in json.loads((SOURCE/'tasks.json').read_text())['tasks']]
    assert len(tasks)==136
    hashes={str(p):base.sha(p) for p in sources};hashes.update(base.legacy.refit.HASHES)
    d=dict(reference='WIDTH-P90-COMPACT-2026-11-03',panels=panels,params=old['params'],seed=old['seed'],
        sources=hashes,inputs={str(OUT/f'{s}_input.parquet'):base.sha(OUT/f'{s}_input.parquet') for s in ['real','synthetic']},
        response='centroid_shift_hours',depth='fixed Manning',jobs=len(tasks),validation=old['validation'],
        change='Replace only log1p(child/parent mean reach width) with child length-weighted width P90 in metres. Other four inputs, RF, rows, weights, targets and folds unchanged. With G and without G (area-change alternative) each remain five inputs.',
        scope='Original closed B domains for ordinary comparisons; collapsed-block edited footprints for full-network controls. Historical P90 reused only on matched ordinary domains.',
        limitation='Post-hoc exploratory comparison on reused validation data. Not an independent final test or a refit of the entire newly selected four-variable ridge panel.')
    fp=hashlib.sha256(json.dumps(d,sort_keys=True).encode()).hexdigest()
    base.save(OUT/'tasks.json',dict(fingerprint=fp,tasks=tasks));base.save(OUT/'design.json',d)
    return d,fp

def fit(t,d,fp):
    base.OUT=OUT
    return base.fit_task(t,d,fp)

def report():
    spec=importlib.util.spec_from_file_location('p90_score_helpers',Path(base.__file__).with_name('summarize.py'))
    helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
    d=json.loads((OUT/'design.json').read_text());fp=hashlib.sha256(json.dumps(d,sort_keys=True).encode()).hexdigest()
    parts=[]
    for f in sorted((OUT/'checkpoints').rglob('completion.json')):
        c=json.loads(f.read_text());assert c['fingerprint']==fp
        p=f.parent/'predictions.parquet';assert base.sha(p)==c['prediction_sha256']
        parts.append(pd.read_parquet(p))
    assert len(parts)==136
    old=pd.read_parquet(SOURCE/'predictions.parquet')
    old=old[old.model.isin(['five_mean_width_with_G','five_mean_width_without_G','old9','revised9'])]
    p=pd.concat([old,*parts],ignore_index=True)
    assert not p.duplicated(['dataset','strategy','repeat','model','example','state']).any()
    p.to_parquet(OUT/'predictions.parquet',index=False)
    rows=[]
    for (dataset,strategy,repeat,model),g in p.groupby(['dataset','strategy','repeat','model']):
        populations=[('all',g)] if dataset=='synthetic' else [('ordinary',g[g.cohort.eq('original')]),('full_network',g[g.cohort.eq('segmented')]),('all',g)]
        for population,z in populations:rows.append(dict(dataset=dataset,strategy=strategy,repeat=repeat,model=model,population=population,**helper.scores(z,dataset=='real' and population=='all')))
    s=pd.DataFrame(rows);s.to_csv(OUT/'repeat_scores.csv',index=False)
    summary=s.groupby(['dataset','strategy','population','model']).agg(n=('n','first'),repeats=('repeat','nunique'),r2=('r2','mean'),r2_min=('r2','min'),r2_max=('r2','max'),mae_minutes=('mae_minutes','mean')).reset_index()
    summary.to_csv(OUT/'summary.csv',index=False)
    paired=[]
    for variant in ['with_G','without_G']:
        keys=['dataset','strategy','repeat','population']
        a=s[s.model.eq('five_child_p90_'+variant)].merge(s[s.model.eq('five_mean_width_'+variant)],on=keys,suffixes=('_p90','_mean'))
        assert (a.n_p90==a.n_mean).all()
        a['delta_r2']=a.r2_p90-a.r2_mean;a['delta_mae_minutes']=a.mae_minutes_p90-a.mae_minutes_mean;a['variant']=variant;paired.append(a)
    pd.concat(paired).to_csv(OUT/'paired_changes.csv',index=False)
    r=p[p.dataset.eq('real')].copy();r['absolute_error_minutes']=60*abs(r.predicted_hours-r.centroid_shift_hours)
    r.groupby(['strategy','repeat','model','cohort','example']).absolute_error_minutes.mean().rename('mae_minutes').to_csv(OUT/'per_river_errors.csv')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    q=summary[summary.dataset.eq('real')&summary.population.eq('ordinary')]
    order=['five_mean_width_with_G','five_child_p90_with_G','five_mean_width_without_G','five_child_p90_without_G']
    table=q.pivot(index='strategy',columns='model',values='r2').reindex(columns=order)
    fig,ax=plt.subplots(figsize=(10,4),layout='constrained');im=ax.imshow(table,vmin=0,vmax=1,cmap='viridis',aspect='auto')
    for i in range(len(table)):
        for j in range(len(order)):ax.text(j,i,f'{table.iloc[i,j]:.3f}',ha='center',va='center',color='white')
    ax.set_xticks(range(4),['Mean width + G','Child P90 + G','Mean width + area','Child P90 + area']);ax.set_yticks(range(len(table)),table.index)
    ax.set_title('Matched width replacement · ordinary real-network comparisons');fig.colorbar(im,ax=ax,label='Held-out R²');fig.savefig(OUT/'comparison.png',dpi=180);plt.close(fig)
    lines=['# Matched width replacement results','','Reference: WIDTH-P90-COMPACT-2026-11-03. Fixed-depth Manning centroid; identical fixed RF and validation splits. Only width predictor replaced. Mean-width baselines reused from verified completed predictions.','','| Dataset | Strategy | Population | Set | R² | MAE (min) |','|---|---|---|---|---:|---:|']
    for r in summary.itertuples():lines.append(f'| {r.dataset} | {r.strategy} | {r.population} | {r.model} | {r.r2:.4f} | {r.mae_minutes:.4f} |')
    lines+=['','Ordinary population: 2,170 comparisons. Fifteen full-network controls are reported separately. Pooled scores use the existing cohort/river weighting. Five-repeat ranges are descriptive, not independent confidence intervals. This is post-hoc selection on reused real data, not a new independent validation.','']
    (OUT/'report.md').write_text('\n'.join(lines))
    return summary

if __name__=='__main__':
    d,fp=prepare()
    with (OUT/'run.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        tasks=json.loads((OUT/'tasks.json').read_text())['tasks'];start=time.monotonic()
        with ProcessPoolExecutor(max_workers=2) as pool:
            futures=[pool.submit(fit,t,d,fp) for t in tasks]
            for i,f in enumerate(as_completed(futures),1):
                f.result();base.save(OUT/'progress.json',dict(status='running',completed_jobs=i,total_jobs=len(tasks),elapsed_seconds=time.monotonic()-start))
                print(f'{i}/136 completed; {(time.monotonic()-start)/60:.1f} minutes',flush=True)
        print(report().to_string(index=False),flush=True)
        for p,h in d['sources'].items():assert base.sha(p)==h,p
        base.save(OUT/'completion.json',dict(complete=True,fingerprint=fp,jobs=136,seconds=time.monotonic()-start))
        base.save(OUT/'progress.json',dict(status='complete',completed_jobs=136,total_jobs=136))
