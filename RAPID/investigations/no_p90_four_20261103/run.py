"""Evaluate the exact four-input no-P90 synthetic selection on real networks."""
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/no-p90-four-mpl')
from pathlib import Path
import sys,json,hashlib,time,fcntl,importlib.util
from concurrent.futures import ProcessPoolExecutor,as_completed
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'RAPID/investigations/streamlined_predictors_20261103'))
import run_comparison as base
import numpy as np
import pandas as pd
SOURCE=ROOT/'RAPID/outputs/width_p90_compact_20261103'
OUT=ROOT/'RAPID/outputs/no_p90_four_20261103'
COLUMNS=['B_delta_route_length_mean','parent_slope_slow','B_parent_width_mean','B_parent_bifurcations']
MODEL='synthetic_selected_four_no_p90'

def prepare():
    OUT.mkdir(exist_ok=True)
    if (OUT/'design.json').exists():
        d=json.loads((OUT/'design.json').read_text())
        for p,h in d['sources'].items():assert base.sha(p)==h,p
        for p,h in d['inputs'].items():assert base.sha(p)==h,p
        return d,hashlib.sha256(json.dumps(d,sort_keys=True).encode()).hexdigest()
    assert json.loads((SOURCE/'completion.json').read_text())['complete']
    old=json.loads((SOURCE/'design.json').read_text())
    for p,h in old['inputs'].items():assert base.sha(p)==h,p
    r=pd.read_parquet(SOURCE/'real_input.parquet')
    rec_path=base.legacy.refit.BASE/'validation_group_inventory_20261023/records.parquet'
    records=pd.read_parquet(rec_path).set_index(['example','state'])
    historic_path=ROOT/'RAPID/outputs/synthetic_selected_sets_20261102/input.parquet'
    historic=pd.read_parquet(historic_path).set_index(['example','state'])
    sources=[Path(__file__),Path(base.__file__),SOURCE/'design.json',SOURCE/'completion.json',SOURCE/'predictions.parquet',SOURCE/'tasks.json',SOURCE/'real_input.parquet',rec_path,historic_path]
    audit=[]
    for ex,z in r.groupby('example'):
        parent=base.legacy.refit.active(ex,'S000_base')
        for i,row in z.iterrows():
            if row.cohort=='segmented':
                p=base.legacy.refit.EXT/f'variants/{ex}/{row.state}/lineage.csv';sources.append(p)
                lineage=pd.read_csv(p);ids=set(lineage.loc[lineage.role.eq('collapsed_block'),'parent_link_id'].astype(int))
            else:ids=set(map(int,records.loc[(ex,row.state),'descriptor_links']))
            f=parent[parent.id_link.isin(ids)]
            assert len(f)==len(ids)>0
            L=f.link_length_m.to_numpy();W=f.width.to_numpy();assert (L>0).all() and (W>0).all()
            mean=float(np.average(W,weights=L));bifs=int((f.groupby('u').size()>1).sum())
            if row.cohort=='original':assert bifs==historic.loc[(ex,row.state),'B_parent_bifurcations']
            else:
                # The edited mask must contain all outgoing branches at every
                # included split; connector exclusion cannot invent a split.
                for node,n in f.groupby('u').size().items():
                    total=int(parent.u.eq(node).sum())
                    if total>1:assert n==total
            r.at[i,'B_parent_width_mean']=mean;r.at[i,'B_parent_bifurcations']=bifs
            audit.append(dict(example=ex,state=row.state,cohort=row.cohort,n_parent_links=len(f),parent_width_mean_m=mean,parent_bifurcations=bifs))
    assert np.isfinite(r[COLUMNS]).all().all()
    r.to_parquet(OUT/'real_input.parquet',index=False);pd.DataFrame(audit).to_csv(OUT/'geometry_audit.csv',index=False)
    original=json.loads((SOURCE/'tasks.json').read_text())['tasks']
    tasks=[dict(t,model=MODEL) for t in original if t['dataset']=='real' and t['model']=='five_child_p90_with_G']
    assert len(tasks)==65
    hashes={str(p):base.sha(p) for p in sources};hashes.update(base.legacy.refit.HASHES)
    d=dict(reference='NO-P90-FOUR-2026-11-03',sources=hashes,inputs={str(OUT/'real_input.parquet'):base.sha(OUT/'real_input.parquet')},
        panels={MODEL:COLUMNS},params=old['params'],seed=old['seed'],validation=old['validation'],response='centroid_shift_hours',jobs=65,
        description='Exact four-input fixed-Manning centroid coverage selection after excluding P90 and summed equivalent width. Parent mean width is absolute length-weighted mean in metres, not a ratio. Parent bifurcation count, not a complexity difference. No G or area input.',
        scope='Ordinary parent closed B domains; full-network controls use edited parent footprints excluding unchanged connectors. Fixed RF; all rows, weights and real splits match previous comparisons.',
        caveat='Exploratory comparison on reused real validation data after interpretability exclusions; no independent final validation or hyperparameter tuning.')
    fp=hashlib.sha256(json.dumps(d,sort_keys=True).encode()).hexdigest()
    base.save(OUT/'design.json',d);base.save(OUT/'tasks.json',dict(fingerprint=fp,tasks=tasks))
    return d,fp

def fit(t,d,fp):
    base.OUT=OUT
    return base.fit_task(t,d,fp)

def report(d,fp):
    spec=importlib.util.spec_from_file_location('four_score_helpers',Path(base.__file__).with_name('summarize.py'))
    helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
    parts=[]
    for f in sorted((OUT/'checkpoints').rglob('completion.json')):
        c=json.loads(f.read_text());assert c['fingerprint']==fp
        p=f.parent/'predictions.parquet';assert base.sha(p)==c['prediction_sha256'];parts.append(pd.read_parquet(p))
    assert len(parts)==65
    prior=pd.read_parquet(SOURCE/'predictions.parquet');prior=prior[prior.dataset.eq('real')]
    p=pd.concat([prior,*parts],ignore_index=True)
    assert not p.duplicated(['strategy','repeat','model','example','state']).any()
    p.to_parquet(OUT/'predictions.parquet',index=False)
    rows=[]
    for (strategy,repeat,model),g in p.groupby(['strategy','repeat','model']):
        for population,z in [('ordinary',g[g.cohort.eq('original')]),('full_network',g[g.cohort.eq('segmented')]),('all',g)]:
            rows.append(dict(strategy=strategy,repeat=repeat,model=model,population=population,**helper.scores(z,population=='all')))
    s=pd.DataFrame(rows);s.to_csv(OUT/'repeat_scores.csv',index=False)
    summary=s.groupby(['strategy','population','model']).agg(n=('n','first'),repeats=('repeat','nunique'),r2=('r2','mean'),r2_min=('r2','min'),r2_max=('r2','max'),mae_minutes=('mae_minutes','mean')).reset_index()
    summary.to_csv(OUT/'summary.csv',index=False)
    p['absolute_error_minutes']=60*abs(p.predicted_hours-p.centroid_shift_hours)
    p.groupby(['strategy','repeat','model','cohort','example']).absolute_error_minutes.mean().rename('mae_minutes').to_csv(OUT/'per_river_errors.csv')
    lines=['# Exact no-P90 four-input RF evaluation','','Reference: NO-P90-FOUR-2026-11-03. Fixed-depth Manning centroid, same RF and real validation splits. Parent mean width and parent bifurcation count are recomputed on the existing selected domains.','','| Validation | Population | Predictor set | R² | MAE (min) |','|---|---|---|---:|---:|']
    for r in summary.itertuples():lines.append(f'| {r.strategy} | {r.population} | {r.model} | {r.r2:.4f} | {r.mae_minutes:.4f} |')
    lines+=['','No hyperparameter selection or routing reruns. The four-input set lacks G and area; differences from five-input models cannot be attributed to width alone. Real data have been inspected repeatedly, so this is exploratory validation. Ordinary comparisons and full-network controls are reported separately.','']
    (OUT/'report.md').write_text('\n'.join(lines));return summary

if __name__=='__main__':
    d,fp=prepare()
    with (OUT/'run.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        tasks=json.loads((OUT/'tasks.json').read_text())['tasks'];start=time.monotonic()
        with ProcessPoolExecutor(max_workers=2) as pool:
            jobs=[pool.submit(fit,t,d,fp) for t in tasks]
            for i,f in enumerate(as_completed(jobs),1):
                f.result();base.save(OUT/'progress.json',dict(status='running',completed_jobs=i,total_jobs=65,elapsed_seconds=time.monotonic()-start));print(f'{i}/65 fits complete',flush=True)
        print(report(d,fp).to_string(index=False),flush=True)
        for p,h in d['sources'].items():assert base.sha(p)==h,p
        base.save(OUT/'completion.json',dict(complete=True,jobs=65,fingerprint=fp,seconds=time.monotonic()-start))
        base.save(OUT/'progress.json',dict(status='complete',completed_jobs=65,total_jobs=65))
