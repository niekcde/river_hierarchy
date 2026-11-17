"""Add only Manning geometric accumulation to the frozen no-P90 four."""
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
SOURCE=ROOT/'RAPID/outputs/no_p90_four_20261103'
OUT=ROOT/'RAPID/outputs/no_p90_four_plus_g_20261103'
COLUMNS=['B_delta_route_length_mean','parent_slope_slow','B_parent_width_mean','B_parent_bifurcations','delta_G_manning']
MODEL='synthetic_selected_four_no_p90_plus_G'

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
    import shutil
    shutil.copy2(SOURCE/'real_input.parquet',OUT/'real_input.parquet')
    r=pd.read_parquet(OUT/'real_input.parquet');assert len(r)==2185
    assert np.isfinite(r[COLUMNS]).all().all()
    assert COLUMNS[:-1]==old['panels']['synthetic_selected_four_no_p90']
    tasks=[dict(t,model=MODEL) for t in json.loads((SOURCE/'tasks.json').read_text())['tasks']]
    assert len(tasks)==65
    files=[Path(__file__),Path(base.__file__),SOURCE/'design.json',SOURCE/'completion.json',SOURCE/'predictions.parquet',SOURCE/'tasks.json',SOURCE/'real_input.parquet']
    d=dict(reference='NO-P90-FOUR-PLUS-G-2026-11-03',sources={str(p):base.sha(p) for p in files},
        inputs={str(OUT/'real_input.parquet'):base.sha(OUT/'real_input.parquet')},
        panels={MODEL:COLUMNS},params=old['params'],seed=old['seed'],validation=old['validation'],response='centroid_shift_hours',jobs=65,
        description='Add only delta_G_manning to the exact four-input no-P90 panel; preserve feature order of the original four, all rows, geometry, responses, folds, weights, RF and seed.',
        scope=old['scope'],caveat='Exploratory incremental-information test on reused validation data. No routing or feature-selection reruns.')
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
    keys=['strategy','repeat','population']
    paired=s[s.model.eq(MODEL)].merge(s[s.model.eq('synthetic_selected_four_no_p90')],on=keys,suffixes=('_plus_G','_four'),validate='one_to_one')
    assert (paired.n_plus_G==paired.n_four).all()
    paired['delta_r2']=paired.r2_plus_G-paired.r2_four
    paired['delta_mae_minutes']=paired.mae_minutes_plus_G-paired.mae_minutes_four
    paired.to_csv(OUT/'paired_changes.csv',index=False)
    p['absolute_error_minutes']=60*abs(p.predicted_hours-p.centroid_shift_hours)
    p.groupby(['strategy','repeat','model','cohort','example']).absolute_error_minutes.mean().rename('mae_minutes').to_csv(OUT/'per_river_errors.csv')
    lines=['# Add G to the exact no-P90 four-input RF','','Reference: NO-P90-FOUR-PLUS-G-2026-11-03. Fixed-depth Manning centroid, same RF and real validation splits. Add only delta_G_manning; prepared input data reused byte-for-byte.','','| Validation | Population | Predictor set | R² | MAE (min) |','|---|---|---|---:|---:|']
    for r in summary.itertuples():lines.append(f'| {r.strategy} | {r.population} | {r.model} | {r.r2:.4f} | {r.mae_minutes:.4f} |')
    lines+=['','No hyperparameter selection or routing reruns. The new five-input set differs from the previous exact four only by adding delta_G_manning. Comparisons with other five-input panels also change width and topology representations. Real data have been inspected repeatedly, so this is exploratory validation. Ordinary comparisons and full-network controls are reported separately.','']
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
