"""Replace only complexity change with allocated entropy change, without G."""
import os
for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
os.environ.setdefault('MPLCONFIGDIR','/tmp/no-p90-four-mpl')
from pathlib import Path
import sys,json,hashlib,time,shutil,fcntl,importlib.util
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures,StandardScaler
from sklearn.linear_model import Ridge
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'RAPID/investigations/streamlined_predictors_20261103'))
import run_comparison as base
SOURCE=ROOT/'RAPID/outputs/mean_width_compact_20261103'
OUT=ROOT/'RAPID/outputs/entropy_replacement_20261104'
BASEMODEL='five_mean_width_without_G'

def prepare():
    OUT.mkdir(exist_ok=True)
    if (OUT/'design.json').exists():
        d=json.loads((OUT/'design.json').read_text())
        for p,h in {**d['sources'],**d['inputs']}.items():assert base.sha(p)==h,p
        return d,hashlib.sha256(json.dumps(d,sort_keys=True).encode()).hexdigest()
    old=json.loads((SOURCE/'design.json').read_text());receipt=json.loads((SOURCE/'completion.json').read_text());assert receipt['complete']
    assert base.sha(SOURCE/'predictions.parquet')==receipt['files'][str(SOURCE/'predictions.parquet')]
    for p,h in old['inputs'].items():assert base.sha(p)==h,p
    original=old['panels'][BASEMODEL]
    entropy=['delta_entropy' if v=='path_delta_topologic_complexity_score' else v for v in original]
    assert sum(a!=b for a,b in zip(original,entropy))==1 and 'delta_G_manning' not in entropy
    panels={'complexity_ridge':original,'entropy_ridge':entropy,'entropy_RF':entropy}
    shutil.copy2(SOURCE/'real_input.parquet',OUT/'real_input.parquet')
    frame=pd.read_parquet(OUT/'real_input.parquet');assert len(frame)==2185
    assert np.isfinite(frame[list(set(original+entropy))]).all().all()
    reference=[t for t in json.loads((SOURCE/'tasks.json').read_text())['tasks'] if t['dataset']=='real' and t['model']==BASEMODEL]
    assert len(reference)==65
    tasks=[dict(t,model=m) for m in panels for t in reference]
    files=[Path(__file__),Path(base.__file__),SOURCE/'design.json',SOURCE/'tasks.json',SOURCE/'completion.json',SOURCE/'predictions.parquet',SOURCE/'real_input.parquet']
    d=dict(reference='ENTROPY-REPLACEMENT-2026-11-04',sources={str(p):base.sha(p) for p in files},
        inputs={str(OUT/'real_input.parquet'):base.sha(OUT/'real_input.parquet')},panels=panels,params=old['params'],seed=old['seed'],validation=old['validation'],
        response='Fixed-depth Manning centroid shift',jobs=195,rf_jobs=65,ridge_jobs=130,alpha_per_row=1e-4,
        scope='Same existing matched domains and edited-footprint conventions; use existing delta_entropy, with width-based allocation and natural logarithms. No geometry or routing recomputation.',
        description='No-G panel: delta allocated mean route length, parent slope descriptor, log1p mean-width ratio, topology/entropy change, delta planform-area proxy. Only topology descriptor changes.',
        ridge='Degree-two main effects, squares and interactions; weighted training-only standardization; alpha=1e-4*sum(weights), mean-one equal-river weights; unpenalized intercept. Fixed penalty, no tuning.',
        caveat='Exploratory comparisons on reused real data. Both ridge baseline and replacement newly fitted. Completed complexity RF predictions reused after hash verification.')
    fp=hashlib.sha256(json.dumps(d,sort_keys=True).encode()).hexdigest()
    base.save(OUT/'design.json',d);base.save(OUT/'tasks.json',dict(fingerprint=fp,tasks=tasks))
    return d,fp

def fit(t,d,fp):
    base.OUT=OUT
    if t['model'].endswith('_RF'):return base.fit_task(t,d,fp)
    folder=OUT/'ridge_checkpoints'/t['strategy']/f"r{t['repeat']}"/f"f{t['fold']}"/t['model']
    final=folder/'completion.json';predfile=folder/'predictions.parquet'
    if final.exists():
        c=json.loads(final.read_text());assert c['fingerprint']==fp and c['prediction_sha256']==base.sha(predfile)
        return str(final)
    frame=pd.read_parquet(OUT/'real_input.parquet');tr=frame.iloc[t['train_indices']];te=frame.iloc[t['test_indices']]
    columns=d['panels'][t['model']];poly=PolynomialFeatures(2,include_bias=False)
    X=poly.fit_transform(tr[columns]);Xt=poly.transform(te[columns])
    y=tr.centroid_shift_hours.to_numpy();w=base.legacy.refit.tuning.b.engine.weights(tr)
    sc=StandardScaler().fit(X,sample_weight=w);good=sc.var_>0;X=sc.transform(X)[:,good];Xt=sc.transform(Xt)[:,good]
    alpha=d['alpha_per_row']*w.sum()
    model=Ridge(alpha=alpha,solver='svd').fit(X,y,sample_weight=w);pred=model.predict(Xt)
    xm=np.average(X,axis=0,weights=w);ym=np.average(y,weights=w);xc=X-xm
    beta=np.linalg.solve(xc.T@(w[:,None]*xc)+alpha*np.eye(X.shape[1]),xc.T@(w*(y-ym)))
    np.testing.assert_allclose(pred,ym+(Xt-xm)@beta,rtol=1e-7,atol=1e-7)
    assert np.isfinite(pred).all()
    z=te[['example','state','cohort','edit_type','centroid_shift_hours']].copy();z['predicted_hours']=pred
    for k in ['dataset','strategy','support','repeat','fold','model']:z[k]=t[k]
    folder.mkdir(parents=True,exist_ok=True);tmp=predfile.with_suffix('.tmp.parquet');z.to_parquet(tmp,index=False);tmp.replace(predfile)
    base.save(final,dict(fingerprint=fp,prediction_sha256=base.sha(predfile),normal_equation_check=True))
    return str(final)

def report(d,fp):
    parts=[]
    paths=list((OUT/'checkpoints').rglob('completion.json'))+list((OUT/'ridge_checkpoints').rglob('completion.json'))
    assert len(paths)==195
    for f in paths:
        c=json.loads(f.read_text());p=f.parent/'predictions.parquet'
        assert c['fingerprint']==fp and c['prediction_sha256']==base.sha(p)
        parts.append(pd.read_parquet(p))
    prior=pd.read_parquet(SOURCE/'predictions.parquet');prior=prior[prior.dataset.eq('real')&prior.model.eq(BASEMODEL)].copy();prior['model']='complexity_RF'
    p=pd.concat([prior,*parts],ignore_index=True)
    assert not p.duplicated(['strategy','repeat','model','example','state']).any()
    p.to_parquet(OUT/'predictions.parquet',index=False)
    spec=importlib.util.spec_from_file_location('entropy_score_helpers',Path(base.__file__).with_name('summarize.py'));helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
    rows=[]
    for (strategy,repeat,model),g in p.groupby(['strategy','repeat','model']):
        for population,z in [('ordinary',g[g.cohort.eq('original')]),('full_network',g[g.cohort.eq('segmented')]),('all',g)]:
            rows.append(dict(strategy=strategy,repeat=repeat,model=model,population=population,**helper.scores(z,population=='all')))
    s=pd.DataFrame(rows);s.to_csv(OUT/'repeat_scores.csv',index=False)
    summary=s.groupby(['strategy','population','model']).agg(n=('n','first'),repeats=('repeat','nunique'),r2=('r2','mean'),r2_min=('r2','min'),r2_max=('r2','max'),mae_minutes=('mae_minutes','mean')).reset_index();summary.to_csv(OUT/'summary.csv',index=False)
    paired=[]
    for estimator in ['RF','ridge']:
        z=s[s.model.eq('entropy_'+estimator)].merge(s[s.model.eq('complexity_'+estimator)],on=['strategy','repeat','population'],suffixes=('_entropy','_complexity'),validate='one_to_one')
        assert (z.n_entropy==z.n_complexity).all();z['delta_r2']=z.r2_entropy-z.r2_complexity;z['delta_mae_minutes']=z.mae_minutes_entropy-z.mae_minutes_complexity;z['estimator']=estimator;paired.append(z)
    pd.concat(paired).to_csv(OUT/'paired_changes.csv',index=False)
    p['absolute_error_minutes']=60*abs(p.predicted_hours-p.centroid_shift_hours)
    p.groupby(['strategy','repeat','model','cohort','example']).absolute_error_minutes.mean().rename('mae_minutes').to_csv(OUT/'per_river_errors.csv')
    p.sort_values('absolute_error_minutes',ascending=False).groupby(['strategy','model']).head(10).to_csv(OUT/'largest_errors.csv',index=False)
    lines=['# Complexity change versus allocated entropy change','','Reference ENTROPY-REPLACEMENT-2026-11-04. Same five-input no-G panel, replacing only complexity change with entropy change. Fifth input remains area-proxy change. Same populations, targets, weights and saved splits; quadratic ridge and RF trained separately on each real training fold.','','| Validation | Population | Model | R² | MAE (min) |','|---|---|---|---:|---:|']
    for r in summary.itertuples():lines.append(f'| {r.strategy} | {r.population} | {r.model} | {r.r2:.4f} | {r.mae_minutes:.4f} |')
    lines+=['','Ordinary n=2,170; full-network controls n=15. Ridge penalty fixed at 1e-4 per training row, not tuned. Twenty quadratic terms plus intercept. No G input, no routing reruns, no exclusions or clipping of large prediction errors. Repeated inspection of real results makes this exploratory rather than independent validation.','']
    (OUT/'report.md').write_text('\n'.join(lines));return summary

if __name__=='__main__':
    d,fp=prepare()
    with (OUT/'run.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);start=time.monotonic()
        tasks=json.loads((OUT/'tasks.json').read_text())['tasks']
        with ProcessPoolExecutor(max_workers=2) as pool:
            jobs=[pool.submit(fit,t,d,fp) for t in tasks]
            for i,f in enumerate(as_completed(jobs),1):
                f.result();base.save(OUT/'progress.json',dict(status='running',completed_jobs=i,total_jobs=195,elapsed_seconds=time.monotonic()-start));print(f'{i}/195 fits complete',flush=True)
        print(report(d,fp).to_string(index=False))
        for p,h in d['sources'].items():assert base.sha(p)==h,p
        base.save(OUT/'completion.json',dict(complete=True,jobs=195,seconds=time.monotonic()-start,fingerprint=fp))
        base.save(OUT/'progress.json',dict(status='complete',completed_jobs=195,total_jobs=195))
