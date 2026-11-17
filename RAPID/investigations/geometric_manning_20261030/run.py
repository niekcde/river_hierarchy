"""Paired addition of whole-network geometry-only G_M on frozen compound splits."""
import os
for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
from pathlib import Path
import sys,json,hashlib,time,importlib.util
from concurrent.futures import ProcessPoolExecutor,as_completed
ROOT=Path(__file__).resolve().parents[3]
spec=importlib.util.spec_from_file_location('hydrography_baseline',ROOT/'RAPID/investigations/hydrography_rf_20261029/run.py')
hydro=importlib.util.module_from_spec(spec);spec.loader.exec_module(hydro)
nested=hydro.nested;refit=hydro.refit
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from rapid_tools.geometric_manning import manning_geometry_factor
from rapid_tools import geometric_manning
OUT=Path('/Volumes/SyntheticResults20261027/analysis/real_geometry_manning_20261030')
OLD=hydro.DEST
COLUMN='delta_G_manning'
PANEL='geometry_ratios_9'

def load(p):return nested.load(p)
def save(p,v):refit.save(p,v)

def prepare():
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'design.json').exists():
        design=load(OUT/'design.json')
        for p,h in design['sources'].items():assert refit.sha(p)==h,p
        assert refit.sha(OUT/'input.parquet')==design['input_sha256']
        return design,hashlib.sha256(json.dumps(design,sort_keys=True).encode()).hexdigest(),load(OLD/'splits.json')['tasks']
    olddesign=load(OLD/'design.json');completion=load(OLD/'completion.json');splits=load(OLD/'splits.json')
    oldfp=hashlib.sha256(json.dumps(olddesign,sort_keys=True).encode()).hexdigest()
    assert completion['complete'] and completion['fingerprint']==oldfp==splits['fingerprint']
    for p,h in olddesign['sources'].items():assert refit.sha(p)==h,p
    for p,h in completion['files'].items():assert refit.sha(p)==h,p
    d=load(OLD/'input.parquet');assert refit.sha(OLD/'input.parquet')==olddesign['input_sha256']
    assert len(d)==2185 and d.group_id.nunique()==450
    rows=[];source={str(OLD/n):refit.sha(OLD/n) for n in ['input.parquet','design.json','completion.json','splits.json','predictions.parquet']}
    for ex,g in d.groupby('example'):
        parent=refit.active(ex,'S000_base');gp=manning_geometry_factor(parent)
        for r in g.itertuples():
            child=refit.active(ex,r.state);gc=manning_geometry_factor(child)
            rows.append(dict(example=ex,state=r.state,parent_G_manning=gp,child_G_manning=gc,delta_G_manning=gc-gp,parent_links=len(parent),child_links=len(child)))
        pd.DataFrame(rows).to_parquet(OUT/'geometry.partial.parquet',index=False)
        print('Whole-network geometry',ex,len(g),flush=True)
    geo=pd.DataFrame(rows);geo.to_parquet(OUT/'geometry.parquet',index=False)
    d=d.merge(geo,on=['example','state'],validate='one_to_one');assert np.isfinite(d[COLUMN]).all()
    original=load(OLD/'input.parquet');pd.testing.assert_frame_equal(d[original.columns],original)
    tasks=splits['tasks'];assert len(tasks)==25
    for t in tasks:nested.audit_task(d,t)
    cols=olddesign['panels'][PANEL];assert len(cols)==9
    params={}
    for t in tasks:
        p=OLD/'checkpoints'/t['strategy']/f"r{t['repeat']}"/f"f{t['fold']}"/PANEL/'outer.json'
        old=load(p);assert old['fingerprint']==oldfp
        params[f"{t['repeat']}_{t['fold']}"]=old['best']['params'];source[str(p)]=refit.sha(p)
    d.to_parquet(OUT/'input.parquet',index=False)
    source.update(refit.HASHES)
    for p in [Path(__file__),Path(geometric_manning.__file__),Path(nested.__file__),Path(refit.__file__),Path(refit.tuning.b.__file__),Path(refit.tuning.b.forest.__file__),Path(refit.tuning.b.engine.__file__),ROOT/'RAPID/src/rapid_tools/current_domains.py']:
        source[str(p)]=refit.sha(p)
    design=dict(sources=source,input_sha256=refit.sha(OUT/'input.parquet'),baseline_panel=PANEL,baseline_columns=cols,new_columns=cols+[COLUMN],params_by_fold=params,
        feature='G_M=sum over WHOLE network L*B^.4*f^.6*S^(-.3); f independently recomputed from width/connectivity for each parent and child. No n, depth, simulated discharge or response used.',
        target=olddesign['target'],validation=olddesign['validation'],caveats=olddesign['caveats'],
        addition_test='Paired ablation: reuse baseline nine-input inner-selected settings separately in each outer fold; add G_M without retuning. Same seeds, weights and preprocessing. This does not evaluate a freshly tuned ten-input procedure.',
        baseline='Training-only network-weighted affine regression on delta_G_manning, one global slope/intercept per outer training fold.',
        rows=len(d),tasks=25,production_modified=False,versions=olddesign['versions'])
    save(OUT/'design.json',design);fp=hashlib.sha256(json.dumps(design,sort_keys=True).encode()).hexdigest()
    save(OUT/'splits.json',dict(fingerprint=fp,tasks=tasks));return design,fp,tasks


def task(t,design,fp):
    dest=OUT/'checkpoints'/f"r{t['repeat']}_f{t['fold']}.json"
    if dest.exists():assert load(dest)['fingerprint']==fp;return str(dest)
    start=time.monotonic();d=load(OUT/'input.parquet');nested.audit_task(d,t)
    tr,te=d.iloc[t['train_indices']],d.iloc[t['test_indices']]
    b=refit.tuning.b;params=design['params_by_fold'][f"{t['repeat']}_{t['fold']}"]
    # Verify exact reproducibility of baseline pipeline once; reuse its verified OOF predictions elsewhere.
    if t['repeat']==0 and t['fold']==0:
        b.COLUMNS=design['baseline_columns'];fit=b.fit(tr,params,jobs=1);pred=b.forest.predict(fit,te)
        old=load(OLD/'predictions.parquet');old=old[(old.model==PANEL)&(old.repeat==0)&(old.fold==0)].set_index(['example','state'])
        expected=old.loc[pd.MultiIndex.from_frame(te[['example','state']]),'predicted_hours'].to_numpy()
        np.testing.assert_allclose(pred,expected,rtol=1e-10,atol=1e-10)
    b.COLUMNS=design['new_columns'];fit=b.fit(tr,params,jobs=1);p_rf=b.forest.predict(fit,te)
    linear=LinearRegression().fit(tr[[COLUMN]],tr.centroid_shift_hours,sample_weight=b.engine.weights(tr));p_lin=linear.predict(te[[COLUMN]])
    rows=[]
    for model,pred in [('geometry_9_plus_GM',p_rf),('linear_GM',p_lin)]:
        for r,v in zip(te.itertuples(),pred):rows.append(dict(strategy=t['strategy'],repeat=t['repeat'],fold=t['fold'],model=model,example=r.example,state=r.state,cohort=r.cohort,edit_type=r.edit_type,centroid_shift_hours=r.centroid_shift_hours,predicted_hours=float(v)))
    save(dest,dict(fingerprint=fp,seconds=time.monotonic()-start,params=params,linear_coefficient=float(linear.coef_[0]),linear_intercept=float(linear.intercept_),rows=rows))
    return str(dest)


def summarize(design,fp):
    records=[];files=sorted((OUT/'checkpoints').glob('*.json'));assert len(files)==25
    for p in files:
        r=load(p);assert r['fingerprint']==fp;records.extend(r['rows'])
    old=load(OLD/'predictions.parquet');old=old[old.model.isin([PANEL,'linear_length','linear_geometry_J'])]
    pred=pd.concat([old,pd.DataFrame(records)],ignore_index=True);assert not pred.duplicated(['repeat','model','example','state']).any()
    assert len(pred)==2185*5*5;pred.to_parquet(OUT/'predictions.parquet',index=False)
    rows=[]
    for (repeat,model),g in pred.groupby(['repeat','model']):
        assert len(g)==2185
        for cohort,z in [('all',g),('original',g[g.cohort=='original']),('segmented',g[g.cohort=='segmented'])]:
            rows.append(dict(repeat=repeat,model=model,cohort=cohort,**nested.score(z,'equal_cohort_network' if cohort=='all' else 'equal_network')))
    scores=pd.DataFrame(rows);scores.to_csv(OUT/'repeat_scores.csv',index=False)
    summary=scores.groupby(['model','cohort'])[['r2','mae_minutes','relative_error_pct']].agg(['mean','min','max'])
    summary.columns=['_'.join(c) for c in summary.columns];summary.reset_index().to_csv(OUT/'summary.csv',index=False)
    paired=scores[scores.model=='geometry_9_plus_GM'].merge(scores[scores.model==PANEL],on=['repeat','cohort'],suffixes=('_plus','_nine'),validate='one_to_one')
    for metric in ['r2','mae_minutes','relative_error_pct']:paired['delta_'+metric]=paired[metric+'_plus']-paired[metric+'_nine']
    paired.to_csv(OUT/'paired_changes.csv',index=False)
    for p,h in design['sources'].items():assert refit.sha(p)==h,p
    save(OUT/'completion.json',dict(complete=True,fingerprint=fp,outer_tasks=25,new_forest_fits=26,linear_fits=25,production_modified=False,outputs={str(OUT/n):refit.sha(OUT/n) for n in ['predictions.parquet','summary.csv','paired_changes.csv']}))
    print(summary.to_string(),flush=True)


def main():
    design,fp,tasks=prepare();start=time.monotonic()
    with ProcessPoolExecutor(max_workers=2) as pool:
        pending=[pool.submit(task,t,design,fp) for t in tasks]
        for i,f in enumerate(as_completed(pending),1):
            p=f.result();save(OUT/'progress.json',dict(status='running',completed=i,total=25,elapsed_seconds=time.monotonic()-start,last=p));print('Completed',i,'/25',flush=True)
    summarize(design,fp);save(OUT/'progress.json',dict(status='complete',completed=25,total=25,elapsed_seconds=time.monotonic()-start))
if __name__=='__main__':main()
