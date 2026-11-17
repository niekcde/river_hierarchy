"""Fresh nested tuning on Manning targets under all three established splits."""
import os
for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
import argparse,fcntl,json,time,hashlib,platform
from concurrent.futures import ProcessPoolExecutor,as_completed
from pathlib import Path
import numpy as np
import pandas as pd
import sklearn
import run as refit
from augmented_tests import objective
from ten_input_outlier_audit import score

BASE=refit.BASE
DEST=BASE/'fixed_manning_rf_nested_20261029'
OLD=BASE/'alternative_validation_20261023'
MODELS=['geometry_9','depth_informed_10']

def load(p):
 p=Path(p)
 if p.suffix=='.json':return json.loads(p.read_text())
 return pd.read_parquet(p) if p.suffix=='.parquet' else pd.read_csv(p,dtype={'example':str})

def audit_task(d,t):
 ti=np.asarray(t['train_indices']);vi=np.asarray(t['test_indices']);inner=np.asarray(t['inner_fold'])
 assert not set(ti)&set(vi) and set(ti)|set(vi)==set(range(len(d)))
 assert len(inner)==len(ti) and set(inner)=={0,1,2}
 tr=d.iloc[ti].reset_index(drop=True);te=d.iloc[vi];col=t['group_col']
 assert set(tr[col]).isdisjoint(te[col])
 for fold in range(3):
  a=tr[inner!=fold];b=tr[inner==fold];assert set(a[col]).isdisjoint(b[col])
  assert set(a.cohort)==set(b.cohort)=={'original','segmented'}
 assert set(te.cohort)=={'original','segmented'}
 if col=='example':assert set(tr.example).isdisjoint(te.example)

def prepare():
 DEST.mkdir(exist_ok=True)
 current=load(refit.DEST/'input.parquet');completed=load(refit.DEST/'completion.json')
 assert refit.sha(refit.DEST/'input.parquet')==completed['files'][str(refit.DEST/'input.parquet')]
 old=load(OLD/'input.parquet');design_old=load(OLD/'design.json');splits_old=load(OLD/'splits.json')
 expected=hashlib.sha256(json.dumps(design_old,sort_keys=True).encode()).hexdigest();assert splits_old['fingerprint']==expected
 # Freeze the exact historical row ordering before reusing index-based partitions.
 keys=['example','state'];groups=old[keys+['cohort','group_id','edit_type','response_bin']].rename(columns={'response_bin':'historical_response_bin'})
 d=groups.merge(current.drop(columns=['cohort','edit_type']),on=keys,how='left',validate='one_to_one')
 assert len(d)==2185 and d.example.nunique()==15 and d.group_id.nunique()==450
 assert np.array_equal(d[keys].to_numpy(),old[keys].to_numpy())
 check=d.merge(current[keys+['cohort','edit_type','centroid_shift_hours']],on=keys,suffixes=('_d','_current'),validate='one_to_one')
 for col in ['cohort','edit_type','centroid_shift_hours']:np.testing.assert_array_equal(check[col+'_d'],check[col+'_current'])
 panels=design_old['panels'];candidates=design_old['candidates'];assert len(candidates)==6
 assert np.isfinite(d[list(set(sum(panels.values(),[])))+['centroid_shift_hours']]).all().all()
 nine=load(BASE/'nine_input_tuning/selections.json');nine={x['held']:x for x in nine if x['variant']=='same_grid'}
 ten={x['held']:x for x in load(BASE/'ten_input_tuning_20261023/selections.json')}
 tasks=[]
 for fold,ex in enumerate(sorted(d.example.unique())):
  ti=np.flatnonzero(d.example.ne(ex));vi=np.flatnonzero(d.example.eq(ex));tr=d.iloc[ti]
  assert nine[ex]['inner_groups']==ten[ex]['inner_groups']
  inner=np.full(len(tr),-1)
  for i,g in enumerate(nine[ex]['inner_groups']):
   assert ex not in g['train']+g['validation'];inner[tr.example.isin(g['validation'])]=i
   assert set(tr.loc[~tr.example.isin(g['validation']),'example'])==set(g['train'])
  tasks.append(dict(strategy='leave_one_network_out',repeat=0,fold=fold,held=ex,group_col='example',train_indices=ti.tolist(),test_indices=vi.tolist(),inner_fold=inner.tolist()))
 for original in splits_old['tasks']:
  t=dict(original);t['group_col']='example' if t['strategy']=='network_balanced' else 'group_id';tasks.append(t)
 assert len(tasks)==65
 for t in tasks:audit_task(d,t)
 # Every repetition predicts every comparison exactly once.
 for strategy in ['leave_one_network_out','network_balanced','compound_mixed']:
  repetitions=sorted({t['repeat'] for t in tasks if t['strategy']==strategy})
  assert len(repetitions)==(1 if strategy=='leave_one_network_out' else 5)
  for repeat in repetitions:
   indices=sum([t['test_indices'] for t in tasks if t['strategy']==strategy and t['repeat']==repeat],[])
   assert sorted(indices)==list(range(len(d)))
 d['current_response_bin']=pd.cut(d.centroid_shift_hours*60,[-np.inf,-60,-5,-1,1,5,60,np.inf],labels=['advance_over60','advance5to60','advance1to5','near_zero','delay1to5','delay5to60','delay_over60'])
 deps=[refit.DEST/'input.parquet',refit.DEST/'preparation.json',refit.DEST/'completion.json',OLD/'input.parquet',OLD/'design.json',OLD/'splits.json',
  BASE/'nine_input_tuning/selections.json',BASE/'ten_input_tuning_20261023/selections.json',
  Path(__file__),Path(refit.__file__),Path(refit.tuning.b.__file__),Path(refit.tuning.b.engine.__file__),Path(refit.tuning.b.forest.__file__),
  refit.ROOT/'RAPID/investigations/cap_investigation_20261020/augmented_tests.py',refit.ROOT/'RAPID/investigations/cap_investigation_20261020/ten_input_outlier_audit.py']
 design=dict(sources={str(p):refit.sha(p) for p in deps},panels=panels,candidates=candidates,rows=len(d),networks=15,
  seed=refit.tuning.b.engine.SEED,versions=dict(python=platform.python_version(),numpy=np.__version__,pandas=pd.__version__,sklearn=sklearn.__version__),
  outer_tasks=65,model_outer_jobs=130,forest_fits=2470,
  validation='All three strategies freshly tuned on Manning responses. Six candidates x three inner folds plus selected outer refit. Frozen historical outer AND inner partitions, identical across the two predictor sets.',
  grouping='Whole network for leave-one-out and grouped-network. Compound mixed retains435 regular family groups plus15 full-network singleton exceptions.',
  balance='Historical partitions were response/type-balanced on earlier BASED outcomes. Do not regenerate partitions after seeing new model errors; report current margins separately.',
  weighting='Network-balanced training. Equal average of regular and segmented network-balanced relative error, denominator floor1minute, for selection.',
  reporting='Score each complete repetition separately; mean/range across repeats. No ensemble-before-scoring. Six candidates also for nine-input LOEO, matching previous expanded-grid control.',
  caveats=['Compound-mixed is known-network related-state prediction; full-collapse singleton permits overlap with constituent regular states.',
   'Exploratory reused networks/features, not external validation; repeated-fold spread is not a confidence interval.',
   'Depth-informed features assume held-network reference calibration is available.'],production_modified=False)
 fp=hashlib.sha256(json.dumps(design,sort_keys=True).encode()).hexdigest()
 if (DEST/'design.json').exists():assert load(DEST/'design.json')==design,'Use a fresh version if inputs change'
 refit.save(DEST/'design.json',design);refit.save(DEST/'splits.json',dict(fingerprint=fp,tasks=tasks))
 d.to_parquet(DEST/'input.parquet',index=False)
 audits=[]
 for t in tasks:
  te=d.iloc[t['test_indices']];tr=d.iloc[t['train_indices']]
  audits.append(dict(strategy=t['strategy'],repeat=t['repeat'],fold=t['fold'],test_rows=len(te),train_rows=len(tr),
   test_networks=te.example.nunique(),test_groups=te.group_id.nunique(),same_network_train_rows=int(te.example.isin(tr.example).sum()),
   current_delays_gt1min=int((te.centroid_shift_hours*60>1+1e-6).sum()),segmented_test=int(te.cohort.eq('segmented').sum())))
 pd.DataFrame(audits).to_csv(DEST/'fold_audit.csv',index=False)
 return design,fp,tasks

def fit_task(t,model,design,fp):
 d=load(DEST/'input.parquet');audit_task(d,t)
 folder=DEST/'checkpoints'/t['strategy']/f"r{t['repeat']}"/f"f{t['fold']}"/model
 final=folder/'outer.json'
 if final.exists():assert load(final)['fingerprint']==fp;return str(final)
 tr=d.iloc[t['train_indices']].reset_index(drop=True);te=d.iloc[t['test_indices']];inner=np.asarray(t['inner_fold'])
 b=refit.tuning.b;b.COLUMNS=list(design['panels'][model]);ranked=[];start=time.monotonic()
 for i,candidate in enumerate(design['candidates']):
  path=folder/f'candidate_{i}.json'
  if path.exists():
   trial=load(path);assert trial['fingerprint']==fp
  else:
   pred=np.full(len(tr),np.nan)
   for f in range(3):
    a=tr[inner!=f];v=tr[inner==f];assert set(a[t['group_col']]).isdisjoint(v[t['group_col']])
    artifact=b.fit(a,candidate['params'],jobs=1);pred[inner==f]=b.forest.predict(artifact,v)
   assert np.isfinite(pred).all();loss,cohorts=objective(tr,pred)
   trial=dict(fingerprint=fp,index=i,name=candidate['name'],params=candidate['params'],score=loss,cohort_scores=cohorts)
   refit.save(path,trial)
  ranked.append(trial)
 best=min(ranked,key=lambda x:(x['score'],x['index']))
 artifact=b.fit(tr,best['params'],jobs=1);pred=b.forest.predict(artifact,te);assert np.isfinite(pred).all()
 rows=[dict(strategy=t['strategy'],repeat=t['repeat'],fold=t['fold'],model=model,example=r.example,state=r.state,cohort=r.cohort,
  edit_type=r.edit_type,centroid_shift_hours=r.centroid_shift_hours,predicted_hours=float(v),same_network_in_training=r.example in set(tr.example)) for r,v in zip(te.itertuples(),pred)]
 refit.save(final,dict(fingerprint=fp,best=best,seconds=time.monotonic()-start,rows=rows))
 return str(final)

def summarize(fp,complete=False):
 rows=[];selections=[]
 for path in sorted((DEST/'checkpoints').glob('*/*/*/*/outer.json')):
  result=load(path);assert result['fingerprint']==fp;rows.extend(result['rows']);r=result['rows'][0]
  selections.append(dict(strategy=r['strategy'],repeat=r['repeat'],fold=r['fold'],model=r['model'],**result['best']))
 if not rows:return
 p=pd.DataFrame(rows);assert not p.duplicated(['strategy','repeat','model','example','state']).any()
 if complete:assert len(selections)==130 and len(p)==48070
 p.to_parquet(DEST/('predictions.parquet' if complete else 'partial_predictions.parquet'),index=False)
 refit.save(DEST/'selections.json',selections)
 scores=[];delays=[];types=[];network=[]
 for (strategy,repeat,model),g in p.groupby(['strategy','repeat','model']):
  if len(g)!=2185:continue
  for cohort,z in [('all',g),('original',g[g.cohort=='original']),('segmented',g[g.cohort=='segmented'])]:
   scores.append(dict(strategy=strategy,repeat=repeat,model=model,cohort=cohort,**score(z,'equal_cohort_network' if cohort=='all' else 'equal_network')))
  for kind,z in g.groupby('edit_type'):types.append(dict(strategy=strategy,repeat=repeat,model=model,edit_type=kind,**score(z,'equal_network')))
  for (cohort,ex),z in g.groupby(['cohort','example']):network.append(dict(strategy=strategy,repeat=repeat,model=model,cohort=cohort,example=ex,**score(z,'equal_network')))
  actual=g.centroid_shift_hours*60>1+1e-6;called=g.predicted_hours*60>1+1e-6;tp=int((actual&called).sum())
  delays.append(dict(strategy=strategy,repeat=repeat,model=model,actual_delays=int(actual.sum()),predicted_delays=int(called.sum()),
   detected=tp,missed=int((actual&~called).sum()),false_delays=int((~actual&called).sum()),recall=tp/actual.sum(),precision=tp/called.sum() if called.sum() else None))
 if scores:
  s=pd.DataFrame(scores);s.to_csv(DEST/'repeat_scores.csv',index=False)
  agg=s.groupby(['strategy','model','cohort'])[['r2','mae_minutes','relative_error_pct']].agg(['mean','min','max','std'])
  agg.columns=['_'.join(c) for c in agg.columns];agg.reset_index().to_csv(DEST/'summary.csv',index=False)
  pd.DataFrame(delays).to_csv(DEST/'delay_detection.csv',index=False)
  pd.DataFrame(types).to_csv(DEST/'type_scores.csv',index=False);pd.DataFrame(network).to_csv(DEST/'network_scores.csv',index=False)
 if complete:
  for path,h in load(DEST/'design.json')['sources'].items():assert refit.sha(path)==h,path
  refit.save(DEST/'completion.json',dict(complete=True,fingerprint=fp,model_outer_jobs=130,forest_fits=2470,predictions=48070,production_modified=False,
   files={str(DEST/x):refit.sha(DEST/x) for x in ['predictions.parquet','repeat_scores.csv','summary.csv','delay_detection.csv']}))

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--run',action='store_true');ap.add_argument('--workers',type=int,default=2);args=ap.parse_args()
 design,fp,tasks=prepare();print('Verified all3strategies:130model/foldjobs,2470freshforestfits',flush=True)
 if not args.run:return
 with (DEST/'run.lock').open('a+') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);start=time.monotonic()
  with ProcessPoolExecutor(max_workers=args.workers) as pool:
   pending=[pool.submit(fit_task,t,model,design,fp) for t in tasks for model in MODELS]
   for i,future in enumerate(as_completed(pending),1):
    result=future.result();elapsed=time.monotonic()-start
    refit.save(DEST/'progress.json',dict(status='running',completed_model_outer_jobs=i,total_model_outer_jobs=len(pending),elapsed_seconds=elapsed,last_checkpoint=result,workers=args.workers))
    print(i,'/130 complete',result,'elapsed minutes',round(elapsed/60,1),flush=True)
    if i%5==0:summarize(fp)
  summarize(fp,complete=True)
  refit.save(DEST/'progress.json',dict(status='complete',completed_model_outer_jobs=130,total_model_outer_jobs=130,elapsed_seconds=time.monotonic()-start,workers=args.workers))

if __name__=='__main__':main()
