"""Numerically safe monotone width representation; same nested RF protocol."""
import compare as raw
import json
import hashlib
import time
import fcntl
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
import pandas as pd

OUT=raw.ROOT/'RAPID/outputs/matched_centroid_sets_20261103_logwidth'
WIDTH='path_parent_equivalent_width'
LOG='log1p__path_parent_equivalent_width'
PANELS={name:[LOG if c==WIDTH else c for c in cols] for name,cols in raw.PANELS.items()}

def prepare():
    design,fp,tasks=raw.prepare()
    raw.check_baseline(tasks)
    OUT.mkdir(exist_ok=True)
    if (OUT/'design.json').exists():
        safe=raw.nested.load(OUT/'design.json')
        for p,h in safe['sources'].items():assert raw.legacy.refit.sha(p)==h,p
        assert raw.legacy.refit.sha(OUT/'input.parquet')==safe['input_sha256']
        return safe,raw.nested.load(OUT/'splits.json')['fingerprint'],tasks
    data=pd.read_parquet(raw.OUT/'input.parquet')
    data[LOG]=np.log1p(data[WIDTH])
    assert np.isfinite(data[LOG]).all()
    data.to_parquet(OUT/'input.parquet',index=False)
    sources=design['sources'] | {str(Path(__file__)):raw.legacy.refit.sha(Path(__file__)),
        str(raw.OUT/'input.parquet'):raw.legacy.refit.sha(raw.OUT/'input.parquet')}
    safe=dict(design,panels=PANELS,sources=sources,input_sha256=raw.legacy.refit.sha(OUT/'input.parquet'),
        numerical_correction='log1p of positive equivalent path width before unchanged training-only StandardScaler. Raw width collapsed to 1–2 float32 values among hundreds of ordinary-domain values. This monotone transformation preserves geometric ordering and is chosen from input precision diagnostics, not outcome scores.',
        raw_diagnostic='First repeated five-fold partition only, retained separately; not five-repeat headline results.')
    fp=hashlib.sha256(json.dumps(safe,sort_keys=True).encode()).hexdigest()
    raw.legacy.refit.save(OUT/'design.json',safe)
    raw.legacy.refit.save(OUT/'splits.json',dict(fingerprint=fp,tasks=tasks))
    checks=[]
    for t in tasks:
        tr=data.iloc[t['train_indices']]
        pre=raw.legacy.refit.tuning.b.engine.Preprocess(PANELS['centroid_with_G'],None,None).fit(tr,tr.centroid_shift_hours)
        x=pre.transform(tr);v=x[:,pre.kept.index(LOG)]
        z=v[tr.cohort.eq('original')]
        assert len(np.unique(z.astype('float32')))>300
        checks.append(dict(repeat=t['repeat'],fold=t['fold'],ordinary_unique_float32=len(np.unique(z.astype('float32'))),standardized_range=float(np.ptp(z))))
    pd.DataFrame(checks).to_csv(OUT/'width_preprocessing_audit.csv',index=False)
    return safe,fp,tasks

def fit_task(t,m,design,fp,destination):
    raw.nested.DEST=Path(destination)
    return raw.nested.fit_task(t,m,design,fp)

def run_jobs(design,fp,tasks,destination,raw_diagnostic=False):
    destination=Path(destination)
    with (destination/'run.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        start=time.monotonic();total=len(tasks)*2
        with ProcessPoolExecutor(max_workers=2) as pool:
            futures=[pool.submit(fit_task,t,m,design,fp,str(destination)) for t in tasks for m in PANELS]
            for i,f in enumerate(as_completed(futures),1):
                checkpoint=f.result()
                raw.legacy.refit.save(destination/'progress.json',dict(status='running',completed_jobs=i,total_jobs=total,
                    elapsed_seconds=time.monotonic()-start,last_checkpoint=checkpoint,workers=2))
                print(destination.name,f'{i}/{total}; {(time.monotonic()-start)/60:.1f} minutes',flush=True)
                original=raw.OUT;raw.OUT=destination
                raw.summarize(fp)
                raw.OUT=original
        original=raw.OUT;raw.OUT=destination
        raw.summarize(fp,complete=not raw_diagnostic)
        raw.OUT=original
        raw.legacy.refit.save(destination/'progress.json',dict(status='diagnostic_complete' if raw_diagnostic else 'complete',
            completed_jobs=total,total_jobs=total,elapsed_seconds=time.monotonic()-start))

def main():
    design,fp,tasks=prepare()
    # Finish one prespecified raw-width repeat using existing checkpoints; no
    # selection between raw and log based on outcome performance.
    rd,rf,rt=raw.prepare()
    run_jobs(rd,rf,[t for t in rt if t['repeat']==0],raw.OUT,True)
    run_jobs(design,fp,tasks,OUT)
    import report
    report.OUT=OUT
    report.main()

if __name__=='__main__':main()
