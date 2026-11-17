"""Matched real-only centroid RF comparison; historical nested compound splits."""
import os
for name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[name] = '1'
import argparse
import hashlib
import importlib.util
import json
import time
import fcntl
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result

legacy = module('legacy_geometry_driver', ROOT/'RAPID/investigations/hydrography_rf_20261029/run.py')
geometry = module('selected_geometry', ROOT/'RAPID/investigations/synthetic_selected_sets_20261102/prepare.py')
nested = legacy.nested
OLD = legacy.DEST
OUT = ROOT/'RAPID/outputs/matched_centroid_sets_20261103'
COMMON = ['B_delta_route_length_mean', 'parent_slope_slow', 'path_parent_equivalent_width',
          'path_delta_topologic_complexity_score']
PANELS = dict(centroid_with_G=COMMON+['delta_G_manning'],
              centroid_without_G=COMMON+['B_delta_area_proxy'])

def save(name, value):
    legacy.refit.save(OUT/name, value)

def prepare():
    OUT.mkdir(exist_ok=True)
    if (OUT/'design.json').exists():
        design = nested.load(OUT/'design.json')
        for path, digest in design['sources'].items():
            assert legacy.refit.sha(path) == digest, path
        assert legacy.refit.sha(OUT/'input.parquet') == design['input_sha256']
        splits = nested.load(OUT/'splits.json')
        return design, splits['fingerprint'], splits['tasks']
    old_design = nested.load(OLD/'design.json')
    old_splits = nested.load(OLD/'splits.json')
    assert old_splits['fingerprint'] == hashlib.sha256(json.dumps(old_design, sort_keys=True).encode()).hexdigest()
    d = pd.read_parquet(OLD/'input.parquet')
    assert legacy.refit.sha(OLD/'input.parquet') == old_design['input_sha256']
    fresh = pd.read_parquet(geometry.OUT/'input.parquet').set_index(['example','state'])
    aligned = fresh.loc[pd.MultiIndex.from_frame(d[['example','state']])].reset_index()
    np.testing.assert_allclose(aligned.centroid_shift_minutes, d.centroid_shift_hours*60, atol=1e-10)
    for col in set(sum(PANELS.values(), [])):
        if col == 'parent_slope_slow':
            np.testing.assert_allclose(d.loc[d.cohort.eq('original'), col], aligned.loc[d.cohort.eq('original'), col])
        else:
            d[col] = aligned[col].to_numpy()
    # Historical full-network controls exclude unchanged connectors from length,
    # width and area rewards. Keep graph connectivity for exact path enumeration.
    audit = []
    for i, row in d[d.cohort.eq('segmented')].iterrows():
        p = geometry.active(row.example, 'S000_base')
        c = geometry.active(row.example, row.state)
        lineage = geometry.read(legacy.refit.EXT/f'variants/{row.example}/{row.state}/lineage.csv')
        lineage = lineage[lineage.role.eq('collapsed_block')]
        pmask = p.id_link.isin(set(lineage.parent_link_id.astype(int))).to_numpy()
        cmask = c.id_link.isin(set(lineage.child_link_id.astype(int))).to_numpy()
        parts = list(geometry.graph(p))
        parts[4] = parts[4] * pmask
        width = geometry.equivalent_width(tuple(parts), f'{row.example}/edited-parent')
        area = float((c.loc[cmask,'link_length_m']*c.loc[cmask,'width']).sum() -
                     (p.loc[pmask,'link_length_m']*p.loc[pmask,'width']).sum())
        audit.append(dict(example=row.example, state=row.state,
            full_mean_delta=d.at[i,'B_delta_route_length_mean'], edited_mean_delta=1000*row.length_delta_km,
            full_equivalent_width=d.at[i,'path_parent_equivalent_width'], edited_equivalent_width=width,
            full_area_delta=d.at[i,'B_delta_area_proxy'], edited_area_delta=area,
            parent_connector_edges=int((~pmask).sum()), child_connector_edges=int((~cmask).sum())))
        d.at[i,'B_delta_route_length_mean'] = 1000*row.length_delta_km
        d.at[i,'path_parent_equivalent_width'] = width
        d.at[i,'B_delta_area_proxy'] = area
    np.testing.assert_allclose(d.B_delta_route_length_mean, d.length_delta_km*1000, atol=1e-6)
    assert np.isfinite(d[list(set(sum(PANELS.values(),[])))]).all().all()
    assert len(d)==2185 and d.group_id.nunique()==450
    for task in old_splits['tasks']:
        nested.audit_task(d,task)
    d.to_parquet(OUT/'input.parquet',index=False)
    pd.DataFrame(audit).to_csv(OUT/'domain_alignment.csv',index=False)
    pd.DataFrame(geometry.WIDTH_AUDIT).to_csv(OUT/'width_checks.csv',index=False)
    deps = [Path(__file__), Path(legacy.__file__), Path(nested.__file__), Path(geometry.__file__),
            Path(legacy.refit.tuning.b.__file__), Path(legacy.refit.tuning.b.engine.__file__),
            Path(legacy.refit.tuning.b.forest.__file__), OLD/'input.parquet', OLD/'design.json',
            OLD/'splits.json', OLD/'predictions.parquet', geometry.OUT/'input.parquet']
    design = dict(panels=PANELS,candidates=old_design['candidates'],
        sources=geometry.HASHES | {str(p):legacy.refit.sha(p) for p in deps},
        input_sha256=legacy.refit.sha(OUT/'input.parquet'),
        validation=old_design['validation'], training=old_design['training'],
        target=old_design['target'],
        domains='Same local B domains. Full-network controls: edited-edge length/area/width rewards on connected full graph, matching historical length and slope convention; full connected topology retained. Manning delta G remains whole-network by definition.',
        baseline='Saved geometry_normalized_9 predictions; first outer refit checked against saved predictions before reuse.',
        caveats=old_design['caveats'], jobs=50,forest_fits=950,production_modified=False)
    fp=hashlib.sha256(json.dumps(design,sort_keys=True).encode()).hexdigest()
    save('design.json',design)
    save('splits.json',dict(fingerprint=fp,tasks=old_splits['tasks']))
    return design,fp,old_splits['tasks']

def check_baseline(tasks):
    if (OUT/'baseline_parity.json').exists():
        return
    t=tasks[0]; model='geometry_normalized_9'
    path=OLD/'checkpoints'/t['strategy']/f"r{t['repeat']}"/f"f{t['fold']}"/model/'outer.json'
    result=nested.load(path)
    d=pd.read_parquet(OLD/'input.parquet'); b=legacy.refit.tuning.b
    b.COLUMNS=nested.load(OLD/'design.json')['panels'][model]
    a=b.fit(d.iloc[t['train_indices']],result['best']['params'],jobs=1)
    prediction=b.forest.predict(a,d.iloc[t['test_indices']])
    old=np.array([r['predicted_hours'] for r in result['rows']])
    np.testing.assert_allclose(prediction,old,rtol=1e-12,atol=1e-12)
    save('baseline_parity.json',dict(max_absolute_error_hours=float(abs(prediction-old).max()),
        fold=t['fold'],repeat=t['repeat'],rows=len(old),checkpoint_sha256=legacy.refit.sha(path)))
    print('Historical outer-refit parity passed',flush=True)

def fit_task(task,model,design,fp):
    nested.DEST=OUT
    return nested.fit_task(task,model,design,fp)

def summarize(fp,complete=False):
    old=pd.read_parquet(OLD/'predictions.parquet')
    old=old[old.model.eq('geometry_normalized_9')].copy()
    rows=[]
    for path in sorted((OUT/'checkpoints').glob('*/*/*/*/outer.json')):
        r=nested.load(path);assert r['fingerprint']==fp;rows.extend(r['rows'])
    p=pd.concat([old,pd.DataFrame(rows)],ignore_index=True)
    scores=[]
    for (model,repeat),g in p.groupby(['model','repeat']):
        if len(g)!=2185:continue
        for cohort,z in [('original',g[g.cohort.eq('original')]),('segmented',g[g.cohort.eq('segmented')]),('all',g)]:
            scores.append(dict(model=model,repeat=repeat,cohort=cohort,
                **nested.score(z,'equal_cohort_network' if cohort=='all' else 'equal_network')))
    s=pd.DataFrame(scores)
    s.to_csv(OUT/'repeat_scores.csv',index=False)
    agg=s.groupby(['model','cohort'])[['r2','mae_minutes']].agg(['mean','min','max'])
    agg.columns=['_'.join(c) for c in agg.columns]
    agg.reset_index().to_csv(OUT/'summary.csv',index=False)
    p.to_parquet(OUT/('predictions.parquet' if complete else 'partial_predictions.parquet'),index=False)
    if complete:
        assert len(rows)==2185*5*2 and len(s)==45
        save('completion.json',dict(complete=True,fingerprint=fp,jobs=50,forest_fits=950,
            files={str(OUT/x):legacy.refit.sha(OUT/x) for x in ['predictions.parquet','repeat_scores.csv','summary.csv']}))

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',action='store_true');ap.add_argument('--workers',type=int,default=2)
    args=ap.parse_args();design,fp,tasks=prepare();check_baseline(tasks)
    if not args.run:return
    with (OUT/'run.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        start=time.monotonic()
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures=[pool.submit(fit_task,t,m,design,fp) for t in tasks for m in PANELS]
            for i,future in enumerate(as_completed(futures),1):
                path=future.result()
                save('progress.json',dict(status='running',completed_jobs=i,total_jobs=50,
                    elapsed_seconds=time.monotonic()-start,last_checkpoint=path,workers=args.workers))
                print(f'{i}/50 outer fits; {(time.monotonic()-start)/60:.1f} minutes',flush=True)
                summarize(fp)
        summarize(fp,True)
        save('progress.json',dict(status='complete',completed_jobs=50,total_jobs=50,elapsed_seconds=time.monotonic()-start))

if __name__=='__main__':main()
