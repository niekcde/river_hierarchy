"""Fixed-setting held-network domain/exposure diagnostic; no new tuning."""
from pathlib import Path
import argparse,json,hashlib
import numpy as np
import pandas as pd
import final_forest_search as forest
from build import descriptors

engine=forest.engine;ROOT=engine.ROOT
SOURCE=forest.SOURCE
SCOPES=ROOT/'RAPID/outputs/hydrograph_predictor_scopes_20261014'
OUT=ROOT/'RAPID/outputs/hydrograph_closed_domain_test_20261019'
METRICS=['route_length_mean','route_length_sd','width_mean','width_cv','route_entropy']
VARIANTS=['current','smallest_closed','smallest_closed_exposure_one']


def local_inputs(d):
    """Choose validated closed U when available; otherwise retain validated B."""
    a=d.copy();use=d.U_closed.eq(True)
    for metric in METRICS:
        ratio=d.loc[use,f'U_child_{metric}']/d.loc[use,f'U_parent_{metric}']
        if not np.isfinite(ratio).all() or (ratio<0).any():raise ValueError(f'Undefined local ratio: {metric}')
        a.loc[use,f'log1p__B_ratio_{metric}']=np.log1p(ratio)
    a.loc[use,'domain_size']=d.loc[use,'U_parent_route_length_mean']/1000
    a.loc[use,'B_parent_dominance_mean']=d.loc[use,'U_parent_dominance_mean']
    a.loc[use,'B_exposure']=d.loc[use,'U_exposure']
    assert np.isfinite(a[forest.sweep.COLUMNS]).all().all()
    return a,use


def prepare(out):
    d=pd.read_parquet(SOURCE/'input.parquet').reset_index(drop=True)
    assert len(d)==2170 and not d.duplicated(['example','state']).any()
    info=json.loads((forest.OUT/'final_model.json').read_text())
    assert engine.sha(forest.OUT/'final_model.joblib')==info['model_sha256']
    graphs=pd.read_csv(SCOPES/'domain_graphs.csv',dtype={'example':str}).set_index(['example','state','scope'])
    alternative,use=local_inputs(d);audit=[];hashes={};cache={}
    def graph(key):
        p=SCOPES/'graphs'/f'{key}.parquet'
        if key not in cache:
            f=pd.read_parquet(p);cache[key]=(f,descriptors(f));hashes[str(p)]=engine.sha(p)
        return cache[key]
    for i,r in d.iterrows():
        old=graphs.loc[(r.example,r.state,r.B_scope)]
        selected='U' if use.iloc[i] else r.B_scope
        new=graphs.loc[(r.example,r.state,selected)]
        assert bool(new.parent_closed) and bool(new.child_closed)
        changed=(old.parent_graph!=new.parent_graph or old.child_graph!=new.child_graph)
        if changed:
            # Recompute all geometric quantities from the saved matched domain pair;
            # verify the cached values before using them in the comparison.
            for side in ['parent','child']:
                f,m=graph(new[f'{side}_graph']);outer=pd.read_parquet(SCOPES/'graphs'/f"{old[f'{side}_graph']}.parquet")
                assert set(f.id_link)<=set(outer.id_link)
                for metric in METRICS+['dominance_mean']:
                    value=m[metric];cached=r[f'U_{side}_{metric}']
                    np.testing.assert_allclose(value,cached,rtol=1e-9,atol=1e-9,equal_nan=True)
        audit.append(dict(example=r.example,state=r.state,edit_type=r.edit_type,old_scope=r.B_scope,new_scope=selected,changed_domain=changed,
          reason='closed selected U' if use.iloc[i] else 'retain containing domain: local U is open or unavailable',
          old_parent_graph=old.parent_graph,new_parent_graph=new.parent_graph,old_child_graph=old.child_graph,new_child_graph=new.child_graph,
          parent_closed=bool(new.parent_closed),child_closed=bool(new.child_closed),old_exposure=r.B_exposure,new_exposure=alternative.loc[i,'B_exposure']))
    audit=pd.DataFrame(audit)
    # All actually changed domains are genuinely nested inside the old graph;
    # compare unchanged domains numerically to ensure no accidental feature edits.
    np.testing.assert_allclose(d.loc[~audit.changed_domain,forest.sweep.COLUMNS],alternative.loc[~audit.changed_domain,forest.sweep.COLUMNS],atol=1e-9,rtol=1e-9)
    d['changed_domain']=audit.changed_domain;alternative['changed_domain']=audit.changed_domain
    forced=alternative.copy();forced['B_exposure']=1.
    frames={'current':d,'smallest_closed':alternative,'smallest_closed_exposure_one':forced}
    files=[Path(__file__),Path(forest.__file__),Path(engine.__file__),SOURCE/'input.parquet',SCOPES/'domain_graphs.csv',forest.OUT/'final_model.json']
    hashes.update({str(p):engine.sha(p) for p in files})
    design=dict(params=info['params'],columns=forest.sweep.COLUMNS,target=forest.TARGET,variants=VARIANTS,
      domain='Smallest among existing validated U/C candidates: closed selected U if available in both states, otherwise retain containing C. Reuses established selector policy; not a global search for new intermediate domains.',
      checks='Recomputed changed-domain geometry from graph pairs, checked cached values, closure flags and containment. Exposure uses saved whole-parent-network allocation, not local normalization.',
      validation='15 whole-example holdouts, identical rows and fixed parameters for all variants; refit preprocessing and forest inside each training fold.',
      limitation='Parameters were already selected on all examples using current domains. This is a paired exploratory domain diagnostic, not nested model-selection validation or a new independent estimate. Extreme-case and without54 evaluation subsets do not remove these cases from training.',
      weighting='Equal-example training and evaluation weights. Signed full-centroid target; peak evaluated as transfer. Floors 1,2,5 minutes.',
      sources=hashes)
    out.mkdir(parents=True,exist_ok=True)
    if (out/'design.json').exists() and list((out/'checkpoints').glob('**/*.json')):
        if json.loads((out/'design.json').read_text())!=design:raise ValueError('Changed design; select a new output directory')
    engine.dump(out/'design.json',design);audit.to_csv(out/'domain_audit.csv',index=False)
    for name,frame in frames.items():frame.to_parquet(out/f'input_{name}.parquet',index=False)
    fp=hashlib.sha256(json.dumps(design,sort_keys=True).encode()).hexdigest()
    print(f'{len(d)} matched rows; {int(audit.changed_domain.sum())} changed domains; 45 fixed-setting fits.',flush=True)
    return frames,info['params'],fp


def run(frames,params,out,fp,jobs):
    for ex in sorted(frames['current'].example.unique()):
      for variant,d in frames.items():
        path=out/'checkpoints'/variant/f'{ex}.json'
        if forest.checked(path,fp) is not None:continue
        tr=d[d.example!=ex];te=d[d.example==ex]
        fitted=forest.fit(tr,params,jobs);pred=forest.predict(fitted,te)
        rows=[dict(example=r.example,state=r.state,centroid_shift_hours=r.centroid_shift_hours,peak_shift_hours=r.peak_shift_hours,edit_type=r.edit_type,has_nested_structure=bool(r.has_nested_structure),changed_domain=bool(r.changed_domain),predicted_hours=float(p)) for r,p in zip(te.itertuples(),pred)]
        engine.dump(path,dict(fingerprint=fp,held=ex,variant=variant,predictions=rows))
        print(ex,variant,'saved',flush=True)


def summarize(frames,out,fp):
    rows=[]
    for path in (out/'checkpoints').glob('*/*.json'):
        c=forest.checked(path,fp)
        rows.extend(dict(variant=c['variant'],**r) for r in c['predictions'])
    if not rows:return
    d=pd.DataFrame(rows);assert not d.duplicated(['variant','example','state']).any()
    d.to_parquet(out/'predictions.parquet',index=False);scores=[];networks=[]
    for variant,g in d.groupby('variant'):
      for response in ['centroid_shift_hours','peak_shift_hours']:
       for floor in [1.,2.,5.]:
        y=g[response]*60
        masks=dict(all=np.ones(len(g),bool),changed_domain=g.changed_domain,unchanged_domain=~g.changed_domain,
          nested=g.has_nested_structure,compound_member=g.edit_type.eq('Compound member'),unchanged=abs(y)<=floor,
          outside_buffer=abs(y)>floor,advance=y < -floor,delay=y>floor,
          excluding_extreme=~((g.example=='37')&(g.state=='CB001_bubble_3')),excluding54=g.example!='54')
        for subset,mask in masks.items():
          z=g[mask]
          if z.empty:continue
          yy=z[response].to_numpy();pp=z.predicted_hours.to_numpy();w=engine.weights(z)
          scores.append(dict(variant=variant,response=response,floor_minutes=floor,subset=subset,n=len(z),examples=z.example.nunique(),complete=len(g)==len(frames[variant]),relative_error_pct=forest.sweep.relative_score(yy,pp,w,floor),mape_pct=float(100*np.average(abs((pp-yy)/yy),weights=w)) if subset=='outside_buffer' else None,**engine.score(yy*60,pp*60,w)))
      for ex,z in g.groupby('example'):
        networks.append(dict(variant=variant,example=ex,relative_error_pct=forest.sweep.relative_score(z.centroid_shift_hours,z.predicted_hours,np.ones(len(z)),1),**engine.score(z.centroid_shift_hours.to_numpy()*60,z.predicted_hours.to_numpy()*60,np.ones(len(z)))))
    pd.DataFrame(scores).to_csv(out/'scores.csv',index=False);pd.DataFrame(networks).to_csv(out/'network_scores.csv',index=False)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',action='store_true');p.add_argument('--jobs',type=int,default=1);p.add_argument('--output',type=Path,default=OUT)
    args=p.parse_args()
    if args.jobs<1:p.error('jobs must be positive')
    frames,params,fp=prepare(args.output)
    if args.run:run(frames,params,args.output,fp,args.jobs)
    summarize(frames,args.output,fp)

if __name__=='__main__':main()
