"""Prepare/resume outer-domain size and crest-cap response experiments.

Default prepares only. --run fits models, with checkpointed nested validation.
"""
from pathlib import Path
import argparse, hashlib, json
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
import model_search as engine
import focused_models as focused

ROOT=engine.ROOT
OUT=ROOT/'RAPID/outputs/hydrograph_cap_size_models_20261019'
SOURCE=ROOT/'RAPID/outputs/hydrograph_residual_review_20261018/diagnostic_rows.parquet'
CAP=ROOT/'RAPID/outputs/hydrograph_cap_centroid_20261019/responses.parquet'
SCOPES=ROOT/'RAPID/outputs/hydrograph_predictor_scopes_20261014'
TARGETS=['cap80_hours','peak_shift_hours','centroid_shift_hours','peak_excess_change_pct','cap70_hours','cap90_hours']
COMMON=['log1p__B_ratio_route_length_sd','log1p__B_ratio_width_mean','log1p__B_ratio_width_cv',
        'log1p__B_ratio_route_entropy','B_parent_dominance_mean','B_exposure','B_K_log_a1']


def panels():
    revised=['log1p__B_ratio_route_length_mean','domain_size']+COMMON
    return dict(current=['B_exposed_length_change','B_mean_length_fraction']+COMMON,
                revised=revised,
                no_resistance=[c for c in revised if c!='B_K_log_a1'],
                no_mean_width=[c for c in revised if c!='log1p__B_ratio_width_mean'],
                size_width=revised+['size_log_width'],size_resistance=revised+['size_resistance'])


def specifications():
    result=[]
    for target in TARGETS:
        selected=list(panels()) if target=='cap80_hours' else (['revised'] if target in ['cap70_hours','cap90_hours'] else ['current','revised'])
        for panel in selected:
            for model,grid in focused.GRIDS.items():
                result.append(dict(name=f'{target}__{panel}__{model}',target=target,panel=panel,model=model,
                                   columns=panels()[panel],grid=grid,corr=None,top_k=None))
    return result


def overlap_components(frame, memberships):
    """Union edits sharing any parent-domain reach; transitive overlap retained."""
    parent={i:i for i in frame.index}
    def find(i):
        while parent[i]!=i:
            parent[i]=parent[parent[i]];i=parent[i]
        return i
    owner={}
    for i,r in frame.iterrows():
        for link in memberships[i]:
            key=(r.example,int(link))
            if key in owner:parent[find(i)]=find(owner[key])
            else:owner[key]=i
    return pd.Series({i:f'{frame.loc[i,"example"]}:{find(i)}' for i in frame.index})


def assign_within_folds(d):
    assignments=pd.Series(-1,index=d.index,dtype=int)
    for ex,g in d.groupby('example'):
        n=g.overlap_group.nunique()
        if n<2:continue
        for fold,(_,test) in enumerate(GroupKFold(n_splits=min(5,n)).split(g,groups=g.overlap_group)):
            assignments.loc[g.index[test]]=fold
    return assignments


def outer_splits(d,validation):
    if validation=='network':
        return [(str(ex),d[d.example.ne(ex)],g) for ex,g in d.groupby('example')]
    fold=assign_within_folds(d)
    result=[]
    for k in sorted(set(fold)-{-1}):
        tr,te=d[fold.ne(k)],d[fold.eq(k)]
        assert set(tr.overlap_group).isdisjoint(te.overlap_group)
        assert set(te.example)<=set(tr.example)
        result.append((str(k),tr,te))
    return result


def inner_splits(train,validation):
    groups=train.example if validation=='network' else train.overlap_group
    splitter=GroupKFold(n_splits=min(3,groups.nunique()))
    return list(splitter.split(train,groups=groups))


def prepare(out):
    d=pd.read_parquet(SOURCE);d=d[d.scenario.eq('all')].copy().reset_index(drop=True)
    assert not d.duplicated(['example','state']).any()
    cap=pd.read_parquet(CAP)
    table=cap.pivot(index=['example','state'],columns='cap_fraction',values='cap_centroid_shift_minutes')
    table=table.rename(columns={.7:'cap70_hours',.8:'cap80_hours',.9:'cap90_hours'})/60
    joined=d.merge(table.reset_index(),on=['example','state'],how='outer',indicator=True,validate='one_to_one')
    audit=joined[['example','state','_merge']].copy()
    assert not audit._merge.eq('left_only').any()
    d=joined[joined._merge.eq('both')].drop(columns='_merge').reset_index(drop=True)
    d['B_ratio_route_length_mean']=d.B_child_route_length_mean/d.B_parent_route_length_mean
    d['log1p__B_ratio_route_length_mean']=np.log1p(d.B_ratio_route_length_mean)
    d['domain_size']=d.B_parent_route_length_mean/1000.
    d['size_log_width']=d.domain_size*np.log(d.B_ratio_width_mean)
    d['size_resistance']=d.domain_size*d.B_K_log_a1
    required=list(set(sum(panels().values(),[])))+TARGETS
    assert np.isfinite(d[required]).all().all()
    graphs=pd.read_csv(SCOPES/'domain_graphs.csv',dtype={'example':str}).set_index(['example','state','scope'])
    memberships={};cache={}
    for i,r in d.iterrows():
        entry=graphs.loc[(r.example,r.state,r.B_scope)]
        assert isinstance(entry,pd.Series)
        key=entry.parent_graph
        if key not in cache:cache[key]=pd.read_parquet(SCOPES/'graphs'/f'{key}.parquet').id_link.to_numpy()
        memberships[i]=cache[key]
    d['overlap_group']=overlap_components(d,memberships)
    d['within_fold']=assign_within_folds(d)
    # Units-only change must vanish under the same training-weighted scaler.
    w=engine.weights(d)
    size_cols=['domain_size','size_log_width','size_resistance']
    x=d[size_cols]
    np.testing.assert_allclose(StandardScaler().fit_transform(x,sample_weight=w),
                               StandardScaler().fit_transform(x/10,sample_weight=w),atol=1e-12)
    inputs=[SOURCE,CAP,SCOPES/'domain_graphs.csv',Path(__file__),Path(engine.__file__),Path(focused.__file__)]
    graph_hashes={k:engine.sha(SCOPES/'graphs'/f'{k}.parquet') for k in cache}
    design=dict(specifications=specifications(),sources={str(p):engine.sha(p) for p in inputs},graph_hashes=graph_hashes,
                reference_length_m=1000,reference_10km_scaling_check='passed at 1e-12',
                domains='Same validated outer domains as prior cohort; no hydraulic predictors.',
                validation='Network: leave one example out. Within: up to 5 folds per example, grouping transitively overlapping parent-domain links. All aliases remain canonical. Inner tuning uses example groups or overlap groups, respectively.',
                weights='Equal-example training and inner validation MSE; report equal-example and pair metrics.',
                targets='Cap80 primary timing; peak, full centroid and peak excess compare current/revised; cap70/90 sensitivity on revised only.',
                limitations='Within-network validation estimates known-network interpolation. Examples with only one overlap component have no within-network test rows; retained as training context. Same parent forcing can occur on both sides of this diagnostic.',
                transforms='Natural log1p geometric ratios; size stays linear. Width interaction uses size*ln(width ratio); resistance interaction size*(-log10 resistance ratio).')
    out.mkdir(parents=True,exist_ok=True)
    path=out/'design.json'
    if path.exists() and json.loads(path.read_text())!=design and any((out/'checkpoints').glob('**/*.json')):
        raise ValueError('Design changed with checkpoints present; use a fresh output directory')
    engine.dump(path,design);d.to_parquet(out/'input.parquet',index=False)
    audit.to_csv(out/'join_audit.csv',index=False)
    d[['example','state','overlap_group','within_fold']].to_csv(out/'within_group_audit.csv',index=False)
    plan=[]
    for scenario in ['all','without54']:
        cohort=d if scenario=='all' else d[d.example.ne('54')]
        for validation in ['network','within']:
            folds=outer_splits(cohort,validation)
            for target in TARGETS:
                specs=[s for s in specifications() if s['target']==target]
                fits=sum(sum(len(inner_splits(tr,validation))*len(s['grid'])+1 for s in specs) for _,tr,_ in folds)
                plan.append(dict(scenario=scenario,validation=validation,target=target,folds=len(folds),test_rows=sum(len(te) for _,_,te in folds),specifications=len(specs),fits=fits))
    pd.DataFrame(plan).to_csv(out/'run_plan.csv',index=False)
    print(pd.DataFrame(plan).to_string(index=False),flush=True)
    print('Prepared. No model fits without --run. Full planned fits:',sum(r['fits'] for r in plan),flush=True)
    return d,design


def summarize(out,design):
    fingerprint=hashlib.sha256(json.dumps(design,sort_keys=True).encode()).hexdigest()
    rows=[];settings=[]
    for path in (out/'checkpoints').glob('*/*/*/*.json'):
        r=json.loads(path.read_text());assert r['fingerprint']==fingerprint
        common={k:r[k] for k in ['scenario','validation','held','spec','target','panel','model']}
        rows.extend([dict(**common,**p) for p in r['predictions']])
        settings.append(dict(**common,params=json.dumps(r['params']),inner_mse=r['inner_mse']))
    if not rows:return
    pred=pd.DataFrame(rows);pred.to_parquet(out/'predictions.parquet',index=False)
    pd.DataFrame(settings).to_csv(out/'chosen_hyperparameters.csv',index=False)
    scores=[]
    plan=pd.read_csv(out/'run_plan.csv')
    for keys,g in pred.groupby(['scenario','validation','spec','target']):
        common=dict(zip(['scenario','validation','spec','target'],keys))
        expected=plan[plan.scenario.eq(keys[0])&plan.validation.eq(keys[1])&plan.target.eq(keys[3])].iloc[0]
        assert not g.duplicated(['example','state']).any()
        common['complete']=bool(len(g)==expected.test_rows and g.held.nunique()==expected.folds)
        common['completed_folds']=int(g.held.nunique())
        common['expected_folds']=int(expected.folds)
        timing=keys[-1]!='peak_excess_change_pct'
        for buffer in ([1,2,5] if timing else [None]):
            y=g.observed*60 if timing else g.observed
            masks={'all':np.ones(len(g),bool)}
            if timing:masks.update(advance=y.lt(-buffer),unchanged=y.abs().le(buffer),delay=y.gt(buffer),large_shift=y.abs().ge(60),outside_buffer=y.abs().gt(buffer))
            for name,mask in masks.items():
                z=g[mask]
                if z.empty:continue
                for weighting,w in [('equal_example',engine.weights(z)),('pair',np.ones(len(z)))]:
                    scale=60 if timing else 1
                    yy=z.observed.to_numpy()*scale;pp=z.predicted.to_numpy()*scale
                    row=dict(**common,buffer_minutes=buffer,subset=name,weighting=weighting,n=len(z),examples=z.example.nunique(),unit='minutes' if timing else 'percent',bias=float(np.average(pp-yy,weights=w)),**engine.score(yy,pp,w))
                    if timing and name=='outside_buffer':row['mape_pct']=float(100*np.average(abs((pp-yy)/yy),weights=w))
                    scores.append(row)
    pd.DataFrame(scores).to_csv(out/'scores.csv',index=False)


def run(d,design,out,args):
    fingerprint=hashlib.sha256(json.dumps(design,sort_keys=True).encode()).hexdigest()
    specs=[s for s in specifications() if s['model'] in args.models and s['target'] in args.targets]
    for scenario in args.scenarios:
        cohort=d if scenario=='all' else d[d.example.ne('54')]
        for validation in args.validations:
            for held,tr,te in outer_splits(cohort,validation):
                for s in specs:
                    path=out/'checkpoints'/scenario/validation/held/(s['name']+'.json')
                    if path.exists():
                        assert json.loads(path.read_text())['fingerprint']==fingerprint;continue
                    trials=[];splits=inner_splits(tr,validation)
                    for params in s['grid']:
                        prediction=np.full(len(tr),np.nan)
                        for ti,vi in splits:
                            prediction[vi],_=engine.fit_predict(tr.iloc[ti],tr.iloc[vi],s['target'],s,params,args.jobs)
                        assert np.isfinite(prediction).all()
                        trials.append(dict(params=params,mse=float(np.average((tr[s['target']]-prediction)**2,weights=engine.weights(tr)))))
                    best=min(trials,key=lambda x:x['mse'])
                    prediction,audit=engine.fit_predict(tr,te,s['target'],s,best['params'],args.jobs)
                    rows=[dict(example=r.example,state=r.state,observed=float(getattr(r,s['target'])),predicted=float(p)) for r,p in zip(te.itertuples(),prediction)]
                    engine.dump(path,dict(fingerprint=fingerprint,scenario=scenario,validation=validation,held=held,
                                         spec=s['name'],target=s['target'],panel=s['panel'],model=s['model'],params=best['params'],
                                         inner_mse=best['mse'],trials=trials,audit=audit,predictions=rows))
                    print(scenario,validation,held,s['name'],'saved',flush=True)
                summarize(out,design)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',action='store_true');p.add_argument('--output',type=Path,default=OUT)
    p.add_argument('--jobs',type=int,default=1)
    p.add_argument('--models',nargs='+',choices=list(focused.GRIDS),default=list(focused.GRIDS))
    p.add_argument('--targets',nargs='+',choices=TARGETS,default=TARGETS)
    p.add_argument('--validations',nargs='+',choices=['network','within'],default=['network','within'])
    p.add_argument('--scenarios',nargs='+',choices=['all','without54'],default=['all','without54'])
    args=p.parse_args()
    if args.jobs<1:p.error('--jobs must be positive')
    d,design=prepare(args.output)
    if args.run:run(d,design,args.output,args)
    summarize(args.output,design)

if __name__=='__main__':main()
