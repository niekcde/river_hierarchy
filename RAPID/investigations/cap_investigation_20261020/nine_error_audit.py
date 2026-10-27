"""Audit dominant held-out errors of the nine-input same-grid tuning procedure."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from tune_nine_inputs import b, OUT, SOURCE, COLUMNS, fit, save, sha

DEST = OUT/'nine_error_audit'


def alternative_fit(tr, params, variant):
    b.COLUMNS=list(COLUMNS)
    if variant=='no_bootstrap':
        return fit(tr,dict(params,bootstrap=False,max_samples=None),4)
    y=tr.centroid_shift_hours.to_numpy()
    # Equal original/segmented mass within each network. Diagnostic only.
    w=1/tr.groupby(['example','cohort']).example.transform('size').to_numpy(float)
    w=w/w.mean()
    pre=b.engine.Preprocess(COLUMNS,None,None).fit(tr,y)
    mean=float(np.average(y,weights=w))
    sd=float(np.sqrt(np.average((y-mean)**2,weights=w)))
    estimator=RandomForestRegressor(**params,random_state=b.engine.SEED,n_jobs=4)
    estimator.fit(pre.transform(tr),(y-mean)/sd,sample_weight=w)
    return dict(preprocessor=pre,estimator=estimator,target_mean=mean,target_sd=sd,columns=COLUMNS)


def main():
    DEST.mkdir(exist_ok=True)
    files=[SOURCE/'input.parquet',OUT/'nine_input_tuning/predictions.parquet',
           OUT/'nine_input_tuning/selections.json',Path(__file__),Path(b.__file__),Path(b.engine.__file__)]
    sources={str(p.resolve()):sha(p) for p in files}
    d=pd.read_parquet(files[0])
    p=pd.read_parquet(files[1]);p=p[p.variant.eq('same_grid')].copy()
    selections=json.loads(files[2].read_text())
    p['error_h']=p.predicted_hours-p.centroid_shift_hours
    p['relative_error']=abs(p.error_h)/np.maximum(abs(p.centroid_shift_hours),1/60)
    p['weight']=1/p.groupby(['cohort','example']).example.transform('size')
    for cohort,g in p.groupby('cohort'):
        p.loc[g.index,'sse_share_pct']=100*g.weight*g.error_h**2/(g.weight*g.error_h**2).sum()
        p.loc[g.index,'relative_error_share_pct']=100*g.weight*g.relative_error/(g.weight*g.relative_error).sum()
    p.to_csv(DEST/'error_attribution.csv',index=False)
    p.groupby(['cohort','example'])[['sse_share_pct','relative_error_share_pct']].sum().to_csv(DEST/'network_attribution.csv')
    top=pd.concat([p[p.cohort.eq('original')].nlargest(8,'sse_share_pct'),
                   p[p.cohort.eq('segmented')].nlargest(4,'sse_share_pct')])
    top.to_csv(DEST/'priority_cases.csv',index=False)
    support,neighbors,diagnostics,leafrows=[],[],[],[]
    for ex in sorted(top.example.unique()):
        tr=d[d.example.ne(ex)].reset_index(drop=True)
        te=d[d.example.eq(ex)&d.state.isin(top[top.example.eq(ex)].state)].copy()
        choice=next(c for c in selections if c['variant']=='same_grid' and c['held']==ex)
        model=fit(tr,choice['params'],4)
        pred=b.forest.predict(model,te)
        check=te[['example','state']].merge(p,on=['example','state'],validate='one_to_one')
        np.testing.assert_allclose(pred,check.predicted_hours,atol=1e-12,rtol=1e-12)
        X=model['preprocessor'].transform(tr);Z=model['preprocessor'].transform(te)
        w=b.engine.weights(tr)
        trees=model['estimator'].estimators_
        samples=model['estimator'].estimators_samples_
        for i,row in enumerate(te.itertuples()):
            y=row.centroid_shift_hours
            lo,hi=float(tr.centroid_shift_hours.min()),float(tr.centroid_shift_hours.max())
            bounds=[]
            for c in COLUMNS:
                v=float(te.iloc[i][c])
                if v<tr[c].min() or v>tr[c].max():
                    bounds.append(dict(feature=c,value=v,low=float(tr[c].min()),high=float(tr[c].max())))
            large=tr[tr.centroid_shift_hours<=y] if y<0 else tr[tr.centroid_shift_hours>=y]
            support.append(dict(example=ex,state=row.state,actual=y,predicted=float(pred[i]),
                training_target_min=lo,training_target_max=hi,
                unavoidable_range_error_h=max(lo-y,y-hi,0),
                training_rows_at_least_as_extreme=len(large),training_networks_at_least_as_extreme=large.example.nunique(),
                outside_features=json.dumps(bounds)))
            dist=np.sqrt(((X-Z[i])**2).sum(axis=1))
            near=tr.copy();near['distance']=dist
            for pool in ['all','segmented']:
                z=near if pool=='all' else near[near.cohort.eq('segmented')]
                for r in z.nsmallest(5,'distance').itertuples():
                    neighbors.append(dict(held_example=ex,held_state=row.state,pool=pool,
                        neighbor_example=r.example,neighbor_state=r.state,cohort=r.cohort,
                        distance=r.distance,target=r.centroid_shift_hours))
            mass=np.zeros(len(tr));encounters=np.zeros(len(tr));treevalues=[];leafsizes=[]
            for tree,sample in zip(trees,samples):
                leaf=tree.apply(Z[i:i+1])[0]
                match=tree.apply(X)==leaf
                counts=np.bincount(sample,minlength=len(tr))
                weight=counts*w*match
                assert weight.sum()>0
                mass+=weight/weight.sum()/len(trees)
                encounters+=(counts>0)&match
                treevalues.append(float(tree.predict(Z[i:i+1])[0]*model['target_sd']+model['target_mean']))
                leafsizes.append(int(tree.tree_.n_node_samples[leaf]))
            for j in np.argsort(-mass)[:12]:
                r=tr.iloc[j]
                leafrows.append(dict(held_example=ex,held_state=row.state,neighbor_example=r.example,
                    neighbor_state=r.state,cohort=r.cohort,target=r.centroid_shift_hours,
                    mean_bootstrap_leaf_weight=float(mass[j]),inbag_shared_leaf_fraction=float(encounters[j]/len(trees)),
                    median_leaf_unique_samples=float(np.median(leafsizes)),tree_prediction_min=min(treevalues),
                    tree_prediction_max=max(treevalues),tree_prediction_median=float(np.median(treevalues))))
        for variant in ['baseline','no_bootstrap','cohort_balanced']:
            artifact=model if variant=='baseline' else alternative_fit(tr,choice['params'],variant)
            prediction=b.forest.predict(artifact,te)
            for r,v in zip(te.itertuples(),prediction):
                diagnostics.append(dict(example=ex,state=r.state,variant=variant,actual=r.centroid_shift_hours,predicted=float(v)))
        print(ex,'support, bootstrap leaves and training-weight diagnostics complete',flush=True)
    for name,rows in [('support',support),('neighbors',neighbors),('bootstrap_leaf_neighbors',leafrows),('targeted_training_checks',diagnostics)]:
        pd.DataFrame(rows).to_csv(DEST/f'{name}.csv',index=False)
    physical(top,d)
    for path,h in sources.items():
        assert sha(path)==h
    save(DEST/'manifest.json',dict(complete=True,sources=sources,
         selection='Eight largest partial and four largest segmented contributions to within-cohort network-weighted squared error.',
         diagnostics='Fixed selected settings, each target network entirely excluded. No-bootstrap and cohort-balanced fits are targeted checks, not globally validated candidate models.',
         leaf_measure='Bootstrap-count times training-weight mass within each reached leaf, averaged across trees. Not exact predictive attribution for absolute-error/median trees.',
         physics='Routed travel times, celerities and depth are diagnostic only, not added predictors.',
         production_modified=False))


def physical(top,d):
    root=OUT.parents[2]
    plans=[root/'RAPID/outputs/event_window_plan_20261013/plan.json',
           root/'RAPID/outputs/hydrograph_closed_unit_extension_20261019/routing_plan.json',
           OUT/'routing_plan.json']
    jobs={}
    for path in plans:
        for case in json.loads(path.read_text())['cases']:
            jobs.setdefault(case['example'],{}).update({j['state_id']:j for j in case['jobs']})
    rows=[]
    for row in top.itertuples():
        ex,state=row.example,row.state
        parent=pd.read_parquet(jobs[ex]['S000_base']['parameters'])
        child=pd.read_parquet(jobs[ex][state]['parameters'])
        entry=dict(example=ex,state=state,centroid_shift_h=row.centroid_shift_hours,predicted_h=row.predicted_hours)
        for label,f in [('parent',parent),('child',child)]:
            rL=f.reference_flow_fraction*f.link_length_m
            entry[label+'_full_route_time_h']=float((rL/f.rapid_celerity_mps).sum()/3600)
        entry['route_time_change_h']=entry['child_full_route_time_h']-entry['parent_full_route_time_h']
        entry['n_manning']=float(parent.n_manning.iloc[0])
        rows.append(entry)
    pd.DataFrame(rows).to_csv(DEST/'physical_checks.csv',index=False)
    features=top[['example','state','cohort','centroid_shift_hours','predicted_hours']].merge(
        d[['example','state',*COLUMNS,'parent_width_km','parent_H','parent_CV_L']],
        on=['example','state'],validate='one_to_one')
    for col in COLUMNS:
        if col.startswith('log1p__'):
            features[col.removeprefix('log1p__')]=np.expm1(features[col])
    features.to_csv(DEST/'priority_features.csv',index=False)


if __name__=='__main__':
    main()
