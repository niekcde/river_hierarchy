"""Controlled slope and path-arrangement extension of the synthetic screen.

Reuses the earlier geometry-group split for matched exploratory comparison.
It is not a new untouched validation dataset. No routing is performed.
"""
from pathlib import Path
import json,hashlib,itertools
import numpy as np
import pandas as pd
import networkx as nx
from functools import lru_cache
from sklearn.ensemble import HistGradientBoostingRegressor
from threadpoolctl import threadpool_limits
from geometry_product_search import BASE,YCOLS,ScalarCurve,score
ROOT=Path(__file__).resolve().parents[2]
RUN=BASE.parent.parent
OUT=BASE/'slope_path_arrangement'


def signed_association(lengths,widths):
    l=np.asarray(lengths,float);w=np.asarray(widths,float)
    value=float(np.average(l,weights=w)/l.mean()-1)
    return 0. if abs(value)<1e-12 else value

@lru_cache(None)
def parent_paths(rid,start,end):
    net=json.loads((RUN/'jobs'/f'{rid}.json').read_text())['network']
    g=nx.MultiDiGraph()
    for i,(u,v,l,w,s) in enumerate(net['edges']):g.add_edge(u,v,key=i,length=l,width=w)
    u=[i for i,n in enumerate(net['nodes']) if np.isclose(n[0],start)]
    v=[i for i,n in enumerate(net['nodes']) if np.isclose(n[0],end)]
    assert len(u)==len(v)==1
    paths=list(nx.all_simple_edge_paths(g,u[0],v[0]));assert paths
    lengths=[];widths=[]
    for path in paths:
        ls=np.array([g.edges[e]['length'] for e in path]);ws=np.array([g.edges[e]['width'] for e in path])
        lengths.append(ls.sum());widths.append(ls.sum()/(ls/ws).sum())
    dh=net['nodes'][u[0]][1]-net['nodes'][v[0]][1]
    return lengths,widths,dh


def estimator():
    return HistGradientBoostingRegressor(max_iter=80,max_leaf_nodes=7,min_samples_leaf=20,
        l2_regularization=1.,early_stopping=False,random_state=20260928)


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    d=pd.read_parquet(BASE/'geometry_features.parquet').reset_index(drop=True)
    vals=[]
    for r in d.itertuples():
        lengths,widths,dh=parent_paths(r.parent_run_id,r.domain_start_m,r.domain_end_m)
        assert np.isclose(np.average(lengths,weights=widths),r.parent_equivalent_length,rtol=1e-10)
        slope=float(format(dh/(r.domain_end_m-r.domain_start_m),'.12g'))
        assert np.isclose(slope,r.reference_slope,rtol=1e-10)
        vals.append(dict(parent_path_length_mean=np.mean(lengths),
            signed_length_width_association=signed_association(lengths,widths),
            domain_reference_slope=slope,domain_head_loss_m=dh))
    d=pd.concat([d,pd.DataFrame(vals)],axis=1)
    split=pd.read_csv(BASE/'products/splits.csv')
    d=d.merge(split[['configuration_id','child_state','geometry_group','split','development_fold']],
              on=['configuration_id','child_state','geometry_group'],validate='one_to_one')
    assert len(d)==15309 and d.groupby('geometry_group').split.nunique().max()==1
    d.to_parquet(OUT/'features.parquet',index=False)
    dev=np.flatnonzero(d.split=='development');test=np.flatnonzero(d.split=='test')
    folds=[(np.flatnonzero(d.iloc[dev].development_fold.to_numpy()!=k),
            np.flatnonzero(d.iloc[dev].development_fold.to_numpy()==k)) for k in range(4)]
    for tr,va in folds:assert not set(d.geometry_group.iloc[dev[tr]])&set(d.geometry_group.iloc[dev[va]])
    y=d[YCOLS].to_numpy();states=d.child_state.to_numpy()
    X='resistance_predictor';L='parent_equivalent_length';EL='parent_elongation';D='parent_dominant_width_fraction'
    CV='parent_path_length_cv';A='signed_length_width_association';S='domain_reference_slope'
    compact=[X,L,EL,D]
    sets={'current_product':[X], 'compact':compact,
          'compact_plus_CV':compact+[CV], 'compact_plus_association':compact+[A],
          'compact_plus_CV_association':compact+[CV,A],
          'compact_plus_slope':compact+[S],
          'compact_plus_slope_CV':compact+[S,CV],
          'compact_plus_slope_association':compact+[S,A],
          'compact_plus_all':compact+[CV,A,S],
          'separate_components_plus_all':['resistance_log','length_fraction','width_fraction',L,EL,D,CV,A,S],
          'compact_without_length_plus_all':[X,EL,D,CV,A,S],
          'compact_without_elongation_plus_all':[X,L,D,CV,A,S]}
    cvrows=[];testrows=[];predictions=[]
    for name,cols in sets.items():
        pred=np.zeros((len(dev),3))
        for tr,va in folds:
            for j in range(3):
                model=estimator().fit(d[cols].iloc[dev[tr]],y[dev[tr],j])
                pred[va,j]=model.predict(d[cols].iloc[dev[va]])
        cvrows.extend(dict(model=name,**r) for r in score(y[dev],pred,states[dev]))
        pred=np.zeros((len(test),3))
        for j in range(3):pred[:,j]=estimator().fit(d[cols].iloc[dev],y[dev,j]).predict(d[cols].iloc[test])
        testrows.extend(dict(model=name,**r) for r in score(y[test],pred,states[test]))
        for j,response in enumerate(YCOLS):
            predictions.append(pd.DataFrame(dict(model=name,response=response,state=states[test],
                configuration_id=d.configuration_id.iloc[test].to_numpy(),observed=y[test,j],predicted=pred[:,j])))
        print(name,'done',flush=True)
    pd.DataFrame(cvrows).to_csv(OUT/'development_scores.csv',index=False)
    pd.DataFrame(testrows).to_csv(OUT/'held_out_scores.csv',index=False)
    pd.concat(predictions,ignore_index=True).to_parquet(OUT/'held_out_predictions.parquet',index=False)
    # Single-axis products; all selected using development CV only.
    candidates=['resistance_log','length_fraction','width_fraction','Lambda_matched',X,L,EL,D,CV,A,S]
    raw={};seen={};aliases=[]
    for k in (1,2,3):
        for combo in itertools.combinations(candidates,k):
            label=' * '.join(combo);xx=d[list(combo)].prod(axis=1).to_numpy()
            maximum=np.abs(xx).max();key=hashlib.sha256(np.round(xx/maximum if maximum else xx,10).tobytes()).hexdigest()
            if key in seen:aliases.append(dict(product=label,canonical=seen[key]));continue
            seen[key]=label;raw[label]=xx
    productrows=[]
    for number,(label,xx) in enumerate(raw.items(),1):
        pred=np.zeros((len(dev),3))
        for tr,va in folds:pred[va]=ScalarCurve().fit(xx[dev[tr]],y[dev[tr]]).predict(xx[dev[va]])
        productrows.extend(dict(product=label,**r) for r in score(y[dev],pred,states[dev]))
        if number%50==0:print('products',number,'/',len(raw),flush=True)
    product_scores=pd.DataFrame(productrows);product_scores.to_csv(OUT/'product_development_scores.csv',index=False)
    ranking=product_scores.groupby(['product','response']).r2.mean().unstack().reindex(columns=YCOLS)
    ranking['mean_r2']=ranking.mean(axis=1);ranking['minimum_response_r2']=ranking[YCOLS].min(axis=1)
    ranking=ranking.sort_values('mean_r2',ascending=False);ranking.to_csv(OUT/'product_ranking.csv')
    selected={'shared':ranking.index[0],
              'balanced_shared':ranking.minimum_response_r2.idxmax(),
              **{r:ranking[r].idxmax() for r in YCOLS}}
    scalarrows=[];scalarpred=[];curves=[]
    for name,label in selected.items():
        xx=raw[label];model=ScalarCurve().fit(xx[dev],y[dev]);pred=model.predict(xx[test])
        scalarrows.extend(dict(selection=name,product=label,**r) for r in score(y[test],pred,states[test]))
        grid=np.linspace(xx[dev].min(),xx[dev].max(),300);curve=model.predict(grid)
        for j,response in enumerate(YCOLS):
            scalarpred.append(pd.DataFrame(dict(selection=name,response=response,state=states[test],
                x=xx[test],observed=y[test,j],predicted=pred[:,j])))
            curves.append(pd.DataFrame(dict(selection=name,response=response,x=grid,predicted=curve[:,j])))
    pd.DataFrame(scalarrows).to_csv(OUT/'product_held_out_scores.csv',index=False)
    pd.concat(scalarpred,ignore_index=True).to_parquet(OUT/'product_held_out_predictions.parquet',index=False)
    pd.concat(curves,ignore_index=True).to_csv(OUT/'product_curves.csv',index=False)
    pd.DataFrame(aliases).to_csv(OUT/'equivalent_products.csv',index=False)
    (OUT/'provenance.json').write_text(json.dumps(dict(feature_sets=sets,product_candidates=candidates,
      unique_products=len(raw),selected_products=selected,
      association='width-weighted mean path length / unweighted mean path length - 1',
      slope='parent domain boundary head drop / centerline downstream span; equal to imposed reference slope in this synthetic design',
      validation='same previous grouped development/holdout partition; exploratory continuation, not new untouched validation',
      model='fixed HGB: 80 iterations, 7 leaves, min leaf 20, L2=1; same as previous compact comparison',
      source_sha256=hashlib.sha256((BASE/'geometry_features.parquet').read_bytes()).hexdigest(),
      script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),indent=2))
    print(pd.DataFrame(testrows).groupby(['model','response']).r2.mean().unstack().round(3).to_string())
    print('PRODUCTS',json.dumps(selected,indent=2))
    print(pd.DataFrame(scalarrows).groupby(['selection','response']).r2.mean().unstack().round(3).to_string())

if __name__=='__main__':
    with threadpool_limits(limits=1):main()
