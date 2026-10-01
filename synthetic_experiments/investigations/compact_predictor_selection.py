"""Development-only compact predictor selection for the three current responses.
Standalone scores/correlations, exhaustive pairs/triples, bounded forward extension,
and separately specified parent-child/context blocks. No new routing.
"""
from pathlib import Path
from functools import lru_cache
from itertools import combinations
import sys,json,hashlib
import numpy as np
import pandas as pd
import networkx as nx
from scipy.stats import pearsonr,spearmanr
from sklearn.metrics import r2_score,mean_absolute_error
from threadpoolctl import threadpool_limits
from current_synthetic_analysis import OUT as BASE,RUN,Y,FULL,W,R,LF,E,L,EL,D,CV,A,S,estimator
OUT=BASE/'compact_predictors'
CANDIDATES=[W,R,LF,E,L,EL,D,CV,A,S]
CHANGE=['log_equivalent_length_ratio','log_elongation_ratio','change_path_length_cv','change_width_dominance']
CONTEXT=['upstream_shortest_length_m','downstream_shortest_length_m']
TOLERANCE=.02

@lru_cache(maxsize=128)
def network(rid):
    data=json.loads((RUN/'jobs'/f'{rid}.json').read_text())['network'];g=nx.MultiDiGraph()
    for i,(u,v,length,width,slope) in enumerate(data['edges']):g.add_edge(u,v,key=i,length=length)
    return data,g

def extend_geometry(d):
    d=d.copy()
    d[CHANGE[0]]=np.log(d.child_equivalent_length/d.parent_equivalent_length)
    d[CHANGE[1]]=np.log(d.child_elongation/d.parent_elongation)
    d[CHANGE[2]]=d.child_path_length_cv-d.parent_path_length_cv
    d[CHANGE[3]]=d.child_dominant_width_fraction-d.parent_dominant_width_fraction
    records={}
    for rid,start,end in d[['parent_run_id','domain_start_m','domain_end_m']].drop_duplicates().itertuples(index=False,name=None):
        net,g=network(rid)
        source=[i for i,node in enumerate(net['nodes']) if np.isclose(node[0],start)]
        sink=[i for i,node in enumerate(net['nodes']) if np.isclose(node[0],end)]
        inlets=[n for n in g if g.in_degree(n)==0];outlets=[n for n in g if g.out_degree(n)==0]
        assert len(source)==len(sink)==len(inlets)==len(outlets)==1
        records[rid,start,end]=(nx.shortest_path_length(g,inlets[0],source[0],weight='length'),
                                nx.shortest_path_length(g,sink[0],outlets[0],weight='length'))
    d[CONTEXT]=[records[row.parent_run_id,row.domain_start_m,row.domain_end_m] for row in d.itertuples()]
    assert np.isfinite(d[CANDIDATES+CHANGE+CONTEXT]).all().all()
    return d

def scores(obs,pred,states):
    rows=[]
    for state in sorted(set(states)):
        m=states==state
        for j,y in enumerate(Y):
            rows.append(dict(child_state=state,response=y,n=int(m.sum()),r2=r2_score(obs[m,j],pred[m,j]),mae_pp=mean_absolute_error(obs[m,j],pred[m,j])))
    return rows

def predict(train,test,cols):
    out=np.empty((len(test),len(Y)))
    for j,y in enumerate(Y):
        model=estimator('boosted').fit(train[list(cols)],train[y])
        out[:,j]=model.predict(test[list(cols)])
    return out

def treatment_transfer(d,named,registry):
    """A separate omitted-treatment diagnostic; it does not select subsets."""
    rows=[]
    labels=['compact_reference','selected_shared','compact_plus_change','selected_plus_change','selected_plus_change_context']
    for state in sorted(d.child_state.unique()):
        te=d.child_state==state;train=d[~te];test=d[te]
        assert not set(train.geometry_group)&set(test.geometry_group)
        for label in labels:
            key=named[label];p=predict(train,test,registry[key])
            rows.extend(dict(label=label,feature_set=key,**r) for r in scores(test[Y].to_numpy(),p,test.child_state.to_numpy()))
    pd.DataFrame(rows).to_csv(OUT/'omitted_treatment_scores.csv',index=False)

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    d=extend_geometry(pd.read_parquet(BASE/'current_comparisons.parquet'))
    d.to_parquet(OUT/'features.parquet',index=False)
    dev=d[d.split=='development'].reset_index(drop=True);test=d[d.split=='test'].reset_index(drop=True)
    assert not set(dev.geometry_group)&set(test.geometry_group)
    folds=sorted(dev.development_fold.unique());assert len(folds)==4
    for fold in folds:
        assert not set(dev.loc[dev.development_fold==fold,'geometry_group'])&set(dev.loc[dev.development_fold!=fold,'geometry_group'])
    correlations=[]
    for state,g in list(dev.groupby('child_state'))+[('pooled',dev)]:
        for feature in CANDIDATES+CHANGE+CONTEXT:
            for y in Y:
                constant=(np.ptp(g[feature])<=1e-12*max(1.,float(g[feature].abs().max()))) or g[y].nunique()==1
                correlations.append(dict(child_state=state,feature=feature,response=y,
                    pearson=np.nan if constant else pearsonr(g[feature],g[y]).statistic,
                    spearman=np.nan if constant else spearmanr(g[feature],g[y]).statistic,
                    constant=constant))
    pd.DataFrame(correlations).to_csv(OUT/'development_correlations.csv',index=False)
    registry={};cvrows=[];means={};order={c:i for i,c in enumerate(CANDIDATES+CHANGE+CONTEXT)}
    def evaluate(cols,origin):
        cols=tuple(sorted(set(cols),key=lambda c:order[c]));key=' + '.join(cols)
        if key in registry:return key
        prediction=np.empty((len(dev),len(Y)))
        for fold in folds:
            va=dev.development_fold==fold;tr=~va
            prediction[va]=predict(dev[tr],dev[va],cols)
        rows=scores(dev[Y].to_numpy(),prediction,dev.child_state.to_numpy())
        cvrows.extend(dict(feature_set=key,n_axes=len(cols),origin=origin,**r) for r in rows)
        registry[key]=list(cols)
        means[key]={y:float(np.mean([r['r2'] for r in rows if r['response']==y])) for y in Y}
        means[key]['mean_r2']=float(np.mean(list(means[key].values())))
        return key
    # Fixed candidate list. Search never reads held-out outcomes.
    for size in [1,2,3]:
        for cols in combinations(CANDIDATES,size):evaluate(cols,'exhaustive_up_to_three')
        print('Completed size',size,'total sets',len(registry),flush=True)
    small=list(registry)
    best_by_size={size:max([k for k in small if len(registry[k])==size],key=lambda k:means[k]['mean_r2']) for size in [1,2,3]}
    chain=best_by_size[3]
    for size in [4,5,6]:
        candidates=[evaluate(registry[chain]+[c],'forward_extension') for c in CANDIDATES if c not in registry[chain]]
        chain=max(candidates,key=lambda k:means[k]['mean_r2']);best_by_size[size]=chain
        print('Forward size',size,'done',flush=True)
    named={'full_reference':evaluate(FULL,'reference'),
           'compact_reference':evaluate([W,L,EL,D],'reference'),
           'width_length_arrangement':evaluate([D,CV,A],'reference'),
           'omega_plus_arrangement':evaluate([W,D,CV,A],'reference')}
    # Complement forward addition with backward removal so 7/8-axis near-full
    # candidates are considered instead of jumping directly from six to nine.
    backward=named['full_reference']
    for size in range(len(FULL)-1,2,-1):
        candidates=[evaluate([x for x in registry[backward] if x!=c],'backward_reduction') for c in registry[backward]]
        backward=max(candidates,key=lambda k:means[k]['mean_r2'])
        if size not in best_by_size or means[backward]['mean_r2']>means[best_by_size[size]]['mean_r2']:
            best_by_size[size]=backward
        print('Backward size',size,'done',flush=True)
    core=[k for k in registry]
    best=max(core,key=lambda k:means[k]['mean_r2'])
    # Near-best means within .02 R² separately for EVERY response, not just mean.
    qualifying=[k for k in core if all(means[k][y]>=means[best][y]-TOLERANCE for y in Y)]
    shared=min(qualifying,key=lambda k:(len(registry[k]),-means[k]['mean_r2']))
    named['selected_shared']=shared
    for y in Y:
        winner=max(core,key=lambda k:means[k][y]);eligible=[k for k in core if means[k][y]>=means[winner][y]-TOLERANCE]
        named['selected_'+y]=min(eligible,key=lambda k:(len(registry[k]),-means[k][y]))
    for size,key in best_by_size.items():named[f'best_{size}_axes']=key
    # Separate candidate blocks tested on a prespecified compact reference and
    # the selected shared set. Do not feed them back into the original search.
    for base_name,cols in [('compact',[W,L,EL,D]),('selected',registry[shared])]:
        for label,extra in [('change',CHANGE),('context',CONTEXT),('change_context',CHANGE+CONTEXT)]:
            named[base_name+'_plus_'+label]=evaluate(cols+extra,'block_extension')
    for c in CHANGE+CONTEXT:named['standalone_'+c]=evaluate([c],'additional_standalone')
    for c in CANDIDATES:named['standalone_'+c]=' + '.join([c])
    pd.DataFrame(cvrows).to_csv(OUT/'development_scores.csv',index=False)
    ranking=pd.DataFrame([dict(feature_set=k,n_axes=len(registry[k]),**v) for k,v in means.items()])
    ranking.sort_values('mean_r2',ascending=False).to_csv(OUT/'development_ranking.csv',index=False)
    # One evaluation of each unique preselected set on the reused holdout.
    held=[];predrows=[]
    for key in sorted(set(named.values())):
        p=predict(dev,test,registry[key]);rows=scores(test[Y].to_numpy(),p,test.child_state.to_numpy())
        held.extend(dict(feature_set=key,n_axes=len(registry[key]),**r) for r in rows)
        for j,y in enumerate(Y):
            predrows.append(pd.DataFrame(dict(configuration_id=test.configuration_id,child_state=test.child_state,
                feature_set=key,response=y,observed=test[y],predicted=p[:,j])))
    pd.DataFrame(held).to_csv(OUT/'held_out_scores.csv',index=False)
    pd.concat(predrows,ignore_index=True).to_parquet(OUT/'held_out_predictions.parquet',index=False)
    (OUT/'selection.json').write_text(json.dumps(dict(named_sets=named,features=registry,best_development_set=best,
        tolerance_r2=TOLERANCE,selection='smallest evaluated set within .02 of best-mean set for every response; ties by development mean R2',
        search='all 1/2/3-axis combinations from 10 candidates; greedy shared-mean extension to 6 and backward reduction from 9 to 3; fixed reference sets; not exhaustive global minimization',
        omega_counts_as_one_axis_but_uses_three_components=True,
        validation='4 geometry-group development folds; reused exploratory geometry holdout; no new untouched confirmation',
        model='fixed shallow histogram gradient boosting matching current benchmark',
        context='shortest directed physical path from network inlet to parent-domain upstream boundary and domain downstream boundary to outlet; not all outside channel length',
        change='natural log child/parent equivalent length and elongation; child minus parent CV and dominance',
        selection_on_development_only=True,code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),indent=2))
    treatment_transfer(d,named,registry)
    print('Selected:',registry[shared],means[shared],flush=True)
    print('Best development:',registry[best],means[best],flush=True)
if __name__=='__main__':
    with threadpool_limits(limits=1):main()
