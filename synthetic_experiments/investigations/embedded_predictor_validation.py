"""Test existing predictor specifications on the separate embedding extension.
Rebuild unchanged models from the ORIGINAL development rows only. No extension
rows enter fitting, feature selection or tuning. Original-grid anchors excluded
from extension scores. No routing is performed.
"""
from pathlib import Path
import sys,json,hashlib
from functools import lru_cache
import numpy as np
import pandas as pd
import networkx as nx
from sklearn.metrics import r2_score,mean_absolute_error
from threadpoolctl import threadpool_limits
ROOT=Path(__file__).resolve().parents[2]
for p in (ROOT,ROOT/'synthetic_experiments/src',ROOT/'RAPID/src'):sys.path.insert(0,str(p))
from synthetic_experiments.networks import make_states
from geometry_predictor_screen import domain_metrics
from embedded_width_length_extension import embed
from current_synthetic_analysis import estimator,Y
from analysis_paths import RUN, EXTENSION as EXT
BASE=RUN/'current_analysis'
OUT=EXT/'predictor_validation'


def load_extension():
    plan=pd.read_csv(EXT/'plan.csv');keys=['configuration_id','child_state','parent_run_id','child_run_id']
    for phase in ['full','pilot']:
        path=EXT/f'{phase}_results.csv'
        if not path.exists():continue
        d=pd.read_csv(path);expected=plan if phase=='full' else plan[plan.is_pilot]
        if len(d)==len(expected) and len(d[keys].merge(expected[keys],on=keys,validate='one_to_one'))==len(expected):
            return d,phase
    raise RuntimeError('No complete extension results matching the current plan')

@lru_cache(None)
def states(a,m):
    return make_states(dict(total_width_m=500,reference_slope=1e-4,inner_fraction=.5,
                           total_length_m=20000,internal_location='central',outer_a_fraction=a,length_multiplier=m))[0]

def geometry(d):
    rows=[]
    for r in d.itertuples():
        graphs=states(r.outer_a_fraction,r.length_multiplier)
        shift=(r.network_extent_m-20000)/2;a=r.domain_start_m-shift;b=r.domain_end_m-shift
        row={}
        for side,state in [('parent',r.parent_state),('child',r.child_state)]:
            g=graphs[state];metrics=domain_metrics(g,a,b)
            row.update({side+'_'+key:value for key,value in metrics.items()})
        parent=graphs[r.parent_state]
        u=next(n for n in parent if np.isclose(parent.nodes[n]['x'],a))
        v=next(n for n in parent if np.isclose(parent.nodes[n]['x'],b))
        paths=list(nx.all_simple_edge_paths(parent,u,v))
        lengths=[sum(parent.edges[e]['length_m'] for e in path) for path in paths]
        row['signed_length_width_association']=row['parent_equivalent_length']/np.mean(lengths)-1
        row['domain_reference_slope']=(parent.nodes[u]['head_m']-parent.nodes[v]['head_m'])/(b-a)
        assert np.isclose(row['domain_reference_slope'],r.reference_slope)
        embedded=embed(parent,r.network_extent_m,r.reference_slope)
        inlet=next(n for n in embedded if embedded.in_degree(n)==0);outlet=next(n for n in embedded if embedded.out_degree(n)==0)
        row['upstream_shortest_length_m']=nx.shortest_path_length(embedded,inlet,u,weight='length_m')
        row['downstream_shortest_length_m']=nx.shortest_path_length(embedded,v,outlet,weight='length_m')
        row['log_equivalent_length_ratio']=np.log(row['child_equivalent_length']/row['parent_equivalent_length'])
        row['log_elongation_ratio']=np.log(row['child_elongation']/row['parent_elongation'])
        row['change_path_length_cv']=row['child_path_length_cv']-row['parent_path_length_cv']
        row['change_width_dominance']=row['child_dominant_width_fraction']-row['parent_dominant_width_fraction']
        row['resistance_log']=-np.log10(r.Lambda_matched)
        rows.append(row)
    result=pd.concat([d.reset_index(drop=True),pd.DataFrame(rows)],axis=1)
    result['is_original_grid_anchor']=(result.network_extent_m==20000)&result.outer_a_fraction.isin([.5,.8,.9])&result.length_multiplier.isin([1.,1.25,1.5])
    return result

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    raw,phase=load_extension();d=geometry(raw)
    selection=json.loads((BASE/'compact_predictors/selection.json').read_text())
    labels={'omega':['omega']}
    for label,key in [('three_axes','best_3_axes'),('selected_shared','selected_shared'),('selected_plus_change','selected_plus_change'),('selected_plus_change_context','selected_plus_change_context')]:
        labels[label]=selection['features'][selection['named_sets'][key]]
    allcols=list(dict.fromkeys(c for cols in labels.values() for c in cols))
    train=pd.read_parquet(BASE/'compact_predictors/features.parquet')
    original=train.copy();train=train[train.split=='development']
    assert np.isfinite(d[allcols+Y]).all().all()
    # The 20 km baseline must reproduce the original descriptors and responses.
    keys=['child_state','outer_a_fraction','length_multiplier']
    anchors=d[d.is_original_grid_anchor]
    old=original[(original.total_length_m==20000)&(original.total_width_m==500)&(original.reference_slope==1e-4)&(original.inner_fraction==.5)&(original.internal_location=='central')]
    check=anchors.merge(old,on=keys,suffixes=('_extension','_original'),validate='one_to_one')
    assert len(check)==len(anchors)
    for c in allcols+Y:assert np.allclose(check[c+'_extension'],check[c+'_original'],rtol=1e-8,atol=1e-8),c
    d.to_parquet(OUT/f'{phase}_features.parquet',index=False)
    predictions=[];scores=[];ranges=[]
    novel=d[~d.is_original_grid_anchor]
    for col in allcols:
        lo,hi=train[col].min(),train[col].max()
        ranges.append(dict(feature=col,training_min=lo,training_max=hi,extension_min=novel[col].min(),extension_max=novel[col].max(),outside_fraction=((novel[col]<lo-1e-12)|(novel[col]>hi+1e-12)).mean()))
    with threadpool_limits(limits=1):
      for label,cols in labels.items():
        for y in Y:
            # Reconstruct the existing fit exactly; do not fit to extension data.
            fit=estimator('boosted').fit(train[cols],train[y]);p=fit.predict(d[cols])
            frame=d[['configuration_id','child_state','outer_a_fraction','length_multiplier','network_extent_m','omega','is_original_grid_anchor','within_one_dt',y]].copy().rename(columns={y:'observed'})
            frame['predicted']=p;frame['model']=label;frame['response']=y
            predictions.append(frame)
            for scope,g in [('novel_all',frame[~frame.is_original_grid_anchor]),('novel_20km',frame[(~frame.is_original_grid_anchor)&(frame.network_extent_m==20000)]),('novel_embedded',frame[frame.network_extent_m>20000])]:
                for state,h in list(g.groupby('child_state'))+[('pooled',g)]:
                    if len(h)<2:continue
                    scores.append(dict(model=label,response=y,scope=scope,child_state=state,n=len(h),
                        r2=r2_score(h.observed,h.predicted) if h.observed.var(ddof=0)>1e-20 else np.nan,
                        mae_pp=mean_absolute_error(h.observed,h.predicted),bias_pp=(h.predicted-h.observed).mean()))
    pd.concat(predictions,ignore_index=True).to_parquet(OUT/f'{phase}_predictions.parquet',index=False)
    pd.DataFrame(scores).to_csv(OUT/f'{phase}_scores.csv',index=False)
    pd.DataFrame(ranges).to_csv(OUT/f'{phase}_feature_ranges.csv',index=False)
    (OUT/f'{phase}_provenance.json').write_text(json.dumps(dict(phase=phase,plan_id=json.loads((EXT/'manifest.json').read_text())['plan_id'],
        n_original_development_rows=len(train),n_extension_rows=len(d),n_scored_novel_rows=len(novel),n_excluded_anchors=len(anchors),
        feature_sets=labels,training='original development rows only, fixed boosted model, no feature reselection or parameter tuning',
        warning='out-of-range diagnostic on a deliberately selected extension, not untouched confirmatory validation',
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),indent=2))
    print('Extension validation:',phase,len(novel),'novel comparisons;',len(anchors),'anchors excluded')
    print(pd.DataFrame(scores).query("scope=='novel_all' and child_state!='pooled'").groupby(['model','response'])[['r2','mae_pp']].mean().round(3).to_string())
if __name__=='__main__':main()
