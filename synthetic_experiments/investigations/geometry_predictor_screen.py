"""Geometry-only predictor screen on the balanced synthetic grid.

Recompute existing unit definitions over matched domains in parent and child.
Group identical parent/child geometries (ignoring slope and clock) into the same
validation fold. This is within-family interpolation, not real-network validation.
Run from repository root. No routing or source-package changes.
"""
from pathlib import Path
import sys,json,hashlib
from functools import lru_cache
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/'synthetic_experiments/src'))
from synthetic_experiments.networks import make_states
import numpy as np
import pandas as pd
import networkx as nx
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error
from threadpoolctl import threadpool_limits
from hierarchy_level_definition.metrics.unit_metrics import (
    weighted_harmonic_mean_width,normalized_entropy,coefficient_of_variation,weighted_mean)
RUN=ROOT/'synthetic_experiments/outputs/full_grid_island'
OUT=RUN/'factor_analysis/geometry_predictors'
RESPONSES=['lag_shift_pct','temporal_spread_change_pct','peak_excess_change_pct']
GEOMETRY=['equivalent_length','equivalent_width','elongation','path_length_cv',
          'path_length_range_norm','effective_n_paths_width','dominant_width_fraction',
          'topologic_complexity_score','branching_density_by_length']

@lru_cache(None)
def network(rid):
    return json.loads((RUN/'jobs'/f'{rid}.json').read_text())['network']

def domain_metrics(g,start,end):
    sources=[n for n,d in g.nodes(data=True) if np.isclose(d['x'],start)]
    sinks=[n for n,d in g.nodes(data=True) if np.isclose(d['x'],end)]
    assert len(sources)==len(sinks)==1
    paths=list(nx.all_simple_edge_paths(g,sources[0],sinks[0]));assert paths
    lengths=[];widths=[];selected=set()
    for path in paths:
        ls=[g.edges[e]['length_m'] for e in path];ws=[g.edges[e]['w'] for e in path]
        lengths.append(sum(ls));widths.append(weighted_harmonic_mean_width(ls,ws));selected.update(path)
    domain=g.edge_subgraph(selected)
    ib=sum(domain.out_degree(n)>1 for n in domain if n!=sources[0])
    ic=sum(domain.in_degree(n)>1 for n in domain if n!=sinks[0])
    eq_l=weighted_mean(lengths,widths);eq_w=sum(widths)
    entropy,evenness,effective=normalized_entropy(widths)
    return dict(equivalent_length=eq_l,equivalent_width=eq_w,elongation=eq_l/eq_w,
        path_length_cv=coefficient_of_variation(lengths),path_length_range_norm=np.ptp(lengths)/eq_l,
        effective_n_paths_width=effective,dominant_width_fraction=max(widths)/eq_w,
        topologic_complexity_score=np.log1p(len(paths))+np.log1p(ib)+np.log1p(ic),
        branching_density_by_length=(ib+ic)/eq_l,n_valid_paths=len(paths),
        width_entropy=entropy,width_evenness=evenness)

@lru_cache(None)
def geometry_identity(rid):
    net=network(rid)
    # Preserve node IDs/connectivity and geometric lengths/widths, omit head/slope.
    return dict(x=[n[0] for n in net['nodes']],edges=[e[:4] for e in net['edges']])

def features(frame):
    records={}
    factors=['total_width_m','reference_slope','outer_a_fraction','inner_fraction',
             'length_multiplier','total_length_m','internal_location']
    # Canonical routing jobs remove redundant junctions. Reconstruct the original
    # state graphs so matched boundaries remain explicit, including inner collapse.
    for cid,part in frame.groupby('configuration_id',sort=False):
        states,_=make_states(part.iloc[0][factors].to_dict())
        for index,r in zip(part.index,part.itertuples()):
            parent=domain_metrics(states[r.parent_state],r.domain_start_m,r.domain_end_m)
            child=domain_metrics(states[r.child_state],r.domain_start_m,r.domain_end_m)
            identity=[geometry_identity(r.parent_run_id),geometry_identity(r.child_run_id),r.domain_start_m,r.domain_end_m]
            group=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()[:24]
            row=dict(geometry_group=group)
            for key in parent:
                row['parent_'+key]=parent[key];row['child_'+key]=child[key];row['delta_'+key]=child[key]-parent[key]
            records[index]=row
    rows=[records[i] for i in frame.index]
    f=pd.concat([frame.reset_index(drop=True),pd.DataFrame(rows)],axis=1)
    f['resistance_log']=-np.log10(f.Lambda_matched)
    f['length_exposure']=f.length_fraction*f.width_fraction
    f['length_resistance']=f.length_fraction*f.resistance_log
    f['exposure_resistance']=f.width_fraction*f.resistance_log
    assert np.allclose(f.resistance_predictor,f.resistance_log*f.length_fraction*f.width_fraction)
    return f

def sets():
    base=['resistance_log','length_fraction','width_fraction']
    geom=['parent_'+k for k in GEOMETRY]
    changes=['delta_'+k for k in GEOMETRY]
    return {
      'mean_baseline':[],
      'lambda_only':['Lambda_matched'],
      'resistance_only':['resistance_log'],
      'length_fraction_only':['length_fraction'],
      'exposure_only':['width_fraction'],
      'resistance_plus_length':base[:2],
      'resistance_plus_exposure':[base[0],base[2]],
      'length_plus_exposure':base[1:],
      'three_components':base,
      'current_weighted_predictor':['resistance_predictor'],
      'length_resistance_product':['length_resistance'],
      'exposure_resistance_product':['exposure_resistance'],
      'length_exposure_product':['length_exposure'],
      **{k+'_only':['parent_'+k] for k in GEOMETRY},
      'existing_geometry':geom,
      'components_plus_geometry':base+geom,
      'weighted_predictor_plus_geometry':['resistance_predictor']+geom,
      'components_plus_geometry_changes':base+geom+changes,
    }

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    frame=pd.read_csv(RUN/'factor_analysis/event_shape_comparisons.csv')
    f=features(frame);f.to_parquet(OUT/'geometry_features.parquet',index=False)
    feature_sets=sets();scores=[];correlations=[];predictions=[];fold_rows=[]
    columns=sorted(set(c for fs in feature_sets.values() for c in fs))
    assert np.isfinite(f[columns+RESPONSES]).all().all()
    for state,g in f.groupby('child_state',sort=False):
        g=g.reset_index(drop=True)
        assert g.geometry_group.nunique()>=5
        splits=list(GroupKFold(5).split(g,groups=g.geometry_group))
        fold=np.full(len(g),-1)
        for i,(train,test) in enumerate(splits):
            assert not set(g.geometry_group.iloc[train])&set(g.geometry_group.iloc[test])
            fold[test]=i
        for cid,group,idx in zip(g.configuration_id,g.geometry_group,fold):fold_rows.append(dict(state=state,configuration_id=cid,geometry_group=group,fold=int(idx)))
        for response in RESPONSES:
            y=g[response].to_numpy()
            for col in columns:
                rho=spearmanr(g[col],y).statistic if np.ptp(g[col])>1e-12 else np.nan
                correlations.append(dict(state=state,response=response,feature=col,spearman_rho=rho))
            for label,cols in feature_sets.items():
                pred=np.zeros(len(g));active_counts=[]
                for train,test in splits:
                    active=[c for c in cols if np.ptp(g[c].iloc[train])>1e-12]
                    active_counts.append(len(active))
                    if not active:pred[test]=y[train].mean();continue
                    model=HistGradientBoostingRegressor(max_iter=80,max_leaf_nodes=7,
                        min_samples_leaf=20,l2_regularization=1.,learning_rate=.1,
                        early_stopping=False,random_state=20260928)
                    model.fit(g[active].iloc[train],y[train]);pred[test]=model.predict(g[active].iloc[test])
                scores.append(dict(state=state,response=response,feature_set=label,
                    r2=r2_score(y,pred),mae_pp=mean_absolute_error(y,pred),
                    rmse_pp=np.sqrt(mean_squared_error(y,pred)),n=len(g),
                    n_geometry_groups=g.geometry_group.nunique(),min_active_features=min(active_counts)))
                predictions.append(pd.DataFrame(dict(state=state,response=response,feature_set=label,
                    configuration_id=g.configuration_id,fold=fold,observed=y,predicted=pred)))
        print(state,'complete',flush=True)
    scores=pd.DataFrame(scores)
    scores.to_csv(OUT/'cv_scores.csv',index=False)
    pd.DataFrame(correlations).to_csv(OUT/'single_variable_correlations.csv',index=False)
    pd.DataFrame(fold_rows).to_csv(OUT/'validation_folds.csv',index=False)
    pd.concat(predictions,ignore_index=True).to_parquet(OUT/'held_out_predictions.parquet',index=False)
    manifest=json.loads((RUN/'manifest.json').read_text())
    (OUT/'provenance.json').write_text(json.dumps(dict(plan_id=manifest['plan_id'],
      script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
      feature_sets=feature_sets,responses=RESPONSES,
      method='5-fold grouped CV per collapse type; identical parent-child geometry across slopes in same fold',
      model='HistGradientBoostingRegressor: 80 iterations, 7 leaves, min leaf 20, l2 1, learning rate .1; fixed settings',
      metrics='existing path/unit definitions applied to matched domain, parent and child separately; changes child minus parent',
      excluded='slope, head, celerity, routing K, input pulse duration, hydrograph-derived predictors',
      scope='synthetic within-family generalization; no real-network validation; model selection is exploratory'),indent=2))
    print(scores.groupby(['response','feature_set']).r2.mean().unstack(0).round(3).to_string())

if __name__=='__main__':
    with threadpool_limits(limits=1):main()
