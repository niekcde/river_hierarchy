"""Select parsimonious geometry products and audit added geometric information.

One scalar-to-response curve across ALL collapse types, with three separate
responses. No response-derived predictors. Products contain 1-3 distinct existing
variables. A grouped 25% split is reserved from this search; selection uses only
four-fold development CV. Metrics macro-average R2 across collapse types.
"""
from pathlib import Path
import json,hashlib,itertools
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit,GroupKFold
from sklearn.preprocessing import SplineTransformer
from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import r2_score,mean_absolute_error
from threadpoolctl import threadpool_limits
ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'synthetic_experiments/outputs/full_grid_island/factor_analysis/geometry_predictors'
OUT=BASE/'products'
YCOLS=['lag_shift_pct','temporal_spread_change_pct','peak_excess_change_pct']
GEOM=['parent_'+k for k in ['equivalent_length','equivalent_width','elongation','path_length_cv',
      'path_length_range_norm','effective_n_paths_width','dominant_width_fraction',
      'topologic_complexity_score','branching_density_by_length']]
COMPONENTS=['resistance_log','length_fraction','width_fraction']
CANDIDATES=COMPONENTS+['Lambda_matched','resistance_predictor']+GEOM

class ScalarCurve:
    """Fixed small quadratic spline with training-only knots, no tuning."""
    def fit(self,x,y):
        x=np.asarray(x,float).reshape(-1);self.mean=np.mean(y,axis=0)
        self.center=float(np.mean(x));self.scale=float(np.std(x))
        self.spline=None
        if self.scale<=1e-12*max(1,abs(self.center)):return self
        z=(x-self.center)/self.scale
        knots=np.unique(np.quantile(z,[0,1/3,2/3,1]))
        if len(knots)<2:return self
        self.spline=SplineTransformer(degree=2,knots=knots[:,None],include_bias=False,extrapolation='linear')
        b=self.spline.fit_transform(z[:,None])
        self.model=Ridge(alpha=1.).fit(b,y)
        return self
    def predict(self,x):
        x=np.asarray(x,float).reshape(-1)
        if self.spline is None:return np.tile(self.mean,(len(x),1))
        b=self.spline.transform(((x-self.center)/self.scale)[:,None])
        return self.model.predict(b)


def score(y,pred,states):
    rows=[]
    for state in sorted(set(states)):
        mask=np.asarray(states)==state
        for j,response in enumerate(YCOLS):
            rows.append(dict(state=state,response=response,r2=r2_score(y[mask,j],pred[mask,j]),
                             mae_pp=mean_absolute_error(y[mask,j],pred[mask,j]),n=int(mask.sum())))
    return rows


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    d=pd.read_parquet(BASE/'geometry_features.parquet').reset_index(drop=True)
    y=d[YCOLS].to_numpy();states=d.child_state.to_numpy();groups=d.geometry_group.to_numpy()
    dev,test=next(GroupShuffleSplit(n_splits=1,test_size=.25,random_state=20260928).split(d,groups=groups))
    assert not set(groups[dev])&set(groups[test])
    folds=list(GroupKFold(4).split(dev,groups=groups[dev]))
    assignment=np.full(len(d),'test',dtype=object);assignment[dev]='development'
    fold_assignment=np.full(len(d),-1)
    for fold,(tr,va) in enumerate(folds):
        assert not set(groups[dev[tr]])&set(groups[dev[va]])
        fold_assignment[dev[va]]=fold
    d[['configuration_id','child_state','geometry_group']].assign(split=assignment,development_fold=fold_assignment).to_csv(OUT/'splits.csv',index=False)
    combinations=[c for size in (1,2,3) for c in itertools.combinations(CANDIDATES,size)]
    raw={};canonical={};aliases=[];seen={}
    # Collapse algebraically/proportionally identical products using predictors only.
    for combo in combinations:
        label=' * '.join(combo);x=d[list(combo)].prod(axis=1).to_numpy()
        scale=np.max(np.abs(x))
        key=hashlib.sha256(np.round(x/scale if scale else x,10).tobytes()).hexdigest()
        if key in seen:aliases.append(dict(product=label,canonical=seen[key]));continue
        seen[key]=label;raw[label]=x;canonical[label]=list(combo)
    pd.DataFrame(aliases).to_csv(OUT/'equivalent_products.csv',index=False)
    cvrows=[]
    for number,(label,x) in enumerate(raw.items(),1):
        pred=np.zeros((len(dev),3))
        for tr,va in folds:
            pred[va]=ScalarCurve().fit(x[dev[tr]],y[dev[tr]]).predict(x[dev[va]])
        cvrows.extend(dict(product=label,n_factors=len(canonical[label]),**r) for r in score(y[dev],pred,states[dev]))
        if number%50==0:print('Products',number,'/',len(raw),flush=True)
    cv=pd.DataFrame(cvrows);cv.to_csv(OUT/'product_development_scores.csv',index=False)
    ranking=cv.groupby(['product','response']).r2.mean().unstack('response').reindex(columns=YCOLS)
    ranking['mean_r2']=ranking.mean(axis=1);ranking['worst_response_r2']=ranking[YCOLS].min(axis=1)
    ranking['n_factors']=[len(canonical[k]) for k in ranking.index]
    ranking=ranking.sort_values(['mean_r2','n_factors'],ascending=[False,True]);ranking.to_csv(OUT/'product_ranking.csv')
    common=ranking.index[0]
    balanced=ranking.sort_values(['worst_response_r2','mean_r2'],ascending=False).index[0]
    specific={response:ranking[response].idxmax() for response in YCOLS}
    baseline=' * '.join(COMPONENTS)
    # Locate the exact primary product if algebraic deduplication gave it an alias.
    if baseline not in raw:baseline=next(r['canonical'] for r in aliases if r['product']==baseline)
    selected={'current_product':baseline,'shared_product':common,'balanced_shared_product':balanced,
              **{f'best_for_{r}':p for r,p in specific.items()}}
    holdout=[];predrows=[];curves=[]
    for label,product in selected.items():
        model=ScalarCurve().fit(raw[product][dev],y[dev]);pred=model.predict(raw[product][test])
        holdout.extend(dict(model=label,product=product,**r) for r in score(y[test],pred,states[test]))
        for j,response in enumerate(YCOLS):
            predrows.append(pd.DataFrame(dict(model=label,product=product,response=response,
                configuration_id=d.configuration_id.iloc[test].to_numpy(),state=states[test],
                x=raw[product][test],observed=y[test,j],predicted=pred[:,j])))
        xx=np.linspace(raw[product][dev].min(),raw[product][dev].max(),300);yy=model.predict(xx)
        for j,response in enumerate(YCOLS):curves.append(pd.DataFrame(dict(model=label,response=response,x=xx,predicted=yy[:,j])))
    # Added-information ablations use a fixed flexible model. They are not scalar formulas.
    full=COMPONENTS+GEOM
    sets={'current_product':['resistance_predictor'],'all_geometry':full,
          **{'add_'+g:['resistance_predictor',g] for g in GEOM},
          **{'drop_'+g:[v for v in full if v!=g] for g in GEOM}}
    blocks={'size':['parent_equivalent_length','parent_equivalent_width','parent_elongation'],
            'length_diversity':['parent_path_length_cv','parent_path_length_range_norm'],
            'width_balance':['parent_effective_n_paths_width','parent_dominant_width_fraction'],
            'topology':['parent_topologic_complexity_score','parent_branching_density_by_length']}
    sets.update({'drop_block_'+k:[v for v in full if v not in cols] for k,cols in blocks.items()})
    compact={
        'compact_X_elongation_dominance':['resistance_predictor','parent_elongation','parent_dominant_width_fraction'],
        'compact_X_length_dominance':['resistance_predictor','parent_equivalent_length','parent_dominant_width_fraction'],
        'compact_X_length_elongation_dominance':['resistance_predictor','parent_equivalent_length','parent_elongation','parent_dominant_width_fraction']}
    sets.update(compact)
    ablations=[]
    for label,cols in sets.items():
        pred=np.zeros((len(dev),3))
        for tr,va in folds:
            active=[c for c in cols if np.ptp(d[c].iloc[dev[tr]])>1e-12]
            for j in range(3):
                if not active:pred[va,j]=y[dev[tr],j].mean();continue
                model=HistGradientBoostingRegressor(max_iter=80,max_leaf_nodes=7,min_samples_leaf=20,
                    l2_regularization=1.,early_stopping=False,random_state=20260928)
                model.fit(d[active].iloc[dev[tr]],y[dev[tr],j]);pred[va,j]=model.predict(d[active].iloc[dev[va]])
        ablations.extend(dict(feature_set=label,**r) for r in score(y[dev],pred,states[dev]))
        print('Ablation',label,flush=True)
    pd.DataFrame(ablations).to_csv(OUT/'ablation_development_scores.csv',index=False)
    # Select a compact shared input set on development data only.
    compact_scores=pd.DataFrame(ablations)
    compact_scores=compact_scores[compact_scores.feature_set.isin(compact)].groupby('feature_set').r2.mean()
    compact_winner=compact_scores.idxmax()
    compact_holdout=[]
    for label,cols in compact.items():
        pp=np.zeros((len(test),3))
        for j in range(3):
            model=HistGradientBoostingRegressor(max_iter=80,max_leaf_nodes=7,min_samples_leaf=20,l2_regularization=1.,early_stopping=False,random_state=20260928)
            model.fit(d[cols].iloc[dev],y[dev,j]);pp[:,j]=model.predict(d[cols].iloc[test])
        compact_holdout.extend(dict(model=label,product='separate inputs',**r) for r in score(y[test],pp,states[test]))
    holdout.extend(compact_holdout)
    # One predeclared high-information comparator on the held-out split.
    pred=np.zeros((len(test),3))
    for j in range(3):
        model=HistGradientBoostingRegressor(max_iter=80,max_leaf_nodes=7,min_samples_leaf=20,l2_regularization=1.,early_stopping=False,random_state=20260928)
        model.fit(d[full].iloc[dev],y[dev,j]);pred[:,j]=model.predict(d[full].iloc[test])
    holdout.extend(dict(model='all_geometry_multivariable',product='separate inputs',**r) for r in score(y[test],pred,states[test]))
    pd.DataFrame(holdout).to_csv(OUT/'held_out_scores.csv',index=False)
    pd.concat(predrows,ignore_index=True).to_parquet(OUT/'held_out_scalar_predictions.parquet',index=False)
    pd.concat(curves,ignore_index=True).to_csv(OUT/'scalar_curves.csv',index=False)
    (OUT/'provenance.json').write_text(json.dumps(dict(candidates=CANDIDATES,raw_products=len(combinations),
      unique_products=len(raw),selected=selected,ablation_sets=sets,blocks=blocks,
      compact_sets=compact,compact_selected_on_development=compact_winner,
      selection='maximize mean development CV R2, equally weighting responses and collapse types; balanced variant maximizes minimum response mean R2',
      scalar_model='quadratic spline with up to four training-quantile knots; ridge alpha 1; one curve across all collapse types per response',
      validation='25% geometry groups reserved from this search; remaining groups used for 4-fold development CV',
      feature_source_sha256=hashlib.sha256((BASE/'geometry_features.parquet').read_bytes()).hexdigest(),
      script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
      limits='synthetic interpolation; no real-world validation; raw dimensional products depend on chosen units; no slope or forcing inputs'),indent=2))
    print('SELECTED',json.dumps(selected,indent=2))
    print(pd.DataFrame(holdout).groupby(['model','response']).r2.mean().unstack().round(3).to_string())

if __name__=='__main__':
    with threadpool_limits(limits=1):main()
