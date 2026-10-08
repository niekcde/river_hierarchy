"""Bounded model/feature selection inside whole-river outer validation."""
from pathlib import Path
import json,sys
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler,PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import r2_score,mean_absolute_error
sys.path.insert(0,str(Path(__file__).parent))
from deduplicated_response_benchmark import EXTENDED,RESPONSES,COMPARISON_KEY,OUT as SOURCE
OUT=SOURCE.parent/'nested_predictor_search_20261005'
FEATURES={key:EXTENDED[key] for key in ['omega','components','selected_six','selected_plus_change_context']}

def model(kind):
    if kind=='linear':return make_pipeline(StandardScaler(),Ridge(alpha=1))
    if kind=='quadratic':return make_pipeline(StandardScaler(),PolynomialFeatures(2,include_bias=False),StandardScaler(),Ridge(alpha=10))
    return HistGradientBoostingRegressor(max_iter=80,max_leaf_nodes=7,min_samples_leaf=20,l2_regularization=1,early_stopping=False,random_state=20261005)

def main():
    OUT.mkdir(parents=True,exist_ok=True);d=pd.read_parquet(SOURCE/'common_features.parquet').reset_index(drop=True)
    assert d.example_id.nunique()==16 and len(d)==1128
    candidates=[(name,kind) for name in FEATURES for kind in ['linear','quadratic','boosted']]
    choices=[];parts=[];audit=[]
    for response in RESPONSES:
        for river in sorted(d.example_id.unique()):
            train=d[d.example_id.ne(river)];test=d[d.example_id.eq(river)];inner=list(GroupKFold(4).split(train,groups=train.example_id));ranking=[]
            for name,kind in candidates:
                errors=[]
                for fold,(a,b) in enumerate(inner):
                    development=train.iloc[a];validation=train.iloc[b]
                    assert not set(development.example_id)&set(validation.example_id) and river not in set(train.example_id)
                    fit=model(kind).fit(development[FEATURES[name]],development[response]);pred=fit.predict(validation[FEATURES[name]])
                    e=pd.DataFrame({'river':validation.example_id.to_numpy(),'error':abs(pred-validation[response].to_numpy())}).groupby('river').error.mean()
                    errors.extend(e.tolist())
                loss=float(np.mean(errors));ranking.append((loss,len(FEATURES[name]),name,kind))
                audit.append(dict(response=response,outer_river=river,feature_set=name,model=kind,inner_equal_river_mae=loss))
            loss,_,name,kind=min(ranking)
            fit=model(kind).fit(train[FEATURES[name]],train[response]);pred=fit.predict(test[FEATURES[name]])
            part=test[COMPARISON_KEY+['plot_category']].copy();part['response']=response;part['observed']=test[response];part['predicted']=pred;part['training_mean']=float(train[response].mean());part['training_median']=float(train[response].median());part['selected_feature_set']=name;part['selected_model']=kind;parts.append(part)
            choices.append(dict(response=response,outer_river=river,feature_set=name,model=kind,inner_equal_river_mae=loss))
        print(response,'complete',flush=True)
    predictions=pd.concat(parts,ignore_index=True);predictions.to_parquet(OUT/'predictions.parquet',index=False)
    pd.DataFrame(choices).to_csv(OUT/'selections.csv',index=False);pd.DataFrame(audit).to_csv(OUT/'inner_scores.csv',index=False);rows=[]
    for response,g in predictions.groupby('response'):
        for scope,key,part in [('pooled','all',g)]+[('river',ex,z) for ex,z in g.groupby('example_id')]:
            y=part.observed.to_numpy();mae=mean_absolute_error(y,part.predicted);zero=float(abs(y).mean());mean=mean_absolute_error(y,part.training_mean);median=mean_absolute_error(y,part.training_median)
            rows.append(dict(response=response,scope=scope,group=key,n=len(part),r2=r2_score(y,part.predicted) if len(y)>1 and np.ptp(y)>0 else np.nan,mae=mae,zero_mae=zero,mean_mae=mean,median_mae=median,skill_vs_zero=1-mae/zero if zero>0 else np.nan,skill_vs_mean=1-mae/mean if mean>0 else np.nan))
    scores=pd.DataFrame(rows);scores.to_csv(OUT/'scores.csv',index=False)
    (OUT/'provenance.json').write_text(json.dumps(dict(reference='NESTED-PREDICTOR-SEARCH-2026-10-05',source=str(SOURCE/'common_features.parquet'),n=len(d),rivers=16,features=FEATURES,candidates=candidates,outer='Leave one river out',inner='4 grouped folds within outer training rivers; selection by equal-river mean absolute error; ties prefer fewer features',models='Standardized Ridge alpha1; standardized degree2 polynomial Ridge alpha10; histogram boosting 80 iterations,7 leaves,l2=1,minleaf20',limitations='Exploratory re-use of these rivers after previous analyses. Nested selection prevents direct outer-fold tuning, but is not an untouched external validation set. Restricted path cohort only. No routing changes.'),indent=2))
    print(scores[scores.scope.eq('pooled')].to_string(index=False),flush=True)

if __name__=='__main__':main()
