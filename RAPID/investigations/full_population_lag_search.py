"""Expanded lag-only search on the all distinct parent-child pairs, nested by river."""
from pathlib import Path
import sys,json,hashlib,inspect
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler,PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingRegressor,ExtraTreesRegressor
from sklearn.metrics import r2_score,mean_absolute_error
sys.path.insert(0,str(Path(__file__).parent))
from deduplicated_response_benchmark import EXTENDED,COMPARISON_KEY,FULL,CHANGE,CONTEXT
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'RAPID/validation/all_pairs_lag_20261005'
RESPONSES=['interpolated_lag_shift_pct','interpolated_peak_shift_hours']
FEATURES={key:EXTENDED[key] for key in ['omega','components','selected_six','selected_plus_context','selected_plus_change_context']}
FEATURES['omega_size_slope']=['omega','parent_equivalent_length','parent_elongation','domain_reference_slope']
FEATURES['all_geometry']=list(dict.fromkeys(FULL+CHANGE+CONTEXT))
LOG_COLUMNS=['parent_equivalent_length','parent_elongation','parent_dominant_width_fraction','domain_reference_slope']
FEATURES['log_geometry']=[('log10_1p_'+c if c in CONTEXT else 'log10_'+c if c in LOG_COLUMNS else c) for c in FEATURES['all_geometry']]
for key in ['dominance','association','slope']:FEATURES['product_'+key]=['omega_times_'+key]
KINDS=['ridge_0.1','ridge_10','quadratic_1','quadratic_100','boosted_7','boosted_15']
CANDIDATES=[(name,kind) for name in FEATURES for kind in (['ridge_10','quadratic_100','boosted_7'] if name.startswith('product_') else KINDS)]
CANDIDATES += [(name,'extra_trees') for name in ['selected_six','selected_plus_change_context','all_geometry','log_geometry']]

def enrich(d):
    d=d.copy()
    for c in LOG_COLUMNS:
        if (d[c]<=0).any():raise ValueError(f'Cannot take geometry log: {c}')
        d['log10_'+c]=np.log10(d[c])
    for c in CONTEXT:
        if (d[c]<0).any():raise ValueError(f'Negative surrounding length: {c}')
        d['log10_1p_'+c]=np.log10(1+d[c])
    for key,c in [('dominance','parent_dominant_width_fraction'),('association','signed_length_width_association'),('slope','domain_reference_slope')]:d['omega_times_'+key]=d.omega*d[c]
    assert np.isfinite(d[list(dict.fromkeys(sum(FEATURES.values(),[])))]).all().all()
    return d


def model(kind):
    if kind.startswith('ridge_'):return make_pipeline(StandardScaler(),Ridge(alpha=float(kind.split('_')[1])))
    if kind.startswith('quadratic_'):return make_pipeline(StandardScaler(),PolynomialFeatures(2,include_bias=False),StandardScaler(),Ridge(alpha=float(kind.split('_')[1])))
    if kind=='extra_trees':return ExtraTreesRegressor(n_estimators=96,min_samples_leaf=5,max_features=1.,n_jobs=1,random_state=20261005)
    leaves=int(kind.split('_')[1]);return HistGradientBoostingRegressor(max_iter=80 if leaves==7 else 160,max_leaf_nodes=leaves,min_samples_leaf=20,l2_regularization=1,early_stopping=False,random_state=20261005)


def score_frame(p):
    rows=[]
    for (response,method),g in p.groupby(['response','method']):
        groups=[('pooled','all',g)]+[('river',ex,z) for ex,z in g.groupby('example_id')]+[('category',cat,z) for cat,z in g.groupby('plot_category')]
        for scope,key,z in groups:
            y=z.observed.to_numpy();mae=mean_absolute_error(y,z.predicted);zero=abs(y).mean();baseline=mean_absolute_error(y,z.training_mean)
            rows.append(dict(response=response,method=method,scope=scope,group=key,n=len(z),r2=r2_score(y,z.predicted) if len(y)>1 and np.ptp(y)>0 else np.nan,mae=mae,zero_mae=zero,mean_mae=baseline,skill_vs_zero=1-mae/zero if zero>0 else np.nan,skill_vs_mean=1-mae/baseline if baseline>0 else np.nan))
    return pd.DataFrame(rows)


def main():
    signature=hashlib.sha256((OUT/'features.parquet').read_bytes()+json.dumps([FEATURES,CANDIDATES],sort_keys=True).encode()+inspect.getsource(model).encode()).hexdigest()
    d=enrich(pd.read_parquet(OUT/'features.parquet'));assert len(d)==2214 and d.example_id.nunique()==16
    preds=[];choices=[];audit=[]
    (OUT/'search_plan.json').write_text(json.dumps(dict(reference='ALL-PAIRS-LAG-2026-10-05',rows=len(d),rivers=16,responses=RESPONSES,features=FEATURES,candidates=CANDIDATES,outer='leave one complete river out',inner='4 grouped folds; equal-river MAE; scaling fitted only on training rows',scope='All 2214 canonical parent-child pairs. No edge-compound or path-count exclusions. No shape-model search.',limits='Exploratory study-river reuse; not untouched external validation. Fixed candidate outer scores are descriptive; only nested selection accounts for this run\u2019s model/feature selection.'),indent=2))
    for response in RESPONSES:
        for river in sorted(d.example_id.unique()):
            cache=OUT/f'fold_{response}_{river}.parquet';choicefile=OUT/f'choice_{response}_{river}.json';auditfile=OUT/f'inner_{response}_{river}.csv'
            if cache.exists() and choicefile.exists() and auditfile.exists() and json.loads(choicefile.read_text()).get('input_signature')==signature:
                preds.append(pd.read_parquet(cache));choices.append(json.loads(choicefile.read_text()));audit.extend(pd.read_csv(auditfile).to_dict('records'));continue
            train=d[d.example_id.ne(river)];test=d[d.example_id.eq(river)];splits=list(GroupKFold(4).split(train,groups=train.example_id));rank=[];innerrows=[];outer=[]
            for name,kind in CANDIDATES:
                errors=[];cols=FEATURES[name]
                for a,b in splits:
                    t=train.iloc[a];v=train.iloc[b];assert not set(t.example_id)&set(v.example_id) and river not in set(train.example_id)
                    fit=model(kind).fit(t[cols],t[response]);prediction=fit.predict(v[cols])
                    errors.extend(pd.DataFrame({'river':v.example_id.to_numpy(),'error':abs(prediction-v[response].to_numpy())}).groupby('river').error.mean().tolist())
                loss=float(np.mean(errors));rank.append((loss,len(cols),name,kind));innerrows.append(dict(response=response,outer_river=river,feature_set=name,model=kind,inner_equal_river_mae=loss))
                fit=model(kind).fit(train[cols],train[response]);part=test[COMPARISON_KEY+['plot_category']].copy();part['response']=response;part['method']=name+'/'+kind;part['observed']=test[response].to_numpy();part['predicted']=fit.predict(test[cols]);part['training_mean']=float(train[response].mean());outer.append(part)
            loss,_,name,kind=min(rank);chosen=dict(input_signature=signature,response=response,outer_river=river,feature_set=name,model=kind,inner_equal_river_mae=loss)
            nested=next(z for z in outer if z.method.iloc[0]==name+'/'+kind).copy();nested['method']='nested_selected';outer.append(nested)
            part=pd.concat(outer,ignore_index=True);assert np.isfinite(part[['observed','predicted']]).all().all()
            part.to_parquet(cache,index=False);choicefile.write_text(json.dumps(chosen));pd.DataFrame(innerrows).to_csv(auditfile,index=False)
            preds.append(part);choices.append(chosen);audit.extend(innerrows);print(response,river,name,kind,'complete',flush=True)
    p=pd.concat(preds,ignore_index=True);p.to_parquet(OUT/'predictions.parquet',index=False);pd.DataFrame(choices).to_csv(OUT/'selections.csv',index=False);pd.DataFrame(audit).to_csv(OUT/'inner_scores.csv',index=False)
    scores=score_frame(p);scores.to_csv(OUT/'scores.csv',index=False)
    print(scores.query("method=='nested_selected' and scope=='pooled'").to_string(index=False),flush=True)

if __name__=='__main__':main()
