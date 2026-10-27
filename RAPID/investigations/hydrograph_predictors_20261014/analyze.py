"""Predeclared small panels, six responses, leave-one-example-out validation."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from build import OUT, ROOT, RESPONSES, digest

TARGETS=list(RESPONSES)
# Fixed before inspecting associations or model scores. No target-based screening.
NETWORK_PANELS={
 'length':['network_ratio_route_length_mean','network_parent_route_length_mean','network_delta_route_length_cv'],
 'length_width':['network_ratio_route_length_mean','network_parent_route_length_mean','network_delta_route_length_cv','network_ratio_width_mean','network_delta_width_cv'],
 'length_width_topology':['network_ratio_route_length_mean','network_parent_route_length_mean','network_delta_route_length_cv','network_ratio_width_mean','network_delta_width_cv','network_delta_expected_splits'],
}
LOCAL_PANELS={
 'length_exposure':['exposed_length_change','domain_parent_length_over_network','exposed_route_sd_change','domain_entry_exposure'],
 'length_width_exposure':['exposed_length_change','domain_parent_length_over_network','exposed_route_sd_change','domain_entry_exposure','domain_ratio_width_mean','domain_parent_dominance_mean'],
 'length_width_topology_exposure':['exposed_length_change','domain_parent_length_over_network','exposed_route_sd_change','domain_entry_exposure','domain_ratio_width_mean','domain_parent_dominance_mean','domain_delta_expected_splits'],
}


def weights(d):
    w=1/d.groupby('example').example.transform('size').to_numpy(float)
    return w/w.mean()


def metrics(y,p,w):
    mu=np.average(y,weights=w);den=np.sum(w*(y-mu)**2)
    return dict(mae=float(np.average(abs(y-p),weights=w)),rmse=float(np.sqrt(np.average((y-p)**2,weights=w))),
        r2=float(1-np.sum(w*(y-p)**2)/den) if den>0 else np.nan)


def fit_predict(train,test,cols):
    # All preprocessing is fitted inside each training fold; equal-example weights.
    x=train[cols].replace([np.inf,-np.inf],np.nan);z=test[cols].replace([np.inf,-np.inf],np.nan)
    med=x.median();keep=med.notna();x=x.loc[:,keep].fillna(med[keep]);z=z.loc[:,keep].fillna(med[keep])
    keep=x.std().gt(1e-12);x=x.loc[:,keep];z=z.loc[:,keep]
    w=weights(train);scaler=StandardScaler().fit(x,sample_weight=w)
    model=Ridge(alpha=10.).fit(scaler.transform(x),train[TARGETS],sample_weight=w)
    return model.predict(scaler.transform(z)),list(x.columns),model.coef_


def main():
    d=pd.read_parquet(OUT/'linked_responses.parquet')
    sensitivity=pd.read_parquet(ROOT/'RAPID/outputs/hydrograph_response/sensitivity_responses.parquet')
    assert not sensitivity.duplicated(['example','state','window']).any()
    primary=sensitivity[sensitivity.window.eq('0.5%')].set_index(['example','state'])
    np.testing.assert_allclose(primary[TARGETS].sort_index(),d.set_index(['example','state'])[TARGETS].sort_index())
    # Label sensitivity: absolute differences, not unstable percentages of near-zero shifts.
    stability=[]
    for win in ['1%','2%']:
        s=sensitivity[sensitivity.window.eq(win)].set_index(['example','state']).reindex(primary.index)
        assert s[TARGETS].notna().all().all()
        for target in TARGETS:
            delta=s[target]-primary[target]
            for scope,mask in [('all',np.ones(len(s),bool)),('54',s.index.get_level_values('example')=='54'),('without54',s.index.get_level_values('example')!='54')]:
                dd=delta[mask]
                stability.append(dict(window=win,response=target,scope=scope,n=len(dd),median_absolute_change=dd.abs().median(),max_absolute_change=dd.abs().max(),
                    sign_reversals=int(((s[target][mask]*primary[target][mask]<0)&(dd.abs()>1e-10)).sum())))
    pd.DataFrame(stability).to_csv(OUT/'window_sensitivity.csv',index=False)
    # Explicitly exploratory associations, pooled and separately within each example.
    assoc=[];redundancy=[]
    pools={'network':(d[d.comparison.eq('partial collapse')],NETWORK_PANELS),
           'domain':(d[d.comparison.eq('partial collapse')&d.status.eq('validated')],LOCAL_PANELS)}
    for scope,(data,panels) in pools.items():
        cols=list(dict.fromkeys(c for v in panels.values() for c in v))
        # These alternatives are reviewed, not added automatically to fitted panels.
        cols+= [scope+'_parent_route_entropy',scope+'_parent_effective_branches_mean',scope+'_parent_split_impact_mean']
        cols=list(dict.fromkeys(c for c in cols if c in data))
        corr=data[cols].corr(method='spearman')
        for i,a in enumerate(cols):
            for b in cols[i+1:]:
                if abs(corr.loc[a,b])>=.85:redundancy.append(dict(scope=scope,a=a,b=b,spearman=corr.loc[a,b]))
        for feature in cols:
            for target in TARGETS:
                for ex,g in [('pooled',data),*list(data.groupby('example'))]:
                    a=g[[feature,target]].dropna()
                    r=spearmanr(a[feature],a[target]).statistic if len(a)>2 and a.nunique().min()>1 else np.nan
                    assoc.append(dict(scope=scope,example=ex,feature=feature,response=target,n=len(a),spearman=r))
    pd.DataFrame(assoc).to_csv(OUT/'individual_associations.csv',index=False)
    pd.DataFrame(redundancy,columns=['scope','a','b','spearman']).to_csv(OUT/'redundancy.csv',index=False)
    predictions=[];folds=[];coefficients=[]
    # Partial collapses are the main cohort; centerlines are an explicit expanded-cohort check.
    for cohort in ['partial','including_centerlines']:
      for scope,panels in [('network',NETWORK_PANELS),('network_on_domain',NETWORK_PANELS),('domain',LOCAL_PANELS)]:
        source=d if cohort=='including_centerlines' else d[d.comparison.eq('partial collapse')]
        if scope in ['domain','network_on_domain']:source=source[source.status.eq('validated')]
        for include54 in [True,False]:
          data=source if include54 else source[source.example.ne('54')]
          run=f'{cohort}/{scope}/'+('all' if include54 else 'without54')
          for held in sorted(data.example.unique()):
            train=data[data.example.ne(held)];test=data[data.example.eq(held)]
            assert not set(train.example)&set(test.example)
            means=np.average(train[TARGETS],axis=0,weights=weights(train))
            preds={'zero':np.zeros((len(test),len(TARGETS))), 'training_mean':np.tile(means,(len(test),1))}
            for panel,cols in panels.items():
                pred,kept,coef=fit_predict(train,test,cols);preds[panel]=pred
                folds.append(dict(run=run,held_example=held,panel=panel,train_rows=len(train),test_rows=len(test),features=' | '.join(kept)))
                for ti,target in enumerate(TARGETS):
                    for fi,feature in enumerate(kept):coefficients.append(dict(run=run,held_example=held,panel=panel,response=target,feature=feature,standardized_x_coefficient=coef[ti,fi]))
            for panel,pred in preds.items():
                for j,target in enumerate(TARGETS):
                    block=test[['example','state']].copy();block['run']=run;block['panel']=panel;block['response']=target
                    block['observed']=test[target].to_numpy();block['predicted']=pred[:,j];predictions.append(block)
          print(run,'validated',flush=True)
    pred=pd.concat(predictions,ignore_index=True);pred.to_parquet(OUT/'held_example_predictions.parquet',index=False)
    pd.DataFrame(folds).to_csv(OUT/'fold_audit.csv',index=False);pd.DataFrame(coefficients).to_csv(OUT/'fold_coefficients.csv',index=False)
    scores=[]
    for (run,panel,target),g in pred.groupby(['run','panel','response']):
        for weighting,w in [('pair',np.ones(len(g))),('equal_example',weights(g))]:
            scores.append(dict(run=run,panel=panel,response=target,weighting=weighting,n=len(g),examples=g.example.nunique(),**metrics(g.observed,g.predicted,w)))
        for ex,gg in g.groupby('example'):
            scores.append(dict(run=run,panel=panel,response=target,weighting='example_'+ex,n=len(gg),examples=1,**metrics(gg.observed,gg.predicted,np.ones(len(gg)))))
    pd.DataFrame(scores).to_csv(OUT/'validation_scores.csv',index=False)
    # Same out-of-example predictions evaluated against alternate-window labels; no retuning.
    sensitivity_scores=[]
    for win in ['1%','2%']:
        alt=sensitivity[sensitivity.window.eq(win)]
        for target,g in pred.groupby('response'):
            v=g.merge(alt[['example','state',target]],on=['example','state'],validate='many_to_one')
            for (run,panel),gg in v.groupby(['run','panel']):
                sensitivity_scores.append(dict(window=win,run=run,panel=panel,response=target,**metrics(gg[target],gg.predicted,weights(gg))))
    pd.DataFrame(sensitivity_scores).to_csv(OUT/'sensitivity_validation_scores.csv',index=False)
    config=dict(targets=TARGETS,primary=['peak_shift_hours','peak_excess_change_pct'],network_panels=NETWORK_PANELS,domain_panels=LOCAL_PANELS,
        estimator='Ridge alpha=10; fixed panels, no tuning or target-based feature selection',validation='Leave one entire example out; rerun after excluding 54, including training exclusion',
        preprocessing='Training-only median imputation, constant removal, equal-example weighted standardization and fitting',
        missingness='No-split conditional means remain NaN in source; training medians used only in model pipeline',
        units={k:v[1] for k,v in RESPONSES.items()},
        limitations=['Exploratory panel comparison, not selection-adjusted final performance','Local validated domains may be containing domains; see domain_type and audit','Geometry-only hours prediction is conditional on this event/model cohort; geometric lengths alone do not supply a velocity scale','Child crest diagnostics and all six responses excluded from predictor columns','No causal claims from correlated predictors'],
        input_sha256={str(OUT/'linked_responses.parquet'):digest(OUT/'linked_responses.parquet'),str(Path(__file__)):digest(__file__)})
    (OUT/'validation_design.json').write_text(json.dumps(config,indent=2)+'\n')


if __name__=='__main__':main()
