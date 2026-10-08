"""Exploration helpers: descriptive association and fixed-panel held-out checks.

Feature selection by response is deliberately absent. Correlation pruning for
prediction is fitted separately inside each training fold.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold
from sklearn.metrics import r2_score

COLORS={'02':'#19718b','03':'#d39127','05':'#925d9d','21':'#629448','37':'#b4504d'}
RESPONSES=['relative_shift_pct','peak_excess_change_pct','event_duration_change_pct','shift_h','shift_over_event_T']


def weighted_rank_corr(a,b,w):
    a,b,w=np.asarray(a,float),np.asarray(b,float),np.asarray(w,float);ok=np.isfinite(a)&np.isfinite(b)&np.isfinite(w)
    a,b,w=a[ok],b[ok],w[ok]
    if len(a)<3 or np.ptp(a)==0 or np.ptp(b)==0:return np.nan
    a=pd.Series(a).rank().to_numpy();b=pd.Series(b).rank().to_numpy();a-=np.average(a,weights=w);b-=np.average(b,weights=w)
    return float(np.average(a*b,weights=w)/np.sqrt(np.average(a*a,weights=w)*np.average(b*b,weights=w)))


def associations(data,features,targets=RESPONSES):
    rows=[]
    scopes=[('pooled',data)]+list(data.groupby('example_id'))
    for scope,d in scopes:
        for feature in features:
            valid=d[feature].notna()
            for target in targets:
                for weighting in (['pair','equal_group'] if scope=='pooled' else ['pair']):
                    w=np.ones(len(d)) if weighting=='pair' else 1/d.groupby('group').group.transform('size').to_numpy()
                    rows.append(dict(scope=scope,feature=feature,response=target,weighting=weighting,n_valid=int((valid&d[target].notna()).sum()),spearman=weighted_rank_corr(d[feature],d[target],w)))
    return pd.DataFrame(rows)


def scatter_grid(data,features,target,linthresh=.01):
    fig,axes=plt.subplots(int(np.ceil(len(features)/2)),2,figsize=(12,3.6*int(np.ceil(len(features)/2))),squeeze=False,layout='constrained')
    for ax,col in zip(axes.flat,features):
        for ex,g in data.groupby('example_id'):ax.scatter(g[col],g[target],s=15,alpha=.55,color=COLORS[ex],label=ex)
        ax.axhline(0,color='gray',lw=.6);ax.axvline(0,color='gray',lw=.6);ax.set(xlabel=col.replace('_',' '),ylabel=target.replace('_',' '),yscale='symlog');ax.set_xscale('symlog',linthresh=linthresh)
    for ax in list(axes.flat)[len(features):]:ax.set_visible(False)
    axes[0,0].legend(title='Network',ncol=5,fontsize=8)
    return fig


def redundancy(data,features,threshold=.95):
    rows=[];ranks=[]
    for scope,d in [('pooled',data)]+list(data.groupby('example_id')):
        c=d[features].corr(method='spearman');ii,jj=np.where(np.triu(np.abs(c.to_numpy())>=threshold,1))
        rows.extend(dict(scope=scope,feature_a=c.index[i],feature_b=c.columns[j],spearman=c.iloc[i,j]) for i,j in zip(ii,jj))
        valid=d[features].replace([np.inf,-np.inf],np.nan);valid=valid.dropna(axis=1,how='all');valid=valid.fillna(valid.median());std=valid.std(ddof=0);valid=valid.loc[:,std>1e-12];scaled=(valid-valid.mean())/valid.std(ddof=0)
        s=np.linalg.svd(scaled,compute_uv=False)
        ranks.append(dict(scope=scope,rows=len(d),active_features=len(valid.columns),rank=int(np.linalg.matrix_rank(scaled)),condition=float(s[0]/s[-1]) if len(s) and s[-1]>0 else np.inf))
    return pd.DataFrame(rows),pd.DataFrame(ranks)


def train_filter(train,cols,threshold=.97):
    x=train[cols].replace([np.inf,-np.inf],np.nan);usable=[c for c in cols if x[c].notna().any() and x[c].nunique()>1]
    med=x[usable].median();x=x[usable].fillna(med);cor=x.corr(method='spearman').abs();keep=[];drops=[]
    for c in usable:
        previous=next((k for k in keep if cor.loc[c,k]>=threshold),None)
        if previous:drops.append((c,previous))
        else:keep.append(c)
    assert keep
    return keep,med,drops


def fixed_panels(legacy):
    dimensional=['domain_parent_route_length_mean','domain_delta_route_length_mean','domain_parent_width_mean','domain_delta_width_mean','domain_parent_width_cv','domain_delta_width_cv','domain_parent_dominance_mean','domain_delta_dominance_mean','domain_parent_split_impact_sd','domain_delta_split_impact_sd','domain_parent_expected_splits','domain_delta_expected_splits','domain_parent_route_length_cv','domain_delta_route_length_cv','domain_inlet_exposure_parent','selection_domain_ratio','upstream_distance_over_network','downstream_distance_over_network']
    dimkeys=['route_length_mean','width_mean']
    dimensionless=[c+'_nd' if any(c.endswith(k) for k in dimkeys) else c for c in dimensional]
    delta=['domain_delta_route_length_mean_nd','domain_delta_width_mean_nd','domain_delta_width_cv','domain_delta_dominance_mean','domain_delta_split_impact_sd','domain_delta_expected_splits','domain_delta_route_length_cv','domain_inlet_exposure_parent','selection_domain_ratio','upstream_distance_over_network','downstream_distance_over_network']
    hydraulic=dimensionless+['domain_parent_route_time_mean_nd','domain_delta_route_time_mean_nd','domain_parent_route_time_cv','domain_delta_route_time_cv','domain_parent_total_time_spread_nd','domain_delta_total_time_spread_nd','domain_delta_celerity_mean_nd','exposed_delta_within_dispersion']
    return {'legacy15':legacy,'paired_geometry_dimensional':dimensional,'paired_geometry_dimensionless':dimensionless,'changes_geometry_dimensionless':delta,'paired_plus_Cunge_hydraulics':hydraulic}


def evaluate_fixed(data,panels,target='relative_shift_pct',model='extra_trees'):
    assert data.method.nunique()==1 and data.method.iloc[0]=='cunge','Hydraulic panel is Cunge-specific'
    d=data.sort_values(['example_id','fingerprint']).reset_index(drop=True);records=[];audits=[]
    splits={'grouped':list(GroupKFold(5).split(d,groups=d.group)), 'network':[(np.flatnonzero(d.example_id.ne(ex)),np.flatnonzero(d.example_id.eq(ex))) for ex in sorted(d.example_id.unique())]}
    for validation,folds in splits.items():
        for name,cols in panels.items():
            for fold,(a,b) in enumerate(folds):
                tr,te=d.iloc[a],d.iloc[b];assert set(tr.group).isdisjoint(te.group)
                if name=='legacy15':keep=cols;med=tr[cols].median();drops=[]
                else:keep,med,drops=train_filter(tr,cols)
                fit=ExtraTreesRegressor(n_estimators=96,min_samples_leaf=5,random_state=20261005,n_jobs=1) if model=='extra_trees' else make_pipeline(StandardScaler(),Ridge(alpha=10))
                fit.fit(tr[keep].fillna(med),tr[target]);p=fit.predict(te[keep].fillna(med))
                part=te[['example_id','fingerprint','state_id','group','parent_lag_h']].copy();part['observed']=te[target];part['predicted']=p;part['training_mean']=tr[target].mean();part['validation']=validation;part['panel']=name;part['fold']=fold;part['response']=target;records.append(part)
                audits.append(dict(validation=validation,panel=name,fold=fold,train_rows=len(tr),test_rows=len(te),train_groups=tr.group.nunique(),test_groups=te.group.nunique(),retained=','.join(keep),dropped_correlations=str(drops),retained_count=len(keep)))
    p=pd.concat(records,ignore_index=True);scores=[]
    for (validation,panel),g in p.groupby(['validation','panel']):
        for units in ['target']+(['hours'] if target=='relative_shift_pct' else []):
            for scope,gg in [('pooled',g)]+list(g.groupby('example_id')):
                for weighting in ['pair','equal_group','equal_network'] if scope=='pooled' else ['pair']:
                    w=np.ones(len(gg)) if weighting=='pair' else 1/gg.groupby('group' if weighting=='equal_group' else 'example_id')['observed'].transform('size').to_numpy()
                    scale=gg.parent_lag_h.to_numpy()/100 if units=='hours' else 1.
                    y=gg.observed.to_numpy()*scale;v=gg.predicted.to_numpy()*scale;baseline=gg.training_mean.to_numpy()*scale
                    mae=np.average(abs(y-v),weights=w);zero=np.average(abs(y),weights=w)
                    scores.append(dict(validation=validation,panel=panel,scope=scope,weighting=weighting,units=units,r2=r2_score(y,v,sample_weight=w),mae=mae,zero_mae=zero,training_mean_mae=np.average(abs(y-baseline),weights=w),skill_vs_zero=1-mae/zero if zero else np.nan))
    return p,pd.DataFrame(scores),pd.DataFrame(audits)
