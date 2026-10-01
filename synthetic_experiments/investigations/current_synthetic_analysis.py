"""Current three-response synthetic synthesis: duration, factorial effects,
omega residuals and a fixed predictor benchmark. Saved routing is never changed.
"""
from pathlib import Path
from collections import defaultdict
from itertools import combinations,product
import json,hashlib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.preprocessing import StandardScaler,SplineTransformer
from sklearn.feature_selection import VarianceThreshold
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score,mean_absolute_error
from threadpoolctl import threadpool_limits
from event_duration_verification import duration,checks
ROOT=Path(__file__).resolve().parents[2]
from analysis_paths import RUN
SOURCE=RUN/'factor_analysis/geometry_predictors/slope_path_arrangement/features.parquet'
OUT=RUN/'current_analysis'
FACTORS=['total_width_m','reference_slope','outer_a_fraction','inner_fraction','length_multiplier','total_length_m','internal_location']
Y=['lag_shift_pct','event_duration_change_pct','peak_excess_change_pct']
W='omega';R='resistance_log';LF='length_fraction';E='width_fraction'
L='parent_equivalent_length';EL='parent_elongation';D='parent_dominant_width_fraction';CV='parent_path_length_cv';A='signed_length_width_association';S='domain_reference_slope'
FULL=[R,LF,E,L,EL,D,CV,A,S]
SETS={'resistance_only':[R],'length_fraction_only':[LF],'exposure_only':[E],
      'omega':[W],'components':[R,LF,E],'compact':[W,L,EL,D],
      'omega_plus_all':[W,L,EL,D,CV,A,S],'components_plus_all':FULL}

def durations(d):
    # Cache signature includes analysis code, boundary code and all saved run stats.
    signature=hashlib.sha256(Path(__file__).read_bytes())
    signature.update((Path(__file__).parent/'event_duration_verification.py').read_bytes())
    signature.update(str(SOURCE.stat()).encode())
    ids=sorted(set(d.parent_run_id)|set(d.child_run_id))
    for rid in ids:
        for filename in ['hydrograph.parquet','metrics.json']:
            p=RUN/'runs'/rid/filename;st=p.stat();signature.update(f'{rid}/{filename}:{st.st_size}:{st.st_mtime_ns}'.encode())
    fp=signature.hexdigest();cache=OUT/'duration_comparisons.parquet';meta=OUT/'duration_provenance.json'
    if cache.exists() and meta.exists() and json.loads(meta.read_text())['fingerprint']==fp:
        return pd.read_parquet(cache)
    tasks=defaultdict(set)
    for row in d.itertuples():
        for f in [.005,.01,.02]:
            threshold=float(f*row.parent_peak_excess_cms)
            tasks[row.parent_run_id].add(threshold);tasks[row.child_run_id].add(threshold)
    endpoints={};qa={}
    base=json.loads((RUN/'manifest.json').read_text())['config']['routing']['baseflow_cms']
    for n,(rid,thresholds) in enumerate(tasks.items(),1):
        folder=RUN/'runs'/rid;mt=json.loads((folder/'metrics.json').read_text())
        if max(abs(mt['volume_error_fraction']),mt['pre_event_relative_error'],mt['final_relative_error'])>1e-4:raise ValueError(rid)
        h=pd.read_parquet(folder/'hydrograph.parquet');dt=mt['dt_seconds'];onset=round(mt['buffer_hours']*3600/dt)*dt
        mask=h.time_seconds.to_numpy()>=onset;t=h.time_seconds.to_numpy()[mask]-onset+dt/2;q=h.outlet_cms.to_numpy()[mask]
        for threshold in thresholds:
            endpoints[rid,threshold]=duration(t,q,base,base,threshold,threshold)
        qa[rid]={k:mt[k] for k in ['n_celerity_capped','n_length_floored','dt_seconds']}
        if n%1000==0:print('Duration runs',n,'/',len(tasks),flush=True)
    rows=[]
    for row in d.itertuples():
        r=dict(configuration_id=row.configuration_id,child_state=row.child_state)
        for f,label in [(.005,'005'),(.01,'010'),(.02,'020')]:
            tol=float(f*row.parent_peak_excess_cms);p=endpoints[row.parent_run_id,tol];c=endpoints[row.child_run_id,tol]
            r[f'duration_change_{label}_pct']=100*(c['duration_seconds']/p['duration_seconds']-1)
            if f==.01:
                for role,values in [('parent',p),('child',c)]:
                    for k,v in values.items():r[f'{role}_event_{k}']=v
        for side in ['parent','child']:
            for k,v in qa[getattr(row,side+'_run_id')].items():r[side+'_'+k]=v
        rows.append(r)
    result=pd.DataFrame(rows);result.to_parquet(cache,index=False)
    meta.write_text(json.dumps(dict(fingerprint=fp,n_runs=len(tasks),n_comparisons=len(rows),thresholds=[.005,.01,.02],
        definition='parent amplitude thresholds shared by pair; first departure and last return; 1h minimum final plateau',checks=checks()),indent=2))
    return result

def decompose(frame,factors,response):
    y=frame[response].to_numpy(float);mean=y.mean();var=np.mean((y-mean)**2)
    if var<=np.finfo(float).eps**2*max(1,np.mean(y*y)):return [],[],dict(variance=var,main_share=0.,pair_share=0.,higher_share=0.)
    terms={};main=[];pairs=[]
    for f in factors:
        term=frame.groupby(f,observed=True)[response].transform('mean').to_numpy()-mean;terms[f]=term
        rest=[x for x in factors if x!=f];conditional=frame.groupby(rest,observed=True)[response].transform('mean').to_numpy()
        main.append(dict(factor=f,main_share=np.mean(term**2)/var,total_share=np.mean((y-conditional)**2)/var))
    residual=y-mean-sum(terms.values())
    for a,b in combinations(factors,2):
        term=frame.groupby([a,b],observed=True)[response].transform('mean').to_numpy()-mean-terms[a]-terms[b]
        residual-=term;pairs.append(dict(factor_a=a,factor_b=b,pair_share=np.mean(term**2)/var))
    summary=dict(variance=var,main_share=sum(x['main_share'] for x in main),pair_share=sum(x['pair_share'] for x in pairs),higher_share=np.mean(residual**2)/var)
    assert np.isclose(summary['main_share']+summary['pair_share']+summary['higher_share'],1,atol=1e-10)
    return main,pairs,summary

def factorial(d):
    toy=pd.DataFrame(product([-1.,0.,1.],repeat=3),columns=['a','b','c']);toy['y']=2*toy.a+3*toy.a*toy.b
    m,p,s=decompose(toy,['a','b','c'],'y');assert np.isclose(m[0]['main_share'],.4) and np.isclose(p[0]['pair_share'],.6)
    main=[];pairs=[];part=[];levels=[];contrasts=[]
    for state,g in d.groupby('child_state'):
        assert len(g)==3**7 and not g.duplicated(FACTORS).any()
        for y in Y:
            tags=dict(child_state=state,response=y);m,p,s=decompose(g,FACTORS,y)
            main.extend(dict(**tags,**r) for r in m);pairs.extend(dict(**tags,**r) for r in p);part.append(dict(**tags,**s))
            for f in FACTORS:
                order=['upstream','central','downstream'] if f=='internal_location' else sorted(g[f].unique())
                for level,vals in g.groupby(f)[y]:
                    levels.append(dict(**tags,factor=f,level=str(level),mean_response=vals.mean(),centered_effect_pp=vals.mean()-g[y].mean()))
                tab=g.pivot(index=[k for k in FACTORS if k!=f],columns=f,values=y);delta=tab[order[-1]]-tab[order[0]]
                contrasts.append(dict(**tags,factor=f,median=delta.median(),p10=delta.quantile(.1),p90=delta.quantile(.9),minimum=delta.min(),maximum=delta.max(),mean=delta.mean()))
    for file,rows in [('main_total_effects',main),('pair_interactions',pairs),('variance_partition',part),('signed_level_effects',levels),('matched_contrasts',contrasts)]:
        pd.DataFrame(rows).to_csv(OUT/f'{file}.csv',index=False)

def omega_audit(d):
    # Finite factorial levels yield repeated omega values. Diagnose variation at
    # effectively identical omega separately within each collapse treatment.
    d=d.copy();d['omega_band']=d.omega.round(10)
    means=d.groupby(['child_state','omega_band']).lag_shift_pct.transform('mean')
    d['within_omega_lag_residual']=d.lag_shift_pct-means
    bands=[];effects=[];matched=[]
    for state,g in d.groupby('child_state'):
        var=g.lag_shift_pct.var(ddof=0)
        for scope,h in [('all',g),('uncapped',g[~g.any_cap])]:
            if len(h)==0:continue
            hm=h.groupby('omega_band').lag_shift_pct.transform('mean')
            bands.append(dict(child_state=state,scope=scope,n=len(h),n_bands=h.omega_band.nunique(),
                within_variance_fraction=np.mean((h.lag_shift_pct-hm)**2)/h.lag_shift_pct.var(ddof=0),
                max_lag_range_pp=h.groupby('omega_band').lag_shift_pct.agg(lambda x:x.max()-x.min()).max(),
                max_omega_rounding_span=h.groupby('omega_band').omega.agg(lambda x:x.max()-x.min()).max()))
        m,p,s=decompose(g,FACTORS,'within_omega_lag_residual')
        effects.extend(dict(child_state=state,**r) for r in m)
        for f in FACTORS:
            context=[k for k in FACTORS if k!=f]
            for scope,h in [('all',g),('uncapped',g[~g.any_cap])]:
                grouped=h.groupby(context,observed=True)
                stats=grouped.agg(n=(f,'size'),omega_min=('omega','min'),omega_max=('omega','max'),lag_min=('lag_shift_pct','min'),lag_max=('lag_shift_pct','max'))
                stats=stats[stats.n==3];unchanged=stats[(stats.omega_max-stats.omega_min)<1e-9]
                matched.append(dict(child_state=state,factor=f,scope=scope,n_complete_contexts=len(stats),n_unchanged_omega_contexts=len(unchanged),
                    median_lag_range_pp=(unchanged.lag_max-unchanged.lag_min).median(),p90_lag_range_pp=(unchanged.lag_max-unchanged.lag_min).quantile(.9)))
    d[['configuration_id','child_state',*FACTORS,W,'lag_shift_pct','omega_band','within_omega_lag_residual','any_cap','any_floor','within_one_dt']].to_csv(OUT/'omega_residuals.csv',index=False)
    pd.DataFrame(bands).to_csv(OUT/'omega_bands.csv',index=False)
    pd.DataFrame(effects).to_csv(OUT/'omega_residual_factor_effects.csv',index=False)
    pd.DataFrame(matched).to_csv(OUT/'omega_matched_controls.csv',index=False)

def estimator(model):
    if model=='linear':return make_pipeline(StandardScaler(),Ridge(alpha=1.))
    if model=='additive_spline':return make_pipeline(VarianceThreshold(),SplineTransformer(n_knots=3,degree=2,knots='quantile',include_bias=False,extrapolation='linear'),StandardScaler(),Ridge(alpha=1.))
    return HistGradientBoostingRegressor(max_iter=80,max_leaf_nodes=7,min_samples_leaf=20,l2_regularization=1.,early_stopping=False,random_state=20260929)

def benchmark(d):
    train=d.split.eq('development');test=~train
    assert not set(d.loc[train,'geometry_group'])&set(d.loc[test,'geometry_group'])
    sets={**SETS,**{'drop_'+c:[x for x in FULL if x!=c] for c in FULL}}
    scores=[];predictions=[]
    def fit_score(model,label,cols,tr,te,validation):
      for y in Y:
        fit=estimator(model).fit(d.loc[tr,cols],d.loc[tr,y]);pred=fit.predict(d.loc[te,cols])
        frame=d.loc[te,['configuration_id','child_state',y]].copy().rename(columns={y:'observed'});frame['predicted']=pred
        for state,g in list(frame.groupby('child_state'))+[('pooled',frame)]:
            scores.append(dict(validation=validation,model=model,feature_set=label,response=y,child_state=state,n=len(g),r2=r2_score(g.observed,g.predicted),mae_pp=mean_absolute_error(g.observed,g.predicted)))
        if validation=='geometry_group_holdout' and label in SETS:
            predictions.append(frame.assign(validation=validation,model=model,feature_set=label,response=y))
    for label,cols in sets.items():
        for model in (['linear','additive_spline','boosted'] if label in SETS else ['boosted']):
            fit_score(model,label,cols,train,test,'geometry_group_holdout')
        print('Predictors',label,flush=True)
    # Unseen collapse treatment is a stricter transfer check, not an independent river test.
    for state in sorted(d.child_state.unique()):
        te=d.child_state.eq(state);tr=~te
        for label in ['omega','compact','components_plus_all']:
            fit_score('boosted',label,SETS[label],tr,te,'leave_collapse_type_out')
    pd.DataFrame(scores).to_csv(OUT/'predictor_scores.csv',index=False)
    pd.concat(predictions,ignore_index=True).to_parquet(OUT/'predictor_predictions.parquet',index=False)
    (OUT/'predictor_features.json').write_text(json.dumps(sets,indent=2))

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    selected=json.loads((RUN/'manifest.json').read_text()).get('config',{}).get('selected_hydraulic_reference')
    if selected:
        # The versioned runner already recomputes the current plateau metric,
        # all response fields, local-kb resistance and imports only geometry/splits.
        d=pd.read_parquet(OUT/'current_comparisons.parquet')
        assert np.allclose(d.depth_alpha,selected['depth_alpha'])
    else:
        d=pd.read_parquet(SOURCE)
        # Historical name meant cumulative-volume t01–t99, NOT plateau duration.
        d=d.rename(columns={'event_duration_change_pct':'volume_quantile_duration_change_pct'})
        d=d.merge(durations(d),on=['configuration_id','child_state'],validate='one_to_one')
    d['event_duration_change_pct']=d.duration_change_010_pct;d[W]=d.resistance_predictor
    assert np.allclose(d.omega,d[LF]*d[E]*d[R],atol=1e-12)
    d['any_cap']=(d.parent_n_celerity_capped>0)|(d.child_n_celerity_capped>0)
    d['any_floor']=(d.parent_n_length_floored>0)|(d.child_n_length_floored>0)
    d['within_one_dt']=d.lag_shift_seconds.abs()<=d.timing_resolution_seconds
    assert len(d)==15309 and np.isfinite(d[Y].to_numpy()).all()
    d.to_parquet(OUT/'current_comparisons.parquet',index=False)
    factorial(d);omega_audit(d)
    with threadpool_limits(limits=1):benchmark(d)
    (OUT/'provenance.json').write_text(json.dumps(dict(responses=Y,omega='length_fraction * exposure * (-log10 Lambda_K)',
        features=SETS,source=str((OUT/'current_comparisons.parquet' if selected else SOURCE).relative_to(ROOT)),selected_hydraulic_reference=selected,code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        validation='same previously explored geometry-group holdout; plus leave-collapse-type-out; no hyperparameter search; not fresh confirmation',
        models=['standardized Ridge alpha1','additive quadratic splines 3 knots + Ridge alpha1','fixed shallow histogram gradient boosting'],
        no_predictors_from_routing_outputs=True,scope='postprocess the explicitly resolved full-grid version; synthetic only; real-case selection deferred'),indent=2))
    print('Saved',OUT,flush=True)
if __name__=='__main__':main()
