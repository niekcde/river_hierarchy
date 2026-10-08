"""Fixed, grouped four-response benchmark; no routing or real-data tuning."""
from pathlib import Path
import sys,json,hashlib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import r2_score,mean_absolute_error
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'RAPID/src'),str(Path(__file__).parent)]
from rapid_tools.comparison_population import COMPARISON_KEY,canonical_comparisons,plotted_comparisons
from rapid_tools.refreshed_analysis import interpolated_peak
from refreshed_predictor_validation import estimator,FULL,SELECTED,CHANGE,CONTEXT

SOURCE=ROOT/'RAPID/validation/real_predictors_20261005'
OUT=ROOT/'RAPID/validation/deduplicated_response_benchmark_20261005'
SYNTH=ROOT/'synthetic_experiments/outputs/hydraulic_alpha065_20260930/full_grid_island'
RESPONSES=['interpolated_lag_shift_pct','interpolated_peak_shift_hours','event_duration_change_pct','peak_excess_change_pct']
LABELS=['Lag change (%)','Timing difference (h)','Event duration change (%)','Peak excess change (%)']
SCALAR={'resistance_only':['resistance_log'],'length_fraction_only':['length_fraction'],
        'exposure_only':['width_fraction'],'omega':['omega'],
        'components':['resistance_log','length_fraction','width_fraction']}
EXTENDED={**SCALAR,'omega_plus_size':['omega','parent_equivalent_length','parent_elongation'],
          'omega_plus_dominance':['omega','parent_dominant_width_fraction'],
          'omega_plus_association':['omega','signed_length_width_association'],
          'omega_plus_slope':['omega','domain_reference_slope'],
          'omega_plus_context':['omega']+CONTEXT,
          'selected_six':SELECTED,'selected_plus_context':SELECTED+CONTEXT,
          'selected_plus_change_context':SELECTED+CHANGE+CONTEXT}
RESIDUAL_FEATURES=['parent_equivalent_length','parent_elongation','parent_dominant_width_fraction',
                   'signed_length_width_association','domain_reference_slope']+CONTEXT


def real_frames():
    original=pd.read_parquet(SOURCE/'analysis.parquet')
    full,aliases=canonical_comparisons(original)
    plot,_=plotted_comparisons(original,exclude_edge_compounds=True,require_shape_metrics=True)
    aliases.to_csv(OUT/'selection_aliases.csv',index=False)
    full.to_parquet(OUT/'canonical_all.parquet',index=False)
    plot['omega']=plot.len_fraction_width_weighted_log_lambda
    plot['resistance_log']=plot.neg_log10_lambda
    plot.to_parquet(OUT/'plotted_population.parquet',index=False)
    geometry=pd.read_parquet(SOURCE/'features_all.parquet')
    extra=list(dict.fromkeys(FULL+CHANGE+CONTEXT))
    extra=[c for c in extra if c not in plot]
    geometry=geometry.drop_duplicates(COMPARISON_KEY)
    merged=plot.merge(geometry[COMPARISON_KEY+extra],on=COMPARISON_KEY,how='left',validate='one_to_one',indicator=True)
    ok=merged._merge.eq('both') & np.isfinite(merged[extra].apply(pd.to_numeric,errors='coerce')).all(axis=1)
    excluded=merged.loc[~ok,COMPARISON_KEY+['mode','state_id','plot_category']].copy()
    excluded['reason']='No valid cached path descriptors (single-source/sink and <=10000 paths required)'
    excluded.to_csv(OUT/'excluded_extended_geometry.csv',index=False)
    common=merged.loc[ok].drop(columns='_merge').copy()
    common.to_parquet(OUT/'common_features.parquet',index=False)
    counts=plot.groupby(['plot_category','hydraulic_domain_type']).size().rename('comparisons').reset_index()
    counts.to_csv(OUT/'population_counts.csv',index=False)
    print('Real population',len(plot),'common geometry',len(common),'excluded',len(excluded),flush=True)
    return plot,common


def scores(obs,pred,null,meta,**tags):
    groups=[('pooled','all',np.ones(len(meta),bool))]
    for column in ['example_id','plot_category']:
        groups += [(column,str(key),meta[column].eq(key).to_numpy()) for key in sorted(meta[column].unique())]
    for scope,group,mask in groups:
        y=np.asarray(obs)[mask];p=np.asarray(pred)[mask];b=np.asarray(null)[mask]
        mae=mean_absolute_error(y,p);base=mean_absolute_error(y,b)
        yield dict(**tags,scope=scope,group=group,n=len(y),r2=r2_score(y,p) if len(y)>1 and np.ptp(y)>0 else np.nan,
                   mae=mae,baseline_mae=base,mae_skill=1-mae/base if base>0 else np.nan)


def validate_real(frame,sets,cohort):
    results=[];predictions=[]
    for label,cols in sets.items():
        assert np.isfinite(frame[cols+RESPONSES]).all().all()
        for response in RESPONSES:
            obs=frame[response].to_numpy(float)
            for kind in ['linear','boosted']:
                pred=np.full(len(frame),np.nan);null=pred.copy()
                for ex in sorted(frame.example_id.unique()):
                    test=frame.example_id.eq(ex).to_numpy();train=~test
                    fit=estimator(kind).fit(frame.loc[train,cols],obs[train])
                    pred[test]=fit.predict(frame.loc[test,cols]);null[test]=obs[train].mean()
                assert np.isfinite(pred).all()
                results.extend(scores(obs,pred,null,frame,cohort=cohort,feature_set=label,model=kind,response=response))
                part=frame[COMPARISON_KEY+['state_id','mode','plot_category']].copy()
                part=part.assign(cohort=cohort,feature_set=label,model=kind,response=response,observed=obs,predicted=pred,baseline_prediction=null,residual=obs-pred)
                predictions.append(part)
        print(cohort,label,'complete',flush=True)
    return pd.DataFrame(results),pd.concat(predictions,ignore_index=True)


def residual_audit(common,predictions):
    base=predictions.query("cohort == 'common_geometry' and feature_set == 'omega' and model == 'boosted'")
    detail=base.merge(common[COMPARISON_KEY+RESIDUAL_FEATURES],on=COMPARISON_KEY,validate='many_to_one')
    detail['absolute_residual']=detail.residual.abs();rows=[]
    for response,g in detail.groupby('response'):
        for scope,key in [('pooled',None),('example_id','example_id'),('plot_category','plot_category')]:
            groups=[('all',g)] if key is None else g.groupby(key)
            for group,part in groups:
                for feature in RESIDUAL_FEATURES:
                    for outcome in ['residual','absolute_residual']:
                        rho=spearmanr(part[feature],part[outcome]).statistic if len(part)>=5 and part[feature].nunique()>1 and part[outcome].nunique()>1 else np.nan
                        rows.append(dict(response=response,scope=scope,group=group,feature=feature,residual_kind=outcome,n=len(part),spearman=rho))
    detail.to_parquet(OUT/'omega_residuals.parquet',index=False)
    pd.DataFrame(rows).to_csv(OUT/'residual_associations.csv',index=False)
    detail.sort_values('absolute_residual',ascending=False).groupby('response',sort=False).head(12).to_csv(OUT/'high_residual_comparisons.csv',index=False)


def synthetic_features():
    s=pd.read_parquet(SYNTH/'current_analysis/compact_predictors/features.parquet')
    run_ids=sorted(set(s.parent_run_id)|set(s.child_run_id)); cache=OUT/'synthetic_peak_cache.parquet';sig=hashlib.sha256(b'guarded-three-point-peak-v1')
    for rid in run_ids:
        for name in ['metrics.json','hydrograph.parquet']:
            st=(SYNTH/'runs'/rid/name).stat();sig.update(f'{rid}/{name}:{st.st_size}:{st.st_mtime_ns}'.encode())
    signature=sig.hexdigest();meta=OUT/'synthetic_peak_cache.json'
    if cache.exists() and meta.exists() and json.loads(meta.read_text())['signature']==signature:
        peaks=pd.read_parquet(cache)
    else:
        rows=[]
        for j,rid in enumerate(run_ids,1):
            folder=SYNTH/'runs'/rid;h=pd.read_parquet(folder/'hydrograph.parquet');m=json.loads((folder/'metrics.json').read_text())
            onset=round(m['buffer_hours']*3600/m['dt_seconds'])*m['dt_seconds']
            peak,status=interpolated_peak(h.time_seconds,h.outlet_cms)
            input_peak=float(h.time_seconds.iloc[int(np.argmax(h.inlet_cms))])
            rows.append(dict(run_id=rid,peak_relative_onset_s=peak-onset,input_peak_relative_onset_s=input_peak-onset,status=status))
            if j%1000==0:print('Synthetic peak extraction',j,'/',len(run_ids),flush=True)
        peaks=pd.DataFrame(rows);peaks.to_parquet(cache,index=False);meta.write_text(json.dumps(dict(signature=signature,runs=len(peaks)),indent=2))
    lookup=peaks.set_index('run_id');p=lookup.reindex(s.parent_run_id).reset_index(drop=True);c=lookup.reindex(s.child_run_id).reset_index(drop=True)
    matching=np.isclose(p.input_peak_relative_onset_s,c.input_peak_relative_onset_s,rtol=0,atol=1e-8)
    valid=p.status.eq('ok') & c.status.eq('ok') & matching
    delta=c.peak_relative_onset_s-p.peak_relative_onset_s;lag=p.peak_relative_onset_s-p.input_peak_relative_onset_s
    valid &= lag.gt(0)
    s['interpolated_peak_shift_hours']=delta/3600
    s['interpolated_lag_shift_pct']=100*delta/lag
    s.loc[~valid,['interpolated_peak_shift_hours','interpolated_lag_shift_pct']]=np.nan
    s['example_id']=s.configuration_id.astype(str)
    key=['parent_run_id','child_run_id','domain_mode','domain_start_m','domain_end_m']
    if s.groupby(key).split.nunique().max()>1:raise ValueError('Synthetic physical duplicate spans training/holdout')
    s=s.drop_duplicates(key).copy()
    s=s.loc[np.isfinite(s[list(dict.fromkeys(SELECTED+CHANGE+CONTEXT+['omega','resistance_log','length_fraction','width_fraction']+RESPONSES))]).all(axis=1)].copy()
    s.to_parquet(OUT/'synthetic_matched_features.parquet',index=False)
    return s,dict(runs=len(peaks),run_status_counts=peaks.status.value_counts().to_dict(),unmatched_input_clocks=int((~matching).sum()),unresolved_pairs=int((~valid).sum()),unique_valid_comparisons=len(s))


def synthetic_transfer(s,real):
    train=s[s.split.eq('development')];test=s[~s.split.eq('development')];results=[];ranges=[]
    for label in ['omega','components','selected_six','selected_plus_change_context']:
        cols=EXTENDED[label]
        for response in RESPONSES:
            fit=estimator('boosted').fit(train[cols],train[response]);null=float(train[response].mean())
            for name,frame in [('synthetic_geometry_holdout',test),('synthetic_to_real',real)]:
                pred=fit.predict(frame[cols]);results.extend(scores(frame[response].to_numpy(),pred,np.full(len(frame),null),frame,validation=name,feature_set=label,model='boosted',response=response))
        print('Matched synthetic transfer',label,flush=True)
    for feature in list(dict.fromkeys(FULL+CHANGE+CONTEXT)):
        lo,hi=train[feature].min(),train[feature].max()
        ranges.append(dict(feature=feature,synthetic_min=lo,synthetic_max=hi,real_min=real[feature].min(),real_max=real[feature].max(),outside_fraction=float((~real[feature].between(lo,hi)).mean())))
    pd.DataFrame(results).to_csv(OUT/'synthetic_transfer_scores.csv',index=False)
    pd.DataFrame(ranges).to_csv(OUT/'feature_ranges.csv',index=False)


def figures(score,predictions):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    names=['omega','components','omega_plus_size','omega_plus_dominance','omega_plus_association','omega_plus_slope','omega_plus_context','selected_six','selected_plus_context','selected_plus_change_context']
    fig,axes=plt.subplots(1,4,figsize=(17,6),constrained_layout=True,sharey=True)
    for ax,response,title in zip(axes,RESPONSES,LABELS):
        for kind,marker in [('linear','o'),('boosted','s')]:
            g=score.query("cohort == 'common_geometry' and scope == 'pooled'")
            g=g[g.response.eq(response)&g.model.eq(kind)].set_index('feature_set').reindex(names)
            ax.scatter(g.mae_skill,range(len(names)),label=kind,marker=marker,s=30)
        ax.axvline(0,c='gray',lw=.7);ax.set(title=title,xlabel='MAE skill vs training-mean baseline');ax.grid(axis='x',alpha=.2)
    axes[0].set_yticks(range(len(names)),names);axes[0].invert_yaxis();axes[-1].legend();fig.savefig(OUT/'response_skill.png',dpi=140);plt.close(fig)
    fig,axes=plt.subplots(1,4,figsize=(16,5),constrained_layout=True)
    for ax,response,title in zip(axes,RESPONSES,LABELS):
        g=predictions.query("cohort == 'common_geometry' and feature_set == 'selected_six' and model == 'boosted'")
        g=g[g.response.eq(response)]
        for label,part in g.groupby('plot_category'):ax.scatter(part.observed,part.predicted,s=10,alpha=.55,label=label)
        limits=[min(g.observed.min(),g.predicted.min()),max(g.observed.max(),g.predicted.max())];ax.plot(limits,limits,'k--',lw=.8)
        ax.set(title=title,xlabel='Observed model response',ylabel='Held-out predicted response')
    axes[-1].legend(fontsize=6);fig.savefig(OUT/'held_out_predictions.png',dpi=140);plt.close(fig)
    associations=pd.read_csv(OUT/'residual_associations.csv');rows=associations.query("scope == 'example_id' and residual_kind == 'residual'")
    heat=rows.groupby(['feature','response']).spearman.median().unstack().reindex(index=RESIDUAL_FEATURES,columns=RESPONSES)
    fig,ax=plt.subplots(figsize=(10,5),constrained_layout=True);im=ax.imshow(heat,cmap='RdBu_r',vmin=-1,vmax=1,aspect='auto')
    ax.set_xticks(range(4),LABELS,rotation=15,ha='right');ax.set_yticks(range(len(heat)),heat.index)
    for i in range(len(heat)):
        for j in range(4):ax.text(j,i,f'{heat.iloc[i,j]:.2f}',ha='center',va='center')
    ax.set_title('Median within-river Spearman association with held-out omega residuals');fig.colorbar(im,ax=ax,label='Spearman ρ');fig.savefig(OUT/'residual_geometry.png',dpi=140);plt.close(fig)


def main():
    OUT.mkdir(parents=True,exist_ok=True);full,common=real_frames()
    scores_all=[];predictions=[]
    for frame,sets,cohort in [(full,SCALAR,'full_plotted'),(common,EXTENDED,'common_geometry')]:
        score,pred=validate_real(frame,sets,cohort);scores_all.append(score);predictions.append(pred)
    score=pd.concat(scores_all,ignore_index=True);pred=pd.concat(predictions,ignore_index=True)
    score.to_csv(OUT/'scores.csv',index=False);pred.to_parquet(OUT/'predictions.parquet',index=False)
    residual_audit(common,pred);figures(score,pred)
    synth,audit=synthetic_features();synthetic_transfer(synth,common)
    (OUT/'provenance.json').write_text(json.dumps(dict(reference='DEDUPLICATED-RESPONSES-2026-10-05',source=str(SOURCE),synthetic_source=str(SYNTH),comparison_key=COMPARISON_KEY,canonical_preference=['indep','compound','island'],exclude_edge_compounds=True,plotted_comparisons=len(full),common_geometry_comparisons=len(common),responses=RESPONSES,feature_sets=EXTENDED,models='Fixed Ridge(alpha=1) and 80-iteration shallow histogram boosting from the prior benchmark. No tuning.',validation='Whole real rivers held out; same cohort for all four responses and feature sets within each cohort. MAE skill=1-MAE/model-specific-held-out-training-mean-MAE. Synthetic development/geometry holdout fixed; training on refined targets without real tuning.',synthetic_peak_audit=audit,limitations='Path features exclude >10000 paths and invalid domains. Residual associations are descriptive, not independent causal effects. Width-path measures retain shared-edge limitations. Shape definitions unchanged (1% duration, shared plateau excess); synthetic peaks now use identical guarded outlet interpolation. Common forcing timing checked per synthetic pair. Grid/forcing/routing dependence remains; synthetic celerity upper cap differs but was inactive. No RAPID rerouting.'),indent=2))
    print('COMPLETE',flush=True)

if __name__=='__main__':main()
