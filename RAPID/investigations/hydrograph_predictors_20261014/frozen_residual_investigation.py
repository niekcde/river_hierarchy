"""Descriptive geometry screen against frozen seven-input held-network residuals.

No new predictive fits, no outcome-derived predictors and no change to baseline.
Associations guide a shortlist; they are not estimates of added predictive skill.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
import model_search as engine
from drop_exposure_check import COLUMNS

ROOT=engine.ROOT
INPUT=ROOT/'RAPID/outputs/hydrograph_relative_sweep_20261019/input.parquet'
BASE=ROOT/'RAPID/outputs/hydrograph_no_exposure_20261019'
OUT=ROOT/'RAPID/outputs/hydrograph_frozen_residuals_20261019'


def correlation(x,y,w):
    x,y,w=map(lambda a:np.asarray(a,float),(x,y,w))
    x=x-np.average(x,weights=w);y=y-np.average(y,weights=w)
    den=np.sqrt(np.sum(w*x*x)*np.sum(w*y*y))
    return float(np.sum(w*x*y)/den) if den>1e-12 else np.nan


def inventory(d):
    rows=[]
    for metric,(family,symbol,unit,_) in engine.METRICS.items():
      for kind in ['parent','ratio','delta']:
        c=f'B_{kind}_{metric}'
        if c not in d:continue
        if (kind=='ratio' and 'log1p__'+c in COLUMNS) or c in ['B_parent_route_length_mean','B_parent_dominance_mean']:continue
        label=rf'({symbol})_p^D' if kind=='parent' else rf'R_{{{symbol}}}^D' if kind=='ratio' else rf'\Delta({symbol})^D'
        rows.append(dict(feature=c,family=family,symbol=label,scope='outer D',representation=kind,unit='1' if kind=='ratio' else unit))
    # Existing context measurements are screened separately from outer-domain geometry.
    for c,family,symbol,scope in [('B_K_log_a1','R',r'-\log_{10}\lambda_K','outer D'),('U_exposure','X',r'r_{in}^{U}','selected U entrance'),('selected_unit_count','T',r'N_{selected}','selected units'),('max_depth','T',r'depth_{max}','selected units')]:
        rows.append(dict(feature=c,family=family,symbol=symbol,scope=scope,representation='context',unit='1'))
    for metric in ['route_length_mean','width_mean','route_length_sd','expected_splits']:
        c=f'N_parent_{metric}'
        rows.append(dict(feature=c,family='N',symbol=rf'({metric})_p^N',scope='whole network',representation='parent',unit='context'))
    return pd.DataFrame(rows)


def rank_partial(g,c,target,control_magnitude=False):
    """Within-network ranks, controlling baseline ranks and collapse-type indicators."""
    ranks=g.groupby('example')[[c,target]+COLUMNS].rank(pct=True)
    x=ranks[c].to_numpy();y=ranks[target].to_numpy();w=engine.weights(g)
    controls=np.column_stack([np.ones(len(g)),ranks[COLUMNS].to_numpy(),pd.get_dummies(g.edit_type,drop_first=True,dtype=float).to_numpy()])
    if control_magnitude:
        magnitude=g.assign(magnitude=abs(g.observed_minutes)).groupby('example').magnitude.rank(pct=True)
        controls=np.column_stack([controls,magnitude])
    sw=np.sqrt(w)
    bx=np.linalg.lstsq(controls*sw[:,None],x*sw,rcond=None)[0]
    by=np.linalg.lstsq(controls*sw[:,None],y*sw,rcond=None)[0]
    return correlation(x-controls@bx,y-controls@by,w)


def screen(d,cat):
    summary=[];individual=[]
    views={'all':d,'without_extreme':d[~((d.example=='37')&(d.state=='CB001_bubble_3'))],
           'full_compound':d[d.edit_type=='Full compound'],'members':d[d.edit_type=='Compound member']}
    for view,cohort in views.items():
      for row in cat.to_dict('records'):
        c=row['feature'];valid=np.isfinite(cohort[c].to_numpy(float));g=cohort[valid].copy()
        coverage=float(cohort.assign(valid=valid).groupby('example').valid.mean().mean())
        if coverage<.7 or len(g)<30 or g[c].nunique()<3:
            summary.append(dict(**row,view=view,status='insufficient coverage or variation',coverage=coverage,n=len(g)));continue
        basecorr={b:abs(correlation(g[c].rank(),g[b].rank(),engine.weights(g))) for b in COLUMNS}
        closest=max(basecorr,key=lambda b:basecorr[b] if np.isfinite(basecorr[b]) else -1)
        for target in ['error_minutes','absolute_error_minutes','relative_error_pct']:
          correlations=[]
          for ex,z in g.groupby('example'):
            if len(z)<(5 if view=='full_compound' else 12) or z[c].nunique()<3 or z[target].nunique()<3:continue
            r=correlation(z[c].rank(),z[target].rank(),np.ones(len(z)))
            correlations.append(r);individual.append(dict(view=view,feature=c,target=target,example=ex,n=len(z),rho=r))
          cor=np.asarray(correlations);mean=float(np.mean(cor)) if len(cor) else np.nan
          consistency=float(np.mean(np.sign(cor)==np.sign(mean))) if len(cor) else np.nan
          partial=rank_partial(g,c,target)
          summary.append(dict(**row,view=view,target=target,status='screened',coverage=coverage,n=len(g),networks=len(cor),
            pooled_equal_network_rho=correlation(g[c].rank(),g[target].rank(),engine.weights(g)),
            mean_within_network_rho=mean,median_within_network_rho=float(np.median(cor)) if len(cor) else np.nan,
            direction_consistency=consistency,partial_rank_rho=partial,
            partial_controlling_response_magnitude=rank_partial(g,c,target,True),
            strongest_baseline_correlation=basecorr[closest],closest_baseline=closest,
            priority=abs(mean)*consistency if len(cor)>=8 else np.nan))
    return pd.DataFrame(summary),pd.DataFrame(individual)


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    data=pd.read_parquet(INPUT)
    pred=pd.read_parquet(BASE/'predictions.parquet');pred=pred[pred.variant=='seven_inputs'].copy()
    assert len(pred)==len(data)==2170
    drop=[c for c in ['predicted_hours','observed_hours','scenario','variant','error_min','abs_error_min','baseline_error_min'] if c in data]
    d=pred[['example','state','predicted_hours']].merge(data.drop(columns=drop),on=['example','state'],validate='one_to_one',how='outer',indicator=True)
    d[['example','state','_merge']].to_csv(OUT/'join_audit.csv',index=False);assert d._merge.eq('both').all();d=d.drop(columns='_merge')
    d['observed_minutes']=d.centroid_shift_hours*60;d['predicted_minutes']=d.predicted_hours*60
    d['error_minutes']=d.predicted_minutes-d.observed_minutes;d['absolute_error_minutes']=abs(d.error_minutes)
    d['relative_error_pct']=100*abs(d.error_minutes)/np.maximum(abs(d.observed_minutes),1)
    d['response_group']=np.where(d.observed_minutes<-1,'advance',np.where(d.observed_minutes>1,'delay','unchanged'))
    w=engine.weights(d);w=w/w.sum();sse=w*d.error_minutes**2
    d['sse_share_pct']=100*sse/sse.sum();d['relative_points']=100*w*abs(d.error_minutes)/np.maximum(abs(d.observed_minutes),1)
    outside=abs(d.observed_minutes)>1;wm=engine.weights(d[outside]);wm=wm/wm.sum()
    d['mape_points']=0.;d.loc[outside,'mape_points']=100*wm*abs(d.loc[outside,'error_minutes']/d.loc[outside,'observed_minutes'])
    d['ape_pct']=np.where(outside,100*abs(d.error_minutes)/np.maximum(abs(d.observed_minutes),1),np.nan)
    d.to_parquet(OUT/'diagnostic_rows.parquet',index=False)
    attribution=[]
    for by in ['example','edit_type','response_group']:
      for key,g in d.groupby(by):
        ww=engine.weights(g)
        attribution.append(dict(grouping=by,group=key,n=len(g),sse_share_pct=g.sse_share_pct.sum(),mape_points=g.mape_points.sum(),relative_points=g.relative_points.sum(),bias_minutes=np.average(g.error_minutes,weights=ww),mae_minutes=np.average(abs(g.error_minutes),weights=ww)))
    pd.DataFrame(attribution).to_csv(OUT/'error_attribution.csv',index=False)
    rankings=[]
    for label,col,mask in [('absolute','absolute_error_minutes',np.ones(len(d),bool)),('squared_error','sse_share_pct',np.ones(len(d),bool)),('percentage','ape_pct',outside),('MAPE_contribution','mape_points',outside),('delays','absolute_error_minutes',d.response_group=='delay'),('unchanged','absolute_error_minutes',d.response_group=='unchanged')]:
      z=d[mask].nlargest(10,col).copy();z['ranking']=label;rankings.append(z)
    pd.concat(rankings).to_csv(OUT/'worst_cases.csv',index=False)
    cat=inventory(d);cat.to_csv(OUT/'candidate_inventory.csv',index=False)
    scores,individual=screen(d,cat);scores.to_csv(OUT/'candidate_screen.csv',index=False);individual.to_csv(OUT/'within_network_associations.csv',index=False)
    neighbors=[]
    for ex,state in [('41','S130_unit_126'),('46','S142_unit_37'),('55','CB001_bubble_38')]:
        train=d[d.example!=ex];query=d[(d.example==ex)&(d.state==state)]
        pre=engine.Preprocess(COLUMNS,None,None).fit(train,train.centroid_shift_hours)
        distance=cdist(pre.transform(query),pre.transform(train))[0]
        order=np.argsort(distance)[:3];a=train.iloc[order].copy()
        a['distance']=distance[order];a['query_example']=ex;a['query_state']=state;neighbors.append(a)
    pd.concat(neighbors).to_csv(OUT/'nearest_baseline_geometry.csv',index=False)
    engine.dump(OUT/'manifest.json',dict(target='Full centroid shift residual = prediction - modelled response, in minutes.',
      baseline='Frozen seven-input settings, held-network predictions. Settings had been selected on all examples; diagnostic only.',
      sources={str(p):engine.sha(p) for p in [Path(__file__),Path(engine.__file__),INPUT,BASE/'predictions.parquet',BASE/'final_model.json']},
      methodology='Geometry-only allowlist. Raw parent, ratio and difference representations kept explicit. No response-derived variables in candidate inventory. Equal-network weighted coverage >=70%; within-network Spearman correlations, partial rank association controlling seven predictors and collapse type. Priority requires >=8 networks, and equals abs(mean within-network rho)*sign consistency.',
      limitations='Exploratory multiple-variable screening without significance claims, no new predictive fits, no causal inference. Correlation can miss nonlinear effects. Partial control removes only linear rank associations. Response-magnitude-adjusted association is diagnostic only and never a predictor. Parent-only descriptors repeated across edits can be confounded with domain identity. Whole-network constants cannot establish within-network explanatory power. U exposure is only an entrance descriptor; U may itself be open.',
      exclusion='37/CB001_bubble_3 excluded only in sensitivity screening, never removed from frozen training data.'))
    print('Screened',len(cat),'candidate representations; saved',OUT,flush=True)

if __name__=='__main__':main()
