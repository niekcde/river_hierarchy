"""Descriptive attribution of existing held-network predictions; no fitting."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
import model_search as engine

ROOT=engine.ROOT
SOURCE=ROOT/'RAPID/outputs/hydrograph_relative_sweep_20261019'
OUT=ROOT/'RAPID/outputs/hydrograph_relative_error_review_20261019'
LABELS={'log1p__B_ratio_route_length_mean':'ln(1+R_muL)', 'domain_size':'muL_Dp / 1 km',
'log1p__B_ratio_route_length_sd':'ln(1+R_sigmaL)', 'log1p__B_ratio_width_mean':'ln(1+R_W)',
'log1p__B_ratio_width_cv':'ln(1+R_CVW)', 'log1p__B_ratio_route_entropy':'ln(1+R_H)',
'B_parent_dominance_mean':'d_Dp', 'B_exposure':'r_in'}

def group(y,b):
    return np.where(y < -b,'advance',np.where(y>b,'delay','unchanged'))

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    data=pd.read_parquet(SOURCE/'input.parquet')
    pred=pd.read_parquet(SOURCE/'predictions.parquet')
    design=json.loads((SOURCE/'design.json').read_text());cols=design['columns']
    pred=pred[(pred.validation=='network')&(pred.target=='centroid_shift_hours')&(pred.floor_minutes==1)&pred.model.isin(['forest','boosting'])].copy()
    metadata=['example','state','edit_type','aliases','centroid_shift_hours','peak_shift_hours','duration80_change_pct','asymmetry_change','peak_excess_change_pct']+cols
    d=pred.merge(data[metadata],on=['example','state'],how='left',validate='many_to_one',indicator=True)
    d[['scenario','model','example','state','_merge']].to_csv(OUT/'join_audit.csv',index=False)
    assert d._merge.eq('both').all()
    np.testing.assert_allclose(d.observed_hours,d.centroid_shift_hours)
    allrows=[];summaries=[];attribution=[]
    for (scenario,model),g in d.groupby(['scenario','model']):
      assert len(g)==len(data if scenario=='all' else data[data.example!='54'])
      for response in ['centroid_shift_hours','peak_shift_hours']:
       for buffer in [1.,2.,5.]:
        z=g.copy();z['response']=response;z['buffer_minutes']=buffer
        z['y']=z[response]*60;z['prediction']=z.predicted_hours*60;z['error']=z.prediction-z.y
        z['group']=group(z.y,buffer);z['predicted_group']=group(z.prediction,buffer)
        z['shift_size']=pd.cut(abs(z.y),[0,1,5,15,60,np.inf],include_lowest=True).astype(str)
        w=engine.weights(z);w=w/w.sum();z['weight']=w
        sse=w*z.error**2;sst=w*(z.y-np.average(z.y,weights=w))**2
        z['sse_share_pct']=100*sse/sse.sum();z['sst_share_pct']=100*sst/sst.sum()
        z['r2_loss']=sse/sst.sum()
        z['relative_points']=100*w*abs(z.error)/np.maximum(abs(z.y),buffer)
        outside=z.y.abs()>buffer
        z['mape_points']=0.
        wm=engine.weights(z[outside]);wm=wm/wm.sum()
        z.loc[outside,'mape_points']=100*wm*abs(z.loc[outside,'error']/z.loc[outside,'y'])
        key=dict(scenario=scenario,model=model,response=response,buffer_minutes=buffer)
        summaries.append(dict(**key,n=len(z),r2=1-z.r2_loss.sum(),relative_error_pct=z.relative_points.sum(),mape_pct=z.mape_points.sum(),mae=np.average(abs(z.error),weights=w)))
        for by in ['example','edit_type','group','shift_size']:
          for name,a in z.groupby(by):
            attribution.append(dict(**key,by=by,category=name,n=len(a),sse_share_pct=a.sse_share_pct.sum(),sst_share_pct=a.sst_share_pct.sum(),r2_loss=a.r2_loss.sum(),relative_points=a.relative_points.sum(),mape_points=a.mape_points.sum()))
        allrows.append(z.drop(columns='_merge'))
    rows=pd.concat(allrows,ignore_index=True);rows.to_parquet(OUT/'error_rows.parquet',index=False)
    pd.DataFrame(summaries).to_csv(OUT/'metrics.csv',index=False)
    pd.DataFrame(attribution).to_csv(OUT/'attribution.csv',index=False)
    # Check exact reproduction of the sweep's scores, including subset reweighting.
    scores=pd.read_csv(SOURCE/'scores.csv')
    for r in summaries:
      s=scores[(scores.scenario==r['scenario'])&(scores.model==r['model'])&(scores.validation=='network')&(scores.target=='centroid_shift_hours')&(scores.evaluated_target==r['response'])&(scores.floor_minutes==1)&(scores.evaluation_floor_minutes==r['buffer_minutes'])&(scores.weighting=='equal_example')]
      a=s[s.subset=='all'].iloc[0];b=s[s.subset=='outside_buffer'].iloc[0]
      np.testing.assert_allclose([r['r2'],r['relative_error_pct'],r['mape_pct']],[a.r2,a.relative_error_pct,b.mape_pct])
    core=rows[(rows.scenario=='all')&(rows.response=='centroid_shift_hours')&(rows.buffer_minutes==1)].copy()
    f=core[core.model=='forest'].copy();b=core[core.model=='boosting']
    paired=f.merge(b[['example','state','error','prediction']],on=['example','state'],suffixes=('','_boosting'),validate='one_to_one')
    paired['same_error_sign']=paired.error*paired.error_boosting>0
    paired['both_error_over_5min']=(paired.error.abs()>5)&(paired.error_boosting.abs()>5)
    paired.to_csv(OUT/'paired_errors.csv',index=False)
    # Descriptive separation and within-example associations; no causal claims or held-out classifier scores.
    stats=[];medians=[];corr=[]
    for c in cols:
      for category in ['unchanged','delay']:
        flag=(f.group==category).astype(int)
        auc=roc_auc_score(flag,f[c],sample_weight=engine.weights(f))
        aucs=[]
        for ex,a in f.groupby('example'):
          y=(a.group==category).astype(int)
          if y.nunique()==2:aucs.append(roc_auc_score(y,a[c]))
        stats.append(dict(feature=c,symbol=LABELS[c],category=category,pooled_auc=auc,within_example_auc_mean=np.mean(aucs),examples=len(aucs)))
      for cat,a in f.groupby('group'):
        medians.append(dict(feature=c,symbol=LABELS[c],group=cat,n=len(a),q25=a[c].quantile(.25),median=a[c].median(),q75=a[c].quantile(.75)))
      for ex,a in f.groupby('example'):
        if a[c].nunique()>1:
          corr.append(dict(feature=c,symbol=LABELS[c],example=ex,signed_error_rho=a[c].corr(a.error,method='spearman'),abs_error_rho=a[c].corr(a.error.abs(),method='spearman')))
    pd.DataFrame(stats).to_csv(OUT/'geometry_separation.csv',index=False)
    contrasts=[]
    for c in cols:
      a=f[f.group!='advance'];w=engine.weights(a);aucs=[]
      for _,z in a.groupby('example'):
        if (z.group=='delay').nunique()==2:aucs.append(roc_auc_score(z.group=='delay',z[c]))
      contrasts.append(dict(feature=c,symbol=LABELS[c],pooled_auc=roc_auc_score(a.group=='delay',a[c],sample_weight=w),within_example_auc_mean=np.mean(aucs),examples=len(aucs)))
    pd.DataFrame(contrasts).to_csv(OUT/'delay_vs_unchanged_geometry.csv',index=False)
    pd.crosstab(f.group,f.predicted_group).to_csv(OUT/'response_group_counts.csv')
    pd.DataFrame(medians).to_csv(OUT/'geometry_distribution.csv',index=False)
    pd.DataFrame(corr).to_csv(OUT/'within_network_associations.csv',index=False)
    # Cases for inspection: distinct reasons, with aliases and both model errors.
    choices=[]
    for reason,col,mask in [('squared_error','sse_share_pct',np.ones(len(paired),bool)),('MAPE','mape_points',paired.group!='unchanged'),('unchanged','relative_points',paired.group=='unchanged'),('delay','mape_points',paired.group=='delay')]:
      a=paired[mask].nlargest(3,col).copy();a['reason']=reason;choices.append(a)
    pd.concat(choices).to_csv(OUT/'review_shortlist.csv',index=False)
    # Lightweight interactive review: use existing out-of-fold predictions only.
    import plotly.express as px
    fig=px.scatter(paired,x='y',y='prediction',color='group',symbol='edit_type',hover_data=['example','state','aliases','error','error_boosting','mape_points','sse_share_pct'],labels={'y':'Modelled centroid shift (min)','prediction':'Held-network forest prediction (min)'})
    lo=min(paired.y.min(),paired.prediction.min());hi=max(paired.y.max(),paired.prediction.max())
    fig.add_shape(type='line',x0=lo,y0=lo,x1=hi,y1=hi,line=dict(color='gray',dash='dash'))
    fig.write_html(OUT/'prediction_review.html',include_plotlyjs=True)
    engine.dump(OUT/'manifest.json',dict(sources={str(p):engine.sha(p) for p in [Path(__file__),SOURCE/'input.parquet',SOURCE/'predictions.parquet',SOURCE/'design.json']},scope='Descriptive analysis only, existing nested held-network predictions; no fits. Peak evaluations reuse centroid predictions.',weighting='Equal-example. MAPE excludes |y| <= buffer and reweights examples within that subset. Additive contributions retain these fixed weights when grouping; group contributions are not subgroup-specific scores.',r2='1 - sum(weight*error^2)/sum(weight*(y-global_weighted_mean)^2); loss contributions sum to 1-R2. SST shares show variance leverage; removing cases changes denominator, so contributions are not removal experiments.',geometry='AUC is univariate descriptive discrimination, not a fitted or validated classifier. Values below .5 indicate reversed direction. Medians are unweighted. Associations do not establish missing physics.',joins='All canonical example/state joins matched; aliases retained, no expansion. Scores reproduced against saved sweep scores.'))
    print(OUT)

if __name__=='__main__':main()
