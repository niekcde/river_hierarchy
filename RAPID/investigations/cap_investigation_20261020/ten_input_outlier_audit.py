"""Pooled scores and post-hoc exclusions for the ten-input depth-informed panel."""
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from slope_ratio_audit import OUT

SOURCE=OUT/'resistance_inverse_depth_panels_20261023'
DEST=OUT/'ten_input_outlier_audit_20261023'
PANEL='5_replace_width_with_Rh_plus_Gh'


def weights(z,mode):
    if mode=='equal_case':return np.ones(len(z))
    if mode=='equal_network':return 1/z.groupby('example').example.transform('size').to_numpy()
    w=1/z.groupby(['cohort','example']).example.transform('size').to_numpy()
    counts=z.groupby('cohort').example.nunique()
    return w/z.cohort.map(counts).to_numpy()


def score(z,mode):
    w=weights(z,mode);y=z.centroid_shift_hours.to_numpy();e=z.predicted_hours.to_numpy()-y
    sse=np.sum(w*e**2);sst=np.sum(w*(y-np.average(y,weights=w))**2)
    return dict(n=len(z),networks=z.example.nunique(),regular=int(z.cohort.eq('original').sum()),
        segmented=int(z.cohort.eq('segmented').sum()),r2=1-sse/sst,
        mae_minutes=60*np.average(abs(e),weights=w),rmse_minutes=60*np.sqrt(np.average(e**2,weights=w)),
        relative_error_pct=100*np.average(abs(e)/np.maximum(abs(y),1/60),weights=w))


def main():
    DEST.mkdir(exist_ok=True)
    design=json.loads((SOURCE/'design.json').read_text())
    completion=json.loads((SOURCE/'completion.json').read_text())
    assert completion['complete'] and len(design['panels'][PANEL])==10
    p=pd.read_parquet(SOURCE/'predictions.parquet');p=p[p.panel.eq(PANEL)].copy()
    assert len(p)==2185 and not p.duplicated(['example','state']).any()
    p['error_hours']=p.predicted_hours-p.centroid_shift_hours
    p['abs_error_hours']=abs(p.error_hours)
    p['relative_error_pct']=100*p.abs_error_hours/np.maximum(abs(p.centroid_shift_hours),1/60)
    for mode in ['equal_network','equal_case','equal_cohort_network']:
        v=weights(p,mode)*p.error_hours.to_numpy()**2
        p[mode+'_sse_share_pct']=100*v/v.sum()
        p[mode+'_sse_rank']=pd.Series(-v,index=p.index).rank(method='first').astype(int)
    parts=[]
    for cohort,g in p.groupby('cohort'):
        g=g.copy();w=weights(g,'equal_network');v=w*g.error_hours.to_numpy()**2
        g['within_cohort_sse_share_pct']=100*v/v.sum()
        g['within_cohort_sse_rank']=pd.Series(-v,index=g.index).rank(method='first').astype(int)
        rel=w*g.relative_error_pct.to_numpy()
        g['within_cohort_relative_share_pct']=100*rel/rel.sum()
        parts.append(g)
    p=pd.concat(parts)
    p.to_parquet(DEST/'ranked_predictions.parquet',index=False)
    rank=p.sort_values('equal_network_sse_rank')
    rank.head(15).to_csv(DEST/'largest_pooled_sse.csv',index=False)
    p.sort_values('abs_error_hours',ascending=False).head(15).to_csv(DEST/'largest_absolute_errors.csv',index=False)
    p.sort_values('within_cohort_sse_rank').groupby('cohort').head(10).to_csv(DEST/'largest_cohort_sse.csv',index=False)
    p.sort_values('within_cohort_relative_share_pct',ascending=False).groupby('cohort').head(10).to_csv(DEST/'largest_relative_contributions.csv',index=False)
    rows=[];removed=[]
    def evaluate(z,population,label):
        for mode in ['equal_network','equal_case','equal_cohort_network']:
            rows.append(dict(population=population,subset=label,weighting=mode,**score(z,mode)))
    evaluate(p,'all','all')
    evaluate(p[p.example.ne('54')],'all','without54')
    for k in [1,2,3,5,10]:
        omit=rank.head(k)
        evaluate(p.drop(omit.index),'all',f'without_top{k}_pooled_weighted_sse')
        for r in omit.itertuples():removed.append(dict(population='all',subset=f'without_top{k}_pooled_weighted_sse',example=r.example,state=r.state))
    for cohort,g in p.groupby('cohort'):
        evaluate(g,cohort,'all')
        for k in [1,2,3,5]:
            omit=g.nsmallest(k,'within_cohort_sse_rank')
            evaluate(g.drop(omit.index),cohort,f'without_top{k}_cohort_weighted_sse')
            for r in omit.itertuples():removed.append(dict(population=cohort,subset=f'without_top{k}_cohort_weighted_sse',example=r.example,state=r.state))
    omit=p[p.within_cohort_sse_rank.le(3)]
    assert len(omit)==6
    evaluate(p.drop(omit.index),'all','without_top3_each_cohort_six_cases')
    for r in omit.itertuples():removed.append(dict(population='all',subset='without_top3_each_cohort_six_cases',example=r.example,state=r.state))
    s=pd.DataFrame(rows);s.to_csv(DEST/'scores_and_exclusions.csv',index=False)
    pd.DataFrame(removed).to_csv(DEST/'excluded_records.csv',index=False)
    sources=[SOURCE/'predictions.parquet',SOURCE/'design.json',SOURCE/'completion.json',Path(__file__)]
    manifest=dict(sources={str(x):hashlib.sha256(x.read_bytes()).hexdigest() for x in sources},
        panel=PANEL,columns=design['panels'][PANEL],fits=0,
        interpretation='Frozen whole-network-held-out predictions. Models trained with these ten features; settings inherited from nine-input tuning. No ten-input hyperparameter search or final all-data artifact in this experiment.',
        weighting='equal_network: equal total per example across all its states. equal_case: every comparison equal. equal_cohort_network: equal cohorts, then equal examples within cohort. Pooled R2 uses the respective weighted global target mean, not average cohort R2.',
        exclusions='Post-hoc diagnostic only, no retraining. Rank using original population equal-network weighted squared-error contributions; remove top k, then recompute weights, target mean and variance for retained subset. Exclusions must not replace headline performance.',production_modified=False)
    (DEST/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(s[(s.population.eq('all')) | ((s.weighting.eq('equal_network')) & s.subset.isin(['all','without_top2_cohort_weighted_sse','without_top3_cohort_weighted_sse']))].to_string(index=False))
    print(rank.head(10)[['example','state','cohort','centroid_shift_hours','predicted_hours','abs_error_hours','equal_network_sse_share_pct']].to_string(index=False))


if __name__=='__main__':main()
