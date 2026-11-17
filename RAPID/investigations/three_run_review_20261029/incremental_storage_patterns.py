"""Separate ΔJ flattening and lower-right departures using saved outputs only."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from incremental_storage import ROOT, SOURCE, OUT, digest

DEST=OUT/'patterns'


def moment(t,q):
    # Exact piecewise-linear first moment, relative to t[0]. No clipping.
    x=t-t[0];h=np.diff(x);a=q[:-1];b=q[1:];m=h*(a+b)/2
    if np.min(q)<-1e-7 or m.sum()<=0:raise ValueError('Invalid excess-flow window')
    return float((x[:-1]*m+h*h*(a/2+(b-a)/3)).sum()/m.sum()),float(m.sum())


def fit(g,target):
    x=g.delta_J_minutes.to_numpy();y=g[target].to_numpy()
    ols=np.polyfit(x,y,1)
    robust=least_squares(lambda p:p[0]*x+p[1]-y,[1,0],loss='huber',f_scale=1).x
    return dict(n=len(g),ols_slope=ols[0],ols_intercept_min=ols[1],huber_slope=robust[0],huber_intercept_min=robust[1])


def main():
    DEST.mkdir(parents=True,exist_ok=True)
    source=OUT/'comparisons.parquet';d=pd.read_parquet(source)
    sources={str(source):digest(source)}
    def wave(ex,state):
        path=SOURCE/'states'/ex/state/'outlet.parquet'
        rec=json.loads((path.parent/'completion.json').read_text());sha=digest(path)
        assert sha==rec['files'][str(path)]
        sources[str(path)]=sha
        f=pd.read_parquet(path)
        return f.time_utc.astype('int64').to_numpy()/1e9,f.q_outlet_cms.to_numpy(float)
    primary=d[d.window.eq('0.5%')]
    flagged=primary[(primary.delta_J_minutes>1)&(primary.centroid_shift_minutes< -1)][['example','state']]
    marked=set(map(tuple,flagged.to_numpy()))
    d['lower_right_primary_gt1min']=[(e,s) in marked for e,s in zip(d.example,d.state)]
    predictions=[];sensitivities=[];decomposition=[]
    for (ex,window),g in d.groupby(['example','window']):
        t,q=wave(ex,'S000_base');r=g.iloc[0]
        a=pd.Timestamp(r.onset_utc).timestamp();b=pd.Timestamp(r.end_utc).timestamp();baseline=r.baseline_cms
        tt=np.r_[a,t[(t>a)&(t<b)],b]
        mu,mass=moment(tt,np.interp(tt,t,q)-baseline)
        assert abs((a+mu)-r.parent_centroid_seconds)<.001
        def translation(delay_minutes):
            delay=delay_minutes*60
            # Shift every knot, preserve the saved parent shape, then measure in the same fixed window.
            knots=t+delay
            z=np.r_[a,knots[(knots>a)&(knots<b)],b]
            shifted=np.interp(z-delay,t,q)-baseline
            cm,vm=moment(z,shifted)
            return (cm-mu)/60
        sensitivity=(translation(.1)-translation(-.1))/.2
        sensitivities.append(dict(example=ex,window=window,window_translation_sensitivity=sensitivity))
        for row in g.itertuples():
            shifted=translation(row.delta_J_minutes)
            predictions.append(dict(example=ex,state=row.state,window=window,translation_only_centroid_minutes=shifted))
            if window=='0.5%' and (ex,row.state) in marked:
                tc,qc=wave(ex,row.state)
                knots=np.unique(np.r_[a,t[(t>a)&(t<b)],tc[(tc>a)&(tc<b)],a+mu,b])
                qp=np.interp(knots,t,q)-baseline;child=np.interp(knots,tc,qc)-baseline
                muc,mc=moment(knots,child)
                assert abs((muc-mu)/60-row.centroid_shift_minutes)<1e-4
                delta=child-qp;x=knots-a;h=np.diff(x)
                dm=h*(delta[:-1]+delta[1:])/2
                centered=(x[:-1]-mu)*dm+h*h*(delta[:-1]/2+(delta[1:]-delta[:-1])/3)
                early=knots[1:]<=a+mu+1e-6
                early_min=centered[early].sum()/mc/60;late_min=centered[~early].sum()/mc/60
                np.testing.assert_allclose(early_min+late_min,row.centroid_shift_minutes,atol=1e-4)
                decomposition.append(dict(example=ex,state=row.state,early_contribution_min=early_min,late_contribution_min=late_min,
                    early_volume_change_m3=dm[early].sum(),late_volume_change_m3=dm[~early].sum()))
    d=d.merge(pd.DataFrame(predictions),on=['example','state','window'],validate='one_to_one')
    d['shape_departure_minutes']=d.centroid_shift_minutes-d.translation_only_centroid_minutes
    fits=[]
    for window,g in d.groupby('window'):
        for name,sub in [('all',g),('abs_delta_J_le20',g[g.delta_J_minutes.abs()<=20]),('without_primary_sign_reversals',g[~g.lower_right_primary_gt1min]),*list(g.groupby('example'))]:
            for target in ['centroid_shift_minutes','translation_only_centroid_minutes']:
                fits.append(dict(window=window,scope=name,target=target,**fit(sub,target)))
    counts=[]
    for window,g in d.groupby('window'):
        for threshold in [0,.25,1]:
            f=g[(g.delta_J_minutes>threshold)&(g.centroid_shift_minutes< -threshold)]
            counts.append(dict(window=window,threshold_min=threshold,n=len(f),networks=f.example.nunique()))
    labels_path=ROOT/'RAPID/outputs/uncapped_event_window_20261020/validation_group_inventory_20261023/records.parquet'
    sources[str(labels_path)]=digest(labels_path)
    labels=pd.read_parquet(labels_path)[['example','state','edit_type','selected_unit_ids','compound_ids','nested']]
    bad=d[d.lower_right_primary_gt1min].merge(labels,on=['example','state'],how='left',validate='many_to_one')
    bad=bad.merge(pd.DataFrame(decomposition),on=['example','state'],validate='many_to_one')
    d.to_parquet(DEST/'pattern_comparisons.parquet',index=False)
    pd.DataFrame(fits).to_csv(DEST/'descriptive_slopes.csv',index=False)
    pd.DataFrame(sensitivities).to_csv(DEST/'window_translation_sensitivity.csv',index=False)
    pd.DataFrame(counts).to_csv(DEST/'sign_reversal_thresholds.csv',index=False)
    bad.to_csv(DEST/'lower_right_cases.csv',index=False)
    scores=[]
    for window,g in d.groupby('window'):
        for predictor in ['delta_J_minutes','translation_only_centroid_minutes']:
            err=g.centroid_shift_minutes-g[predictor]
            scores.append(dict(window=window,predictor=predictor,mae_min=err.abs().mean(),max_abs_min=err.abs().max(),
                r2_no_fit=1/(1) - (err**2).sum()/((g.centroid_shift_minutes-g.centroid_shift_minutes.mean())**2).sum()))
    pd.DataFrame(scores).to_csv(DEST/'translation_check_scores.csv',index=False)
    (DEST/'provenance.json').write_text(json.dumps(dict(source_sha256=sources,script_sha256=digest(__file__),
        scope='Descriptive diagnostic only. No routing or prediction model training; OLS and Huber lines describe the plotted data.',
        threshold='Primary lower-right set: ΔJ > 1 min and 0.5% centroid shift < -1 min. Descriptive threshold, not uncertainty certification; sensitivity at 0 and 0.25 min retained.',
        translation='Shift saved parent by ΔJ, retain its shape, recompute exact piecewise-linear centroid in original window and baseline. Uses a simulated parent waveform; this is a diagnostic, not a hydrography-only predictor.',
        decomposition='Exact change in finite-window centroid: integral (t-parent_centroid)*(child-parent) / child_excess_volume. Early/late split at parent centroid. Describes location of shape changes, not their hydraulic cause.'),indent=2)+'\n')
    print(pd.DataFrame(fits).query("window=='0.5%' and scope=='all'").to_string(index=False))
    print(pd.DataFrame(scores).to_string(index=False))
    print(bad[bad.window.eq('0.5%')][['example','state','edit_type','delta_J_minutes','centroid_shift_minutes','translation_only_centroid_minutes','early_contribution_min','late_contribution_min']].to_string(index=False))

if __name__=='__main__':main()
