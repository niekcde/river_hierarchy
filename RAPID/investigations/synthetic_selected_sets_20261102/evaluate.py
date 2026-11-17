"""Real-only fits of four frozen sets, leave one entire river out."""
import os
for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
os.environ.setdefault('MPLCONFIGDIR','/tmp/river-similarity-mpl')
from pathlib import Path
import json,hashlib,time
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'RAPID/outputs/synthetic_selected_sets_20261102'
common=['B_delta_route_length_mean','parent_slope_slow']
PANELS={
 'centroid_with_G':common+['path_parent_equivalent_width','path_delta_topologic_complexity_score','delta_G_manning'],
 'centroid_without_G':common+['path_parent_equivalent_width','path_delta_topologic_complexity_score','B_delta_area_proxy'],
 'peak_with_G':common+['B_child_width_p90','B_parent_bifurcations','delta_G_manning','B_delta_route_length_cv','path_parent_slope_mean'],
 'peak_without_G':common+['B_child_width_p90','B_parent_bifurcations','path_parent_slope_mean','path_delta_slope_mean','B_delta_area_proxy','B_ratio_route_length_mean']}
RF=dict(n_estimators=200,max_depth=12,min_samples_leaf=3,max_features=1.0,criterion='squared_error',random_state=20261102,n_jobs=2)
def weights(d,mode):
    if mode=='equal_cohort_network':
        w=1/d.groupby(['cohort','example']).example.transform('size').to_numpy(float)
        counts=d.groupby('cohort').example.nunique()
        return w/d.cohort.map(counts).to_numpy()
    if mode=='equal_network':return 1/d.groupby('example').example.transform('size').to_numpy(float)
    return np.ones(len(d))
def score(z,mode):
    y=z.observed.to_numpy();p=z.predicted.to_numpy();w=weights(z,mode);den=np.sum(w*(y-np.average(y,weights=w))**2)
    return dict(n=len(z),networks=z.example.nunique(),r2=float(1-np.sum(w*(y-p)**2)/den) if den else np.nan,
                mae_minutes=float(np.average(abs(y-p),weights=w)),rmse_minutes=float(np.sqrt(np.average((y-p)**2,weights=w))))
def polynomial(x):
    return np.column_stack([x]+[x[:,i]*x[:,j] for i in range(x.shape[1]) for j in range(i,x.shape[1])])
def main():
    start=time.monotonic();d=pd.read_parquet(OUT/'input.parquet')
    assert len(d)==2185 and d.example.nunique()==15
    assert np.isfinite(d[list(set(sum(PANELS.values(),[])))]).all().all()
    design=dict(reference='REAL-SELECTED-SETS-PREDICTION-2026-11-02',panels=PANELS,
        validation='15 leave-one-entire-river-out fits for each of four frozen sets, each fitted exclusively on real cases; no synthetic coefficient transfer.',
        peak_with_G_choice='Seven-input set selected in two synthetic folds, fixed before real performance inspection.',
        weighting='Equal total training weight per river; equal-network, equal-cohort-network and unweighted metrics reported; cohorts separate.',
        quadratic='Main effects, squares and all pair interactions; expanded columns training-standardised; ridge alpha=1e-4*n, same regularisation rule as synthetic.',
        forest=RF,forest_transform='Training-median-scale signed log1p on each input. Monotonic, preserves tree split ordering; avoids float32 overflow/constant-feature precision collapse for astronomical path-width sums. No target transform.',
        limitations='Fixed sets and fixed model settings, not newly optimised RF. Previously investigated rivers, not independent external validation. Fifteen full-network segmented controls use full physical domains including connectors.',
        input_sha256=hashlib.sha256((OUT/'input.parquet').read_bytes()).hexdigest())
    (OUT/'evaluation_design.json').write_text(json.dumps(design,indent=2))
    predictions=[];folds=[];ranges=[]
    for panel,cols in PANELS.items():
        target='centroid_shift_minutes' if panel.startswith('centroid') else 'peak_shift_minutes'
        for ex in sorted(d.example.unique()):
            tr=d[d.example!=ex];te=d[d.example==ex]
            assert set(tr.example).isdisjoint(te.example) and tr.example.nunique()==14
            x=tr[cols].to_numpy();v=te[cols].to_numpy();y=tr[target].to_numpy()
            w=weights(tr,'equal_network');w*=len(tr)/w.sum()
            for j,c in enumerate(cols):ranges.append(dict(panel=panel,held=ex,variable=c,train_min=x[:,j].min(),train_max=x[:,j].max(),test_min=v[:,j].min(),test_max=v[:,j].max(),outside=int(((v[:,j]<x[:,j].min())|(v[:,j]>x[:,j].max())).sum())))
            # Fixed quadratic regression, with no clipping of held-out extrapolation.
            scale=np.maximum(np.max(abs(x),axis=0),1.)
            a=polynomial(x/scale);b=polynomial(v/scale)
            means=np.average(a,axis=0,weights=w)
            std=np.sqrt(np.average((a-means)**2,axis=0,weights=w));good=std>1e-12
            model=Ridge(alpha=1e-4*len(tr)).fit((a[:,good]-means[good])/std[good],y,sample_weight=w)
            pq=model.predict((b[:,good]-means[good])/std[good])
            # Fixed forest comparator; monotonic input compression only.
            scale=np.maximum(np.median(abs(x),axis=0),1e-12)
            tx=np.sign(x)*np.log1p(abs(x)/scale);tv=np.sign(v)*np.log1p(abs(v)/scale)
            forest=RandomForestRegressor(**RF).fit(tx,y,sample_weight=w)
            pf=forest.predict(tv)
            assert np.isfinite(pq).all() and np.isfinite(pf).all()
            for name,pred in [('quadratic',pq),('RF_fixed',pf),('training_mean',np.full(len(te),np.average(y,weights=w)))]:
                z=te[['example','state','cohort','edit_type']].copy();z['panel']=panel;z['model']=name;z['target']=target;z['observed']=te[target].to_numpy();z['predicted']=pred
                predictions.append(z);folds.append(dict(panel=panel,model=name,held=ex,**score(z,'equal_network')))
        print(panel,'done;',round(time.monotonic()-start,1),'s',flush=True)
    pred=pd.concat(predictions,ignore_index=True)
    assert len(pred)==2185*4*3 and not pred.duplicated(['panel','model','example','state']).any()
    pred.to_parquet(OUT/'real_only_predictions.parquet',index=False)
    pd.DataFrame(folds).to_csv(OUT/'real_only_fold_scores.csv',index=False)
    pd.DataFrame(ranges).to_csv(OUT/'held_river_feature_ranges.csv',index=False)
    results=[]
    for (panel,model),z in pred.groupby(['panel','model']):
        for cohort,h in [('all',z)]+list(z.groupby('cohort')):
            for weighting in ['equal_network','equal_cohort_network','unweighted']:
                results.append(dict(panel=panel,model=model,cohort=cohort,weighting=weighting,**score(h,weighting)))
    result=pd.DataFrame(results);result.to_csv(OUT/'real_only_scores.csv',index=False)
    error=pred.copy();error['absolute_error_minutes']=abs(error.predicted-error.observed)
    error.sort_values('absolute_error_minutes',ascending=False).groupby(['panel','model']).head(10).to_csv(OUT/'largest_real_errors.csv',index=False)
    fig,axes=plt.subplots(2,2,figsize=(10,9),constrained_layout=True)
    for ax,panel in zip(axes.flat,PANELS):
        z=pred[(pred.panel==panel)&(pred.model=='RF_fixed')]
        for cohort,h in z.groupby('cohort'):ax.scatter(h.observed,h.predicted,s=14,alpha=.5,label=cohort)
        lim=[min(z.observed.min(),z.predicted.min()),max(z.observed.max(),z.predicted.max())]
        ax.plot(lim,lim,'k--',lw=1);ax.set(title=panel.replace('_',' '),xlabel='RAPID child–parent shift (min)',ylabel='Held-river RF prediction (min)');ax.grid(alpha=.15)
    axes[0,0].legend();fig.suptitle('Frozen predictor sets · real-only fits · whole rivers held out')
    fig.savefig(OUT/'real_only_RF_predictions.png',dpi=170);plt.close(fig)
    (OUT/'evaluation_completion.json').write_text(json.dumps(dict(complete=True,seconds=time.monotonic()-start,
        cases=2185,rivers=15,panels=4,quadratic_fits=60,forest_fits=60,baseline_predictions=True,
        sources={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),OUT/'input.parquet',OUT/'evaluation_design.json']},
        no_synthetic_coefficients=True,no_routing_runs=True),indent=2))
    print(result[(result.cohort=='all')&(result.weighting=='equal_cohort_network')].to_string(index=False))
if __name__=='__main__':main()
