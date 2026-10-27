"""Reconstruct one saved held-network forest and audit its extrapolation limit."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
import relative_model_sweep as sweep

engine=sweep.engine
OUT=engine.ROOT/'RAPID/outputs/hydrograph_extreme_extrapolation_20261019'
SOURCE=sweep.OUT

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    d=pd.read_parquet(SOURCE/'input.parquet')
    tr=d[d.example!='37'].copy().reset_index(drop=True)
    te=d[(d.example=='37')&(d.state=='CB001_bubble_3')].copy()
    assert len(te)==1
    checkpoint=SOURCE/'checkpoints/all/network/37/centroid_shift_hours__forest.json'
    saved=json.loads(checkpoint.read_text())
    selected=next(r for r in saved['results'] if r['floor_minutes']==1)
    expected=next(r['predicted_hours'] for r in selected['predictions'] if r['state']=='CB001_bubble_3')*60
    y=tr.centroid_shift_hours.to_numpy(float);w=engine.weights(tr)
    pre=engine.Preprocess(sweep.COLUMNS,None,None).fit(tr,y)
    x,xt=pre.transform(tr),pre.transform(te)
    mean=np.average(y,weights=w);sd=np.sqrt(np.average((y-mean)**2,weights=w))
    model=engine.estimator('forest',selected['params'],1)
    model.fit(x,(y-mean)/sd,sample_weight=w)
    prediction=float((model.predict(xt)[0]*sd+mean)*60)
    np.testing.assert_allclose(prediction,expected,rtol=1e-10)
    observed=float(te.centroid_shift_hours.iloc[0]*60)
    ranges=[]
    for c in sweep.COLUMNS:
        log=c.startswith('log1p__');convert=np.expm1 if log else lambda a:a
        vals=convert(tr[c]);value=float(convert(te[c].iloc[0]))
        ranges.append(dict(feature=c,display_feature=c.removeprefix('log1p__'),value=value,train_min=vals.min(),train_max=vals.max(),train_p95=vals.quantile(.95),train_percentile=float((vals<=value).mean()*100),outside=bool(value<vals.min()-1e-10 or value>vals.max()+1e-10)))
    pd.DataFrame(ranges).to_csv(OUT/'feature_ranges.csv',index=False)
    # Exact forest prediction as a convex combination of training responses.
    # Bootstrap multiplicity and sample weights are both included.
    influence=np.zeros(len(tr));tree_rows=[]
    for tree,sampled in zip(model.estimators_,model.estimators_samples_):
        leaf=tree.apply(xt)[0];members=tree.apply(x)==leaf
        weights=w*np.bincount(sampled,minlength=len(tr))*members
        weights=weights/weights.sum()
        influence+=weights/len(model.estimators_)
        tree_rows.append(dict(leaf_rows=int(members.sum()),bootstrap_unique_rows=int((weights>0).sum()),prediction_minutes=float(np.sum(weights*y)*60)))
    np.testing.assert_allclose(influence.sum(),1.)
    np.testing.assert_allclose(np.sum(influence*y)*60,prediction,atol=1e-8)
    tr['forest_weight']=influence;tr['centroid_minutes']=y*60
    tr['prediction_contribution_minutes']=influence*y*60
    tr.sort_values('forest_weight',ascending=False).to_parquet(OUT/'forest_training_influence.parquet',index=False)
    pd.DataFrame(tree_rows).to_csv(OUT/'tree_predictions.csv',index=False)
    # Geometry-only neighbours in the actual transformed feature space.
    # Drop machine-precision constants from this descriptive distance only;
    # exact reconstruction above keeps the original model preprocessing intact.
    used=[i for i,c in enumerate(pre.kept) if np.ptp(tr[c])>1e-10]
    ignored=[c for c in pre.kept if np.ptp(tr[c])<=1e-10]
    dist=cdist(xt[:,used],x[:,used])[0]
    tr['geometry_distance']=dist
    nearest=tr.nsmallest(15,'geometry_distance')
    nearest.to_csv(OUT/'nearest_geometry.csv',index=False)
    tr[tr.edit_type=='Full compound'].nlargest(15,'domain_size').to_csv(OUT/'largest_training_compounds.csv',index=False)
    tr.nsmallest(15,'centroid_minutes').to_csv(OUT/'largest_training_advances.csv',index=False)
    distances=cdist(x[:,used],x[:,used]);same=tr.example.to_numpy()[:,None]==tr.example.to_numpy()[None,:]
    distances[same]=np.inf
    nn=distances.min(axis=1)
    stats=dict(observed_minutes=observed,predicted_minutes=prediction,training_min_minutes=float(y.min()*60),training_max_minutes=float(y.max()*60),
       total_underprediction_minutes=prediction-observed,unavoidable_range_error_minutes=float(y.min()*60-observed),
       range_bound_fraction_of_error=float((y.min()*60-observed)/(prediction-observed)),
       nearest_geometry_distance=float(dist.min()),training_cross_network_nearest_p95=float(np.quantile(nn,.95)),
       training_cross_network_nearest_p99=float(np.quantile(nn,.99)),nearest_distance_percentile=float(np.mean(nn<=dist.min())*100),
       distance_ignored_numerical_constants=ignored,tree_prediction_min_minutes=min(r['prediction_minutes'] for r in tree_rows),tree_prediction_max_minutes=max(r['prediction_minutes'] for r in tree_rows),
       effective_training_rows=float(1/np.sum(influence**2)),support_rows=int((influence>0).sum()),params=selected['params'],
       note='One diagnostic reconstruction, no tuning. Lower bound conditional on these training data and convex-average forest class; not a causal estimate of benefit from more data. Euclidean distances are descriptive, training-scaled and not a formal applicability domain.',
       sources={str(p):engine.sha(p) for p in [Path(__file__),checkpoint,SOURCE/'input.parquet',Path(sweep.__file__),Path(engine.__file__)]})
    engine.dump(OUT/'audit.json',stats)
    print(json.dumps(stats,indent=2))
    print('Nearest:',nearest[['example','state','edit_type','domain_size','centroid_minutes','geometry_distance']].to_string(index=False))
    print('Influential:',tr.nlargest(8,'forest_weight')[['example','state','centroid_minutes','forest_weight']].to_string(index=False))

if __name__=='__main__':main()
