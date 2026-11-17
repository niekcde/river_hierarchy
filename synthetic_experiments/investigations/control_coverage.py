"""Matched control coverage, conditioned on resolved response changes; no fitting."""
import numpy as np
import pandas as pd

def coverage_table(features,grid,predictors,response='centroid_shift_minutes'):
    factors=list(grid);index=pd.MultiIndex.from_product(grid.values(),names=factors)
    shape=tuple(len(grid[k]) for k in factors);records=[]
    # Descriptive screening spans, consistent with full-grid response resolution screens.
    response_span=4/60 if response=='centroid_shift_minutes' else .5
    for (method,state),g in features.groupby(['method','child_state']):
        g=g.set_index(factors).reindex(index)
        y=g[response].to_numpy().reshape(shape)
        for i,factor in enumerate(factors):
            yy=np.moveaxis(y,i,-1).reshape(-1,shape[i])
            changing=np.isfinite(yy).all(axis=1)&(np.ptp(yy,axis=1)>response_span)
            for predictor in predictors:
                a=g[predictor].to_numpy().reshape(shape)
                aa=np.moveaxis(a,i,-1).reshape(-1,shape[i])
                finite=np.isfinite(aa).all(axis=1);eligible=changing&finite
                values=a[np.isfinite(a)];scale=max(float(np.max(abs(values))) if len(values) else 0.,1.)
                tol=1e-10*scale;sd=np.std(values) if len(values) else np.nan
                spans=np.ptp(aa[eligible],axis=1);changed=spans>tol
                records.append(dict(method=method,child_state=state,response=response,factor=factor,predictor=predictor,
                    contexts=len(yy),response_changing_contexts=int(changing.sum()),valid_predictor_contexts=int(eligible.sum()),
                    response_changes_pct=100*changing.mean(),predictor_available_pct=100*eligible.sum()/changing.sum() if changing.any() else np.nan,
                    predictor_changes_pct=100*changed.mean() if len(spans) else np.nan,
                    median_predictor_span_sd=float(np.median(spans)/sd) if len(spans) and sd>tol else 0. if len(spans) else np.nan,
                    response_span_threshold_minutes=response_span,predictor_numerical_tolerance=tol))
    return pd.DataFrame(records)
