"""Descriptive control/predictor associations; no fitting or routing."""
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def correlations(features, grid, predictors, method, statistic='spearman'):
    if statistic not in ('spearman','pearson'):raise ValueError(statistic)
    rows=[]
    for state,g in features[features.method.eq(method)].groupby('child_state'):
        for factor,levels in grid.items():
            # Location is explicitly ordered upstream -> central -> downstream.
            x=g[factor].map(dict(zip(levels,range(len(levels))))) if factor=='internal_location' else g[factor]
            for predictor in predictors:
                y=g[predictor];valid=np.isfinite(x)&np.isfinite(y)
                xx=x[valid];yy=y[valid]
                tol=1e-10*max(float(yy.abs().max()) if len(yy) else 0.,1.)
                varying=len(yy)>1 and np.ptp(yy)>tol and xx.nunique()>1
                # Suppress floating-point differences between analytically tied values.
                yy=(yy/tol).round()*tol
                rho=float(xx.corr(yy,method=statistic)) if varying else np.nan
                rows.append(dict(child_state=state,factor=factor,predictor=predictor,correlation=rho,n_finite=int(valid.sum()),n_total=len(g),finite_pct=100*valid.mean()))
    return pd.DataFrame(rows)


def matrices(features,effects,grid,predictors,method,treatment=None,statistic='spearman'):
    corr=correlations(features,grid,predictors,method,statistic)
    terms=effects[effects.method.eq(method)&effects.variable.isin(predictors)]
    if treatment is not None:
        corr=corr[corr.child_state.eq(treatment)]
        terms=terms[terms.child_state.eq(treatment)]
    result={}
    for name,table,index,column,value,order in [
        ('correlation',corr,'factor','predictor','correlation',list(grid)),
        ('main',terms[terms.order.eq(1)],'term','variable','variance_pct',list(grid)),
        ('interaction',terms[terms.order.eq(2)],'term','variable','variance_pct',[' × '.join(p) for p in itertools.combinations(grid,2)])]:
        # Means here only combine per-treatment summaries, never raw mixed populations.
        grouped=table.groupby([index,column])[value]
        result[name]=grouped.mean().unstack().reindex(index=order,columns=predictors)
        result[name+'_counts']=grouped.count().unstack().reindex(index=order,columns=predictors).fillna(0).astype(int)
    return result,corr


def figure(data,title,labels,factor_labels,signed=False):
    rows,cols=data.shape
    fig,ax=plt.subplots(figsize=(14,max(5,rows*.43)),constrained_layout=True)
    cmap=plt.get_cmap('coolwarm' if signed else 'viridis').copy();cmap.set_bad('#e5e7eb')
    values=data.to_numpy();finite=values[np.isfinite(values)]
    bound=1. if signed else max(float(finite.max()) if len(finite) else 0.,1.)
    im=ax.imshow(values,aspect='auto',cmap=cmap,vmin=-1 if signed else 0,vmax=bound)
    for i in range(rows):
        for j in range(cols):
            v=values[i,j]
            if np.isfinite(v):
                rgba=im.cmap(im.norm(v));light=.2126*rgba[0]+.7152*rgba[1]+.0722*rgba[2]
                ax.text(j,i,f'{v:.2f}' if signed else ('<0.1' if 1e-8<v<.05 else f'{v:.1f}'),ha='center',va='center',fontsize=7,color='black' if light>.55 else 'white')
            else:ax.text(j,i,'—',ha='center',va='center',fontsize=8,color='#555')
    ax.set(xticks=range(cols),xticklabels=[labels.get(c,c) for c in data.columns],yticks=range(rows),yticklabels=[' × '.join(factor_labels.get(t,t) for t in r.split(' × ')) for r in data.index],title=title)
    plt.setp(ax.get_xticklabels(),rotation=40,ha='right')
    fig.colorbar(im,ax=ax,label='Correlation (−1 to +1)' if signed else '% of predictor variance (linear colour scale; figure-specific maximum)',shrink=.75)
    return fig
