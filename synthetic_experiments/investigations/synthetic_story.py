"""Two presentation figures linking experimental controls to predictor mechanisms."""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

CONTROLS={'total_width_m':'Reference width','reference_slope':'Reference slope','outer_a_fraction':'Outer allocation','inner_fraction':'Inner/crossover allocation','length_multiplier':'Branch length','total_length_m':'Network extent','internal_location':'Internal location','branch_width_factor':'Branch-width expansion'}
RESPONSES=['centroid_shift_minutes','peak_shift_minutes']

def figures(effects,coverage,method,predictors,labels):
    resolved=coverage[(coverage.method==method)&coverage.variable.isin(RESPONSES)].pivot(index='child_state',columns='variable',values='resolved')
    types=resolved.index[resolved.all(axis=1)].tolist()
    if not types:raise ValueError('No jointly resolved response types')
    chosen=effects[effects.method.eq(method)&effects.child_state.isin(types)]
    grouped=chosen.groupby(['term','variable']).variance_pct
    mean=grouped.mean().unstack();count=grouped.count().unstack()
    mains=mean.reindex(CONTROLS)[RESPONSES].max(axis=1).sort_values(ascending=False).index.tolist()
    pairs=mean[mean.index.str.contains(' × ')][RESPONSES].max(axis=1).sort_values(ascending=False).head(4).index.tolist()
    terms=mains+pairs
    termlabels=[' × '.join(CONTROLS.get(t,t) for t in term.split(' × ')) for term in terms]
    responses=mean.reindex(index=terms,columns=RESPONSES)
    # Figure 1 uses disjoint main/pair contributions, never overlapping total effects.
    fig1,ax=plt.subplots(figsize=(12,7.7),constrained_layout=True)
    y=np.arange(len(terms));maxval=responses.max().max()
    for i,(col,label,color) in enumerate(zip(RESPONSES,['Centroid shift','Peak shift'],['#2563eb','#d97706'])):
        v=responses[col].to_numpy();yy=y+(i-.5)*.34
        ax.barh(yy,v,height=.32,label=label,color=color)
        for pos,value in zip(yy,v):ax.text(value+.35,pos,f'{value:.1f}%',fontsize=9,va='center')
    ax.axhline(len(mains)-.5,color='#64748b',lw=1)
    ax.set(yticks=y,yticklabels=termlabels,xlim=(0,maxval*1.19),xlabel='Contribution to response variance (%)',title='1. Length, scale and allocation drive timing changes')
    ax.invert_yaxis();ax.legend(loc='lower right');ax.grid(axis='x',alpha=.15)
    fig1.supxlabel('Eight main effects above the divider; four strongest response pair interactions below.\nEqual weight across '+str(len(types))+' resolved collapse types; remaining interactions are not shown.',fontsize=9)
    # Figure 2: same terms, all original and candidate predictors.
    data=mean.reindex(index=terms,columns=predictors).T
    fig2,ax=plt.subplots(figsize=(16,9),constrained_layout=True)
    cmap=plt.get_cmap('viridis').copy();cmap.set_bad('#e5e7eb');norm=Normalize(0,100)
    im=ax.imshow(data,aspect='auto',cmap=cmap,norm=norm)
    for i in range(len(data)):
        for j in range(len(terms)):
            v=data.iloc[i,j]
            if np.isfinite(v):
                rgb=cmap(norm(v));light=.2126*rgb[0]+.7152*rgb[1]+.0722*rgb[2]
                ax.text(j,i,f'{v:.1f}',ha='center',va='center',fontsize=8,color='black' if light>.55 else 'white')
            else:ax.text(j,i,'—',ha='center',va='center',color='#555',fontsize=9)
    counts=count.reindex(index=terms,columns=predictors).fillna(0).max(axis=0).astype(int)
    names=[labels.get(p,p)+'  [n='+str(counts[p])+']' for p in predictors]
    ax.set(yticks=range(len(predictors)),yticklabels=names,xticks=range(len(terms)),xticklabels=termlabels,title='2. Predictors represent different parts of those timing mechanisms')
    plt.setp(ax.get_xticklabels(),rotation=40,ha='right',fontsize=9)
    ax.axvline(len(mains)-.5,color='white',lw=2)
    ax.axhline(9.5,color='white',lw=2)
    fig2.colorbar(im,ax=ax,label='Contribution to each predictor’s own variance (%)',shrink=.8)
    fig2.supxlabel('Original ten predictors above the horizontal divider; three new difference candidates below.\nColumns match Figure 1. n = available types; missing/constant quantities are gray. Similarity is mechanistic evidence, not a prediction score.',fontsize=9)
    return (fig1,fig2),dict(types=types,terms=terms,mean=mean,counts=count)
