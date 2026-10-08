"""Full within-network distributions for the seven failing base networks."""
from pathlib import Path
import json
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, ScalarFormatter
ROOT=Path(__file__).resolve().parents[2];RUN=ROOT/'RAPID/assets/bounded_robust_slopes_20261002';OUT=RUN/'distributions';OUT.mkdir(exist_ok=True)
CAP_MAX=2.524

def quantile(v,w,qs):
 i=np.argsort(v);v=np.asarray(v)[i];w=np.asarray(w)[i];return np.interp(qs,(np.cumsum(w)-.5*w)/w.sum(),v)

def ecdf(ax,v,w,**kw):
 i=np.argsort(v);ax.step(np.asarray(v)[i],np.cumsum(np.asarray(w)[i])/np.sum(w),where='post',**kw)

def main():
 d=pd.read_csv(RUN/'stability_link_details.csv',dtype={'example':str});choices=pd.read_csv(RUN/'stability_conflict_counts.csv',dtype={'example':str}).set_index('example');refs=pd.read_csv(RUN/'hydraulic_references.csv',dtype={'example':str}).set_index('example');summaries=[];tails=[];flagged=[]
 variables=[('wid_adj_routing','Width (m)'),('depth_m','Estimated depth (m)'),('slope','Slope (m/m)'),('celerity_raw_mps','Raw celerity (m/s)'),('K_seconds','Used segment K (s)'),('effective_length_m','Effective segment length (m)')]
 for ex,g in d.groupby('example'):
  g=g.copy();L=g.actual_length_m.to_numpy();dt=int(choices.loc[ex,'representative_dt_s']);g['diagnostic_side']=np.where(g.dt_lower_s>dt,'slow',np.where(g.dt_upper_s<dt,'fast','compatible'))
  fig,axes=plt.subplots(2,3,figsize=(14,8),constrained_layout=True)
  for ax,(column,label) in zip(axes.flat,variables):
   values=g[column].to_numpy();ecdf(ax,values,np.ones(len(g)),color='#475569',lw=1.5,label='Equal original links');ecdf(ax,values,L,color='#059669',lw=2,label='Channel-length weighted');ax.set_xscale('log');ax.set(xlabel=label,ylabel='Cumulative fraction',ylim=(-.02,1.07));ax.grid(alpha=.15)
   for weightname,weights in [('links',np.ones(len(g))),('length',L)]:
    qs=quantile(values,weights,[.01,.05,.25,.5,.75,.95,.99]);row=dict(example=ex,variable=column,weighting=weightname,minimum=values.min(),maximum=values.max());row.update(dict(zip(['q01','q05','q25','q50','q75','q95','q99'],qs)));summaries.append(row)
   for side,color,y in [('slow','#dc2626',1.01),('fast','#2563eb',1.035)]:
    ax.plot(g.loc[g.diagnostic_side==side,column],np.full((g.diagnostic_side==side).sum(),y),'|',color=color,markersize=6,label=f'{side.title()} conflict at {dt} s')
   if column in ['wid_adj_routing','depth_m']:
    ref=refs.loc[ex,'reference_width_m' if column=='wid_adj_routing' else 'stored_depth_m'];ax.axvline(ref,color='#9333ea',ls=':',lw=1,label='BASED reference')
   if column=='celerity_raw_mps':
    ecdf(ax,g.celerity_mps,L,color='#f59e0b',ls='--',lw=1.3,label='Used celerity, length weighted');ax.axvline(CAP_MAX,color='#f59e0b',ls=':',lw=1)
   if column=='effective_length_m':
    ax.set_xscale('linear');ax.xaxis.set_major_locator(MaxNLocator(nbins=5));ax.xaxis.set_major_formatter(ScalarFormatter())
   if column=='K_seconds':ax.axvspan(dt/1.8,dt/.2,color='#22c55e',alpha=.08,label='Compatible K at diagnostic dt')
  axes[0,0].legend(fontsize=7,loc='upper left');axes[1,0].legend(fontsize=7,loc='lower right');axes[1,1].legend(fontsize=7,loc='lower right');fig.suptitle(f'Case {ex}: full distributions; {len(g)} original links; diagnostic dt = {dt} s\nRed/blue ticks mark numerical conflicts, not presumed bad observations',fontsize=12);fig.savefig(OUT/f'{ex}_distributions.png',dpi=140);plt.close(fig)
  for column in ['celerity_raw_mps','K_seconds']:
   for weighting,weights in [('links',np.ones(len(g))),('length',L)]:
    lq=quantile(np.log10(g[column].to_numpy()),weights,[.25,.5,.75]);iqr=lq[2]-lq[0];lo=10**(lq[0]-1.5*iqr);hi=10**(lq[2]+1.5*iqr);g[f'{column}_{weighting}_lower_tail']=g[column]<lo;g[f'{column}_{weighting}_upper_tail']=g[column]>hi
    for side,mask in [('lower',g[column]<lo),('upper',g[column]>hi)]:tails.append(dict(example=ex,variable=column,weighting=weighting,side=side,lower_fence=lo,upper_fence=hi,count=int(mask.sum()),link_pct=100*mask.mean(),length_pct=100*L[mask].sum()/L.sum(),ids=';'.join(map(str,g.loc[mask,'id_link']))))
  for side,mask in [('slow',g.diagnostic_side=='slow'),('fast',g.diagnostic_side=='fast'),('slow_only',g.too_slow_for_current_fastest),('fast_only',g.too_fast_for_current_slowest)]:
   tails.append(dict(example=ex,variable='numerical_conflict',weighting='diagnostic_dt' if side in ['slow','fast'] else 'opposite_extreme_fixed',side=side,count=int(mask.sum()),link_pct=100*mask.mean(),length_pct=100*L[mask].sum()/L.sum(),ids=';'.join(map(str,g.loc[mask,'id_link']))))
  flagged.extend(g.to_dict('records'))
 pd.DataFrame(summaries).to_csv(OUT/'quantiles.csv',index=False);pd.DataFrame(tails).to_csv(OUT/'tail_summary.csv',index=False);pd.DataFrame(flagged).to_csv(OUT/'links_with_distribution_flags.csv',index=False)
 # Compare the seven networks on one consistent travel-time axis.
 nrows=int(np.ceil(d.example.nunique()/2));fig,axes=plt.subplots(nrows,2,figsize=(13,3.25*nrows),constrained_layout=True)
 for ax,(ex,g) in zip(axes.flat,d.groupby('example')):
  ecdf(ax,g.K_seconds,np.ones(len(g)),color='#475569',label='Equal links');ecdf(ax,g.K_seconds,g.actual_length_m,color='#059669',label='Length weighted');ecdf(ax,g.effective_length_m/g.celerity_raw_mps,g.actual_length_m,color='#f59e0b',ls='--',label='Uncapped K, length weighted');dt=int(choices.loc[ex,'representative_dt_s']);ax.axvspan(dt/1.8,dt/.2,color='#22c55e',alpha=.12);ax.set(xscale='log',xlim=(20,8000),ylim=(0,1.02),xlabel='Segment travel time K (s)',ylabel='Cumulative fraction',title=f'Case {ex} · shaded window at {dt} s');ax.grid(alpha=.15)
 axes.flat[0].legend(fontsize=8)
 if d.example.nunique()%2:
  axes.flat[-1].axis('off');axes.flat[-1].text(0,.9,'Left tail = fast; right tail = slow.\n\nShading spans a feasible K ratio of 9\nat a diagnostic integer timestep.\n\nLength weighting sums physical channel lengths;\nit is not a discharge or centerline weighting.\n\nNo hydraulic inputs were changed.',va='top',fontsize=11)
 fig.savefig(OUT/'all_K_distributions.png',dpi=140);plt.close(fig)
 print('Distribution audit complete')
if __name__=='__main__':
 import argparse
 parser=argparse.ArgumentParser();parser.add_argument('--run-dir',default=str(RUN));parser.add_argument('--cap-max',type=float,default=2.524);args=parser.parse_args();RUN=Path(args.run_dir);OUT=RUN/'distributions';OUT.mkdir(exist_ok=True);CAP_MAX=args.cap_max;main()
