"""Show the very small background mismatches in networks 22 and 45."""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from review_revised_backgrounds_20261013 import OUT, pairs, read_wave

definitions=json.loads((OUT/'candidate_definitions.json').read_text())
fig,axes=plt.subplots(2,2,figsize=(12,7),sharex='col')
rows=[]
for col,ex in enumerate(['22','45']):
    variants=definitions['cases'][ex]
    old=next(c for c in variants if c['key']=='proposed')
    alt=next(c for c in variants if c['key']==('lowered_end' if ex=='22' else 'held_ramp'))
    t,q=read_wave(pairs[pairs.example_id.eq(ex)].iloc[0].parent_outlet_parquet)
    end=max(old['knots'][-1][0],alt['knots'][-1][0])
    start=old['knots'][-1][0]-(3 if ex=='22' else 4)*3600
    tt=np.unique(np.r_[start,t[(t>=start)&(t<=end)],*[x for x,y in old['knots'] if start<=x<=end],*[x for x,y in alt['knots'] if start<=x<=end],end])
    qq=np.interp(tt,t,q);dates=pd.to_datetime(tt,unit='s',utc=True)
    ax=axes[0,col];ax.plot(dates,qq,color='#ce3d45',label='Model Q',lw=1.8)
    for c,label,color in [(old,'Current straight background','#9b59b6'),(alt,'Candidate horizontal finish','#168a65')]:
        kt,kq=np.array(c['knots']).T
        bg=np.interp(tt,kt,kq);bg[(tt<kt[0])|(tt>kt[-1])]=np.nan
        ax.plot(dates,bg,color=color,ls='--',label=label,lw=1.5)
        axes[1,col].plot(dates,qq-bg,color=color,label=label,lw=1.5)
        if c is old:
            ax.fill_between(dates,qq,bg,where=qq<bg,color='#e68c28',alpha=.4,interpolate=True)
            axes[1,col].fill_between(dates,qq-bg,0,where=qq<bg,color='#e68c28',alpha=.4,interpolate=True)
    ax.set(title=f"{ex} · {pd.Timestamp(end,unit='s',tz='UTC').strftime('%Y-%m-%d')} (UTC)",ylabel='Q (m³/s)')
    axes[1,col].axhline(0,color='#444',lw=.8)
    axes[1,col].set(ylabel='Q − B (m³/s)',xlabel='Time (UTC)')
    axes[1,col].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    for a in axes[:,col]:a.grid(alpha=.2)
    ax.legend(fontsize=8)
    hold=alt['knots'][-1][0]-alt['knots'][-2][0]
    rows.append(dict(example=ex,maximum_background_excess_cms=-old['minimum_excess_cms'],
        maximum_excess_pct_of_end_Q=100*(-old['minimum_excess_cms'])/old['end_cms'],
        deficit_pct_of_positive_volume=old['deficit_to_positive_volume_pct'],
        current_end_cms=old['end_cms'],candidate_end_cms=alt['end_cms'],
        horizontal_finish_seconds=hold,candidate_end_utc=alt['end_utc']))
fig.suptitle('Tiny background mismatches: visible in Q − B, barely visible in Q\nThese are analysis-background differences, not evidence of a routing error.',fontsize=13)
fig.tight_layout(rect=(0,0,1,.92));fig.savefig(OUT/'22_45_recession_detail.png',dpi=170);plt.close(fig)
pd.DataFrame(rows).to_csv(OUT/'22_45_recession_detail.csv',index=False)
print(pd.DataFrame(rows).to_string(index=False))
