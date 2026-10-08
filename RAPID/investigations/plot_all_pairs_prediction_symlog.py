"""Signed-log presentation view of existing held-out predictions; no refitting."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from sklearn.metrics import r2_score
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'RAPID/validation/all_pairs_lag_20261005'
p=pd.read_parquet(OUT/'predictions.parquet')
p=p[p.method.eq('nested_selected')]
labels={'single simple':'Simple unit','nested unit':'Nested unit','part of compound unit':'Member of compound','compound unit compound':'Whole compound','compound indep':'Compound (independent mode)','island-only collapse':'Island-only collapse'}
fig,axes=plt.subplots(1,2,figsize=(13,6),layout='constrained')
for ax,response,label,threshold,ticks in zip(axes,['interpolated_lag_shift_pct','interpolated_peak_shift_hours'],['Percentage lag change (%)','Absolute peak-time change (h)'],[.01,1/60],[[-100,-10,-1,-.1,-.01,0,.01,.1,1,10],[-10,-1,-.1,-1/60,0,1/60,.1,1,10]]):
 d=p[p.response.eq(response)]
 assert len(d)==2214
 for category,z in d.groupby('plot_category'):
  ax.scatter(z.observed,z.predicted,s=18,alpha=.6,edgecolors='none',label=labels.get(category,category))
 bound=max(abs(d.observed).max(),abs(d.predicted).max())*1.18
 ax.set_xscale('symlog',linthresh=threshold,linscale=.7)
 ax.set_yscale('symlog',linthresh=threshold,linscale=.7)
 ax.plot([-bound,bound],[-bound,bound],color='black',ls='--',lw=1,label='Perfect prediction')
 ax.set(xlim=(-bound,bound),ylim=(-bound,bound),xlabel='RAPID '+label,ylabel='Predicted '+label)
 chosen=[v for v in ticks if abs(v)<bound];ax.set_xticks(chosen);ax.set_yticks(chosen)
 fmt=FuncFormatter(lambda v,pos: '0' if v==0 else f'{v:.3g}')
 ax.xaxis.set_major_formatter(fmt);ax.yaxis.set_major_formatter(fmt)
 ax.tick_params(axis='x',labelrotation=40,labelsize=9);ax.tick_params(axis='y',labelsize=9)
 ax.grid(alpha=.15);ax.set_aspect('equal',adjustable='box')
 ax.set_title(f'{label}\nHeld-out R² = {r2_score(d.observed,d.predicted):.2f} · n = {len(d):,}',fontsize=12)
axes[1].legend(loc='lower right',fontsize=8,framealpha=.95)
fig.suptitle('Geometry-based prediction of parent–child timing changes',fontsize=16)
fig.supxlabel('Signed logarithmic axes; linear near zero (±0.01 percentage points / ±1 minute).\nNegative = earlier child peak. R² is calculated on original signed responses, not log values.',fontsize=10)
for ext in ['png','pdf']:fig.savefig(OUT/f'held_out_timing_symlog.{ext}',dpi=200)
print(OUT/'held_out_timing_symlog.png')
