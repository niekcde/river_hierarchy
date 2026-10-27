"""Summarise paired fixed/variable Manning results without refitting predictors."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'synthetic_experiments/outputs/fixed_variable_manning_20261027'
if not (OUT/'completion.json').exists():raise RuntimeError('Comparison run is not complete')
f=pd.read_csv(OUT/'responses.csv');keys=['case','branch_width_factor','family','child_state']
a=f[f.method=='fixed_manning'];b=f[f.method=='variable_manning']
if a.duplicated(keys).any() or b.duplicated(keys).any():raise ValueError('Duplicate pair keys')
x=a.merge(b,on=keys,suffixes=('_fixed','_variable'),validate='one_to_one')
assert len(x)==len(a)==len(b)
x['subset']=np.where(x['case'].str.startswith('core_'),'core','stress')
summary=[]
for metric in ['centroid_shift_minutes','peak_shift_minutes']:
 x[metric+'_difference']=x[metric+'_variable']-x[metric+'_fixed']
 x[metric+'_abs_difference']=x[metric+'_difference'].abs()
 for subset,rows in [('all',x),('core',x[x.subset=='core']),('stress',x[x.subset=='stress'])]:
  u=rows[metric+'_fixed'];v=rows[metric+'_variable'];d=v-u
  sign=lambda s:np.where(s.abs()<=.25,0,np.sign(s))
  eligible=u.abs()>=1
  rel=d[eligible].abs()/u[eligible].abs()*100
  summary.append(dict(metric=metric,subset=subset,n=len(rows),median_abs_difference_min=float(d.abs().median()),p90_abs_difference_min=float(d.abs().quantile(.9)),max_abs_difference_min=float(d.abs().max()),median_signed_difference_min=float(d.median()),spearman=float(spearmanr(u,v).statistic),sign_agreement_with_025min_deadband=float(np.mean(sign(u)==sign(v))),opposite_nonzero_signs=int(np.sum(sign(u)*sign(v)==-1)),n_relative_eligible=int(eligible.sum()),median_relative_difference_pct=float(rel.median()),max_relative_difference_pct=float(rel.max())))
x.to_csv(OUT/'method_pairs.csv',index=False);summary=pd.DataFrame(summary);summary.to_csv(OUT/'method_summary.csv',index=False)
simple=x[x.child_state=='simple/full'][keys+['subset']+[metric+suffix for metric in ['centroid_shift_minutes','peak_shift_minutes'] for suffix in ['_fixed','_variable','_difference']]]
simple.to_csv(OUT/'simple_full_summary.csv',index=False)
fig,axes=plt.subplots(1,2,figsize=(11,4.8),constrained_layout=True)
for ax,metric,label in zip(axes,['centroid_shift_minutes','peak_shift_minutes'],['Centroid shift','Peak shift']):
 for subset,color,marker in [('core','#2563eb','o'),('stress','#d97706','^')]:
  r=x[x.subset==subset];ax.scatter(r[metric+'_fixed'],r[metric+'_variable'],s=28,alpha=.65,c=color,marker=marker,label=f'{subset} geometries')
 vals=x[[metric+'_fixed',metric+'_variable']].to_numpy();low=min(vals.min(),0);high=max(vals.max(),0);margin=.05*max(1,high-low)
 ax.plot([low-margin,high+margin],[low-margin,high+margin],'k--',lw=1,label='Equal parent–child effect')
 ax.set(xlabel='Fixed Manning: child − parent (min)',ylabel='Variable Manning: child − parent (min)',title=label)
 ax.grid(alpha=.15);ax.legend(fontsize=8)
fig.suptitle('Fixed versus conservative variable Manning · 2:1 pulse')
fig.savefig(OUT/'fixed_vs_variable.png',dpi=160);plt.close(fig)
print(summary.round(4).to_string(index=False))
print('\nSIMPLE/FULL, width factor 1')
print(simple[simple.branch_width_factor==1].round(3).to_string(index=False))
metrics=pd.concat([pd.read_csv(p) for p in OUT.glob('groups/*/state_metrics.csv')],ignore_index=True)
def markdown(frame):
 return '\n'.join(['| '+' | '.join(map(str,frame.columns))+' |', '| '+' | '.join(['---']*len(frame.columns))+' |']+['| '+' | '.join(map(str,row))+' |' for row in frame.itertuples(index=False,name=None)])

lines=['# Fixed versus conservative variable Manning — 27 October 2026','',
'Purpose: test whether fixed-reference Manning approximates the parent–child timing effects obtained with conservative variable-depth routing. No predictor fitting or production-setting change.','',
'Four configurations within the core ranges: balanced split with length multiplier 1.25; dominant split with multiplier 1.5; that dominant geometry at low (5e-5) and high (5e-4) reference slopes. Two additional stress cases: 5% long branch and 95% long branch, each multiplier 3. All use width factors 1, 1.1 and 1.2, all seven collapse treatments, and a 2:1 forcing pulse. Total: 126 matched geometry/treatment pairs, 252 method-specific response rows. These are deterministic targeted checks, not independent field samples or generalisation scores.','',
'Fixed method uses midpoint reference Q=750 m3/s; variable method uses local Manning normal depth and storage-conservative MCT. Both use n=.025, diffusion multiplier 1, width-based allocation, 200m target cells, 15s native steps, and adaptive tail/centroid convergence. The core and stress configurations must be interpreted separately.','',
'## Paired results','',markdown(summary.round(5)),'',
'Absolute differences are variable minus fixed. Negative shifts mean collapse advances the response. Relative differences use only fixed shifts with magnitude >=1 min. Direction agreement uses a +/-0.25 min near-zero band. Spearman is descriptive ranking agreement on this targeted set, not a predictive-validation R2.','',
'## Simple/full examples at width factor 1','',markdown(simple[simple.branch_width_factor==1].round(4)),'',
'## Numerical checks','',
f"Maximum storage-aware native mass residual: {metrics.network_mass_residual_m3.abs().max():.6g} m3.",
f"Maximum excess-event volume error: {metrics.volume_error_fraction.abs().max():.6g} fraction.",
f"Maximum centroid convergence discrepancy: {metrics.centroid_convergence_error_seconds.abs().max():.6g} seconds.",'',
'## Interpretation','',
'The comparison tests magnitude, direction and rank of the collapse response, not agreement of absolute outlet timings. A strong ranking match does not make magnitude bias irrelevant. Check larger changes individually and avoid inflated relative errors for nearly zero responses.','',
'Both methods can produce a small positive peak shift while the centroid shifts earlier (notably the low-slope simple/full case). Hydraulic consistency does not guarantee all peak shifts are negative. This shared outcome is not by itself evidence that it occurs in real rivers; both methods share simplified diffusion-wave/junction assumptions.','',
'Use fixed Manning as a candidate simpler model only if the observed differences are acceptable for the study aim. The full factorial, other pulse shapes/durations, n values and real-network transfer remain outside this test.','',
'Artifacts: method_pairs.csv (all matched responses), method_summary.csv, simple_full_summary.csv, fixed_vs_variable.png; original group traces, parameters and convergence histories remain under groups/. Configuration: configs/fixed_variable_manning_20261027.json. Reproduce routing with run_depth_comparison.py and summarise with investigations/compare_fixed_variable_manning.py.']
(OUT/'REPORT.md').write_text('\n'.join(lines)+'\n')
