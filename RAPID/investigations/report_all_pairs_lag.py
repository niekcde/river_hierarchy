"""Report the unfiltered, deduplicated 2,214-pair timing analysis."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'RAPID/validation/all_pairs_lag_20261005'
KEY=['example_id','fingerprint','matched_base_link_ids','child_link_ids']

def table(d):return '\n'.join('| '+' | '.join(row)+' |' for row in [list(d.columns),['---']*len(d.columns),*d.astype(str).values.tolist()])

def main():
    f=pd.read_parquet(OUT/'features.parquet');canonical=pd.read_parquet(ROOT/'RAPID/validation/deduplicated_response_benchmark_20261005/canonical_all.parquet')
    assert len(f)==2214 and set(map(tuple,f[KEY].to_numpy()))==set(map(tuple,canonical[KEY].to_numpy()))
    old=pd.read_parquet(ROOT/'RAPID/validation/full_population_lag_20261005/features.parquet');coverage=pd.concat([old.groupby('example_id').size().rename('previous_filtered'),f.groupby('example_id').size().rename('all_pairs')],axis=1);coverage['restored']=coverage.all_pairs-coverage.previous_filtered;assert coverage.restored.sum()==562;coverage.to_csv(OUT/'coverage.csv')
    f.groupby('plot_category').size().rename('pairs').to_csv(OUT/'categories.csv')
    scores=pd.read_csv(OUT/'scores.csv');p=pd.read_parquet(OUT/'predictions.parquet');assert p.groupby(['method','response']).size().eq(2214).all()
    fixed=['omega/boosted_7','components/boosted_7','selected_six/boosted_7','nested_selected']
    summary=scores[scores.scope.eq('pooled')&scores.method.isin(fixed)][['response','method','n','r2','mae','skill_vs_zero']];summary.to_csv(OUT/'main_comparison.csv',index=False)
    equal=scores[scores.scope.eq('river')&scores.method.isin(fixed)].groupby(['response','method']).agg(equal_example_mae=('mae','mean'),median_example_r2=('r2','median'),examples_beating_zero=('skill_vs_zero',lambda v:int((v>0).sum()))).reset_index();equal.to_csv(OUT/'equal_example_summary.csv',index=False)
    selected=p[p.method.eq('nested_selected')].merge(f[KEY+['state_id','mode']],on=KEY,validate='many_to_one');selected['absolute_error']=(selected.observed-selected.predicted).abs();selected.sort_values('absolute_error',ascending=False).groupby('response').head(12).to_csv(OUT/'largest_errors.csv',index=False)
    responses=['interpolated_lag_shift_pct','interpolated_peak_shift_hours'];labels=['Percentage lag change (%)','Absolute timing change (h)']
    fig,axes=plt.subplots(1,2,figsize=(12,5),constrained_layout=True)
    for ax,response,label in zip(axes,responses,labels):
        d=selected[selected.response.eq(response)]
        for category,z in d.groupby('plot_category'):ax.scatter(z.observed,z.predicted,s=10,alpha=.45,label=category)
        lim=[min(d.observed.min(),d.predicted.min()),max(d.observed.max(),d.predicted.max())];ax.plot(lim,lim,'k--',lw=.8);ax.set(title=label,xlabel='Routed response',ylabel='Held-out prediction');ax.grid(alpha=.12)
    axes[1].legend(fontsize=7);fig.suptitle('All 2,214 parent–child pairs · no edge-compound exclusion');fig.savefig(OUT/'held_out_timing.png',dpi=150);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,4),constrained_layout=True)
    for ax,response,label in zip(axes,responses,labels):
        z=summary[summary.response.eq(response)].set_index('method').reindex(fixed);ax.barh(['Omega','Separate components','Selected six','Nested selection'],z.skill_vs_zero);ax.axvline(0,c='k',lw=.7);ax.set(title=label,xlabel='MAE skill versus no change')
    fig.savefig(OUT/'timing_skill.png',dpi=150);plt.close(fig)
    t=pd.read_csv(OUT/'synthetic_transfer_scores.csv');ts=t[t.model_label.eq('selected')][['response','dataset','n','r2','mae','skill_vs_zero']]
    fig,axes=plt.subplots(1,2,figsize=(12,4),constrained_layout=True)
    for ax,response,label in zip(axes,responses,labels):
        z=ts[ts.response.eq(response)];ax.barh(z.dataset,z.skill_vs_zero);ax.axvline(0,c='k',lw=.7);ax.set(title=label,xlabel='MAE skill versus no change')
    fig.savefig(OUT/'synthetic_transfer_skill.png',dpi=150);plt.close(fig)
    text='''# All parent–child pairs — corrected analysis, 5 October 2026

Reference **ALL-PAIRS-LAG-2026-10-05**. The earlier 1,652-row analysis was incorrectly described as the full population: it retained an inherited first/last compound-group filter. The user explicitly requested every distinct pair. That filter is now removed from active plotting and prediction. This report supersedes the filtered-population timing results.

**2,214 canonical parent–child pairs, 16 network examples, zero edge-compound exclusions, zero path-count exclusions, zero missing-feature exclusions.** Duplicate representations remain collapsed to one comparison; aliases are retained in the original identity audit. All lag responses are valid. Exact comparison identities are checked against the complete canonical inventory, not just a row count.

## Coverage

'''+table(coverage.reset_index())+'''

Example 37 was especially affected: 395 pairs were removed with a first/last compound group, leaving only one. The old filter removed member units as well as whole-compound collapses; it was not a scientific reason to discard their model results.

Scalable path descriptors retain the same equations, including shared-edge path counting. The 1,128 old enumerated records still agree within rtol=1e-8/atol=1e-10. Single-path invariants and quadrature convergence are checked. No geometry imputation or path sampling is used. The largest path counts now exceed 1e25, which reinforces the need to interpret path-based width/dominance cautiously rather than as counts of independent channels.

## Real timing search

The same 61 candidate model/feature combinations are compared on every pair. Outer validation holds out an entire network example; four grouped inner folds select by equal-example MAE. Scalers are fitted within training folds. The logarithmic surrounding-length candidates use log10(1+L/metre), which can retain zero-length context if a domain touches an external boundary. Other positive geometric variables use log10. This does not change any raw descriptor or drop boundary cases.

'''+table(summary.round(4))+'''

'''+table(equal.round(4))+'''

Pooled R² and MAE answer different questions; model selection optimizes inner equal-example MAE, not pooled outer R². Fixed-candidate tables are descriptive benchmarks; inspecting them to choose the highest outer R² would introduce selection optimism. These network examples were previously explored and may belong to related basins. This is grouped exploratory validation, not untouched basin-independent confirmation.

## Synthetic selection and transfer

Selection uses only original synthetic development geometries. The original holdout and the novel embedding extension are excluded from training/model choice; 63 original-grid extension anchors are excluded from extension scoring. Transfer now evaluates **all 2,214 real pairs**, not the old filtered subset. The same guarded peak-timing definitions are used. These previously examined test populations remain exploratory.

'''+table(ts.round(4))+'''

Shape predictor search remains paused. Hydraulic-domain results are unchanged by this population correction; no routing was rerun. The earlier 1,652/1,128-row reports remain historical, explicitly filtered analyses. Current notebook plots and the active timing-search sections use all 2,214 pairs.

Reproduce: `full_population_features.py`, `full_population_lag_search.py`, `full_population_synthetic_transfer.py`, `report_all_pairs_lag.py`. `coverage.csv` lists every restored count. `features.parquet`, fold predictions, `scores.csv`, `selections.csv`, `search_plan.json`, geometry parity and provenance are retained here. The input-signature guard prevents resuming cached folds against different features/model definitions.
'''
    (OUT/'README.md').write_text(text);print(summary.to_string(index=False))

if __name__=='__main__':main()
