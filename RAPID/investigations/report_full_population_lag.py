"""Summarize completed full-coverage timing searches; preserve older audits."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'RAPID/validation/full_population_lag_20261005'

def table(d):
    rows=[list(d.columns),['---']*len(d.columns),*d.astype(str).values.tolist()]
    return '\n'.join('| '+' | '.join(row)+' |' for row in rows)

def main():
    scores=pd.read_csv(OUT/'scores.csv');pred=pd.read_parquet(OUT/'predictions.parquet');f=pd.read_parquet(OUT/'features.parquet');plan=json.loads((OUT/'search_plan.json').read_text());geo=json.loads((OUT/'geometry_provenance.json').read_text())
    chosen=pred[pred.method.eq('nested_selected')].copy();chosen=chosen.merge(f[['example_id','fingerprint','matched_base_link_ids','child_link_ids','state_id','mode']],on=['example_id','fingerprint','matched_base_link_ids','child_link_ids'],validate='many_to_one');chosen['absolute_error']=(chosen.observed-chosen.predicted).abs();chosen.sort_values('absolute_error',ascending=False).groupby('response').head(12).to_csv(OUT/'largest_errors.csv',index=False);assert len(chosen)==3304 and chosen.groupby('response').size().eq(1652).all()
    assert np.isfinite(pred[['observed','predicted']]).all().all();assert not pred.duplicated(['response','method','example_id','fingerprint','matched_base_link_ids','child_link_ids']).any()
    fixed=['omega/boosted_7','components/boosted_7','selected_six/boosted_7','nested_selected']
    summary=scores[scores.scope.eq('pooled')&scores.method.isin(fixed)][['response','method','n','r2','mae','skill_vs_zero']];summary.to_csv(OUT/'main_comparison.csv',index=False)
    rivers=scores[scores.scope.eq('river')&scores.method.isin(fixed)].groupby(['response','method']).agg(equal_example_mae=('mae','mean'),median_example_r2=('r2','median'),examples_beating_zero=('skill_vs_zero',lambda x:int((x>0).sum()))).reset_index();rivers.to_csv(OUT/'equal_example_summary.csv',index=False)
    responses=plan['responses'];labels=['Percentage lag change (%)','Absolute timing change (h)']
    fig,axes=plt.subplots(1,2,figsize=(12,5),constrained_layout=True)
    for ax,response,label in zip(axes,responses,labels):
        g=chosen[chosen.response.eq(response)]
        for category,z in g.groupby('plot_category'):ax.scatter(z.observed,z.predicted,s=10,alpha=.45,label=category)
        bounds=[min(g.observed.min(),g.predicted.min()),max(g.observed.max(),g.predicted.max())];ax.plot(bounds,bounds,'k--',lw=.8)
        ax.set(xlabel='Routed response',ylabel='Held-out prediction',title=label);ax.grid(alpha=.12)
    axes[1].legend(fontsize=7);fig.suptitle('Full population: nested model selection, 1,652 comparisons');fig.savefig(OUT/'held_out_timing.png',dpi=150);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,4),constrained_layout=True)
    for ax,response,label in zip(axes,responses,labels):
        z=summary[summary.response.eq(response)].set_index('method').reindex(fixed)
        ax.barh(['Omega','Separate components','Selected six','Nested selection'],z.skill_vs_zero,color=['#64748b','#2563eb','#0d9488','#d97706']);ax.axvline(0,c='k',lw=.7);ax.set(title=label,xlabel='MAE skill versus predicting no change')
    fig.savefig(OUT/'timing_skill.png',dpi=150);plt.close(fig)
    transfer=pd.read_csv(OUT/'synthetic_transfer_scores.csv');ts=transfer[transfer.model_label.eq('selected')][['response','dataset','n','r2','mae','skill_vs_zero']]
    fig,axes=plt.subplots(1,2,figsize=(12,4),constrained_layout=True)
    for ax,response,label in zip(axes,responses,labels):
        z=ts[ts.response.eq(response)];ax.barh(z.dataset,z.skill_vs_zero,color=['#2563eb','#0d9488','#d97706']);ax.axvline(0,c='k',lw=.7);ax.set(title=label,xlabel='MAE skill versus no change (negative = worse)')
    fig.savefig(OUT/'synthetic_transfer_skill.png',dpi=150);plt.close(fig)
    text='''# Full-population lag predictor search — 5 October 2026

Reference **FULL-POPULATION-LAG-2026-10-05**. Supersedes the path-limited timing benchmarks for population coverage. Scope is percentage lag and absolute timing only. No RAPID rerouting or response-definition change.

## Complete coverage and unchanged metrics

All **1,652 plotted comparisons across 16 network examples** now have the required descriptors. All 524 formerly excluded cases are recovered. The existing edge-compound filter remains: full population means the complete agreed plotted population, not every raw selection/alias in the repository.

The path algorithm retains the original definitions. It counts suffix paths by dynamic programming, calculates uniform-path length mean/variance, and evaluates E[L/R] and E[L²/R], where R=Σ(l/w), using 1/R=∫exp(−tR)dt. Thus total equivalent width=P E[L/R] and equivalent length=E[L²/R]/E[L/R]. Maximum harmonic path width is obtained by fractional optimization on the DAG. Checked quadrature replaces enumeration, not the metric. There is no path sampling, truncation by count or missing-value filling. Single-path dominance/association/CV invariants are calculated analytically, and association roundoff below 1e-12 is set to zero to avoid artificial near-zero splits in tree models.

All 1,128 previously enumerated records agree at rtol=1e-8/atol=1e-10. Maximum observed quadrature refinement difference is about 1e-13. Synthetic tests check random DAGs against direct enumeration and a billion-path network. Actual maximum counts are approximately 2.28e11 parent paths and 1.04e13 child paths.

**Interpretation remains important:** shared-edge path sums still count shared edges on multiple paths. Path-derived equivalent width, elongation and dominance can become extreme in large compound domains. Eliminating the computational cutoff does not turn these into independent-channel measurements. Log-transformed candidates address scale statistically, not this physical interpretation.

## Expanded real-case validation

61 candidates combine 11 feature sets with Ridge regularization, quadratic Ridge, two boosted-tree complexities and Extra Trees. Candidate sets include omega, separate components, size/slope, selected six, changes/context, all descriptors, logarithmic transforms and the existing single-product axes. `search_plan.json` is the complete specification.

Outer validation holds out one **network example** at a time. Four grouped inner folds in the remaining examples select by equal-example MAE. No outer response enters selection. Example grouping does not certify basin independence: examples may be geographically related. These rivers have been examined previously, so this remains exploratory validation, not an untouched external test.

'''+table(summary.round(4))+'''

The nested search yields R² **0.221 for percentage lag** and **0.472 for absolute timing**. Its MAEs are 0.232 percentage points and 0.135 h (8.1 minutes). Those improve on no change by 22.6% and 41.9%. For percentage lag, the simple separate-component benchmark remains stronger (R² .428, MAE .212). For absolute timing, the fixed selected-six boosted model has higher R² (.554) but slightly worse MAE (.147 h). Inner equal-example MAE selection need not optimize pooled outer R²; report both, rather than choosing whichever score looks best.

The largest held-out errors are complete-compound collapses in examples 15 and 22, recovered by removing the old path cutoff. For example, a −22.31% response in example 15 is predicted as −3.31%; a −8.57 h response in example 22 is predicted as −1.95 h. These are retained, not removed to improve scores. `largest_errors.csv` identifies the cases.

The earlier .68 percentage-lag score belonged to the restricted cohort. The full-population search does not justify carrying that headline forward. Fixed-candidate rankings in `scores.csv` are descriptive candidate comparisons; selecting a winner from them after inspecting outer performance incurs selection optimism. The nested result is the assessment of this run's selection process, not necessarily the strongest single fixed model.

'''+table(rivers.round(4))+'''

## Synthetic selection and transfer

The same 61 candidates are selected using four geometry-grouped folds **only within the original synthetic development set** (9,851 unique comparisons). Models are then frozen and evaluated on the original 3,362-row holdout, the existing 917 novel extension comparisons (63 original anchors excluded), and all 1,652 real comparisons. Extension outlet peaks were reread and interpolated with the same guards; per-pair forcing clocks agree. Extension and real outcomes never select the synthetic model. These are previously explored transfer populations, so do not label them untouched confirmatory tests.

'''+table(ts.round(4))+'''

The synthetic-selected percentage model is Extra Trees with selected-six/change/context features; its original holdout R² is .9966, but extension R² is −.929 and real transfer R² −145.98. Strong interpolation within the original design still does not establish transfer.

The synthetic-selected absolute model uses boosted trees on logged geometry. Its R² is .9824 on the original holdout and .583 on the novel extension. Real transfer R² is only .133, and MAE .352 h is worse than predicting zero change (.232 h). Positive R² alone therefore does not establish useful real transfer.

## Current conclusion and next decision

Keep omega/separate components as readable percentage-lag benchmarks. Absolute timing benefits from dimensional geometry and is more promising on the wider synthetic extension, but synthetic-trained absolute predictions still do not beat no change in real cases. The larger full-population search is now complete; it did not reveal a universal predictor or a consistent benefit of more model complexity.

Next inspect the full-population residuals and category/example dependence before choosing a compact final predictor. The physical applicability issue in the separate hydraulic-domain notebook is unchanged by statistical model performance. Shape model search remains paused by user decision.

Reproduce `full_population_features.py`, `full_population_lag_search.py`, `full_population_synthetic_transfer.py`, then this report script. Outputs, fold choices and candidate scores are retained here. Model fits use one CPU thread; no hydraulic settings, simulation inputs or legacy results were overwritten.
'''
    (OUT/'README.md').write_text(text);print(summary.to_string(index=False))

if __name__=='__main__':main()
