"""Readable report for the nested searches; no response-driven refitting."""
from pathlib import Path
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/river-similarity-mpl')
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT=Path(__file__).resolve().parent/'results'
s=pd.read_csv(OUT/'outer_summary.csv')
sizes=pd.read_csv(OUT/'model_sizes.csv')
freq=pd.read_csv(OUT/'selection_frequency.csv')
pooled=s[s.scope.eq('pooled')].merge(sizes,on=['variant','method','response'],how='left')
variants=['G_alone','unconstrained_with_G','coverage_with_G','unconstrained_without_G','coverage_without_G']
labels={'G_alone':'ΔG alone','unconstrained_with_G':'Unconstrained · ΔG allowed',
        'coverage_with_G':'Four groups · ΔG allowed','unconstrained_without_G':'Unconstrained · without ΔG',
        'coverage_without_G':'Four groups · without ΔG'}
methods=['variable_manning','fixed_manning','fixed_based']
responses=['centroid_shift_minutes','peak_shift_minutes']
columns=[(m,r) for m in methods for r in responses]
tab=pooled.pivot(index='variant',columns=['method','response'],values='r2').reindex(index=variants,columns=pd.MultiIndex.from_tuples(columns))
fig,ax=plt.subplots(figsize=(12,4.7),constrained_layout=True)
im=ax.imshow(tab,vmin=0,vmax=1,cmap='viridis',aspect='auto')
ax.set(yticks=range(5),yticklabels=[labels[v] for v in variants],xticks=range(6),
       xticklabels=[m.replace('_',' ')+'\n'+('Centroid' if r.startswith('centroid') else 'Peak') for m,r in columns],
       title='Nested case-group evaluation · pooled outer-fold R²\nFixed quadratic ridge; synthetic-grid interpolation')
for i in range(5):
    for j in range(6):
        v=tab.iloc[i,j];ax.text(j,i,f'{v:.4f}',ha='center',va='center',color='white' if v<.5 else 'black')
fig.colorbar(im,ax=ax,label='Outer-fold R²');fig.savefig(OUT/'nested_comparison.png',dpi=180);plt.close(fig)
lines=['# Forward selection up to ten geometric predictors','',
'Reference: SYN-FORWARD-SELECTION-2026-11-02. Status: completed internal synthetic model-selection experiment; not a final real-network predictor selection.','',
'## Population and coverage','',
'The accepted standalone cutoff leaves 68 candidates from the 100-class pool. It eliminates all six pure topology candidates. Those six are restored as explicit coverage reserves for BOTH searches, giving the same 74-candidate pool. Without Manning ΔG there are 73. The former 100-class inventory remains archived, not deleted. candidate_pool.csv records each variable, concept, coverage category and restoration status.','',
'Coverage requires at least one explicit length, width, slope and topology representative. Combined variables do not satisfy multiple categories. In particular ΔG, area, elongation, width entropy, head loss and branching density are not allowed to stand in for all four principles. Pure graph-architecture descriptors qualify as topology; actual reach-slope descriptors qualify as slope. Distance distribution descriptors qualify as length even when allocation weights also use width.','',
'## Fixed model and search','',
'The model is quadratic ridge regression: intercept, main effects, squared terms and all pair products among the selected variables. Ten predictors therefore imply 65 expanded non-intercept terms, not ten coefficients. This explicitly permits width–length and other pair interactions. Expanded columns are centred/scaled using training moments; penalty is 1e-4 times training row count. Numeric global scaling is purely for conditioning and cancels under training-moment scaling. No model-family or regularisation search is performed; this is not RF optimisation.','',
'Unconstrained greedy forward selection evaluates every remaining addition up to ten inputs. The coverage version chooses one missing principle at each of its first four steps, then performs up to two same-principle coordinate-swap passes on that seed before continuing unrestricted additions. This avoids forcing missing principles into the final slots, but remains a heuristic rather than an exhaustive constrained optimum. The smallest path model within 0.005 absolute INNER R² of the best path score is selected; this prespecified parsimony tolerance is a practical choice, not a significance threshold. Coverage models require at least four predictors.','',
'There are three outer case-group folds and three inner folds. All seven collapse entries of each complete parameter combination stay together; folds are shared across hydraulic methods, responses and search variants. Selection and size choice use inner predictions only. The selected model is refit on outer training data and evaluated on the untouched outer fold for that search. The objective is pooled R² over all seven types, rather than the previous screen’s average of separately fitted within-type models. Outer per-type metrics are also exported, and unresolved near-null strata retain MAE but do not receive R².','',
'Both searches run with and without ΔG for each of the three depth approaches and both responses. A fixed ΔG-alone quadratic baseline is evaluated on the same outer splits. Sufficient-statistic caching computes exact ridge solutions for this fixed basis; it is not training-data subsampling. Numerical SSE was verified against direct predictions, and the solution against scikit-learn Ridge in every outer split.','',
'## Important interpretation limits','',
'These are nested subset-selection scores CONDITIONAL on an already inspected/frozen candidate inventory. The earlier 0.05 filter and similarity decisions used the full synthetic population; they were not rebuilt inside these outer folds. Therefore this does not constitute unbiased validation of the entire discovery process. The synthetic design has been repeatedly inspected. Cases with irrelevant control differences can also repeat geometry across folds. Scores describe interpolation within this factorial design, not unseen network types, new control ranges, or observed rivers. Independent real-network evaluation remains necessary.','',
'Pure topology variables can act as proxies for the seven synthetic treatment geometries. Their selection does not establish that the same topology mechanism transfers to more complex real cases. Coverage is an imposed interpretability constraint, not evidence that all four groups are independently necessary. Alternative greedy paths and model families may give different selections.','',
'## Results','']
for method in methods:
    lines += [f'### {method}','', '| Search | Response | Outer R² | MAE (min) | Selected inputs across folds |','|---|---|---:|---:|---|']
    for variant in variants:
        for response in responses:
            r=pooled[(pooled.method==method)&(pooled.variant==variant)&(pooled.response==response)].iloc[0]
            n='1' if variant=='G_alone' else str(int(r['min'])) if r['min']==r['max'] else f"{int(r['min'])}–{int(r['max'])}"
            lines.append(f"| {labels[variant]} | {'Centroid' if response.startswith('centroid') else 'Peak'} | {r.r2:.6f} | {r.mae_minutes:.3f} | {n} |")
    lines.append('')
lines += ['## Selected inputs: variable Manning','',
'Counts below are selection frequencies out of three outer training folds. They are not feature importance or confidence intervals. No single final full-data model has been fitted or frozen.','']
for variant in variants[1:]:
    for response in responses:
        lines += [f"### {labels[variant]} — {'centroid' if response.startswith('centroid') else 'peak'}",'']
        z=freq[(freq.variant==variant)&(freq.response==response)&(freq.method=='variable_manning')].sort_values(['selected_folds','variable'],ascending=[False,True])
        for r in z.itertuples():lines.append(f'- `{r.variable}` — {r.category}; {r.selected_folds}/3 folds.')
        lines.append('')
lines += ['## Main findings','',
'- Under variable Manning, ΔG alone explains centroid shift almost exactly. Without ΔG, a stable three-input set (mean route-length change, area change, and inverse-root slope) also performs strongly. Its pair products are part of the model; three inputs do not mean a simple three-term linear law.',
'- Peak shift requires a richer combination. Six unconstrained inputs reach high internal held-out scores; the four-principle constraint typically needs more inputs for similar performance.',
'- The forced-coverage search achieves similar scores in many settings, but that does not demonstrate unique added information from every category. For fixed Manning centroid, the parsimony tolerance accepts a four-input model without ΔG even when ΔG is allowed; the resulting score is slightly below the almost-exact ΔG baseline.',
'- BASED peak shift is harder and reaches the ten-input cap. It is not justified to claim that ten is an optimal universal size.',
'- No final model or universal physical law is selected here. Next review the stable small combinations, test transfer and coverage on real networks, and compare model families if the goal is predictive optimisation.','',
'## Outputs','',
'nested_comparison.png; candidate_pool.csv; case_folds.csv; search_paths.csv; candidate_trials.csv (forward-addition trials; seed-swap trials are evaluated but not individually exported); selected_variables_by_fold.csv; selection_frequency.csv; model_sizes.csv; outer_fold_scores.csv; outer_summary.csv; outer_predictions.parquet; selected_fold_models.json; numerical_checks.csv; provenance.json. The model JSON records exact coefficients and numeric scales for the selected outer-fold models.']
(OUT/'report.md').write_text('\n'.join(lines)+'\n')
assert len(pooled)==30
print(pooled[pooled.method.eq('variable_manning')][['variant','response','r2','mae_minutes','min','max']].to_string(index=False))
