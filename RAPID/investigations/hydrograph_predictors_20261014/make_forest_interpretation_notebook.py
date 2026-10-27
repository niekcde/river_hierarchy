"""Create a compact review notebook from the saved forest interpretation artifacts."""
import nbformat as n
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
b=n.v4.new_notebook();b.metadata['kernelspec']={'display_name':'Python 3','language':'python','name':'python3'}
b.cells=[n.v4.new_markdown_cell('''# What does the forest use?

Full-centroid target, one-stage forest, one-minute relative-error tuning. Each example is held out in turn. The exact saved models are reconstructed and their predictions verified before interpretation. No predictor selection or resistance fits are performed by this notebook.

**Permutation:** 20 shuffles within each held-out network. Positive values mean that disrupting a variable worsens predictions. Related variables are shuffled jointly with the same row permutation. Scores are averaged equally across networks; the R² decrease uses the fixed global weighted variance. This is not the average of individual network R² values. Group importance is not the sum of individual importances.

**SHAP:** exact, tree-path-dependent TreeSHAP on the held-out predictions. Values are in minutes relative to each model's training-path expectation. Negative values move predictions toward advances; positive values move them toward delays. Absolute SHAP measures how much the model uses a variable, not whether using it helps prediction. These are neither causal effects nor a refitted variable-removal experiment. [TreeExplainer documentation](https://shap.readthedocs.io/en/latest/generated/shap.TreeExplainer.html).

Within-network permutation cannot assess information that is constant within that network. Both methods can divide importance among correlated variables. The interaction plots show how the fitted model behaves, not new physical laws.'''),
n.v4.new_code_cell('''from pathlib import Path
import numpy as np
import pandas as pd
import plotly.express as px
ROOT=Path.cwd()
while not (ROOT/'RAPID').exists():
    if ROOT==ROOT.parent: raise RuntimeError('Run from inside river-hierarchy')
    ROOT=ROOT.parent
OUT=ROOT/'RAPID/outputs/hydrograph_forest_interpretation_20261019'
p=pd.read_csv(OUT/'permutation_summary.csv')
s=pd.read_csv(OUT/'shap_summary.csv')
shap=pd.read_parquet(OUT/'shap_values.parquet')
interactions=pd.read_parquet(OUT/'shap_interactions.parquet')
by=pd.read_csv(OUT/'permutation_by_network.csv',dtype={'example':str})
print('Individual predictors; equal-network means')
display(p[~p.group.str.endswith('_group')].sort_values('relative_increase_pp',ascending=False))
display(s)
print('Joint permutations of related predictors')
display(p[p.group.str.endswith('_group')].sort_values('relative_increase_pp',ascending=False))'''),
n.v4.new_markdown_cell('''## Symbols and domains

All D values use the agreed outer comparison domain. R denotes child/parent. Model inputs use ln(1+R), while SHAP dependence plots below display the untransformed ratios.

| Symbol | Quantity |
|---|---|
| R_muL | Mean travelled-length ratio |
| muL_Dp / 1 km | Parent domain mean travelled length, divided by 1 km |
| R_sigmaL | Route-length SD ratio |
| R_W | Length-weighted mean-width ratio |
| R_CVW | Width-CV ratio |
| R_H | Route-entropy ratio |
| d_Dp | Parent mean split dominance |
| r_in | Domain exposure |

The grouped permutations cover length (ratio, SD ratio, parent size), width (mean ratio, CV ratio, dominance), length plus width ratios, and width plus entropy. The matrix below helps identify shared information. Exposure differences at machine precision are listed explicitly; the original preprocessing has been retained to reproduce the evaluated model.'''),
n.v4.new_code_cell('''fig=px.bar(p[~p.group.str.endswith('_group')].sort_values('relative_increase_pp'),x='relative_increase_pp',y='group',orientation='h',labels={'relative_increase_pp':'Increase in floored relative error (percentage points)','group':'Predictor'})
fig.show()
fig=px.imshow(by.pivot(index='example',columns='group',values='delta_relative_pp'),color_continuous_scale='RdBu_r',color_continuous_midpoint=0,labels={'color':'Relative-error increase (pp)'})
fig.show()
corr=pd.read_csv(OUT/'predictor_spearman.csv',index_col=0)
px.imshow(corr,color_continuous_scale='RdBu_r',zmin=-1,zmax=1,title='Pooled Spearman correlations; descriptive').show()
support=pd.read_csv(OUT/'predictor_support.csv',dtype={'example':str})
display(support.groupby(['symbol','numerically_constant']).size().rename('network_count').reset_index())'''),
n.v4.new_markdown_cell('''## SHAP direction and size interactions

Each point is explained by the forest that did not train on its network. Plots show every comparison, so dense networks occupy more visual space; the summary table above gives each network equal weight. Fold-specific baselines differ, so these plots should not be read as one universal response curve. The pair interaction shown is twice the off-diagonal SHAP interaction entry.'''),
n.v4.new_code_cell('''size=shap[shap.symbol=='muL_Dp / 1 km'][['example','state','value']].rename(columns={'value':'parent_size_km'})
for symbol in ['R_muL','R_W','R_H','muL_Dp / 1 km']:
    a=shap[shap.symbol==symbol].merge(size,on=['example','state'],validate='one_to_one')
    px.scatter(a,x='value',y='shap_minutes',color='parent_size_km',hover_data=['example','state','edit_type'],title=symbol,labels={'value':symbol,'shap_minutes':'SHAP contribution (minutes)'}).show()
for pair,a in interactions.groupby('pair'):
    px.scatter(a,x='ratio',y='interaction_minutes',color='domain_size',hover_data=['example','state'],title=pair).show()
print('Extreme compound: attribution of its predicted shift, not of its prediction error')
a=shap[(shap.example=='37')&(shap.state=='CB001_bubble_3')]
display(a[['symbol','value','shap_minutes','base_minutes','predicted_minutes']])'''),
n.v4.new_markdown_cell('''## Focused resistance comparison

Three panels: current predictors; current + −log10(λK); and resistance replacing mean-width ratio. Resistance uses the existing outer-domain definition. The two new panels select their forest settings independently inside the same grouped inner folds, using the same one-minute relative-error objective. The baseline reuses the completed forest predictions. Both all-example and without-54 scenarios are planned.

The comparison is prepared, but requires the separate `--run-resistance` command. A result marked `complete=False` must not be compared with a complete panel. Peak scores reuse centroid predictions and are not separate peak-trained models.'''),
n.v4.new_code_cell('''scores=pd.read_csv(OUT/'resistance_scores.csv')
display(scores[(scores.scenario=='all')&(~scores.exclude_extreme)&(scores.subset=='all')])
print('Planned fits:',pd.read_csv(OUT/'resistance_run_plan.csv').fits.sum())''')]
n.write(b,ROOT/'RAPID/notebooks/hydrograph_forest_interpretation.ipynb')
print(ROOT/'RAPID/notebooks/hydrograph_forest_interpretation.ipynb')
