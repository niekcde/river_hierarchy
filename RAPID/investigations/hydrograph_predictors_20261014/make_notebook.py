"""Create a separate current-response investigation; preserve existing notebooks."""
import json
import nbformat as nb
from build import ROOT, OUT

cells=[]
def md(s):cells.append(nb.v4.new_markdown_cell(s))
def code(s):cells.append(nb.v4.new_code_cell(s))

md('''# Geometry predictors linked to six finalized hydrograph responses

14 October 2026. **Peak-time shift is the leading target; peak-excess change is the second primary target.** Centroid, duration, volume and asymmetry help distinguish translation, spreading and redistribution. This notebook uses the completed reruns and fixed-baseline **0.5% shared window**, with 1% and 2% sensitivity.

This continues the earlier predictor investigation without replacing its notebook or the user-edited metric examples. The old five-network response tables and hydraulic predictors are not imported. Physical geometry is recalculated from the parameter files whose hashes were recorded in the completed-run plan. No routing, calibration or response definitions were changed.

The results below are **exploratory fixed-panel benchmarks**, not a final selected predictor model. Broad candidate associations are for review; they did not select the fitted columns.''')
code('''from pathlib import Path
import sys,json
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots
from IPython.display import display
ROOT=next(p for p in [Path.cwd(),*Path.cwd().parents] if (p/'RAPID/src/rapid_tools').is_dir())
OUT=ROOT/'RAPID/outputs/hydrograph_predictors_20261014'
sys.path.insert(0,str(ROOT/'RAPID/src'))
from rapid_tools.hydrograph_response import load, METRICS, threshold_figure
pio.renderers.default='plotly_mimetype'
states,responses=load('0.5%')
d=pd.read_parquet(OUT/'linked_responses.parquet')
manifest=json.loads((OUT/'manifest.json').read_text())
design=json.loads((OUT/'validation_design.json').read_text())
scores=pd.read_csv(OUT/'validation_scores.csv')
associations=pd.read_csv(OUT/'individual_associations.csv',dtype={'example':str})
audit=pd.read_csv(OUT/'local_domain_audit.csv',dtype={'example':str})
assert not d.duplicated(['example','state']).any()
assert len(d)==len(responses)==2187
display(pd.Series({k:manifest[k] for k in ['comparisons','examples','matched','unmatched','historical_unmatched_aliases']}))''')
md(r'''## 1. The six responses remain fixed

For each comparison, use one scalar per response. Peak time is the **first saved global maximum in the common analysis window**; peak excess is that maximum minus the shared baseline. Neither target changes to a matched-crest definition in example 54.

Signed changes are child minus parent. Percentages are $100(X_c/X_p-1)$. Timing units are hours; percentage errors are percentage points. No interpolation of a different peak time is introduced.

The baseline is the fixed minimum across each example's completed states after input start and initial/final equilibria. The window starts at the earliest sustained departure across states and ends at the latest return to final equilibrium within the chosen tolerance. Endpoint excess is retained. Distribution metrics describe this finite window, including the transition to a higher final plateau in 54.''')
code('''response_contract=pd.DataFrame([
 ('Timing and duration','peak_shift_hours','PRIMARY: time of maximum discharge','Length change, width change, resistance; crest dominance can also matter'),
 ('Timing and duration','centroid_shift_hours','Supporting: excess-volume centre','Allocation-weighted distance; exposure; retained late volume'),
 ('Timing and duration','duration80_change_pct','Supporting: central 80% volume duration','Route-length variability, branch balance and repeated splitting'),
 ('Magnitude','peak_excess_change_pct','PRIMARY: maximum above shared baseline','Width/resistance changes, route variability, allocation balance'),
 ('Magnitude','volume_change_pct','Supporting: finite-window excess volume','Check truncation/equilibrium transition before interpreting geometry'),
 ('Temporal shape','asymmetry_change','Supporting: quantile asymmetry','Unequal routes and balance; finite-window effects'),
],columns=['group','response','role','candidate physical connections — hypotheses'])
display(response_contract)
display(d[list(METRICS)].describe().T[['count','min','50%','max']])''')
md('''## 2. Join and domain audit before interpreting predictors

Canonical keys are **example + state**, resolved through the completed-run plan and response aliases. Historical aliases are cross-checked against canonical fingerprints; aliases never create extra model rows. All 2,187 responses join to current whole-network geometry, including 15 single-centerline comparisons. Historical base-state aliases are labeled `parent_only`, not unmatched children.

Local-domain measurements require matching physical endpoints, a closed single-source/sink DAG, no external crossings through internal nodes, and the same parent/child anchors. Failure is retained in the audit rather than replaced with a guessed local domain.

**Domain type matters:** `closed_unit` and `closed_compound_unit` describe closed unit comparisons; `containing_compound_context` describes the containing compound before/after a member edit, not necessarily the edited footprint. Whole-network summaries are separately prefixed `network_`. Entry exposure uses the whole parent network; allocation inside a local domain is conditional on entering that domain.''')
code('''display(pd.crosstab(audit.example,audit.status))
display(audit.groupby(['status','domain_type'],dropna=False).size().rename('comparisons'))
failures=audit[audit.status.eq('unavailable')].copy()
failures['reason_group']=failures.reason.str.replace(r'No saved local domain membership for .*','No saved local domain membership',regex=True)
display(failures.reason_group.value_counts())
print('Complete records:',OUT/'response_predictor_join_audit.csv')
print('Unmatched records (empty when successful):',OUT/'unmatched_records.csv')
print('Historical alias audit:',OUT/'historical_alias_join_audit.csv')''')
md(r'''## 3. Which representations enter this first comparison?

The full descriptive table retains parent and child values, differences and defined ratios. Fitted panels deliberately use a small subset. Positive length/width measures can use ratios; signed or bounded measures use differences. A zero count ratio cannot distinguish 1→0 from 4→0, so expected-split change remains informative. Conditional split means are undefined for a child with no splits; they are stored as NaN, not zero. The fitted local width panel uses **parent** dominance.

The local length panel uses entry-exposed mean-length change normalized by parent network mean route length, parent domain/network length ratio, similarly exposed route-SD change, and entry exposure. Width adds the child/parent mean-width ratio and parent dominance. Topology adds change in expected bifurcations. These are prescribed panels, not winners selected from the association plots.

**Resistance remains an important combined candidate.** It is not imported from older tables because its geometry law/provenance must be consistent. Under wide Chezy and $h=aW^\alpha$, geometry weights are $L/W^{2+3\alpha}$. A new cohort-wide equivalent-resistance calculation, with explicit $\alpha=0.65$ versus $1$ sensitivity and valid domains, is still needed. Harmonic width is not a substitute. The present panels therefore do not settle whether resistance is the better predictor.

P10/P90, junction excess and density remain secondary. Route variance and branch-contrast variance are the same ingredient; route entropy mixes repeated splitting with balance. Dominance, normalized local entropy and effective branch count should first be compared as alternatives. No hydraulic celerity, depth, Manning n, beta, transit time, output-crest flag or hydrograph response is used as a predictor.''')
code('''panel_rows=[]
for scope,key in [('network','network_panels'),('domain','domain_panels')]:
    for panel,features in design[key].items():
        panel_rows.append(dict(scope=scope,panel=panel,predictors='; '.join(features)))
display(pd.DataFrame(panel_rows))
display(pd.read_csv(OUT/'redundancy.csv').sort_values('spearman',key=abs,ascending=False).head(20))''')
md('''## 4. Individual associations with all six responses

Use these to inspect mechanisms, nonlinearities and network dependence—not to claim causal effects or select predictors before reusing the validation score. Pooled rank association can conceal between-example differences. The next table retains each example separately; constants and insufficient data yield NaN. Children sharing a parent are not independent replicates, so no naive significance tests are attached.''')
code('''SCOPE='domain'  # 'network' for whole-network summaries
TARGET='peak_shift_hours'
a=associations[(associations.scope==SCOPE)&associations.example.eq('pooled')]
matrix=a.pivot(index='feature',columns='response',values='spearman').reindex(columns=list(METRICS))
fig=px.imshow(matrix,zmin=-1,zmax=1,color_continuous_scale='RdBu_r',aspect='auto',
              labels=dict(color='Spearman'),title=f'{SCOPE}: exploratory pooled associations with six responses')
fig.update_layout(height=570);fig.show()
within=associations[(associations.scope==SCOPE)&associations.response.eq(TARGET)&~associations.example.eq('pooled')]
display(within.pivot(index='feature',columns='example',values='spearman').round(3))''')
code('''FEATURE='exposed_length_change'
plot=d[d.comparison.eq('partial collapse')&d.status.eq('validated')].copy()
fig=make_subplots(rows=1,cols=2,subplot_titles=['Peak-time shift — primary','Peak-excess change — primary'])
colors=px.colors.qualitative.Dark24
for i,(ex,g) in enumerate(plot.groupby('example')):
    for col,target in enumerate(['peak_shift_hours','peak_excess_change_pct'],start=1):
        fig.add_trace(go.Scatter(x=g[FEATURE],y=g[target],mode='markers',name=ex,legendgroup=ex,showlegend=col==1,
            marker=dict(size=5,color=colors[i%len(colors)],opacity=.65),customdata=g[['state','domain_type']].to_numpy(),
            hovertemplate='%{customdata[0]}<br>%{customdata[1]}<br>x=%{x}<br>y=%{y}<extra>'+ex+'</extra>'),row=1,col=col)
        fig.update_xaxes(title_text=FEATURE,row=1,col=col)
fig.update_yaxes(title_text='hours',row=1,col=1);fig.update_yaxes(title_text='%',row=1,col=2)
fig.update_layout(template='plotly_white',height=480);fig.show()''')
md('''## 5. Prediction: hold out entire examples

Each fold holds out all children of one example. Training-only median imputation, constant removal and standardization precede fixed Ridge regression (alpha=10). Training examples have equal total weight. Zero change and the training-example-weighted mean are explicit baselines. No feature selection or hyperparameter search uses the held-out examples.

Results report both pair-weighted and equal-example errors. Prefer the latter for transfer: a large network should not dominate just because it supplies more collapses. R² and MAE answer different questions; a model can improve squared error while losing to zero change on typical absolute error.

`partial` is the main cohort. `including_centerlines` adds the 15 global collapses and tests their leverage. Every cohort is refitted with 54 excluded from both training and evaluation. `network_on_domain` restricts the network panels to exactly the available local-domain records for fair scope comparisons. Local partial-collapse validation covers only the examples with validated local records; missing examples are not claimed as tested.

These panels were fixed before viewing current scores. Further decisions based on these scores make this cohort exploratory; subsequent variable selection needs nested example-level validation. Geometric lengths alone provide no physical velocity scale, so predicting hours remains conditional on the event/routing regimes represented here.''')
code('''COHORT='partial'  # also 'including_centerlines'
view=scores[scores.weighting.eq('equal_example')&scores.run.str.startswith(COHORT+'/')&scores.response.isin(['peak_shift_hours','peak_excess_change_pct'])].copy()
view['mae_minutes_for_timing']=np.where(view.response.eq('peak_shift_hours'),60*view.mae,np.nan)
display(view[['run','panel','response','n','examples','mae','mae_minutes_for_timing','rmse','r2']].round(4))
# All six targets remain available; avoid choosing a different target to rescue a weak primary result.
display(scores[scores.weighting.eq('equal_example')&scores.run.eq(COHORT+'/domain/all')][['panel','response','mae','r2']].round(4))''')
code('''RUN='partial/domain/all'
PANEL='length_width_exposure'
TARGET='peak_shift_hours'
pred=pd.read_parquet(OUT/'held_example_predictions.parquet')
v=pred[(pred.run==RUN)&(pred.panel==PANEL)&(pred.response==TARGET)]
fig=px.scatter(v,x='observed',y='predicted',color='example',hover_data=['state'],
    title=f'Unseen-example predictions: {TARGET} — {RUN} / {PANEL}')
lo=min(v.observed.min(),v.predicted.min());hi=max(v.observed.max(),v.predicted.max())
fig.add_shape(type='line',x0=lo,x1=hi,y0=lo,y1=hi,line=dict(dash='dash',color='grey'))
fig.update_layout(template='plotly_white',height=520);fig.show()
display(scores[(scores.run==RUN)&(scores.panel==PANEL)&(scores.response==TARGET)&scores.weighting.str.startswith('example_')][['weighting','n','mae','rmse','r2']].round(4))''')
md('''## 6. Example 54: global maxima and a separate diagnostic

The parent has two close-in-height crests around 13:43:30 and 22:45:30 UTC on 22 June 2023, approximately 1,393.39 and 1,396.39 m³/s. The small dip after the first is not evidence that it is numerically absent; a large prominence cutoff would erase it.

**Provisional diagnostic:** find local maxima separated by at least two hours and within 1% of the hydrograph's maximum excess. Match each of the two parent crests to exactly one candidate within a specified time radius, require chronological order, and require the child's global maximum to belong to the matched pair. Compare which ordered crest is dominant. No prominence cutoff is used. Default matching radius: 4.5 h; sensitivity: 2 h and 6 h.

The flag is three-state: resolved switch, resolved no switch, or unresolved. It is a heuristic based on temporal correspondence, not proof that a water parcel or the same physical wave was tracked. A child-derived flag is **diagnostic only**, never a predictor. The 1% here is a crest-height diagnostic tolerance, unrelated to the analysis-window tolerance.

All 81 partial-collapse children resolve as no switch under all three radii. The single centerline is unresolved at 2 and 4.5 h but classified as a switch at 6 h. Its large peak shift therefore needs qualified interpretation, not a forced binary label. Neither this flag nor matched-crest timing replaces any of the six response values.

54 ends above its baseline. Centroid, duration, asymmetry and volume describe the finite-window excess including that transition, not an isolated pulse returning to its starting discharge.''')
code('''crest=pd.read_csv(OUT/'crest_diagnostic_54.csv',dtype={'example':str})
display(pd.read_csv(OUT/'parent_crests_54.csv'))
display(crest.groupby(['matching_radius_hours','status','crest_switch_flag'],dropna=False).size().rename('comparisons'))
display(crest[crest.state.eq('fully_collapsed')])
display(d[d.example.eq('54')][['state',*METRICS]].sort_values('peak_shift_hours').head(8))
threshold_figure(states,'54','fully_collapsed',view='peak',gauges=False).show()''')
md('''## 7. Window sensitivity

The baseline is held fixed. Both global-peak targets are identical across the three finalized windows. The four distribution responses can change; inspect maximum as well as median differences, because a small median can conceal an influential centerline or equilibrium transition.

`sensitivity_validation_scores.csv` evaluates the same held-example predictions trained on 0.5% responses against 1% and 2% labels. It is a label-sensitivity check, not a separately tuned model.''')
code('''sensitivity=pd.read_csv(OUT/'window_sensitivity.csv')
display(sensitivity[sensitivity.scope.eq('all')].round(6))
display(sensitivity[sensitivity.scope.eq('54')].round(6))''')
md('''## 8. What this pass supports—and what comes next

1. Review peak timing first, then peak excess, using the same geometry panels. Use centroid/duration/asymmetry to interpret cases where peak timing alone is misleading; none is an input to the geometric predictor.
2. Resolve domain coverage before final local-variable selection: missing membership, changed endpoints and anchor correspondence are different problems. Examples 13 and 15 currently have no validated partial-collapse domains. The 1,078 containing-compound contexts must remain distinct from the 491 closed-unit domains.
3. Rebuild equivalent resistance under the explicit width–depth laws on validated domains, then compare its compact panel on the same cohort. Do not substitute historical resistance values or calibrated hydraulic timing.
4. Compare alternative representations within length, width balance and topology, with selection inside training examples. Keep a variable for a clear mechanism and repeatable incremental value, not just a pooled correlation. Investigate residual patterns without adding child-derived diagnostics as predictors.
5. Examine weak amplitude predictions explicitly. A small positive R² is not sufficient if zero change has a lower MAE. Sparse near-zero responses and a few large cases can reward different loss functions.

Scripts in `RAPID/investigations/hydrograph_predictors_20261014/` reproduce extraction, audits and validation. Inputs and definitions are hashed in the output manifest. Existing response and worked-example notebooks are preserved.''')
md('''## 9. Read the cohort comparison before quoting a score

For partial collapses, the whole-network length panel covers all 15 examples (2,172 comparisons). The local length/width/exposure panel covers 13 examples (1,569 comparisons); these two scores are not a head-to-head comparison. Use `network_on_domain` for that comparison.

The local panel's partial-collapse timing R² is approximately 0.50, with MAE about 6.2 minutes. Excluding 54 leaves R² approximately 0.50. This is a limited-cohort benchmark, not evidence for a universal predictor.

**Including single centerlines causes the local panel to fail badly** (negative R²), even after excluding 54. The response scale and geometric changes are much larger, and incomplete local coverage gives examples 13 and 15 only one available observation each—the centerline—which receives an entire example's training weight. Inspect this coverage and leverage before drawing conclusions about global-collapse physics. A simple pooled linear local model is not justified across both regimes by these results.

The network panel remains available on all records, with full-cohort errors reported below. Peak-excess models have positive R² in some panels but worse MAE than zero change: predictive value is not established uniformly across losses.''')
code('''summary=scores[scores.weighting.eq('equal_example')&scores.response.eq('peak_shift_hours')&(
    (scores.run.str.contains('/domain/')&scores.panel.eq('length_width_exposure'))|
    (scores.run.str.contains('/network/')&scores.panel.eq('length')))].copy()
summary['mae_minutes']=60*summary.mae
display(summary[['run','panel','n','examples','mae_minutes','r2']].round(4))''')
n=nb.v4.new_notebook(cells=cells,metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'}})
path=ROOT/'RAPID/notebooks/hydrograph_predictor_response_investigation.ipynb'
nb.write(n,path)
print(path)
