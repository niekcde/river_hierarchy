"""Promote the completed grid to four main notebooks; archive previous notebooks."""
from pathlib import Path
import shutil
import nbformat as nbf
ROOT=Path(__file__).resolve().parents[2];NB=ROOT/'synthetic_experiments/notebooks'
ARCHIVE=NB/'legacy/pre_full_depth_grid_20261028';ARCHIVE.mkdir(parents=True,exist_ok=True)
def md(s):return nbf.v4.new_markdown_cell(s)
def code(s):return nbf.v4.new_code_cell(s)
intro='''**SYN-FULL-DEPTH-GRID-2026-10-28 · Current completed analysis.**
6,561 geometries × 3 hydraulic methods × 7 parent–child treatments = **137,781 comparisons**.
All analyses below read saved results; **Run All does not rerun routing**.
Previous versions, including the old fixed-X analyses and pilot figures, are preserved in
[the dated archive](legacy/pre_full_depth_grid_20261028/). Their fitted scores are not results for this grid.

The results are conditional on the tested factor ranges, 2:1 upstream pulse, fixed width-based allocation,
wide-rectangular hydraulics and routing implementations. The three depth methods are separate strata.
'''
setup='''from pathlib import Path
import sys, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from IPython.display import display, Markdown
from IPython.display import Image
from io import BytesIO
def display_figure(fig):
    buffer = BytesIO()
    fig.savefig(buffer, format='png', dpi=140, bbox_inches='tight')
    display(Image(data=buffer.getvalue()))
ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p/'synthetic_experiments').is_dir())
for p in (ROOT, ROOT/'RAPID/src', ROOT/'synthetic_experiments/src'):
    if str(p) not in sys.path: sys.path.insert(0, str(p))
from synthetic_experiments.full_grid_review import (load_analysis, RESPONSES, METHODS, LABELS,
    FACTOR_LABELS, TREATMENTS, summary_figure, heatmaps, agreement_figure, physical_figure)
OUTPUT = Path('/Volumes/SyntheticResults20261027')  # mounted container physically on the PhD drive
manifest, results, ANALYSIS = load_analysis(OUTPUT)
FIGURES = ANALYSIS/'figures'
FIGURES.mkdir(exist_ok=True)
def table(name): return pd.read_csv(ANALYSIS/(name+'.csv'))
def show(fig, name):
    if name.endswith(('_main_effects','_total_effects','_28_pairs','_signed_contrasts')):
        labels=['Simple\\nfull','Nested\\ninner','Nested\\nisland','Nested\\nfull','Cross\\nup','Cross\\ndown','Cross\\nfull']
        for ax in fig.axes:
            if len(ax.get_xticklabels())==7: ax.set_xticks(range(7),labels=labels)
    fig.savefig(FIGURES/(name+'.png'), dpi=150, bbox_inches='tight')
    display_figure(fig)
    plt.close(fig)
print(f"{results.group_id.nunique():,} completed groups; {len(results):,} comparisons; source: {OUTPUT}")
'''
responses=r'''## Responses and sign convention

Both responses are **child minus parent at the outlet**; negative means the collapsed child responds earlier.

$$\Delta T_{\rm peak} = T_{\rm peak,c}-T_{\rm peak,p},\qquad
\Delta T_{\rm centroid}=\bar t_c-\bar t_p,$$
$$\bar t=\frac{\int t[Q(t)-Q_0]dt}{\int[Q(t)-Q_0]dt}.$$

Centroids use native traces over the converged full event. Peak timing uses the **first native maximum**,
so ties or competing peaks can change which feature is selected. Native timestep is 15 seconds;
stored hydrographs are 60-second interval means. All primary plots use minutes, not rounded reporting-bin peaks.
The saved centroid-lag percentage is $100(\bar t_c-\bar t_p)/(\bar t_p-\bar t_{\rm input})$;
it is available as a secondary normalization and is not a peak-lag percentage.
'''
settings=code('''display(pd.DataFrame([{'Factor': FACTOR_LABELS[k], 'Levels': str(v)} for k,v in manifest['config']['grid'].items()]))
s = manifest['jobs'][0]['settings']
display(pd.Series({k:s[k] for k in ['baseflow_cms','peak_cms','n_manning','depth_alpha','beta','segment_length_m','dt_seconds','report_seconds','spinup_hours','rise_hours','fall_hours']}, name='Shared settings').to_frame())
display(results.groupby('method').agg(groups=('group_id','nunique'), comparisons=('group_id','size'), maximum_tail_hours=('tail_hours','max')))
''')
synthesis=[md('# Synthetic results synthesis — completed three-method grid\n\n'+intro),code(setup),md('## 1. Experimental setup\n\nEight three-level factors; identical geometries are compared across hydraulic methods. Fixed BASED uses peak discharge to set reference depth and width scaling with exponent 0.65. Fixed Manning uses midpoint discharge 750 m³/s. Variable Manning updates normal depth through conservative MCT. All methods use n=0.025. Inlet and outlet widths remain equal; internal branch-width expansion is preserved in the collapsed child.'),settings,md(responses),md('## 2. Magnitude and direction by collapse treatment\n\nDots are medians and bars show P05–P95 across the uniformly sampled grid, **not confidence intervals**. Treatments and geometries are deterministic, not independent field observations.'),code('''summary = table('response_summary')
show(summary_figure(summary), 'synthesis_response_ranges')
display(summary[['method','child_state','response','median','earlier_pct','near_zero_pct','later_pct']].round(3))
'''),md('Near-zero bands are ±15 seconds for peak shift and ±2 seconds for centroid shift. These are reporting/interpretation bands, not a full discretization-error estimate. The centroid band reflects the sum of the two saved 1-second tail-convergence tolerances; spatial/time refinement uncertainty remains separate.'),md('## 3. Does the depth approach change the response?\n\nEach density plot matches the same geometry and collapse across two methods. The dashed line is equal response. Density reflects the factorial sampling distribution. Agreement, rank correlation and sign consistency are descriptive—not out-of-sample predictive skill.'),code('''show(agreement_figure(results), 'synthesis_method_agreement')
agreement = table('method_agreement')
display(agreement[(agreement.method_left=='fixed_manning') & (agreement.method_right=='variable_manning')].round(3))
'''),md('## 4. What controls the responses?\n\nVariance fractions describe the selected grid ranges. Main effects, all 28 two-factor interactions and higher-order residuals form the partition. Signed contrasts give direction separately. See the [factor notebook](synthetic_factor_interactions.ipynb) for the full breakdown.'),code('''partition = table('variance_partition')
display(partition.round(3))
main = table('main_total_effects')
display(main[main.response_variation_resolved].sort_values('total_pct',ascending=False).groupby(['method','child_state','response'],sort=False).head(2)[['method','child_state','response','factor','main_pct','total_pct']].round(2))
'''),md('## 5. Interpretation and next use\n\nUse centroid shift to describe full-event timing and peak shift to expose changes in which peak dominates. A later peak does not automatically imply a later centroid. Use the three-method comparison to identify conclusions dependent on the depth closure.\n\nThe next explanatory check is in [predictor validation](synthetic_predictor_validation.ipynb): a geometry-and-slope travel-time expression, evaluated without fitting. Fixed-Manning centroid agreement is an internal identity check, not independent physical validation. Real-network transfer and a new multivariable model search are not claimed here. Peak height, duration, multicollinearity and hydraulic applicability remain outside this update.')]

factor=[md('# Synthetic factor interactions — eight factors, three depth methods\n\n'+intro),code(setup),md(responses),md(r'''## 1. Meaning of the variance decomposition
For each method and treatment, average uniformly over the other tested levels:
$$f_i(x_i)=E[Y\mid x_i]-E[Y],\qquad
f_{ij}(x_i,x_j)=E[Y\mid x_i,x_j]-E[Y]-f_i(x_i)-f_j(x_j).$$
Main and pair percentages are $100\,\mathrm{Var}(f_i)/\mathrm{Var}(Y)$ and $100\,\mathrm{Var}(f_{ij})/\mathrm{Var}(Y)$.
The higher-order remainder contains interactions of order three and above.
Total sensitivity is $100\{1-\mathrm{Var}(E[Y\mid x_{-i}])/\mathrm{Var}(Y)\}$ and includes every interaction containing factor $i$.
**Total sensitivities overlap and must not be summed. Variance contributions are not signed.**

Column abbreviations: Nested island = main-island collapse; Cross up/down = upstream-only/downstream-only collapse. Full treatment names are retained in the tables.

Gray cells and undefined scores flag a response range no larger than twice the interpretation band (4 seconds for centroid, 30 seconds for peak). Raw changes are retained. Normalizing numerical-scale variation would create misleading sensitivities; this screening rule is not a substitute for refinement checks.
'''),code('''main, pairs, partition, contrasts = [table(n) for n in ['main_total_effects','pair_interactions','variance_partition','signed_contrasts']]
resolved_partition=partition[partition.response_variation_resolved]
np.testing.assert_allclose(resolved_partition[['main_pct','pair_pct','higher_order_pct']].sum(axis=1),100,atol=1e-7)
display(partition.round(3))
for response in RESPONSES:
    show(heatmaps(main,response,'main_pct','factor',list(FACTOR_LABELS),'Main effects'), response+'_main_effects')
    show(heatmaps(main,response,'total_pct','factor',list(FACTOR_LABELS),'Total sensitivity (overlapping)'), response+'_total_effects')
'''),md('## 2. All 28 pair interactions\n\nThese are the additional two-factor effects after removing both main effects. Colour scales are shared across methods within each figure. Higher-order interactions are reported in the table above rather than assigned to a pair.'),code('''pairs = pairs.copy()
pairs['pair'] = pairs.factor_a.map(FACTOR_LABELS)+' × '+pairs.factor_b.map(FACTOR_LABELS)
pair_order = pairs['pair'].drop_duplicates().tolist()
assert len(pair_order)==28
for response in RESPONSES:
    show(heatmaps(pairs,response,'variance_pct','pair',pair_order,'Two-factor interactions'),response+'_28_pairs')
'''),md('## 3. Direction: matched highest-minus-lowest contrasts\n\nFor each factor, change its lowest listed level to its highest while holding the other seven factors fixed. The heatmap shows the mean change in response across these matched contexts, in minutes. Negative means a more negative child-minus-parent shift. A near-zero mean can hide opposing context-dependent effects; inspect P10–P90 in the table.'),code('''for response in RESPONSES:
    show(heatmaps(contrasts,response,'mean_minutes','factor',list(FACTOR_LABELS),'Matched signed contrasts',cmap='RdBu_r',signed=True),response+'_signed_contrasts')
display(contrasts[['method','child_state','response','factor','from_level','to_level','median_minutes','p10_minutes','p90_minutes']].round(3))
'''),md('## 4. Inspect a specific interaction\n\nEdit the response, treatment and two factors below. Each line averages over the other six factors. These are conditional mean curves, not a single baseline slice.'),code('''RESPONSE='centroid_shift_minutes'
TREATMENT='simple/full'
FACTOR_X='outer_a_fraction'
FACTOR_LINES='length_multiplier'
fig,axes=plt.subplots(1,3,figsize=(15,4),constrained_layout=True)
for ax,method in zip(axes,METHODS):
    g=results[(results.method==method)&(results.child_state==TREATMENT)]
    marginal=g.groupby([FACTOR_X,FACTOR_LINES])[RESPONSE].mean().unstack(FACTOR_LINES)
    for level in manifest['config']['grid'][FACTOR_LINES]: ax.plot(marginal.index,marginal[level],'o-',label=str(level))
    ax.set(title=LABELS[method],xlabel=FACTOR_LABELS[FACTOR_X],ylabel=RESPONSES[RESPONSE])
    ax.legend(title=FACTOR_LABELS[FACTOR_LINES],fontsize=8);ax.grid(alpha=.15)
show(fig,'selected_factor_interaction')
''')]

predictor=[md('# Physical predictor diagnostics — completed three-method grid\n\n'+intro),code(setup),md('## 1. What is being tested now?\n\nThis update follows the agreed **physical-explanation-first** approach. It does not fit a new random forest, perform model selection or reuse old R² claims. The first test is whether a known Manning travel-time scaling explains centroid and peak changes, and how that explanation changes under the depth methods. It uses the full 137,781-row population.'),md(r'''## 2. Geometry, allocation and slope expression
For edge $e$, let $L_e$ be physical length, $B_e$ width, $S_e$ slope and $f_e=Q_e/Q_{\rm ref}$ the fixed width-based allocation fraction reaching that edge.
Under wide-rectangular Manning normal depth,
$$h_e=\left(\frac{nQ_{\rm ref}f_e}{B_e\sqrt{S_e}}\right)^{3/5},\qquad
c_e=\frac{5}{3n}h_e^{2/3}\sqrt{S_e}.$$
For a fixed linear routing network, the expected travel time is
$$T=\sum_e f_e\frac{L_e}{c_e}
=\frac35 n^{3/5}Q_{\rm ref}^{-2/5}\underbrace{\sum_e L_e B_e^{2/5}f_e^{3/5}S_e^{-3/10}}_{G_M}.$$
Thus the **unfitted** timing axis is
$$\widehat{\Delta T}=\frac35 n^{3/5}Q_{\rm ref}^{-2/5}(G_{M,c}-G_{M,p}),\qquad Q_{\rm ref}=750\;\mathrm{m^3/s}.$$
$G_M$ uses geometry, slope and the prescribed geometry-based allocation rule. The conversion to time additionally uses $n$ and reference discharge. This is **conditional on the Manning closure**, not a universal resistance law.

All sums here use the **whole parent/child network**. Common unchanged upstream/downstream contributions cancel; this does not redefine compound-unit length or silently substitute this quantity for omega. For reference we also export the simpler allocated mean route-length change $\Delta\sum_e f_eL_e$.
'''),code('''features=pd.read_parquet(ANALYSIS/'geometry_diagnostic.parquet')
assert len(features)==len(results)
scores=table('physical_diagnostic_scores')
show(physical_figure(features),'physical_manning_timing_diagnostic')
display(scores.round(5))
'''),md('## 3. How to read these scores\n\n`r2_no_fit` compares the specified formula directly with the model response; no coefficients were estimated. Near-perfect fixed-Manning centroid agreement is expected from the construction and the full-event linear-routing first moment. **It is not held-out prediction or independent validation.**\n\nApplying that same axis to variable Manning tests the approximation of a changing-depth event by one reference discharge. Applying it to fixed BASED deliberately exposes the different depth closure. Peak timing need not obey the centroid identity, especially when dispersion changes or competing peaks exchange dominance. Negative R² means worse squared error than the response mean within that method/treatment. R² is withheld when the response span is at most 4 seconds for centroid or 30 seconds for peak; inspect absolute errors instead. This avoids scoring near-zero numerical variation, notably the Manning main-island centroid response.'),code('''features['centroid_formula_residual_minutes']=features.centroid_shift_minutes-features.manning_reference_shift_minutes
features['peak_formula_residual_minutes']=features.peak_shift_minutes-features.manning_reference_shift_minutes
display(features.groupby(['method','child_state'])[['centroid_formula_residual_minutes','peak_formula_residual_minutes']].agg(['median','min','max']).round(3))
fixed=features[features.method=='fixed_manning']
print('Maximum fixed-Manning centroid discrepancy (seconds):',60*fixed.centroid_formula_residual_minutes.abs().max())
'''),md('## 4. Which settings explain the remaining discrepancy?\n\nThe curves below average the variable-Manning centroid residual over treatments and the other seven factors. They locate systematic biases of the reference-flow approximation; they are not unique causal attributions or predictor importances.'),code('''g=features[features.method=='variable_manning']
fig,axes=plt.subplots(2,4,figsize=(16,7),constrained_layout=True)
for ax,factor in zip(axes.flat,FACTOR_LABELS):
    levels=manifest['config']['grid'][factor]
    grouped=g.groupby(factor).centroid_formula_residual_minutes
    mean=grouped.mean().reindex(levels);lo=grouped.quantile(.1).reindex(levels);hi=grouped.quantile(.9).reindex(levels)
    x=np.arange(len(levels));ax.plot(x,mean,'o-');ax.fill_between(x,lo,hi,alpha=.15)
    ax.axhline(0,color='gray',lw=.7);ax.set(xticks=x,xticklabels=levels,xlabel=FACTOR_LABELS[factor],ylabel='Centroid residual (min)')
show(fig,'variable_manning_formula_residual_controls')
'''),md('## 5. What remains before a predictive-model claim?\n\nRebuild the agreed real-network geometric descriptors for this population with matching domain definitions; retain resistance, length fraction and exposure separately and as omega. Do not transfer old BASED-depth resistance values to Manning runs without labelling their meaning.\n\nThen freeze candidate features, group all hydraulic-method rows and related collapse treatments from one geometry into the same validation fold, and evaluate centroid and peak separately. A held-out geometry or treatment is a stronger test than randomly splitting comparison rows. Real-network validation remains a separate step. Multicollinearity and a broader model-family search are deferred, as agreed.\n\nHistorical predictor selections and fitted scores are preserved in the archived notebook; they have not been silently relabelled as current results.')]

routing=[md('# Synthetic routing results — completed three-method grid\n\n'+intro),code(setup),md(responses),md('## 1. Choose a geometry and inspect matched hydrographs\n\nThe default is a central 20-km network with width 500 m, slope 1e-4, outer fraction 0.8, inner fraction 0.5, length multiplier 1.25 and branch-width factor 1.1. Change any value to one of the tested grid levels. All three methods use the same geometry and forcing.'),code('''SELECTION=dict(total_width_m=500,reference_slope=1e-4,outer_a_fraction=.8,inner_fraction=.5,length_multiplier=1.25,total_length_m=20000,internal_location='central',branch_width_factor=1.1)
mask=pd.Series(True,index=results.index)
for key,value in SELECTION.items():
    mask &= results[key].eq(value) if isinstance(value,str) else np.isclose(results[key],value)
selected=results.loc[mask]
assert selected['case'].nunique()==1
CASE=selected['case'].iloc[0]
display(selected[['method','child_state','centroid_shift_minutes','peak_shift_minutes','centroid_lag_change_pct','tail_hours']].round(4))
'''),md('## 2. Parent–child overlays\n\nSelect specific treatments or set `UNIT_SELECTION=None` to show all seven up to `MAX_PAIRS`. The hydrographs show 60-second interval means; the timing table uses the native 15-second output. The displayed time window is a visual zoom, not the metric integration window.'),code('''from synthetic_experiments.depth_review import hydrograph_figure, geometry_figure
UNIT_SELECTION=['simple/full','nested/main_island_only','crossover/full']
MAX_PAIRS=3
SHOW_GEOMETRY=False
chosen=TREATMENTS if UNIT_SELECTION is None else UNIT_SELECTION
assert set(chosen)<=set(TREATMENTS)
print(f'Showing {min(len(chosen),MAX_PAIRS)} of {len(chosen)} requested comparisons.')
for child in chosen[:MAX_PAIRS]:
    family=child.split('/')[0]
    if SHOW_GEOMETRY: show(geometry_figure(SELECTION,family,child),'geometry_'+child.replace('/','_'))
    show(hydrograph_figure(OUTPUT,case=CASE,width_factor=SELECTION['branch_width_factor'],parent=family+'/original',child=child),'hydrograph_'+child.replace('/','_'))
'''),md('## 3. Parent and multiple collapses in one panel\n\nAll selected children of one family share the same original parent. The uniform centerline is shown as an additional control; when branch-width factor exceeds one it is not generally identical to the full collapsed state.'),code('''FAMILY='nested'
CHILD_SELECTION=None  # or ['nested/inner_only','nested/full']
MAX_CHILDREN=3
family_children=[s for s in TREATMENTS if s.startswith(FAMILY+'/')]
children=family_children if CHILD_SELECTION is None else CHILD_SELECTION
assert set(children)<=set(family_children)
fig,axes=plt.subplots(1,3,figsize=(16,4),constrained_layout=True)
for ax,method in zip(axes,METHODS):
    job=next(j for j in manifest['jobs'] if j['case']==CASE and j['settings']['method']==method)
    wave=pd.read_parquet(OUTPUT/'groups'/job['id']/'hydrographs.parquet')
    s=job['settings'];wave=wave[wave.time_seconds.le((s['spinup_hours']+s['rise_hours']+s['fall_hours']+48)*3600)]
    x=wave.time_seconds/3600
    ax.plot(x,wave[FAMILY+'/original'],color='black',lw=2,label='Parent')
    ax.plot(x,wave['centerline/original'],color='gray',ls=':',label='Uniform centerline')
    for child in children[:MAX_CHILDREN]:ax.plot(x,wave[child],label=child.split('/')[1])
    ax.set(title=LABELS[method],xlabel='Time since simulation start (h)',ylabel='Discharge (m³/s)');ax.legend(fontsize=8)
show(fig,'multiple_collapses_'+FAMILY)
'''),md('## 4. Locate peak/centroid disagreement for follow-up\n\nThese rows flag a later peak but an earlier full-event centroid beyond the display deadbands. They identify hydrographs to inspect; they do not prove a secondary peak without looking at the trace. All stored statistics are native-derived.'),code('''opposed=results[(results.peak_shift_minutes>.25)&(results.centroid_shift_minutes< -2/60)]
display(opposed.groupby(['method','child_state']).size().rename('opposed_sign_cases').to_frame())
display(opposed.sort_values('peak_shift_minutes',ascending=False).head(15)[['case','method','child_state',*FACTOR_LABELS,'peak_shift_minutes','centroid_shift_minutes']])
''')]
for name,cells in [('synthetic_results_synthesis',synthesis),('synthetic_factor_interactions',factor),('synthetic_predictor_validation',predictor),('synthetic_routing_results',routing)]:
 path=NB/(name+'.ipynb')
 if not (ARCHIVE/path.name).exists():shutil.copy2(path,ARCHIVE/path.name)
 nb=nbf.v4.new_notebook(cells=cells,metadata={'kernelspec':{'display_name':'Python (river-hierarchy-rivgraph)','language':'python','name':'python3'},'analysis_reference':'SYN-FULL-DEPTH-GRID-2026-10-28'})
 for cell in nb.cells:cell.metadata['tags']=['full-depth-grid-20261028']
 nbf.write(nb,path);print(path.name,len(cells),'cells')

# Keep the all-pairs effect-size screen when regenerating the current notebooks.
from screen_depth_grid_interactions import install_notebook_section
install_notebook_section()

from install_control_predictor_notebook import install as install_control_predictors
install_control_predictors()

from install_broad_descriptor_notebook import install as install_broad_descriptor_audit
install_broad_descriptor_audit()

from install_manning_candidate_notebook import install as install_manning_candidate
install_manning_candidate()

from install_candidate_reduction_notebook import install as install_candidate_reduction
install_candidate_reduction()
