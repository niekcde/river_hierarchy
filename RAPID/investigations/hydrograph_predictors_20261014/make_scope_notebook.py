"""Current resistance/domain comparison, using symbols and existing response definitions."""
import nbformat as nb
from scopes import ROOT,OUT
cells=[]
def md(s):cells.append(nb.v4.new_markdown_cell(s))
def code(s):cells.append(nb.v4.new_code_cell(s))
md(r'''# Hydrograph predictors: resistance and the unit–compound comparison

**Current comparison, 14 October 2026.** Peak-time shift is the leading target; peak-excess change is the second primary target. The other four responses support interpretation. This replaces the preliminary restricted-domain benchmark as the current investigation; the worked-example metric definitions remain unchanged.

We now reuse reviewed parent–child node identities, existing unit membership and the existing compound selector. We compare resistance and geometry on **identical records**, then compare $U$ and $C$ for the **same member edits**. No routing, calibration, or six-response definitions were changed.''')
code('''from pathlib import Path
import sys,json
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from IPython.display import display,Markdown
ROOT=next(p for p in [Path.cwd(),*Path.cwd().parents] if (p/'RAPID/src/rapid_tools').is_dir())
OUT=ROOT/'RAPID/outputs/hydrograph_predictor_scopes_20261014'
sys.path.insert(0,str(ROOT/'RAPID/src'))
from rapid_tools.hydrograph_response import load,METRICS
states,responses=load('0.5%')
d=pd.read_parquet(OUT/'linked_responses.parquet')
scores=pd.read_csv(OUT/'scores.csv')
audit=pd.read_csv(OUT/'domain_audit.csv',dtype={'example':str})
graphs=pd.read_csv(OUT/'domain_graphs.csv',dtype={'example':str})
design=json.loads((OUT/'comparison_manifest.json').read_text())
pio.renderers.default='plotly_mimetype'
assert len(d)==len(responses)==2187
assert not d.duplicated(['example','state']).any()
np.testing.assert_allclose(d.set_index(['example','state'])[list(METRICS)].sort_index(),responses.set_index(['example','state'])[list(METRICS)].sort_index())
RESPONSE={
 'peak_shift_hours':r'$\\Delta t_{\\mathrm{peak}}$',
 'peak_excess_change_pct':r'$100(R_{E_{\\max}}-1)$',
 'centroid_shift_hours':r'$\\Delta\\mu_t$',
 'duration80_change_pct':r'$100(R_{D_{80}}-1)$',
 'volume_change_pct':r'$100(R_V-1)$','asymmetry_change':r'$\\Delta A$',
}
PANEL={
 'R_only':r'$z_K$', 'R_weighted_only':r'$z_{K,w}$', 'R_context':r'$\\{z_K,z_{K,w}\\}$',
 'geometry':r'$G(L,W,T)$', 'R_plus_L':r'$\\{z_K,z_{K,w}\\}+G(L)$',
 'R_plus_LW':r'$\\{z_K,z_{K,w}\\}+G(L,W)$',
 'R_plus_LWT':r'$\\{z_K,z_{K,w}\\}+G(L,W,T)$',
 'R065_context':r'$\\{z_K,z_{K,w}\\},\\;\\alpha=0.65$',
 'R065_plus_LWT':r'$\\{z_K,z_{K,w}\\}+G(L,W,T),\\;\\alpha=0.65$',
 'R_C':r'$\\{z_K^C,z_{K,w}^C\\}$', 'U_geometry':r'$G^U$', 'C_geometry':r'$G^C$',
 'R_C_plus_U':r'$\\{z_K^C,z_{K,w}^C\\}+G^U$', 'R_C_plus_C':r'$\\{z_K^C,z_{K,w}^C\\}+G^C$',
 'zero':'Zero change','training_mean':'Training-example mean',
}
def show(frame):
    z=frame.copy()
    z=z.rename(columns=RESPONSE)
    for c in ['response','panel']:
        if c in z:z[c]=z[c].map(RESPONSE if c=='response' else PANEL).fillna(z[c])
    lines=['| '+' | '.join(map(str,z.columns))+' |','| '+' | '.join(['---']*len(z.columns))+' |']
    for row in z.itertuples(index=False,name=None):lines.append('| '+' | '.join(str(x).replace('|',r'\\|') for x in row)+' |')
    display(Markdown('\\n'.join(lines)))
def summary(run):
    z=scores[(scores.run==run)&scores.weighting.eq('equal_example')]
    timing=z[z.response.eq('peak_shift_hours')].set_index('panel')
    peak=z[z.response.eq('peak_excess_change_pct')].set_index('panel')
    return pd.DataFrame({'panel':timing.index,'n':timing.n,'examples':timing.examples,
        'Timing $R^2$':timing.r2,'Timing MAE [min]':60*timing.mae,
        'Peak-excess $R^2$':peak.r2,'Peak-excess MAE [percentage points]':peak.mae}).reset_index(drop=True).round(4)
show(pd.DataFrame([dict(cohort=k,**v) for k,v in design['cohorts'].items()]))''')
md(r'''## 1. What $U$, $C$ and the resistance domain mean

- $U$: the selected unit or selected unit set, before and after the edit.
- $C$: its **full containing compound**, before and after that **same edit**. This does not mean collapsing the entire compound.
- $N$: the whole river network. Single-centerline comparisons use $N$.
- For the all-case resistance benchmark, $D=C$ for compound members and $D=U$ for standalone closed units; $D=N$ for single centerlines. This is an explicit common rule, not whichever domain happened to load.

$$R_X^S=\frac{X_c^S}{X_p^S},\qquad \Delta X^S=X_c^S-X_p^S,\quad S\in\{U,C,N\}.$$

Current geometry and the reviewed node identity tables agree with the routed physical graph. Existing unit/compound membership supplies the parent domains; mapped boundaries identify complete child source-to-sink domains. Different numeric node IDs are allowed.

**Open $U$ domains:** some compound members exchange flow through internal nodes. Their route metrics describe a *conditional geometric allocation*: start at the unit inlet, normalize outgoing widths over reaches inside $U$, and exclude external exchange. These are not the full compound's actual allocation. No isolated $U$ resistance is computed for an open unit. $C$ resistance always uses a closed parent and child domain.

The paired comparison requires both domains and excludes cases where $U$ and $C$ are the same graph. It therefore tests the effect of scope rather than comparing two copies of a variable.

For island edits, $U$ is the recorded source/anchor unit. It is not the island polygon or an inferred set of all modified raster pixels.''')
code('''display(graphs.groupby(['scope','parent_closed','child_closed']).size().rename('comparisons'))
show(pd.read_csv(OUT/'benchmark_exclusions.csv',dtype={'example':str}))
print('All response records matched; exclusions concern domains or numerical resistance quality, not missing response rows.')
print('Detailed domain audit:',OUT/'domain_audit.csv')''')
md(r'''## 2. Resistance is now an explicit benchmark

With a wide-section Chezy law and $h=aW^\alpha$,

$$K_e\propto\frac{L_e}{W_e^{2+3\alpha}},\qquad
R_{K_{\mathrm{eq}}}^D=\frac{K_{\mathrm{eq},c}^D}{K_{\mathrm{eq},p}^D},\qquad
z_K^D=-\log_{10}R_{K_{\mathrm{eq}}}^D.$$

A common scale factor cancels within each parent–child ratio. We compare **$\alpha=1$**, the older constant width/depth assumption, with **$\alpha=0.65$** as a separate fixed-law sensitivity. These are geometry-derived benchmarks conditional on the chosen law—not measured resistance or a claim that every current depth follows that law.

We also retain the earlier exposure/size weighting:

$$f_L^D=\frac{L_{\mathrm{tot},p}^{D}}{L_{\mathrm{tot},p}^{N}},\qquad
z_{K,w}^D=f_L^D\,r_{\mathrm{in},p}^D\,z_K^D.$$

The comparison includes $z_K$ alone, $z_{K,w}$ alone, both together, geometry alone, and resistance plus geometry. In the tables “+” means adding input columns to the fitted model, **not summing the predictor values**.

The static calculation uses connectivity and physical lengths/widths. It does not force the route metrics' width-proxy splits. Neither calibrated Manning n/celerity nor a child's hydrograph enters resistance.''')
md(r'''## 3. The geometry combinations use the worked-example metrics

| Set | Inputs for scope $S$ |
|---|---|
| $G^S(L)$ | $r_{\mathrm{in},p}^S\Delta\mu_L^S/\mu_{L,p}^N$, $\mu_{L,p}^S/\mu_{L,p}^N$, $r_{\mathrm{in},p}^S\Delta\sigma_L^S/\mu_{L,p}^N$, $r_{\mathrm{in},p}^S$ |
| Width additions | $R_{\bar W}^S$, $\bar d_p^S$ |
| Topology addition | $\Delta\mathbb E[N_b]^S$ |

Thus $G(L,W,T)$ contains seven inputs, **not three new metrics**. $\mu_L$, $\sigma_L$, $\bar W$, $\bar d$, and $\mathbb E[N_b]$ are the same definitions as in the worked examples. Exposure and scale transformations are shown explicitly here.

These small panels are fixed exploratory comparisons. They are not a final selected set, and not every candidate in the metric dictionary is included. Strong correlations are reported below; adding resistance to geometric ingredients can be redundant.''')
md(r'''## 4. Validation and the two primary targets

Each fold holds out an entire example. All children sharing its parent stay together. Median imputation (only needed for undefined conditional means), constant removal and standardization are fitted on training examples only. Ridge regression uses fixed $\alpha_{\mathrm{ridge}}=10$, distinct from the hydraulic width–depth exponent $\alpha$.

Training and the main reported errors give equal total weight to each example. Zero change and training-example mean predictions are explicit baselines. No tuning or feature ranking uses the held-out example. The comparison is a **linear benchmark**; it does not establish a ceiling on nonlinear predictive performance.

All panels within a cohort use identical records. The main partial-collapse cohort contains **2,170 comparisons across all 15 examples**. Excluding 54 means refitting without it in both training and evaluation. The 0.5% window and global-maximum response definitions remain fixed.''')
code('''show(summary('partial/all'))
show(summary('partial/without54'))''')
md(r'''For partial collapses, resistance with exposure/size context has timing $R^2\approx0.38$ and MAE about 6.4 minutes. Adding length reaches approximately $R^2=0.45$, although MAE does not improve. The geometry-only combination gives a similar $R^2$. Adding every geometric input to resistance does not consistently improve the score.

These results support resistance as a useful baseline, but do not establish independent added value over its length/width/topology ingredients. The alternative width–depth law changes the scores modestly. Peak-excess prediction remains weak under MAE: zero change is better than the fitted panels on the partial-collapse cohort.''')
md(r'''## 5. Direct $U$ versus $C$ comparison

**1,577 identical member-edit cases across all 15 examples**, with distinct available $U$ and $C$. We compare $G^U$ with $G^C$, then add the same $C$-scale resistance terms to each. This avoids giving one scope an easier or smaller test population.

On this cohort, $G^C$ gives timing $R^2\approx0.37$ versus approximately $0.09$ for $G^U$. The compound-scale descriptors are more useful in this tested linear panel. This is evidence about the chosen descriptors and allocation conventions—not a conclusion that every possible local geometric descriptor is uninformative.''')
code('''show(summary('paired_U_C/all'))
show(summary('paired_U_C/without54'))''')
md(r'''## 6. Single centerlines and example 54

The expanded cohort includes the 15 single-centerline collapses. Its response range is much larger, so a higher $R^2$ does not imply a lower timing error. The resistance-plus-geometry panel has timing $R^2\approx0.68$ here, with MAE about 17 minutes. The result remains positive without 54. This supersedes the earlier failure of the incomplete-domain model; coverage, scope policy and predictors have changed, so it is not an isolated proof that adding resistance caused the improvement.

54's two near-equal crests retain the **global maximum** definition. Its 81 partial-collapse children previously resolved as no switch under the tested diagnostic settings; the single centerline's crest classification was matching-radius-sensitive. The existing diagnostic is kept separate and never used as a predictor. The finite-window distribution metrics retain the transition to its higher final equilibrium.''')
code('''show(summary('including_centerlines/all'))
show(summary('including_centerlines/without54'))''')
md(r'''## 7. Inspect held-example predictions and individual relationships

The selected plot uses only predictions made while that entire example was held out. Change the response to inspect one of the other five; the response definitions do not change. Associations and redundancy below are descriptive, not a feature-selection rule.''')
code('''RUN='partial/all'
PANEL_CHOICE='R_plus_LW'
TARGET='peak_shift_hours'
pred=pd.read_parquet(OUT/'predictions.parquet')
v=pred[(pred.run==RUN)&(pred.panel==PANEL_CHOICE)&(pred.response==TARGET)]
fig=px.scatter(v,x='observed',y='predicted',color='example',hover_data=['state'],
    title='Held-example predictions: '+METRICS[TARGET][0],opacity=.65)
lo=min(v.observed.min(),v.predicted.min());hi=max(v.observed.max(),v.predicted.max())
fig.add_shape(type='line',x0=lo,y0=lo,x1=hi,y1=hi,line=dict(dash='dash',color='grey'))
fig.update_layout(template='plotly_white',height=520,xaxis_title='Observed ['+METRICS[TARGET][1]+']',yaxis_title='Predicted ['+METRICS[TARGET][1]+']')
fig.show()
show(scores[(scores.run==RUN)&(scores.panel==PANEL_CHOICE)&(scores.response==TARGET)&scores.weighting.str.startswith('example_')][['weighting','n','mae','r2']].round(4))''')
md(r'''## 8. All six responses and window sensitivity

The six targets are $\Delta t_{\mathrm{peak}}$, $100(R_{E_{\max}}-1)$, $\Delta\mu_t$, $100(R_{D_{80}}-1)$, $100(R_V-1)$ and $\Delta A$. All timing differences are child minus parent, and all percentages are $100(X_c/X_p-1)$.

The two peak targets are unchanged across the 0.5%, 1% and 2% windows. The four distribution responses can vary with the finite window. Below, sensitivity scores evaluate the **same primary-window held-example predictions** against alternate-window labels; they are not separately tuned models.''')
code('''show(scores[scores.run.eq('partial/all')&scores.panel.eq('R_plus_LW')&scores.weighting.eq('equal_example')][['response','n','mae','rmse','r2']].round(5))
sensitivity=pd.read_csv(OUT/'sensitivity_scores.csv')
show(sensitivity[sensitivity.run.eq('partial/all')&sensitivity.panel.eq('R_plus_LW')][['window','response','mae','r2']].round(5))''')
md(r'''## 9. Numerical and source audit

The earlier head-form resistance solver had convergence failures on large compounds. An isolated unit-flow formulation solves the **same signed quadratic edge law**, with mass conservation and independent primal/dual energy checks. It does not change the production solver.

For unit source-to-sink flow $q$, the primal quantity is $K_{\mathrm{upper}}=\sum_e K_e|q_e|^3$. For trial node potentials $h$, write $S(h)=\sum_e |h_u-h_v|^{3/2}/\sqrt{K_e}$ and $b$ for unit source/sink injection. Optimizing the dual potential scale gives $K_{\mathrm{lower}}=(b^Th)^3/S(h)^2$. Numerical acceptance requires mass-balance error below $10^{-8}$ and relative agreement of these energy bounds within $10^{-8}$. The implemented laws are checked against analytic circuits, the existing solver on a braided core, and the two real worked examples.

Of 5,904 graph-and-law solves, 5,903 pass. `37 / island_37_098` is excluded because its $\alpha=1$ compound calculation does not meet that tolerance. `41 / S061_unit_57` is excluded because a compound boundary lacks a surviving reviewed correspondence. Neither missing value is imputed into the resistance benchmark.

All 2,187 response rows still match canonical geometry records. The exclusions are explicit quality/coverage exclusions, not alias duplicates or unmatched hydrographs.''')
code('''solver=pd.read_csv(OUT/'resistance_solver_audit.csv')
show(solver.groupby('alpha').agg(solves=('success','size'),passed=('success','sum'),max_energy_gap=('relative_energy_gap','max'),max_mass_error=('continuity','max')).reset_index())
show(solver[~solver.success][['graph','alpha','relative_energy_gap','continuity']])
print('Provenance:',OUT/'geometry_manifest.json')
print('Numerical assumptions:',OUT/'resistance_manifest.json')
print('Validation design:',OUT/'comparison_manifest.json')''')
md(r'''## 10. Code-name lookup and redundancy

Symbols are used for interpretation; implementation names below allow reproducibility. $D$ follows the prescribed closed-domain policy in the all-case benchmark. In the paired comparison, the superscript is explicitly $U$ or $C$.''')
code(r'''def feature_symbol(key):
    scope,rest=key.split('_',1);s='D' if scope=='B' else scope
    form={
      'K_log_a1':rf'z_K^{{{s}}}', 'weighted_K_log_a1':rf'z_{{K,w}}^{{{s}}}',
      'K_log_a065':rf'z_K^{{{s}}}(\alpha=0.65)', 'weighted_K_log_a065':rf'z_{{K,w}}^{{{s}}}(\alpha=0.65)',
      'exposed_length_change':rf'r_{{\mathrm{{in}},p}}^{{{s}}}\Delta\mu_L^{{{s}}}/\mu_{{L,p}}^N',
      'mean_length_fraction':rf'\mu_{{L,p}}^{{{s}}}/\mu_{{L,p}}^N',
      'exposed_sd_change':rf'r_{{\mathrm{{in}},p}}^{{{s}}}\Delta\sigma_L^{{{s}}}/\mu_{{L,p}}^N',
      'exposure':rf'r_{{\mathrm{{in}},p}}^{{{s}}}', 'ratio_width_mean':rf'R_{{\bar W}}^{{{s}}}',
      'parent_dominance_mean':rf'\bar d_p^{{{s}}}', 'delta_expected_splits':rf'\Delta\mathbb E[N_b]^{{{s}}}',
    }
    return '$'+form.get(rest,rest)+'$'
features=list(dict.fromkeys(x for category in ['panels','paired_panels'] for cols in design[category].values() for x in cols))
show(pd.DataFrame({'Symbol':[feature_symbol(x) for x in features],'Code':['`'+x+'`' for x in features]}))
redundancy=pd.read_csv(OUT/'redundancy.csv')
redundancy['feature']=redundancy.feature.map(feature_symbol);redundancy['other']=redundancy.other.map(feature_symbol)
show(redundancy.sort_values('rho',key=abs,ascending=False).head(15).round(3))
a=pd.read_csv(OUT/'associations.csv',dtype={'example':str})
a=a[(a.cohort=='partial')&(a.example=='pooled')].copy()
a['feature']=a.feature.map(feature_symbol);a['response']=a.response.map(RESPONSE)
mat=a.pivot(index='feature',columns='response',values='rho')
fig=px.imshow(mat,zmin=-1,zmax=1,color_continuous_scale='RdBu_r',aspect='auto',title='Exploratory individual associations — six responses')
fig.update_layout(height=600);fig.show()''')
md(r'''## Next scientific decision

The clearest finding is that **compound-scale descriptors outperform the current unit-scale descriptors for member edits on the same records**. Resistance is useful, especially with exposure/size context, but its geometric ingredients overlap with the other predictors. Keep both MAE and $R^2$ visible: the panel with better squared error is not always better for a typical small shift.

The next selection stage should compare alternative representations within each mechanism and use nested example-level validation. Do not promote the largest panel just because it contains more variables, and do not infer physical causality from these associations. The current results are substantially broader than the earlier restricted-domain benchmark, but remain conditional on this event/model cohort and the stated allocation and resistance laws.''')
n=nb.v4.new_notebook(cells=cells,metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'}})
p=ROOT/'RAPID/notebooks/hydrograph_predictor_scope_comparison.ipynb';nb.write(n,p);print(p)
