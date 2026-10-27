"""Build the preparation/results notebook; fitting is opt-in in a separate cell."""
from pathlib import Path
import nbformat as nb

ROOT = Path(__file__).resolve().parents[3]
cells = []
def md(s): cells.append(nb.v4.new_markdown_cell(s))
def code(s): cells.append(nb.v4.new_code_cell(s))

md(r'''# Broad hydrograph predictor benchmark — prepared, not fitted

**Purpose:** establish a broad predictive benchmark, then quantify how much performance survives simplification. The two primary targets are $\Delta t_{\mathrm{peak}}$ [h] and $100(R_{E_{\max}}-1)$ [%]. The other four responses support interpretation. The finalized shared 0.5% window, shared baseline and global maxima are unchanged. Percentages are $100(X_c/X_p-1)$; differences are child minus parent.

No extensive model search has been run during preparation. Run cells 1–3 to inspect the inputs and design. Cell 4 is explicitly disabled until you choose to fit. Later cells read completed results without fitting.

Related: [current small-panel comparison](hydrograph_predictor_scope_comparison.ipynb) and [worked metric definitions](geometry_metric_worked_examples_20261012.ipynb).''')
code('''from pathlib import Path
import json, sys, os, subprocess
import numpy as np
import pandas as pd
from IPython.display import display, Markdown
ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p/'RAPID/src/rapid_tools').is_dir())
OUT = ROOT/'RAPID/outputs/hydrograph_model_search_20261014'
SCRIPT = ROOT/'RAPID/investigations/hydrograph_predictors_20261014/model_search.py'
design = json.loads((OUT/'design.json').read_text())
catalog = pd.read_csv(OUT/'feature_catalog.csv')
plan = pd.read_csv(OUT/'run_plan.csv')
def table(frame):
    lines=['| '+' | '.join(map(str,frame.columns))+' |','| '+' | '.join(['---']*len(frame.columns))+' |']
    for row in frame.itertuples(index=False,name=None):
        lines.append('| '+' | '.join(str(x).replace('|',' / ') for x in row)+' |')
    display(Markdown('\\n'.join(lines)))
display(Markdown('**Status:** '+('completed result tables available' if (OUT/'scores.csv').exists() else 'prepared; no completed model-search results')))
table(plan)
table(pd.read_csv(OUT/'exclusions.csv'))
join = pd.read_csv(OUT/'join_audit.csv')
assert join['_merge'].eq('both').all()
print(f'{len(join):,} canonical parent–child comparisons matched; no unmatched records.')''')
md(r'''## 1. Predictor pool and domains

$U$ is the selected unit domain; $C$ is the entire containing compound before and after **the same edit**. The general benchmark uses $D=C$ for compound members, closed $U$ for standalone units, and $N$ for centerlines. The paired benchmark compares $U$, $C$, and $U+C$ on identical member-edit records. Open $U$ route descriptors condition on entry and normalize widths inside $U$; they omit external exchange. For island edits $U$ is the recorded anchor unit, not the island polygon.

The broad pool has four mechanism families: **length $L$**, **width/splitting $W$**, **topological complexity $T$**, and **resistance $R$**. Context $X$ supplies exposure and relative size. Each metric has a parent value plus one change representation: ratios for strictly positive scales; differences for bounded or zero-prone quantities. Child values are not added as a third algebraically redundant representation.

Resistance uses $z_K=-\log_{10}(K_c/K_p)$ and $z_{K,w}=r_{\mathrm{in},p}(L_{\mathrm{tot},p}^D/L_{\mathrm{tot},p}^N)z_K$, for both fixed laws $K_e\propto L_e/W_e^{2+3\alpha}$, $\alpha\in\{1,0.65\}$. These are geometric resistance hypotheses. In paired comparisons resistance always uses closed $C$, never open $U$.

No example ID, state label, output-hydrograph variable, crest diagnostic or saved hydraulic model parameter is a predictor. Saved slope is not yet established here as an independently measured geometric feature. Width quantiles remain excluded following the earlier discussion. Junction excess duplicates cycle rank in these single-source/sink DAGs and is omitted. Cycle rank and bifurcation count are both retained: nonbinary junctions make them different in many compounds. Route variance duplicated as branch variance is omitted. Area elongation and bifurcation density remain exploratory candidates; predictive value is not assumed.

Ratios are dimensionless, but parent length/width and some differences retain their physical units. Statistical standardization is **not** physical nondimensionalization, and this search does not establish Buckingham-$\Pi$ completeness.''')
code('''table(catalog.groupby(['scope','family']).size().rename('candidate variables').reset_index())
SCOPE = 'B'  # B uses the displayed D convention; change to U or C for the paired pool.
FAMILY = 'L' # L, W, T, R or X
table(catalog[(catalog.scope==SCOPE)&(catalog.family==FAMILY)][['symbol','unit','representation','metric','finite_fraction_all_rows']])
display(Markdown('Coverage above is descriptive across all rows, **not a global feature-selection filter**. Structural unavailability is handled in each training fold.'))''')
md(r'''## 2. Models, combinations and honest selection

**Ridge, Elastic Net, random forest and histogram gradient boosting** are fitted separately for each response. All use equal total training weight per example. Three modest parameter settings per model are tested, with fixed seeds. Tree boosting has no random-row early-stopping split. Four previous small Ridge panels remain fixed references.

General panels include context alone; $L$, $W$, $T$, $R$ with context; $L+W$, $L+R$, $L+W+T$, all four families; and removal of each family from the full pool. Paired panels compare $U$, $C$, $U+C$, and each with $C$ resistance. Some ablations intentionally coincide with named family combinations, making their interpretation explicit.

**Outer validation:** leave an entire example out. **Inner validation:** three grouped folds on the remaining examples choose settings. In each outer fold, `nested_selected` also chooses the model and panel using inner errors only. This is the main estimate for the whole selection procedure. Simply choosing the best row from the outer score table would introduce selection optimism.

Every training fold independently removes features with <70% example-balanced finite coverage, constants and exact duplicates, imputes remaining missing values with training medians and scales with training weights. No missingness indicators are introduced. Some local split means are undefined in unsplit states; imputation does not turn those quantities into physical observations. Fold logs record removed features.

**Compact alternatives:** within each training fold, prune $|\rho_{\mathrm{Spearman}}|\geq0.95$ using a fixed catalog order, then retain at most 7 or 14 predictors by weighted marginal association. Each model is tested on these alternatives. `nested_compact` chooses among them in inner validation. Elastic Net provides an additional sparse alternative. These are reproducible compact baselines, not exhaustive subset search: marginal screening can discard interaction-only predictors, which the unfiltered broad panels retain.

Inner selection minimizes equal-example MSE, separately per target; outer reports show $R^2$, MAE and RMSE with both equal-example and pair weights, plus individual examples. Interpret fewer variables and worse MAE separately from improved squared error. Retained-feature frequency and standardized coefficients are stability diagnostics, not causal importance. Correlated family ablations are more informative than treating each coefficient as independent evidence.

The partial-collapse cohort is primary. Excluding 54 refits both training and test sets. Centerlines are a separate expanded cohort. Example 54 keeps global-maximum timing, including possible crest-dominance changes; no diagnostic enters fitting. Its finite-window distribution metrics include the equilibrium transition. Sensitivity evaluates the same out-of-example predictions against 1% and 2% labels, without tuning to them.''')
code('''pipeline_rows=[]
for scope, specs in [('general',design['full']),('paired',design['paired'])]:
    for s in specs:
        pipeline_rows.append({'scope':scope,'panel':s['panel'],'model':s['model'],
                              'input variables':len(s['columns']),'compact maximum':s['top_k'],
                              'settings':len(s['grid'])})
table(pd.DataFrame(pipeline_rows))
fits = int(plan.loc[plan.cohort.eq('partial'),'fits_per_target'].sum()*2)
print(f'Default run: {fits:,} estimator fits (two primary targets; all/without54).')
print('Fit counts exclude zero/mean baselines. Runtime is not estimated from unrun models.')''')
md('''## 3. Run separately when ready

From the repository root, using the project environment:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 /opt/anaconda3/envs/river-hierarchy-rivgraph/bin/python RAPID/investigations/hydrograph_predictors_20261014/model_search.py --run --jobs 4
```

The default runs both primary targets on partial collapses, with and without 54. To start smaller, append `--targets peak_shift_hours --scenarios all`.

Subsequent batches reuse completed checkpoints:

- **Matched domains:** append `--cohorts paired_U_C`.
- **Centerline sensitivity:** append `--cohorts including_centerlines`.
- **Supporting responses:** append `--targets centroid_shift_hours duration80_change_pct volume_change_pct asymmetry_change` (add a cohort option if desired).

Without `--run`, the script prepares/audits only. Repeating the same run skips completed candidate/outer-fold checkpoints. Changing inputs, code or design requires a fresh `--output` directory if checkpoints already exist. Do not launch concurrent processes writing the same output directory. Failures stop visibly; they are not silently dropped. Scores are produced only for complete cohort/scenario/target batches. Checkpoints still preserve interrupted work.

Alternatively, enable the following cell. It launches the same process using the notebook kernel's Python. No plotting cell fits models.''')
code('''RUN_EXTENSIVE = False
COHORTS = ['partial']
SCENARIOS = ['all', 'without54']
TARGETS = ['peak_shift_hours', 'peak_excess_change_pct']
JOBS = 4
command = [sys.executable, str(SCRIPT), '--run', '--jobs', str(JOBS),
           '--cohorts', *COHORTS, '--scenarios', *SCENARIOS, '--targets', *TARGETS]
if RUN_EXTENSIVE:
    env = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
    subprocess.run(command, cwd=ROOT, env=env, check=True)
else:
    print('Fitting disabled. Enable RUN_EXTENSIVE or use the terminal command above.')''')
md(r'''## 4. Results after your runs

Read `nested_selected` and `nested_compact` first. The other rows diagnose families and model behavior; ranking them retrospectively is exploratory. Compare paired scopes only within the paired cohort. High $R^2$ in the expanded centerline cohort is not directly comparable to partial-collapse $R^2$.

Timing MAE in the score file is in hours; the display converts it to minutes. Peak-excess MAE is in percentage points. Asymmetry is dimensionless. Inspect per-example performance before concluding that a gain generalizes.''')
code('''COHORT = 'partial'
SCENARIO = 'all'
RESPONSE = 'peak_shift_hours'
if (OUT/'scores.csv').exists():
    scores=pd.read_csv(OUT/'scores.csv')
    selected=scores[(scores.cohort==COHORT)&(scores.scenario==SCENARIO)&(scores.response==RESPONSE)&(scores.weighting=='equal_example')].copy()
    if RESPONSE.endswith('_hours'):
        selected['MAE [min]']=60*selected.mae
    table(selected.sort_values('rmse')[['pipeline','n','r2','mae','rmse']+(['MAE [min]'] if 'MAE [min]' in selected else [])].round(4))
    display(Markdown('**Nested selection across held examples:**'))
    choices=pd.read_csv(OUT/'selected_pipelines.csv',dtype={'held_example':str})
    table(choices[(choices.cohort==COHORT)&(choices.scenario==SCENARIO)&(choices.response==RESPONSE)])
else:
    print('No completed runs yet.')''')
code('''if (OUT/'predictions.parquet').exists():
    import plotly.express as px
    pred=pd.read_parquet(OUT/'predictions.parquet')
    z=pred[(pred.cohort==COHORT)&(pred.scenario==SCENARIO)&(pred.response==RESPONSE)&pred.pipeline.isin(['nested_selected','nested_compact'])]
    if len(z):
        fig=px.scatter(z,x='observed',y='predicted',color='example',facet_col='pipeline',hover_data=['state'],template='plotly_white')
        lo=min(z.observed.min(),z.predicted.min());hi=max(z.observed.max(),z.predicted.max())
        fig.add_shape(type='line',x0=lo,y0=lo,x1=hi,y1=hi,line=dict(dash='dash',color='grey'),row='all',col='all')
        fig.update_layout(title='Held-example predictions: '+RESPONSE,height=500)
        fig.show()
    features=pd.read_csv(OUT/'retained_features.csv')
    f=features[(features.cohort==COHORT)&(features.scenario==SCENARIO)&(features.response==RESPONSE)&features.pipeline.eq('nested_compact')&features.active]
    frequency=f.groupby('feature').held_example.nunique().rename('held folds retained').reset_index()
    frequency=frequency.merge(catalog[['feature','symbol','family']],on='feature',validate='one_to_one')
    table(frequency.sort_values('held folds retained',ascending=False)[['symbol','family','held folds retained']])
else:
    print('Prediction and stability plots will become available after fitting.')''')
md('''## 5. Provenance and interpretation limits

`design.json` records the exact panels, settings, preprocessing rules, versions and input hashes. `feature_catalog.csv` maps symbols to implementation names and units. `join_audit.csv` and `exclusions.csv` preserve coverage. Each checkpoint records training/validation examples, inner scores, chosen settings and retained/dropped features. `sensitivity_scores.csv` uses unchanged predictions against alternate-window labels.

The same two quality exclusions as the current resistance benchmark apply to every panel, including geometry-only panels, for fair comparison. These fits are conditional on 15 available examples, their events, the width-allocation rule and fixed resistance laws. A broad benchmark is not a proven prediction ceiling. Selecting a new final formula after reading these results remains exploratory and should ultimately be checked on new examples.

Next review: assess the broad-versus-compact performance gap, per-example consistency, the contributions of each mechanism, $U$ versus $C$, and stability of the retained variables. Use those findings to propose a physically interpretable compact model.''')

n = nb.v4.new_notebook(cells=cells, metadata={'kernelspec': {'display_name':'Python 3','language':'python','name':'python3'}})
path = ROOT/'RAPID/notebooks/hydrograph_model_search.ipynb'
nb.write(n, path)
print(path)
