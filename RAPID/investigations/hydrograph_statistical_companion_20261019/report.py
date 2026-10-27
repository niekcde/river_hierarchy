"""Generate reproducible static figures, results, and a dedicated read-only notebook."""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from run import ROOT,HERE,OUT,COLS,weights,quantile
LABELS=['Mean length ratio','Parent length (km)','Length SD ratio','Mean width ratio','Width CV ratio','Route entropy ratio','Parent split dominance']

def table(df):
    # Avoid optional tabulate dependency.
    fmt=lambda v:f'{v:.3f}' if isinstance(v,(float,np.floating)) else str(v)
    return '| '+' | '.join(df.columns)+' |\n| '+' | '.join(['---']*len(df.columns))+' |\n'+'\n'.join('| '+' | '.join(fmt(v) for v in row)+' |' for row in df.itertuples(index=False,name=None))

def main():
    scores=pd.read_csv(OUT/'scores.csv');eff=pd.read_csv(OUT/'effects_uncertainty.csv');coef=pd.read_csv(OUT/'coefficients.csv');diag=pd.read_csv(OUT/'fit_diagnostics.csv');boot=pd.read_csv(OUT/'bootstrap_diagnostics.csv');data=pd.read_parquet(OUT/'residual_diagnostics.parquet')
    fig,ax=plt.subplots(figsize=(10,5))
    for model,offset,color in [('mixed',-.12,'#275dad'),('wls',.12,'#c36718')]:
        g=eff[eff.model.eq(model)].set_index('term').loc[COLS];y=np.arange(7)+offset
        ax.errorbar(g.iqr_effect_minutes,y,xerr=[g.iqr_effect_minutes-g.iqr_low,g.iqr_high-g.iqr_effect_minutes],fmt='o',color=color,label=model,capsize=3)
    ax.axvline(0,color='gray',lw=1);ax.set_yticks(range(7),LABELS);ax.invert_yaxis();ax.set_xlabel('Conditional Q25 → Q75 contrast (minutes), pointwise bootstrap 95% interval');ax.legend();ax.set_title('Exploratory associations; correlated predictors held fixed');fig.tight_layout();fig.savefig(OUT/'effect_contrasts.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(2,4,figsize=(15,7))
    for j,c in enumerate(['predicted_hours']+COLS):
        ax=axes.flat[j]
        for model,color in [('mixed','#275dad'),('wls','#c36718')]:
            g=data[data.model.eq(model)].copy();w=weights(g);x=g[c].to_numpy();edges=np.unique([quantile(x,w,q) for q in np.linspace(0,1,9)]);bins=np.digitize(x,edges[1:-1]);xx=[];yy=[]
            for k in np.unique(bins):
                z=bins==k;xx.append(np.average(x[z],weights=w[z]));yy.append(np.average(g.error_minutes.to_numpy()[z],weights=w[z]))
            ax.plot(xx,yy,'o-',label=model,color=color,ms=4)
        ax.axhline(0,color='gray',lw=.7);ax.set_title('Prediction (hours)' if j==0 else LABELS[j-1]);ax.set_ylabel('Prediction − response (min)');ax.set_xlabel('ln(1 + child/parent)' if c.startswith('log1p__') else '')
    axes.flat[0].legend();fig.suptitle('Held-network binned signed residuals: descriptive, no confidence bands');fig.tight_layout();fig.savefig(OUT/'residual_patterns.png',dpi=180);plt.close(fig)
    # Report how far a marginal IQR contrast moves rows from observed joint geometry.
    base=data[data.model.eq('mixed')];x=base[COLS].to_numpy();w=weights(base);mu=np.average(x,axis=0,weights=w);sd=np.sqrt(np.average((x-mu)**2,axis=0,weights=w));z=(x-mu)/sd
    from scipy.spatial import cKDTree
    tree=cKDTree(z);support=[]
    for i,c in enumerate(COLS):
        for q in [.25,.75]:
            endpoint=quantile(x[:,i],w,q);changed=z.copy();changed[:,i]=(endpoint-mu[i])/sd[i];distance,_=tree.query(changed)
            support.append(dict(term=c,quantile=q,endpoint=endpoint,median_nearest_distance=quantile(distance,w,.5),p90_nearest_distance=quantile(distance,w,.9),note='Standardized Euclidean distance to any observed row; descriptive, not proof of conditional support'))
    pd.DataFrame(support).to_csv(OUT/'contrast_support.csv',index=False)
    primary=scores.query("target=='centroid' and training_scenario=='all' and evaluation=='all'")[['model','r2','mae_minutes','relative_error_pct']]
    sens=scores.query("target=='centroid'")[['model','training_scenario','evaluation','r2','mae_minutes','relative_error_pct']]
    effects=eff[['model','term','iqr_effect_minutes','iqr_low','iqr_high','lono_same_sign']].copy();effects['term']=effects.term.map(dict(zip(COLS,LABELS)))
    stability=coef[(coef.held=='full')].pivot_table(index=['model','term'],columns='scenario',values='coefficient_hours')*60;stability.to_csv(OUT/'refit_coefficient_sensitivity.csv')
    text='''# Statistical companion: first-stage findings

Run: 19 October 2026. Research intent and amendment rules: [DESIGN.md](DESIGN.md). These are exploratory conditional associations in simulated hydrograph responses, not causal effects or independent validation.

## Predictive performance

Identical held-network folds; unseen network intercepts are zero. All metrics use equal network weights. MAE is minutes; relative error uses a one-minute denominator floor.

'''+table(primary)+'''

Both linear companions lose substantial predictive skill. Equal-network WLS performs better than the mixed model here, but neither approaches the forest. Fitting squared error rather than relative loss is also relevant to the high floored relative errors. These results do not demonstrate that geometric information is insufficient.

## Cohort and dependence

All 2,170 input/prediction keys and response values match; all seven transformations reproduce exactly. The finalized 0.5% response table is verified separately. Whole-network states remain excluded; response_coverage.csv records the 15 whole-network cases and two other omissions without claiming their exclusion provenance has been resolved. Network sizes range from 21 to 408.

Each network shares one S000_base response parent. Geometric parent signatures also recur: network 37 has 395 rows but only three signatures/overlap groups; network 22 has 408 rows but 14. A network intercept does not capture every dependence induced by overlapping edits. Bootstrap sampling retains each whole network, including all these dependencies; it assumes networks are the independent sampling units.

Equal-network pooled entropy VIF is 12.29, versus 9.77 after within-network centering. Pooled mean-width and width-CV VIFs are 7.80 and 6.95. The scaled design condition number is 8.60 (within-network: 7.48). This is full rank, but individual conditional coefficients are difficult to isolate. These coefficients combine within- and between-network information; they are not pure within-network effects.

## Conditional contrasts and uncertainty

The table reports equal-network marginal Q25→Q75 contrasts, holding the other six predictors constant. Intervals are pointwise 95% whole-network percentile-bootstrap intervals; they are not seven-test simultaneous intervals or significance tests. All 2,000 draws succeeded for both models. Only 15 original groups are available; percentile tail coverage and post-selection bias remain unresolved. LONO sign agreement describes stability, not a probability of a physical effect.

'''+table(effects)+'''

Mean width, width CV and entropy have negative conditional coefficients in both fits; parent split dominance is positive. This describes later/earlier simulated centroid timing conditional on correlated covariates, not what independently changing channel geometry would cause. Mean-length direction is much less stable in WLS, and length-SD association is sensitive to the model/weighting choice. The mixed-model mean-length bootstrap interval spans zero despite all 15 single-network deletions retaining its sign: deleting one network and resampling networks measure different aspects of stability.

Some IQR contrasts are large compared with prediction errors because they hold strongly correlated variables fixed in combinations that may be poorly supported. contrast_support.csv records descriptive joint-distance diagnostics. The reference contrasts in effects_uncertainty.csv (ratio 0.5→1, parent length +1 km, dominance +0.1) are algebraic benchmarks, not guaranteed feasible interventions.

## Explicit exclusions and refits

training_scenario=all with a reduced evaluation is evaluation-only exclusion. A reduced training_scenario means the entire LONO procedure and full coefficient fit were rerun without those observations. Saved forest predictions are only rescored; no forest refits are implied.

'''+table(sens)+'''

The extreme 37 case matters for fitting as well as scoring: removing it only from mixed-model evaluation gives R² −0.009, whereas refitting without it gives R² 0.156 on that same reduced cohort. Excluding 54 does not rescue either linear model. Refit coefficient changes are retained in refit_coefficient_sensitivity.csv; they must be considered alongside bootstrap and LONO stability.

## Adequacy and next decision

residual_patterns.png shows descriptive binned held-network errors versus predictions and each input. Systematic patterns and the performance gap warrant considering a restrained nonlinear companion, but do not uniquely diagnose nonlinearity: correlated predictors, heterogeneous network relationships, influence and extrapolation may also contribute. No GAM was fitted in this stage. A later GAM requires a documented extension with low-complexity smooths and training-only selection; these same data cannot become an untouched test.

Peak-transfer metrics are in scores.csv, explicitly using centroid predictions; no peak model was trained. The separate whole-network transfer check remains separate.

## Verification and limitations

Independent statsmodels REML checks on the full cohort and two network-deletion cohorts agree in fixed-effect predictions within 3.3e-9 hours. WLS agrees with statsmodels, and population predictions are invariant to test outcomes and group labels. statsmodels emits a small-variance boundary warning on these fits; optimized variance estimates are positive, and variance diagnostics are saved rather than suppressed. Custom-fit boundary counts are in bootstrap_diagnostics.csv. Source hashes include the frozen serialized model and are verified unchanged at run completion.

No naive independent-row p-values are reported. Model-based prediction intervals, simultaneous inference and causal claims are outside this stage.
'''
    HERE.joinpath('RESULTS.md').write_text(text)
    cells=[]
    def md(s):cells.append(dict(cell_type='markdown',metadata={},source=s.splitlines(True)))
    def code(s):cells.append(dict(cell_type='code',metadata={},source=s.splitlines(True),outputs=[],execution_count=None))
    md('# Hydrograph statistical companion\nSeparate investigation; frozen model unchanged. Read DESIGN.md for the durable research contract and RESULTS.md for interpretation. All inference is exploratory; no causal effects or significance tests.\n')
    code("from pathlib import Path\nimport pandas as pd\nfrom IPython.display import display, Image\nROOT=Path.cwd()\nwhile not (ROOT/'RAPID').exists():\n    if ROOT == ROOT.parent: raise RuntimeError('Run inside repository')\n    ROOT=ROOT.parent\nOUT=ROOT/'RAPID/outputs/hydrograph_statistical_companion_20261019'\nscores=pd.read_csv(OUT/'scores.csv')\ndisplay(scores.query(\"target=='centroid' and training_scenario=='all' and evaluation=='all'\"))\n")
    md('## Associations and uncertainty\nPointwise whole-network bootstrap intervals, 2,000 draws, only 15 observed networks. Marginal IQR contrasts can have poor joint support under collinearity.\n')
    code("display(pd.read_csv(OUT/'effects_uncertainty.csv'))\ndisplay(Image(filename=str(OUT/'effect_contrasts.png')))\n")
    md('## Dependence and collinearity\nResponse parents are shared within networks; geometric parent signatures recur across edits.\n')
    code("display(pd.read_csv(OUT/'dependence_audit.csv',dtype={'example':str}))\ndisplay(pd.read_csv(OUT/'collinearity.csv'))\n")
    md('## Exclusions versus refits\ntraining_scenario=all retains original training. Other scenarios remove observations before every fit. Forest rows are evaluation-only rescoring.\n')
    code("display(scores.query(\"target=='centroid'\"))\ndisplay(pd.read_csv(OUT/'refit_coefficient_sensitivity.csv'))\n")
    md('## Linear adequacy\nDescriptive held-network binned residuals; no smooth model fitted. Patterns are not proof of a particular nonlinear or physical mechanism.\n')
    code("display(Image(filename=str(OUT/'residual_patterns.png')))\ndisplay(pd.read_csv(OUT/'contrast_support.csv'))\n")
    md('## Peak transfer\nThese are centroid predictions evaluated against peak shifts, not a separately fitted peak model. Whole-network states remain outside this cohort.\n')
    code("display(scores.query(\"target=='peak_transfer' and training_scenario=='all' and evaluation=='all'\"))\n")
    notebook=dict(cells=cells,metadata=dict(kernelspec=dict(display_name='Python 3',language='python',name='python3')),nbformat=4,nbformat_minor=4)
    (ROOT/'RAPID/notebooks/hydrograph_statistical_companion_20261019.ipynb').write_text(json.dumps(notebook,indent=1)+'\n')
    print('Wrote results, figures and notebook')
if __name__=='__main__':main()
