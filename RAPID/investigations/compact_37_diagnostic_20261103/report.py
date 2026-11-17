"""Present exact donor tracing and explicitly post-hoc example-37 probes."""
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/compact37_mpl')
from pathlib import Path
import json,hashlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'RAPID/outputs/compact_37_diagnostic_20261103'

def main():
    d=json.loads((OUT/'design.json').read_text())
    for p,h in d['sources'].items():assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==h,p
    scores=pd.read_csv(OUT/'diagnostic_scores.csv')
    p=pd.read_parquet(OUT/'diagnostic_predictions.parquet')
    features=pd.read_parquet(OUT/'example37_features.parquet')
    z=features[features.group_id.eq('37_9485f774f602')]
    comparisons=['five_with_G','drop_equivalent_width','replace_width_with_domain_size','revised9']
    names=['Compact five with ΔGₘ','Remove equivalent width (4)','Replace width with edited size (5)','Revised nine']
    fig,axes=plt.subplots(1,4,figsize=(15,4.4),layout='constrained')
    for ax,model,name in zip(axes,comparisons,names):
        a=p[p.model.eq(model)&p.cohort.eq('original')]
        score=scores[scores.model.eq(model)&scores.population.eq('ordinary')].iloc[0]
        ax.scatter(60*a.centroid_shift_hours,60*a.predicted_hours,s=18,alpha=.45)
        ax.plot([-160,20],[-160,20],ls='--',color='black',lw=1)
        ax.set_xscale('symlog',linthresh=1);ax.set_yscale('symlog',linthresh=1)
        ax.set_xlim(-160,20);ax.set_ylim(-160,20)
        ax.set_xlabel('Modelled shift (min)');ax.set_ylabel('Predicted shift (min)')
        ax.set_title(f'{name}\nMAE {score.mae_minutes:.2f} min',fontsize=10)
        ax.grid(alpha=.2)
    fig.suptitle('Example 37 held out from training · 395 ordinary comparisons\nPost-hoc diagnostic; edited size = affected footprint length',fontsize=12)
    fig.savefig(OUT/'example37_prediction_diagnostic.png',dpi=180);plt.close(fig)
    donors=pd.read_csv(OUT/'training_cohort_contributions.csv')
    a=donors[donors.state.eq('S315_unit_274')&donors.model.isin(['five_with_G','revised9'])]
    fig,ax=plt.subplots(figsize=(8,4.2),layout='constrained')
    for j,model in enumerate(['five_with_G','revised9']):
        g=a[a.model.eq(model)].set_index('training_cohort')
        for dx,cohort,color in [(-.17,'original','C0'),(.17,'segmented','C1')]:
            v=g.loc[cohort,'contribution_minutes'];ax.bar(j+dx,v,.32,color=color,label=cohort if j==0 else None)
            ax.text(j+dx,v+(1 if v>=0 else -1),f'{v:+.2f}',ha='center',va='bottom' if v>=0 else 'top')
    actual=60*features.loc[features.state.eq('S315_unit_274'),'centroid_shift_hours'].iloc[0]
    ax.axhline(actual,color='black',ls='--',label=f'Modelled shift: +{actual:.2f} min')
    ax.set_xticks([0,1],['Compact five with ΔGₘ','Revised nine'])
    ax.set_ylabel('Contribution to RF prediction (min)');ax.set_ylim(-45,15)
    ax.set_title('Why S315_unit_274 receives the wrong sign\nExact training-case contributions; no river-37 cases in training')
    handles,labels=ax.get_legend_handles_labels();labels=[{'original':'Ordinary training cases','segmented':'Full-network training controls'}.get(x,x) for x in labels]
    ax.legend(handles,labels,loc='lower right',fontsize=8)
    fig.savefig(OUT/'training_contributions.png',dpi=180);plt.close(fig)
    lines=['# Example 37 compact-model error diagnosis','',
        'Reference: COMPACT-37-DIAGNOSTIC-2026-11-03. Fixed-depth Manning, fixed RF configuration, held-out example 37. This is a post-hoc explanation of a known failure; no final predictor set is selected.','',
        '## Finding','',
        'Equivalent parent path width directs a substantial fraction of RF prediction weight toward full-network controls with large negative timing shifts. This generates a systematic negative offset across many small collapses sharing a large parent domain. Removing that variable largely resolves the observed example-37 failure under this fixed RF.','',
        f'{len(z)} states share parent equivalent path width {z.path_parent_equivalent_width.iloc[0]:.6g} m and parent allocated mean length {z.parent_mean_length_km.iloc[0]:.3f} km. Their median edited footprint length is only {z.domain_size_m.median():.1f} m. Equivalent path width is a sum across complete routes, including shared reaches; this enormous value is not a physical channel width. Combinatorial route multiplicity makes it difficult to interpret as local edit magnitude.','',
        'The compact set contains parent extent indirectly through equivalent width, but no explicit edited-footprint size. Full-network controls are valid training cases; their influence here is a representation/generalization issue, not train/test leakage. All river-37 cases are excluded from the fitted training set.','',
        '## Exact prediction trace','',
        f'S315_unit_274 has modelled centroid shift +{actual:.3f} min. The compact five predict −35.743 min: ordinary training cases contribute +1.971 min, while full-network controls contribute −37.715 min. Full-network controls carry 24.8% of the RF prediction weight. The revised nine draw effectively no full-network prediction weight for this state and predict +1.858 min.','',
        'The donor decomposition includes bootstrap multiplicities and the original river-balancing weights. Baseline refits reproduced the stored predictions exactly; donor reconstruction agreed to approximately 4e-15 hours. Thus this trace explains the fitted predictions, rather than relying on a feature-importance ranking.','',
        '## Controlled probes','',
        'All probes use the same held-out river, RF configuration, random seed and targets. Only the named input or training-cohort change is made. Scores below cover the same 395 ordinary river-37 comparisons; they are not whole-population validation scores.','',
        '| Model/probe | Inputs | MAE (min) | RMSE (min) | R² within river 37 |',
        '|---|---:|---:|---:|---:|']
    for r in scores[scores.population.eq('ordinary')].itertuples():
        lines.append(f'| {r.model} | {len(d["panels"][r.model])} | {r.mae_minutes:.3f} | {r.rmse_minutes:.3f} | {r.r2:.3f} |')
    lines+=['','Adding edited-domain size while retaining equivalent width helps but leaves large errors. Removing equivalent width, or replacing it with edited-domain size, largely resolves the systematic offset. Excluding full-network controls from training is a mechanism check, not a recommended population change: it worsens prediction of the complete compound/full-network collapse.','',
        'The repair does not explain every response. For S315_unit_274, removal predicts about +2.59 min versus +9.46 min modelled; for S256_unit_229 it predicts about +2.09 min versus +7.86 min. These remaining underpredictions are much smaller than the erroneous large advances.','',
        '## Consequence for predictor selection','',
        'This revises the interpretation of the compact-set failure: it is strong evidence against this equivalent-path-width representation in the current mixed population, not proof that the compact mean-length/slope/geometry descriptors lack all necessary complexity information. Entropy/dominance additions retained the problematic width and therefore did not remove this failure.','',
        'Next evaluate the two frozen alternatives (remove equivalent width; replace it with edited-domain size) across the full real population and all three validation strategies, and check their synthetic performance. These one-river post-hoc scores cannot establish a universal replacement or an unbiased performance gain. Keep all comparisons and controls in the primary evaluation.','',
        'The target and its finite event window were unchanged. These diagnostics require no routing reruns, hydrograph smoothing or peak reassignment.','',
        '## Files','',
        '- example37_prediction_diagnostic.png and training_contributions.png: visual summary.',
        '- diagnostic_scores.csv / diagnostic_predictions.parquet: all controlled probes.',
        '- top_training_donors.csv / training_cohort_contributions.csv: exact prediction origins.',
        '- reproduction_checks.csv: refit and donor-reconstruction parity.',
        '- state_errors_all_strategies.csv: original frozen errors across all validation schemes.',
        '- training_support.csv and example37_features.parquet: geometric ranges and domain context.',
        '- impurity_importance_diagnostic.csv: descriptive RF importance only; not evidence of physical causality.','']
    (OUT/'report.md').write_text('\n'.join(lines))
    (OUT/'completion.json').write_text(json.dumps(dict(complete=True,diagnostic_fits=7,report_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        files={str(OUT/n):hashlib.sha256((OUT/n).read_bytes()).hexdigest() for n in ['report.md','example37_prediction_diagnostic.png','training_contributions.png','diagnostic_scores.csv']}),indent=2))

if __name__=='__main__':main()
