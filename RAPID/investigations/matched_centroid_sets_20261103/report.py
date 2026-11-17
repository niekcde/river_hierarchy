"""Report the complete matched comparison without pooling away cohort failures."""
from pathlib import Path
import json
import hashlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'RAPID/outputs/matched_centroid_sets_20261103'
MODELS=['geometry_normalized_9','centroid_with_G','centroid_without_G']
NAMES={'geometry_normalized_9':'Previous nine inputs','centroid_with_G':'Five inputs including ΔGₘ',
       'centroid_without_G':'Five inputs, without ΔGₘ'}

def main():
    receipt=json.loads((OUT/'completion.json').read_text());assert receipt['complete']
    design=json.loads((OUT/'design.json').read_text())
    for path,digest in design['sources'].items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==digest,path
    for path,digest in receipt['files'].items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==digest,path
    p=pd.read_parquet(OUT/'predictions.parquet');s=pd.read_csv(OUT/'repeat_scores.csv')
    assert len(p)==2185*5*3 and len(s)==45
    assert not p.duplicated(['model','repeat','example','state']).any()
    baseline=s[s.model.eq(MODELS[0])].set_index(['repeat','cohort'])
    paired=[]
    for r in s[~s.model.eq(MODELS[0])].itertuples():
        b=baseline.loc[(r.repeat,r.cohort)]
        paired.append(dict(model=r.model,repeat=r.repeat,cohort=r.cohort,
            delta_r2=r.r2-b.r2,delta_mae_minutes=r.mae_minutes-b.mae_minutes))
    pd.DataFrame(paired).to_csv(OUT/'paired_changes.csv',index=False)
    selections=[]
    for path in sorted((OUT/'checkpoints').glob('*/*/*/*/outer.json')):
        result=json.loads(path.read_text());first=result['rows'][0]
        selections.append(dict(model=first['model'],repeat=first['repeat'],fold=first['fold'],
            selected_configuration=result['best']['name'],inner_relative_error=result['best']['score']))
    pd.DataFrame(selections).to_csv(OUT/'selected_configurations.csv',index=False)
    # Show all repeat scores: ranges describe split sensitivity, not confidence intervals.
    fig,axes=plt.subplots(2,2,figsize=(11,7),layout='constrained')
    for col,cohort in enumerate(['original','segmented']):
        for row,metric in enumerate(['r2','mae_minutes']):
            ax=axes[row,col]
            for i,model in enumerate(MODELS):
                z=s[s.model.eq(model)&s.cohort.eq(cohort)].sort_values('repeat')
                ax.scatter(i+np.linspace(-.09,.09,5),z[metric],s=24,alpha=.7,color=f'C{i}')
                ax.plot([i-.2,i+.2],[z[metric].mean()]*2,color=f'C{i}',lw=3)
            ax.set_xticks(range(3),['Previous 9','New 5 + ΔGₘ','New 5 − ΔGₘ'])
            ax.set_ylabel('Held-out R²' if row==0 else 'MAE (minutes)')
            ax.grid(axis='y',alpha=.2)
            if row==0:ax.set_title('2,170 ordinary comparisons' if col==0 else '15 full-network controls')
    fig.suptitle('Matched compound-family validation · five repeats\nPoints: individual repeats; bars: means')
    fig.savefig(OUT/'matched_scores.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(2,3,figsize=(13,8),layout='constrained')
    for row,cohort in enumerate(['original','segmented']):
        for col,model in enumerate(MODELS):
            z=p[p.model.eq(model)&p.cohort.eq(cohort)]
            # Average repeated held-out predictions for display only. Scores above
            # remain the mean of per-repeat scores, not this ensemble's score.
            z=z.groupby(['example','state']).agg(actual=('centroid_shift_hours','first'),predicted=('predicted_hours','mean'))*60
            ax=axes[row,col];ax.scatter(z.actual,z.predicted,s=12 if row==0 else 28,alpha=.45)
            extent=max(abs(z.to_numpy()).max(),1)*1.1
            ax.plot([-extent,extent],[-extent,extent],color='black',ls='--',lw=1)
            ax.set_xscale('symlog',linthresh=1);ax.set_yscale('symlog',linthresh=1)
            ax.set_xlim(-extent,extent);ax.set_ylim(-extent,extent)
            ax.set_xlabel('Modelled centroid shift (min)');ax.set_ylabel('Predicted centroid shift (min)')
            ax.set_title(NAMES[model]+('\nOrdinary comparisons' if row==0 else '\nFull-network controls'))
            ax.grid(alpha=.2)
    fig.suptitle('Out-of-fold predictions · signed-log display, linear-target training\nFive-repeat prediction means for display; no synthetic-to-real coefficient transfer')
    fig.savefig(OUT/'matched_predictions.png',dpi=180);plt.close(fig)
    lines=['# Matched centroid predictor comparison — 3 November 2026','',
        'Reference: MATCHED-CENTROID-SETS-2026-11-03. Requested after the synthetic-selected sets were tested with whole-river holdout, which was not directly comparable to the existing notebook.','',
        '## Results','',
        '| Predictor set | Cohort | Mean R² [repeat range] | Mean MAE (min) |',
        '|---|---|---:|---:|']
    for model in MODELS:
        for cohort in ['original','segmented','all']:
            z=s[s.model.eq(model)&s.cohort.eq(cohort)]
            lines.append(f'| {NAMES[model]} | {cohort} | {z.r2.mean():.3f} [{z.r2.min():.3f}, {z.r2.max():.3f}] | {z.mae_minutes.mean():.3f} |')
    lines += ['', 'Original = ordinary unit/island/compound comparisons. Segmented = all-closed-unit full-network controls. All = equal weight to the two cohorts, then equal weight to rivers within each cohort. The all-cohort R² is not the average of cohort R²; between-cohort variation enlarges its denominator.','',
        '## Method and limits','',
        '- All 2,185 canonical comparisons from 15 rivers retained. No modelled routing reruns and no synthetic-trained coefficients.',
        '- Exact notebook compound-family partitions: five repeats, five outer folds, three inner folds. Same six 500-tree RF configurations, training weights, training-only standardization/imputation, target standardization, random seed and one-minute-floored relative-error tuning objective.',
        '- Historical nine-input predictions reused after exact numerical parity of a fresh first-fold outer refit. Two new panels require 50 outer jobs and 950 forest fits.',
        '- All ordinary domains match. For the 15 full-network controls, unchanged connectors are excluded from length/area/width rewards, while connected topology is retained. Whole-network ΔGₘ includes changes in allocation on surviving reaches. See domain_alignment.csv.',
        '- This is known-river, held-out-compound-family prediction. It does not replace the earlier whole-river-holdout result or demonstrate transfer to unseen rivers.',
        '- Historical partitions used response/type balancing. Full-network singleton controls can share constituents with training cases. Feature sets were selected after prior analysis: treat this as an exploratory benchmark, not independent confirmation.',
        '- Five repeats reuse observations; repeat ranges are split sensitivity, not independent-sample confidence intervals. No significance claim is made.',
        '- Equivalent path width spans many orders of magnitude. In the primary numerically corrected test, log1p(width) precedes the unchanged training-only scaler. This preserves order and prevents distinct widths from rounding to the same float32 value. Selection of this transformation used input precision diagnostics, not response scores. The raw-width first repeat is a separate diagnostic.',
        '- The earlier leave-river-out test used a fixed RF and signed-log preprocessing of all inputs. Differences from that test cannot be attributed to validation grouping alone.',
        '', '## Frozen new inputs','',
        'Both sets: allocated mean route-length change (m), parent allocated-length-weighted S⁻¹ᐟ², parent sum of harmonic path widths (m), and change in topology complexity log(1+route count)+log(1+internal bifurcations)+log(1+internal confluences). Fifth input: whole-network geometry-only Manning ΔGₘ, or channel planform-area change (m²).','',
        'The previous nine: route-length ratio, domain size, normalized route-spread change, width-mean ratio, width-CV ratio, entropy change, parent width dominance, parent slope descriptor, and allocated mean route-length change. The ratios retain the notebook log1p transformation.','',
        '## Files','',
        '- matched_scores.png: cohort-specific comparison with all repeat scores.',
        '- matched_predictions.png: repeated held-out prediction means, signed-log display only.',
        '- paired_changes.csv: per-repeat differences relative to the previous nine-input set.',
        '- design.json / splits.json / baseline_parity.json: provenance and reproducibility.',
        '- checkpoints/: all inner tuning scores and selected outer configurations.','']
    (OUT/'report.md').write_text('\n'.join(lines))
    print(pd.read_csv(OUT/'summary.csv').to_string(index=False))

if __name__=='__main__':main()
