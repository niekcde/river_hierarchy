"""Scores, paired descriptor tests and residual diagnostics for the frozen panels."""
from pathlib import Path
import json,hashlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from run_comparison import OUT,PANELS

def scores(g,balanced_cohorts=False):
    y=g.centroid_shift_hours.to_numpy(float)*60
    pred=g.predicted_hours.to_numpy(float)*60
    if balanced_cohorts:
        counts=g.groupby('cohort').example.nunique()
        w=1/g.groupby(['cohort','example']).example.transform('size').to_numpy()/g.cohort.map(counts).to_numpy()
    else:w=1/g.groupby('example').example.transform('size').to_numpy()
    error=pred-y;mean=np.average(y,weights=w);sst=np.sum(w*(y-mean)**2)
    return dict(n=len(g),groups=g.example.nunique(),r2=1-np.sum(w*error**2)/sst if sst>1e-20 else np.nan,
        mae_minutes=np.average(abs(error),weights=w),rmse_minutes=np.sqrt(np.average(error**2,weights=w)))

def build():
    design=json.loads((OUT/'design.json').read_text())
    fp=hashlib.sha256(json.dumps(design,sort_keys=True).encode()).hexdigest()
    paths=sorted((OUT/'checkpoints').rglob('completion.json'));assert len(paths)==design['jobs']
    parts=[]
    for path in paths:
        receipt=json.loads(path.read_text());assert receipt['fingerprint']==fp
        file=path.parent/'predictions.parquet'
        assert hashlib.sha256(file.read_bytes()).hexdigest()==receipt['prediction_sha256']
        parts.append(pd.read_parquet(file))
    p=pd.concat(parts,ignore_index=True)
    keys=['dataset','strategy','support','repeat','model','example','state']
    assert not p.duplicated(keys).any()
    p.to_parquet(OUT/'predictions.parquet',index=False)
    rows=[]
    for group,g in p.groupby(['dataset','strategy','support','repeat','model']):
        dataset,strategy,support,repeat,model=group
        subsets=[('all',g)] if dataset=='synthetic' else [('ordinary',g[g.cohort.eq('original')]),('full_network',g[g.cohort.eq('segmented')]),('all',g)]
        for population,z in subsets:
            rows.append(dict(dataset=dataset,strategy=strategy,support=support,repeat=repeat,model=model,population=population,
                **scores(z,dataset=='real' and population=='all')))
    s=pd.DataFrame(rows);s.to_csv(OUT/'repeat_scores.csv',index=False)
    group=['dataset','strategy','support','population','model']
    summary=s.groupby(group).agg(n=('n','first'),repeats=('repeat','nunique'),r2=('r2','mean'),r2_min=('r2','min'),r2_max=('r2','max'),
        mae_minutes=('mae_minutes','mean'),rmse_minutes=('rmse_minutes','mean')).reset_index()
    summary.to_csv(OUT/'summary.csv',index=False)
    comparisons=[('revised9','old9','CV difference replaces CV ratio'),
                 ('seven_with_G','five_with_G','Add entropy and dominance, with G'),
                 ('seven_without_G','five_without_G','Add entropy and dominance, without G'),
                 ('revised9','revised_without_allocation7','Retain entropy and dominance in revised nine'),
                 ('revised9','revised_without_spread8','Retain normalized route-spread change')]
    paired=[]
    for a,b,label in comparisons:
        cols=['dataset','strategy','support','population','repeat']
        z=s[s.model.eq(a)].merge(s[s.model.eq(b)],on=cols,suffixes=('_a','_b'),validate='one_to_one')
        for r in z.itertuples():
            assert r.n_a==r.n_b
            paired.append(dict(dataset=r.dataset,strategy=r.strategy,support=r.support,population=r.population,repeat=r.repeat,
                contrast=label,model_a=a,model_b=b,delta_r2=r.r2_a-r.r2_b,delta_mae_minutes=r.mae_minutes_a-r.mae_minutes_b))
    pd.DataFrame(paired).to_csv(OUT/'paired_changes.csv',index=False)
    # Per-river error contributions retain repeat-wise weighting and expose
    # whether aggregate improvements are confined to one difficult network.
    real=p[p.dataset.eq('real')].copy()
    real['absolute_error_minutes']=60*abs(real.predicted_hours-real.centroid_shift_hours)
    real['squared_error_minutes']=(60*(real.predicted_hours-real.centroid_shift_hours))**2
    rg=['strategy','support','model','repeat','cohort','example']
    river=real.groupby(rg).agg(n=('state','size'),mae_minutes=('absolute_error_minutes','mean'),mse_minutes=('squared_error_minutes','mean')).reset_index()
    denom=river.groupby(['strategy','support','model','repeat','cohort']).mse_minutes.transform('sum')
    river['cohort_sse_share']=river.mse_minutes/denom
    river.to_csv(OUT/'river_error_contributions.csv',index=False)
    real.sort_values('absolute_error_minutes',ascending=False).groupby(['strategy','model','repeat']).head(10).to_csv(OUT/'largest_errors.csv',index=False)
    syn=pd.read_parquet(OUT/'synthetic_input.parquet')
    controls=['length_multiplier','outer_a_fraction','inner_fraction','total_width_m','reference_slope','total_length_m','internal_location','branch_width_factor']
    z=p[p.dataset.eq('synthetic')].merge(syn[['example','state']+controls],on=['example','state'],validate='many_to_one')
    z['abs_error_minutes']=60*abs(z.predicted_hours-z.centroid_shift_hours)
    z['squared_error_minutes']=(60*(z.predicted_hours-z.centroid_shift_hours))**2
    control_records=[]
    for control in controls:
        a=z.groupby(['support','model',control],dropna=False).agg(n=('state','size'),mae_minutes=('abs_error_minutes','mean'),mse_minutes=('squared_error_minutes','mean')).reset_index().rename(columns={control:'value'})
        a['control']=control;control_records.append(a)
    pd.concat(control_records,ignore_index=True).to_csv(OUT/'synthetic_control_residuals.csv',index=False)
    z.groupby(['support','model','length_multiplier','outer_a_fraction']).agg(n=('state','size'),mae_minutes=('abs_error_minutes','mean'),mse_minutes=('squared_error_minutes','mean')).reset_index().to_csv(OUT/'length_allocation_residuals.csv',index=False)
    # Matrix directly answers the user's strategy-by-predictor-set question.
    panel_order=list(PANELS)
    matrix=summary[(summary.dataset.eq('real')&summary.population.eq('ordinary'))|
                   summary.dataset.eq('synthetic')].copy()
    matrix['row']=np.where(matrix.dataset.eq('real'),'Real / '+matrix.strategy,'Synthetic / '+matrix.support)
    table=matrix.pivot(index='row',columns='model',values='r2').reindex(columns=panel_order)
    fig,ax=plt.subplots(figsize=(14,5),layout='constrained')
    values=np.ma.masked_invalid(table.to_numpy());im=ax.imshow(values,cmap='RdYlGn',vmin=-1,vmax=1,aspect='auto')
    for i in range(len(table)):
        for j in range(len(panel_order)):
            v=table.iloc[i,j]
            ax.text(j,i,'—' if pd.isna(v) else f'{v:.3f}',ha='center',va='center',fontsize=9)
    ax.set_xticks(range(len(panel_order)),panel_order,rotation=30,ha='right')
    ax.set_yticks(range(len(table)),table.index)
    ax.set_title('Fixed-depth Manning · same fixed RF configuration\nReal: ordinary comparisons; synthetic old nine uses common valid support')
    fig.colorbar(im,ax=ax,label='Held-out R² (colour clipped at −1; labels show actual values)')
    fig.savefig(OUT/'comparison_matrix.png',dpi=180);plt.close(fig)
    lines=['# Streamlined synthetic–real predictor comparison','',
        'Reference: STREAMLINED-PREDICTORS-2026-11-03. Completed fixed-depth Manning, centroid-shift comparison. Each dataset is fitted separately.','',
        '## What was fixed','',design['model'],'',design['validation'],'',design['coverage'],'',
        'No features were selected based on this experiment’s scores. The old nine remain the reference; seven-input panels add entropy and dominance to the synthetic-selected five. Two joint removal tests assess the revised nine. Variable-depth Manning is reserved for sensitivity.','',
        '## Results','',
        '| Dataset | Validation | Support | Population | Set | R² | MAE (min) |',
        '|---|---|---|---|---|---:|---:|']
    for r in summary.itertuples():lines.append(f'| {r.dataset} | {r.strategy} | {r.support} | {r.population} | {r.model} | {r.r2:.3f} | {r.mae_minutes:.3f} |')
    lines+=['','## How to interpret','',
        '- Compare original and revised nine on common synthetic support; all other primary synthetic panels use the full population. Different supports must not be ranked as if they were identical.',
        '- Use the old-nine RF fitted HERE as the performance reference for this fixed-configuration test. Earlier nested-tuned scores use a different procedure and remain historical context.',
        '- Positive paired delta R² and negative paired delta MAE favour the first model in paired_changes.csv. Both models use identical held-out rows within each contrast.',
        '- Main real comparison is the strategy-by-set matrix. Ordinary and full-network cohorts are additionally reported because pooled R² can hide failures. Subgroup results do not define different validation schemes.',
        '- Inspect river_error_contributions.csv and largest_errors.csv before attributing an aggregate improvement to a general mechanism. No outliers were excluded.',
        '- Inspect synthetic_control_residuals.csv and length_allocation_residuals.csv for errors under changing controls. These are residual diagnostics, not new feature selection or proof of causality.',
        '- Response integration differs: saved synthetic targets use the full converged excess-volume pulse; saved real targets use the finite 0.5% event window. Panels are compared on identical targets WITHIN each dataset, but cross-dataset differences cannot be assigned solely to network complexity. In particular, an identity for full-pulse centroid travel time need not hold for a truncated-window centroid. This experiment does not change previously agreed event definitions.',
        '- Repeated-fold ranges describe split sensitivity, not confidence intervals. Synthetic folds test interpolation across parameter combinations, not new topology classes. Previous whole-population screens and inspected real results make this an exploratory reconciliation.',
        '', '## Frozen sets','']
    for model,cols in PANELS.items():lines.append(f'- **{model}** ({len(cols)}): '+', '.join(f'`{c}`' for c in cols))
    lines+=['','## Equations','',
        'Normalized route-spread change = (σ_L,child − σ_L,parent) / μ_L,parent. Width-CV difference = CV_W,child − CV_W,parent, with CV_W = σ_W / mean(W), using the existing length-weighted reach-width moments. Zero width CV is valid. Parent-zero CV makes the historical child/parent ratio undefined.',
        '', 'A favourable result supports these frozen representations under this model and these validation schemes. It does not prove every retained variable necessary, establish a universal law, or constitute independent validation after feature discovery.','']
    (OUT/'report.md').write_text('\n'.join(lines))
    (OUT/'report_provenance.json').write_text(json.dumps(dict(script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),input_fingerprint=fp),indent=2))
    print(summary.to_string(index=False))

if __name__=='__main__':build()
