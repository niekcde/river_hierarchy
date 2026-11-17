"""Append the paired real-network Manning geometry benchmark to the synthetic screen."""
from pathlib import Path
import nbformat
ROOT=Path(__file__).resolve().parents[2]
TAG='geometry-manning-paired-test-20261030'
def install():
 p=ROOT/'synthetic_experiments/notebooks/synthetic_predictor_validation.ipynb'
 n=nbformat.read(p,4);n.cells=[c for c in n.cells if TAG not in c.metadata.get('tags',[])]
 cells=[nbformat.v4.new_markdown_cell(r'''## 9. Does Manning geometry add information to the real-network RF?

**GEOMETRY-MANNING-PAIRED-2026-10-30.** Section 8 screens the synthetic candidates. This separate paired test evaluates the proposed tenth input on the **2,185 real-network comparisons, 450 groups, 15 networks** already used by the current nine-input hydrography RF. Target: **centroid shift in hours**, using the fixed-Manning runs and the existing finite 0.5% event window.

\[
\Delta G_M=\left[\sum_e L_e B_e^{2/5} f_e^{3/5}S_e^{-3/10}\right]_c-
\left[\sum_e L_e B_e^{2/5} f_e^{3/5}S_e^{-3/10}\right]_p.
\]

Both sums cover the **whole network**. Each state's allocation is independently recomputed from connectivity and width, including accumulation at confluences. Unchanged contributions cancel; changed allocation on surviving reaches remains. No calibrated roughness, depth, simulated discharge or response enters this descriptor. Its physical interpretation is conditional on wide-channel Manning hydraulics and the allocation rule.

**Paired design:** reuse the exact five repeats × five outer compound-grouped folds. In each fold, retain the nine-input model's training-only, inner-selected RF settings and add G_M; do not retune the new model. The training-only affine G_M baseline uses the same folds and network weights. Thus this tests the incremental feature under the current RF recipe, not the best possible newly tuned ten-input RF. A baseline fold is refitted and checked against its saved predictions.

These are **known-network, related-state predictions**, not held-out-network transfer. The inherited 15 full-network singleton exceptions and previously reused partitions remain limitations. Synthetic Spearman rankings are descriptive and must not be called held-out R². The tables below evaluate the real centroid response only; they do not establish peak-shift improvement. No production model or routing result is replaced.'''),
 nbformat.v4.new_code_cell('''gm_path=Path('/Volumes/SyntheticResults20261027/analysis/real_geometry_manning_20261030')
gm_design=json.loads((gm_path/'design.json').read_text())
gm_completion=json.loads((gm_path/'completion.json').read_text())
assert gm_completion['complete']
assert hashlib.sha256(json.dumps(gm_design,sort_keys=True).encode()).hexdigest()==gm_completion['fingerprint']
for path,digest in gm_completion['outputs'].items():
    assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==digest,path
gm_summary=pd.read_csv(gm_path/'summary.csv')
gm_paired=pd.read_csv(gm_path/'paired_changes.csv')
gm_models=['geometry_ratios_9','geometry_9_plus_GM','linear_GM']
gm_names={'geometry_ratios_9':'RF: nine geometry inputs','geometry_9_plus_GM':'RF: nine + Manning ΔG','linear_GM':'Manning ΔG alone (affine)'}
print('Nine existing inputs:')
display(pd.DataFrame({'predictor':gm_design['baseline_columns']}))
print('Same grouped-compound splits; RF addition uses the nine-input fold settings without retuning.')
display(gm_summary[gm_summary.model.isin(gm_models)][['model','cohort','r2_mean','r2_min','r2_max','mae_minutes_mean','relative_error_pct_mean']].round(3))
print('Paired change: positive R² is better; negative error is better. Range is across repeats, not a confidence interval.')
display(gm_paired.groupby('cohort')[['delta_r2','delta_mae_minutes','delta_relative_error_pct']].agg(['mean','min','max']).round(3))
'''),nbformat.v4.new_code_cell('''gm_plot=gm_summary[(gm_summary.cohort=='all')&gm_summary.model.isin(gm_models)].set_index('model').loc[gm_models]
fig,axes=plt.subplots(1,3,figsize=(16,4.3),constrained_layout=True)
for ax,metric,label in zip(axes,['r2','mae_minutes','relative_error_pct'],['Held-out R²','MAE (minutes)','Relative-error score (%)']):
    mean=gm_plot[metric+'_mean'];lo=gm_plot[metric+'_min'];hi=gm_plot[metric+'_max']
    ax.barh(range(3),mean,xerr=np.vstack([mean-lo,hi-mean]),color=['#64748b','#2563eb','#d97706'],capsize=4)
    ax.set(yticks=range(3),yticklabels=[gm_names[x] for x in gm_models] if ax is axes[0] else ['']*3,xlabel=label)
    ax.invert_yaxis();ax.axvline(0,color='black',lw=.7);ax.grid(axis='x',alpha=.15)
fig.suptitle('Real-network centroid prediction · identical compound-grouped splits\\nMeans and repeat ranges; equal cohort/network weighting')
display_figure(fig);plt.close(fig)
''')]
 for c in cells:c.metadata['tags']=[TAG]
 n.cells.extend(cells);nbformat.write(n,p)
if __name__=='__main__':install()
