"""Effect-size screen of all 28 factorial interactions; no p-values or routing."""
from pathlib import Path
import sys,json,hashlib
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[2]
for p in (ROOT,ROOT/'RAPID/src',ROOT/'synthetic_experiments/src'):sys.path.insert(0,str(p))
from synthetic_experiments.full_grid_review import load_analysis

def build():
    manifest,d,out=load_analysis()
    pairs=pd.read_csv(out/'pair_interactions.csv');partition=pd.read_csv(out/'variance_partition.csv')
    keys=['method','child_state','response']
    pairs=pairs.merge(partition[keys+['variance_minutes_squared']],on=keys,validate='many_to_one')
    pairs['interaction_rms_minutes']=np.sqrt(pairs.variance_minutes_squared*pairs.variance_pct/100)
    rows=[]
    for (method,response,a,b),g in pairs.groupby(['method','response','factor_a','factor_b']):
        valid=g[g.response_variation_resolved]
        rows.append(dict(method=method,response=response,factor_a=a,factor_b=b,n_resolved_treatments=len(valid),mean_variance_pct=valid.variance_pct.mean(),median_variance_pct=valid.variance_pct.median(),max_variance_pct=valid.variance_pct.max(),median_rms_minutes=valid.interaction_rms_minutes.median(),max_rms_minutes=valid.interaction_rms_minutes.max(),n_rms_above_one_minute=int(valid.interaction_rms_minutes.gt(1).sum())))
    spectrum=pd.DataFrame(rows)
    spectrum['variance_rank']=spectrum.groupby(['method','response']).mean_variance_pct.rank(ascending=False,method='min')
    spectrum['magnitude_rank']=spectrum.groupby(['method','response']).median_rms_minutes.rank(ascending=False,method='min')
    pairs.to_csv(out/'interaction_magnitudes.csv',index=False);spectrum.to_csv(out/'interaction_spectrum.csv',index=False)
    stability=[]
    for response in spectrum.response.unique():
        a=spectrum[(spectrum.method=='fixed_manning')&(spectrum.response==response)].set_index(['factor_a','factor_b'])
        b=spectrum[(spectrum.method=='variable_manning')&(spectrum.response==response)].set_index(['factor_a','factor_b']).reindex(a.index)
        stability.append(dict(response=response,variance_rank_spearman=a.mean_variance_pct.corr(b.mean_variance_pct,method='spearman'),magnitude_rank_spearman=a.median_rms_minutes.corr(b.median_rms_minutes,method='spearman')))
    pd.DataFrame(stability).to_csv(out/'interaction_rank_stability.csv',index=False)
    print(spectrum[spectrum.method.eq('variable_manning')].sort_values(['response','variance_rank']).groupby('response').head(8).round(3).to_string(index=False))
    print(stability)
    (out/'interaction_screen_provenance.json').write_text(json.dumps(dict(reference='SYN-INTERACTION-SPECTRUM-2026-10-28',analysis_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),inputs={name:hashlib.sha256((out/name).read_bytes()).hexdigest() for name in ['pair_interactions.csv','variance_partition.csv']},interpretation='Exact balanced-grid effect sizes; no stochastic replication, no p-value claims. One-minute threshold is illustrative, not a significance test.'),indent=2))


def install_notebook_section():
    import nbformat
    path=ROOT/'synthetic_experiments/notebooks/synthetic_factor_interactions.ipynb'
    nb=nbformat.read(path,as_version=4);tag='interaction-spectrum-20261028'
    nb.cells=[c for c in nb.cells if tag not in c.metadata.get('tags',[])]
    text=r'''## 5. Choose interactions across the full spectrum

This screen includes **all 28 pairs**. It reports the mean percentage of variance
and the median interaction RMS magnitude across resolved collapse treatments:
$\mathrm{RMS}_{ij}=\sqrt{\mathrm{Var}(f_{ij})}$, in minutes.
The two axes distinguish relative importance from the absolute size of the timing effect.
Top labels are selected by variance rank, rather than by inspecting attractive plots.
The table also retains maximum magnitudes to expose treatment-specific effects.

There are no stochastic replicates or observation-error model in this factorial.
The decomposition is exact for the tested grid; conventional significance p-values
would require assumptions not established here. These are **effect-size and
cross-method consistency checks**, not inference to all rivers. The illustrative
one-minute count in the CSV is not a statistical significance threshold.
Higher-order interactions remain in the previously reported remainder.

`METHOD` below selects the view; all three methods are retained in the saved table.
'''
    cell='''METHOD='variable_manning'
spectrum=table('interaction_spectrum')
rank_stability=table('interaction_rank_stability')
labels={'length_multiplier':'Branch length','total_length_m':'Network scale','inner_fraction':'Inner allocation','outer_a_fraction':'Outer allocation','reference_slope':'Slope','total_width_m':'Width','internal_location':'Location','branch_width_factor':'Width expansion'}
fig,axes=plt.subplots(1,2,figsize=(13,5),constrained_layout=True)
for ax,response in zip(axes,RESPONSES):
    g=spectrum[(spectrum.method==METHOD)&(spectrum.response==response)]
    ax.scatter(g.median_rms_minutes,g.mean_variance_pct,s=40,alpha=.7)
    for _,r in g.nsmallest(4,'variance_rank').iterrows():
        ax.annotate(labels[r.factor_a]+' × '+labels[r.factor_b],(r.median_rms_minutes,r.mean_variance_pct),xytext=(5,-18 if r.variance_rank==4 else 10),textcoords='offset points',fontsize=8)
    ax.set(xlabel='Median interaction RMS across treatments (min)',ylabel='Mean interaction variance contribution (%)',title=RESPONSES[response])
    ax.set_xlim(0,max(g.median_rms_minutes)*1.7);ax.set_ylim(0,max(g.mean_variance_pct)*1.2);ax.grid(alpha=.15)
fig.suptitle(LABELS[METHOD]+' · all 28 interactions')
show(fig,'interaction_effect_size_spectrum')
display(rank_stability.round(3))
display(spectrum[spectrum.method==METHOD].sort_values(['response','variance_rank']).groupby('response').head(8)[['response','factor_a','factor_b','mean_variance_pct','median_rms_minutes','max_rms_minutes','n_resolved_treatments']].round(3))
'''
    for c in [nbformat.v4.new_markdown_cell(text),nbformat.v4.new_code_cell(cell)]:
        c.metadata['tags']=[tag];nb.cells.append(c)
    nbformat.write(nb,path)

if __name__=='__main__':
    build()
    install_notebook_section()
