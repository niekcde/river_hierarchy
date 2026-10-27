"""Targeted notebook updates preserving user-authored mathematical notation."""
from pathlib import Path
from datetime import datetime
import sys,json,shutil,hashlib
import numpy as np
import pandas as pd
import nbformat as nb

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from hierarchy_level_definition.metrics.equivalent_resistance import compute_K_chezy,solve_bubble_equivalent_K
OUT=ROOT/'RAPID/outputs/real_geometry_examples_20261012'
cases=json.loads((OUT/'cases.json').read_text())
res=[];splitrows=[]

for case,meta in cases.items():
    frames={state:pd.read_csv(OUT/f"{meta['tag']}_{state}_reaches.csv") for state in ['parent','child']}
    allf=pd.concat(list(frames.values()))
    alpha=.65;coefficient=allf.depth_m/allf.width**alpha
    np.testing.assert_allclose(coefficient,coefficient.iloc[0],rtol=1e-10)
    # Same reference scales on both graphs: the normalization cancels in Kc/Kp.
    L0=frames['parent'].link_length_m.sum();W0=np.average(frames['parent'].width,weights=frames['parent'].link_length_m)
    calculated={}
    for state,f in frames.items():
        normalized=(f.link_length_m/L0)/(f.width/W0)**(2+3*alpha)
        physical=compute_K_chezy(f.link_length_m,f.width,22.,f.width/f.depth_m,geometry_method='wide',g=9.81)
        solved={}
        for name,k in [('geometry',normalized),('current_depth_chezy',physical),('constant_aspect_ratio',(f.link_length_m/L0)/(f.width/W0)**5)]:
            edges=pd.DataFrame(dict(edge_id=f.id_link,u=f.id_us_node,v=f.id_ds_node,K_e=k))
            result=solve_bubble_equivalent_K(None,edges,meta['source'],meta['sink'],solver_backend='energy_sparse',solver_cache_dir=None)
            if not result['solver_success'] or result['residual_relative']>1e-8:raise ValueError('Resistance continuity check failed')
            solved[name]=float(result['K_eq'])
            result['edge_flows'].to_csv(OUT/f"{meta['tag']}_{state}_{name}_resistance_edges.csv",index=False)
        calculated[state]=solved
    ratio=calculated['child']['geometry']/calculated['parent']['geometry']
    physical_ratio=calculated['child']['current_depth_chezy']/calculated['parent']['current_depth_chezy']
    np.testing.assert_allclose(ratio,physical_ratio,rtol=1e-8)
    archived=pd.read_csv(Path(meta['membership_path']).parent/'bubble_equivalent_state_comparison.csv').iloc[0]
    constant_aspect_ratio=calculated['child']['constant_aspect_ratio']/calculated['parent']['constant_aspect_ratio']
    np.testing.assert_allclose(constant_aspect_ratio,archived.Lambda_K,rtol=1e-8)
    res.append(dict(case=case,alpha=alpha,width_power=2+3*alpha,shared_depth_coefficient=coefficient.iloc[0],
        parent_K_normalized=calculated['parent']['geometry'],child_K_normalized=calculated['child']['geometry'],
        child_over_parent=ratio,relative_change_pct=100*(ratio-1),resistance_log=-np.log10(ratio),
        parent_K_current_depth=calculated['parent']['current_depth_chezy'],child_K_current_depth=calculated['child']['current_depth_chezy'],
        constant_aspect_ratio=constant_aspect_ratio,archived_ratio_reference_only=archived.Lambda_K))
    edges=pd.read_csv(OUT/f"{meta['tag']}_parent_edges.csv")
    for b,g in edges.groupby('u'):
        if len(g)<2:continue
        p=g.p_outgoing.to_numpy();r=g.r_edge.sum();H=-p@np.log(p)
        splitrows.append(dict(case=case,bifurcation=int(b),r_b=r,
            outgoing_fractions=' / '.join(f'{x:.5f}' for x in p),dominance=p.max(),
            local_entropy=H,normalized_entropy=H/np.log(len(p)),entropic_effective_branches=np.exp(H),
            inverse_simpson_branches=1/(p@p),route_entropy_contribution=r*H,
            split_impact=r*(1-p@p)))
pd.DataFrame(res).to_csv(OUT/'resistance_current_geometry.csv',index=False)
pd.DataFrame(splitrows).to_csv(OUT/'split_explanation.csv',index=False)
(OUT/'resistance_provenance.json').write_text(json.dumps(dict(
    scope='Static equivalent-resistance calculation only; no RAPID routing, recalibration, or production writes.',
    geometry_law='wide Chezy; depth=a*width^0.65; common a and Cz across each pair; normalized edge weight=(L/L0)/(W/W0)^3.95',
    validation='Current saved depths verify a common a; physical current-depth Chezy and geometry-only normalized ratios agree at rtol=1e-8; relative continuity residual <1e-8.',
    qualification='Geometry-derived ratio conditional on fixed depth-width exponent and wide-section Chezy law. Not assumption-free. Archived ratios retained only to identify provenance differences.',
    source_hashes={str(OUT/f"{meta['tag']}_{state}_reaches.csv"):hashlib.sha256((OUT/f"{meta['tag']}_{state}_reaches.csv").read_bytes()).hexdigest() for meta in cases.values() for state in ['parent','child']}
),indent=2))

p=ROOT/'RAPID/notebooks/geometry_metric_worked_examples_20261012.ipynb'
backup=OUT/('geometry_metric_before_ratio_review_'+datetime.now().strftime('%H%M%S')+'.ipynb')
shutil.copy2(p,backup);n=nb.read(p,as_version=4)

notes={
 'width_p10':'**Priority: exploratory only.** No clear added mechanism beyond width spread/constriction measures has been established. Keep out of the initial shortlist. For a two-reach unit these quantiles often reduce to choosing one reach width.',
 'width_p90':'**Priority: exploratory only**, for the same reason as width P10. This is not a finding of zero predictive value; it is a choice to start with more interpretable summaries.',
 'route_length_quantiles':'**Priority: exploratory only.** With two or five discrete routes, P10/P90 can mainly pick individual paths. Mean and SD are the first route-distribution candidates to review.',
 'junction_excess':'**Priority: redundancy diagnostic, not a separate initial predictor.** In the 15 base graphs, 1,740/1,746 branching nodes are binary (99.66%). For binary splits, junction excess equals bifurcation count. It differs only at the few higher-degree split nodes.',
 'cycle_rank':'**Priority: redundancy diagnostic.** On these connected single-sink DAGs it equals junction excess exactly; do not select both.',
 'bifurcation_density':'**Priority: secondary.** Simple = 0.2675 splits/km of unique channel; compound = 0.2547 splits/km. Similar values indicate similar splitting per channel length, although the counts are 1 and 4. Two examples cannot establish predictive usefulness; start with count and physical length separately. Density is their ratio, not an independent ingredient.',
 'route_length_mean':r'**This is exactly $\mu_L$.** `route_length_mean` is the code name for mean travelled distance, not another metric.',
 'route_entropy':r'''**Plain interpretation:** total uncertainty about which complete route an allocated particle follows. A split that is almost always visited contributes more than one rarely visited.

$$H_{route}=\sum_b r_bH_b=-\sum_kP_k\ln P_k.$$

$r_b$ is the fraction reaching bifurcation $b$, conditional on entering the unit (confluences already included). $H_b=-\sum_jp_{bj}\ln p_{bj}$ describes balance at that split. Local split entropy measures balance; route entropy combines balance **and the number of split encounters**. Keep route entropy as a secondary diagnostic initially, since this combination is less direct to interpret.''',
 'dominance_mean':r'''At one split, dominance is the largest outgoing share: 90:10 gives 0.9; 50:50 gives 0.5.

$$\bar d=\frac{\sum_b r_b\max_jp_{bj}}{\sum_b r_b}.$$

This is the average largest outgoing share at an encountered bifurcation. Simple unit: **0.88089** (88.1:11.9 at its only split). Small compound: **0.78162**, a visit-weighted average over four split nodes. It is not the probability of one dominant complete route. For a child with no split it is undefined, rather than zero or a forced value of one.''',
 'entropic_branches_mean':r'''Translate local entropy back to an effective number of branches: $N_{1,b}=\exp(H_b)$. A 50:50 split behaves like 2 equally used branches; a 90:10 split like 1.384. Values near one indicate strong dominance.

$$\bar N_1=\frac{\sum_b r_b\exp(H_b)}{\sum_b r_b}.$$

Simple parent: **1.44073**; small compound parent: **1.64914**. These are averages of effective branches **at split encounters**, not total route counts. With no child bifurcations, there is no encounter to average over: NaN is appropriate.

This uses the effective-number transformation underlying [the entropic braiding index](https://agupubs.onlinelibrary.wiley.com/doi/abs/10.1029/2022GL099681). Published eBI uses channel fractions at cross-sections. We apply it at bifurcations and average over visits, so this is an **adaptation**, not a measured cross-sectional eBI. Do not call exponentiated route entropy a simultaneous channel count.''',
 'entropic_branches_sd':'Spread of those local effective branch counts: do the bifurcations all have similar balance, or do some strongly favour one branch while others divide evenly? This is optional variability around the mean, not another mean braiding index.',
 'path_harmonic_width_min':'Among the listed complete routes, choose the smallest harmonic effective width. It identifies the narrowest effective route, even if that route carries little allocation. Keep as a secondary constriction diagnostic.',
 'path_harmonic_width_max':'Among the listed complete routes, choose the largest harmonic effective width. It identifies the widest effective route, not necessarily the most probable route. Compare its added information against simpler width summaries.',
 'upstream_distance':'Parent-context variable only: how far the allocated flow travels from the network inlet before reaching this domain. No child/parent ratio is shown because this teaching table has only the parent context.',
 'downstream_distance':'Parent-context variable only: expected remaining distance after leaving this domain to the network outlet. This is not a distance inside the unit.',
 'network_entry_exposure':'Parent-context variable only: fraction of the whole-network allocation reaching the domain inlet. In contrast, r_b starts from one at the local domain inlet.',
 'modified_length_fraction':'Parent-context variable only: unique length removed/modified by this full unit collapse divided by unique parent-network length. This is already a fraction; do not divide by a nonexistent child version.',
}

intro=r'''## Reading the comparisons: ratios first, parent structure retained

For a nonnegative metric with positive parent value, show $R_X=X_c/X_p$ and $100(R_X-1)$. Thus 0.8 means the child retains 80%, a 20% decrease. Positive quantities may optionally use $\ln R_X$, but no log is defined when the child is zero. Never add an arbitrary epsilon to create one.

**A ratio alone is insufficient for complete collapses:** bifurcation count 1→0 and 4→0 both give ratio zero; both route SDs also collapse to zero. Retain parent magnitude to distinguish the removed structure. The tables retain parent and child values. Signed skewness uses a difference; conditional split means remain undefined for a no-split child. Context metrics are parent-only.

Start with mean travelled distance ($\mu_L$), route spread, width size/spread, one split-balance descriptor, bifurcation count/expected split encounters, domain exposure/position, and the explicitly parameterized resistance ratio. Deprioritize width/route quantiles and bifurcation density; do not select algebraically redundant topology measures together. This is a discussion shortlist, not a fitted selection result.

Use dominance **or** entropic effective branches initially. They describe related balance information; normalized entropy and inverse-Simpson count are additional alternatives, not automatically additional predictors.
'''
insert_at=next(i for i,c in enumerate(n.cells) if c.cell_type=='markdown' and c.source.startswith('## Each metric:'))
n.cells.insert(insert_at,nb.v4.new_markdown_cell(intro))

split_md=r'''## Split metrics in one small table

Read the columns in this order: **how much reaches the split** ($r_b$), **how it divides** ($p_{bj}$), **largest share** (dominance), **effective branch count** ($\exp H_b$). These are different descriptions of the same splits, not four unrelated mechanisms.

The real compound has four split nodes, but an expected 2.7626 encounters per allocated route: not every route visits every bifurcation. The visit-weighted means divide by $\sum_b r_b$; route entropy instead sums the contributions without that normalization.
'''
where=next(i for i,c in enumerate(n.cells) if c.cell_type=='markdown' and c.source.startswith('### route_entropy'))
n.cells[where:where]=[nb.v4.new_markdown_cell(split_md),nb.v4.new_code_cell("splits=pd.read_csv(OUT/'split_explanation.csv')\ndisplay(splits.round(5))")]

helper='''
def metric_comparison(metric):
    shown=values[values.metric.eq(metric)].pivot(index='case',columns='state',values='value').reindex(cases)
    if metric in {'upstream_distance','downstream_distance','network_entry_exposure','modified_length_fraction'}:
        return shown[['parent']].rename(columns={'parent':'parent_context'})
    for state in ['parent','child']:
        if state not in shown:shown[state]=np.nan
    if metric=='route_length_quantiles':return shown[['parent','child']]
    shown=shown.apply(pd.to_numeric,errors='coerce')
    if metric in {'route_length_skew','log1p_paths','dominance_mean','dominance_sd','split_entropy_mean','split_entropy_sd','split_exposure_mean','split_exposure_sd'}:
        shown['child_minus_parent']=shown.child-shown.parent
    else:
        valid=shown.parent.gt(0)&shown.child.ge(0)
        shown['child_over_parent']=(shown.child/shown.parent.where(valid)).where(valid)
        shown['relative_change_pct']=100*(shown.child_over_parent-1)
    return shown
'''
for c in n.cells:
    if c.cell_type=='code' and 'cases=json.loads' in c.source:c.source+=helper
    if c.cell_type=='code' and c.source.startswith("metric='"):
        first=c.source.splitlines()[0];c.source=first+'\ndisplay(metric_comparison(metric))'
    if c.cell_type=='markdown' and c.source.startswith('### '):
        key=c.source.splitlines()[0][4:]
        if key in notes:c.source+='\n\n'+notes[key]
    if c.cell_type=='markdown' and c.source.startswith('## Resistance stays distinct'):
        c.source=r'''## Resistance restored: matched-domain ratio under the stated geometry law

For a wide section, $A_e=W_eh_e$, $R_{h,e}\approx h_e$. The repository's Chezy resistance is

$$K_e=\frac{L_e}{C_z^2g\,W_e^2h_e^3}.$$

With $h_e=aW_e^{\alpha}$ and a common $a,C_z$ in parent and child,

$$K_e=\frac{1}{C_z^2ga^3}\frac{L_e}{W_e^{2+3\alpha}}.$$

The common prefactor scales the entire network's equivalent resistance and therefore cancels in $K_c/K_p$. For the current $\alpha=0.65$, the width power is **3.95**. Both real examples verify a common $a$ across their parent and child reaches to numerical precision.

We can therefore calculate a **geometry-derived resistance ratio conditional on this fixed depth–width/Chezy law** using normalized edge weights

$$\widetilde K_e=\frac{L_e/L_0}{(W_e/W_0)^{3.95}},\qquad R_K=\frac{\widetilde K_c}{\widetilde K_p},\qquad resistance\_log=-\log_{10}R_K.$$

$L_0,W_0$ are the same on both sides. The static unit-head network calculation includes series/parallel topology and internal confluence continuity; it does **not** force the width-proxy split probabilities used by route metrics. Resistance is thus not a harmonic-width substitute. No hydrograph, calibrated Manning n, or celerity enters this geometry ratio. The exponent and wide-section law are explicit modeling assumptions and require sensitivity assessment.

The normalized-geometry result was checked against a calculation using the saved current depths and $C_z=22$; ratios agree at relative tolerance $10^{-8}$ and continuity residuals are below $10^{-8}$. This is a static equivalent-resistance calculation, not a RAPID reroute or recalibration. No production files were changed.

Archived selected-domain ratios are printed separately for provenance. They differ from the current-depth-law calculation and must not be mixed into the new table as if they were the same version.
'''

n.cells.append(nb.v4.new_code_cell("resistance=pd.read_csv(OUT/'resistance_current_geometry.csv').set_index('case')\ndisplay(resistance[['parent_K_normalized','child_K_normalized','child_over_parent','relative_change_pct','resistance_log']])\nprint('Archived ratios, reference only:')\ndisplay(resistance[['archived_ratio_reference_only']])"))
n.cells.append(nb.v4.new_markdown_cell('''The metrics above remain local-domain examples, not a rebuilt full-population local feature dataset. Parent-only context metrics remain separate. The earlier whole-network snapshot is unchanged. User-edited mathematical formulas elsewhere in this notebook have been retained.'''))
for c in n.cells:
    if c.cell_type=='code':c.outputs=[];c.execution_count=None
n.metadata.pop('execution_validation',None)
nb.validate(n);nb.write(n,p)
print(pd.DataFrame(res)[['case','child_over_parent','relative_change_pct','resistance_log']].to_string(index=False))
print('Notebook updated:',p,'\nPreserved input:',backup)
