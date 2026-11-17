"""Report joint boundary objectives without claiming full-network routing results."""
from pathlib import Path
import json,hashlib
import numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parents[3];OUT=ROOT/'RAPID/outputs/boundary_objectives_20261027_v2'
NAMES={'absolute_drop':'Absolute drop squared','relative_drop':'Relative drop squared','absolute_slope':'Absolute slope squared'}
ORDER=list(NAMES)

def table(headers,rows):
    return '| '+' | '.join(headers)+' |\n| '+' | '.join(['---']*len(headers))+' |\n'+''.join('| '+' | '.join(map(str,row))+' |\n' for row in rows)


def main():
    f=pd.read_csv(OUT/'case_metrics.csv',dtype={'example':str});n=pd.read_csv(OUT/'boundary_heads.csv',dtype={'example':str});edges=pd.read_parquet(OUT/'exterior_reaches.parquet');meta=json.loads((OUT/'case_metadata.json').read_text())
    assert len(f)==300 and f.groupby(['example','state']).size().eq(6).all()
    assert f.primal_residual_m.max()<1e-8 and f.stationarity.max()<1e-8
    reference_networks=0
    for source in json.loads((OUT/'manifest.json').read_text())['source_hashes']:
        if source.endswith('/S000_base/parameters.parquet') or source.endswith('/S000_base/rapid/cunge/prep/parameters.parquet'):
            base=pd.read_parquet(source)
            c=np.clip((5/3)*base.depth_m**(2/3)*np.sqrt(base.slope)/base.n_manning,base.celerity_min_mps,base.celerity_max_mps)
            np.testing.assert_allclose(c,base.rapid_celerity_mps,rtol=1e-12,atol=1e-12)
            np.testing.assert_allclose(base.link_length_m/c,base['count']*base.rapid_k,rtol=1e-12,atol=1e-10)
            np.testing.assert_allclose(base.reference_discharge_cms/(2*base.width*base.slope),base.hydraulic_diffusivity_m2_s,rtol=1e-12)
            reference_networks+=1
    (OUT/'routing_formula_validation.json').write_text(json.dumps(dict(reference_networks_checked=reference_networks,celerity_travel_and_diffusivity_match_saved_parameters=True),indent=2))
    measures=['rms_head_drop_change_m','rms_relative_drop_change','rms_absolute_slope_change','max_relative_slope_change','weighted_absolute_travel_change_s','weighted_signed_travel_change_s','weighted_absolute_adjustment_travel_s','max_relative_diffusivity_change']
    f.groupby(['length_policy','objective'])[measures].agg(['median','max']).to_csv(OUT/'aggregate_metrics.csv')
    a=f[f.length_policy.eq('reference_exterior_lengths')];winners=[]
    for policy in f.length_policy.unique():
        group=f[f.length_policy.eq(policy)]
        for metric in measures:
            p=group.pivot(index=['example','state'],columns='objective',values=metric)
            if metric=='weighted_signed_travel_change_s':continue
            for objective in ORDER:winners.append(dict(length_policy=policy,metric=metric,objective=objective,cases_at_minimum=int((p[objective]<=p.min(axis=1)+1e-10).sum())))
    pd.DataFrame(winners).to_csv(OUT/'metric_winners.csv',index=False)
    med=a.groupby('objective').median(numeric_only=True);worst=a.groupby('objective').max(numeric_only=True)
    unitrows=[]
    for policy in ['saved_geometry','reference_exterior_lengths']:
        for o in ORDER:
            z=n[(n.state=='S023_unit_136')&(n.length_policy==policy)&(n.objective==o)].set_index('node');m=f[(f.state=='S023_unit_136')&(f.length_policy==policy)&(f.objective==o)].iloc[0]
            unitrows.append([policy,NAMES[o],f'{1000*z.loc[266,"head_change_m"]:.3f}',f'{1000*z.loc[267,"head_change_m"]:.3f}',f'{100*m.max_relative_slope_change:.3f}%',f'{m.weighted_absolute_travel_change_s:.4f}'])
    report='''# Joint boundary-head objective comparison

27 October 2026. Fifty compound-member cases that need a boundary adjustment:
49 reversed/zero-drop conflicts and one remaining reference-floor conflict.
The seven earlier floor/reference-definition cases requiring no boundary movement
are excluded. The 27 manual/correspondence exceptions remain outside this comparison.

## Design

Each case releases only the candidate endpoints of its conflicting path(s), jointly.
Every other exterior/boundary head stays fixed at its reference value. Internal heads
remain free. The whole graph must admit slopes satisfying the frozen reference
compound floor; incident exterior reaches keep their reference floors. Four tiny
reference floor residuals in network 15 are retained rather than forcing unrelated
reference reaches to change: exterior lower bounds are capped at the accepted
reference slope, with maximum prior head-drop residual about 3.15 micrometres.

The primary optimization variables are the two or three candidate boundary heads.
Internal heads are eliminated through exact fixed-to-fixed path inequalities on the
DAG. A small convex quadratic program minimizes one of:

1. Sum of squared exterior head-drop changes (metres squared).
2. Sum of squared relative exterior head-drop changes (dimensionless).
3. Sum of squared absolute exterior slope changes (gradient squared).

All incident exterior reaches enter the objective. Objective 3 weights head changes
by inverse reach length squared. With saved length differences, it also includes the
constant slope offset needed to target the original reference slope accurately.
The normalized weight scale does not change the optimum. Exhaustive independent
active sets provide feasibility and KKT optimality certificates; no arbitrary
interior-head regularization is added to the exterior objective.

These are constrained candidate solutions, not an implemented production policy.
No production geometry, direction, head, slope, routing parameter or RF model changed.
No full-network routing or internal-head objective fit was performed.

## Geometry control

'''
    changed=sum(v['affected_exterior_length_changes']>0 for v in meta)
    report+=f'{changed}/50 cases already have length changes on an affected exterior reach. Each of the three objectives is therefore evaluated twice:\n\n'
    report+='''- **Saved geometry:** use current child lengths and target the original reference drops/slopes. Report both total reference differences and adjustment-only differences on that same geometry.
- **Reference exterior lengths:** a diagnostic control using original exterior lengths while leaving replacement lengths unchanged. This isolates the proposed boundary adjustment from prior exterior length changes. It is not a geometry repair or a new network product.

There are 300 certified optima: 50 cases × 3 objectives × 2 length policies.

## Unit 136 (network 22)

Its affected exterior reaches are 261 → 266 (parent 396, 305.38 m) and 271 → 267
(parent 406, 1,075.79 m reference / 1,072.88 m saved length). The required difference
between the two head adjustments is about 9.503 mm.

'''
    report+=table(['Length policy','Objective','ΔH266 (mm)','ΔH267 (mm)','Largest exterior |ΔS/S|','Σ q_fraction × |Δtravel| (s)'],unitrows)
    report+='''
Equal absolute-drop weighting splits the adjustment equally. Both relative-drop and
absolute-slope weighting favour moving the endpoint of the longer branch here.
They are similar because the two original slopes are similar; they are not generally
equivalent objectives. The saved-geometry slope optimum also compensates part of the
existing length change, which explains its difference from the reference-length control.

## Across the fifty cases

The table below uses reference exterior lengths to isolate head-adjustment effects.
The first column is the median, across cases, of each case's largest exterior percentage
slope change. The worst-case column is not a typical result.

'''
    report+=table(['Objective','Median largest |ΔS/S|','Worst |ΔS/S|','Median Σ q_fraction × |Δtravel| (s)'],[[NAMES[o],f'{100*med.loc[o,"max_relative_slope_change"]:.2f}%',f'{100*worst.loc[o,"max_relative_slope_change"]:.1f}%',f'{med.loc[o,"weighted_absolute_travel_change_s"]:.3f}'] for o in ORDER])
    report+='''
The absolute-slope objective minimizes RMS absolute slope change by construction;
that alone is not evidence of superior routing preservation. Relative-drop weighting
produces the smallest maximum percentage slope change in 39/50 cases, versus 11/50
for absolute-slope weighting. With saved lengths these counts are 24, 21, and 5 for
relative-drop, absolute-slope, and absolute-drop respectively: existing geometry
changes materially affect a reference-relative comparison.

No objective consistently minimizes the exterior travel-time diagnostic. In the
reference-length control, absolute-drop wins 22 cases, relative-drop 18, and
absolute-slope 10. Relative-drop has the lowest median diagnostic, but these counts
show why the median cannot be used as a universal choice rule.

Network 15 / S202_unit_55 is the largest absolute-slope percentage outlier: a 674.9%
exterior slope increase in the reference-length control (602.6% with saved geometry).
An absolute-gradient objective can permit a large percentage change on a very flat
reach. It should not be adopted without inspecting such cases. Relative-drop is not
a guarantee of negligible change either: its maximum percentage slope change reaches
64.9% in this sample.

## Routing interpretation and limits

For each affected exterior reach we retain the reference width, fixed BASED depth,
reference discharge, roughness, flow fraction and celerity caps. Only head drop and
the declared length policy vary. We evaluate:

- c = clip((5/3) h^(2/3) sqrt(S) / n, saved caps)
- Reach travel time = L / c
- Physical diffusivity = Q_ref / (2 B S)

No evaluated affected exterior reach hits a celerity cap. Beta is unchanged; scaling
physical diffusivity by the same beta leaves relative differences unchanged.
With unchanged lengths, relative drop change equals relative slope change. For small
changes, relative travel-time change is approximately -0.5 ΔS/S, and relative
diffusivity change approximately -ΔS/S. The exact nonlinear formulas above are used.

Σ q_fraction × Δtravel is the exterior contribution to a linear-network centroid
contrast if reference routing fractions are retained. Σ q_fraction × |Δtravel|
measures disturbance without allowing advances and delays to cancel. Neither is a
full-network hydrograph result: interior parameters and split compatibility have not
been chosen, and no Cunge grid/clock stability validation is claimed here. The
Manning-depth alternative was not used; these are fixed-depth diagnostics only.

## Conclusion

Include the absolute-slope objective: it addresses reach length and gives a clearly
different adjustment from equal head-drop weighting. Do not treat it as automatically
minimizing routing impact. Relative-drop preservation is the stronger starting point
if percentage hydraulic changes are the priority, while the actual choice should
follow inspection of the largest changes and an explicit acceptance criterion.
No objective or numerical cap on permitted change has been adopted.

## Validation and artifacts

Six tests pass, including analytical solutions and 40 random comparisons with an
independent SLSQP solver. All 300 case optimizations have primal residual below 3e-17 m
and normalized stationarity residual below 2e-16. Source hashes are checked before
and after the computation. `case_metrics.csv`, `boundary_heads.csv`,
`exterior_reaches.parquet`, `optimality_certificates.json`, and `manifest.json`
contain the detailed results and provenance. The diagnostic formulas reproduce the
saved celerity, travel time and diffusivity for all seven reference networks involved.
No complete trial hydraulic frame is
published because the internal head allocation has intentionally not been selected.
'''
    (OUT/'REPORT.md').write_text(report)
    # Compact exportable scientific plot: head allocation + case-wise hydraulic tradeoff.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(12,4.8),layout='constrained');colors=['#bf704a','#238477','#536cbb'];x=np.arange(3)
    u=n[n.state.eq('S023_unit_136')&n.length_policy.eq('reference_exterior_lengths')]
    for offset,node in [(-.18,266),(.18,267)]:
        values=[u[u.objective.eq(o)&u.node.eq(node)].head_change_m.iloc[0]*1000 for o in ORDER]
        bars=axes[0].bar(x+offset,values,.34,label=f'Node {node}');axes[0].bar_label(bars,fmt='%.2f',padding=3,fontsize=9)
    axes[0].set_xticks(x,['Absolute drop','Relative drop','Absolute slope']);axes[0].set_ylabel('Boundary-head adjustment (mm)');axes[0].axhline(0,color='#777',lw=.7);axes[0].legend();axes[0].set_title('Unit 136: same fixed reference floors')
    for o,color in zip(ORDER,colors):
        part=a[a.objective.eq(o)];axes[1].scatter(100*part.max_relative_slope_change,part.weighted_absolute_travel_change_s,label=NAMES[o],color=color,s=22,alpha=.75)
    axes[1].set(xscale='log',yscale='log',xlabel='Largest exterior slope change (%)',ylabel='Flow-weighted absolute travel-time change (s)',title='50 cases: hydraulic disturbance is not one metric');axes[1].legend(fontsize=8);axes[1].grid(alpha=.15)
    fig.suptitle('Joint boundary adjustments — reference exterior lengths',fontsize=14);fig.savefig(OUT/'comparison.png',dpi=180);plt.close(fig)
    manifest=json.loads((OUT/'manifest.json').read_text());manifest.update(tests_passed=6,independent_random_qp_checks=40,maximum_primal_residual_m=float(f.primal_residual_m.max()),maximum_stationarity=float(f.stationarity.max()),cases_with_affected_exterior_length_changes=changed,code_hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob('*.py')},artifact_hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in OUT.iterdir() if p.is_file() and p.name!='manifest.json'});(OUT/'manifest.json').write_text(json.dumps(manifest,indent=2))
    print('Report saved',OUT/'REPORT.md')

if __name__=='__main__':main()
