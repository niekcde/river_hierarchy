"""Build two teaching/review notebooks; preserve the original foundation snapshot."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import nbformat as nb

ROOT=Path(__file__).resolve().parents[3]
SOURCE=ROOT/'RAPID/outputs/geometry_response_foundation_20261012T120140'
OUT=ROOT/'RAPID/outputs/response_metric_worked_examples_20261012'
OUT.mkdir(exist_ok=True)

def notebook(title):
    return [nb.v4.new_markdown_cell('# '+title)]
def md(c,s): c.append(nb.v4.new_markdown_cell(s))
def code(c,s): c.append(nb.v4.new_code_cell(s))
def save(c,name):
    p=ROOT/'RAPID/notebooks'/name
    nb.write(nb.v4.new_notebook(cells=c,metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'}}),p)
    print(p)

setup=f'''from pathlib import Path
import sys, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display
ROOT=Path({str(ROOT)!r}); SOURCE=Path({str(SOURCE)!r}); OUT=Path({str(OUT)!r})
states=pd.read_parquet(SOURCE/'canonical_states.parquet')
pairs=pd.read_parquet(SOURCE/'canonical_pairs.parquet')
'''
h=notebook('Hydrograph response variables — definitions and worked examples')
md(h,r'''Read this notebook before feature selection. A **response (y, dependent variable)** describes the hydrograph change. Geometry supplies predictors (x, independent variables). We will not choose a large response set by default.

Initial core: (1) reviewed crest-time shift in seconds, with sampled maximum separately retained; (2) total peak-discharge percentage change. Next review one widening measure and one bulk-timing measure after agreeing the event baseline/window. Alternatives below are a catalogue for discussion, not sixteen prediction targets. No model is fitted.

All real examples compare a canonical child against the same network's base. Negative timing means earlier; negative height change means lower. Network 54 uses the two-lobe centre as its general timing value and remains a separate focused case with left/right timing and heights. No recalibration.''')
code(h,setup)
md(h,r'''## First verify the disputed later peaks

The printed UTC timestamps label centres of maximum 60-second mean-discharge intervals. `peak_total_change_pct` is **height**, not time. `sampled_peak_shift_minutes` is a signed time difference; it is neither a percentage nor an absolute-value operation. The zoom plots show each curve's discharge deficit from its own maximum to make very flat crests visible without conflating height and timing.''')
code(h,'''audit=[]
fig,axes=plt.subplots(2,2,figsize=(12,7))
for row,(ex,sid) in enumerate([('02','S013_unit_20'),('15','S127_unit_205')]):
    br=states[(states.example==ex)&states.is_base].iloc[0]
    cr=states[(states.example==ex)&(states.state_id==sid)].iloc[0]
    waves=[pd.read_parquet(r.outlet_parquet) for r in [br,cr]]
    t0=waves[0].time_utc.iloc[waves[0].q_outlet_cms.argmax()]
    record=dict(example=ex,state_id=sid)
    for label,w in zip(['parent','child'],waves):
        i=int(w.q_outlet_cms.argmax());t=w.time_utc.iloc[i];q=w.q_outlet_cms.iloc[i]
        record[label+'_peak_utc']=str(t);record[label+'_peak_cms']=q
        x=(w.time_utc-t0).dt.total_seconds()/60
        event=x.abs()<=24*60; zoom=x.abs()<=12
        axes[row,0].plot(x[event]/60,w.q_outlet_cms[event],label=label)
        axes[row,1].plot(x[zoom],(w.q_outlet_cms-q)[zoom],'.-',label=label)
        axes[row,1].axvline((t-t0).total_seconds()/60,ls=':',alpha=.6)
    record['sampled_shift_minutes']=(pd.Timestamp(record['child_peak_utc'])-pd.Timestamp(record['parent_peak_utc'])).total_seconds()/60
    record['peak_total_change_pct']=100*(record['child_peak_cms']/record['parent_peak_cms']-1)
    audit.append(record)
    axes[row,0].set(title=f'{ex} {sid}: event context',xlabel='Hours from parent maximum',ylabel='Q (m³/s)')
    axes[row,1].set(title='Crest zoom, own maximum removed',xlabel='Minutes from parent maximum',ylabel='Q − own maximum (m³/s)')
    for ax in axes[row]: ax.legend()
plt.tight_layout();plt.show()
audit=pd.DataFrame(audit);display(audit);audit.to_csv(OUT/'disputed_peak_audit.csv',index=False)
assert np.allclose(audit.sampled_shift_minutes,[4,6])''')
md(h,r'''## What the three-point quadratic calculation actually does

Take the maximum bin at time $t_0$ and its neighbours $(t_0-h,q_-),(t_0,q_0),(t_0+h,q_+)$, with $h=60$ seconds. The unique fitted parabola has a vertex offset
$$\delta=\frac{h}{2}\frac{q_- - q_+}{q_- -2q_0+q_+}.$$
We calculate this separately for parent and child. Their interpolated timing difference is the sampled-bin time difference plus $\delta_c-\delta_p$. The −33-second example is **a difference between two fitted offsets**, not a 33-second shift read directly from saved output.

The three values are interval means. The vertex estimates a local crest of the averaged signal under a quadratic assumption; it does not recover the exact instantaneous hydrograph. Unequal rising/falling slopes affect the fit. Thus a changed shape can produce a changed vertex even when the largest bin is unchanged. This does not make the estimate useless, but it does change the question it answers.

**Policy for now:** retain sampled and reviewed timing; store the quadratic estimate as a clearly separate sensitivity diagnostic. Do not replace zero timings automatically. If subminute crest timing becomes a target, compare fit-window and reporting-resolution sensitivities first. Distinct limb/width responses describe shape changes directly; they do not need to be relabelled as peak shifts.

For signed-log visualization use `symlog` or `asinh(y/s)` with an explicit scale $s$. Both retain zero and signed subminute values. Never add invented tiny shifts just to permit ordinary logarithms.''')
code(h,'''triplets=[]
for label,mask in [('parent',(states.example=='02')&states.is_base),('child',(states.example=='02')&(states.state_id=='S004_unit_5'))]:
    w=pd.read_parquet(states[mask].iloc[0].outlet_parquet);i=int(w.q_outlet_cms.argmax())
    qm,q0,qp=w.q_outlet_cms.iloc[i-1:i+2]
    offset=30*(qm-qp)/(qm-2*q0+qp)
    triplets.append(dict(curve=label,peak_bin_utc=str(w.time_utc.iloc[i]),q_minus=qm,q_zero=q0,q_plus=qp,vertex_offset_seconds=offset))
display(pd.DataFrame(triplets))
print('Child minus parent fitted offset (s):',triplets[1]['vertex_offset_seconds']-triplets[0]['vertex_offset_seconds'])''')
md(h,r'''## B(t): no production baseline has been chosen

$B(t)$ is a proposed event-background discharge. It is not automatically the initial value, the curve minimum, a fitted arbitrary straight line, or a value already stored for every event. In the current outputs, initial and final flows can differ, and some curves dip below initial flow. `tail_settled` means the routing tail settled relative to its forcing, not that the event returned to its initial background.

For real data we must first inspect forcing and outlet together, select an event window, identify defensible pre/post-event background information, and apply a documented common rule to parent/child. A constant baseline is appropriate only when a stable background is supported. A varying baseline needs an independently justified construction and sensitivity analysis. A routed background component would be another explicit experiment, not something this notebook assumes or reroutes. If no defensible event decomposition exists, do not publish excess-volume moments as event properties. Total-Q window moments would be a different, window-dependent diagnostic.

The examples below use a **known synthetic constant background of 3 m³/s**. No synthetic baseline is assigned to real networks.

With $q=Q-B\geq0$, $V=\int q(t)dt$, the function $f(t)=q(t)/V$ is a normalized volume density over time. Therefore
$$\mu=\int tf(t)dt,\quad \sigma_t=\sqrt{\int(t-\mu)^2f(t)dt}.$$
$\mu$ is the discharge-volume-weighted mean time (centroid); $\sigma_t$ is its temporal standard deviation. These are standard distribution moments, applied to event excess. Centroid-based hydrograph timing has a [USGS precedent](https://www.usgs.gov/publications/estimating-basin-lagtime-and-hydrograph-timing-indexes-used-characterize-stormflows). This source does not establish every proposed spread metric as a universal hydrological standard.

In $\int^{t_a}q(t)dt=aV$, **a is a fraction**, e.g. $a=0.1,0.5,0.9$. $t_{0.5}$ is the median volume time: half the volume has passed. It need not equal the centroid. [USGS volume timing](https://pubs.usgs.gov/publication/sir20235064G/full) uses median and interquartile timing for annual hydrographs; our event application is an adaptation.''')
md(h,r'''## One transparent synthetic parent–child pair

Parent excess hydrograph: triangle starting at 2 s, peak at 4 s with height 12 m³/s, ending at 10 s. Background is 3 m³/s. Child transforms time by $t_c=1+1.25t_p$ and excess height by a factor 0.8. Hence child starts at 3.5 s, peaks at 6 s with excess 9.6 m³/s, and ends at 13.5 s. This is an illustrative signal, not a routed river experiment.

Because $0.8\times1.25=1$, excess volume is unchanged. Timing and duration change, but standardized shape does not. Exact triangular formulas below provide examples for every proposed scalar response.''')
code(h,'''def triangle_stats(start,mode,end,height,background=3):
    mu=(start+mode+end)/3
    sd=np.sqrt((start**2+mode**2+end**2-start*mode-start*end-mode*end)/18)
    def quantile(a):
        split=(mode-start)/(end-start)
        return start+np.sqrt(a*(end-start)*(mode-start)) if a<=split else end-np.sqrt((1-a)*(end-start)*(end-mode))
    q10,q25,q50,q75,q90=[quantile(a) for a in [.1,.25,.5,.75,.9]]
    return dict(peak_time=mode,total_peak=background+height,excess_peak=height,volume=height*(end-start)/2,
        centroid=mu,temporal_sd=sd,t10=q10,t25=q25,t50=q50,t75=q75,t90=q90,
        volume_width80=q90-q10,volume_width50=q75-q25,crest_width50=.5*(end-start),crest_width90=.1*(end-start),
        rise50=.5*(mode-start),fall50=.5*(end-mode),asymmetry=(q90+q10-2*q50)/(q90-q10))
parent=triangle_stats(2,4,10,12);child=triangle_stats(3.5,6,13.5,9.6)
display(pd.DataFrame({'parent':parent,'child':child,'child_minus_parent':{k:child[k]-parent[k] for k in parent}}))
t=np.linspace(0,16,1601)
def triangle(t,a,m,b,H): return H*np.maximum(0,np.minimum((t-a)/(m-a),(b-t)/(b-m)))
plt.figure(figsize=(10,3));plt.plot(t,3+triangle(t,2,4,10,12),label='parent');plt.plot(t,3+triangle(t,3.5,6,13.5,9.6),label='child')
plt.axhline(3,color='grey',ls='--',label='known B(t)=3');plt.xlabel('Time (s)');plt.ylabel('Q (m³/s)');plt.legend();plt.show()
assert np.isclose(parent['volume'],child['volume'])
assert np.isclose(child['temporal_sd']/parent['temporal_sd'],1.25)
''')
examples={
 'sampled_peak_shift':("child['peak_time']-parent['peak_time']",'s','2 s later in this example; a discrete sampled version additionally depends on the output bins.'),
 'reviewed_feature_shift':("child['peak_time']-parent['peak_time']",'s','For these single triangular crests the reviewed feature is the peak. For 54 it is the lobe centre instead.'),
 'relative_timing':("100*(child['peak_time']-parent['peak_time'])/(parent['peak_time']-1)",'%','Illustrative inlet peak at 1 s gives parent lag 3 s; shift 2 s / lag 3 s = 66.67%. This is conditional on a defined inlet feature.'),
 'total_peak_change_pct':("100*(child['total_peak']/parent['total_peak']-1)",'%','Total peak goes from 15 to 12.6 m³/s: −16%. This is not the −20% excess-peak change.'),
 'peak_excess_change_pct':("100*(child['excess_peak']/parent['excess_peak']-1)",'%','Excess peak goes from 12 to 9.6 m³/s: −20%. Requires the baseline.'),
 'centroid_shift':("child['centroid']-parent['centroid']",'s','Volume mean moves from 5.3333 to 7.6667 s: +2.3333 s. The mean can move differently from the mode.'),
 'volume_median_shift':("child['t50']-parent['t50']",'s','Compare the printed t50 values: this tracks half-volume passage, not a maximum.'),
 'temporal_sd_change':("child['temporal_sd']-parent['temporal_sd']",'s','Temporal SD increases by 25%; translation alone would not change it. Exact standard distribution variance, not fitted.'),
 'volume_duration80_change':("child['volume_width80']-parent['volume_width80']",'s','The time containing the middle 80% of volume increases by 25%; uses the 10% and 90% cumulative-volume crossings.'),
 'volume_duration50_change':("child['volume_width50']-parent['volume_width50']",'s','The middle 50% volume duration also increases by 25%. This is not width at half peak height.'),
 'connected_crest_width':("{'delta_W50_s':child['crest_width50']-parent['crest_width50'],'delta_W90_s':child['crest_width90']-parent['crest_width90']}",'s','Half-excess-height width grows from 4 to 5 s; 90%-height width from 0.8 to 1 s. Use nearest crossings around the selected crest; lobe-specific for multimodal curves.'),
 'quantile_asymmetry_change':("child['asymmetry']-parent['asymmetry']",'dimensionless','Zero change: translation and uniform stretching preserve this asymmetry, despite altered timing and duration. This specific quantile diagnostic is a project proposal.'),
 'rise_fall_duration_change':("{'delta_rise50_s':child['rise50']-parent['rise50'],'delta_fall50_s':child['fall50']-parent['fall50']}",'s','Rise at half excess height grows from 1 to 1.25 s; fall from 3 to 3.75 s. These measure limb changes explicitly.'),
 'volume_change_pct':("100*(child['volume']/parent['volume']-1)",'%','Both volumes are 48 m³: zero change. This is a conservation/window diagnostic, not a principal prediction target.'),
 'standardized_shape_distance':("0.0",'dimensionless','Exactly zero analytically: after removing volume, centroid and SD, these two affine-transformed triangles coincide. Optional proposed diagnostic; no general real-data implementation is implied.'),
}
catalog=pd.read_csv(SOURCE/'response_definitions.csv')
for r in catalog.itertuples():
    md(h,f'### {r.name}\n\nDefinition: `{r.equation}`.\n\n{r.interpretation}\n\nReal-data status: **{r.status}**.')
    if r.name=='54_left_right_center':
        md(h,'Use actual 54/unit 19: left −2 min, right 0 min, centre −1 min. This cannot be represented by our single-triangle toy. The next table retains all three clocks and heights for focused review.')
        code(h,"lobes=pd.read_csv(SOURCE/'54_lobe_audit.csv');display(lobes[lobes.state_id.isin(['S001_unit_19','S027_unit_2','island_54_033'])][['state_id','left_shift_minutes','center_shift_minutes','right_shift_minutes','left_change_pct','right_change_pct']])")
    else:
        expression,unit,explanation=examples[r.name]
        md(h,'**Worked interpretation:** '+explanation)
        code(h,f"print('Example result ({unit}):', {expression})")
md(h,r'''## A shape-only counterexample to interpreting every difference as peak movement

Take two triangles with the same start (2 s), peak time (4 s) and peak excess (12 m³/s), but endpoints 10 and 12 s. Their peak timing and height changes are exactly zero, whereas falling duration, centroid and temporal spread change. Thus separate responses are essential. A local quadratic estimate on sampled asymmetric crests can also respond to the change of slope; that does not justify reclassifying every shape change as a resolved peak displacement.''')
code(h,"shape_only=triangle_stats(2,4,12,12)\ndisplay(pd.DataFrame({'parent':parent,'longer_falling_limb':shape_only}).loc[['peak_time','excess_peak','centroid','temporal_sd','rise50','fall50']])")
md(h,'''## Review order and what is deliberately undecided

1. Agree primary timing convention: reviewed feature, sampled maximum retained, quadratic sensitivity separate. Preserve all values in seconds without rounding; use signed plots.
2. Review total peak height separately from excess peak height. Height and time units are explicit everywhere.
3. Inspect real forcing/outlet events and baseline/window choices together. Until then, real excess centroid, volume median and widening remain uncomputed.
4. Choose one bulk timing and one widening response after worked examples, rather than treating every catalogue entry as a required model target. Keep 54's focused analysis separate.

No response selection, model fitting, rerouting, or recalibration has occurred. Proposed metrics are not yet an agreed final contract.''')
save(h,'hydrograph_response_worked_examples_20261012.ipynb')

g=notebook('Geometry metrics — one worked example per metric')
md(g,r'''This companion contains **predictors (x)** only. Hydrograph responses (y) are in `hydrograph_response_worked_examples_20261012.ipynb`. The earlier foundation notebook remains a preserved snapshot.

Every metric below has an equation, numerical value on a known graph, interpretation, and computation scope/status. Synthetic numbers teach the definition; they are not fitted predictions or measurements from a real network. The four toy routes can be explicitly enumerated; production moment calculations use DAG recursions instead.''')
code(g,setup+"\nsys.path.insert(0,str(ROOT/'RAPID/investigations/parent_child_predictor_exploration'))\nfrom features import describe\n")
md(g,r'''## Domain first

Our toy containing compound runs from node 0 to node 4:

- First split, 0→1: two alternatives, A = 100 m / 30 m wide; B = 300 m / 10 m wide.
- Shared reach C, 1→2: 50 m / 40 m wide.
- Second split, 2→3: D = 100 m / 20 m wide; E = 200 m / 20 m wide.
- Shared reach F, 3→4: 50 m / 40 m wide.

Toy child replaces A/B by one 150 m / 40 m reach, leaving C–F unchanged. This is a stipulated geometric example, not an assertion about the production collapse rule.

**Actual edit domain:** 0→1, including A/B in parent and replacement in child. **Containing compound:** 0→4 before and after this edit. **Whole network:** add an upstream 60 m reach and downstream 80 m reach, each 40 m wide. External distances for the compound are therefore 60 and 80 m. For the local edit, downstream expected distance is 50+150+50+80=330 m. For the whole network, external distances are zero.

Current real-data foundation columns are **whole-network** summaries. Matched local/compound computations are not silently implied: their boundary audit remains pending. Every worked metric card below uses the containing compound unless explicitly stated.''')
code(g,'''toy=pd.DataFrame(dict(id_link=range(6),id_us_node=[0,0,1,2,2,3],id_ds_node=[1,1,2,3,3,4],link_length_m=[100.,300.,50.,100.,200.,50.],width=[30.,10.,40.,20.,20.,40.]))
childtoy=pd.concat([pd.DataFrame(dict(id_link=[10],id_us_node=[0],id_ds_node=[1],link_length_m=[150.],width=[40.])),toy.iloc[2:]],ignore_index=True)
display(toy.assign(reach=list('ABCDEF')))
values,_=describe(toy,False);childvalues,_=describe(childtoy,False)
routes=pd.DataFrame(dict(route=['A-C-D-F','A-C-E-F','B-C-D-F','B-C-E-F'],probability=[.375,.375,.125,.125],length_m=[300.,400.,500.,600.]))
routes['weighted_length_m']=routes.probability*routes.length_m
routes['harmonic_width_m']=[300/(100/30+50/40+100/20+50/40),400/(100/30+50/40+200/20+50/40),500/(300/10+50/40+100/20+50/40),600/(300/10+50/40+200/20+50/40)]
display(routes)
assert np.isclose((routes.probability*routes.length_m).sum(),values['route_length_mean'])
assert np.isclose((routes.probability*(routes.length_m-values['route_length_mean'])**2).sum(),values['route_length_sd']**2)
np.testing.assert_allclose([values['total_length'],values['area_proxy'],values['width_mean'],values['route_length_mean'],values['route_length_sd'],values['route_length_skew']],[800,16000,20,400,100,.75])
display(pd.DataFrame({'parent_compound':values,'child_compound':childvalues}).dropna())''')
md(g,r'''## How P_k is calculated; what shared reaches mean

$P_k=\prod_{b\in k}p_{b,j(k)}$ over encountered splits. First split fractions are 30/(30+10)=0.75 and 0.25; second split fractions are 0.5 and 0.5. Thus route probabilities are 0.375, 0.375, 0.125 and 0.125. This assumes local width-proportional allocation, not measured flow.

All four routes contain C and F. Listing them in each route is correct. In a **probability-weighted expectation**, their total contribution is multiplied by $\sum_kP_k=1$: C contributes 50 m, not 200 m. Equivalently $\mu_L=\sum_e r_eL_e$. An unweighted sum over all four routes counts C four times. The old sum of complete-path effective widths also repeats paths sharing C/F; those paths are not independent parallel channels. The issue is the aggregation and its physical interpretation, not merely using paths.

There is still possible structural sensitivity in the width-allocation assumptions; avoiding unweighted duplication does not prove that width is the true flow partition.

Skewness $E[(L-\mu_L)^3]/\sigma_L^3$ is the **standard third standardized central moment**, not a newly invented exponent. Cubing retains deviation signs and emphasizes tails; dividing by SD cubed makes it dimensionless. We use population probabilities, not a sample-bias correction. [NIST definition](https://www.itl.nist.gov/div898/software/dataplot/refman2/ch2/skewness.pdf).''')
code(g,'''H=np.array([-sum(p*np.log(p) for p in [.75,.25]),np.log(2)])
values['entropic_branches_mean']=float(np.exp(H).mean());values['entropic_branches_sd']=float(np.exp(H).std())
values['path_harmonic_width_min']=routes.harmonic_width_m.min();values['path_harmonic_width_max']=routes.harmonic_width_m.max()
values['route_length_quantiles']='P10=300 m; P50=400 m; P90=600 m (inverse discrete CDF)'
values['upstream_distance']=60.;values['downstream_distance']=80.;values['network_entry_exposure']=1.
values['modified_length_fraction']=400/(800+60+80)
print('Route mean:',values['route_length_mean'],'m; SD:',values['route_length_sd'],'m; skewness:',values['route_length_skew'])
print('Local edit mean/SD:',describe(toy.iloc[:2],False)[0]['route_length_mean'],describe(toy.iloc[:2],False)[0]['route_length_sd'])
''')
catalogg=pd.read_csv(SOURCE/'geometry_dictionary.csv')
interpret={
 'total_length':'Unique compound reaches sum to 800 m; summing route lengths would give 1800 m and measure something different.',
 'area_proxy':'Sum of reach length times width is 16000 m²; this is a channel-area proxy, not a polygon-union area.',
 'route_length_mean':'0.375×300+0.375×400+0.125×500+0.125×600 = 400 m. Child compound mean remains 400 m even though route spread decreases.',
 'route_length_sd':'SD is 100 m: variation among complete travelled distances. The child retains the second split, so its SD is 50 m, not zero.',
 'route_length_cv':'100/400 = 0.25. Relative route-length spread; shares information with mean and SD.',
 'route_length_skew':'Positive skew means the longer alternatives form the lighter tail. The next printed value comes from the four weighted routes, not edge-length skewness.',
 'branch_length_contrast_mean':'First split contrast is sqrt(.75*.25)*200 = 86.6025 m; second is 50 m. Both are fully visited, so their mean is 68.3013 m.',
 'branch_length_contrast_sd':'SD across the two local contrast values (86.6025 and 50 m), about 18.3013 m; distinct from route SD.',
 'branch_length_variance_sum':'86.6025²+50² = 10000 m², exactly route SD squared; not a separate mechanism.',
 'width_mean':'16000/800 = 20 m; all unique reaches weighted by physical length, not allocation probability.',
 'width_sd':'The displayed SD describes reach widths around 20 m, with long reaches carrying more weight.',
 'width_cv':'Width SD divided by 20 m; dimensionless reach-width heterogeneity.',
 'width_harmonic':'800 divided by the sum L/w; constrictions contribute strongly. On a branched graph this is a descriptive aggregate, not equivalent hydraulic resistance.',
 'width_p10':'Sort widths and accumulate reach lengths. The 10th length percentile lies in the 10-m-wide reach.',
 'width_p90':'The 90th length percentile lies in the 40-m-wide reaches; does not describe a dominant complete path.',
 'bifurcations':'Two physical nodes split (0 and 2), regardless of the number of numerical routing cells.',
 'cycle_rank':'E=6, V=5, C=1, giving 2 independent undirected loops. This DAG has no directed recirculation.',
 'junction_excess':'Each binary split contributes 1: total 2, equal to cycle rank here.',
 'bifurcation_density':'2/800 = 0.0025 m⁻¹, or 2.5 splits per km of unique channel length.',
 'log1p_paths':'Four routes give ln(5)=1.60944. Serial branching grows route count multiplicatively; log compresses that growth without creating a capacity metric.',
 'expected_splits':'Both split nodes are reached with probability 1: expected encounters = 2. In general, rarely visited splits contribute less than 1.',
 'dominance_mean':'Local dominances .75 and .5 average to .625; the full compound has no single analogous width fraction.',
 'dominance_sd':'SD of .75 and .5 is .125; the splits differ in balance.',
 'effective_branches_mean':'First split: 1/(.75²+.25²)=1.6; second: 2; mean 1.8.',
 'effective_branches_sd':'SD of 1.6 and 2 is .2 effective branches.',
 'split_entropy_mean':'Average normalized entropies of the 75:25 and 50:50 splits; 1 would mean both fully balanced.',
 'split_entropy_sd':'Variability of those normalized local balance values; not accumulated route entropy.',
 'route_entropy':'Sum H(75:25)+H(50:50), because both splits are visited. Units are nats; exponentiating would give effective route alternatives, not simultaneous channels.',
 'split_exposure_mean':'Both conditional node visit probabilities are 1, so their mean is 1.',
 'split_exposure_sd':'Both conditional exposures equal 1, giving zero SD; this does not imply zero branching.',
 'split_impact_mean':'Impacts are 1−(.75²+.25²)=.375 and .5; mean .4375. No route-length contrast enters this metric.',
 'split_impact_sd':'SD of .375 and .5 is .0625.',
 'split_impact_sum':'Sum .875 = 2×.4375. With more splits it can exceed 1; do not interpret as a single fraction.',
 'area_elongation':'400²/16000 = 10. Proposed dimensionless geometric shape proxy, not the historical path-sum elongation.',
 'sinuosity_proxy':'400/300 = 1.3333. Allocation-weighted detour relative to shortest graph route, not map sinuosity.',
 'route_length_min':'Shortest route A-C-D-F is 300 m; it carries .375 allocation probability.',
 'route_length_max':'Longest route B-C-E-F is 600 m; it carries .125 probability.',
 'entropic_branches_mean':'Mean exp(H) for 75:25 and 50:50; compares with inverse-Simpson count 1.8. This is a local eBI adaptation, not transect eBI.',
 'entropic_branches_sd':'Spread of the two local exp(H) values. Keep encounter count separately.',
 'path_harmonic_width_min':'Minimum of the four printed path harmonic widths; serial constrictions on the B route make it narrow despite a long geometric length.',
 'path_harmonic_width_max':'Maximum of the four printed path harmonic widths; it is not a sum over shared-route capacities.',
 'route_length_quantiles':'Use cumulative probabilities .375,.75,.875,1. The 90% point is 600 m, even though the most likely lengths are 300 and 400 m.',
 'upstream_distance':'For compound 0→4, external upstream length is 60 m. This is a contextual metric, not computed solely inside the compound.',
 'downstream_distance':'For compound 0→4, external downstream length is 80 m. For edit 0→1 it is instead 330 m in expectation.',
 'network_entry_exposure':'All network flow enters the compound in this toy, so exposure is 1. A bypass would reduce it.',
 'modified_length_fraction':'Actual parent edit removes A/B (400 m); whole parent has 940 m. Fraction 400/940 ≈ .4255, not containing-compound fraction 800/940.',
}
assert set(catalogg.name)==set(interpret)
for r in catalogg.itertuples():
    md(g,f'### {r.name}\n\n**Family:** {r.family}. **Equation:** `{r.equation}`.\n\n{interpret[r.name]}\n\n**Real-data computation status/scope:** {r.computed_scope}.\n\n**Qualification:** {r.redundancy_note}')
    code(g,f"print('Worked example:',values[{r.name!r}])"+(f"\nprint('Child compound:',childvalues[{r.name!r}])" if r.name in set(catalogg.name[:36]) else ''))
md(g,r'''## Resistance belongs explicitly in the inventory

The earlier dictionary omitted the existing `resistance_log`; that omission should not imply it is unimportant. It combines length, width, network arrangement and hydraulic assumptions, so it is not merely a width statistic.

The repository's edge laws are
$$K_e^{Manning}=\frac{n_e^2L_e}{A_e^2R_{h,e}^{4/3}},\qquad
K_e^{Chezy}=\frac{L_e}{A_e^2C_{z,e}^2gR_{h,e}},$$
with $\Delta h=KQ|Q|$. Cross-sectional area and hydraulic radius come from the configured width–depth assumptions. Series K values add; parallel K values combine as $K_{parallel}=(\sum_jK_j^{-1/2})^{-2}$, not the linear-resistor formula. General compounds require the existing nonlinear network solver with the same matched boundaries.

The historical predictor is **$-\log_{10}(K_c/K_p)$**. Positive means lower child resistance; negative means higher child resistance. It belongs in a separate explicitly assumption-dependent feature family until we decide what geometry-only resistance surrogate is justified. A common roughness multiplier may cancel in a ratio; depth laws and state/domain choices need checking before declaring the ratio purely geometric. We should not replace it by harmonic width.

Worked quadratic-network example: parallel branch resistances 4 and 16 have equivalent K = (1/2+1/4)⁻² = 16/9. Add a series reach of K=1 to get parent K=25/9. A child with K=2 has resistance_log = −log10(2/(25/9)) ≈ +0.1427. These are illustrative consistent K units, not values from a routed case.

Source: `hierarchy_level_definition/metrics/equivalent_resistance.py`, functions `compute_K_manning`, `compute_K_chezy`, `edge_flow`, `reduce_simple_parallel`; historical sign convention in `RAPID/investigations/prediction_audit_20261006/report.py`.''')
code(g,"kp=1+(4**-.5+16**-.5)**-2;kc=2\nprint('Parent K:',kp,'Child K:',kc,'resistance_log:',-np.log10(kc/kp))")
md(g,'''## Review gate

Review definitions, examples and domains before selecting features. Decide whether geometry-only resistance under fixed assumptions belongs beside the strict planform descriptors; do not silently mix calibrated parameters into them. Keep old snapshots intact. These examples introduce no new prediction scores and do not establish which variables explain the real hydrograph responses.''')
save(g,'geometry_metric_worked_examples_20261012.ipynb')
