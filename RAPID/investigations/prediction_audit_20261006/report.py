"""Supplement a completed audit with interpretation and reference-allocation checks."""
from pathlib import Path
import sys,json,hashlib
import numpy as np,pandas as pd
import networkx as nx
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[3]
OUT=Path(sys.argv[1]).resolve()
assert (OUT/'provenance.json').exists(), 'Run audit must complete first'
assert not (OUT/'REPORT.txt').exists(), 'Do not overwrite a completed report'
SOURCE=ROOT/'RAPID/outputs/cunge_fixed_comparison_plus37_20261006'
PREV=ROOT/'RAPID/outputs/cunge_fixed_extra_trees_plus37_20261006'
raw=pd.read_parquet(SOURCE/'paired_results.parquet');d=raw[raw.method.eq('cunge')].copy()
groups=pd.read_csv(OUT/'groups.csv',dtype={'example_id':str});sizes=pd.read_csv(OUT/'group_sizes.csv',dtype={'example_id':str})
scores=pd.read_csv(OUT/'exploratory_scores.csv');folds=pd.read_csv(OUT/'grouped_fold_audit.csv')
foot=pd.read_csv(OUT/'modification_footprint_audit.csv',dtype={'example':str});topo=pd.read_csv(OUT/'topology_candidates.csv',dtype={'example':str})
pred=pd.read_parquet(PREV/'held_out_predictions.parquet');m=pd.read_parquet(PREV/'matched_features_responses.parquet');cols=json.loads((PREV/'provenance.json').read_text())['geometry_features']
checks=[];extra_hashes={str(Path(__file__)):hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
old_scores_path=ROOT/'RAPID/outputs/cunge_fixed_extra_trees_20261006/scores_in_hours.csv'
extra_hashes[str(old_scores_path)]=hashlib.sha256(old_scores_path.read_bytes()).hexdigest()
old_scores=pd.read_csv(old_scores_path)
old_r2=float(old_scores[(old_scores.routing=='cunge')&(old_scores.trained_response=='relative_shift_pct')&(old_scores.features=='geometry_15')&(old_scores.scope=='pooled')].r2.iloc[0])
assert abs(old_r2-.503458)<1e-6
for source in json.loads((SOURCE/'provenance.json').read_text())['sources']:
    path=Path(source['source'])/'states'/source['base_fingerprint']/'parameters.parquet';extra_hashes[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest();f=pd.read_parquet(path)
    p=f.width/f.groupby('u').width.transform('sum');np.testing.assert_allclose(p,f.split_fraction,rtol=1e-12,atol=1e-12)
    graph=nx.DiGraph(zip(f.u,f.v));order=list(nx.topological_sort(graph));share={order[0]:1.};allocation={}
    for u in order:
        for r in f[f.u.eq(u)].itertuples():
            q=share[u]*r.split_fraction;allocation[r.id_link]=q;share[r.v]=share.get(r.v,0.)+q
    np.testing.assert_allclose(f.id_link.map(allocation),f.reference_flow_fraction,rtol=1e-12,atol=1e-12)
    checks.append(dict(example=source['example'],links=len(f),max_split_error=float(abs(p-f.split_fraction).max()),max_allocation_error=float(abs(f.id_link.map(allocation)-f.reference_flow_fraction).max()),outlet_share=share[order[-1]]))
pd.DataFrame(checks).to_csv(OUT/'reference_allocation_verification.csv',index=False)
# Full predictor dictionary: exact transforms and scope, no feature is silently renamed.
definitions=[
('resistance_log','-log10(K_child/K_parent)','Matched hydraulic domain','Reference-hydraulic derived; not pure planform'),
('length_fraction','sum(parent matched-domain physical edge lengths)/sum(base network lengths)','Matched domain, often containing compound','Geometric context exposure, not local edit size'),
('width_fraction','1 for closed_compound_unit or containing_compound_context; otherwise width-allocated base flow entering domain','Matched domain','Geometric allocation policy proxy; forced exposure can discard local size'),
('log10_parent_equivalent_length','log10(sum_p L_p W_p / sum_p W_p)','Matched parent domain','W_p = L_p / sum_e(L_e/w_e)'),
('log10_parent_elongation','log10(L_eq / sum_p W_p)','Matched parent domain','Complete-path width sum counts shared edges repeatedly'),
('log10_parent_dominant_width_fraction','log10(max_p W_p / sum_p W_p)','Matched parent domain','Shares denominator with elongation'),
('parent_path_length_cv','SD_uniform_paths(L_p)/mean_uniform_paths(L_p)','Matched parent domain','Uniform complete-path weighting, not routed flow weighting'),
('signed_length_width_association','L_eq/mean_uniform_paths(L_p) - 1','Matched parent domain','Cov(L,W)/(mean(L)*mean(W)); not Pearson correlation'),
('log10_domain_reference_slope','log10((head_source-head_sink)/shortest_path_length)','Matched parent domain','Reference-head reconstruction dependent'),
('log_equivalent_length_ratio','ln(child_L_eq/parent_L_eq)','Matched paired domains','Natural log, unlike log10 parent feature'),
('log_elongation_ratio','ln(child_elongation/parent_elongation)','Matched paired domains','Combines length and path-sum width changes'),
('change_path_length_cv','child_CV-parent_CV','Matched paired domains','Uniform-path geometry change'),
('change_width_dominance','child_dominant_fraction-parent_dominant_fraction','Matched paired domains','Absolute difference; can be tiny in large compounds'),
('log10_1p_upstream_shortest_length_m','log10(1+shortest distance from network inlet to parent-domain source)','Matched-domain position','Containing-domain position can repeat for local edits'),
('log10_1p_downstream_shortest_length_m','log10(1+shortest distance from parent-domain sink to network outlet)','Matched-domain position','Containing-domain position can repeat for local edits')]
pd.DataFrame(definitions,columns=['feature','definition','scope','qualification']).to_csv(OUT/'feature_dictionary.csv',index=False)
# Independently check principal numeric claims, including interval and giant context.
a=d[d.example_id.eq('37')];old=d[~d.example_id.eq('37')]
z=pred[(pred.routing=='cunge')&(pred.response=='relative_shift_pct')&(pred.features=='geometry_15')&pred.example_id.eq('37')].copy();z['sqerr']=(z.observed_shift_h-z.predicted_shift_h)**2
large=z[z.state_id.eq('CB001_bubble_3')].iloc[0];context=a[a.hydraulic_domain_type.eq('containing_compound_context')]
findings=dict(pairs=len(d),pairs_37=len(a),below_755_seconds=int((abs(a.shift_h)*3600<755).sum()),largest_squared_error_share_pct=float(100*large.sqerr/z.sqerr.sum()),containing_context_rows=len(context),rows_using_largest_parent_context=int(np.isclose(a.length_fraction,.956977548884169).sum()),context_max_length_fraction=float(context.length_fraction.max()),parent_paths_max=float(a.parent_paths.max()),old_parent_paths_max=float(old.parent_paths.max()),old_parent_leq_max=float(old.parent_equivalent_length.max()),old_length_fraction_max=float(old.length_fraction.max()),selection_records=int(d.selection_count.sum()),additional_alias_records=int(d.selection_count.sum()-len(d)),topology_groups=len(sizes),previous_four_network_r2=old_r2)
assert findings['below_755_seconds']==378
assert abs(findings['largest_squared_error_share_pct']-72.249015)<1e-5
(OUT/'verified_findings.json').write_text(json.dumps(findings,indent=2))
# Exact constant direction in the rank-deficient river and strongest empirical dependencies.
svd_rows=[]
for scope,xx in [('pooled',m[m.method.eq('cunge')])]+list(m[m.method.eq('cunge')].groupby('example_id')):
    standardized=(xx[cols]-xx[cols].mean())/xx[cols].std(ddof=0);u,s,vh=np.linalg.svd(standardized,full_matrices=False)
    for feature,value in zip(cols,vh[-1]):svd_rows.append(dict(scope=scope,feature=feature,smallest_direction_loading=value,smallest_singular_value=s[-1]))
pd.DataFrame(svd_rows).to_csv(OUT/'smallest_singular_directions.csv',index=False)
cor=pd.read_csv(OUT/'feature_correlations.csv',dtype={'scope':str});strong=cor[(cor.method=='spearman')&cor.correlation.abs().ge(.95)].copy();strong.to_csv(OUT/'strong_correlations.csv',index=False)
view=scores[(scores.routing=='cunge')&(scores.response=='relative_shift_pct')&(scores.units=='hours')&(scores.scope=='pooled')].copy();view['mae_minutes']=60*view.mae;view['zero_mae_minutes']=60*view.zero_mae
view[['validation','features','weighting','r2','mae_minutes','zero_mae_minutes','skill_vs_zero']].to_csv(OUT/'cunge_relative_to_hours_summary.csv',index=False)
# Plot the audited evidence; no visual introduces a new statistical claim.
fig,axes=plt.subplots(2,2,figsize=(12,9),layout='constrained')
ss=sizes.groupby('example_id').agg(groups=('group','size'),largest=('pairs','max'),pairs=('pairs','sum'))
axes[0,0].bar(ss.index,ss['pairs'],color='#a8c6d7',label='All comparisons');axes[0,0].bar(ss.index,ss.largest,color='#28566b',label='Largest indivisible group');axes[0,0].set(title='Most of network 37 remains one group',ylabel='Unique comparisons');axes[0,0].legend()
ff=foot[foot.example.eq('37')];axes[0,1].scatter(ff.selected_length_fraction*100,ff.matched_length_fraction*100,s=13,alpha=.45);axes[0,1].plot([.05,100],[.05,100],color='gray',ls='--');axes[0,1].set(xscale='log',yscale='log',xlabel='Selected unit-support length (% network)',ylabel='Matched context length (% network)',title='Unit support and context are different scopes')
base=view[view.features.eq('geometry_15')];labels=[f'{r.validation}\n{r.weighting}' for r in base.itertuples()];axes[1,0].bar(labels,base.mae_minutes,color='#28566b');axes[1,0].scatter(labels,base.zero_mae_minutes,color='#b8683e',marker='_',s=300,label='No-change baseline');axes[1,0].set(title='Conditional percentage predictions in hours',ylabel='MAE (minutes)');axes[1,0].legend()
syn=pd.read_csv(OUT/'synthetic_geometry_checks.csv');axes[1,1].semilogy(syn.serial_diamonds,syn.elongation,'o-',label='Implemented elongation');axes[1,1].semilogy(syn.serial_diamonds,10*syn.serial_diamonds,'o--',label='Single-route L / width');axes[1,1].set(xlabel='Successive equal two-branch stages',ylabel='Ratio (log scale)',title='Path multiplicity overwhelms the shape descriptor');axes[1,1].legend()
fig.savefig(OUT/'audit_summary.png',dpi=180);plt.close(fig)
summary=sizes.groupby('example_id').agg(groups=('group','size'),pairs=('pairs','sum'),largest_group=('pairs','max'))
q=foot.merge(d[['example_id','state_id','hydraulic_domain_type']],left_on=['example','state_id'],right_on=['example_id','state_id'],validate='one_to_one');q=q[(q.example=='37')&q.hydraulic_domain_type.eq('containing_compound_context')]
text=f'''PREDICTION LIMITATIONS, GROUPING AND GEOMETRY AUDIT — 6 October 2026

Outcome and scope
The existing five-network findings were independently reproduced from held-out predictions. Whole-network Cunge R2 is 0.042102 for hours, 0.069668 for percentage, and 0.126290 for percentage converted to hours with known parent lag. This remains conditional on the parent hydrograph and existing network calibration. No routing, calibration, targets, production inputs, or prior snapshots were changed. The new model fits are fixed exploratory diagnostics; no model or seven-feature subset has been selected.

1. Verified evidence
{json.dumps(findings,indent=2)}
Saved per-method, per-target recomputations, exclusions without refitting, and predictor/response ranges are in verified_existing_scores.csv and range_audit.csv. The 755-second comparison is a resolution diagnostic, not a confidence interval. CB001_bubble_3 remains in all primary scores. Supplying n remains unhelpful. Extra Trees predictions were asserted to remain inside each training response range.

2. Membership semantics and a defensible grouping
The hierarchy code builds an undirected graph whose vertices are units and whose edges mean shared base-network edges. compound_bubble_id labels its connected components; these are not simply primary-parent trees. Transitive overlap joins units even if their immediate supports do not touch. Standalone units also receive a bubble ID, with in_compound_bubble=False. IDs must be qualified by network.
The stored membership was reconstructed from unit_link_membership.csv and agreed exactly with every stored bubble component. All explicit child relationships stay within these components. All selection aliases were traced back to independent/compound registries or island source units and unioned, rather than counted as extra observations. We also audited base selection-support overlaps and matched-domain overlaps and conservatively keep overlapping contexts together. groups.csv is the complete resulting assignment.
{summary.to_string()}

Network 37 compound 3 includes 316 base units and 384 canonical comparisons. This is a real shared-edge component, not a mistaken global compound label. Preserving that component prevents distributing its examples across folds. The five-fold assignment is:
{folds.to_string(index=False)}
The split is structure-only GroupKFold, frozen across all routing methods and targets. target_type and signed percentage bins [-infinity,-5,-1,-0.1,0.1,1,5,infinity) are reported in stratification_diagnostic.csv. Approximate stratification cannot repair the giant-group imbalance, so no response-based fold search was performed and no correlated group was split. This is pooled-network grouped evaluation, not balanced within-network evaluation; some folds contain only one giant group. It shares forcing and calibration wherever a network occurs on both sides and cannot replace network transfer testing.

3. Local modification, exposure, position and context
A containing compound is not a local edit. For network 37's containing-context rows, selected-unit-support length fraction has median {q.selected_length_fraction.median():.6f} and range {q.selected_length_fraction.min():.6f}–{q.selected_length_fraction.max():.6f}, whereas the matched context can cover 0.956978 of the network. These support measures are intended modification footprints, not final measured geometric differences. Compound collapse supports and island source-unit supports may themselves be broad.
Cross-state numeric link IDs are not stable. A naive comparison falsely merges most of 37 and all of 21: for S013_unit_251, unchanged edges are renumbered. The completed audit uses stable base supports and contexts for grouping. modification_footprint_audit.csv records renumbering and partial lineage-mapped changes separately. {int((foot.unmapped_child_edges>0).sum())} states contain child edges absent from the saved lineage table (e.g. later anchor splits), so it would be wrong to label these partial mapped changes as the actual complete edited geometry. Reconcile those splits before introducing actual-edit predictors or relaxing conservative grouping.
Use four separate predictor blocks in the next version: (a) lineage-resolved removed/modified/replacement geometry, (b) physical length and reference-flow exposure of that edit, (c) edit-local network position, (d) surrounding closed-compound topology and hydraulics. Keep context prefixes explicit. The current exposure feature is forced to one for containing_compound_context and closed_compound_unit; it cannot by itself measure local modification exposure.

4. Mathematical audit of all 15 features
feature_dictionary.csv gives exact definitions and their scope. Let L_p=sum_e L_e, R_p=sum_e L_e/w_e, W_p=L_p/R_p, N=number of complete directed paths. Then L_eq=sum(L_p W_p)/sum(W_p), W_sum=sum(W_p), elongation=L_eq/W_sum, and dominance=max(W_p)/W_sum. The path length CV uses uniform complete-path probabilities. Association=L_eq/mean(L)-1=Cov(L,W)/(mean(L)mean(W)), not a Pearson coefficient. Parent logs are base ten; child/parent ratio logs are natural logs. The slope is reference head drop divided by shortest domain path, not a mean bed slope. Resistance is -log10(K_child/K_parent) from reference-hydraulic domain solves, not a pure planform property.
Shared edges enter every path containing them. k successive identical two-branch stages, each branch length 100 m and width 10 m, have N=2^k, L_eq=100k, W_sum=10*2^k, elongation=10k/2^k and dominance=2^-k. The implementation agrees analytically through k=32 and under degree-two subdivision. These small values are mathematically consistent. However, W_sum is a sum over route alternatives, not a cross-sectional width or independently flowing channel inventory. Path multiplicity can dominate a nominal shape variable.
Elongation and dominance share W_sum: log(elongation)-log(dominance)=log(L_eq/max W_p). Association also reuses L_eq, and ratio features reuse both parent and child quantities. These algebraic couplings are not necessarily exact linear dependencies among the 15 supplied columns. feature_rank.csv and strong_correlations.csv quantify actual dependencies pooled and within each network; pooled full rank does not imply physical independence. SVD loadings are descriptive, not a fitted feature selector. omega, though not one of the 15, is exactly resistance_log*length_fraction*width_fraction.

5. Candidate topology and flow descriptors
We calculated bifurcation count, bifurcations per physical kilometre, independent undirected cycle rank E-V+C (parallel physical channels retained), log(1+complete path count), width-allocation route entropy, and width-allocation route-length CV for selected-unit supports and matched contexts separately. See topology_candidates.csv. Non-single-source/sink supports receive no route-probability descriptors; they are not silently forced into a closed-domain model.
Bifurcation count, cycle rank, log path count and entropy are invariant to simple degree-two subdivision; physical-length density and route CV are invariant too. Counts must use physical links, not routing cells, and parallel channels must not be collapsed into a simple graph. Per-node/edge density is segmentation-sensitive and should be avoided. Cycle count grows linearly in successive diamonds, while path count grows exponentially; log path count is an explicit complexity measure, not a width surrogate. Parallel-edge splitting into duplicate channels would change these metrics and is not an invariant transformation.
The entropy is H=E_route[sum_nodes -log(p_chosen)] = sum_nodes visit_probability*H(local_split). Route-length moments use the same transition probabilities; shared reaches are traversed once by a particle. This is an expected route-choice entropy in nats, not the entropy of treating all reaches as mutually exclusive. Serial and nested decisions increase entropy; a normalized per-length variant answers a different question. Route-length CV captures geometric route spread, not travel-time spread or hydrograph diffusion. Under uniform scaling of lengths, route CV and entropy stay fixed, cycle/path counts stay fixed, and bifurcations/km changes inversely with length. Absolute size must remain separately available.
The prototype uses p_e=width_e/sum(outgoing widths). We verified against all five stored Cunge base parameter tables that this is exactly their reference split/allocation policy, including conservation to the outlet. Thus these are geometric allocation proxies that coincide with current reference-flow routing weights; they are not independently measured flow. They do not depend on supplied Manning n under the current fixed-width allocation rule. Under a hydraulic discharge allocation, both entropy and route CV would become flow-state dependent and must be labelled accordingly. Domain-conditioned values still need inlet exposure as a separate feature. No candidate was added to model selection in this audit.

6. Fixed held-out ablations, not feature selection
All fits use the previous 96-tree Extra Trees configuration, min_samples_leaf=5, seed=20261005 and one CPU thread. Cunge has the full 15, supplied-n, and six leave-one-family-out variants, under both grouped and whole-network outer evaluation and both targets. The fixed-X controls get full geometry and supplied-n grouped evaluation; their saved network tests remain available. The independently re-fitted Cunge network baselines reproduce saved predictions to 1e-11 relative tolerance. No tuning, imputation or feature selection used test folds.
Each group's row weights sum to one in equal-group scores; weighted R2 uses a weighted global mean, not an average of unstable per-group R2 values. Equal-group MAE equals the mean of group MAEs. Training is still pair-weighted to preserve the estimator; equal-group training weights have not been tested. Equal-group evaluation also does not imply equal-network evaluation.
Cunge percentage predictions converted to hours:
{view[['validation','features','weighting','r2','mae_minutes','zero_mae_minutes']].to_string(index=False)}
Read ablation changes jointly across held-out networks/groups, both target formulations, and zero-change skill. Do not pick a subset from these outer scores and quote its score as selection-adjusted performance. Correlated feature blocks can substitute for one another; ordinary individual permutation importance can understate this. Block ablations expose a different dependence but are not causal evidence. SHAP has not been computed: it could explain the fitted prediction for a failure case but cannot explain the physical routed response or make trees extrapolate.

7. Concrete staged continuation
Stage A — completed: preserve and hash current sources, reproduce findings, audit membership/aliases/context overlap, keep network and grouped tests separate, quantify range and redundancy, and test path-count/subdivision behavior.
Stage B — next priority: reconcile post-lineage anchor splits and verify changed-edge geometry against actual base/child lineage. Produce explicit edit geometry, edit exposure and edit position, separate from compound context. Inspect CB001_bubble_3 and the two large delayed island cases using existing hydrographs only. Assess interpolation/sampled-peak sensitivity without changing targets or rerouting.
Stage C — predeclare candidate families: local edit/exposure/position; bounded context geometry; explicit log path count, bifurcation and cycle density; and flow-allocation entropy/route CV. Compare equal-group and pair-weighted fitting as separate fixed candidates. Avoid forcing seven features before determining stable scopes and dependencies.
Stage D — nested evaluation: retain whole-network outer folds and the published conservative grouped outer assignment as two distinct questions. Within each outer training partition, use grouped inner validation only; keep aliases/compound components intact. Fit scaling, imputation if any, feature selection and all tuning inside inner training. Select with equal-group MAE in hours (percentage predictions use known parent lag), with zero-change and training-only mean baselines; refit the selected pipeline once and score outer test once. Audit giant-group feasibility in every inner split and report unavailable or imbalanced validation rather than splitting it. No new model winner is declared from this exploratory audit.
Stage E — interpretation and stability: examine per-group losses and paired block-ablation differences, sensitivity to the giant compound without deleting it from primary scores, and joint/per-family permutation diagnostics. Use SHAP only to explain model predictions with a training-fold background. With only five networks, uncertainty across network transfer is weakly identified; many pair rows do not create more independent rivers. Additional network/event evidence, not a different split alone, is needed to demonstrate transfer.

Reproduction and limitations
Run run.py with /opt/anaconda3/envs/river-hierarchy-rivgraph/bin/python, then report.py <completed output directory>. Every run creates a new directory. The earlier directories from this investigation are retained as superseded/incomplete attempts; only this completed, lineage-aware report is the deliverable. Sources are hashed and rechecked; prior routing/calibration/model snapshots were read only. This is exploratory reuse of an existing cohort, not untouched external validation. Group independence is limited to audited topology/selection/context overlap, not independence of network forcing or hydraulic calibration.
'''
(OUT/'REPORT.txt').write_text(text)
assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in extra_hashes.items())
(OUT/'report_provenance.json').write_text(json.dumps(dict(source_hashes=extra_hashes,source_unchanged=True),indent=2))
print(summary.to_string());print(view[view.features.eq('geometry_15')][['validation','weighting','r2','mae_minutes','zero_mae_minutes']].to_string(index=False));print('REPORT',OUT)
