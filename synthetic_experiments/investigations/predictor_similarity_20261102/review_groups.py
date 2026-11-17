from pathlib import Path
import pandas as pd
p=Path('synthetic_experiments/investigations/predictor_similarity_20261102/results')
d=pd.read_csv(p/'clusters_with_all_variables.csv');d=d[(d.criterion=='rank_and_linear')&(d.threshold==.9)]
pairs=pd.read_csv(p/'class_pair_similarity.csv')
reviews={
1:('Parent allocation diversity','H_p versus I_p. Both accumulate exposure-weighted local splitting diversity.','Prefer parent route entropy H_p. Keep intensity as a documented alternative; Shannon and quadratic diversity are not universally equivalent.'),
3:('Parent path-width diversity','H_q versus exp(H_q): exact nonlinear re-expression, not affine equivalence.','Prefer H_q; effective path count can be derived for interpretation. This is the same information for a split-order tree model.'),
5:('Parent local allocation asymmetry','Visit-weighted means of max(a), 1/sum(a²), and local normalised Shannon entropy.','Provisionally prefer parent dominance for direct interpretation. Binary splits dominate this design; dominance does not describe the distribution among all minor branches at multiway splits. Recheck these alternatives in real networks.'),
9:('Slope scale','Parent and child mean/minimum slope, a parent-child maximum-slope affine class, and parent allocated-length-weighted S^(-1/2).','Do not reduce all six classes to one yet. Retain parent mean slope, parent minimum slope and inverse-root slope as distinct candidates pending real-network checks. Parent/child means can be represented by parent mean plus the existing mean-slope change. Maximum slope is a design-limited alternative. Common reference-slope variation drives much of the correlation; inverse-root slope emphasises flat reaches.'),
11:('Parent distance heterogeneity','Visit-weighted mean local branch contrast versus whole-route SD sigma_L,p.','Prefer parent route SD. It is the aggregate route-distance heterogeneity. Local mean contrast is not the same as the sum of contrast variances; distributed branching can break the observed similarity.'),
13:('Parent relative heterogeneity','Unweighted route CV, route range/L_eq, and slope CV.','Among the two unweighted distance measures, prefer CV as the fuller distribution summary, pending comparison with weighted CV. Retain slope CV separately until real-network checks: conserved head loss couples slopes and lengths here. Do not call the unweighted descriptor mathematically wrong.'),
19:('Parent and child area','A_p versus A_c, where A=sum(LB).','Prefer parent area as scale only if the existing delta A is retained: A_c=A_p+delta A exactly. High parent-child correlation alone does not show the difference is irrelevant.'),
20:('Width scale','Parent length-weighted reach width versus parent and child sum of harmonic route widths.','Retain parent mean reach width and parent equivalent path width as distinct candidates for now. The latter counts widths across enumerated routes and can repeat shared reaches. Child equivalent width can be reconstructed if its existing difference is retained.'),
29:('Elongation','Parent/child mu_L²/A and parent/child L_eq/W_eq.','Retain one parent representative of EACH definition pending comparison: area elongation and equivalent elongation. They are different shape definitions. Child forms can be reconstructed with their respective existing differences, if retained; high correlation does not choose a definition.'),
32:('Absolute length scale','Parent and child mean/max/equivalent lengths, parent total channel length, and the common parent-child shortest length.','Use parent allocated mean length as the primary scale representative, with existing delta mean length retaining child information. Preserve total length, shortest/longest length and equivalent length as real-network challengers: they have different meanings despite common scale variation on this design. This is a provisional preference, not proof of interchangeability.'),
41:('Change in allocation diversity','delta H versus delta I.','Prefer route-entropy change for consistency with parent entropy; splitting intensity remains a nonlinear-diversity alternative.'),
51:('Change in distance heterogeneity','delta longest route length versus delta route SD.','Prefer delta route SD: it accounts for allocation, whereas the longest route can carry negligible flow. Recheck long, low-allocation routes in real networks before treating the longest-route change as redundant.'),
53:('Distance change versus Manning combination','delta mu_L versus delta G_M.','Retain BOTH. G_M has explicit width/allocation/slope scaling and whole-network scope; delta mu_L describes the evaluated B domain. Correlation does not establish that the additional scaling is dispensable. This is a key comparison, not a safe deletion.'),
55:('Relative distance change','delta(mu_L/L_min), an affine class containing mean-length/area/elongation/sinuosity ratios, and log1p(mean-length ratio).','Prefer raw mean-route-length ratio as canonical; log1p is an exact invertible transformation on this positive domain. Equalities with area and elongation ratios are design-specific alternatives requiring real-network checks.'),
56:('Relative heterogeneity change','delta allocated route CV versus child/parent longest-route length.','Provisionally prefer delta allocated route CV for allocation-aware heterogeneity. The maximum-length ratio measures an extremal route change, not the same quantity; retain as a real-network challenger.'),
59:('Change in relative heterogeneity','delta unweighted path CV versus delta slope CV.','Retain both for now. Conserved head loss couples these on this grid; heterogeneous real slopes can separate them. The unweighted measure should later be compared with its weighted counterpart.'),
68:('Relative allocation diversity','Child/parent entropy ratio, child/parent splitting-intensity ratio, log1p entropy ratio.','Prefer raw route-entropy ratio; log1p adds no independent information. The quadratic-diversity ratio remains a documented alternative, not a universal duplicate.'),
69:('Child diversity','Child allocated-route entropy, summed splitting intensity, and harmonic-path-width dominance/effective count/entropy.','Provisionally prefer child allocated-route entropy. Child H can also be reconstructed from parent H and delta H if both are retained. Recheck route-weight definitions and multiway children in real networks before discarding harmonic-width alternatives.'),
70:('Child distance spread','Child branch variance sum V versus child route SD sigma_L. For this single-source/sink DAG allocation, V=sigma_L².','Prefer route SD in metres. The variance sum is mathematically determined by SD; this is a strong consolidation with stated graph/allocation assumptions.'),
71:('Child relative heterogeneity','Child unweighted route CV, normalised range and slope CV.','Prefer route CV over range within the unweighted family, but retain slope CV separately until real-network checks. Same distinction as the parent group.'),
77:('Heterogeneity of local allocation','Visit-weighted SD of local dominance versus SD of local effective branch count.','Provisionally prefer dominance SD, consistent with mean dominance. Averages/SDs of nonlinear transforms are not equivalent in general; real multiway splits require a check.'),
84:('Narrow-width emphasis','Child harmonic reach width versus length-weighted 10th-percentile width.','Retain both for now: harmonic width uses the full distribution with emphasis on narrow reaches; P10 is a quantile. Similarity on a grid with few width classes need not survive real width distributions.')}
lines=['# Review of the 22 predictor-similarity groups','', 'Reference: SYN-SIMILARITY-GROUP-REVIEW-2026-11-02.','',
'Population: 134 empirical affine classes; joint absolute Spearman/Pearson threshold 0.90, pooled and within informative treatments under the consistency rules in the screen. There are 69 singleton classes and 22 multi-class clusters containing 65 classes. One representative per cluster gives 91 slots, but formula review can justify more. Cluster IDs below are the actual IDs in cluster_membership.csv, not consecutive row numbers.','',
'These are recommendations for discussion, not applied deletions or a newly selected model. They use definitions and predictor similarity only; no response performance is used. Canonical lengths are in metres. Existing affine aliases are displayed in full so that old empirical equivalences are not mistaken for verified identities.','',
'## Equation key','',
r'- $\mu_L=\sum_r\pi_rL_r$; $\sigma_L^2=\sum_r\pi_r(L_r-\mu_L)^2$; allocated $CV_L=\sigma_L/\mu_L$.',
r'- $A=\sum_e L_eB_e$; $\bar B=\sum_eL_eB_e/\sum_eL_e$; $\bar S=\sum_eL_eS_e/\sum_eL_e$. Reach CVs use the same length weights.',
r'- $a_{vj}=B_{vj}/\sum_kB_{vk}$; $p_v$ is visit probability; $D_v=\max_j a_{vj}$; $N_v=1/\sum_ja_{vj}^2$; $h_v=-\sum_ja_{vj}\ln a_{vj}/\ln k_v$. Local means and SDs are weighted by $p_v$.',
r'- $H=-\sum_r\pi_r\ln\pi_r$; $I=\sum_{v\in\mathrm{bif}}p_v(1-\sum_ja_{vj}^2)$.',
r'- $W_r=L_r/\sum_{e\in r}(L_e/B_e)$; $q_r=W_r/\sum_sW_s$; $W_{eq}=\sum_rW_r$; $L_{eq}=\sum_rq_rL_r$; $H_q=-\sum_rq_r\ln q_r$; effective width-path count $=\exp(H_q)$. These weights differ from routed allocation $\pi_r$.',
r'- Unweighted $CV_0=\sqrt{N^{-1}\sum_r(L_r-\bar L_0)^2}/\bar L_0$; $\bar L_0=N^{-1}\sum_rL_r$; normalised range $=(L_{max}-L_{min})/L_{eq}$.',
r'- Local distance contrast $c_v^2=\sum_ja_{vj}(L_{vj}+\mu_{L,j}-\bar d_v)^2$; mean contrast $=\sum_vp_vc_v/\sum_vp_v$; variance sum $=\sum_vp_vc_v^2=\sigma_L^2$.',
r'- Parent inverse-root slope $=\sum_ef_eL_eS_e^{-1/2}/\sum_ef_eL_e$.',
r'- $G_M=\sum_eL_eB_e^{2/5}f_e^{3/5}S_e^{-3/10}$ over the whole network; candidate is $\Delta G_M$.',
r'- Harmonic reach width $=\sum_eL_e/\sum_e(L_e/B_e)$; P10 is the length-weighted width quantile.',
r'- $\Delta x=x_c-x_p$; ratio $=x_c/x_p$; $p,c$ denote parent and child.','']
rows=[]
for k,z in d.groupby('cluster'):
 if z.affine_class.nunique()<2:continue
 title,meaning,advice=reviews[k]
 ids=set(z.affine_class)
 pp=pairs[pairs.a.isin(ids)&pairs.b.isin(ids)]
 lines += [f'## Group {k}: {title}', '',f"{len(ids)} affine classes. Minimum conservative Spearman: {pp.conservative_spearman.min():.4f}; minimum absolute Pearson floor: {pp.absolute_pearson_floor.min():.4f}.",'',meaning,'']
 for a,zz in z.groupby('affine_class'):lines.append('- '+a+': '+ '; '.join('`'+x+'`' for x in zz.variable))
 lines += ['', '**Recommendation:** '+advice,'']
 rows.append(dict(group=k,classes=len(ids),topic=title,minimum_spearman=pp.conservative_spearman.min(),minimum_pearson=pp.absolute_pearson_floor.min(),recommendation=advice))
assert len(rows)==22 and sum(x['classes'] for x in rows)==65
lines += ['## Overall assessment','',
'The strongest consolidations are invertible transformations (entropy/effective count, raw/log1p ratios), variance/SD identities, and explicit parent-plus-difference reconstruction when the difference is retained. Other groupings mostly identify plausible alternatives, not proven redundancy. In particular do not merge distance change with Manning geometry, mean slope with slow-reach slope emphasis, or route heterogeneity with slope heterogeneity solely because this grid correlates them. The 91 slots are a screening organisation, not a forced final predictor count.','',
'Sources: predictor_similarity_20261102/results cluster and pair tables; RAPID/investigations/parent_child_predictor_exploration/features.py; synthetic_experiments/investigations/geometry_predictor_screen.py; synthetic_experiments/investigations/current_predictor_bridge.py; concept inventory dated 2026-11-02.']
(p/'group_review_090.md').write_text('\n'.join(lines)+'\n')
pd.DataFrame(rows).to_csv(p/'group_review_090.csv',index=False)
print(pd.DataFrame(rows)[['group','classes','topic','minimum_spearman','minimum_pearson']].to_string(index=False))
