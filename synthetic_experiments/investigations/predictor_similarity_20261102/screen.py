"""Response-blind similarity screen of the 134 empirical affine classes."""
from pathlib import Path
import json
import hashlib
import itertools
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[3]
SOURCE = Path('/Volumes/SyntheticResults20261027/analysis/candidate_reduction_20261030')
OUT = Path(__file__).parent / 'results'
OUT.mkdir(exist_ok=True)
inv = pd.read_csv(SOURCE / 'inventory.csv')
pairs = pd.read_csv(SOURCE / 'predictor_pairs.csv')
aff = pd.read_csv(SOURCE / 'affine_equalities.csv')
concepts = json.loads((ROOT / 'synthetic_experiments/notes/geometry_concept_inventory_20261102.json').read_text())['concepts']
concept = {v: g['name'] for g in concepts for v in g['candidates']}
eligible = sorted(inv.loc[(inv.minimum_type_coverage == 100) & inv.global_varying, 'variable'])
assert len(eligible) == 161
parent = {v: v for v in eligible}
def root(v):
    while parent[v] != v:
        v = parent[v]
    return v
for r in aff.itertuples():
    parent[root(r.b)] = root(r.a)
components = {}
for v in eligible:
    components.setdefault(root(v), []).append(v)
groups = sorted(components.values(), key=lambda x: tuple(x))
assert len(groups) == 134
# Labels only: no retained representative is selected. All members enter comparisons.
classes = {f'A{i:03d}': vs for i, vs in enumerate(groups, 1)}
ids = list(classes)
lookup = {frozenset((r.a, r.b)): r for r in pairs.itertuples()}
members = pd.DataFrame([dict(affine_class=k, variable=v, concept=concept[v],
                            class_size=len(vs)) for k, vs in classes.items() for v in vs])
members.to_csv(OUT / 'affine_class_members.csv', index=False)
records = []
for a, b in itertools.combinations(ids, 2):
    rs = [lookup[frozenset((x, y))] for x in classes[a] for y in classes[b]]
    # Worst member-pair rule makes the screen independent of arbitrary representatives.
    spearman = min(r.conservative_similarity for r in rs)
    pearson_values = []
    for r in rs:
        value = min(abs(r.pearson_pooled), r.min_abs_pearson) if np.isfinite(r.min_abs_pearson) else abs(r.pearson_pooled)
        pearson_values.append(value)
    pearson = min(pearson_values)
    records.append(dict(a=a, b=b, conservative_spearman=spearman,
                        absolute_pearson_floor=pearson,
                        joint_similarity=min(spearman, pearson),
                        pooled_abs_spearman=min(abs(r.spearman_pooled) for r in rs),
                        same_active_types=all(r.same_active_types for r in rs),
                        consistent_spearman_sign=all(r.consistent_sign for r in rs)))
cp = pd.DataFrame(records)
cp.to_csv(OUT / 'class_pair_similarity.csv', index=False)
summary = []
assignments = []
for criterion, column in [('rank', 'conservative_spearman'), ('rank_and_linear', 'joint_similarity')]:
    similarity = np.eye(len(ids))
    indices = {k: i for i, k in enumerate(ids)}
    for r in cp.itertuples():
        similarity[indices[r.a], indices[r.b]] = similarity[indices[r.b], indices[r.a]] = getattr(r, column)
    tree = linkage(squareform(1-similarity, checks=True), method='complete')
    for threshold in [.85, .90, .95, .99]:
        labels = fcluster(tree, 1-threshold+1e-12, criterion='distance')
        for k, label in zip(ids, labels):
            assignments.append(dict(criterion=criterion, threshold=threshold, affine_class=k, cluster=int(label)))
        sizes = pd.Series(labels).value_counts()
        summary.append(dict(criterion=criterion, threshold=threshold, groups=len(sizes),
                            possible_slot_reduction=134-len(sizes),
                            multi_class_groups=int((sizes>1).sum()), singleton_groups=int((sizes==1).sum()),
                            largest_group=int(sizes.max())))
        # Verify the complete-linkage guarantee for every resulting group.
        for label in sizes.index:
            ii = np.flatnonzero(labels == label)
            assert similarity[np.ix_(ii, ii)].min() >= threshold-1e-10
assign = pd.DataFrame(assignments)
assign.to_csv(OUT / 'cluster_membership.csv', index=False)
summ = pd.DataFrame(summary)
summ.to_csv(OUT / 'threshold_summary.csv', index=False)
details = assign.merge(members, on='affine_class')
details.to_csv(OUT / 'clusters_with_all_variables.csv', index=False)
cp[(cp.pooled_abs_spearman>=.95) & (cp.conservative_spearman<.95)].to_csv(OUT/'pooled_only_matches.csv',index=False)
fig, ax = plt.subplots(figsize=(8,4.5), constrained_layout=True)
for name,z in summ.groupby('criterion',sort=False):
    ax.plot(z.threshold,z.groups,'o-',label={'rank':'Spearman: consistent within types','rank_and_linear':'Spearman + Pearson requirement'}[name])
    for r in z.itertuples(): ax.annotate(str(r.groups),(r.threshold,r.groups),xytext=(0,7),textcoords='offset points',ha='center')
ax.axhline(134,color='gray',ls='--',label='134 starting affine classes')
ax.set(xlabel='Minimum absolute similarity',ylabel='Remaining predictor groups',xticks=[.85,.90,.95,.99],title='Predictor-only redundancy screen · no response-based selection')
ax.legend(loc='lower right');ax.grid(alpha=.15)
fig.savefig(OUT/'similarity_thresholds.png',dpi=180)
plt.close(fig)
lines=['# Predictor similarity screen — 2 November 2026', '',
       'Reference: SYN-PREDICTOR-SIMILARITY-2026-11-02.', '',
       '## Scope and method', '',
       'All 134 empirical affine classes are included (161 complete nonconstant representations). No earlier shortlist, centroid/peak responses, feature importance or predictive score is used. The 40 partial candidates and four constants remain outside this correlation screen. Source geometry is counted once across hydraulic approaches.', '',
       'Class-to-class similarity is the minimum over ALL constituent variable pairs, so no arbitrary class representative determines the result. The rank rule requires matching sets of treatments with variation, consistent Spearman sign, and the threshold in the pooled population AND every informative treatment. Complete linkage requires every pair inside a proposed cluster to satisfy the rule; chains of weakly related endpoints cannot merge. If both variables vary only between treatments, the evidence is pooled only and cannot establish within-treatment redundancy.', '',
       'A second, stricter screen additionally requires absolute Pearson correlation above the same threshold, pooled and within informative treatments. It measures linear as well as rank similarity. Thresholds are sensitivity choices, not significance tests. These clusters indicate empirical similarity, not universal equivalence or demonstrated lack of incremental prediction value. No candidate is deleted.', '',
       '## Counts', '', '| Rule | Threshold | Groups | Potential reduction from 134 | Singleton groups |', '|---|---:|---:|---:|---:|']
for r in summ.itertuples(): lines.append(f'| {r.criterion} | {r.threshold:.2f} | {r.groups} | {r.possible_slot_reduction} | {r.singleton_groups} |')
lines += ['', '## Detailed conservative starting point: rank and linear similarity ≥ 0.95', '',
          'Every member is listed; choosing a retained representative remains a separate formula/coverage decision. Classes containing several names are existing affine alternatives, not newly selected variables.', '']
sel=details[(details.criterion=='rank_and_linear') & (details.threshold==.95)]
for cluster,z in sel.groupby('cluster'):
    if z.affine_class.nunique()<2: continue
    lines += [f'### Similarity group {cluster}: {z.affine_class.nunique()} affine classes', '']
    for cid, zz in z.groupby('affine_class'):
        lines.append('- '+cid+': '+ '; '.join('`'+v+'`' for v in zz.variable))
    lines.append('')
lines += ['## Interpretation and next decisions', '',
          'Important: parent and child area, equivalent width and elongation can be strongly correlated because between-case scale dominates their differences. This does not make the parent-child difference uninformative. Preserve explicit change candidates and do not interpret a similarity cluster as permission to discard a state without checking its comparison role. Likewise, high correlation between slope variability and route variability follows partly from conserved head loss. These are conditional redundancy findings, not universal identities.', '',
          'Start review with the joint 0.95 clusters, compare the rank-only clusters and 0.90 sensitivity, and retain singletons. Choose canonical SI representations only after formula review. Correlated but physically different descriptors stay as documented alternatives until real-network checks or grouped predictive comparisons distinguish them. Topology-rich real networks can break equalities observed on this grid. This screen does not establish whether any candidate predicts the response.', '',
          'The complete cluster membership for all thresholds, class memberships, pairwise scores and pooled-only matches are exported beside this report.']
(OUT/'report.md').write_text('\n'.join(lines)+'\n')
sources=[SOURCE/x for x in ['inventory.csv','affine_equalities.csv','predictor_pairs.csv','coverage_by_type.csv']]
(OUT/'provenance.json').write_text(json.dumps(dict(reference='SYN-PREDICTOR-SIMILARITY-2026-11-02',
    sources={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
    script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    classes=134,representations=161,selection_uses_responses=False),indent=2))
print(summ.to_string(index=False))
print('Detailed report:',OUT/'report.md')
