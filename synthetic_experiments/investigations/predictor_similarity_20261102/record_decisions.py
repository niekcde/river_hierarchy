"""Record user-accepted concept review without deleting source features."""
from pathlib import Path
import pandas as pd

HERE = Path(__file__).resolve().parent
OUT = HERE / 'results'
members = pd.read_csv(OUT / 'clusters_with_all_variables.csv')
members = members[(members.criterion == 'rank_and_linear') & (members.threshold == .9)]
reviews = pd.read_csv(OUT / 'group_review_090.csv').set_index('group')
# Explicit representatives from the accepted review. Multiple entries preserve
# distinctions, not a response-based optimisation. Affine aliases remain attached.
picks = {
    1: ['A054'], 3: ['A133'], 5: ['A049'],
    9: ['A131', 'A132', 'A088'], 11: ['A058'],
    13: ['A127', 'A130'], 19: ['A043'], 20: ['A069', 'A126'],
    29: ['A042', 'A124'], 32: ['A057'], 41: ['A028'],
    51: ['A032'], 53: ['A031', 'A084'], 55: ['A073'],
    56: ['A029'], 59: ['A112', 'A115'], 68: ['A076'],
    69: ['A006'], 70: ['A011'], 71: ['A096', 'A099'],
    77: ['A050'], 84: ['A016', 'A018'],
}
# Choose SI/canonical labels where this review explicitly chooses an affine member.
preferred = {'A028': 'B_delta_route_entropy', 'A031': 'B_delta_route_length_mean',
             'A032': 'B_delta_route_length_sd', 'A029': 'B_delta_route_length_cv',
             'A073': 'B_ratio_route_length_mean'}
rows = []
for cluster, z in members.groupby('cluster'):
    multi = z.affine_class.nunique() > 1
    if multi:
        assert cluster in picks
        assert set(picks[cluster]) <= set(z.affine_class)
    for cid, zz in z.groupby('affine_class'):
        selected = not multi or cid in picks[cluster]
        names = sorted(zz.variable)
        representative = preferred.get(cid, names[0] if len(names) == 1 else '')
        rows.append(dict(affine_class=cid, similarity_group=cluster,
                         status='working_candidate' if selected else 'documented_alternative',
                         representative=representative,
                         affine_members='; '.join(names),
                         reason=reviews.loc[cluster, 'recommendation'] if multi else 'Singleton at joint 0.90; unchanged by group review.',
                         stage='user-accepted group review, 2026-11-02'))
ledger = pd.DataFrame(rows)
assert len(ledger) == 134
assert (ledger.status == 'working_candidate').sum() == 100
assert sum(map(len, picks.values())) == 31
ledger.to_csv(OUT / 'accepted_class_decisions.csv', index=False)
ledger[ledger.status == 'working_candidate'].to_csv(OUT / 'working_100_classes.csv', index=False)
print(ledger.status.value_counts().to_string())
