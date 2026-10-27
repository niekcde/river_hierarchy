"""Freeze strategy-3 groups with one singleton per segmented full collapse."""
import hashlib
import json
from pathlib import Path
import pandas as pd
from slope_ratio_audit import OUT

SOURCE=OUT/'validation_group_inventory_20261023'
DEST=OUT/'validation_groups_full_collapse_singletons_20261023'


def main():
    DEST.mkdir(exist_ok=True)
    paths=[SOURCE/'records.parquet',SOURCE/'group_assignments.csv',SOURCE/'manifest.json',Path(__file__)]
    hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    records=pd.read_parquet(SOURCE/'records.parquet')
    groups=pd.read_csv(SOURCE/'group_assignments.csv',dtype={'example':str})
    regular=groups[groups.scenario.eq('family_and_reach')][['example','state','group_id']].copy()
    segmented=records[records.cohort.eq('segmented')][['example','state']].copy()
    segmented['group_id']=segmented.example+'_full_collapse_'+segmented.state
    assert len(segmented)==15 and segmented.example.nunique()==15
    assignments=pd.concat([regular,segmented],ignore_index=True)
    assert len(assignments)==2185 and not assignments.duplicated(['example','state']).any()
    assert assignments.group_id.nunique()==450
    d=records.merge(assignments,on=['example','state'],how='outer',validate='one_to_one',indicator=True)
    assert d._merge.eq('both').all();d=d.drop(columns='_merge')
    sizes=d.groupby('group_id').size()
    assert sizes.loc[segmented.group_id].eq(1).all()
    assert d.groupby('group_id').example.nunique().eq(1).all()
    assert set(segmented.group_id).isdisjoint(regular.group_id)
    pd.testing.assert_frame_equal(d[d.cohort.eq('original')][regular.columns].sort_values(['example','state']).reset_index(drop=True),
        regular.sort_values(['example','state']).reset_index(drop=True))
    d['full_collapse_overlap_exception']=d.cohort.eq('segmented')
    d.to_parquet(DEST/'records.parquet',index=False)
    d[['example','state','cohort','edit_type','response_bin','group_id','full_collapse_overlap_exception']].to_csv(DEST/'group_assignments.csv',index=False)
    d.groupby('example').agg(records=('state','size'),groups=('group_id','nunique'),
        full_collapse_singletons=('full_collapse_overlap_exception','sum')).to_csv(DEST/'network_summary.csv')
    d.groupby(['edit_type','response_bin']).agg(records=('state','size'),groups=('group_id','nunique'),networks=('example','nunique')).to_csv(DEST/'stratum_support.csv')
    manifest=dict(sources=hashes,records=2185,regular_groups=435,full_collapse_singletons=15,total_groups=450,
        decision='User-authorized: each segmented full-network collapse is its own group in strategy 3. Regular compound/nested/member/island grouping is preserved exactly.',
        overlap='Intentional full-collapse exception: its constituent regular groups may be on the other side of a split. This is known-network related-state prediction, not non-overlapping or unseen-network validation.',
        strategies_1_2='Whole network remains the group, including its full collapse. This exception does not apply.',
        fitting='No folds selected, no response/feature/weight changes and no fits performed. Group assignment does not impose special weight or a separate model on full-collapse cases.',
        production_modified=False)
    for path,h in hashes.items():assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==h
    (DEST/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print('Prepared 2185 records in 450 groups: 435 unchanged regular groups + 15 full-collapse singleton groups.')


if __name__=='__main__':main()
