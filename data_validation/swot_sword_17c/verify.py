"""Independent arithmetic, join, and source-read checks for the saved validation."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
D = Path(__file__).resolve().parent
m = json.loads((D/'run_metadata.json').read_text())
n = pd.read_csv(D/'unique_node_comparison.csv')
e = pd.read_csv(D/'example_node_comparison.csv')
s = pd.read_csv(D/'example_summary.csv')
f = pd.read_csv(D/'file_manifest.csv')
a = pd.read_csv(D/'example_reaches.csv')
o = pd.read_parquet(D/'target_observations.parquet')
assert len(f)==m['files_scanned'] and f.status.eq('ok').all()
assert f.target_rows.sum()==len(o)==m['target_rows']
assert not n.duplicated(['node_id','reach_id']).any()
assert not e.duplicated(['example_id','node_id','reach_id']).any()
assert pd.read_csv(D/'reaches_absent_from_sword.csv').empty
assert set(a.example_id)==set(s.example_id)
for r in s.itertuples():
 assert r.sword_nodes==r.sword_valid+r.sword_missing==r.swot_valid+r.swot_missing
 assert r.net_gain==r.swot_valid-r.sword_valid==r.recovered_by_swot-r.sword_only
 assert r.sword_missing==r.recovered_by_swot+r.neither
# Re-read one whole granule without Parquet predicate pushdown, then compare selection.
picked = f.loc[f.target_rows.gt(0),'file'].iloc[len(f.loc[f.target_rows.gt(0)])//2]
source = pq.read_table(Path('/Volumes/PhD/SWOT/RiverSP_D_parq/node')/picked,columns=['node_id','reach_id','wse_sm','wse_sm_u','dark_frac','xtrk_dist']).to_pandas()
source['reach_id'] = source.reach_id.astype('int64')
source = source.loc[source.reach_id.isin(a.reach_id)]
saved = o.loc[o.source_file.eq(picked)]
assert len(source)==len(saved)
assert set(source.node_id.astype('int64'))==set(saved.node_id)
keep = source.wse_sm_u.le(1)&source.dark_frac.le(.5)&source.xtrk_dist.abs().between(10000,60000)&source.wse_sm.gt(0)
assert int(keep.sum())==int(saved.passes_repository_filter.sum())
# Recompute 100 medians from individual observation arrays using NumPy.
sample = n.loc[n.swot_valid].sample(n=min(100,int(n.swot_valid.sum())),random_state=17)
for r in sample.itertuples():
 values = o.loc[o.node_id.eq(r.node_id)&o.reach_id.eq(r.reach_id)&o.passes_repository_filter,'wse_sm'].to_numpy()
 assert len(values)==r.swot_observations
 assert np.isclose(np.quantile(values,.5),r.swot_p50,rtol=0,atol=1e-9)
checks = {'passed':True,'files_accounted_for':len(f),'examples_accounted_for':len(s),'unique_nodes':len(n),'independent_full_granule_read':picked,'independent_numpy_medians':len(sample),'coverage_identities':'passed','duplicate_join_check':'passed','all_example_reaches_present_in_sword':True}
(D/'verification.json').write_text(json.dumps(checks,indent=2)+'\n')
print(json.dumps(checks,indent=2))
