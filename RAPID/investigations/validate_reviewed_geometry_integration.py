"""Validate the integrated release and its alias/preparation behavior."""
from pathlib import Path
import sys,json
import pandas as pd
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT),str(ROOT/'RAPID/src')]
from network_variants.reviewed_geometry import ReviewedGeometry,digest
OUT=ROOT/'RAPID/assets/reviewed_geometry_20261009'
release=ReviewedGeometry(OUT/'manifest.json')
manifest=json.loads((OUT/'manifest.json').read_text())
for path,expected in manifest['input_checksums'].items():
    assert digest(path)==expected, f'Integration input changed: {path}'
seen=set();aliases=0;duplicates=0;pending=0
for record in release.records.values():
    canonical=release.canonical(record)
    aliases+=bool(record.get('canonical_key'))
    duplicates+=bool(record.get('separate_observation_excluded'))
    if canonical.get('bundle') and canonical['geometry_id'] not in seen:
        links,nodes,contract,matches,_=release.load(record['example'],record['family'],record['state'])
        needed=set(contract.get('reviewed_required_boundary_nodes',contract['boundary_nodes']))
        assert needed<=set(nodes.parent_identity)
        seen.add(canonical['geometry_id'])
    if record['geometry_status']!='excluded':
        assert Path(canonical['links']).exists() and Path(canonical['nodes']).exists()
    pending+=bool(canonical.get('requires_hydraulic_regeneration') and not record.get('separate_observation_excluded'))
assert duplicates==22 and aliases==1283
assert len(seen)==970
# Explicitly exercise the original 41/unit4 missing-anchor bug and island6 alias.
links,nodes,contract,matches,_=release.load('41','island','island_41_006')
assert {5,7}<=set(nodes.parent_identity)
assert release.canonical(release.lookup('41','island','island_41_006'))['state']=='S016_unit_4'
# Manual provisional and internal decisions survive the integrated representation.
for ex,state,forbidden in [('22','S135_unit_211',{418}),('37','S051_unit_256',{509}),('37','S289_unit_143',{296}),('41','S061_unit_57',{112,115})]:
    _,nodes,_,_,_=release.load(ex,'indep',state)
    assert not forbidden & set(nodes.parent_identity)
# All exported pending rows must have NO routable stale frame.
for p in OUT.glob('*/inventory.csv'):
    f=pd.read_csv(p,dtype={'example':str})
    need=f.status.eq('pending_reviewed_geometry_hydraulics')
    assert f.loc[need,'frame'].isna().all()
    assert f.loc[f.status.eq('excluded_duplicate'),'frame'].isna().all()
# The actual old network41 hierarchy must now be rejected by the guard.
old=pd.read_csv(ROOT/'RAPID/assets/bounded_robust_slopes_20261002/child_head_reconstruction_20261003/41/inventory.csv',dtype={'example':str})
try:release.assert_hydraulics_current(old)
except ValueError as e:stale_message=str(e)
else:raise AssertionError('Old hydraulic inventory incorrectly accepted')
result=dict(manifest_sha256=digest(OUT/'manifest.json'),logical_states=len(release.records),verified_geometry_objects=len(seen),canonical_aliases=aliases,
            excluded_duplicate_observations=duplicates,states_pending_hydraulics=pending,
            input_checksums_valid=True,bundle_checksums_valid=True,manual_decisions_preserved=True,
            missing_anchor_regression_passed=True,stale_hydraulics_rejected=True,stale_message=stale_message,
            routing_ready=False)
(OUT/'validation.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
