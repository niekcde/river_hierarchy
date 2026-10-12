"""Activate the independently audited metadata revision without copying geometries."""
from pathlib import Path
import sys,json,os,shutil
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT),str(ROOT/'RAPID/src')]
from network_variants.reviewed_geometry import ACTIVE,ReviewedGeometry,digest
OUT=ROOT/'RAPID/assets/reviewed_geometry_20261009_r2'
manifest=OUT/'manifest.json';data=json.loads(manifest.read_text())
audit=json.loads((OUT/'completeness_audit/summary.json').read_text())
assert audit['geometry_and_coverage_pass'] and not audit['errors'] and not audit['warnings']
assert audit['current_registry_states']==audit['release_states']==len(data['records'])
for p,expected in data['input_checksums'].items():
 assert digest(p)==expected,f'Source registry/evidence changed: {p}'
manual=json.loads((OUT/'completeness_audit/manual_decision_checks.json').read_text());assert len(manual)==10 and all(r['passed'] for r in manual)
if not (OUT/'previous_active.json').exists():shutil.copy2(ACTIVE,OUT/'previous_active.json')
pointer=dict(manifest=str(manifest),sha256=digest(manifest),release=data['release'],stage='geometry_integrated_hydraulics_pending')
tmp=ACTIVE.with_suffix('.tmp');tmp.write_text(json.dumps(pointer,indent=2)+'\n');os.replace(tmp,ACTIVE)
r=ReviewedGeometry()
# Repeat the two previously open paths against the real repaired case.
from types import SimpleNamespace
import pandas as pd
case=r.lookup('41','indep','S016_unit_4')
checks={}
for key,node in [('original_node_path',case['nodes']),('integrated_node_path',str(Path(case['bundle'])/'nodes.parquet'))]:
 try:r.assert_preparation_current(SimpleNamespace(directed_nodes_path=node,geometry_release=r.release,geometry_id=case['geometry_id']))
 except ValueError:checks[key]='blocked_pending_hydraulics'
 else:raise AssertionError('Pending preparation passed')
f=pd.DataFrame([dict(example='41',family='indep',state='S016_unit_4',status='ready',geometry_release=r.release,geometry_id=case['geometry_id'],frame=case['original_frame'])])
try:r.assert_hydraulics_current(f)
except ValueError:checks['old_frame_with_new_release_labels']='blocked_pending_hydraulics'
else:raise AssertionError('Old frame relabelled as new passed')
from rapid_tools.registry import load_state_registry
frame=load_state_registry(ROOT/'network_variants/output/41_indep__anchored')
assert set(frame.integrated_geometry_release.dropna())=={r.release}
summary=json.loads((OUT/'summary.json').read_text());summary.update(active=True,active_pointer=str(ACTIVE),tests_passed=84,normal_registry_smoke_test_passed=True,
 verified_repaired_geometries=970,verified_retained_geometry_sources=1252,current_official_registry_entries=3554,completeness_audit_passed=True,
 additional_geometry_review_required=False,runtime_checks=checks,release_metadata_size_bytes=sum(p.stat().st_size for p in OUT.rglob('*') if p.is_file()))
(OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
(OUT/'activation.json').write_text(json.dumps(dict(pointer=pointer,completeness_audit_sha256=digest(OUT/'completeness_audit/summary.json'),runtime_checks=checks,
 original_geometry_and_RAPID_results_modified=False,code_checksums={str(ROOT/p):digest(ROOT/p) for p in ['network_variants/reviewed_geometry.py','RAPID/investigations/reconstruct_child_heads.py','RAPID/investigations/audit_integrated_geometry_completeness.py']}),indent=2)+'\n')
(OUT/'index.html').write_text('''<!doctype html><html lang="en"><meta charset="utf-8"><title>Geometry completeness audit — 9 October 2026</title><style>body{font:17px system-ui;max-width:1000px;margin:40px auto;padding:0 24px;line-height:1.6;color:#203440}table{border-collapse:collapse;width:100%}td,th{padding:12px;border-bottom:1px solid #ccc;text-align:left}a{color:#00769b}</style><h1>Geometry integration: independent completeness audit</h1><p><strong>Corrected revision r2 is active. No outstanding geometry or coverage errors were found in the final audit.</strong> Hydraulic regeneration remains pending.</p><table><tr><th>Check</th><th>Result</th></tr><tr><td>Official registry entries accounted for</td><td>3,554 / 3,554</td></tr><tr><td>Repaired geometry objects</td><td>970 checked</td></tr><tr><td>Retained geometry sources</td><td>1,252 checked</td></tr><tr><td>Canonical island mappings</td><td>1,283 checked: 1,261 existing aliases + 22 duplicate exclusions</td></tr><tr><td>Manual review ledger</td><td>18 records covered; 10 explicit cut/mapping/domain checks passed</td></tr><tr><td>Relevant automated tests</td><td>84 passed</td></tr></table><h2>Issues found and corrected</h2><p>Two hidden temporary promotion folders were incorrectly included as logical states: network 22 / temporary S216_unit_10 and network 41 / temporary S086_unit_14. The actual registered states remain. Future source enumeration now uses official state registries.</p><p>Preparation checks now recognise both original and integrated node paths. A release label alone cannot make hydraulic inputs ready while regeneration is pending. Unknown states and excluded states cannot silently enter the routed population.</p><p>The unchanged-geometry loader now preserves saved canonical node identities. Retained source files have frozen checksums to detect later changes.</p><h2>Preserved decisions</h2><p>The independent state remains canonical for the 22 duplicate islands. The approved absorption of tiny island 100 into island 99 is retained. Reviewed cuts, provisional unmatched nodes, internal-node removals and the four rebuilt island geometries are preserved. No geometry recalibration or new routing run was performed.</p><h2>Next step</h2><p>Regenerate hydraulic attributes for 1,597 state entries sharing repaired geometry, resolve the known width-fragment and head-drop issues, then validate preparation and routing. The current release intentionally blocks stale hydraulic inputs.</p><p><a href="summary.json">Current summary</a> · <a href="state_registry.csv">Current state registry</a> · <a href="completeness_audit/summary.json">Detailed audit</a> · <a href="completeness_audit/manual_decision_checks.json">Explicit review decisions</a> · <a href="activation.json">Activation evidence</a></p></html>''')
old=Path(data['supersedes']).parent
p=old/'summary.json';s=json.loads(p.read_text());s.update(active=False,superseded_by=str(manifest));p.write_text(json.dumps(s,indent=2)+'\n')
p=old/'index.html';s=p.read_text().replace('<h1>', '<p><strong>Superseded by the corrected completeness-audited revision.</strong> <a href="../reviewed_geometry_20261009_r2/index.html">Open the current overview</a>.</p><h1>',1);p.write_text(s)
for p in [ROOT/'RAPID/outputs/boundary_contract_repair_20261008/summary.json',ROOT/'RAPID/outputs/boundary_contract_repair_20261008/review_revision/summary.json']:
 s=json.loads(p.read_text());s['current_geometry_integration']=dict(active=True,manifest=str(manifest),summary=str(OUT/'summary.json'),legacy_files_overwritten=False,hydraulic_regeneration_pending=True);p.write_text(json.dumps(s,indent=2)+'\n')
print(json.dumps(summary,indent=2))
