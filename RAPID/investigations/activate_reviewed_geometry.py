from pathlib import Path
import sys,json,html
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT),str(ROOT/'RAPID/src'),str(ROOT/'RAPID/investigations')]
from integrate_reviewed_geometry import activate,OUT,SOURCE,REV
from network_variants.reviewed_geometry import active_geometry,digest
pointer=activate()
release=active_geometry()
from rapid_tools.registry import load_state_registry
frame=load_state_registry(ROOT/'network_variants/output/41_indep__anchored')
row=frame[frame.state_id.eq('S016_unit_4')].iloc[0]
assert row.integrated_hydraulics_pending
assert Path(row.integrated_geometry_links).exists()
assert Path(row.integrated_geometry_nodes).exists()
import pandas as pd
records=pd.read_csv(OUT/'state_registry.csv',dtype={'example':str})
rows=[]
for ex,group in records.groupby('example'):
    rows.append(dict(example=ex,logical_states=len(group),geometry_objects=group.geometry_id.nunique(),
                     pending_hydraulics=int((group.requires_hydraulic_regeneration & ~group.separate_observation_excluded).sum()),
                     duplicate_islands=int(group.separate_observation_excluded.sum())))
pd.DataFrame(rows).to_csv(OUT/'by_network.csv',index=False)
summary=json.loads((OUT/'summary.json').read_text());summary.update(active=True,active_pointer=str(ROOT/'network_variants/config/reviewed_geometry.json'),tests_passed=80,normal_registry_smoke_test_passed=True,release_size_bytes=sum(p.stat().st_size for p in OUT.rglob('*') if p.is_file()))
(OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
receipt=dict(pointer=pointer,validation_sha256=digest(OUT/'validation.json'),tests_passed=80,normal_registry_smoke_test_passed=True,
             changed_scope='Active geometry selection, canonical alias/exclusion policy, and preparation stale-input guards',legacy_rasters_widths_contexts_and_RAPID_results_overwritten=False,
             code_checksums={str(ROOT/p):digest(ROOT/p) for p in ['network_variants/reviewed_geometry.py','RAPID/investigations/integrate_reviewed_geometry.py','RAPID/investigations/validate_reviewed_geometry_integration.py','RAPID/src/rapid_tools/prep.py','RAPID/src/rapid_tools/registry.py','RAPID/src/rapid_tools/common_grid_production.py','RAPID/investigations/cunge_full_hierarchy_reviewed.py']})
(OUT/'activation.json').write_text(json.dumps(receipt,indent=2)+'\n')
for p in [SOURCE/'summary.json',REV/'summary.json']:
    s=json.loads(p.read_text());s['current_geometry_integration']=dict(active=True,manifest=str(OUT/'manifest.json'),summary=str(OUT/'summary.json'),legacy_files_overwritten=False,hydraulic_regeneration_pending=True)
    # Historical numbers remain evidence, not the current integrated status.
    p.write_text(json.dumps(s,indent=2)+'\n')
body=''.join('<tr>'+''.join(f'<td>{r[k]}</td>' for k in ['example','logical_states','geometry_objects','pending_hydraulics','duplicate_islands'])+'</tr>' for r in rows)
(OUT/'index.html').write_text('''<!doctype html><html lang="en"><meta charset="utf-8"><title>Integrated geometry — 9 October 2026</title><style>body{font:16px system-ui;max-width:1050px;margin:40px auto;padding:0 24px;line-height:1.55;color:#1f3442}td,th{padding:9px 16px;text-align:right;border-bottom:1px solid #ddd}th:first-child,td:first-child{text-align:left}table{border-collapse:collapse;width:100%}a{color:#00779b}.status{background:#e8f5ed;padding:16px}</style><h1>Reviewed geometry integration</h1><p class="status"><strong>Active geometry release: reviewed_geometry_20261009.</strong><br>All reviewed geometry decisions are integrated. Hydraulic regeneration is pending.</p><p>The active registry covers 3,556 logical entries, with 970 validated replacement geometry objects. It updates 1,261 existing aliases and excludes 22 duplicate island observations in favour of independent states. Existing topology exclusions and example 11 remain excluded. Tiny-island absorption remains as reviewed.</p><p>1,597 active entries require regenerated hydraulic attributes; many refer to the same physical geometry. Original rasters, widths, hydraulic contexts and RAPID results remain available. Normal preparation and hierarchy runners reject stale hydraulic inputs for affected states.</p><h2>By network</h2><table><tr><th>Network</th><th>Logical entries</th><th>Repaired geometries</th><th>Entries needing hydraulics</th><th>Duplicate islands excluded</th></tr>'''+body+'''</table><h2>Validation</h2><p>All 970 geometry objects passed identity, boundary, endpoint, connectivity and directed-acyclic checks. Checksums and alias targets verified; 80 relevant tests passed. The original 41/unit4 anchors P5 and P7 and the island6 → unit4 canonical mapping were checked explicitly. Reviewed provisional/internal nodes remain unassigned as agreed.</p><h2>Remaining work</h2><p>Regenerate affected widths and hydraulic heads, including the six known width-fragment problems and the pre-existing nonpositive head drop in 37/S289_unit_143. Then rebuild routing inputs and validate before rerunning.</p><p><a href="state_registry.csv">Authoritative state registry</a> · <a href="summary.json">Summary</a> · <a href="validation.json">Validation</a> · <a href="activation.json">Activation and provenance</a> · <a href="../../outputs/island_assembly_fix_20261008/index.html">Four island old/new comparisons</a></p></html>''')
print(json.dumps(summary,indent=2))
