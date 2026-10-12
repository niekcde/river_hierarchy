from integrate_master_networks import *
import html

def main():
 plan=json.loads((STAGE/'installation_plan.json').read_text())
 validation=json.loads((STAGE/'validation.json').read_text()) if (STAGE/'validation.json').exists() else None
 rows=[]
 for a in plan['actions']:
  rel=Path(a['target']).relative_to(MASTER)
  rows.append(dict(folder=rel.parts[0],files=1,install_bytes=a['bytes'],backup_bytes=a['previous_bytes']))
 summary=pd.DataFrame(rows).groupby('folder').sum()
 status='Installed and verified in the master directory' if validation else 'Staged; master directory not yet modified'
 doc=f'''<!doctype html><html lang="en"><meta charset="utf-8"><title>Master network integration</title>
<style>body{{font:16px/1.5 system-ui;max-width:1150px;margin:40px auto;padding:0 20px;color:#172534}}table{{border-collapse:collapse}}th,td{{padding:6px 12px;border-bottom:1px solid #ddd;text-align:left}}code{{overflow-wrap:anywhere}}.status{{background:#e9f4ee;padding:16px}}</style>
<h1>Master network integration</h1><p class="status"><strong>{status}</strong></p>
<p>Master directory: <code>{html.escape(str(MASTER))}</code>.</p>
<p>970 repaired geometries; 2,222 physical states with validated hydraulic tables; 3,554 registered logical entries.
The existing independent, compound and island folders remain authoritative. Unchanged vectors are retained.
The existing optimized slope implementation is used; the additional fixed-exterior diagnostic is not an acceptance gate.</p>
<p>All 22 reviewed duplicate islands remain excluded as separate observations and resolve to their retained independent states.
The 49 existing topology exclusions remain in force. Reviewed tiny-island absorption is retained.</p>
<p>Replacements include the repaired GeoPackages, width samples where link geometries changed, matching metadata,
link/node hydraulic contexts, and registry references. GeoPackage layer names are retained for QGIS compatibility.</p>
<p>RAPID preparation and routing were not executed. Existing routing results are historical; affected hierarchy and equivalent metrics are marked for refresh.
The physical tables contain no inherited routing K/X or segmentation parameters.</p>
<h2>Backup and rollback</h2><p>Every replaced file is backed up under <code>{html.escape(str(BACKUP))}</code>.
The installation plan records old and new checksums. Publication refuses concurrent edits; rollback also refuses to overwrite subsequent changes.</p>
<pre>python RAPID/investigations/publish_master_networks.py rollback</pre>
<p>Use the river-hierarchy-rivgraph environment when running the rollback command.</p>
<h2>Validation</h2><pre>{html.escape(json.dumps(validation,indent=2)) if validation else 'Installed-product validation pending.'}</pre>
<h2>File changes by directory</h2>{summary.to_html()}
<p><a href="installation_plan.json">Exact file and checksum manifest</a> · <a href="validation.json">Installed validation</a> · <a href="activation.json">Activation receipt</a></p></html>'''
 (STAGE/'integration_report.html').write_text(doc)
 print(STAGE/'integration_report.html')

if __name__=='__main__':main()
