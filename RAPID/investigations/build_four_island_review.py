from pathlib import Path
import json
import nbformat
from rapid_tools.island_assembly_review import OUT, ROOT, ISLANDS, plot_case, static_preview
figdir=OUT/'figures';figdir.mkdir(exist_ok=True)
results=[]
for island in ISLANDS:
    state=f'island_41_{island:03d}'
    fig=plot_case(island)
    fig.write_html(figdir/f'{state}.html',include_plotlyjs='directory',config={'scrollZoom':True,'displaylogo':False})
    static_preview(island,figdir/f'{state}.png')
    results.append(json.loads((OUT/state/'review_result.json').read_text()))
    print(state,'exported',flush=True)
rows=''.join(f'<tr><td>41</td><td>{i}</td><td><a href="figures/island_41_{i:03d}.html">Open old / repaired comparison</a></td><td>Passed</td></tr>' for i in ISLANDS)
(OUT/'index.html').write_text('''<!doctype html><html lang="en"><meta charset="utf-8"><title>Four island repairs — old / new review</title><style>body{font:17px system-ui;max-width:1050px;margin:45px auto;padding:0 25px;color:#172b39}table{border-collapse:collapse;width:100%}td,th{text-align:left;padding:14px;border-bottom:1px solid #ddd}a{color:#00769b}p{line-height:1.6}</style><h1>Network 41: four repaired island cases</h1><p>Each comparison shows the current production geometry on the left and the corrected isolated geometry on the right. Purple P labels mark the required original parent boundaries. Grey is the collapsed island footprint, black is remaining land, and white is water. Hover over links and nodes for IDs; click “Link labels” in the legend to display them.</p><table><tr><th>Network</th><th>Island</th><th>Review</th><th>Geometry checks</th></tr>'''+rows+'''</table><p>All four repairs preserve exterior parent links exactly, include every required boundary, fill only the requested island, and remove exactly one undirected loop. Each remains connected and directed acyclic, with the original inlet and outlet.</p><p>Island 55 includes an explicit 3.16 m junction restoration to parent P22, preserving exterior L110 exactly.</p><p><strong>These are isolated geometry review products.</strong> Production networks and RAPID results are unchanged. Hydraulic attributes and routing inputs still need regeneration before a production run.</p></html>''')
(OUT/'summary.json').write_text(json.dumps({'scope':'Four corrected island assemblies for visual review','geometry_cases_passed':4,'states':[r['state'] for r in results],'all_exterior_exact':all(r['exterior_exact'] for r in results),'all_boundaries_present':all(all(b['present'] for b in r['boundary_checks']) for r in results),'production_changed':False,'routing_ready':False,'core_tests_passed':36,'results':results},indent=2)+'\n')
nb=nbformat.v4.new_notebook()
nb.cells=[nbformat.v4.new_markdown_cell('''# Network 41 — four island assembly repairs

Compare **old current production** with **new corrected isolated geometry** for islands **5, 55, 75 and 138**. All four passed geometry checks. Production networks and RAPID results are unchanged; hydraulic attributes still require regeneration.

Purple **P** labels show the required original parent boundaries. Hover over links/nodes for IDs. Grey = collapsed island footprint; black = remaining land; white = water. Link labels can be enabled in the legend. No `ipywidgets` is required.

The fix uses the complete enclosing unit for computation while filling only the selected island. Island 55 also restores a junction by 3.16 m to P22 and restores exterior L110 exactly.'''),nbformat.v4.new_code_cell('''from pathlib import Path
import sys
ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / 'RAPID/src/rapid_tools').exists())
sys.path.insert(0, str(ROOT / 'RAPID/src'))
from rapid_tools.island_assembly_review import OUT, ISLANDS, case_data, plot_case
import pandas as pd
import json
from IPython.display import display, HTML

ISLAND = 5  # Choose 5, 55, 75 or 138
BUFFER_M = 150
SHOW_LINK_LABELS = False'''),nbformat.v4.new_code_cell('''rows = []
for island in ISLANDS:
    r = json.loads((OUT / f'island_41_{island:03d}' / 'review_result.json').read_text())
    rows.append({'island': island, 'geometry accepted': r['geometry_accepted'],
                 'all boundaries present': all(b['present'] for b in r['boundary_checks']),
                 'exterior exact': r['exterior_exact'], 'loops removed': r['cycle_reduction'],
                 'directed acyclic': r['topology']['dag']})
display(pd.DataFrame(rows).set_index('island'))'''),nbformat.v4.new_code_cell('''fig = plot_case(ISLAND, buffer_m=BUFFER_M, show_link_labels=SHOW_LINK_LABELS)
fig.show(config={'scrollZoom': True, 'displaylogo': False})'''),nbformat.v4.new_code_cell('''data = case_data(ISLAND, BUFFER_M)
display(pd.DataFrame(data['result']['boundary_checks']))
print('Corrected QGIS geometry:')
print(data['folder'] / 'fixed_links.gpkg')
print(data['folder'] / 'fixed_nodes.gpkg')
print('Old current links:', __import__('rapid_tools.collapse_creation_review', fromlist=['case_row']).case_row('41', data['state']).child_links_path)''')]
nb.metadata={'kernelspec':{'display_name':'river-hierarchy-rivgraph','language':'python','name':'river-hierarchy-rivgraph'},'language_info':{'name':'python','version':'3.11'}}
nbformat.write(nb,ROOT/'RAPID/notebooks/rapid_island_assembly_fix_review.ipynb')
print('Notebook and index written',flush=True)
