"""Review B(t) failures without changing selected plateau levels or clipping Q-B."""
from pathlib import Path
import base64
import gzip
import json
import re

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import nbformat

from calculate_reviewed_distribution_20261013 import ROOT, OUT, cases, pairs, read_wave

source = ROOT / 'RAPID/outputs/calibration_review/15_network_plateau_review.html'
original = source.read_text()
payload = json.loads(gzip.decompress(base64.b64decode(re.search(r'<script id="payload"[^>]*>(.*?)</script>', original, re.S)[1])))
states = pd.read_csv(OUT / 'state_metrics.csv', dtype={'example_id': str})
base = states[states.state_id.eq('S000_base')].set_index('example_id')
fig, axes = plt.subplots(5, 3, figsize=(15, 17))
for panel, ax in zip(payload['panels'], axes.flat):
    ex = panel['example']
    row = base.loc[ex]
    origin = pd.Timestamp(panel['origin']).timestamp()
    start, end = pd.Timestamp(row.baseline_start_utc).timestamp(), pd.Timestamp(row.baseline_end_utc).timestamp()
    t, q = read_wave(pairs[pairs.example_id.eq(ex)].iloc[0].parent_outlet_parquet)
    tt = np.r_[start, t[(t > start) & (t < end)], end]
    qq = np.interp(tt, t, q)
    bg = np.interp(tt, [start, end], [row.baseline_start_cms, row.baseline_end_cms])
    tx = (tt - origin) / 86400
    panel['background_note'] = f"Network {ex}: minimum Q−B = {row.minimum_excess_cms:.6g} m³/s; below-background volume / positive volume = {row.negative_to_positive_volume_pct:.6g}%. Base status: {row.distribution_status}. Child backgrounds must also pass. No negative excess is discarded."
    panel['series'].append(dict(type='scatter', mode='lines', x=tx.tolist(), y=bg.tolist(),
        name='Shared linear background B(t)', legendgroup='linear_background',
        line=dict(color='#9b59b6', width=2, dash='dash'), hovertemplate='B(t): %{y:.3f} m³/s<extra></extra>'))
    bad = qq < bg - 1e-8
    # Paired traces show only the vertical region where the background exceeds Q.
    panel['series'].extend([
        dict(type='scatter', mode='lines', x=tx.tolist(), y=np.where(bad, qq, np.nan).tolist(),
             name='Below-background Q', legendgroup='deficit', showlegend=False,
             line=dict(width=0), hoverinfo='skip'),
        dict(type='scatter', mode='lines', x=tx.tolist(), y=np.where(bad, bg, np.nan).tolist(),
             name='Background exceeds hydrograph', legendgroup='deficit', fill='tonexty',
             fillcolor='rgba(230,140,40,0.35)', line=dict(width=0), hoverinfo='skip')])
    for trace in panel['series']:
        trace['y'] = [None if v is None or not np.isfinite(v) else v for v in trace['y']]
    # Event view is now the reviewed window, so the background can be inspected.
    panel['event_x'] = [float(tx[0]), float(tx[-1])]
    px = (tt - start) / 86400
    ax.plot(px, qq, color='#ce3d45', lw=1.3, label='Base Cunge')
    ax.plot(px, bg, color='#9b59b6', lw=1.3, ls='--', label='Shared B(t)')
    ax.fill_between(px, qq, bg, where=bad, color='#e68c28', alpha=.5, interpolate=True)
    ax.set(title=f"{ex} · deficit / positive volume: {row.negative_to_positive_volume_pct:.4g}%",
           xlabel='Days from reviewed start', ylabel='Q (m³/s)')
    ax.grid(alpha=.2)
axes.flat[0].legend(fontsize=8)
fig.suptitle('Review the shared straight-line background\nOrange: B(t) exceeds the base hydrograph; no clipping applied', fontsize=16)
fig.tight_layout(rect=(0,0,1,.96))
fig.savefig(OUT / 'linear_background_review.png', dpi=140)
plt.close(fig)

library = original.split('<script>', 1)[1].split('</script>', 1)[0]
packed = base64.b64encode(gzip.compress(json.dumps(payload, separators=(',', ':'), allow_nan=False).encode())).decode()
script = r'''
let payload;const el=id=>document.getElementById(id);
function render(){
 const single=el('network').value,mode=el('view').value,all=single==='all';
 const panels=all?payload.panels:payload.panels.filter(p=>p.example===single);
 let data=[],seen=new Set();
 panels.forEach((p,i)=>p.series.forEach(t=>{const suffix=all&&i?String(i+1):'';const show=t.showlegend!==false&&!seen.has(t.legendgroup);if(show)seen.add(t.legendgroup);data.push({...t,xaxis:'x'+suffix,yaxis:'y'+suffix,showlegend:show});}));
 let layout;
 if(all){layout=JSON.parse(JSON.stringify(payload.layouts[mode]));layout.shapes=[];layout.annotations.forEach((a,i)=>a.text='Network '+panels[i].example);panels.forEach((p,i)=>{const s=i?String(i+1):'';layout['xaxis'+s].range=p[mode+'_x'];layout['yaxis'+s].range=p[mode+'_y'];});}
 else{const p=panels[0];layout={height:680,margin:{l:80,r:30,t:80,b:70},xaxis:{range:p[mode+'_x'],title:{text:'Days from '+p.origin+' (UTC)'}},yaxis:{range:p[mode+'_y'],title:{text:'Q (m³/s)'}},legend:{orientation:'h',y:1.1}};}
 layout.paper_bgcolor='white';layout.plot_bgcolor='white';el('plot').style.height=layout.height+'px';Plotly.react(el('plot'),data,layout,{responsive:true,scrollZoom:true});
 el('notes').textContent=all?'Select a network to inspect its background and deficit. Full, reviewed event and peak views are available.':panels[0].background_note;
}
(async()=>{const bytes=Uint8Array.from(atob(el('payload').textContent.trim()),c=>c.charCodeAt(0));payload=JSON.parse(await new Response(new Blob([bytes]).stream().pipeThrough(new DecompressionStream('gzip'))).text());payload.panels.forEach(p=>el('network').add(new Option('Network '+p.example,p.example)));el('network').onchange=render;el('view').onchange=render;render();})();
'''
table_columns = ['minimum_excess_cms', 'negative_to_positive_volume_pct', 'distribution_status']
table = base[table_columns].to_html(float_format=lambda x: f'{x:.6g}')
html = '''<!doctype html><meta charset="utf-8"><title>15-network background review</title>
<style>body{font:14px Arial;color:#202a34;margin:24px}p{max-width:1200px;line-height:1.5}.controls{position:sticky;top:0;background:#f4f6f9;padding:12px;z-index:20}select{padding:8px}table{border-collapse:collapse}th,td{padding:8px;border-bottom:1px solid #ddd}</style>
<h1>Shared straight-line background: review before volume metrics</h1>
<p>Blue: applied input. Red: base Cunge. Purple: B(t), connecting the accepted start and end levels on their selected dates. Orange marks where B(t) exceeds the base hydrograph. Start/end plateau choices are unchanged. The same line and absolute-time window apply to each parent–child pair; children require their own check.</p>
<p>Negative excess is not discarded. “Deficit / positive volume” compares the integrated below-background deficit with the integrated positive excess; it is not a flood-volume response. A tiny deficit and a substantial baseline mismatch are distinguished by the numerical values. The linear reconstruction of one-minute means is used consistently for endpoint crossings and volume integrals.</p>
<div class="controls">Network <select id="network"><option value="all">All 15</option></select> View <select id="view"><option value="full">Full view</option><option value="event" selected>Reviewed event</option><option value="peak">Peak</option></select></div>
<p id="notes"></p><div id="plot"></div><h2>Base hydrograph checks</h2>TABLE
<script>LIBRARY</script><script id="payload" type="application/octet-stream">PAYLOAD</script><script>SCRIPT</script>'''
(OUT / '15_network_background_review.html').write_text(html.replace('TABLE',table).replace('LIBRARY',library).replace('PAYLOAD',packed).replace('SCRIPT',script))

book = ROOT / 'RAPID/notebooks/hydrograph_response_worked_examples_20261012.ipynb'
notebook = nbformat.read(book, as_version=4)
tag = 'reviewed_background_results_20261013'
notebook.cells = [c for c in notebook.cells if c.metadata.get('audit_section') != tag]
coverage = pd.read_csv(OUT / 'full_coverage.csv', dtype={'example_id': str})
valid = int(coverage.valid_distributions.sum())
intro = f'''## Reviewed plateaus and background check — 13 October 2026

This section supersedes earlier **baseline pending** notes for this explicitly reviewed calculation only. Selected plateaus are saved in `calibration_review/plateau_review_20261013.json`. The user approved a **shared straight-line B(t)** between the base's reviewed endpoints, then requested **review of problematic backgrounds before clipping or finalizing volume metrics**.

All **2,172 unique base–child comparisons across 15 networks** have been evaluated. Peak timing, total height, scalar-reference height, linear-background peak excess and connected crest widths are exported. Network 54 retains separate ordered lobe timings and widths. Volume-distribution metrics pass nonnegativity for **{valid} pairs**; other pairs have explicit missing values and deficit diagnostics. These are provisional calculations, not an accepted final response dataset.

`Q − B(t)` is not silently clipped. Only negative roundoff within 1e-8 m³/s is zeroed. The volume, centroid, quantiles, temporal SD, asymmetry and standardized shape distance use exact integrals of the **linear reconstruction of saved one-minute interval means** on the common reviewed window. Interval-weighted signed volume is retained as a reconstruction sensitivity diagnostic. Signed volume remains a separate quantity even when an event distribution is invalid.

The fixed reviewed start to each state's downward end crossing is reported separately. **It is not the historical 1% departure-to-permanent-settlement duration** and must not replace that column. Crossing failure is recorded rather than repaired by extending into a later event. A scalar `max(Q0,Qf)` peak reference is also distinct from the time-varying B(t) peak-excess definition.

The background-review HTML includes full/event/peak views. Review the larger mismatches in **05, 13, 37 and 54**, then smaller recession deficits such as **46**, before choosing another background or explicitly defining positive-only excess. The approved plateau levels have not changed.
'''
code = '''from pathlib import Path
import pandas as pd
from IPython.display import display, Image, HTML
RESPONSE_ROOT=next(p for p in [Path.cwd(),*Path.cwd().parents] if (p/'RAPID').is_dir())
RESPONSE_OUT=RESPONSE_ROOT/'RAPID/outputs/reviewed_responses_20261013'
reviewed_responses=pd.read_parquet(RESPONSE_OUT/'responses.parquet')
display(pd.read_csv(RESPONSE_OUT/'full_coverage.csv', dtype={'example_id':str}))
display(Image(filename=str(RESPONSE_OUT/'linear_background_review.png')))
example=reviewed_responses[(reviewed_responses.example_id=='46') & reviewed_responses.state_aliases.str.contains('CB006_bubble_101',regex=False)]
assert len(example)==1
display(example[['state_id','state_aliases','sampled_peak_shift_minutes','peak_total_change_pct','peak_excess_cms_change_pct','crest50_width_seconds_change','crest90_width_seconds_change','event_distribution_status','parent_minimum_excess_cms','child_minimum_excess_cms']])
'''
notebook.cells[1:1] = [nbformat.v4.new_markdown_cell(intro, metadata={'audit_section':tag}),
    nbformat.v4.new_code_cell(code, metadata={'audit_section':tag})]
nbformat.validate(notebook)
nbformat.write(notebook, book)
print('Background review HTML, PNG and notebook section written.')
