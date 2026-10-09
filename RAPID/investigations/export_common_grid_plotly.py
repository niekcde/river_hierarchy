"""Offline Plotly versions of the full-event and crest-detail target figures.

Keep every one-minute point. Embed one compressed dataset plus Plotly itself in
each HTML; nothing is fetched from a CDN and no routing is performed.
"""
from pathlib import Path
import base64
import gzip
import hashlib
import html
import json
import shutil

import numpy as np
import pandas as pd
from plotly.offline import get_plotlyjs
import plotly.graph_objects as go
from plotly.subplots import make_subplots

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/'RAPID/outputs/common_grid_validation_20261008'
STYLES = {
    'upstream_cms': ('Upstream gauge', '#2580a5', 'dash'),
    'downstream_cms': ('Downstream gauge', '#18212b', 'solid'),
    'fixed_x_cms': ('Fixed-X model', '#dc8a16', 'solid'),
    'cunge_cms': ('Cunge model', '#ce3d45', 'solid'),
}


def clean(values):
    return [float(v) if np.isfinite(v) else None for v in values]


def build(dest):
    freeze = dest/'baseline'
    target_path = freeze/'RAPID/reviews/calibration_targets_20261006/targets.json'
    targets = {r['example']: r for r in json.loads(target_path.read_text())['records']}
    table = pd.read_csv(freeze/'RAPID/outputs/cunge_calibration_reviewed_20261006/comparison.csv', dtype={'example': str})
    examples = sorted(set(table.example)-{'11'})
    panels = []
    for ex in examples:
        f = pd.read_parquet(dest/ex/'60s.parquet')
        target = targets[ex]
        origin = pd.Timestamp(target['plot_origin_utc'])
        nominal = pd.Timestamp(target['target_utc'])
        lo, hi = pd.Timestamp(target['crest_start_utc']), pd.Timestamp(target['crest_end_utc'])
        folder = freeze/Path(table[table.example == ex].iloc[0].output_dir).relative_to(ROOT)
        raw = pd.read_parquet(folder/'observed_downstream.parquet')
        raw.time = pd.to_datetime(raw.time, utc=True)
        valid = f.downstream_cms.notna().to_numpy()
        t = (pd.to_datetime(f.time_utc, utc=True)-origin).dt.total_seconds().to_numpy()/86400
        xr = [(raw.time.min()-origin).total_seconds()/86400, (raw.time.max()-origin).total_seconds()/86400]
        lx, hx, tx = [(stamp-origin).total_seconds()/86400 for stamp in (lo,hi,nominal)]
        pad = max(.15, (hx-lx)*.4)
        zx = [max(xr[0],lx-pad), min(xr[1],hx+pad)]
        series, zoom_values, full_values = [], [], []
        reference = float(json.loads((dest/ex/'60s.json').read_text())[0]['initial_cms'])
        downstream_initial = float(f.loc[valid, 'downstream_cms'].iloc[0])
        downstream_offset = reference-downstream_initial
        assert reference > 0
        utc = pd.to_datetime(f.time_utc,utc=True).dt.strftime('%Y-%m-%d %H:%M:%S UTC').to_numpy()
        for key,(name,color,dash) in STYLES.items():
            q = f[key].to_numpy(float)
            usable = valid & np.isfinite(q)
            offset = downstream_offset if key == 'downstream_cms' else 0.
            y = (q+offset)/reference
            full_values.extend(y[usable].tolist())
            if key != 'upstream_cms':
                zoom_values.extend(y[usable & (t>=zx[0]) & (t<=zx[1])].tolist())
            flags = f[key.replace('_cms','_gap')].to_numpy() if key in ['upstream_cms','downstream_cms'] else np.zeros(len(f),bool)
            custom = [[str(utc[i]), float(q[i]) if np.isfinite(q[i]) else None,
                       'Long sampling interval: diagnostic interpolation' if flags[i] else 'One-minute interval mean']
                      for i in np.flatnonzero(valid)]
            hover = f'<b>{ex} · {name}</b><br>%{{customdata[0]}}<br>Day %{{x:.6f}}<br>Q: %{{customdata[1]:.6f}} m³/s<br>Normalized Q: %{{y:.6f}}<br>%{{customdata[2]}}<extra></extra>'
            main = y.copy(); main[flags] = np.nan
            series.append(dict(type='scatter', mode='lines', name=name, legendgroup=key,
                x=t[valid].tolist(), y=clean(main[valid]), customdata=custom, hovertemplate=hover,
                line=dict(color=color,dash=dash,width=1.6), connectgaps=False))
            if flags.any():
                diagnostic = np.where(flags,y,np.nan)
                # Adjacent original points make the flagged segments connect exactly
                # to valid intervals; their hover retains the true per-bin flag.
                near = flags.copy()
                near[1:] |= flags[:-1]; near[:-1] |= flags[1:]
                diagnostic[near] = y[near]
                series.append(dict(type='scatter',mode='lines',name=name+' · long sampling intervals',
                    legendgroup=key,showlegend=False,x=t[valid].tolist(),y=clean(diagnostic[valid]),
                    customdata=custom,hovertemplate=hover,line=dict(color=color,dash='dot',width=1.3),opacity=.5,
                    connectgaps=False))
            if key == 'downstream_cms':
                rawx = (raw.time-origin).dt.total_seconds().to_numpy()/86400
                rawq = raw.discharge_cms.to_numpy(float)
                series.append(dict(type='scatter',mode='markers',name='Original downstream samples',legendgroup='raw',
                    x=rawx.tolist(),y=((rawq+downstream_offset)/reference).tolist(),
                    customdata=list(map(list,zip(raw.time.dt.strftime('%Y-%m-%d %H:%M:%S UTC'),rawq.tolist()))),
                    marker=dict(color='#637080',size=4,opacity=.55),
                    hovertemplate=f'<b>{ex} · original downstream observation</b><br>%{{customdata[0]}}<br>Q: %{{customdata[1]:.6f}} m³/s<extra></extra>'))
        full_lo, full_hi = min(full_values), max(full_values)
        full_pad = max((full_hi-full_lo)*.08, .02)
        low, high = min(zoom_values),max(zoom_values)
        yp = max((high-low)*.15,.0001)
        title = f'{ex} · target {nominal:%d %b %H:%M:%S}'
        if ex in ['21','22']:
            title += ' · clock provisional'
        panels.append(dict(example=ex,title=title,origin=origin.isoformat(),full_x=xr,crest_x=zx,
            normalization=dict(forcing_initial_cms=reference,downstream_initial_mean_cms=downstream_initial,
                downstream_offset_cms=downstream_offset,rule='(Q + downstream-only offset) / initial forcing'),
            full_y=[full_lo-full_pad,full_hi+full_pad],crest_y=[low-yp,high+yp],target_x=tx,span=[lx,hx],target=target,series=series))
    assert len(panels)==15
    return panels


def overview(panels, mode):
    fig = make_subplots(rows=5,cols=3,subplot_titles=[p['title'] for p in panels],
                        horizontal_spacing=.06,vertical_spacing=.055)
    seen = set()
    for i,p in enumerate(panels):
        row,col=divmod(i,3);row+=1;col+=1
        for trace in p['series']:
            item=dict(trace)
            # Validate style and build subplot layout without repeatedly copying
            # millions of points; the HTML uses the complete panel data directly.
            item['x'] = item['x'][:1]
            item['y'] = item['y'][:1]
            if 'customdata' in item:
                item['customdata'] = item['customdata'][:1]
            group=item['legendgroup']
            show=item.get('showlegend',True) and group not in seen
            item['showlegend']=show
            if show:seen.add(group)
            fig.add_trace(go.Scatter(**{k:v for k,v in item.items() if k!='type'}),row=row,col=col)
        yr=p[mode+'_y']
        target=p['target']
        fig.add_trace(go.Scatter(x=[p['target_x']]*2,y=yr,mode='lines',name='Excel calibration target',
            legendgroup='target',showlegend=i==0,line=dict(color='#8353ad',dash='dot',width=2),
            hovertemplate=f"<b>{p['example']} · selected target</b><br>{html.escape(target['target_utc'])}<br>{html.escape(target['selection_rule'])}<br>{html.escape(target['rationale'])}<extra></extra>"),row=row,col=col)
        if p['span'][1]>p['span'][0]:
            fig.add_vrect(x0=p['span'][0],x1=p['span'][1],fillcolor='#8353ad',opacity=.12,
                          line_width=0,layer='below',row=row,col=col)
        fig.update_xaxes(range=p[mode+'_x'],title_text='Days from panel origin (UTC)',nticks=6,row=row,col=col)
        fig.update_yaxes(range=yr,title_text='Q / initial forcing' if col==1 else None,nticks=6,row=row,col=col)
    fig.update_layout(template='plotly_white',height=2050,margin=dict(l=70,r=25,t=100,b=60),
        legend=dict(orientation='h',x=.5,xanchor='center',y=1.045,yanchor='bottom',groupclick='togglegroup'),
        font=dict(family='Arial, sans-serif',size=12),dragmode='zoom',hovermode='closest')
    for a in fig.layout.annotations:a.font=dict(size=12)
    # Python construction validates the schema; both overview layouts are saved
    # compactly. Series data are embedded only once per document.
    return fig.to_plotly_json()['layout']


SCRIPT = r'''
const panelSelect=document.getElementById('panel'), reset=document.getElementById('reset');
let payload, current='all', mode=DEFAULT_MODE;
const gd=document.getElementById('plot');
const config={responsive:true,scrollZoom:true,displaylogo:false,modeBarButtonsToRemove:['lasso2d','select2d'],
  toImageButtonOptions:{format:'svg',filename:'common_grid_target_review'}};
function targetTrace(p,yr,show){return {type:'scatter',mode:'lines',x:[p.target_x,p.target_x],y:yr,
 name:'Excel calibration target',legendgroup:'target',showlegend:show,line:{color:'#8353ad',dash:'dot',width:2},
 hovertemplate:'<b>'+p.example+' · selected target</b><br>'+p.target.target_utc+'<br>'+p.target.selection_rule+'<br>'+p.target.rationale+'<extra></extra>'};}
async function render(){
  const all=current==='all';const panels=all?payload.panels:payload.panels.filter(p=>p.example===current);
  let data=[],seen=new Set();
  panels.forEach((p,i)=>{const suffix=all&&i>0?String(i+1):'';
    p.series.forEach(t=>{let show=t.showlegend!==false&&!seen.has(t.legendgroup);
      if(show)seen.add(t.legendgroup);
      data.push({...t,showlegend:show,xaxis:'x'+suffix,yaxis:'y'+suffix});});
    data.push({...targetTrace(p,p[mode+'_y'],i===0),xaxis:'x'+suffix,yaxis:'y'+suffix});
  });
  let layout;
  if(all){layout=JSON.parse(JSON.stringify(payload.layouts[mode]));}
  else {const p=panels[0];layout={template:'plotly_white',height:Math.max(650,window.innerHeight-210),
    margin:{l:80,r:35,t:110,b:75},title:{text:p.title+'<br><sup>Day zero: '+p.origin+'</sup>',x:.06},
    xaxis:{range:p[mode+'_x'],title:{text:'Days from panel origin (UTC)'},showgrid:true},
    yaxis:{range:p[mode+'_y'],title:{text:'Q / initial forcing (downstream shifted)'},showgrid:true},
    legend:{orientation:'h',y:1.08,groupclick:'togglegroup'},dragmode:'zoom',hovermode:'closest',
    shapes:p.span[1]>p.span[0]?[{type:'rect',xref:'x',yref:'paper',x0:p.span[0],x1:p.span[1],y0:0,y1:1,
      fillcolor:'#8353ad',opacity:.12,line:{width:0},layer:'below'}]:[]};}
  layout.paper_bgcolor='white';layout.plot_bgcolor='white';
  gd.style.height=layout.height+'px';
  await Plotly.react(gd,data,layout,config);
  document.getElementById('status').textContent=all?'15 panels · full one-minute data':'Network '+current+' · full one-minute data';
  document.getElementById('context').textContent=all?'Select a network to inspect it at full width.':
    'Selected target: '+panels[0].target.target_utc+' · '+panels[0].target.rationale+' · Initial forcing: '+panels[0].normalization.forcing_initial_cms.toFixed(3)+' m³/s; downstream shift: '+panels[0].normalization.downstream_offset_cms.toFixed(3)+' m³/s';
}
(async()=>{try{
  const bytes=Uint8Array.from(atob(document.getElementById('payload').textContent.trim()),c=>c.charCodeAt(0));
  const stream=new Blob([bytes]).stream().pipeThrough(new DecompressionStream('gzip'));
  payload=JSON.parse(await new Response(stream).text());
  for(const p of payload.panels){const o=document.createElement('option');o.value=p.example;o.textContent='Network '+p.example;panelSelect.appendChild(o);}
  panelSelect.addEventListener('change',()=>{current=panelSelect.value;render();});
  document.getElementById('view').value=mode;
  document.getElementById('view').addEventListener('change',e=>{mode=e.target.value;render();});
  reset.addEventListener('click',render);
  await render();
}catch(e){document.getElementById('status').textContent='Could not load plot: '+e.message;}})();
'''


def main():
    dest=Path(json.loads((OUT/'latest.json').read_text())['directory'])
    panels=build(dest)
    layouts={mode:overview(panels,mode) for mode in ['full','crest']}
    from plotly.utils import PlotlyJSONEncoder
    serialized=json.dumps(dict(panels=panels,layouts=layouts),cls=PlotlyJSONEncoder,separators=(',',':'),allow_nan=False)
    packed=base64.b64encode(gzip.compress(serialized.encode(),compresslevel=6,mtime=0)).decode()
    assert json.loads(gzip.decompress(base64.b64decode(packed)))['panels'][0]['example']=='02'
    library=get_plotlyjs()
    files=[]
    for mode,stem,title in [('full','15_panel_common_grid_targets','Complete events'),
                             ('crest','15_panel_common_grid_target_crests','Crest detail')]:
        document='''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Common-grid target review — TITLE</title><style>
body{margin:0;background:#fff;color:#202a34;font-family:Arial,sans-serif}header{padding:20px 28px 12px}
h1{font-size:23px;margin:0 0 8px}p{line-height:1.45;margin:8px 0;font-size:14px;max-width:1300px}
.controls{position:sticky;top:0;background:#f4f6f9;padding:12px 28px;z-index:20;border-bottom:1px solid #d7dde5;display:flex;align-items:center;gap:12px;flex-wrap:wrap}
select,button{font:inherit;padding:7px 10px;border:1px solid #b7c2cd;border-radius:5px;background:white}
#status{font-size:13px;color:#4a5867}#context{padding:5px 28px;color:#594579;min-height:20px}
#plot{width:100%;min-height:650px}.note{padding:5px 28px 22px;color:#4a5867}
</style><script>LIBRARY</script></head><body><header>
<h1>Common-grid calibration review · TITLE</h1>
<p>15 retained examples · one-minute interval means · existing parameters, <strong>not recalibrated</strong>.
Purple lines mark unchanged Excel targets; shading marks selected crest spans.</p>
<p>Drag to zoom; use the toolbar to pan; click legend entries to toggle series. Hover for UTC, discharge and sampling flags.
Both files contain the full one-minute series; neither is a static image.</p></header>
<div class="controls"><label>Show <select id="panel"><option value="all">All 15 networks</option></select></label>
<label>View <select id="view"><option value="full">Complete event</option><option value="crest">Crest detail</option></select></label>
<button id="reset">Reset view</button><span id="status">Loading full-resolution data…</span></div>
<p id="context"></p><div id="plot"></div><div class="note">
<p>All curves use the initial forcing discharge Q₀ as their common scale: upstream and models show Q/Q₀.
Only downstream observations are shifted: (Q + Q₀ − initial downstream mean)/Q₀, so their first supported one-minute mean equals 1.
This is a display adjustment, not a correction to the measurements or calibration. No time alignment is applied.
Original downstream point observations are shown as grey markers; hover reports discharge in m³/s.</p>
<p>Faint dotted gauge segments span more than twice the whole-event nominal sampling interval. These flags can indicate
a sampling-regime change rather than missing data: example 55 is hourly around its crests. Examples 21/22 retain provisional source clocks.
Example 46 is the equilibrium-start pilot shown in the PNG figures; the separate antecedent-forcing checks are not substituted here.
Example 11 is excluded. The original target choices and simulation data are unchanged.</p>
<p>Self-contained offline file. Full data retained for zooming. Requires a modern browser with gzip decompression support.</p></div>
<script id="payload" type="application/octet-stream">PAYLOAD</script><script>const DEFAULT_MODE="MODE";SCRIPT</script></body></html>'''
        document=document.replace('TITLE',title).replace('"MODE"',json.dumps(mode)).replace('SCRIPT',SCRIPT).replace('PAYLOAD',packed).replace('LIBRARY',library)
        path=dest/(stem+'.html')
        backup=dest/(stem+'_independent_minmax.html')
        if path.exists() and not backup.exists():
            shutil.copy2(path,backup)
        path.write_text(document)
        files.append(dict(path=str(path),bytes=path.stat().st_size,sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
        print(path,round(path.stat().st_size/1024**2,2),'MiB')
    (dest/'plotly_export_manifest.json').write_text(json.dumps(dict(files=files,panels=15,
        normalization='shared initial forcing, additive downstream shift',resolution_seconds=60,decimation=False,offline=True,source_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        source_data_hashes={str(dest/p['example']/'60s.parquet'):hashlib.sha256((dest/p['example']/'60s.parquet').read_bytes()).hexdigest() for p in panels}),indent=2))


if __name__=='__main__':main()
