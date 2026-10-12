"""Canonical offline 15-network overview, physical and shared-scale discharge."""
from pathlib import Path
import sys,json,gzip,base64,hashlib
import numpy as np,pandas as pd
from plotly.offline import get_plotlyjs
from plotly.utils import PlotlyJSONEncoder
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'RAPID/src'),str(Path(__file__).parent)]
import export_common_grid_plotly as base
from rapid_tools.common_grid_production import model_feature
BUNDLE=ROOT/'RAPID/outputs/bounded_beta_calibration_20261009'
OUT=ROOT/'RAPID/outputs/calibration_review'

def main():
 source=Path(json.loads((BUNDLE/'protocol.json').read_text())['source_directory'])
 panels=base.build(source)
 table=pd.read_csv(source/'baseline/RAPID/outputs/cunge_calibration_reviewed_20261006/comparison.csv',dtype={'example':str})
 for p in panels:
  ex=p['example'];f=pd.read_parquet(BUNDLE/ex/'wave.parquet');r=json.loads((BUNDLE/ex/'result.json').read_text());valid=f.downstream_cms.notna().to_numpy()
  q0=p['normalization']['forcing_initial_cms'];offset=p['normalization']['downstream_offset_cms'];origin=pd.Timestamp(p['origin'])
  t=(pd.to_datetime(f.time_utc,utc=True)-origin).dt.total_seconds().to_numpy()/86400
  utc=pd.to_datetime(f.time_utc,utc=True).dt.strftime('%Y-%m-%d %H:%M:%S UTC').to_numpy()
  series=[]
  fixed=table[table.example.eq(ex)&table.method.eq('fixed_x')].iloc[0]
  p['title']=f"{ex} · β={r['beta']:.3g} · n={r['n_manning']:.4f}"+(' · provisional clock' if ex in ['21','22'] else '')
  p['target']=dict(p['target']);p['target']['rationale']+=f" | Cunge n={r['n_manning']:.7f}, beta={r['beta']:.6g}. Single-X benchmark: X=0.1, calibrated n={fixed.n_manning:.7f}."
  definitions=[('forcing_cms','Input hydrograph','#2580a5','dash',0.,'upstream_gap'),
   ('downstream_cms','Downstream · baseline aligned','#18212b','solid',offset,'downstream_gap'),
   ('downstream_cms','Downstream · unshifted observations','#8b8f98','dash',0.,'downstream_gap'),
   ('calibrated_cunge_cms','Calibrated Cunge','#ce3d45','solid',0.,None),
   ('fixed_x_cms','Calibrated single X (X=0.1)','#dc8a16','solid',0.,None)]
  for i,(column,name,color,dash,shift,flagcol) in enumerate(definitions):
   q=f[column].to_numpy();flags=f[flagcol].to_numpy() if flagcol else np.zeros(len(f),bool);plotq=q+shift
   custom=[[str(utc[j]),float(q[j]),float(plotq[j]),'Diagnostic interpolation / long sampling interval' if flags[j] else 'One-minute interval mean'] for j in np.flatnonzero(valid)]
   hover=f'<b>{ex} · {name}</b><br>%{{customdata[0]}}<br>Original Q: %{{customdata[1]:.3f}} m³/s<br>Displayed Q: %{{customdata[2]:.3f}} m³/s<br>%{{customdata[3]}}<extra></extra>'
   group=f'curve_{i}';y=plotq.copy();y[flags]=np.nan
   series.append(dict(type='scatter',mode='lines',name=name,legendgroup=group,x=t[valid].tolist(),y=base.clean(y[valid]),customdata=custom,hovertemplate=hover,line=dict(color=color,dash=dash,width=1.8),connectgaps=False))
   if flags.any():
    near=flags.copy();near[1:]|=flags[:-1];near[:-1]|=flags[1:]
    series.append(dict(type='scatter',mode='lines',name=name+' · sampling flags',legendgroup=group,showlegend=False,x=t[valid].tolist(),y=base.clean(np.where(near,plotq,np.nan)[valid]),customdata=custom,hovertemplate=hover,line=dict(color=color,dash='dot',width=1.4),opacity=.55,connectgaps=False))
  for column,name,color,symbol,group in [('forcing_cms','Input peak','#2580a5','circle','input_peak'),('calibrated_cunge_cms','Base Cunge peak','#ce3d45','diamond','model_peak')]:
   tm,h,kind=model_feature(f[column].to_numpy()[valid],t[valid]*86400,'peak',{})
   stamp=origin+pd.Timedelta(seconds=tm)
   series.append(dict(type='scatter',mode='markers',name=name,legendgroup=group,x=[tm/86400],y=[h],marker=dict(color=color,symbol=symbol,size=9,line=dict(color='white',width=1)),hovertemplate=f'{name}<br>{stamp.isoformat()}<br>{h:.3f} m³/s<br>{kind}<extra></extra>'))
  if ex=='54':
   tm=(pd.Timestamp(r['feature_utc'])-origin).total_seconds()/86400
   series.append(dict(type='scatter',mode='markers',name='54 model two-crest midpoint',legendgroup='model_feature54',x=[tm],y=[r['peak_cms']],marker=dict(color='#ce3d45',symbol='diamond-open',size=12),hovertemplate=f"Reviewed model feature: two-crest midpoint<br>{r['feature_utc']}<extra></extra>"))
  p['series']=series;p['calibration']=r
  for mode in ['full','crest']:
   vals=[]
   for tr in series:
    for x,y in zip(tr['x'],tr['y']):
     if y is not None and p[mode+'_x'][0]<=x<=p[mode+'_x'][1]:vals.append(y)
   low,high=min(vals),max(vals);pad=max((high-low)*.08,q0*.005)
   p[mode+'_y']=[low-pad,high+pad]
 layouts={mode:base.overview(panels,mode) for mode in ['full','crest']}
 for layout in layouts.values():
  for key,value in layout.items():
   if key.startswith('yaxis') and value.get('title'):value['title']['text']='Discharge (m³/s)'
  layout['margin']['t']=145;layout['legend']['y']=1.055
 payload=dict(panels=panels,layouts=layouts)
 packed=base64.b64encode(gzip.compress(json.dumps(payload,cls=PlotlyJSONEncoder,separators=(',',':'),allow_nan=False).encode(),mtime=0)).decode()
 script=base.SCRIPT.replace("let payload, current='all', mode=DEFAULT_MODE;", "let payload, current='all', mode=DEFAULT_MODE;\nlet units='physical';")
 script=script.replace("const all=current==='all';const panels=all?payload.panels:payload.panels.filter(p=>p.example===current);", """const all=current==='all';const original=all?payload.panels:payload.panels.filter(p=>p.example===current);
  const panels=original.map(p=>{const scale=units==='physical'?1:p.normalization.forcing_initial_cms;
    return {...p,full_y:p.full_y.map(y=>y/scale),crest_y:p.crest_y.map(y=>y/scale),
      series:p.series.map(t=>({...t,y:t.y.map(y=>y===null?null:y/scale)}))};});""")
 script=script.replace("if(all){layout=JSON.parse(JSON.stringify(payload.layouts[mode]));}", """if(all){layout=JSON.parse(JSON.stringify(payload.layouts[mode]));
    panels.forEach((p,i)=>{const key='yaxis'+(i?i+1:'');layout[key].range=p[mode+'_y'];if(layout[key].title)layout[key].title.text=units==='physical'?'Discharge (m³/s)':'Q / initial forcing';});}""")
 script=script.replace("title:{text:'Q / initial forcing (downstream shifted)'}", "title:{text:units==='physical'?'Discharge (m³/s)':'Q / initial forcing'}")
 script=script.replace("reset.addEventListener('click',render);", "document.getElementById('units').addEventListener('change',e=>{units=e.target.value;render();});\n  reset.addEventListener('click',render);")
 html='''<!doctype html><html lang="en"><head><meta charset="utf-8"><title>15-network calibrated hydrographs</title><style>body{font:14px Arial;margin:0;color:#202a34}header,.note{padding:18px 28px}p{line-height:1.5;max-width:1400px}.controls{position:sticky;top:0;background:#f4f6f9;padding:12px 28px;z-index:20;display:flex;gap:14px;flex-wrap:wrap;align-items:center}select,button{font:inherit;padding:7px}#context{padding:8px 28px}#plot{width:100%}</style><script>LIBRARY</script></head><body><header><h1>15-network calibrated hydrographs</h1><p>Blue: input. Black: downstream observations aligned to the input’s initial plateau. Grey: unshifted downstream observations. Red: calibrated Cunge (β 0.5–2). Orange: the existing Manning-calibrated single-X benchmark (X=0.1).</p><p>Blue circles mark input peaks; red diamonds mark base Cunge peaks. Purple lines mark the reviewed calibration target; purple shading marks its crest interval. For 54, the open red diamond separately marks the model’s two-crest midpoint used for calibration.</p><p>The downstream alignment is an additive display shift only; measurements and times are unchanged. Discharge units show the unshifted observations in their original m³/s. The optional shared-scale view divides every curve by the same initial input discharge. Hover retains original and displayed discharge.</p></header><div class="controls"><label>Show <select id="panel"><option value="all">All 15 networks</option></select></label><label>View <select id="view"><option value="full">Complete event</option><option value="crest">Crest detail</option></select></label><label>Units <select id="units"><option value="physical">Discharge (m³/s)</option><option value="shared">Shared initial-forcing scale</option></select></label><button id="reset">Reset view</button><span id="status">Loading…</span></div><p id="context"></p><div id="plot"></div><div class="note"><p>One-minute interval means; full data retained. Dotted gauge segments flag long sampling intervals. 21/22 source clocks remain provisional; 55 has hourly crest sampling. These curves retain the reviewed calibration geometry and equilibrium start. In 46, actual 7/30-day antecedent forcing shifts the new crest about 14–15 minutes later with less than 0.5% peak-height change. Example 11 remains a separate unresolved diagnostic.</p><p>Single offline file with both views. Network selection opens a full-width panel; legend entries toggle curves. No soft height bands are displayed.</p></div><script id="payload" type="application/octet-stream">PAYLOAD</script><script>const DEFAULT_MODE="full";SCRIPT</script></body></html>'''
 OUT.mkdir(parents=True,exist_ok=True);path=OUT/'15_network_overview.html';path.write_text(html.replace('SCRIPT',script).replace('PAYLOAD',packed).replace('LIBRARY',get_plotlyjs()))
 # Verify serialized scientific values, required series, and marker positions.
 decoded=json.loads(gzip.decompress(base64.b64decode(packed)))
 for p in decoded['panels']:
  assert {s['legendgroup'] for s in p['series']} >= {'curve_0','curve_1','curve_2','curve_3','curve_4','input_peak','model_peak'}
  assert not any('height' in s['legendgroup'] for s in p['series'])
  f=pd.read_parquet(BUNDLE/p['example']/'wave.parquet');valid=f.downstream_cms.notna().to_numpy()
  for group,column,offset in [('curve_0','forcing_cms',0),('curve_1','downstream_cms',p['normalization']['downstream_offset_cms']),('curve_2','downstream_cms',0),('curve_3','calibrated_cunge_cms',0),('curve_4','fixed_x_cms',0)]:
   tr=next(s for s in p['series'] if s['legendgroup']==group);np.testing.assert_allclose([c[2] for c in tr['customdata']],f[column].to_numpy()[valid]+offset)
  for tr in p['series']:assert len(tr['x'])==len(tr['y'])
 manifest=dict(bundle=str(BUNDLE),panels=15,offline=True,full_resolution=True,soft_height_bands=False,physical_units_default=True,
  input_peak='Midpoint of first contiguous exact global-maximum block within displayed observation extent',
  model_peak='Same global-maximum rule; 54 calibration midpoint displayed separately',
  fixed_x='Existing calibrated Manning n, constant X=0.1; replayed on common reporting support, not newly fitted',
  source_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),html_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
  wave_sha256={p['example']:hashlib.sha256((BUNDLE/p['example']/'wave.parquet').read_bytes()).hexdigest() for p in panels},serialized_values_verified=True)
 (OUT/'overview_manifest.json').write_text(json.dumps(manifest,indent=2));print(path,flush=True)
if __name__=='__main__':main()
