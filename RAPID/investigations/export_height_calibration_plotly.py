"""Offline full-resolution comparison of calibrated waves and fixed height targets."""
from pathlib import Path
import sys,json,gzip,base64,hashlib
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(Path(__file__).parent))
import export_common_grid_plotly as base
from plotly.offline import get_plotlyjs
from plotly.utils import PlotlyJSONEncoder

def main(dest=None, previous=None, bounded=False):
 dest=Path(dest) if dest is not None else Path(json.loads((ROOT/'RAPID/outputs/common_grid_height_calibration_20261008/latest.json').read_text())['directory'])
 protocol=json.loads((dest/'protocol.json').read_text());source=Path(protocol['source_directory'])
 panels=base.build(source)
 for p in panels:
  ex=p['example'];r=json.loads((dest/ex/'result.json').read_text());f=pd.read_parquet(dest/ex/'wave.parquet');valid=f.downstream_cms.notna().to_numpy()
  q=f.calibrated_cunge_cms.to_numpy()[valid];q0=p['normalization']['forcing_initial_cms']
  old=next(t for t in p['series'] if t['legendgroup']=='cunge_cms')
  original=dict(old);original.update(name='Original Cunge',legendgroup='original_cunge',line=dict(color='#9b7477',dash='dash',width=1.2),opacity=.7)
  original['hovertemplate']=old['hovertemplate'].replace('Cunge model','Original Cunge')
  if previous is not None:
   prior=pd.read_parquet(Path(previous)/ex/'wave.parquet').calibrated_cunge_cms.to_numpy()[valid]
   original['name']='Previous unrestricted fit';original['y']=(prior/q0).tolist()
   original['customdata']=[[c[0],float(prior[i]),'Previous unrestricted fit'] for i,c in enumerate(old['customdata'])]
   original['hovertemplate']=old['hovertemplate'].replace('Cunge model','Previous unrestricted fit')
  p['series'].append(original)
  old['name']='Calibrated Cunge';old['y']=(q/q0).tolist()
  old['customdata']=[[c[0],float(q[i]),'Calibrated one-minute mean'] for i,c in enumerate(old['customdata'])]
  old['hovertemplate']=old['hovertemplate'].replace('Cunge model','Calibrated Cunge')
  p['calibration']=r
  p['target']=dict(p['target']);p['target']['rationale']+=f" | Fit beta={r['beta']:.5g}, n={r['n_manning']:.7f}; time error {r['timing_error_minutes']:.2f} min; height shortfall {r['height_distance_cms']:.2f} m3/s."
  p['title']+=f" · β={r['beta']:.3g}"
  low=r['height_low_cms']/q0;high=r['height_high_cms']/q0
  if high>low:
   p['series'].insert(0,dict(type='scatter',mode='lines',name='Soft height range',legendgroup='height_band',x=p['full_x'],y=[low,low],line=dict(color='rgba(37,150,90,0.25)',width=.5),showlegend=False,hoverinfo='skip'))
   p['series'].insert(1,dict(type='scatter',mode='lines',name='Soft height range',legendgroup='height_band',x=p['full_x'],y=[high,high],line=dict(color='rgba(37,150,90,0.25)',width=.5),fill='tonexty',fillcolor='rgba(37,150,90,0.09)',hovertemplate=f"Soft height range: {r['height_low_cms']:.2f}–{r['height_high_cms']:.2f} m³/s<extra></extra>"))
  else:
   p['series'].append(dict(type='scatter',mode='lines',name='Soft height point',legendgroup='height_point',x=p['full_x'],y=[low,low],line=dict(color='#25965a',dash='dot',width=1.2),hovertemplate=f"Soft height target: {r['height_low_cms']:.2f} m³/s<extra></extra>"))
  p['full_y']=[min(p['full_y'][0],float(min(q/q0))*.95,low*.98),max(p['full_y'][1],float(max(q/q0))*1.03,high*1.03)]
  near=(np.array(old['x'])>=p['crest_x'][0])&(np.array(old['x'])<=p['crest_x'][1])
  vals=np.r_[q[near]/q0,low,high,p['crest_y']]
  p['crest_y']=[float(vals.min()-(vals.max()-vals.min())*.08),float(vals.max()+(vals.max()-vals.min())*.08)]
  # Mark the fitted feature separately from the unchanged observed timing target.
  model_x=(pd.Timestamp(r['feature_utc'])-pd.Timestamp(p['origin'])).total_seconds()/86400
  p['series'].append(dict(type='scatter',mode='markers',name='Fitted timing feature',legendgroup='fit_feature',x=[model_x],y=[r['peak_cms']/q0],marker=dict(color='#ce3d45',symbol='diamond',size=7),hovertemplate=f"Fitted {r['feature_kind']}<br>{r['feature_utc']}<br>Error {r['timing_error_minutes']:.2f} min<extra></extra>"))
 layouts={mode:base.overview(panels,mode) for mode in ['full','crest']}
 payload=json.dumps(dict(panels=panels,layouts=layouts),cls=PlotlyJSONEncoder,separators=(',',':'),allow_nan=False)
 packed=base64.b64encode(gzip.compress(payload.encode(),compresslevel=6,mtime=0)).decode()
 # Validate actual embedded numeric data before writing HTML.
 decoded=json.loads(gzip.decompress(base64.b64decode(packed)));assert len(decoded['panels'])==15
 for p in decoded['panels']:
  for tr in p['series']:assert len(tr['x'])==len(tr['y'])
  tr=next(t for t in p['series'] if t['legendgroup']=='cunge_cms')
  np.testing.assert_allclose(tr['y'],np.array([c[1] for c in tr['customdata']])/p['normalization']['forcing_initial_cms'])
 library=get_plotlyjs()
 for mode in ['full','crest']:
  template='''<!doctype html><html><head><meta charset="utf-8"><title>Calibrated timing and soft heights</title><style>body{font:14px Arial;margin:0;color:#202a34}header,.note{padding:18px 28px}p{line-height:1.5;max-width:1400px}.controls{position:sticky;top:0;background:#f4f6f9;padding:12px 28px;z-index:20;display:flex;gap:14px;align-items:center;flex-wrap:wrap}select,button{font:inherit;padding:7px}#context{padding:8px 28px}#plot{width:100%}</style><script>LIBRARY</script></head><body><header><h1>15-example calibration: timing primary, height soft</h1><p>Red: calibrated Cunge. Dashed muted red: original Cunge. Orange: unchanged fixed-X comparator. Green bands/lines: your height targets. Purple vertical lines: unchanged timing targets. Red diamonds: fitted timing feature (54 uses the two-lobe midpoint).</p><p>Shared initial-forcing scale; downstream alone receives its fixed initial-plateau shift. Hover retains original discharge. Full one-minute data; no decimation. Equilibrium-start base fits; provisional clocks and sampling flags retained. These are review candidates, not hierarchy or final production validation.</p></header><div class="controls"><label>Show <select id="panel"><option value="all">All 15 networks</option></select></label><label>View <select id="view"><option value="full">Complete event</option><option value="crest">Crest detail</option></select></label><button id="reset">Reset view</button><span id="status">Loading…</span></div><p id="context"></p><div id="plot"></div><div class="note"><p>Timing numerical tolerance: 1 minute, not observational uncertainty. Height intervals are modelling acceptance ranges; point targets use a 0.1% numerical search tolerance. 13/37/48 use the proposed ±1% current-height bands. 21/22 source clocks remain provisional; 55 is hourly at its crest. Diagnostic long-interval interpolation is retained and flagged. 05 may retain a soft-target shortfall; consult results. Target intervals were not widened. No production parameters or original results overwritten.</p><p>Self-contained Plotly HTML. Use network selection for a full-width view. Requires a modern browser with gzip decompression support.</p></div><script id="payload" type="application/octet-stream">PAYLOAD</script><script>const DEFAULT_MODE="MODE";SCRIPT</script></body></html>'''
  if bounded:
   template=template.replace('Dashed muted red: original Cunge.', 'Dashed muted red: previous unrestricted-beta fit.')
   template=template.replace('Green bands/lines: your height targets.', 'Green bands/lines: former soft height preferences, now allowed to be missed.')
   template=template.replace('Height intervals are modelling acceptance ranges; point targets use a 0.1% numerical search tolerance. 13/37/48 use the proposed ±1% current-height bands.', 'Beta is constrained to 0.5–2.0. Former height preferences are retained only as soft comparison references; no height-band pass is required. Timing remains primary.')
   template=template.replace('Target intervals were not widened.', 'Height-reference misses are explicitly accepted for this bounded-beta check; no new observational targets were invented.')
   template=template.replace('15-example calibration: timing primary, height soft', '15 networks: beta 0.5–2, timing primary, height soft')
  h=template.replace('"MODE"',json.dumps(mode)).replace('SCRIPT',base.SCRIPT).replace('PAYLOAD',packed).replace('LIBRARY',library)
  path=dest/f'15_panel_calibrated_{mode}.html';path.write_text(h)
  print(path,round(path.stat().st_size/1024**2,1),'MiB',flush=True)
 (dest/'plot_manifest.json').write_text(json.dumps(dict(panels=15,offline=True,decimation=False,normalization='Fixed initial forcing; fixed downstream additive shift; original physical discharge hover',script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),waves={p['example']:hashlib.sha256((dest/p['example']/'wave.parquet').read_bytes()).hexdigest() for p in panels}),indent=2))
if __name__=='__main__':main()
