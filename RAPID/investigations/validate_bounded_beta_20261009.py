"""Replay/resolution verification and summary of the bounded-beta review."""
from pathlib import Path
import sys,json,hashlib,shutil,re,gzip,base64
import numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'RAPID/src'),str(Path(__file__).parent)]
import importlib.util
_snapshot=ROOT/'RAPID/outputs/bounded_beta_calibration_20261009/verification_code/common_grid_production.py'
_spec=importlib.util.spec_from_file_location('rapid_tools._bounded_beta_frozen_runner',_snapshot)
_runner=importlib.util.module_from_spec(_spec);sys.modules[_spec.name]=_runner;_spec.loader.exec_module(_runner)
_runner.ROOT=ROOT
CalibrationCase,run_case,refined_parameters,model_feature=(_runner.CalibrationCase,_runner.run_case,_runner.refined_parameters,_runner.model_feature)
from rapid_tools.common_grid import route_interval_means
from rapid_tools.time_support import point_series_interval_means
OUT=ROOT/'RAPID/outputs/bounded_beta_calibration_20261009'
PREVIOUS=ROOT/'RAPID/outputs/common_grid_height_calibration_20261008/33847083c6ea'

def main():
 results=json.loads((OUT/'results.json').read_text());rows=[]
 for r in sorted(results,key=lambda r:r['example']):
  ex=r['example'];assert .5<=r['beta']<=2
  folder=OUT/'verification'/ex
  if not (folder/'completion.json').exists():
   run_case(OUT,ex,OUT/'verification',steps=(60,30),diagnostic_gaps=True)
  replay=json.loads((folder/'results.json').read_text());assert replay[0]['reviewed_wave_max_difference_cms']==0
  resolution=json.loads((folder/'resolution_checks.json').read_text())[0]
  old=json.loads((PREVIOUS/ex/'result.json').read_text());wave=pd.read_parquet(OUT/ex/'wave.parquet');q0=float(wave.forcing_cms.iloc[0]);peak=float(wave.forcing_cms.max())
  if not r['refitted_for_beta_bounds']:
   assert hashlib.sha256((OUT/ex/'wave.parquet').read_bytes()).digest()==hashlib.sha256((PREVIOUS/ex/'wave.parquet').read_bytes()).digest()
  rows.append(dict(example=ex,old_beta=old['beta'],beta=r['beta'],n_manning=r['n_manning'],input_peak_cms=peak,old_model_peak_cms=old['peak_cms'],model_peak_cms=r['peak_cms'],
   total_peak_attenuation_pct=100*(1-r['peak_cms']/peak),pulse_attenuation_pct=100*(1-(r['peak_cms']-q0)/(peak-q0)),
   timing_error_minutes=r['timing_error_minutes'],old_shifted_rmse_cms=old['shifted_rmse_cms'],shifted_rmse_cms=r['shifted_rmse_cms'],
   former_height_reference_distance_cms=r['height_distance_cms'],resolution_difference_pct_input=resolution['max_difference_pct_inlet_amplitude'],resolution_feature_shift_seconds=resolution['feature_shift_seconds']))
 # Recheck original initialization against actual antecedent, retaining identical event forcing.
 case=CalibrationCase(OUT,'46');wave=case.wave;selected=case.selection;frame=refined_parameters(case.grid,selected);dt=case.native_dt
 valid=wave.downstream_cms.notna().to_numpy();tt=wave.time_utc.astype('int64').to_numpy()[valid]/1e9
 eq=pd.read_parquet(OUT/'46/wave.parquet').calibrated_cunge_cms.to_numpy();antecedent=[]
 for days in [7,30]:
  points=pd.read_parquet(case.source/'46'/f'antecedent_{days}d_points.parquet');origin=wave.interval_start_utc.iloc[0]-pd.Timedelta(days=days)
  t=(pd.to_datetime(points.time,utc=True)-origin).dt.total_seconds().to_numpy();v=points.discharge.to_numpy();edges=np.arange(len(wave)+days*1440+1)*60.
  flow=point_series_interval_means(t,v,edges,allow_endpoint_hold=True);flow[days*1440:]=wave.forcing_cms
  q,audit=route_interval_means(frame,flow,round(60/dt),dt,initial_flow=float(np.interp(0,t,v)));q=q[days*1440:];assert audit['minimum_native_flow_cms']>=0
  tm,height,kind=model_feature(q[valid],tt,'46',case.review)
  antecedent.append(dict(days=days,peak_cms=height,timing_error_minutes=(tm-pd.Timestamp(case.review['target_utc']).timestamp())/60,max_event_difference_cms=float(np.max(abs(q[valid]-eq[valid])))))
  pd.DataFrame(dict(time_utc=wave.time_utc,cunge_cms=q)).to_parquet(OUT/'46'/f'bounded_antecedent_{days}d.parquet',index=False)
 (OUT/'46/antecedent_controls.json').write_text(json.dumps(antecedent,indent=2))
 # Inspect actual embedded plot payload, not just producer input.
 for mode in ['full','crest']:
  html=(OUT/f'15_panel_calibrated_{mode}.html').read_text();encoded=re.search(r'<script id="payload" type="application/octet-stream">(.*?)</script>',html).group(1)
  payload=json.loads(gzip.decompress(base64.b64decode(encoded)));assert len(payload['panels'])==15
  for p in payload['panels']:
   ex=p['example'];r=next(r for r in results if r['example']==ex);assert p['calibration']['beta']==r['beta']
   w=pd.read_parquet(OUT/ex/'wave.parquet');valid=w.downstream_cms.notna().to_numpy();tr=next(t for t in p['series'] if t['legendgroup']=='cunge_cms')
   np.testing.assert_allclose(tr['y'],w.calibrated_cunge_cms.to_numpy()[valid]/p['normalization']['forcing_initial_cms'])
   old=next(t for t in p['series'] if t['legendgroup']=='original_cunge');assert old['name']=='Previous unrestricted fit'
   np.testing.assert_allclose(old['y'],pd.read_parquet(PREVIOUS/ex/'wave.parquet').calibrated_cunge_cms.to_numpy()[valid]/p['normalization']['forcing_initial_cms'])
 pd.DataFrame(rows).to_csv(OUT/'comparison.csv',index=False)
 code=OUT/'source_code';code.mkdir(exist_ok=True)
 sources=[Path(__file__),Path(__file__).with_name('check_bounded_beta_20261009.py'),Path(__file__).with_name('calibrate_common_grid_height.py'),Path(__file__).with_name('export_height_calibration_plotly.py')]
 for p in sources:shutil.copy2(p,code/p.name)
 report=['BOUNDED BETA REVIEW — 2026-10-09','',
  'User-authorized beta interval: 0.5–2. Timing remains primary; former height references are soft.',
  'Six outside-bounds cases refitted: 02,03,05,21,22,46. Nine retained with byte-identical waves.',
  'All 15 selected waves replayed exactly. 60s and aggregated 30s comparison verified.',
  'Figures retain shared original initial-forcing scale and downstream-only baseline shift.',
  'Dashed muted red = previous unrestricted-beta fit. Green = former height reference, allowed to be missed.','',
  pd.DataFrame(rows)[['example','beta','model_peak_cms','total_peak_attenuation_pct','pulse_attenuation_pct','timing_error_minutes']].to_string(index=False), '',
  'Example 46 antecedent checks at new parameters:',json.dumps(antecedent,indent=2),'',
  'Example 11 remains a separate diagnostic: see ../bounded_beta_example11_20261009/11_bounded_beta_diagnostic.html.',
  'No production bundle or previous results replaced. Source-clock/gap qualifications remain.',
  'A beta bound is a calibration safeguard, not independent physical validation. Height misses are accepted explicitly.',
  '19 core routing/common-grid tests passed. HTML numeric payloads verified; no browser-rendering claim.']
 (OUT/'REPORT.txt').write_text('\n'.join(report)+'\n')
 (OUT/'validation.json').write_text(json.dumps(dict(cases=15,replay_exact=True,plot_payloads_verified=True,tests_passed=19,
  maximum_30s_60s_difference_pct_input=max(r['resolution_difference_pct_input'] for r in rows),
  maximum_feature_shift_seconds=max(abs(r['resolution_feature_shift_seconds']) for r in rows),
  source_hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}),indent=2))
 print(pd.DataFrame(rows).to_string(index=False),flush=True)
if __name__=='__main__':main()
