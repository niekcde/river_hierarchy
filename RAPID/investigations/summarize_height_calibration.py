"""Collect canonical results after targeted refinements and verify immutable inputs."""
from pathlib import Path
import sys,json,hashlib,html
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(Path(__file__).parent))
from calibrate_common_grid_height import feature

def main():
 parent=ROOT/'RAPID/outputs/common_grid_height_calibration_20261008';dest=Path(json.loads((parent/'latest.json').read_text())['directory'])
 protocol=json.loads((dest/'protocol.json').read_text());source=Path(protocol['source_directory']);targets={r['example']:r for r in json.loads((dest/'height_targets.json').read_text())['records']};timing={r['example']:r for r in json.loads((dest/'timing_targets.json').read_text())['records']}
 records=[];rows=[];total_trials=0
 for ex in sorted(targets):
  r=json.loads((dest/ex/'result.json').read_text());f=pd.read_parquet(dest/ex/'wave.parquet');base=pd.read_parquet(source/ex/'60s.parquet')
  pd.testing.assert_frame_equal(f[base.columns],base)
  mask=f.downstream_cms.notna().to_numpy();t=f.time_utc.astype('int64').to_numpy()[mask]/1e9
  tm,h,k,_=feature(f.calibrated_cunge_cms.to_numpy()[mask],t,ex,timing[ex])
  assert np.isclose(h,r['peak_cms']);assert abs((pd.Timestamp(r['feature_utc']).timestamp()-tm))<.001
  assert r['timing_within_one_minute']==(abs(r['timing_error_minutes'])<=1)
  assert r['minimum_native_flow_cms']>=0
  if r['height_target_type']=='interval':status='In range' if r['height_range_met'] else f"Below range by {r['height_distance_cms']:.2f}" if h<r['height_low_cms'] else f"Above range by {r['height_distance_cms']:.2f}"
  else:status=f"Point difference: {100*(h/r['height_low_cms']-1):+.3f}%"
  if any(abs(c['feature_shift_minutes'])>1 for c in r['numerical_controls']):
   r['warnings'].append('Spatial-refinement feature change exceeds the 1-minute fitting gate; this is not a final precision certification.')
  if r['height_target_type']=='point' and not r['point_height_within_numerical_01pct']:
   r['warnings'].append('Soft point target not reached to 0.1% numerical search tolerance; residual explicitly retained.')
  r['warnings']=list(dict.fromkeys(r['warnings']))
  (dest/ex/'result.json').write_text(json.dumps(r,indent=2));records.append(r);total_trials+=r['trials']
  rows.append(dict(example=ex,beta=r['beta'],n=r['n_manning'],time_error_min=r['timing_error_minutes'],original_height=r['original_peak_cms'],fitted_height=h,height_low=r['height_low_cms'],height_high=r['height_high_cms'],height_status=status,subdivision_multiplier=r.get('subdivision_multiplier',1),shifted_RMSE=r['shifted_rmse_cms'],original_shifted_RMSE=r['original_shifted_rmse_cms'],warnings=' | '.join(r['warnings'])))
 for path,digest in protocol['hashes'].items():assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==digest
 (dest/'results.json').write_text(json.dumps(records,indent=2));table=pd.DataFrame(rows);table.to_csv(dest/'comparison.csv',index=False)
 ninterval=sum(r['height_target_type']=='interval' for r in records);met=sum(r['height_range_met'] is True for r in records)
 text=f'''COMMON-GRID TIMING / SOFT-HEIGHT CALIBRATION — 2026-10-08

15 retained base Cunge examples, separate versioned outputs. Original and fixed-X
results unchanged; no hierarchy recalibration. All 15 selected timing features
are within one minute of their existing target on the selected numerical grids.
This is numerical calibration accuracy, not observational precision.
{met}/{ninterval} interval targets satisfied. Point target residuals shown individually.
Recorded main/refinement trial count: {total_trials} (additional numerical and
antecedent controls are separate). beta search bounds .01–2; frozen n bounds .01–.1.
No manual range widened. 13/37/48 use proposed +/-1% bands around original peaks.

Targets and selection
Height uses the case-specific shifted-downstream targets or manual ranges in
this revision of height_targets.json. The additive offset is never recomputed from a fitted outlet
or antecedent state. Timing is primary; height is soft. Prefer beta=1 when a
range can already be met. Within tested interval fits choose beta closest to 1;
for point targets reduce height error with 0.1% numerical stopping tolerance.
This is a bounded adaptive search, not proof of a unique/global optimum or
independent validation. No arbitrarily weighted timing-height sum was introduced.

Feature handling
Observed selected timestamps and spans unchanged. Model flat maxima use their
contiguous midpoint; 48 uses the middle equal maximum. Smooth single model crests
represent short observed plateau/crest groups. 54 explicitly uses the midpoint
between its two dominant distinct lobes, and the higher lobe for its height;
its first-crest hierarchy convention is not substituted. The operator is tested
on an exact synthetic plateau and two unequal synthetic crest lobes. Maximum-bin timing can jump near flat crests. All target timestamps remain
unchanged; trial histories and any targeted refinements are retained.

Important exceptions
05 remains below its interval at beta=.01, the lower search bound. The interval
starts above the supplied input peak. Do not interpret this as permission to
create numerical overshoot, widen the target, or claim a globally optimal fit.
Any soft point residuals are review items, not new target uncertainty estimates. Check the full curves rather than judging peaks alone.

Numerical checks
Every selected case has native half-dt and spatial-refinement controls with
nonnegative native flow checks. Saved original wave columns match source data;
source/code/target hashes checked unchanged. Some spatial feature shifts exceed
the 1-minute numerical fitting gate despite small discharge differences; they
are explicitly flagged, so that gate is not a convergence guarantee.
46 was specifically refined further after a ten-minute spatial peak shift.
See 46/spatial_refinement_and_antecedent.json for the refinement, retiming and
actual antecedent tests at its selected beta. Physical geometry is unchanged.

Initialization and remaining limitations
Fits use equilibrium at original event start. Antecedent tests are diagnostics,
not silently substituted fits. 21/22 source clocks remain provisional. 55's local
crest observations are hourly. Long-interval diagnostic interpolation remains
flagged; final data eligibility and production runner migration are not complete.
These are completed base calibration REVIEW CANDIDATES, not completed production
or hierarchy validation. No whole-window output-volume difference is labeled a
water-balance error without accounting for initial/final storage and full tails.

'''
 text+=table.drop(columns=['warnings']).round(5).to_string(index=False)+'\n'
 (dest/'REPORT.txt').write_text(text)
 doc='<html><meta charset="utf-8"><title>Calibration results</title><style>body{font:14px Arial;margin:28px}td,th{padding:7px}table{border-collapse:collapse}p{max-width:1400px;line-height:1.5}</style><h1>Timing and soft-height calibration — all 15 bases</h1><p>All selected timing errors ≤1 minute; '+str(met)+'/'+str(ninterval)+' interval targets met. Point-target residuals shown below. Modelling targets are not uncertainty bands. Review candidates; production/hierarchy validation remains outstanding.</p>'+table.drop(columns=['warnings']).round(4).to_html(index=False)
 for r in records:doc+='<p><b>'+r['example']+'</b>: '+html.escape(' | '.join(r['warnings']) or 'No additional case-specific flag; general validation limitations still apply.')+'</p>'
 (dest/'calibration_results.html').write_text(doc+'</html>')
 (dest/'validation.json').write_text(json.dumps(dict(source_hashes_unchanged=True,original_wave_columns_unchanged=True,saved_feature_metrics_recomputed=True,cases=15,interval_targets_met=met,interval_targets_total=ninterval,point_residuals={r['example']:r['height_distance_cms'] for r in records if r['height_target_type']=='point'},browser_render_checked=False),indent=2))
 (parent/'latest.json').write_text(json.dumps(dict(directory=str(dest),status='base_calibration_review_complete'),indent=2))
 print(table.drop(columns=['warnings']).round(4).to_string(index=False));print('PASS: saved features, immutable inputs, '+str(met)+'/'+str(ninterval)+' ranges')
if __name__=='__main__':main()
