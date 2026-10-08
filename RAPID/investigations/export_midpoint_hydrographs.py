"""Publish versioned midpoint-labelled means without changing discharge or routing."""
from pathlib import Path
import sys,json
import numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT/'RAPID/src'),str(ROOT)]
from rapid_tools.time_support import interval_bounds_and_midpoints
OUT=ROOT/'RAPID/outputs/temporal_and_target_audit_20261008';dest=OUT/'midpoint_calibration_hydrographs';dest.mkdir(parents=True,exist_ok=True)
for r in pd.read_csv(ROOT/'RAPID/outputs/cunge_calibration_reviewed_20261006/comparison.csv',dtype={'example':str}).itertuples():
 src=Path(r.output_dir)/f'{r.method}_best_wave.parquet';h=pd.read_parquet(src);bounds,mid=interval_bounds_and_midpoints(h.time_seconds,float(r.forcing_step_s));start=pd.to_datetime(h.time_utc,utc=True);d=pd.DataFrame(dict(time_utc=start+pd.to_timedelta(r.forcing_step_s/2,unit='s'),time_seconds=mid,interval_start_utc=start,interval_end_utc=start+pd.to_timedelta(r.forcing_step_s,unit='s'),interval_start_seconds=bounds[:,0],interval_end_seconds=bounds[:,1],q_inflow_mean_cms=h.q_inflow_cms,q_outlet_mean_cms=h.q_outlet_cms))
 np.testing.assert_array_equal(d.q_outlet_mean_cms,h.q_outlet_cms);np.testing.assert_array_equal(d.q_inflow_mean_cms,h.q_inflow_cms);d.to_parquet(dest/f'{r.example}_{r.method}.parquet',index=False)
(dest/'time_support.json').write_text(json.dumps(dict(time_utc='Geometric interval midpoint, for both input and output means.',bounds='Explicit UTC start/end; engine holds inlet over each interval.',outlet_quadrature='Arithmetic mean of native right-endpoint solver states; sample-time centroid is midpoint + dt/2.',inlet_preparation='Historical arithmetic mean of gauge samples per bin, with interpolation of empty bins.',native_values_modified=False,calibration_refitted=False,peak_policy='Use native outlet point times for peak timing; do not substitute maximum-mean bin midpoint for instantaneous peak.',gauge_comparison='Native outlet interpolated to recorded gauge timestamps; raw gauge timestamps are unchanged.'),indent=2));print('exported 32 midpoint hydrographs')
