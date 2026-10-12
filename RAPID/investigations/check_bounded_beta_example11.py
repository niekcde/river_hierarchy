"""Separate diagnostic for unresolved example 11 on matched common-grid support."""
from pathlib import Path
import sys,json,shutil
import numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(Path(__file__).parent),str(ROOT/'RAPID/src')]
import validate_common_grid as prepare
import calibrate_common_grid_height as fit
from rapid_tools import cunge
from rapid_tools.common_grid import route_interval_means
OUT=ROOT/'RAPID/outputs/bounded_beta_example11_20261009'
SOURCE=fit.SOURCE

def main():
 OUT.mkdir(parents=True,exist_ok=False)
 inputs=OUT/'inputs';inputs.mkdir();(inputs/'baseline').symlink_to(SOURCE/'baseline',target_is_directory=True)
 table=pd.read_csv(SOURCE/'baseline/RAPID/outputs/cunge_calibration_reviewed_20261006/comparison.csv',dtype={'example':str})
 timing=json.loads((SOURCE/'baseline/RAPID/reviews/calibration_targets_20261006/targets.json').read_text())
 review=next(r for r in timing['records'] if r['example']=='11')
 prepare.VARIANTS=[('60s',60,0,1),('30s',30,0,1)]
 prepare.run_case('11',table[table.example.eq('11')],review,inputs)
 data=pd.read_parquet(inputs/'11/60s.parquet');meta=next(r for r in json.loads((inputs/'11/60s.json').read_text()) if r['method']=='cunge')
 target=dict(example='11',low_cms=0.,high_cms=float(data.forcing_cms.max()),downstream_display_offset_cms=float(meta['initial_cms']-data.downstream_cms.dropna().iloc[0]),soft_target_type='diagnostic_no_height_fit')
 (OUT/'height_targets.json').write_text(json.dumps(dict(records=[target]),indent=2));(OUT/'timing_targets.json').write_text(json.dumps(timing,indent=2))
 fit.SOURCE=inputs
 result=fit.calibrate('11',str(OUT),beta_bounds=(.5,2.))
 (OUT/'diagnostic_scope.json').write_text(json.dumps(dict(status='diagnostic only; no reinstatement or production update',timing_target=review,beta_bounds=[.5,2.],height='No observational height fit: only nonamplification reference; assess timing and shape separately',source_directory=str(SOURCE)),indent=2))
 print('11 DIAGNOSTIC',result,flush=True)

if __name__=='__main__':main()
