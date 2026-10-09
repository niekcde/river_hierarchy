"""Replay every selected candidate from parameters, including refined numerical grids."""
from pathlib import Path
import sys,json,hashlib
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'RAPID/src')]
from rapid_tools import cunge
from rapid_tools.common_grid import route_interval_means

def main():
 dest=Path(json.loads((ROOT/'RAPID/outputs/common_grid_height_calibration_20261008/latest.json').read_text())['directory']);source=Path(json.loads((dest/'protocol.json').read_text())['source_directory'])
 comp=pd.read_csv(source/'baseline/RAPID/outputs/cunge_calibration_reviewed_20261006/comparison.csv',dtype={'example':str});checks=[]
 for path in sorted(dest.glob('*/result.json')):
  ex=path.parent.name;r=json.loads(path.read_text());row=comp[(comp.example==ex)&(comp.method=='cunge')].iloc[0]
  grid=pd.read_parquet(source/'baseline'/Path(row.output_dir).relative_to(ROOT)/'frozen_grid.parquet');mult=r.get('subdivision_multiplier',1);grid['count']*=mult;grid['rapid_effective_length_m']=grid.link_length_m/grid['count']
  data=pd.read_parquet(source/ex/'60s.parquet');meta=next(m for m in json.loads((source/ex/'60s.json').read_text()) if m['method']=='cunge');dt=r.get('native_dt_s',meta['dt_s'])
  f=cunge.parameters(grid,r['n_manning']);f['rapid_x']=.5-r['beta']*(.5-f.rapid_x)
  np.testing.assert_allclose(f.rapid_celerity_mps*f.rapid_effective_length_m*(.5-f.rapid_x),r['beta']*f.hydraulic_diffusivity_m2_s,rtol=1e-12)
  q,a=route_interval_means(f,data.forcing_cms.to_numpy(),int(round(60/dt)),dt,initial_flow=meta['initial_cms'])
  wave=pd.read_parquet(path.parent/'wave.parquet');np.testing.assert_allclose(q,wave.calibrated_cunge_cms,rtol=1e-12,atol=1e-9);assert a['minimum_native_flow_cms']>=0
  r['minimum_native_flow_cms']=a['minimum_native_flow_cms'];r['native_dt_s']=dt;r['subdivision_multiplier']=mult;r['selected_wave_replayed']=True
  path.write_text(json.dumps(r,indent=2));checks.append(dict(example=ex,beta=r['beta'],n=r['n_manning'],native_dt_s=dt,subdivision_multiplier=mult,max_replay_difference_cms=float(np.max(abs(q-wave.calibrated_cunge_cms))),**a))
  print('replay',ex,'PASS',flush=True)
 (dest/'replay_validation.json').write_text(json.dumps(checks,indent=2))
 (dest/'replay_validation_source_sha256.txt').write_text(hashlib.sha256(Path(__file__).read_bytes()).hexdigest()+'\n')
if __name__=='__main__':main()
