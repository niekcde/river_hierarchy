"""Bounded beta revision: timing first, former height preferences explicitly soft."""
from pathlib import Path
import sys,json,shutil,hashlib
from concurrent.futures import ProcessPoolExecutor,as_completed
import pandas as pd
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(Path(__file__).parent),str(ROOT/'RAPID/src')]
import calibrate_common_grid_height as fit
PREVIOUS=ROOT/'RAPID/outputs/common_grid_height_calibration_20261008/33847083c6ea'
OUT=ROOT/'RAPID/outputs/bounded_beta_calibration_20261009'

def main():
 OUT.mkdir(parents=True,exist_ok=False)
 for name in ['height_targets.json','timing_targets.json']:shutil.copy2(PREVIOUS/name,OUT/name)
 records={p.parent.name:json.loads(p.read_text()) for p in PREVIOUS.glob('*/result.json')}
 changed=[ex for ex,r in records.items() if not .5<=r['beta']<=2]
 protocol=dict(source_directory=str(fit.SOURCE),previous_directory=str(PREVIOUS),beta_bounds=[.5,2.],refitted_examples=sorted(changed),
  height='Former references retained as soft preferences only: accept residuals within beta bounds; no mandatory height pass or invented widened target.',
  selection='Timing within one minute first; retain former within-bounds fits. Refit outside-bounds cases, prefer height match then beta nearest one among timing-feasible candidates.',
  controls='Inherit selected spatial refinement and native clock; halve native dt and spatial cell length for selected refits.',
  example11='Separate diagnostic, not included in 15 or promoted to production')
 paths=[Path(__file__),Path(fit.__file__),ROOT/'RAPID/src/rapid_tools/cunge.py',ROOT/'RAPID/src/rapid_tools/common_grid.py',PREVIOUS/'height_targets.json',PREVIOUS/'timing_targets.json']
 protocol['hashes']={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
 (OUT/'protocol.json').write_text(json.dumps(protocol,indent=2))
 for ex,r in records.items():
  if ex not in changed:shutil.copytree(PREVIOUS/ex,OUT/ex)
 with ProcessPoolExecutor(max_workers=2) as pool:
  futures={pool.submit(fit.calibrate,ex,str(OUT),beta_bounds=(.5,2.),previous_selection=records[ex]):ex for ex in sorted(changed)}
  for future in as_completed(futures):
   r=future.result();print('BOUNDED SELECTED',r['example'],r['beta'],r['n_manning'],r['peak_cms'],r['timing_error_minutes'],flush=True)
 results=[]
 for ex in sorted(records):
  p=OUT/ex/'result.json';r=json.loads(p.read_text());r.update(height_reference_is_soft=True,beta_bounds=[.5,2.],refitted_for_beta_bounds=ex in changed)
  p.write_text(json.dumps(r,indent=2));results.append(r)
 (OUT/'results.json').write_text(json.dumps(results,indent=2))
 for path,digest in protocol['hashes'].items():assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==digest
 from export_height_calibration_plotly import main as plot
 plot(OUT,previous=PREVIOUS,bounded=True)
 (OUT/'completion.json').write_text(json.dumps(dict(status='bounded_beta_review',examples=15,production_updated=False),indent=2))

if __name__=='__main__':main()
