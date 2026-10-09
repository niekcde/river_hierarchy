"""User-requested second target revision; refit four cases, preserve others."""
from pathlib import Path
import sys,json,hashlib,shutil
from concurrent.futures import ProcessPoolExecutor,as_completed
import pandas as pd
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(Path(__file__).parent))
import calibrate_common_grid_height as engine

def main():
 parent=ROOT/'RAPID/outputs/common_grid_height_calibration_20261008'
 previous=parent/'fca75ef22cf5'
 targets=json.loads((previous/'height_targets.json').read_text())
 changes={'02':2800.,'03':2550.,'21':14350.,'41':7500.}
 for r in targets['records']:
  ex=r['example']
  if ex not in changes:continue
  r['previous_low_cms']=r['low_cms'];r['previous_high_cms']=r['high_cms'];r['previous_nominal_cms']=r.get('height_target_nominal_cms')
  r['low_cms']=changes[ex]
  if ex=='03':
   r['high_cms']=r['forcing_peak_cms']
   r.pop('height_target_nominal_cms',None)
   r['height_objective_basis']='user-defined near-input modelling range, replacing shifted-downstream point'
   r['soft_target_type']='interval'
   r['notes'].append('User requested lower bound 2550 and closer to input. Working upper bound is the one-minute input peak (2631.413 m3/s); explicitly communicated assumption, not a new downstream observation.')
  r['status']='user_revised_range'
  r['notes'].append('Revision 2: timing remains primary; choose smallest tested change from beta=1 that meets the revised interval.')
 targets['status']='Revision 2: user changed lower bounds of 02,03,21,41; remaining fits retained'
 target_path=ROOT/'RAPID/reviews/common_grid_production_20261008/manual_height_targets_revision2.json'
 target_path.write_text(json.dumps(targets,indent=2))
 protocol=json.loads((previous/'protocol.json').read_text())
 protocol.update(revision=2,previous_directory=str(previous),refitted_examples=list(changes),retained_examples=[r['example'] for r in targets['records'] if r['example'] not in changes],height='Revision 2 intervals for 02/03/21/41. Other cases retain prior targets and waves; 05 retained as user accepted the result despite its target shortfall.',revision_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
 protocol['hashes'][str(target_path)]=hashlib.sha256(target_path.read_bytes()).hexdigest()
 sig=hashlib.sha256(json.dumps(dict(targets=targets,protocol=protocol),sort_keys=True).encode()).hexdigest()[:12]
 dest=parent/sig;dest.mkdir(parents=True,exist_ok=True)
 shutil.copy2(target_path,dest/'height_targets.json');shutil.copy2(previous/'timing_targets.json',dest/'timing_targets.json')
 (dest/'protocol.json').write_text(json.dumps(protocol,indent=2));shutil.copy2(__file__,dest/'revision_source.py');shutil.copy2(engine.__file__,dest/'calibration_source.py')
 for ex in protocol['retained_examples']:
  shutil.copytree(previous/ex,dest/ex,dirs_exist_ok=True)
 (parent/'latest.json').write_text(json.dumps(dict(directory=str(dest),status='revision2_running'),indent=2))
 with ProcessPoolExecutor(max_workers=2) as pool:
  futures={pool.submit(engine.calibrate,ex,str(dest)):ex for ex in changes}
  for future in as_completed(futures):
   result=future.result();print('REVISED',result['example'],result['beta'],result['n_manning'],result['peak_cms'],result['timing_error_minutes'],result['height_range_met'],flush=True)
 for path,digest in protocol['hashes'].items():assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==digest
 rows=[]
 for ex in changes:
  old=json.loads((previous/ex/'result.json').read_text());new=json.loads((dest/ex/'result.json').read_text())
  rows.append(dict(example=ex,old_beta=old['beta'],new_beta=new['beta'],old_n=old['n_manning'],new_n=new['n_manning'],old_height=old['peak_cms'],new_height=new['peak_cms'],height_low=new['height_low_cms'],height_high=new['height_high_cms'],timing_error_minutes=new['timing_error_minutes'],range_met=new['height_range_met'],old_shifted_rmse=old['shifted_rmse_cms'],new_shifted_rmse=new['shifted_rmse_cms']))
 pd.DataFrame(rows).to_csv(dest/'revision_comparison.csv',index=False)
 for ex in protocol['retained_examples']:
  assert hashlib.sha256((dest/ex/'wave.parquet').read_bytes()).hexdigest()==hashlib.sha256((previous/ex/'wave.parquet').read_bytes()).hexdigest()
 (parent/'latest.json').write_text(json.dumps(dict(directory=str(dest),status='revision2_calibration_complete'),indent=2))
 print(pd.DataFrame(rows).to_string(index=False));print('OUTPUT',dest,flush=True)
if __name__=='__main__':main()
