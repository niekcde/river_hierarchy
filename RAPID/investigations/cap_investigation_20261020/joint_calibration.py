"""Isolated uncapped n/beta calibration using the archived selection algorithm."""
from pathlib import Path
import sys,ast,json,hashlib,math,argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from audit import ROOT,uncapped_grid,sha,cunge,shared_clock
from rapid_tools.common_grid import route_interval_means

BUNDLE=ROOT/'RAPID/outputs/bounded_beta_calibration_20261009'
SOURCE=Path(json.loads((BUNDLE/'protocol.json').read_text())['source_directory'])
DEST=ROOT/'RAPID/outputs/uncapped_joint_calibration_20261020'
ARCHIVE=BUNDLE/'source_code/calibrate_common_grid_height.py'
JOBS=json.loads((ROOT/'RAPID/outputs/cap_investigation_20261020/jobs.json').read_text())
PARENTS={j['example']:j for j in JOBS if j['state_id']=='S000_base'}

def replacement_grid(ex):
 f=pd.read_parquet(PARENTS[ex]['parameters'])
 grid=uncapped_grid(f)
 assert (grid.celerity_times_n/.01 < grid.celerity_max_mps).all()
 return grid

# Extract only the archived functions, avoiding its original main and mutable latest pointers.
tree=ast.parse(ARCHIVE.read_text())
for name in ['feature','calibrate']:
 node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name)
 text=ast.get_source_segment(ARCHIVE.read_text(),node)
 if name=='calibrate':
  start=text.index('    comp=pd.read_csv(');end=text.index("    data=pd.read_parquet(",start)
  text=text[:start]+"    grid=replacement_grid(ex)\n    subdivision=1\n"+text[end:]
  original="orig_n=meta['n_manning']"
  assert text.count(original)==1
  text=text.replace(original,"orig_n=(previous_selection or {}).get('n_manning',meta['n_manning'])\n    dt=shared_clock([cunge.parameters(grid,n,beta=b) for n in [.01,.1] for b in beta_bounds],[30,60],dt)")
  # Always include the active beta and previous conditional n as candidates.
  text=text.replace("baseline=fit_beta(1.)","fit_beta((previous_selection or {}).get('beta',1.))\n    baseline=fit_beta(1.)")
  text=text.replace("best.update(native_dt_s=dt,subdivision_multiplier=subdivision,beta_bounds=list(beta_bounds))",
    "best.update(native_dt_s=dt,subdivision_multiplier=subdivision,beta_bounds=list(beta_bounds),n_bounds=[.01,.1],n_at_search_boundary=bool(np.isclose(best['n_manning'],.01) or np.isclose(best['n_manning'],.1)),upper_cap_removed=True,grid_cells=int(grid['count'].sum()))")
 exec(compile(text,str(ARCHIVE)+' [isolated uncapped adapter]','exec'),globals())

def worker(ex):
 prior=json.loads((BUNDLE/ex/'result.json').read_text())
 return calibrate(ex,str(DEST),beta_bounds=(.5,2.),previous_selection=prior)

def summarize():
 rows=[]
 for ex in sorted(PARENTS):
  path=DEST/ex/'result.json'
  if not path.exists():continue
  new=json.loads(path.read_text());old=json.loads((BUNDLE/ex/'result.json').read_text())
  rows.append(dict(example=ex,status=new.get('status','joint_refit'),old_n=old['n_manning'],new_n=new['n_manning'],old_beta=old['beta'],new_beta=new['beta'],timing_error_minutes=new['timing_error_minutes'],old_height_distance_cms=old['height_distance_cms'],new_height_distance_cms=new['height_distance_cms'],beta_at_boundary=new['beta'] in [.5,2.],n_at_boundary=bool(np.isclose(new['n_manning'],.01) or np.isclose(new['n_manning'],.1)),trials=new.get('trials',0)))
 d=pd.DataFrame(rows);d.to_csv(DEST/'comparison.csv',index=False);return d

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--run',action='store_true');parser.add_argument('--jobs',type=int,default=2);args=parser.parse_args()
 DEST.mkdir(exist_ok=True)
 paths=[ARCHIVE,Path(__file__),ROOT/'RAPID/investigations/cap_investigation_20261020/audit.py',ROOT/'RAPID/src/rapid_tools/cunge.py',ROOT/'RAPID/src/rapid_tools/common_grid.py']
 paths+=list(BUNDLE.rglob('*'));paths=[p for p in paths if p.is_file()]
 paths += [Path(j['parameters']) for j in PARENTS.values()]
 paths += [SOURCE/ex/'60s.parquet' for ex in PARENTS]
 paths += [SOURCE/ex/'60s.json' for ex in PARENTS]
 hashes={str(p):sha(p) for p in paths}
 for name in ['height_targets.json','timing_targets.json']:
  content=(BUNDLE/name).read_bytes();target=DEST/name
  if target.exists():assert target.read_bytes()==content
  else:target.write_bytes(content)
 affected=[]
 for ex,j in sorted(PARENTS.items()):
  f=pd.read_parquet(j['parameters']);active=bool((f.rapid_celerity_mps_raw>f.celerity_max_mps).any())
  if active:affected.append(ex)
  else:
   folder=DEST/ex;folder.mkdir(exist_ok=True)
   old=json.loads((BUNDLE/ex/'result.json').read_text())
   old.update(status='retained_no_active_parent_upper_cap',upper_cap_removed=True,qualification='No refit: removing the upper cap does not alter parameters at the existing selected n/beta. Not a new uncapped search.')
   (folder/'result.json').write_text(json.dumps(old,indent=2))
 protocol=dict(n_bounds=[.01,.1],beta_bounds=[.5,2.],affected_examples=affected,
  objective='Archived timing-first search, <=1min timing tolerance; original height references soft, beta nearest1 preference as archived. No bounds widened.',
  scope='Seven affected parent networks jointly refit; eight unaffected selections retained. Lower celerity bound retained. No child reruns or production changes.',
  numerical='Uncapped grid valid across n bounds; common clock over n/beta corners; archived half-dt and half-dx/half-dt checks.',
  caveat='Bounds are existing modelling constraints, not independent physical validation. A boundary optimum signals unresolved preference beyond the tested range.',
  source_hashes=hashes)
 (DEST/'protocol.json').write_text(json.dumps(protocol,indent=2))
 print('Affected examples:',affected,flush=True)
 if not args.run:return
 pending=[ex for ex in affected if not (DEST/ex/'result.json').exists()]
 with ProcessPoolExecutor(max_workers=args.jobs) as pool:
  futures={pool.submit(worker,ex):ex for ex in pending}
  for future in as_completed(futures):
   r=future.result();print('COMPLETED',r['example'],flush=True);summarize()
 assert all(sha(p)==h for p,h in hashes.items()),'Source file changed'
 d=summarize();print(d.to_string(index=False))
 (DEST/'completion.json').write_text(json.dumps(dict(complete=True,production_modified=False,sources_unchanged=True,examples=len(d)),indent=2))

if __name__=='__main__':main()
