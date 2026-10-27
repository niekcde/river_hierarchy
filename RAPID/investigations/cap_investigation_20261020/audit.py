"""Read-only source audit; all new files go to a separate investigation directory."""
from pathlib import Path
import sys,json,hashlib
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[3]
sys.path[:0]=[str(ROOT),str(ROOT/'RAPID/src'),str(ROOT/'RAPID/investigations/closed_unit_extension_20261019')]
from rapid_tools import cunge
from rapid_tools.common_grid_production import shared_clock
OUT=ROOT/'RAPID/outputs/cap_investigation_20261020';EXT=ROOT/'RAPID/outputs/hydrograph_closed_unit_extension_20261019'
SOURCES={}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):
 p=Path(p);SOURCES[str(p)]=sha(p)
 return json.loads(p.read_text()) if p.suffix=='.json' else pd.read_parquet(p)
def save(p,value):p=Path(p);p.parent.mkdir(exist_ok=True,parents=True);p.write_text(json.dumps(value,indent=2))
def uncapped_grid(f):
 # Finite ceiling above every raw speed throughout the declared n range, retaining lower cap.
 ceiling=max(float((f.celerity_times_n/f.grid_n_min).max())*1.01,5.01)
 root=cunge.topology(f)[2];q=float(f.loc[f.u.eq(root),'reference_discharge_cms'].sum())
 grid=cunge.freeze_grid(f,q,n_bounds=(float(f.grid_n_min.iloc[0]),float(f.grid_n_max.iloc[0])),alpha=float(f.cunge_alpha.iloc[0]),min_celerity=float(f.celerity_min_mps.iloc[0]),max_celerity=ceiling)
 grid['count']=np.maximum(grid['count'],f['count'].to_numpy());grid['rapid_effective_length_m']=grid.link_length_m/grid['count']
 return grid

def main():
 OUT.mkdir(exist_ok=True);plan=read(ROOT/'RAPID/outputs/event_window_plan_20261013/plan.json');ext=read(EXT/'routing_plan.json');rows=[];jobsout=[]
 protected=[]
 for directory in [ROOT/'RAPID/outputs/hydrograph_response',ROOT/'RAPID/outputs/common_grid_height_calibration_20261008/33847083c6ea',ROOT/'RAPID/outputs/hydrograph_no_exposure_20261019']:
  protected.extend(p for p in directory.rglob('*') if p.is_file())
 snapshot={str(p):sha(p) for p in protected};save(OUT/'protected_before.json',snapshot)
 for case in plan['cases']:
  ex=case['example'];ec=next(c for c in ext['cases'] if c['example']==ex);jobs=[*case['jobs'],next(j for j in ec['jobs'] if j['state_id']=='all_closed_units_collapsed')]
  for j in jobs:
   f=read(j['parameters']);L=f.reference_flow_fraction*f.link_length_m;current=float((L/f.rapid_celerity_mps).sum()/3600);no_upper=np.maximum(f.rapid_celerity_mps_raw,f.celerity_min_mps);raw=float((L/no_upper).sum()/3600);grid=uncapped_grid(f);g=cunge.parameters(grid,float(f.n_manning.iloc[0]),beta=float(f.diffusion_multiplier.iloc[0]));assert (g.rapid_celerity_mps_raw<=g.celerity_max_mps).all()
   dt=shared_clock([g],[30,60],j['native_dt_seconds']);cohort='parent' if j['state_id']=='S000_base' else 'segmented' if j['state_id']=='all_closed_units_collapsed' else 'uniform' if j['state_id']=='single_centerline' else 'original'
   row=dict(example=ex,state=j['state_id'],cohort=cohort,links=len(f),upper_capped_links=int((f.rapid_celerity_mps_raw>f.celerity_max_mps).sum()),lower_capped_links=int((f.rapid_celerity_mps_raw<f.celerity_min_mps).sum()),upper_capped_length_fraction=float(L[f.rapid_celerity_mps_raw>f.celerity_max_mps].sum()/L.sum()),max_raw_celerity=float(f.rapid_celerity_mps_raw.max()),mean_route_hours=current,uncapped_route_hours=raw,cap_delay_hours=current-raw,n_manning=float(f.n_manning.iloc[0]),beta=float(f.diffusion_multiplier.iloc[0]),old_cells=int(f['count'].sum()),uncapped_cells=int(g['count'].sum()),old_dt_seconds=j['native_dt_seconds'],uncapped_dt_seconds=dt,parameters=j['parameters'])
   rows.append(row);jobsout.append(dict(example=ex,**j))
  print(ex,'audited',len(jobs),'states',flush=True)
 d=pd.DataFrame(rows);parent=d[d.cohort.eq('parent')].set_index('example').cap_delay_hours;d['change_in_collapse_shift_hours']=d.example.map(parent)-d.cap_delay_hours
 d.to_csv(OUT/'state_cap_audit.csv',index=False);save(OUT/'jobs.json',jobsout);save(OUT/'source_hashes.json',SOURCES)
 group=d.groupby('example').agg(states=('state','size'),affected_states=('upper_capped_links',lambda x:int((x>0).sum())),max_raw_celerity=('max_raw_celerity','max'),max_abs_shift_effect_hours=('change_in_collapse_shift_hours',lambda x:float(abs(x).max())))
 group.to_csv(OUT/'network_cap_audit.csv');print(group.to_string())
if __name__=='__main__':main()
