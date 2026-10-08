from pathlib import Path
import sys,json,time
import pandas as pd,numpy as np
R=Path('/Users/6256481/Code/river-hierarchy');sys.path.insert(0,str(R));sys.path.insert(0,str(R/'RAPID/src'))
from rapid_tools import cunge
c=pd.read_csv(R/'RAPID/outputs/cunge_calibration_alpha2/comparison.csv',dtype={'example':str});c=c[c.method=='cunge'];rows=[]
for r in c.itertuples():
 ex=r.example;p=next((p for p in [R/'RAPID/outputs/cunge_full_hierarchy_pilot'/ex,Path('/Volumes/PhD/river_hierarchy/cunge_full_hierarchy_runs')/ex] if (p/'completion.json').exists()),None)
 if p:
  m=json.loads((p/'experiment.json').read_text());done=json.loads((p/'completion.json').read_text());u=pd.read_csv(p/'unique_run_registry.csv');nt=m['report_steps'];sections=int(u.n_sections.sum());nsub=m['nsub'];dt=m['routing_dt_s'];count=len(u);actual=done['actual_bytes'];engine_h=u.runtime_seconds.sum()/3600;wall_h=((p/'completion.json').stat().st_mtime-(p/'exterior_invariance.csv').stat().st_mtime)/3600
 else:
  folder=Path(r.output_dir);m=json.loads((folder/'case.json').read_text());nt=len(pd.read_parquet(folder/'cunge_best_wave.parquet'));inv=pd.read_csv(R/'RAPID/assets/bounded_robust_slopes_20261002/child_head_reconstruction_20261003'/ex/'inventory.csv');inv=inv[inv.status=='ready'].drop_duplicates('fingerprint');sections=0;dtmax=15
  for i,s in enumerate(inv.itertuples()):
   g=cunge.freeze_grid(pd.read_parquet(s.frame),m['reference_discharge_cms']);sections+=int(g['count'].sum());dtmax=min(dtmax,cunge.common_clock(g,m['forcing_step_s'])[0])
   if i%100==0:print(ex,i,'/',len(inv),flush=True)
  nsub=int(np.ceil(m['forcing_step_s']/dtmax));dt=m['forcing_step_s']/nsub;count=len(inv);actual=None;engine_h=None;wall_h=None
 row=dict(example=ex,unique_states=count,report_steps=nt,sections_total=sections,nsub=nsub,dt_s=dt,qout_gib=nt*sections*8/1024**3,prep_plus_output_gib=nt*sections*16/1024**3,native_cell_steps=nt*sections*nsub,actual_gib=actual/1024**3 if actual else None,recorded_engine_hours=engine_h,tail_and_two_worker_wall_hours=wall_h)
 rows.append(row);Path('/private/tmp/cunge_resource_estimate.json').write_text(json.dumps(rows,indent=2));print('DONE',ex,row,flush=True)
