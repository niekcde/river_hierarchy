"""Stage or publish the approved master calibration/run-folder addition."""
from pathlib import Path
import sys,json,shutil,argparse
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'RAPID/src')]
import pandas as pd
from rapid_tools.run_manifest import digest,load_run_manifest
MASTER=Path('/Volumes/PhD/river_hierarchy/network_variants/output')
STAGE=ROOT/'RAPID/assets/master_run_configuration_20261009'
BUNDLE=ROOT/'RAPID/outputs/bounded_beta_calibration_20261009'
CAL=Path('calibration/bounded_beta_20261009')
RUN=Path('rapid_runs/common_grid_20261009')

def save(path,data):
 path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(data,indent=2)+'\n')

def stage():
 STAGE.mkdir(parents=True,exist_ok=False)
 for name in ['reviewed_master_manifest.json','hydraulic_inventory.csv']:
  shutil.copy2(MASTER/name,STAGE/name)
 source=Path(json.loads((BUNDLE/'protocol.json').read_text())['source_directory'])
 table=pd.read_csv(source/'baseline/RAPID/outputs/cunge_calibration_reviewed_20261006/comparison.csv',dtype={'example':str})
 parameters={};inputs={};casefiles={}
 def copy(src,rel):
  dest=STAGE/rel;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dest);inputs[str(rel)]=digest(dest);return str(rel)
 for p in sorted(BUNDLE.glob('*/result.json')):
  ex=p.parent.name;r=json.loads(p.read_text());row=table[table.example.eq(ex)&table.method.eq('cunge')].iloc[0]
  folder=source/'baseline'/Path(row.output_dir).relative_to(ROOT)
  meta=next(v for v in json.loads((source/ex/'60s.json').read_text()) if v['method']=='cunge')
  fixed=table[table.example.eq(ex)&table.method.eq('fixed_x')].iloc[0]
  parameters[ex]=dict(cunge=dict(n_manning=r['n_manning'],beta=r['beta'],subdivision_multiplier=r.get('subdivision_multiplier',1),native_dt_cap_seconds=r.get('native_dt_s',meta['dt_s'])),
   single_x=dict(x=.1,n_manning=float(fixed.n_manning)),initial_discharge_cms=meta['initial_cms'],reference_discharge_cms=json.loads((folder/'case.json').read_text())['reference_discharge_cms'])
  casefiles[ex]={}
  for name in ['observed_upstream.parquet','observed_downstream.parquet','frozen_grid.parquet','case.json']:
   casefiles[ex][name]=copy(folder/name,CAL/ex/name)
  for name in ['result.json','wave.parquet']:casefiles[ex][name]=copy(BUNDLE/ex/name,CAL/ex/name)
  for name in ['60s.parquet','60s.json']:casefiles[ex][name]=copy(source/ex/name,CAL/ex/name)
 for name in ['timing_targets.json','height_targets.json','comparison.csv','validation.json','REPORT.txt','protocol.json']:
  copy(BUNDLE/name,CAL/name)
 save(STAGE/CAL/'parameters.json',parameters);inputs[str(CAL/'parameters.json')]=digest(STAGE/CAL/'parameters.json')
 copy(ROOT/'RAPID/outputs/calibration_review/15_network_overview.html',RUN/'reports/15_network_overview.html')
 # The report is a convenience artifact, not a routing input.
 inputs.pop(str(RUN/'reports/15_network_overview.html'))
 for name in ['reviewed_master_manifest.json','hydraulic_inventory.csv']:inputs[name]=digest(STAGE/name)
 release=json.loads((STAGE/'reviewed_master_manifest.json').read_text())['release']
 manifest=dict(schema_version='rapid-master-run-v1',run_id=RUN.name,master_root=str(MASTER),status='configured_not_prepared',geometry_release=release,
  geometry_manifest='reviewed_master_manifest.json',hydraulic_inventory='hydraulic_inventory.csv',calibration_parameters=str(CAL/'parameters.json'),calibration_cases=casefiles,
  examples=sorted(parameters),methods=['cunge'],exclusions=dict(examples=['11'],inventory_statuses=['excluded_topology_review','excluded_duplicate']),
  time_policy=dict(report_interval_seconds=60,alignment='UTC midnight',labels='interval midpoint',gauge_reconstruction='piecewise-linear exact interval integral',
   forcing='held interval mean',outlet_averaging='linear_state_interval_mean',native_clock='shared across compared states/methods; exact divisor; bound evaluated after beta and subdivision'),
  initialization=dict(mode='equilibrium_at_original_first_forcing',extra_antecedent=False),
  parameter_inheritance='Base n/beta inherited by children; no child calibration; preserve calibrated refinement policy',
  quality_policy=dict(diagnostic_gap_fill=True,observation_clock_unresolved=['21','22'],hourly_crest_sampling=['55'],example46_antecedent_sensitivity_minutes=[14,15],final_scientific_acceptance=False,
   required_preflight=['active inventory and exclusion checks','base hydraulic/calibration consistency','changed-child and shared-clock checks','shared tail settling before volume/duration metrics']),
  outputs=dict(prep=str(RUN/'prep'),results=str(RUN/'results'),reports=str(RUN/'reports'),prep_receipt=str(RUN/'prep/completion.json')),
  input_sha256=inputs)
 save(STAGE/RUN/'run_manifest.json',manifest)
 for name in ['prep','results']:(STAGE/RUN/name).mkdir(parents=True,exist_ok=True)
 save(STAGE/'calibration/current.json',dict(bundle=str(CAL),parameters=str(CAL/'parameters.json')))
 save(STAGE/'rapid_runs/current.json',dict(run_manifest=str(RUN/'run_manifest.json'),sha256=digest(STAGE/RUN/'run_manifest.json'),status='configured_not_prepared'))
 (STAGE/RUN/'README.txt').write_text('Shared configuration for PREP and RUN: run_manifest.json.\nStatus: configured; no preparation or routing performed.\nThe 15-panel overview is in reports/.\nPREP and RUN must load this manifest using rapid_tools.run_manifest.load_run_manifest.\nExisting legacy runners are not automatically redirected; their end-to-end adapter remains to be connected.\nRUN requires a checksum-matched completed PREP receipt.\n')
 load_run_manifest(STAGE/RUN/'run_manifest.json',master_root=STAGE)
 try:load_run_manifest(STAGE/RUN/'run_manifest.json',stage='run',master_root=STAGE)
 except ValueError as error:assert 'PREP is not complete' in str(error)
 else:raise AssertionError('Unprepared run accepted')
 files={str(p.relative_to(STAGE)):digest(p) for folder in ['calibration','rapid_runs'] for p in (STAGE/folder).rglob('*') if p.is_file()}
 save(STAGE/'publication_plan.json',dict(master_root=str(MASTER),files=files,expected_geometry_sha256=digest(MASTER/'reviewed_master_manifest.json'),expected_inventory_sha256=digest(MASTER/'hydraulic_inventory.csv')))
 print('Staged',len(files),'files; 15 calibrations validated; unprepared RUN correctly rejected.',flush=True)

def publish():
 plan=json.loads((STAGE/'publication_plan.json').read_text())
 assert digest(MASTER/'reviewed_master_manifest.json')==plan['expected_geometry_sha256']
 assert digest(MASTER/'hydraulic_inventory.csv')==plan['expected_inventory_sha256']
 for rel,sha in plan['files'].items():
  assert digest(STAGE/rel)==sha
  dest=MASTER/rel
  if dest.exists() and digest(dest)!=sha:raise FileExistsError('Refusing to replace '+str(dest))
 for rel,sha in plan['files'].items():
  dest=MASTER/rel;dest.parent.mkdir(parents=True,exist_ok=True)
  if not dest.exists():shutil.copy2(STAGE/rel,dest)
  assert digest(dest)==sha
 for name in ['prep','results']:(MASTER/RUN/name).mkdir(parents=True,exist_ok=True)
 load_run_manifest(MASTER/RUN/'run_manifest.json')
 save(ROOT/'RAPID/config/production_run.json',dict(manifest=str(MASTER/RUN/'run_manifest.json'),sha256=digest(MASTER/RUN/'run_manifest.json'),status='configured_not_prepared'))
 save(STAGE/'publication_receipt.json',dict(published=True,files=len(plan['files']),manifest=str(MASTER/RUN/'run_manifest.json'),prep_executed=False,run_executed=False))
 print('Published and verified:',MASTER/RUN/'run_manifest.json',flush=True)

if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('action',choices=['stage','publish']);args=parser.parse_args();stage() if args.action=='stage' else publish()
