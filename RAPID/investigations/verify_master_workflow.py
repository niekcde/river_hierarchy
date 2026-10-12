"""Run isolated real-data PREP/RUN smoke checks without writing master outputs."""
from pathlib import Path
import sys,json,shutil
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT),str(ROOT/'RAPID/src')]
from rapid_tools.prep import prepare_master_run
from rapid_tools.engine import run_master_run
from rapid_tools.run_manifest import digest
from rapid_tools.master_workflow import inventory_summary
import pandas as pd
OUT=ROOT/'RAPID/outputs/master_workflow_verification_20261009'

def main():
 ptr=json.loads((ROOT/'RAPID/config/production_run.json').read_text());source=Path(ptr['manifest']);m=json.loads(source.read_text());master=Path(m['master_root'])
 OUT.mkdir(parents=True,exist_ok=False)
 for rel,sha in m['input_sha256'].items():
  dest=OUT/rel;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(master/rel,dest);assert digest(dest)==sha
 m['master_root']=str(OUT);m['run_id']='isolated_workflow_smoke_20261009'
 path=OUT/'run_manifest.json';path.write_text(json.dumps(m,indent=2))
 inv=pd.read_csv(OUT/m['hydraulic_inventory'],dtype={'example':str});summary=inventory_summary(inv,m['examples']);summary.to_csv(OUT/'inventory_overview.csv')
 receipt=prepare_master_run(path,examples=['03','46'],max_states=2)
 results=run_master_run(path,workers=2)
 assert receipt['scope']['full_batch'] is False and len(results)==8
 (OUT/'verification.json').write_text(json.dumps(dict(passed=True,examples=['03','46'],unique_states=4,method_jobs=8,prep_seconds=receipt['runtime_seconds'],routing_seconds_sum=sum(r['runtime_seconds'] for r in results),master_unmodified=True),indent=2))
 print(summary.to_string());print('PASS master workflow smoke',flush=True)
if __name__=='__main__':main()
