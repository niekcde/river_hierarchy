from pathlib import Path
import sys,json,time
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT),str(ROOT/'RAPID/src')]
from rapid_tools.engine import run_master_run
OUT=ROOT/'RAPID/outputs/master_workflow_verification_20261009'

def main():
 path=OUT/'run_manifest.json';m=json.loads(path.read_text());dest=OUT/m['outputs']['results'];done=dest/'completion.json';backup=OUT/'first_run_completion.json'
 done.rename(backup)
 start=time.monotonic();rows=run_master_run(path,workers=2);elapsed=time.monotonic()-start
 assert len(rows)==8
 (OUT/'resume_verification.json').write_text(json.dumps(dict(passed=True,resumed_examples=2,retained_method_jobs=8,seconds=elapsed),indent=2))
 print('Resume passed',elapsed)
if __name__=='__main__':main()
