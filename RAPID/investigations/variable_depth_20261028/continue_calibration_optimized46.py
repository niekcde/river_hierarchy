"""Resume cached v2 examples 22/41, then use the checked optimised solver for 46."""
import os
for key in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS']:os.environ[key]='1'
import fcntl,json,subprocess,sys,time
from pathlib import Path
from prototype import DEST,HERE,r

def main():
 folder=DEST/'calibration_optimized46_continuation';folder.mkdir(exist_ok=True)
 with (folder/'continuation.lock').open('a+') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  evidence=json.loads((DEST/'optimization46_benchmark.json').read_text())
  for source,sha in evidence['source_hashes'].items():assert r.digest(HERE/source)==sha
  assert evidence['max_wave_difference_cms']<1e-6
  status=dict(started_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),stages=[])
  stages=[('remaining_v2','calibrate_references_v2.py',['22','41'],'reference_calibration_v2'),
          ('optimized46','calibrate_references_optimized.py',['46'],'reference_calibration_optimized')]
  for label,script,examples,output in stages:
   dest=DEST/output;dest.mkdir(exist_ok=True)
   status['active_stage']=label;r.write_json(folder/'status.json',status)
   with (dest/'run.log').open('a') as log:
    log.write('\nCONTINUATION '+label+' '+time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())+'\n');log.flush()
    process=subprocess.Popen([sys.executable,'-u',str(HERE/script),'--examples',*examples],stdout=log,stderr=subprocess.STDOUT)
    status['child_pid']=process.pid;r.write_json(folder/'status.json',status)
    code=process.wait()
   status['stages'].append(dict(stage=label,returncode=code))
   if code:
    status['active_stage']='stopped_on_process_error';r.write_json(folder/'status.json',status)
    raise RuntimeError(f'{label} exited {code}; inspect its log')
  status['active_stage']='finished';status['child_pid']=None;r.write_json(folder/'status.json',status)

if __name__=='__main__':main()
