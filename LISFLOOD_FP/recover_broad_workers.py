"""Recover failed calibration workers after the missing-metric serialization fix."""
import re,time
import broad_study as b
handled={'13'} # Already recovering in a dedicated process.
start=time.monotonic()
while not (b.OUT/'stages_complete.json').exists():
 p=b.ROOT/'LISFLOOD_FP/results/broad_calibration.log'
 failures=set(re.findall(r'WORKER FAILED (\d{2}) ',p.read_text()))
 for ex in sorted(failures-handled):
  print('RECOVER',ex,flush=True);r=b.calibrate(ex);handled.add(ex);print('RECOVERED',ex,r['status'],flush=True)
 if time.monotonic()-start>86400:raise RuntimeError('Recovery monitor exceeded one day')
 time.sleep(30)
