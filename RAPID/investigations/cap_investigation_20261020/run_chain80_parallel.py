"""Execution-only launcher: six independent candidates with scheduled tree parallelism.

The scientific script/checkpoint fingerprint is unchanged. This does not alter
seeds, settings, splits, training data or the order used to rank completed fits.
"""
import concurrent.futures
import hashlib
import json
import runpy
import sys
import threading
from pathlib import Path

HERE=Path(__file__).resolve().parent
OriginalExecutor=concurrent.futures.ThreadPoolExecutor
class SixWorkers(OriginalExecutor):
    def __init__(self,max_workers=None,*args,**kwargs):
        super().__init__(max_workers=6,*args,**kwargs)

if __name__=='__main__':
    from chain_augmentation import b
    original_fit=b.fit
    def scheduled_fit(train,params,jobs=4):
        return original_fit(train,params,jobs=6 if threading.current_thread() is threading.main_thread() else 2)
    b.fit=scheduled_fit
    concurrent.futures.ThreadPoolExecutor=SixWorkers
    script=HERE/'chain80_tuning.py'
    out=HERE.parents[1]/'outputs/uncapped_event_window_20261020/chain80_tuning'
    out.mkdir(parents=True,exist_ok=True)
    (out/'execution.json').write_text(json.dumps(dict(workers=6,forest_threads_per_concurrent_fit=2,forest_threads_per_serial_fit=6,launcher_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),scientific_script_sha256=hashlib.sha256(script.read_bytes()).hexdigest(),note='Execution-only concurrency override; reuse identical scientific checkpoints.'),indent=2)+'\n')
    sys.argv=[str(script),'--run','--tune-only']
    runpy.run_path(str(script),run_name='__main__')
