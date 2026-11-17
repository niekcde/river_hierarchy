"""Finalize saved fits with an explicit local report import.

The legacy fitting imports add other investigation directories to sys.path;
an unqualified `import summarize` can therefore resolve to a different module.
Keep the frozen fitting source and fingerprint intact and finalize by path.
"""
from pathlib import Path
import importlib.util
import json
import time
import fcntl
import run_comparison as run

def main():
    with (run.OUT/'run.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        design,fp=run.prepare()
        path=Path(__file__).with_name('summarize.py')
        spec=importlib.util.spec_from_file_location('streamlined_local_report',path)
        report=importlib.util.module_from_spec(spec);spec.loader.exec_module(report)
        report.build()
        for p,h in design['sources'].items():assert run.sha(p)==h,p
        progress=json.loads((run.OUT/'progress.json').read_text())
        run.save(run.OUT/'completion.json',dict(complete=True,fingerprint=fp,jobs=design['jobs'],
            fitting_elapsed_seconds=progress['elapsed_seconds'],
            finalizer_sha256=run.sha(__file__),reporting_repair='Explicit report module path; no fit changes or reruns.',
            files={str(run.OUT/x):run.sha(run.OUT/x) for x in ['summary.csv','report.md','predictions.parquet','comparison_matrix.png']}))
        run.save(run.OUT/'progress.json',dict(status='complete',completed_jobs=design['jobs'],total_jobs=design['jobs'],
            elapsed_seconds=progress['elapsed_seconds'],completed_at=time.strftime('%Y-%m-%dT%H:%M:%S%z')))

if __name__=='__main__':main()
