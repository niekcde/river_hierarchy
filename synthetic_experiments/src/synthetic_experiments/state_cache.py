"""Process-safe cache of native-derived metrics and reported hydrographs."""
from pathlib import Path
import fcntl
import json
import os
import tempfile
import numpy as np


def cached_state(root, key, calculate):
    folder=Path(root)/key[:2];folder.mkdir(parents=True,exist_ok=True)
    target=folder/(key+'.npz')
    # flock releases automatically on worker exit, including interruption.
    with (folder/(key+'.lock')).open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        if target.exists():
            with np.load(target,allow_pickle=False) as data:
                return data['reported'].copy(),json.loads(str(data['metrics'].item()))
        wave,metrics=calculate()
        fd,name=tempfile.mkstemp(dir=folder,suffix='.tmp')
        try:
            with os.fdopen(fd,'wb') as stream:
                np.savez_compressed(stream,reported=wave,metrics=json.dumps(metrics,allow_nan=False))
            os.replace(name,target)
        finally:
            if os.path.exists(name):os.unlink(name)
        return wave,metrics
