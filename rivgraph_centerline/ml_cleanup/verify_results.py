"""Independent artifact checks after model evaluation has completed."""
import hashlib,json
from pathlib import Path
from collections import Counter
import numpy as np
from core import OUT,CROP,load_tile,dump

def main():
    rec=json.loads((OUT/'tiles.json').read_text());lookup={r['file']:r for r in rec};splits=json.loads((OUT/'splits.json').read_text())
    checked=0;hashes={}
    for s in splits:
        folder=OUT/f"fold_{s['fold']}";assert(folder/'DONE.json').exists()
        rows=json.loads((folder/'metrics.json').read_text());sel=json.loads((folder/'selection.json').read_text())
        expected={r['file'] for r in rec if r['river'] in s['test']}
        assert len(rows)==len(expected)*6
        assert all(n==1 for n in Counter((r['method'],r['tile']) for r in rows).values())
        for method in {r['method'] for r in rows}:assert {r['tile'] for r in rows if r['method']==method}==expected
        for method,data in sel.items():
            assert data['threshold']==max(data['validation'],key=lambda x:x['macro_iou'])['threshold']
        for r in rows:
            z=load_tile(lookup[r['tile']]);y=z['target'][CROP];b=z['base'][CROP]
            with np.load(folder/'predictions'/r['method']/(Path(r['tile']).stem+'.npz')) as f:
                p=f['prediction'];prob=f['probability']
                assert p.shape==y.shape and p.dtype==bool
                assert np.isfinite(prob).all() and prob.min()>=0 and prob.max()<=1
                assert r['tp']==int((p&y).sum()) and r['fp']==int((p&~y).sum()) and r['fn']==int((~p&y).sum())
                assert r['damage']==int(((p!=b)&(b==y)).sum())
                assert r['river'] in s['test'] and r['river'] not in s['train']+s['validation']
            checked+=1
        for p in folder.iterdir():
            if p.suffix in ['.pt','.joblib']:hashes[str(p.relative_to(OUT))]=hashlib.sha256(p.read_bytes()).hexdigest()
    dump(OUT/'verification.json',dict(predictions_verified=checked,all_four_folds_complete=True,model_hashes=hashes))
    print('Verified',checked,'held-out predictions against saved metric counts and split membership')
if __name__=='__main__':main()
