"""Save the same-grid research model alongside the expanded-grid artifact."""
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from tune_nine_inputs import DEST, SOURCE, COLUMNS, fit, b, sha, save


def main():
    completion=json.loads((DEST/'completion.json').read_text())
    assert completion['complete']
    trials=[json.loads((DEST/'inner/all_data'/f'{i:02d}.json').read_text()) for i in range(4)]
    assert all(t['fingerprint']==completion['fingerprint'] for t in trials)
    best=min(trials,key=lambda t:(t['score'],t['index']))
    d=pd.read_parquet(SOURCE/'input.parquet')
    artifact=fit(d,best['params'],4)
    artifact['fingerprint']=completion['fingerprint']
    path=DEST/'research_model_same_grid.joblib'
    joblib.dump(artifact,path,compress=3)
    restored=joblib.load(path)
    np.testing.assert_allclose(b.forest.predict(artifact,d),b.forest.predict(restored,d),rtol=1e-12,atol=1e-12)
    save(DEST/'research_model_same_grid.json',dict(selected=best,columns=COLUMNS,sha256=sha(path),
         training_rows=len(d),source_sha256={str(SOURCE/'input.parquet'):sha(SOURCE/'input.parquet'),
         str(Path(__file__).resolve()):sha(__file__)},
         warning='Separate research artifact for same-grid tuning procedure. Use nested same_grid scores; all-data selection score is not a test score. No production replacement.'))
    print(best['name'],best['params'])


if __name__=='__main__':
    main()
