"""Full base Cunge engine verification after reviewed-target calibration."""
from pathlib import Path
import sys,json,time
from concurrent.futures import ProcessPoolExecutor
import numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT),str(ROOT/'RAPID/src'),str(Path(__file__).parent)]
import cunge_calibration_reviewed as cal
from rapid_tools.engine import run_prepared_state

def verify(ex):
    r=pd.read_csv(cal.OUT/'comparison.csv',dtype={'example':str});r=r[r.example.eq(ex)&r.method.eq('cunge')].iloc[0]
    folder=Path(r.output_dir);prep=folder/'cunge_prep';m=json.loads((prep/'rapid_prep_manifest.json').read_text())
    dest=folder/'cunge_run';dest.mkdir(exist_ok=True);path=dest/'Qout.nc';started=time.monotonic()
    _,q=run_prepared_state(prep,output_path=path,return_qout=True)
    outlet=q[:,np.array(m['outlet_reach_ids'])-1].sum(axis=1);wave=pd.read_parquet(folder/'cunge_best_wave.parquet');error=float(abs(outlet-wave.q_outlet_cms.to_numpy()).max());audit=json.loads(path.with_suffix('.audit.json').read_text())
    assert error<=1e-8*max(1,float(np.ptp(wave.q_inflow_cms)))
    assert audit['minimum_substep_flow_cms']>=-1e-9
    t=pd.to_datetime(wave.time_utc,utc=True);peak=t.iloc[int(outlet.argmax())]
    target=json.loads((folder/'case.json').read_text())['target'];score=cal.timing_fields(peak,target)
    assert abs(score['error_h']-r.error_h)<1e-9
    result=dict(example=ex,method='cunge',n_manning=float(r.n_manning),max_engine_cascade_error_cms=error,runtime_seconds=time.monotonic()-started,**score,**audit)
    (folder/'full_engine_verification.json').write_text(json.dumps(result,indent=2));print('verified',ex,error,flush=True);return result

if __name__=='__main__':
    c=pd.read_csv(cal.OUT/'comparison.csv',dtype={'example':str});assert len(c)==32
    records=[]
    with ProcessPoolExecutor(max_workers=2) as pool:
        for record in pool.map(verify,sorted(c.example.unique())):
            records.append(record);pd.DataFrame(records).to_csv(cal.OUT/'prepared_verification.csv',index=False)
