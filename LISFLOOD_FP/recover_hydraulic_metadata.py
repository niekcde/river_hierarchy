"""Recover a completed diagnostic after a NumPy-bool JSON serialization error.
Require exact agreement with the saved hydrograph before accepting kernel output.
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import broad_study as b

def recover(folder,ex,state,bed_source,partition):
    from investigate_junctions37 import partition_widths
    folder=Path(folder);wave=pd.read_parquet(folder/'wave.parquet');found=[]
    for p in folder.parent.glob('_output_*.npz'):
        with np.load(p) as d:
            if len(d['wave'])!=len(wave)+1:continue
            means=np.diff(d['wave'][:,4])/60
            if np.array_equal(means,wave.q_outlet_cms.to_numpy()):found.append((p,d['wave'].copy(),d['audit'].copy()))
    if len(found)!=1:raise RuntimeError(f'Expected one exact completed kernel match for {folder}, found {len(found)}')
    p,numerical,audit=found[0];audit=dict(zip(b.v.AUDIT,map(float,audit)));forcing=b.forcing(ex);origin=forcing.interval_start_utc.iloc[0];event=forcing[forcing.phase.eq('input_event')];t=(wave.time_utc-origin).dt.total_seconds().to_numpy();baseline=np.interp(t,[(event.interval_start_utc.iloc[0]-origin).total_seconds(),(event.interval_end_utc.iloc[-1]-origin).total_seconds()],[event.forcing_cms.iloc[0],event.forcing_cms.iloc[-1]])
    cal=json.loads((b.OUT/ex/'calibration.json').read_text());f=pd.read_parquet(bed_source) if bed_source else b.frame(ex,state);cells=b.v.grid(f,'max100');m=b.build(cells,cal['selected_n']);changes=partition_widths(cells,m) if partition else []
    relative=float(audit['maximum_mass_residual_m3']/numerical[-1,3])
    r=dict(example=ex,state=state,valid=bool(audit['limited_face_steps']==0 and audit['clipping_volume_m3']<1e-6 and relative<1e-9),n=cal['selected_n'],bed_source=str(bed_source) if bed_source else None,bed_source_sha256=b.sha(bed_source) if bed_source else None,partitioned_junction=partition,relative_mass_residual=relative,changes=changes,seconds=None,folder=str(folder),metadata_recovered_from=str(p),kernel_sha256=b.sha(p),recovery_check='Exact interval-mean hydrograph match; no hydraulic rerun or output alteration.',**b.v.metrics(t,wave.q_outlet_cms.to_numpy(),baseline),**audit)
    b.save(folder/'result.json',r);print('RECOVERED COMPLETED NUMERICAL RESULT',ex,state,flush=True);return r
