"""Postprocess saved interval-average synthetic hydrographs; no routing changes."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
from scipy.signal import find_peaks


def volume_shape(time_seconds, excess, dt):
    """Moments/quantiles of nonnegative, piecewise-constant interval volumes.

    Timestamps are interval starts. Include dt²/12 within-interval variance;
    interpolate quantiles within their mass-containing intervals.
    """
    t=np.asarray(time_seconds,float);q=np.asarray(excess,float)
    if len(t)!=len(q) or not len(t) or not np.isfinite(q).all() or not np.isfinite(t).all():
        raise ValueError('Invalid event arrays')
    if dt<=0 or (len(t)>1 and not np.allclose(np.diff(t),dt)):
        raise ValueError('Expected a uniform positive interval duration')
    tolerance=1e-8*max(1.,float(q.max()))
    if q.min() < -tolerance:
        raise ValueError('Materially negative event excess')
    mass=np.maximum(q,0)*dt
    if mass.sum()<=0: raise ValueError('No event volume')
    w=mass/mass.sum();mid=t+dt/2
    mean=float(np.dot(w,mid))
    sd=float(np.sqrt(np.dot(w,(mid-mean)**2)+dt**2/12))
    cumulative=np.cumsum(mass)
    def quantile(p):
        target=p*cumulative[-1];i=min(int(np.searchsorted(cumulative,target)),len(t)-1)
        before=cumulative[i-1] if i else 0.
        return float(t[i]+dt*(target-before)/mass[i])
    return dict(event_volume_m3=float(mass.sum()),centroid_seconds=mean,
                temporal_sd_seconds=sd,duration_t01_t99_seconds=quantile(.99)-quantile(.01))


def load_run_shapes(run, results, manifest, out):
    """Cache once per unique run; invalidate on inputs or implementation change."""
    run=Path(run);out=Path(out)
    ids=sorted(set(results.parent_run_id)|set(results.child_run_id))
    signature=hashlib.sha256(Path(__file__).read_bytes())
    signature.update(json.dumps(manifest,sort_keys=True).encode())
    for rid in ids:
        for filename in ('metrics.json','hydrograph.parquet'):
            p=run/'runs'/rid/filename;s=p.stat()
            signature.update(f'{rid}/{filename}:{s.st_size}:{s.st_mtime_ns}'.encode())
    fingerprint=signature.hexdigest();meta=out/'event_shape_cache.json';csv=out/'event_shape_runs.csv'
    if meta.exists() and csv.exists() and json.loads(meta.read_text()).get('fingerprint')==fingerprint:
        cached=pd.read_csv(csv)
        if set(cached.run_id)==set(ids) and not cached.run_id.duplicated().any():return cached
    base=manifest['config']['routing']['baseflow_cms']
    amplitude=manifest['config']['routing']['peak_cms']-base
    rows=[]
    for index,rid in enumerate(ids):
        folder=run/'runs'/rid;metrics=json.loads((folder/'metrics.json').read_text())
        if max(abs(metrics['volume_error_fraction']),metrics['pre_event_relative_error'],metrics['final_relative_error'])>1e-4:
            raise ValueError(f'Incomplete/unconserved event: {rid}')
        h=pd.read_parquet(folder/'hydrograph.parquet');dt=metrics['dt_seconds']
        onset=round(metrics['buffer_hours']*3600/dt)*dt
        event=h.time_seconds.to_numpy()>=onset
        t=h.time_seconds.to_numpy()[event]-onset;q=h.outlet_cms.to_numpy()[event]-base
        row=volume_shape(t,q,dt)
        if not np.isclose(row['event_volume_m3'],metrics['event_volume_out_m3'],rtol=1e-7):
            raise ValueError(f'Event volume mismatch: {rid}')
        peaks,_=find_peaks(q,prominence=.01*amplitude)
        rows.append(dict(run_id=rid,peak_excess_cms=float(q.max()),screened_peak_count=len(peaks),**row))
        if (index+1)%1000==0:print(f'Event shapes: {index+1:,}/{len(ids):,}')
    frame=pd.DataFrame(rows);frame.to_csv(csv,index=False)
    meta.write_text(json.dumps(dict(fingerprint=fingerprint,n_runs=len(ids),
        background_cms=base,peak_prominence_fraction_of_input_excess=.01,
        interval_model='piecewise constant; within-interval variance dt^2/12'),indent=2))
    return frame
