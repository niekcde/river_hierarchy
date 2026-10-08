"""Parent-only Cunge alpha=2 calibration and numerical/observational checks.

Run with the project's scientific Python environment. All new products are
separate from the historical calibration and state directories. No children
are calibrated or routed by this investigation.
"""
from pathlib import Path
import argparse
import hashlib
import json
import math
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT)); sys.path.insert(0,str(ROOT/'RAPID/src'))
sys.path.insert(0,str(Path(__file__).parent))
from rapid_tools import cunge
from rapid_tools.engine import run_prepared_state
import travel_time_calibration as original

OUT = ROOT/'RAPID/outputs/cunge_calibration_alpha2'
BOUNDS = (.01,.1)
METHODS = ['fixed_x','cunge']


def digest(paths):
    h=hashlib.sha256()
    for p in paths:h.update(Path(p).read_bytes())
    return h


def load_case(ex):
    schedule=original.read_schedule();w=schedule.loc[ex]
    old=pd.read_csv(ROOT/'RAPID/outputs/travel_time_calibration_refined/refined_candidates.csv',dtype={'example':str}).set_index('example').loc[ex]
    provenance=ROOT/'RAPID/validation/cunge_x_20261005/provenance.json'
    src=next(s for s in json.loads(provenance.read_text())['sources'] if s['example']==ex)
    frame=pd.read_parquet(src['frame'])
    reference=json.loads(Path(src['policy_manifest']).read_text())['reference_kb_summary']
    wave=pd.read_parquet(old.hydrograph_path);times=pd.to_datetime(wave.time_utc,utc=True)
    event=wave.loc[(times>=w.first_sample)&(times<=w.last_sample),'q_inflow_cms'].to_numpy()
    step=float(old.forcing_step_s)
    np.testing.assert_allclose([event[0],event[-1]],[w.q_initial,w.q_final])
    assert len(event)>2 and len(event)==round((w.last_sample-w.first_sample).total_seconds()/step)+1
    # Read the workbook directly, avoiding pandas' optional Excel-engine
    # version requirement while using the exact same reviewed station table.
    from openpyxl import load_workbook
    from rapid_tools.analysis import GAUGE_QUALITY_XLSX
    book=load_workbook(GAUGE_QUALITY_XLSX,read_only=True,data_only=True)
    try:
        rows=list(book['Hydrograph_quality'].values)
        quality=pd.DataFrame(rows[1:],columns=rows[0])
    finally:book.close()
    target=original.observed_target(ex,w,quality=quality)
    if not target['target_usable']:raise ValueError(f'Unusable target for {ex}: {target["target_notes"]}')
    grid=cunge.freeze_grid(frame,reference['reference_discharge_cms'],n_bounds=BOUNDS,alpha=2.)
    dt,nsub=cunge.common_clock(grid,step)
    source_paths=[src['frame'],src['policy_manifest'],old.hydrograph_path,original.WORKBOOK,
                  Path(__file__),Path(cunge.__file__),ROOT/'RAPID/src/rapid_tools/engine.py',
                  ROOT/'RAPID/src/rapid_tools/k_values.py',Path(original.__file__)]
    h=digest(source_paths)
    for series in target['series'].values():h.update(pd.util.hash_pandas_object(series,index=False).values.tobytes())
    h.update(json.dumps(dict(bounds=BOUNDS,alpha=2,dt=dt,step=step)).encode())
    signature=h.hexdigest();dest=OUT/ex/signature[:16];dest.mkdir(parents=True,exist_ok=True)
    grid.to_parquet(dest/'frozen_grid.parquet',index=False)
    for name,series in target['series'].items():series.to_parquet(dest/f'observed_{name}.parquet',index=False)
    manifest=dict(example=ex,signature=signature,alpha=2,n_bounds=BOUNDS,routing_dt_s=dt,
                  forcing_step_s=step,nsub=nsub,original_links=len(grid),numerical_sections=int(grid['count'].sum()),
                  reference_discharge_cms=reference['reference_discharge_cms'],
                  source_hashes={str(p):hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in source_paths},
                  target={k:str(v) if isinstance(v,pd.Timestamp) else v for k,v in target.items() if k!='series'},
                  first_sample=str(w.first_sample),last_sample=str(w.last_sample),
                  scope='Parent only. Constant-reference MC; fixed grid and clock across n and methods; no lateral inflow; upstream two-time boundary.')
    (dest/'case.json').write_text(json.dumps(manifest,indent=2))
    return dict(ex=ex,window=w,old=old,target=target,grid=grid,event=event,step=step,dt=dt,nsub=nsub,dest=dest,manifest=manifest)


def shape_metrics(case,t,q,flow):
    w=case['window'];step=case['step'];g=case['target']['series']['downstream']
    obs_t=(pd.to_datetime(g.time,utc=True)-w.first_sample).dt.total_seconds().to_numpy()
    mask=(obs_t>=0)&(obs_t<=t[-1]);obs=g.discharge_cms.to_numpy()[mask];pred=np.interp(obs_t[mask],t,q)
    def normalized(y):return (y-y.min())/max(np.ptp(y),1e-12)
    def span(y,tt):
        threshold=y[0]+.5*(y.max()-y[0]);ids=np.flatnonzero(y>=threshold)
        return float((tt[ids[-1]]-tt[ids[0]])/3600) if len(ids)>1 else 0.
    return dict(normalized_shape_rmse=float(np.sqrt(np.mean((normalized(pred)-normalized(obs))**2))),
                raw_rmse_cms=float(np.sqrt(np.mean((pred-obs)**2))),
                model_to_observed_peak_ratio=float(pred.max()/obs.max()),
                model_peak_excess_ratio=float((q.max()-flow[0])/max(flow.max()-flow[0],1e-12)),
                model_half_height_span_h=span(q,t),observed_half_height_span_h=span(obs,obs_t[mask]),
                shape_scope='Same calibration event, observed UTC; no peak alignment; diagnostic only')


def trial(case,method,n,*,clock_factor=1,grid_factor=1,save_wave=False):
    n=float(n);dest=case['dest'];name=f'{method}_n{n:.12f}_dt{clock_factor}_grid{grid_factor}'
    path=dest/(name+'.json')
    if path.exists() and not save_wave:return json.loads(path.read_text()),None,None
    grid=case['grid'].copy()
    if grid_factor!=1:
        grid['count']*=grid_factor;grid['rapid_effective_length_m']=grid.link_length_m/grid['count']
    f=cunge.parameters(grid,n,method=method);dt=case['dt']/clock_factor;nsub=case['nsub']*clock_factor
    from rapid_tools.k_values import compute_dt_bounds
    upper=compute_dt_bounds(f.rapid_k,x_value=f.rapid_x,coefficient_policy='signed_inflow')[1]
    if dt>upper:
        extra=int(math.ceil(dt/(.99*upper)));nsub*=extra;dt=case['step']/nsub
    event=case['event'];w=case['window'];step=case['step']
    count=max(int(round((w.simulation_end-w.last_sample).total_seconds()/step)),int(math.ceil(3*86400/step)))
    started=time.perf_counter();settled=False
    for extension in range(7):
        flow=np.r_[event,np.full(count,event[-1])]
        q,audit=cunge.route(f,flow,nsub,dt);t=np.arange(len(flow))*step
        amplitude=max(float(q.max()-max(flow[0],flow[-1])),1e-12)
        last_change=(len(event)-1)*step
        outside=np.flatnonzero((t>=last_change)&(abs(q-flow[-1])>.001*amplitude))
        j=int(outside[-1]+1) if len(outside) else len(event)-1
        settled=j<len(t) and t[j]+2*86400<=t[-1]
        if settled:break
        count=max(count*2,count+int(math.ceil(2*86400/step)))
    peak=original.peak_summary(w.first_sample+pd.to_timedelta(t,unit='s'),q)
    lag=(int(q.argmax())-int(event.argmax()))*step/3600
    target=case['target'];error=lag-target['observed_lag_h']
    valid=bool(settled and not peak['boundary'] and audit['minimum_native_flow_cms']>=-1e-10*max(1,float(q.max())))
    r=dict(example=case['ex'],method=method,n_manning=n,lag_h=lag,error_h=error,abs_error_h=abs(error),
           valid=valid,tail_settled=bool(settled),tail_extensions=extension,simulation_days=t[-1]/86400,
           peak_boundary=bool(peak['boundary']),peak_tied_span_h=peak['tied_span_h'],peak_cms=float(q.max()),
           observed_lag_h=target['observed_lag_h'],timing_resolution_h=target['timing_resolution_h'],
           within_sampling=bool(valid and abs(error)<=target['timing_resolution_h']),
           dt_s=dt,nsub=nsub,forcing_step_s=step,grid_factor=grid_factor,clock_factor=clock_factor,
           x_min=float(f.rapid_x.min()),x_max=float(f.rapid_x.max()),
           cap_fraction=float(np.average(f.rapid_celerity_capped,weights=f['count'])),
           n_sections=int(f['count'].sum()),routing_seconds=time.perf_counter()-started,
           target_notes=target['target_notes'],**audit,**shape_metrics(case,t,q,flow))
    path.write_text(json.dumps(r,indent=2))
    wave=pd.DataFrame(dict(time_utc=w.first_sample+pd.to_timedelta(t,unit='s'),time_seconds=t,q_inflow_cms=flow,q_outlet_cms=q))
    if save_wave:wave.to_parquet(dest/(name+'.parquet'),index=False)
    return r,wave,f


def fit(case,method):
    cache={}
    def evaluate(n):
        key=round(float(n),12)
        if key not in cache:
            cache[key]=trial(case,method,key)[0]
            pd.DataFrame(cache.values()).sort_values('n_manning').to_csv(case['dest']/f'{method}_trials.csv',index=False)
        return cache[key]
    for n in np.unique(np.r_[.01,.0125,.015,.02,.025,.03,.035,.04,.05,.06,.08,.1,case['old'].n_manning]):evaluate(n)
    def entries(lo,hi):return sorted([r for n,r in cache.items() if lo<=n<=hi and r['valid']],key=lambda r:r['n_manning'])
    def best(lo,hi):return min(entries(lo,hi),key=lambda r:(r['abs_error_h'],abs(r['n_manning']-.035)))
    if not entries(*BOUNDS):raise ValueError(f'No valid calibration trials: {case["ex"]}/{method}')
    # Refine both the best neighborhood and every observed sign crossing;
    # peak timing can be discontinuous or nonmonotonic in Manning n.
    values=entries(*BOUNDS);b=best(*BOUNDS);ix=values.index(b)
    low=values[max(0,ix-1)]['n_manning'];high=values[min(len(values)-1,ix+1)]['n_manning']
    for n in np.linspace(low,high,7):evaluate(n)
    for lo,hi in [BOUNDS,(.025,.06)]:
        for _ in range(9):
            vals=entries(lo,hi)
            if best(lo,hi)['abs_error_h']<=case['step']/7200+1e-12:break
            crossings=[(a,b) for a,b in zip(vals[:-1],vals[1:]) if a['error_h']*b['error_h']<0 and b['n_manning']-a['n_manning']>1e-7]
            if not crossings:break
            for a,b in crossings:evaluate((a['n_manning']+b['n_manning'])/2)
    chosen=best(*BOUNDS);band=best(.025,.06)
    result,wave,frame=trial(case,method,chosen['n_manning'],save_wave=True)
    half=trial(case,method,chosen['n_manning'],clock_factor=2)[0]
    # Grid refinement of fixed-X alters its nominal diffusivity; only MC
    # refinement is an accuracy check of the same physical equation.
    fine=trial(case,method,chosen['n_manning'],grid_factor=2,save_wave=True)[0] if method=='cunge' else None
    eligible=[r['n_manning'] for r in entries(*BOUNDS) if r['within_sampling']]
    result.update(best_at_search_boundary=result['n_manning'] in BOUNDS,
                  common_band_n=band['n_manning'],common_band_error_h=band['abs_error_h'],
                  common_band_within_sampling=band['within_sampling'],
                  half_clock_lag_change_s=3600*(half['lag_h']-result['lag_h']),
                  fine_grid_lag_change_s=None if fine is None else 3600*(fine['lag_h']-result['lag_h']),
                  fine_grid_peak_change_pct=None if fine is None else 100*(fine['peak_cms']/result['peak_cms']-1),
                  sampled_n_min_within_tolerance=min(eligible) if eligible else None,
                  sampled_n_max_within_tolerance=max(eligible) if eligible else None,
                  old_n=float(case['old'].n_manning),old_error_h=float(case['old'].abs_error_h),
                  output_dir=str(case['dest']),independent_event_validation=False)
    wave.to_parquet(case['dest']/f'{method}_best_wave.parquet',index=False)
    manifest=cunge.write_prepared_state(frame,wave.q_inflow_cms,case['step'],case['dt'],case['dest']/f'{method}_prep',metadata=case['manifest'])
    # File-based sparse engine cross-check includes a changing part of the
    # event. Full final runs can be selected separately without refitting.
    changing=np.flatnonzero(abs(np.diff(case['event']))>0)
    stop=min(len(case['event']),int(changing[0])+32) if len(changing) else len(case['event'])
    probe=case['event'][:stop]
    probe_dir=case['dest']/f'{method}_engine_probe'
    pm=cunge.write_prepared_state(frame,probe,case['step'],case['dt'],probe_dir,metadata=case['manifest'])
    _,q=run_prepared_state(probe_dir,return_qout=True)
    actual=q[:,np.array(pm['outlet_reach_ids'])-1].sum(axis=1)
    expected,_=cunge.route(frame,probe,case['nsub'],case['dt'])
    result['engine_probe_max_error_cms']=float(np.max(abs(actual-expected)))
    if result['engine_probe_max_error_cms']>1e-8*max(1,float(np.ptp(probe))):raise ValueError('Prepared engine/cascade mismatch')
    (case['dest']/f'{method}_best.json').write_text(json.dumps(result,indent=2))
    print(case['ex'],method,'n',result['n_manning'],'error_min',result['abs_error_h']*60,'fine_s',result['fine_grid_lag_change_s'],flush=True)
    return result


def run(ex):
    case=load_case(ex);print(ex,'loaded',int(case['grid']['count'].sum()),'sections dt',case['dt'],flush=True)
    results=[fit(case,method) for method in METHODS]
    pd.DataFrame(results).to_csv(case['dest']/'best.csv',index=False)
    return results


def main():
    a=argparse.ArgumentParser();a.add_argument('--examples',nargs='+');a.add_argument('--workers',type=int,default=2);args=a.parse_args()
    examples=args.examples or original.read_schedule().index.tolist()
    OUT.mkdir(parents=True,exist_ok=True)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:results=list(pool.map(run,examples))
    pd.DataFrame([r for group in results for r in group]).to_csv(OUT/'comparison.csv',index=False)


if __name__=='__main__':main()
