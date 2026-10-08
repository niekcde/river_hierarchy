"""Independent-event screening and full prepared-engine verification."""
from pathlib import Path
import argparse
import json
import math
import sys
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from openpyxl import load_workbook

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'RAPID/src'));sys.path.insert(0,str(Path(__file__).parent))
import cunge_calibration as cal
from rapid_tools import cunge
from rapid_tools.engine import run_prepared_state
from rapid_tools.analysis import GAUGE_QUALITY_XLSX,_load_station_timeseries_for_observed_lag
OUT=cal.OUT


def select_events():
    """Freeze selection using gauges only, before looking at routed scores."""
    dest=OUT/'independent_events';dest.mkdir(parents=True,exist_ok=True)
    book=load_workbook(GAUGE_QUALITY_XLSX,read_only=True,data_only=True)
    try:r=list(book['Hydrograph_quality'].values);quality=pd.DataFrame(r[1:],columns=r[0])
    finally:book.close()
    schedule=cal.original.read_schedule();rows=[]
    for ex,w in schedule.iterrows():
        rec=quality[pd.to_numeric(quality.Example,errors='coerce').eq(int(ex))].iloc[0]
        series={side:_load_station_timeseries_for_observed_lag(str(rec[key])).dropna(subset=['time','discharge_cms']).sort_values('time').drop_duplicates('time') for side,key in [('upstream','Up'),('downstream','Down')]}
        up=series['upstream'];six=up.set_index('time').discharge_cms.resample('6h').median()
        prominence=.15*(six.quantile(.95)-six.quantile(.05))
        peaks,_=find_peaks(six.fillna(-np.inf).to_numpy(),distance=28*4,prominence=prominence)
        center=w.requested_start+(w.requested_end_exclusive-w.requested_start)/2
        candidates=sorted(six.index[peaks],key=lambda t:abs((t-center).total_seconds()))
        chosen=[];rejections={}
        for peak in candidates:
            start=peak-pd.Timedelta(days=7);end=peak+pd.Timedelta(days=14)
            if not (end<w.requested_start-pd.Timedelta(days=7) or start>w.requested_end_exclusive+pd.Timedelta(days=7)):
                continue
            if any(not (end<a or start>b) for a,b in chosen):continue
            windows={side:g[g.time.between(start,end)].copy() for side,g in series.items()}
            reason=None
            for side,g in windows.items():
                if len(g)<50:reason='insufficient samples';break
                gaps=np.diff(pd.to_datetime(g.time,utc=True).astype('int64'))/1e9
                if ((pd.Timestamp(g.time.iloc[0])-start).total_seconds()>6*3600
                    or (end-pd.Timestamp(g.time.iloc[-1])).total_seconds()>6*3600
                    or gaps.max()>6*3600):reason='coverage gap over six hours';break
            if reason:
                rejections[reason]=rejections.get(reason,0)+1;continue
            ug=windows['upstream'];actual_peak=ug.time.iloc[int(ug.discharge_cms.to_numpy().argmax())]
            if abs((actual_peak-peak).total_seconds())>24*3600:
                rejections['another upstream maximum dominates window']=rejections.get('another upstream maximum dominates window',0)+1;continue
            selected=len(chosen)+1;event=f'{ex}_{selected}';folder=dest/event;folder.mkdir(exist_ok=True)
            for side,g in windows.items():g.to_parquet(folder/f'{side}.parquet',index=False)
            chosen.append((start,end));rows.append(dict(example=ex,event=event,start=str(start),end=str(end),selection_peak_utc=str(peak),folder=str(folder),status='selected'))
            if len(chosen)==2:break
        if not chosen:rows.append(dict(example=ex,event=None,status='no eligible independent event'))
        (dest/f'{ex}_selection_audit.json').write_text(json.dumps(dict(selected=len(chosen),candidate_peaks=len(candidates),rejections=rejections),indent=2))
        print('selected',ex,len(chosen),flush=True)
    table=pd.DataFrame(rows);table.to_csv(dest/'selection.csv',index=False)
    (dest/'selection_policy.json').write_text(json.dumps(dict(
        selection='Two temporally nearest distinct upstream peaks outside calibration plus seven-day guard; no model scores used.',
        smoothing='Six-hour median for peak detection only',minimum_peak_spacing_days=28,
        minimum_prominence='0.15*(upstream 95th minus 5th percentile)',window_days_before=7,window_days_after=14,
        coverage='Both stations: >=50 samples; no internal or endpoint gap above6 h; upstream global maximum within24 h of detection peak.',
        calibration_parameters='n, Q_reference, slopes, widths, depths, alpha, grid and timestep all inherited without refitting.',
        limitation='Automatically screened events need hydrological review; rating changes, tributaries and stage/backwater effects are not resolved.'),indent=2))
    return table


def independent(ex):
    comparison=pd.read_csv(OUT/'comparison.csv',dtype={'example':str});best=comparison[comparison.example==ex]
    dest=Path(best.iloc[0].output_dir);manifest=json.loads((dest/'case.json').read_text());grid=pd.read_parquet(dest/'frozen_grid.parquet')
    selected=pd.read_csv(OUT/'independent_events/selection.csv',dtype={'example':str});selected=selected[(selected.example==ex)&selected.status.eq('selected')];records=[]
    for event in selected.itertuples():
        folder=Path(event.folder);up=pd.read_parquet(folder/'upstream.parquet');down=pd.read_parquet(folder/'downstream.parquet')
        start=max(pd.Timestamp(up.time.iloc[0]),pd.Timestamp(down.time.iloc[0]));end=min(pd.Timestamp(up.time.iloc[-1]),pd.Timestamp(down.time.iloc[-1]))
        step=manifest['forcing_step_s'];dt=manifest['routing_dt_s'];nsub=manifest['nsub'];t=np.arange(int((end-start).total_seconds()/step)+1)*step
        ut=(pd.to_datetime(up.time,utc=True)-start).dt.total_seconds().to_numpy();dtime=(pd.to_datetime(down.time,utc=True)-start).dt.total_seconds().to_numpy()
        uq=up.discharge_cms.to_numpy();dq=down.discharge_cms.to_numpy();flow=np.interp(t,ut,uq)
        up_peak=ut[uq.argmax()];down_peak=dtime[dq.argmax()];observed_lag=(down_peak-up_peak)/3600
        usable=bool(observed_lag>0 and 86400<up_peak<t[-1]-86400 and 86400<down_peak<t[-1]-86400)
        for r in best.itertuples():
            f=cunge.parameters(grid,r.n_manning,method=r.method)
            q,audit=cunge.route(f,flow,nsub,dt);peak=t[q.argmax()];lag=(peak-t[flow.argmax()])/3600
            inside=(dtime>=0)&(dtime<=t[-1]);obs=dq[inside];pred=np.interp(dtime[inside],t,q)
            norm=lambda y:(y-y.min())/max(np.ptp(y),1e-12)
            valid=bool(usable and 0<q.argmax()<len(q)-1 and audit['minimum_native_flow_cms']>=-1e-8)
            result=dict(example=ex,event=event.event,method=r.method,n_manning=r.n_manning,
                        observed_lag_h=observed_lag,model_lag_h=lag,abs_error_h=abs(lag-observed_lag),
                        observed_target_usable=usable,valid=valid,calibration_within_sampling=bool(r.within_sampling),
                        normalized_shape_rmse=float(np.sqrt(np.mean((norm(pred)-norm(obs))**2))),
                        model_to_observed_peak_ratio=float(pred.max()/obs.max()),
                        forcing_peak_offset_h=(t[flow.argmax()]-up_peak)/3600,
                        **audit)
            records.append(result);pd.DataFrame(dict(time_utc=start+pd.to_timedelta(t,unit='s'),q_inflow_cms=flow,q_outlet_cms=q)).to_parquet(folder/f'{r.method}_wave.parquet',index=False)
        print('holdout',event.event,'observed_lag',observed_lag,flush=True)
    pd.DataFrame(records).to_csv(OUT/'independent_events'/f'{ex}_results.csv',index=False)
    return records


def prepared(ex):
    comparison=pd.read_csv(OUT/'comparison.csv',dtype={'example':str});r=comparison[(comparison.example==ex)&comparison.method.eq('cunge')].iloc[0]
    dest=Path(r.output_dir);prep=dest/'cunge_prep';m=json.loads((prep/'rapid_prep_manifest.json').read_text());run=dest/'cunge_run';run.mkdir(exist_ok=True)
    out=run/'Qout.nc';_,q=run_prepared_state(prep,output_path=out,return_qout=True)
    outlet=q[:,np.array(m['outlet_reach_ids'])-1].sum(axis=1);wave=pd.read_parquet(dest/'cunge_best_wave.parquet')
    error=float(np.max(abs(outlet-wave.q_outlet_cms.to_numpy())))
    if error>1e-8*max(1,float(np.ptp(wave.q_inflow_cms))):raise ValueError(('Full sparse mismatch',ex,error))
    wave['q_outlet_cms']=outlet;wave.to_csv(run/'outlet_hydrograph.csv',index=False)
    audit=json.loads(out.with_suffix('.audit.json').read_text())
    result=dict(example=ex,max_engine_cascade_error_cms=error,**audit)
    # Positive pulses expose below-baseline artifacts that positive total Q
    # alone can conceal. Same fitted n/grid/clock; no calibration to pulses.
    f=pd.read_parquet(prep/'hydraulic_links.parquet');ref=m['metadata']['reference_discharge_cms'];step=m['routing']['forcing_dt_seconds'];dt=m['routing']['routing_dt_seconds'];nsub=round(step/dt)
    pulses=[]
    for rise_h in [12,48]:
        tail_days=max(30,float(r.simulation_days));t=np.arange(int((3+2*rise_h/24+tail_days)*86400/step)+1)*step
        p=np.sin(np.pi*np.clip((t-86400)/(2*rise_h*3600),0,1))**2;p[(t<86400)|(t>86400+2*rise_h*3600)]=0
        flow=ref*(.1+.9*p);y,a=cunge.route(f,flow,nsub,dt)
        pulses.append(dict(example=ex,rise_h=rise_h,finite_window_excess_volume_ratio=float(np.sum(y-flow[0])/np.sum(flow-flow[0])),
                           final_excess_fraction=float(abs(y[-1]-flow[0])/(.9*ref)),**a))
    pd.DataFrame(pulses).to_csv(dest/'fitted_pulse_checks.csv',index=False)
    (dest/'full_engine_verification.json').write_text(json.dumps(result,indent=2))
    print('full engine',ex,'error',error,flush=True)
    return result


def main():
    a=argparse.ArgumentParser();a.add_argument('mode',choices=['select','independent','prepared']);a.add_argument('--workers',type=int,default=2);a.add_argument('--examples',nargs='+');args=a.parse_args()
    if args.mode=='select':select_events();return
    examples=args.examples or pd.read_csv(OUT/'comparison.csv',dtype={'example':str}).example.unique().tolist()
    func=independent if args.mode=='independent' else prepared
    with ProcessPoolExecutor(max_workers=args.workers) as pool:results=list(pool.map(func,examples))
    if args.mode=='independent':results=[r for group in results for r in group]
    pd.DataFrame(results).to_csv(OUT/f'{args.mode}_verification.csv',index=False)


if __name__=='__main__':main()
