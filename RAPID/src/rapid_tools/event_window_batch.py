"""Audit held-tail convergence and sequentially route reviewed input events.

Uses existing calibrated parameters and native clocks; writes a separate run.
"""
from pathlib import Path
import argparse
import hashlib
import json
import math
import time

import numpy as np
import pandas as pd

from . import cunge, common_grid

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_WINDOWS = ROOT/'RAPID/outputs/input_window_review_20261013/proposed_input_windows.json'
DEFAULT_PLAN = ROOT/'RAPID/outputs/event_window_plan_20261013/plan.json'
DEFAULT_OUTPUT = ROOT/'RAPID/outputs/event_window_runs_20261013'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    tmp.replace(path)


def equilibrium_check(frame, dt):
    """Verify the declared steady initialization for every native subreach."""
    f=frame.reset_index(drop=True)
    k,x=f.rapid_k.to_numpy(float),f.rapid_x.to_numpy(float)
    den=dt+2*k*(1-x)
    d=(2*k*(1-x)-dt)/den
    if not np.isfinite(d).all() or (d<0).any() or (d>=1).any():
        raise ValueError('Unstable native routing clock.')
    order,edges,root,end=cunge.topology(f)
    values={root:1.};error=0.
    for u in order:
        for i in edges.get(u,[]):
            q=values[u]*float(f.split_fraction.iloc[i])
            error=max(error,abs(q-f.reference_flow_fraction.iloc[i]))
            values[f.v.iloc[i]]=values.get(f.v.iloc[i],0.)+q
    if error>1e-10 or abs(values[end]-1)>1e-10:
        raise ValueError('Inconsistent steady initialization.')
    return error


def settling_hours(times, q, held_start, final, tolerance, guard_hours=96):
    """Last exceedance plus 48 h settling confirmation and 48 h extra check.

    Returns a measured required duration and whether the saved trace covers it.
    A finite-window diagnostic, not an infinite-horizon mathematical guarantee.
    """
    seconds=(pd.DatetimeIndex(times)-pd.Timestamp(held_start)).total_seconds().to_numpy()
    if not np.isfinite(q).all() or not np.isfinite(seconds).all() or not (np.diff(seconds)>0).all():
        raise ValueError("Settlement requires finite chronological output.")
    valid=seconds>=0
    if not valid.any():
        return float(guard_hours),False
    outside=np.flatnonzero(valid & (abs(np.asarray(q)-final)>tolerance))
    last=max(0.,float(seconds[outside[-1]])+60) if len(outside) else 0.
    required=last/3600+guard_hours
    return required,bool(seconds[-1]+30>=required*3600)


def event_frame(window, forcing_path):
    f = pd.read_parquet(forcing_path)
    start, end = pd.Timestamp(window['start_utc']), pd.Timestamp(window['end_utc'])
    event = f[f.time_utc.between(start, end)].copy().reset_index(drop=True)
    if len(event) < 3 or event.time_utc.iloc[0] != start or event.time_utc.iloc[-1] != end:
        raise ValueError('Selected timestamps absent from source forcing.')
    if event.forcing_endpoint_hold.any() or not np.isfinite(event.forcing_cms).all():
        raise ValueError('Event includes held or invalid input.')
    np.testing.assert_allclose(np.diff(event.time_utc.astype('int64'))/1e9, 60., atol=0, rtol=0)
    np.testing.assert_allclose((event.interval_end_utc-event.interval_start_utc).dt.total_seconds(), 60.)
    np.testing.assert_allclose(event.forcing_cms.iloc[[0, -1]], [window['start_cms'], window['end_cms']], rtol=0, atol=1e-7)
    return event


def inventory(ex):
    folder = ROOT/f'network_variants/output/{ex}_indep__anchored'
    sources = {}
    tables = []
    for family in ['indep','compound_bubbles','island']:
        path = ROOT/f'network_variants/output/{ex}_{family}__anchored/rapid_run_registry.csv'
        if not path.exists():
            continue
        table = pd.read_csv(path)
        if not table.method.eq('cunge').all() or not table.status.eq('ran').all() or table.state_id.duplicated().any():
            raise ValueError('Expected completed unique Cunge registry states.')
        table['family'] = family
        tables.append(table)
        sources[str(path.resolve())] = sha(path)
    registry = pd.concat(tables,ignore_index=True)
    jobs = []
    for fingerprint, group in registry.groupby('fingerprint',sort=False):
        if group.parameters_parquet.nunique()!=1 or group.routing_dt_seconds.nunique()!=1:
            raise ValueError('Aliased routing parameters disagree.')
        r=group.iloc[0]
        jobs.append(dict(state_id=r.state_id, fingerprint=fingerprint,
            aliases=[f'{v.family}/{v.state_id}' for v in group.itertuples()],
            parameters=str(Path(r.parameters_parquet).resolve()), outlet=str(Path(r.outlet_parquet).resolve()), native_dt_seconds=float(r.routing_dt_seconds)))
    if len({j['state_id'] for j in jobs})!=len(jobs):
        raise ValueError('Different fingerprints share a state name.')
    prep = folder/'reference_section/rapid/cunge/prep'
    receipt = prep/'completion.json'
    meta = json.loads(receipt.read_text())
    jobs.append(dict(state_id='single_centerline', parameters=str((prep/'parameters.parquet').resolve()),
                     outlet=str((folder/'reference_section/rapid/cunge/run/outlet.parquet').resolve()), native_dt_seconds=float(meta['native_dt_s']), fingerprint='single_centerline', aliases=['reference_section/single_centerline']))
    base = registry[registry.state_id.eq('S000_base')].iloc[0]
    sources[str(receipt.resolve())] = sha(receipt)
    if registry.forcing_parquet.nunique()!=1:
        raise ValueError('All families must share one source forcing.')
    return jobs, str(Path(base.forcing_parquet).resolve()), sources


def code_sources():
    return {str(Path(p).resolve()): sha(p) for p in [__file__, cunge.__file__, common_grid.__file__]}


def plan(windows_path, output):
    windows_path,output=Path(windows_path).resolve(),Path(output).resolve()
    windows=json.loads(windows_path.read_text())['windows']
    cases=[];rows=[]
    for w in windows:
        ex=w['example'];jobs,forcing,sources=inventory(ex)
        sources[forcing]=sha(forcing)
        event=event_frame(w,forcing)
        full=pd.read_parquet(forcing)
        held_start=full.loc[~full.forcing_endpoint_hold,'interval_end_utc'].iloc[-1]
        old_final=float(full.forcing_cms.iloc[-1])
        q0,qe=float(event.forcing_cms.iloc[0]),float(event.forcing_cms.iloc[-1])
        tolerance=.001*float(np.ptp(event.forcing_cms))
        if tolerance<=0:raise ValueError('Degenerate event.')
        for job in jobs:
            p=job['parameters'];sources[p]=sha(p)
            f=pd.read_parquet(p)
            if not np.isclose(60/job['native_dt_seconds'],round(60/job['native_dt_seconds']),rtol=0,atol=1e-8):
                raise ValueError('Native clock must divide 60 s.')
            error=equilibrium_check(f,job['native_dt_seconds'])
            old=pd.read_parquet(job['outlet'])
            hours,covered=settling_hours(old.time_utc,old.q_outlet_cms,held_start,old_final,tolerance)
            job.update(equilibrium_fraction_error=error,n_segments=int(f['count'].sum()),
                       saved_required_tail_hours=hours,saved_tail_covers_guard=covered,
                       saved_outlet_sha256=sha(job['outlet']))
            rows.append(dict(example=ex,**job))
        slowest=max(jobs,key=lambda j:j['saved_required_tail_hours'])
        # Existing tails seed the pilot; they do not certify the changed input.
        tail_hours=max(120,math.ceil(slowest['saved_required_tail_hours']/24)*24)
        selected=[]
        for name in ['S000_base','single_centerline',slowest['state_id']]:
            if name not in selected:selected.append(name)
        pilots=[]
        for name in selected:
            job=next(j for j in jobs if j['state_id']==name)
            f=pd.read_parquet(job['parameters']);dt=job['native_dt_seconds']
            for attempt in range(9):
                flow=np.r_[event.forcing_cms,np.full(tail_hours*60,qe)]
                begin=time.monotonic()
                q,audit=common_grid.route_interval_means(f,flow,round(60/dt),dt,initial_flow=q0)
                if audit['minimum_native_flow_cms'] < -1e-9:
                    raise ValueError('Negative pilot flow: '+ex+'/'+name)
                times=event.time_utc.iloc[0]+pd.to_timedelta(np.arange(len(q))*60,unit='s')
                hours,covered=settling_hours(times,q,event.interval_end_utc.iloc[-1],qe,tolerance)
                if covered:break
                tail_hours=max(tail_hours+96,math.ceil(hours/24)*24)
            else:raise ValueError('Pilot failed to settle: '+ex+'/'+name)
            dest=output.parent/'pilots'/ex/name;dest.mkdir(parents=True,exist_ok=True)
            pd.DataFrame(dict(time_utc=times,q_outlet_cms=q)).to_parquet(dest/'outlet.parquet',index=False)
            rec=dict(example=ex,state_id=name,required_tail_hours=hours,simulated_tail_hours=tail_hours,
                     minimum_native_flow_cms=audit['minimum_native_flow_cms'],seconds=time.monotonic()-begin)
            pilots.append(rec);save(dest/'result.json',rec)
            print(ex,'pilot',name,f'{hours:.1f} h required; {tail_hours} h checked',flush=True)
        # Keep the longer of the all-state historical requirement and fresh pilots.
        tail_hours=max(tail_hours,math.ceil(max(p['required_tail_hours'] for p in pilots)/24)*24)
        event_end=event.interval_end_utc.iloc[-1]
        cases.append(dict(example=ex,window=w,forcing=forcing,jobs=jobs,pilots=pilots,
            initial_hours=24,tail_hours=tail_hours,guard_hours=96,tolerance_cms=tolerance,
            initial_cms=q0,final_cms=qe,event_intervals=len(event),event_duration_hours=len(event)/60,
            simulation_start_utc=(event.interval_start_utc.iloc[0]-pd.Timedelta(hours=24)).isoformat(),
            simulation_end_utc=(event_end+pd.Timedelta(hours=tail_hours)).isoformat(),
            limiting_saved_state=slowest['state_id'],input_gap_bins=int(event.upstream_gap.sum()),source_sha256=sources))
        pd.DataFrame(rows).to_csv(output.parent/'all_state_audit.csv',index=False)
        save(output.parent/'planning_progress.json',dict(completed_cases=cases))
        print(ex,len(jobs),'states; before 24 h; planned after',tail_hours,'h',flush=True)
    doc=dict(schema='reviewed-event-window-batch-v2',status='pilot_checked_full_batch_pending',
        initialization='steady_first_inflow; constant 24-hour prefix',
        tail_rule='Initial plan: all-state old-tail audit and new-event parent/centerline/slowest-old-state pilots. Every new state must pass 96 continuous hours within 0.1% input range; extend common tail by at least 96 hours if needed.',
        qualification='Finite-tail convergence checks, not an infinite-horizon bound. Final durations can exceed plan.',
        event_bin_policy='Include both selected midpoint bins; preserve original 60-second means and phase.',
        analysis_warning='Padding is not a response-metric integration window. No metrics changed.',
        source_sha256={str(windows_path):sha(windows_path)},code_sha256=code_sources(),cases=cases)
    save(output,doc)
    pd.DataFrame([{k:c[k] for k in ['example','initial_hours','tail_hours','event_duration_hours','limiting_saved_state','tolerance_cms','simulation_start_utc','simulation_end_utc','input_gap_bins']}|{'states':len(c['jobs'])} for c in cases]).to_csv(output.parent/'buffer_summary.csv',index=False)
    return doc


def verify(sources):
    for p, expected in sources.items():
        if sha(p) != expected:
            raise ValueError('Input/code changed since plan: '+p)


def forcing_with_buffers(case):
    event = event_frame(case['window'], case['forcing'])
    nb, na = int(case['initial_hours']*60), int(case['tail_hours']*60)
    origin = event.interval_start_utc.iloc[0]-pd.Timedelta(minutes=nb)
    n = nb+len(event)+na
    wave = pd.DataFrame({'interval_start_utc':origin+pd.to_timedelta(np.arange(n)*60, unit='s'),
        'interval_end_utc':origin+pd.to_timedelta((np.arange(n)+1)*60, unit='s'),
        'time_utc':origin+pd.to_timedelta((np.arange(n)+.5)*60, unit='s'),
        'forcing_cms':np.r_[np.full(nb,case['initial_cms']),event.forcing_cms,np.full(na,case['final_cms'])],
        'phase':['spinup']*nb+['input_event']*len(event)+['settling']*na})
    wave['forcing_endpoint_hold']=wave.phase.ne('input_event')
    wave['upstream_gap']=False
    wave.loc[nb:nb+len(event)-1,'upstream_gap']=event.upstream_gap.to_numpy()
    assert wave.time_utc.iloc[nb] == event.time_utc.iloc[0]
    assert wave.interval_end_utc.iloc[-1] == pd.Timestamp(case['simulation_end_utc'])
    return wave, nb


def run(plan_path, output, examples=None):
    plan_path, output = Path(plan_path).resolve(), Path(output).resolve()
    doc = json.loads(plan_path.read_text()); plan_hash = sha(plan_path)
    if doc.get('schema') != 'reviewed-event-window-batch-v2':
        raise ValueError('Unsupported event-window plan schema.')
    verify(doc['source_sha256']); verify(doc['code_sha256'])
    chosen = examples or [c['example'] for c in doc['cases']]
    if len(set(chosen)) != len(chosen) or not set(chosen) <= {c['example'] for c in doc['cases']}:
        raise ValueError('Select unique planned examples.')
    # Refuse production or source destinations; this command writes isolated runs.
    if not output.is_relative_to((ROOT/'RAPID/outputs').resolve()):
        raise ValueError('Output must be a new directory within RAPID/outputs.')
    output.mkdir(parents=True, exist_ok=True)
    batch = output/'batch.json'
    if batch.exists():
        if json.loads(batch.read_text())['plan_sha256'] != plan_hash:
            raise ValueError('Existing output belongs to a different plan.')
    elif any(output.iterdir()):
        raise ValueError('Output directory is not an empty new batch.')
    else:
        save(batch, dict(plan=str(plan_path),plan_sha256=plan_hash,selection=chosen))
    for ex in chosen:
        case=dict(next(c for c in doc['cases'] if c['example']==ex))
        verify(case['source_sha256'])
        folder=output/ex;folder.mkdir(exist_ok=True)
        if (folder/'completion.json').exists():
            done=json.loads((folder/'completion.json').read_text())
            if done['plan_sha256']!=plan_hash:raise ValueError('Example belongs to another plan.')
            verify(done['files'])
            print(ex,'complete; hashes verified',flush=True);continue
        for attempt in range(9):
            hours=case['tail_hours']
            dest=folder/f'tail_{hours}h';dest.mkdir(exist_ok=True)
            wave,nb=forcing_with_buffers(case)
            wave.to_parquet(dest/'forcing.parquet',index=False)
            dynamic=wave.forcing_cms.to_numpy()[nb:]
            completed=[]
            for j,job in enumerate(case['jobs']):
                target=dest/job['state_id'];receipt=target/'completion.json'
                if receipt.exists():
                    done=json.loads(receipt.read_text())
                    if done['plan_sha256']!=plan_hash:raise ValueError('State belongs to another plan.')
                    verify(done['files']);completed.append(done)
                    print(ex,job['state_id'],'verified resume',flush=True);continue
                target.mkdir(exist_ok=True)
                f=pd.read_parquet(job['parameters']);dt=job['native_dt_seconds']
                begin=time.monotonic()
                q,audit=common_grid.route_interval_means(f,dynamic,round(60/dt),dt,initial_flow=case['initial_cms'])
                if not np.isfinite(q).all() or audit['minimum_native_flow_cms'] < -1e-9:
                    raise ValueError('Invalid routed flow: '+ex+'/'+job['state_id'])
                required,settled=settling_hours(wave.time_utc.iloc[nb:],q,
                    wave.interval_start_utc.iloc[nb+case['event_intervals']],case['final_cms'],case['tolerance_cms'])
                observed=float(np.max(abs(q[-96*60:]-case['final_cms'])))
                q=np.r_[np.full(nb,case['initial_cms']),q]
                out=wave[['interval_start_utc','interval_end_utc','time_utc','phase']].copy();out['q_outlet_cms']=q
                out.to_parquet(target/'outlet.parquet',index=False)
                done=dict(example=ex,state_id=job['state_id'],aliases=job['aliases'],plan_sha256=plan_hash,
                    complete=True,settled=settled,required_tail_hours=required,
                    final_96h_max_error_cms=observed,tolerance_cms=case['tolerance_cms'],
                    seconds=time.monotonic()-begin,audit=audit,
                    files={str(target/'outlet.parquet'):sha(target/'outlet.parquet')})
                save(receipt,done);completed.append(done)
                print(ex,f'{j+1}/{len(case["jobs"])}',job['state_id'],f'{done["seconds"]:.1f}s; settled={settled}',flush=True)
            if all(c['settled'] for c in completed):break
            case['tail_hours']=max(hours+96,math.ceil(max(c['required_tail_hours'] for c in completed)/24)*24)
            event_end=pd.Timestamp(case['window']['end_utc'])+pd.Timedelta(seconds=30)
            case['simulation_end_utc']=(event_end+pd.Timedelta(hours=case['tail_hours'])).isoformat()
            print(ex,'EXTENDING shared tail to',case['tail_hours'],'hours',flush=True)
        else:raise ValueError('Nine tail attempts failed; no completed example published.')
        verify(case['source_sha256'])
        files={str(dest/'forcing.parquet'):sha(dest/'forcing.parquet')}
        for c in completed:files.update(c['files'])
        save(folder/'completion.json',dict(complete=True,plan_sha256=plan_hash,states=len(completed),
            final_tail_hours=case['tail_hours'],final_directory=str(dest),
            max_final_96h_error_cms=max(c['final_96h_max_error_cms'] for c in completed),files=files))
    print('Selected examples complete:', ' '.join(chosen), flush=True)


def cli():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='action', required=True)
    p = sub.add_parser('plan'); p.add_argument('--windows',type=Path,default=DEFAULT_WINDOWS); p.add_argument('--output',type=Path,default=DEFAULT_PLAN)
    p = sub.add_parser('run'); p.add_argument('--plan',type=Path,default=DEFAULT_PLAN); p.add_argument('--output',type=Path,default=DEFAULT_OUTPUT); p.add_argument('--examples',nargs='+')
    args = parser.parse_args()
    if args.action=='plan': plan(args.windows,args.output)
    else: run(args.plan,args.output,args.examples)

if __name__=='__main__': cli()
