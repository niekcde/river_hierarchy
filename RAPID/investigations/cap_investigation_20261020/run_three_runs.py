"""Sequential frozen BASED correction / BASED midpoint / Manning midpoint batch.

Default: verify configuration only. --run executes, with resumable hashed
receipts, one process pool at a time and an exclusive output-directory lock.
"""
import os
for _key in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:
    os.environ[_key]='1'
import argparse,ast,fcntl,hashlib,json,math,platform,shutil,sys,time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from manning_calibrated_network import ROOT,OUT,BUNDLE,digest
from manning_network_pilot import state_metrics
from staged_routing_merge import assemble
from convergence_moments import metrics as convergence_metrics
from rapid_tools import cunge
from rapid_tools.common_grid import route_interval_means
from rapid_tools.reference_section import predict_based_depth
from rapid_tools.hydrograph_response import response_rows
from rapid_tools.event_window_batch import settling_hours

SETUP=OUT/'three_run_setup_20261028'
PREFLIGHT=SETUP/'preflight_20261028'
RUNS=['01_based_slope_geometry_correction','02_based_midpoint','03_manning_midpoint']
CTX=None
CACHE={}


def write_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,indent=2,default=str)+'\n');os.replace(tmp,path)


def stamp(value):return hashlib.sha256(json.dumps(value,sort_keys=True,default=str).encode()).hexdigest()


def ensure_space(root,reserve=20):
    free=shutil.disk_usage(root).free/2**30
    if free<reserve:raise RuntimeError(f'Free space {free:.2f} GiB below {reserve} GiB reserve; free space and resume.')
    return free


def verify_inputs():
    manifest=json.loads((PREFLIGHT/'manifest.json').read_text())
    for kind in ['source_hashes','output_hashes']:
        for p,h in manifest[kind].items():
            if digest(p)!=h:raise RuntimeError(f'Preflight fingerprint changed: {p}; rerun/review preflight before launching.')
    completion=json.loads((PREFLIGHT/'completion.json').read_text())
    if not completion['ready_to_start_calibration_and_runs']:raise RuntimeError('Preflight is not ready')
    files=dict(manifest['source_hashes'])
    for p in [SETUP/'run_setup_v2.json',SETUP/'frozen_reference_discharges.csv',PREFLIGHT/'completion.json',PREFLIGHT/'shared_midpoint_grid_checks.csv',PREFLIGHT/'state_checks.csv',
              OUT/'manning_all_networks_preparation_20261027/run_specification.json',OUT/'manning_all_networks_preparation_20261027/response_windows.parquet',
              BUNDLE/'source_code/calibrate_common_grid_height.py']:
        files[str(p)]=digest(p)
    plan=json.loads((OUT/'manning_all_networks_preparation_20261027/run_specification.json').read_text())
    for c in plan['cases']:
        for p in [c['forcing'],c['calibration_forcing'],c['calibration_metadata']]:files[p]=digest(p)
    for p in [Path(__file__),Path(__file__).with_name('staged_routing_merge.py'),Path(__file__).with_name('manning_network_pilot.py'),Path(__file__).with_name('convergence_moments.py'),
              ROOT/'RAPID/src/rapid_tools/cunge.py',ROOT/'RAPID/src/rapid_tools/common_grid.py',ROOT/'RAPID/src/rapid_tools/event_window_batch.py',
              ROOT/'RAPID/src/rapid_tools/hydrograph_response.py',ROOT/'RAPID/src/rapid_tools/reference_section.py']:
        files[str(p.resolve())]=digest(p)
    return dict(files=files,python=platform.python_version(),numpy=np.__version__,pandas=pd.__version__)


def init_worker(output,fingerprint):
    global CTX,CACHE
    spec=json.loads((OUT/'manning_all_networks_preparation_20261027/run_specification.json').read_text())
    indexpath=json.loads((SETUP/'run_setup_v2.json').read_text())['shared']['staged_slope_index']
    index=json.loads(Path(indexpath).read_text())
    CTX=dict(output=Path(output),fingerprint=fingerprint,cases={c['example']:c for c in spec['cases']},
        slopes={s['case']:s for s in index['cases']},values=pd.read_csv(SETUP/'frozen_reference_discharges.csv',dtype={'example':str}).set_index('example'),
        windows=pd.read_parquet(OUT/'manning_all_networks_preparation_20261027/response_windows.parquet'),
        clocks=pd.read_csv(PREFLIGHT/'state_checks.csv',dtype={'example':str}).set_index(['example','state']),
        midpoint=pd.read_csv(PREFLIGHT/'shared_midpoint_grid_checks.csv',dtype={'example':str}).set_index(['example','state']))
    CACHE={}


def parent_inputs(ex):
    if ex not in CACHE:
        case=CTX['cases'][ex];job=next(j for j in case['jobs'] if j['state']=='S000_base')
        base=pd.read_parquet(job['source_parameters']);suffix='_direction130' if ex=='15' else ''
        kb=json.loads((ROOT/f'network_variants/output/{ex}_indep{suffix}__anchored/reference_section/reference_kb_summary.json').read_text())
        q=CTX['values'].loc[ex,'midpoint_Qref_cms']
        anchor=predict_based_depth(discharge_cms=q,width_m=kb['reference_width_m'],slope=kb['reference_slope'])
        CACHE[ex]=(base,kb,anchor)
    return CACHE[ex]


def prepare_frame(ex,state,run):
    base,kb,anchor=parent_inputs(ex);case=CTX['cases'][ex]
    job=next(j for j in case['jobs'] if j['state']==state)
    f=pd.read_parquet(job['source_parameters']);key=ex+'/'+state
    if key in CTX['slopes']:
        item=CTX['slopes'][key];s=pd.read_parquet(item['slope_file'])
        actions=json.loads((Path(item['accepted_geometry_candidate']).parent/'geometry_actions.json').read_text()) if 'accepted_geometry_candidate' in item else []
        f,_,_,_=assemble(f,base,s,actions)
    order,edges,root,end=cunge.topology(f);shares={root:1.};alloc=np.zeros(len(f));splits=np.zeros(len(f))
    w=f.width.to_numpy();v=f.v.to_numpy()
    for u in order:
        ids=edges.get(u,[]);total=w[ids].sum()
        for i in ids:
            splits[i]=w[i]/total;alloc[i]=shares[u]*splits[i];shares[v[i]]=shares.get(v[i],0)+alloc[i]
    if not np.isclose(shares[end],1,atol=1e-10,rtol=0):raise ValueError('Reference allocation is not conservative')
    qnet=CTX['values'].loc[ex,'legacy_routing_Qref_cms' if run==RUNS[0] else 'midpoint_Qref_cms']
    f['reference_flow_fraction']=alloc;f['split_fraction']=splits;f['reference_discharge_cms']=qnet*alloc
    if run!=RUNS[0]:
        # Identical frozen spatial grid for the two midpoint depth formulations.
        q=f.reference_discharge_cms.to_numpy();s=f.slope.to_numpy();L=f.link_length_m.to_numpy()
        hbased=anchor*(w/kb['reference_width_m'])**.65
        cb=np.clip(5/(3*.01)*hbased**(2/3)*np.sqrt(s),f.celerity_min_mps,f.celerity_max_mps)
        hm=(.01*q/(w*np.sqrt(s)))**.6;cm=5/(3*.01)*hm**(2/3)*np.sqrt(s)
        alpha=float(f.cunge_alpha.iloc[0])
        required=np.maximum(np.ceil(L/(alpha*q/(w*s*cb))),np.ceil(L/(alpha*q/(w*s*cm)))).astype(int)
        f['count']=np.maximum(f['count'].to_numpy().astype(int),required)
        f['depth_m']=hbased
        record=CTX['midpoint'].loc[ex,state]
        if int(f['count'].sum())!=int(record.shared_midpoint_cells):raise ValueError('Shared-grid preflight mismatch')
        dt=float(record.shared_dt_seconds)
    else:dt=float(CTX['clocks'].loc[(ex,state),'based_legacy_dt_seconds'])
    f['count']=f['count'].astype(int);f['rapid_effective_length_m']=f.link_length_m/f['count']
    return f,dt,job


def hydraulic_parameters(grid,run,n,beta):
    f=grid.copy();f['n_manning']=float(n);f['diffusion_multiplier']=float(beta)
    q=f.reference_discharge_cms;w=f.width;s=f.slope
    if run==RUNS[2]:
        f['depth_m']=(n*q/(w*np.sqrt(s)))**.6
        np.testing.assert_allclose(w/n*f.depth_m**(5/3)*np.sqrt(s),q,rtol=1e-11)
    f['celerity_times_n']=5/3*f.depth_m**(2/3)*np.sqrt(s)
    f['rapid_celerity_mps_raw']=f.celerity_times_n/n
    f['rapid_celerity_mps']=f.rapid_celerity_mps_raw if run==RUNS[2] else np.clip(f.rapid_celerity_mps_raw,f.celerity_min_mps,f.celerity_max_mps)
    c=f.rapid_celerity_mps;dx=f.rapid_effective_length_m
    f['rapid_celerity_capped']=~np.isclose(c,f.rapid_celerity_mps_raw,rtol=1e-12,atol=1e-12)
    f['hydraulic_length_scale_m']=q/(w*s*c);f['rapid_k']=dx/c;f['rapid_x']=.5-beta*q/(2*w*s*c*dx)
    f['hydraulic_diffusivity_m2_s']=q/(2*w*s)
    if 'kb_local' in f:f=f.drop(columns='kb_local')
    f['depth_policy']='wide_rectangular_Manning_at_frozen_midpoint_Q' if run==RUNS[2] else 'BASED_reference_width_scaling'
    f['routing_method']='cunge'
    if not np.isfinite(f[['depth_m','rapid_k','rapid_x']]).all().all():raise ValueError('Nonfinite hydraulic coefficients')
    return f


def read_receipt(path,key):
    if not path.exists():return None
    r=json.loads(path.read_text())
    if r.get('key')!=key:raise RuntimeError(f'Resume fingerprint mismatch: {path}')
    for p,h in r.get('files',{}).items():
        if digest(p)!=h:raise RuntimeError(f'Resume output mismatch: {p}')
    return r


def solve_wave(frame,flow,dt,initial):
    q,audit=route_interval_means(frame,np.asarray(flow),round(60/dt),dt,initial_flow=initial)
    if not np.isfinite(q).all() or audit['minimum_native_flow_cms']< -1e-9:raise ValueError('Nonfinite or negative routed discharge')
    return q,audit


def assess_calibration_convergence(run,ex,control,best,check):
    change=check['timing_error_minutes']-best['timing_error_minutes']
    peak_rel=abs(check['peak_cms']/best['peak_cms']-1)
    passed=bool(abs(change)<=1.000001 and peak_rel<=.001)
    exception=bool(run==RUNS[1] and ex=='05' and control=='half_dx_half_dt'
        and abs(best['n_manning']-.039486035241004865)<1e-12
        and abs(best['beta']-.5)<1e-12
        and abs(best['timing_error_minutes']+.5)<1e-6
        and abs(check['timing_error_minutes']-1.5)<1e-6 and peak_rel<=.001)
    return dict(timing_change_minutes=change,peak_relative_change=peak_rel,
        thresholds_passed=passed,accepted_exception=exception,accepted=passed or exception,
        review_reason=('User accepted example-05 BASED-midpoint calibration +2 min spatial shift; '
            'single broad crest, stable at 2x/4x/8x; refined target error +1.5 min. '
            'See calibration05_resolution_diagnosis/results.json.' if exception else ''))


def calibrate_task(args):
    run,ex=args;ensure_space(CTX['output']);folder=CTX['output']/run/'calibration'/ex;folder.mkdir(parents=True,exist_ok=True)
    key=stamp([CTX['fingerprint'],run,ex,'calibration']);receipt=folder/'completion.json'
    done=read_receipt(receipt,key)
    if done:return json.loads((folder/'result.json').read_text())
    grid,dt,_=prepare_frame(ex,'S000_base',run);base,_,_=parent_inputs(ex);case=CTX['cases'][ex]
    if run==RUNS[0]:
        f=hydraulic_parameters(grid,run,float(base.n_manning.iloc[0]),float(base.diffusion_multiplier.iloc[0]))
        np.testing.assert_allclose(f[['depth_m','rapid_k','rapid_x','rapid_celerity_mps']],base[['depth_m','rapid_k','rapid_x','rapid_celerity_mps']],rtol=1e-10,atol=1e-10)
        result=dict(example=ex,selected=dict(n_manning=float(base.n_manning.iloc[0]),beta=float(base.diffusion_multiplier.iloc[0])),retained_existing_calibration=True)
    else:
        data=pd.read_parquet(case['calibration_forcing']);meta=next(x for x in json.loads(Path(case['calibration_metadata']).read_text()) if x['method']=='cunge')
        timing=case['timing_target'];height=case['height_target'];goal=pd.Timestamp(timing['target_utc']).timestamp()
        low=height['low_cms'];high=height['high_cms']
        if low is None:low=high=height['height_target_nominal_cms']
        valid=data.downstream_cms.notna().to_numpy();times=data.time_utc.astype('int64').to_numpy()/1e9
        archive=BUNDLE/'source_code/calibrate_common_grid_height.py';node=next(x for x in ast.parse(archive.read_text()).body if isinstance(x,ast.FunctionDef) and x.name=='feature')
        scope={'np':np,'pd':pd,'find_peaks':find_peaks};exec(compile(ast.Module(body=[node],type_ignores=[]),str(archive),'exec'),scope);feature=scope['feature']
        cache={};trials=[];fits={}
        trialpath=folder/'trials.json'
        if trialpath.exists():
            prior=json.loads(trialpath.read_text())
            if prior['key']!=key:raise RuntimeError('Calibration trial fingerprint mismatch')
            trials=prior['trials'];cache={(x['n_manning'],x['beta']):x for x in trials}
        def trial(n,beta,clock=1,subdivision=1,return_wave=False):
            n=float(n);beta=float(beta);pair=(n,beta)
            if clock==subdivision==1 and pair in cache and not return_wave:return cache[pair]
            g=grid.copy();g['count']*=subdivision;g['rapid_effective_length_m']=g.link_length_m/g['count']
            try:
                q,audit=solve_wave(hydraulic_parameters(g,run,n,beta),data.forcing_cms.to_numpy(),dt/clock,meta['initial_cms'])
                tm,h,kind,_=feature(q[valid],times[valid],ex,timing)
                record=dict(n_manning=n,beta=beta,valid=True,timing_error_minutes=(tm-goal)/60,peak_cms=h,
                    height_distance_cms=max(low-h,0.,h-high),feature_kind=kind,**audit)
            except ValueError as error:
                if clock!=1 or subdivision!=1 or return_wave:raise
                record=dict(n_manning=n,beta=beta,valid=False,error=str(error))
            if clock==subdivision==1 and pair not in cache:
                cache[pair]=record;trials.append(record);write_json(trialpath,dict(key=key,trials=trials))
                print(f'{run} {ex} trial n={n:.7f} beta={beta:g}: '+str(record.get('timing_error_minutes',record.get('error'))),flush=True)
            return (record,q) if return_wave else record
        def fit_beta(beta):
            if beta in fits:return fits[beta]
            ns=sorted(set([.01,.0175,.025,.04,.07,.1,float(base.n_manning.iloc[0])]+[r['n_manning'] for r in fits.values()]))
            candidates=[trial(n,beta) for n in ns]
            for a,b in zip(ns[:-1],ns[1:]):
                ra,rb=trial(a,beta),trial(b,beta)
                if not ra['valid'] or not rb['valid'] or ra['timing_error_minutes']*rb['timing_error_minutes']>=0:continue
                for _ in range(18):
                    ea,eb=ra['timing_error_minutes'],rb['timing_error_minutes'];n=a+float(np.clip(-ea/(eb-ea),.15,.85))*(b-a)
                    r=trial(n,beta);candidates.append(r)
                    if not r['valid']:break
                    if abs(r['timing_error_minutes'])<=.5:break
                    if ea*r['timing_error_minutes']<=0:b=n;rb=r
                    else:a=n;ra=r
            validc=[r for r in candidates if r['valid']]
            if not validc:raise RuntimeError(f'No valid calibration candidates: {run}/{ex}/beta={beta}')
            best=min(validc,key=lambda r:(abs(r['timing_error_minutes']),abs(r['n_manning']-float(base.n_manning.iloc[0]))))
            fits[beta]=best;pd.DataFrame(fits.values()).to_csv(folder/'beta_profile.csv',index=False);return best
        baseline=fit_beta(1.)
        for beta in [.75,.5,1.5,2.]:fit_beta(beta)
        feasible=[r for r in fits.values() if abs(r['timing_error_minutes'])<=1 and r['height_distance_cms']==0]
        if feasible and baseline['height_distance_cms']>0:
            success=min(feasible,key=lambda r:abs(math.log(r['beta'])))['beta'];failed=1.
            for _ in range(3):
                beta=math.sqrt(success*failed);r=fit_beta(beta)
                if abs(r['timing_error_minutes'])<=1 and r['height_distance_cms']==0:success=beta
                else:failed=beta
        eligible=[r for r in fits.values() if abs(r['timing_error_minutes'])<=1]
        if not eligible:raise RuntimeError(f'Timing tolerance not met for {run}/{ex}; inspect calibration trials; batch stopped.')
        best=min(eligible,key=lambda r:(r['height_distance_cms'],abs(math.log(r['beta'])),abs(r['timing_error_minutes'])))
        _,native=trial(best['n_manning'],best['beta'],return_wave=True);checks=[]
        for name,clock,sub in [('half_dt',2,1),('half_dx_half_dt',2,2)]:
            check,q=trial(best['n_manning'],best['beta'],clock,sub,return_wave=True)
            review=assess_calibration_convergence(run,ex,name,best,check)
            checks.append(dict(control=name,**review,max_wave_change_cms=float(abs(q-native).max())))
            write_json(folder/'numerical_checks.json',checks)
            if not review['accepted']:raise RuntimeError(f'Calibration convergence check failed for {run}/{ex}: {name}; inspect numerical_checks.json')
        wave=data.copy();wave['selected_outlet_cms']=native;wave.to_parquet(folder/'waveforms.parquet',index=False)
        result=dict(example=ex,selected=best,beta1=baseline,numerical_checks=checks,timing_tolerance_met=True,
            soft_height_target_met=best['height_distance_cms']==0,trials=len(trials),native_dt_seconds=dt,
            independent_validation=False,reference_Qref_cms=float(CTX['values'].loc[ex,'midpoint_Qref_cms']))
    write_json(folder/'result.json',result)
    write_json(receipt,dict(key=key,files={str(p):digest(p) for p in folder.iterdir() if p.is_file() and p.name!='completion.json'}))
    return result


def metrics_for(ex,state,aliases,force,q):
    t=force.time_utc.astype('int64').to_numpy()/1e9
    windows=CTX['windows'][CTX['windows'].example.eq(ex)]
    canonical='fully_collapsed' if state=='single_centerline' else state
    rows=[]
    for _,w in windows.iterrows():
        r=state_metrics(t,q,w,canonical,aliases);r['example']=ex;rows.append(r)
    return pd.DataFrame(rows)


def route_state(args):
    run,ex,state,fit=args;ensure_space(CTX['output']);folder=CTX['output']/run/'states'/ex/state;folder.mkdir(parents=True,exist_ok=True)
    key=stamp([CTX['fingerprint'],run,ex,state,fit]);receipt=folder/'completion.json'
    done=read_receipt(receipt,key)
    if done:return done
    grid,dt,job=prepare_frame(ex,state,run);frame=hydraulic_parameters(grid,run,fit['n_manning'],fit['beta'])
    case=CTX['cases'][ex];force=pd.read_parquet(case['forcing']);nb=int(force.phase.eq('spinup').sum());initial=float(force.forcing_cms.iloc[0])
    final=case['final_cms'];hold_start=force.loc[force.phase.eq('settling'),'interval_start_utc'].iloc[0];extensions=0
    while True:
        q,audit=solve_wave(frame,force.forcing_cms.to_numpy()[nb:],dt,initial)
        required,settled=settling_hours(force.time_utc.iloc[nb:],q,hold_start,final,case['tolerance_cms'])
        if settled:break
        if extensions>=5:raise RuntimeError(f'Settling failed: {run}/{ex}/{state}')
        ensure_space(CTX['output']);extra=max(96,int(math.ceil(required-(force.interval_end_utc.iloc[-1]-hold_start).total_seconds()/3600))+48)
        origin=force.interval_end_utc.iloc[-1];count=extra*60
        tail=pd.DataFrame(dict(interval_start_utc=origin+pd.to_timedelta(np.arange(count)*60,unit='s'),
            interval_end_utc=origin+pd.to_timedelta((np.arange(count)+1)*60,unit='s'),time_utc=origin+pd.to_timedelta((np.arange(count)+.5)*60,unit='s'),forcing_cms=final,phase='settling'))
        force=pd.concat([force,tail],ignore_index=True);extensions+=1
    q=np.r_[np.full(nb,initial),q];metrics=metrics_for(ex,state,job['aliases'],force,q)
    outlet=force[['time_utc','interval_start_utc','interval_end_utc','phase']].copy();outlet['q_outlet_cms']=q
    frame.to_parquet(folder/'parameters.parquet',index=False);outlet.to_parquet(folder/'outlet.parquet',index=False);metrics.to_parquet(folder/'metrics.parquet',index=False)
    record=dict(key=key,example=ex,state=state,native_dt_seconds=dt,n=fit['n_manning'],beta=fit['beta'],settled=True,tail_extensions=extensions,
                required_tail_hours=required,**audit,files={str(p):digest(p) for p in [folder/'parameters.parquet',folder/'outlet.parquet',folder/'metrics.parquet']})
    write_json(receipt,record);return record


def check_state(args):
    run,ex,state=args;folder=CTX['output']/run/'states'/ex/state
    force=pd.read_parquet(CTX['cases'][ex]['forcing']);nb=int(force.phase.eq('spinup').sum());initial=float(force.forcing_cms.iloc[0])
    f=pd.read_parquet(folder/'parameters.parquet');receipt=json.loads((folder/'completion.json').read_text());dt=receipt['native_dt_seconds']
    native=pd.read_parquet(folder/'metrics.parquet').set_index('window');out=[]
    for label,sub in [('half_dt',1),('half_dx_half_dt',2)]:
        g=f.copy();g['count']*=sub;g['rapid_effective_length_m']=g.link_length_m/g['count'];g=hydraulic_parameters(g,run,receipt['n'],receipt['beta'])
        q,_=solve_wave(g,force.forcing_cms.to_numpy()[nb:],dt/2,initial)
        m=convergence_metrics(ex,state,force,np.r_[np.full(nb,initial),q],CTX['windows'][CTX['windows'].example.eq(ex)]).set_index('window')
        for window in m.index:
            out.append(dict(example=ex,state=state,control=label,window=window,
                centroid_change_seconds=float(m.loc[window,'centroid_seconds']-native.loc[window,'centroid_seconds']),
                peak_change_seconds=float(m.loc[window,'peak_seconds']-native.loc[window,'peak_seconds']),
                peak_relative_change=float(abs(m.loc[window,'peak_cms']/native.loc[window,'peak_cms']-1)),
                minimum_excess_cms=float(m.loc[window,'minimum_excess_cms']),
                negative_excess_volume_m3=float(m.loc[window,'negative_excess_volume_m3']),
                negative_volume_fraction=float(m.loc[window,'negative_volume_fraction']),
                diagnostic_method=m.loc[window,'diagnostic_method']))
    return out


def pool_tasks(function,tasks,workers,output,fingerprint,label):
    results=[]
    with ProcessPoolExecutor(max_workers=workers,initializer=init_worker,initargs=(str(output),fingerprint)) as pool:
        futures=[pool.submit(function,t) for t in tasks]
        try:
            for future in as_completed(futures):
                results.append(future.result())
                if len(results)%10==0 or len(results)==len(tasks):print(f'{label}: {len(results)}/{len(tasks)}',flush=True)
        except BaseException:
            for f in futures:f.cancel()
            raise
    return results


def assess_convergence(run,checks):
    """Keep raw failures visible; apply only the specifically reviewed exception."""
    checks=checks.copy()
    checks['required_for_acceptance']=~checks.state.isin(['single_centerline','fully_collapsed'])
    checks['centroid_passed']=checks.centroid_contrast_change_seconds.abs().le(60)
    checks['peak_timing_passed']=checks.peak_contrast_change_seconds.abs().le(60)
    checks['peak_height_passed']=checks.peak_relative_change.le(.001)
    checks['thresholds_passed']=checks[['centroid_passed','peak_timing_passed','peak_height_passed']].all(axis=1)
    # Approval applies to the measured +180 s spatial check in run 1 only.
    # New failures in midpoint runs, other controls, or other metrics still stop.
    checks['accepted_exception']=(
        (run==RUNS[0]) & checks.example.eq('54') & checks.state.eq('all_closed_units_collapsed')
        & checks.control.eq('half_dx_half_dt') & checks.window.isin(['0.5%','1%','2%'])
        & checks.peak_contrast_change_seconds.sub(180).abs().le(1e-6)
        & checks.centroid_passed & checks.peak_height_passed)
    checks['review_reason']=''
    checks.loc[~checks.required_for_acceptance,'review_reason']='Unused non-segmented single-centreline control; diagnostics retained, excluded from required gate.'
    checks.loc[checks.accepted_exception,'review_reason']='User accepted run-1 example-54 +180 s peak contrast discrepancy; broad crests, stable dominant lobes, 4x/8x convergence verified. See peak_resolution_diagnosis/diagnosis.json.'
    checks['accepted']=~checks.required_for_acceptance | checks.thresholds_passed | checks.accepted_exception
    required=checks[checks.required_for_acceptance]
    summary=dict(convergence_checks_passed=bool(required.thresholds_passed.all()),
        convergence_accepted=bool(checks.accepted.all()),
        excluded_diagnostic_rows=int((~checks.required_for_acceptance).sum()),
        accepted_exception_rows=int(checks.accepted_exception.sum()),
        unresolved_rows=int((~checks.accepted).sum()))
    return checks,summary


def finish_stage(run,output,fingerprint,workers,plan):
    folder=output/run
    frames=[pd.read_parquet(folder/'states'/c['example']/j['state']/'metrics.parquet') for c in plan['cases'] for j in c['jobs']]
    states=pd.concat(frames,ignore_index=True);assert len(states)==2215*3 and not states.duplicated(['example','state','window']).any()
    responses=response_rows(states);states.to_parquet(folder/'state_metrics.parquet',index=False);responses.to_csv(folder/'responses.csv',index=False)
    # Parent, both full-network controls, and centroid extrema for each network.
    representatives=set()
    for c in plan['cases']:
        ex=c['example'];d=responses[(responses.example==ex)&(responses.window=='0.5%')]
        for state in ['S000_base','all_closed_units_collapsed','single_centerline',d.loc[d.centroid_shift_hours.idxmin(),'state'],d.loc[d.centroid_shift_hours.idxmax(),'state']]:
            representatives.add((run,ex,'single_centerline' if state=='fully_collapsed' else state))
    checks=pool_tasks(check_state,sorted(representatives),workers,output,fingerprint,run+' convergence')
    checks=pd.DataFrame([r for group in checks for r in group]);checks.to_csv(folder/'convergence_checks.csv',index=False)
    # A run is review-required, not complete, if its extrema depend materially on resolution.
    parent=checks[checks.state.eq('S000_base')][['example','control','window','centroid_change_seconds','peak_change_seconds']]
    checks=checks.merge(parent,on=['example','control','window'],suffixes=('','_parent'),validate='many_to_one')
    checks['centroid_contrast_change_seconds']=checks.centroid_change_seconds-checks.centroid_change_seconds_parent
    checks['peak_contrast_change_seconds']=checks.peak_change_seconds-checks.peak_change_seconds_parent
    checks,review=assess_convergence(run,checks)
    checks.to_csv(folder/'convergence_checks.csv',index=False)
    write_json(folder/'convergence_review.json',review)
    if not review['convergence_accepted']:raise RuntimeError(f'{run}: convergence review required; inspect convergence_checks.csv. Subsequent runs not started.')
    write_json(folder/'completion.json',dict(status='complete',fingerprint=fingerprint,states=2215,response_rows=len(responses),
        **review,files={str(folder/n):digest(folder/n) for n in ['state_metrics.parquet','responses.csv','convergence_checks.csv','convergence_review.json']}))


def review_diagnostic_compatibility(old,sources,output):
    path=SETUP/'convergence_baseline_diagnosis/diagnostic_compatibility.json'
    if not path.exists():raise RuntimeError('Sources changed; no reviewed compatibility record exists')
    review=json.loads(path.read_text())
    if old['fingerprint']!=review['original_fingerprint'] or stamp(sources)!=review['updated_source_fingerprint']:
        raise RuntimeError('Source change is not the reviewed diagnostic-only update')
    if old['sources']!=review['original_sources']:
        raise RuntimeError('Original batch provenance does not match reviewed update')
    for p,h in review['review_artifacts'].items():
        if digest(p)!=h:raise RuntimeError('Diagnostic review artifact changed: '+p)
    # Keep data and parameter receipts immutable, and verify their complete files.
    for receipt in output.glob('*/states/*/*/completion.json'):
        r=json.loads(receipt.read_text())
        for p,h in r['files'].items():
            if digest(p)!=h:raise RuntimeError('Saved state changed: '+p)
    for receipt in output.glob('*/calibration/*/completion.json'):
        r=json.loads(receipt.read_text())
        for p,h in r['files'].items():
            if digest(p)!=h:raise RuntimeError('Saved calibration changed: '+p)
    return dict(review_path=str(path),review_sha256=digest(path),
                updated_source_fingerprint=stamp(sources),kind='resolution diagnostic only; production routing and metrics unchanged')


def run_batch(output,workers,sources):
    output.mkdir(parents=True,exist_ok=True)
    with (output/'batch.lock').open('a+') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise RuntimeError('Another launcher is using this output directory')
        fingerprint=stamp(sources);manifest=output/'manifest.json'
        if manifest.exists():
            old=json.loads(manifest.read_text())
            if old['fingerprint']!=fingerprint:
                review=review_diagnostic_compatibility(old,sources,output)
                fingerprint=old['fingerprint']
                write_json(manifest,dict(old,current_sources=sources,diagnostic_revision=review))
                print('Verified diagnostic-only update; retaining saved calibration/state receipts.',flush=True)
        else:
            if any(p.name not in ['batch.lock','run.log'] for p in output.iterdir()):raise RuntimeError('Output directory contains unidentified results')
            ensure_space(output,40);write_json(manifest,dict(fingerprint=fingerprint,sources=sources,runs=RUNS,created_utc=pd.Timestamp.now(tz='UTC').isoformat(),
                numerical_acceptance=dict(max_timing_or_centroid_contrast_change_seconds=60,max_relative_peak_change=.001)))
        init_worker(output,fingerprint);plan=dict(cases=list(CTX['cases'].values()))
        for run in RUNS:
            # Recheck protected inputs between stages, including resume.
            for p,h in sources['files'].items():
                if digest(p)!=h:raise RuntimeError('Source changed during batch: '+p)
            ensure_space(output);print('\nSTART '+run,flush=True)
            receipt=output/run/'completion.json'
            if receipt.exists():
                done=json.loads(receipt.read_text())
                if done.get('fingerprint')!=fingerprint:raise RuntimeError('Stage fingerprint mismatch')
                for p,h in done['files'].items():
                    if digest(p)!=h:raise RuntimeError('Stage output mismatch: '+p)
                print('Verified completed stage; skipping '+run,flush=True);continue
            fits=pool_tasks(calibrate_task,[(run,c['example']) for c in plan['cases']],workers,output,fingerprint,run+' calibration')
            byex={f['example']:f['selected'] for f in fits}
            tasks=[(run,c['example'],j['state'],byex[c['example']]) for c in plan['cases'] for j in c['jobs']]
            pool_tasks(route_state,tasks,workers,output,fingerprint,run+' states')
            finish_stage(run,output,fingerprint,workers,plan)
            print('COMPLETE '+run,flush=True)
        keys=['example','state','window'];fields=['centroid_shift_hours','peak_shift_hours','peak_excess_change_pct','duration80_change_pct','asymmetry_change','volume_change_pct'];comparison=None
        for i,run in enumerate(RUNS,1):
            d=pd.read_csv(output/run/'responses.csv',dtype={'example':str})[keys+fields].rename(columns={f:f+'_run'+str(i) for f in fields})
            comparison=d if comparison is None else comparison.merge(d,on=keys,validate='one_to_one')
        comparison.to_csv(output/'three_run_comparison.csv',index=False)
        write_json(output/'completion.json',dict(status='complete',fingerprint=fingerprint,runs=RUNS,states_per_run=2215,
            comparison_sha256=digest(output/'three_run_comparison.csv')))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workers',type=int,default=4)
    parser.add_argument('--output',type=Path,default=Path('/Volumes/PhD/river_hierarchy/three_run_comparison_20261028'))
    parser.add_argument('--run',action='store_true',help='Execute all three stages sequentially; otherwise verify only')
    args=parser.parse_args()
    if not 1<=args.workers<=8:parser.error('--workers must be 1..8 local processes')
    output=args.output.expanduser().resolve();parent=output
    while not parent.exists():parent=parent.parent
    if str(output).startswith('/Volumes/PhD/') and not Path('/Volumes/PhD').is_mount():raise RuntimeError('PhD drive is not mounted')
    sources=verify_inputs();free=ensure_space(parent,20 if (output/'manifest.json').exists() else 40)
    print(f'Verified preflight and frozen inputs. {free:.1f} GiB free. Workers={args.workers}; BLAS threads=1. Output={output}',flush=True)
    if args.run:run_batch(output,args.workers,sources)
    else:print('Check only: no calibration/routing started. Add --run to execute.',flush=True)

if __name__=='__main__':main()
