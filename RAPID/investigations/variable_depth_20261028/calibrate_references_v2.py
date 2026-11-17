"""Separate resumable 15-reference variable-depth calibration; existing grids only.

Search/targets match the fixed-depth batch. Failures are recorded per example.
Calibration acceptance covers timing and half-dt only, not spatial convergence.
No depth clipping, no slope changes, no collapse routing, no automatic exceptions.
"""
import os
for key in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[key]='1'
import ast,json,math,fcntl,traceback
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from prototype import r,DEST,HERE,engine,route
from run_three_runs import (stamp,read_receipt,write_json,digest,prepare_frame,parent_inputs,hydraulic_parameters,RUNS,BUNDLE)
CTX=None
FUN=None
RUN=RUNS[2]
OUTPUT=DEST/'reference_calibration_v2'
LABEL='04_manning_variable_depth'

def ensure_space(path):return r.ensure_space(path,5)

def solve_wave(frame,flow,dt,initial):
    try:
        q,a,links=route(FUN,frame,flow,dt,initial,float(frame.diffusion_multiplier.iloc[0]))
        if a['residual_pct_excess_volume'] is not None and abs(a['residual_pct_excess_volume'])>1e-5:
            raise ValueError('Mass balance residual exceeds 1e-5 percent of excess input volume')
        return q,dict(a,depth_width_ratio_max=float(links.depth_width_ratio_max.max()))
    except RuntimeError as e:raise ValueError(str(e)) from e

def assess_calibration_convergence(run,ex,name,best,check):
    change=check['timing_error_minutes']-best['timing_error_minutes']
    peak=abs(check['peak_cms']/best['peak_cms']-1)
    passed=abs(change)<=1.000001 and peak<=.001
    return dict(timing_change_minutes=change,peak_relative_change=peak,accepted=bool(passed),thresholds_passed=bool(passed),accepted_exception=False)

def calibrate_task(args):
    run,ex=args;ensure_space(CTX['output']);folder=CTX['output']/LABEL/'calibration'/ex;folder.mkdir(parents=True,exist_ok=True)
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
                print(f'{LABEL} {ex} trial n={n:.7f} beta={beta:g}: '+str(record.get('timing_error_minutes',record.get('error'))),flush=True)
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
            if not validc:
                write_json(folder/f'invalid_beta_{beta:g}.json',dict(beta=beta,reason='No valid candidates; continue other beta values',trials=len(candidates)))
                print(f'{LABEL} {ex}: skipping beta={beta:g}, no valid candidates',flush=True)
                return None
            best=min(validc,key=lambda r:(abs(r['timing_error_minutes']),abs(r['n_manning']-float(base.n_manning.iloc[0]))))
            fits[beta]=best;pd.DataFrame(fits.values()).to_csv(folder/'beta_profile.csv',index=False);return best
        baseline=fit_beta(1.)
        for beta in [.75,.5,1.5,2.]:fit_beta(beta)
        feasible=[r for r in fits.values() if abs(r['timing_error_minutes'])<=1 and r['height_distance_cms']==0]
        if feasible and (baseline is None or baseline['height_distance_cms']>0):
            success=min(feasible,key=lambda r:abs(math.log(r['beta'])))['beta'];failed=1.
            for _ in range(3):
                beta=math.sqrt(success*failed);r=fit_beta(beta)
                if r is not None and abs(r['timing_error_minutes'])<=1 and r['height_distance_cms']==0:success=beta
                else:failed=beta
        if not fits:raise RuntimeError(f'No valid calibration candidates at any tested beta for {ex}')
        eligible=[r for r in fits.values() if abs(r['timing_error_minutes'])<=1]
        timing_met=bool(eligible)
        if not eligible:eligible=list(fits.values())
        best=min(eligible,key=lambda r:((0 if timing_met else abs(r['timing_error_minutes'])),r['height_distance_cms'],abs(math.log(r['beta'])),abs(r['timing_error_minutes'])))
        _,native=trial(best['n_manning'],best['beta'],return_wave=True);checks=[]
        for name,clock,sub in [('half_dt',2,1)]:
            check,q=trial(best['n_manning'],best['beta'],clock,sub,return_wave=True)
            review=assess_calibration_convergence(run,ex,name,best,check)
            checks.append(dict(control=name,**review,max_wave_change_cms=float(abs(q-native).max())))
            write_json(folder/'numerical_checks.json',checks)

        wave=data.copy();wave['selected_outlet_cms']=native;wave.to_parquet(folder/'waveforms.parquet',index=False)
        result=dict(example=ex,selected=best,beta1=baseline,numerical_checks=checks,timing_tolerance_met=timing_met,
            calibration_accepted=timing_met and all(x['accepted'] for x in checks),spatial_convergence_certified=False,
            soft_height_target_met=best['height_distance_cms']==0,trials=len(trials),native_dt_seconds=dt,
            independent_validation=False,reference_Qref_cms=float(CTX['values'].loc[ex,'midpoint_Qref_cms']))
    write_json(folder/'result.json',result)
    write_json(receipt,dict(key=key,files={str(p):digest(p) for p in folder.iterdir() if p.is_file() and p.name!='completion.json'}))
    return result

def summarize():
    rows=[]
    for ex in sorted(CTX['cases']):
        folder=OUTPUT/LABEL/'calibration'/ex
        if (folder/'completion.json').exists():
            a=json.loads((folder/'result.json').read_text());s=a['selected']
            rows.append(dict(example=ex,status='accepted' if a['calibration_accepted'] else 'review_required',**s,
                timing_tolerance_met=a['timing_tolerance_met'],soft_height_target_met=a['soft_height_target_met'],
                half_dt_passed=all(x['accepted'] for x in a['numerical_checks']),native_dt_seconds=a['native_dt_seconds'],trials=a['trials']))
        elif (folder/'failure.json').exists():
            a=json.loads((folder/'failure.json').read_text());rows.append(dict(example=ex,status='failed',error=a['error']))
        else:rows.append(dict(example=ex,status='pending'))
    pd.DataFrame(rows).to_csv(OUTPUT/'summary.csv',index=False)
    return rows

def main():
    global CTX,FUN
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument('--examples',nargs='+',default=['02','05','15','22','41','46'])
    args=parser.parse_args()
    OUTPUT.mkdir(exist_ok=True);ensure_space(OUTPUT)
    with (OUTPUT/'calibration.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        sources=r.verify_inputs()
        for p in [HERE/'conservative_v2.cpp',HERE/'prototype.py',Path(__file__),DEST/'hydraulic_envelope.csv']:
            sources['files'][str(p.resolve())]=digest(p)
        fingerprint=stamp(sources);manifest=OUTPUT/'manifest.json'
        if manifest.exists():
            if json.loads(manifest.read_text())['fingerprint']!=fingerprint:raise RuntimeError('Calibration source/input fingerprint changed')
        else:write_json(manifest,dict(fingerprint=fingerprint,sources=sources,solver='conservative_variable_depth',grid='existing shared midpoint',workers=1,
            run_directory_label=LABEL,n_bounds=[.01,.1],beta_bounds=[.5,2.],depth_caps=False,spatial_refinement=False))
        r.init_worker(OUTPUT,fingerprint);CTX=r.CTX
        envelope=pd.read_csv(DEST/'hydraulic_envelope.csv',dtype={'example':str}).set_index(['example','state'])
        for ex in CTX['cases']:
            CTX['midpoint'].loc[(ex,'S000_base'),'shared_dt_seconds']=float(envelope.loc[(ex,'S000_base'),'envelope_dt_seconds'])
        FUN=engine('conservative_v2.cpp')
        for ex in sorted(args.examples):
            if ex not in CTX['cases']:raise ValueError('Unknown example '+ex)
            for p,h in sources['files'].items():
                if digest(p)!=h:raise RuntimeError('Source/input changed: '+p)
            print('START VARIABLE CALIBRATION '+ex,flush=True)
            folder=OUTPUT/LABEL/'calibration'/ex
            try:
                if (folder/'failure.json').exists():(folder/'failure.json').unlink()
                result=calibrate_task((RUN,ex))
                print('FINISHED '+ex+' accepted='+str(result['calibration_accepted']),flush=True)
            except Exception as error:
                write_json(folder/'failure.json',dict(error=str(error),traceback=traceback.format_exc()))
                print('FAILED '+ex+': '+str(error),flush=True)
            summarize()
        rows=summarize()
        write_json(OUTPUT/'batch_summary.json',dict(status='selected_references_attempted',selected_examples=args.examples,accepted=sum(x['status']=='accepted' for x in rows),
            review_required=sum(x['status']=='review_required' for x in rows),failed=sum(x['status']=='failed' for x in rows),examples=len(rows),
            ready_for_full_collapse_batch=False,depth_review_required=True))
if __name__=='__main__':main()
