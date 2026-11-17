"""Versioned base-only beta/n fits with reviewed timing and manual soft heights."""
from pathlib import Path
import sys,json,hashlib,shutil,os,time,math
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
import pandas as pd
from scipy.signal import find_peaks
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'RAPID/src')]
from rapid_tools import cunge
from rapid_tools.common_grid import route_interval_means
SOURCE=Path(json.loads((ROOT/'RAPID/outputs/common_grid_validation_20261008/latest.json').read_text())['directory'])
TARGET_PATH=ROOT/'RAPID/reviews/common_grid_production_20261008/manual_height_targets_proposal.json'
TIMING_PATH=SOURCE/'baseline/RAPID/reviews/calibration_targets_20261006/targets.json'
PROTOCOL=dict(beta_bounds=[.01,2.],n_bounds=[.01,.1],timing_tolerance_minutes=1.,point_height_numerical_tolerance_fraction=.001,initialization='equilibrium at original event start; no antecedent substitution',height='shifted downstream selected crest for 03/46/54; recorded manual intervals otherwise; proposed +/-1% current bands used for 13/37/48',selection='Timing gate first; minimize relative height distance outside interval (point distance for nominal targets). Within an interval prefer beta nearest 1; point searches stop at 0.1% numerical residual, not an observational acceptance band.',feature='54: midpoint of two highest distinct prominent lobes, separation >= one-third reviewed crest span. If merged, use single crest and flag. Other cases: midpoint of contiguous exact maximum; 48 uses middle chronological equal-maximum sample. Smoothed single crests represent observed short plateaus/crest groups; no observed target changes.',production_modified=False,scope='15 retained base Cunge examples only; fixed-X comparator unchanged; no hierarchy reruns')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def feature(q,t,ex,review):
    maximum=float(np.max(q));tol=max(np.ptp(q)*1e-10,1e-9)
    top=np.flatnonzero(abs(q-maximum)<=tol)
    if ex=='54':
        span=(pd.Timestamp(review['crest_end_utc'])-pd.Timestamp(review['crest_start_utc'])).total_seconds()/60
        peaks,_=find_peaks(q,prominence=max(np.ptp(q)*1e-4,1e-8),distance=max(1,int(span/3)))
        peaks=peaks[q[peaks]>=q.min()+.8*np.ptp(q)]
        best=peaks[np.argsort(q[peaks])[-2:]]
        if len(best)==2:
            best=np.sort(best)
            return float((t[best[0]]+t[best[1]])/2),float(max(q[best])), 'two_lobe_midpoint', [float(t[i]) for i in best]
        return float(t[top[0]]),maximum,'merged_or_unresolved_two_lobes',[float(t[top[0]])]
    if ex=='48':idx=top[len(top)//2];return float(t[idx]),maximum,'middle_equal_maximum',[float(t[idx])]
    # First contiguous numerically exact maximum block; a rounded crest is one bin.
    end=0
    while end+1<len(top) and top[end+1]==top[end]+1:end+=1
    center=float((t[top[0]]+t[top[end]])/2)
    return center,maximum,'exact_plateau_midpoint' if end else 'single_smoothed_crest',[float(t[top[0]]),float(t[top[end]])]

def calibrate(ex,dest_string, *, beta_bounds=(.01,2.), previous_selection=None):
    dest=Path(dest_string);out=dest/ex;out.mkdir(exist_ok=True)
    target=next(r for r in json.loads((dest/'height_targets.json').read_text())['records'] if r['example']==ex)
    review=next(r for r in json.loads((dest/'timing_targets.json').read_text())['records'] if r['example']==ex)
    comp=pd.read_csv(SOURCE/'baseline/RAPID/outputs/cunge_calibration_reviewed_20261006/comparison.csv',dtype={'example':str})
    row=comp[(comp.example==ex)&(comp.method=='cunge')].iloc[0]
    folder=SOURCE/'baseline'/Path(row.output_dir).relative_to(ROOT)
    grid=pd.read_parquet(folder/'frozen_grid.parquet')
    subdivision=(previous_selection or {}).get('subdivision_multiplier',1)
    grid['count']*=subdivision;grid['rapid_effective_length_m']=grid.link_length_m/grid['count']
    data=pd.read_parquet(SOURCE/ex/'60s.parquet')
    meta=next(r for r in json.loads((SOURCE/ex/'60s.json').read_text()) if r['method']=='cunge')
    valid=data.downstream_cms.notna().to_numpy();ids=np.flatnonzero(valid)
    t=(data.time_utc.astype('int64').to_numpy()/1e9);tt=t[valid]
    goal=pd.Timestamp(review['target_utc']).timestamp();q0=meta['initial_cms'];dt=(previous_selection or {}).get('native_dt_s',meta['dt_s']);orig_n=meta['n_manning']
    lower=target['low_cms'];upper=target['high_cms'];is_point=lower is None
    if is_point:lower=upper=target['height_target_nominal_cms']
    scale=max((lower+upper)/2,1e-12)
    records=[];cache={};beta_fits={};errors=[]
    def run(beta,n,keep=False):
        beta=float(beta);n=float(n)
        if not beta_bounds[0]<=beta<=beta_bounds[1]:raise ValueError('Beta outside declared search bounds')
        key=(beta,n)
        if key in cache and not keep:return cache[key]
        f=cunge.parameters(grid,n,method='cunge',beta=beta)
        try:
            q,audit=route_interval_means(f,data.forcing_cms.to_numpy(),int(round(60/dt)),dt,initial_flow=q0)
            if audit['minimum_native_flow_cms'] < -1e-8:raise ValueError('Negative native discharge')
            tm,height,kind,lobes=feature(q[valid],tt,ex,review)
            distance=max(lower-height,0.,height-upper)
            rec=dict(example=ex,beta=beta,n_manning=n,feature_utc=pd.Timestamp(tm,unit='s',tz='UTC').isoformat(),timing_error_minutes=(tm-goal)/60,peak_cms=height,height_distance_cms=distance,height_distance_fraction=distance/scale,height_signed_residual=(height-(lower if height<lower else upper if height>upper else height))/scale,feature_kind=kind,feature_lobes_utc=[pd.Timestamp(x,unit='s',tz='UTC').isoformat() for x in lobes],minimum_native_flow_cms=audit['minimum_native_flow_cms'])
            cache[key]=rec
            if not keep:records.append(rec)
            return (rec,q) if keep else rec
        except ValueError as e:
            errors.append(dict(beta=beta,n=n,error=str(e)));return None
    def fit_beta(beta):
        beta=float(beta)
        if beta in beta_fits:return beta_fits[beta]
        # Coarse n scan catches peak switches; refine every sign-changing bracket.
        ns=np.unique(np.r_[.01,.0175,.025,.04,.07,.1,orig_n,(previous_selection or {}).get('n_manning',orig_n)])
        if beta_fits:
            nearest=min(beta_fits,key=lambda b:abs(math.log(b/beta)))
            ns=np.unique(np.r_[ns,beta_fits[nearest]['n_manning']])
        candidates=[]
        for n in ns:
            rec=run(beta,n)
            if rec is not None:candidates.append(rec)
        for a,b in zip(ns[:-1],ns[1:]):
            a=float(a);b=float(b);ra=run(beta,a);rb=run(beta,b)
            if ra is None or rb is None or ra['timing_error_minutes']*rb['timing_error_minutes']>=0:continue
            for iteration in range(18):
                ea=ra['timing_error_minutes'];eb=rb['timing_error_minutes']
                # Safeguarded secant: never assume smoothness through peak switches.
                frac=np.clip(-ea/(eb-ea),.15,.85);n=float(a+frac*(b-a));r=run(beta,n)
                if r is None:break
                candidates.append(r)
                if abs(r['timing_error_minutes'])<=.5:break
                if ea*r['timing_error_minutes']<=0:b=n;rb=r
                else:a=n;ra=r
        if not candidates:raise RuntimeError(f'No admissible trial for {ex} beta={beta}')
        best=min(candidates,key=lambda r:(abs(r['timing_error_minutes']),abs(r['n_manning']-orig_n)))
        beta_fits[beta]=best
        pd.DataFrame(records).to_json(out/'trials.json',orient='records',indent=2)
        print(ex,'beta',round(beta,6),'n',round(best['n_manning'],7),'time',round(best['timing_error_minutes'],3),'height',round(best['peak_cms'],3),flush=True)
        return best
    baseline=fit_beta(1.)
    need=baseline['height_distance_fraction']>(.001 if is_point else 0) or abs(baseline['timing_error_minutes'])>1
    if need:
        # Explore both directions when needed; stop intervals at first feasible beta.
        order=[.75,.5,.25,.1,.05,.01,1.5,2.] if baseline['peak_cms']<lower else [1.5,2.,.75,.5,.25,.1,.05,.01]
        order=[b for b in order if beta_bounds[0]<=b<=beta_bounds[1]]
        for beta in order:
            r=fit_beta(beta)
            feasible=[x for x in beta_fits.values() if abs(x['timing_error_minutes'])<=1]
            if not is_point and any(x['height_distance_cms']==0 for x in feasible):break
            signed=[x['peak_cms']-lower for x in feasible]
            if is_point and signed and min(signed)<=0<=max(signed):break
            if is_point and r['height_distance_fraction']<=.001 and abs(r['timing_error_minutes'])<=1:break
        if is_point:
            ordered=sorted(beta_fits)
            brackets=[(a,b) for a,b in zip(ordered[:-1],ordered[1:]) if abs(beta_fits[a]['timing_error_minutes'])<=1 and abs(beta_fits[b]['timing_error_minutes'])<=1 and (beta_fits[a]['peak_cms']-lower)*(beta_fits[b]['peak_cms']-lower)<0]
            for a,b in brackets:
                for _ in range(9):
                    if min(x['height_distance_fraction'] for x in beta_fits.values() if abs(x['timing_error_minutes'])<=1)<=.001:break
                    beta=math.sqrt(a*b);r=fit_beta(beta)
                    if abs(r['timing_error_minutes'])>1:break
                    if (beta_fits[a]['peak_cms']-lower)*(r['peak_cms']-lower)<=0:b=beta
                    else:a=beta
        else:
            # Refine the first successful change towards beta=1, avoiding gratuitous attenuation changes.
            feasible=[x for x in beta_fits.values() if abs(x['timing_error_minutes'])<=1 and x['height_distance_cms']==0]
            if feasible:
                success=min(feasible,key=lambda r:abs(math.log(r['beta'])))['beta'];failed=1.
                for _ in range(3):
                    beta=math.sqrt(success*failed);r=fit_beta(beta)
                    if abs(r['timing_error_minutes'])<=1 and r['height_distance_cms']==0:success=beta
                    else:failed=beta
    feasible=[r for r in beta_fits.values() if abs(r['timing_error_minutes'])<=1]
    pool=feasible or list(beta_fits.values())
    if feasible:
        if is_point:
            close=[r for r in pool if r['height_distance_fraction']<=.001]
            best=min(close,key=lambda r:abs(math.log(r['beta']))) if close else min(pool,key=lambda r:(r['height_distance_fraction'],abs(math.log(r['beta']))))
        else:best=min(pool,key=lambda r:(r['height_distance_fraction'],abs(math.log(r['beta'])),abs(r['timing_error_minutes'])))
    else:best=min(pool,key=lambda r:(abs(r['timing_error_minutes']),r['height_distance_fraction']))
    best,q=run(best['beta'],best['n_manning'],keep=True)
    before_tm,before_height,before_kind,_=feature(data.cunge_cms.to_numpy()[valid],tt,ex,review)
    fitted=data.copy();fitted['calibrated_cunge_cms']=q
    fitted.to_parquet(out/'wave.parquet',index=False)
    shifted=data.downstream_cms.to_numpy()+target['downstream_display_offset_cms']
    best.update(height_low_cms=lower,height_high_cms=upper,height_target_type='point' if is_point else 'interval',timing_within_one_minute=abs(best['timing_error_minutes'])<=1,height_range_met=best['height_distance_cms']==0 if not is_point else None,point_height_within_numerical_01pct=best['height_distance_fraction']<=.001 if is_point else None,original_n=orig_n,original_peak_cms=before_height,original_feature_utc=pd.Timestamp(before_tm,unit='s',tz='UTC').isoformat(),original_timing_error_minutes=(before_tm-goal)/60,shifted_rmse_cms=float(np.sqrt(np.mean((q[valid]-shifted[valid])**2))),original_shifted_rmse_cms=float(np.sqrt(np.mean((data.cunge_cms.to_numpy()[valid]-shifted[valid])**2))),raw_rmse_cms=float(np.sqrt(np.mean((q[valid]-data.downstream_cms.to_numpy()[valid])**2))),trials=len(records),beta_at_search_boundary=best['beta'] in beta_bounds,initialization='equilibrium at original first forcing',calibration_ready_for_production=False)
    warnings=[]
    if ex in ['21','22']:warnings.append('Source clock provisional; no timezone correction applied.')
    if ex=='55':warnings.append('Hourly crest sampling: numerical minute timing fit is not observational certainty.')
    if ex=='46':warnings.append('Antecedent sensitivity needs rechecking at these final fitted parameters.')
    if not best['timing_within_one_minute']:warnings.append('No one-minute timing fit found; do not present as satisfied.')
    if best['height_distance_cms']>0 and not is_point:warnings.append('Height range unmet in tested beta/n bounds; range not widened.')
    if best['feature_kind']=='merged_or_unresolved_two_lobes':warnings.append('54 two-lobe feature merged/unresolved; explicit manual review required.')
    if data.upstream_gap.any() or data.downstream_gap.any():warnings.append('Diagnostic interpolation flags retained; final sampling/gap eligibility not resolved.')
    best['warnings']=warnings
    checks=[];half=None
    for name,mult in [('half_dt',1),('half_dx_half_dt',2)]:
        g=grid.copy();g['count']*=mult;g['rapid_effective_length_m']=g.link_length_m/g['count']
        f=cunge.parameters(g,best['n_manning'],method='cunge');f['rapid_x']=.5-best['beta']*(.5-f.rapid_x)
        qc,audit=route_interval_means(f,data.forcing_cms.to_numpy(),int(round(120/dt)),dt/2,initial_flow=q0)
        assert audit['minimum_native_flow_cms']>=-1e-8
        ref=q if half is None else half
        tm,h,kind,_=feature(qc[valid],tt,ex,review);rt,_,_,_=feature(ref[valid],tt,ex,review)
        checks.append(dict(control=name,max_event_difference_cms=float(np.max(abs(qc[valid]-ref[valid]))),feature_shift_minutes=(tm-rt)/60,feature_kind=kind))
        if half is None:half=qc
    best['numerical_controls']=checks
    best.update(native_dt_s=dt,subdivision_multiplier=subdivision,beta_bounds=list(beta_bounds))
    (out/'result.json').write_text(json.dumps(best,indent=2))
    (out/'beta_candidates.json').write_text(json.dumps(list(beta_fits.values()),indent=2))
    (out/'rejected_trials.json').write_text(json.dumps(errors,indent=2))
    print('FINISHED',ex,'beta',best['beta'],'n',best['n_manning'],'height gap',best['height_distance_cms'],flush=True)
    return best

def main():
    source_files=[TARGET_PATH,TIMING_PATH,Path(__file__),ROOT/'RAPID/src/rapid_tools/cunge.py',ROOT/'RAPID/src/rapid_tools/common_grid.py']
    hashes={str(p):sha(p) for p in source_files}
    for ex in [r['example'] for r in json.loads(TARGET_PATH.read_text())['records']]:hashes[str(SOURCE/ex/'60s.parquet')]=sha(SOURCE/ex/'60s.parquet')
    sig=hashlib.sha256(json.dumps(dict(hashes=hashes,protocol=PROTOCOL),sort_keys=True).encode()).hexdigest()[:12]
    parent=ROOT/'RAPID/outputs/common_grid_height_calibration_20261008';dest=parent/sig;dest.mkdir(parents=True,exist_ok=True)
    shutil.copy2(TARGET_PATH,dest/'height_targets.json');shutil.copy2(TIMING_PATH,dest/'timing_targets.json');shutil.copy2(__file__,dest/'calibration_source.py')
    (dest/'protocol.json').write_text(json.dumps(dict(**PROTOCOL,source_directory=str(SOURCE),hashes=hashes),indent=2))
    (parent/'latest.json').write_text(json.dumps(dict(directory=str(dest),status='running'),indent=2))
    examples=[r['example'] for r in json.loads(TARGET_PATH.read_text())['records']]
    results=[]
    with ProcessPoolExecutor(max_workers=2) as pool:
        futures={pool.submit(calibrate,ex,str(dest)):ex for ex in examples}
        for future in as_completed(futures):
            results.append(future.result());pd.DataFrame(results).sort_values('example').to_json(dest/'results.json',orient='records',indent=2)
    for p,digest in hashes.items():assert sha(p)==digest
    (parent/'latest.json').write_text(json.dumps(dict(directory=str(dest),status='calibration_complete'),indent=2))
    print('OUTPUT',dest,flush=True)
if __name__=='__main__':main()
