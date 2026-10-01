"""Verify plateau-to-plateau event duration on existing focused synthetic runs.
No routing, grid or existing response definitions are changed.
"""
from pathlib import Path
import json,hashlib
import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from event_shape_metrics import volume_shape
ROOT=Path(__file__).resolve().parents[2]
RUN=ROOT/'synthetic_experiments/outputs/side_channel_width_length'
OUT=RUN/'event_duration_verification'
THRESHOLDS=[.005,.01,.02]

def duration(t,q,initial,final,rise_tolerance,fall_tolerance,min_settled_seconds=3600):
    """First departure to final sustained return, interpolating sampled crossings.

    t is the post-spin-up clock. First departure uses |Q-initial|; settlement
    uses |Q-final|. Each tolerance is fixed from the parent's peak-to-plateau
    amplitude and is reused for its child. A final return must follow the peak
    and remain settled for the rest of the record (at least the minimum hold).
    """
    t=np.asarray(t,float);q=np.asarray(q,float)
    if len(t)<3 or not np.isfinite(t).all() or not np.isfinite(q).all() or np.any(np.diff(t)<=0):
        raise ValueError('Invalid hydrograph')
    if min(rise_tolerance,fall_tolerance)<=0:raise ValueError('Nonpositive tolerance')
    first=np.flatnonzero(abs(q-initial)>rise_tolerance)
    outside=np.flatnonzero(abs(q-final)>fall_tolerance)
    if not len(first) or not len(outside):raise ValueError('No resolved event')
    i,j=int(first[0]),int(outside[-1])
    if i==0 or j==len(t)-1:raise ValueError('Censored boundary')
    def crossing(k,level,values):
        return float(t[k]+(t[k+1]-t[k])*(level-values[k])/(values[k+1]-values[k]))
    start=crossing(i-1,rise_tolerance,abs(q-initial))
    end=crossing(j,fall_tolerance,abs(q-final))
    if end<t[np.argmax(q)] or end<=start or t[-1]-end<min_settled_seconds:
        raise ValueError('Insufficient final settlement')
    return dict(start_seconds=start,end_seconds=end,duration_seconds=end-start,
                settled_tail_seconds=t[-1]-end)

def checks():
    t=np.arange(0.,101.,.01)
    pulse=lambda center,height:height*np.maximum(1-abs(t-center)/5,0)
    q=pulse(20,100)
    a=duration(t,q,0,0,1,1,1)
    shifted=duration(t,pulse(40,100),0,0,1,1,1)
    assert np.isclose(a['duration_seconds'],9.9)
    assert np.isclose(a['duration_seconds'],shifted['duration_seconds'])
    # A late second peak extends the event despite an intervening flat valley.
    above=duration(t,q+pulse(60,2),0,0,1,1,1)
    below=duration(t,q+pulse(60,.5),0,0,1,1,1)
    assert above['end_seconds']>60
    assert np.isclose(below['duration_seconds'],a['duration_seconds'])
    try:duration(t[:2100],q[:2100],0,0,1,1,1)
    except ValueError:pass
    else:raise AssertionError('Censored tail was accepted')
    return {'analytic_triangle':'passed','pure_translation_invariance':'passed',
            'late_peak_after_flat_valley':'passed','subthreshold_peak_ignored':'passed',
            'censored_tail_rejected':'passed'}

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    validation=checks()
    manifest=json.loads((RUN/'manifest.json').read_text());base=manifest['settings']['baseflow_cms']
    amplitude=manifest['settings']['peak_cms']-base
    runs={};runrows=[]
    for key in ['child']+[f'a{a:g}_m{m:g}' for a in manifest['shares'] for m in manifest['multipliers']]:
        mt=json.loads((RUN/f'{key}_metrics.json').read_text());h=pd.read_parquet(RUN/f'{key}_hydrograph.parquet')
        if max(abs(mt['volume_error_fraction']),mt['pre_event_relative_error'],mt['final_relative_error'])>1e-4:
            raise ValueError(f'Invalid event: {key}')
        dt=mt['dt_seconds'];onset=round(mt['buffer_hours']*3600/dt)*dt
        mask=h.time_seconds.to_numpy()>=onset
        ts=h.time_seconds.to_numpy()[mask]-onset;q=h.outlet_cms.to_numpy()[mask]
        shape=volume_shape(ts,q-base,dt)
        peaks,_=find_peaks(q-base,prominence=.01*amplitude)
        runs[key]=dict(t=ts+dt/2,q=q,dt=dt,mt=mt,shape=shape)
        runrows.append(dict(run=key,screened_peak_count=len(peaks),peak_excess=q.max()-base,
            temporal_sd_hours=shape['temporal_sd_seconds']/3600,volume_error_fraction=mt['volume_error_fraction'],
            capped_segments=mt['n_celerity_capped']))
    pairs=[]
    for a in manifest['shares']:
        for m in manifest['multipliers']:
            pairs.append(('collapse',a,m,f'a{a:g}_m{m:g}','child'))
    for a in [.2,.05]:
        for m in [1.5,1.25,1.]:
            pairs.append(('shortening',a,m,f'a{a:g}_m1.5',f'a{a:g}_m{m:g}'))
    rows=[]
    for kind,a,m,parent,child in pairs:
        p,c=runs[parent],runs[child]
        rise=p['q'].max()-base;fall=rise # Both synthetic plateaus are exactly 100.
        for fraction in THRESHOLDS:
            pdur=duration(p['t'],p['q'],base,base,fraction*rise,fraction*fall)
            cdur=duration(c['t'],c['q'],base,base,fraction*rise,fraction*fall)
            # Sensitivity to saved-output resolution, not a routing convergence test.
            p2=duration(p['t'][::2],p['q'][::2],base,base,fraction*rise,fraction*fall)
            c2=duration(c['t'][::2],c['q'][::2],base,base,fraction*rise,fraction*fall)
            change=100*(cdur['duration_seconds']/pdur['duration_seconds']-1)
            rows.append(dict(kind=kind,outer_a_fraction=a,length_multiplier=m,parent=parent,child=child,
                threshold_fraction=fraction,tolerance_cms=fraction*rise,
                parent_duration_hours=pdur['duration_seconds']/3600,child_duration_hours=cdur['duration_seconds']/3600,
                parent_start_hours=pdur['start_seconds']/3600,parent_end_hours=pdur['end_seconds']/3600,
                child_start_hours=cdur['start_seconds']/3600,child_end_hours=cdur['end_seconds']/3600,
                duration_change_pct=change,
                temporal_spread_change_pct=100*(c['shape']['temporal_sd_seconds']/p['shape']['temporal_sd_seconds']-1),
                peak_excess_change_pct=100*((c['q'].max()-base)/(p['q'].max()-base)-1),
                lag_shift_pct=100*(c['mt']['peak_lag_seconds']/p['mt']['peak_lag_seconds']-1),
                coarsening_change_difference_pp=100*(c2['duration_seconds']/p2['duration_seconds']-1)-change,
                minimum_settled_tail_hours=min(pdur['settled_tail_seconds'],cdur['settled_tail_seconds'])/3600))
    d=pd.DataFrame(rows)
    for _,g in d[d.kind=='collapse'].groupby(['outer_a_fraction','threshold_fraction']):
        assert g.sort_values('length_multiplier').parent_duration_hours.diff().dropna().gt(0).all()
    assert d[(d.kind=='shortening')&(d.length_multiplier==1.5)].duration_change_pct.eq(0).all()
    assert d[d.kind=='collapse'].duration_change_pct.lt(0).all()
    validation.update(monotonic_length_response='passed',identity_controls='passed',collapse_shortens_duration='passed')
    d.to_csv(OUT/'comparisons.csv',index=False)
    pd.DataFrame(runrows).to_csv(OUT/'run_metrics.csv',index=False)
    wide=d.pivot(index=['kind','outer_a_fraction','length_multiplier'],columns='threshold_fraction',values='duration_change_pct')
    sensitivity=wide.copy();sensitivity['range_pp']=wide.max(axis=1)-wide.min(axis=1)
    sensitivity['same_sign']=np.sign(wide).nunique(axis=1)==1
    sensitivity.to_csv(OUT/'threshold_sensitivity.csv')
    (OUT/'verification.json').write_text(json.dumps(dict(checks=validation,n_runs=len(runs),n_comparisons=len(pairs),
        thresholds=THRESHOLDS,all_threshold_signs_agree=bool(sensitivity.same_sign.all()),
        max_threshold_range_pp=float(sensitivity.range_pp.max()),
        max_sampling_difference_pp=float(d.coarsening_change_difference_pp.abs().max()),
        minimum_settled_tail_hours=float(d.minimum_settled_tail_hours.min()),
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        scope='20 topological collapses; 4 actual shortening comparisons plus 2 identity controls; saved 200m routing only',
        interpolation='interval-average discharges at interval midpoints; linear threshold crossings',
        tolerance='parent peak-to-initial and peak-to-final plateau amplitudes, reused unchanged for child',
        settlement='after last excursion outside final tolerance, through all remaining samples; minimum 1 hour'),indent=2))
    print(d[d.threshold_fraction==.01][['kind','outer_a_fraction','length_multiplier','parent_duration_hours','child_duration_hours','duration_change_pct','temporal_spread_change_pct']].round(3).to_string(index=False))
    print((OUT/'verification.json').read_text())
if __name__=='__main__':main()
