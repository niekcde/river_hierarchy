"""Full network-22 beta=1 sensitivity check using the completed reference calibration."""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import ast
import hashlib
import json
import time

import numpy as np
import pandas as pd

from delay_side_project import ROOT, OUT, read, SOURCES
from manning_network_pilot import manning_reference, state_metrics, table
from rapid_tools.common_grid import route_interval_means
from rapid_tools.hydrograph_response import response_rows
from rapid_tools.event_window_batch import settling_hours

DEST=OUT/'manning_beta1_network22_20261027'
BUNDLE=ROOT/'RAPID/outputs/bounded_beta_calibration_20261009'


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def parameters(source,n):
    f=source.copy();f['n_manning']=n;f['diffusion_multiplier']=1.0
    beta=f.diffusion_multiplier.to_numpy()
    f=manning_reference(f)
    f['rapid_x']=.5-beta*(.5-f.rapid_x)
    # Physical diffusivity above is unscaled; retain beta as a separate modifier.
    np.testing.assert_allclose(1-2*f.rapid_x,beta*f.hydraulic_length_scale_m/f.rapid_effective_length_m,rtol=1e-12)
    return f


def run_state(args):
    job,n,forcing_path,windows,final,tolerance=args
    state=job['state_id'];canonical='fully_collapsed' if state=='single_centerline' else state
    folder=DEST/'states'/state;folder.mkdir(parents=True,exist_ok=True)
    receipt=folder/'completion.json'
    if receipt.exists():
        done=json.loads(receipt.read_text())
        assert done['n']==n and done['beta']==1.0 and done['source_sha256']==digest(job['parameters'])
        assert all(digest(p)==h for p,h in done['files'].items())
        return done
    start=time.monotonic();source=pd.read_parquet(job['parameters']);f=parameters(source,n)
    force=pd.read_parquet(forcing_path);nb=int(force.phase.eq('spinup').sum());flow=force.forcing_cms.to_numpy()[nb:]
    initial=float(force.forcing_cms.iloc[0]);dt=float(job['native_dt_seconds'])
    q,audit=route_interval_means(f,flow,round(60/dt),dt,initial_flow=initial)
    if audit['minimum_native_flow_cms']< -1e-9:raise ValueError('Negative flow: '+state)
    hold_start=force.loc[force.phase.eq('settling'),'interval_start_utc'].iloc[0]
    required,settled=settling_hours(force.time_utc.iloc[nb:],q,hold_start,final,tolerance)
    # Extend the held final forcing if this new parameterization needs more tail.
    extensions=0
    while not settled and extensions<5:
        extra=max(96,int(np.ceil(required-(force.interval_end_utc.iloc[-1]-hold_start).total_seconds()/3600))+48)
        origin=force.interval_end_utc.iloc[-1];count=extra*60
        tail=pd.DataFrame({'interval_start_utc':origin+pd.to_timedelta(np.arange(count)*60,unit='s'),
                           'interval_end_utc':origin+pd.to_timedelta((np.arange(count)+1)*60,unit='s'),
                           'time_utc':origin+pd.to_timedelta((np.arange(count)+.5)*60,unit='s'),
                           'forcing_cms':final,'phase':'settling'})
        force=pd.concat([force,tail],ignore_index=True);flow=force.forcing_cms.to_numpy()[nb:]
        q,audit=route_interval_means(f,flow,round(60/dt),dt,initial_flow=initial)
        assert audit['minimum_native_flow_cms']>=-1e-9
        required,settled=settling_hours(force.time_utc.iloc[nb:],q,hold_start,final,tolerance);extensions+=1
    if not settled:raise RuntimeError('Tail failed to settle: '+state)
    q=np.r_[np.full(nb,initial),q];t=force.time_utc.astype('int64').to_numpy()/1e9
    metrics=[state_metrics(t,q,pd.Series(w),canonical,job['aliases']) for w in windows]
    out=force[['interval_start_utc','interval_end_utc','time_utc','phase']].copy();out['q_outlet_cms']=q
    f.to_parquet(folder/'parameters.parquet',index=False);out.to_parquet(folder/'outlet.parquet',index=False)
    pd.DataFrame(metrics).to_parquet(folder/'metrics.parquet',index=False)
    done=dict(state=state,canonical=canonical,n=n,beta=float(f.diffusion_multiplier.iloc[0]),source_sha256=digest(job['parameters']),
              seconds=time.monotonic()-start,settled=settled,tail_extensions=extensions,required_tail_hours=required,
              depth_min=float(f.depth_m.min()),depth_max=float(f.depth_m.max()),celerity_min=float(f.rapid_celerity_mps.min()),celerity_max=float(f.rapid_celerity_mps.max()),
              mean_travel_h=float((f.reference_flow_fraction*f.link_length_m/f.rapid_celerity_mps).sum()/3600),**audit,
              files={str(p):digest(p) for p in [folder/'parameters.parquet',folder/'outlet.parquet',folder/'metrics.parquet']})
    receipt.write_text(json.dumps(done,indent=2)+'\n')
    return done


def main():
    DEST.mkdir(exist_ok=True)
    doc=read(OUT/'routing_plan.json');case=next(c for c in doc['cases'] if c['example']=='22')
    jobs=case['jobs'];parent=next(j for j in jobs if j['state_id']=='S000_base')
    old=read(OUT/'review/uncapped_state_metrics.parquet');old=old[old.example.eq('22')].copy()
    inventory=read(OUT/'validation_group_inventory_20261023/records.parquet');inventory=inventory[inventory.example.eq('22')]
    windows=old[old.state.eq('S000_base')].to_dict('records')
    forcing_path=Path(windows[0]['outlet']).parents[1]/'forcing.parquet';force=read(forcing_path)
    source=read(parent['parameters'])
    for job in jobs:SOURCES[str(Path(job['parameters']))]=digest(job['parameters'])
    joint=read(OUT/'manning_joint_reference22_20261027/result.json')
    calibration={'selected':joint['beta1'],'old_beta':1.0}
    (DEST/'calibration.json').write_text(json.dumps(calibration,indent=2)+'\n')
    (DEST/'design.json').write_text(json.dumps(dict(network='22',jobs=len(jobs),accepted_comparisons=len(inventory),
        beta_fixed=1.0,depth='Manning normal depth at allocated reference discharge, frozen in time',
        calibration='Reuse beta=1 reference-only calibration; freeze n and beta across all collapses',
        response='Saved common baseline and 0.5/1/2% windows; six existing response measures',
        production_modified=False,RF_fits=0),indent=2)+'\n')
    n=calibration['selected']['n_manning']
    print('Selected n',n,'beta',calibration['old_beta'],'routing',len(jobs),'states',flush=True)
    args=[(j,n,str(forcing_path),windows,case['final_cms'],case['tolerance_cms']) for j in jobs]
    results=[]
    with ProcessPoolExecutor(max_workers=4) as pool:
        futures={pool.submit(run_state,arg):arg[0]['state_id'] for arg in args}
        for future in as_completed(futures):
            result=future.result();results.append(result)
            if len(results)%10==0 or len(results)==len(jobs):
                print('Completed',len(results),'/',len(jobs),'last',result['state'],flush=True)
                pd.DataFrame([{k:v for k,v in r.items() if k!='files'} for r in results]).to_csv(DEST/'routing_audit.csv',index=False)
    states=pd.concat([pd.read_parquet(DEST/'states'/j['state_id']/'metrics.parquet') for j in jobs],ignore_index=True)
    assert set(states.state)==set(old.state) and len(states)==len(old)
    new_response=response_rows(states);old_response=response_rows(old)
    keys=['example','state','window'];fields=['centroid_shift_hours','peak_shift_hours','peak_excess_change_pct','duration80_change_pct','asymmetry_change','volume_change_pct']
    comparison=old_response[keys+fields].merge(new_response[keys+fields],on=keys,suffixes=('_based','_manning'),validate='one_to_one')
    comparison=comparison.merge(inventory[['state','edit_type','cohort']],on='state',how='left',validate='many_to_one')
    comparison['edit_type']=comparison.edit_type.fillna('Single-centerline reference control')
    for side in ['based','manning']:
        comparison['centroid_minutes_'+side]=comparison['centroid_shift_hours_'+side]*60
        comparison['response_class_'+side]=np.select([comparison['centroid_minutes_'+side]>1,comparison['centroid_minutes_'+side]<-1],['delay','advance'],default='near_zero')
    states.to_parquet(DEST/'state_metrics.parquet',index=False);new_response.to_csv(DEST/'responses.csv',index=False)
    comparison.to_csv(DEST/'comparison.csv',index=False)
    main=comparison[comparison.window.eq('0.5%')].copy()
    main.groupby(['edit_type','response_class_based','response_class_manning']).size().rename('count').to_csv(DEST/'class_transitions.csv')
    # Select clock checks by predeclared prior cases plus new largest effects.
    reps=set(['S000_base','S057_unit_5','island_22_018','S296_unit_30','CB001_bubble_9','all_closed_units_collapsed','single_centerline'])
    reps.update(main.nlargest(2,'centroid_minutes_manning').state)
    reps.update(main.nsmallest(2,'centroid_minutes_manning').state)
    reps={'single_centerline' if s=='fully_collapsed' else s for s in reps}
    bystate={j['state_id']:j for j in jobs};checks=[];parent_half=None
    for state in ['S000_base']+sorted(reps-{'S000_base'}):
        j=bystate[state];folder=DEST/'states'/state;f=pd.read_parquet(folder/'parameters.parquet');saved=pd.read_parquet(folder/'outlet.parquet')
        # Only the common saved support is needed for fixed response windows.
        nb=int(force.phase.eq('spinup').sum());flow=force.forcing_cms.to_numpy()[nb:];dt=j['native_dt_seconds']/2
        q,a=route_interval_means(f,flow,round(60/dt),dt,initial_flow=float(force.forcing_cms.iloc[0]));assert a['minimum_native_flow_cms']>=-1e-9
        q=np.r_[np.full(nb,force.forcing_cms.iloc[0]),q];t=force.time_utc.astype('int64').to_numpy()/1e9
        m=state_metrics(t,q,pd.Series(windows[0]),state,j['aliases'])
        if state=='S000_base':parent_half=m
        canonical='fully_collapsed' if state=='single_centerline' else state
        exact=states[(states.state==canonical)&states.window.eq(windows[0]['window'])].iloc[0]
        pexact=states[states.state.eq('S000_base')&states.window.eq(windows[0]['window'])].iloc[0]
        checks.append(dict(state=state,centroid_contrast_change_s=(m['centroid_seconds']-parent_half['centroid_seconds'])-(exact.centroid_seconds-pexact.centroid_seconds),
                           peak_contrast_change_s=(m['peak_seconds']-parent_half['peak_seconds'])-(exact.peak_seconds-pexact.peak_seconds)))
        print('Clock check',state,checks[-1]['centroid_contrast_change_s'],flush=True)
    pd.DataFrame(checks).to_csv(DEST/'clock_checks.csv',index=False)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(8,6),constrained_layout=True)
    for kind,g in main.groupby('edit_type'):
        ax.scatter(g.centroid_minutes_based,g.centroid_minutes_manning,s=20,alpha=.65,label=kind)
    lo=min(main.centroid_minutes_based.min(),main.centroid_minutes_manning.min());hi=max(main.centroid_minutes_based.max(),main.centroid_minutes_manning.max())
    ax.plot([lo,hi],[lo,hi],color='grey',linestyle='--');ax.axhline(0,color='grey',lw=.5);ax.axvline(0,color='grey',lw=.5)
    ax.set(xlabel='Existing BASED centroid shift (min)',ylabel='Recalibrated Manning centroid shift (min)',title='Network 22: all saved collapse comparisons');ax.legend(fontsize=8);ax.grid(alpha=.2)
    fig.savefig(DEST/'centroid_comparison.png',dpi=160);plt.close(fig)
    summary=[]
    for kind,g in main.groupby('edit_type'):
        summary.append(dict(type=kind,cases=len(g),based_delays=int((g.centroid_minutes_based>1).sum()),manning_delays=int((g.centroid_minutes_manning>1).sum()),
                            based_median_min=float(g.centroid_minutes_based.median()),manning_median_min=float(g.centroid_minutes_manning.median())))
    pd.DataFrame(summary).to_csv(DEST/'summary.csv',index=False)
    for path in [Path(__file__),Path(__file__).with_name('manning_network_pilot.py'),Path(__file__).with_name('delay_side_project.py'),BUNDLE/'source_code/calibrate_common_grid_height.py',ROOT/'RAPID/src/rapid_tools/cunge.py',ROOT/'RAPID/src/rapid_tools/common_grid.py',ROOT/'RAPID/src/rapid_tools/hydrograph_response.py']:
        SOURCES[str(path)]=digest(path)
    for path,h in SOURCES.items():assert digest(path)==h
    (DEST/'manifest.json').write_text(json.dumps(dict(complete=True,source_hashes=SOURCES,states=len(jobs),comparisons=len(main),accepted_comparisons=len(inventory),
                 selected_n=n,beta=calibration['old_beta'],rf_fits=0,production_modified=False,
                 maximum_centroid_clock_difference_s=max(abs(x['centroid_contrast_change_s']) for x in checks)),indent=2)+'\n')
    print('SUMMARY',pd.DataFrame(summary).to_string(index=False),flush=True)
    print(main[main.state.isin(['S057_unit_5','island_22_018','S296_unit_30','CB001_bubble_9','all_closed_units_collapsed'])][['state','centroid_minutes_based','centroid_minutes_manning']].to_string(index=False),flush=True)


if __name__=='__main__':main()
