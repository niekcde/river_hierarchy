"""Network 22: current versus Manning reference depths, four collapse states.

Runs the same constant-reference RAPID/Cunge DAG cascade used by the accepted
event-window workflow. No RF fits, calibration, or production replacement.
"""
from pathlib import Path
import hashlib
import json
import time

import numpy as np
import pandas as pd

from delay_side_project import ROOT, OUT, read, SOURCES, routing_summary
from rapid_tools.common_grid import route_interval_means
from rapid_tools.hydrograph_response import distribution, response_rows
from rapid_tools.event_window_batch import settling_hours

DEST=OUT/'manning_network22_pilot_20261027'
SELECTION={'S000_base':'Uncollapsed network','S057_unit_5':'Simple unit 5',
           'island_22_018':'Island 018','S296_unit_30':'Compound member 30',
           'CB001_bubble_9':'Full compound 9'}


def manning_reference(source):
    f=source.copy()
    for col in ['depth_m','rapid_k','rapid_x','rapid_celerity_mps','kb_local']:
        if col in f:f['source_'+col]=f[col]
    # kb_local is not used by the router; do not leave its old-depth value
    # labeled as a current hydraulic quantity in this diagnostic parameter file.
    if 'kb_local' in f:f=f.drop(columns='kb_local')
    q=f.reference_discharge_cms
    f['depth_m']=(q*f.n_manning/(f.width*np.sqrt(f.slope)))**.6
    f['celerity_times_n']=5/3*f.depth_m**(2/3)*np.sqrt(f.slope)
    f['rapid_celerity_mps_raw']=f.celerity_times_n/f.n_manning
    f['rapid_celerity_mps']=f.rapid_celerity_mps_raw
    f['rapid_celerity_capped']=False
    f['hydraulic_length_scale_m']=q/(f.width*f.slope*f.rapid_celerity_mps)
    f['rapid_k']=f.rapid_effective_length_m/f.rapid_celerity_mps
    f['rapid_x']=.5*(1-f.hydraulic_length_scale_m/f.rapid_effective_length_m)
    f['hydraulic_diffusivity_m2_s']=q/(2*f.width*f.slope)
    f['depth_policy']='wide_rectangular_Manning_at_allocated_reference_Q'
    f['diagnostic_celerity_policy']='unclipped'
    np.testing.assert_allclose(f.width/f.n_manning*f.depth_m**(5/3)*np.sqrt(f.slope),q,rtol=1e-12)
    for col in ['id_link','u','v','link_length_m','width','slope','count','split_fraction','reference_flow_fraction','reference_discharge_cms','n_manning']:
        np.testing.assert_array_equal(f[col],source[col])
    return f


def state_metrics(t,q,window,state,aliases):
    a,b,base=window.rise_seconds,window.fall_seconds,window.baseline_cms
    tt=np.r_[a,t[(t>a)&(t<b)],b]
    excess=np.interp(tt,t,q)-base
    if excess.min()<0:raise ValueError('Negative excess under frozen baseline; needs explicit review')
    d=distribution(tt,excess);peak=int(excess.argmax())
    return dict(example='22',state=state,aliases=aliases,window=window.window,
                rise_seconds=a,fall_seconds=b,baseline_cms=base,rise_utc=window.rise_utc,fall_utc=window.fall_utc,
                threshold_fraction=window.threshold_fraction,peak_seconds=tt[peak],peak_cms=excess[peak]+base,
                peak_excess_cms=excess[peak],duration_hours=(b-a)/3600,**d)


def table(d):
    header='| '+' | '.join(d.columns)+' |\n| '+' | '.join(['---']*len(d.columns))+' |'
    rows=['| '+' | '.join(f'{v:.3f}' if isinstance(v,float) else str(v) for v in row)+' |' for row in d.itertuples(index=False,name=None)]
    return header+'\n'+'\n'.join(rows)


def main():
    DEST.mkdir(exist_ok=True)
    case=next(c for c in read(OUT/'routing_plan.json')['cases'] if c['example']=='22')
    jobs={j['state_id']:j for j in case['jobs']};assert set(SELECTION)<=set(jobs)
    prior=read(OUT/'review/uncapped_state_metrics.parquet')
    prior=prior[prior.example.eq('22') & prior.state.isin(SELECTION)]
    windows=prior[prior.state.eq('S000_base')].sort_values('threshold_fraction')
    original=prior.set_index(['state','window'])
    forcing=read(Path(original.loc['S000_base','0.5%'].outlet).parents[1]/'forcing.parquet')
    nb=int(forcing.phase.eq('spinup').sum());flow=forcing.forcing_cms.to_numpy()[nb:]
    initial=float(forcing.forcing_cms.iloc[0]);t=forcing.time_utc.astype('int64').to_numpy()/1e9
    hold_start=forcing.loc[forcing.phase.eq('settling'),'interval_start_utc'].iloc[0]
    forcing.to_parquet(DEST/'forcing.parquet',index=False)
    state_rows,audits,hydraulics=[],[],[];curves={}
    for option in ['current_depth','manning_depth']:
        for state in SELECTION:
            source=read(jobs[state]['parameters']);assert not source.rapid_celerity_capped.any()
            f=source if option=='current_depth' else manning_reference(source)
            folder=DEST/option/state;folder.mkdir(parents=True,exist_ok=True)
            f.to_parquet(folder/'parameters.parquet',index=False)
            hydraulic=dict(option=option,state=state,depth_min=f.depth_m.min(),depth_median=f.depth_m.median(),depth_max=f.depth_m.max(),
                           roughness=float(f.n_manning.iloc[0]),**routing_summary(f))
            hydraulics.append(hydraulic)
            for clock in [1,2]:
                dt=float(jobs[state]['native_dt_seconds'])/clock
                started=time.monotonic()
                routed,audit=route_interval_means(f,flow,round(60/dt),dt,initial_flow=initial)
                assert audit['minimum_native_flow_cms']>=-1e-9
                q=np.r_[np.full(nb,initial),routed]
                required,settled=settling_hours(forcing.time_utc.iloc[nb:],routed,hold_start,case['final_cms'],case['tolerance_cms'])
                if not settled:raise RuntimeError(f'Tail extension required: {option}/{state}/{clock}')
                reproduction=np.nan
                if option=='current_depth' and clock==1:
                    rr=original.loc[state,'0.5%'];saved=read(rr.outlet)
                    assert SOURCES[str(Path(rr.outlet))]==rr.outlet_sha256
                    np.testing.assert_array_equal(saved.time_utc,forcing.time_utc)
                    reproduction=float(abs(saved.q_outlet_cms.to_numpy()-q).max())
                    np.testing.assert_allclose(q,saved.q_outlet_cms,rtol=0,atol=1e-7)
                for _,window in windows.iterrows():
                    m=state_metrics(t,q,window,state,jobs[state]['aliases'])
                    if option=='current_depth' and clock==1:
                        old=original.loc[state,window.window]
                        for field in ['centroid_seconds','volume_m3','duration80_hours','asymmetry','peak_seconds','peak_excess_cms']:
                            np.testing.assert_allclose(m[field],old[field],rtol=1e-10,atol=1e-7)
                    state_rows.append(dict(option=option,clock_divisor=clock,dt_s=dt,**m))
                if clock==1:
                    out=forcing[['time_utc','interval_start_utc','interval_end_utc','phase']].copy();out['q_outlet_cms']=q
                    out.to_parquet(folder/'outlet.parquet',index=False);curves[option,state]=q
                audits.append(dict(option=option,state=state,clock_divisor=clock,dt_s=dt,seconds=time.monotonic()-started,
                                   settled=settled,required_tail_hours=required,saved_outlet_max_error_cms=reproduction,**audit))
                print(option,state,clock,'settled',settled,flush=True)
    states=pd.DataFrame(state_rows)
    response=[]
    for (option,clock),g in states.groupby(['option','clock_divisor']):
        r=response_rows(g);r['option']=option;r['clock_divisor']=clock;response.append(r)
    responses=pd.concat(response,ignore_index=True)
    states.to_parquet(DEST/'state_metrics.parquet',index=False)
    responses.to_csv(DEST/'responses.csv',index=False)
    pd.DataFrame(audits).to_csv(DEST/'routing_audit.csv',index=False)
    pd.DataFrame(hydraulics).to_csv(DEST/'hydraulic_summary.csv',index=False)
    check=responses[responses.clock_divisor.eq(1)].merge(responses[responses.clock_divisor.eq(2)],on=['option','state','window'],suffixes=('_native','_half'),validate='one_to_one')
    check['centroid_clock_difference_s']=3600*(check.centroid_shift_hours_half-check.centroid_shift_hours_native)
    check.to_csv(DEST/'clock_check.csv',index=False)
    main=responses[responses.clock_divisor.eq(1)&responses.window.eq('0.5%')]
    fields=['centroid_shift_hours','peak_shift_hours','peak_excess_change_pct','duration80_change_pct','volume_change_pct','asymmetry_change']
    compare=main[main.option.eq('current_depth')][['state']+fields].merge(main[main.option.eq('manning_depth')][['state']+fields],on='state',suffixes=('_current','_manning'),validate='one_to_one')
    compare.insert(1,'case',compare.state.map(SELECTION))
    for suffix in ['current','manning']:
        compare['centroid_minutes_'+suffix]=compare['centroid_shift_hours_'+suffix]*60
        compare['peak_minutes_'+suffix]=compare['peak_shift_hours_'+suffix]*60
    compare.to_csv(DEST/'comparison.csv',index=False)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(12,7),constrained_layout=True)
    start=windows.iloc[0].rise_seconds;hours=(t-start)/3600
    for ax,state in zip(axes.flat,[s for s in SELECTION if s!='S000_base']):
        for option,color in [('current_depth','#555555'),('manning_depth','#0072B2')]:
            ax.plot(hours,curves[option,state]-curves[option,'S000_base'],label=option,color=color,linewidth=1.2)
        ax.axhline(0,color='grey',linewidth=.5);ax.set(xlim=(0,255),title=SELECTION[state],xlabel='Hours from original event-window start',ylabel='Child − parent discharge (m³/s)');ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.savefig(DEST/'collapse_effects.png',dpi=150);plt.close(fig)
    view=compare[['case','centroid_minutes_current','centroid_minutes_manning','peak_minutes_current','peak_minutes_manning']]
    max_clock=check.centroid_clock_difference_s.abs().max()
    report=f'''# Network 22 Manning-depth routing pilot — 27 October 2026

One uncollapsed network and four collapse states were routed using current
depths and Manning-derived fixed reference depths. Both parent and child are
recomputed under each option. This is a routing sensitivity experiment, with
no RF fits, roughness recalibration, or replacement of production products.

## Timing results

Positive shifts mean later; negative shifts mean earlier. All values below are
minutes, child minus the corresponding uncollapsed parent, in the original
shared 0.5% event window.

{table(view)}

## Other responses

Values are percentage changes except asymmetry, which is a dimensionless
difference. Current and Manning columns use their respective parent runs.

{table(compare[['case','peak_excess_change_pct_current','peak_excess_change_pct_manning','duration80_change_pct_current','duration80_change_pct_manning','volume_change_pct_current','volume_change_pct_manning','asymmetry_change_current','asymmetry_change_manning']])}

![Change in outlet hydrograph caused by each collapse](collapse_effects.png)

## Controlled assumptions

Network 22 was chosen because it includes the previously examined simple delay,
a large island delay, a compound-member delay and a full-compound advance.
The cases were selected before seeing their Manning rerun results. The four
comparisons are descriptive examples, not independent samples.

The Manning option uses h=(n Q_ref / (B sqrt(S)))^(3/5), a wide rectangular
normal-flow closure. It then recomputes celerity, K and X consistently on every
reach of each whole-network state. Its parameters remain constant through time.
The current option uses the saved parameters. Geometry, slopes, numerical cell
counts, reference-flow allocation, forcing, initial conditions and roughness
remain unchanged within each state. Roughness is approximately 0.01624238.
No effective-depth or roughness adjustment forces the parent and child to agree.
Manning celerity is not clipped; the original selected states have no active clips.

This uses `rapid_tools.common_grid.route_interval_means`, the same RAPID/Cunge
DAG cascade used for the accepted event-window runs. It is a fresh routing
calculation, not a hydrograph translation or prediction. The saved original
outlets are reproduced to 1e−7 m³/s and their response metrics are independently
reproduced before comparison.

## Verification and interpretation limits

All 20 runs settle within the saved forcing/tail support and have nonnegative
discharge. The second run of every state uses half its original timestep.
The maximum change in centroid contrast across all three saved windows is
{max_clock:.6f} seconds. Full clock results, including peak-bin sensitivity,
are in clock_check.csv.

The original baseline and shared 0.5%, 1% and 2% windows are retained across
options. They are not reselected after seeing new hydrographs. This provides
a controlled response comparison; it does not establish a new event definition.
Depth and celerity ranges and exact network transfer moments are in
hydraulic_summary.csv. No variable-depth solver is used in this pilot.

The same n was calibrated under the original depth formulation. Manning normal
depth, fixed widths/slopes and fixed split fractions remain assumptions; these
results do not establish better agreement with observed river behaviour.

## Files and reproduction

Run `RAPID/investigations/cap_investigation_20261020/manning_network_pilot.py`
with the `river-hierarchy-rivgraph` Python environment. Separate option/state
folders contain recalculated parameters and new outlet hydrographs. The shared
forcing, state metrics, all six response metrics, comparison tables, routing
audit, hydraulic summaries, and clock checks are retained here. Source and
output hashes are in manifest.json. RF inputs and artifacts were not changed.
'''
    (DEST/'REPORT.md').write_text(report)
    for source in [Path(__file__),ROOT/'RAPID/src/rapid_tools/cunge.py',ROOT/'RAPID/src/rapid_tools/common_grid.py',ROOT/'RAPID/src/rapid_tools/hydrograph_response.py',ROOT/'RAPID/src/rapid_tools/event_window_batch.py',Path(__file__).with_name('delay_side_project.py')]:
        SOURCES[str(source)]=hashlib.sha256(source.read_bytes()).hexdigest()
    for path,digest in SOURCES.items():assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==digest
    outputs={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in DEST.rglob('*') if p.is_file() and p.name!='manifest.json'}
    (DEST/'manifest.json').write_text(json.dumps(dict(example='22',selection=SELECTION,options=['current_depth','manning_depth'],routing_runs=20,rf_fits=0,
                                                    sources=SOURCES,outputs=outputs),indent=2)+'\n')
    print(view.to_string(index=False),flush=True)
    print('Maximum centroid clock difference (s):',max_clock,flush=True)


if __name__=='__main__':main()
