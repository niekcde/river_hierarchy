"""Isolated moment controls for the two simple centroid delays. No production edits."""
import json
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from delay_side_project import ROOT, OUT, EXT, read, SOURCES, centroid
from rapid_tools.common_grid import route_interval_means
from rapid_tools.cunge import route

DEST = OUT / 'delay_kx_controls_20261027'


def moments(f):
    """These selected local units consist entirely of parallel single reaches."""
    assert f.u.nunique()==f.v.nunique()==1
    p=f.split_fraction.to_numpy()
    np.testing.assert_allclose(p.sum(),1)
    mu=f['count'].to_numpy()*f.rapid_k.to_numpy()
    variance=f['count'].to_numpy()*f.rapid_k.to_numpy()**2*(1-2*f.rapid_x.to_numpy())
    mean=float(p@mu)
    return mean,float(p@(mu-mean)**2),float(p@variance)


def metric(t,q,a,b,base):
    ts=np.unique(np.r_[a,t[(t>a)&(t<b)],b])
    excess=np.interp(ts,t,q)-base
    mu,vol=centroid(ts,excess)
    peak=int(excess.argmax())
    # Crossing is an event-window rise diagnostic, relative to this run's peak.
    above=np.flatnonzero(excess[:peak+1]>=.1*excess[peak])
    j=int(above[0])
    rise=np.nan if j==0 else float(np.interp(.1*excess[peak],excess[j-1:j+1],ts[j-1:j+1]))
    return dict(centroid_s=mu,peak_s=ts[peak],rise10_s=rise,rise10_left_censored=j==0,volume=vol,peak_excess=excess[peak])


def main():
    DEST.mkdir(exist_ok=True)
    jobs={}
    for path in [ROOT/'RAPID/outputs/event_window_plan_20261013/plan.json',EXT/'routing_plan.json',OUT/'routing_plan.json']:
        for case in read(path)['cases']:
            for job in case['jobs']:
                for name in {job['state_id'],*job.get('aliases',[])}:
                    jobs[case['example'],name]=job
    old,new=read(EXT/'state_metrics.parquet'),read(OUT/'review/uncapped_state_metrics.parquet')
    states=pd.concat([old[~old.example.isin(new.example)],new])
    states=states[states.window.eq('0.5%')].set_index(['example','state'])
    rows,settings,waves,kernels,spatial_rows=[],[],[],[],[]
    for ex,state in [('02','S013_unit_20'),('22','S057_unit_5')]:
        parent=read(jobs[ex,'S000_base']['parameters'])
        child=read(jobs[ex,state]['parameters'])
        removed=parent.id_link[~parent.id_link.isin(child.id_link)]
        added=child.id_link[~child.id_link.isin(parent.id_link)]
        local=parent[parent.id_link.isin(removed)].copy()
        actual=child[child.id_link.isin(added)].copy()
        np.testing.assert_array_equal(local[['u','v']].drop_duplicates(),actual[['u','v']].drop_duplicates())
        shared=parent.id_link[parent.id_link.isin(child.id_link)]
        cols=['u','v','count','rapid_k','rapid_x','split_fraction','reference_flow_fraction']
        np.testing.assert_allclose(parent.set_index('id_link').loc[shared,cols],child.set_index('id_link').loc[shared,cols],rtol=0,atol=1e-12)
        np.testing.assert_allclose(local.reference_flow_fraction,local.split_fraction)
        pm,pv,pd_=moments(local);cm,cv,cd=moments(actual)
        target_variance=cv+cd
        frames={'parent':parent}
        for label,mean,var in [('mean_only',cm,pv+pd_),('spreading_only',pm,target_variance),('combined',cm,target_variance)]:
            scale=mean/pm
            factor=(var-scale**2*pv)/(scale**2*pd_)
            assert factor>0
            f=parent.copy();sel=f.id_link.isin(removed)
            f.loc[sel,'rapid_k']*=scale
            f.loc[sel,'rapid_x']=.5-factor*(.5-f.loc[sel,'rapid_x'])
            m,v,d=moments(f[sel])
            np.testing.assert_allclose([m,v+d],[mean,var],rtol=1e-12)
            frames[label]=f
            settings.append(dict(example=ex,state=state,control=label,K_multiplier=scale,X_spreading_multiplier=factor,
                                 local_mean_min=m/60,local_path_variance_min2=v/3600,local_diffusion_variance_min2=d/3600))
        frames['actual_child']=child
        pr,cr=states.loc[(ex,'S000_base')],states.loc[(ex,state)]
        forcing=read(Path(pr.outlet).parents[1]/'forcing.parquet')
        nb=int(forcing.phase.eq('spinup').sum())
        flow=forcing.forcing_cms.to_numpy()[nb:];initial=float(forcing.forcing_cms.iloc[0])
        t=forcing.time_utc.astype('int64').to_numpy()/1e9
        a,b,base=pr.rise_seconds,pr.fall_seconds,pr.baseline_cms
        dt=float(jobs[ex,'S000_base']['native_dt_seconds'])
        assert dt==jobs[ex,state]['native_dt_seconds']
        reference={}
        for clock in [1,2]:
            for label,f in frames.items():
                step=dt/clock
                q,audit=route_interval_means(f,flow,round(60/step),step,initial_flow=initial)
                assert audit['minimum_native_flow_cms']>=-1e-9
                q=np.r_[np.full(nb,initial),q]
                check=np.nan
                if clock==1 and label in ['parent','actual_child']:
                    rr=pr if label=='parent' else cr
                    saved=read(rr.outlet)
                    assert SOURCES[str(Path(rr.outlet))]==rr.outlet_sha256
                    np.testing.assert_array_equal(saved.time_utc,forcing.time_utc)
                    check=float(np.max(abs(q-saved.q_outlet_cms.to_numpy())))
                    np.testing.assert_allclose(q,saved.q_outlet_cms,rtol=0,atol=1e-7)
                for window,lo,hi in [('event',a,b),('saved_support',t[0],t[-1])]:
                    values=metric(t,q,lo,hi,base)
                    if label=='parent':reference[clock,window]=values
                    ref=reference[clock,window]
                    record=dict(example=ex,state=state,control=label,clock_divisor=clock,dt_s=step,window=window,
                                centroid_shift_min=(values['centroid_s']-ref['centroid_s'])/60,
                                peak_shift_min=(values['peak_s']-ref['peak_s'])/60,
                                rise10_shift_min=(values['rise10_s']-ref['rise10_s'])/60,
                                rise10_left_censored=values['rise10_left_censored'] or ref['rise10_left_censored'],
                                volume_change_pct=100*(values['volume']/ref['volume']-1),
                                saved_max_error_cms=check,**audit)
                    rows.append(record)
                    if window=='event':print({k:record[k] for k in ['example','control','clock_divisor','centroid_shift_min','peak_shift_min']},flush=True)
                if clock==1:
                    waves.append(pd.DataFrame(dict(example=ex,state=state,control=label,hours=(t-a)/3600,q=q-base)))
        # Refine only the edited local reaches; retain physical c and diffusivity.
        # N*=factor, K/=factor, (1-2X)*=factor preserves the first two moments.
        for refinement in [2,4]:
            ref=None
            for label,original,ids in [('parent',parent,removed),('actual_child',child,added)]:
                f=original.copy();sel=f.id_link.isin(ids)
                before=moments(f[sel])
                f.loc[sel,'count']*=refinement
                f.loc[sel,'rapid_k']/=refinement
                f.loc[sel,'rapid_effective_length_m']/=refinement
                f.loc[sel,'rapid_x']=.5-refinement*(.5-f.loc[sel,'rapid_x'])
                np.testing.assert_allclose(moments(f[sel]),before,rtol=1e-12,atol=1e-7)
                q,audit=route_interval_means(f,flow,round(120/dt),dt/2,initial_flow=initial)
                assert audit['minimum_native_flow_cms']>=-1e-9
                q=np.r_[np.full(nb,initial),q]
                values=metric(t,q,a,b,base)
                if ref is None:ref=values
                spatial_rows.append(dict(example=ex,state=state,control=label,local_subdivision_factor=refinement,
                                    centroid_shift_min=(values['centroid_s']-ref['centroid_s'])/60,
                                    peak_shift_min=(values['peak_s']-ref['peak_s'])/60,**audit))
            print('Local refinement',spatial_rows[-1],flush=True)
        # Local step response: illustrate early response versus tails, not water velocity.
        for label,f in frames.items():
            lf=f[f.id_link.isin(added if label=='actual_child' else removed)].copy()
            m,v,d=moments(lf)
            step_flow=np.r_[0.,np.ones(int(72*3600/dt))]
            response,audit=route(lf,step_flow,1,dt,return_native=True)
            k,x=lf.rapid_k.to_numpy(),lf.rapid_x.to_numpy()
            den=dt+2*k*(1-x);aa=(dt-2*k*x)/den;bb=(dt+2*k*x)/den;dd=(2*k*(1-x)-dt)/den
            nonnegative=bool((aa>=0).all() and (dd>=0).all() and (bb+aa*dd>=0).all())
            settings.append(dict(example=ex,state=state,control=label,local_mean_min=m/60,
                                 local_path_variance_min2=v/3600,local_diffusion_variance_min2=d/3600,
                                 local_x_min=x.min(),local_x_max=x.max(),prompt_fraction=response[1],
                                 tail_fraction_72h=1-response[-1],nonnegative_local_kernel=nonnegative))
            stride=max(1,int(60/dt))
            kernels.append(pd.DataFrame(dict(example=ex,control=label,hours=np.arange(len(response)-1)[::stride]*dt/3600,
                                             fraction=response[1:][::stride])))
        pd.DataFrame(rows).to_csv(DEST/'results.csv',index=False)
    result=pd.DataFrame(rows)
    paired=result[result.clock_divisor.eq(1)].merge(result[result.clock_divisor.eq(2)],on=['example','state','control','window'],suffixes=('_native','_half'),validate='one_to_one')
    paired['centroid_clock_difference_s']=60*(paired.centroid_shift_min_half-paired.centroid_shift_min_native)
    paired.to_csv(DEST/'clock_check.csv',index=False)
    pd.DataFrame(settings).to_csv(DEST/'control_parameters.csv',index=False)
    pd.DataFrame(spatial_rows).to_csv(DEST/'local_grid_check.csv',index=False)
    pd.concat(waves).to_parquet(DEST/'hydrographs.parquet',index=False)
    pd.concat(kernels).to_parquet(DEST/'local_steps.parquet',index=False)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(12,7),constrained_layout=True)
    colors={'parent':'#555555','mean_only':'#009E73','spreading_only':'#CC79A7','combined':'#E69F00','actual_child':'#0072B2'}
    for i,ex in enumerate(['02','22']):
        for f in kernels:
            if f.example.iloc[0]!=ex:continue
            label=f.control.iloc[0]
            axes[i,0].plot(f.hours,f.fraction,label=label,color=colors[label])
        axes[i,0].set(xlim=(0,24),ylim=(.7,1.005),xlabel='Hours after a local unit step',ylabel='Fraction of final response',title=f'{ex}: local response')
        panel=result[(result.example==ex)&(result.window=='event')&(result.clock_divisor==1)]
        axes[i,1].bar(panel.control,panel.centroid_shift_min,color=[colors[c] for c in panel.control])
        axes[i,1].axhline(0,color='grey',linewidth=.8)
        axes[i,1].tick_params(axis='x',rotation=25)
        axes[i,1].set(ylabel='Centroid shift (minutes)',title=f'{ex}: fixed event window')
        axes[i,0].legend(fontsize=8);axes[i,0].grid(alpha=.2)
    fig.savefig(DEST/'controls.png',dpi=160);plt.close(fig)
    for path in [Path(__file__),ROOT/'RAPID/src/rapid_tools/cunge.py',ROOT/'RAPID/src/rapid_tools/common_grid.py',Path(__file__).with_name('delay_side_project.py')]:
        SOURCES[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
    for path,digest in SOURCES.items():assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==digest
    (DEST/'manifest.json').write_text(json.dumps(dict(sources=SOURCES,event_routing_runs=28,local_step_runs=10,
       design='Parent topology/splits/exterior fixed. Mean-only matches child local mean while preserving parent total variance; spreading-only matches child total local variance with parent mean; combined matches both. K scales uniformly; 1-2X compensates to hit variance. No peak/centroid fitting.',
       limits='Moment controls also change higher moments. They are numerical interventions, not self-consistent alternative geometries. Half-clock and 2x/4x local grid checks do not establish global spatial convergence or validation against observations. Saved-support centroid is still finite and may include equilibrium transitions.'),indent=2)+'\n')
    print('Maximum centroid clock difference (seconds):',paired.centroid_clock_difference_s.abs().max(),flush=True)


if __name__=='__main__':main()
