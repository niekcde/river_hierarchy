"""Isolated steady-buffer tests and runtime projections; production untouched."""
from pathlib import Path
import sys, json, time
from functools import partial
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'RAPID/src'))
sys.path.insert(0,str(Path(__file__).parent))
from rapid_tools.hydrograph_review import base_directory,event_clock
from rapid_tools.refreshed_analysis import DEFAULT_REFRESH_ROOT
from check_fractional_routing_clocks import cascade,sparse_benchmark,peak
cascade=partial(cascade,serialized_allocations=True)
sparse_benchmark=partial(sparse_benchmark,serialized_allocations=True)

OUT=ROOT/'RAPID/validation/event_buffer_tests_20261005'
DAY=86400.


def settle(t,q,after,final,tolerance,dwell=DAY,buffer=DAY):
    """Last exceedance, then full dwell + extra buffer; no early transient crossing."""
    tail=np.flatnonzero(t>=after)
    outside=tail[np.abs(q[tail]-final)>tolerance]
    i=int(outside[-1]+1) if len(outside) else int(tail[0])
    if i>=len(t):return np.nan,np.nan
    arrival=float(t[i]);end=arrival+dwell+buffer
    return arrival,end if end<=t[-1] else np.nan


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    windows=pd.read_csv(ROOT/'RAPID/reviews/proposed_event_windows_20261005.csv',dtype={'example':str}).set_index('example')
    scans=[];summary=[];tests=[];selections=[]
    for ex,w in windows.iterrows():
        h=pd.read_csv(base_directory(ex)/'run/outlet_hydrograph.csv')
        t=h.time_seconds.to_numpy();flow=h.q_inflow_cms.to_numpy();dt=float(np.median(np.diff(t)))
        np.testing.assert_allclose(np.diff(t),dt,atol=1e-8,rtol=0)
        clock,_=event_clock(ex,h)
        changed=np.flatnonzero(flow!=flow[0]);lastchanged=np.flatnonzero(flow!=flow[-1])
        initial=int(changed[0]);event_end=int(lastchanged[-1]+1)
        A=float(h.q_outlet_cms.max()-max(h.q_outlet_cms.iloc[0],h.q_outlet_cms.iloc[-1]));assert A>0
        inv=pd.read_csv(DEFAULT_REFRESH_ROOT/ex/'inventory.csv')
        good=inv[inv.status.eq('ready')].drop_duplicates('fingerprint')
        basefp=base_directory(ex).name
        local=[]
        for r in good.itertuples():
            run=DEFAULT_REFRESH_ROOT/ex/r.fingerprint/'run'
            saved=pd.read_csv(run/'outlet_hydrograph.csv',usecols=['time_seconds','q_outlet_cms'])
            np.testing.assert_allclose(saved.time_seconds,t,rtol=0,atol=1e-8)
            arrive,end=settle(t,saved.q_outlet_cms.to_numpy(),t[event_end],flow[-1],.001*A)
            local.append(dict(example=ex,state=r.state,fingerprint=r.fingerprint,n_segments=r.n_segments,
                              old_tail_settle_days=(arrive-t[event_end])/DAY,
                              old_tail_required_days=(end-t[event_end])/DAY,
                              old_tail_censored=not np.isfinite(end)))
        scans.extend(local);audit=pd.DataFrame(local)
        pd.DataFrame(scans).to_csv(OUT/'all_state_saved_settlement.csv',index=False)
        start,end=pd.to_datetime(w.proposed_start_utc),pd.to_datetime(w.proposed_end_exclusive_utc)
        idx=np.flatnonzero(clock.between(start,end,inclusive='left').to_numpy());assert len(idx)>3
        event=flow[idx];q0,qe=float(event[0]),float(event[-1])
        old_pad_days=(t[initial]-t[0])/DAY
        old_dynamic_steps=len(t)-initial
        chosen={}
        def add(fp,reason):
            chosen.setdefault(fp,[]).append(reason)
        add(basefp,'parent')
        add(good.loc[good.max_travel_seconds.idxmax(),'fingerprint'],'largest cumulative travel-time bound')
        add(audit.loc[audit.old_tail_settle_days.idxmax(),'fingerprint'],'latest saved-output settlement')
        if ex=='37':
            for state in ['CB001_bubble_3','S274_unit_280','island_37_272']:
                add(good[good.state.eq(state)].iloc[0].fingerprint,'example-37 shape/peak case')
        clock_config=json.loads((base_directory(ex)/'run/Qout.audit.json').read_text())
        routing_dt=float(clock_config['routing_dt_seconds']);nsub=round(dt/routing_dt)
        assert abs(nsub*routing_dt-dt)<1e-8
        # Bound is for an isolated candidate test, not the proposed final buffer.
        tail_days=max(4.,3*float(good.max_travel_seconds.max())/DAY+3)
        nr=int(np.ceil(tail_days*DAY/dt));new_flow=np.r_[event,np.full(nr,qe)]
        new_t=np.arange(len(new_flow))*dt
        traces={};local_tests=[];bench=None
        for fp,reasons in chosen.items():
            r=good[good.fingerprint.eq(fp)].iloc[0];f=pd.read_parquet(r.frame)
            manifest=json.loads((DEFAULT_REFRESH_ROOT/ex/fp/'policy_manifest.json').read_text())
            assert manifest['prep_config']['initial_flow_mode']=='steady_first_inflow'
            assert manifest['prep_config']['x_value']==.1
            selections.append(dict(example=ex,state=r.state,fingerprint=fp,reasons='; '.join(reasons)))
            # Cross-check the spatial recurrence against production sparse operations
            # on the new event, and against saved production output on the old forcing.
            probe=event[:32]
            sparse,seconds,_=sparse_benchmark(f,probe,nsub,routing_dt,q0)
            fast,_=cascade(f,probe,nsub,routing_dt,q0)
            sparse_error=float(np.max(abs(fast-sparse)));assert sparse_error<1e-7*max(1.,np.ptp(event))
            old_probe=flow[initial:initial+128]
            old_out,_=cascade(f,old_probe,nsub,routing_dt,float(flow[0]))
            saved=pd.read_csv(DEFAULT_REFRESH_ROOT/ex/fp/'run/outlet_hydrograph.csv',usecols=['q_outlet_cms']).q_outlet_cms.to_numpy()
            saved_error=float(np.max(abs(old_out-saved[initial:initial+len(old_probe)])))
            assert saved_error<1e-6*max(1.,np.ptp(flow)),saved_error
            if fp==basefp:bench=seconds/len(probe)
            # Same event sample phase in each test; nominal buffer rounded up to dt.
            spin_errors={}
            for hours in [0,6,24]:
                n=int(np.ceil(hours*3600/dt))
                test,_=cascade(f,np.r_[np.full(n,q0),event[:128]],nsub,routing_dt,q0)
                ref,_=cascade(f,event[:128],nsub,routing_dt,q0)
                spin_errors[hours]=float(np.max(abs(test[n:]-ref)))
                assert spin_errors[hours]<1e-7*max(1.,np.ptp(event))
            begin=time.perf_counter();wave,_=cascade(f,new_flow,nsub,routing_dt,q0);elapsed=time.perf_counter()-begin
            traces[fp]=wave
            pd.DataFrame(dict(time_since_event_start_seconds=new_t,inflow_cms=new_flow,outlet_cms=wave)).to_parquet(OUT/f'{ex}_{fp}.parquet',index=False)
            local_tests.append(dict(example=ex,state=r.state,fingerprint=fp,n_segments=r.n_segments,
                                    sparse_parity_max_cms=sparse_error,saved_parity_max_cms=saved_error,
                                    spin0_max_cms=spin_errors[0],spin6_max_cms=spin_errors[6],spin24_max_cms=spin_errors[24],
                                    sparse_seconds_per_interval=seconds/len(probe),cascade_elapsed_seconds=elapsed))
        parent=traces[basefp];amplitude=float(parent.max()-max(q0,qe));assert amplitude>0
        cuts=[]
        for rec in local_tests:
            wave=traces[rec['fingerprint']]
            arrive,cut=settle(new_t,wave,new_t[len(event)-1],qe,.001*amplitude)
            assert np.isfinite(cut),'Extend candidate tail before interpreting'
            rec.update(new_settle_tail_days=(arrive-new_t[len(event)-1])/DAY,
                       new_required_tail_days=(cut-new_t[len(event)-1])/DAY,
                       final_error_parent_amplitude=abs(wave[-1]-qe)/amplitude,
                       new_peak_change_vs_parent_hours=(peak(new_t,wave)-peak(new_t,parent))/3600)
            cuts.append(cut)
        common=max(cuts);cut_index=int(np.searchsorted(new_t,common))
        for rec in local_tests:
            wave=traces[rec['fingerprint']]
            rec['omitted_tail_max_parent_amplitude']=float(np.max(abs(wave[cut_index:]-qe))/amplitude)
            rec['extra_day_tail_max_parent_amplitude']=float(np.max(abs(wave[cut_index-min(cut_index,round(DAY/dt)):cut_index+1]-qe))/amplitude)
            rec['peak_kept']=bool(np.argmax(wave)<=cut_index)
            assert rec['peak_kept'] and rec['omitted_tail_max_parent_amplitude']<=.001
        tests.extend(local_tests)
        nspin=int(np.ceil(DAY/dt));new_steps=cut_index+1+nspin
        # Estimates combine selected-state tail and all-state node/substep costs.
        old_span=(t[-1]-t[0])/DAY;new_span=(new_steps-1)*dt/DAY
        summary.append(dict(example=ex,total_states=len(good),tested_states=len(chosen),
                            old_total_days=old_span,old_initial_plateau_days=old_pad_days,
                            event_days=(len(event)-1)*dt/DAY,new_common_tail_days=(common-new_t[len(event)-1])/DAY,
                            proposed_total_days=new_span,duration_reduction_pct=100*(1-new_span/old_span),
                            old_dynamic_intervals=old_dynamic_steps,proposed_dynamic_intervals=cut_index+1,
                            dynamic_work_reduction_pct=100*(1-(cut_index+1)/old_dynamic_steps),
                            node_substeps_old=old_dynamic_steps*nsub*good.n_segments.sum(),
                            node_substeps_proposed=(cut_index+1)*nsub*good.n_segments.sum(),
                            output_values_old=len(t)*good.n_segments.sum(),output_values_proposed=new_steps*good.n_segments.sum(),
                            parent_sparse_seconds_per_interval=bench,
                            all_state_saved_tail_censored=int(audit.old_tail_censored.sum()),
                            new_parent_tail_days=next(r['new_required_tail_days'] for r in local_tests if r['fingerprint']==basefp)))
        pd.DataFrame(tests).to_csv(OUT/'selected_state_tests.csv',index=False)
        pd.DataFrame(summary).to_csv(OUT/'duration_runtime_projection.csv',index=False)
        pd.DataFrame(selections).to_csv(OUT/'selection.csv',index=False)
        print(ex,'tested',len(chosen),'of',len(good),'duration reduction',round(summary[-1]['duration_reduction_pct'],1),'dynamic reduction',round(summary[-1]['dynamic_work_reduction_pct'],1),flush=True)


if __name__=='__main__':main()
