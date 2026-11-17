"""Focused launcher checks: real parameter prep/routing and mocked orchestration."""
import ast,json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import run_three_runs_control_fix as runner

def test_calibration05_exception_only_accepts_reviewed_result():
    best=dict(n_manning=.039486035241004865,beta=.5,timing_error_minutes=-.5,peak_cms=982.528101739409)
    fine=dict(timing_error_minutes=1.5,peak_cms=982.5705732644797)
    def assess(run=runner.RUNS[1],ex='05',control='half_dx_half_dt',b=best,c=fine):
        return runner.assess_calibration_convergence(run,ex,control,b,c)
    review=assess()
    assert review['accepted'] and review['accepted_exception'] and not review['thresholds_passed']
    for kw in [dict(run=runner.RUNS[2]),dict(ex='54'),dict(control='half_dt'),
               dict(b=dict(best,beta=1.)),dict(c=dict(fine,timing_error_minutes=2.5)),
               dict(c=dict(fine,peak_cms=990.))]:
        assert not assess(**kw)['accepted']
    normal=assess(c=dict(fine,timing_error_minutes=.5))
    assert normal['accepted'] and normal['thresholds_passed'] and not normal['accepted_exception']

def test_reviewed_convergence_exception_is_narrow():
    row=dict(example='54',state='all_closed_units_collapsed',control='half_dx_half_dt',window='0.5%',
             centroid_contrast_change_seconds=1.,peak_contrast_change_seconds=180.,peak_relative_change=.0001)
    checks,review=runner.assess_convergence(runner.RUNS[0],pd.DataFrame([row]))
    assert review['convergence_accepted'] and not review['convergence_checks_passed']
    assert checks.accepted_exception.all()
    for change in [dict(example='03'),dict(state='S001_unit_1'),dict(control='half_dt'),
                   dict(peak_contrast_change_seconds=240.),dict(centroid_contrast_change_seconds=61.),
                   dict(peak_relative_change=.002)]:
        _,review=runner.assess_convergence(runner.RUNS[0],pd.DataFrame([dict(row,**change)]))
        assert not review['convergence_accepted']
    for run in runner.RUNS[1:]:
        _,review=runner.assess_convergence(run,pd.DataFrame([row]))
        assert not review['convergence_accepted']


def test_single_centerline_remains_visible_but_not_required():
    rows=[dict(example='03',state=s,control='half_dx_half_dt',window='0.5%',
               centroid_contrast_change_seconds=0.,peak_contrast_change_seconds=300.,peak_relative_change=.0001)
          for s in ['single_centerline','fully_collapsed']]
    checks,review=runner.assess_convergence(runner.RUNS[1],pd.DataFrame(rows))
    assert review['convergence_accepted'] and review['excluded_diagnostic_rows']==2
    assert not checks.thresholds_passed.any() and not checks.accepted_exception.any()

@pytest.fixture
def ctx(tmp_path,monkeypatch):
    monkeypatch.setattr(runner,'ensure_space',lambda *a,**kw:100.)
    runner.init_worker(tmp_path,'test-fingerprint')
    return tmp_path


def test_parent_reproduction_and_midpoint_pair(ctx):
    for ex in ['02','22','54']:
        base,_,_=runner.parent_inputs(ex)
        n=float(base.n_manning.iloc[0]);beta=float(base.diffusion_multiplier.iloc[0])
        g,dt,_=runner.prepare_frame(ex,'S000_base',runner.RUNS[0])
        f=runner.hydraulic_parameters(g,runner.RUNS[0],n,beta)
        np.testing.assert_allclose(f[['rapid_k','rapid_x','depth_m']],base[['rapid_k','rapid_x','depth_m']],rtol=1e-10,atol=1e-10)
        a,da,_=runner.prepare_frame(ex,'S000_base',runner.RUNS[1]);b,db,_=runner.prepare_frame(ex,'S000_base',runner.RUNS[2])
        np.testing.assert_array_equal(a['count'],b['count']);assert da==db
        for run,grid in [(runner.RUNS[1],a),(runner.RUNS[2],b)]:
            f=runner.hydraulic_parameters(grid,run,n,beta)
            q,audit=runner.solve_wave(f,np.full(3,1000.),da,1000.)
            np.testing.assert_allclose(q,1000,atol=1e-7)
            assert audit['minimum_native_flow_cms']>=0


def test_restored_selected_link(ctx):
    for ex,state,newid,pid in [('22','S135_unit_211',-632,632),('37','S051_unit_256',-769,769),('37','S289_unit_143',-445,445)]:
        g,_,_=runner.prepare_frame(ex,state,runner.RUNS[0]);base,_,_=runner.parent_inputs(ex)
        assert g.set_index('id_link').loc[newid,'width']==base.set_index('id_link').loc[pid,'width']


def test_resume_detects_corruption(ctx):
    result=runner.calibrate_task((runner.RUNS[0],'22'))
    assert result['retained_existing_calibration']
    assert runner.calibrate_task((runner.RUNS[0],'22'))==result
    p=ctx/runner.RUNS[0]/'calibration/22/result.json';p.write_text('{}')
    with pytest.raises(RuntimeError,match='Resume output mismatch'):runner.calibrate_task((runner.RUNS[0],'22'))


@pytest.mark.parametrize('ex',['03','54'])
def test_calibration_point_targets_and_feature_scope(ctx,monkeypatch,ex):
    # Keep real archived targets and feature function, use a deterministic narrow
    # crest at its target to test selection/checkpoints without fitting real data.
    case=runner.CTX['cases'][ex];goal=pd.Timestamp(case['timing_target']['target_utc'])
    t=pd.date_range(goal-pd.Timedelta(minutes=5),periods=11,freq='min')
    height=case['height_target'];peak=height.get('height_target_nominal_cms') or height['low_cms']
    q=np.array([peak*.5]*5+[peak]+[peak*.5]*5)
    data=pd.DataFrame(dict(time_utc=t,forcing_cms=q,downstream_cms=q))
    original=pd.read_parquet
    monkeypatch.setattr(pd,'read_parquet',lambda path,*a,**kw:data.copy() if str(path)==case['calibration_forcing'] else original(path,*a,**kw))
    monkeypatch.setattr(runner,'solve_wave',lambda f,flow,dt,initial:(q.copy(),dict(minimum_native_flow_cms=1.)))
    result=runner.calibrate_task((runner.RUNS[1],ex))
    assert result['timing_tolerance_met'] and len(result['numerical_checks'])==2
    assert runner.calibrate_task((runner.RUNS[1],ex))==result


def test_three_stage_order_and_failure_stops(ctx,monkeypatch):
    seen=[]
    monkeypatch.setattr(runner,'ensure_space',lambda *a,**kw:100.)
    # Exercise driver sequencing with its real lock/manifest, small fake stages.
    real_init=runner.init_worker
    def init(*args):
        real_init(*args)
        runner.CTX['cases']={'22':runner.CTX['cases']['22']}
        runner.CTX['cases']['22']['jobs']=[runner.CTX['cases']['22']['jobs'][0]]
    monkeypatch.setattr(runner,'init_worker',init)
    def pool(fn,tasks,*args):
        run=tasks[0][0];seen.append((run,fn.__name__))
        if fn==runner.calibrate_task:
            if run==runner.RUNS[1]:raise RuntimeError('intentional second-stage failure')
            return [dict(example='22',selected=dict(n_manning=.02,beta=1.))]
        return []
    monkeypatch.setattr(runner,'pool_tasks',pool)
    monkeypatch.setattr(runner,'finish_stage',lambda *args:None)
    with pytest.raises(RuntimeError,match='intentional second-stage failure'):runner.run_batch(ctx,1,dict(files={}))
    assert [x[0] for x in seen]==[runner.RUNS[0],runner.RUNS[0],runner.RUNS[1]]
    assert not any(x[0]==runner.RUNS[2] for x in seen)


def smoke_calibrate(args):
    runner.ensure_space=lambda *a,**kw:100.  # Tiny temporary-storage test only.
    return runner.calibrate_task(args)

def smoke_route(args):
    runner.ensure_space=lambda *a,**kw:100.  # Tiny temporary-storage test only.
    return runner.route_state(args)

def test_two_worker_actual_route_and_resume(ctx):
    run=runner.RUNS[0]
    fits=runner.pool_tasks(smoke_calibrate,[(run,'48')],2,ctx,'test-fingerprint','test calibration')
    fit=fits[0]['selected'];args=[(run,'48','S000_base',fit)]
    result=runner.pool_tasks(smoke_route,args,2,ctx,'test-fingerprint','test routing')[0]
    assert result['settled']
    assert runner.pool_tasks(smoke_route,args,2,ctx,'test-fingerprint','test resume')[0]==result
    produced=pd.read_parquet(ctx/run/'states/48/S000_base/metrics.parquet').set_index('window')
    saved=runner.CTX['windows'][runner.CTX['windows'].example.eq('48')].set_index('window')
    np.testing.assert_allclose(produced.loc[saved.index,['centroid_seconds','peak_cms','volume_m3']],saved[['centroid_seconds','peak_cms','volume_m3']],rtol=1e-10,atol=1e-6)


def test_diagnostic_signed_moments_and_material_negative_rejected():
    from convergence_moments import metrics
    times=pd.date_range('2026-01-01',periods=5,freq='min',tz='UTC')
    force=pd.DataFrame(dict(time_utc=times));t=times.astype('int64').to_numpy()/1e9
    w=pd.DataFrame([dict(window='0.5%',rise_seconds=t[0],fall_seconds=t[-1],baseline_cms=1000.,threshold_fraction=.005,rise_utc=str(times[0]),fall_utc=str(times[-1]))])
    q=np.array([1000.,1100.,1200.,1100.,1000.])
    exact=runner.state_metrics(t,q,w.iloc[0],'S000_base',[])
    d=metrics('22','S000_base',force,q,w).iloc[0]
    assert d.centroid_seconds==exact['centroid_seconds'] and d.peak_cms==exact['peak_cms']
    q[0]-=1e-6
    d=metrics('22','S000_base',force,q,w).iloc[0]
    assert d.minimum_excess_cms<0 and d.negative_excess_volume_m3>0
    assert d.signed_excess_volume_m3==pytest.approx(np.trapz(q-1000,t))
    q[0]=999.
    with pytest.raises(ValueError,match='baseline review required'):metrics('22','S000_base',force,q,w)


def test_actual_unused_control_baseline_failure_keeps_required_states_strict(ctx):
    path=runner.SETUP/'midpoint_baseline_diagnosis/22_single_centerline.parquet'
    wave=pd.read_parquet(path)
    q=wave.q_outlet_cms.to_numpy();original=q.copy()
    metrics=runner.metrics_for('22','single_centerline',[],wave,q)
    assert len(metrics)==3 and metrics.metrics_status.eq('unavailable_unused_control').all()
    assert metrics.centroid_seconds.isna().all() and metrics.volume_m3.isna().all()
    np.testing.assert_array_equal(q,original)
    with pytest.raises(ValueError,match='Negative excess'):
        runner.metrics_for('22','S001_unit_1',[],wave,q)
    # Missing response metrics must survive downstream response calculation.
    w=runner.CTX['windows'][runner.CTX['windows'].example.eq('22')]
    safe=np.maximum(q,float(w.baseline_cms.max())+1.)
    parent=runner.metrics_for('22','S000_base',[],wave,safe)
    responses=runner.response_rows(pd.concat([parent,metrics],ignore_index=True))
    assert len(responses)==3 and responses.centroid_shift_hours.isna().all()


def test_roundoff_actual48_is_audited_without_modifying_wave(ctx):
    wave=pd.read_parquet(runner.SETUP/'manning48_baseline_diagnosis/48_S045_unit_2.parquet')
    q=wave.q_outlet_cms.to_numpy();before=q.copy()
    metrics=runner.metrics_for('48','S045_unit_2',[],wave,q)
    row=metrics[metrics.window.eq('0.5%')].iloc[0]
    assert row.roundoff_negative_samples==2 and row.roundoff_added_volume_fraction<1e-12
    assert np.isfinite(metrics.centroid_seconds).all()
    np.testing.assert_array_equal(before,q)
    from manning_network_pilot import state_metrics as strict
    t=wave.time_utc.astype('int64').to_numpy()/1e9
    for _,w in runner.CTX['windows'][runner.CTX['windows'].example.eq('48') & runner.CTX['windows'].window.ne('0.5%')].iterrows():
        expected=strict(t,q,w,'S045_unit_2',[])
        actual=metrics[metrics.window.eq(w.window)].iloc[0]
        assert actual.centroid_seconds==expected['centroid_seconds']


def test_roundoff_rejects_material_deficit_and_excessive_relative_volume():
    from roundoff_metrics import state_metrics
    t=np.arange(5,dtype=float)*60
    w=pd.Series(dict(rise_seconds=0.,fall_seconds=240.,baseline_cms=1000.,window='0.5%',rise_utc='a',fall_utc='b',threshold_fraction=.005))
    for q in [np.array([999.9999,1100,1200,1100,1000]),np.array([1000-1e-10,1000,1000+1e-8,1000,1000])]:
        with pytest.raises(ValueError,match='Negative excess'):state_metrics(t,q,w,'S000_base',[])
