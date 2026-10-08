"""Scientific invariants for the isolated travel-time calibration notebook."""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'investigations'))
import travel_time_calibration as calibration


def test_peak_definition_preserves_disconnected_ties_and_first_maximum():
    t = pd.date_range('2025-01-01', periods=7, freq='h', tz='UTC')
    p = calibration.peak_summary(t, [1, 3, 3, 2, 3, 2, 1])
    assert p['first'] == t[1]
    assert p['last'] == t[4]
    assert p['tied_span_h'] == 3
    assert p['disconnected_ties'] and not p['boundary']


def test_roughness_changes_time_without_changing_geometry_and_respects_caps():
    f = pd.DataFrame(dict(width=[100.,100.], slope=[.0001,.01],
                          rapid_effective_length_m=[200.,200.], link_length_m=[500.,500.],
                          depth_m=[2.,2.], count=[3,3]))
    reference = dict(reference_width_m=100.,reference_depth_m=2.)
    a = calibration.candidate_frame(f,reference,.035)
    b = calibration.candidate_frame(f,reference,.07)
    assert b.rapid_k.iloc[0] == pytest.approx(2*a.rapid_k.iloc[0])
    assert a.rapid_celerity_mps.iloc[1] == 5.
    pd.testing.assert_series_equal(a.link_length_m,f.link_length_m)
    pd.testing.assert_series_equal(b.depth_m,f.depth_m)
    with pytest.raises(ValueError, match='positive'):
        calibration.candidate_frame(f,reference,-.01)


def test_settlement_rejects_temporary_plateau_and_requires_full_buffer():
    t = np.arange(9)*86400.
    q = np.array([20,15,10,10,12,10,10,10,10.])
    arrival, end = calibration.settle(t,q,t[1],10,.01)
    assert arrival == t[5] and end == t[7]
    _, censored = calibration.settle(t[:7],q[:7],t[1],10,.01)
    assert np.isnan(censored)


def test_trial_extends_tail_and_uses_forcing_relative_lag(tmp_path, monkeypatch):
    t = pd.date_range('2025-01-02', periods=4, freq='h', tz='UTC')
    event = np.array([10.,20.,15.,10.])
    frame = pd.DataFrame(dict(rapid_k=[3600.], count=[1], rapid_celerity_capped=[False]))
    case = dict(signature='test', example='01', bounds=(.015,.08), frame=frame, reference={},
                routing_dt_s=600.,forcing_step_s=3600.,nsub=6,flow=event,times=t,
                window=pd.Series(dict(first_sample=t[0],last_sample=t[-1],
                                      simulation_start=t[0]-pd.Timedelta(days=1),
                                      simulation_end=t[-1]+pd.Timedelta(hours=2))),
                forcing_peak=dict(first=t[1]),target=dict(observed_lag_h=2.,target_usable=True,
                    target_notes='',timing_resolution_h=1.,upstream_peak_utc=t[1],
                    downstream_peak_utc=t[3]))
    def fake_cascade(f,flow,nsub,dt,q0,**kwargs):
        # One forcing-interval delay, including an exactly steady initial condition.
        q = np.r_[q0,np.asarray(flow)[:-1]]
        return q,[dict(minimum=10.,maximum=20.)]
    monkeypatch.setattr(calibration,'candidate_frame',lambda *args:frame)
    monkeypatch.setattr(calibration,'cascade',fake_cascade)
    monkeypatch.setattr(calibration,'sparse_benchmark',lambda *args,**kwargs:(fake_cascade(*args)[0],0,{}))
    result,wave = calibration.run_trial(case,.035,tmp_path)
    assert result['tail_extensions'] > 0
    assert result['simulated_lag_h'] == 1.
    assert result['abs_error_h'] == 1.
    assert wave.time_utc.iloc[-1] >= t[-1]+pd.Timedelta(days=2)
    cached,_ = calibration.run_trial(case,.035,tmp_path)
    assert cached == result


def test_search_reports_boundary_without_claiming_interior_optimum(tmp_path, monkeypatch):
    case = dict(example='01',signature='test-search',bounds=(.015,.08),
                target=dict(target_usable=True))
    def trial(case,n,output):
        return dict(n_manning=float(n),abs_error_h=float(n*100),timing_resolution_h=.1), None
    monkeypatch.setattr(calibration,'run_trial',trial)
    trials,best = calibration.calibrate(case,[.015,.035,.08],tmp_path)
    assert len(trials) == 3
    assert best['best_at_search_boundary']
    assert best['n_manning'] == .015
    with pytest.raises(ValueError,match='exceeds'):
        calibration.calibrate(case,[.01,.035,.08],tmp_path)


def test_refinement_does_not_repeat_coarse_values_due_to_float_roundoff(tmp_path, monkeypatch):
    case = dict(example='01',signature='test-refinement',bounds=(.015,.08),
                target=dict(target_usable=True))
    def trial(case,n,output):
        return dict(n_manning=float(n),abs_error_h=float(abs(n-.06)),timing_resolution_h=.001), None
    monkeypatch.setattr(calibration,'run_trial',trial)
    trials,_ = calibration.calibrate(case,[.015,.035,.05,.06,.08],tmp_path)
    assert (np.diff(trials.n_manning)>1e-12).all()
