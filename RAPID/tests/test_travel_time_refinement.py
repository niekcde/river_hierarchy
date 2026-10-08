from pathlib import Path
import sys
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'investigations'))
import travel_time_refinement as refinement


@pytest.mark.parametrize('n,phrase',[(.011,'Very low'),(.014,'Low'),(.035,'common'),(.08,'High')])
def test_context_is_not_automatic_physical_acceptance(n,phrase):
    assert phrase in refinement.roughness_context(n)


def test_refines_low_boundary_and_keeps_common_band_tradeoff(tmp_path,monkeypatch):
    case = dict(example='01',bounds=(.01,.1),signature='low-case',forcing_step_s=3600.,
                target=dict(target_usable=True))
    calls=[]
    def trial(case,n,output):
        calls.append(n)
        error=n*1000-12.5
        return dict(example='01',n_manning=n,abs_error_h=abs(error),signed_error_h=error,
                    simulated_lag_h=n*1000,timing_resolution_h=.25,
                    forcing_minus_observed_upstream_peak_h=0.),None
    monkeypatch.setattr(refinement.cal,'run_trial',trial)
    previous=pd.DataFrame([trial(case,n,tmp_path)[0] for n in [.015,.035,.08]])
    calls.clear()
    trials,best=refinement.refine_case(case,previous,tmp_path)
    assert best['n_manning']==pytest.approx(.0125)
    assert best['abs_error_h']==0
    assert not best['best_at_search_boundary']
    assert not best['site_roughness_verified']
    assert best['common_band_best_n']==.025
    assert best['common_band_abs_error_h']==12.5
    assert not best['common_band_within_sampling_tolerance']
    assert len(calls)==len(set(calls))==len(trials)


def test_unreachable_target_stays_flagged_at_expanded_boundary(tmp_path,monkeypatch):
    case=dict(example='02',bounds=(.01,.1),signature='unreachable',forcing_step_s=3600.,
              target=dict(target_usable=True))
    def trial(case,n,output):
        error=n*1000-1
        return dict(example='02',n_manning=n,abs_error_h=abs(error),signed_error_h=error,
                    simulated_lag_h=n*1000,timing_resolution_h=.5,
                    forcing_minus_observed_upstream_peak_h=2.),None
    monkeypatch.setattr(refinement.cal,'run_trial',trial)
    previous=pd.DataFrame([trial(case,n,tmp_path)[0] for n in [.015,.035,.08]])
    _,best=refinement.refine_case(case,previous,tmp_path)
    assert best['n_manning']==.01
    assert best['best_at_search_boundary']
    assert not best['within_observation_sampling_tolerance']
    assert best['significant_forcing_peak_offset']
    assert 'no unresolved bracketed crossing' in best['search_stop']
