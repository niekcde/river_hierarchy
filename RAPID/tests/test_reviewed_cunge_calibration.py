"""UTC calibration must not mix gauge and prepared-inlet peak references."""
from pathlib import Path
import sys
import pandas as pd
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'investigations'))
from cunge_calibration_reviewed import timing_fields


def test_utc_objective_ignores_inlet_reference_offset():
    target=dict(downstream_peak_utc='2025-01-01T12:00:00Z',crest_start_utc='2025-01-01T11:00:00Z',crest_end_utc='2025-01-01T13:00:00Z',target_lag_from_prepared_inlet_h=5.)
    # Gauge peak 06:00 and prepared peak 07:00: subtracting those different
    # references would erroneously give a one-hour error for a perfect fit.
    result=timing_fields('2025-01-01T12:00:00Z',target)
    assert result['error_h']==0 and result['within_crest_interval']
    assert result['target_lag_from_prepared_inlet_h']==5
    inside=timing_fields('2025-01-01T12:30:00Z',target)
    assert inside['error_h']==.5 and inside['crest_interval_error_h']==0
    outside=timing_fields('2025-01-01T14:00:00Z',target)
    assert outside['error_h']==2 and outside['crest_interval_error_h']==1
    assert not outside['within_crest_interval']


def test_case48_actual_middle_peak_is_not_replaced_by_interval_midpoint():
    target=dict(downstream_peak_utc='2025-06-02T23:45:00Z',crest_start_utc='2025-06-02T22:15:00Z',crest_end_utc='2025-06-03T01:30:00Z',target_lag_from_prepared_inlet_h=8.)
    assert timing_fields('2025-06-02T23:45:00Z',target)['error_h']==0
    assert timing_fields('2025-06-02T23:52:30Z',target)['error_h']==pytest.approx(.125)
