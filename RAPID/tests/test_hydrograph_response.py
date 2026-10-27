import numpy as np
import pytest
from rapid_tools.hydrograph_response import distribution


def test_symmetric_triangle_has_known_mass_moments_and_quantiles():
    m=distribution([0,1,2],[0,1,0])
    assert m['volume_m3']==pytest.approx(1)
    assert m['centroid_seconds']==pytest.approx(1)
    assert m['temporal_sd_seconds']==pytest.approx(np.sqrt(1/6))
    assert m['t10_seconds']==pytest.approx(np.sqrt(.2))
    assert m['t90_seconds']==pytest.approx(2-np.sqrt(.2))
    assert m['asymmetry']==pytest.approx(0,abs=1e-12)


def test_translation_stretching_and_height_scaling_have_distinct_effects():
    t=np.array([0.,1.,4.]);q=np.array([0.,3.,0.]);a=distribution(t,q)
    b=distribution(1.7e9+2*t,5*q)
    assert b['volume_m3']==pytest.approx(10*a['volume_m3'])
    assert b['centroid_seconds']==pytest.approx(1.7e9+2*a['centroid_seconds'],abs=1e-6)
    assert b['duration80_hours']==pytest.approx(2*a['duration80_hours'],rel=1e-6)
    assert b['asymmetry']==pytest.approx(a['asymmetry'],abs=1e-6)


def test_negative_excess_is_not_silently_accepted():
    with pytest.raises(ValueError):distribution([0,1,2],[0,1,-.01])


def test_shared_window_retains_positive_endpoints_and_volume_difference(tmp_path):
    import pandas as pd
    from rapid_tools.hydrograph_response import pulse
    t=pd.to_datetime([0,60,120],unit='s',utc=True)
    path=tmp_path/'outlet.parquet'
    pd.DataFrame({'time_utc':t,'q_outlet_cms':[12.,15.,13.]}).to_parquet(path)
    row=dict(outlet=str(path),rise_seconds=0.,fall_seconds=120.,baseline_cms=10.)
    seconds,e,adjustment=pulse(row)
    np.testing.assert_array_equal(e,[2.,5.,3.])
    assert adjustment==0
    a=distribution(seconds,e);b=distribution(seconds,e+[1.,2.,1.])
    assert a['volume_m3']==pytest.approx(450.)
    assert b['volume_m3']-a['volume_m3']==pytest.approx(np.trapz([1.,2.,1.],seconds))


def test_small_late_volume_shifts_centroid_by_volume_times_time_separation():
    # Two disjoint triangular pulses: area 1 at t=1, area .001 at t=1001.
    m=distribution([0.,1.,2.,1000.,1001.,1002.],[0.,1.,0.,0.,.001,0.])
    assert m['volume_m3']==pytest.approx(1.001)
    assert m['centroid_seconds']-1==pytest.approx(.001/1.001*1000.)
