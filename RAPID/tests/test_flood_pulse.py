import numpy as np
import pandas as pd
import pytest
from shapely.geometry import LineString
from rapid_tools.flood_pulse import prepare_pulse_field, pulse_intensity, interpolate_discharge


def test_rise_and_recession_follow_local_discharge_on_shared_clock():
    reaches=pd.DataFrame({'reference_flow_fraction':[1.,0.25],
        'geometry':[LineString([(0,0),(1,0)]),LineString([(1,0),(2,0)])]})
    t=np.array([0.,1.,2.,3.])
    q=np.array([[4.,1.],[8.,1.],[4.,2.],[4.,1.]])
    field=prepare_pulse_field(reaches,q,t,0.,3.,4.)
    np.testing.assert_allclose(pulse_intensity(field,0),[0,0])
    np.testing.assert_allclose(pulse_intensity(field,1),[1,0])
    np.testing.assert_allclose(pulse_intensity(field,1.5),[0.5,0.5])
    np.testing.assert_allclose(pulse_intensity(field,2),[0,1])
    np.testing.assert_allclose(pulse_intensity(field,3),[0,0])
    np.testing.assert_allclose(interpolate_discharge(t,q,1.5),[6,1.5])
    with pytest.raises(ValueError,match='outside'):
        pulse_intensity(field,4)


def test_normalization_uses_reviewed_window_and_retains_final_excess():
    reaches=pd.DataFrame({'reference_flow_fraction':[1.],
        'geometry':[LineString([(0,0),(1,0)])]})
    t=np.arange(5.)
    q=np.array([[100.],[2.],[6.],[4.],[100.]])
    field=prepare_pulse_field(reaches,q,t,1.,3.,3.)
    assert field['baseline_lowered_count']==1
    np.testing.assert_allclose(field['baseline'],[2.])
    np.testing.assert_allclose(pulse_intensity(field,2),[1.])
    np.testing.assert_allclose(pulse_intensity(field,3),[0.5])


def test_local_background_returns_each_reach_to_grey_with_changed_final_flow():
    reaches=pd.DataFrame({'reference_flow_fraction':[1.,0.25],
        'geometry':[LineString([(0,0),(1,0)]),LineString([(1,0),(2,0)])]})
    t=np.arange(5.)
    q=np.array([[2.,1.],[3.,4.],[8.,2.],[5.,2.5],[4.,3.]])
    field=prepare_pulse_field(reaches,q,t,0.,4.,2.,background='local_endpoints')
    np.testing.assert_allclose(pulse_intensity(field,0),[0.,0.])
    np.testing.assert_allclose(pulse_intensity(field,4),[0.,0.])
    np.testing.assert_allclose(pulse_intensity(field,2),[1.,0.])
    assert pulse_intensity(field,1)[1]==1.
    # No change to modelled discharge: only the display field changes.
    np.testing.assert_allclose(interpolate_discharge(t,field['qout'],4),[4.,3.])
