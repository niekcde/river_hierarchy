import numpy as np
import pytest
from rapid_tools.time_support import interval_bounds_and_midpoints, point_series_interval_means, native_outlet_at_times
from rapid_tools.time_support import native_outlet_interval_means


def test_irregular_linear_observations_have_exact_means_and_volume():
    t=np.array([0.,1.,7.,10.]); q=2*t+3; edges=np.array([0.,2.,5.,10.])
    means=point_series_interval_means(t,q,edges)
    np.testing.assert_allclose(means,2*(edges[:-1]+edges[1:])/2+3)
    assert np.dot(means,np.diff(edges)) == pytest.approx(130.)


def test_triangle_integral_across_unsampled_bin_boundaries():
    means=point_series_interval_means([0,3,8],[0,6,0],[0,2,5,8])
    assert np.dot(means,[2,3,3]) == pytest.approx(24.)


def test_native_first_state_is_at_end_of_first_step():
    np.testing.assert_allclose(native_outlet_at_times([12,14,16],2,[0,1,2,5,6],initial_flow=10),[10,11,12,15,16])
    with pytest.raises(ValueError):native_outlet_at_times([12],2,[3],initial_flow=10)


def test_midpoint_relabelling_does_not_change_bin_volume_or_pair_shifts():
    bounds,t=interval_bounds_and_midpoints([0,1540,3080],1540)
    np.testing.assert_allclose(t,[770,2310,3850]);np.testing.assert_allclose(np.diff(bounds,axis=1).ravel(),1540)
    assert t[2]-t[1] == 1540


def test_extrapolation_is_explicit_and_duplicate_times_rejected():
    with pytest.raises(ValueError):point_series_interval_means([0,1],[2,4],[-1,2])
    np.testing.assert_allclose(point_series_interval_means([0,1],[2,4],[-1,0,1,2],allow_endpoint_hold=True),[2,3,4])
    with pytest.raises(ValueError):point_series_interval_means([0,0],[2,4],[0,1])


def test_gap_limit_only_checks_intersecting_observation_gaps():
    with pytest.raises(ValueError, match='gap'):
        point_series_interval_means([0, 2, 20], [0, 2, 20], [1, 3], max_gap_seconds=5)
    np.testing.assert_allclose(point_series_interval_means(
        [0, 2, 20], [0, 2, 20], [0, 2], max_gap_seconds=5), [1])


def test_endpoint_hold_is_bounded_separately_from_internal_gaps():
    with pytest.raises(ValueError, match='hold'):
        point_series_interval_means([0, 2], [2, 4], [-3, 2],
                                    allow_endpoint_hold=True, max_endpoint_hold_seconds=2)
    np.testing.assert_allclose(point_series_interval_means(
        [0, 2], [2, 4], [-2, 0, 2, 4], allow_endpoint_hold=True,
        max_endpoint_hold_seconds=2, max_gap_seconds=2), [2, 3, 4])


def test_interval_observations_use_exact_bounds_not_midpoint_samples():
    # Triangle: mean 1, midpoint point value 2, historical endpoint mean 1.
    np.testing.assert_allclose(native_outlet_interval_means(
        [2, 0], 1, [[0, 2], [0.5, 1.5], [1.5, 2]], initial_flow=0), [1, 1.5, .5])
    # Ramp: endpoint mean 3 differs from the integral mean 2.
    np.testing.assert_allclose(native_outlet_interval_means(
        [2, 4], 1, [[0, 2]], initial_flow=0), [2])


@pytest.mark.parametrize('bounds', [[[0, 3]], [[-1, 1]], [[1, 1]], [[2, 1]], [[0, np.nan]]])
def test_interval_comparison_rejects_invalid_or_unsupported_bounds(bounds):
    with pytest.raises(ValueError):
        native_outlet_interval_means([2, 4], 1, bounds, initial_flow=0)
