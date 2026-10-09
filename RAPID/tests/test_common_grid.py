import numpy as np
import pytest
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import factorized

from rapid_tools.common_grid import resample_points, outlet_interval_means, route_interval_means
from rapid_tools.cunge import freeze_grid, parameters, common_clock
from rapid_tools.engine import ccc_mat, rte_mat, mus_rte
import pandas as pd


def test_irregular_resampling_conserves_linear_reconstruction_volume():
    t = [0., 60., 210., 300.]
    q = [2., 5., 8., 3.]
    volumes = []
    for step in (300, 60, 30):
        means, gap, endpoint = resample_points(t, q, np.arange(0, 301, step),
                                               maximum_gap_seconds=180)
        volumes.append(means.sum()*step)
        assert not gap.any() and not endpoint.any()
    np.testing.assert_allclose(volumes, [1680]*3)


def test_gaps_and_endpoints_are_distinct_and_not_silently_accepted():
    with pytest.raises(ValueError, match='gap'):
        resample_points([0, 60, 300], [1, 2, 3], [0, 60, 120, 300], maximum_gap_seconds=120)
    q, gap, endpoint = resample_points([0, 60, 300], [1, 2, 3], [-60, 0, 60, 120, 300, 360],
        maximum_gap_seconds=120, endpoint_hold_seconds=60, diagnostic_gap_fill=True)
    np.testing.assert_array_equal(gap, [False, False, True, True, False])
    np.testing.assert_array_equal(endpoint, [True, False, False, False, True])
    assert q[0] == 1 and q[-1] == 3
    with pytest.raises(ValueError, match='hold'):
        resample_points([0, 60], [1, 2], [-61, 0], maximum_gap_seconds=120, endpoint_hold_seconds=60)


def test_model_means_use_initial_state_and_preserve_triangle_volume():
    np.testing.assert_allclose(outlet_interval_means([1, 2, 1, 0], 2, initial_flow=0), [1, 1])
    np.testing.assert_allclose(outlet_interval_means([2, 4], 2, initial_flow=0), [2])


@pytest.mark.parametrize('method', ['cunge', 'fixed_x'])
def test_common_means_and_explicit_initialization_match_sparse_solver(method):
    frame = pd.DataFrame(dict(id_link=[1], u=[0], v=[1], link_length_m=[100.],
                              width=[10.], slope=[.001], depth_m=[2.]))
    grid = freeze_grid(frame, 20.)
    f = parameters(grid, .035, method=method)
    assert int(f['count'].iloc[0]) == 1
    dt, ns = common_clock(grid, 60.)
    flow = np.array([4., 8., 6., 2.])
    actual, audit = route_interval_means(f, flow, ns, dt, initial_flow=2.)
    c1, c2, c3 = ccc_mat(f.rapid_k.to_numpy(), f.rapid_x.to_numpy(), dt)
    mats = rte_mat(csc_matrix((1, 1)), c1, c2, c3)
    solve = factorized(mats[0]); state = np.array([2.]); previous = np.array([2.])
    expected = []
    for q in flow:
        start = state.copy()
        rectangle, state = mus_rte(*mats, ns, state, np.array([q]), solve,
                                   previous_external=previous, previous_inflow_matrix=c2)
        expected.append(float((rectangle+(start-state)/(2*ns))[0]))
        previous = np.array([q])
    np.testing.assert_allclose(actual, expected, rtol=1e-11, atol=1e-11)
    assert audit['initial_outlet_cms'] == 2.
