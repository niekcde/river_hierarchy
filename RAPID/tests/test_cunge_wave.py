import numpy as np
from rapid_tools.cunge_wave import midpoint_discharge


def test_spatial_average_cannot_choose_midpoint_peak():
    q = np.array([[10, 10, 0, 10, 10, 10], [0, 0, 9, 0, 0, 0]])
    assert q.mean(axis=1).argmax() == 0
    assert midpoint_discharge(q).argmax() == 1


def test_midpoint_interpolates_endpoint_samples_and_single_cell_proxy():
    np.testing.assert_allclose(midpoint_discharge([[2, 4, 6]]), [3])
    np.testing.assert_allclose(midpoint_discharge([[2, 4, 6, 8]]), [4])
    np.testing.assert_allclose(midpoint_discharge([[2], [4]]), [2, 4])


def test_centroid_exact_linear_moment_on_shared_window():
    from rapid_tools.cunge_wave import wave_centroid
    # Rising triangular excess: centroid at 2/3 of the duration, not peak time.
    centroid, baseline = wave_centroid([0., 3.], [5., 8.], 0., 3., 5.)
    np.testing.assert_allclose(centroid, 2.)
    assert baseline == 5.
    # Boundary interpolation gives the same result on a cropped interval.
    centroid, _ = wave_centroid([0., 3.], [5., 8.], 1., 3., 6.)
    np.testing.assert_allclose(centroid, 1 + 4/3)


def test_centroid_reports_lowered_local_baseline():
    from rapid_tools.cunge_wave import wave_centroid
    centroid, baseline = wave_centroid([0., 1., 2.], [1., 3., 1.], 0., 2., 2.)
    assert baseline == 1.
    np.testing.assert_allclose(centroid, 1.)
