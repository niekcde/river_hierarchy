import numpy as np, pandas as pd
import unittest
from network_variants._local_head_qp import fit_primary, fit_internal


class Tests(unittest.TestCase):
    def test_unpenalised_boundary_tie(self):
        x, qa = fit_primary(
            np.array([[0.0, 1.0]]),
            np.array([1.0]),
            np.array([[1.0, -1.0]]),
            np.array([2.0]),
        )
        np.testing.assert_allclose(x, [2.0, 0.0], atol=1e-8)
        self.assertTrue(qa["rank_deficient"])

    def test_joint_relative_drops(self):
        x, qa = fit_primary(
            np.eye(2), np.array([2.0, 1.0]), np.array([[1.0, -1.0]]), np.array([1.0])
        )
        np.testing.assert_allclose(x, [0.2, -0.8], atol=1e-8)

    def test_internal_shared_head_fit(self):
        f = pd.DataFrame(
            {
                "id_link": [1, 2],
                "u": [0, 1],
                "v": [1, 2],
                "link_length_m": [100.0, 200.0],
            }
        )
        h, qa = fit_internal(f, {0: 1.0, 2: 0.0}, 0.003, np.array([0.01, 0.02]))
        self.assertAlmostEqual(h[1], 0.68, places=7)

    def test_active_internal_floor(self):
        f = pd.DataFrame(
            {
                "id_link": [1, 2],
                "u": [0, 1],
                "v": [1, 2],
                "link_length_m": [100.0, 200.0],
            }
        )
        h, qa = fit_internal(f, {0: 1.0, 2: 0.0}, 0.003, np.array([0.8, 0.02]))
        self.assertAlmostEqual(h[1], 0.2, places=7)

    def test_zero_width_feasible_face(self):
        f = pd.DataFrame(
            {
                "id_link": [1, 2],
                "u": [0, 1],
                "v": [1, 2],
                "link_length_m": [100.0, 200.0],
            }
        )
        h, qa = fit_internal(f, {0: 1.0, 2: 0.0}, 0.003, np.array([0.4, 0.6]))
        self.assertAlmostEqual(h[1], 0.6, places=7)

    def test_infeasible_not_softened(self):
        f = pd.DataFrame(
            {
                "id_link": [1, 2],
                "u": [0, 1],
                "v": [1, 2],
                "link_length_m": [100.0, 200.0],
            }
        )
        with self.assertRaises(ValueError):
            fit_internal(f, {0: 1.0, 2: 0.0}, 0.003, np.array([0.8, 0.8]))


if __name__ == "__main__":
    unittest.main()
