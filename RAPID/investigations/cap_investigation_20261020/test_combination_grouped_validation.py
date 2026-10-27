import unittest

import numpy as np
import pandas as pd

from combination_expansion import resistance, depth_features
from combination_grouped_validation import singleton_folds, training_mask


class CombinationValidationTests(unittest.TestCase):
    def test_singleton_assignment_balanced_and_reproducible(self):
        d = pd.DataFrame(dict(example=['01']*40+['02']*40,
                              state=[str(i) for i in range(80)],
                              response_bin=['near_zero_<=1min', 'advance_5to60min']*40))
        a = singleton_folds(d, 42)
        np.testing.assert_array_equal(a, singleton_folds(d, 42))
        np.testing.assert_array_equal(np.bincount(a), [16]*5)
        # Network identity is deliberately NOT the split group.
        self.assertEqual(len(set(a[:40])), 5)
        self.assertFalse(np.array_equal(a, singleton_folds(d, 43)))

    def test_holdout_family_never_split_and_singletons_independent(self):
        d = pd.DataFrame(dict(cohort=['original']*3+['segmented']+['combined']*3,
            previous80=[False]*4+[True, True, False],
            group_id=['family_a', 'family_a', 'family_b', 'full', 'combo_a', 'combo_b', 'combo_c']))
        folds = np.array([0, 0, 1, 1, 0, 1, 1])
        np.testing.assert_array_equal(np.flatnonzero(training_mask(d, folds, 0, 'baseline')), [2, 3])
        np.testing.assert_array_equal(np.flatnonzero(training_mask(d, folds, 0, 'plus80')), [2, 3, 5])
        np.testing.assert_array_equal(np.flatnonzero(training_mask(d, folds, 0, 'plus240')), [2, 3, 5, 6])

    def test_quadratic_resistance_parallel_and_series(self):
        # Two equal parallel K=8 branches have equivalent K=2.
        f = pd.DataFrame(dict(id_us_node=[0, 0], id_ds_node=[1, 1],
                              link_length_m=[8., 8.], width=[1., 1.], depth_m=[1., 1.]))
        self.assertAlmostEqual(resistance(f)['K_eq'], 2., places=7)
        f.id_us_node = [0, 1]
        f.id_ds_node = [1, 2]
        self.assertAlmostEqual(resistance(f)['K_eq'], 16., places=7)

    def test_depth_delta_excludes_unchanged_connector(self):
        parent = pd.DataFrame(dict(id_link=[1, 2], id_us_node=[0, 1], id_ds_node=[1, 2],
            link_length_m=[8., 100.], width=[1., 1.], depth_m=[2., 1.], slope=[.25, .01],
            reference_flow_fraction=[1., 1.]))
        child = parent.copy()
        child.loc[0, 'depth_m'] = 4.
        lineage = pd.DataFrame(dict(role=['collapsed_block', 'unchanged'], parent_link_id=[1, 2], child_link_id=[1, 2]))
        features, _ = depth_features(parent, child, lineage, resistance(parent))
        self.assertAlmostEqual(features['delta_G_inverse_depth'], -4.)
        self.assertAlmostEqual(features['resistance_explicit_depth'], -np.log10(100.125/101.), places=7)


if __name__ == '__main__':
    unittest.main()
