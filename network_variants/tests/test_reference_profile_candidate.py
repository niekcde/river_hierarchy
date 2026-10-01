from __future__ import annotations

import numpy as np
import pandas as pd

from network_variants.reference_profile_candidate import (
    DOMAIN_MINIMUM_REFERENCE_SLOPE_FRACTION,
    DOMAIN_OBJECTIVE_NAME,
    solve_boundary_conditioned_domain,
)


def test_boundary_conditioned_parallel_domain_preserves_every_path_head_loss() -> None:
    links = pd.DataFrame(
        {
            "id_link": [1, 2, 3, 4],
            "id_us_node": [10, 11, 10, 12],
            "id_ds_node": [11, 20, 12, 20],
            "actual_length_m": [100.0, 100.0, 150.0, 150.0],
        }
    )
    solved, audit, heads = solve_boundary_conditioned_domain(
        links,
        source=10,
        sink=20,
        source_head_m=110.0,
        sink_head_m=100.0,
        reference_distance_m=1_000.0,
    )
    assert set(heads) == {10, 11, 12, 20}
    assert (solved["candidate_slope_m_per_m"] > 0.0).all()
    assert audit["n_paths"] == 2
    assert audit["max_abs_path_residual_m"] < 1e-8
    assert audit["objective_name"] == DOMAIN_OBJECTIVE_NAME
    assert audit["minimum_reference_slope_fraction"] == DOMAIN_MINIMUM_REFERENCE_SLOPE_FRACTION
    assert (solved["candidate_slope_m_per_m"] >= 0.01 * audit["local_reference_slope_m_per_m"]).all()
    assert np.isclose(audit["minimum_path_drop_m"], 10.0)
    assert np.isclose(audit["maximum_path_drop_m"], 10.0)


def test_boundary_conditioned_direct_link_uses_exact_boundary_drop() -> None:
    links = pd.DataFrame(
        {
            "id_link": [5],
            "id_us_node": [1],
            "id_ds_node": [2],
            "actual_length_m": [250.0],
        }
    )
    solved, audit, _ = solve_boundary_conditioned_domain(
        links,
        source=1,
        sink=2,
        source_head_m=101.0,
        sink_head_m=100.0,
        reference_distance_m=1_000.0,
    )
    assert np.isclose(float(solved.iloc[0]["candidate_slope_m_per_m"]), 1.0 / 250.0)
    assert audit["optimizer_message"] == "direct_boundary_solution"
