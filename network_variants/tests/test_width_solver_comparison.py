from __future__ import annotations

import numpy as np
import pandas as pd

from network_variants.width_solver_comparison import (
    AdaptiveSolverConfig,
    _evaluate_method,
    adaptive_evidence_parameters,
    solve_adaptive_two_ended,
)


def test_adaptive_evidence_relaxes_bounds_when_correction_evidence_is_stronger() -> None:
    candidates = pd.DataFrame(
        {
            "width_model_probability": [0.50, 0.99],
            "width_model_threshold": [0.45, 0.45],
            "upstream_continuity_abs_deviation": [0.1, 5.0],
            "opposition_cos_trimmed_mean": [-1.0, 0.9],
        }
    )

    evidence = adaptive_evidence_parameters(candidates, config=AdaptiveSolverConfig())

    assert evidence.loc[1, "adaptive_min_scale"] < evidence.loc[0, "adaptive_min_scale"]
    assert evidence.loc[1, "adaptive_max_scale"] > evidence.loc[0, "adaptive_max_scale"]
    assert evidence.loc[1, "adaptive_raw_penalty_weight"] < evidence.loc[0, "adaptive_raw_penalty_weight"]


def test_adaptive_solver_uses_positive_soft_compromise_when_fixed_sibling_consumes_flow() -> None:
    # At node 10 the fixed accepted sibling already consumes most of the incoming
    # width. An exact upstream allocation would nearly zero link 2, while its
    # downstream junction favors retaining some width.
    links = pd.DataFrame(
        {
            "id_link": [1, 2, 3, 4],
            "id_us_node": [9, 10, 10, 11],
            "id_ds_node": [10, 11, 12, 12],
            "wid_adj_wet": [100.0, 80.0, 95.0, 80.0],
            "routing_context_width_us": [100.0, 80.0, 95.0, 80.0],
            "routing_context_width_ds": [100.0, 80.0, 95.0, 80.0],
            "width_model_prediction": [0, 1, 0, 0],
            "width_model_probability": [0.0, 0.99, 0.0, 0.0],
            "width_model_threshold": [0.45] * 4,
            "upstream_continuity_abs_deviation": [0.0, 4.0, 0.0, 0.0],
            "opposition_cos_trimmed_mean": [-1.0, 0.5, -1.0, -1.0],
        }
    )

    solved, diagnostics = solve_adaptive_two_ended(links, config=AdaptiveSolverConfig())
    scale = float(solved.loc[solved["id_link"].eq(2), "comparison_scale_factor"].iloc[0])

    assert diagnostics["success"]
    assert np.isfinite(scale)
    assert 0.05 <= scale < 1.0
    assert float(solved.loc[solved["id_link"].eq(3), "comparison_scale_factor"].iloc[0]) == 1.0


def test_evaluation_keeps_equal_node_ids_in_different_examples_independent() -> None:
    # Both networks use node IDs 1 and 2, but their widths differ. Pooling them
    # would incorrectly make both upstream deviations equal.
    frame = pd.DataFrame(
        {
            "example_id": ["01", "01", "02", "02"],
            "id_link": [1, 2, 1, 2],
            "id_us_node": [0, 1, 0, 1],
            "id_ds_node": [1, 2, 1, 2],
            "wid_adj_wet": [100.0, 50.0, 10.0, 20.0],
            "routing_context_width_us": [100.0, 50.0, 10.0, 20.0],
            "routing_context_width_ds": [100.0, 50.0, 10.0, 20.0],
        }
    )
    predicted = frame["id_link"].eq(2)

    rows, _ = _evaluate_method(
        frame,
        method="test",
        scales=pd.Series(1.0, index=frame.index),
        predicted_mask=predicted,
    )

    by_example = rows.set_index("example_id")["upstream_deviation_before"]
    assert np.isclose(by_example.loc["01"], 0.5)
    assert np.isclose(by_example.loc["02"], 1.0)
