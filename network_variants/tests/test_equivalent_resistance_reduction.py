from __future__ import annotations

import pandas as pd
import pytest

from network_variants.equivalent_resistance_reduction import (
    reduce_series_parallel_resistance_graph,
)
from hierarchy_level_definition.metrics.equivalent_resistance import (
    solve_bubble_equivalent_K_continuation,
)


def test_exact_series_parallel_reduction_preserves_known_equivalent_resistance() -> None:
    # Two serial paths in parallel: (1 + 3) || (2 + 2) = 4 || 4 = 1.
    edges = pd.DataFrame(
        {
            "edge_id": [10, 11, 12, 13],
            "u": [1, 2, 1, 3],
            "v": [2, 4, 3, 4],
            "K_e": [1.0, 3.0, 2.0, 2.0],
        }
    )

    result = reduce_series_parallel_resistance_graph(
        edges, source_node=1, sink_node=4
    )

    assert result.summary["n_series_contractions"] == 2
    assert result.summary["n_parallel_contractions"] == 1
    assert result.summary["n_reduced_nodes"] == 2
    assert result.summary["n_reduced_edges"] == 1
    assert result.reduced_edges.iloc[0]["K_e"] == 1.0


def test_reduction_retains_boundary_nodes_and_stops_at_nonreducible_braid() -> None:
    # A four-node diamond has no internal degree-two node once its terminals
    # are excluded, so this is intentionally retained as a true braided core.
    edges = pd.DataFrame(
        {
            "edge_id": [10, 11, 12, 13, 14],
            "u": [1, 1, 2, 3, 2],
            "v": [2, 3, 4, 4, 3],
            "K_e": [1.0, 1.0, 1.0, 1.0, 1.0],
        }
    )

    result = reduce_series_parallel_resistance_graph(
        edges, source_node=1, sink_node=4
    )

    assert result.summary["n_series_contractions"] == 0
    assert result.summary["n_parallel_contractions"] == 0
    assert result.summary["n_reduced_nodes"] == 4
    assert result.summary["n_reduced_edges"] == 5


def test_continuation_retains_target_stage_and_exact_simple_solution() -> None:
    edges = pd.DataFrame(
        {
            "edge_id": [1, 2],
            "u": [1, 2],
            "v": [2, 3],
            "K_e": [1.0, 3.0],
        }
    )
    result = solve_bubble_equivalent_K_continuation(
        pd.DataFrame({"node_id": [1, 2, 3]}),
        edges,
        upstream_node=1,
        downstream_node=3,
        eps_schedule=(1e-4, 1e-8),
        solver_backend="energy_sparse",
    )

    assert result["solver_success"]
    assert result["continuation_used"]
    assert result["continuation_eps_schedule"] == [1e-4, 1e-8]
    assert len(result["continuation_stages"]) == 2
    assert result["K_eq"] == pytest.approx(4.0)
