from __future__ import annotations

import math

import networkx as nx
import pandas as pd
import pytest

from hierarchy_level_definition.metrics.equivalent_resistance import (
    compute_K_chezy,
    compute_K_manning,
    reduce_simple_parallel,
    reduce_simple_series,
    solve_bubble_equivalent_K,
)


def test_compute_K_manning_matches_expected_formula() -> None:
    length = 1000.0
    width = 50.0
    n_value = 0.035
    kb = 20.0
    depth = width / kb
    area = width * depth
    hydraulic_radius = depth
    expected = (n_value**2 * length) / (area**2 * hydraulic_radius ** (4.0 / 3.0))

    result = compute_K_manning(length, width, n_value, kb, geometry_method="wide")

    assert result == pytest.approx(expected)


def test_compute_K_chezy_matches_expected_formula() -> None:
    length = 1000.0
    width = 50.0
    cz_value = 22.0
    kb = 20.0
    gravity = 9.81
    depth = width / kb
    area = width * depth
    hydraulic_radius = depth
    expected = length / (area**2 * cz_value**2 * gravity * hydraulic_radius)

    result = compute_K_chezy(length, width, cz_value, kb, geometry_method="wide", g=gravity)

    assert result == pytest.approx(expected)


def test_reduce_simple_series_sums_resistances() -> None:
    assert reduce_simple_series([10.0, 20.0]) == pytest.approx(30.0)


def test_reduce_simple_parallel_matches_quadratic_parallel_formula() -> None:
    assert reduce_simple_parallel([10.0, 10.0]) == pytest.approx(2.5)


def test_general_solver_reproduces_series_solution() -> None:
    nodes = pd.DataFrame({"id_node": [1, 2, 3]})
    edges = pd.DataFrame(
        {
            "id_link": [101, 102],
            "id_us_node": [1, 2],
            "id_ds_node": [2, 3],
            "K_e": [10.0, 20.0],
        }
    )

    result = solve_bubble_equivalent_K(nodes, edges, upstream_node=1, downstream_node=3)

    assert result["solver_success"] is True
    assert result["K_eq"] == pytest.approx(30.0, rel=1e-7)
    assert result["G_eq"] == pytest.approx(1.0 / 30.0, rel=1e-7)
    assert result["Q_unit"] == pytest.approx(1.0 / math.sqrt(30.0), rel=1e-7)
    assert result["internal_heads"][2] == pytest.approx(2.0 / 3.0, rel=1e-7)


def test_general_solver_reproduces_parallel_solution() -> None:
    nodes = pd.DataFrame({"id_node": [1, 2]})
    edges = pd.DataFrame(
        {
            "id_link": [201, 202],
            "id_us_node": [1, 1],
            "id_ds_node": [2, 2],
            "K_e": [10.0, 10.0],
        }
    )

    result = solve_bubble_equivalent_K(nodes, edges, upstream_node=1, downstream_node=2)

    assert result["solver_success"] is True
    assert result["K_eq"] == pytest.approx(2.5, rel=1e-7)
    assert result["Q_unit"] == pytest.approx(2.0 / math.sqrt(10.0), rel=1e-7)


def test_general_solver_handles_compound_four_node_bubble() -> None:
    nodes = pd.DataFrame({"id_node": [1, 2, 3, 4]})
    edges = pd.DataFrame(
        {
            "id_link": [301, 302, 303, 304, 305],
            "id_us_node": [1, 1, 2, 2, 3],
            "id_ds_node": [2, 3, 3, 4, 4],
            "K_e": [8.0, 10.0, 6.0, 9.0, 7.0],
        }
    )

    result = solve_bubble_equivalent_K(nodes, edges, upstream_node=1, downstream_node=4)

    assert result["solver_success"] is True
    assert math.isfinite(result["K_eq"])
    assert result["K_eq"] > 0.0
    assert result["Q_unit"] > 0.0
    assert result["residual_norm"] < 1e-7
    node_heads = result["node_heads"].set_index("node_id")
    assert 0.0 < node_heads.loc[2, "head_norm"] < 1.0
    assert 0.0 < node_heads.loc[3, "head_norm"] < 1.0


def test_solver_does_not_require_model_q_or_hydrograph_outputs() -> None:
    graph = nx.MultiGraph()
    graph.add_node(1)
    graph.add_node(2)
    graph.add_edge(1, 2, key="edge-1", K_e=4.0)

    result = solve_bubble_equivalent_K(None, graph, upstream_node=1, downstream_node=2)

    assert result["solver_success"] is True
    assert result["K_eq"] == pytest.approx(4.0, rel=1e-7)
    assert result["Q_unit"] == pytest.approx(0.5, rel=1e-7)
    assert "Q_model" not in result["edge_flows"].columns
