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


def test_sparse_solver_matches_dense_legacy_solution() -> None:
    nodes = pd.DataFrame({"id_node": [1, 2, 3, 4]})
    edges = pd.DataFrame(
        {
            "id_link": [301, 302, 303, 304, 305],
            "id_us_node": [1, 1, 2, 2, 3],
            "id_ds_node": [2, 3, 3, 4, 4],
            "K_e": [8.0, 10.0, 6.0, 9.0, 7.0],
        }
    )

    dense = solve_bubble_equivalent_K(
        nodes, edges, upstream_node=1, downstream_node=4,
        solver_backend="dense_legacy",
    )
    sparse = solve_bubble_equivalent_K(
        nodes, edges, upstream_node=1, downstream_node=4,
        solver_backend="sparse",
    )

    assert dense["solver_backend_resolved"] == "dense_legacy"
    assert sparse["solver_backend_resolved"] == "sparse_least_squares"
    assert sparse["solver_method"] == "least_squares_sparse_lsmr"
    assert sparse["solver_success"] is True
    assert sparse["K_eq"] == pytest.approx(dense["K_eq"], rel=1e-9)
    assert sparse["Q_unit"] == pytest.approx(dense["Q_unit"], rel=1e-9)
    assert sparse["residual_norm"] < 1e-7


def test_auto_solver_selects_backend_from_internal_node_threshold() -> None:
    nodes = pd.DataFrame({"id_node": [1, 2, 3]})
    edges = pd.DataFrame(
        {
            "id_link": [101, 102],
            "id_us_node": [1, 2],
            "id_ds_node": [2, 3],
            "K_e": [10.0, 20.0],
        }
    )

    dense = solve_bubble_equivalent_K(
        nodes, edges, upstream_node=1, downstream_node=3,
        solver_backend="auto", sparse_min_internal_nodes=2,
    )
    sparse = solve_bubble_equivalent_K(
        nodes, edges, upstream_node=1, downstream_node=3,
        solver_backend="auto", sparse_min_internal_nodes=1,
    )

    assert dense["solver_backend_requested"] == "auto"
    assert dense["solver_backend_resolved"] == "dense_legacy"
    assert sparse["solver_backend_requested"] == "auto"
    assert sparse["solver_backend_resolved"] == "energy_sparse"


def test_energy_sparse_solver_matches_dense_and_records_both_stages() -> None:
    nodes = pd.DataFrame({"id_node": [1, 2, 3, 4]})
    edges = pd.DataFrame(
        {
            "id_link": [301, 302, 303, 304, 305],
            "id_us_node": [1, 1, 2, 2, 3],
            "id_ds_node": [2, 3, 3, 4, 4],
            "K_e": [8.0, 10.0, 6.0, 9.0, 7.0],
        }
    )

    dense = solve_bubble_equivalent_K(
        nodes, edges, 1, 4, solver_backend="dense_legacy"
    )
    energy = solve_bubble_equivalent_K(
        nodes, edges, 1, 4, solver_backend="energy_sparse"
    )

    assert energy["solver_success"] is True
    assert energy["solver_backend_resolved"] == "energy_sparse"
    assert energy["solver_method"] == "energy_lbfgsb_then_sparse_lsmr"
    assert energy["energy_stage_nfev"] > 0
    assert energy["polish_attempted"] is True
    assert energy["polish_nfev"] > 0
    assert energy["solver_solution_stage"] in {
        "energy_minimization", "sparse_least_squares_polish"
    }
    assert energy["K_eq"] == pytest.approx(dense["K_eq"], rel=1e-8)
    assert energy["residual_relative"] < 1e-6


def test_solver_cache_reuses_only_an_identical_hydraulic_problem(tmp_path) -> None:
    nodes = pd.DataFrame({"id_node": [1, 2, 3]})
    edges = pd.DataFrame(
        {
            "id_link": [101, 102],
            "id_us_node": [1, 2],
            "id_ds_node": [2, 3],
            "K_e": [10.0, 20.0],
        }
    )

    first = solve_bubble_equivalent_K(
        nodes, edges, 1, 3,
        solver_backend="sparse", solver_cache_dir=tmp_path,
    )
    repeated = solve_bubble_equivalent_K(
        nodes, edges, 1, 3,
        solver_backend="sparse", solver_cache_dir=tmp_path,
    )

    assert first["solver_cache_hit"] is False
    assert repeated["solver_cache_hit"] is True
    assert repeated["solver_cache_key"] == first["solver_cache_key"]
    assert repeated["K_eq"] == pytest.approx(first["K_eq"], rel=1e-12)
    assert repeated["edge_flows"].shape == first["edge_flows"].shape
    assert repeated["node_heads"].shape == first["node_heads"].shape

    changed_edges = edges.copy()
    changed_edges.loc[0, "K_e"] = 11.0
    changed = solve_bubble_equivalent_K(
        nodes, changed_edges, 1, 3,
        solver_backend="sparse", solver_cache_dir=tmp_path,
    )
    different_backend = solve_bubble_equivalent_K(
        nodes, edges, 1, 3,
        solver_backend="dense_legacy", solver_cache_dir=tmp_path,
    )

    assert changed["solver_cache_hit"] is False
    assert changed["solver_cache_key"] != first["solver_cache_key"]
    assert different_backend["solver_cache_hit"] is False
    assert different_backend["solver_cache_key"] != first["solver_cache_key"]


def test_solver_cache_remaps_an_unchanged_child_domain_to_child_node_ids(tmp_path) -> None:
    base_nodes = pd.DataFrame({"id_node": [10, 20, 30]})
    base_edges = pd.DataFrame(
        {
            "id_link": [101, 102],
            "id_us_node": [10, 20],
            "id_ds_node": [20, 30],
            "K_e": [10.0, 20.0],
        }
    )
    base = solve_bubble_equivalent_K(
        base_nodes, base_edges, 10, 30,
        solver_backend="sparse", solver_cache_dir=tmp_path,
    )

    child_nodes = pd.DataFrame({"id_node": [1, 2, 3]})
    child_edges = pd.DataFrame(
        {
            "id_link": [201, 202],
            "id_us_node": [1, 2],
            "id_ds_node": [2, 3],
            "K_e": [10.0, 20.0],
            "solver_cache_edge_alias": [101, 102],
            "solver_cache_u_alias": [10, 20],
            "solver_cache_v_alias": [20, 30],
        }
    )
    child = solve_bubble_equivalent_K(
        child_nodes, child_edges, 1, 3,
        solver_backend="sparse", solver_cache_dir=tmp_path,
    )

    assert base["solver_cache_hit"] is False
    assert child["solver_cache_hit"] is True
    assert child["solver_cache_key"] == base["solver_cache_key"]
    assert child["upstream_node"] == 1
    assert child["downstream_node"] == 3
    assert set(child["node_heads"]["node_id"].astype(int)) == {1, 2, 3}
    assert set(child["edge_flows"]["edge_id"].astype(int)) == {201, 202}
    assert child["K_eq"] == pytest.approx(base["K_eq"], rel=1e-12)


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
