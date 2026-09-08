from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

import hierarchy_level_definition.metrics.bubble_equivalent_metrics as bubble_metrics_module
from hierarchy_level_definition.metrics import (
    BubbleEquivalentResistanceConfig,
    LENGTH_FIELD_CANDIDATES,
    compute_bubble_equivalent_metrics_from_units,
    compute_K_manning,
    resolve_length_field_candidates,
)
from hierarchy_level_definition.run_unit_workflow import UnitWorkflowOutputs, write_unit_workflow_outputs
from hierarchy_level_definition.unit_detection.bifurcation_confluence_units import StructuralUnit, UnitPath


def _parallel_unit(*, unit_id: int, edge_ids: list[int], upstream: int, downstream: int) -> StructuralUnit:
    paths = [
        UnitPath(
            path_id=index,
            node_path=[upstream, downstream],
            edge_path=[(upstream, downstream, edge_id)],
            id_links=[edge_id],
            total_length=100.0 + 20.0 * (index - 1),
        )
        for index, edge_id in enumerate(edge_ids, start=1)
    ]
    return StructuralUnit(
        unit_id=unit_id,
        bifurcation=upstream,
        confluence=downstream,
        n_paths=len(paths),
        unit_class="simple_bifurcation_confluence_pair",
        min_path_length=min(path.total_length for path in paths),
        max_path_length=max(path.total_length for path in paths),
        paths=paths,
        node_set={upstream, downstream},
        edge_set={(upstream, downstream, edge_id) for edge_id in edge_ids},
        children=[],
        parents=[],
    )


def test_compound_bubble_domain_includes_nonunit_connectors() -> None:
    first = StructuralUnit(
        unit_id=1,
        bifurcation=0,
        confluence=2,
        n_paths=2,
        unit_class="compound",
        min_path_length=1.0,
        max_path_length=2.0,
        paths=[],
        node_set={0, 1, 2},
        edge_set={(0, 1, 1), (1, 2, 2)},
        children=[],
        parents=[],
    )
    second = StructuralUnit(
        unit_id=2,
        bifurcation=0,
        confluence=4,
        n_paths=2,
        unit_class="compound",
        min_path_length=1.0,
        max_path_length=3.0,
        paths=[],
        node_set={0, 1, 3, 4},
        edge_set={(0, 1, 1), (1, 3, 3), (3, 4, 4)},
        children=[],
        parents=[],
    )
    prepared = pd.DataFrame(
        [
            {"edge_id": 1, "u": 0, "v": 1},
            {"edge_id": 2, "u": 1, "v": 2},
            {"edge_id": 3, "u": 1, "v": 3},
            {"edge_id": 4, "u": 3, "v": 4},
            # Ordinary single-channel continuation: not owned by either unit.
            {"edge_id": 5, "u": 2, "v": 4},
        ]
    )

    keys, connector_ids = bubble_metrics_module._complete_compound_bubble_edge_keys(
        prepared,
        [first, second],
    )

    assert (2, 4, 5) in keys
    assert connector_ids == [5]


def _fake_solve_result(*, success: bool, residual: float, k_eq: float) -> dict:
    return {
        "n_nodes": 3,
        "n_edges": 2,
        "solver_success": success,
        "solver_method": "least_squares",
        "solver_message": "ok" if success else "maximum evaluations reached",
        "solver_nfev": 20 if success else 4000,
        "K_eq": k_eq,
        "G_eq": 1.0 / k_eq,
        "Q_unit": 1.0 / k_eq**0.5,
        "residual_norm": residual,
        "residual_relative": residual,
        "edge_flows": pd.DataFrame(
            {"edge_id": [1, 2], "q_unit": [1.0, 1.0], "dh": [0.5, 0.5]}
        ),
        "node_heads": pd.DataFrame(
            {
                "node_id": [1, 2, 3],
                "head_norm": [1.0, 0.5, 0.0],
                "is_boundary": [True, False, True],
                "is_upstream_boundary": [True, False, False],
                "is_downstream_boundary": [False, False, True],
                "head_source": ["boundary", "solved", "boundary"],
                "continuity_residual": [pd.NA, residual, pd.NA],
            }
        ),
    }


def test_hierarchy_bubble_solver_retries_with_normalized_neutral_heads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    guesses = []
    results = [
        _fake_solve_result(success=False, residual=0.3, k_eq=99.0),
        _fake_solve_result(success=True, residual=1e-12, k_eq=4.0),
    ]

    def fake_solver(*args, optional_H_guess=None, **kwargs):
        guesses.append(optional_H_guess)
        return results.pop(0)

    monkeypatch.setattr(bubble_metrics_module, "solve_bubble_equivalent_K", fake_solver)
    nodes = pd.DataFrame(
        {"node_id": [1, 2, 3], "sword_wse": [10.0, 9.8, 9.0]}
    )
    edges = pd.DataFrame(
        {"edge_id": [1, 2], "u": [1, 2], "v": [2, 3], "K_e": [2.0, 2.0]}
    )

    solved = bubble_metrics_module._solve_with_neutral_head_fallback(
        bubble_nodes=nodes,
        bubble_edges=edges,
        upstream_node=1,
        downstream_node=3,
        config=BubbleEquivalentResistanceConfig(),
    )

    assert guesses[0] is nodes
    assert guesses[1] == {1: 1.0, 2: 0.5, 3: 0.0}
    assert solved["solver_success"] is True
    assert solved["K_eq"] == pytest.approx(4.0)
    assert solved["solver_initialization"] == "neutral_internal_heads_fallback"
    assert solved["primary_solver_success"] is False
    assert solved["primary_residual_relative"] == pytest.approx(0.3)
    assert solved["neutral_fallback_attempted"] is True


def test_hierarchy_bubble_solver_masks_provisional_values_when_both_attempts_fail(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    results = [
        _fake_solve_result(success=False, residual=0.3, k_eq=99.0),
        _fake_solve_result(success=False, residual=0.2, k_eq=88.0),
    ]
    monkeypatch.setattr(
        bubble_metrics_module,
        "solve_bubble_equivalent_K",
        lambda *args, **kwargs: results.pop(0),
    )

    solved = bubble_metrics_module._solve_with_neutral_head_fallback(
        bubble_nodes=pd.DataFrame({"node_id": [1, 2, 3]}),
        bubble_edges=pd.DataFrame(
            {"edge_id": [1, 2], "u": [1, 2], "v": [2, 3], "K_e": [2.0, 2.0]}
        ),
        upstream_node=1,
        downstream_node=3,
        config=BubbleEquivalentResistanceConfig(),
    )

    assert solved["solver_success"] is False
    assert pd.isna(solved["K_eq"])
    assert pd.isna(solved["G_eq"])
    assert pd.isna(solved["Q_unit"])
    assert solved["edge_flows"][["q_unit", "dh"]].isna().all().all()


def test_hierarchy_bubble_solver_conditionally_accepts_stable_near_convergence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    results = [
        _fake_solve_result(success=False, residual=0.008, k_eq=100.0),
        _fake_solve_result(success=False, residual=0.006, k_eq=101.0),
    ]
    monkeypatch.setattr(
        bubble_metrics_module,
        "solve_bubble_equivalent_K",
        lambda *args, **kwargs: results.pop(0),
    )

    solved = bubble_metrics_module._solve_with_neutral_head_fallback(
        bubble_nodes=pd.DataFrame({"node_id": [1, 2, 3]}),
        bubble_edges=pd.DataFrame(
            {"edge_id": [1, 2], "u": [1, 2], "v": [2, 3], "K_e": [2.0, 2.0]}
        ),
        upstream_node=1,
        downstream_node=3,
        config=BubbleEquivalentResistanceConfig(),
    )

    assert solved["solver_success"] is True
    assert solved["solver_strict_success"] is False
    assert solved["solver_conditional_acceptance"] is True
    assert solved["solver_acceptance_tier"] == "conditional_stable"
    assert solved["K_eq"] == pytest.approx(101.0)
    assert solved["candidate_K_relative_difference"] == pytest.approx(1.0 / 101.0)


def test_hierarchy_bubble_solver_rejects_near_convergence_when_k_is_unstable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    results = [
        _fake_solve_result(success=False, residual=0.008, k_eq=100.0),
        _fake_solve_result(success=False, residual=0.006, k_eq=110.0),
    ]
    monkeypatch.setattr(
        bubble_metrics_module,
        "solve_bubble_equivalent_K",
        lambda *args, **kwargs: results.pop(0),
    )

    solved = bubble_metrics_module._solve_with_neutral_head_fallback(
        bubble_nodes=pd.DataFrame({"node_id": [1, 2, 3]}),
        bubble_edges=pd.DataFrame(
            {"edge_id": [1, 2], "u": [1, 2], "v": [2, 3], "K_e": [2.0, 2.0]}
        ),
        upstream_node=1,
        downstream_node=3,
        config=BubbleEquivalentResistanceConfig(),
    )

    assert solved["solver_success"] is False
    assert solved["solver_acceptance_tier"] == "rejected"
    assert pd.isna(solved["K_eq"])


def test_hierarchy_bubble_solver_keeps_primary_when_neutral_attempt_is_worse(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    results = [
        _fake_solve_result(success=False, residual=0.006, k_eq=100.0),
        _fake_solve_result(success=False, residual=0.008, k_eq=101.0),
    ]
    monkeypatch.setattr(
        bubble_metrics_module,
        "solve_bubble_equivalent_K",
        lambda *args, **kwargs: results.pop(0),
    )

    solved = bubble_metrics_module._solve_with_neutral_head_fallback(
        bubble_nodes=pd.DataFrame({"node_id": [1, 2, 3]}),
        bubble_edges=pd.DataFrame(
            {"edge_id": [1, 2], "u": [1, 2], "v": [2, 3], "K_e": [2.0, 2.0]}
        ),
        upstream_node=1,
        downstream_node=3,
        config=BubbleEquivalentResistanceConfig(),
    )

    assert solved["solver_success"] is True
    assert solved["solver_initialization"] == "available_node_heads"
    assert solved["K_eq"] == pytest.approx(100.0)
    assert solved["residual_relative"] == pytest.approx(0.006)


def test_compute_bubble_equivalent_metrics_from_units_reduces_standalone_parallel_unit() -> None:
    links = pd.DataFrame(
        [
            {"id_link": 1, "id_us_node": 1, "id_ds_node": 2, "len_adj": 100.0, "wid_adj": 50.0},
            {"id_link": 2, "id_us_node": 1, "id_ds_node": 2, "len_adj": 100.0, "wid_adj": 50.0},
        ]
    )
    nodes = pd.DataFrame(
        [
            {"id_node": 1, "sword_wse": 10.0},
            {"id_node": 2, "sword_wse": 9.0},
        ]
    )
    unit = _parallel_unit(unit_id=10, edge_ids=[1, 2], upstream=1, downstream=2)
    config = BubbleEquivalentResistanceConfig(kb_value=20.0, default_manning_n=0.035)

    bubble_summary, bubble_edges, bubble_nodes = compute_bubble_equivalent_metrics_from_units(
        links,
        nodes,
        [unit],
        config=config,
    )

    branch_k = compute_K_manning(100.0, 50.0, 0.035, 20.0)
    assert len(bubble_summary) == 1
    assert bool(bubble_summary.loc[0, "solver_success"]) is True
    assert bubble_summary.loc[0, "K_eq"] == pytest.approx(branch_k / 4.0, rel=1e-7)
    assert bubble_summary.loc[0, "G_eq"] == pytest.approx(4.0 / branch_k, rel=1e-7)
    assert len(bubble_edges) == 2
    assert set(bubble_edges["edge_id"].tolist()) == {1, 2}
    assert len(bubble_nodes) == 2
    assert set(bubble_nodes["node_id"].tolist()) == {1, 2}


def test_bubble_resistance_prefers_effective_routing_width() -> None:
    links = pd.DataFrame(
        [
            {
                "id_link": 1,
                "id_us_node": 1,
                "id_ds_node": 2,
                "len_adj": 100.0,
                "wid_adj_routing": 25.0,
                "wid_adj_wet": 50.0,
                "wid_adj": 80.0,
            },
            {
                "id_link": 2,
                "id_us_node": 1,
                "id_ds_node": 2,
                "len_adj": 100.0,
                "wid_adj_routing": 25.0,
                "wid_adj_wet": 50.0,
                "wid_adj": 80.0,
            },
        ]
    )
    nodes = pd.DataFrame(
        [{"id_node": 1, "sword_wse": 10.0}, {"id_node": 2, "sword_wse": 9.0}]
    )
    unit = _parallel_unit(unit_id=10, edge_ids=[1, 2], upstream=1, downstream=2)

    _, bubble_edges, _ = compute_bubble_equivalent_metrics_from_units(
        links,
        nodes,
        [unit],
        config=BubbleEquivalentResistanceConfig(kb_value=20.0, default_manning_n=0.035),
    )

    assert set(bubble_edges["width_source"]) == {"wid_adj_routing"}
    assert set(bubble_edges["w"]) == {25.0}


def test_compute_bubble_equivalent_metrics_from_units_groups_compound_bubbles() -> None:
    links = pd.DataFrame(
        [
            {"id_link": 1, "id_us_node": 1, "id_ds_node": 4, "len_adj": 100.0, "wid_adj": 60.0},
            {"id_link": 2, "id_us_node": 1, "id_ds_node": 2, "len_adj": 120.0, "wid_adj": 40.0},
            {"id_link": 3, "id_us_node": 2, "id_ds_node": 4, "len_adj": 70.0, "wid_adj": 30.0},
            {"id_link": 4, "id_us_node": 2, "id_ds_node": 3, "len_adj": 80.0, "wid_adj": 25.0},
            {"id_link": 5, "id_us_node": 2, "id_ds_node": 5, "len_adj": 90.0, "wid_adj": 50.0},
            {"id_link": 6, "id_us_node": 5, "id_ds_node": 3, "len_adj": 110.0, "wid_adj": 45.0},
            {"id_link": 7, "id_us_node": 9, "id_ds_node": 10, "len_adj": 100.0, "wid_adj": 55.0},
            {"id_link": 8, "id_us_node": 9, "id_ds_node": 10, "len_adj": 120.0, "wid_adj": 35.0},
            {"id_link": 9, "id_us_node": 3, "id_ds_node": 4, "len_adj": 60.0, "wid_adj": 30.0},
        ]
    )
    nodes = pd.DataFrame(
        [{"id_node": node_id, "sword_wse": 100.0 - node_id} for node_id in [1, 2, 3, 4, 5, 9, 10]]
    )

    root_unit = StructuralUnit(
        unit_id=10,
        bifurcation=1,
        confluence=4,
        n_paths=2,
        unit_class="compound_or_nested_complex",
        min_path_length=100.0,
        max_path_length=190.0,
        paths=[
            UnitPath(1, [1, 4], [(1, 4, 1)], [1], 100.0),
            UnitPath(2, [1, 2, 4], [(1, 2, 2), (2, 4, 3)], [2, 3], 190.0),
        ],
        node_set={1, 2, 3, 4, 5},
        edge_set={(1, 4, 1), (1, 2, 2), (2, 4, 3), (2, 3, 4), (2, 5, 5), (3, 4, 9)},
        internal_bifurcations=[2],
        internal_confluences=[],
        children=[20],
        parents=[],
    )
    child_unit = StructuralUnit(
        unit_id=20,
        bifurcation=2,
        confluence=3,
        n_paths=2,
        unit_class="simple_bifurcation_confluence_pair",
        min_path_length=80.0,
        max_path_length=200.0,
        paths=[
            UnitPath(1, [2, 3], [(2, 3, 4)], [4], 80.0),
            UnitPath(2, [2, 5, 3], [(2, 5, 5), (5, 3, 6)], [5, 6], 200.0),
        ],
        node_set={2, 3, 5},
        edge_set={(2, 3, 4), (2, 5, 5), (5, 3, 6)},
        internal_bifurcations=[],
        internal_confluences=[],
        children=[],
        parents=[10],
    )
    standalone_unit = _parallel_unit(unit_id=30, edge_ids=[7, 8], upstream=9, downstream=10)

    bubble_summary, bubble_edges, bubble_nodes = compute_bubble_equivalent_metrics_from_units(
        links,
        nodes,
        [root_unit, child_unit, standalone_unit],
        config=BubbleEquivalentResistanceConfig(kb_value=20.0),
    )

    assert len(bubble_summary) == 2
    compound_row = bubble_summary.set_index("bubble_root_unit_id").loc[10]
    assert compound_row["n_member_units"] == 2
    assert compound_row["upstream_boundary_node"] == 1
    assert compound_row["downstream_boundary_node"] == 4
    assert compound_row["n_edges"] == 7
    assert bool(compound_row["solver_success"]) is True
    assert compound_row["K_eq"] > 0.0
    assert compound_row["residual_relative"] < 1e-3

    standalone_row = bubble_summary.set_index("bubble_root_unit_id").loc[30]
    assert standalone_row["n_member_units"] == 1
    assert bool(standalone_row["solver_success"]) is True

    compound_bubble_id = int(compound_row["compound_bubble_id"])
    assert len(bubble_edges.loc[bubble_edges["compound_bubble_id"] == compound_bubble_id]) == 7
    assert set(bubble_nodes.loc[bubble_nodes["compound_bubble_id"] == compound_bubble_id, "node_id"]) >= {1, 2, 3, 4, 5}


def test_write_unit_workflow_outputs_writes_equivalent_resistance_files(tmp_path: Path) -> None:
    results = UnitWorkflowOutputs(
        unit_summary=pd.DataFrame({"unit_id": [1]}),
        unit_metrics=pd.DataFrame({"unit_id": [1]}),
        path_metrics=pd.DataFrame({"unit_id": [1], "path_id": [1]}),
        hierarchy_level_metrics=pd.DataFrame({"collapse_level": [0]}),
        collapse_ranking=pd.DataFrame({"unit_id": [1]}),
        merge_tree=pd.DataFrame({"n_groups": [1], "group_index": [1]}),
        bubble_summary=pd.DataFrame({"compound_bubble_id": [1]}),
        bubble_equivalent_metrics=pd.DataFrame({"compound_bubble_id": [1], "K_eq": [2.5]}),
        bubble_edge_resistance=pd.DataFrame({"compound_bubble_id": [1], "edge_id": [1]}),
        bubble_node_heads=pd.DataFrame({"compound_bubble_id": [1], "node_id": [1]}),
        group_count_summary=pd.DataFrame({"n_groups": [1], "is_optimal_n_groups": [True]}),
        selected_groups=pd.DataFrame({"group_index": [1]}),
    )
    results.bubble_equivalent_metrics.attrs["equivalent_resistance_config"] = {"kb_value": 20.0}
    results.bubble_equivalent_metrics.attrs["resolved_kb_context"] = {"kb_value": 20.0, "kb_mode": "fixed"}

    write_unit_workflow_outputs(tmp_path, results)

    assert (tmp_path / "bubble_equivalent_metrics.csv").exists()
    assert (tmp_path / "bubble_edge_resistance.csv").exists()
    assert (tmp_path / "bubble_node_heads.csv").exists()

    manifest = json.loads((tmp_path / "unit_workflow_manifest.json").read_text())
    assert "bubble_equivalent_metrics.csv" in manifest["files"]
    assert manifest["n_equivalent_bubbles"] == 1
    assert manifest["resolved_kb_context"]["kb_value"] == 20.0


def test_resolve_length_field_candidates_prioritizes_requested_field() -> None:
    assert LENGTH_FIELD_CANDIDATES[0] == "len"
    assert resolve_length_field_candidates(None) == LENGTH_FIELD_CANDIDATES
    assert resolve_length_field_candidates("len") == ("len", "len_adj", "link_length_m")
    with pytest.raises(ValueError, match="Unsupported equivalent-resistance length field"):
        resolve_length_field_candidates("not_a_length_field")
