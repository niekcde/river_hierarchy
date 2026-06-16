from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

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
        edge_set={(1, 4, 1), (1, 2, 2), (2, 4, 3), (2, 3, 4), (2, 5, 5)},
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
    assert compound_row["n_edges"] == 6
    assert bool(compound_row["solver_success"]) is True
    assert compound_row["K_eq"] > 0.0
    assert compound_row["residual_relative"] < 1e-3

    standalone_row = bubble_summary.set_index("bubble_root_unit_id").loc[30]
    assert standalone_row["n_member_units"] == 1
    assert bool(standalone_row["solver_success"]) is True

    compound_bubble_id = int(compound_row["compound_bubble_id"])
    assert len(bubble_edges.loc[bubble_edges["compound_bubble_id"] == compound_bubble_id]) == 6
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
    assert resolve_length_field_candidates(None) == LENGTH_FIELD_CANDIDATES
    assert resolve_length_field_candidates("len") == ("len", "len_adj", "link_length_m")
    with pytest.raises(ValueError, match="Unsupported equivalent-resistance length field"):
        resolve_length_field_candidates("not_a_length_field")
