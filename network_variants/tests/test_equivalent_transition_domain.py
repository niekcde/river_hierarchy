from __future__ import annotations

import pandas as pd
import pytest

from hierarchy_level_definition.metrics import BubbleEquivalentResistanceConfig
from network_variants import equivalent_transition_domain as domains


def _links(records: list[tuple[int, int, int]]) -> pd.DataFrame:
    return pd.DataFrame(records, columns=["id_link", "id_us_node", "id_ds_node"])


def test_closed_unit_is_selected_without_compound_expansion() -> None:
    links = _links([(1, 1, 2), (2, 2, 4), (3, 1, 3), (4, 3, 4), (5, 0, 1), (6, 4, 5)])
    unit_summary = pd.DataFrame(
        [{"unit_id": 7, "bifurcation": 1, "confluence": 4, "compound_bubble_id": pd.NA}]
    )
    membership = pd.DataFrame({"unit_id": [7, 7, 7, 7], "id_link": [1, 2, 3, 4]})

    domain = domains.select_smallest_closed_parent_domain(
        full_links=links,
        unit_summary=unit_summary,
        unit_link_membership=membership,
        bubble_edge_resistance=pd.DataFrame(),
        selected_unit_ids=[7],
    )

    assert domain.domain_type == "closed_unit"
    assert domain.source_node == 1
    assert domain.sink_node == 4
    assert set(domain.links["id_link"]) == {1, 2, 3, 4}
    assert domain.selected_unit_is_closed is True


def test_open_member_unit_expands_to_complete_containing_compound() -> None:
    links = _links(
        [
            (1, 1, 2),
            (2, 2, 4),
            (3, 2, 3),
            (4, 3, 4),
            (5, 0, 1),
            (6, 4, 5),
        ]
    )
    unit_summary = pd.DataFrame(
        [{"unit_id": 7, "bifurcation": 1, "confluence": 4, "compound_bubble_id": 9}]
    )
    membership = pd.DataFrame({"unit_id": [7, 7], "id_link": [1, 2]})
    bubble_edges = pd.DataFrame(
        {
            "compound_bubble_id": [9, 9, 9, 9],
            "u": [1, 2, 2, 3],
            "v": [2, 4, 3, 4],
        }
    )

    domain = domains.select_smallest_closed_parent_domain(
        full_links=links,
        unit_summary=unit_summary,
        unit_link_membership=membership,
        bubble_edge_resistance=bubble_edges,
        selected_unit_ids=[7],
    )

    assert domain.domain_type == "containing_compound_context"
    assert domain.compound_bubble_id == 9
    assert set(domain.links["id_link"]) == {1, 2, 3, 4}
    assert domain.selected_unit_is_closed is False
    assert domain.selected_unit_violating_link_ids == (3, 4)


def test_child_domain_uses_matched_parent_anchors_and_all_directed_paths() -> None:
    parent = domains.HydraulicDomain(
        domain_type="containing_compound_context",
        source_node=1,
        sink_node=4,
        links=_links([(1, 1, 2), (2, 2, 4)]),
        selected_unit_ids=(7,),
        compound_bubble_id=9,
        selected_unit_is_closed=False,
        selected_unit_violating_link_ids=(3,),
    )
    child_links = _links(
        [(11, 10, 20), (12, 20, 40), (13, 10, 30), (14, 30, 40), (15, 0, 10), (16, 40, 50)]
    )

    child = domains.build_matched_child_domain(
        parent_domain=parent,
        child_links=child_links,
        parent_to_child_node={1: 10, 4: 40},
    )

    assert child.source_node == 10
    assert child.sink_node == 40
    assert set(child.links["id_link"]) == {11, 12, 13, 14}


def test_anchor_mapping_falls_back_to_directed_link_lineage(
    tmp_path,
) -> None:
    node_match_path = tmp_path / "node_match.csv"
    lineage_path = tmp_path / "link_lineage.csv"
    pd.DataFrame(
        [
            {
                "child_id_node": 7,
                "matched_parent_node_id": 8,
                "match_distance": 0.0,
                "match_within_tolerance": True,
            }
        ]
    ).to_csv(node_match_path, index=False)
    pd.DataFrame(
        [
            {
                "id_link": 10,
                "matched_parent_us_node": 31,
                "matched_parent_ds_node": 5,
            }
        ]
    ).to_csv(lineage_path, index=False)
    child_links = _links([(10, 100, 101)])

    resolved, methods = domains.load_parent_to_child_anchor_map(
        node_match_path=node_match_path,
        child_links=child_links,
        link_lineage_path=lineage_path,
    )

    assert resolved[8] == 7
    assert methods[8] == "stable_node_match"
    assert resolved[31] == 100
    assert resolved[5] == 101
    assert methods[31] == "directed_link_lineage_endpoint"
    assert methods[5] == "directed_link_lineage_endpoint"


def test_solver_retries_with_neutral_internal_heads(monkeypatch: pytest.MonkeyPatch) -> None:
    domain = domains.HydraulicDomain(
        domain_type="closed_unit",
        source_node=1,
        sink_node=3,
        links=_links([(1, 1, 2), (2, 2, 3)]),
        selected_unit_ids=(7,),
        compound_bubble_id=None,
        selected_unit_is_closed=True,
        selected_unit_violating_link_ids=(),
    )
    nodes = pd.DataFrame({"id_node": [1, 2, 3]})

    monkeypatch.setattr(
        domains,
        "_compute_equivalent_resistance",
        lambda *args, **kwargs: (
            {
                "solver_success": False,
                "residual_relative": 0.25,
                "K_eq": 99.0,
            },
            None,
        ),
    )
    monkeypatch.setattr(
        domains,
        "prepare_bubble_resistance_edges",
        lambda *args, **kwargs: (
            pd.DataFrame({"u": [1, 2], "v": [2, 3], "K": [1.0, 1.0]}),
            pd.Series({"k_b": 1.0}),
        ),
    )

    result, _ = domains.solve_hydraulic_domain(
        domain=domain,
        nodes=nodes,
        config=BubbleEquivalentResistanceConfig(),
    )

    assert result["solver_success"] is True
    assert result["initialization_used"] == "neutral_internal_heads_fallback"
    assert result["primary_solver_success"] is False
    assert result["primary_residual_relative"] == pytest.approx(0.25)
