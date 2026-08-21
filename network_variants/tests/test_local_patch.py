from __future__ import annotations

import json

import pandas as pd
import pytest

from network_variants.local_patch import (
    LOCAL_PATCH_CONTRACT_VERSION,
    LocalPatchCollar,
    LocalPatchContract,
    LocalPatchMergePolicy,
    plan_local_patch,
    write_local_patch_contract,
)


def _example_56_unit_6_contract() -> LocalPatchContract:
    return LocalPatchContract(
        example_id="56",
        selection_type="unit",
        selected_unit_ids=(6,),
        domain_type="closed_unit",
        core_link_ids=(16, 17),
        upstream_collar=LocalPatchCollar(
            role="upstream",
            base_link_id=18,
            base_upstream_node_id=13,
            base_downstream_node_id=12,
            domain_boundary_node_id=12,
            outer_splice_anchor_node_id=13,
        ),
        downstream_collar=LocalPatchCollar(
            role="downstream",
            base_link_id=14,
            base_upstream_node_id=11,
            base_downstream_node_id=9,
            domain_boundary_node_id=11,
            outer_splice_anchor_node_id=9,
        ),
    )


def test_closed_unit_contract_protects_outer_nodes_and_keeps_collars_mutable() -> None:
    contract = _example_56_unit_6_contract()

    assert contract.source_splice_anchor_node_id == 13
    assert contract.sink_splice_anchor_node_id == 9
    assert contract.collar_link_ids == (18, 14)
    assert contract.local_context_link_ids == (16, 17, 18, 14)
    assert contract.merge_policy.exact_match_action == "retain_base"
    assert contract.merge_policy.collar_match_required is False
    assert (
        contract.upstream_collar.change_policy
        == "allow_retain_replace_split_or_merge"
    )


def test_compound_member_contract_uses_complete_compound_and_external_collars() -> None:
    contract = LocalPatchContract(
        example_id="15",
        selection_type="unit",
        selected_unit_ids=(131,),
        domain_type="containing_compound",
        core_link_ids=tuple(range(49, 421)),
        upstream_collar=LocalPatchCollar(
            role="upstream",
            base_link_id=421,
            base_upstream_node_id=280,
            base_downstream_node_id=279,
            domain_boundary_node_id=279,
            outer_splice_anchor_node_id=280,
        ),
        downstream_collar=LocalPatchCollar(
            role="downstream",
            base_link_id=48,
            base_upstream_node_id=33,
            base_downstream_node_id=32,
            domain_boundary_node_id=33,
            outer_splice_anchor_node_id=32,
        ),
        selected_compound_bubble_id=4,
    )

    assert contract.domain_type == "containing_compound"
    assert contract.selected_compound_bubble_id == 4
    assert contract.source_splice_anchor_node_id == 280
    assert contract.sink_splice_anchor_node_id == 32
    assert contract.raster_policy.expected_unauthorized_boundary_sections == 0
    assert contract.raster_policy.cut_geometry == "nearest_main_banks_via_reference_centerline_context"
    assert contract.raster_policy.minimum_collar_context_m == 1000.0
    assert contract.raster_policy.unsafe_plan_action == "fall_back_to_full_raster"


def test_contract_rejects_a_required_unchanged_collar_match() -> None:
    with pytest.raises(ValueError, match="cannot be required"):
        LocalPatchMergePolicy(collar_match_required=True)


def test_contract_rejects_a_collar_with_reversed_flow_orientation() -> None:
    with pytest.raises(ValueError, match="upstream collar"):
        LocalPatchCollar(
            role="upstream",
            base_link_id=18,
            base_upstream_node_id=13,
            base_downstream_node_id=12,
            domain_boundary_node_id=13,
            outer_splice_anchor_node_id=12,
        )


def test_contract_manifest_records_interface_and_merge_policy(tmp_path) -> None:
    output_path = write_local_patch_contract(
        _example_56_unit_6_contract(), tmp_path / "local_patch_plan.json"
    )
    manifest = json.loads(output_path.read_text())

    assert manifest["contract_version"] == LOCAL_PATCH_CONTRACT_VERSION
    assert manifest["source_splice_anchor_node_id"] == 13
    assert manifest["sink_splice_anchor_node_id"] == 9
    assert manifest["collar_link_ids"] == [18, 14]
    assert manifest["local_context_link_ids"] == [16, 17, 18, 14]
    assert manifest["computational_domain_link_ids"] == [16, 17]
    assert manifest["merge_policy"]["outside_splice_domain_action"] == "retain_base"
    assert manifest["merge_policy"]["collar_match_required"] is False
    assert (
        manifest["merge_policy"]["collar_change_policy"]
        == "allow_retain_replace_split_or_merge"
    )


def _links(records: list[tuple[int, int, int]]) -> pd.DataFrame:
    return pd.DataFrame(records, columns=["id_link", "id_us_node", "id_ds_node"])


def test_planner_selects_closed_unit_and_immediate_unowned_collars() -> None:
    links = _links(
        [
            (18, 13, 12),
            (16, 12, 11),
            (17, 12, 11),
            (14, 11, 9),
        ]
    )
    unit_summary = pd.DataFrame(
        [{"unit_id": 6, "bifurcation": 12, "confluence": 11}]
    )
    membership = pd.DataFrame(
        {"unit_id": [6, 6], "id_link": [16, 17]}
    )

    result = plan_local_patch(
        example_id="56",
        full_links=links,
        unit_summary=unit_summary,
        unit_link_membership=membership,
        bubble_edge_resistance=pd.DataFrame(),
        selected_unit_ids=[6],
    )

    assert result.is_safe
    assert result.contract is not None
    assert set(result.contract.core_link_ids) == {16, 17}
    assert result.contract.collar_link_ids == (18, 14)
    assert result.contract.source_splice_anchor_node_id == 13
    assert result.contract.sink_splice_anchor_node_id == 9
    assert result.contract.upstream_connector_link_ids == ()
    assert result.contract.downstream_connector_link_ids == ()


def test_planner_outer_context_changes_core_but_not_selected_replacement() -> None:
    links = _links(
        [
            (9, 0, 1),
            (1, 1, 2),
            (2, 1, 2),
            (3, 2, 3),
            (4, 1, 3),
            (5, 3, 4),
        ]
    )
    unit_summary = pd.DataFrame(
        [
            {"unit_id": 1, "bifurcation": 1, "confluence": 2},
            {"unit_id": 2, "bifurcation": 1, "confluence": 3},
        ]
    )
    membership = pd.DataFrame(
        {
            "unit_id": [1, 1, 2, 2, 2, 2],
            "id_link": [1, 2, 1, 2, 3, 4],
        }
    )

    result = plan_local_patch(
        example_id="synthetic",
        full_links=links,
        unit_summary=unit_summary,
        unit_link_membership=membership,
        bubble_edge_resistance=pd.DataFrame(),
        selected_unit_ids=[1],
        computational_context_unit_id=2,
    )

    assert result.is_safe
    assert result.contract is not None
    assert result.contract.domain_type == "containing_compound"
    assert set(result.contract.core_link_ids) == {1, 2, 3, 4}
    assert set(result.contract.selected_link_ids) == {1, 2}
    assert result.contract.anchored_replacement_link_ids == (1, 2)
    assert result.audit["computational_context_unit_id"] == 2


def test_planner_uses_closed_compound_context_for_open_member() -> None:
    links = _links(
        [
            (5, 0, 1),
            (1, 1, 2),
            (2, 2, 4),
            (3, 2, 3),
            (4, 3, 4),
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

    result = plan_local_patch(
        example_id="synthetic",
        full_links=links,
        unit_summary=unit_summary,
        unit_link_membership=membership,
        bubble_edge_resistance=bubble_edges,
        selected_unit_ids=[7],
    )

    assert result.is_safe
    assert result.contract is not None
    assert result.contract.domain_type == "containing_compound"
    assert result.contract.selected_compound_bubble_id == 9
    assert set(result.contract.computational_domain_link_ids) == {1, 2, 3, 4}
    assert set(result.contract.selected_link_ids) == {1, 2}
    assert result.contract.anchored_replacement_link_ids == (1, 2)
    assert result.contract.anchored_upstream_boundary_node_ids == (1,)
    assert result.contract.anchored_downstream_boundary_node_ids == (2, 4)
    assert result.contract.collar_link_ids == (5, 6)
    assert result.audit["selected_unit_is_closed"] is False


def test_planner_accepts_an_explicit_complete_compound_selection() -> None:
    links = _links(
        [
            (5, 0, 1),
            (1, 1, 2),
            (2, 2, 4),
            (3, 2, 3),
            (4, 3, 4),
            (6, 4, 5),
        ]
    )
    unit_summary = pd.DataFrame(
        [
            {"unit_id": 7, "bifurcation": 1, "confluence": 4, "compound_bubble_id": 9},
            {"unit_id": 8, "bifurcation": 2, "confluence": 4, "compound_bubble_id": 9},
        ]
    )
    membership = pd.DataFrame(
        {"unit_id": [7, 7, 8, 8], "id_link": [1, 2, 3, 4]}
    )
    bubble_edges = pd.DataFrame(
        {
            "compound_bubble_id": [9, 9, 9, 9],
            "u": [1, 2, 2, 3],
            "v": [2, 4, 3, 4],
        }
    )

    result = plan_local_patch(
        example_id="synthetic",
        full_links=links,
        unit_summary=unit_summary,
        unit_link_membership=membership,
        bubble_edge_resistance=bubble_edges,
        selected_unit_ids=[7, 8],
        selection_type="compound_bubble",
        selected_compound_bubble_id=9,
    )

    assert result.is_safe
    assert result.contract is not None
    assert result.contract.domain_type == "selected_compound"
    assert result.contract.selected_compound_bubble_id == 9
    assert set(result.contract.core_link_ids) == {1, 2, 3, 4}
    assert result.contract.collar_link_ids == (5, 6)


def test_planner_uses_immediate_owned_link_as_temporary_collar_context() -> None:
    links = _links(
        [
            (8, -2, -1),
            (7, -1, 0),
            (5, 0, 1),
            (1, 1, 2),
            (2, 1, 2),
            (6, 2, 3),
        ]
    )
    unit_summary = pd.DataFrame(
        [
            {"unit_id": 1, "bifurcation": 1, "confluence": 2},
            {"unit_id": 2, "bifurcation": -1, "confluence": 1},
        ]
    )
    membership = pd.DataFrame(
        {"unit_id": [1, 1, 2, 2], "id_link": [1, 2, 7, 5]}
    )

    result = plan_local_patch(
        example_id="synthetic",
        full_links=links,
        unit_summary=unit_summary,
        unit_link_membership=membership,
        bubble_edge_resistance=pd.DataFrame(),
        selected_unit_ids=[1],
    )

    assert result.is_safe
    assert result.contract is not None
    assert result.contract.upstream_connector_link_ids == ()
    assert result.contract.upstream_collar.base_link_id == 5
    assert result.contract.upstream_collar.unit_membership_count == 1
    assert result.contract.source_splice_anchor_node_id == 0
    assert result.contract.local_context_link_ids == (1, 2, 5, 6)


def test_planner_requests_full_raster_for_ambiguous_external_interface() -> None:
    links = _links(
        [
            (5, 0, 1),
            (7, 9, 1),
            (1, 1, 2),
            (2, 1, 2),
            (6, 2, 3),
        ]
    )
    unit_summary = pd.DataFrame(
        [{"unit_id": 1, "bifurcation": 1, "confluence": 2}]
    )
    membership = pd.DataFrame({"unit_id": [1, 1], "id_link": [1, 2]})

    result = plan_local_patch(
        example_id="synthetic",
        full_links=links,
        unit_summary=unit_summary,
        unit_link_membership=membership,
        bubble_edge_resistance=pd.DataFrame(),
        selected_unit_ids=[1],
    )

    assert not result.is_safe
    assert result.contract is None
    assert result.status == "fallback_full_raster"
    assert result.fallback_code == "ambiguous_upstream_interface"
