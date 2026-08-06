from __future__ import annotations

import pandas as pd
import pytest

from network_variants.width_routing_solver import (
    build_rejected_correction_report,
    prepare_hybrid_solver_links,
    solve_routing_widths,
)


def _links(rows: list[tuple[int, int, int, float, float, float]]) -> pd.DataFrame:
    return pd.DataFrame(
        rows,
        columns=[
            "id_link",
            "id_us_node",
            "id_ds_node",
            "wid_adj_wet",
            "routing_context_width_us",
            "routing_context_width_ds",
        ],
    )


def test_prepare_hybrid_context_uses_endpoints_only_for_eligible_links() -> None:
    endpoints = pd.DataFrame(
        {
            "example_id": ["02", "02", "02"],
            "id_link": [1, 2, 3],
            "id_us_node": [0, 1, 1],
            "id_ds_node": [1, 2, 3],
            "whole_link_mean_wet_width": [100.0, 200.0, 300.0],
            "width_near_us_node_wet": [110.0, 210.0, 310.0],
            "width_near_ds_node_wet": [120.0, 220.0, 320.0],
            "length_over_wet_width": [12.0, 9.0, 12.0],
            "endpoint_windows_nonoverlapping": [True, True, False],
        }
    )

    result = prepare_hybrid_solver_links(endpoints).set_index("id_link")

    assert result.loc[1, "routing_context_width_us"] == 110.0
    assert result.loc[1, "routing_context_width_ds"] == 120.0
    assert bool(result.loc[1, "routing_context_uses_endpoint"]) is True
    assert result.loc[2, "routing_context_width_us"] == 200.0
    assert result.loc[3, "routing_context_width_us"] == 300.0


def test_single_flagged_outgoing_receives_remaining_incoming_width() -> None:
    links = _links(
        [
            (1, 0, 1, 1000.0, 1000.0, 1000.0),
            (2, 1, 2, 800.0, 800.0, 800.0),
            (3, 1, 3, 300.0, 300.0, 300.0),
        ]
    )
    decisions = pd.DataFrame(
        {"id_link": [2, 3], "width_model_prediction": [1, 0]}
    )

    solved, junctions = solve_routing_widths(links, decisions)
    result = solved.set_index("id_link")
    node = junctions.set_index("id_node").loc[1]

    assert result.loc[2, "wid_adj_wet"] == 800.0
    assert result.loc[2, "wid_adj_routing"] == pytest.approx(700.0)
    assert result.loc[2, "width_correction_scale_factor"] == pytest.approx(0.875)
    assert bool(result.loc[2, "width_correction_applied"]) is True
    assert result.loc[3, "wid_adj_routing"] == 300.0
    assert node["continuity_abs_deviation_before"] == pytest.approx(0.1)
    assert node["continuity_abs_deviation_after"] == pytest.approx(0.0)


def test_multiple_flagged_siblings_are_allocated_jointly_by_context_proportion() -> None:
    links = _links(
        [
            (1, 0, 1, 1000.0, 1000.0, 1000.0),
            (2, 1, 2, 900.0, 900.0, 900.0),
            (3, 1, 3, 300.0, 300.0, 300.0),
            (4, 1, 4, 200.0, 200.0, 200.0),
        ]
    )
    decisions = pd.DataFrame(
        {"id_link": [2, 3, 4], "width_model_prediction": [1, 1, 0]}
    )

    solved, _ = solve_routing_widths(links, decisions)
    result = solved.set_index("id_link")

    assert result.loc[2, "wid_adj_routing"] == pytest.approx(600.0)
    assert result.loc[3, "wid_adj_routing"] == pytest.approx(200.0)
    assert result.loc[2, "wid_adj_routing"] + result.loc[3, "wid_adj_routing"] == 800.0


def test_upstream_correction_scale_propagates_to_downstream_endpoint_context() -> None:
    links = _links(
        [
            (1, 0, 1, 1000.0, 1000.0, 1000.0),
            (2, 1, 2, 800.0, 800.0, 400.0),
            (3, 1, 3, 300.0, 300.0, 300.0),
            (4, 2, 4, 500.0, 500.0, 500.0),
        ]
    )
    decisions = pd.DataFrame(
        {"id_link": [2, 3, 4], "width_model_prediction": [1, 0, 1]}
    )

    solved, junctions = solve_routing_widths(links, decisions)
    result = solved.set_index("id_link")
    nodes = junctions.set_index("id_node")

    assert result.loc[2, "width_correction_scale_factor"] == pytest.approx(0.875)
    assert nodes.loc[2, "incoming_context_width_sum"] == pytest.approx(350.0)
    assert result.loc[4, "wid_adj_routing"] == pytest.approx(350.0)


def test_unsafe_joint_target_retains_all_flagged_raw_widths() -> None:
    links = _links(
        [
            (1, 0, 1, 100.0, 100.0, 100.0),
            (2, 1, 2, 1000.0, 1000.0, 1000.0),
        ]
    )
    decisions = pd.DataFrame({"id_link": [2], "width_model_prediction": [1]})

    solved, junctions = solve_routing_widths(links, decisions)
    result = solved.set_index("id_link").loc[2]

    assert result["wid_adj_routing"] == 1000.0
    assert result["width_qc_decision"] == "retain_raw_flag"
    assert result["width_qc_reason"] == "joint_allocation_outside_scale_bounds"
    assert bool(result["width_qc_flag"]) is True
    assert bool(result["width_correction_applied"]) is False
    assert result["width_correction_candidate_routing_width"] == pytest.approx(100.0)
    assert result["width_correction_candidate_scale_factor"] == pytest.approx(0.1)
    assert result["width_correction_scale_bound_violation"] == "below_min"

    solved.insert(0, "example_id", "02")
    junctions.insert(0, "example_id", "02")
    report = build_rejected_correction_report(solved, junctions)
    assert report["id_link"].tolist() == [2]
    assert report.loc[0, "incoming_context_width_sum"] == 100.0
    assert report.loc[0, "outgoing_link_ids"] == "2"


def test_flagged_source_link_is_not_corrected_without_incoming_context() -> None:
    links = _links([(1, 0, 1, 100.0, 100.0, 100.0)])
    decisions = pd.DataFrame({"id_link": [1], "width_model_prediction": [1]})

    solved, _ = solve_routing_widths(links, decisions)
    result = solved.iloc[0]

    assert result["wid_adj_routing"] == 100.0
    assert result["width_qc_decision"] == "retain_raw_flag"
    assert result["width_qc_reason"] == "no_incoming_width_at_source_node"


def test_accepted_siblings_cannot_leave_a_nonpositive_flagged_allocation() -> None:
    links = _links(
        [
            (1, 0, 1, 100.0, 100.0, 100.0),
            (2, 1, 2, 150.0, 150.0, 150.0),
            (3, 1, 3, 50.0, 50.0, 50.0),
        ]
    )
    decisions = pd.DataFrame(
        {"id_link": [2, 3], "width_model_prediction": [0, 1]}
    )

    solved, _ = solve_routing_widths(links, decisions)
    result = solved.set_index("id_link").loc[3]

    assert result["wid_adj_routing"] == 50.0
    assert result["width_qc_reason"] == "nonpositive_width_remaining_after_accepted_outgoing"


def test_flagged_link_in_directed_cycle_is_retained_and_flagged() -> None:
    links = _links(
        [
            (1, 1, 2, 100.0, 100.0, 100.0),
            (2, 2, 1, 100.0, 100.0, 100.0),
        ]
    )
    decisions = pd.DataFrame({"id_link": [1], "width_model_prediction": [1]})

    solved, _ = solve_routing_widths(links, decisions)
    result = solved.set_index("id_link").loc[1]

    assert result["width_qc_decision"] == "retain_raw_flag"
    assert result["width_qc_reason"] == "directed_topology_cycle"
    assert bool(result["width_qc_flag"]) is True
