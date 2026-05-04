from __future__ import annotations

import pandas as pd
import pytest

from hierarchy_level_definition.collapse_decisions.unit_collapse_decisions import (
    build_constrained_merge_tree,
    compute_collapse_decisions_from_unit_metrics,
    rank_unit_collapse_priority,
    summarize_collapse_bubbles,
    summarize_group_count_selection,
)
from hierarchy_level_definition.run_unit_workflow import select_optimal_groups


def _make_unit_metrics() -> pd.DataFrame:
    return pd.DataFrame.from_records(
        [
            {
                "unit_id": 10,
                "compound_bubble_id": 1,
                "primary_parent_id": pd.NA,
                "compound_unit_id": 10,
                "compound_bubble_role": "bubble_root",
                "unit_node_ids": "1,2,3,4",
                "n_valid_paths": 3,
                "effective_n_paths_width": 2.6,
                "path_disparity_width": 1.15,
                "equivalent_length": 800.0,
                "elongation": 4.0,
                "topologic_complexity_score": 3.1,
            },
            {
                "unit_id": 20,
                "compound_bubble_id": 1,
                "primary_parent_id": 10,
                "compound_unit_id": 10,
                "compound_bubble_role": "bubble_member",
                "unit_node_ids": "2,5,6",
                "n_valid_paths": 2,
                "effective_n_paths_width": 1.25,
                "path_disparity_width": 1.6,
                "equivalent_length": 120.0,
                "elongation": 0.9,
                "topologic_complexity_score": 1.1,
            },
            {
                "unit_id": 40,
                "compound_bubble_id": 1,
                "primary_parent_id": 10,
                "compound_unit_id": 10,
                "compound_bubble_role": "bubble_member",
                "unit_node_ids": "2,7,8",
                "n_valid_paths": 2,
                "effective_n_paths_width": 1.55,
                "path_disparity_width": 1.28,
                "equivalent_length": 260.0,
                "elongation": 2.2,
                "topologic_complexity_score": 1.35,
            },
            {
                "unit_id": 30,
                "compound_bubble_id": 2,
                "primary_parent_id": pd.NA,
                "compound_unit_id": pd.NA,
                "compound_bubble_role": "standalone",
                "unit_node_ids": "9,10",
                "n_valid_paths": 2,
                "effective_n_paths_width": 1.28,
                "path_disparity_width": 1.56,
                "equivalent_length": 125.0,
                "elongation": 0.95,
                "topologic_complexity_score": 1.12,
            },
        ]
    )


def test_rank_unit_collapse_priority_orders_more_redundant_units_first() -> None:
    unit_metrics = _make_unit_metrics()

    ranking = rank_unit_collapse_priority(unit_metrics)

    ranked_ids = ranking.sort_values("collapse_order_global")["unit_id"].tolist()
    assert ranked_ids[:3] == [20, 30, 40]
    assert ranking.set_index("unit_id").loc[20, "collapse_order_in_bubble"] == 1
    assert ranking.set_index("unit_id").loc[40, "collapse_order_in_bubble"] == 2
    assert ranking.set_index("unit_id").loc[10, "collapse_order_in_bubble"] == 3


def test_build_constrained_merge_tree_merges_adjacent_units_in_global_ranking() -> None:
    unit_metrics = _make_unit_metrics()

    merge_tree = build_constrained_merge_tree(unit_metrics)

    assert set(merge_tree["n_groups"].tolist()) == {1, 2, 3, 4}
    assert (merge_tree["grouping_rule"] == "optimal_contiguous_partition").all()

    groups_for_one = merge_tree.loc[merge_tree["n_groups"] == 1].sort_values("group_index")
    assert len(groups_for_one) == 1
    assert groups_for_one.iloc[0]["group_size"] == 4
    assert groups_for_one.iloc[0]["rank_start"] == 1
    assert groups_for_one.iloc[0]["rank_end"] == 4

    groups_for_two = merge_tree.loc[merge_tree["n_groups"] == 2].sort_values("group_index")
    assert len(groups_for_two) == 2
    assert groups_for_two.iloc[0]["rank_start"] == 1
    assert groups_for_two.iloc[0]["rank_end"] + 1 == groups_for_two.iloc[1]["rank_start"]
    assert groups_for_two.iloc[1]["rank_end"] == 4
    assert set(groups_for_two.iloc[0]["unit_ids"].split(",")) == {"20", "30", "40"}
    assert set(groups_for_two.iloc[1]["unit_ids"].split(",")) == {"10"}

    groups_for_four = merge_tree.loc[merge_tree["n_groups"] == 4].sort_values("group_index")
    assert len(groups_for_four) == 4
    assert groups_for_four["group_size"].tolist() == [1, 1, 1, 1]


def test_compute_collapse_decisions_from_unit_metrics_returns_all_outputs() -> None:
    unit_metrics = _make_unit_metrics()

    collapse_ranking, merge_tree, bubble_summary = compute_collapse_decisions_from_unit_metrics(unit_metrics)

    assert len(collapse_ranking) == 4
    assert len(merge_tree) == 10
    assert len(bubble_summary) == 2

    bubble_summary_by_id = bubble_summary.set_index("compound_bubble_id")
    assert bubble_summary_by_id.loc[1, "bubble_root_unit_id"] == 10
    assert bubble_summary_by_id.loc[1, "n_units"] == 3
    assert bubble_summary_by_id.loc[2, "n_group_rows"] >= 1


def test_summarize_collapse_bubbles_uses_ranking_and_merge_tree() -> None:
    unit_metrics = _make_unit_metrics()
    ranking = rank_unit_collapse_priority(unit_metrics)
    merge_tree = build_constrained_merge_tree(unit_metrics)

    bubble_summary = summarize_collapse_bubbles(ranking, merge_tree)

    bubble1 = bubble_summary.set_index("compound_bubble_id").loc[1]
    assert bubble1["n_group_rows"] >= 1
    assert bubble1["min_n_groups"] == 1
    assert bubble1["mean_collapse_priority_score"] == pytest.approx(
        ranking.loc[ranking["compound_bubble_id"] == 1, "collapse_priority_score"].mean()
    )


def test_summarize_group_count_selection_returns_one_optimal_group_count() -> None:
    unit_metrics = _make_unit_metrics()
    partitions = build_constrained_merge_tree(unit_metrics)

    summary = summarize_group_count_selection(partitions)

    assert set(summary["n_groups"].tolist()) == {1, 2, 3, 4}
    assert summary["is_optimal_n_groups"].sum() == 1
    assert "n_valid_paths" not in partitions.attrs["collapse_config"]["merge_feature_columns"]
    assert summary.attrs["collapse_config"]["optimal_n_groups"] in {1, 2, 3, 4}
    assert "min_relative_cost_reduction" in summary.attrs["collapse_config"]


def test_select_optimal_groups_filters_to_selected_partition() -> None:
    unit_metrics = _make_unit_metrics()
    partitions = build_constrained_merge_tree(unit_metrics)
    summary = summarize_group_count_selection(partitions)

    selected_groups = select_optimal_groups(partitions, summary)

    assert not selected_groups.empty
    optimal_n_groups = int(summary.loc[summary["is_optimal_n_groups"], "n_groups"].iloc[0])
    assert set(selected_groups["n_groups"].tolist()) == {optimal_n_groups}
    assert selected_groups["group_index"].tolist() == list(range(1, len(selected_groups) + 1))


def test_select_optimal_groups_can_keep_single_group_for_homogeneous_state() -> None:
    unit_metrics = pd.DataFrame.from_records(
        [
            {
                "unit_id": unit_id,
                "compound_bubble_id": pd.NA,
                "primary_parent_id": pd.NA,
                "compound_unit_id": pd.NA,
                "compound_bubble_role": "standalone",
                "unit_node_ids": f"{unit_id},{unit_id + 1}",
                "n_valid_paths": 2,
                "effective_n_paths_width": 1.8,
                "path_disparity_width": 1.1,
                "equivalent_length": 500.0,
                "elongation": 2.0,
                "topologic_complexity_score": 1.2,
            }
            for unit_id in (1, 2, 3, 4)
        ]
    )

    partitions = build_constrained_merge_tree(unit_metrics)
    summary = summarize_group_count_selection(partitions)
    selected_groups = select_optimal_groups(partitions, summary)

    assert int(summary.loc[summary["is_optimal_n_groups"], "n_groups"].iloc[0]) == 1
    assert len(selected_groups) == 1
    assert selected_groups.iloc[0]["group_label"] == "G1_1"
    assert selected_groups.iloc[0]["group_size"] == 4
