from __future__ import annotations

from pathlib import Path
from typing import Any

import geopandas as gpd
import pandas as pd

from hierarchy_level_definition.run_unit_workflow import UnitWorkflowOutputs, write_unit_workflow_outputs
from network_variants.collapse_experiment import run_collapse_experiment
from network_variants.variant_generation import NetworkVariantOutputs


def _workflow_outputs(
    unit_ids: list[int],
    *,
    group_specs: list[tuple[str, list[int]]] | None = None,
) -> UnitWorkflowOutputs:
    unit_summary = pd.DataFrame({"unit_id": unit_ids})
    unit_metrics = pd.DataFrame({"unit_id": unit_ids})
    path_metrics = pd.DataFrame(columns=["unit_id", "path_id"])
    hierarchy_level_metrics = pd.DataFrame(columns=["collapse_level"])

    collapse_ranking = pd.DataFrame(
        [
            {
                "unit_id": unit_id,
                "collapse_order_global": index,
                "collapse_priority_score": 1.0 - (index - 1) * 0.1,
            }
            for index, unit_id in enumerate(unit_ids, start=1)
        ]
    )

    group_rows: list[dict[str, Any]] = []
    if group_specs:
        rank_start = 1
        n_groups = len(group_specs)
        for group_index, (group_label, member_unit_ids) in enumerate(group_specs, start=1):
            rank_end = rank_start + len(member_unit_ids) - 1
            group_rows.append(
                {
                    "n_groups": n_groups,
                    "group_index": group_index,
                    "group_label": group_label,
                    "group_size": len(member_unit_ids),
                    "unit_ids": ",".join(str(unit_id) for unit_id in member_unit_ids),
                    "rank_start": rank_start,
                    "rank_end": rank_end,
                    "compound_bubble_ids": "",
                    "mean_collapse_priority_score": 0.5,
                    "mean_path_disparity_width": 1.0,
                    "mean_effective_n_paths_width": 1.0,
                    "mean_n_valid_paths": float(len(member_unit_ids)),
                    "mean_equivalent_length": 100.0,
                    "mean_elongation": 1.0,
                    "mean_topologic_complexity_score": 1.0,
                    "within_group_cost": 1.0,
                    "partition_total_cost": 1.0,
                    "grouping_rule": "optimal_contiguous_partition",
                }
            )
            rank_start = rank_end + 1

    merge_tree = pd.DataFrame(group_rows)
    selected_groups = merge_tree.copy()
    group_count_summary = (
        pd.DataFrame(
            [
                {
                    "n_groups": len(group_specs),
                    "partition_total_cost": 1.0,
                    "cost_reduction_from_prev": 0.0,
                    "relative_cost_reduction_from_prev": 0.0,
                    "normalized_partition_total_cost": 0.0,
                    "elbow_score": 1.0,
                    "is_optimal_n_groups": True,
                }
            ]
        )
        if group_specs
        else pd.DataFrame(
            columns=[
                "n_groups",
                "partition_total_cost",
                "cost_reduction_from_prev",
                "relative_cost_reduction_from_prev",
                "normalized_partition_total_cost",
                "elbow_score",
                "is_optimal_n_groups",
            ]
        )
    )

    bubble_summary = pd.DataFrame(columns=["compound_bubble_id"])
    return UnitWorkflowOutputs(
        unit_summary=unit_summary,
        unit_metrics=unit_metrics,
        path_metrics=path_metrics,
        hierarchy_level_metrics=hierarchy_level_metrics,
        collapse_ranking=collapse_ranking,
        merge_tree=merge_tree,
        bubble_summary=bubble_summary,
        group_count_summary=group_count_summary,
        selected_groups=selected_groups,
    )


def _make_workflow_runner(outputs: list[UnitWorkflowOutputs]):
    remaining = list(outputs)
    calls: list[tuple[str, str]] = []

    def runner(links_path: str | Path, nodes_path: str | Path, **_: Any) -> UnitWorkflowOutputs:
        calls.append((str(links_path), str(nodes_path)))
        if not remaining:
            raise AssertionError("workflow runner called more times than expected")
        return remaining.pop(0)

    runner.calls = calls
    runner.remaining = remaining
    return runner


def _make_variant_runner():
    calls: list[dict[str, Any]] = []

    def runner(**kwargs: Any) -> NetworkVariantOutputs:
        calls.append(dict(kwargs))
        output_dir = Path(kwargs["output_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "mask").mkdir(exist_ok=True)
        (output_dir / "matching").mkdir(exist_ok=True)
        (output_dir / "directed").mkdir(exist_ok=True)
        (output_dir / "rivgraph").mkdir(exist_ok=True)
        (output_dir / "widths").mkdir(exist_ok=True)
        (output_dir / "summary").mkdir(exist_ok=True)

        variant_id = str(kwargs["variant_id"])
        collapsed_mask_path = output_dir / "mask" / f"{variant_id}_collapsed.tif"
        directed_links_path = output_dir / "directed" / f"{variant_id}_directed_links.gpkg"
        directed_nodes_path = output_dir / "directed" / f"{variant_id}_directed_nodes.gpkg"
        rivgraph_links_path = output_dir / "rivgraph" / f"{variant_id}_links.gpkg"
        rivgraph_nodes_path = output_dir / "rivgraph" / f"{variant_id}_nodes.gpkg"

        for path in (
            collapsed_mask_path,
            directed_links_path,
            directed_nodes_path,
            rivgraph_links_path,
            rivgraph_nodes_path,
        ):
            path.touch()

        empty_frame = pd.DataFrame()
        empty_gdf = gpd.GeoDataFrame(geometry=[], crs="EPSG:3857")
        return NetworkVariantOutputs(
            collapse_components=empty_frame,
            edit_geometries=empty_gdf,
            node_match=empty_frame,
            link_match=empty_frame,
            link_lineage=empty_frame,
            link_width_families=empty_frame,
            link_width_samples=empty_frame,
            enriched_links=empty_gdf,
            directed_links=empty_gdf,
            directed_nodes=empty_gdf,
            output_dir=output_dir,
            collapsed_mask_path=collapsed_mask_path,
            rivgraph_links_path=rivgraph_links_path,
            rivgraph_nodes_path=rivgraph_nodes_path,
            directed_links_path=directed_links_path,
            directed_nodes_path=directed_nodes_path,
            rivgraph_centerline_path=None,
            sword_reaches_path=None,
            sword_nodes_path=None,
            manifest={"variant_id": variant_id},
        )

    runner.calls = calls
    return runner


def test_sequential_units_recomputes_hierarchy_on_each_child_state(tmp_path: Path) -> None:
    workflow_runner = _make_workflow_runner(
        [
            _workflow_outputs([5, 7]),
            _workflow_outputs([2, 9]),
            _workflow_outputs([]),
        ]
    )
    variant_runner = _make_variant_runner()

    results = run_collapse_experiment(
        "sequential-units",
        cleaned_mask_path=tmp_path / "base_cleaned.tif",
        reviewed_links_path=tmp_path / "base_links.gpkg",
        reviewed_nodes_path=tmp_path / "base_nodes.gpkg",
        exit_sides="NS",
        output_dir=tmp_path / "experiment",
        unit_workflow_runner=workflow_runner,
        unit_workflow_writer=write_unit_workflow_outputs,
        variant_runner=variant_runner,
    )

    assert [call["unit_ids"] for call in variant_runner.calls] == [[5], [2]]
    assert Path(variant_runner.calls[1]["reviewed_links_path"]).name == "S001_unit_5_directed_links.gpkg"
    assert list(results.state_registry["state_id"]) == ["S000_base", "S001_unit_5", "S002_unit_2"]
    assert results.manifest["stop_reason"] == "no_units_remaining"
    assert list(results.transition_registry["selected_unit_ids"]) == ["5", "2"]


def test_sequential_groups_uses_current_state_group_labels(tmp_path: Path) -> None:
    workflow_runner = _make_workflow_runner(
        [
            _workflow_outputs([1, 2, 3], group_specs=[("G3_1", [1, 2]), ("G3_2", [3])]),
            _workflow_outputs([4, 5], group_specs=[("G2_1", [4, 5])]),
            _workflow_outputs([], group_specs=None),
        ]
    )
    variant_runner = _make_variant_runner()

    results = run_collapse_experiment(
        "sequential-groups",
        cleaned_mask_path=tmp_path / "base_cleaned.tif",
        reviewed_links_path=tmp_path / "base_links.gpkg",
        reviewed_nodes_path=tmp_path / "base_nodes.gpkg",
        exit_sides="NS",
        output_dir=tmp_path / "experiment",
        unit_workflow_runner=workflow_runner,
        unit_workflow_writer=write_unit_workflow_outputs,
        variant_runner=variant_runner,
    )

    assert [call["group_label"] for call in variant_runner.calls] == ["G3_1", "G2_1"]
    assert [call["example_id"] for call in variant_runner.calls] == ["base", "base"]
    assert Path(variant_runner.calls[1]["workflow_output_dir"]).name == "hierarchy"
    assert "S001_group_G3_1" in str(Path(variant_runner.calls[1]["workflow_output_dir"]))
    assert list(results.state_registry["state_id"]) == ["S000_base", "S001_group_G3_1", "S002_group_G2_1"]
    assert results.manifest["stop_reason"] == "no_units_remaining"
    assert list(results.transition_registry["selected_group_label"]) == ["G3_1", "G2_1"]


def test_independent_units_creates_one_child_per_base_unit(tmp_path: Path) -> None:
    workflow_runner = _make_workflow_runner(
        [
            _workflow_outputs([5, 7, 2]),
            _workflow_outputs([1]),
            _workflow_outputs([1]),
            _workflow_outputs([1]),
        ]
    )
    variant_runner = _make_variant_runner()

    results = run_collapse_experiment(
        "independent-units",
        cleaned_mask_path=tmp_path / "base_cleaned.tif",
        reviewed_links_path=tmp_path / "base_links.gpkg",
        reviewed_nodes_path=tmp_path / "base_nodes.gpkg",
        exit_sides="NS",
        output_dir=tmp_path / "experiment",
        unit_workflow_runner=workflow_runner,
        unit_workflow_writer=write_unit_workflow_outputs,
        variant_runner=variant_runner,
    )

    assert [call["unit_ids"] for call in variant_runner.calls] == [[5], [7], [2]]
    assert all(str(call["reviewed_links_path"]).endswith("base_links.gpkg") for call in variant_runner.calls)
    assert list(results.state_registry["parent_state_id"]) == ["", "S000_base", "S000_base", "S000_base"]
    assert list(results.state_registry["depth"]) == [0, 1, 1, 1]
    assert results.manifest["stop_reason"] == "all_base_units_processed"
