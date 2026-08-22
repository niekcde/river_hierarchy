from __future__ import annotations
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import LineString
from shapely.ops import unary_union

from hierarchy_level_definition.metrics import BubbleEquivalentResistanceConfig
from hierarchy_level_definition.run_unit_workflow import UnitWorkflowOutputs, write_unit_workflow_outputs
import network_variants.collapse_experiment as collapse_experiment_module
from network_variants.collapse_experiment import (
    BaseStateVariantOutputs,
    ExperimentStateContext,
    _build_unit_link_membership,
    _copy_boundary_component_artifacts,
    _ensure_seed_direction_columns,
    _normalize_numeric_id_columns,
    resolve_execution_backend_plan,
    _variant_analysis_links_path,
    _write_equivalent_transition_outputs,
    main,
    run_collapse_experiment,
)
from network_variants.hydraulic_context import StateHydraulicContextOutputs
from network_variants.variant_generation import NetworkVariantOutputs, VariantDirectionValidationError
from rapid_tools.forcing import ForcingConfig


@pytest.mark.parametrize(
    ("requested", "base", "child", "fallback"),
    [
        ("full_raster", "full_raster", "full_raster", None),
        ("panelized", "panelized", "panelized", None),
        ("local_patch", "full_raster", "local_patch", "full_raster"),
        (("panelized", "local_patch"), "panelized", "local_patch", "panelized"),
        (("local_patch", "panelized"), "panelized", "local_patch", "panelized"),
    ],
)
def test_resolve_execution_backend_plan(
    requested: str | tuple[str, ...],
    base: str,
    child: str,
    fallback: str | None,
) -> None:
    plan = resolve_execution_backend_plan(requested)

    assert plan.base_backend == base
    assert plan.child_backend == child
    assert plan.fallback_backend == fallback


@pytest.mark.parametrize(
    "requested",
    [(), ("full_raster", "local_patch"), ("full_raster", "panelized"), ("unknown",)],
)
def test_resolve_execution_backend_plan_rejects_unsupported_combinations(
    requested: tuple[str, ...],
) -> None:
    with pytest.raises(ValueError):
        resolve_execution_backend_plan(requested)


def test_panelized_reference_section_builder_collapses_every_base_unit(
    tmp_path: Path,
) -> None:
    calls: list[dict[str, Any]] = []
    variant_dir = tmp_path / "reference_section" / "variant"

    def panelized_runner(**kwargs: Any):
        calls.append(kwargs)
        return SimpleNamespace(
            output_dir=variant_dir,
            directed_links_path=variant_dir / "directed_links.gpkg",
            directed_nodes_path=variant_dir / "directed_nodes.gpkg",
            rivgraph_centerline_path=None,
            collapsed_mask_path=variant_dir / "collapsed_mask.vrt",
            boundary_component_labels_path=tmp_path / "boundary_labels.tif",
            manifest={},
        )

    base_context = SimpleNamespace(
        workflow=SimpleNamespace(
            unit_summary=pd.DataFrame({"unit_id": [7, 3, 7, 11]})
        ),
        hierarchy_output_dir=tmp_path / "states" / "S000_base" / "hierarchy",
    )
    base_outputs = SimpleNamespace(
        collapsed_mask_path=tmp_path / "base_mask.tif",
        directed_nodes_path=tmp_path / "base_nodes.gpkg",
    )
    builder = collapse_experiment_module._make_panelized_reference_section_builder(
        panelized_variant_runner=panelized_runner,
        base_context=base_context,
        base_variant_outputs=base_outputs,
        base_analysis_links_path=tmp_path / "base_links.gpkg",
        example_id="41",
        exit_sides="EW",
        preferred_width_field="wid_adj_routing",
        footprint_buffer_scale=0.5,
        all_touched=True,
        bank_candidate_neighbors_max=2048,
        sword_node_source_path=None,
        sword_wse_field="wse_obs_p50",
        sword_match_tolerance=None,
        sword_example_station_source_path=None,
        sword_station_match_source_path=None,
        sword_reach_buffer_steps=2,
    )

    summary = builder(reference_paths={"variant": variant_dir})

    assert len(calls) == 1
    assert calls[0]["unit_ids"] == [3, 7, 11]
    assert calls[0]["collapse_corridor_mode"] == "realized"
    assert calls[0]["panelized_mask_publication"] == "virtual_patch"
    assert calls[0]["selection_type"] == "reference_section_all_units"
    assert summary["construction_backend"] == "panelized"
    assert summary["unit_ids"] == [3, 7, 11]


@pytest.mark.parametrize("reset_name", ["_reset_state_dir", "_reset_experiment_output_dir"])
def test_reset_directory_ignores_missing_entry_race(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    reset_name: str,
) -> None:
    target = tmp_path / "output"
    target.mkdir()
    (target / "stale.txt").write_text("stale", encoding="utf-8")
    real_rmtree = collapse_experiment_module.shutil.rmtree

    def racing_rmtree(path: Path, *, onexc: Any) -> None:
        onexc(Path.unlink, str(Path(path) / "already-gone"), FileNotFoundError("gone"))
        real_rmtree(path)

    monkeypatch.setattr(collapse_experiment_module.shutil, "rmtree", racing_rmtree)

    getattr(collapse_experiment_module, reset_name)(target)

    assert target.is_dir()
    assert list(target.iterdir()) == []


@pytest.mark.parametrize("reset_name", ["_reset_state_dir", "_reset_experiment_output_dir"])
def test_reset_directory_does_not_hide_permission_errors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    reset_name: str,
) -> None:
    target = tmp_path / "output"
    target.mkdir()

    def failing_rmtree(path: Path, *, onexc: Any) -> None:
        del path
        onexc(Path.unlink, "protected", PermissionError("denied"))

    monkeypatch.setattr(collapse_experiment_module.shutil, "rmtree", failing_rmtree)

    with pytest.raises(PermissionError, match="denied"):
        getattr(collapse_experiment_module, reset_name)(target)


def _workflow_outputs(
    unit_ids: list[int],
    *,
    group_specs: list[tuple[str, list[int]]] | None = None,
    compound_rows: list[dict[str, Any]] | None = None,
    bubble_equivalent_rows: list[dict[str, Any]] | None = None,
) -> UnitWorkflowOutputs:
    unit_summary = pd.DataFrame({"unit_id": unit_ids})
    unit_metrics = pd.DataFrame({"unit_id": unit_ids})
    if compound_rows:
        compound_frame = pd.DataFrame(compound_rows)
        if "unit_id" not in compound_frame.columns:
            raise ValueError("compound_rows must contain unit_id values.")
        unit_summary = unit_summary.merge(compound_frame, on="unit_id", how="left", validate="1:1")
        unit_metrics = unit_metrics.merge(compound_frame, on="unit_id", how="left", validate="1:1")
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
    bubble_equivalent_metrics = pd.DataFrame.from_records(bubble_equivalent_rows or [])
    if bubble_equivalent_metrics.empty:
        bubble_equivalent_metrics = pd.DataFrame(
            columns=[
                "compound_bubble_id",
                "bubble_root_unit_id",
                "upstream_boundary_node",
                "downstream_boundary_node",
                "n_member_units",
                "member_unit_ids",
                "n_nodes",
                "n_edges",
                "solver_success",
                "solver_method",
                "solver_message",
                "solver_nfev",
                "K_eq",
                "G_eq",
                "Q_unit",
                "residual_norm",
                "residual_relative",
                "resistance_law",
                "geometry_method",
                "kb_mode",
                "kb_value",
                "kb_source_method",
            ]
        )
    bubble_edge_resistance = pd.DataFrame(columns=["compound_bubble_id", "edge_id"])
    bubble_node_heads = pd.DataFrame(columns=["compound_bubble_id", "node_id"])
    return UnitWorkflowOutputs(
        unit_summary=unit_summary,
        unit_metrics=unit_metrics,
        path_metrics=path_metrics,
        hierarchy_level_metrics=hierarchy_level_metrics,
        collapse_ranking=collapse_ranking,
        merge_tree=merge_tree,
        bubble_summary=bubble_summary,
        bubble_equivalent_metrics=bubble_equivalent_metrics,
        bubble_edge_resistance=bubble_edge_resistance,
        bubble_node_heads=bubble_node_heads,
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


def test_variant_analysis_links_prefer_width_family_product(tmp_path: Path) -> None:
    variant_dir = tmp_path / "variant"
    directed_path = variant_dir / "directed" / "state_directed_links.gpkg"
    width_family_path = variant_dir / "widths" / "links_with_width_families.gpkg"
    directed_path.parent.mkdir(parents=True)
    directed_path.touch()

    assert _variant_analysis_links_path(
        variant_output_dir=variant_dir,
        directed_links_path=directed_path,
    ) == directed_path

    width_family_path.parent.mkdir(parents=True)
    width_family_path.touch()
    assert _variant_analysis_links_path(
        variant_output_dir=variant_dir,
        directed_links_path=directed_path,
    ) == width_family_path


def test_copied_base_localizes_nested_routing_audit_reference(tmp_path: Path) -> None:
    source = tmp_path / "source" / "states" / "S000_base" / "variant"
    for directory in ("summary", "widths", "directed", "mask"):
        (source / directory).mkdir(parents=True, exist_ok=True)
    (source / "summary" / "variant_manifest.json").write_text(
        json.dumps({"output_paths": {"output_dir": str(source.resolve())}}),
        encoding="utf-8",
    )
    (source / "widths" / "routing_width_application.csv").write_text(
        "id_link,wid_adj_routing\n1,10\n", encoding="utf-8"
    )
    (source / "widths" / "routing_width_application_manifest.json").write_text(
        json.dumps(
            {
                "files": {
                    "link_audit": str(
                        (source / "widths" / "routing_width_application.csv").resolve()
                    )
                }
            }
        ),
        encoding="utf-8",
    )
    (source / "directed" / "99__S000_base_directed_links.gpkg").touch()
    (source / "directed" / "99__S000_base_directed_nodes.gpkg").touch()
    (source / "mask" / "99__S000_base_collapsed.tif").touch()
    (source / "directed" / "._99__S000_base_directed_links.gpkg").touch()
    (source / "directed" / "._99__S000_base_directed_nodes.gpkg").touch()
    (source / "mask" / "._99__S000_base_collapsed.tif").touch()
    state_dir = tmp_path / "destination" / "states" / "S000_base"
    context = SimpleNamespace(state_dir=state_dir)

    copied = collapse_experiment_module._materialize_copied_base_state_variant(
        context=context,
        source_variant_dir=source,
        example_id="99",
        collapse_corridor_mode="anchored",
        boundary_component_labels_path=None,
    )

    copied_manifest = json.loads(
        (
            state_dir
            / "variant"
            / "widths"
            / "routing_width_application_manifest.json"
        ).read_text(encoding="utf-8")
    )
    assert copied_manifest["files"]["link_audit"] == "routing_width_application.csv"
    assert not copied.directed_links_path.name.startswith("._")
    assert not copied.directed_nodes_path.name.startswith("._")
    assert not copied.collapsed_mask_path.name.startswith("._")
    assert not any(
        path.name.startswith("._")
        for path in (state_dir / "variant").rglob("*")
    )


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
        width_scope_mask_path = output_dir / "mask" / "width_scope_mask.tif"
        directed_links_path = output_dir / "directed" / f"{variant_id}_directed_links.gpkg"
        directed_nodes_path = output_dir / "directed" / f"{variant_id}_directed_nodes.gpkg"
        rivgraph_links_path = output_dir / "rivgraph" / f"{variant_id}_links.gpkg"
        rivgraph_nodes_path = output_dir / "rivgraph" / f"{variant_id}_nodes.gpkg"

        for path in (
            collapsed_mask_path,
            width_scope_mask_path,
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
            node_sword_match=empty_frame,
            link_match=empty_frame,
            link_lineage=empty_frame,
            link_width_families=empty_frame,
            link_width_samples=empty_frame,
            enriched_links=empty_gdf,
            directed_links=empty_gdf,
            directed_nodes=empty_gdf,
            output_dir=output_dir,
            collapsed_mask_path=collapsed_mask_path,
            width_scope_mask_path=width_scope_mask_path,
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


def _make_variant_outputs(output_dir: Path, variant_id: str, *, direction_valid: bool = True) -> NetworkVariantOutputs:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "mask").mkdir(exist_ok=True)
    (output_dir / "matching").mkdir(exist_ok=True)
    (output_dir / "directed").mkdir(exist_ok=True)
    (output_dir / "rivgraph").mkdir(exist_ok=True)
    (output_dir / "widths").mkdir(exist_ok=True)
    (output_dir / "summary").mkdir(exist_ok=True)

    collapsed_mask_path = output_dir / "mask" / f"{variant_id}_collapsed.tif"
    width_scope_mask_path = output_dir / "mask" / "width_scope_mask.tif"
    directed_links_path = output_dir / "directed" / f"{variant_id}_directed_links.gpkg"
    directed_nodes_path = output_dir / "directed" / f"{variant_id}_directed_nodes.gpkg"
    rivgraph_links_path = output_dir / "rivgraph" / f"{variant_id}_links.gpkg"
    rivgraph_nodes_path = output_dir / "rivgraph" / f"{variant_id}_nodes.gpkg"
    direction_validation_report = output_dir / "directed" / "direction_validation_report.json"

    for path in (
        collapsed_mask_path,
        width_scope_mask_path,
        directed_links_path,
        directed_nodes_path,
        rivgraph_links_path,
        rivgraph_nodes_path,
        direction_validation_report,
    ):
        path.touch()

    empty_frame = pd.DataFrame()
    empty_gdf = gpd.GeoDataFrame(geometry=[], crs="EPSG:3857")
    return NetworkVariantOutputs(
        collapse_components=empty_frame,
        edit_geometries=empty_gdf,
        node_match=empty_frame,
        node_sword_match=empty_frame,
        link_match=empty_frame,
        link_lineage=empty_frame,
        link_width_families=empty_frame,
        link_width_samples=empty_frame,
        enriched_links=empty_gdf,
        directed_links=empty_gdf,
        directed_nodes=empty_gdf,
        output_dir=output_dir,
        collapsed_mask_path=collapsed_mask_path,
        width_scope_mask_path=width_scope_mask_path,
        rivgraph_links_path=rivgraph_links_path,
        rivgraph_nodes_path=rivgraph_nodes_path,
        directed_links_path=directed_links_path,
        directed_nodes_path=directed_nodes_path,
        rivgraph_centerline_path=None,
        sword_reaches_path=None,
        sword_nodes_path=None,
        manifest={
            "variant_id": variant_id,
            "direction_summary": {"is_valid": direction_valid, "issues": [] if direction_valid else ["invalid flow graph"]},
            "output_paths": {"direction_validation_report": str(direction_validation_report.resolve())},
        },
    )


def _make_base_state_materializer():
    calls: list[dict[str, Any]] = []

    def materializer(**kwargs: Any) -> BaseStateVariantOutputs:
        calls.append(dict(kwargs))
        context = kwargs["context"]
        output_dir = context.state_dir / "variant"
        directed_dir = output_dir / "directed"
        mask_dir = output_dir / "mask"
        for directory in (output_dir, directed_dir, mask_dir):
            directory.mkdir(parents=True, exist_ok=True)
        collapsed_mask_path = mask_dir / "S000_base_collapsed.tif"
        directed_links_path = directed_dir / "S000_base_directed_links.gpkg"
        directed_nodes_path = directed_dir / "S000_base_directed_nodes.gpkg"
        for path in (collapsed_mask_path, directed_links_path, directed_nodes_path):
            path.touch()
        return BaseStateVariantOutputs(
            output_dir=output_dir,
            collapsed_mask_path=collapsed_mask_path,
            directed_links_path=directed_links_path,
            directed_nodes_path=directed_nodes_path,
        )

    materializer.calls = calls
    return materializer


def _make_state_hydraulic_context_runner():
    calls: list[dict[str, Any]] = []

    def runner(**kwargs: Any) -> StateHydraulicContextOutputs:
        calls.append(dict(kwargs))
        output_dir = Path(kwargs["output_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)
        link_context_path = output_dir / "link_hydraulic_context.csv"
        node_context_path = output_dir / "node_hydraulic_context.csv"
        manifest_path = output_dir / "hydraulic_context_manifest.json"
        pd.DataFrame(columns=["id_link", "slope_used"]).to_csv(link_context_path, index=False)
        pd.DataFrame(columns=["id_node", "rapid_wse_for_slope"]).to_csv(node_context_path, index=False)
        manifest_path.write_text("{}", encoding="utf-8")
        return StateHydraulicContextOutputs(
            output_dir=output_dir,
            link_context_path=link_context_path,
            node_context_path=node_context_path,
            manifest_path=manifest_path,
            link_context=pd.DataFrame(columns=["id_link", "slope_used"]),
            node_context=pd.DataFrame(columns=["id_node", "rapid_wse_for_slope"]),
            manifest={},
        )

    runner.calls = calls
    return runner


def _make_equivalent_forcing_table_runner():
    calls: list[dict[str, Any]] = []

    def runner(forcing_path: str | Path, **kwargs: Any) -> tuple[pd.DataFrame, dict[str, object]]:
        calls.append({"forcing_path": str(forcing_path), **dict(kwargs)})
        forcing = pd.DataFrame(
            {
                "time": pd.to_datetime(["2023-01-01T00:00:00Z", "2023-01-01T00:15:00Z"], utc=True),
                "discharge_cms": [100.0, 125.0],
            }
        )
        return forcing, {"selected_station_key": "TEST:1", "normalized_row_count": 2}

    runner.calls = calls
    return runner


def _make_equivalent_reference_kb_summary_runner():
    calls: list[dict[str, Any]] = []

    def runner(
        experiment_dir: str | Path,
        *,
        forcing: pd.DataFrame,
        width_sample_field: str,
        width_percentile: float,
        model_path: str | Path | None = None,
    ) -> dict[str, Any]:
        calls.append(
            {
                "experiment_dir": str(experiment_dir),
                "forcing_rows": int(len(forcing)),
                "width_sample_field": width_sample_field,
                "width_percentile": width_percentile,
                "model_path": None if model_path is None else str(model_path),
            }
        )
        return {
            "kb_source_method": "based_reference_section",
            "kb_value": 7.5,
            "reference_discharge_cms": 125.0,
            "reference_width_m": 40.0,
            "reference_slope": 1e-4,
            "reference_depth_m": 5.333333333333333,
            "reference_section_summary_path": str(Path(experiment_dir) / "reference_section" / "reference_section_summary.json"),
            "based_model_path": "" if model_path is None else str(model_path),
        }

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
    base_state_materializer = _make_base_state_materializer()
    state_hydraulic_context_runner = _make_state_hydraulic_context_runner()

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
        base_state_variant_materializer=base_state_materializer,
        state_hydraulic_context_runner=state_hydraulic_context_runner,
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
    base_state_materializer = _make_base_state_materializer()
    state_hydraulic_context_runner = _make_state_hydraulic_context_runner()

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
        base_state_variant_materializer=base_state_materializer,
        state_hydraulic_context_runner=state_hydraulic_context_runner,
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
    base_state_materializer = _make_base_state_materializer()
    state_hydraulic_context_runner = _make_state_hydraulic_context_runner()

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
        base_state_variant_materializer=base_state_materializer,
        state_hydraulic_context_runner=state_hydraulic_context_runner,
    )

    assert [call["unit_ids"] for call in variant_runner.calls] == [[5], [7], [2]]
    assert all(str(call["reviewed_links_path"]).endswith("S000_base_directed_links.gpkg") for call in variant_runner.calls)
    assert list(results.state_registry["parent_state_id"]) == ["", "S000_base", "S000_base", "S000_base"]
    assert list(results.state_registry["depth"]) == [0, 1, 1, 1]
    assert results.manifest["stop_reason"] == "all_base_units_processed"
    for state_id in ("S001_unit_5", "S002_unit_7", "S003_unit_2"):
        timing_path = tmp_path / "experiment" / "states" / state_id / "stage_timings.json"
        timing = json.loads(timing_path.read_text(encoding="utf-8"))
        assert timing["schema_version"] == "collapse-state-stage-timings-v1"
        assert timing["state_id"] == state_id
        assert timing["variant_generation_seconds"] >= 0
        assert timing["hierarchy_seconds"] >= 0
        assert timing["hydraulic_context_seconds"] >= 0
        assert timing["state_total_seconds"] >= 0


def test_independent_units_can_defer_child_hierarchy(tmp_path: Path) -> None:
    workflow_runner = _make_workflow_runner([_workflow_outputs([5])])
    state_hydraulic_context_runner = _make_state_hydraulic_context_runner()

    results = run_collapse_experiment(
        "independent-units",
        cleaned_mask_path=tmp_path / "base_cleaned.tif",
        reviewed_links_path=tmp_path / "base_links.gpkg",
        reviewed_nodes_path=tmp_path / "base_nodes.gpkg",
        exit_sides="NS",
        output_dir=tmp_path / "experiment",
        unit_workflow_runner=workflow_runner,
        unit_workflow_writer=write_unit_workflow_outputs,
        variant_runner=_make_variant_runner(),
        base_state_variant_materializer=_make_base_state_materializer(),
        state_hydraulic_context_runner=state_hydraulic_context_runner,
        defer_child_hierarchy=True,
    )

    assert len(workflow_runner.calls) == 1
    child = results.state_registry.loc[results.state_registry["state_id"].eq("S001_unit_5")].iloc[0]
    assert child["hierarchy_status"] == "deferred"
    assert child["status"] == "hydraulic_complete_hierarchy_deferred"
    assert not (tmp_path / "experiment" / "states" / "S001_unit_5" / "hierarchy").exists()
    assert results.transition_registry.iloc[0]["child_hierarchy_status"] == "deferred"
    assert results.equivalent_transition_registry.empty
    assert results.manifest["n_hierarchy_deferred_states"] == 1
    assert results.manifest["options"]["hierarchy"]["defer_child_hierarchy"] is True
    assert "parent_link_context_path" in state_hydraulic_context_runner.calls[1]


def test_sequential_mode_rejects_deferred_child_hierarchy(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="sequential modes require"):
        run_collapse_experiment(
            "sequential-units",
            cleaned_mask_path=tmp_path / "base_cleaned.tif",
            reviewed_links_path=tmp_path / "base_links.gpkg",
            reviewed_nodes_path=tmp_path / "base_nodes.gpkg",
            exit_sides="NS",
            output_dir=tmp_path / "experiment",
            defer_child_hierarchy=True,
        )


def test_independent_units_can_filter_to_requested_unit_ids(tmp_path: Path) -> None:
    workflow_runner = _make_workflow_runner(
        [
            _workflow_outputs([5, 7, 2]),
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
        requested_unit_ids=[7, 2],
        unit_workflow_runner=workflow_runner,
        unit_workflow_writer=write_unit_workflow_outputs,
        variant_runner=variant_runner,
        base_state_variant_materializer=_make_base_state_materializer(),
        state_hydraulic_context_runner=_make_state_hydraulic_context_runner(),
    )

    assert [call["unit_ids"] for call in variant_runner.calls] == [[7], [2]]
    assert list(results.state_registry["state_id"]) == ["S000_base", "S001_unit_7", "S002_unit_2"]
    assert results.manifest["options"]["variant"]["requested_unit_ids"] == [7, 2]


def test_compound_bubbles_creates_one_child_per_compound_bubble(tmp_path: Path) -> None:
    workflow_runner = _make_workflow_runner(
        [
            _workflow_outputs(
                [5, 7, 2, 9, 11],
                compound_rows=[
                    {
                        "unit_id": 5,
                        "compound_bubble_id": 13,
                        "in_compound_bubble": True,
                        "compound_bubble_role": "bubble_root",
                        "unit_node_count": 12,
                    },
                    {
                        "unit_id": 7,
                        "compound_bubble_id": 13,
                        "in_compound_bubble": True,
                        "compound_bubble_role": "bubble_member",
                        "unit_node_count": 9,
                    },
                    {
                        "unit_id": 2,
                        "compound_bubble_id": 8,
                        "in_compound_bubble": True,
                        "compound_bubble_role": "bubble_root",
                        "unit_node_count": 7,
                    },
                    {
                        "unit_id": 9,
                        "compound_bubble_id": 8,
                        "in_compound_bubble": True,
                        "compound_bubble_role": "bubble_member",
                        "unit_node_count": 6,
                    },
                    {
                        "unit_id": 11,
                        "compound_bubble_id": 21,
                        "in_compound_bubble": False,
                        "compound_bubble_role": "standalone",
                        "unit_node_count": 5,
                    },
                ],
            ),
            _workflow_outputs([1]),
            _workflow_outputs([1]),
        ]
    )
    variant_runner = _make_variant_runner()
    base_state_materializer = _make_base_state_materializer()
    state_hydraulic_context_runner = _make_state_hydraulic_context_runner()

    results = run_collapse_experiment(
        "compound-bubbles",
        cleaned_mask_path=tmp_path / "base_cleaned.tif",
        reviewed_links_path=tmp_path / "base_links.gpkg",
        reviewed_nodes_path=tmp_path / "base_nodes.gpkg",
        exit_sides="NS",
        output_dir=tmp_path / "experiment",
        unit_workflow_runner=workflow_runner,
        unit_workflow_writer=write_unit_workflow_outputs,
        variant_runner=variant_runner,
        base_state_variant_materializer=base_state_materializer,
        state_hydraulic_context_runner=state_hydraulic_context_runner,
    )

    assert [call["unit_ids"] for call in variant_runner.calls] == [[5, 7], [2, 9]]
    assert [call["selection_type"] for call in variant_runner.calls] == [
        "compound_bubble",
        "compound_bubble",
    ]
    assert [call["compound_bubble_id"] for call in variant_runner.calls] == [13, 8]
    assert list(results.state_registry["state_id"]) == ["S000_base", "CB001_bubble_13", "CB002_bubble_8"]
    derived = results.state_registry.loc[results.state_registry["state_role"] == "derived"].copy()
    assert derived["selection_type"].tolist() == ["compound_bubble", "compound_bubble"]
    assert derived["is_compound_state"].tolist() == [True, True]
    assert derived["compound_bubble_id"].astype(int).tolist() == [13, 8]
    assert derived["bubble_root_unit_id"].astype(int).tolist() == [5, 2]
    assert derived["total_unit_nodes_selected"].astype(int).tolist() == [21, 13]
    assert results.manifest["stop_reason"] == "all_compound_bubbles_processed"
    assert results.manifest["n_compound_states"] == 2


def test_compound_bubbles_can_filter_to_requested_bubble_ids(tmp_path: Path) -> None:
    compound_rows = [
        {
            "unit_id": unit_id,
            "compound_bubble_id": bubble_id,
            "in_compound_bubble": True,
            "compound_bubble_role": role,
            "unit_node_count": 4,
        }
        for unit_id, bubble_id, role in [
            (5, 13, "bubble_root"),
            (7, 13, "bubble_member"),
            (2, 8, "bubble_root"),
            (9, 8, "bubble_member"),
        ]
    ]
    workflow_runner = _make_workflow_runner(
        [
            _workflow_outputs([5, 7, 2, 9], compound_rows=compound_rows),
            _workflow_outputs([1]),
        ]
    )
    variant_runner = _make_variant_runner()

    results = run_collapse_experiment(
        "compound-bubbles",
        cleaned_mask_path=tmp_path / "base_cleaned.tif",
        reviewed_links_path=tmp_path / "base_links.gpkg",
        reviewed_nodes_path=tmp_path / "base_nodes.gpkg",
        exit_sides="NS",
        output_dir=tmp_path / "experiment",
        requested_compound_bubble_ids=[8],
        unit_workflow_runner=workflow_runner,
        unit_workflow_writer=write_unit_workflow_outputs,
        variant_runner=variant_runner,
        base_state_variant_materializer=_make_base_state_materializer(),
        state_hydraulic_context_runner=_make_state_hydraulic_context_runner(),
    )

    assert [call["unit_ids"] for call in variant_runner.calls] == [[2, 9]]
    assert list(results.state_registry["state_id"]) == ["S000_base", "CB001_bubble_8"]
    assert results.manifest["options"]["variant"]["requested_compound_bubble_ids"] == [8]


def test_panelized_runner_accepts_multi_unit_compound_selection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    links_path = tmp_path / "links.gpkg"
    links = gpd.GeoDataFrame(
        {
            "id_link": [1, 2],
            "id_nodes": ["10,11", "11,12"],
            "wid_adj_routing": [2.0, 10.0],
            "wid_adj_wet": [2.0, 10.0],
            "wid_adj": [2.0, 10.0],
            "geometry": [
                LineString([(0.0, 0.0), (2.0, 0.0)]),
                LineString([(20.0, 0.0), (22.0, 0.0)]),
            ],
        },
        crs="EPSG:3857",
    )
    links.to_file(links_path, driver="GPKG")
    path_metrics = tmp_path / "path_metrics.csv"
    pd.DataFrame(
        [
            {"unit_id": 112, "path_id": 1, "id_links": "1"},
            {"unit_id": 113, "path_id": 1, "id_links": "2"},
        ]
    ).to_csv(path_metrics, index=False)
    captured: dict[str, Any] = {}

    def fake_materialize(**kwargs: Any) -> dict[str, Any]:
        captured["materialize"] = kwargs
        return {
            "panel_plan": {"affected": ["P009"]},
            "merged": {},
            "collapsed_mask": str(tmp_path / "collapsed.tif"),
        }

    def fake_finalize(**kwargs: Any) -> SimpleNamespace:
        captured["finalize"] = kwargs
        return SimpleNamespace(output_dir=Path(kwargs["state_dir"]) / "variant")

    monkeypatch.setattr(
        collapse_experiment_module,
        "materialize_panelized_child_state",
        fake_materialize,
    )
    monkeypatch.setattr(
        collapse_experiment_module,
        "_finalize_panelized_child_result",
        fake_finalize,
    )
    runner = collapse_experiment_module._make_panelized_child_runner(
        source_panel_root=tmp_path / "panels",
        panel_plan_path=tmp_path / "panel_plan.gpkg",
        panel_breaks_path=tmp_path / "panel_breaks.gpkg",
        boundary_match_plan_path=tmp_path / "boundary_match_plan.json",
        parent_merged_links_path=tmp_path / "merged_links.gpkg",
        boundary_component_labels_path=tmp_path / "labels.tif",
        exit_sides_by_panel={"P009": "SE"},
        run_name="46",
    )

    runner(
        unit_ids=[112, 113],
        group_label=None,
        selection_type="compound_bubble",
        compound_bubble_id=7,
        bubble_root_unit_id=112,
        reviewed_links_path=links_path,
        reviewed_nodes_path=tmp_path / "nodes.gpkg",
        cleaned_mask_path=tmp_path / "mask.tif",
        panelized_path_metrics=path_metrics,
        output_dir=tmp_path / "state" / "variant",
        variant_id="CB001_bubble_7",
        example_id="46",
        preferred_width_field="wid_adj_routing",
        footprint_buffer_scale=0.5,
        all_touched=False,
    )

    expected = unary_union(
        [links.geometry.iloc[0].buffer(1.0), links.geometry.iloc[1].buffer(5.0)]
    )
    assert captured["materialize"]["footprint"].symmetric_difference(expected).area == pytest.approx(0.0)
    assert captured["materialize"]["all_touched"] is False
    assert captured["finalize"]["unit_ids"] == [112, 113]
    assert captured["finalize"]["selected_link_ids"] == [1, 2]
    assert captured["finalize"]["selected_node_ids"] == [10, 11, 12]
    assert captured["finalize"]["selection_type"] == "compound_bubble"
    assert captured["finalize"]["compound_bubble_id"] == 7


def test_base_state_is_materialized_as_local_variant_outputs(tmp_path: Path) -> None:
    workflow_runner = _make_workflow_runner([_workflow_outputs([5]), _workflow_outputs([])])
    variant_runner = _make_variant_runner()
    base_state_materializer = _make_base_state_materializer()
    state_hydraulic_context_runner = _make_state_hydraulic_context_runner()

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
        base_state_variant_materializer=base_state_materializer,
        state_hydraulic_context_runner=state_hydraulic_context_runner,
    )

    base_row = results.state_registry.loc[results.state_registry["state_id"] == "S000_base"].iloc[0]
    assert Path(base_row["variant_output_dir"]).name == "variant"
    assert Path(base_row["hydraulic_context_output_dir"]).name == "hydraulic_context"
    assert Path(base_row["directed_links_path"]).name == "S000_base_directed_links.gpkg"
    assert Path(base_row["directed_nodes_path"]).name == "S000_base_directed_nodes.gpkg"
    assert "states/S000_base/variant" in base_row["variant_output_dir"]
    assert base_state_materializer.calls
    assert base_state_materializer.calls[0]["exit_sides"] == "NS"
    assert state_hydraulic_context_runner.calls


def test_base_only_stops_after_materializing_base_state(tmp_path: Path) -> None:
    workflow_runner = _make_workflow_runner([_workflow_outputs([5])])
    variant_runner = _make_variant_runner()
    base_state_materializer = _make_base_state_materializer()
    state_hydraulic_context_runner = _make_state_hydraulic_context_runner()

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
        base_state_variant_materializer=base_state_materializer,
        state_hydraulic_context_runner=state_hydraulic_context_runner,
        base_only=True,
    )

    assert list(results.state_registry["state_id"]) == ["S000_base"]
    assert results.transition_registry.empty
    assert results.manifest["stop_reason"] == "base_only_requested"
    assert results.manifest["options"]["variant"]["base_only"] is True
    assert len(workflow_runner.calls) == 1
    assert not variant_runner.calls
    assert base_state_materializer.calls
    assert state_hydraulic_context_runner.calls


def test_seed_direction_columns_uses_raw_id_nodes_when_explicit_direction_missing() -> None:
    links = gpd.GeoDataFrame(
        [
            {
                "id_link": 1,
                "id_nodes": "10, 11",
                "id_us_node": pd.NA,
                "id_ds_node": pd.NA,
                "geometry": None,
            },
            {
                "id_link": 2,
                "id_nodes": "11, 12",
                "id_us_node": pd.NA,
                "id_ds_node": pd.NA,
                "geometry": None,
            },
        ],
        geometry="geometry",
        crs="EPSG:3857",
    )
    nodes = gpd.GeoDataFrame(
        [
            {"id_node": 10, "is_inlet": True, "is_outlet": False, "geometry": None},
            {"id_node": 11, "is_inlet": False, "is_outlet": False, "geometry": None},
            {"id_node": 12, "is_inlet": False, "is_outlet": True, "geometry": None},
        ],
        geometry="geometry",
        crs="EPSG:3857",
    )

    seeded = _ensure_seed_direction_columns(links, nodes)

    assert seeded["id_us_node"].astype(int).tolist() == [10, 11]
    assert seeded["id_ds_node"].astype(int).tolist() == [11, 12]
    assert seeded["is_inlet"].tolist() == [True, False]
    assert seeded["is_outlet"].tolist() == [False, True]
    assert seeded["type_io"].tolist() == ["inlet", "outlet"]
    assert seeded["direction_assignment_method"].tolist() == ["raw_id_nodes_seed", "raw_id_nodes_seed"]


def test_anchored_collapse_corridor_mode_is_passed_to_variant_generation(tmp_path: Path) -> None:
    workflow_runner = _make_workflow_runner([_workflow_outputs([5]), _workflow_outputs([])])
    variant_runner = _make_variant_runner()
    base_state_materializer = _make_base_state_materializer()
    state_hydraulic_context_runner = _make_state_hydraulic_context_runner()

    results = run_collapse_experiment(
        "independent-units",
        cleaned_mask_path=tmp_path / "base_cleaned.tif",
        reviewed_links_path=tmp_path / "base_links.gpkg",
        reviewed_nodes_path=tmp_path / "base_nodes.gpkg",
        exit_sides="NS",
        output_dir=tmp_path / "experiment",
        collapse_corridor_mode="anchored",
        unit_workflow_runner=workflow_runner,
        unit_workflow_writer=write_unit_workflow_outputs,
        variant_runner=variant_runner,
        base_state_variant_materializer=base_state_materializer,
        state_hydraulic_context_runner=state_hydraulic_context_runner,
    )

    assert results.manifest["options"]["variant"]["collapse_corridor_mode"] == "anchored"
    assert base_state_materializer.calls[0]["collapse_corridor_mode"] == "anchored"
    assert variant_runner.calls[0]["collapse_corridor_mode"] == "anchored"


def test_based_reference_section_kb_is_resolved_for_equivalent_resistance(tmp_path: Path) -> None:
    workflow_runner = _make_workflow_runner(
        [
            _workflow_outputs([5]),
            _workflow_outputs(
                [5],
                bubble_equivalent_rows=[
                    {
                        "compound_bubble_id": 1,
                        "bubble_root_unit_id": 5,
                        "upstream_boundary_node": 1,
                        "downstream_boundary_node": 2,
                        "n_member_units": 1,
                        "member_unit_ids": "5",
                        "n_nodes": 2,
                        "n_edges": 1,
                        "solver_success": True,
                        "solver_method": "boundary_only",
                        "solver_message": "",
                        "solver_nfev": 0,
                        "K_eq": 7.5,
                        "G_eq": 1.0 / 7.5,
                        "Q_unit": 0.3651483717,
                        "residual_norm": 0.0,
                        "residual_relative": 0.0,
                        "resistance_law": "manning",
                        "geometry_method": "wide",
                        "kb_mode": "based_reference_section",
                        "kb_value": 7.5,
                        "kb_source_method": "based_reference_section",
                    }
                ],
            ),
            _workflow_outputs([]),
        ]
    )
    variant_runner = _make_variant_runner()
    base_state_materializer = _make_base_state_materializer()
    state_hydraulic_context_runner = _make_state_hydraulic_context_runner()
    forcing_table_runner = _make_equivalent_forcing_table_runner()
    reference_kb_summary_runner = _make_equivalent_reference_kb_summary_runner()

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
        base_state_variant_materializer=base_state_materializer,
        state_hydraulic_context_runner=state_hydraulic_context_runner,
        equivalent_resistance_config=BubbleEquivalentResistanceConfig(
            resistance_law="manning",
            geometry_method="wide",
            kb_mode="based_reference_section",
            kb_value=20.0,
            default_manning_n=0.035,
        ),
        equivalent_forcing_path=tmp_path / "forcing.parquet",
        equivalent_forcing_config=ForcingConfig(
            time_column="time",
            discharge_column="discharge",
            station_key_column="station_key",
            station_key="TEST:1",
            start_time="2023-01-01T00:00:00Z",
            end_time="2023-01-01T01:00:00Z",
            resample_minutes=15,
            output_cache_dir=str(tmp_path / "forcing_cache"),
        ),
        equivalent_kb_model_path=tmp_path / "based_model.ubj",
        equivalent_kb_width_sample_field="width_wet",
        equivalent_kb_width_percentile=90.0,
        equivalent_forcing_table_runner=forcing_table_runner,
        equivalent_reference_kb_summary_runner=reference_kb_summary_runner,
    )

    assert len(workflow_runner.calls) == 3
    assert len(forcing_table_runner.calls) == 1
    assert len(reference_kb_summary_runner.calls) == 1
    assert reference_kb_summary_runner.calls[0]["width_sample_field"] == "width_wet"
    assert reference_kb_summary_runner.calls[0]["width_percentile"] == pytest.approx(90.0)

    manifest_config = results.manifest["options"]["hierarchy"]["equivalent_resistance_config"]
    assert manifest_config["kb_mode"] == "based_reference_section"
    assert manifest_config["reference_kb_summary"]["kb_value"] == pytest.approx(7.5)
    assert results.manifest["options"]["hierarchy"]["equivalent_reference_kb_summary"]["kb_value"] == pytest.approx(7.5)
    assert results.manifest["reference_kb_prepass"] == {
        "required": True,
        "complete": True,
        "construction_backend": "full_raster",
    }
    assert results.manifest["stop_reason"] != "reference_kb_prepass_pending"

    base_row = results.state_registry.loc[results.state_registry["state_id"] == "S000_base"].iloc[0]
    assert int(base_row["n_equivalent_bubbles"]) == 1
    bubble_metrics = pd.read_csv(Path(base_row["bubble_equivalent_metrics_path"]))
    assert bubble_metrics["kb_mode"].iloc[0] == "based_reference_section"
    assert bubble_metrics["kb_value"].iloc[0] == pytest.approx(7.5)


def test_main_runs_both_collapse_corridor_modes_with_suffixed_outputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, Any]] = []

    def fake_run_collapse_experiment(mode: str, **kwargs: Any) -> SimpleNamespace:
        calls.append({"mode": mode, **dict(kwargs)})
        output_dir = Path(kwargs["output_dir"])
        return SimpleNamespace(
            output_dir=output_dir,
            state_registry=pd.DataFrame([{"state_id": "S000_base"}]),
            transition_registry=pd.DataFrame(),
            equivalent_transition_registry=pd.DataFrame(),
            manifest={"mode": mode, "stop_reason": "test_complete", "terminal_state_id": "S000_base"},
        )

    monkeypatch.setattr(collapse_experiment_module, "run_collapse_experiment", fake_run_collapse_experiment)

    exit_code = main(
        [
            "independent-units",
            "--cleaned-mask",
            str(tmp_path / "base_cleaned.tif"),
            "--reviewed-links",
            str(tmp_path / "base_links.gpkg"),
            "--reviewed-nodes",
            str(tmp_path / "base_nodes.gpkg"),
            "--exit-sides",
            "NS",
            "--experiment-id",
            "demo",
            "--output-dir",
            str(tmp_path / "demo_output"),
            "--collapse-corridor-mode",
            "both",
        ]
    )

    assert exit_code == 0
    assert [call["collapse_corridor_mode"] for call in calls] == ["realized", "anchored"]
    assert [call["experiment_id"] for call in calls] == ["demo__realized", "demo__anchored"]
    assert [Path(call["output_dir"]).name for call in calls] == ["demo_output__realized", "demo_output__anchored"]
    assert "variant_runner" not in calls[0]
    assert callable(calls[1]["variant_runner"])
    assert Path(calls[1]["boundary_component_artifact_source_dir"]) == (
        tmp_path / "demo_output__realized" / "boundary_components"
    ).resolve()


def test_main_builds_only_anchored_from_completed_realized_experiment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, Any]] = []
    realized_source = tmp_path / "demo_output__realized"
    (realized_source / "states" / "S000_base" / "variant").mkdir(parents=True)
    (realized_source / "boundary_components").mkdir()

    def fake_run_collapse_experiment(mode: str, **kwargs: Any) -> SimpleNamespace:
        calls.append({"mode": mode, **dict(kwargs)})
        return SimpleNamespace(
            output_dir=Path(kwargs["output_dir"]),
            state_registry=pd.DataFrame([{"state_id": "S000_base"}]),
            transition_registry=pd.DataFrame(),
            equivalent_transition_registry=pd.DataFrame(),
            manifest={"mode": mode, "stop_reason": "test_complete", "terminal_state_id": "S000_base"},
        )

    monkeypatch.setattr(collapse_experiment_module, "run_collapse_experiment", fake_run_collapse_experiment)

    exit_code = main(
        [
            "compound-bubbles",
            "--cleaned-mask",
            str(tmp_path / "base_cleaned.tif"),
            "--reviewed-links",
            str(tmp_path / "base_links.gpkg"),
            "--reviewed-nodes",
            str(tmp_path / "base_nodes.gpkg"),
            "--exit-sides",
            "NS",
            "--experiment-id",
            "demo",
            "--output-dir",
            str(tmp_path / "demo_output"),
            "--collapse-corridor-mode",
            "anchored",
            "--anchored-source-experiment",
            str(realized_source),
        ]
    )

    assert exit_code == 0
    assert len(calls) == 1
    assert calls[0]["collapse_corridor_mode"] == "anchored"
    assert calls[0]["experiment_id"] == "demo__anchored"
    assert Path(calls[0]["output_dir"]).name == "demo_output__anchored"
    assert callable(calls[0]["variant_runner"])
    assert Path(calls[0]["base_state_variant_source_dir"]) == (
        realized_source / "states" / "S000_base" / "variant"
    ).resolve()
    assert Path(calls[0]["boundary_component_artifact_source_dir"]) == (
        realized_source / "boundary_components"
    ).resolve()


def test_copy_boundary_component_artifacts_localizes_panelized_manifest(tmp_path: Path) -> None:
    source = tmp_path / "realized" / "boundary_components"
    source.mkdir(parents=True)
    labels = source / "46_full_extent_land_water_labels_with_snapped_buffered_border_cut.tif"
    summary = source / "46_full_extent_land_component_summary_with_snapped_buffered_border_cut.csv"
    extensions = source / "46_centerline_border_extensions_snapped.gpkg"
    labels.write_bytes(b"labels")
    summary.write_text("land_component_id,raster_value\n1,2\n", encoding="utf-8")
    extensions.write_bytes(b"extensions")
    manifest_path = source / "46_panelized_boundary_component_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "algorithm": "panelwise_local_labels_overlap_union",
                "labels": str(labels.resolve()),
                "summary": str(summary.resolve()),
                "extensions": str(extensions.resolve()),
            }
        ),
        encoding="utf-8",
    )

    destination = tmp_path / "anchored" / "boundary_components"
    copied_labels = _copy_boundary_component_artifacts(
        source_dir=source,
        output_dir=destination,
        example_id="46",
    )

    assert copied_labels == destination.resolve() / labels.name
    assert copied_labels.read_bytes() == b"labels"
    copied_manifest = json.loads(
        (destination / "46_panelized_boundary_component_manifest.json").read_text(encoding="utf-8")
    )
    assert copied_manifest["labels"] == str((destination / labels.name).resolve())
    assert copied_manifest["summary"] == str((destination / summary.name).resolve())
    assert copied_manifest["extensions"] == str((destination / extensions.name).resolve())


def test_infer_boundary_artifacts_from_reused_base_variant(tmp_path: Path) -> None:
    experiment = tmp_path / "panelized_base__realized"
    source_variant = experiment / "states" / "S000_base" / "variant"
    boundary_components = experiment / "boundary_components"
    source_variant.mkdir(parents=True)
    boundary_components.mkdir()

    inferred = collapse_experiment_module._infer_boundary_component_artifact_source_dir(
        source_variant
    )

    assert inferred == boundary_components.resolve()


def test_full_raster_base_source_reuses_complete_base_products_and_boundaries(
    tmp_path: Path,
) -> None:
    source_experiment = tmp_path / "source_experiment"
    source_state = source_experiment / "states" / "S000_base"
    source_variant = source_state / "variant"
    for directory in (
        source_variant / "directed",
        source_variant / "mask",
        source_variant / "widths",
        source_variant / "summary",
        source_state / "hydraulic_context",
        source_experiment / "boundary_components",
    ):
        directory.mkdir(parents=True, exist_ok=True)

    (source_variant / "directed" / "22__S000_base_directed_links.gpkg").write_bytes(b"links")
    (source_variant / "directed" / "22__S000_base_directed_nodes.gpkg").write_bytes(b"nodes")
    (source_variant / "mask" / "22__S000_base_collapsed.tif").write_bytes(b"mask")
    (source_variant / "widths" / "links_with_width_families.gpkg").write_bytes(b"widths")
    (source_variant / "summary" / "variant_manifest.json").write_text(
        json.dumps(
            {
                "output_paths": {
                    "output_dir": str(source_variant.resolve()),
                    "directed_links": str(
                        (
                            source_variant
                            / "directed"
                            / "22__S000_base_directed_links.gpkg"
                        ).resolve()
                    ),
                }
            }
        ),
        encoding="utf-8",
    )
    labels = (
        source_experiment
        / "boundary_components"
        / "22_full_extent_land_water_labels_with_snapped_buffered_border_cut.tif"
    )
    labels.write_bytes(b"labels")

    write_unit_workflow_outputs(
        source_state / "hierarchy",
        _workflow_outputs(unit_ids=[1]),
    )
    (source_state / "hydraulic_context" / "link_hydraulic_context.csv").write_text(
        "id_link,slope_used\n1,0.001\n", encoding="utf-8"
    )
    (source_state / "hydraulic_context" / "node_hydraulic_context.csv").write_text(
        "id_node,rapid_wse_for_slope\n1,10\n2,9\n", encoding="utf-8"
    )
    (source_state / "hydraulic_context" / "hydraulic_context_manifest.json").write_text(
        json.dumps(
            {
                "state_id": "S000_base",
                "output_paths": {
                    "link_hydraulic_context": str(
                        (
                            source_state
                            / "hydraulic_context"
                            / "link_hydraulic_context.csv"
                        ).resolve()
                    )
                },
            }
        ),
        encoding="utf-8",
    )

    cleaned_mask = tmp_path / "22_cleaned.tif"
    reviewed_links = tmp_path / "22_links.gpkg"
    reviewed_nodes = tmp_path / "22_nodes.gpkg"
    for path in (cleaned_mask, reviewed_links, reviewed_nodes):
        path.write_bytes(b"input")
    (source_experiment / "experiment_manifest.json").write_text(
        json.dumps(
            {
                "mode": "independent-units",
                "example_id": "22",
                "exit_sides": "NS",
                "code_version": "source-test-revision",
                "n_complete_states": 1,
                "n_failed_states": 0,
                "stop_reason": "base_state_only",
                "root_paths": {
                    "cleaned_mask": str(cleaned_mask.resolve()),
                    "reviewed_links": str(reviewed_links.resolve()),
                    "reviewed_nodes": str(reviewed_nodes.resolve()),
                },
                "options": {"hierarchy": {}, "variant": {}},
            }
        ),
        encoding="utf-8",
    )

    def fail_workflow_runner(*args: Any, **kwargs: Any) -> UnitWorkflowOutputs:
        raise AssertionError("Complete base hierarchy should have been reused.")

    def fail_hydraulic_runner(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("Complete base hydraulic context should have been reused.")

    output_dir = tmp_path / "reused_output"
    results = run_collapse_experiment(
        "independent-units",
        cleaned_mask_path=cleaned_mask,
        reviewed_links_path=reviewed_links,
        reviewed_nodes_path=reviewed_nodes,
        exit_sides="NS",
        output_dir=output_dir,
        collapse_corridor_mode="realized",
        base_only=True,
        base_state_variant_source_dir=source_variant,
        unit_workflow_runner=fail_workflow_runner,
        state_hydraulic_context_runner=fail_hydraulic_runner,
    )

    copied_labels = (
        output_dir
        / "boundary_components"
        / "22_full_extent_land_water_labels_with_snapped_buffered_border_cut.tif"
    )
    assert copied_labels.read_bytes() == b"labels"
    assert (output_dir / "states" / "S000_base" / "hierarchy" / "unit_summary.csv").exists()
    assert (
        output_dir
        / "states"
        / "S000_base"
        / "hydraulic_context"
        / "link_hydraulic_context.csv"
    ).exists()
    assert len(results.state_registry) == 1
    reuse = results.manifest["base_state_reuse"]
    assert reuse["variant_reused"] is True
    assert reuse["hierarchy_reused"] is True
    assert reuse["hydraulic_context_reused"] is True
    assert reuse["boundary_components_reused"] is True
    assert reuse["compatibility_validated"] is True
    reuse_audit = json.loads(Path(reuse["audit_manifest"]).read_text(encoding="utf-8"))
    assert reuse_audit["validation_passed"] is True
    assert reuse_audit["non_json_content_verified"] is True
    assert reuse_audit["non_json_content_mismatches"] == []
    assert reuse_audit["source_code_version"] == "source-test-revision"

    copied_state = output_dir / "states" / "S000_base"
    copied_variant_manifest = json.loads(
        (copied_state / "variant" / "summary" / "variant_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert copied_variant_manifest["output_paths"]["output_dir"] == str(
        (copied_state / "variant").resolve()
    )
    copied_hydraulic_manifest = json.loads(
        (copied_state / "hydraulic_context" / "hydraulic_context_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert copied_hydraulic_manifest["output_paths"]["link_hydraulic_context"] == str(
        (copied_state / "hydraulic_context" / "link_hydraulic_context.csv").resolve()
    )


def test_incomplete_base_source_is_rejected_before_destination_reset(
    tmp_path: Path,
) -> None:
    source_experiment = tmp_path / "source"
    source_variant = source_experiment / "states" / "S000_base" / "variant"
    source_variant.mkdir(parents=True)
    (source_experiment / "boundary_components").mkdir()
    output_dir = tmp_path / "destination"
    output_dir.mkdir()
    marker = output_dir / "existing-output-must-survive.txt"
    marker.write_text("preserve", encoding="utf-8")
    cleaned_mask = tmp_path / "22_cleaned.tif"
    reviewed_links = tmp_path / "22_links.gpkg"
    reviewed_nodes = tmp_path / "22_nodes.gpkg"
    for path in (cleaned_mask, reviewed_links, reviewed_nodes):
        path.write_bytes(b"input")
    (source_experiment / "experiment_manifest.json").write_text(
        json.dumps(
            {
                "mode": "independent-units",
                "example_id": "22",
                "exit_sides": "NS",
                "n_complete_states": 1,
                "n_failed_states": 0,
                "root_paths": {
                    "cleaned_mask": str(cleaned_mask.resolve()),
                    "reviewed_links": str(reviewed_links.resolve()),
                    "reviewed_nodes": str(reviewed_nodes.resolve()),
                },
                "options": {"hierarchy": {}, "variant": {}},
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(FileNotFoundError, match="source is incomplete"):
        run_collapse_experiment(
            "compound-bubbles",
            cleaned_mask_path=cleaned_mask,
            reviewed_links_path=reviewed_links,
            reviewed_nodes_path=reviewed_nodes,
            exit_sides="NS",
            output_dir=output_dir,
            collapse_corridor_mode="anchored",
            base_only=True,
            base_state_variant_source_dir=source_variant,
        )

    assert marker.read_text(encoding="utf-8") == "preserve"


def test_run_copies_boundary_artifacts_before_base_materialization(tmp_path: Path) -> None:
    source = tmp_path / "realized" / "boundary_components"
    source.mkdir(parents=True)
    source_labels = source / "46_full_extent_land_water_labels_with_snapped_buffered_border_cut.tif"
    source_labels.write_bytes(b"labels")
    (source / "46_full_extent_land_component_summary_with_snapped_buffered_border_cut.csv").write_text(
        "land_component_id,raster_value\n1,2\n",
        encoding="utf-8",
    )
    output_dir = tmp_path / "anchored"
    workflow_runner = _make_workflow_runner([_workflow_outputs([])])
    base_materializer = _make_base_state_materializer()

    run_collapse_experiment(
        "independent-units",
        cleaned_mask_path=tmp_path / "46_cleaned.tif",
        reviewed_links_path=tmp_path / "46_links.gpkg",
        reviewed_nodes_path=tmp_path / "46_nodes.gpkg",
        exit_sides="NS",
        output_dir=output_dir,
        collapse_corridor_mode="anchored",
        base_only=True,
        boundary_component_artifact_source_dir=source,
        unit_workflow_runner=workflow_runner,
        unit_workflow_writer=write_unit_workflow_outputs,
        base_state_variant_materializer=base_materializer,
        state_hydraulic_context_runner=_make_state_hydraulic_context_runner(),
    )

    copied_labels = (
        output_dir
        / "boundary_components"
        / "46_full_extent_land_water_labels_with_snapped_buffered_border_cut.tif"
    )
    assert copied_labels.read_bytes() == b"labels"
    assert Path(base_materializer.calls[0]["boundary_component_labels_path"]) == copied_labels.resolve()


def test_main_passes_base_only_into_run_collapse_experiment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, Any]] = []

    def fake_run_collapse_experiment(mode: str, **kwargs: Any) -> SimpleNamespace:
        calls.append({"mode": mode, **dict(kwargs)})
        output_dir = Path(kwargs["output_dir"])
        return SimpleNamespace(
            output_dir=output_dir,
            state_registry=pd.DataFrame([{"state_id": "S000_base"}]),
            transition_registry=pd.DataFrame(),
            equivalent_transition_registry=pd.DataFrame(),
            manifest={"mode": mode, "stop_reason": "base_only_requested", "terminal_state_id": "S000_base"},
        )

    monkeypatch.setattr(collapse_experiment_module, "run_collapse_experiment", fake_run_collapse_experiment)

    exit_code = main(
        [
            "independent-units",
            "--cleaned-mask",
            str(tmp_path / "base_cleaned.tif"),
            "--reviewed-links",
            str(tmp_path / "base_links.gpkg"),
            "--reviewed-nodes",
            str(tmp_path / "base_nodes.gpkg"),
            "--exit-sides",
            "NS",
            "--output-dir",
            str(tmp_path / "demo_output"),
            "--base-only",
        ]
    )

    assert exit_code == 0
    assert len(calls) == 1
    assert calls[0]["base_only"] is True


def test_main_base_only_anchored_spawns_realized_and_anchored_base_runs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, Any]] = []

    def fake_run_collapse_experiment(mode: str, **kwargs: Any) -> SimpleNamespace:
        calls.append({"mode": mode, **dict(kwargs)})
        output_dir = Path(kwargs["output_dir"])
        return SimpleNamespace(
            output_dir=output_dir,
            state_registry=pd.DataFrame([{"state_id": "S000_base"}]),
            transition_registry=pd.DataFrame(),
            equivalent_transition_registry=pd.DataFrame(),
            manifest={"mode": mode, "stop_reason": "base_only_requested", "terminal_state_id": "S000_base"},
        )

    monkeypatch.setattr(collapse_experiment_module, "run_collapse_experiment", fake_run_collapse_experiment)

    exit_code = main(
        [
            "independent-units",
            "--cleaned-mask",
            str(tmp_path / "base_cleaned.tif"),
            "--reviewed-links",
            str(tmp_path / "base_links.gpkg"),
            "--reviewed-nodes",
            str(tmp_path / "base_nodes.gpkg"),
            "--exit-sides",
            "NS",
            "--output-dir",
            str(tmp_path / "demo_output"),
            "--collapse-corridor-mode",
            "anchored",
            "--base-only",
        ]
    )

    assert exit_code == 0
    assert len(calls) == 2
    assert [call["collapse_corridor_mode"] for call in calls] == ["realized", "anchored"]
    assert [Path(call["output_dir"]).name for call in calls] == ["demo_output__realized", "demo_output__anchored"]
    assert all(call["base_only"] is True for call in calls)


def test_main_anchored_runs_realized_and_anchored_with_suffixed_outputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, Any]] = []

    def fake_run_collapse_experiment(mode: str, **kwargs: Any) -> SimpleNamespace:
        calls.append({"mode": mode, **dict(kwargs)})
        output_dir = Path(kwargs["output_dir"])
        return SimpleNamespace(
            output_dir=output_dir,
            state_registry=pd.DataFrame([{"state_id": "S000_base"}]),
            transition_registry=pd.DataFrame(),
            equivalent_transition_registry=pd.DataFrame(),
            manifest={"mode": mode, "stop_reason": "test_complete", "terminal_state_id": "S000_base"},
        )

    monkeypatch.setattr(collapse_experiment_module, "run_collapse_experiment", fake_run_collapse_experiment)

    exit_code = main(
        [
            "independent-units",
            "--cleaned-mask",
            str(tmp_path / "base_cleaned.tif"),
            "--reviewed-links",
            str(tmp_path / "base_links.gpkg"),
            "--reviewed-nodes",
            str(tmp_path / "base_nodes.gpkg"),
            "--exit-sides",
            "NS",
            "--experiment-id",
            "demo",
            "--output-dir",
            str(tmp_path / "demo_output"),
            "--collapse-corridor-mode",
            "anchored",
            "--slope-profile-mode",
            "reference_section",
            "--reference-section-summary-path",
            str(tmp_path / "reference_section" / "reference_section_summary.json"),
            "--min-slope",
            "1e-5",
            "--max-slope-for-k",
            "0.004",
            "--section-slope-ratio-min",
            "0.5",
            "--section-slope-ratio-max",
            "4.0",
            "--disable-section-slope-fallback",
            "--disable-compound-bubble-slope-qc",
            "--disable-compound-bubble-slope-repair",
        ]
    )

    assert exit_code == 0
    assert [call["collapse_corridor_mode"] for call in calls] == ["realized", "anchored"]
    assert [call["experiment_id"] for call in calls] == ["demo__realized", "demo__anchored"]
    assert [Path(call["output_dir"]).name for call in calls] == ["demo_output__realized", "demo_output__anchored"]
    assert "variant_runner" not in calls[0]
    assert callable(calls[1]["variant_runner"])
    for call in calls:
        config = call["state_hydraulic_context_kwargs"]["config"]
        assert config.slope_profile_mode == "reference_section"
        assert config.reference_section_summary_path == str(
            tmp_path / "reference_section" / "reference_section_summary.json"
        )
        assert config.min_slope == pytest.approx(1e-5)
        assert config.max_slope_for_k == pytest.approx(0.004)
        assert config.section_slope_ratio_min == pytest.approx(0.5)
        assert config.section_slope_ratio_max == pytest.approx(4.0)
        assert config.use_section_slope_fallback is False
        assert config.use_compound_bubble_slope_qc is False
        assert config.use_compound_bubble_slope_repair is False


def test_main_passes_equivalent_length_field_into_equivalent_resistance_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, Any]] = []

    def fake_run_collapse_experiment(mode: str, **kwargs: Any) -> SimpleNamespace:
        calls.append({"mode": mode, **dict(kwargs)})
        output_dir = Path(kwargs["output_dir"])
        return SimpleNamespace(
            output_dir=output_dir,
            state_registry=pd.DataFrame([{"state_id": "S000_base"}]),
            transition_registry=pd.DataFrame(),
            equivalent_transition_registry=pd.DataFrame(),
            manifest={"mode": mode, "stop_reason": "test_complete", "terminal_state_id": "S000_base"},
        )

    monkeypatch.setattr(collapse_experiment_module, "run_collapse_experiment", fake_run_collapse_experiment)

    exit_code = main(
        [
            "independent-units",
            "--cleaned-mask",
            str(tmp_path / "base_cleaned.tif"),
            "--reviewed-links",
            str(tmp_path / "base_links.gpkg"),
            "--reviewed-nodes",
            str(tmp_path / "base_nodes.gpkg"),
            "--exit-sides",
            "NS",
            "--experiment-id",
            "demo",
            "--output-dir",
            str(tmp_path / "demo_output"),
            "--compute-equivalent-resistance",
            "--equivalent-length-field",
            "len",
        ]
    )

    assert exit_code == 0
    assert len(calls) == 1
    config = calls[0]["equivalent_resistance_config"]
    assert config is not None
    assert config.length_field_candidates == ("len", "len_adj", "link_length_m")


def test_main_passes_equivalent_forcing_resample_seconds(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, Any]] = []

    def fake_run_collapse_experiment(mode: str, **kwargs: Any) -> SimpleNamespace:
        calls.append({"mode": mode, **dict(kwargs)})
        output_dir = Path(kwargs["output_dir"])
        return SimpleNamespace(
            output_dir=output_dir,
            state_registry=pd.DataFrame([{"state_id": "S000_base"}]),
            transition_registry=pd.DataFrame(),
            equivalent_transition_registry=pd.DataFrame(),
            manifest={"mode": mode, "stop_reason": "test_complete", "terminal_state_id": "S000_base"},
        )

    monkeypatch.setattr(collapse_experiment_module, "run_collapse_experiment", fake_run_collapse_experiment)

    exit_code = main(
        [
            "independent-units",
            "--cleaned-mask",
            str(tmp_path / "base_cleaned.tif"),
            "--reviewed-links",
            str(tmp_path / "base_links.gpkg"),
            "--reviewed-nodes",
            str(tmp_path / "base_nodes.gpkg"),
            "--exit-sides",
            "NS",
            "--experiment-id",
            "demo",
            "--output-dir",
            str(tmp_path / "demo_output"),
            "--compute-equivalent-resistance",
            "--equivalent-kb-mode",
            "based_reference_section",
            "--equivalent-forcing-resample-seconds",
            "980",
        ]
    )

    assert exit_code == 0
    assert len(calls) == 1
    config = calls[0]["equivalent_forcing_config"]
    assert config is not None
    assert config.resample_seconds == 980
    assert config.resample_minutes is None
    assert calls[0]["equivalent_kb_width_sample_field"] == "wid_adj_routing"
    assert calls[0]["equivalent_kb_width_percentile"] is None


def test_independent_units_writes_equivalent_transition_comparisons(tmp_path: Path) -> None:
    workflow_runner = _make_workflow_runner(
        [
            _workflow_outputs(
                [5],
                bubble_equivalent_rows=[
                    {
                        "compound_bubble_id": 1,
                        "bubble_root_unit_id": 5,
                        "upstream_boundary_node": 1,
                        "downstream_boundary_node": 2,
                        "n_member_units": 1,
                        "member_unit_ids": "5",
                        "n_nodes": 2,
                        "n_edges": 2,
                        "solver_success": True,
                        "solver_method": "boundary_only",
                        "solver_message": "",
                        "solver_nfev": 0,
                        "K_eq": 4.0,
                        "G_eq": 0.25,
                        "Q_unit": 0.5,
                        "residual_norm": 0.0,
                        "residual_relative": 0.0,
                        "resistance_law": "manning",
                        "geometry_method": "wide",
                        "kb_mode": "fixed",
                        "kb_value": 20.0,
                        "kb_source_method": "fixed_scalar",
                    }
                ],
            ),
            _workflow_outputs(
                [],
                bubble_equivalent_rows=[
                    {
                        "compound_bubble_id": 9,
                        "bubble_root_unit_id": 50,
                        "upstream_boundary_node": 1,
                        "downstream_boundary_node": 2,
                        "n_member_units": 1,
                        "member_unit_ids": "50",
                        "n_nodes": 2,
                        "n_edges": 1,
                        "solver_success": True,
                        "solver_method": "boundary_only",
                        "solver_message": "",
                        "solver_nfev": 0,
                        "K_eq": 8.0,
                        "G_eq": 0.125,
                        "Q_unit": 0.3535533906,
                        "residual_norm": 0.0,
                        "residual_relative": 0.0,
                        "resistance_law": "manning",
                        "geometry_method": "wide",
                        "kb_mode": "fixed",
                        "kb_value": 20.0,
                        "kb_source_method": "fixed_scalar",
                    }
                ],
            ),
        ]
    )
    variant_runner = _make_variant_runner()
    base_state_materializer = _make_base_state_materializer()
    state_hydraulic_context_runner = _make_state_hydraulic_context_runner()

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
        base_state_variant_materializer=base_state_materializer,
        state_hydraulic_context_runner=state_hydraulic_context_runner,
    )

    assert len(results.equivalent_transition_registry) == 1
    row = results.equivalent_transition_registry.iloc[0]
    assert row["match_method"] == "upstream_boundary_node__downstream_boundary_node"
    assert row["Lambda_K"] == pytest.approx(2.0)
    assert row["percent_change_K"] == pytest.approx(100.0)

    transition = results.transition_registry.iloc[0]
    assert int(transition["n_equivalent_bubbles_matched"]) == 1
    assert Path(transition["equivalent_transition_comparison_csv"]).exists()
    assert Path(transition["equivalent_transition_summary_json"]).exists()

    derived_row = results.state_registry.loc[results.state_registry["state_id"] == "S001_unit_5"].iloc[0]
    assert int(derived_row["n_equivalent_bubbles_matched_to_parent"]) == 1
    assert Path(derived_row["equivalent_transition_comparison_csv"]).exists()
    assert (tmp_path / "experiment" / "equivalent_transition_registry.csv").exists()


def test_equivalent_transition_matching_uses_parent_mapped_child_boundary_nodes(tmp_path: Path) -> None:
    parent_workflow = _workflow_outputs(
        [13, 15],
        bubble_equivalent_rows=[
            {
                "compound_bubble_id": 3,
                "bubble_root_unit_id": 13,
                "upstream_boundary_node": 26,
                "downstream_boundary_node": 25,
                "n_member_units": 1,
                "member_unit_ids": "13",
                "n_nodes": 2,
                "n_edges": 2,
                "solver_success": True,
                "solver_method": "boundary_only",
                "solver_message": "",
                "solver_nfev": 0,
                "K_eq": 1.0,
                "G_eq": 1.0,
                "Q_unit": 1.0,
                "residual_norm": 0.0,
                "residual_relative": 0.0,
                "resistance_law": "manning",
                "geometry_method": "wide",
                "kb_mode": "fixed",
                "kb_value": 20.0,
                "kb_source_method": "fixed_scalar",
            },
            {
                "compound_bubble_id": 1,
                "bubble_root_unit_id": 15,
                "upstream_boundary_node": 30,
                "downstream_boundary_node": 29,
                "n_member_units": 1,
                "member_unit_ids": "15",
                "n_nodes": 2,
                "n_edges": 2,
                "solver_success": True,
                "solver_method": "boundary_only",
                "solver_message": "",
                "solver_nfev": 0,
                "K_eq": 10.0,
                "G_eq": 0.1,
                "Q_unit": 0.3162277660,
                "residual_norm": 0.0,
                "residual_relative": 0.0,
                "resistance_law": "manning",
                "geometry_method": "wide",
                "kb_mode": "fixed",
                "kb_value": 20.0,
                "kb_source_method": "fixed_scalar",
            },
        ],
    )
    child_workflow = _workflow_outputs(
        [13],
        bubble_equivalent_rows=[
            {
                "compound_bubble_id": 1,
                "bubble_root_unit_id": 13,
                "upstream_boundary_node": 26,
                "downstream_boundary_node": 25,
                "n_member_units": 1,
                "member_unit_ids": "13",
                "n_nodes": 2,
                "n_edges": 2,
                "solver_success": True,
                "solver_method": "boundary_only",
                "solver_message": "",
                "solver_nfev": 0,
                "K_eq": 10.0,
                "G_eq": 0.1,
                "Q_unit": 0.3162277660,
                "residual_norm": 0.0,
                "residual_relative": 0.0,
                "resistance_law": "manning",
                "geometry_method": "wide",
                "kb_mode": "fixed",
                "kb_value": 20.0,
                "kb_source_method": "fixed_scalar",
            }
        ],
    )

    parent_state_dir = tmp_path / "parent_state"
    child_state_dir = tmp_path / "child_state"
    child_variant_dir = child_state_dir / "variant"
    (child_variant_dir / "matching").mkdir(parents=True)
    pd.DataFrame(
        [
            {"child_id_node": 25, "matched_parent_node_id": 29, "match_within_tolerance": True},
            {"child_id_node": 26, "matched_parent_node_id": 30, "match_within_tolerance": True},
        ]
    ).to_csv(child_variant_dir / "matching" / "node_match.csv", index=False)

    parent_context = ExperimentStateContext(
        state_id="S000_base",
        parent_state_id=None,
        depth=0,
        state_dir=parent_state_dir,
        hierarchy_output_dir=parent_state_dir / "hierarchy",
        cleaned_mask_path=parent_state_dir / "mask.tif",
        reviewed_links_path=parent_state_dir / "links.gpkg",
        reviewed_nodes_path=parent_state_dir / "nodes.gpkg",
        workflow=parent_workflow,
        variant_output_dir=parent_state_dir / "variant",
    )
    child_context = ExperimentStateContext(
        state_id="S001_unit_13",
        parent_state_id="S000_base",
        depth=1,
        state_dir=child_state_dir,
        hierarchy_output_dir=child_state_dir / "hierarchy",
        cleaned_mask_path=child_state_dir / "mask.tif",
        reviewed_links_path=child_state_dir / "links.gpkg",
        reviewed_nodes_path=child_state_dir / "nodes.gpkg",
        workflow=child_workflow,
        variant_output_dir=child_variant_dir,
    )

    comparison = _write_equivalent_transition_outputs(
        transition_id="T001",
        parent_context=parent_context,
        child_context=child_context,
    )

    assert len(comparison) == 1
    row = comparison.iloc[0]
    assert row["match_method"] == "matched_parent_upstream_boundary_node__matched_parent_downstream_boundary_node"
    assert row["parent_upstream_boundary_node"] == 30
    assert row["parent_downstream_boundary_node"] == 29
    assert row["child_upstream_boundary_node"] == 26
    assert row["child_downstream_boundary_node"] == 25
    assert row["K_before"] == pytest.approx(10.0)
    assert row["K_after"] == pytest.approx(10.0)
    assert row["Lambda_K"] == pytest.approx(1.0)


def test_selected_hydraulic_domain_replaces_primary_comparison_and_keeps_diagnostic(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parent_state_dir = tmp_path / "parent"
    child_state_dir = tmp_path / "child"
    parent = ExperimentStateContext(
        state_id="S000_base",
        parent_state_id=None,
        depth=0,
        state_dir=parent_state_dir,
        hierarchy_output_dir=parent_state_dir / "hierarchy",
        cleaned_mask_path=parent_state_dir / "mask.tif",
        reviewed_links_path=parent_state_dir / "links.gpkg",
        reviewed_nodes_path=parent_state_dir / "nodes.gpkg",
        workflow=_workflow_outputs([]),
    )
    child = ExperimentStateContext(
        state_id="S001_unit_7",
        parent_state_id="S000_base",
        depth=1,
        state_dir=child_state_dir,
        hierarchy_output_dir=child_state_dir / "hierarchy",
        cleaned_mask_path=child_state_dir / "mask.tif",
        reviewed_links_path=child_state_dir / "links.gpkg",
        reviewed_nodes_path=child_state_dir / "nodes.gpkg",
        workflow=_workflow_outputs([]),
    )
    selected = pd.DataFrame.from_records(
        [
            {
                "transition_id": "T001",
                "parent_state_id": "S000_base",
                "child_state_id": "S001_unit_7",
                "match_method": "smallest_closed_hydraulic_domain__matched_parent_anchors",
                "comparison_scope": "selected_hydraulic_domain",
                "hydraulic_domain_type": "containing_compound_context",
                "K_before": 4.0,
                "K_after": 2.0,
                "Lambda_K": 0.5,
                "Lambda_G": 2.0,
                "percent_change_K": -50.0,
                "percent_change_G": 100.0,
            }
        ]
    )
    audit = pd.DataFrame(
        [{"transition_id": "T001", "domain_role": "parent", "id_link": 1}]
    )
    monkeypatch.setattr(
        collapse_experiment_module,
        "_selected_hydraulic_domain_comparison",
        lambda **kwargs: (selected, audit),
    )

    comparison = _write_equivalent_transition_outputs(
        transition_id="T001",
        parent_context=parent,
        child_context=child,
        equivalent_resistance_config=BubbleEquivalentResistanceConfig(),
    )

    assert comparison.iloc[0]["comparison_scope"] == "selected_hydraulic_domain"
    assert comparison.iloc[0]["Lambda_K"] == pytest.approx(0.5)
    output_dir = child_state_dir / "equivalent_resistance"
    assert (output_dir / "redetected_bubble_matches.csv").exists()
    assert (output_dir / "selected_hydraulic_domain_links.csv").exists()
    summary = json.loads(
        (output_dir / "bubble_equivalent_state_comparison_summary.json").read_text()
    )
    assert summary["comparison_method"] == "smallest_closed_hydraulic_domain"
    assert summary["n_selected_hydraulic_domain_comparisons"] == 1

    failed_state_dir = tmp_path / "failed_child"
    child.state_dir = failed_state_dir
    monkeypatch.setattr(
        collapse_experiment_module,
        "_selected_hydraulic_domain_comparison",
        lambda **kwargs: (_ for _ in ()).throw(
            ValueError("No unambiguous child anchor")
        ),
    )
    failed = _write_equivalent_transition_outputs(
        transition_id="T002",
        parent_context=parent,
        child_context=child,
        equivalent_resistance_config=BubbleEquivalentResistanceConfig(),
    )
    assert failed.iloc[0]["selected_domain_status"] == "failed"
    assert "No unambiguous child anchor" in failed.iloc[0]["selected_domain_error"]
    failed_summary = json.loads(
        (
            failed_state_dir
            / "equivalent_resistance"
            / "bubble_equivalent_state_comparison_summary.json"
        ).read_text()
    )
    assert failed_summary["selected_hydraulic_domain_status"] == "failed"


def test_independent_units_records_failed_variant_and_continues(tmp_path: Path) -> None:
    workflow_runner = _make_workflow_runner(
        [
            _workflow_outputs([5, 7, 2]),
            _workflow_outputs([1]),
            _workflow_outputs([1]),
        ]
    )
    base_state_materializer = _make_base_state_materializer()
    state_hydraulic_context_runner = _make_state_hydraulic_context_runner()
    calls: list[dict[str, Any]] = []

    def variant_runner(**kwargs: Any) -> NetworkVariantOutputs:
        calls.append(dict(kwargs))
        output_dir = Path(kwargs["output_dir"])
        variant_id = str(kwargs["variant_id"])
        outputs = _make_variant_outputs(output_dir, variant_id, direction_valid=variant_id != "S002_unit_7")
        if variant_id == "S002_unit_7":
            raise VariantDirectionValidationError(outputs, ["invalid flow graph"])
        return outputs

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
        base_state_variant_materializer=base_state_materializer,
        state_hydraulic_context_runner=state_hydraulic_context_runner,
    )

    assert [call["unit_ids"] for call in calls] == [[5], [7], [2]]
    statuses = results.state_registry.set_index("state_id")["status"].to_dict()
    assert statuses["S001_unit_5"] == "complete"
    assert statuses["S002_unit_7"] == "failed"
    assert statuses["S003_unit_2"] == "complete"
    failed_row = results.state_registry.loc[results.state_registry["state_id"] == "S002_unit_7"].iloc[0]
    assert Path(failed_row["direction_validation_report"]).exists()
    assert Path(failed_row["failure_artifact_path"]).exists()
    assert results.manifest["n_failed_states"] == 1


def test_independent_units_rerun_cleans_existing_state_directories(tmp_path: Path) -> None:
    def make_runner() -> Any:
        return _make_workflow_runner([_workflow_outputs([5]), _workflow_outputs([])])

    variant_runner = _make_variant_runner()
    base_state_materializer = _make_base_state_materializer()
    state_hydraulic_context_runner = _make_state_hydraulic_context_runner()
    output_dir = tmp_path / "experiment"

    first = run_collapse_experiment(
        "independent-units",
        cleaned_mask_path=tmp_path / "base_cleaned.tif",
        reviewed_links_path=tmp_path / "base_links.gpkg",
        reviewed_nodes_path=tmp_path / "base_nodes.gpkg",
        exit_sides="NS",
        output_dir=output_dir,
        unit_workflow_runner=make_runner(),
        unit_workflow_writer=write_unit_workflow_outputs,
        variant_runner=variant_runner,
        base_state_variant_materializer=base_state_materializer,
        state_hydraulic_context_runner=state_hydraulic_context_runner,
    )

    base_stale = output_dir / "states" / "S000_base" / "stale.txt"
    child_stale = output_dir / "states" / "S001_unit_5" / "stale.txt"
    stale_state_dir = output_dir / "states" / "S999_stale"
    stale_registry = output_dir / "rapid_run_registry.csv"
    stale_transition = output_dir / "equivalent_transition_registry.csv"
    base_stale.write_text("old")
    child_stale.write_text("old")
    stale_state_dir.mkdir(parents=True)
    (stale_state_dir / "stale.txt").write_text("old")
    stale_registry.write_text("old")
    stale_transition.write_text("old")

    second = run_collapse_experiment(
        "independent-units",
        cleaned_mask_path=tmp_path / "base_cleaned.tif",
        reviewed_links_path=tmp_path / "base_links.gpkg",
        reviewed_nodes_path=tmp_path / "base_nodes.gpkg",
        exit_sides="NS",
        output_dir=output_dir,
        unit_workflow_runner=make_runner(),
        unit_workflow_writer=write_unit_workflow_outputs,
        variant_runner=variant_runner,
        base_state_variant_materializer=base_state_materializer,
        state_hydraulic_context_runner=state_hydraulic_context_runner,
    )

    assert first.manifest["n_complete_states"] == 2
    assert second.manifest["n_complete_states"] == 2
    assert not base_stale.exists()
    assert not child_stale.exists()
    assert not stale_state_dir.exists()
    assert not stale_registry.exists()
    assert stale_transition.exists()
    assert stale_transition.read_text() != "old"


def test_independent_units_records_exception_type_for_child_build_failures(tmp_path: Path) -> None:
    workflow_runner = _make_workflow_runner([_workflow_outputs([5])])
    base_state_materializer = _make_base_state_materializer()
    state_hydraulic_context_runner = _make_state_hydraulic_context_runner()

    def variant_runner(**kwargs: Any) -> NetworkVariantOutputs:
        raise KeyError(6)

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
        base_state_variant_materializer=base_state_materializer,
        state_hydraulic_context_runner=state_hydraulic_context_runner,
    )

    failed_row = results.state_registry.loc[results.state_registry["state_id"] == "S001_unit_5"].iloc[0]
    assert failed_row["status"] == "failed"
    assert failed_row["error"] == "KeyError: 6"
    assert "KeyError" in failed_row["error_traceback"]


def test_build_unit_link_membership_derives_link_to_unit_mapping() -> None:
    path_metrics = pd.DataFrame(
        {
            "unit_id": [1, 1, 2],
            "path_id": [1, 2, 1],
            "id_links": ["2,6", "3,5,8", "1,3"],
        }
    )
    unit_metrics = pd.DataFrame(
        {
            "unit_id": [1, 2],
            "root_unit_id": [1, 2],
            "collapse_level": [0, 1],
            "compound_unit_id": [pd.NA, 20],
            "compound_bubble_id": [10, 20],
            "class": ["compound_or_nested_complex", "simple_bifurcation_confluence_pair"],
            "unit_topodynamic_class": ["intermediate_unit", "balanced_simple_split"],
        }
    )

    membership, link_summary = _build_unit_link_membership(
        path_metrics=path_metrics,
        unit_metrics=unit_metrics,
    )

    row = membership.loc[(membership["unit_id"] == 1) & (membership["id_link"] == 3)].iloc[0]
    assert row["path_ids"] == "2"
    assert int(row["n_paths_through_link"]) == 1
    assert row["unit_topodynamic_class"] == "intermediate_unit"

    link_row = link_summary.loc[link_summary["id_link"] == 3].iloc[0]
    assert link_row["unit_ids"] == "1,2"
    assert int(link_row["n_units"]) == 2
    assert link_row["collapse_levels"] == "0,1"


def test_normalize_numeric_id_columns_coerces_base_link_ids_from_object() -> None:
    frame = pd.DataFrame(
        {
            "id_link": ["48", "52"],
            "id_us_node": ["31", "34"],
            "id_ds_node": ["34", "36"],
            "label": ["a", "b"],
        }
    )

    normalized = _normalize_numeric_id_columns(frame)

    assert str(normalized["id_link"].dtype) == "int64"
    assert str(normalized["id_us_node"].dtype) == "int64"
    assert str(normalized["id_ds_node"].dtype) == "int64"
    assert normalized.loc[0, "id_us_node"] == 31
    assert normalized.loc[1, "id_ds_node"] == 36


def test_local_patch_child_runner_returns_local_result_without_full_fallback(tmp_path: Path) -> None:
    sentinel = object()
    calls = {"local": 0, "full": 0}

    def local_runner(**kwargs: Any):
        calls["local"] += 1
        return sentinel

    def full_runner(**kwargs: Any):
        calls["full"] += 1
        raise AssertionError("full-raster fallback must not run for a successful local patch")

    runner = collapse_experiment_module._make_local_patch_child_runner(
        fallback_variant_runner=full_runner,
        local_variant_runner=local_runner,
    )

    assert runner(output_dir=tmp_path / "variant") is sentinel
    assert calls == {"local": 1, "full": 0}


def test_local_patch_child_runner_audits_ambiguous_merge_and_falls_back(tmp_path: Path) -> None:
    output_dir = tmp_path / "variant"
    source_audit = output_dir / "local_patch_attempt" / "merge" / "merge_audit.json"
    source_audit.parent.mkdir(parents=True)
    source_audit.write_text("{}\n", encoding="utf-8")

    def local_runner(**kwargs: Any):
        raise collapse_experiment_module.LocalPatchFallbackRequested(
            stage="merge",
            code="ambiguous_interface",
            reason="Two local terminals match the upstream cut.",
            audit_path=source_audit,
        )

    def full_runner(**kwargs: Any):
        summary_dir = output_dir / "summary"
        summary_dir.mkdir(parents=True, exist_ok=True)
        return SimpleNamespace(output_dir=output_dir, manifest={"variant_id": "S001_unit_7"})

    runner = collapse_experiment_module._make_local_patch_child_runner(
        fallback_variant_runner=full_runner,
        local_variant_runner=local_runner,
    )
    outputs = runner(output_dir=output_dir)

    audit_path = output_dir / "local_patch_attempt" / "fallback_to_full_raster.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    assert audit["execution_mode_used"] == "full_raster"
    assert audit["fallback_stage"] == "merge"
    assert audit["fallback_code"] == "ambiguous_interface"
    assert outputs.manifest["execution_mode_requested"] == "local_patch"
    assert outputs.manifest["local_patch_fallback_audit"] == str(audit_path.resolve())


def test_local_patch_child_runner_can_use_panelized_fallback(tmp_path: Path) -> None:
    output_dir = tmp_path / "variant"

    def local_runner(**kwargs: Any):
        raise collapse_experiment_module.LocalPatchFallbackRequested(
            stage="planning",
            code="unsafe_crop",
            reason="The local crop crosses an unsafe interface.",
            audit_path=output_dir / "local_patch_attempt" / "planning.json",
        )

    def panelized_runner(**kwargs: Any):
        summary_dir = output_dir / "summary"
        summary_dir.mkdir(parents=True, exist_ok=True)
        return SimpleNamespace(output_dir=output_dir, manifest={"variant_id": "S001_unit_7"})

    runner = collapse_experiment_module._make_local_patch_child_runner(
        fallback_variant_runner=panelized_runner,
        local_variant_runner=local_runner,
        fallback_execution_mode="panelized",
    )
    outputs = runner(output_dir=output_dir)

    audit_path = output_dir / "local_patch_attempt" / "fallback_to_panelized.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    assert audit["schema_version"] == "local-patch-fallback-v2"
    assert audit["execution_mode_requested"] == "local_patch"
    assert audit["execution_mode_used"] == "panelized"
    assert outputs.manifest["local_patch_fallback_audit"] == str(audit_path.resolve())
