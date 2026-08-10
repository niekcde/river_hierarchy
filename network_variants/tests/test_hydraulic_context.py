from __future__ import annotations

import json
from pathlib import Path

import geopandas as gpd
import networkx as nx
import pandas as pd
import pytest
from shapely.geometry import LineString, Point

from network_variants.hydraulic_context import (
    SlopeConfig,
    _apply_collapsed_unit_head_solver,
    _summarize_dag_path_drops,
    compute_hydraulic_context,
    generate_state_hydraulic_context,
)


def test_dag_path_drop_summary_matches_explicit_path_enumeration() -> None:
    graph = nx.DiGraph()
    graph.add_edge(0, 1, drop_m=1.0)
    graph.add_edge(1, 3, drop_m=2.0)
    graph.add_edge(0, 2, drop_m=4.0)
    graph.add_edge(2, 3, drop_m=1.0)
    graph.add_edge(0, 4, drop_m=2.0)
    graph.add_edge(4, 2, drop_m=1.0)

    explicit_drops = [
        sum(
            float(graph[upstream][downstream]["drop_m"])
            for upstream, downstream in zip(path[:-1], path[1:], strict=False)
        )
        for path in nx.all_simple_paths(graph, 0, 3)
    ]
    summary = _summarize_dag_path_drops(graph, 0, 3)

    assert summary is not None
    assert summary["n_paths"] == len(explicit_drops) == 3
    assert summary["drop_min_m"] == pytest.approx(min(explicit_drops))
    assert summary["drop_max_m"] == pytest.approx(max(explicit_drops))
    assert summary["drop_mean_m"] == pytest.approx(sum(explicit_drops) / len(explicit_drops))
    assert summary["drop_spread_m"] == pytest.approx(max(explicit_drops) - min(explicit_drops))
    assert summary["is_consistent"] is False
    assert summary["summary_method"] == "dag_dynamic_programming"


def _simple_nodes() -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(
        [
            {
                "id_node": 1,
                "is_inlet": True,
                "is_outlet": False,
                "sword_wse": 10.0,
                "sword_wse_field": "wse",
                "sword_wse_fill_method": "requested_field",
                "geometry": Point(0.0, 0.0),
            },
            {
                "id_node": 2,
                "is_inlet": False,
                "is_outlet": True,
                "sword_wse": 9.0,
                "sword_wse_field": "wse",
                "sword_wse_fill_method": "requested_field",
                "geometry": Point(100.0, 0.0),
            },
        ],
        geometry="geometry",
        crs="EPSG:3857",
    )


def _simple_links() -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(
        [
            {
                "id_link": 1,
                "id_us_node": 1,
                "id_ds_node": 2,
                "len": 100.0,
                "geometry": LineString([(0.0, 0.0), (100.0, 0.0)]),
            }
        ],
        geometry="geometry",
        crs="EPSG:3857",
    )


def _write_reference_section_summary(tmp_path: Path) -> Path:
    experiment_dir = tmp_path / "experiment"
    reference_dir = experiment_dir / "reference_section" / "variant"
    directed_dir = reference_dir / "directed"
    rivgraph_dir = reference_dir / "rivgraph"
    directed_dir.mkdir(parents=True)
    rivgraph_dir.mkdir(parents=True)

    centerline_path = rivgraph_dir / "reference_centerline.gpkg"
    directed_nodes_path = directed_dir / "reference_directed_nodes.gpkg"
    sword_nodes_path = rivgraph_dir / "reference_SWORD_nodes.gpkg"
    base_nodes_path = experiment_dir / "states" / "S000_base" / "variant" / "directed" / "base_directed_nodes.gpkg"
    base_nodes_path.parent.mkdir(parents=True)

    centerline = gpd.GeoDataFrame(
        [{"geometry": LineString([(0.0, 0.0), (100.0, 0.0)])}],
        geometry="geometry",
        crs="EPSG:3857",
    )
    centerline.to_file(centerline_path, driver="GPKG")

    reference_nodes = gpd.GeoDataFrame(
        [
            {"id_node": 100, "is_inlet": True, "is_outlet": False, "geometry": Point(0.0, 0.0)},
            {"id_node": 101, "is_inlet": False, "is_outlet": True, "geometry": Point(100.0, 0.0)},
        ],
        geometry="geometry",
        crs="EPSG:3857",
    )
    reference_nodes.to_file(directed_nodes_path, driver="GPKG")

    reference_sword_nodes = gpd.GeoDataFrame(
        [
            {"wse": 12.0, "wse_fill_method": "requested_field", "geometry": Point(0.0, 0.0)},
            {"wse": 10.5, "wse_fill_method": "requested_field", "geometry": Point(50.0, 0.0)},
            {"wse": 9.0, "wse_fill_method": "requested_field", "geometry": Point(100.0, 0.0)},
        ],
        geometry="geometry",
        crs="EPSG:3857",
    )
    reference_sword_nodes.to_file(sword_nodes_path, driver="GPKG")

    base_nodes = gpd.GeoDataFrame(
        [
            {"id_node": 1, "is_inlet": True, "is_outlet": False, "geometry": Point(0.0, 0.0)},
            {"id_node": 2, "is_inlet": False, "is_outlet": True, "geometry": Point(100.0, 0.0)},
        ],
        geometry="geometry",
        crs="EPSG:3857",
    )
    base_nodes.to_file(base_nodes_path, driver="GPKG")

    pd.DataFrame(
        [
            {
                "state_id": "S000_base",
                "state_role": "base",
                "directed_nodes_path": str(base_nodes_path.resolve()),
            }
        ]
    ).to_csv(experiment_dir / "state_registry.csv", index=False)

    summary_path = experiment_dir / "reference_section" / "reference_section_summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "paths": {
                    "centerline_path": str(centerline_path.resolve()),
                    "directed_nodes_path": str(directed_nodes_path.resolve()),
                    "reference_section_dir": str((experiment_dir / "reference_section").resolve()),
                }
            }
        ),
        encoding="utf-8",
    )
    return summary_path


def test_compute_hydraulic_context_builds_profile_and_link_slopes() -> None:
    result = compute_hydraulic_context(_simple_links(), _simple_nodes())

    assert list(result.node_context["rapid_wse_for_slope"]) == pytest.approx([10.0, 9.0])
    assert result.node_context["rapid_wse_source_method"].tolist() == [
        "profile_monotone_requested_field_nodes",
        "profile_monotone_requested_field_nodes",
    ]

    row = result.link_context.iloc[0]
    assert row["wse_us"] == pytest.approx(10.0)
    assert row["wse_ds"] == pytest.approx(9.0)
    assert row["slope_used"] == pytest.approx(0.01)
    assert row["slope_source_method"] == "raw_wse"
    assert row["bubble_slope_qc_reason"] == "unavailable_unit_summary"
    assert result.section_slope_summary["slope"] == pytest.approx(0.01)


def test_generate_state_hydraulic_context_writes_artifacts(tmp_path: Path) -> None:
    links_path = tmp_path / "links.gpkg"
    nodes_path = tmp_path / "nodes.gpkg"
    _simple_links().to_file(links_path, driver="GPKG")
    _simple_nodes().to_file(nodes_path, driver="GPKG")

    outputs = generate_state_hydraulic_context(
        directed_links_path=links_path,
        directed_nodes_path=nodes_path,
        hierarchy_output_dir=None,
        output_dir=tmp_path / "hydraulic_context",
        state_id="S000_base",
    )

    assert outputs.link_context_path.exists()
    assert outputs.node_context_path.exists()
    assert outputs.manifest_path.exists()
    assert outputs.manifest["state_id"] == "S000_base"
    assert outputs.manifest["head_concepts"]["normalized_solver_heads"] == "not written here"

    written_links = pd.read_csv(outputs.link_context_path)
    written_nodes = pd.read_csv(outputs.node_context_path)
    assert float(written_links.loc[0, "slope_used"]) == pytest.approx(0.01)
    assert "rapid_wse_for_slope" in written_nodes.columns


def test_generate_state_hydraulic_context_inherits_family_representative_slope(tmp_path: Path) -> None:
    links = gpd.GeoDataFrame(
        [
            {
                "id_link": 1,
                "id_us_node": 1,
                "id_ds_node": 2,
                "len": 10.0,
                "family_width_inheritance_applied": True,
                "family_width_inherited_from_link": 2,
                "geometry": LineString([(0.0, 0.0), (10.0, 0.0)]),
            },
            {
                "id_link": 2,
                "id_us_node": 2,
                "id_ds_node": 3,
                "len": 100.0,
                "family_width_inheritance_applied": False,
                "family_width_inherited_from_link": pd.NA,
                "geometry": LineString([(10.0, 0.0), (110.0, 0.0)]),
            },
        ],
        geometry="geometry",
        crs="EPSG:3857",
    )
    nodes = gpd.GeoDataFrame(
        [
            {
                "id_node": 1,
                "is_inlet": True,
                "is_outlet": False,
                "sword_wse": 10.0,
                "sword_wse_field": "wse",
                "sword_wse_fill_method": "requested_field",
                "geometry": Point(0.0, 0.0),
            },
            {
                "id_node": 2,
                "is_inlet": False,
                "is_outlet": False,
                "sword_wse": 9.5,
                "sword_wse_field": "wse",
                "sword_wse_fill_method": "requested_field",
                "geometry": Point(10.0, 0.0),
            },
            {
                "id_node": 3,
                "is_inlet": False,
                "is_outlet": True,
                "sword_wse": 9.0,
                "sword_wse_field": "wse",
                "sword_wse_fill_method": "requested_field",
                "geometry": Point(110.0, 0.0),
            },
        ],
        geometry="geometry",
        crs="EPSG:3857",
    )

    links_path = tmp_path / "links.gpkg"
    nodes_path = tmp_path / "nodes.gpkg"
    links.to_file(links_path, driver="GPKG")
    nodes.to_file(nodes_path, driver="GPKG")

    outputs = generate_state_hydraulic_context(
        directed_links_path=links_path,
        directed_nodes_path=nodes_path,
        hierarchy_output_dir=None,
        output_dir=tmp_path / "hydraulic_context",
        state_id="S010_unit_8",
    )

    by_link = outputs.link_context.set_index("id_link")
    assert float(by_link.loc[2, "slope_used"]) == pytest.approx(0.005)
    assert float(by_link.loc[1, "slope_used"]) == pytest.approx(0.005)
    assert str(by_link.loc[1, "slope_source_method"]) == "family_representative_link"
    assert bool(by_link.loc[1, "family_slope_inheritance_applied"]) is True
    assert int(by_link.loc[1, "family_slope_inherited_from_link"]) == 2
    assert outputs.manifest["family_slope_inheritance"]["applied_link_ids"] == [1]


def test_generate_state_hydraulic_context_inherits_strict_parent_slope(tmp_path: Path) -> None:
    links = _simple_links().copy()
    links["lineage_type"] = "unchanged_1to1"
    links["dominant_parent_link_id"] = 10
    links["dominant_parent_overlap_fraction"] = 1.0
    links["matched_parent_us_node"] = 101
    links["matched_parent_ds_node"] = 102

    links_path = tmp_path / "links.gpkg"
    nodes_path = tmp_path / "nodes.gpkg"
    parent_context_path = tmp_path / "parent_link_hydraulic_context.csv"
    links.to_file(links_path, driver="GPKG")
    _simple_nodes().to_file(nodes_path, driver="GPKG")
    pd.DataFrame(
        [
            {
                "id_link": 10,
                "id_us_node": 101,
                "id_ds_node": 102,
                "slope_used": 0.0025,
                "slope_source_method": "compound_bubble_profile",
                "slope_reason": "filled_from_compound_bubble_profile",
                "slope_minimum_applied": False,
            }
        ]
    ).to_csv(parent_context_path, index=False)

    outputs = generate_state_hydraulic_context(
        directed_links_path=links_path,
        directed_nodes_path=nodes_path,
        hierarchy_output_dir=None,
        output_dir=tmp_path / "hydraulic_context",
        state_id="S001_unit_1",
        parent_link_context_path=parent_context_path,
    )

    row = outputs.link_context.iloc[0]
    assert float(row["raw_slope"]) == pytest.approx(0.01)
    assert float(row["slope_preinheritance_value"]) == pytest.approx(0.01)
    assert float(row["slope_used"]) == pytest.approx(0.0025)
    assert str(row["slope_source_method"]) == "parent_unchanged_1to1"
    assert bool(row["slope_parent_inheritance_applied"]) is True
    assert int(row["slope_inherited_from_parent_link"]) == 10
    assert str(row["slope_parent_source_method"]) == "compound_bubble_profile"
    assert outputs.manifest["parent_slope_inheritance"]["applied_link_ids"] == [1]


def test_parent_slope_inheritance_rejects_reversed_parent_direction(tmp_path: Path) -> None:
    links = _simple_links().copy()
    links["lineage_type"] = "unchanged_1to1"
    links["dominant_parent_link_id"] = 10
    links["dominant_parent_overlap_fraction"] = 1.0
    links["matched_parent_us_node"] = 102
    links["matched_parent_ds_node"] = 101

    links_path = tmp_path / "links.gpkg"
    nodes_path = tmp_path / "nodes.gpkg"
    parent_context_path = tmp_path / "parent_link_hydraulic_context.csv"
    links.to_file(links_path, driver="GPKG")
    _simple_nodes().to_file(nodes_path, driver="GPKG")
    pd.DataFrame(
        [{"id_link": 10, "id_us_node": 101, "id_ds_node": 102, "slope_used": 0.0025}]
    ).to_csv(parent_context_path, index=False)

    outputs = generate_state_hydraulic_context(
        directed_links_path=links_path,
        directed_nodes_path=nodes_path,
        hierarchy_output_dir=None,
        output_dir=tmp_path / "hydraulic_context",
        state_id="S001_unit_1",
        parent_link_context_path=parent_context_path,
    )

    row = outputs.link_context.iloc[0]
    assert float(row["slope_used"]) == pytest.approx(0.01)
    assert bool(row["slope_parent_inheritance_applied"]) is False
    assert str(row["slope_parent_inheritance_reason"]) == "parent_direction_mismatch"
    assert outputs.manifest["parent_slope_inheritance"]["direction_mismatch_link_ids"] == [1]


def test_topology_conflict_solver_uses_unchanged_downstream_parent_target() -> None:
    frame = pd.DataFrame(
        [
            {
                "id_link": 1,
                "id_us_node": 10,
                "id_ds_node": 11,
                "link_length_m": 20.0,
                "wse_us": 9.0,
                "wse_ds": 10.0,
                "raw_slope": -0.05,
                "slope_used": 1e-6,
                "slope_preinheritance_value": 1e-6,
                "slope_parent_value": 0.02,
                "slope_parent_direction_matches": False,
                "slope_parent_inheritance_applied": False,
                "slope_section_ref": 0.001,
            },
            {
                "id_link": 2,
                "id_us_node": 11,
                "id_ds_node": 12,
                "link_length_m": 100.0,
                "wse_us": 10.0,
                "wse_ds": 9.5,
                "raw_slope": 0.005,
                "slope_used": 0.003,
                "slope_preinheritance_value": 0.005,
                "slope_parent_value": 0.003,
                "slope_parent_direction_matches": True,
                "slope_parent_inheritance_applied": True,
                "slope_section_ref": 0.001,
            },
            {
                "id_link": 3,
                "id_us_node": 20,
                "id_ds_node": 21,
                "link_length_m": 50.0,
                "wse_us": 8.0,
                "wse_ds": 7.9,
                "raw_slope": 0.002,
                "slope_used": 0.002,
                "slope_preinheritance_value": 0.002,
                "slope_parent_value": 0.002,
                "slope_parent_direction_matches": True,
                "slope_parent_inheritance_applied": True,
                "slope_section_ref": 0.001,
            },
        ]
    )
    solved, summary = _apply_collapsed_unit_head_solver(
        frame,
        collapsed_unit_boundary_analysis=[
            {
                "component_id": "C1",
                "variant_local_link_ids": [1],
                "variant_boundary_heads": [
                    {"variant_node": 10, "head_m": 9.0},
                    {"variant_node": 11, "head_m": 10.0},
                ],
            }
        ],
        min_slope=1e-6,
    )

    by_link = solved.set_index("id_link")
    assert float(by_link.loc[1, "slope_used"]) == pytest.approx(0.003)
    assert float(by_link.loc[2, "slope_used"]) == pytest.approx(0.003)
    assert float(by_link.loc[3, "slope_used"]) == pytest.approx(0.002)
    assert by_link.loc[1, "topology_conflict_target_source"] == "downstream_unchanged_parent_continuation"
    assert summary["topology_conflict_link_ids"] == [1]
    assert summary["topology_context_link_ids"] == [2]


def test_collapsed_head_solver_never_replaces_positive_fallback_with_zero() -> None:
    frame = pd.DataFrame(
        [
            {
                "id_link": 1,
                "id_us_node": 10,
                "id_ds_node": 11,
                "link_length_m": 100.0,
                "raw_slope": 0.0,
                "slope_used": 1e-6,
                "slope_source_method": "minimum_slope",
                "slope_reason": "below_minimum_raw",
                "slope_minimum_applied": True,
            }
        ]
    )
    solved, summary = _apply_collapsed_unit_head_solver(
        frame,
        collapsed_unit_boundary_analysis=[
            {
                "component_id": "C1",
                "variant_local_link_ids": [1],
                "variant_boundary_heads": [
                    {"variant_node": 10, "head_m": 9.0},
                    {"variant_node": 11, "head_m": 9.0},
                ],
            }
        ],
        min_slope=1e-6,
    )

    row = solved.iloc[0]
    assert float(row["slope_used"]) == pytest.approx(1e-6)
    assert float(row["slope_used"]) > 0
    assert bool(row["collapsed_profile_repair_applied"]) is False
    assert str(row["collapsed_profile_repair_reason"]) == "missing_variant_path_targets"
    assert summary["applied_link_ids"] == []


def test_path_relative_solver_combines_multiple_targets_for_shared_link() -> None:
    frame = pd.DataFrame(
        [
            {
                "id_link": 10,
                "id_us_node": 1,
                "id_ds_node": 2,
                "link_length_m": 100.0,
                "raw_slope": 0.001,
                "slope_used": 0.001,
                "slope_source_method": "raw_wse",
                "slope_reason": "ok",
                "slope_minimum_applied": False,
            },
            {
                "id_link": 11,
                "id_us_node": 2,
                "id_ds_node": 3,
                "link_length_m": 100.0,
                "raw_slope": 0.009,
                "slope_used": 0.009,
                "slope_source_method": "raw_wse",
                "slope_reason": "ok",
                "slope_minimum_applied": False,
            },
            {
                "id_link": 12,
                "id_us_node": 2,
                "id_ds_node": 4,
                "link_length_m": 100.0,
                "raw_slope": 0.019,
                "slope_used": 0.019,
                "slope_source_method": "raw_wse",
                "slope_reason": "ok",
                "slope_minimum_applied": False,
            },
        ]
    )
    solved, summary = _apply_collapsed_unit_head_solver(
        frame,
        collapsed_unit_boundary_analysis=[
            {
                "component_id": "C1",
                "variant_local_link_ids": [10, 11, 12],
                "variant_boundary_heads": [
                    {"variant_node": 1, "head_m": 10.0},
                    {"variant_node": 3, "head_m": 9.0},
                    {"variant_node": 4, "head_m": 8.0},
                ],
                "variant_paths": [
                    {
                        "source_boundary_node": 1,
                        "target_boundary_node": 3,
                        "node_path": [1, 2, 3],
                        "link_path": [10, 11],
                    },
                    {
                        "source_boundary_node": 1,
                        "target_boundary_node": 4,
                        "node_path": [1, 2, 4],
                        "link_path": [10, 12],
                    },
                ],
            }
        ],
        min_slope=1e-6,
    )

    by_link = solved.set_index("id_link")
    assert set(by_link["slope_source_method"]) == {"collapsed_unit_path_relative_solver"}
    assert all(pd.to_numeric(by_link["slope_used"], errors="raise") >= 1e-6)
    assert by_link.loc[10, "path_relative_target_paths"] == "P1:1->3,P2:1->4"
    assert by_link.loc[10, "path_relative_target_slopes"] == "0.005,0.01"
    assert float(by_link.loc[10, "path_relative_objective_after"]) < float(
        by_link.loc[10, "path_relative_objective_before"]
    )
    assert (
        float(by_link.loc[10, "collapsed_reference_drop_m"])
        + float(by_link.loc[11, "collapsed_reference_drop_m"])
    ) == pytest.approx(1.0)
    assert (
        float(by_link.loc[10, "collapsed_reference_drop_m"])
        + float(by_link.loc[12, "collapsed_reference_drop_m"])
    ) == pytest.approx(2.0)
    assert summary["path_relative_component_ids"] == ["C1"]
    assert summary["path_relative_failed_components"] == []


def test_compute_hydraulic_context_can_use_reference_section_profile(tmp_path: Path) -> None:
    summary_path = _write_reference_section_summary(tmp_path)
    nodes = _simple_nodes().copy()
    nodes["matched_parent_node_id"] = [1, 2]
    nodes["match_within_tolerance"] = [True, True]

    result = compute_hydraulic_context(
        _simple_links(),
        nodes,
        config=SlopeConfig(
            slope_profile_mode="reference_section",
            reference_section_summary_path=str(summary_path),
        ),
    )

    assert "node_reference_section_dist_out_m" in result.node_context.columns
    assert result.node_context["rapid_wse_source_method"].tolist() == [
        "reference_section_monotone_requested_field_sword_nodes",
        "reference_section_monotone_requested_field_sword_nodes",
    ]
    row = result.link_context.iloc[0]
    assert row["wse_us"] == pytest.approx(12.0)
    assert row["wse_ds"] == pytest.approx(9.0)
    assert row["slope_used"] == pytest.approx(0.03)


def test_generate_state_hydraulic_context_applies_collapsed_unit_head_solver(tmp_path: Path) -> None:
    experiment_dir = tmp_path / "experiment"
    base_state_dir = experiment_dir / "states" / "S000_base"
    child_state_dir = experiment_dir / "states" / "S001_unit_1"

    base_hc_dir = base_state_dir / "hydraulic_context"
    base_hc_dir.mkdir(parents=True)
    pd.DataFrame(
        [
            {"id_node": 11, "rapid_wse_for_slope": 177.94389},
            {"id_node": 8, "rapid_wse_for_slope": 177.245404},
        ]
    ).to_csv(base_hc_dir / "node_hydraulic_context.csv", index=False)
    pd.DataFrame(
        [
            {
                "id_link": 30,
                "id_us_node": 11,
                "id_ds_node": 8,
                "link_length_m": 1447.522362,
                "slope_used": (177.94389 - 177.245404) / 1447.522362,
            }
        ]
    ).to_csv(base_hc_dir / "link_hydraulic_context.csv", index=False)

    links = gpd.GeoDataFrame(
        [
            {
                "id_link": 11,
                "id_us_node": 50,
                "id_ds_node": 51,
                "len": 1447.522362,
                "geometry": LineString([(0.0, 0.0), (1447.522362, 0.0)]),
            }
        ],
        geometry="geometry",
        crs="EPSG:3857",
    )
    nodes = gpd.GeoDataFrame(
        [
            {
                "id_node": 50,
                "is_inlet": True,
                "is_outlet": False,
                "sword_wse": 177.94389,
                "sword_wse_field": "wse",
                "sword_wse_fill_method": "requested_field",
                "geometry": Point(0.0, 0.0),
            },
            {
                "id_node": 51,
                "is_inlet": False,
                "is_outlet": True,
                "sword_wse": 177.48722,
                "sword_wse_field": "wse",
                "sword_wse_fill_method": "requested_field",
                "geometry": Point(1447.522362, 0.0),
            },
        ],
        geometry="geometry",
        crs="EPSG:3857",
    )

    links_path = child_state_dir / "variant" / "directed" / "child_directed_links.gpkg"
    nodes_path = child_state_dir / "variant" / "directed" / "child_directed_nodes.gpkg"
    links_path.parent.mkdir(parents=True, exist_ok=True)
    nodes_path.parent.mkdir(parents=True, exist_ok=True)
    links.to_file(links_path, driver="GPKG")
    nodes.to_file(nodes_path, driver="GPKG")

    matching_dir = child_state_dir / "variant" / "matching"
    matching_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {"child_id_node": 50, "matched_parent_node_id": 11, "match_within_tolerance": True},
            {"child_id_node": 51, "matched_parent_node_id": 8, "match_within_tolerance": True},
        ]
    ).to_csv(matching_dir / "node_match.csv", index=False)
    summary_dir = child_state_dir / "variant" / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "component_id": "C1",
                "link_ids": "30",
                "node_ids": "8,11",
                "collapsed_selection_type": "unit",
                "collapsed_selection_label": "collapsed_unit_1",
                "collapsed_unit_ids": "1",
                "collapsed_unit_count": 1,
            }
        ]
    ).to_csv(summary_dir / "collapse_components.csv", index=False)

    outputs = generate_state_hydraulic_context(
        directed_links_path=links_path,
        directed_nodes_path=nodes_path,
        hierarchy_output_dir=None,
        output_dir=child_state_dir / "hydraulic_context",
        state_id="S001_unit_1",
    )

    written = outputs.link_context.set_index("id_link")
    expected_slope = (177.94389 - 177.245404) / 1447.522362
    assert float(written.loc[11, "raw_slope"]) == pytest.approx((177.94389 - 177.48722) / 1447.522362)
    assert float(written.loc[11, "slope_used"]) == pytest.approx(expected_slope)
    assert str(written.loc[11, "slope_source_method"]) == "collapsed_unit_path_relative_solver"
    assert bool(written.loc[11, "collapsed_profile_repair_applied"]) is True
    assert float(written.loc[11, "collapsed_reference_drop_m"]) == pytest.approx(177.94389 - 177.245404)
    assert outputs.manifest["collapsed_unit_head_solver"]["applied_link_ids"] == [11]
    assert outputs.manifest["collapsed_unit_head_solver"]["path_relative_component_ids"] == ["C1"]


def test_generate_state_hydraulic_context_reports_collapsed_unit_boundary_analysis_from_slope_used(
    tmp_path: Path,
) -> None:
    experiment_dir = tmp_path / "experiment"
    base_state_dir = experiment_dir / "states" / "S000_base"
    child_state_dir = experiment_dir / "states" / "S001_unit_1"

    base_hc_dir = base_state_dir / "hydraulic_context"
    base_hc_dir.mkdir(parents=True)
    pd.DataFrame(
        [
            {"id_node": 1, "rapid_wse_for_slope": 9.0},
            {"id_node": 2, "rapid_wse_for_slope": 9.5},
            {"id_node": 3, "rapid_wse_for_slope": 10.0},
        ]
    ).to_csv(base_hc_dir / "node_hydraulic_context.csv", index=False)
    pd.DataFrame(
        [
            {
                "id_link": 10,
                "id_us_node": 3,
                "id_ds_node": 2,
                "link_length_m": 100.0,
                "slope_used": 0.01,
            },
            {
                "id_link": 11,
                "id_us_node": 2,
                "id_ds_node": 1,
                "link_length_m": 100.0,
                "slope_used": 0.02,
            },
        ]
    ).to_csv(base_hc_dir / "link_hydraulic_context.csv", index=False)

    links = gpd.GeoDataFrame(
        [
            {
                "id_link": 1,
                "id_us_node": 30,
                "id_ds_node": 40,
                "len": 50.0,
                "geometry": LineString([(0.0, 0.0), (50.0, 0.0)]),
            },
            {
                "id_link": 2,
                "id_us_node": 40,
                "id_ds_node": 31,
                "len": 50.0,
                "geometry": LineString([(50.0, 0.0), (100.0, 0.0)]),
            },
            {
                "id_link": 3,
                "id_us_node": 31,
                "id_ds_node": 32,
                "len": 100.0,
                "geometry": LineString([(100.0, 0.0), (200.0, 0.0)]),
            },
        ],
        geometry="geometry",
        crs="EPSG:3857",
    )
    nodes = gpd.GeoDataFrame(
        [
            {
                "id_node": 30,
                "is_inlet": True,
                "is_outlet": False,
                "sword_wse": 8.0,
                "sword_wse_field": "wse",
                "sword_wse_fill_method": "requested_field",
                "geometry": Point(0.0, 0.0),
            },
            {
                "id_node": 40,
                "is_inlet": False,
                "is_outlet": False,
                "sword_wse": 7.5,
                "sword_wse_field": "wse",
                "sword_wse_fill_method": "requested_field",
                "geometry": Point(50.0, 0.0),
            },
            {
                "id_node": 31,
                "is_inlet": False,
                "is_outlet": False,
                "sword_wse": 7.0,
                "sword_wse_field": "wse",
                "sword_wse_fill_method": "requested_field",
                "geometry": Point(100.0, 0.0),
            },
            {
                "id_node": 32,
                "is_inlet": False,
                "is_outlet": True,
                "sword_wse": 6.5,
                "sword_wse_field": "wse",
                "sword_wse_fill_method": "requested_field",
                "geometry": Point(200.0, 0.0),
            },
        ],
        geometry="geometry",
        crs="EPSG:3857",
    )

    links_path = child_state_dir / "variant" / "directed" / "child_directed_links.gpkg"
    nodes_path = child_state_dir / "variant" / "directed" / "child_directed_nodes.gpkg"
    links_path.parent.mkdir(parents=True, exist_ok=True)
    links.to_file(links_path, driver="GPKG")
    nodes.to_file(nodes_path, driver="GPKG")

    matching_dir = child_state_dir / "variant" / "matching"
    matching_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {"child_id_node": 30, "matched_parent_node_id": 3, "match_within_tolerance": True},
            {"child_id_node": 31, "matched_parent_node_id": 2, "match_within_tolerance": True},
            {"child_id_node": 32, "matched_parent_node_id": 1, "match_within_tolerance": True},
        ]
    ).to_csv(matching_dir / "node_match.csv", index=False)

    summary_dir = child_state_dir / "variant" / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "component_id": "C1",
                "link_ids": "10,11",
                "node_ids": "1,2,3",
                "collapsed_selection_type": "unit",
                "collapsed_selection_label": "collapsed_unit_1",
                "collapsed_unit_ids": "1",
                "collapsed_unit_count": 1,
            }
        ]
    ).to_csv(summary_dir / "collapse_components.csv", index=False)

    outputs = generate_state_hydraulic_context(
        directed_links_path=links_path,
        directed_nodes_path=nodes_path,
        hierarchy_output_dir=None,
        output_dir=child_state_dir / "hydraulic_context",
        state_id="S001_unit_1",
    )

    analysis = outputs.manifest["collapsed_unit_boundary_analysis"]
    assert len(analysis) == 1
    component = analysis[0]
    assert component["variant_boundary_nodes"] == [30, 31, 32]
    assert component["variant_entry_nodes"] == [30]
    assert component["variant_exit_nodes"] == [32]

    boundary_heads = {
        item["variant_node"]: item["head_m"]
        for item in component["variant_boundary_heads"]
    }
    assert boundary_heads[32] == pytest.approx(9.0)
    assert boundary_heads[31] == pytest.approx(11.0)
    assert boundary_heads[30] == pytest.approx(12.0)

    base_consistency = component["base_path_drop_consistency"]
    assert base_consistency["3->1"]["drop_mean_m"] == pytest.approx(3.0)
    assert base_consistency["3->2"]["drop_mean_m"] == pytest.approx(1.0)
    assert base_consistency["2->1"]["drop_mean_m"] == pytest.approx(2.0)


def test_reference_section_profile_avoids_statewise_corridor_anchor_load(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    summary_path = _write_reference_section_summary(tmp_path)
    nodes = _simple_nodes().copy()
    nodes["matched_parent_node_id"] = [1, 2]
    nodes["match_within_tolerance"] = [True, True]

    def fail_if_called(*args, **kwargs):
        raise AssertionError("Statewise corridor SWORD anchor loading should not run in reference_section mode.")

    monkeypatch.setattr(
        "network_variants.hydraulic_context._load_reference_section_corridor_sword_anchors",
        fail_if_called,
    )

    result = compute_hydraulic_context(
        _simple_links(),
        nodes,
        config=SlopeConfig(
            slope_profile_mode="reference_section",
            reference_section_summary_path=str(summary_path),
        ),
    )

    row = result.link_context.iloc[0]
    assert row["wse_us"] == pytest.approx(12.0)
    assert row["wse_ds"] == pytest.approx(9.0)
    assert row["slope_used"] == pytest.approx(0.03)
