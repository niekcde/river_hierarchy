from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Point

from network_variants.sword_matching import _resolve_example_reach_filter, match_variant_nodes_to_sword


def _node_frame(rows: list[dict[str, object]]) -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(rows, geometry="geometry", crs="EPSG:3857")


def test_match_variant_nodes_to_sword_matches_nearest_source_nodes(tmp_path: Path) -> None:
    directed_nodes = _node_frame(
        [
            {"id_node": 10, "geometry": Point(0.0, 0.0)},
            {"id_node": 11, "geometry": Point(100.0, 0.0)},
        ]
    )
    parent_nodes = _node_frame(
        [
            {"id_node": 1, "geometry": Point(-10.0, 0.0)},
            {"id_node": 2, "geometry": Point(110.0, 0.0)},
        ]
    )
    node_match = pd.DataFrame(
        {
            "child_id_node": [10, 11],
            "matched_parent_node_id": [pd.NA, pd.NA],
            "parent_node_order": [pd.NA, pd.NA],
        }
    )

    sword_nodes = gpd.GeoDataFrame(
        {
            "node_id": [501, 502],
            "reach_id": [9001, 9002],
            "sword_region": ["na", "na"],
            "dist_out": [1234.0, 5678.0],
            "wse": [4.2, 3.9],
        },
        geometry=[Point(1.0, 0.0), Point(101.0, 0.0)],
        crs="EPSG:3857",
    )
    sword_path = tmp_path / "sword_nodes.gpkg"
    sword_nodes.to_file(sword_path, driver="GPKG")

    enriched_nodes, match_frame, match_metadata = match_variant_nodes_to_sword(
        directed_nodes=directed_nodes,
        parent_nodes=parent_nodes,
        node_match=node_match,
        sword_node_source_path=sword_path,
        sword_wse_field="wse",
    )

    assert list(enriched_nodes["sword_node_id"].astype("Int64")) == [501, 502]
    assert set(enriched_nodes["sword_match_method"]) == {"nearest_sword_node"}
    assert np.allclose(match_frame["sword_match_distance"].astype(float).values, [1.0, 1.0])
    assert set(enriched_nodes["sword_wse_field"]) == {"wse"}
    assert set(enriched_nodes["sword_wse_fallback_used"]) == {False}
    assert set(enriched_nodes["sword_wse_fill_method"]) == {"requested_field"}
    assert match_metadata["scope"] == "bbox_only"


def test_match_variant_nodes_to_sword_propagates_parent_matches_and_falls_back_to_nearest(tmp_path: Path) -> None:
    directed_nodes = _node_frame(
        [
            {"id_node": 10, "geometry": Point(0.0, 0.0)},
            {"id_node": 11, "geometry": Point(100.0, 0.0)},
        ]
    )
    parent_nodes = _node_frame(
        [
            {
                "id_node": 1,
                "geometry": Point(-10.0, 0.0),
                "sword_node_id": 501,
                "sword_reach_id": 9001,
                "sword_region": "na",
                "sword_dist_out": 1234.0,
                "sword_wse": 4.2,
                "sword_wse_field": "wse",
                "sword_source_file": "/tmp/source.parquet",
            },
            {"id_node": 2, "geometry": Point(110.0, 0.0)},
        ]
    )
    node_match = pd.DataFrame(
        {
            "child_id_node": [10, 11],
            "matched_parent_node_id": [1, pd.NA],
            "parent_node_order": [0, pd.NA],
        }
    )

    sword_nodes = gpd.GeoDataFrame(
        {
            "node_id": [501, 502],
            "reach_id": [9001, 9002],
            "sword_region": ["na", "na"],
            "dist_out": [1234.0, 5678.0],
            "wse": [4.2, 3.9],
        },
        geometry=[Point(1.0, 0.0), Point(101.0, 0.0)],
        crs="EPSG:3857",
    )
    sword_path = tmp_path / "sword_nodes.gpkg"
    sword_nodes.to_file(sword_path, driver="GPKG")

    enriched_nodes, match_frame, match_metadata = match_variant_nodes_to_sword(
        directed_nodes=directed_nodes,
        parent_nodes=parent_nodes,
        node_match=node_match,
        sword_node_source_path=sword_path,
        sword_wse_field="wse",
    )

    row_10 = match_frame.loc[match_frame["id_node"] == 10].iloc[0]
    row_11 = match_frame.loc[match_frame["id_node"] == 11].iloc[0]

    assert int(row_10["sword_node_id"]) == 501
    assert row_10["sword_match_method"] == "propagated_parent"
    assert bool(row_10["sword_match_from_parent"]) is True
    assert int(row_11["sword_node_id"]) == 502
    assert row_11["sword_match_method"] == "nearest_sword_node"
    assert enriched_nodes.loc[enriched_nodes["id_node"] == 10, "sword_region"].iloc[0] == "na"
    assert bool(enriched_nodes.loc[enriched_nodes["id_node"] == 10, "sword_wse_fallback_used"].iloc[0]) is False
    assert enriched_nodes.loc[enriched_nodes["id_node"] == 10, "sword_wse_fill_method"].iloc[0] == "requested_field"
    assert match_metadata["scope"] == "bbox_only"


def test_match_variant_nodes_to_sword_propagates_parent_match_with_missing_fill_method(tmp_path: Path) -> None:
    directed_nodes = _node_frame(
        [
            {"id_node": 10, "geometry": Point(0.0, 0.0)},
        ]
    )
    parent_nodes = _node_frame(
        [
            {
                "id_node": 1,
                "geometry": Point(-10.0, 0.0),
                "sword_node_id": 501,
                "sword_reach_id": 9001,
                "sword_region": "na",
                "sword_dist_out": 1234.0,
                "sword_wse": 4.2,
                "sword_wse_field": "wse_obs_p50",
                "sword_wse_fallback_used": False,
                "sword_wse_fill_method": pd.NA,
                "sword_source_file": "/tmp/source.parquet",
            },
        ]
    )
    node_match = pd.DataFrame(
        {
            "child_id_node": [10],
            "matched_parent_node_id": [1],
            "parent_node_order": [0],
        }
    )

    enriched_nodes, match_frame, _ = match_variant_nodes_to_sword(
        directed_nodes=directed_nodes,
        parent_nodes=parent_nodes,
        node_match=node_match,
        sword_node_source_path=None,
        sword_wse_field="wse_obs_p50",
    )

    row = enriched_nodes.iloc[0]
    match_row = match_frame.iloc[0]
    assert int(row["sword_node_id"]) == 501
    assert row["sword_match_method"] == "propagated_parent"
    assert row["sword_wse_fill_method"] == "requested_field"
    assert match_row["sword_wse_fill_method"] == "requested_field"


def test_match_variant_nodes_to_sword_skips_parent_rows_with_string_na_sword_node_id(tmp_path: Path) -> None:
    directed_nodes = _node_frame(
        [
            {"id_node": 10, "geometry": Point(0.0, 0.0)},
            {"id_node": 11, "geometry": Point(100.0, 0.0)},
        ]
    )
    parent_nodes = _node_frame(
        [
            {
                "id_node": 1,
                "geometry": Point(-10.0, 0.0),
                "sword_node_id": "<NA>",
                "sword_wse_fill_method": "requested_field",
            },
            {
                "id_node": 2,
                "geometry": Point(110.0, 0.0),
                "sword_node_id": 502,
                "sword_reach_id": 9002,
                "sword_region": "na",
                "sword_dist_out": 5678.0,
                "sword_wse": 3.9,
                "sword_wse_field": "wse",
                "sword_wse_fill_method": "requested_field",
            },
        ]
    )
    node_match = pd.DataFrame(
        {
            "child_id_node": [10, 11],
            "matched_parent_node_id": [1, 2],
            "parent_node_order": [0, 1],
        }
    )

    enriched_nodes, match_frame, _ = match_variant_nodes_to_sword(
        directed_nodes=directed_nodes,
        parent_nodes=parent_nodes,
        node_match=node_match,
        sword_node_source_path=None,
        sword_wse_field="wse",
    )

    row_10 = match_frame.loc[match_frame["id_node"] == 10].iloc[0]
    row_11 = match_frame.loc[match_frame["id_node"] == 11].iloc[0]

    assert pd.isna(row_10["sword_node_id"])
    assert row_10["sword_match_method"] == "unmatched"
    assert int(row_11["sword_node_id"]) == 502
    assert row_11["sword_match_method"] == "propagated_parent"
    assert pd.isna(enriched_nodes.loc[enriched_nodes["id_node"] == 10, "sword_node_id"]).iloc[0]


def test_match_variant_nodes_to_sword_falls_back_from_requested_wse_field_to_wse(tmp_path: Path) -> None:
    directed_nodes = _node_frame(
        [
            {"id_node": 10, "geometry": Point(0.0, 0.0)},
            {"id_node": 11, "geometry": Point(100.0, 0.0)},
        ]
    )
    parent_nodes = _node_frame(
        [
            {"id_node": 1, "geometry": Point(-10.0, 0.0)},
            {"id_node": 2, "geometry": Point(110.0, 0.0)},
        ]
    )
    node_match = pd.DataFrame(
        {
            "child_id_node": [10, 11],
            "matched_parent_node_id": [pd.NA, pd.NA],
            "parent_node_order": [pd.NA, pd.NA],
        }
    )

    sword_nodes = gpd.GeoDataFrame(
        {
            "node_id": [501, 502],
            "reach_id": [9001, 9002],
            "sword_region": ["na", "na"],
            "dist_out": [1234.0, 5678.0],
            "wse_obs_p50": [np.nan, 5.5],
            "wse": [4.2, 3.9],
        },
        geometry=[Point(1.0, 0.0), Point(101.0, 0.0)],
        crs="EPSG:3857",
    )
    sword_path = tmp_path / "sword_nodes_fallback.gpkg"
    sword_nodes.to_file(sword_path, driver="GPKG")

    enriched_nodes, match_frame, _ = match_variant_nodes_to_sword(
        directed_nodes=directed_nodes,
        parent_nodes=parent_nodes,
        node_match=node_match,
        sword_node_source_path=sword_path,
        sword_wse_field="wse_obs_p50",
    )

    row_10 = enriched_nodes.loc[enriched_nodes["id_node"] == 10].iloc[0]
    row_11 = enriched_nodes.loc[enriched_nodes["id_node"] == 11].iloc[0]

    assert float(row_10["sword_wse"]) == 4.2
    assert row_10["sword_wse_field"] == "wse"
    assert bool(row_10["sword_wse_fallback_used"]) is True
    assert row_10["sword_wse_fill_method"] == "same_node_wse"

    assert float(row_11["sword_wse"]) == 5.5
    assert row_11["sword_wse_field"] == "wse_obs_p50"
    assert bool(row_11["sword_wse_fallback_used"]) is False
    assert row_11["sword_wse_fill_method"] == "requested_field"

    fallback_flags = match_frame.sort_values("id_node")["sword_wse_fallback_used"].tolist()
    assert fallback_flags == [True, False]


def test_match_variant_nodes_to_sword_fills_from_nearest_same_reach_node_when_both_fields_null(tmp_path: Path) -> None:
    directed_nodes = _node_frame(
        [
            {"id_node": 10, "geometry": Point(0.0, 0.0)},
        ]
    )
    parent_nodes = _node_frame(
        [
            {"id_node": 1, "geometry": Point(-10.0, 0.0)},
        ]
    )
    node_match = pd.DataFrame(
        {
            "child_id_node": [10],
            "matched_parent_node_id": [pd.NA],
            "parent_node_order": [pd.NA],
        }
    )

    sword_nodes = gpd.GeoDataFrame(
        {
            "node_id": [501, 502, 503],
            "reach_id": [9001, 9001, 9001],
            "sword_region": ["na", "na", "na"],
            "node_order": [2, 3, 10],
            "dist_out": [1234.0, 1200.0, 800.0],
            "wse_obs_p50": [np.nan, 5.5, 6.6],
            "wse": [1.1, 3.9, 2.2],
        },
        geometry=[Point(1.0, 0.0), Point(1.2, 0.0), Point(10.0, 0.0)],
        crs="EPSG:3857",
    )
    sword_path = tmp_path / "sword_nodes_neighbor_fill.gpkg"
    sword_nodes.to_file(sword_path, driver="GPKG")

    enriched_nodes, match_frame, _ = match_variant_nodes_to_sword(
        directed_nodes=directed_nodes,
        parent_nodes=parent_nodes,
        node_match=node_match,
        sword_node_source_path=sword_path,
        sword_wse_field="wse_obs_p50",
    )

    row = enriched_nodes.iloc[0]
    assert int(row["sword_node_id"]) == 501
    assert float(row["sword_wse"]) == 5.5
    assert row["sword_wse_field"] == "wse_obs_p50"
    assert bool(row["sword_wse_fallback_used"]) is True
    assert row["sword_wse_fill_method"] == "nearest_reach_node"
    assert int(row["sword_wse_source_node_id"]) == 502

    match_row = match_frame.iloc[0]
    assert match_row["sword_wse_fill_method"] == "nearest_reach_node"
    assert int(match_row["sword_wse_source_node_id"]) == 502


def test_resolve_example_reach_filter_uses_dist_out_and_endpoint_buffer(tmp_path: Path, monkeypatch) -> None:
    sword_dir = tmp_path / "sword"
    sword_dir.mkdir()

    example_stations = gpd.GeoDataFrame(
        {
            "station_key": ["BR:3652880", "BR:3652890"],
            "example_ids": ["3", "3"],
            "down_values": ["True", ""],
        },
        geometry=[Point(-0.01, 0.0), Point(0.21, 0.0)],
        crs="EPSG:4326",
    )
    example_station_path = tmp_path / "hierarchy_examples_filtered_subdaily_manual_updates_final.gpkg"
    example_stations.to_file(example_station_path, driver="GPKG")

    station_matches = gpd.GeoDataFrame(
        {
            "station_key": ["BR:3652880", "BR:3652890"],
            "reach_id": [100, 80],
            "sword_region": ["sa", "sa"],
        },
        geometry=[Point(-0.01, 0.0), Point(0.21, 0.0)],
        crs="EPSG:4326",
    )
    station_match_path = tmp_path / "selected_event_stations_same_main_path.gpkg"
    station_matches.to_file(station_match_path, driver="GPKG")

    topology = pd.DataFrame(
        {
            "reach_id": [100, 90, 80, 70, 60],
            "rch_id_dn": [90, 80, 70, pd.NA, pd.NA],
            "dist_out": [300.0, 200.0, 100.0, 0.0, 250.0],
        }
    )

    def fake_build_downstream_adjacency(reaches, *, reach_id_col="reach_id", downstream_col=None):
        return {
            int(row[reach_id_col]): {int(row["rch_id_dn"])} if not pd.isna(row["rch_id_dn"]) else set()
            for _, row in reaches.iterrows()
        }

    def fake_find_reaches_between(reaches, upstream_reach_ids, downstream_reach_ids, *, reach_id_col="reach_id", downstream_col=None):
        assert list(upstream_reach_ids) == [100]
        assert list(downstream_reach_ids) == [80]
        return {100, 90, 80}

    def fake_normalize_reach_id(value):
        if pd.isna(value):
            return None
        return int(value)

    monkeypatch.setattr(
        "network_variants.sword_matching._maybe_import_reach_tools",
        lambda: (
            fake_build_downstream_adjacency,
            fake_find_reaches_between,
            lambda catalog, region, columns=(): topology.copy(),
            fake_normalize_reach_id,
            lambda path: object(),
        ),
    )

    metadata = _resolve_example_reach_filter(
        example_id="sarl_03",
        sword_node_source_path=sword_dir,
        example_station_source_path=example_station_path,
        station_match_source_path=station_match_path,
        reach_buffer_steps=1,
    )

    assert metadata["scope"] == "example_reach_corridor"
    assert metadata["candidate_region"] == "sa"
    assert metadata["candidate_reach_ids"] == [70, 80, 90, 100]
    assert metadata["candidate_reach_count"] == 4
    assert metadata["upstream_station_key"] == "BR:3652880"
    assert metadata["downstream_station_key"] == "BR:3652890"
    assert metadata["upstream_reach_id"] == 100
    assert metadata["downstream_reach_id"] == 80


def test_match_variant_nodes_to_sword_applies_example_reach_filter_to_geo_file(tmp_path: Path, monkeypatch) -> None:
    directed_nodes = _node_frame(
        [
            {"id_node": 10, "geometry": Point(10.4, 0.0)},
        ]
    )
    parent_nodes = _node_frame(
        [
            {"id_node": 1, "geometry": Point(0.0, 0.0)},
        ]
    )
    node_match = pd.DataFrame(
        {
            "child_id_node": [10],
            "matched_parent_node_id": [pd.NA],
            "parent_node_order": [pd.NA],
        }
    )

    sword_nodes = gpd.GeoDataFrame(
        {
            "node_id": [901, 601],
            "reach_id": [90, 60],
            "sword_region": ["sa", "sa"],
            "dist_out": [200.0, 250.0],
            "wse": [9.0, 9.5],
        },
        geometry=[Point(10.0, 0.0), Point(10.5, 0.0)],
        crs="EPSG:3857",
    )
    sword_path = tmp_path / "sword_nodes.gpkg"
    sword_nodes.to_file(sword_path, driver="GPKG")

    monkeypatch.setattr(
        "network_variants.sword_matching._resolve_example_reach_filter",
        lambda **kwargs: {
            "scope": "example_reach_corridor",
            "reason": "",
            "example_numeric_id": 3,
            "candidate_region": "sa",
            "candidate_reach_ids": [70, 80, 90, 100],
            "candidate_reach_count": 4,
            "upstream_station_key": "BR:3652880",
            "downstream_station_key": "BR:3652890",
            "upstream_reach_id": 100,
            "downstream_reach_id": 80,
            "reach_buffer_steps": 1,
            "example_station_source": "/tmp/example.gpkg",
            "station_match_source": "/tmp/matches.gpkg",
        },
    )

    enriched_nodes, match_frame, match_metadata = match_variant_nodes_to_sword(
        directed_nodes=directed_nodes,
        parent_nodes=parent_nodes,
        node_match=node_match,
        sword_node_source_path=sword_path,
        sword_wse_field="wse",
        example_id="sarl_03",
    )

    row = match_frame.iloc[0]
    assert int(row["sword_node_id"]) == 901
    assert int(row["sword_reach_id"]) == 90
    assert row["sword_match_method"] == "nearest_sword_node"
    assert match_metadata["scope"] == "example_reach_corridor"
    assert match_metadata["candidate_reach_ids"] == [70, 80, 90, 100]
    assert enriched_nodes.loc[0, "sword_region"] == "sa"
