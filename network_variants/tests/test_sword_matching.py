from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Point

from network_variants.sword_matching import match_variant_nodes_to_sword


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

    enriched_nodes, match_frame = match_variant_nodes_to_sword(
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

    enriched_nodes, match_frame = match_variant_nodes_to_sword(
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
