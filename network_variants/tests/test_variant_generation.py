from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import LineString, Point

from hierarchy_level_definition.unit_detection.bifurcation_confluence_units import StructuralUnit
from network_variants.variant_generation import (
    _apply_collapsed_selection_metadata,
    _apply_collapse_components_to_mask,
    _assign_variant_directions,
    _componentize_units,
    _compute_family_stats,
    _infer_parent_node_order,
    _match_child_links_to_parent_links,
    _match_child_nodes_to_parent_nodes,
    _resolve_link_by_progression,
    _resolve_link_lineage,
    _resolve_single_remaining_link,
    _trim_slice,
)


def _unit(unit_id: int, node_ids: set[int], edge_keys: set[tuple[int, int, int]]) -> StructuralUnit:
    return StructuralUnit(
        unit_id=unit_id,
        bifurcation=min(node_ids),
        confluence=max(node_ids),
        n_paths=2,
        unit_class="simple_bifurcation_confluence_pair",
        min_path_length=1.0,
        max_path_length=1.0,
        node_set=node_ids,
        edge_set=edge_keys,
    )


def test_componentize_units_groups_by_shared_nodes() -> None:
    units_by_id = {
        1: _unit(1, {1, 2, 3}, {(1, 3, 10)}),
        2: _unit(2, {3, 4, 5}, {(3, 5, 11)}),
        3: _unit(3, {9, 10}, {(9, 10, 12)}),
    }

    components = _componentize_units([1, 2, 3], units_by_id)

    assert [component["unit_ids"] for component in components] == [[1, 2], [3]]
    assert components[0]["link_ids"] == [10, 11]
    assert components[1]["link_ids"] == [12]


def test_trim_slice_matches_rivgraph_style_end_trimming() -> None:
    widths = np.asarray([4.0, 4.0, 4.0, 4.0, 4.0, 4.0], dtype=float)
    distances = np.asarray([0.0, 3.0, 6.0, 9.0, 12.0, 15.0], dtype=float)

    trim = _trim_slice(widths, distances)

    assert trim.start == 1
    assert trim.stop == 4


def test_compute_family_stats_uses_trimmed_values() -> None:
    values = np.asarray([1.0, 2.0, 10.0, 20.0], dtype=float)
    trim = slice(1, 3)

    stats = _compute_family_stats(values, trim)

    assert np.isclose(stats["adj"], 6.0)
    assert np.isclose(stats["med"], 6.0)
    assert np.isclose(stats["p50"], 6.0)


def test_apply_collapsed_selection_metadata_adds_collapsed_unit_columns() -> None:
    frame = pd.DataFrame({"id_link": [10, 11]})

    result = _apply_collapsed_selection_metadata(frame, unit_ids=[5], group_label=None)

    assert result["collapsed_selection_label"].tolist() == ["collapsed_unit_5", "collapsed_unit_5"]
    assert result["collapsed_unit_ids"].tolist() == ["5", "5"]
    assert result["collapsed_selection_type"].tolist() == ["unit", "unit"]


def test_apply_collapse_components_to_mask_adds_pixels(tmp_path: Path) -> None:
    mask = np.zeros((20, 20), dtype=np.uint8)
    mask[4, 4:16] = 1
    mask[15, 4:16] = 1
    mask[4:16, 4] = 1
    mask[4:16, 15] = 1

    mask_path = tmp_path / "parent_mask.tif"
    transform = from_origin(0, 20, 1, 1)
    with rasterio.open(
        mask_path,
        "w",
        driver="GTiff",
        height=mask.shape[0],
        width=mask.shape[1],
        count=1,
        dtype="uint8",
        crs="EPSG:3857",
        transform=transform,
    ) as dst:
        dst.write(mask, 1)

    reviewed_links = gpd.GeoDataFrame(
        [
            {"id_link": 1, "wid_adj": 1.0, "geometry": LineString([(4.5, 15.5), (15.5, 15.5)])},
            {"id_link": 2, "wid_adj": 1.0, "geometry": LineString([(4.5, 4.5), (15.5, 4.5)])},
            {"id_link": 3, "wid_adj": 1.0, "geometry": LineString([(4.5, 4.5), (4.5, 15.5)])},
            {"id_link": 4, "wid_adj": 1.0, "geometry": LineString([(15.5, 4.5), (15.5, 15.5)])},
        ],
        geometry="geometry",
        crs="EPSG:3857",
    )

    components = [
            {
                "component_id": "C1",
                "unit_ids": [1],
                "link_ids": [1, 2, 3, 4],
                "node_ids": [10, 11],
                "compound_bubble_ids": [],
                "n_units": 1,
                "n_links": 4,
                "n_nodes": 2,
            }
        ]

    collapsed_mask, summary, component_frame, edit_geometries = _apply_collapse_components_to_mask(
        mask_path,
        components,
        reviewed_links,
        preferred_width_field="wid_adj",
        buffer_scale=0.5,
    )

    assert summary["pixels_added"] > 0
    assert int(collapsed_mask.sum()) > int(mask.sum())
    assert int(component_frame.loc[0, "added_pixels"]) == summary["pixels_added"]
    assert set(edit_geometries["geometry_role"]) == {"footprint", "edit"}


def test_resolve_link_by_progression_prefers_upstream_source_distance() -> None:
    resolved = _resolve_link_by_progression(
        node_a=7,
        node_b=6,
        source_distance={6: 3283.5, 7: 3093.6},
        sink_distance={6: 63615.7, 7: 63657.4},
        geometry_order=(6, 7),
        allow_geometry_fallback=True,
    )

    assert resolved == ((7, 6), "global_progression_rule")


def test_resolve_single_remaining_link_uses_progression_when_both_directions_validate() -> None:
    child_nodes = gpd.GeoDataFrame(
        [
            {"id_node": 0, "is_inlet": True, "is_outlet": False, "geometry": Point(0, 0)},
            {"id_node": 1, "is_inlet": False, "is_outlet": False, "geometry": Point(10, 10)},
            {"id_node": 2, "is_inlet": False, "is_outlet": False, "geometry": Point(10, 0)},
            {"id_node": 3, "is_inlet": False, "is_outlet": False, "geometry": Point(20, 10)},
            {"id_node": 5, "is_inlet": False, "is_outlet": True, "geometry": Point(30, 0)},
        ],
        geometry="geometry",
        crs="EPSG:3857",
    )

    child_links = gpd.GeoDataFrame(
        [
            {"id_link": 100, "len": 10.0, "geometry": LineString([(0, 0), (10, 0)])},
            {"id_link": 101, "len": 14.0, "geometry": LineString([(0, 0), (10, 10)])},
            {"id_link": 102, "len": 20.0, "geometry": LineString([(10, 0), (30, 0)])},
            {"id_link": 103, "len": 14.0, "geometry": LineString([(10, 10), (20, 10)])},
            {"id_link": 104, "len": 14.0, "geometry": LineString([(20, 10), (30, 0)])},
            {"id_link": 105, "len": 14.0, "geometry": LineString([(20, 10), (10, 0)])},
        ],
        geometry="geometry",
        crs="EPSG:3857",
    )

    orientation_by_link = {
        100: (0, 2),
        101: (0, 1),
        102: (2, 5),
        103: (1, 3),
        104: (3, 5),
    }

    resolved = _resolve_single_remaining_link(
        link_id=105,
        edge_nodes={105: (2, 3)},
        geometry_order_by_link={105: (3, 2)},
        orientation_by_link=orientation_by_link,
        child_nodes=child_nodes,
        child_inlet_nodes={0},
        child_outlet_nodes={5},
        child_links=child_links,
    )

    assert resolved == ((2, 3), "global_progression_rule")


def test_matching_and_direction_assignment_populates_upstream_downstream_nodes() -> None:
    parent_nodes = gpd.GeoDataFrame(
        [
            {"id_node": 0, "is_inlet": True, "is_outlet": False, "geometry": Point(0, 0)},
            {"id_node": 1, "is_inlet": False, "is_outlet": False, "geometry": Point(10, 0)},
            {"id_node": 2, "is_inlet": False, "is_outlet": False, "geometry": Point(20, 0)},
            {"id_node": 3, "is_inlet": False, "is_outlet": True, "geometry": Point(30, 0)},
        ],
        geometry="geometry",
        crs="EPSG:3857",
    )
    parent_nodes["idx_node"] = np.arange(len(parent_nodes))
    parent_nodes["id_links"] = ["0", "0,1", "1,2", "2"]
    parent_nodes["n_links"] = [1, 2, 2, 1]
    parent_nodes["type_io"] = ["inlet", "neither", "neither", "outlet"]
    parent_nodes["schema_rg"] = "rg-v1"

    parent_links = gpd.GeoDataFrame(
        [
            {"id_link": 0, "id_nodes": "0, 1", "id_us_node": 0, "id_ds_node": 1, "geometry": LineString([(0, 0), (10, 0)])},
            {"id_link": 1, "id_nodes": "1, 2", "id_us_node": 1, "id_ds_node": 2, "geometry": LineString([(10, 0), (20, 0)])},
            {"id_link": 2, "id_nodes": "2, 3", "id_us_node": 2, "id_ds_node": 3, "geometry": LineString([(20, 0), (30, 0)])},
        ],
        geometry="geometry",
        crs="EPSG:3857",
    )
    parent_links["idx_link"] = ""
    parent_links["n_nodes"] = 2
    parent_links["is_inlet"] = [True, False, False]
    parent_links["is_outlet"] = [False, False, True]
    parent_links["type_io"] = ["inlet", "neither", "outlet"]
    parent_links["schema_rg"] = "rg-v1"
    parent_links["link_conn"] = ""
    parent_links["wid_pix"] = ""
    parent_links["len"] = [10.0, 10.0, 10.0]
    parent_links["wid"] = [4.0, 4.0, 4.0]
    parent_links["wid_adj"] = [4.0, 4.0, 4.0]
    parent_links["wid_med"] = [4.0, 4.0, 4.0]
    parent_links["len_adj"] = [10.0, 10.0, 10.0]
    parent_links["sinuosity"] = [1.0, 1.0, 1.0]

    child_nodes = gpd.GeoDataFrame(
        [
            {"id_node": 10, "is_inlet": True, "is_outlet": False, "geometry": Point(0, 0)},
            {"id_node": 11, "is_inlet": False, "is_outlet": False, "geometry": Point(10, 0)},
            {"id_node": 12, "is_inlet": False, "is_outlet": False, "geometry": Point(20, 0)},
            {"id_node": 13, "is_inlet": False, "is_outlet": True, "geometry": Point(30, 0)},
        ],
        geometry="geometry",
        crs="EPSG:3857",
    )
    child_nodes["idx_node"] = np.arange(len(child_nodes))
    child_nodes["id_links"] = ["10", "10,11", "11,12", "12"]
    child_nodes["n_links"] = [1, 2, 2, 1]
    child_nodes["type_io"] = ["inlet", "neither", "neither", "outlet"]
    child_nodes["schema_rg"] = "rg-v1"

    child_links = gpd.GeoDataFrame(
        [
            {"id_link": 10, "id_nodes": "11, 10", "geometry": LineString([(0, 0), (10, 0)])},
            {"id_link": 11, "id_nodes": "12, 11", "geometry": LineString([(10, 0), (20, 0)])},
            {"id_link": 12, "id_nodes": "13, 12", "geometry": LineString([(20, 0), (30, 0)])},
        ],
        geometry="geometry",
        crs="EPSG:3857",
    )
    child_links["idx_link"] = ""
    child_links["n_nodes"] = 2
    child_links["id_us_node"] = pd.NA
    child_links["id_ds_node"] = pd.NA
    child_links["is_inlet"] = [True, False, False]
    child_links["is_outlet"] = [False, False, True]
    child_links["type_io"] = ["inlet", "neither", "outlet"]
    child_links["schema_rg"] = "rg-v1"
    child_links["link_conn"] = ""
    child_links["wid_pix"] = ""
    child_links["len"] = [10.0, 10.0, 10.0]
    child_links["wid"] = [4.0, 4.0, 4.0]
    child_links["wid_adj"] = [4.0, 4.0, 4.0]
    child_links["wid_med"] = [4.0, 4.0, 4.0]
    child_links["len_adj"] = [10.0, 10.0, 10.0]
    child_links["sinuosity"] = [1.0, 1.0, 1.0]

    parent_order, _, _ = _infer_parent_node_order(parent_links, parent_nodes)
    node_match = _match_child_nodes_to_parent_nodes(
        parent_nodes,
        child_nodes,
        match_tolerance=1.0,
        parent_node_order=parent_order,
    )
    link_match = _match_child_links_to_parent_links(
        parent_links,
        child_links,
        match_tolerance=1.0,
    )
    directed_links, directed_nodes, summary = _assign_variant_directions(
        child_links=child_links,
        child_nodes=child_nodes,
        parent_links=parent_links,
        parent_nodes=parent_nodes,
        node_match=node_match,
        link_match=link_match,
    )
    _, parent_graph, _ = _infer_parent_node_order(parent_links, parent_nodes)
    link_lineage = _resolve_link_lineage(
        parent_graph=parent_graph,
        directed_child_links=directed_links,
        node_match=node_match,
        link_match=link_match,
        match_tolerance=1.0,
    )

    oriented_pairs = {
        int(row.id_link): (int(row.id_us_node), int(row.id_ds_node))
        for row in directed_links.itertuples(index=False)
    }
    assert oriented_pairs == {
        10: (10, 11),
        11: (11, 12),
        12: (12, 13),
    }
    assert summary["source_node"] == 10
    assert summary["sink_node"] == 13
    assert directed_nodes.loc[directed_nodes["id_node"] == 10, "is_inlet"].iloc[0]
    assert directed_nodes.loc[directed_nodes["id_node"] == 13, "is_outlet"].iloc[0]
    assert link_match["child_id_link"].nunique() == 3
    coords = list(directed_links.loc[directed_links["id_link"] == 10].iloc[0].geometry.coords)
    assert coords[0] == (0.0, 0.0)
    assert coords[-1] == (10.0, 0.0)
    lineage_row = link_lineage.loc[link_lineage["id_link"] == 10].iloc[0]
    assert lineage_row["lineage_type"] == "unchanged_1to1"
    assert lineage_row["matched_parent_link_ids"] == "0"
    assert lineage_row["touch_parent_link_ids"] == "1"
    assert lineage_row["secondary_parent_link_ids"] == ""
    assert np.isclose(lineage_row["matched_parent_overlap_fraction"], 1.0)


def test_resolved_lineage_uses_parent_node_path_and_keeps_parallel_edges() -> None:
    parent_nodes = gpd.GeoDataFrame(
        [
            {"id_node": 0, "is_inlet": True, "is_outlet": False, "geometry": Point(0, 0)},
            {"id_node": 1, "is_inlet": False, "is_outlet": False, "geometry": Point(10, 0)},
            {"id_node": 2, "is_inlet": False, "is_outlet": False, "geometry": Point(20, 0)},
            {"id_node": 3, "is_inlet": False, "is_outlet": True, "geometry": Point(30, 0)},
        ],
        geometry="geometry",
        crs="EPSG:3857",
    )
    parent_nodes["idx_node"] = np.arange(len(parent_nodes))
    parent_nodes["id_links"] = ["0", "0,1,2", "1,2,3", "3"]
    parent_nodes["n_links"] = [1, 3, 3, 1]
    parent_nodes["type_io"] = ["inlet", "neither", "neither", "outlet"]
    parent_nodes["schema_rg"] = "rg-v1"

    parent_links = gpd.GeoDataFrame(
        [
            {"id_link": 0, "id_nodes": "0, 1", "id_us_node": 0, "id_ds_node": 1, "geometry": LineString([(0, 0), (10, 0)])},
            {"id_link": 1, "id_nodes": "1, 2", "id_us_node": 1, "id_ds_node": 2, "geometry": LineString([(10, 0), (20, 0)])},
            {"id_link": 2, "id_nodes": "1, 2", "id_us_node": 1, "id_ds_node": 2, "geometry": LineString([(10, 0), (20, 0)])},
            {"id_link": 3, "id_nodes": "2, 3", "id_us_node": 2, "id_ds_node": 3, "geometry": LineString([(20, 0), (30, 0)])},
        ],
        geometry="geometry",
        crs="EPSG:3857",
    )
    parent_links["idx_link"] = ""
    parent_links["n_nodes"] = 2
    parent_links["is_inlet"] = [True, False, False, False]
    parent_links["is_outlet"] = [False, False, False, True]
    parent_links["type_io"] = ["inlet", "neither", "neither", "outlet"]
    parent_links["schema_rg"] = "rg-v1"
    parent_links["link_conn"] = ""
    parent_links["wid_pix"] = ""
    parent_links["len"] = [10.0, 10.0, 10.0, 10.0]
    parent_links["wid"] = [4.0, 4.0, 4.0, 4.0]
    parent_links["wid_adj"] = [4.0, 4.0, 4.0, 4.0]
    parent_links["wid_med"] = [4.0, 4.0, 4.0, 4.0]
    parent_links["len_adj"] = [10.0, 10.0, 10.0, 10.0]
    parent_links["sinuosity"] = [1.0, 1.0, 1.0, 1.0]

    child_nodes = gpd.GeoDataFrame(
        [
            {"id_node": 10, "is_inlet": True, "is_outlet": False, "geometry": Point(0, 0)},
            {"id_node": 11, "is_inlet": False, "is_outlet": True, "geometry": Point(30, 0)},
        ],
        geometry="geometry",
        crs="EPSG:3857",
    )
    child_nodes["idx_node"] = np.arange(len(child_nodes))
    child_nodes["id_links"] = ["20", "20"]
    child_nodes["n_links"] = [1, 1]
    child_nodes["type_io"] = ["inlet", "outlet"]
    child_nodes["schema_rg"] = "rg-v1"

    child_links = gpd.GeoDataFrame(
        [
            {"id_link": 20, "id_nodes": "11, 10", "geometry": LineString([(0, 0), (30, 0)])},
        ],
        geometry="geometry",
        crs="EPSG:3857",
    )
    child_links["idx_link"] = ""
    child_links["n_nodes"] = 2
    child_links["id_us_node"] = pd.NA
    child_links["id_ds_node"] = pd.NA
    child_links["is_inlet"] = [True]
    child_links["is_outlet"] = [True]
    child_links["type_io"] = ["both"]
    child_links["schema_rg"] = "rg-v1"
    child_links["link_conn"] = ""
    child_links["wid_pix"] = ""
    child_links["len"] = [30.0]
    child_links["wid"] = [4.0]
    child_links["wid_adj"] = [4.0]
    child_links["wid_med"] = [4.0]
    child_links["len_adj"] = [30.0]
    child_links["sinuosity"] = [1.0]

    parent_order, parent_graph, _ = _infer_parent_node_order(parent_links, parent_nodes)
    node_match = _match_child_nodes_to_parent_nodes(
        parent_nodes,
        child_nodes,
        match_tolerance=1.0,
        parent_node_order=parent_order,
    )
    link_match = _match_child_links_to_parent_links(
        parent_links,
        child_links,
        match_tolerance=1.0,
    )
    directed_links, _, _ = _assign_variant_directions(
        child_links=child_links,
        child_nodes=child_nodes,
        parent_links=parent_links,
        parent_nodes=parent_nodes,
        node_match=node_match,
        link_match=link_match,
    )
    link_lineage = _resolve_link_lineage(
        parent_graph=parent_graph,
        directed_child_links=directed_links,
        node_match=node_match,
        link_match=link_match,
        match_tolerance=1.0,
    )

    lineage_row = link_lineage.loc[link_lineage["id_link"] == 20].iloc[0]
    assert lineage_row["lineage_method"] == "matched_parent_node_path"
    assert lineage_row["matched_parent_link_ids"] == "0,1,2,3"
    assert lineage_row["primary_parent_link_ids"] == "0,1,2,3"
    assert lineage_row["secondary_parent_link_ids"] == ""
    assert lineage_row["lineage_type"] == "collapsed_many_to_one"
    assert np.isclose(lineage_row["matched_parent_overlap_fraction"], 1.0)
