from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import geopandas as gpd
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import LineString, Point

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "rivgraph_centerline" / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from rivgraph_centerline.panelization import _provisional_panel_grid, assign_core_network_direction, create_panel_breaks, plan_panels, prepare_panel_masks, rebuild_core_network_nodes


def test_panel_breaks_and_plan_create_four_irregular_core_outputs(tmp_path: Path) -> None:
    mask = tmp_path / "mask.tif"
    with rasterio.open(mask, "w", driver="GTiff", height=100, width=100, count=1, dtype="uint8", crs="EPSG:32611", transform=from_origin(0, 100, 1, 1), nodata=0) as dst:
        dst.write(np.ones((100, 100), dtype="uint8"), 1)
    breaks = tmp_path / "breaks.gpkg"
    plan = tmp_path / "panel_plan.gpkg"
    assert create_panel_breaks(mask_path=mask, output_path=breaks, rows=2, columns=2)["break_count"] == 2
    summary = plan_panels(mask_path=mask, breaks_path=breaks, output_path=plan, overlap_m=5)
    assert summary["panel_count"] == 4
    assert summary["over_budget_panel_ids"] == []
    assert plan.exists()
    assert plan.with_suffix(".json").exists()
    prepared = prepare_panel_masks(mask_path=mask, plan_path=plan, output_dir=tmp_path / "panels", panel_ids={"P001"})
    assert prepared["prepared_panel_ids"] == ["P001"]
    assert (tmp_path / "panels" / "P001" / "mask.tif").exists()


def test_rebuild_core_network_nodes_creates_global_endpoint_topology(tmp_path: Path) -> None:
    source = tmp_path / "merged_core_links.gpkg"
    links = gpd.GeoDataFrame(
        {"global_link_id": [0, 1, 2]},
        geometry=[
            LineString([(0, 0), (10, 0)]),
            LineString([(10, 0), (20, 0)]),
            LineString([(10, 0), (10, 10)]),
        ],
        crs="EPSG:32611",
    )
    links.to_file(source, layer="merged_core_links", driver="GPKG")

    summary = rebuild_core_network_nodes(merged_links_path=source, output_dir=tmp_path / "reconciled")

    assert summary["is_valid"] is True
    assert summary["link_count"] == 3
    assert summary["node_count"] == 4
    assert summary["max_degree"] == 3
    assert summary["connected_component_count"] == 1
    reconciled = gpd.read_file(tmp_path / "reconciled" / "merged_links.gpkg", layer="merged_links")
    assert {"id_node_a", "id_node_b", "id_nodes", "direction_status"}.issubset(reconciled.columns)


def test_provisional_panel_grid_is_budget_based_and_aspect_aware() -> None:
    rows, columns = _provisional_panel_grid(height=48_000, width=40_500, max_panel_pixels=160_000_000)
    assert (rows, columns) == (4, 4)
    assert np.ceil(48_000 / rows) * np.ceil(40_500 / columns) <= 160_000_000


def test_assign_core_network_direction_uses_distinct_downstream_node(tmp_path: Path) -> None:
    core_dir = tmp_path / "core"
    panel_rivgraph = tmp_path / "panels" / "P001" / "rivgraph"
    core_dir.mkdir(parents=True)
    panel_rivgraph.mkdir(parents=True)
    crs = "EPSG:32611"
    gpd.GeoDataFrame(
        {"global_link_id": [0], "source_segments": ["P001:1:0"], "id_node_a": [10], "id_node_b": [11]},
        geometry=[LineString([(0, 0), (10, 0)])], crs=crs,
    ).to_file(core_dir / "merged_links.gpkg", layer="merged_links", driver="GPKG")
    gpd.GeoDataFrame(
        {"segment_id": ["P001:1:0"], "panel_id": ["P001"], "source_link_id": [1]},
        geometry=[LineString([(0, 0), (10, 0)])], crs=crs,
    ).to_file(core_dir / "core_segments.gpkg", layer="core_segments", driver="GPKG")
    gpd.GeoDataFrame(
        {"id_link": [1], "id_us_node": [5], "id_ds_node": [6]},
        geometry=[LineString([(0, 0), (10, 0)])], crs=crs,
    ).to_file(panel_rivgraph / "x_P001_links.gpkg", driver="GPKG")
    gpd.GeoDataFrame({"id_node": [5, 6]}, geometry=[Point(0, 0), Point(10, 0)], crs=crs).to_file(panel_rivgraph / "x_P001_nodes.gpkg", driver="GPKG")

    summary = assign_core_network_direction(
        merged_links_path=core_dir / "merged_links.gpkg",
        core_segments_path=core_dir / "core_segments.gpkg",
        panel_root=tmp_path / "panels",
        run_name="x",
        output_dir=core_dir,
    )

    assert summary["assigned_link_count"] == 1
    directed = gpd.read_file(core_dir / "merged_links_directed.gpkg", layer="merged_links")
    assert directed.loc[0, "id_us_node"] == 10
    assert directed.loc[0, "id_ds_node"] == 11
