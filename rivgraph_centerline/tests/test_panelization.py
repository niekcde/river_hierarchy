from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import geopandas as gpd
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import LineString

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "rivgraph_centerline" / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from rivgraph_centerline.panelization import _provisional_panel_grid, create_panel_breaks, plan_panels, prepare_panel_masks, rebuild_core_network_nodes


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
