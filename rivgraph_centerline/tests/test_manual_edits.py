from __future__ import annotations

import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import box


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "rivgraph_centerline" / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from rivgraph_centerline.manual_edits import apply_manual_edits


def test_apply_manual_edits_can_keep_only_largest_water_component(tmp_path) -> None:
    base_mask = tmp_path / "base.tif"
    edits_path = tmp_path / "edits.gpkg"
    output = tmp_path / "cleaned.tif"

    data = np.zeros((12, 16), dtype="uint8")
    data[3:9, 4:10] = 1  # Main component: 36 pixels.
    data[0:2, 0:2] = 1  # Disconnected clutter that would otherwise expand the crop.
    profile = {
        "driver": "GTiff", "height": data.shape[0], "width": data.shape[1], "count": 1,
        "dtype": "uint8", "crs": "EPSG:32631", "transform": from_origin(0, 12, 1, 1), "nodata": 0,
    }
    with rasterio.open(base_mask, "w", **profile) as dst:
        dst.write(data, 1)
    gpd.GeoDataFrame({"action": ["remove"]}, geometry=[box(100, 100, 101, 101)], crs="EPSG:32631").to_file(edits_path, driver="GPKG")

    summary = apply_manual_edits(
        base_mask=base_mask, edits=edits_path, output=output, keep_largest_water_component=True,
    )

    with rasterio.open(output) as cleaned:
        # crop_binary_mask retains a one-pixel land buffer around the water.
        assert cleaned.read(1).shape == (8, 8)
        assert int(cleaned.read(1).sum()) == 36
    component_filter = summary["largest_water_component_filter"]
    assert component_filter["enabled"] is True
    assert component_filter["removed_component_count"] == 1
    assert component_filter["removed_water_pixels"] == 4


def test_apply_manual_edits_crops_output_mask(tmp_path) -> None:
    base_mask = tmp_path / "base.tif"
    edits_path = tmp_path / "edits.gpkg"
    output = tmp_path / "cleaned.tif"

    data = np.zeros((10, 12), dtype="uint8")
    data[2:8, 3:9] = 1
    profile = {
        "driver": "GTiff",
        "height": data.shape[0],
        "width": data.shape[1],
        "count": 1,
        "dtype": "uint8",
        "crs": "EPSG:32631",
        "transform": from_origin(0, 10, 1, 1),
        "nodata": 0,
    }
    with rasterio.open(base_mask, "w", **profile) as dst:
        dst.write(data, 1)

    edits = gpd.GeoDataFrame(
        {"action": ["remove"]},
        geometry=[box(0, 4, 8, 10)],
        crs="EPSG:32631",
    )
    edits.to_file(edits_path, driver="GPKG")

    summary = apply_manual_edits(base_mask=base_mask, edits=edits_path, output=output)

    with rasterio.open(output) as cleaned:
        cleaned_data = cleaned.read(1)
        assert cleaned_data.shape[0] < data.shape[0]
        assert cleaned_data.shape[1] < data.shape[1]
        assert int(cleaned_data.sum()) < int(data.sum())
        assert cleaned.transform.c > profile["transform"].c
        assert cleaned.transform.f < profile["transform"].f

    assert summary["output_shape"] == [int(cleaned_data.shape[0]), int(cleaned_data.shape[1])]


def test_apply_manual_edits_removes_tiny_main_river_land_and_exports_small_remaining(tmp_path) -> None:
    base_mask = tmp_path / "base.tif"
    edits_path = tmp_path / "edits.gpkg"
    output = tmp_path / "cleaned.tif"
    export_path = tmp_path / "main_river_small_land_4_to_10px.gpkg"
    diagonal_path = tmp_path / "main_river_diagonal_passages.gpkg"

    data = np.zeros((20, 20), dtype="uint8")
    data[2:18, 2:18] = 1
    data[5, 5] = 0
    data[10:12, 10:12] = 0
    data[14, 14] = 0
    data[15, 15] = 0
    data[16, 16] = 0
    data[17, 17] = 0
    data[2:7, 14:19] = 0
    data[3:6, 15:18] = 1
    data[4, 16] = 0

    profile = {
        "driver": "GTiff",
        "height": data.shape[0],
        "width": data.shape[1],
        "count": 1,
        "dtype": "uint8",
        "crs": "EPSG:32631",
        "transform": from_origin(0, 20, 1, 1),
        "nodata": 0,
    }
    with rasterio.open(base_mask, "w", **profile) as dst:
        dst.write(data, 1)

    edits = gpd.GeoDataFrame(
        {"action": ["remove"]},
        geometry=[box(100, 100, 101, 101)],
        crs="EPSG:32631",
    )
    edits.to_file(edits_path, driver="GPKG")

    summary = apply_manual_edits(
        base_mask=base_mask,
        edits=edits_path,
        output=output,
        tiny_land_remove_max_pixels=3,
        tiny_land_export_min_pixels=4,
        tiny_land_export_max_pixels=10,
        tiny_land_export_path=export_path,
        diagonal_passage_export_path=diagonal_path,
    )

    with rasterio.open(output) as cleaned:
        cleaned_data = cleaned.read(1)
        row_off = int((profile["transform"].f - cleaned.transform.f) / abs(profile["transform"].e))
        col_off = int((cleaned.transform.c - profile["transform"].c) / profile["transform"].a)
        restored_main_island = cleaned_data[5 - row_off, 5 - col_off]
        retained_four_pixel_island = cleaned_data[10 - row_off, 10 - col_off]
        retained_disconnected_island = cleaned_data[4 - row_off, 16 - col_off]
        assert restored_main_island == 1
        assert retained_four_pixel_island == 0
        assert retained_disconnected_island == 0

    cleanup = summary["main_river_small_land_cleanup"]
    assert cleanup["removed_component_count"] == 1
    assert cleanup["removed_pixel_count"] == 1
    assert cleanup["export_component_count"] == 1
    assert cleanup["export_path"] == str(export_path)

    exported = gpd.read_file(export_path, layer="main_river_small_land")
    assert len(exported) == 1
    assert exported.iloc[0]["size_pixels"] == 4

    diagonal = summary["main_river_diagonal_passages"]
    assert diagonal["count"] == 3
    assert diagonal["export_path"] == str(diagonal_path)

    diag_points = gpd.read_file(diagonal_path, layer="main_river_diagonal_passages")
    assert len(diag_points) == 3
