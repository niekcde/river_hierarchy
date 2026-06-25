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
