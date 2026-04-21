from pathlib import Path

import geopandas as gpd
import pandas as pd
from shapely.geometry import LineString, Point

from gauge_sword_match.qa_exports import export_qgis_package
from gauge_sword_match.sword_io import scan_sword_parquet_dir


def test_export_qgis_package_writes_gpkg_layers(tmp_path: Path):
    reaches = gpd.GeoDataFrame(
        {
            "reach_id": [101],
            "river_name": ["Main Stem"],
            "facc": [500.0],
            "stream_order": [5],
            "reach_length": [2_000.0],
        },
        geometry=[LineString([(0.0, 0.0), (0.0, 0.08)])],
        crs="EPSG:4326",
    )
    reaches.to_parquet(tmp_path / "na_sword_v17c_beta_reaches.parquet")

    gauges_gdf = gpd.GeoDataFrame(
        {
            "station_key": ["US:g1"],
            "station_id": ["g1"],
            "station_name": ["Gauge 1"],
            "lat": [0.01],
            "lon": [0.01],
            "country": ["US"],
            "agency": ["demo"],
            "drainage_area": [450.0],
            "river_name": ["Main Stem"],
        },
        geometry=[Point(0.01, 0.01)],
        crs="EPSG:4326",
    )

    best_matches = pd.DataFrame(
        [
            {
                "station_key": "US:g1",
                "reach_id": 101,
                "sword_region": "na",
                "sword_node_id": pd.NA,
                "distance_m": 100.0,
                "total_score": 0.95,
                "second_best_score": 0.50,
                "score_gap": 0.45,
                "confidence_class": "high",
                "review_flag": False,
                "node_distance_m": pd.NA,
            }
        ]
    )

    catalog = scan_sword_parquet_dir(tmp_path)
    output_path = tmp_path / "matched_qgis.gpkg"
    export_qgis_package(best_matches, gauges_gdf, catalog, output_path)

    assert output_path.exists()

    matched_gauges = gpd.read_file(output_path, layer="matched_gauges")
    matched_reaches = gpd.read_file(output_path, layer="matched_reaches")

    assert len(matched_gauges) == 1
    assert len(matched_reaches) == 1
    assert matched_reaches.iloc[0]["reach_id"] == 101
