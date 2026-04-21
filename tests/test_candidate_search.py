from pathlib import Path

import geopandas as gpd
import pandas as pd
from shapely.geometry import LineString

from gauge_sword_match.candidate_search import search_reach_candidates
from gauge_sword_match.gauge_io import clean_gauges, gauges_to_geodataframe
from gauge_sword_match.sword_io import load_reaches, scan_sword_parquet_dir


def test_search_reach_candidates_returns_nearest_reach(tmp_path: Path):
    reaches = gpd.GeoDataFrame(
        {
            "reach_id": [101, 202],
            "river_name": ["Main Stem", "Far River"],
            "facc": [500.0, 50.0],
            "stream_order": [5, 2],
            "reach_length": [2_000.0, 1_500.0],
        },
        geometry=[
            LineString([(0.0, 0.0), (0.0, 0.08)]),
            LineString([(1.0, 1.0), (1.1, 1.1)]),
        ],
        crs="EPSG:4326",
    )
    reach_path = tmp_path / "na_sword_v17c_beta_reaches.parquet"
    reaches.to_parquet(reach_path)

    gauges = pd.DataFrame(
        [
            {
                "station_id": "g-1",
                "station_name": "Gauge 1",
                "lat": 0.01,
                "lon": 0.01,
                "country": "US",
                "agency": "demo",
                "drainage_area": 450.0,
                "river_name": "Main Stem",
            }
        ]
    )
    cleaned = clean_gauges(gauges)
    gauges_gdf = gauges_to_geodataframe(cleaned)
    catalog = scan_sword_parquet_dir(tmp_path)

    candidates = search_reach_candidates(
        gauges=gauges_gdf,
        catalog=catalog,
        search_radius_m=5_000,
        max_candidates=4,
    )

    assert len(candidates) == 1
    assert candidates.iloc[0]["reach_id"] == 101
    assert candidates.iloc[0]["distance_m"] < 2_000


def test_load_reaches_accepts_float_reach_ids(tmp_path: Path):
    reaches = gpd.GeoDataFrame(
        {
            "reach_id": [101, 202],
            "river_name": ["Main Stem", "Far River"],
            "facc": [500.0, 50.0],
            "stream_order": [5, 2],
            "reach_length": [2_000.0, 1_500.0],
        },
        geometry=[
            LineString([(0.0, 0.0), (0.0, 0.08)]),
            LineString([(1.0, 1.0), (1.1, 1.1)]),
        ],
        crs="EPSG:4326",
    )
    reach_path = tmp_path / "na_sword_v17c_beta_reaches.parquet"
    reaches.to_parquet(reach_path)
    catalog = scan_sword_parquet_dir(tmp_path)

    loaded = load_reaches(
        catalog,
        regions=["na"],
        columns=["reach_id", "river_name"],
        reach_ids=[101.0],
    )

    assert len(loaded) == 1
    assert loaded.iloc[0]["reach_id"] == 101


def test_scan_sword_parquet_dir_infers_region_from_sword_prefix(tmp_path: Path):
    reaches = gpd.GeoDataFrame(
        {"reach_id": [101]},
        geometry=[LineString([(0.0, 0.0), (0.0, 0.08)])],
        crs="EPSG:4326",
    )
    reaches.to_parquet(tmp_path / "sword_NA_v17c_beta_0.0.8_reaches.parquet")

    catalog = scan_sword_parquet_dir(tmp_path)

    assert [item.region for item in catalog.reach_files] == ["na"]
