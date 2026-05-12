from pathlib import Path

import geopandas as gpd
import pandas as pd
from shapely.geometry import LineString, Point

from gauge_sword_match import example_bboxes


def test_write_example_bbox_layer_builds_buffered_rectangles(tmp_path: Path, monkeypatch):
    station_summary_path = tmp_path / "examples.gpkg"
    output_path = tmp_path / "example_bboxes.gpkg"

    stations = gpd.GeoDataFrame(
        {
            "station_key": ["g-101", "g-202"],
            "example_ids": ["101", "202"],
        },
        geometry=[Point(-60.0000, -3.0000), Point(-60.5000, -3.5000)],
        crs="EPSG:4326",
    )
    stations.to_file(station_summary_path, layer="subdaily_station_summary", driver="GPKG")

    pd.DataFrame(
        {
            "example_id": [101, 202],
            "reaches_between": ["[9001, 9002]", "[9003]"],
            "route_found": [True, False],
        }
    ).to_parquet(tmp_path / "hierarchy_example_reach_lists.parquet")
    pd.DataFrame(
        {
            "example_id": [101, 202],
            "sword_region": ["sa", "sa"],
            "route_status": ["route_found", "stations_only"],
            "route_found": [True, False],
        }
    ).to_csv(tmp_path / "hierarchy_example_reach_summary.csv", index=False)

    sword_reaches = gpd.GeoDataFrame(
        {
            "reach_id": [9001, 9002, 9003],
            "sword_region": ["sa", "sa", "sa"],
            "width_obs_p50": [80.0, 120.0, 30.0],
        },
        geometry=[
            LineString([(-60.0015, -3.0010), (-60.0010, -3.0000)]),
            LineString([(-59.9995, -3.0005), (-59.9990, -2.9995)]),
            LineString([(-60.5005, -3.5005), (-60.4995, -3.4995)]),
        ],
        crs="EPSG:4326",
    )

    monkeypatch.setattr(example_bboxes, "scan_sword_parquet_dir", lambda _path: {"dummy": "catalog"})

    def fake_load_reaches(_catalog, *, regions=None, reach_ids=None, columns=None):
        reach_ids = set(reach_ids or [])
        return sword_reaches.loc[sword_reaches["reach_id"].isin(reach_ids)].copy()

    monkeypatch.setattr(example_bboxes, "load_reaches", fake_load_reaches)

    layer = example_bboxes.write_example_bbox_layer(
        station_summary_path,
        output_path,
        buffer_multiplier=1.5,
        fallback_buffer_m=250.0,
    ).sort_values("example_id").reset_index(drop=True)

    assert output_path.exists()
    assert len(layer) == 2

    first = layer.iloc[0]
    second = layer.iloc[1]

    assert first["example_id"] == 101
    assert first["geometry_source"] == "stations_and_reaches"
    assert first["loaded_reach_count"] == 2
    assert first["max_width_m"] == 120.0
    assert first["buffer_m"] == 180.0
    assert first.geometry.geom_type == "Polygon"
    assert first.geometry.covers(stations.iloc[0].geometry)

    assert second["example_id"] == 202
    assert second["geometry_source"] == "stations_only"
    assert second["loaded_reach_count"] == 0
    assert second["buffer_m"] == 250.0
    assert second.geometry.geom_type == "Polygon"
    assert second.geometry.covers(stations.iloc[1].geometry)

    written = gpd.read_file(output_path, layer="example_bboxes").sort_values("example_id").reset_index(drop=True)
    assert list(written["example_id"]) == [101, 202]
