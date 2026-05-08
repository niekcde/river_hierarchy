import zipfile
from pathlib import Path

import geopandas as gpd
import pandas as pd
from shapely.geometry import Point

from gauge_sword_match.us_manual_download import build_manual_us_station_row, load_us_manual_archive


def test_load_us_manual_archive_combines_zip_chunks(tmp_path: Path):
    manual_dir = tmp_path / "manual_download"
    manual_dir.mkdir(parents=True, exist_ok=True)

    payload_a = "\n".join(
        [
            "x,y,id,time_series_id,monitoring_location_id,parameter_code,statistic_id,time,value,unit_of_measure,approval_status,qualifier,last_modified",
            ",,1,ts-1,USGS-15564860,00060,00011,2024-01-01 03:00:00+00:00,66300,ft^3/s,Approved,,2026-01-06 00:41:24.168601+00:00",
            ",,2,ts-1,USGS-15564860,00060,00011,2024-01-01 03:15:00+00:00,66400,ft^3/s,Approved,,2026-01-06 00:41:24.168601+00:00",
        ]
    )
    payload_b = "\n".join(
        [
            "x,y,id,time_series_id,monitoring_location_id,parameter_code,statistic_id,time,value,unit_of_measure,approval_status,qualifier,last_modified",
            ",,3,ts-1,USGS-15564860,00060,00011,2024-01-01 03:15:00+00:00,66500,ft^3/s,Approved,,2026-02-01 00:00:00+00:00",
            ",,4,ts-1,USGS-15564860,00060,00011,2024-01-01 03:30:00+00:00,66600,ft^3/s,Approved,,2026-02-01 00:00:00+00:00",
        ]
    )

    for name, payload in {
        "mlp_continuous_USGS-15564860_20260501T071748.zip": payload_a,
        "mlp_continuous_USGS-15564860_20260501T071926.zip": payload_b,
    }.items():
        with zipfile.ZipFile(manual_dir / name, "w") as archive:
            archive.writestr("primary-time-series.csv", payload)

    frame, notes = load_us_manual_archive("15564860", manual_dir)

    assert len(frame) == 3
    assert frame["time"].min().isoformat() == "2024-01-01T03:00:00+00:00"
    assert frame["time"].max().isoformat() == "2024-01-01T03:30:00+00:00"
    assert frame["discharge"].notna().all()
    assert frame["unit_of_measure"].eq("m3/s").all()
    assert "2 file(s)" in str(notes)


def test_build_manual_us_station_row_assigns_nearest_example(tmp_path: Path):
    examples = gpd.GeoDataFrame(
        {
            "station_key": ["US:4103450", "US:15389100"],
            "lat": [64.7405, 66.651468],
            "lon": [-155.4919, -145.099148],
            "down": [None, True],
            "example_id": [36.0, 41.0],
        },
        geometry=[Point(-155.4919, 64.7405), Point(-145.099148, 66.651468)],
        crs="EPSG:4326",
    )
    gauges = pd.DataFrame(
        [
            {
                "station_key": "US:15453500",
                "station_id": "15453500",
                "lat": 65.875101,
                "lon": -149.720349,
                "country": "US",
            }
        ]
    )
    gauges_path = tmp_path / "gauges_cleaned.parquet"
    gauges.to_parquet(gauges_path, index=False)

    row = build_manual_us_station_row(
        "15453500",
        examples=examples,
        gauges_cleaned_path=gauges_path,
    )

    assert row["station_key"] == "US:15453500"
    assert row["example_id"] == 41
    assert row["status"] == "subdaily_found"
