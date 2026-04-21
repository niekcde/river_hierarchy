from pathlib import Path

import pandas as pd

from gauge_sword_match.grdc_io import (
    build_grdc_request_table,
    prepare_grdc_catalog,
    write_grdc_request_station_names,
)


def test_prepare_grdc_catalog_filters_daily_rows_and_normalizes_columns():
    frame = pd.DataFrame(
        [
            {
                "grdc_no": 1010010,
                "station": "Alpha Gauge",
                "river": "Alpha",
                "country_code": "nl",
                "lat": 52.1,
                "long": 4.3,
                "area": 1250.0,
                "d_start": 1980,
                "d_end": 1999,
                "Unnamed: 12": 20,
                "lta_discharge": "12.5",
                "r_volume_yr": "6.307200000000001e-05",
                "r_height_yr": "n.a.",
            },
            {
                "grdc_no": 2020020,
                "station": "Monthly Gauge",
                "river": "Beta",
                "country_code": "de",
                "lat": 51.0,
                "long": 8.0,
                "area": 850.0,
                "d_start": pd.NA,
                "d_end": pd.NA,
                "Unnamed: 12": pd.NA,
                "lta_discharge": "n.a.",
                "r_volume_yr": "n.a.",
                "r_height_yr": "n.a.",
            },
        ]
    )

    prepared = prepare_grdc_catalog(frame, daily_only=True, min_daily_years=1)

    assert len(prepared) == 1
    row = prepared.iloc[0]
    assert row["station_id"] == "1010010"
    assert row["grdc_no"] == 1010010
    assert row["station_name"] == "Alpha Gauge"
    assert row["river_name"] == "Alpha"
    assert row["country"] == "NL"
    assert row["lon"] == 4.3
    assert row["drainage_area"] == 1250.0
    assert row["agency"] == "GRDC"
    assert row["d_yrs"] == 20
    assert row["lta_discharge"] == 12.5
    assert row["r_volume_yr"] == 6.307200000000001e-05
    assert pd.isna(row["r_height_yr"])


def test_build_grdc_request_table_and_station_names_preserve_duplicates(tmp_path: Path):
    best_matches = pd.DataFrame(
        [
            {
                "station_name": "Shared Name",
                "grdc_no": "1111111",
                "station_id": "1111111",
                "country": "FR",
                "river_name": "Alpha",
                "lat": 43.0,
                "lon": 1.0,
                "drainage_area": 1000.0,
                "d_start": 1980,
                "d_end": 2000,
                "d_yrs": 21,
                "d_miss": 1.0,
                "reach_id": 101,
                "sword_region": "eu",
                "sword_node_id": 10,
                "confidence_class": "high",
                "distance_m": 150.0,
                "total_score": 0.92,
                "score_gap": 0.20,
                "review_flag": False,
            },
            {
                "station_name": "Shared Name",
                "grdc_no": "2222222",
                "station_id": "2222222",
                "country": "FR",
                "river_name": "Alpha",
                "lat": 43.1,
                "lon": 1.1,
                "drainage_area": 1200.0,
                "d_start": 1985,
                "d_end": 2005,
                "d_yrs": 21,
                "d_miss": 2.0,
                "reach_id": 202,
                "sword_region": "eu",
                "sword_node_id": 20,
                "confidence_class": "medium",
                "distance_m": 400.0,
                "total_score": 0.74,
                "score_gap": 0.08,
                "review_flag": False,
            },
            {
                "station_name": "Other Name",
                "grdc_no": "3333333",
                "station_id": "3333333",
                "country": "ES",
                "river_name": "Beta",
                "lat": 41.0,
                "lon": -3.0,
                "drainage_area": 900.0,
                "d_start": 1975,
                "d_end": 1995,
                "d_yrs": 21,
                "d_miss": 5.0,
                "reach_id": 303,
                "sword_region": "eu",
                "sword_node_id": 30,
                "confidence_class": "low",
                "distance_m": 950.0,
                "total_score": 0.58,
                "score_gap": 0.03,
                "review_flag": True,
            },
            {
                "station_name": "Unmatched",
                "grdc_no": "4444444",
                "station_id": "4444444",
                "country": "ES",
                "river_name": "Gamma",
                "lat": 40.0,
                "lon": -4.0,
                "drainage_area": 700.0,
                "d_start": 1970,
                "d_end": 1990,
                "d_yrs": 21,
                "d_miss": 3.0,
                "reach_id": pd.NA,
                "sword_region": pd.NA,
                "sword_node_id": pd.NA,
                "confidence_class": "unmatched",
                "distance_m": pd.NA,
                "total_score": pd.NA,
                "score_gap": pd.NA,
                "review_flag": True,
            },
        ]
    )

    request_table = build_grdc_request_table(best_matches)

    assert request_table["grdc_no"].tolist() == ["1111111", "2222222", "3333333"]
    assert request_table["confidence_class"].tolist() == ["high", "medium", "low"]

    names_path = tmp_path / "request_station_names.txt"
    write_grdc_request_station_names(request_table, names_path)

    assert names_path.read_text(encoding="utf-8").splitlines() == [
        "Shared Name",
        "Shared Name",
        "Other Name",
    ]
