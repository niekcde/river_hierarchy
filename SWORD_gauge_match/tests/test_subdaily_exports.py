from pathlib import Path

import geopandas as gpd
import pandas as pd
from shapely.geometry import Point

from gauge_sword_match.qa_exports import export_subdaily_hierarchy_package


def test_export_subdaily_hierarchy_package_writes_joined_layers(tmp_path: Path):
    examples = gpd.GeoDataFrame(
        {
            "station_key": ["BR:1", "BR:1", "US:2"],
            "lat": [-1.0, -1.0, 40.0],
            "lon": [-50.0, -50.0, -120.0],
            "down": ["True", None, None],
            "example_id": [5.0, 6.0, 7.0],
        },
        geometry=[Point(-50.0, -1.0), Point(-50.0, -1.0), Point(-120.0, 40.0)],
        crs="EPSG:4326",
    )
    input_gpkg = tmp_path / "examples.gpkg"
    examples.to_file(input_gpkg, layer="hierarchy_examples_filtered", driver="GPKG")

    audit = pd.DataFrame(
        [
            {
                "station_key": "BR:1",
                "country": "BR",
                "provider": "brazil_ana",
                "status": "subdaily_found",
                "daily_available_explicit": True,
                "daily_audit_class": "daily_yes_explicit",
                "resolved_site_number": "16661000",
                "reason_summary": "Daily and subdaily are both exposed.",
            },
            {
                "station_key": "US:2",
                "country": "US",
                "provider": "usgs",
                "status": "resolved_no_subdaily",
                "daily_available_explicit": True,
                "daily_audit_class": "daily_yes_explicit",
                "resolved_site_number": "12345678",
                "reason_summary": "Daily found at a resolved station; no subdaily exposed.",
            },
        ]
    )
    audit_path = tmp_path / "audit.csv"
    audit.to_csv(audit_path, index=False)

    output_path = tmp_path / "subdaily_examples.gpkg"
    export_subdaily_hierarchy_package(input_gpkg, audit_path, output_path)

    assert output_path.exists()

    points_layer = gpd.read_file(output_path, layer="hierarchy_examples_filtered")
    summary_layer = gpd.read_file(output_path, layer="subdaily_station_summary")

    assert len(points_layer) == 3
    assert len(summary_layer) == 2

    br_points = points_layer[points_layer["station_key"] == "BR:1"]
    assert br_points["subdaily_found"].all()
    assert set(br_points["status"]) == {"subdaily_found"}

    br_summary = summary_layer[summary_layer["station_key"] == "BR:1"].iloc[0]
    assert br_summary["subdaily_found"]
    assert br_summary["example_ids"] == "5;6"
    assert int(br_summary["example_count"]) == 2
    assert int(br_summary["occurrence_count"]) == 2


def test_export_subdaily_hierarchy_package_joins_download_manifests(tmp_path: Path):
    examples = gpd.GeoDataFrame(
        {
            "station_key": ["CA:1", "US:2"],
            "lat": [50.0, 40.0],
            "lon": [-80.0, -120.0],
            "down": [None, None],
            "example_id": [5.0, 7.0],
        },
        geometry=[Point(-80.0, 50.0), Point(-120.0, 40.0)],
        crs="EPSG:4326",
    )
    input_gpkg = tmp_path / "examples.gpkg"
    examples.to_file(input_gpkg, layer="hierarchy_examples_filtered", driver="GPKG")

    audit = pd.DataFrame(
        [
            {
                "station_key": "CA:1",
                "country": "CA",
                "provider": "canada_wateroffice",
                "status": "subdaily_found",
                "daily_available_explicit": True,
                "daily_audit_class": "daily_yes_explicit",
                "resolved_site_number": "04HA001",
            },
            {
                "station_key": "US:2",
                "country": "US",
                "provider": "usgs",
                "status": "subdaily_found",
                "daily_available_explicit": True,
                "daily_audit_class": "daily_yes_explicit",
                "resolved_site_number": "12345678",
            },
        ]
    )
    audit_path = tmp_path / "audit.csv"
    audit.to_csv(audit_path, index=False)

    manifests_dir = tmp_path / "subdaily_values"
    (manifests_dir / "CA").mkdir(parents=True, exist_ok=True)
    (manifests_dir / "US").mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "station_key": "CA:1",
                "provider_station_id": "04HA001",
                "download_status": "ok",
                "window_strategy": "latest_10y",
                "selected_start": "2015-01-01T00:00:00Z",
                "selected_end": "2025-01-01T00:00:00Z",
                "selected_row_count": 100,
                "notes": "Canada ok",
            }
        ]
    ).to_csv(manifests_dir / "CA" / "subdaily_download_manifest.csv", index=False)
    pd.DataFrame(
        [
            {
                "station_key": "US:2",
                "provider_station_id": "12345678",
                "download_status": "ok",
                "window_strategy": "full_available_short_record",
                "selected_start": "2020-01-01T00:00:00Z",
                "selected_end": "2020-06-01T00:00:00Z",
                "selected_row_count": 20,
                "notes": "US ok",
            }
        ]
    ).to_csv(manifests_dir / "US" / "subdaily_download_manifest.csv", index=False)

    output_path = tmp_path / "subdaily_examples.gpkg"
    export_subdaily_hierarchy_package(
        input_gpkg,
        audit_path,
        output_path,
        manifests_dir=manifests_dir,
    )

    points_layer = gpd.read_file(output_path, layer="hierarchy_examples_filtered")
    summary_layer = gpd.read_file(output_path, layer="subdaily_station_summary")

    assert set(points_layer["download_status"]) == {"ok"}
    assert set(summary_layer["download_status"]) == {"ok"}
    ca_summary = summary_layer[summary_layer["station_key"] == "CA:1"].iloc[0]
    assert ca_summary["download_window_strategy"] == "latest_10y"
    assert int(ca_summary["download_selected_row_count"]) == 100
    assert ca_summary["download_manifest_country"] == "CA"
