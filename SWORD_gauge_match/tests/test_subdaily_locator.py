import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from xml.etree import ElementTree as ET

import pandas as pd

from gauge_sword_match.subdaily_locator.bulgaria import BulgariaAppdClient, locate_bulgaria_subdaily_station
from gauge_sword_match.subdaily_locator.brazil import BrazilAnaHydroClient, locate_brazil_subdaily_station
from gauge_sword_match.subdaily_locator.canada import locate_canada_subdaily_station
from gauge_sword_match.subdaily_locator.chile import locate_chile_subdaily_station
from gauge_sword_match.subdaily_locator.colombia import locate_colombia_subdaily_station
from gauge_sword_match.subdaily_locator.french_guiana import locate_french_guiana_subdaily_station
from gauge_sword_match.subdaily_locator.mekong_mrc import (
    locate_cambodia_subdaily_station,
    locate_laos_subdaily_station,
    locate_thailand_subdaily_station,
)
from gauge_sword_match.subdaily_locator.niger_basin_abn import locate_mali_subdaily_station
from gauge_sword_match.subdaily_locator.nigeria import locate_nigeria_subdaily_station
from gauge_sword_match.subdaily_locator.russia import locate_russia_subdaily_station
from gauge_sword_match.subdaily_locator.runner import locate_subdaily_from_hierarchy_examples
from gauge_sword_match.subdaily_locator.seeds import load_hierarchy_example_station_seeds
from gauge_sword_match.subdaily_locator.usgs import MonitoringLocation, locate_usgs_subdaily_station


def test_load_hierarchy_example_station_seeds_aggregates_duplicates(tmp_path: Path):
    gpkg_path = tmp_path / "hierarchy_examples_filtered.gpkg"
    with sqlite3.connect(gpkg_path) as connection:
        connection.execute(
            """
            CREATE TABLE hierarchy_examples_filtered (
                station_key TEXT,
                lat REAL,
                lon REAL,
                down TEXT,
                example_id REAL
            )
            """
        )
        connection.executemany(
            "INSERT INTO hierarchy_examples_filtered VALUES (?, ?, ?, ?, ?)",
            [
                ("US:4105800", 41.1, -88.2, "True", 30.0),
                ("US:4105800", 41.1, -88.2, None, 30.0),
                ("US:13018750", 43.0, -112.0, "", 31.0),
            ],
        )

    seeds = load_hierarchy_example_station_seeds(gpkg_path)

    assert seeds["station_key"].tolist() == ["US:13018750", "US:4105800"]
    assert seeds["occurrence_count"].tolist() == [1, 2]
    assert seeds["example_ids"].tolist() == ["31", "30"]
    assert seeds["down_values"].tolist() == ["", "True"]
    assert seeds["source_station_id"].tolist() == ["13018750", "4105800"]


def test_locate_usgs_subdaily_station_zero_pads_station_id():
    row = pd.Series(
        {
            "station_key": "US:4105800",
            "country": "US",
            "source_station_id": "4105800",
            "lat": 41.0,
            "lon": -88.0,
            "occurrence_count": 1,
            "example_ids": "30",
            "down_values": "True",
        }
    )

    class FakeClient:
        def fetch_discharge_metadata(self, monitoring_location_id: str):
            if monitoring_location_id == "USGS-04105800":
                return [
                    {
                        "geometry": {"type": "Point", "coordinates": [-88.0005, 41.0005]},
                        "properties": {
                            "monitoring_location_id": monitoring_location_id,
                            "computation_identifier": "Instantaneous",
                            "computation_period_identifier": "Points",
                            "begin": "2001-01-01T00:00:00Z",
                            "end": "2025-01-01T00:00:00Z",
                            "primary": "Primary",
                        },
                    }
                ]
            return []

        def fetch_monitoring_locations_by_number(self, monitoring_location_number: str):
            return []

        def fetch_monitoring_locations_nearby(self, *, lon: float, lat: float, radius_m: float, limit: int = 25):
            raise AssertionError("Nearby search should not be used for zero-padded direct resolution")

    result = locate_usgs_subdaily_station(row, client=FakeClient())

    assert result["status"] == "subdaily_found"
    assert result["resolution_method"] == "zero_padded_site_number"
    assert result["candidate_site_numbers"] == "4105800,04105800"
    assert result["resolved_monitoring_location_id"] == "USGS-04105800"
    assert result["resolved_site_number"] == "04105800"
    assert result["subdaily_discharge_found"] is True
    assert result["instantaneous_series_count"] == 1


def test_locate_usgs_subdaily_station_rejects_far_zero_padded_match_without_inventory():
    row = pd.Series(
        {
            "station_key": "US:4105810",
            "country": "US",
            "source_station_id": "4105810",
            "lat": 62.1778,
            "lon": -150.1772,
            "occurrence_count": 1,
            "example_ids": "37",
            "down_values": "",
        }
    )

    class FakeClient:
        def fetch_discharge_metadata(self, monitoring_location_id: str):
            if monitoring_location_id == "USGS-04105810":
                return [
                    {
                        "geometry": {"type": "Point", "coordinates": [-85.355, 42.289]},
                        "properties": {
                            "monitoring_location_id": monitoring_location_id,
                            "computation_identifier": "Mean",
                            "computation_period_identifier": "Daily",
                            "begin": "2001-01-01T00:00:00Z",
                            "end": "2025-01-01T00:00:00Z",
                            "primary": "Primary",
                        },
                    }
                ]
            return []

        def fetch_monitoring_locations_by_number(self, monitoring_location_number: str):
            return []

        def fetch_monitoring_locations_nearby(self, *, lon: float, lat: float, radius_m: float, limit: int = 25):
            return []

    result = locate_usgs_subdaily_station(row, client=FakeClient(), max_resolution_distance_m=5_000.0)

    assert result["status"] == "unresolved"
    assert result["resolution_method"] == "unresolved"
    assert result["resolved_monitoring_location_id"] is None
    assert result["subdaily_discharge_found"] is False


def test_locate_subdaily_from_hierarchy_examples_prefers_inventory_station_over_bad_zero_padding(tmp_path: Path):
    gpkg_path = tmp_path / "hierarchy_examples_filtered.gpkg"
    inventory_path = tmp_path / "gauges_cleaned.csv"
    with sqlite3.connect(gpkg_path) as connection:
        connection.execute(
            """
            CREATE TABLE hierarchy_examples_filtered (
                station_key TEXT,
                lat REAL,
                lon REAL,
                down TEXT,
                example_id REAL
            )
            """
        )
        connection.execute(
            "INSERT INTO hierarchy_examples_filtered VALUES (?, ?, ?, ?, ?)",
            ("US:4105800", 61.5442, -150.5147, "True", 37.0),
        )

    pd.DataFrame(
        [
            {
                "station_id": "15294350",
                "station_name": pd.NA,
                "country": "US",
                "lat": 61.544180,
                "lon": -150.514733,
            }
        ]
    ).to_csv(inventory_path, index=False)

    class FakeClient:
        def fetch_discharge_metadata(self, monitoring_location_id: str):
            if monitoring_location_id == "USGS-15294350":
                return [
                    {
                        "geometry": {"type": "Point", "coordinates": [-150.514733, 61.544180]},
                        "properties": {
                            "monitoring_location_id": monitoring_location_id,
                            "computation_identifier": "Instantaneous",
                            "computation_period_identifier": "Points",
                            "begin": "2011-01-01T00:00:00Z",
                            "end": "2025-01-01T00:00:00Z",
                            "primary": "Primary",
                        },
                    }
                ]
            if monitoring_location_id == "USGS-04105800":
                return [
                    {
                        "geometry": {"type": "Point", "coordinates": [-85.3, 42.2]},
                        "properties": {
                            "monitoring_location_id": monitoring_location_id,
                            "computation_identifier": "Mean",
                            "computation_period_identifier": "Daily",
                            "begin": "2001-01-01T00:00:00Z",
                            "end": "2025-01-01T00:00:00Z",
                            "primary": "Primary",
                        },
                    }
                ]
            return []

        def fetch_monitoring_locations_by_number(self, monitoring_location_number: str):
            return []

        def fetch_monitoring_locations_nearby(self, *, lon: float, lat: float, radius_m: float, limit: int = 25):
            return []

    results = locate_subdaily_from_hierarchy_examples(
        gpkg_path,
        country="US",
        client=FakeClient(),
        inventory_path=inventory_path,
    )

    assert len(results) == 1
    record = results.iloc[0].to_dict()
    assert record["inventory_station_id"] == "15294350"
    assert record["inventory_resolution_method"] == "inventory_nearest_gauge"
    assert record["status"] == "subdaily_found"
    assert record["resolution_method"] == "inventory_nearest_gauge"
    assert record["resolved_site_number"] == "15294350"
    assert record["resolved_monitoring_location_id"] == "USGS-15294350"


def test_locate_subdaily_from_hierarchy_examples_uses_nearby_usgs_fallback(tmp_path: Path):
    gpkg_path = tmp_path / "hierarchy_examples_filtered.gpkg"
    with sqlite3.connect(gpkg_path) as connection:
        connection.execute(
            """
            CREATE TABLE hierarchy_examples_filtered (
                station_key TEXT,
                lat REAL,
                lon REAL,
                down TEXT,
                example_id REAL
            )
            """
        )
        connection.execute(
            "INSERT INTO hierarchy_examples_filtered VALUES (?, ?, ?, ?, ?)",
            ("US:4116192", 40.0, -105.0, "", 41.0),
        )

    class FakeClient:
        def fetch_discharge_metadata(self, monitoring_location_id: str):
            if monitoring_location_id == "USGS-06700000":
                return [
                    {
                        "geometry": {"type": "Point", "coordinates": [-105.0008, 40.0008]},
                        "properties": {
                            "monitoring_location_id": monitoring_location_id,
                            "computation_identifier": "Instantaneous",
                            "computation_period_identifier": "Points",
                            "begin": "2010-01-01T00:00:00Z",
                            "end": "2025-02-01T00:00:00Z",
                            "primary": "Primary",
                        },
                    }
                ]
            return []

        def fetch_monitoring_locations_by_number(self, monitoring_location_number: str):
            return []

        def fetch_monitoring_locations_nearby(self, *, lon: float, lat: float, radius_m: float, limit: int = 25):
            return [
                MonitoringLocation(
                    monitoring_location_id="USGS-06700000",
                    monitoring_location_number="06700000",
                    monitoring_location_name="Nearby Gauge",
                    lat=40.0008,
                    lon=-105.0008,
                )
            ]

    results = locate_subdaily_from_hierarchy_examples(
        gpkg_path,
        country="US",
        client=FakeClient(),
    )

    assert len(results) == 1
    record = results.iloc[0].to_dict()
    assert record["station_key"] == "US:4116192"
    assert record["status"] == "subdaily_found"
    assert record["resolution_method"] == "spatial_nearest_discharge_station"
    assert record["resolved_site_number"] == "06700000"
    assert record["resolved_station_name"] == "Nearby Gauge"
    assert record["subdaily_discharge_found"] is True


def test_locate_canada_subdaily_station_for_exact_inventory_match():
    row = pd.Series(
        {
            "station_key": "CA:04HA001",
            "country": "CA",
            "source_station_id": "04HA001",
            "lat": 51.33056,
            "lon": -83.83333,
            "occurrence_count": 1,
            "example_ids": "12",
            "down_values": "True",
            "inventory_station_id": "04HA001",
            "inventory_station_key": "CA:04HA001",
            "inventory_station_name": pd.NA,
            "inventory_resolution_method": "inventory_exact_station_key",
            "inventory_distance_m": 0.0,
        }
    )

    class FakeClient:
        def fetch_discharge_unit_values(self, station_id: str, *, start_datetime_utc: datetime, end_datetime_utc: datetime):
            assert station_id == "04HA001"
            return pd.DataFrame(
                {
                    "Date": ["2026-04-01 00:00:00", "2026-04-02 00:15:00"],
                    "Value": [10.0, 11.0],
                }
            )

        def fetch_discharge_daily_values(self, station_id: str, *, start_date, end_date):
            assert station_id == "04HA001"
            return pd.DataFrame(
                {
                    "Date": ["2026-04-01", "2026-04-02"],
                    "Value": [10.5, 10.8],
                }
            )

    result = locate_canada_subdaily_station(
        row,
        client=FakeClient(),
        now_utc=datetime(2026, 4, 28, tzinfo=timezone.utc),
    )

    assert result["provider"] == "canada_wateroffice"
    assert result["status"] == "subdaily_found"
    assert result["resolution_method"] == "inventory_exact_station_key"
    assert result["resolved_site_number"] == "04HA001"
    assert result["daily_series_count"] == 1
    assert result["instantaneous_series_count"] == 1
    assert result["subdaily_discharge_found"] is True
    assert result["daily_coverage_type"] == "recent_window"


def test_locate_canada_subdaily_station_for_nearest_inventory_match():
    row = pd.Series(
        {
            "station_key": "CA:4214590",
            "country": "CA",
            "source_station_id": "4214590",
            "lat": 51.85,
            "lon": -82.97,
            "occurrence_count": 1,
            "example_ids": "56",
            "down_values": "True",
            "inventory_station_id": "04HA002",
            "inventory_station_key": "CA:04HA002",
            "inventory_station_name": pd.NA,
            "inventory_resolution_method": "inventory_nearest_gauge",
            "inventory_distance_m": 563.97,
        }
    )

    class FakeClient:
        def fetch_discharge_unit_values(self, station_id: str, *, start_datetime_utc: datetime, end_datetime_utc: datetime):
            assert station_id == "04HA002"
            return pd.DataFrame(
                {
                    "Date": ["2026-04-01 00:00:00", "2026-04-02 00:15:00"],
                    "Value": [10.0, 11.0],
                }
            )

        def fetch_discharge_daily_values(self, station_id: str, *, start_date, end_date):
            assert station_id == "04HA002"
            return pd.DataFrame(
                {
                    "Date": ["2026-04-01", "2026-04-02"],
                    "Value": [10.5, 10.8],
                }
            )

    result = locate_canada_subdaily_station(
        row,
        client=FakeClient(),
        now_utc=datetime(2026, 4, 28, tzinfo=timezone.utc),
    )

    assert result["status"] == "subdaily_found"
    assert result["resolution_method"] == "inventory_nearest_gauge"
    assert result["resolved_site_number"] == "04HA002"
    assert result["subdaily_discharge_found"] is True
    assert result["daily_coverage_type"] == "recent_window"


def test_locate_canada_subdaily_station_historical_daily_only():
    row = pd.Series(
        {
            "station_key": "CA:4203760",
            "country": "CA",
            "source_station_id": "4203760",
            "lat": 61.49,
            "lon": -134.78,
            "occurrence_count": 1,
            "example_ids": "42",
            "down_values": "True",
            "inventory_station_id": "09AF001",
            "inventory_station_key": "CA:09AF001",
            "inventory_station_name": pd.NA,
            "inventory_resolution_method": "inventory_nearest_gauge",
            "inventory_distance_m": 213.0,
        }
    )

    class FakeClient:
        def fetch_discharge_unit_values(self, station_id: str, *, start_datetime_utc: datetime, end_datetime_utc: datetime):
            assert station_id == "09AF001"
            return pd.DataFrame()

        def fetch_discharge_daily_values(self, station_id: str, *, start_date, end_date):
            assert station_id == "09AF001"
            return pd.DataFrame(
                {
                    "Date": ["1971-01-01", "1973-12-31"],
                    "Value": [10.0, 11.0],
                }
            )

    result = locate_canada_subdaily_station(
        row,
        client=FakeClient(),
        now_utc=datetime(2026, 4, 28, tzinfo=timezone.utc),
    )

    assert result["status"] == "resolved_historical_daily_only"
    assert result["resolution_method"] == "inventory_nearest_gauge"
    assert result["resolved_site_number"] == "09AF001"
    assert result["daily_series_count"] == 1
    assert result["instantaneous_series_count"] == 0
    assert result["daily_coverage_type"] == "historical_only"
    assert result["daily_begin"].startswith("1971-01-01")
    assert result["daily_end"].startswith("1973-12-31")


def test_locate_canada_subdaily_station_leaves_grdc_only_seed_unresolved():
    row = pd.Series(
        {
            "station_key": "CA:4214590",
            "country": "CA",
            "source_station_id": "4214590",
            "lat": 51.85,
            "lon": -82.97,
            "occurrence_count": 1,
            "example_ids": "13",
            "down_values": "",
        }
    )

    class FakeClient:
        def fetch_discharge_unit_values(self, station_id: str, *, start_datetime_utc: datetime, end_datetime_utc: datetime):
            raise AssertionError("Client should not be called for a non-direct Canada inventory seed")

        def fetch_discharge_daily_values(self, station_id: str, *, start_date, end_date):
            raise AssertionError("Client should not be called for a non-direct Canada inventory seed")

    result = locate_canada_subdaily_station(
        row,
        client=FakeClient(),
        now_utc=datetime(2026, 4, 28, tzinfo=timezone.utc),
    )

    assert result["status"] == "unresolved"
    assert result["resolution_method"] == "inventory_not_found"
    assert result["monitoring_location_found"] is False
    assert result["subdaily_discharge_found"] is False


def test_brazil_ana_client_parses_subdaily_and_daily_xml():
    class FakeClient(BrazilAnaHydroClient):
        def _get_xml_root(self, operation: str, params):
            if operation == "DadosHidrometeorologicos":
                return ET.fromstring(
                    """
                    <DataTable xmlns="http://MRCS/">
                      <DocumentElement xmlns="">
                        <DadosHidrometereologicos>
                          <CodEstacao>58880001</CodEstacao>
                          <DataHora>2026-04-02 12:00:00</DataHora>
                          <Vazao>123.4</Vazao>
                        </DadosHidrometereologicos>
                        <DadosHidrometereologicos>
                          <CodEstacao>58880001</CodEstacao>
                          <DataHora>2026-04-02 13:00:00</DataHora>
                          <Vazao />
                        </DadosHidrometereologicos>
                      </DocumentElement>
                    </DataTable>
                    """
                )
            return ET.fromstring(
                """
                <DataTable xmlns="http://MRCS/">
                  <DocumentElement xmlns="">
                    <SerieHistorica>
                      <EstacaoCodigo>58880001</EstacaoCodigo>
                      <NivelConsistencia>1</NivelConsistencia>
                      <DataHora>2026-04-01 00:00:00</DataHora>
                      <Vazao01>10.1</Vazao01>
                      <Vazao02>10.2</Vazao02>
                    </SerieHistorica>
                    <SerieHistorica>
                      <EstacaoCodigo>58880001</EstacaoCodigo>
                      <NivelConsistencia>2</NivelConsistencia>
                      <DataHora>2026-04-01 00:00:00</DataHora>
                      <Vazao01>99.9</Vazao01>
                    </SerieHistorica>
                  </DocumentElement>
                </DataTable>
                """
            )

    client = FakeClient()
    subdaily = client.fetch_subdaily_discharge_values("58880001", start_date=datetime(2026, 4, 1).date(), end_date=datetime(2026, 4, 28).date())
    daily = client.fetch_daily_discharge_values("58880001", start_date=datetime(2026, 4, 1).date(), end_date=datetime(2026, 4, 28).date())

    assert subdaily["DateTime"].tolist() == ["2026-04-02 12:00:00"]
    assert subdaily["Vazao"].tolist() == [123.4]
    assert daily["Date"].tolist() == ["2026-04-01", "2026-04-02"]
    assert daily["Vazao"].tolist() == [10.1, 10.2]
    assert daily["NivelConsistencia"].tolist() == [1, 1]


def test_locate_brazil_subdaily_station_for_inventory_match():
    row = pd.Series(
        {
            "station_key": "BR:3652880",
            "country": "BR",
            "source_station_id": "3652880",
            "lat": -21.64528,
            "lon": -41.75222,
            "occurrence_count": 1,
            "example_ids": "3",
            "down_values": "",
            "inventory_station_id": "58880001",
            "inventory_station_key": "BR:58880001",
            "inventory_station_name": pd.NA,
            "inventory_resolution_method": "inventory_nearest_gauge",
            "inventory_distance_m": 3.04,
        }
    )

    class FakeClient:
        def fetch_subdaily_discharge_values(self, station_id: str, *, start_date, end_date):
            assert station_id == "58880001"
            return pd.DataFrame({"DateTime": ["2026-04-01 12:00:00"], "Vazao": [123.4]})

        def fetch_daily_discharge_values(self, station_id: str, *, start_date, end_date):
            assert station_id == "58880001"
            return pd.DataFrame({"Date": ["2026-04-01"], "Vazao": [125.0], "NivelConsistencia": [1]})

    result = locate_brazil_subdaily_station(
        row,
        client=FakeClient(),
        now_utc=datetime(2026, 4, 28, tzinfo=timezone.utc),
    )

    assert result["provider"] == "brazil_ana"
    assert result["status"] == "subdaily_found"
    assert result["resolution_method"] == "inventory_nearest_gauge"
    assert result["resolved_site_number"] == "58880001"
    assert result["instantaneous_series_count"] == 1
    assert result["daily_series_count"] == 1
    assert result["daily_coverage_type"] == "recent_window"


def test_locate_brazil_subdaily_station_historical_daily_only():
    row = pd.Series(
        {
            "station_key": "BR:3652450",
            "country": "BR",
            "source_station_id": "3652450",
            "lat": -16.13860,
            "lon": -40.30690,
            "occurrence_count": 1,
            "example_ids": "4",
            "down_values": "",
            "inventory_station_id": "54780000",
            "inventory_station_key": "BR:54780000",
            "inventory_station_name": pd.NA,
            "inventory_resolution_method": "inventory_nearest_gauge",
            "inventory_distance_m": 322.86,
        }
    )

    class FakeClient:
        def fetch_subdaily_discharge_values(self, station_id: str, *, start_date, end_date):
            assert station_id == "54780000"
            return pd.DataFrame()

        def fetch_daily_discharge_values(self, station_id: str, *, start_date, end_date):
            assert station_id == "54780000"
            return pd.DataFrame(
                {
                    "Date": ["1990-01-01", "1990-01-02"],
                    "Vazao": [10.0, 11.0],
                    "NivelConsistencia": [1, 1],
                }
            )

    result = locate_brazil_subdaily_station(
        row,
        client=FakeClient(),
        now_utc=datetime(2026, 4, 28, tzinfo=timezone.utc),
    )

    assert result["status"] == "resolved_historical_daily_only"
    assert result["resolved_site_number"] == "54780000"
    assert result["daily_series_count"] == 1
    assert result["instantaneous_series_count"] == 0
    assert result["daily_coverage_type"] == "historical_only"
    assert result["daily_begin"].startswith("1990-01-01")
    assert result["daily_end"].startswith("1990-01-02")


def test_locate_brazil_subdaily_station_notes_blank_telemetric_discharge():
    row = pd.Series(
        {
            "station_key": "BR:3635010",
            "country": "BR",
            "source_station_id": "3635010",
            "lat": -4.8972,
            "lon": -60.0253,
            "occurrence_count": 1,
            "example_ids": "10",
            "down_values": "True",
            "inventory_station_id": "15860000",
            "inventory_station_key": "BR:15860000",
            "inventory_station_name": pd.NA,
            "inventory_resolution_method": "inventory_nearest_gauge",
            "inventory_distance_m": 0.0,
        }
    )

    class FakeClient:
        def fetch_subdaily_discharge_values(self, station_id: str, *, start_date, end_date):
            assert station_id == "15860000"
            return pd.DataFrame(
                {
                    "DateTime": ["2026-04-28 08:45:00", "2026-04-28 08:30:00"],
                    "Vazao": [None, None],
                    "has_discharge_value": [False, False],
                }
            )

        def fetch_daily_discharge_values(self, station_id: str, *, start_date, end_date):
            assert station_id == "15860000"
            return pd.DataFrame()

    result = locate_brazil_subdaily_station(
        row,
        client=FakeClient(),
        now_utc=datetime(2026, 4, 28, tzinfo=timezone.utc),
    )

    assert result["status"] == "resolved_no_discharge"
    assert result["subdaily_discharge_found"] is False
    assert "blank" in str(result["notes"]).lower()


def test_locate_subdaily_from_hierarchy_examples_supports_canada_inventory_reconciliation(tmp_path: Path):
    gpkg_path = tmp_path / "hierarchy_examples_filtered.gpkg"
    inventory_path = tmp_path / "gauges_cleaned.csv"
    with sqlite3.connect(gpkg_path) as connection:
        connection.execute(
            """
            CREATE TABLE hierarchy_examples_filtered (
                station_key TEXT,
                lat REAL,
                lon REAL,
                down TEXT,
                example_id REAL
            )
            """
        )
        connection.executemany(
            "INSERT INTO hierarchy_examples_filtered VALUES (?, ?, ?, ?, ?)",
            [
                ("CA:04HA001", 51.33056, -83.83333, "True", 1.0),
                ("CA:4214590", 51.85, -82.97, "", 2.0),
            ],
        )

    pd.DataFrame(
        [
            {
                "station_id": "04HA001",
                "country": "CA",
                "lat": 51.33056,
                "lon": -83.83333,
                "station_name": pd.NA,
            },
            {
                "station_id": "04HA002",
                "country": "CA",
                "lat": 51.85381,
                "lon": -82.97542,
                "station_name": pd.NA,
            },
        ]
    ).to_csv(inventory_path, index=False)

    class FakeClient:
        def fetch_discharge_unit_values(self, station_id: str, *, start_datetime_utc: datetime, end_datetime_utc: datetime):
            return pd.DataFrame({"Date": ["2026-04-01 00:00:00"], "Value": [10.0]})

        def fetch_discharge_daily_values(self, station_id: str, *, start_date, end_date):
            return pd.DataFrame({"Date": ["2026-04-01"], "Value": [10.2]})

    results = locate_subdaily_from_hierarchy_examples(
        gpkg_path,
        country="CA",
        client=FakeClient(),
        inventory_path=inventory_path,
    )

    assert len(results) == 2
    by_key = {row["station_key"]: row for row in results.to_dict(orient="records")}
    assert by_key["CA:04HA001"]["status"] == "subdaily_found"
    assert by_key["CA:04HA001"]["inventory_resolution_method"] == "inventory_exact_station_key"
    assert by_key["CA:4214590"]["status"] == "subdaily_found"
    assert by_key["CA:4214590"]["inventory_resolution_method"] == "inventory_nearest_gauge"
    assert by_key["CA:4214590"]["resolved_site_number"] == "04HA002"


def test_locate_subdaily_from_hierarchy_examples_supports_canada_curated_override(tmp_path: Path):
    gpkg_path = tmp_path / "hierarchy_examples_filtered.gpkg"
    inventory_path = tmp_path / "gauges_cleaned.csv"
    with sqlite3.connect(gpkg_path) as connection:
        connection.execute(
            """
            CREATE TABLE hierarchy_examples_filtered (
                station_key TEXT,
                lat REAL,
                lon REAL,
                down TEXT,
                example_id REAL
            )
            """
        )
        connection.execute(
            "INSERT INTO hierarchy_examples_filtered VALUES (?, ?, ?, ?, ?)",
            ("CA:4208871", 58.2, -111.39, "True", 45.0),
        )

    pd.DataFrame(
        [
            {
                "station_id": "07DD001",
                "country": "CA",
                "lat": 58.31264,
                "lon": -111.51510,
                "station_name": pd.NA,
            }
        ]
    ).to_csv(inventory_path, index=False)

    class FakeClient:
        def fetch_discharge_unit_values(self, station_id: str, *, start_datetime_utc: datetime, end_datetime_utc: datetime):
            assert station_id == "07DD001"
            return pd.DataFrame({"Date": ["2026-04-01 00:00:00"], "Value": [10.0]})

        def fetch_discharge_daily_values(self, station_id: str, *, start_date, end_date):
            assert station_id == "07DD001"
            return pd.DataFrame({"Date": ["2026-04-01"], "Value": [10.2]})

    results = locate_subdaily_from_hierarchy_examples(
        gpkg_path,
        country="CA",
        client=FakeClient(),
        inventory_path=inventory_path,
    )

    assert len(results) == 1
    record = results.iloc[0].to_dict()
    assert record["inventory_station_id"] == "07DD001"
    assert record["inventory_resolution_method"] == "inventory_curated_override"
    assert record["status"] == "subdaily_found"
    assert record["resolved_site_number"] == "07DD001"


def test_locate_subdaily_from_hierarchy_examples_supports_brazil_inventory_reconciliation(tmp_path: Path):
    gpkg_path = tmp_path / "hierarchy_examples_filtered.gpkg"
    inventory_path = tmp_path / "gauges_cleaned.csv"
    with sqlite3.connect(gpkg_path) as connection:
        connection.execute(
            """
            CREATE TABLE hierarchy_examples_filtered (
                station_key TEXT,
                lat REAL,
                lon REAL,
                down TEXT,
                example_id REAL
            )
            """
        )
        connection.execute(
            "INSERT INTO hierarchy_examples_filtered VALUES (?, ?, ?, ?, ?)",
            ("BR:3652880", -21.64528, -41.75222, "", 3.0),
        )

    pd.DataFrame(
        [
            {
                "station_id": "58880001",
                "country": "BR",
                "lat": -21.64530,
                "lon": -41.75220,
                "station_name": pd.NA,
            }
        ]
    ).to_csv(inventory_path, index=False)

    class FakeClient:
        def fetch_subdaily_discharge_values(self, station_id: str, *, start_date, end_date):
            assert station_id == "58880001"
            return pd.DataFrame({"DateTime": ["2026-04-01 00:00:00"], "Vazao": [10.0]})

        def fetch_daily_discharge_values(self, station_id: str, *, start_date, end_date):
            assert station_id == "58880001"
            return pd.DataFrame({"Date": ["2026-04-01"], "Vazao": [10.2], "NivelConsistencia": [1]})

    results = locate_subdaily_from_hierarchy_examples(
        gpkg_path,
        country="BR",
        client=FakeClient(),
        inventory_path=inventory_path,
    )

    assert len(results) == 1
    record = results.iloc[0].to_dict()
    assert record["inventory_station_id"] == "58880001"
    assert record["inventory_resolution_method"] == "inventory_nearest_gauge"
    assert record["status"] == "subdaily_found"
    assert record["resolved_site_number"] == "58880001"


def test_locate_subdaily_from_hierarchy_examples_supports_brazil_curated_override(tmp_path: Path):
    gpkg_path = tmp_path / "hierarchy_examples_filtered.gpkg"
    inventory_path = tmp_path / "gauges_cleaned.csv"
    with sqlite3.connect(gpkg_path) as connection:
        connection.execute(
            """
            CREATE TABLE hierarchy_examples_filtered (
                station_key TEXT,
                lat REAL,
                lon REAL,
                down TEXT,
                example_id REAL
            )
            """
        )
        connection.execute(
            "INSERT INTO hierarchy_examples_filtered VALUES (?, ?, ?, ?, ?)",
            ("BR:3652455", -15.9483, -39.5236, "True", 4.0),
        )

    pd.DataFrame(
        [
            {
                "station_id": "54950000",
                "country": "BR",
                "lat": -16.0147,
                "lon": -39.4253,
                "station_name": pd.NA,
            }
        ]
    ).to_csv(inventory_path, index=False)

    class FakeClient:
        def fetch_subdaily_discharge_values(self, station_id: str, *, start_date, end_date):
            assert station_id == "54950000"
            return pd.DataFrame()

        def fetch_daily_discharge_values(self, station_id: str, *, start_date, end_date):
            assert station_id == "54950000"
            return pd.DataFrame({"Date": ["2025-02-01"], "Vazao": [201.164], "NivelConsistencia": [1]})

    results = locate_subdaily_from_hierarchy_examples(
        gpkg_path,
        country="BR",
        client=FakeClient(),
        inventory_path=inventory_path,
    )

    assert len(results) == 1
    record = results.iloc[0].to_dict()
    assert record["inventory_station_id"] == "54950000"
    assert record["inventory_resolution_method"] == "inventory_curated_override"
    assert record["status"] == "resolved_no_subdaily"
    assert record["resolved_site_number"] == "54950000"


def test_locate_subdaily_from_hierarchy_examples_supports_additional_brazil_manual_overrides(tmp_path: Path):
    gpkg_path = tmp_path / "hierarchy_examples_filtered.gpkg"
    inventory_path = tmp_path / "gauges_cleaned.csv"
    with sqlite3.connect(gpkg_path) as connection:
        connection.execute(
            """
            CREATE TABLE hierarchy_examples_filtered (
                station_key TEXT,
                lat REAL,
                lon REAL,
                down TEXT,
                example_id REAL
            )
            """
        )
        connection.executemany(
            "INSERT INTO hierarchy_examples_filtered VALUES (?, ?, ?, ?, ?)",
            [
                ("BR:3636201", -1.0650, -57.0614, "True", 11.0),
                ("BR:3637150", -6.0453, -57.6428, "True", 9.0),
                ("BR:3637152", -5.1525, -56.8539, "True", 9.0),
            ],
        )

    pd.DataFrame(
        [
            {
                "station_id": "16661000",
                "country": "BR",
                "lat": -1.3706,
                "lon": -56.8519,
                "station_name": pd.NA,
            },
            {
                "station_id": "17710000",
                "country": "BR",
                "lat": -4.6156,
                "lon": -56.3250,
                "station_name": pd.NA,
            },
        ]
    ).to_csv(inventory_path, index=False)

    class FakeClient:
        def fetch_subdaily_discharge_values(self, station_id: str, *, start_date, end_date):
            if station_id == "16661000":
                return pd.DataFrame({"DateTime": ["2026-04-01 03:00:00"], "Vazao": [12.0]})
            if station_id == "17710000":
                return pd.DataFrame({"DateTime": ["2026-04-02 06:00:00"], "Vazao": [8.5]})
            raise AssertionError(f"Unexpected station {station_id}")

        def fetch_daily_discharge_values(self, station_id: str, *, start_date, end_date):
            if station_id == "16661000":
                return pd.DataFrame({"Date": ["2026-04-01"], "Vazao": [11.8], "NivelConsistencia": [1]})
            if station_id == "17710000":
                return pd.DataFrame({"Date": ["2026-04-02"], "Vazao": [8.2], "NivelConsistencia": [1]})
            raise AssertionError(f"Unexpected station {station_id}")

    results = locate_subdaily_from_hierarchy_examples(
        gpkg_path,
        country="BR",
        client=FakeClient(),
        inventory_path=inventory_path,
    )

    assert len(results) == 3
    by_key = {row["station_key"]: row for row in results.to_dict(orient="records")}

    assert by_key["BR:3636201"]["inventory_station_id"] == "16661000"
    assert by_key["BR:3636201"]["inventory_resolution_method"] == "inventory_curated_override"
    assert by_key["BR:3636201"]["status"] == "subdaily_found"

    assert by_key["BR:3637150"]["inventory_station_id"] == "17710000"
    assert by_key["BR:3637150"]["inventory_resolution_method"] == "inventory_curated_override"
    assert by_key["BR:3637150"]["status"] == "subdaily_found"

    assert by_key["BR:3637152"]["inventory_station_id"] == "17710000"
    assert by_key["BR:3637152"]["inventory_resolution_method"] == "inventory_curated_override"
    assert by_key["BR:3637152"]["status"] == "subdaily_found"


def test_locate_chile_subdaily_station_for_inventory_match():
    row = pd.Series(
        {
            "station_key": "CL:3179530",
            "country": "CL",
            "source_station_id": "3179530",
            "lat": -37.7106,
            "lon": -71.9017,
            "occurrence_count": 1,
            "example_ids": "3",
            "down_values": "",
            "inventory_station_id": "08317001",
            "inventory_station_key": "CL:08317001",
            "inventory_station_name": pd.NA,
            "inventory_resolution_method": "inventory_nearest_gauge",
            "inventory_distance_m": 0.0,
        }
    )

    class FakeClient:
        def fetch_inventory_station_records(self, station_id: str):
            assert station_id == "08317001"
            return [
                {
                    "COD_BNA": "08317001",
                    "NOM_ESTACION": "Rio Biobio En Rucalhue",
                    "TIPO_ESTACION": "Fluviometr\u00e9icas",
                    "VIGENCIA": "Vigentes",
                    "INSTITUCION": "DGA",
                    "LATITUD": -37.70965618,
                    "LONGITUD": -71.90231888,
                }
            ]

        def fetch_alert_records_for_station_prefix(self, station_id: str):
            assert station_id == "08317001"
            return [
                {
                    "SITMOP_PROD.SITMOP_DESA.TG_RED_HIDROMETEO.CODBNA": "08317001-8",
                    "SITMOP_PROD.SITMOP_DESA.TG_RED_HIDROMETEO.NOMBRERED": "rio biobio en rucalhue",
                    "SITMOP_PROD.SITMOP_DESA.TG_RED_HIDROMETEO.FLUVIOMETRICA": "vig",
                    "SITMOP_PROD.SITMOP_DESA.TG_RED_HIDROMETEO.TIPOTRASMISION": "OTR",
                    "SITMOP_PROD.SDE.V_DGA_GIS_ALERTAS.mod_codest": "08317001-8",
                    "SITMOP_PROD.SDE.V_DGA_GIS_ALERTAS.mod_fechra": 1777379207000,
                    "SITMOP_PROD.SDE.V_DGA_GIS_ALERTAS.mod_valor": 0.0,
                    "SITMOP_PROD.SDE.V_DGA_GIS_ALERTAS.mod_alerta": 0.0,
                }
            ]

    result = locate_chile_subdaily_station(row, client=FakeClient())

    assert result["provider"] == "chile_dga"
    assert result["status"] == "subdaily_found"
    assert result["resolution_method"] == "inventory_nearest_gauge"
    assert result["resolved_site_number"] == "08317001"
    assert result["resolved_monitoring_location_id"] == "08317001-8"
    assert result["resolved_station_name"] == "Rio Biobio En Rucalhue"
    assert result["instantaneous_series_count"] == 1
    assert result["daily_series_count"] == 0
    assert result["subdaily_discharge_found"] is True
    assert result["instantaneous_begin"] == "2026-04-28T12:26:47Z"
    assert "historical" in str(result["notes"]).lower()


def test_locate_chile_subdaily_station_resolved_without_live_alert():
    row = pd.Series(
        {
            "station_key": "CL:3179500",
            "country": "CL",
            "source_station_id": "3179500",
            "lat": -36.83,
            "lon": -73.07,
            "occurrence_count": 1,
            "example_ids": "4",
            "down_values": "True",
            "inventory_station_id": "08394001",
            "inventory_station_key": "CL:08394001",
            "inventory_station_name": pd.NA,
            "inventory_resolution_method": "inventory_nearest_gauge",
            "inventory_distance_m": 1197.55,
        }
    )

    class FakeClient:
        def fetch_inventory_station_records(self, station_id: str):
            assert station_id == "08394001"
            return [
                {
                    "COD_BNA": "08394001",
                    "NOM_ESTACION": "Rio Biobio En Desembocadura",
                    "TIPO_ESTACION": "Fluviometricas",
                    "VIGENCIA": "Vigentes",
                    "INSTITUCION": "DGA",
                    "LATITUD": -36.83784751,
                    "LONGITUD": -73.06153398,
                }
            ]

        def fetch_alert_records_for_station_prefix(self, station_id: str):
            assert station_id == "08394001"
            return []

    result = locate_chile_subdaily_station(row, client=FakeClient())

    assert result["status"] == "resolved_no_subdaily"
    assert result["resolved_site_number"] == "08394001"
    assert result["resolved_station_name"] == "Rio Biobio En Desembocadura"
    assert result["subdaily_discharge_found"] is False
    assert result["discharge_series_found"] is False
    assert "alertas" in str(result["notes"]).lower()


def test_locate_subdaily_from_hierarchy_examples_supports_chile_inventory_reconciliation(tmp_path: Path):
    gpkg_path = tmp_path / "hierarchy_examples_filtered.gpkg"
    inventory_path = tmp_path / "gauges_cleaned.csv"
    with sqlite3.connect(gpkg_path) as connection:
        connection.execute(
            """
            CREATE TABLE hierarchy_examples_filtered (
                station_key TEXT,
                lat REAL,
                lon REAL,
                down TEXT,
                example_id REAL
            )
            """
        )
        connection.executemany(
            "INSERT INTO hierarchy_examples_filtered VALUES (?, ?, ?, ?, ?)",
            [
                ("CL:3179520", -37.5503, -72.5903, "", 1.0),
                ("CL:3179530", -37.7106, -71.9017, "", 2.0),
            ],
        )

    pd.DataFrame(
        [
            {
                "station_id": "08334001",
                "station_key": "CL:08334001",
                "country": "CL",
                "lat": -37.5503,
                "lon": -72.5903,
                "station_name": pd.NA,
            },
            {
                "station_id": "08317001",
                "station_key": "CL:08317001",
                "country": "CL",
                "lat": -37.7106,
                "lon": -71.9017,
                "station_name": pd.NA,
            },
        ]
    ).to_csv(inventory_path, index=False)

    class FakeClient:
        def fetch_inventory_station_records(self, station_id: str):
            names = {
                "08334001": "Rio Biobio En Coihue",
                "08317001": "Rio Biobio En Rucalhue",
            }
            return [
                {
                    "COD_BNA": station_id,
                    "NOM_ESTACION": names[station_id],
                    "TIPO_ESTACION": "Fluviometricas",
                    "VIGENCIA": "Vigentes",
                    "INSTITUCION": "DGA",
                    "LATITUD": -37.0,
                    "LONGITUD": -72.0,
                }
            ]

        def fetch_alert_records_for_station_prefix(self, station_id: str):
            suffix = {"08334001": "0", "08317001": "8"}[station_id]
            name = {
                "08334001": "rio biobio en coihue",
                "08317001": "rio biobio en rucalhue",
            }[station_id]
            return [
                {
                    "SITMOP_PROD.SITMOP_DESA.TG_RED_HIDROMETEO.CODBNA": f"{station_id}-{suffix}",
                    "SITMOP_PROD.SITMOP_DESA.TG_RED_HIDROMETEO.NOMBRERED": name,
                    "SITMOP_PROD.SITMOP_DESA.TG_RED_HIDROMETEO.FLUVIOMETRICA": "vig",
                    "SITMOP_PROD.SITMOP_DESA.TG_RED_HIDROMETEO.TIPOTRASMISION": "OTR",
                    "SITMOP_PROD.SDE.V_DGA_GIS_ALERTAS.mod_codest": f"{station_id}-{suffix}",
                    "SITMOP_PROD.SDE.V_DGA_GIS_ALERTAS.mod_fechra": 1777379207000,
                    "SITMOP_PROD.SDE.V_DGA_GIS_ALERTAS.mod_valor": 0.0,
                    "SITMOP_PROD.SDE.V_DGA_GIS_ALERTAS.mod_alerta": 0.0,
                }
            ]

    results = locate_subdaily_from_hierarchy_examples(
        gpkg_path,
        country="CL",
        client=FakeClient(),
        inventory_path=inventory_path,
    )

    assert len(results) == 2
    by_key = {row["station_key"]: row for row in results.to_dict(orient="records")}
    assert by_key["CL:3179520"]["status"] == "subdaily_found"
    assert by_key["CL:3179520"]["inventory_resolution_method"] == "inventory_nearest_gauge"
    assert by_key["CL:3179520"]["resolved_site_number"] == "08334001"
    assert by_key["CL:3179530"]["status"] == "subdaily_found"
    assert by_key["CL:3179530"]["resolved_site_number"] == "08317001"


def test_locate_french_guiana_subdaily_station_for_provider_match():
    row = pd.Series(
        {
            "station_key": "GF:3512025",
            "country": "GF",
            "source_station_id": "3512025",
            "lat": 4.27152742,
            "lon": -54.381600578,
            "occurrence_count": 1,
            "example_ids": "9",
            "down_values": "",
        }
    )

    class FakeClient:
        def fetch_department_stations(self, code_departement: str):
            assert code_departement == "973"
            return [
                {
                    "code_station": "5241000101",
                    "libelle_station": "Le Maroni [Le Lawa] à Grand-Santi",
                    "latitude_station": 4.27152742,
                    "longitude_station": -54.381600578,
                    "en_service": True,
                }
            ]

        def fetch_realtime_discharge_bounds(self, station_id: str):
            assert station_id == "5241000101"
            return ("2026-03-29T17:40:48Z", "2026-04-28T14:28:12Z", 2907)

        def fetch_daily_discharge_bounds(self, station_id: str):
            assert station_id == "5241000101"
            return ("1953-08-01", "2026-04-27", 13329)

    result = locate_french_guiana_subdaily_station(
        row,
        client=FakeClient(),
        now_utc=datetime(2026, 4, 28, tzinfo=timezone.utc),
    )

    assert result["provider"] == "france_hubeau"
    assert result["status"] == "subdaily_found"
    assert result["resolution_method"] == "provider_referential_nearest_station"
    assert result["inventory_station_id"] == "5241000101"
    assert result["resolved_site_number"] == "5241000101"
    assert result["resolved_station_name"] == "Le Maroni [Le Lawa] à Grand-Santi"
    assert result["subdaily_discharge_found"] is True
    assert result["instantaneous_begin"] == "2026-03-29T17:40:48Z"
    assert result["instantaneous_end"] == "2026-04-28T14:28:12Z"
    assert result["daily_begin"] == "1953-08-01"
    assert result["daily_end"] == "2026-04-27"
    assert result["daily_coverage_type"] == "recent_window"


def test_locate_french_guiana_subdaily_station_historical_daily_only():
    row = pd.Series(
        {
            "station_key": "GF:3512400",
            "country": "GF",
            "source_station_id": "3512400",
            "lat": 4.986101,
            "lon": -54.43691,
            "occurrence_count": 1,
            "example_ids": "10",
            "down_values": "True",
        }
    )

    class FakeClient:
        def fetch_department_stations(self, code_departement: str):
            assert code_departement == "973"
            return [
                {
                    "code_station": "5041000101",
                    "libelle_station": "Le Maroni à Apatou [Langa Tabiki]",
                    "latitude_station": 4.986100844,
                    "longitude_station": -54.436911882,
                    "en_service": True,
                }
            ]

        def fetch_realtime_discharge_bounds(self, station_id: str):
            assert station_id == "5041000101"
            return (None, None, 0)

        def fetch_daily_discharge_bounds(self, station_id: str):
            assert station_id == "5041000101"
            return ("1951-11-28", "2008-04-29", 26385)

    result = locate_french_guiana_subdaily_station(
        row,
        client=FakeClient(),
        now_utc=datetime(2026, 4, 28, tzinfo=timezone.utc),
    )

    assert result["status"] == "resolved_historical_daily_only"
    assert result["resolved_site_number"] == "5041000101"
    assert result["subdaily_discharge_found"] is False
    assert result["daily_series_count"] == 1
    assert result["instantaneous_series_count"] == 0
    assert result["daily_begin"] == "1951-11-28"
    assert result["daily_end"] == "2008-04-29"
    assert result["daily_coverage_type"] == "historical_only"


def test_locate_subdaily_from_hierarchy_examples_supports_french_guiana_provider_resolution(tmp_path: Path):
    gpkg_path = tmp_path / "hierarchy_examples_filtered.gpkg"
    with sqlite3.connect(gpkg_path) as connection:
        connection.execute(
            """
            CREATE TABLE hierarchy_examples_filtered (
                station_key TEXT,
                lat REAL,
                lon REAL,
                down TEXT,
                example_id REAL
            )
            """
        )
        connection.executemany(
            "INSERT INTO hierarchy_examples_filtered VALUES (?, ?, ?, ?, ?)",
            [
                ("GF:3512020", 3.805435012, -54.142595126, "", 1.0),
                ("GF:3512400", 4.986101, -54.43691, "True", 2.0),
            ],
        )

    class FakeClient:
        def fetch_department_stations(self, code_departement: str):
            assert code_departement == "973"
            return [
                {
                    "code_station": "5221000201",
                    "libelle_station": "Le Maroni [Lawa] à Papaichton",
                    "latitude_station": 3.805435012,
                    "longitude_station": -54.142595126,
                    "en_service": True,
                },
                {
                    "code_station": "5041000101",
                    "libelle_station": "Le Maroni à Apatou [Langa Tabiki]",
                    "latitude_station": 4.986100844,
                    "longitude_station": -54.436911882,
                    "en_service": True,
                },
            ]

        def fetch_realtime_discharge_bounds(self, station_id: str):
            if station_id == "5221000201":
                return ("2026-03-29T17:40:48Z", "2026-04-28T13:09:51Z", 2896)
            if station_id == "5041000101":
                return (None, None, 0)
            raise AssertionError(f"Unexpected station_id {station_id}")

        def fetch_daily_discharge_bounds(self, station_id: str):
            if station_id == "5221000201":
                return ("2016-02-25", "2026-04-27", 3715)
            if station_id == "5041000101":
                return ("1951-11-28", "2008-04-29", 26385)
            raise AssertionError(f"Unexpected station_id {station_id}")

    results = locate_subdaily_from_hierarchy_examples(
        gpkg_path,
        country="GF",
        client=FakeClient(),
        inventory_path=None,
    )

    assert len(results) == 2
    by_key = {row["station_key"]: row for row in results.to_dict(orient="records")}
    assert by_key["GF:3512020"]["status"] == "subdaily_found"
    assert by_key["GF:3512020"]["resolved_site_number"] == "5221000201"
    assert by_key["GF:3512020"]["inventory_resolution_method"] == "provider_referential_nearest_station"
    assert by_key["GF:3512400"]["status"] == "resolved_historical_daily_only"
    assert by_key["GF:3512400"]["resolved_site_number"] == "5041000101"


def test_locate_colombia_subdaily_station_for_provider_match():
    row = pd.Series(
        {
            "station_key": "CO:3103010",
            "country": "CO",
            "source_station_id": "3103010",
            "lat": 4.387777778,
            "lon": -74.838375,
            "occurrence_count": 1,
            "example_ids": "12",
            "down_values": "",
        }
    )

    class FakeClient:
        def fetch_station_inventory(self):
            return [
                {
                    "id": "0021237010",
                    "nombre": "NARI\u00d1O AUTOM [21237010]",
                    "lng": "-74.8382",
                    "lat": "4.3876",
                    "corriente": "MAGDALENA",
                    "municipio": "NARINO",
                    "depart": "CUNDINAMARCA",
                }
            ]

        def fetch_discharge_payload(self, station_id: str):
            assert station_id == "0021237010"
            return {
                "obs": {
                    "label": "Caudal Observado",
                    "data": [
                        ["2024/05/30 06:00", 1399.56],
                        ["2024/05/30 18:00", 1846.84],
                    ],
                },
                "sen": {
                    "label": "Caudal Sensor",
                    "data": [
                        ["2024/05/30 07:00", 1450.0],
                        ["2024/05/30 08:00", 1460.0],
                    ],
                },
            }

    result = locate_colombia_subdaily_station(row, client=FakeClient())

    assert result["provider"] == "colombia_ideam_fews"
    assert result["status"] == "subdaily_found"
    assert result["resolution_method"] == "provider_referential_nearest_station"
    assert result["inventory_station_id"] == "0021237010"
    assert result["inventory_station_key"] == "CO:0021237010"
    assert result["resolved_site_number"] == "0021237010"
    assert result["resolved_station_name"] == "NARI\u00d1O AUTOM [21237010]"
    assert result["subdaily_discharge_found"] is True
    assert result["discharge_series_count"] == 2
    assert result["instantaneous_series_count"] == 2
    assert result["daily_series_count"] == 0
    assert result["instantaneous_begin"] == "2024-05-30T06:00:00"
    assert result["instantaneous_end"] == "2024-05-30T18:00:00"
    assert "jsonq" in str(result["notes"]).lower()


def test_locate_subdaily_from_hierarchy_examples_supports_colombia_provider_resolution(tmp_path: Path):
    gpkg_path = tmp_path / "hierarchy_examples_filtered.gpkg"
    with sqlite3.connect(gpkg_path) as connection:
        connection.execute(
            """
            CREATE TABLE hierarchy_examples_filtered (
                station_key TEXT,
                lat REAL,
                lon REAL,
                down TEXT,
                example_id REAL
            )
            """
        )
        connection.executemany(
            "INSERT INTO hierarchy_examples_filtered VALUES (?, ?, ?, ?, ?)",
            [
                ("CO:3103010", 4.387777778, -74.838375, "", 1.0),
                ("CO:3103500", 6.5, -74.38, "True", 2.0),
            ],
        )

    class FakeClient:
        def fetch_station_inventory(self):
            return [
                {
                    "id": "0021237010",
                    "nombre": "NARI\u00d1O AUTOM [21237010]",
                    "lng": "-74.8382",
                    "lat": "4.3876",
                },
                {
                    "id": "0023097030",
                    "nombre": "PTO BERRIO AUTOMATICA [23097030]",
                    "lng": "-74.4056",
                    "lat": "6.4894",
                },
            ]

        def fetch_discharge_payload(self, station_id: str):
            if station_id == "0021237010":
                return {
                    "obs": {
                        "label": "Caudal Observado",
                        "data": [
                            ["2024/05/30 06:00", 1399.56],
                            ["2024/05/30 18:00", 1846.84],
                        ],
                    },
                    "sen": {"label": "Caudal Sensor", "data": []},
                }
            if station_id == "0023097030":
                return {
                    "obs": {"label": "Caudal Observado", "data": []},
                    "sen": {"label": "Caudal Sensor", "data": []},
                }
            raise AssertionError(f"Unexpected station_id {station_id}")

    results = locate_subdaily_from_hierarchy_examples(
        gpkg_path,
        country="CO",
        client=FakeClient(),
        inventory_path=None,
    )

    assert len(results) == 2
    by_key = {row["station_key"]: row for row in results.to_dict(orient="records")}
    assert by_key["CO:3103010"]["status"] == "subdaily_found"
    assert by_key["CO:3103010"]["resolved_site_number"] == "0021237010"
    assert by_key["CO:3103010"]["inventory_resolution_method"] == "provider_referential_nearest_station"
    assert by_key["CO:3103500"]["status"] == "resolved_no_discharge"
    assert by_key["CO:3103500"]["resolved_site_number"] == "0023097030"
    assert by_key["CO:3103500"]["monitoring_location_found"] is True
    assert by_key["CO:3103500"]["subdaily_discharge_found"] is False


def test_locate_cambodia_subdaily_station_uses_curated_station_override_and_live_spacing():
    row = pd.Series(
        {
            "station_key": "KH:2569004",
            "country": "KH",
            "source_station_id": "2569004",
            "lat": 12.4767,
            "lon": 106.0150,
            "occurrence_count": 1,
            "example_ids": "88",
            "down_values": "",
        }
    )

    class FakeClient:
        def fetch_time_series_inventory(self):
            return [
                {
                    "uniqueId": "kh-014901-q",
                    "country": "Cambodia",
                    "countryCode": "KH",
                    "stationName": "Kratie",
                    "stationCode": "014901",
                    "parameter": "Discharge",
                    "label": "Calculated daily discharge",
                    "interval": "Unknown",
                    "correctedStartTime": "1924-01-01T00:00:00Z",
                    "correctedEndTime": "2026-04-27T00:00:00Z",
                    "latitude": 12.48141003,
                    "longitude": 106.0176163,
                },
                {
                    "uniqueId": "kh-014901-h",
                    "country": "Cambodia",
                    "countryCode": "KH",
                    "stationName": "Kratie",
                    "stationCode": "014901",
                    "parameter": "Water Level",
                    "label": "Telemetry",
                    "interval": "15 minutes",
                    "correctedStartTime": "2018-05-13T01:45:00Z",
                    "correctedEndTime": "2026-04-28T18:30:00Z",
                    "latitude": 12.48141003,
                    "longitude": 106.0176163,
                },
            ]

        def fetch_corrected_time_series_data(self, unique_id: str):
            assert unique_id == "kh-014901-q"
            return {
                "Points": [
                    {"Timestamp": "2026-04-26T07:00:00.0000000+07:00", "Value": {"Numeric": 4600.0}},
                    {"Timestamp": "2026-04-26T19:00:00.0000000+07:00", "Value": {"Numeric": 4550.0}},
                    {"Timestamp": "2026-04-27T07:00:00.0000000+07:00", "Value": {"Numeric": 4339.7}},
                ]
            }

    result = locate_cambodia_subdaily_station(row, client=FakeClient())

    assert result["provider"] == "mrc_timeseries"
    assert result["status"] == "subdaily_found"
    assert result["resolution_method"] == "provider_curated_station_code"
    assert result["inventory_station_id"] == "014901"
    assert result["inventory_station_key"] == "KH:014901"
    assert result["resolved_site_number"] == "014901"
    assert result["resolved_station_name"] == "Kratie"
    assert result["discharge_series_found"] is True
    assert result["subdaily_discharge_found"] is True
    assert result["discharge_series_count"] == 1
    assert result["instantaneous_series_count"] == 1
    assert result["daily_series_count"] == 1
    assert result["daily_begin"] == "1924-01-01T00:00:00+00:00"
    assert result["daily_end"] == "2026-04-27T00:00:00+00:00"
    assert result["instantaneous_begin"] == "2026-04-26T00:00:00+00:00"
    assert result["instantaneous_end"] == "2026-04-27T00:00:00+00:00"
    assert "spacing below 24 hours" in str(result["notes"]).lower()


def test_locate_subdaily_from_hierarchy_examples_supports_cambodia_mrc_resolution(tmp_path: Path):
    gpkg_path = tmp_path / "hierarchy_examples_filtered.gpkg"
    with sqlite3.connect(gpkg_path) as connection:
        connection.execute(
            """
            CREATE TABLE hierarchy_examples_filtered (
                station_key TEXT,
                lat REAL,
                lon REAL,
                down TEXT,
                example_id REAL
            )
            """
        )
        connection.executemany(
            "INSERT INTO hierarchy_examples_filtered VALUES (?, ?, ?, ?, ?)",
            [
                ("KH:2569002", 11.5833, 104.9425, "", 1.0),
                ("KH:2569004", 12.4767, 106.0150, "True", 2.0),
            ],
        )

    class FakeClient:
        def fetch_time_series_inventory(self):
            return [
                {
                    "uniqueId": "kh-019801-h",
                    "country": "Cambodia",
                    "countryCode": "KH",
                    "stationName": "Chroy Chang Var",
                    "stationCode": "019801",
                    "parameter": "Water Level",
                    "label": "Manual",
                    "interval": "1-6 times p/day",
                    "correctedStartTime": "1959-12-31T17:00:00Z",
                    "correctedEndTime": "2012-12-30T17:00:00Z",
                    "latitude": 11.5874,
                    "longitude": 104.93842,
                },
                {
                    "uniqueId": "kh-014901-q",
                    "country": "Cambodia",
                    "countryCode": "KH",
                    "stationName": "Kratie",
                    "stationCode": "014901",
                    "parameter": "Discharge",
                    "label": "Calculated daily discharge",
                    "interval": "Unknown",
                    "correctedStartTime": "1924-01-01T00:00:00Z",
                    "correctedEndTime": "2026-04-27T00:00:00Z",
                    "latitude": 12.48141003,
                    "longitude": 106.0176163,
                },
            ]

        def fetch_corrected_time_series_data(self, unique_id: str):
            assert unique_id == "kh-014901-q"
            return {
                "Points": [
                    {"Timestamp": "2026-04-26T07:00:00.0000000+07:00", "Value": {"Numeric": 4600.0}},
                    {"Timestamp": "2026-04-26T19:00:00.0000000+07:00", "Value": {"Numeric": 4550.0}},
                    {"Timestamp": "2026-04-27T07:00:00.0000000+07:00", "Value": {"Numeric": 4339.7}},
                ]
            }

    results = locate_subdaily_from_hierarchy_examples(
        gpkg_path,
        country="KH",
        client=FakeClient(),
        inventory_path=None,
    )

    assert len(results) == 2
    by_key = {row["station_key"]: row for row in results.to_dict(orient="records")}
    assert by_key["KH:2569002"]["status"] == "resolved_no_discharge"
    assert by_key["KH:2569002"]["resolved_site_number"] == "019801"
    assert by_key["KH:2569002"]["monitoring_location_found"] is True
    assert by_key["KH:2569002"]["subdaily_discharge_found"] is False
    assert by_key["KH:2569004"]["status"] == "subdaily_found"
    assert by_key["KH:2569004"]["resolved_site_number"] == "014901"
    assert by_key["KH:2569004"]["inventory_resolution_method"] == "provider_curated_station_code"
    assert by_key["KH:2569004"]["subdaily_discharge_found"] is True


def test_locate_laos_subdaily_station_uses_curated_station_override_and_live_spacing():
    row = pd.Series(
        {
            "station_key": "LA:2469260",
            "country": "LA",
            "source_station_id": "2469260",
            "lat": 15.1167,
            "lon": 105.8,
            "occurrence_count": 1,
            "example_ids": "22",
            "down_values": "",
        }
    )

    class FakeClient:
        def fetch_time_series_inventory(self):
            return [
                {
                    "uniqueId": "la-013901-q",
                    "country": "Lao PDR",
                    "countryCode": "LA",
                    "stationName": "Pakse",
                    "stationCode": "013901",
                    "parameter": "Discharge",
                    "label": "Calculated daily discharge",
                    "interval": "Unknown",
                    "correctedStartTime": "1923-01-01T00:00:00Z",
                    "correctedEndTime": "2026-04-27T00:00:00Z",
                    "latitude": 15.09976006,
                    "longitude": 105.8131866,
                },
                {
                    "uniqueId": "la-013901-h",
                    "country": "Lao PDR",
                    "countryCode": "LA",
                    "stationName": "Pakse",
                    "stationCode": "013901",
                    "parameter": "Water Level",
                    "label": "Telemetry",
                    "interval": "15 minutes",
                    "correctedStartTime": "2018-04-25T07:30:00Z",
                    "correctedEndTime": "2026-04-28T18:30:00Z",
                    "latitude": 15.09976006,
                    "longitude": 105.8131866,
                },
            ]

        def fetch_corrected_time_series_data(self, unique_id: str):
            assert unique_id == "la-013901-q"
            return {
                "Points": [
                    {"Timestamp": "2026-04-26T07:00:00.0000000+07:00", "Value": {"Numeric": 3600.0}},
                    {"Timestamp": "2026-04-26T19:00:00.0000000+07:00", "Value": {"Numeric": 3550.0}},
                    {"Timestamp": "2026-04-27T07:00:00.0000000+07:00", "Value": {"Numeric": 3497.7}},
                ]
            }

    result = locate_laos_subdaily_station(row, client=FakeClient())

    assert result["provider"] == "mrc_timeseries"
    assert result["status"] == "subdaily_found"
    assert result["resolution_method"] == "provider_curated_station_code"
    assert result["inventory_station_id"] == "013901"
    assert result["inventory_station_key"] == "LA:013901"
    assert result["resolved_site_number"] == "013901"
    assert result["resolved_station_name"] == "Pakse"
    assert result["subdaily_discharge_found"] is True
    assert result["discharge_series_count"] == 1
    assert result["instantaneous_series_count"] == 1
    assert result["daily_series_count"] == 1
    assert result["daily_begin"] == "1923-01-01T00:00:00+00:00"
    assert result["daily_end"] == "2026-04-27T00:00:00+00:00"
    assert result["instantaneous_begin"] == "2026-04-26T00:00:00+00:00"
    assert result["instantaneous_end"] == "2026-04-27T00:00:00+00:00"


def test_locate_subdaily_from_hierarchy_examples_supports_laos_mrc_resolution(tmp_path: Path):
    gpkg_path = tmp_path / "hierarchy_examples_filtered.gpkg"
    with sqlite3.connect(gpkg_path) as connection:
        connection.execute(
            """
            CREATE TABLE hierarchy_examples_filtered (
                station_key TEXT,
                lat REAL,
                lon REAL,
                down TEXT,
                example_id REAL
            )
            """
        )
        connection.executemany(
            "INSERT INTO hierarchy_examples_filtered VALUES (?, ?, ?, ?, ?)",
            [
                ("LA:2469260", 15.1167, 105.8, "", 22.0),
            ],
        )

    class FakeClient:
        def fetch_time_series_inventory(self):
            return [
                {
                    "uniqueId": "la-013901-q",
                    "country": "Lao PDR",
                    "countryCode": "LA",
                    "stationName": "Pakse",
                    "stationCode": "013901",
                    "parameter": "Discharge",
                    "label": "Calculated daily discharge",
                    "interval": "Unknown",
                    "correctedStartTime": "1923-01-01T00:00:00Z",
                    "correctedEndTime": "2026-04-27T00:00:00Z",
                    "latitude": 15.09976006,
                    "longitude": 105.8131866,
                }
            ]

        def fetch_corrected_time_series_data(self, unique_id: str):
            assert unique_id == "la-013901-q"
            return {
                "Points": [
                    {"Timestamp": "2026-04-26T07:00:00.0000000+07:00", "Value": {"Numeric": 3600.0}},
                    {"Timestamp": "2026-04-26T19:00:00.0000000+07:00", "Value": {"Numeric": 3550.0}},
                    {"Timestamp": "2026-04-27T07:00:00.0000000+07:00", "Value": {"Numeric": 3497.7}},
                ]
            }

    results = locate_subdaily_from_hierarchy_examples(
        gpkg_path,
        country="LA",
        client=FakeClient(),
        inventory_path=None,
    )

    assert len(results) == 1
    record = results.iloc[0].to_dict()
    assert record["status"] == "subdaily_found"
    assert record["resolved_site_number"] == "013901"
    assert record["inventory_resolution_method"] == "provider_curated_station_code"
    assert record["subdaily_discharge_found"] is True


def test_locate_thailand_subdaily_station_prefers_curated_nong_khai_discharge():
    row = pd.Series(
        {
            "station_key": "TH:2969090",
            "country": "TH",
            "source_station_id": "2969090",
            "lat": 17.8767,
            "lon": 102.72,
            "occurrence_count": 1,
            "example_ids": "21",
            "down_values": "True",
        }
    )

    class FakeClient:
        def fetch_time_series_inventory(self):
            return [
                {
                    "uniqueId": "th-012001-q",
                    "country": "Thailand",
                    "countryCode": "TH",
                    "stationName": "Nong Khai",
                    "stationCode": "012001",
                    "parameter": "Discharge",
                    "label": "Calculated daily discharge",
                    "interval": "Unknown",
                    "correctedStartTime": "1969-01-01T00:00:00Z",
                    "correctedEndTime": "2026-04-27T00:00:00Z",
                    "latitude": 17.88143921,
                    "longitude": 102.7322006,
                },
                {
                    "uniqueId": "th-012002-h",
                    "country": "Thailand",
                    "countryCode": "TH",
                    "stationName": "Nong Khai 2",
                    "stationCode": "012002",
                    "parameter": "Water Level",
                    "label": "Telemetry",
                    "interval": "15 minutes",
                    "correctedStartTime": "2022-03-14T09:00:00Z",
                    "correctedEndTime": "2026-04-23T00:45:00Z",
                    "latitude": 17.8777504,
                    "longitude": 102.7166672,
                },
            ]

        def fetch_corrected_time_series_data(self, unique_id: str):
            assert unique_id == "th-012001-q"
            return {
                "Points": [
                    {"Timestamp": "2026-04-26T07:00:00.0000000+07:00", "Value": {"Numeric": 2400.0}},
                    {"Timestamp": "2026-04-26T19:00:00.0000000+07:00", "Value": {"Numeric": 2360.0}},
                    {"Timestamp": "2026-04-27T07:00:00.0000000+07:00", "Value": {"Numeric": 2323.3}},
                ]
            }

    result = locate_thailand_subdaily_station(row, client=FakeClient())

    assert result["provider"] == "mrc_timeseries"
    assert result["status"] == "subdaily_found"
    assert result["resolution_method"] == "provider_curated_station_code"
    assert result["inventory_station_id"] == "012001"
    assert result["inventory_station_key"] == "TH:012001"
    assert result["resolved_site_number"] == "012001"
    assert result["resolved_station_name"] == "Nong Khai"
    assert result["subdaily_discharge_found"] is True


def test_locate_thailand_subdaily_station_leaves_ambiguous_nearby_only_seed_unresolved():
    row = pd.Series(
        {
            "station_key": "TH:2969257",
            "country": "TH",
            "source_station_id": "2969257",
            "lat": 17.834389,
            "lon": 102.693111,
            "occurrence_count": 1,
            "example_ids": "21",
            "down_values": "True",
        }
    )

    class FakeClient:
        def fetch_time_series_inventory(self):
            return [
                {
                    "uniqueId": "th-012001-q",
                    "country": "Thailand",
                    "countryCode": "TH",
                    "stationName": "Nong Khai",
                    "stationCode": "012001",
                    "parameter": "Discharge",
                    "label": "Calculated daily discharge",
                    "interval": "Unknown",
                    "correctedStartTime": "1969-01-01T00:00:00Z",
                    "correctedEndTime": "2026-04-27T00:00:00Z",
                    "latitude": 17.88143921,
                    "longitude": 102.7322006,
                },
                {
                    "uniqueId": "th-012002-h",
                    "country": "Thailand",
                    "countryCode": "TH",
                    "stationName": "Nong Khai 2",
                    "stationCode": "012002",
                    "parameter": "Water Level",
                    "label": "Telemetry",
                    "interval": "15 minutes",
                    "correctedStartTime": "2022-03-14T09:00:00Z",
                    "correctedEndTime": "2026-04-23T00:45:00Z",
                    "latitude": 17.8777504,
                    "longitude": 102.7166672,
                },
            ]

        def fetch_corrected_time_series_data(self, unique_id: str):
            raise AssertionError("No corrected discharge call should happen for unresolved nearby-only seeds")

    result = locate_thailand_subdaily_station(row, client=FakeClient(), max_resolution_distance_m=5_000.0)

    assert result["status"] == "unresolved"
    assert result["resolution_method"] == "provider_referential_no_plausible_match"
    assert result["resolved_site_number"] is None
    assert result["monitoring_location_found"] is False
    assert result["subdaily_discharge_found"] is False


def test_locate_subdaily_from_hierarchy_examples_supports_thailand_mrc_resolution(tmp_path: Path):
    gpkg_path = tmp_path / "hierarchy_examples_filtered.gpkg"
    with sqlite3.connect(gpkg_path) as connection:
        connection.execute(
            """
            CREATE TABLE hierarchy_examples_filtered (
                station_key TEXT,
                lat REAL,
                lon REAL,
                down TEXT,
                example_id REAL
            )
            """
        )
        connection.executemany(
            "INSERT INTO hierarchy_examples_filtered VALUES (?, ?, ?, ?, ?)",
            [
                ("TH:2969090", 17.8767, 102.72, "True", 21.0),
                ("TH:2969257", 17.834389, 102.693111, "True", 21.0),
            ],
        )

    class FakeClient:
        def fetch_time_series_inventory(self):
            return [
                {
                    "uniqueId": "th-012001-q",
                    "country": "Thailand",
                    "countryCode": "TH",
                    "stationName": "Nong Khai",
                    "stationCode": "012001",
                    "parameter": "Discharge",
                    "label": "Calculated daily discharge",
                    "interval": "Unknown",
                    "correctedStartTime": "1969-01-01T00:00:00Z",
                    "correctedEndTime": "2026-04-27T00:00:00Z",
                    "latitude": 17.88143921,
                    "longitude": 102.7322006,
                },
                {
                    "uniqueId": "th-012002-h",
                    "country": "Thailand",
                    "countryCode": "TH",
                    "stationName": "Nong Khai 2",
                    "stationCode": "012002",
                    "parameter": "Water Level",
                    "label": "Telemetry",
                    "interval": "15 minutes",
                    "correctedStartTime": "2022-03-14T09:00:00Z",
                    "correctedEndTime": "2026-04-23T00:45:00Z",
                    "latitude": 17.8777504,
                    "longitude": 102.7166672,
                },
            ]

        def fetch_corrected_time_series_data(self, unique_id: str):
            assert unique_id == "th-012001-q"
            return {
                "Points": [
                    {"Timestamp": "2026-04-26T07:00:00.0000000+07:00", "Value": {"Numeric": 2400.0}},
                    {"Timestamp": "2026-04-26T19:00:00.0000000+07:00", "Value": {"Numeric": 2360.0}},
                    {"Timestamp": "2026-04-27T07:00:00.0000000+07:00", "Value": {"Numeric": 2323.3}},
                ]
            }

    results = locate_subdaily_from_hierarchy_examples(
        gpkg_path,
        country="TH",
        client=FakeClient(),
        inventory_path=None,
    )

    assert len(results) == 2
    by_key = {row["station_key"]: row for row in results.to_dict(orient="records")}
    assert by_key["TH:2969090"]["status"] == "subdaily_found"
    assert by_key["TH:2969090"]["resolved_site_number"] == "012001"
    assert by_key["TH:2969090"]["inventory_resolution_method"] == "provider_curated_station_code"
    assert by_key["TH:2969090"]["subdaily_discharge_found"] is True
    assert by_key["TH:2969257"]["status"] == "unresolved"
    assert by_key["TH:2969257"]["subdaily_discharge_found"] is False


def test_bulgaria_appd_client_parses_daily_discharge_and_graph_station_names():
    class FakeClient(BulgariaAppdClient):
        def _get_text(self, url: str) -> str:
            assert url.endswith("/hidrology-en")
            return """
            <h2>Water levels</h2>
            <h3>Water levels on the bulgarian section of the Danube river 28.04.2026 г.<br/>
                Hydrometeorological stations</h3>
            <table>
              <tr>
                <td>station</td>
                <td>kilo­metre</td>
                <td>water level (cm)</td>
                <td>discharge (m3/s)</td>
              </tr>
              <tr>
                <td>Svishtov</td>
                <td>554.30</td>
                <td><span style="color: blue;">173</span></td>
                <td>4277</td>
              </tr>
              <tr>
                <td>Ruse</td>
                <td>495.60</td>
                <td><span style="color: blue;">162</span></td>
                <td>4263</td>
              </tr>
              <tr>
                <td>Silistra</td>
                <td>375.50</td>
                <td><span style="color: blue;">192</span></td>
                <td>4512</td>
              </tr>
            </table>
            <h3>Automated gauging stations</h3>
            <table>
              <tr><td>station</td><td>kilo­metre</td><td>water level [cm]</td></tr>
              <tr><td>Svishtov</td><td>554.30</td><td>168</td></tr>
              <tr><td>Силистра</td><td>375.50</td><td>189</td></tr>
            </table>
            <h3>Water level graphs for the last 24 hours</h3>
            <div><h4>Svishtov (554.30 km)</h4></div>
            <div><h4>Силистра (375.50 km)</h4></div>
            </section>
            """

    snapshot = FakeClient().fetch_hydrology_snapshot()

    assert snapshot.report_date.isoformat() == "2026-04-28"
    assert snapshot.daily_records["svishtov"].discharge_m3s == 4277.0
    assert snapshot.daily_records["ruse"].discharge_m3s == 4263.0
    assert snapshot.daily_records["silistra"].kilometre == 375.5
    assert "svishtov" in snapshot.automated_station_names
    assert "silistra" in snapshot.automated_station_names
    assert "svishtov" in snapshot.graph_station_names
    assert "silistra" in snapshot.graph_station_names
    assert "ruse" not in snapshot.graph_station_names


def test_locate_bulgaria_subdaily_station_daily_only():
    row = pd.Series(
        {
            "station_key": "BG:6842700",
            "country": "BG",
            "source_station_id": "6842700",
            "lat": 43.63,
            "lon": 25.35,
            "occurrence_count": 1,
            "example_ids": "30",
            "down_values": "",
        }
    )

    class FakeClient:
        def fetch_hydrology_snapshot(self):
            return type(
                "Snapshot",
                (),
                {
                    "report_date": pd.Timestamp("2026-04-28").date(),
                    "daily_records": {
                        "svishtov": type(
                            "DailyRecord",
                            (),
                            {
                                "station_name": "Svishtov",
                                "kilometre": 554.3,
                                "water_level_cm": 173.0,
                                "discharge_m3s": 4277.0,
                            },
                        )()
                    },
                    "automated_station_names": {"svishtov"},
                    "graph_station_names": {"svishtov"},
                },
            )()

    result = locate_bulgaria_subdaily_station(row, client=FakeClient())

    assert result["provider"] == "bulgaria_appd"
    assert result["status"] == "resolved_no_subdaily"
    assert result["resolution_method"] == "provider_curated_station_name"
    assert result["resolved_site_number"] == "Svishtov (554.30 km)"
    assert result["daily_series_count"] == 1
    assert result["instantaneous_series_count"] == 0
    assert result["subdaily_discharge_found"] is False
    assert result["daily_begin"] == "2026-04-28"
    assert result["daily_coverage_type"] == "recent_window"
    assert "water-level graph" in str(result["notes"]).lower()


def test_locate_subdaily_from_hierarchy_examples_supports_bulgaria_appd_resolution(tmp_path: Path):
    gpkg_path = tmp_path / "hierarchy_examples_filtered.gpkg"
    with sqlite3.connect(gpkg_path) as connection:
        connection.execute(
            """
            CREATE TABLE hierarchy_examples_filtered (
                station_key TEXT,
                lat REAL,
                lon REAL,
                down TEXT,
                example_id REAL
            )
            """
        )
        connection.executemany(
            "INSERT INTO hierarchy_examples_filtered VALUES (?, ?, ?, ?, ?)",
            [
                ("BG:6842700", 43.63, 25.35, "", 30.0),
                ("BG:6842800", 43.86, 25.95, "True", 30.0),
                ("BG:6842900", 44.13, 27.26, "True", 30.0),
            ],
        )

    class FakeClient:
        def fetch_hydrology_snapshot(self):
            daily_record = lambda name, km, q: type(
                "DailyRecord",
                (),
                {
                    "station_name": name,
                    "kilometre": km,
                    "water_level_cm": 100.0,
                    "discharge_m3s": q,
                },
            )()

            return type(
                "Snapshot",
                (),
                {
                    "report_date": pd.Timestamp("2026-04-28").date(),
                    "daily_records": {
                        "svishtov": daily_record("Svishtov", 554.3, 4277.0),
                        "ruse": daily_record("Ruse", 495.6, 4263.0),
                        "silistra": daily_record("Silistra", 375.5, 4512.0),
                    },
                    "automated_station_names": {"svishtov", "silistra"},
                    "graph_station_names": {"svishtov", "silistra"},
                },
            )()

    results = locate_subdaily_from_hierarchy_examples(
        gpkg_path,
        country="BG",
        client=FakeClient(),
        inventory_path=None,
    )

    assert len(results) == 3
    by_key = {row["station_key"]: row for row in results.to_dict(orient="records")}
    assert by_key["BG:6842700"]["status"] == "resolved_no_subdaily"
    assert by_key["BG:6842700"]["resolved_site_number"] == "Svishtov (554.30 km)"
    assert by_key["BG:6842800"]["status"] == "resolved_no_subdaily"
    assert by_key["BG:6842800"]["resolved_site_number"] == "Ruse (495.60 km)"
    assert by_key["BG:6842900"]["status"] == "resolved_no_subdaily"
    assert by_key["BG:6842900"]["resolved_site_number"] == "Silistra (375.50 km)"


def test_locate_mali_subdaily_station_resolves_exact_station_and_daily_only():
    row = pd.Series(
        {
            "station_key": "ML:1134100",
            "country": "ML",
            "source_station_id": "1134100",
            "lat": 12.8667,
            "lon": -7.55,
            "occurrence_count": 1,
            "example_ids": "30",
            "down_values": "",
        }
    )

    class FakeClient:
        def fetch_discharge_station_features(self):
            return [
                {
                    "geometry": {"type": "Point", "coordinates": [-7.55, 12.86]},
                    "properties": {"query_id": "/hydromet/updated_stations/Koulikoro", "year": 2026},
                }
            ]

        def fetch_place_tabs(self, place_id: str, *, layer_id: str = "discharge"):
            assert place_id == "/hydromet/updated_stations/Koulikoro"
            assert layer_id == "discharge"
            return [
                {
                    "displayName": "Water Level",
                    "id": "rwd1",
                    "xAxis": {"start": "2017-07-03T00:00:00", "end": "2021-03-24T00:00:00"},
                },
                {
                    "displayName": "Discharge",
                    "id": "rwd2",
                    "xAxis": {"start": "2006-03-14T00:00:00", "end": "2026-04-13T00:00:00"},
                },
            ]

        def fetch_place_timeseries(self, place_id: str, *, tab_id: str, start: str | None = None, end: str | None = None):
            assert place_id == "/hydromet/updated_stations/Koulikoro"
            assert tab_id == "rwd2"
            assert start == "2006-03-14T00:00:00"
            assert end == "2026-04-13T00:00:00"
            return [
                {
                    "displayName": "Discharge",
                    "id": "rwd2",
                    "charts": [
                        {
                            "id": "/updated_stations/discharge/87",
                            "valueType": "Instantaneous",
                            "data": [
                                ["2026-04-11T00:00:00", 75.408],
                                ["2026-04-12T00:00:00", 73.809],
                                ["2026-04-13T00:00:00", 73.809],
                            ],
                        }
                    ],
                }
            ]

    result = locate_mali_subdaily_station(row, client=FakeClient(), now_utc=datetime(2026, 4, 29, tzinfo=timezone.utc))

    assert result["provider"] == "niger_basin_abn"
    assert result["status"] == "resolved_no_subdaily"
    assert result["resolution_method"] == "provider_referential_nearest_station"
    assert result["resolved_monitoring_location_id"] == "/hydromet/updated_stations/Koulikoro"
    assert result["resolved_site_number"] == "Koulikoro"
    assert result["resolved_station_name"] == "Koulikoro"
    assert result["resolution_distance_m"] < 1000.0
    assert result["daily_series_count"] == 1
    assert result["instantaneous_series_count"] == 0
    assert result["subdaily_discharge_found"] is False
    assert result["daily_begin"] == "2026-04-11T00:00:00"
    assert result["daily_end"] == "2026-04-13T00:00:00"
    assert result["daily_coverage_type"] == "recent_window"
    assert "daily at midnight" in str(result["notes"]).lower()


def test_locate_subdaily_from_hierarchy_examples_supports_mali_abn_resolution(tmp_path: Path):
    gpkg_path = tmp_path / "hierarchy_examples_filtered.gpkg"
    with sqlite3.connect(gpkg_path) as connection:
        connection.execute(
            """
            CREATE TABLE hierarchy_examples_filtered (
                station_key TEXT,
                lat REAL,
                lon REAL,
                down TEXT,
                example_id REAL
            )
            """
        )
        connection.executemany(
            "INSERT INTO hierarchy_examples_filtered VALUES (?, ?, ?, ?, ?)",
            [
                ("ML:1134100", 12.8667, -7.55, "", 30.0),
                ("ML:1134250", 13.7167, -6.05, "", 31.0),
                ("ML:1134400", 13.95, -5.37, "", 32.0),
            ],
        )

    class FakeClient:
        def fetch_discharge_station_features(self):
            return [
                {
                    "geometry": {"type": "Point", "coordinates": [-7.55, 12.86]},
                    "properties": {"query_id": "/hydromet/updated_stations/Koulikoro", "year": 2026},
                },
                {
                    "geometry": {"type": "Point", "coordinates": [-6.06, 13.7]},
                    "properties": {"query_id": "/hydromet/updated_stations/Kirango Aval", "year": 2026},
                },
                {
                    "geometry": {"type": "Point", "coordinates": [-5.35, 13.96]},
                    "properties": {"query_id": "/hydromet/updated_stations/Ke Macina", "year": 2024},
                },
            ]

        def fetch_place_tabs(self, place_id: str, *, layer_id: str = "discharge"):
            assert layer_id == "discharge"
            if place_id == "/hydromet/updated_stations/Koulikoro":
                return [
                    {"displayName": "Water Level", "id": "rwd1", "xAxis": {"start": "2017-07-03T00:00:00", "end": "2021-03-24T00:00:00"}},
                    {"displayName": "Discharge", "id": "rwd2", "xAxis": {"start": "2006-03-14T00:00:00", "end": "2026-04-13T00:00:00"}},
                ]
            if place_id == "/hydromet/updated_stations/Kirango Aval":
                return [
                    {"displayName": "Water Level", "id": "rwd1", "xAxis": {"start": "2017-07-03T00:00:00", "end": "2021-03-24T00:00:00"}},
                    {"displayName": "Discharge", "id": "rwd2", "xAxis": {"start": "2007-01-01T00:00:00", "end": "2026-04-12T00:00:00"}},
                ]
            if place_id == "/hydromet/updated_stations/Ke Macina":
                return [
                    {"displayName": "Water Level", "id": "rwd1", "xAxis": {"start": "2006-03-14T00:00:00", "end": "2024-09-30T00:00:00"}},
                    {"displayName": "Discharge", "id": "rwd2", "xAxis": {"start": "2020-03-01T00:00:00", "end": "2024-08-31T00:00:00"}},
                ]
            raise AssertionError(f"Unexpected place_id {place_id}")

        def fetch_place_timeseries(self, place_id: str, *, tab_id: str, start: str | None = None, end: str | None = None):
            assert tab_id == "rwd2"
            if place_id == "/hydromet/updated_stations/Koulikoro":
                return [
                    {
                        "displayName": "Discharge",
                        "charts": [
                            {
                                "id": "/updated_stations/discharge/87",
                                "valueType": "Instantaneous",
                                "data": [
                                    ["2026-04-11T00:00:00", 75.408],
                                    ["2026-04-12T00:00:00", 73.809],
                                    ["2026-04-13T00:00:00", 73.809],
                                ],
                            }
                        ],
                    }
                ]
            if place_id == "/hydromet/updated_stations/Kirango Aval":
                return [
                    {
                        "displayName": "Discharge",
                        "charts": [
                            {
                                "id": "/updated_stations/discharge/85",
                                "valueType": "Instantaneous",
                                "data": [
                                    ["2026-04-10T00:00:00", 3.4],
                                    ["2026-04-11T00:00:00", 2.833],
                                    ["2026-04-12T00:00:00", 1.7],
                                ],
                            }
                        ],
                    }
                ]
            if place_id == "/hydromet/updated_stations/Ke Macina":
                return [
                    {
                        "displayName": "Discharge",
                        "charts": [
                            {
                                "id": "/updated_stations/discharge/83",
                                "valueType": "Instantaneous",
                                "data": [
                                    ["2024-08-29T00:00:00", 3578.833],
                                    ["2024-08-30T00:00:00", 3566.867],
                                    ["2024-08-31T00:00:00", 3554.9],
                                ],
                            }
                        ],
                    }
                ]
            raise AssertionError(f"Unexpected place_id {place_id}")

    results = locate_subdaily_from_hierarchy_examples(
        gpkg_path,
        country="ML",
        client=FakeClient(),
        inventory_path=None,
        max_resolution_distance_m=5_000.0,
    )

    assert len(results) == 3
    by_key = {row["station_key"]: row for row in results.to_dict(orient="records")}

    assert by_key["ML:1134100"]["status"] == "resolved_no_subdaily"
    assert by_key["ML:1134100"]["resolved_site_number"] == "Koulikoro"
    assert by_key["ML:1134100"]["daily_coverage_type"] == "recent_window"

    assert by_key["ML:1134250"]["status"] == "resolved_no_subdaily"
    assert by_key["ML:1134250"]["resolved_site_number"] == "Kirango Aval"
    assert by_key["ML:1134250"]["daily_coverage_type"] == "recent_window"

    assert by_key["ML:1134400"]["status"] == "resolved_historical_daily_only"
    assert by_key["ML:1134400"]["resolved_site_number"] == "Ke Macina"
    assert by_key["ML:1134400"]["daily_coverage_type"] == "historical_only"


def test_locate_nigeria_subdaily_station_reports_public_request_only():
    row = pd.Series(
        {
            "station_key": "NG:1837253",
            "country": "NG",
            "source_station_id": "1837253",
            "lat": 11.424033,
            "lon": 9.950833,
            "occurrence_count": 1,
            "example_ids": "18",
            "down_values": "True",
        }
    )

    class FakeClient:
        dashboard_url = "https://nihsa.gov.ng/flood-forecast-dashboard/"
        public_api_url = "https://nihsa.gov.ng/flood-forecast-dashboard/api/data"
        data_request_url = "https://nihsa.gov.ng/data-request/"

    result = locate_nigeria_subdaily_station(row, client=FakeClient())

    assert result["provider"] == "nigeria_nihsa"
    assert result["status"] == "unresolved"
    assert result["resolution_method"] == "provider_station_api_not_public"
    assert result["monitoring_location_found"] is False
    assert result["discharge_series_found"] is False
    assert result["subdaily_discharge_found"] is False
    assert result["daily_coverage_type"] == "none"
    assert result["candidate_site_numbers"] == "1837253,FOGGO"
    assert "manual hydrological data-request workflow" in str(result["notes"])


def test_locate_subdaily_from_hierarchy_examples_supports_nigeria_metadata_only(tmp_path: Path):
    gpkg_path = tmp_path / "hierarchy_examples_filtered.gpkg"
    with sqlite3.connect(gpkg_path) as connection:
        connection.execute(
            """
            CREATE TABLE hierarchy_examples_filtered (
                station_key TEXT,
                lat REAL,
                lon REAL,
                down TEXT,
                example_id REAL
            )
            """
        )
        connection.executemany(
            "INSERT INTO hierarchy_examples_filtered VALUES (?, ?, ?, ?, ?)",
            [
                ("NG:1837253", 11.424033, 9.950833, "True", 18.0),
                ("NG:1837255", 10.930033, 9.60565, "", 18.0),
            ],
        )

    class FakeClient:
        dashboard_url = "https://nihsa.gov.ng/flood-forecast-dashboard/"
        public_api_url = "https://nihsa.gov.ng/flood-forecast-dashboard/api/data"
        data_request_url = "https://nihsa.gov.ng/data-request/"

    results = locate_subdaily_from_hierarchy_examples(
        gpkg_path,
        country="NG",
        client=FakeClient(),
        inventory_path=None,
    )

    assert len(results) == 2
    by_key = {row["station_key"]: row for row in results.to_dict(orient="records")}

    assert by_key["NG:1837253"]["status"] == "unresolved"
    assert by_key["NG:1837253"]["provider"] == "nigeria_nihsa"
    assert by_key["NG:1837253"]["candidate_site_numbers"] == "1837253,FOGGO"

    assert by_key["NG:1837255"]["status"] == "unresolved"
    assert by_key["NG:1837255"]["provider"] == "nigeria_nihsa"
    assert by_key["NG:1837255"]["candidate_site_numbers"] == "1837255,BUNGA"


def test_locate_russia_subdaily_station_reports_closed_access_portal():
    row = pd.Series(
        {
            "station_key": "RU:2909150",
            "country": "RU",
            "source_station_id": "2909150",
            "lat": 67.43,
            "lon": 86.48,
            "occurrence_count": 1,
            "example_ids": "27",
            "down_values": "True",
        }
    )

    class FakeClient:
        legacy_portal_url = "https://gmvo.skniivh.ru/"
        closed_contour_url = "https://sslgis.favr.ru/"
        access_instruction_url = "https://rwec.ru/dl/gis/instruction.pdf"

    result = locate_russia_subdaily_station(row, client=FakeClient())

    assert result["provider"] == "russia_gmvo"
    assert result["status"] == "unresolved"
    assert result["resolution_method"] == "provider_portal_closed_access"
    assert result["monitoring_location_found"] is False
    assert result["discharge_series_found"] is False
    assert result["subdaily_discharge_found"] is False
    assert result["daily_coverage_type"] == "none"
    assert result["candidate_site_numbers"] == "2909150,IGARKA"
    assert "decommissioned" in str(result["notes"]).lower()
    assert "closed contour" in str(result["notes"]).lower()


def test_locate_subdaily_from_hierarchy_examples_supports_russia_metadata_only(tmp_path: Path):
    gpkg_path = tmp_path / "hierarchy_examples_filtered.gpkg"
    with sqlite3.connect(gpkg_path) as connection:
        connection.execute(
            """
            CREATE TABLE hierarchy_examples_filtered (
                station_key TEXT,
                lat REAL,
                lon REAL,
                down TEXT,
                example_id REAL
            )
            """
        )
        connection.executemany(
            "INSERT INTO hierarchy_examples_filtered VALUES (?, ?, ?, ?, ?)",
            [
                ("RU:2909150", 67.43, 86.48, "True", 27.0),
                ("RU:2909152", 61.6, 90.08, "", 27.0),
            ],
        )

    class FakeClient:
        legacy_portal_url = "https://gmvo.skniivh.ru/"
        closed_contour_url = "https://sslgis.favr.ru/"
        access_instruction_url = "https://rwec.ru/dl/gis/instruction.pdf"

    results = locate_subdaily_from_hierarchy_examples(
        gpkg_path,
        country="RU",
        client=FakeClient(),
        inventory_path=None,
    )

    assert len(results) == 2
    by_key = {row["station_key"]: row for row in results.to_dict(orient="records")}

    assert by_key["RU:2909150"]["status"] == "unresolved"
    assert by_key["RU:2909150"]["provider"] == "russia_gmvo"
    assert by_key["RU:2909150"]["candidate_site_numbers"] == "2909150,IGARKA"

    assert by_key["RU:2909152"]["status"] == "unresolved"
    assert by_key["RU:2909152"]["provider"] == "russia_gmvo"
    assert by_key["RU:2909152"]["candidate_site_numbers"] == "2909152,POD. TUNGUSKA"
