import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from xml.etree import ElementTree as ET

import pandas as pd

from gauge_sword_match.subdaily_locator.brazil import BrazilAnaHydroClient, locate_brazil_subdaily_station
from gauge_sword_match.subdaily_locator.canada import locate_canada_subdaily_station
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
