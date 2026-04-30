from datetime import date, datetime, timezone
from pathlib import Path

import pandas as pd

from gauge_sword_match.chile_manual_excel import parse_chile_manual_sheet
from gauge_sword_match.subdaily_download import (
    _select_download_window,
    download_subdaily_from_audit,
    download_subdaily_to_country_outputs,
)


def test_download_subdaily_from_audit_reuses_cached_provider_series_for_shared_brazil_station(tmp_path: Path):
    audit = pd.DataFrame(
        [
            {
                "station_key": "BR:3637150",
                "country": "BR",
                "provider": "brazil_ana",
                "status": "subdaily_found",
                "resolved_site_number": "17710000",
                "resolved_station_name": "Shared Brazil Station",
            },
            {
                "station_key": "BR:3637152",
                "country": "BR",
                "provider": "brazil_ana",
                "status": "subdaily_found",
                "resolved_site_number": "17710000",
                "resolved_station_name": "Shared Brazil Station",
            },
        ]
    )
    audit_path = tmp_path / "audit.csv"
    audit.to_csv(audit_path, index=False)

    class FakeBrazilClient:
        def __init__(self) -> None:
            self.calls = 0

        def fetch_subdaily_discharge_values(self, station_id: str, *, start_date, end_date):
            self.calls += 1
            assert station_id == "17710000"
            return pd.DataFrame(
                {
                    "DateTime": ["2024-01-01T00:00:00Z", "2024-01-01T06:00:00Z"],
                    "Vazao": [10.0, 11.0],
                }
            )

    fake_client = FakeBrazilClient()
    output_path = tmp_path / "timeseries.csv"
    manifest_path = tmp_path / "manifest.csv"
    timeseries, manifest = download_subdaily_from_audit(
        audit_path,
        output_path=output_path,
        manifest_path=manifest_path,
        countries=["BR"],
        clients={"brazil_ana": fake_client},
        now_utc=datetime(2024, 1, 10, tzinfo=timezone.utc),
        target_start_date=date(2024, 1, 1),
    )

    assert fake_client.calls == 1
    assert len(manifest) == 2
    assert set(manifest["download_status"]) == {"ok"}
    assert len(timeseries) == 4
    assert set(timeseries["station_key"]) == {"BR:3637150", "BR:3637152"}
    assert set(timeseries["provider_station_id"]) == {"17710000"}


def test_download_subdaily_from_audit_chunks_brazil_requests_for_long_windows(tmp_path: Path):
    audit = pd.DataFrame(
        [
            {
                "station_key": "BR:3636201",
                "country": "BR",
                "provider": "brazil_ana",
                "status": "subdaily_found",
                "resolved_site_number": "16661000",
                "resolved_station_name": "Chunked Brazil Station",
            }
        ]
    )
    audit_path = tmp_path / "audit.csv"
    audit.to_csv(audit_path, index=False)

    class FakeBrazilClient:
        def __init__(self) -> None:
            self.calls: list[tuple[str, str]] = []

        def fetch_subdaily_discharge_values(self, station_id: str, *, start_date, end_date):
            assert station_id == "16661000"
            self.calls.append((start_date.isoformat(), end_date.isoformat()))
            return pd.DataFrame(
                {
                    "DateTime": [f"{start_date.isoformat()}T00:00:00Z"],
                    "Vazao": [42.0 + len(self.calls)],
                }
            )

    fake_client = FakeBrazilClient()
    timeseries, manifest = download_subdaily_from_audit(
        audit_path,
        countries=["BR"],
        clients={"brazil_ana": fake_client},
        now_utc=datetime(2011, 7, 5, tzinfo=timezone.utc),
    )

    assert len(fake_client.calls) >= 3
    assert len(timeseries) >= 3
    assert manifest.loc[0, "download_status"] == "ok"
    assert timeseries["provider_station_id"].eq("16661000").all()


def test_download_subdaily_from_audit_marks_chile_as_live_snapshot_only(tmp_path: Path):
    audit = pd.DataFrame(
        [
            {
                "station_key": "CL:3179500",
                "country": "CL",
                "provider": "chile_dga",
                "status": "subdaily_found",
                "resolved_site_number": "08394001",
                "resolved_station_name": "Chile Live Gauge",
            }
        ]
    )
    audit_path = tmp_path / "audit.csv"
    audit.to_csv(audit_path, index=False)

    class FakeChileClient:
        def fetch_alert_records_for_station_prefix(self, station_id: str):
            assert station_id == "08394001"
            return [
                {
                    "SITMOP_PROD.SITMOP_DESA.TG_RED_HIDROMETEO.CODBNA": "08394001-8",
                    "SITMOP_PROD.SITMOP_DESA.TG_RED_HIDROMETEO.FLUVIOMETRICA": "VIG",
                    "SITMOP_PROD.SDE.V_DGA_GIS_ALERTAS.mod_codest": "08394001-8",
                    "SITMOP_PROD.SDE.V_DGA_GIS_ALERTAS.mod_fechra": 1_704_067_200_000,
                    "SITMOP_PROD.SDE.V_DGA_GIS_ALERTAS.mod_valor": 25.5,
                }
            ]

    timeseries, manifest = download_subdaily_from_audit(
        audit_path,
        countries=["CL"],
        clients={"chile_dga": FakeChileClient()},
        now_utc=datetime(2025, 1, 10, tzinfo=timezone.utc),
    )

    assert len(timeseries) == 1
    assert manifest.loc[0, "download_status"] == "ok"
    assert manifest.loc[0, "window_strategy"] == "live_snapshot_only"
    assert timeseries.loc[0, "window_strategy"] == "live_snapshot_only"


def test_parse_chile_manual_sheet_extracts_station_rows():
    frame = pd.DataFrame(
        [
            [None, "PERIODO: 01/01/2022 - 31/12/2023", None, None, None, None, None],
            [None, "Estación:", None, None, "RIO BIOBIO EN COIHUE", None, None],
            [None, "Codigo BNA:", None, None, "08334001-0", None, None],
            [None, "MES:", None, "01/2023", None, None, None],
            [None, "DIA", "HORA", "ALTURA (m)", None, "CAUDAL (m3/seg)", "I"],
            [None, 1, "00:00", 0.67, None, 145.8418, "CTO"],
            [None, 1, "03:00", 0.58, None, 114.6910, "CTO"],
        ]
    )

    metadata, parsed = parse_chile_manual_sheet(
        frame,
        source_file="test.xls",
        source_sheet="RIO BIOBIO EN COIHUE",
    )

    assert metadata["station_id"] == "08334001-0"
    assert metadata["station_name"] == "RIO BIOBIO EN COIHUE"
    assert len(parsed) == 2
    assert parsed.loc[0, "discharge"] == 145.8418
    assert str(parsed.loc[0, "time"].tz) == "UTC"


def test_download_subdaily_from_audit_prefers_manual_chile_archive(tmp_path: Path, monkeypatch):
    audit = pd.DataFrame(
        [
            {
                "station_key": "CL:3179520",
                "country": "CL",
                "provider": "chile_dga",
                "status": "subdaily_found",
                "resolved_site_number": "08334001",
                "resolved_station_name": "Chile Manual Gauge",
            }
        ]
    )
    audit_path = tmp_path / "audit.csv"
    audit.to_csv(audit_path, index=False)

    manual_frame = pd.DataFrame(
        {
            "time": pd.to_datetime(["2024-01-01T03:00:00Z", "2024-01-01T06:00:00Z"], utc=True),
            "discharge": [10.0, 12.0],
            "raw_discharge": [10.0, 12.0],
            "unit_of_measure": ["m3/s", "m3/s"],
            "raw_unit_of_measure": ["m3/s", "m3/s"],
            "unit_normalized": [True, True],
            "provider_series_name": ["manual_excel_altura_caudal_instantaneo", "manual_excel_altura_caudal_instantaneo"],
            "provider_series_id": ["08334001", "08334001"],
        }
    )

    def fake_load_chile_manual_archive(station_id: str, archive_dir):
        assert station_id == "08334001"
        return manual_frame, "manual archive used"

    monkeypatch.setattr("gauge_sword_match.subdaily_download.load_chile_manual_archive", fake_load_chile_manual_archive)

    class FakeChileClient:
        def fetch_alert_records_for_station_prefix(self, station_id: str):
            raise AssertionError("live snapshot client should not be called when manual archive is available")

    timeseries, manifest = download_subdaily_from_audit(
        audit_path,
        countries=["CL"],
        clients={"chile_dga": FakeChileClient()},
        provider_contexts={"chile_dga": {"manual_excel_dir": tmp_path / "excel_download"}},
        now_utc=datetime(2025, 1, 10, tzinfo=timezone.utc),
    )

    assert len(timeseries) == 2
    assert manifest.loc[0, "download_status"] == "ok"
    assert "manual archive used" in manifest.loc[0, "notes"]
    assert manifest.loc[0, "window_strategy"] == "full_available_short_record"


def test_download_subdaily_from_audit_widens_window_when_latest_10y_has_large_gaps(tmp_path: Path):
    audit = pd.DataFrame(
        [
            {
                "station_key": "CA:07DD001",
                "country": "CA",
                "provider": "canada_wateroffice",
                "status": "subdaily_found",
                "resolved_site_number": "07DD001",
                "resolved_station_name": "Fallback Canada Gauge",
            }
        ]
    )
    audit_path = tmp_path / "audit.csv"
    audit.to_csv(audit_path, index=False)

    class FakeCanadaClient:
        def fetch_discharge_unit_values(self, station_id: str, *, start_datetime_utc, end_datetime_utc):
            assert station_id == "07DD001"
            historical = pd.DataFrame(
                {
                    "Date": ["2010-01-01T00:00:00Z", "2010-01-02T00:00:00Z"],
                    "Value": [100.0, 101.0],
                }
            )
            recent_dates = pd.date_range("2024-01-01T00:00:00Z", periods=10, freq="D", tz="UTC")
            recent = pd.DataFrame(
                {
                    "Date": recent_dates.astype(str),
                    "Value": [200.0 + idx for idx in range(len(recent_dates))],
                }
            )
            return pd.concat([historical, recent], ignore_index=True)

    timeseries, manifest = download_subdaily_from_audit(
        audit_path,
        countries=["CA"],
        clients={"canada_wateroffice": FakeCanadaClient()},
        now_utc=datetime(2025, 1, 10, tzinfo=timezone.utc),
    )

    assert len(timeseries) == 12
    assert manifest.loc[0, "download_status"] == "ok"
    assert manifest.loc[0, "window_strategy"] == "since_2010_gap_fallback"
    assert manifest.loc[0, "selected_row_count"] == 12


def test_select_download_window_prefers_latest_10y_when_recent_archive_is_solid():
    times = pd.date_range("2014-01-01T00:00:00Z", "2026-01-01T00:00:00Z", freq="30D", tz="UTC")
    frame = pd.DataFrame(
        {
            "time": times,
            "discharge": range(len(times)),
            "raw_discharge": range(len(times)),
            "unit_of_measure": ["m3/s"] * len(times),
            "raw_unit_of_measure": ["m3/s"] * len(times),
            "unit_normalized": [True] * len(times),
            "provider_series_name": ["manual_excel_altura_caudal_instantaneo"] * len(times),
            "provider_series_id": ["08334001-0"] * len(times),
        }
    )

    selected, assessment, strategy = _select_download_window(
        frame,
        provider="chile_dga",
        target_start_date=date(2010, 1, 1),
        minimum_completeness=0.70,
        max_gap_days=183.0,
        fallback_years=(10, 5, 2, 1),
    )

    assert strategy == "latest_10y"
    assert len(selected) < len(frame)
    assert assessment.max_gap_days is not None and assessment.max_gap_days < 183.0


def test_download_subdaily_from_audit_zero_pads_colombia_station_ids(tmp_path: Path):
    audit = pd.DataFrame(
        [
            {
                "station_key": "CO:3103010",
                "country": "CO",
                "provider": "colombia_ideam_fews",
                "status": "subdaily_found",
                # This mirrors the reviewed audit, where CSV serialization dropped leading zeros.
                "resolved_site_number": "21237010",
                "resolved_station_name": "Zero Padded Colombia Gauge",
            }
        ]
    )
    audit_path = tmp_path / "audit.csv"
    audit.to_csv(audit_path, index=False)

    class FakeColombiaClient:
        def fetch_discharge_payload(self, station_id: str):
            assert station_id == "0021237010"
            return {
                "obs": {
                    "data": [
                        ["2024-01-01T00:00:00Z", 1.0],
                        ["2024-01-01T06:00:00Z", 2.0],
                    ]
                }
            }

    timeseries, manifest = download_subdaily_from_audit(
        audit_path,
        countries=["CO"],
        clients={"colombia_ideam_fews": FakeColombiaClient()},
        now_utc=datetime(2025, 1, 10, tzinfo=timezone.utc),
    )

    assert len(timeseries) == 2
    assert manifest.loc[0, "download_status"] == "ok"
    assert manifest.loc[0, "provider_station_id"] == "0021237010"


def test_download_subdaily_from_audit_chunks_usgs_requests_larger_than_limit(tmp_path: Path):
    audit = pd.DataFrame(
        [
            {
                "station_key": "US:13018750",
                "country": "US",
                "provider": "usgs",
                "status": "subdaily_found",
                "resolved_site_number": "13018750",
                "resolved_station_name": "Chunked USGS Gauge",
            }
        ]
    )
    audit_path = tmp_path / "audit.csv"
    audit.to_csv(audit_path, index=False)

    class FakeUSGSClient:
        def __init__(self) -> None:
            self.calls: list[tuple[str, str]] = []

        def fetch_discharge_metadata(self, monitoring_location_id: str):
            assert monitoring_location_id == "USGS-13018750"
            return [
                {
                    "properties": {
                        "id": "ts-1",
                        "monitoring_location_id": monitoring_location_id,
                        "computation_identifier": "Instantaneous",
                        "computation_period_identifier": "Points",
                        "primary": "Primary",
                    }
                }
            ]

        def fetch_continuous_values(self, *, time_series_id: str, start_datetime_utc: str, end_datetime_utc: str, limit: int = 10_000):
            assert time_series_id == "ts-1"
            self.calls.append((start_datetime_utc, end_datetime_utc))
            return pd.DataFrame(
                {
                    "time": [start_datetime_utc],
                    "value": [10.0 + len(self.calls)],
                    "unit_of_measure": ["ft3/s"],
                    "time_series_id": [time_series_id],
                }
            )

    fake_client = FakeUSGSClient()
    timeseries, manifest = download_subdaily_from_audit(
        audit_path,
        countries=["US"],
        clients={"usgs": fake_client},
        now_utc=datetime(2014, 6, 1, tzinfo=timezone.utc),
    )

    assert len(fake_client.calls) >= 2
    assert len(timeseries) >= 1
    assert manifest.loc[0, "download_status"] == "ok"
    assert timeseries["unit_of_measure"].eq("m3/s").all()


def test_download_subdaily_from_audit_handles_non_scalar_usgs_unit_values(tmp_path: Path):
    audit = pd.DataFrame(
        [
            {
                "station_key": "US:13022500",
                "country": "US",
                "provider": "usgs",
                "status": "subdaily_found",
                "resolved_site_number": "13022500",
                "resolved_station_name": "USGS Unit Edge Case",
            }
        ]
    )
    audit_path = tmp_path / "audit.csv"
    audit.to_csv(audit_path, index=False)

    class FakeUSGSClient:
        def fetch_discharge_metadata(self, monitoring_location_id: str):
            return [
                {
                    "properties": {
                        "id": "ts-2",
                        "monitoring_location_id": monitoring_location_id,
                        "computation_identifier": "Instantaneous",
                        "computation_period_identifier": "Points",
                        "primary": "Primary",
                    }
                }
            ]

        def fetch_continuous_values(self, *, time_series_id: str, start_datetime_utc: str, end_datetime_utc: str, limit: int = 10_000):
            return pd.DataFrame(
                {
                    "time": ["2024-01-01T00:00:00Z", "2024-01-01T06:00:00Z"],
                    "value": [12.0, 14.0],
                    "unit_of_measure": [["ft3/s", "cfs"], ["ft3/s", "cfs"]],
                    "time_series_id": [time_series_id, time_series_id],
                }
            )

    timeseries, manifest = download_subdaily_from_audit(
        audit_path,
        countries=["US"],
        clients={"usgs": FakeUSGSClient()},
        now_utc=datetime(2025, 1, 10, tzinfo=timezone.utc),
    )

    assert len(timeseries) == 2
    assert manifest.loc[0, "download_status"] == "ok"
    assert timeseries["unit_of_measure"].eq("m3/s").all()


def test_download_subdaily_from_audit_adaptively_splits_usgs_timeout_windows(tmp_path: Path):
    audit = pd.DataFrame(
        [
            {
                "station_key": "US:14144700",
                "country": "US",
                "provider": "usgs",
                "status": "subdaily_found",
                "resolved_site_number": "14144700",
                "resolved_station_name": "USGS Timeout Split Gauge",
            }
        ]
    )
    audit_path = tmp_path / "audit.csv"
    audit.to_csv(audit_path, index=False)

    class FakeUSGSClient:
        def __init__(self) -> None:
            self.calls: list[tuple[str, str]] = []

        def fetch_discharge_metadata(self, monitoring_location_id: str):
            return [
                {
                    "properties": {
                        "id": "ts-timeout",
                        "monitoring_location_id": monitoring_location_id,
                        "computation_identifier": "Instantaneous",
                        "computation_period_identifier": "Points",
                        "primary": "Primary",
                    }
                }
            ]

        def fetch_continuous_values(self, *, time_series_id: str, start_datetime_utc: str, end_datetime_utc: str, limit: int = 10_000):
            self.calls.append((start_datetime_utc, end_datetime_utc))
            start = pd.Timestamp(start_datetime_utc)
            end = pd.Timestamp(end_datetime_utc)
            if (end - start).days > 30:
                raise RuntimeError("USGS API request failed for fake endpoint: timed out")
            return pd.DataFrame(
                {
                    "time": [start_datetime_utc],
                    "value": [9.0],
                    "unit_of_measure": ["ft3/s"],
                    "time_series_id": [time_series_id],
                }
            )

    fake_client = FakeUSGSClient()
    timeseries, manifest = download_subdaily_from_audit(
        audit_path,
        countries=["US"],
        clients={"usgs": fake_client},
        now_utc=datetime(2010, 4, 15, tzinfo=timezone.utc),
    )

    assert len(fake_client.calls) >= 4
    assert len(timeseries) >= 2
    assert manifest.loc[0, "download_status"] == "ok"


def test_download_subdaily_to_country_outputs_writes_one_folder_per_country(tmp_path: Path):
    audit = pd.DataFrame(
        [
            {
                "station_key": "CA:07DD001",
                "country": "CA",
                "provider": "canada_wateroffice",
                "status": "subdaily_found",
                "resolved_site_number": "07DD001",
                "resolved_station_name": "Canada Station",
            },
            {
                "station_key": "CL:3179500",
                "country": "CL",
                "provider": "chile_dga",
                "status": "subdaily_found",
                "resolved_site_number": "08394001",
                "resolved_station_name": "Chile Station",
            },
        ]
    )
    audit_path = tmp_path / "audit.csv"
    audit.to_csv(audit_path, index=False)

    class FakeCanadaClient:
        def fetch_discharge_unit_values(self, station_id: str, *, start_datetime_utc, end_datetime_utc):
            assert station_id == "07DD001"
            return pd.DataFrame(
                {
                    "Date": ["2024-01-01T00:00:00Z", "2024-01-01T06:00:00Z"],
                    "Value": [1.0, 2.0],
                }
            )

    class FakeChileClient:
        def fetch_alert_records_for_station_prefix(self, station_id: str):
            assert station_id == "08394001"
            return [
                {
                    "SITMOP_PROD.SITMOP_DESA.TG_RED_HIDROMETEO.CODBNA": "08394001-8",
                    "SITMOP_PROD.SITMOP_DESA.TG_RED_HIDROMETEO.FLUVIOMETRICA": "VIG",
                    "SITMOP_PROD.SDE.V_DGA_GIS_ALERTAS.mod_codest": "08394001-8",
                    "SITMOP_PROD.SDE.V_DGA_GIS_ALERTAS.mod_fechra": 1_704_067_200_000,
                    "SITMOP_PROD.SDE.V_DGA_GIS_ALERTAS.mod_valor": 25.5,
                }
            ]

    output_dir = tmp_path / "subdaily_values"
    summary = download_subdaily_to_country_outputs(
        audit_path,
        output_dir=output_dir,
        clients={
            "canada_wateroffice": FakeCanadaClient(),
            "chile_dga": FakeChileClient(),
        },
        now_utc=datetime(2025, 1, 10, tzinfo=timezone.utc),
    )

    assert list(summary["country"]) == ["CA", "CL"]
    assert (output_dir / "CA" / "subdaily_timeseries.parquet").exists()
    assert (output_dir / "CA" / "subdaily_download_manifest.csv").exists()
    assert (output_dir / "CL" / "subdaily_timeseries.parquet").exists()
    assert (output_dir / "CL" / "subdaily_download_manifest.csv").exists()
    assert (output_dir / "subdaily_country_download_summary.csv").exists()


def test_download_subdaily_to_country_outputs_preserves_existing_summary_rows(tmp_path: Path):
    audit = pd.DataFrame(
        [
            {
                "station_key": "CA:07DD001",
                "country": "CA",
                "provider": "canada_wateroffice",
                "status": "subdaily_found",
                "resolved_site_number": "07DD001",
                "resolved_station_name": "Canada Station",
            }
        ]
    )
    audit_path = tmp_path / "audit.csv"
    audit.to_csv(audit_path, index=False)

    output_dir = tmp_path / "subdaily_values"
    output_dir.mkdir(parents=True, exist_ok=True)
    existing_summary = pd.DataFrame(
        [
            {
                "country": "BR",
                "requested_start": "2010-01-01T00:00:00Z",
                "requested_end": "2026-04-29T00:00:00Z",
                "station_count": 23,
                "ok_count": 23,
                "error_count": 0,
                "no_data_count": 0,
                "missing_provider_station_id_count": 0,
                "timeseries_row_count": 123,
                "timeseries_path": "old/br.parquet",
                "manifest_path": "old/br.csv",
            }
        ]
    )
    existing_summary.to_csv(output_dir / "subdaily_country_download_summary.csv", index=False)

    class FakeCanadaClient:
        def fetch_discharge_unit_values(self, station_id: str, *, start_datetime_utc, end_datetime_utc):
            return pd.DataFrame(
                {
                    "Date": ["2024-01-01T00:00:00Z"],
                    "Value": [1.0],
                }
            )

    summary = download_subdaily_to_country_outputs(
        audit_path,
        output_dir=output_dir,
        countries=["CA"],
        clients={"canada_wateroffice": FakeCanadaClient()},
        now_utc=datetime(2025, 1, 10, tzinfo=timezone.utc),
    )

    assert list(summary["country"]) == ["BR", "CA"]
